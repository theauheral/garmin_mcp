"""Regression tests for the 2026-09 composites review.

Each test pins one defect found against live payloads: elapsed time counted
as zone time, a decimated stream dropped wholesale, the day's cool-down walk
analysed instead of its run, a UTC energy curve, old badges shown as recent,
and a threshold field test with no readout.
"""

import datetime

import pytest

from garmin_mcp import composites
from garmin_mcp.composites import (
    _ceiling_share,
    _energy_windows,
    _fetch_stream,
    _gap_cap,
    _integrate_zones,
    _plan_summary,
    _resolve_activity,
    _threshold_test,
    _utc_offset_hours,
)


class FakeClient:
    def __init__(self, details=None, activities=None):
        self._details = details
        self._activities = activities or []

    def get_activity_details(self, activity_id, maxchart=None, maxpoly=None):
        return self._details

    def get_activities_by_date(self, start, end):
        return self._activities


def details(rows, keys=("directHeartRate", "sumDuration", "sumElapsedDuration",
                        "sumDistance", "directSpeed")):
    return {
        "metricDescriptors": [{"metricsIndex": i, "key": k} for i, k in enumerate(keys)],
        "activityDetailMetrics": [{"metrics": list(r)} for r in rows],
    }


# --- timer time, not elapsed -------------------------------------------------


def test_stream_prefers_timer_time_over_elapsed():
    """A 40 s stop at a crossing: elapsed time jumps, timer time does not.
    Garmin's stored zone minutes follow the timer."""
    rows = [(150, 0, 0, 0, 3), (150, 60, 60, 180, 3),
            (150, 61, 101, 183, 3), (150, 120, 160, 360, 3)]
    composites.configure(FakeClient(details=details(rows)))
    stream, error = _fetch_stream(1, 100)
    assert error is None
    assert [r["t"] for r in stream] == [0, 60, 61, 120]


def test_stream_falls_back_to_elapsed_when_no_timer_column():
    rows = [(150, 0), (150, 5)]
    composites.configure(FakeClient(details=details(rows, keys=("directHeartRate", "sumElapsedDuration"))))
    stream, error = _fetch_stream(1, 100)
    assert error is None and [r["t"] for r in stream] == [0, 5]


def test_gap_cap_scales_with_a_decimated_stream():
    """Nine hours at 500 samples is a 65 s spacing — over the 60 s floor, so a
    fixed cap would have dropped every interval and called it unrecomputable."""
    samples = [(t, 140) for t in range(0, 9 * 3600, 65)]
    assert _gap_cap(samples) == 195
    secs, _ = _integrate_zones(samples, [100, 120, 150, 160, 170])
    assert round(sum(secs.values())) == samples[-1][0]


def test_gap_cap_still_drops_a_real_dropout():
    samples = [(0, 140), (1, 140), (2, 140), (300, 140), (301, 140)]
    secs, _ = _integrate_zones(samples, [100, 120, 150, 160, 170])
    assert sum(secs.values()) == 3


# --- the plan's own cap ------------------------------------------------------


def test_ceiling_share_is_time_weighted():
    samples = [(0, 140), (30, 155), (40, 145), (100, 145)]
    out = _ceiling_share(samples, 150)
    assert out["under_pct"] == 90          # 90 of 100 s at or under 150
    assert out["over_min"] == 0.2
    assert out["avg_hr"] == round((140 * 30 + 155 * 10 + 145 * 60) / 100, 1)


def test_ceiling_counts_the_cap_itself_as_under():
    assert _ceiling_share([(0, 150), (10, 150)], 150)["under_pct"] == 100


# --- threshold field test ----------------------------------------------------


def field_test_rows():
    """15 min warm-up at 130, 10 min at 165 then 20 min at 171 running
    4 m/s (250 s/km), 10 min cool-down at 120. Timer time, 5 s samples."""
    rows, dist = [], 0.0
    for t in range(0, 55 * 60 + 1, 5):
        if t < 15 * 60:
            hr, v = 130, 2.8
        elif t < 25 * 60:
            hr, v = 165, 4.0
        elif t < 45 * 60:
            hr, v = 171, 4.0
        else:
            hr, v = 120, 2.5
        rows.append({"t": t, "hr": hr, "dist": dist, "speed": v})
        dist += v * 5
    return rows


def test_threshold_test_reads_the_last_20_minutes_of_the_effort():
    out = _threshold_test(field_test_rows(), garmin_lthr=172)
    assert out["effort_window_min"] == [15.0, 45.0]
    assert out["lthr_estimate_bpm"] == 171
    assert out["first_10_min_avg_hr"] == 165
    assert out["threshold_pace_s_per_km"] == 250
    assert out["lt_zone_floors_bpm"] == [111, 135, 150, 161, 169]
    assert out["garmin_lthr_bpm"] == 172


def test_threshold_test_refuses_a_short_activity():
    rows = [{"t": t, "hr": 160, "dist": None, "speed": None} for t in range(0, 20 * 60, 5)]
    assert "not_applicable" in _threshold_test(rows)


# --- which activity a date means --------------------------------------------


def test_a_date_resolves_to_its_longest_activity_not_its_newest():
    run = {"activityId": 1, "duration": 2700, "activityType": {"typeKey": "running"},
           "startTimeLocal": "2026-09-24 19:00:00"}
    walk = {"activityId": 2, "duration": 600, "activityType": {"typeKey": "walking"},
            "startTimeLocal": "2026-09-24 19:50:00"}
    composites.configure(FakeClient(activities=[walk, run]))  # newest first
    summary, aid, note, others = _resolve_activity(0, "2026-09-24")
    assert aid == 1 and note is None
    assert others == [{"activity_id": 2, "type": "walking", "start": "19:50", "duration_min": 10}]


# --- energy curve ------------------------------------------------------------


def test_utc_offset_comes_from_the_days_own_timestamps():
    bst = {"startTimestampGMT": "2026-09-24T23:00:00.0",
           "startTimestampLocal": "2026-09-25T00:00:00.0"}
    assert _utc_offset_hours(bst) == 1.0
    assert _utc_offset_hours({}) is None


def test_energy_windows_average_adjacent_hours():
    curve = [{"hour": h, "level": lvl} for h, lvl in
             [(8, 60), (9, 90), (10, 80), (11, 40), (12, 20), (15, 95)]]
    peak, trough = _energy_windows(curve)
    assert peak == {"from_hour": 9, "to_hour": 11, "level": 85}
    assert trough == {"from_hour": 11, "to_hour": 13, "level": 30}


# --- plan context ------------------------------------------------------------


def test_plan_summary_names_status_dates_and_position():
    florence = {
        "trainingPlanId": 47608131, "name": "Florence Marathon Plan",
        "trainingPlanCategory": "FBT_ADAPTIVE",
        "trainingStatus": {"statusId": 1, "statusKey": "Scheduled"},
        "durationInWeeks": 11, "startDate": "2026-09-11T00:00:00.0",
        "endDate": "2026-11-29T00:00:00.0", "trainingType": {"typeKey": "Running"},
        "trainingVersion": {"versionName": "Pace"},
    }
    out = _plan_summary(florence, datetime.date(2026, 9, 25))
    assert out["adaptive"] is True
    assert out["status"] == "Scheduled"
    assert out["end_date"] == "2026-11-29"
    assert out["days_to_end"] == 65
    assert out["current_week"] == 3


# --- wins --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wins_only_lists_badges_inside_the_window(monkeypatch):
    from unittest.mock import MagicMock
    from mcp.server.fastmcp import FastMCP

    today = datetime.date.today()
    recent = (today - datetime.timedelta(days=3)).isoformat() + "T09:00:00.0"
    old = (today - datetime.timedelta(days=400)).isoformat() + "T09:00:00.0"
    client = MagicMock()
    client.get_personal_record.return_value = []
    client.get_earned_badges.return_value = [
        {"badgeName": "Old", "badgeEarnedDate": old},
        {"badgeName": "New", "badgeEarnedDate": recent},
    ]
    client.get_race_predictions.return_value = {}
    composites.configure(client)
    app = FastMCP("t")
    composites.register_tools(app)
    result = await app.call_tool("get_wins", {"days": 14})
    import json
    text = result[0][0].text if isinstance(result, tuple) else result[0].text
    assert [b["name"] for b in json.loads(text)["recent_badges"]] == ["New"]


def test_adaptive_plan_detail_keeps_phases_and_only_upcoming_sessions():
    from garmin_mcp.composites import _adaptive_plan_detail

    class Client:
        def get_adaptive_training_plan_by_id(self, plan_id):
            return {
                "adaptivePlanPhases": [
                    {"startDate": "2026-09-11", "endDate": "2026-10-04", "trainingPhase": "BASE", "currentPhase": True},
                    {"startDate": "2026-10-05", "endDate": "2026-11-01", "trainingPhase": "BUILD", "currentPhase": False},
                ],
                "taskList": [
                    {"calendarDate": "2026-09-24", "taskWorkout": {"workoutName": "Base"}},
                    {"calendarDate": "2026-09-26", "longWkt": False, "taskWorkout": {
                        "sportType": {"sportTypeKey": "strength_training"}, "workoutName": "Core Circuit 2",
                        "estimatedDurationInSecs": 1500, "trainingEffectLabel": "INVALID",
                        "restDay": False, "adaptiveCoachingWorkoutStatus": "NOT_COMPLETE",
                        "workoutUuid": "u1"}},
                ],
            }

    composites.configure(Client())
    out = _adaptive_plan_detail(47608131, datetime.date(2026, 9, 25))
    assert out["phases"][0] == {"phase": "BASE", "start": "2026-09-11", "end": "2026-10-04", "current": True}
    assert "current" not in out["phases"][1]
    assert out["upcoming"] == [{"date": "2026-09-26", "sport": "strength_training",
                                "name": "Core Circuit 2", "est_min": 25,
                                "status": "NOT_COMPLETE", "workout_uuid": "u1"}]


def test_energy_windows_never_return_one_window_as_both_peak_and_trough():
    curve = [{"hour": h, "level": lvl} for h, lvl in [(0, 70), (7, 90), (15, 50), (16, 46)]]
    peak, trough = _energy_windows(curve)
    assert peak == {"from_hour": 7, "to_hour": 8, "level": 90}
    assert trough == {"from_hour": 16, "to_hour": 17, "level": 46}
