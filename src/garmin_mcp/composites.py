"""Composite one-call briefs for daily planning and coaching.

These aggregate several Garmin endpoints server-side so an assistant doing a
morning briefing or a weekly training review needs one tool call instead of
five to seven. Sub-sections fail independently: a missing night of sleep data
must not take down the whole brief, so each block is fetched in isolation and
failures are reported in an "errors" map instead of raising.
"""

import datetime
import json

# The garmin_client will be set by the main file
garmin_client = None


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client


def _round(value, digits=1):
    return round(value, digits) if isinstance(value, (int, float)) else None


def _minutes(seconds):
    return round(seconds / 60) if isinstance(seconds, (int, float)) else None


def _local_hhmm(epoch_ms):
    """Garmin 'Local' timestamps are epoch ms pre-shifted to local time."""
    if not isinstance(epoch_ms, (int, float)):
        return None
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%H:%M")


def _descriptor_index(descriptors, key, fallback):
    """Resolve a column index from a Garmin value-descriptor list."""
    if isinstance(descriptors, list):
        for d in descriptors:
            if isinstance(d, dict) and key in {v for v in d.values() if isinstance(v, str)}:
                for v in d.values():
                    if isinstance(v, int) and not isinstance(v, bool):
                        return v
    return fallback


def _current_body_battery_level(day):
    """Precise 0-100 level from the last measured row of the values array."""
    rows = day.get("bodyBatteryValuesArray")
    if not isinstance(rows, list):
        return None
    index = _descriptor_index(
        day.get("bodyBatteryValueDescriptorDTOList"), "bodyBatteryLevel", 2
    )
    for row in reversed(rows):
        if isinstance(row, (list, tuple)) and len(row) > index:
            value = row[index]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return round(value)
    return None


def _summarize_session(activity):
    distance = activity.get("distance")
    start = activity.get("startTimeLocal") or ""
    return {
        "date": start[:10] or None,
        "start": start[11:16] or None,
        "type": (activity.get("activityType") or {}).get("typeKey"),
        "name": activity.get("activityName"),
        "event_type": (activity.get("eventType") or {}).get("typeKey"),
        "distance_km": _round(distance / 1000, 2) if isinstance(distance, (int, float)) else None,
        "duration_min": _minutes(activity.get("duration")),
        "avg_hr": activity.get("averageHR"),
        "training_load": _round(activity.get("activityTrainingLoad")),
        "aerobic_te": activity.get("aerobicTrainingEffect"),
        "anaerobic_te": activity.get("anaerobicTrainingEffect"),
    }


def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


def _sessions_between(start_date, end_date):
    activities = garmin_client.get_activities_by_date(start_date, end_date) or []
    sessions = [_drop_none(_summarize_session(a)) for a in activities if isinstance(a, dict)]
    sessions.sort(key=lambda s: (s.get("date") or "", s.get("start") or ""))
    return sessions


def _days_since_last_session(sessions, as_of):
    dates = [s["date"] for s in sessions if s.get("date")]
    if not dates:
        return None
    last = datetime.date.fromisoformat(max(dates))
    return (datetime.date.fromisoformat(as_of) - last).days


def _load_position(date):
    """Acute load vs Garmin's optimal tunnel, via the training status endpoint."""
    data = garmin_client.get_training_status(date) or {}
    latest_map = (data.get("mostRecentTrainingStatus") or {}).get("latestTrainingStatusData") or {}
    entry = next(iter(latest_map.values()), {}) if isinstance(latest_map, dict) else {}
    acwr = entry.get("acuteTrainingLoadDTO") or {}
    return _drop_none(
        {
            "training_status": entry.get("trainingStatusFeedbackPhrase"),
            "acute_load": acwr.get("dailyTrainingLoadAcute"),
            "chronic_load": acwr.get("dailyTrainingLoadChronic"),
            "acwr": acwr.get("dailyAcuteChronicWorkloadRatio"),
            "acwr_status": acwr.get("acwrStatus"),
            "optimal_load_min": _round(acwr.get("minTrainingLoadChronic")),
            "optimal_load_max": _round(acwr.get("maxTrainingLoadChronic")),
        }
    )


def register_tools(app):
    """Register composite brief tools with the MCP server app"""

    @app.tool()
    async def get_wellness_brief(date: str) -> str:
        """One-call morning brief: training readiness, last night's sleep and
        HRV, current body battery (precise 0-100), yesterday's day summary,
        and recent training with days_since_last_session. Use this FIRST for
        day planning, workout decisions, or "how did I sleep?" — call the
        individual tools only to drill into something specific.

        Args:
            date: Today's date in YYYY-MM-DD format
        """
        day = datetime.date.fromisoformat(date)
        yesterday = (day - datetime.timedelta(days=1)).isoformat()
        week_ago = (day - datetime.timedelta(days=6)).isoformat()
        brief = {"date": date}
        errors = {}

        try:
            readiness_list = garmin_client.get_training_readiness(date) or []
            r = next((e for e in readiness_list if isinstance(e, dict)), {})
            brief["readiness"] = _drop_none(
                {
                    "score": r.get("score"),
                    "level": r.get("level"),
                    "feedback": r.get("feedbackShort"),
                    "factors_pct": _drop_none(
                        {
                            "sleep": r.get("sleepScoreFactorPercent"),
                            "sleep_history": r.get("sleepHistoryFactorPercent"),
                            "hrv": r.get("hrvFactorPercent"),
                            "recovery_time": r.get("recoveryTimeFactorPercent"),
                            "training_load": r.get("acwrFactorPercent"),
                            "stress_history": r.get("stressHistoryFactorPercent"),
                        }
                    ),
                }
            )
        except Exception as e:
            errors["readiness"] = str(e)

        try:
            sleep = garmin_client.get_sleep_data(date) or {}
            dto = sleep.get("dailySleepDTO") or {}
            scores = dto.get("sleepScores") or {}
            duration = dto.get("sleepTimeSeconds")
            brief["sleep"] = _drop_none(
                {
                    "score": (scores.get("overall") or {}).get("value"),
                    "qualifier": (scores.get("overall") or {}).get("qualifierKey"),
                    "duration_h": _round(duration / 3600, 2) if isinstance(duration, (int, float)) else None,
                    "deep_min": _minutes(dto.get("deepSleepSeconds")),
                    "light_min": _minutes(dto.get("lightSleepSeconds")),
                    "rem_min": _minutes(dto.get("remSleepSeconds")),
                    "awake_min": _minutes(dto.get("awakeSleepSeconds")),
                    "bedtime": _local_hhmm(dto.get("sleepStartTimestampLocal")),
                    "wake_time": _local_hhmm(dto.get("sleepEndTimestampLocal")),
                    "resting_hr": sleep.get("restingHeartRate"),
                    "avg_overnight_hrv_ms": sleep.get("avgOvernightHrv"),
                    "body_battery_change": sleep.get("bodyBatteryChange"),
                }
            )
        except Exception as e:
            errors["sleep"] = str(e)

        try:
            hrv = (garmin_client.get_hrv_data(date) or {}).get("hrvSummary") or {}
            baseline = hrv.get("baseline") or {}
            brief["hrv"] = _drop_none(
                {
                    "last_night_ms": hrv.get("lastNightAvg"),
                    "weekly_avg_ms": hrv.get("weeklyAvg"),
                    "balanced_low_ms": baseline.get("balancedLow"),
                    "balanced_upper_ms": baseline.get("balancedUpper"),
                    "status": hrv.get("status"),
                }
            )
        except Exception as e:
            errors["hrv"] = str(e)

        try:
            bb_days = garmin_client.get_body_battery(date, date) or []
            bb = next((d for d in bb_days if isinstance(d, dict)), {})
            feedback = bb.get("bodyBatteryDynamicFeedbackEvent") or {}
            brief["body_battery"] = _drop_none(
                {
                    "current_level": _current_body_battery_level(bb),
                    "label": feedback.get("bodyBatteryLevel"),
                    "charged_today": bb.get("charged"),
                    "drained_today": bb.get("drained"),
                }
            )
        except Exception as e:
            errors["body_battery"] = str(e)

        try:
            summary = garmin_client.get_user_summary(yesterday) or {}
            brief["yesterday"] = _drop_none(
                {
                    "steps": summary.get("totalSteps"),
                    "avg_stress": summary.get("averageStressLevel"),
                    "max_stress": summary.get("maxStressLevel"),
                    "intensity_min": (summary.get("moderateIntensityMinutes") or 0)
                    + 2 * (summary.get("vigorousIntensityMinutes") or 0),
                }
            )
        except Exception as e:
            errors["yesterday"] = str(e)

        try:
            sessions = _sessions_between(week_ago, date)
            days_since = _days_since_last_session(sessions, date)
            brief["training"] = _drop_none(
                {
                    "days_since_last_session": days_since,
                    "note": None if days_since is not None else "no sessions recorded in the last 7 days",
                    "last_7d_sessions": len(sessions),
                    "yesterday_sessions": [s for s in sessions if s.get("date") == yesterday],
                    "today_sessions": [s for s in sessions if s.get("date") == date],
                    "load": _load_position(date),
                }
            )
        except Exception as e:
            errors["training"] = str(e)

        if errors:
            brief["errors"] = errors
        return json.dumps(brief, indent=2)

    @app.tool()
    async def get_training_week(end_date: str) -> str:
        """One-call training adherence view for the 7 days ending at end_date:
        every recorded session (type, duration, load, training effect), totals
        by type, days_since_last_session, and acute load vs Garmin's optimal
        band. Use for weekly reviews and daily "has he actually been
        training?" checks; use get_training_load_trend for the full CTL/ATL
        curve over longer ranges.

        Args:
            end_date: Last day of the window in YYYY-MM-DD format
        """
        end = datetime.date.fromisoformat(end_date)
        start = (end - datetime.timedelta(days=6)).isoformat()
        result = {"window": {"start": start, "end": end_date}}
        errors = {}

        try:
            sessions = _sessions_between(start, end_date)
            by_type = {}
            for s in sessions:
                key = s.get("type") or "unknown"
                by_type[key] = by_type.get(key, 0) + 1
            result["sessions"] = sessions
            result["totals"] = _drop_none(
                {
                    "sessions": len(sessions),
                    "by_type": by_type or None,
                    "total_min": sum(s.get("duration_min") or 0 for s in sessions) or None,
                    "total_km": _round(sum(s.get("distance_km") or 0 for s in sessions), 2) or None,
                    "total_load": _round(sum(s.get("training_load") or 0 for s in sessions)) or None,
                }
            )
            days_since = _days_since_last_session(sessions, end_date)
            result["days_since_last_session"] = days_since
            if days_since is None:
                result["note"] = "no sessions recorded in this window"
        except Exception as e:
            errors["sessions"] = str(e)

        try:
            result["load"] = _load_position(end_date)
        except Exception as e:
            errors["load"] = str(e)

        if errors:
            result["errors"] = errors
        return json.dumps(result, indent=2)

    return app
