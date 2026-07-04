"""Tests for the composite brief tools (get_wellness_brief, get_training_week)."""

import json

import pytest
from mcp.server.fastmcp import FastMCP

from garmin_mcp import composites

READINESS = [{
    "calendarDate": "2026-07-02",
    "score": 84,
    "level": "HIGH",
    "feedbackShort": "READY_FOR_ANYTHING",
    "sleepScoreFactorPercent": 58,
    "sleepHistoryFactorPercent": 91,
    "hrvFactorPercent": 98,
    "recoveryTimeFactorPercent": 87,
    "acwrFactorPercent": 100,
    "stressHistoryFactorPercent": 53,
}]

SLEEP = {
    "dailySleepDTO": {
        "sleepTimeSeconds": 21720,
        "deepSleepSeconds": 3900,
        "lightSleepSeconds": 12780,
        "remSleepSeconds": 5040,
        "awakeSleepSeconds": 1260,
        "sleepStartTimestampLocal": 1782958020000,
        "sleepEndTimestampLocal": 1782981000000,
        "sleepScores": {"overall": {"value": 72, "qualifierKey": "FAIR"}},
    },
    "restingHeartRate": 45,
    "avgOvernightHrv": 59.0,
    "bodyBatteryChange": 51,
}

HRV = {
    "hrvSummary": {
        "lastNightAvg": 59,
        "weeklyAvg": 70,
        "status": "BALANCED",
        "baseline": {"balancedLow": 55, "balancedUpper": 81},
    }
}

BODY_BATTERY = [{
    "date": "2026-07-02",
    "charged": 50,
    "drained": 7,
    "bodyBatteryValueDescriptorDTOList": [
        {"bodyBatteryValueDescriptorIndex": 0, "bodyBatteryValueDescriptorKey": "timestamp"},
        {"bodyBatteryValueDescriptorIndex": 1, "bodyBatteryValueDescriptorKey": "bodyBatteryStatus"},
        {"bodyBatteryValueDescriptorIndex": 2, "bodyBatteryValueDescriptorKey": "bodyBatteryLevel"},
    ],
    "bodyBatteryValuesArray": [
        [1782950400000, "MEASURED", 38],
        [1782972000000, "MEASURED", 64],
    ],
    "bodyBatteryDynamicFeedbackEvent": {"bodyBatteryLevel": "MODERATE"},
}]

USER_SUMMARY = {
    "totalSteps": 16010,
    "averageStressLevel": 33,
    "maxStressLevel": 96,
    "moderateIntensityMinutes": 25,
    "vigorousIntensityMinutes": 45,
}

ACTIVITIES = [
    {
        "activityId": 1,
        "activityName": "City of Westminster Run",
        "startTimeLocal": "2026-07-01 14:31:00",
        "activityType": {"typeKey": "running"},
        "eventType": {"typeKey": "uncategorized"},
        "distance": 8123.4,
        "duration": 2711.2,
        "averageHR": 152.0,
        "activityTrainingLoad": 142.9,
        "aerobicTrainingEffect": 3.4,
        "anaerobicTrainingEffect": 0.4,
    },
    {
        "activityId": 2,
        "activityName": "TRX Circuit A",
        "startTimeLocal": "2026-06-29 15:00:00",
        "activityType": {"typeKey": "strength_training"},
        "duration": 1200.0,
        "activityTrainingLoad": 35.0,
    },
]

TRAINING_STATUS = {
    "mostRecentTrainingStatus": {
        "latestTrainingStatusData": {
            "3627784890": {
                "trainingStatusFeedbackPhrase": "RECOVERY_2",
                "acuteTrainingLoadDTO": {
                    "dailyTrainingLoadAcute": 85,
                    "dailyTrainingLoadChronic": 219,
                    "dailyAcuteChronicWorkloadRatio": 0.3,
                    "acwrStatus": "LOW",
                    "minTrainingLoadChronic": 175.2,
                    "maxTrainingLoadChronic": 328.5,
                },
            }
        }
    },
    "mostRecentVO2Max": {"generic": {"vo2MaxValue": 52.0}, "cycling": None},
}


@pytest.fixture
def app_with_composites(mock_garmin_client):
    mock_garmin_client.get_training_readiness.return_value = READINESS
    mock_garmin_client.get_sleep_data.return_value = SLEEP
    mock_garmin_client.get_hrv_data.return_value = HRV
    mock_garmin_client.get_body_battery.return_value = BODY_BATTERY
    mock_garmin_client.get_user_summary.return_value = USER_SUMMARY
    mock_garmin_client.get_activities_by_date.return_value = ACTIVITIES
    mock_garmin_client.get_training_status.return_value = TRAINING_STATUS
    composites.configure(mock_garmin_client)
    app = FastMCP("Test Composites")
    return composites.register_tools(app)


@pytest.mark.asyncio
async def test_wellness_brief_aggregates_all_sections(app_with_composites, mock_garmin_client):
    result = await app_with_composites.call_tool("get_wellness_brief", {"date": "2026-07-02"})
    data = json.loads(result[0][0].text)

    assert data["readiness"]["score"] == 84
    assert data["readiness"]["factors_pct"]["hrv"] == 98
    assert data["sleep"]["score"] == 72
    assert data["sleep"]["duration_h"] == 6.03
    assert data["hrv"]["last_night_ms"] == 59
    assert data["body_battery"]["current_level"] == 64  # precise, not just MODERATE
    assert data["body_battery"]["label"] == "MODERATE"
    assert data["yesterday"]["steps"] == 16010
    assert data["yesterday"]["intensity_min"] == 25 + 2 * 45
    assert data["training"]["days_since_last_session"] == 1
    assert data["training"]["last_7d_sessions"] == 2
    assert data["training"]["yesterday_sessions"][0]["type"] == "running"
    assert data["training"]["load"]["acute_load"] == 85
    assert data["training"]["load"]["optimal_load_min"] == 175.2
    assert "errors" not in data

    mock_garmin_client.get_user_summary.assert_called_once_with("2026-07-01")
    mock_garmin_client.get_activities_by_date.assert_called_once_with("2026-06-26", "2026-07-02")


@pytest.mark.asyncio
async def test_wellness_brief_sections_fail_independently(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_sleep_data.side_effect = RuntimeError("boom")
    result = await app_with_composites.call_tool("get_wellness_brief", {"date": "2026-07-02"})
    data = json.loads(result[0][0].text)

    assert "sleep" not in data
    assert data["errors"]["sleep"] == "boom"
    assert data["readiness"]["score"] == 84  # other sections unaffected
    assert data["training"]["days_since_last_session"] == 1


@pytest.mark.asyncio
async def test_wellness_brief_no_training_recorded(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = []
    result = await app_with_composites.call_tool("get_wellness_brief", {"date": "2026-07-02"})
    data = json.loads(result[0][0].text)

    assert "days_since_last_session" not in data["training"]  # unknown, not 0
    assert data["training"]["note"] == "no sessions recorded in the last 7 days"
    assert data["training"]["last_7d_sessions"] == 0


@pytest.mark.asyncio
async def test_training_week_totals_and_adherence(app_with_composites, mock_garmin_client):
    result = await app_with_composites.call_tool("get_training_week", {"end_date": "2026-07-02"})
    data = json.loads(result[0][0].text)

    assert data["window"] == {"start": "2026-06-26", "end": "2026-07-02"}
    assert data["totals"]["sessions"] == 2
    assert data["totals"]["by_type"] == {"running": 1, "strength_training": 1}
    assert data["totals"]["total_min"] == 45 + 20
    assert data["totals"]["total_km"] == 8.12
    assert data["totals"]["total_load"] == 177.9
    assert data["days_since_last_session"] == 1
    assert data["sessions"][0]["date"] == "2026-06-29"  # sorted oldest first
    assert data["load"]["acwr_status"] == "LOW"
    assert data["load"]["training_status"] == "RECOVERY_2"


@pytest.mark.asyncio
async def test_coach_report_trends_and_direction(app_with_composites, mock_garmin_client):
    # today's load high, 4w ago low -> acute rising; give distinct snapshots by date
    def status_for(date):
        acute = {"2026-07-02": 130, "2026-06-18": 60, "2026-06-04": 40}.get(date, 100)
        return {
            "mostRecentTrainingStatus": {"latestTrainingStatusData": {"dev": {
                "trainingStatusFeedbackPhrase": "PRODUCTIVE_1",
                "acuteTrainingLoadDTO": {
                    "dailyTrainingLoadAcute": acute, "dailyTrainingLoadChronic": 200,
                    "dailyAcuteChronicWorkloadRatio": round(acute / 200, 2), "acwrStatus": "LOW",
                    "minTrainingLoadChronic": 175.2, "maxTrainingLoadChronic": 328.5,
                }}}},
            "mostRecentVO2Max": {"generic": {"vo2MaxPreciseValue": 52.2 if date == "2026-07-02" else 51.0}, "cycling": None},
        }
    mock_garmin_client.get_training_status.side_effect = status_for
    mock_garmin_client.get_max_metrics.side_effect = lambda d: [{"generic": {"vo2MaxPreciseValue": 52.2 if d == "2026-07-02" else 51.0}}]
    mock_garmin_client.get_endurance_score.return_value = {"enduranceScoreDTO": {"overallScore": 5982}}

    result = await app_with_composites.call_tool("get_coach_report", {"end_date": "2026-07-02", "weeks": 6})
    data = json.loads(result[0][0].text)

    assert data["readiness"]["score"] == 84
    assert data["sleep_last_night"]["duration_h"] == 6.03
    assert data["hrv"]["weekly_avg_ms"] == 70
    assert data["body_battery"]["current_level"] == 64
    assert data["adherence"]["days_since_last_session"] == 1
    assert data["adherence"]["by_type_14d"] == {"running": 1, "strength_training": 1}
    assert data["load"]["now"]["tsb"] == 70            # 200 - 130
    assert data["load"]["acute_direction_4w"] == "rising"   # 130 vs 60
    assert data["fitness"]["vo2max_now"] == 52.2
    assert data["fitness"]["vo2max_change"] == 1.2     # 52.2 - 51.0
    assert data["fitness"]["endurance_score"] == 5982
    assert "errors" not in data


@pytest.mark.asyncio
async def test_coach_report_degrades_on_section_failure(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_endurance_score.side_effect = RuntimeError("500")
    mock_garmin_client.get_max_metrics.return_value = [{"generic": {"vo2MaxValue": 52.0}}]
    result = await app_with_composites.call_tool("get_coach_report", {"end_date": "2026-07-02"})
    data = json.loads(result[0][0].text)
    assert data["errors"]["fitness"] == "500"
    assert data["readiness"]["score"] == 84  # rest still present


@pytest.mark.asyncio
async def test_session_analysis_hr_zones_and_pacing(app_with_composites, mock_garmin_client):
    # real-shaped zone data: mostly Zone 4 (a "hard" run)
    mock_garmin_client.get_activity_hr_in_timezones.return_value = [
        {"zoneNumber": 1, "secsInZone": 27.4, "zoneLowBoundary": 100},
        {"zoneNumber": 2, "secsInZone": 141.8, "zoneLowBoundary": 120},
        {"zoneNumber": 3, "secsInZone": 475.9, "zoneLowBoundary": 140},
        {"zoneNumber": 4, "secsInZone": 2428.4, "zoneLowBoundary": 160},
    ]
    mock_garmin_client.get_activity_typed_splits.return_value = {"splits": [
        {"distance": 1000, "duration": 330, "averageHR": 150},
        {"distance": 1000, "duration": 335, "averageHR": 158},
        {"distance": 1000, "duration": 350, "averageHR": 165},
        {"distance": 1000, "duration": 360, "averageHR": 170},
    ]}
    result = await app_with_composites.call_tool("get_session_analysis", {"date": "2026-07-01"})
    data = json.loads(result[0][0].text)

    assert data["activity_id"] == 1
    z = data["hr_zones"]
    assert z["easy_share_pct"] < 10       # only ~5% in Z1-2 — not an easy run
    assert z["hard_share_pct"] > 90
    assert z["by_zone"]["z4"]["pct"] > 70
    p = data["pacing"]
    assert p["shape"] == "faded"          # slowed in the second half
    assert p["drift_pct"] > 3


@pytest.mark.asyncio
async def test_session_analysis_no_activity(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = []
    result = await app_with_composites.call_tool("get_session_analysis", {"date": "2026-07-04"})
    data = json.loads(result[0][0].text)
    assert "no activity found" in data["note"]


@pytest.mark.asyncio
async def test_plan_context_filters_completed_and_lists_workouts(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_training_plans.return_value = {"trainingPlanList": [
        {"trainingPlanId": 44144473, "name": "Strength Builder", "trainingPlanCategory": "STRENGTH",
         "trainingStatus": {"statusKey": "Completed"}, "durationInWeeks": 4, "avgWeeklyWorkouts": 3},
    ]}
    mock_garmin_client.get_workouts.return_value = [
        {"workoutId": 1470743108, "workoutName": "The Hotel Workout", "sportType": {"sportTypeKey": "strength_training"}},
    ]
    result = await app_with_composites.call_tool("get_plan_context", {})
    data = json.loads(result[0][0].text)

    assert data["active_plans"] == []                 # Completed filtered out
    assert data["all_plans_count"] == 1
    assert "no active Garmin plan" in data["note"]
    assert data["saved_workouts"][0]["name"] == "The Hotel Workout"
    assert data["saved_workouts"][0]["id"] == 1470743108
