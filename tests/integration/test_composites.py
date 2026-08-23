"""Tests for the composite brief/coaching tools (wellness brief, training week,
coach report, session analysis, running dynamics, execution trend, ...)."""

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
    mock_garmin_client.get_activity_splits.return_value = {"lapDTOs": [
        {"lapIndex": 1, "distance": 1000, "duration": 330, "averageHR": 150},
        {"lapIndex": 2, "distance": 1000, "duration": 335, "averageHR": 158},
        {"lapIndex": 3, "distance": 1000, "duration": 350, "averageHR": 165},
        {"lapIndex": 4, "distance": 1000, "duration": 360, "averageHR": 170},
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


@pytest.mark.asyncio
async def test_execution_trend_classifies_and_detects_monotony(app_with_composites, mock_garmin_client):
    # 3 runs, same distance (monotony); zones make them hard/hard/easy
    mock_garmin_client.get_activities.return_value = [
        {"activityId": 1, "activityType": {"typeKey": "running"}, "startTimeLocal": "2026-07-03 19:25", "distance": 7200, "averageHR": 166, "aerobicTrainingEffect": 4.2},
        {"activityId": 2, "activityType": {"typeKey": "running"}, "startTimeLocal": "2026-07-01 14:31", "distance": 7200, "averageHR": 165, "aerobicTrainingEffect": 4.0},
        {"activityId": 3, "activityType": {"typeKey": "running"}, "startTimeLocal": "2026-06-28 10:00", "distance": 7200, "averageHR": 140, "aerobicTrainingEffect": 2.5},
    ]
    def zones_for(aid):
        if aid == 3:  # easy run
            return [{"zoneNumber": 1, "secsInZone": 300}, {"zoneNumber": 2, "secsInZone": 1500}, {"zoneNumber": 3, "secsInZone": 200}]
        return [{"zoneNumber": 3, "secsInZone": 200}, {"zoneNumber": 4, "secsInZone": 2000}]  # hard
    mock_garmin_client.get_activity_hr_in_timezones.side_effect = zones_for

    result = await app_with_composites.call_tool("get_execution_trend", {"count": 3})
    data = json.loads(result[0][0].text)

    assert data["analysed"] == 3
    assert data["distribution"]["hard"] == 2
    assert data["distribution"]["easy"] == 1
    assert data["monotony"]["distinct_distances"] == 1     # all 7.2km — monotone
    assert data["sessions"][0]["execution"] == "hard"
    assert data["sessions"][2]["execution"] == "easy"


@pytest.mark.asyncio
async def test_health_flags_fuses_signals(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_heart_rates.return_value = {"restingHeartRate": 55, "lastSevenDaysAvgRestingHeartRate": 47}
    mock_garmin_client.get_hrv_data.return_value = {"hrvSummary": {"lastNightAvg": 50, "weeklyAvg": 70, "status": "UNBALANCED", "baseline": {"balancedLow": 55}}}
    mock_garmin_client.get_sleep_data.return_value = {"dailySleepDTO": {"sleepTimeSeconds": 18000, "sleepNeed": {"actual": 510}}}
    mock_garmin_client.get_respiration_data.return_value = {"avgWakingRespirationValue": 15}
    mock_garmin_client.get_spo2_data.return_value = {"averageSpO2": None}
    result = await app_with_composites.call_tool("get_health_flags", {"date": "2026-07-04"})
    data = json.loads(result[0][0].text)
    assert data["severity"] == "red"        # RHR + HRV + sleep debt all off
    assert len(data["flags"]) >= 2
    assert data["signals"]["sleep_debt_min"] == 510 - 300


@pytest.mark.asyncio
async def test_health_flags_green_when_clear(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_heart_rates.return_value = {"restingHeartRate": 46, "lastSevenDaysAvgRestingHeartRate": 47}
    mock_garmin_client.get_hrv_data.return_value = {"hrvSummary": {"lastNightAvg": 72, "weeklyAvg": 70, "baseline": {"balancedLow": 55}}}
    mock_garmin_client.get_sleep_data.return_value = {"dailySleepDTO": {"sleepTimeSeconds": 28800, "sleepNeed": {"actual": 480}}}
    mock_garmin_client.get_respiration_data.return_value = {"avgWakingRespirationValue": 14}
    mock_garmin_client.get_spo2_data.return_value = {"averageSpO2": 96}
    result = await app_with_composites.call_tool("get_health_flags", {"date": "2026-07-04"})
    data = json.loads(result[0][0].text)
    assert data["severity"] == "green"
    assert data["flags"] == []


@pytest.mark.asyncio
async def test_energy_curve_finds_peak_and_trough(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_body_battery_events.return_value = [{"event": {"timezoneOffset": 0}}]
    mock_garmin_client.get_body_battery.return_value = [{
        "bodyBatteryValueDescriptorDTOList": [
            {"bodyBatteryValueDescriptorIndex": 0, "bodyBatteryValueDescriptorKey": "timestamp"},
            {"bodyBatteryValueDescriptorIndex": 2, "bodyBatteryValueDescriptorKey": "bodyBatteryLevel"},
        ],
        "bodyBatteryValuesArray": [
            [3600 * 1000, "M", 80],    # 01:00 UTC — high
            [10 * 3600 * 1000, "M", 30],  # 10:00 UTC — low
        ],
    }]
    mock_garmin_client.get_training_readiness.return_value = [{"score": 70, "level": "MODERATE"}]
    result = await app_with_composites.call_tool("get_energy_curve", {"date": "2026-07-04"})
    data = json.loads(result[0][0].text)
    assert data["peak_window"]["around_hour"] == 1
    assert data["trough_window"]["around_hour"] == 10
    assert data["readiness"]["score"] == 70


@pytest.mark.asyncio
async def test_session_analysis_includes_weather(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activity_hr_in_timezones.return_value = [{"zoneNumber": 4, "secsInZone": 2000}]
    mock_garmin_client.get_activity_splits.return_value = {"lapDTOs": []}
    mock_garmin_client.get_activity_weather.return_value = {"temp": 81, "relativeHumidity": 28}
    result = await app_with_composites.call_tool("get_session_analysis", {"date": "2026-07-03"})
    data = json.loads(result[0][0].text)
    assert data["weather"]["temp_c"] == 27
    assert "warm" in data["weather"]["heat_note"]


# --- running form / dynamics fixtures (shapes recorded from live payloads) ---

RUN_ACTIVITY_WRAPPER = {  # get_activity() shape: numbers nested under summaryDTO
    "activityId": 42,
    "activityName": "City of Westminster Run",
    "activityTypeDTO": {"typeKey": "running"},
    "eventTypeDTO": {"typeKey": "uncategorized"},
    "summaryDTO": {
        "startTimeLocal": "2026-08-23T18:24:21.0",
        "distance": 7261.37, "duration": 2541.242, "averageHR": 143.0,
        "activityTrainingLoad": 95.47, "trainingEffect": 3.4000000953674316,
        "anaerobicTrainingEffect": 0.0,
        "averageRunCadence": 166.953125, "maxRunCadence": 176.0,
        "strideLength": 102.22999877929688,        # centimeters
        "verticalOscillation": 8.790000152587892,  # centimeters
        "verticalRatio": 8.609999656677246,
        "groundContactTime": 272.79998779296875,
        "averagePower": 319.0, "normalizedPower": 320.0,
        "directWorkoutFeel": 100, "directWorkoutRpe": 20,  # RPE stored x10
        "directWorkoutComplianceScore": 86,
        "beginPotentialStamina": 100.0, "endPotentialStamina": 59.0,
        "minAvailableStamina": 59.0, "differenceBodyBattery": -10,
        "waterEstimated": 705.0,
        "elevationGain": 35.0, "elevationLoss": 32.0,
        "avgGradeAdjustedSpeed": 2.8480000495910645,
    },
}

RUN_LIST_ITEM = {  # get_activities*() flat shape — same metrics, different names
    "activityId": 42,
    "activityName": "City of Westminster Run",
    "startTimeLocal": "2026-08-23 18:24:21",
    "activityType": {"typeKey": "running"},
    "distance": 7261.37, "duration": 2541.242, "averageHR": 143.0,
    "aerobicTrainingEffect": 3.4,
    "averageRunningCadenceInStepsPerMinute": 166.953125,
    "maxRunningCadenceInStepsPerMinute": 176.0,
    "avgStrideLength": 102.22999877929688,
    "avgVerticalOscillation": 8.790000152587892,
    "avgVerticalRatio": 8.609999656677246,
    "avgGroundContactTime": 272.79998779296875,
    "avgPower": 319.0, "normPower": 320.0,
    "powerTimeInZone_1": 916.918, "powerTimeInZone_2": 1409.9,
    "powerTimeInZone_3": 160.949, "powerTimeInZone_4": 24.049,
    "powerTimeInZone_5": 4.0,
}

LAP_SPLITS = {"lapDTOs": [
    {"lapIndex": 1, "distance": 1000.0, "duration": 340.0, "averageHR": 120.0,
     "averageRunCadence": 162.1, "strideLength": 104.2},
    {"lapIndex": 2, "distance": 1000.0, "duration": 345.0, "averageHR": 140.0,
     "averageRunCadence": 167.3, "strideLength": 101.0},
    {"lapIndex": 3, "distance": 1000.0, "duration": 372.0, "averageHR": 152.0,
     "averageRunCadence": 168.8, "strideLength": 98.0},
    {"lapIndex": 4, "distance": 200.0, "duration": 76.0, "averageHR": 155.0,
     "averageRunCadence": 163.2, "strideLength": 96.0},
]}

POWER_ZONES = [
    {"zoneNumber": 1, "secsInZone": 916.918, "zoneLowBoundary": 254},
    {"zoneNumber": 2, "secsInZone": 1409.9, "zoneLowBoundary": 314},
    {"zoneNumber": 3, "secsInZone": 160.949, "zoneLowBoundary": 353},
    {"zoneNumber": 4, "secsInZone": 24.049, "zoneLowBoundary": 392},
    {"zoneNumber": 5, "secsInZone": 4.0, "zoneLowBoundary": 450},
]

DETAILS = {
    "metricDescriptors": [
        {"metricsIndex": 0, "key": "directTimestamp"},
        {"metricsIndex": 1, "key": "directPerformanceCondition"},
    ],
    "activityDetailMetrics": [
        {"metrics": [0, None]}, {"metrics": [1, 0.0]}, {"metrics": [2, 4.0]},
        {"metrics": [3, 2.0]}, {"metrics": [4, 1.0]},
    ],
}

EXERCISE_SETS = {"exerciseSets": [
    {"setType": "ACTIVE", "repetitionCount": 10, "weight": 20000.0, "duration": 40.0,
     "exercises": [{"category": "CURL", "name": "BICEPS_CURL", "probability": 99.6},
                   {"category": "UNKNOWN", "name": None, "probability": 0.4}]},
    {"setType": "REST", "repetitionCount": None, "weight": None, "duration": 60.0,
     "exercises": []},
    {"setType": "ACTIVE", "repetitionCount": 12, "weight": 0.0, "duration": 35.0,
     "exercises": [{"category": "UNKNOWN", "name": None, "probability": 99.0}]},
]}


@pytest.mark.asyncio
async def test_running_dynamics_by_id_summary_dto_shape(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activity.return_value = RUN_ACTIVITY_WRAPPER
    mock_garmin_client.get_activity_splits.return_value = LAP_SPLITS
    mock_garmin_client.get_activity_details.return_value = DETAILS
    result = await app_with_composites.call_tool("get_running_dynamics", {"activity_id": 42})
    data = json.loads(result[0][0].text)

    assert data["session"]["distance_km"] == 7.26   # summaryDTO flattened
    assert data["session"]["aerobic_te"] == 3.4     # trainingEffect alias, rounded
    f = data["form"]
    assert f["cadence_spm"] == 167.0
    assert f["stride_length_m"] == 1.02             # cm -> m
    assert f["vertical_oscillation_cm"] == 8.8
    assert f["ground_contact_time_ms"] == 272.8
    assert f["normalized_power_w"] == 320.0
    assert f["gct_balance_pct"] is None             # wrist dynamics: no L/R
    assert "chest strap" in f["gct_balance_note"]
    assert data["effort"] == {"rpe_10": 2.0, "feel_pct": 100, "compliance_score": 86}
    assert data["cost"]["stamina_end_pct"] == 59.0
    assert data["cost"]["sweat_loss_ml"] == 705
    assert data["terrain"]["grade_adjusted_pace_s_per_km"] == 351
    assert data["performance_condition"] == {
        "start": 0.0, "end": 1.0, "min": 0.0, "max": 4.0, "delta": 1.0}
    assert [r["lap"] for r in data["by_lap"]] == [1, 2, 3, 4]
    assert data["by_lap"][0]["stride_length_m"] == 1.04
    assert data["fatigue"]["first_third"]["cadence_spm"] == 162.1
    assert data["fatigue"]["last_third"]["cadence_spm"] == 163.2
    assert data["fatigue"]["stride_drift_pct"] == -7.7


@pytest.mark.asyncio
async def test_running_dynamics_list_shape_refetches_for_effort(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [RUN_LIST_ITEM]
    mock_garmin_client.get_activity.return_value = RUN_ACTIVITY_WRAPPER  # the re-fetch
    mock_garmin_client.get_activity_splits.return_value = LAP_SPLITS
    mock_garmin_client.get_activity_details.return_value = DETAILS
    result = await app_with_composites.call_tool("get_running_dynamics", {"date": "2026-08-23"})
    data = json.loads(result[0][0].text)

    assert data["form"]["cadence_spm"] == 167.0     # list-endpoint field names
    assert data["form"]["normalized_power_w"] == 320.0
    assert data["effort"]["rpe_10"] == 2.0          # via the summaryDTO re-fetch
    mock_garmin_client.get_activity.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_running_dynamics_degrades_to_null(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activity.return_value = {
        "activityId": 7, "activityName": "Pool Swim",
        "summaryDTO": {"distance": 1000.0, "duration": 1800.0}}
    mock_garmin_client.get_activity_splits.side_effect = RuntimeError("404")
    mock_garmin_client.get_activity_details.return_value = {}
    result = await app_with_composites.call_tool("get_running_dynamics", {"activity_id": 7})
    data = json.loads(result[0][0].text)

    for key in ("cadence_spm", "stride_length_m", "vertical_oscillation_cm",
                "vertical_ratio_pct", "ground_contact_time_ms", "gct_balance_pct",
                "avg_power_w", "normalized_power_w"):
        assert data["form"][key] is None
    assert data["effort"] is None and data["cost"] is None
    assert data["performance_condition"] is None
    assert "by_lap" not in data
    assert data["errors"]["laps"] == "404"


@pytest.mark.asyncio
async def test_session_analysis_zone_boundaries_power_and_effort(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [RUN_LIST_ITEM]
    mock_garmin_client.get_activity.return_value = RUN_ACTIVITY_WRAPPER
    mock_garmin_client.get_activity_hr_in_timezones.return_value = [
        {"zoneNumber": 1, "secsInZone": 310.8, "zoneLowBoundary": 99},
        {"zoneNumber": 2, "secsInZone": 208.0, "zoneLowBoundary": 118},
        {"zoneNumber": 3, "secsInZone": 1893.4, "zoneLowBoundary": 139},
    ]
    mock_garmin_client.get_activity_power_in_timezones.return_value = POWER_ZONES
    mock_garmin_client.get_activity_splits.return_value = LAP_SPLITS
    result = await app_with_composites.call_tool("get_session_analysis", {"date": "2026-08-23"})
    data = json.loads(result[0][0].text)

    assert data["hr_zones"]["by_zone"]["z3"]["low_bpm"] == 139   # boundaries visible
    assert data["power_zones"]["by_zone"]["z2"]["low_w"] == 314  # watts, not bpm
    assert data["power_zones"]["easy_share_pct"] == 92           # vs 21% by HR
    assert data["form"] == {"cadence_spm": 167.0, "stride_length_m": 1.02}
    assert data["effort"]["rpe_10"] == 2.0
    assert data["cost"]["body_battery_drain"] == -10
    assert data["pacing"]["shape"] == "faded"


@pytest.mark.asyncio
async def test_pacing_is_distance_weighted(app_with_composites, mock_garmin_client):
    # 3 even km + a short slow tail lap: unweighted halves would call this a
    # fade; distance-weighted lap pacing reads it as even.
    mock_garmin_client.get_activities_by_date.return_value = [RUN_LIST_ITEM]
    mock_garmin_client.get_activity_splits.return_value = {"lapDTOs": [
        {"distance": 1000.0, "duration": 350.0},
        {"distance": 1000.0, "duration": 350.0},
        {"distance": 1000.0, "duration": 352.0},
        {"distance": 120.0, "duration": 50.0},   # 417 s/km, but only 120 m
    ]}
    result = await app_with_composites.call_tool("get_session_analysis", {"date": "2026-08-23"})
    data = json.loads(result[0][0].text)
    assert data["pacing"]["shape"] == "even"
    assert data["pacing"]["drift_pct"] < 3


@pytest.mark.asyncio
async def test_session_analysis_strength_sets(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [{
        "activityId": 9, "activityName": "Strength",
        "startTimeLocal": "2026-08-21 20:02:00",
        "activityType": {"typeKey": "strength_training"}, "duration": 1380.0,
    }]
    mock_garmin_client.get_activity_exercise_sets.return_value = EXERCISE_SETS
    result = await app_with_composites.call_tool("get_session_analysis", {"date": "2026-08-21"})
    data = json.loads(result[0][0].text)

    s = data["strength_sets"]
    assert s["total_sets"] == 2                     # ACTIVE only, REST excluded
    assert s["total_reps"] == 22
    assert s["by_exercise"]["BICEPS_CURL"] == {"sets": 1, "reps": 10, "max_weight_kg": 20.0}
    assert s["by_exercise"]["unclassified"] == {"sets": 1, "reps": 12}  # bodyweight
    mock_garmin_client.get_activity_exercise_sets.assert_called_once_with(9)


@pytest.mark.asyncio
async def test_execution_trend_rows_carry_form_power_and_rpe(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_activities.return_value = [RUN_LIST_ITEM]
    mock_garmin_client.get_activity.return_value = RUN_ACTIVITY_WRAPPER
    mock_garmin_client.get_activity_hr_in_timezones.return_value = [
        {"zoneNumber": 2, "secsInZone": 600}, {"zoneNumber": 3, "secsInZone": 1400}]
    result = await app_with_composites.call_tool("get_execution_trend", {"count": 1})
    data = json.loads(result[0][0].text)

    row = data["sessions"][0]
    assert row["cadence_spm"] == 167.0
    assert row["stride_length_m"] == 1.02
    assert row["power_easy_share_pct"] == 92        # from the flat list fields
    assert row["rpe_10"] == 2.0                     # perception vs objective
    assert row["easy_share_pct"] == 30


@pytest.mark.asyncio
async def test_health_flags_unknown_when_unsynced(app_with_composites, mock_garmin_client):
    mock_garmin_client.get_heart_rates.return_value = {}
    mock_garmin_client.get_hrv_data.return_value = {}
    mock_garmin_client.get_sleep_data.return_value = {}
    mock_garmin_client.get_respiration_data.return_value = {}
    mock_garmin_client.get_spo2_data.return_value = {}
    result = await app_with_composites.call_tool("get_health_flags", {"date": "2026-07-04"})
    data = json.loads(result[0][0].text)
    assert data["severity"] == "unknown"
    assert "synced" in data["note"]
