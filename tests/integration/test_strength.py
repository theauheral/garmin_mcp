"""Tests for get_strength_progression: ladders, clean judging, and pattern
set counts, driven through the MCP tool with a mocked Garmin client. Shapes
mirror real payloads (activity 24331499906 / card 1686158824, 2026-09-12)."""

import json

import pytest
from mcp.server.fastmcp import FastMCP

from garmin_mcp import strength


def _step(order, step_type, category=None, name=None, reps=None, secs=None, desc=""):
    step = {
        "type": "ExecutableStepDTO",
        "stepOrder": order,
        "stepType": {"stepTypeKey": step_type},
        "description": desc,
        "category": category,
        "exerciseName": name,
    }
    if reps is not None:
        step.update(endCondition={"conditionTypeKey": "reps"}, endConditionValue=float(reps))
    else:
        step.update(endCondition={"conditionTypeKey": "time"}, endConditionValue=float(secs or 30))
    return step


def _repeat(order, n, *children):
    return {"type": "RepeatGroupDTO", "stepOrder": order, "numberOfIterations": n,
            "workoutSteps": list(children)}


# warm-up push-ups, [TRX row x3], transition, [floor press x2]
CARD = {"workoutSegments": [{"workoutSteps": [
    _step(1, "warmup", "PUSH_UP", "PUSH_UP", secs=45, desc="10 EASY PUSH-UPS"),
    _repeat(2, 3,
            _step(3, "interval", "ROW", "INVERTED_ROW", reps=10,
                  desc="TRX ROW 10. ~50s per set. Past 65s = angle too hard"),
            _step(4, "rest", secs=60)),
    _step(5, "rest", secs=60, desc="TRANSITION"),
    _repeat(6, 2,
            _step(7, "interval", "BENCH_PRESS", "DUMBBELL_FLOOR_PRESS", reps=10),
            _step(8, "rest", secs=60)),
]}]}

# FIT order: warm-up 0, row 1, rest 2, repeat 3, transition 4, press 5, rest 6
ROW_IDX, PRESS_IDX = 1, 5


def _set(idx, category, name, reps, secs, grams=0.0, set_type="ACTIVE"):
    return {
        "exercises": [{"category": category, "name": name, "probability": 99.6}],
        "repetitionCount": reps, "duration": secs, "weight": grams,
        "setType": set_type, "wktStepIndex": idx,
    }


def _activity(aid, date, workout_id=None, type_key="strength_training"):
    return {"activityId": aid, "startTimeLocal": f"{date} 10:00:00",
            "activityName": "Strength", "workoutId": workout_id,
            "activityType": {"typeKey": type_key}}


@pytest.fixture
def app(mock_garmin_client):
    strength.configure(mock_garmin_client)
    return strength.register_tools(FastMCP("Test Strength"))


async def _call(app, **args):
    result = await app.call_tool("get_strength_progression", args)
    return json.loads(result[0][0].text)


def test_fit_step_index_puts_repeat_after_its_children():
    index = strength._fit_step_index(CARD)
    assert index[0]["exerciseName"] == "PUSH_UP"
    assert index[ROW_IDX]["exerciseName"] == "INVERTED_ROW"
    assert index[3]["type"] == "RepeatGroupDTO"
    assert index[4]["description"] == "TRANSITION"
    assert index[PRESS_IDX]["exerciseName"] == "DUMBBELL_FLOOR_PRESS"


def test_every_ladder_key_is_on_exactly_one_rung():
    seen = [k for rungs in strength.LADDERS.values() for rung in rungs for k in rung]
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("category,name,pattern", [
    ("ROW", "INVERTED_ROW", "pull"),
    ("SUSPENSION", "ROW", "pull"),
    ("SUSPENSION", "PUSH_UP", "push"),
    ("SUSPENSION", "GLUTE_BRIDGE", "hinge"),
    ("CORE", "CAT_COW", "accessory"),
    ("CURL", "DUMBBELL_BICEPS_CURL", "accessory"),
    ("SHOULDER_STABILITY", None, "accessory"),
])
def test_pattern(category, name, pattern):
    assert strength._pattern(category, name) == pattern


async def test_carded_sessions_set_the_rung_and_judge_clean(app, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [
        _activity(2, "2026-09-12", workout_id=77),
        _activity(1, "2026-09-05", workout_id=77),
    ]
    mock_garmin_client.get_workout_by_id.return_value = CARD
    sets = {
        # older: all reps, but set 3 ran past the 65 s ceiling -> not clean
        1: [_set(0, "PUSH_UP", "PUSH_UP", 10, 45),
            _set(ROW_IDX, "ROW", "INVERTED_ROW", 10, 120),
            _set(2, "ROW", None, None, 60, set_type="REST"),
            _set(ROW_IDX, "ROW", "INVERTED_ROW", 10, 60),
            _set(ROW_IDX, "ROW", "INVERTED_ROW", 10, 70)],
        # newer: slow set 1 is the transition, sets 2-3 inside the ceiling -> clean
        2: [_set(ROW_IDX, "ROW", "INVERTED_ROW", 10, 110),
            _set(ROW_IDX, "ROW", "INVERTED_ROW", 10, 55),
            _set(ROW_IDX, "ROW", "INVERTED_ROW", 10, 58),
            _set(PRESS_IDX, "BENCH_PRESS", "DUMBBELL_FLOOR_PRESS", 10, 50, 14500.0),
            _set(PRESS_IDX, "BENCH_PRESS", "DUMBBELL_FLOOR_PRESS", 8, 52, 14500.0)],
    }
    mock_garmin_client.get_activity_exercise_sets.side_effect = (
        lambda aid: {"exerciseSets": sets[aid]})

    data = await _call(app, end_date="2026-09-14", weeks=2)

    row = data["exercises"]["ROW/INVERTED_ROW"]
    assert row["ladder"] == "row" and row["rung"] == "2 of 4"
    newest, older = row["history"]
    assert newest["date"] == "2026-09-12"
    assert newest["prescribed_reps"] == 10 and newest["ceiling_s"] == 65
    assert newest["clean"] is True and newest["median_s"] == 56
    assert older["over_ceiling"] == 1 and older["clean"] is False
    assert row["clean_streak"] == 1

    press = data["exercises"]["BENCH_PRESS/DUMBBELL_FLOOR_PRESS"]["history"][0]
    assert press["top_kg"] == 14.5 and press["hit_reps"] is False

    # the warm-up push-ups are skipped: no push_up ladder, no push sets from them
    assert "PUSH_UP/PUSH_UP" not in data["exercises"]
    assert set(data["ladders"]) == {"row"}
    ladder = data["ladders"]["row"]
    assert ladder["rung"] == 2 and ladder["of"] == 4 and ladder["clean_streak"] == 1
    assert "ROW/ELEVATED_FEET_INVERTED_ROW" in ladder["next"]

    assert data["patterns"]["pull"]["sets"] == 6
    assert data["patterns"]["push"]["sets"] == 2
    assert data["patterns"]["squat"] == {"sets": 0, "note": "no sets in window"}
    assert mock_garmin_client.get_workout_by_id.call_count == 1  # cached per card


async def test_uncarded_session_counts_sets_but_never_moves_a_rung(app, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [
        _activity(3, "2026-09-17"),
        _activity(4, "2026-09-17", type_key="running"),
    ]
    mock_garmin_client.get_activity_exercise_sets.return_value = {"exerciseSets": [
        _set(0, "PULL_UP", "PULL_UP", 10, 40),
        _set(1, "LUNGE", "LUNGE", 8, 40),
    ]}

    data = await _call(app, end_date="2026-09-20", weeks=1)

    assert data["ladders"] == {}
    assert data["sessions"] == [{
        "date": "2026-09-17", "activity_id": 3, "name": "Strength",
        "card_status": "none - auto-classified sets", "working_sets": 2,
    }]
    assert data["patterns"]["pull"]["sets"] == 1
    assert data["patterns"]["squat"]["sets"] == 1
    assert "clean" not in data["exercises"]["PULL_UP/PULL_UP"]["history"][0]
    mock_garmin_client.get_activity_exercise_sets.assert_called_once_with(3)


async def test_plan_workout_moves_a_rung_without_a_verdict(app, mock_garmin_client):
    # A Run Coach plan circuit: no workoutId on the list entry, the plan id
    # only in the activity's metadata; 40 s timed sets whose rep counts are noise.
    mock_garmin_client.get_activities_by_date.return_value = [_activity(7, "2026-09-17")]
    mock_garmin_client.get_activity.return_value = {
        "metadataDTO": {"trainingPlanId": 47608131, "associatedWorkoutId": None}}
    mock_garmin_client.get_activity_exercise_sets.return_value = {"exerciseSets": [
        _set(0, "PUSH_UP", "PUSH_UP", 4, 40),
        _set(2, "PUSH_UP", "PUSH_UP", 0, 40),
    ]}

    data = await _call(app, end_date="2026-09-20", weeks=1)

    session = data["sessions"][0]
    assert session["card_status"] == "garmin plan workout" and session["plan"] == 47608131
    assert data["ladders"]["push_up"]["on"] == ["PUSH_UP/PUSH_UP"]
    assert "clean_streak" not in data["ladders"]["push_up"]
    assert data["exercises"]["PUSH_UP/PUSH_UP"]["history"][0]["clean"] is None
    mock_garmin_client.get_workout_by_id.assert_not_called()


async def test_deleted_card_keeps_sets_with_unknown_verdict(app, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [
        _activity(5, "2026-09-01", workout_id=99)]
    mock_garmin_client.get_workout_by_id.side_effect = Exception("404 NotFoundException")
    mock_garmin_client.get_activity_exercise_sets.return_value = {"exerciseSets": [
        _set(0, "CORE", "CAT_COW", 15, 60),
        _set(3, "PUSH_UP", "SUSPENDED_PUSH_UP", 10, 80),
    ]}

    data = await _call(app, end_date="2026-09-02", weeks=1)

    assert data["sessions"][0]["card_status"] == "deleted"
    ladder = data["ladders"]["push_up"]
    assert ladder["on"] == ["PUSH_UP/SUSPENDED_PUSH_UP"]
    assert "clean_streak" not in ladder  # nothing could be judged
    assert data["exercises"]["PUSH_UP/SUSPENDED_PUSH_UP"]["history"][0]["clean"] is None
    assert data["patterns"]["core"]["sets"] == 0  # cat-cow is mobility
    assert data["patterns"]["accessory_sets"] == 1


async def test_errors_are_reported_not_raised(app, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = [
        _activity(6, "2026-09-10", workout_id=None)]
    mock_garmin_client.get_activity_exercise_sets.side_effect = Exception("boom")

    data = await _call(app, end_date="2026-09-10", weeks=1)

    assert data["errors"] == {"sets:6": "boom"}
    assert data["sessions"][0]["activity_id"] == 6

    mock_garmin_client.get_activities_by_date.side_effect = Exception("down")
    data = await _call(app, end_date="2026-09-10")
    assert data["errors"] == {"activities": "down"}


async def test_window_is_clamped(app, mock_garmin_client):
    mock_garmin_client.get_activities_by_date.return_value = []
    data = await _call(app, end_date="2026-09-25", weeks=99)
    assert data["window"] == {"start": "2026-03-28", "end": "2026-09-25", "weeks": 26}
    assert data["note"] == "no strength_training activities in window"
