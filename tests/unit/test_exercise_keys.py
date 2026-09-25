"""Exercise keys Garmin would blank on upload are rejected, with suggestions.

Garmin accepts any category/exerciseName on upload and silently blanks a pair
outside its catalogs — 38 of 83 steps across the coach's cards had lost their
name that way, and with it the exercise's name, rep counting and video on the
watch. Garmin's own workouts: 0 of 242.
"""

import pytest

from garmin_mcp import workouts
from garmin_mcp.workouts import _curate_workout_step, _validate_exercise_keys

CATALOG = {
    "PUSH_UP": {"PUSH_UP", "PIKE_PUSH_UP", "INCLINE_PUSH_UP"},
    "LATERAL_RAISE": {"DUMBBELL_LATERAL_RAISE"},
    "SHOULDER_STABILITY": {"FLOOR_Y_RAISE", "FLOOR_T_RAISE"},
    "WARM_UP": {"ARM_CIRCLES", "STRETCH_CAT_COW"},
    "CORE": {"CAT_COW"},
}


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    monkeypatch.setattr(workouts, "_EXERCISE_CATALOG", CATALOG)


def step(step_type, category=None, name=None):
    s = {"type": "ExecutableStepDTO", "stepType": {"stepTypeKey": step_type}}
    if category:
        s["category"] = category
    if name:
        s["exerciseName"] = name
    return s


def workout(*steps, sport="strength_training"):
    return {
        "sportType": {"sportTypeKey": sport},
        "workoutSegments": [{"sportType": {"sportTypeKey": sport}, "workoutSteps": [
            {"type": "RepeatGroupDTO", "workoutSteps": list(steps)}
        ]}],
    }


def test_valid_keys_and_keyless_rests_pass():
    _validate_exercise_keys(workout(
        step("warmup", "WARM_UP", "ARM_CIRCLES"),
        step("rest"),
        step("interval", "PUSH_UP", "PIKE_PUSH_UP"),
        step("interval", "CORE", "CAT_COW"),   # a Mobility key inside strength
    ))


def test_a_name_filed_under_the_wrong_category_names_its_home():
    with pytest.raises(ValueError) as e:
        _validate_exercise_keys(workout(step("interval", "SHOULDER_STABILITY", "DUMBBELL_LATERAL_RAISE")))
    assert "exists under LATERAL_RAISE" in str(e.value)


def test_an_invented_name_gets_the_nearest_real_ones():
    with pytest.raises(ValueError) as e:
        _validate_exercise_keys(workout(step("interval", "PUSH_UP", "PIKE_PUSHUP")))
    assert "nearest in PUSH_UP: PIKE_PUSH_UP" in str(e.value)
    assert "workoutSegments[0].workoutSteps[0].workoutSteps[0]" in str(e.value)


def test_a_keyless_strength_step_is_rejected_but_a_keyless_run_is_not():
    with pytest.raises(ValueError, match="no category/exerciseName"):
        _validate_exercise_keys(workout(step("interval")))
    _validate_exercise_keys(workout(step("interval"), sport="running"))


def test_unreachable_catalogs_skip_validation(monkeypatch):
    monkeypatch.setattr(workouts, "_EXERCISE_CATALOG", None)
    monkeypatch.setattr(workouts, "_fetch_exercise_catalog", lambda: None)
    _validate_exercise_keys(workout(step("interval", "MADE_UP", "NOTHING")))


def test_read_back_shows_garmins_load_and_side_fields():
    curated = _curate_workout_step({
        "stepOrder": 4, "stepType": {"stepTypeKey": "interval"},
        "category": "PLANK", "exerciseName": "SIDE_PLANK",
        "relativeWeight": "BODY_WEIGHT", "exerciseSide": "LEFT",
    })
    assert curated["relative_weight"] == "BODY_WEIGHT"
    assert curated["exercise_side"] == "LEFT"


@pytest.mark.asyncio
async def test_upload_refuses_before_calling_garmin(mock_garmin_client):
    from mcp.server.fastmcp import FastMCP
    workouts.configure(mock_garmin_client)
    app = FastMCP("t")
    workouts.register_tools(app)
    await app.call_tool("upload_workout", {"workout_data": workout(step("interval", "PUSH_UP", "PIKE_PUSHUP"))})
    mock_garmin_client.upload_workout.assert_not_called()
