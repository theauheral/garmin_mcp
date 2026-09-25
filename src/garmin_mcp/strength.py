"""Strength progression across sessions: which rung of each bodyweight ladder
the athlete is on, whether recent sessions at that rung were clean, and hard
working sets per movement pattern.

Why this exists: bodyweight work scores zero tonnage, and a TRX move gets
harder by changing body angle, which no Garmin field records. So "3 x 10
inverted row" reads the same on the day it was easy and the day it was a
grind. A ladder makes progression legible: each rung is a *different* Garmin
catalog exercise, so stepping up changes the recorded exerciseName and shows
in get_activity_exercise_sets. Within a rung, the signals are reps against
the card's prescription and set duration against the card's ceiling.

Kept out of composites.py on purpose: this module only reads activities,
exercise sets and workouts, and shares no helpers with the endurance
composites.
"""

import datetime
import json
import re
from statistics import median

# The garmin_client will be set by the main file
garmin_client = None


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client


# Default ladders, easiest rung first. Each rung is a set of Garmin catalog
# keys ("CATEGORY/EXERCISE_NAME", https://connect.garmin.com/web-data/
# exercises/Exercises.json) judged to be the same difficulty. These are
# generic calisthenics progressions, not a prescription: the athlete's own
# ladder, standards and target rung belong to the coach's plan. Weighted
# variants of a rung's move are deliberately absent — they progress by load,
# which the per-exercise view already shows.
LADDERS = {
    "row": [
        {"ROW/MODIFIED_INVERTED_ROW"},
        {"ROW/INVERTED_ROW", "ROW/TRX_INVERTED_ROW", "ROW/SUSPENDED_INVERTED_ROW",
         "ROW/RING_ROW", "SUSPENSION/ROW"},
        {"ROW/ELEVATED_FEET_INVERTED_ROW", "ROW/ELEVATED_RING_ROW"},
        {"ROW/SINGLE_ARM_INVERTED_ROW"},
    ],
    "pull_up": [
        {"PULL_UP/BAND_ASSISTED_PULL_UP", "PULL_UP/BAND_ASSISTED_CHIN_UP",
         "PULL_UP/JUMPING_PULL_UPS", "PULL_UP/SUSPENDED_CHIN_UP", "SUSPENSION/PULL_UP"},
        {"PULL_UP/CHIN_UP", "PULL_UP/NEUTRAL_GRIP_CHIN_UP", "PULL_UP/CLOSE_GRIP_CHIN_UP"},
        {"PULL_UP/PULL_UP", "PULL_UP/NEUTRAL_GRIP_PULL_UP", "PULL_UP/WIDE_GRIP_PULL_UP",
         "PULL_UP/MIXED_GRIP_PULL_UP"},
        {"PULL_UP/WEIGHTED_PULL_UP", "PULL_UP/WEIGHTED_CHIN_UP", "PULL_UP/L_PULL_UP"},
        {"LATERAL_RAISE/MUSCLE_UP", "LATERAL_RAISE/BAR_MUSCLE_UP", "LATERAL_RAISE/RING_MUSCLE_UP"},
    ],
    "push_up": [
        {"PUSH_UP/KNEELING_PUSH_UP", "PUSH_UP/INCLINE_PUSH_UP"},
        {"PUSH_UP/PUSH_UP", "PUSH_UP/SUSPENDED_PUSH_UP", "SUSPENSION/PUSH_UP"},
        {"PUSH_UP/DECLINE_PUSH_UP", "PUSH_UP/DIAMOND_PUSH_UP", "PUSH_UP/RING_PUSH_UP"},
        {"PUSH_UP/WEIGHTED_PUSH_UP"},
        {"PUSH_UP/ONE_ARM_PUSH_UP"},
    ],
    "dip": [
        {"TRICEPS_EXTENSION/BENCH_DIP", "TRICEPS_EXTENSION/TABLETOP_DIP",
         "TRICEPS_EXTENSION/INCLINE_DIP"},
        {"TRICEPS_EXTENSION/BODY_WEIGHT_DIP"},
        {"TRICEPS_EXTENSION/SUSPENDED_DIP", "SUSPENSION/DIP", "LATERAL_RAISE/RING_DIP"},
        {"TRICEPS_EXTENSION/WEIGHTED_DIP", "LATERAL_RAISE/WEIGHTED_RING_DIP"},
    ],
    "single_leg_squat": [
        {"SQUAT/AIR_SQUAT", "SUSPENSION/SQUAT"},
        {"LUNGE/LUNGE", "SUSPENSION/LUNGE"},
        {"SUSPENSION/SINGLE_LEG_SQUAT", "SQUAT/PARTIAL_SINGLE_LEG_SQUAT",
         "SQUAT/SINGLE_LEG_BENCH_SQUAT"},
        {"SQUAT/PISTOL_SQUAT", "SQUAT/ELEVATED_SINGLE_LEG_SQUAT"},
    ],
    "hinge": [
        {"HIP_RAISE/HIP_RAISE", "SUSPENSION/GLUTE_BRIDGE"},
        {"HIP_RAISE/SINGLE_LEG_HIP_RAISE", "HIP_RAISE/MARCHING_HIP_RAISE"},
        {"HIP_RAISE/SINGLE_LEG_HIP_RAISE_WITH_FOOT_ON_BENCH", "SUSPENSION/HAMSTRING_CURL"},
    ],
    "core": [
        {"PLANK/KNEELING_PLANK"},
        {"PLANK/PLANK", "CRUNCH/HOLLOW_ROCK"},
        {"CORE/KNEELING_AB_WHEEL", "SUSPENSION/ROLLOUT", "SUSPENSION/PIKE",
         "CORE/TRX_SUSPENDED_JACKKNIFE"},
        {"LEG_RAISE/HANGING_KNEE_RAISE", "CORE/L_SIT"},
        {"LEG_RAISE/HANGING_LEG_RAISE"},
        {"CRUNCH/TOES_TO_BAR"},
    ],
}

_RUNG_OF = {
    key: (ladder, i)
    for ladder, rungs in LADDERS.items()
    for i, rung in enumerate(rungs)
    for key in rung
}

# Movement pattern per catalog category — the unit the coach counts hard sets
# in. Everything unlisted (curls, raises, calves, stability, warm-ups) is an
# accessory: logged, but not a pattern set.
_CATEGORY_PATTERN = {
    "ROW": "pull", "PULL_UP": "pull",
    "PUSH_UP": "push", "BENCH_PRESS": "push", "SHOULDER_PRESS": "push",
    "TRICEPS_EXTENSION": "push", "FLYE": "push",
    "SQUAT": "squat", "LUNGE": "squat", "PLYO": "squat",
    "DEADLIFT": "hinge", "HIP_RAISE": "hinge", "HIP_SWING": "hinge",
    "HYPEREXTENSION": "hinge", "LEG_CURL": "hinge",
    "CARRY": "carry", "SANDBAG": "carry",
    "CORE": "core", "PLANK": "core", "CRUNCH": "core", "LEG_RAISE": "core",
    "SIT_UP": "core", "CHOP": "core",
}

# SUSPENSION's exercise names are generic (ROW, PUSH_UP, ...), so its pattern
# comes from the name, not the category.
_SUSPENSION_PATTERN = {
    "ROW": "pull", "PULL_UP": "pull", "FACE_PULL": "pull", "Y_FLY": "pull",
    "POWER_PULL": "pull", "LAT_PULLOVER": "pull",
    "PUSH_UP": "push", "CHEST_PRESS": "push", "CHEST_FLY": "push",
    "TRICEP_PRESS": "push", "DIP": "push",
    "SQUAT": "squat", "LUNGE": "squat", "SIDE_LUNGE": "squat",
    "SINGLE_LEG_SQUAT": "squat", "SQUAT_JUMP": "squat",
    "GLUTE_BRIDGE": "hinge", "HAMSTRING_CURL": "hinge", "HIP_DROP": "hinge",
    "SINGLE_LEG_DEADLIFT": "hinge",
    "CRUNCH": "core", "PIKE": "core", "PLANK": "core", "SIDE_PLANK": "core",
    "REVERSE_PLANK": "core", "ROLLOUT": "core", "SIT_UP": "core",
    "KNEE_TO_CHEST": "core", "MOUNTAIN_CLIMBER": "core",
    "REVERSE_MOUNTAIN_CLIMBER": "core", "PENDULUM": "core",
}

_PATTERNS = ("pull", "push", "squat", "hinge", "carry", "core")

# Mobility drills filed under a working category. On a card they sit in the
# warm-up and are skipped by step type; this catches them when the card has
# since been deleted and step types are unknown.
_MOBILITY = {"CORE/CAT_COW", "CORE/INCHWORM", "CORE/CRESCENT_CIRCLE"}

# CoS cards state a per-set time ceiling in the step description, e.g.
# "~50s per set. Past 65s = angle too hard".
_CEILING_RE = re.compile(r"\bpast\s+(\d{2,3})\s*s\b", re.IGNORECASE)

_SKIPPED_STEP_TYPES = {"warmup", "cooldown"}


def _pattern(category, name):
    if f"{category}/{name}" in _MOBILITY:
        return "accessory"
    if category == "SUSPENSION":
        return _SUSPENSION_PATTERN.get(name or "", "accessory")
    return _CATEGORY_PATTERN.get(category or "", "accessory")


def _fit_step_index(workout):
    """Map a set's wktStepIndex to its workout step.

    wktStepIndex counts steps in FIT order, where a repeat step comes AFTER
    the steps it repeats (Connect's JSON puts the RepeatGroupDTO first). So a
    card of warm-up, [row, rest] x4, transition records the row sets at index
    1 and the transition at 3 — the repeat node takes index 2.
    """
    flat = []

    def walk(steps):
        for step in sorted(steps or [], key=lambda s: s.get("stepOrder") or 0):
            if step.get("type") == "RepeatGroupDTO":
                walk(step.get("workoutSteps"))
            flat.append(step)

    for segment in (workout or {}).get("workoutSegments") or []:
        walk(segment.get("workoutSteps"))
    return dict(enumerate(flat))


def _prescription(step):
    """(prescribed reps, per-set ceiling seconds) read off a card step."""
    if not step:
        return None, None
    reps = None
    if ((step.get("endCondition") or {}).get("conditionTypeKey")) == "reps":
        value = step.get("endConditionValue")
        reps = int(value) if isinstance(value, (int, float)) else None
    match = _CEILING_RE.search(step.get("description") or "")
    return reps, (int(match.group(1)) if match else None)


def _set_exercise(s):
    """(category, name) of a recorded set — the most probable candidate."""
    cands = [e for e in s.get("exercises") or [] if isinstance(e, dict)]
    if not cands:
        return None, None
    best = max(cands, key=lambda e: e.get("probability") or 0)
    return best.get("category"), best.get("name")


def _kg(grams):
    # Garmin stores exercise-set weight in grams; 0 means bodyweight.
    return round(grams / 1000, 1) if isinstance(grams, (int, float)) and grams > 0 else None


def _judge(entry):
    """Mark one exercise-in-one-session clean / not clean / unknown.

    Clean = every set reached the prescribed reps AND no set after the first
    ran past the card's ceiling. Set 1 is excluded from the time check
    because the transition before it (strap change, dial-in) routinely runs
    into the recorded set. Unknown (None) when the card had no rep target.
    """
    sets = entry["sets"]
    rest = [s["s"] for s in sets[1:] if s.get("s") is not None]
    timed = rest or [s["s"] for s in sets if s.get("s") is not None]
    if timed:
        entry["median_s"] = round(median(timed))
    target = entry.get("prescribed_reps")
    if target is None:
        entry["clean"] = None
        return entry
    entry["hit_reps"] = all((s.get("reps") or 0) >= target for s in sets)
    ceiling = entry.get("ceiling_s")
    if ceiling is not None:
        entry["over_ceiling"] = sum(1 for t in rest if t > ceiling)
    entry["clean"] = entry["hit_reps"] and not entry.get("over_ceiling")
    return entry


def _clean_streak(history):
    """Consecutive clean sessions, newest first; unknowns neither count nor break."""
    streak = 0
    for h in history:
        if h.get("clean") is None:
            continue
        if not h["clean"]:
            break
        streak += 1
    return streak


def _training_plan_id(activity_id):
    """The Garmin training plan an activity was run from, or None.

    Plan workouts (e.g. a Run Coach plan's 40/20 circuits) are addressed by
    UUID and leave the list entry's workoutId empty, so only the activity's
    metadataDTO says the exercise names came from a card. Failures count as
    "no plan": the session is then treated as freestyle, never invented.
    """
    try:
        activity = garmin_client.get_activity(activity_id)
    except Exception:
        return None
    meta = activity.get("metadataDTO") if isinstance(activity, dict) else None
    plan_id = meta.get("trainingPlanId") if isinstance(meta, dict) else None
    return plan_id if isinstance(plan_id, int) else None


def _session_date(activity):
    return (activity.get("startTimeLocal") or "")[:10]


def register_tools(app):
    """Register strength-progression tools with the MCP server app"""

    @app.tool()
    async def get_strength_progression(end_date: str = "", weeks: int = 8) -> str:
        """Strength progression over the last N weeks: the rung the athlete
        is on in each bodyweight ladder (row, pull_up, push_up, dip,
        single_leg_squat, hinge, core), whether recent sessions at each
        exercise were clean, load trend on weighted moves, and hard working
        sets per movement pattern (pull/push/squat/hinge/carry/core). Use for
        a weekly strength review, before progressing or regressing a card,
        and to show "where you are / what unlocks next".

        How to read it:
        - A rung is a distinct Garmin catalog exercise (e.g. INVERTED_ROW ->
          ELEVATED_FEET_INVERTED_ROW), so a rung change is visible in the
          record. Progress WITHIN a rung (TRX angle, tempo) is not — the
          within-rung signals are `hit_reps` against the card's prescription
          and set seconds against the card's "Past NNs" ceiling.
        - `clean` = all sets hit the prescribed reps and no set after the
          first ran past the ceiling (set 1 absorbs the transition). None when
          the card had no rep target or the card was since deleted.
          `clean_streak` counts clean sessions back from the newest.
        - Rungs come only from GUIDED sessions, whose exercise names came from
          a workout rather than the watch's guesswork: a coach card (the
          activity has a workoutId; prescriptions are read from it) or a
          Garmin training-plan workout (`card_status: "garmin plan workout"`;
          the plan's card can't be fetched from the activity, so no
          prescription and `clean: null`). Freestyle sessions have
          auto-classified sets: they count toward pattern sets but never
          move a rung. Warm-up and cool-down steps on a coach card are
          excluded.
        - The ladders are generic defaults; the coach's plan owns the
          athlete's standard for stepping up and the target rung.

        Args:
            end_date: last day of the window, YYYY-MM-DD (default today)
            weeks: window length in weeks (default 8, 1-26; 2-3 API calls
                per strength session)
        """
        weeks = max(1, min(int(weeks or 8), 26))
        end = datetime.date.fromisoformat(end_date) if end_date else datetime.date.today()
        start = end - datetime.timedelta(days=7 * weeks - 1)
        out = {"window": {"start": str(start), "end": str(end), "weeks": weeks}}
        errors = {}

        try:
            activities = garmin_client.get_activities_by_date(str(start), str(end)) or []
        except Exception as e:
            out["errors"] = {"activities": str(e)}
            return json.dumps(out, indent=2)
        activities = sorted(
            (a for a in activities
             if ((a.get("activityType") or {}).get("typeKey")) == "strength_training"),
            key=lambda a: a.get("startTimeLocal") or "",
        )

        cards = {}
        sessions = []
        exercises = {}
        patterns = {p: {"sets": 0, "sessions": set()} for p in _PATTERNS}
        accessory_sets = 0
        ladder_hits = {}  # ladder -> list of (date, rung, key)

        for a in activities:
            aid = a.get("activityId")
            date = _session_date(a)
            wid = a.get("workoutId")
            session = {"date": date, "activity_id": aid, "name": a.get("activityName")}
            steps = {}
            guided = bool(wid)
            if wid:
                session["card"] = wid
                if wid not in cards:
                    try:
                        cards[wid] = _fit_step_index(garmin_client.get_workout_by_id(wid))
                    except Exception:
                        cards[wid] = None  # revised cards are deleted; the sets remain
                steps = cards[wid] or {}
                if cards[wid] is None:
                    session["card_status"] = "deleted"
            else:
                plan_id = _training_plan_id(aid)
                if plan_id:
                    guided = True
                    session["card_status"] = "garmin plan workout"
                    session["plan"] = plan_id
                else:
                    session["card_status"] = "none - auto-classified sets"
            try:
                payload = garmin_client.get_activity_exercise_sets(aid) or {}
            except Exception as e:
                errors[f"sets:{aid}"] = str(e)
                sessions.append(session)
                continue
            raw = payload.get("exerciseSets") if isinstance(payload, dict) else payload

            working = 0
            per_exercise = {}
            for s in raw or []:
                if not isinstance(s, dict) or s.get("setType") != "ACTIVE":
                    continue
                step = steps.get(s.get("wktStepIndex"))
                step_type = ((step or {}).get("stepType") or {}).get("stepTypeKey")
                if step_type in _SKIPPED_STEP_TYPES:
                    continue
                category, name = _set_exercise(s)
                pattern = _pattern(category, name)
                if pattern == "accessory":
                    accessory_sets += 1
                else:
                    patterns[pattern]["sets"] += 1
                    patterns[pattern]["sessions"].add(date)
                working += 1
                if not name:
                    continue
                key = f"{category}/{name}"
                entry = per_exercise.get(key)
                if entry is None:
                    reps, ceiling = _prescription(step) if wid else (None, None)
                    entry = per_exercise[key] = {"date": date, "sets": []}
                    if reps is not None:
                        entry["prescribed_reps"] = reps
                    if ceiling is not None:
                        entry["ceiling_s"] = ceiling
                row = {"reps": s.get("repetitionCount")}
                kg = _kg(s.get("weight"))
                if kg is not None:
                    row["kg"] = kg
                if isinstance(s.get("duration"), (int, float)):
                    row["s"] = round(s["duration"])
                entry["sets"].append(row)

            session["working_sets"] = working
            sessions.append(session)
            for key, entry in per_exercise.items():
                loads = [r["kg"] for r in entry["sets"] if "kg" in r]
                if loads:
                    entry["top_kg"] = max(loads)
                ex = exercises.setdefault(key, {"history": []})
                ex["history"].append(_judge(entry) if guided else entry)
                if guided and key in _RUNG_OF:
                    ladder, rung = _RUNG_OF[key]
                    ladder_hits.setdefault(ladder, []).append((date, rung, key))

        out["sessions"] = sessions

        pattern_out = {}
        for p in _PATTERNS:
            data = patterns[p]
            if data["sets"]:
                pattern_out[p] = {
                    "sets": data["sets"],
                    "sets_per_week": round(data["sets"] / weeks, 1),
                    "sessions": len(data["sessions"]),
                    "last": max(data["sessions"]),
                }
            else:
                pattern_out[p] = {"sets": 0, "note": "no sets in window"}
        pattern_out["accessory_sets"] = accessory_sets
        out["patterns"] = pattern_out

        exercise_out = {}
        for key, ex in exercises.items():
            history = list(reversed(ex["history"]))  # newest first
            category, name = key.split("/", 1)
            item = {"pattern": _pattern(category, name), "last": history[0]["date"],
                    "sessions": len(history), "history": history[:4]}
            if key in _RUNG_OF:
                ladder, rung = _RUNG_OF[key]
                item["ladder"] = ladder
                item["rung"] = f"{rung + 1} of {len(LADDERS[ladder])}"
            if any(h.get("clean") is not None for h in history):
                item["clean_streak"] = _clean_streak(history)
            exercise_out[key] = item
        out["exercises"] = exercise_out

        ladders_out = {}
        for ladder, hits in ladder_hits.items():
            rungs = LADDERS[ladder]
            last_date = max(d for d, _, _ in hits)
            current = max(r for d, r, _ in hits if d == last_date)
            keys = sorted({k for d, r, k in hits if d == last_date and r == current})
            item = {
                "rung": current + 1,
                "of": len(rungs),
                "on": keys,
                "last": last_date,
                "highest_in_window": max(r for _, r, _ in hits) + 1,
            }
            streaks = [exercise_out[k]["clean_streak"] for k in keys
                       if "clean_streak" in exercise_out[k]]
            # Absent, not 0, when no session on this rung could be judged.
            if streaks:
                item["clean_streak"] = max(streaks)
            if current + 1 < len(rungs):
                item["next"] = sorted(rungs[current + 1])
            ladders_out[ladder] = item
        out["ladders"] = ladders_out

        if not activities:
            out["note"] = "no strength_training activities in window"
        if errors:
            out["errors"] = errors
        return json.dumps(out, indent=2)

    return app
