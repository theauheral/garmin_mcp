"""Composite one-call briefs for daily planning and coaching.

These aggregate several Garmin endpoints server-side so an assistant doing a
morning briefing or a weekly training review needs one tool call instead of
five to seven. Sub-sections fail independently: a missing night of sleep data
must not take down the whole brief, so each block is fetched in isolation and
failures are reported in an "errors" map instead of raising.
"""

import asyncio
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean

from .health_wellness import _recovery_from_readiness
from .training import _extract_dated_vo2_measurements, _get_max_metrics_range

# The garmin_client will be set by the main file
garmin_client = None

# Activity HR sample streams, memoised per (activity_id, sample budget) — see
# _hr_stream. Cleared whenever the client is reconfigured.
_STREAM_CACHE = {}


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client
    # Cached streams belong to whoever fetched them. Swapping the client
    # (a different account, or a test's fake) must not serve the old one's
    # samples under the same activity ids.
    _STREAM_CACHE.clear()


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
    # get_activity() nests the numbers under summaryDTO (type under
    # activityTypeDTO, aerobic TE named trainingEffect); the list endpoints
    # are flat. Flatten so both shapes summarize the same way.
    dto = activity.get("summaryDTO")
    if isinstance(dto, dict):
        activity = {
            **dto,
            "activityName": activity.get("activityName"),
            "activityType": activity.get("activityTypeDTO"),
            "eventType": activity.get("eventTypeDTO"),
            "aerobicTrainingEffect": dto.get("trainingEffect"),
        }
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
        "aerobic_te": _round(activity.get("aerobicTrainingEffect")),
        "anaerobic_te": _round(activity.get("anaerobicTrainingEffect")),
    }


def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


def _first_number(d, *keys):
    """First numeric value among aliased keys — the list endpoints and
    summaryDTO name the same running-dynamics fields differently."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _form_metrics(activity):
    """Running-form averages from either payload shape (list item or the
    get_activity wrapper with summaryDTO). Garmin sends stride and vertical
    oscillation in centimeters. Every field is None when the device didn't
    record it; wrist-based dynamics never report L/R ground-contact balance."""
    src = _dto_source(activity)
    stride_cm = _first_number(src, "avgStrideLength", "strideLength")
    return {
        "cadence_spm": _round(_first_number(src, "averageRunningCadenceInStepsPerMinute", "averageRunCadence")),
        "max_cadence_spm": _round(_first_number(src, "maxRunningCadenceInStepsPerMinute", "maxRunCadence")),
        "stride_length_m": _round(stride_cm / 100, 2) if stride_cm is not None else None,
        "vertical_oscillation_cm": _round(_first_number(src, "avgVerticalOscillation", "verticalOscillation")),
        "vertical_ratio_pct": _round(_first_number(src, "avgVerticalRatio", "verticalRatio")),
        "ground_contact_time_ms": _round(_first_number(src, "avgGroundContactTime", "groundContactTime")),
        "gct_balance_pct": _first_number(src, "avgGroundContactBalance", "groundContactBalanceLeft"),
        "avg_power_w": _round(_first_number(src, "avgPower", "averagePower")),
        "normalized_power_w": _round(_first_number(src, "normPower", "normalizedPower")),
    }


def _lap_form_rows(splits):
    """Per-lap cadence/stride from get_activity_splits lapDTOs."""
    laps = splits.get("lapDTOs") if isinstance(splits, dict) else splits
    if not isinstance(laps, list):
        return []
    rows = []
    for i, lap in enumerate(laps, start=1):
        if not isinstance(lap, dict):
            continue
        dist, dur = lap.get("distance"), lap.get("duration") or lap.get("movingDuration")
        stride_cm = _first_number(lap, "strideLength", "avgStrideLength")
        rows.append(_drop_none({
            "lap": lap.get("lapIndex") or i,
            "km": _round(dist / 1000, 2) if isinstance(dist, (int, float)) else None,
            "pace_s_per_km": round(dur / (dist / 1000)) if isinstance(dist, (int, float)) and dist > 0 and isinstance(dur, (int, float)) else None,
            "avg_hr": lap.get("averageHR"),
            "cadence_spm": _round(_first_number(lap, "averageRunCadence", "averageRunningCadenceInStepsPerMinute")),
            "stride_length_m": _round(stride_cm / 100, 2) if stride_cm is not None else None,
        }))
    return rows


def _form_fatigue(rows):
    """Form under fatigue: cadence/stride in the first vs last third of laps.
    A collapsing stride with flat-or-rising cadence late is the classic
    tired-form signature."""
    graded = [r for r in rows if isinstance(r.get("cadence_spm"), (int, float))]
    if len(graded) < 3:
        return None
    third = len(graded) // 3

    def _block(part):
        strides = [r["stride_length_m"] for r in part if isinstance(r.get("stride_length_m"), (int, float))]
        return {
            "cadence_spm": _round(fmean(r["cadence_spm"] for r in part)),
            "stride_length_m": _round(fmean(strides), 2) if strides else None,
        }

    first, last = _block(graded[:third]), _block(graded[-third:])
    out = {"first_third": first, "last_third": last}
    if first["cadence_spm"] and last["cadence_spm"]:
        out["cadence_drift_pct"] = _round(100 * (last["cadence_spm"] - first["cadence_spm"]) / first["cadence_spm"])
    if first["stride_length_m"] and last["stride_length_m"]:
        out["stride_drift_pct"] = _round(100 * (last["stride_length_m"] - first["stride_length_m"]) / first["stride_length_m"])
    return out


def _dto_source(activity):
    """The summaryDTO dict when the payload is the get_activity wrapper,
    else the payload itself (list shape)."""
    if isinstance(activity, dict) and isinstance(activity.get("summaryDTO"), dict):
        return activity["summaryDTO"]
    return activity if isinstance(activity, dict) else {}


def _effort_rating(activity):
    """The athlete's own post-session rating from the watch prompt (RPE is
    stored x10, feel is 0-100) plus the structured-workout compliance score.
    summaryDTO-only fields — the list endpoints never carry them."""
    src = _dto_source(activity)
    rpe = _first_number(src, "directWorkoutRpe")
    return {
        "rpe_10": _round(rpe / 10) if rpe is not None else None,
        "feel_pct": _first_number(src, "directWorkoutFeel"),
        "compliance_score": _first_number(src, "directWorkoutComplianceScore"),
    }


def _session_cost(activity):
    """What the session cost: Garmin stamina depletion (summaryDTO-only),
    body-battery drain, and estimated sweat loss."""
    src = _dto_source(activity)
    sweat = _first_number(src, "waterEstimated")
    return {
        "stamina_start_pct": _round(_first_number(src, "beginPotentialStamina")),
        "stamina_end_pct": _round(_first_number(src, "endPotentialStamina")),
        "stamina_min_pct": _round(_first_number(src, "minAvailableStamina")),
        "body_battery_drain": _first_number(src, "differenceBodyBattery"),
        "sweat_loss_ml": round(sweat) if sweat is not None else None,
    }


def _terrain(activity):
    """Climb and grade-adjusted pace — hills inflate raw pace, so judge
    pacing off the adjusted number on lumpy routes."""
    src = _dto_source(activity)
    gas = _first_number(src, "avgGradeAdjustedSpeed")
    return _drop_none({
        "elev_gain_m": _round(_first_number(src, "elevationGain")),
        "elev_loss_m": _round(_first_number(src, "elevationLoss")),
        "grade_adjusted_pace_s_per_km": round(1000 / gas) if isinstance(gas, (int, float)) and gas > 0 else None,
    }) or None


def _power_share_from_list(activity):
    """Easy-share by POWER zones from the flat powerTimeInZone_N list fields.
    Power has no cardiac lag and no heat inflation, so it cross-checks the
    HR-based read on warm or surging runs."""
    if not isinstance(activity, dict):
        return None
    secs = {z: activity.get(f"powerTimeInZone_{z}") for z in range(1, 6)}
    total = sum(v for v in secs.values() if isinstance(v, (int, float)))
    if total <= 0:
        return None
    easy = sum(v for z, v in secs.items() if z <= 2 and isinstance(v, (int, float)))
    return round(100 * easy / total)


def _summarize_exercise_sets(payload):
    """Compact strength view from get_activity_exercise_sets: what was
    actually lifted. Auto-detected exercises are often unclassified; weight
    arrives in grams, 0 = bodyweight."""
    sets = payload.get("exerciseSets") if isinstance(payload, dict) else payload
    if not isinstance(sets, list):
        return None
    active = [s for s in sets if isinstance(s, dict) and s.get("setType") == "ACTIVE"]
    if not active:
        return None
    by_exercise = {}
    total_reps = 0
    active_sec = 0.0
    for s in active:
        cands = [e for e in s.get("exercises") or [] if isinstance(e, dict)]
        best = max(cands, key=lambda e: e.get("probability") or 0, default={})
        name = best.get("name") or best.get("category") or "UNKNOWN"
        if name == "UNKNOWN":
            name = "unclassified"
        entry = by_exercise.setdefault(name, {"sets": 0, "reps": 0, "max_weight_kg": None})
        entry["sets"] += 1
        reps = s.get("repetitionCount")
        if isinstance(reps, (int, float)):
            entry["reps"] += round(reps)
            total_reps += round(reps)
        w = s.get("weight")
        if isinstance(w, (int, float)) and w > 0:
            entry["max_weight_kg"] = max(entry["max_weight_kg"] or 0, round(w / 1000, 1))
        dur = s.get("duration")
        if isinstance(dur, (int, float)):
            active_sec += dur
    return {
        "total_sets": len(active),
        "total_reps": total_reps or None,
        "active_min": round(active_sec / 60, 1) if active_sec else None,
        "by_exercise": {k: _drop_none(v) for k, v in by_exercise.items()},
    }


def _performance_condition(details):
    """Start-vs-late performance condition from the details time series —
    Garmin's rolling freshness delta vs baseline; sliding late in a run is
    a durability signal."""
    if not isinstance(details, dict):
        return None
    idx = None
    for d in details.get("metricDescriptors") or []:
        if isinstance(d, dict) and d.get("key") == "directPerformanceCondition":
            idx = d.get("metricsIndex")
    if not isinstance(idx, int):
        return None
    vals = []
    for row in details.get("activityDetailMetrics") or []:
        m = row.get("metrics") if isinstance(row, dict) else None
        if isinstance(m, list) and len(m) > idx and isinstance(m[idx], (int, float)):
            vals.append(m[idx])
    if not vals:
        return None
    return {
        "start": _round(vals[0]),
        "end": _round(vals[-1]),
        "min": _round(min(vals)),
        "max": _round(max(vals)),
        "delta": _round(vals[-1] - vals[0]),
    }


def _resolve_activity(activity_id, date):
    """Resolve one activity to (summary_payload, activity_id, note, others):
    an explicit id, else a date's main activity, else the most recent one.
    note is set (and the payload None) when nothing was found.

    A date's "main" activity is its longest by timer time. The list endpoint
    is newest-first, so taking the first item would audit the cool-down walk
    that followed the run. The rest come back in `others` so the caller can
    name them and the athlete can ask for one by id.
    """
    if activity_id:
        return garmin_client.get_activity(activity_id) or {}, activity_id, None, []
    if date:
        acts = garmin_client.get_activities_by_date(date, date) or []
    else:
        acts = garmin_client.get_activities(0, 1) or []
    acts = [a for a in acts if isinstance(a, dict)]
    if not acts:
        return None, None, f"no activity found for {date or 'most recent'}", []
    main = max(acts, key=lambda a: a.get("duration") or 0)
    others = [
        _drop_none({
            "activity_id": a.get("activityId"),
            "type": (a.get("activityType") or {}).get("typeKey"),
            "start": (a.get("startTimeLocal") or "")[11:16] or None,
            "duration_min": _minutes(a.get("duration")),
        })
        for a in acts if a is not main
    ]
    return main, main.get("activityId"), None, others


def _full_activity(summary, aid):
    """The summaryDTO-shaped payload for an already-resolved activity.
    RPE/feel/stamina/compliance exist only in that shape, so when the
    summary came from a flat list endpoint, re-fetch the full activity."""
    if isinstance(summary.get("summaryDTO"), dict):
        return summary
    try:
        return garmin_client.get_activity(aid) or summary
    except Exception:
        return summary


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


def _primary_device_entry(device_map):
    """Pick from a device-id-keyed map, preferring the primary training
    device and falling back to the first entry — the same selection as
    upstream's training-status tools."""
    entry = {}
    if not isinstance(device_map, dict):
        return entry
    for dev_data in device_map.values():
        if not isinstance(dev_data, dict):
            continue
        if dev_data.get("primaryTrainingDevice"):
            return dev_data
        if not entry:
            entry = dev_data
    return entry


def _load_position(date):
    """Acute load vs Garmin's optimal tunnel, via the training status endpoint."""
    data = garmin_client.get_training_status(date) or {}
    latest_map = (data.get("mostRecentTrainingStatus") or {}).get("latestTrainingStatusData") or {}
    entry = _primary_device_entry(latest_map)
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


def _load_snapshot(date):
    """Compact load position for one date incl. TSB (chronic - acute)."""
    pos = _load_position(date)
    acute, chronic = pos.get("acute_load"), pos.get("chronic_load")
    if isinstance(acute, (int, float)) and isinstance(chronic, (int, float)):
        pos["tsb"] = round(chronic - acute)
    return pos


def _direction(now, past):
    if not isinstance(now, (int, float)) or not isinstance(past, (int, float)):
        return None
    delta = now - past
    if abs(delta) < max(2.0, 0.05 * abs(past or 1)):
        return "flat"
    return "rising" if delta > 0 else "falling"


def _zone_distribution(zones, boundary_key="low_bpm"):
    """Summarize a Garmin time-in-zones payload (HR or power — same shape)
    into easy vs hard share. boundary_key labels zoneLowBoundary in the
    output: bpm for HR zones, watts for power zones."""
    if not isinstance(zones, list):
        return None
    total = sum(z.get("secsInZone", 0) or 0 for z in zones if isinstance(z, dict))
    if total <= 0:
        return None
    by_zone = {}
    easy = hard = 0.0
    for z in zones:
        if not isinstance(z, dict):
            continue
        n = z.get("zoneNumber")
        secs = z.get("secsInZone", 0) or 0
        by_zone[f"z{n}"] = {
            "min": round(secs / 60, 1),
            "pct": round(100 * secs / total),
            # which zone model was applied — boundaries must be visible, or
            # easy/hard share is uninterpretable
            boundary_key: z.get("zoneLowBoundary"),
        }
        if isinstance(n, int):
            if n <= 2:
                easy += secs
            else:
                hard += secs
    return {
        "total_min": round(total / 60, 1),
        "by_zone": by_zone,
        "easy_share_pct": round(100 * easy / total),   # Z1-2 = aerobic/easy
        "hard_share_pct": round(100 * hard / total),    # Z3+ = tempo and above
    }


# --- Recomputed time-in-zone -------------------------------------------------
# Garmin freezes hrTimeInZones into each activity AT UPLOAD. Correct the zone
# model afterwards and every older activity keeps its old bands forever —
# re-pulling returns byte-identical zone times. Any easy-share series spanning
# a zone-model change is therefore a broken prefix joined to a clean suffix,
# and easy-share is the headline KPI this server reports. So: integrate
# time-in-zone from the raw HR sample stream against the CURRENT zone model,
# and surface it ALONGSIDE Garmin's stored numbers rather than in place of
# them — being able to compare the two is what shows the correction landed.

_HR_ZONES_PATH = "/biometric-service/heartRateZones"
# Streams are 1 Hz and Garmin decimates above the requested size. Weighting
# each sample by the gap to the next makes the result near-invariant to that
# decimation (measured: identical easy-share from 300 to 5000 samples on a
# 49-minute run), so this is a bandwidth choice, not an accuracy one.
_HR_STREAM_SAMPLES = 2000
# The trend fetches a stream per session, so it pays the payload N times. Time
# weighting makes the result invariant to decimation (identical easy-share from
# 300 to 5000 samples on a 49-minute run), so the trend buys the cheap one.
_HR_STREAM_SAMPLES_TREND = 500
# A gap longer than this is a dropout, not time spent in a zone — counting it
# would credit whichever zone happened to precede it. It is a floor: a heavily
# decimated stream (a long activity at the trend's sample budget) spaces its
# samples wider than this, so the cap scales with the stream's own spacing.
_HR_GAP_CAP_S = 60

# Garmin's zone floors as fractions of the anchor, measured rather than
# documented: live %LTHR floors at LTHR 173 and 171 (112/137/152/163/171 and
# 111/135/150/161/169) both reproduce to the bpm from these ratios, and %max
# zones are Garmin's stock 50/60/70/80/90. They are what turns "LTHR = 172"
# into bands, and what recognises which basis a set of floors was built on.
_LTHR_ZONE_RATIOS = (0.65, 0.79, 0.88, 0.94, 0.99)
_PCT_ZONE_RATIOS = (0.50, 0.60, 0.70, 0.80, 0.90)
_BASIS_TOLERANCE_BPM = 2
# What each training method's floors should look like.
_METHOD_BASIS = {
    "LACTATE_THRESHOLD": "pct_lthr",
    "HR_MAX": "pct_max_hr",
    "HR_RESERVE": "pct_hrr",
}


def _zone_sport(type_key):
    """The zone-config sport for an activity typeKey. Garmin keeps DEFAULT,
    RUNNING and CYCLING zone sets independently; anything else uses DEFAULT."""
    key = (type_key or "").lower()
    if "run" in key:
        return "RUNNING"
    if "cycl" in key or "bik" in key or "ride" in key:
        return "CYCLING"
    return "DEFAULT"


def _floors_basis(floors, lthr=None, max_hr=None, resting_hr=None):
    """Which anchor a set of floors was built from: pct_lthr, pct_max_hr,
    pct_hrr, or custom. Recognised within a couple of bpm of Garmin's rounding."""

    def _fits(expected):
        return all(abs(f - e) <= _BASIS_TOLERANCE_BPM for f, e in zip(floors, expected))

    candidates = []
    if isinstance(lthr, (int, float)) and lthr > 0:
        candidates.append(("pct_lthr", [lthr * r for r in _LTHR_ZONE_RATIOS]))
    if isinstance(max_hr, (int, float)) and max_hr > 0:
        candidates.append(("pct_max_hr", [max_hr * r for r in _PCT_ZONE_RATIOS]))
        if isinstance(resting_hr, (int, float)) and 0 < resting_hr < max_hr:
            candidates.append((
                "pct_hrr",
                [resting_hr + r * (max_hr - resting_hr) for r in _PCT_ZONE_RATIOS],
            ))
    for basis, expected in candidates:
        if _fits(expected):
            return basis
    return "custom"


def _lthr_floors(lthr):
    return [int(round(lthr * r)) for r in _LTHR_ZONE_RATIOS]


def _zone_model_from_config(entries, sport="DEFAULT"):
    """Pull the zone floors for one sport out of Garmin's zone-config payload.

    The config is the live model — what the watch applies today — as opposed to
    the bands frozen into each activity at upload time. A sport without its own
    entry falls back to DEFAULT, which is what the watch does.
    """
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return None
    by_sport = {}
    for e in entries:
        if isinstance(e, dict):
            by_sport.setdefault((e.get("sport") or "").upper(), e)
    wanted = sport.upper()
    chosen = by_sport.get(wanted) or by_sport.get("DEFAULT")
    if chosen is None and by_sport:
        chosen = next(iter(by_sport.values()))
    if not chosen:
        return None
    floors = []
    for n in range(1, 6):
        v = chosen.get(f"zone{n}Floor")
        if not isinstance(v, (int, float)):
            return None
        floors.append(int(v))
    return {
        "floors": floors,
        "lthr": chosen.get("lactateThresholdHeartRateUsed"),
        "max_hr": chosen.get("maxHeartRateUsed"),
        "resting_hr": chosen.get("restingHeartRateUsed"),
        "training_method": chosen.get("trainingMethod"),
        "sport": chosen.get("sport"),
        "sport_requested": wanted,
    }


def _hr_zone_model(lthr=None, sport="DEFAULT"):
    """The zone model to score against: Garmin's live config for `sport`, or
    %LTHR bands built from a caller-supplied LTHR.

    The LTHR is a parameter and never a constant because the sources disagree
    in practice — Garmin auto-detects one value, a field test measures another
    — and ~5 bpm moves every boundary. The model also says which basis its
    floors were really built on: Garmin has been seen labelling a config
    LACTATE_THRESHOLD while serving 50–90% of max HR, which silently turns an
    "easy" cap into a Z4 floor. Returns (model, None) or (None, reason).
    """
    try:
        raw = garmin_client.connectapi(_HR_ZONES_PATH)
    except Exception as e:
        return None, f"zone configuration unavailable: {e}"
    base = _zone_model_from_config(raw, sport)
    if not base:
        return None, "zone configuration returned no usable zone floors"

    model = {
        "lthr": base["lthr"],
        "lthr_source": "garmin_zone_config",
        "training_method": base["training_method"],
        "max_hr": base["max_hr"],
        "resting_hr": base["resting_hr"],
        "sport": base["sport"],
        "floors": list(base["floors"]),
    }
    if base["sport"] and (base["sport"] or "").upper() != base["sport_requested"]:
        model["sport_note"] = (
            f"no {base['sport_requested']} zone set; scored on {base['sport']}"
        )
    basis = _floors_basis(model["floors"], base["lthr"], base["max_hr"], base["resting_hr"])
    model["floors_basis"] = basis
    expected = _METHOD_BASIS.get((base["training_method"] or "").upper())
    if expected and basis != expected:
        model["method_mismatch"] = (
            f"Garmin labels these zones {base['training_method']} but the floors "
            f"{model['floors']} are {basis.replace('_', ' ')}"
            + (f" (max HR {base['max_hr']})" if basis == "pct_max_hr" else "")
            + " — the label is wrong, and every share scored on them inherits that"
        )
    if lthr:
        # Built from Garmin's own %LTHR ratios, never by rescaling the config:
        # rescaling only works when the config is itself %LTHR, and it isn't
        # always (see method_mismatch).
        model["floors"] = _lthr_floors(lthr)
        model["floors_basis"] = "pct_lthr"
        model["lthr_configured"] = base["lthr"]
        model["lthr"] = lthr
        model["lthr_source"] = "caller_override"
        model.pop("method_mismatch", None)
    model["floors_bpm"] = {f"z{i}": f for i, f in enumerate(model["floors"], start=1)}
    return model, None


def _model_summary(model):
    """The model fields every consumer should see beside a recomputed share."""
    return _drop_none({
        "lthr": model.get("lthr"),
        "lthr_source": model.get("lthr_source"),
        "lthr_configured": model.get("lthr_configured"),
        "training_method": model.get("training_method"),
        "floors_basis": model.get("floors_basis"),
        "method_mismatch": model.get("method_mismatch"),
        "max_hr": model.get("max_hr"),
        "sport": model.get("sport"),
        "sport_note": model.get("sport_note"),
        "floors_bpm": model.get("floors_bpm"),
    })


# An activity's samples never change after upload, so a scheduled agent calling
# get_execution_trend repeatedly re-downloads the same bytes. Memoise the stream
# — never the recomputation, which depends on a zone model that does drift —
# and never a failure: a timeout or a 429 is transient, and caching it would
# make the activity unrecomputable until the server restarts.
_STREAM_CACHE_MAX = 64


def _activity_stream(activity_id, max_samples=_HR_STREAM_SAMPLES):
    """Timer-time sample rows for one activity, or (None, reason).

    Rows are dicts with t (timer seconds), hr, and — when recorded — dist
    (metres) and speed (m/s).
    """
    key = (activity_id, max_samples)
    if key in _STREAM_CACHE:
        return _STREAM_CACHE[key], None
    rows, reason = _fetch_stream(activity_id, max_samples)
    if rows is None:
        return None, reason
    if len(_STREAM_CACHE) >= _STREAM_CACHE_MAX:
        _STREAM_CACHE.pop(next(iter(_STREAM_CACHE)))
    _STREAM_CACHE[key] = rows
    return rows, None


def _hr_stream(activity_id, max_samples=_HR_STREAM_SAMPLES):
    """(timer_seconds, bpm) samples for one activity, or (None, reason)."""
    rows, reason = _activity_stream(activity_id, max_samples)
    if rows is None:
        return None, reason
    return [(r["t"], r["hr"]) for r in rows], None


def _fetch_stream(activity_id, max_samples):
    """maxpoly=0 drops the GPS polyline: this needs a few columns of a
    26-column payload and the track is the bulk of it.

    Time is TIMER time (sumDuration), not elapsed: it stops while the activity
    is paused, which is what Garmin's own zone times count. Measured on twelve
    runs: timer time reproduces the stored zone minutes exactly, elapsed time
    over-counts every stop at a crossing into the zone before it.
    """
    try:
        details = garmin_client.get_activity_details(
            activity_id, maxchart=max_samples, maxpoly=0
        )
    except Exception as e:
        return None, f"HR stream unavailable: {e}"
    if not isinstance(details, dict):
        return None, "activity details returned no payload"
    index = {}
    for d in details.get("metricDescriptors") or []:
        if isinstance(d, dict) and isinstance(d.get("metricsIndex"), int):
            index[d.get("key")] = d["metricsIndex"]
    hr_i = index.get("directHeartRate")
    t_i = index.get("sumDuration", index.get("sumElapsedDuration"))
    dist_i = index.get("sumDistance")
    speed_i = index.get("directSpeed")
    if hr_i is None:
        return None, "activity has no heart-rate stream (manual entry, or HR not recorded)"
    if t_i is None:
        return None, "activity stream carries no timer column"

    def _num(m, i):
        if i is None or i >= len(m):
            return None
        v = m[i]
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    rows = []
    for row in details.get("activityDetailMetrics") or []:
        m = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(m, list):
            continue
        hr, t = _num(m, hr_i), _num(m, t_i)
        if hr is None or t is None or hr <= 0:
            continue
        rows.append({"t": t, "hr": hr, "dist": _num(m, dist_i), "speed": _num(m, speed_i)})
    if len(rows) < 2:
        return None, "activity has no usable heart-rate samples"
    rows.sort(key=lambda r: r["t"])
    return rows, None


def _gap_cap(samples):
    """The longest gap still counted as continuous time: the floor, or three
    sample spacings when the stream is decimated wider than the floor."""
    gaps = sorted(t1 - t0 for (t0, _), (t1, _) in zip(samples, samples[1:]) if t1 > t0)
    if not gaps:
        return _HR_GAP_CAP_S
    return max(_HR_GAP_CAP_S, 3 * gaps[len(gaps) // 2])


def _weighted(samples):
    """(bpm, seconds) per sample, weighting each by the gap to the next and
    dropping gaps too long to be continuous recording."""
    cap = _gap_cap(samples)
    for (t0, hr), (t1, _) in zip(samples, samples[1:]):
        dt = t1 - t0
        if 0 < dt <= cap:
            yield hr, dt


def _integrate_zones(samples, floors):
    """Seconds per zone, time-weighting each HR sample by the gap to the next.

    Returns (secs_by_zone, below_z1_secs). HR under the zone-1 floor is counted
    separately and excluded from the shares, which is what Garmin does too —
    its stored totals omit sub-Z1 time rather than folding it into Z1.
    """
    secs = {n: 0.0 for n in range(1, len(floors) + 1)}
    below = 0.0
    for hr, dt in _weighted(samples):
        zone = 0
        for n, floor in enumerate(floors, start=1):
            if hr >= floor:
                zone = n
        if zone:
            secs[zone] += dt
        else:
            below += dt
    return secs, below


def _ceiling_share(samples, ceiling):
    """Time at or under an HR ceiling — the coaching plan's own easy cap,
    independent of whatever zone model Garmin is serving this week."""
    under = over = 0.0
    hr_secs = 0.0
    for hr, dt in _weighted(samples):
        hr_secs += hr * dt
        if hr <= ceiling:
            under += dt
        else:
            over += dt
    total = under + over
    if total <= 0:
        return None
    return {
        "ceiling_bpm": ceiling,
        "under_pct": round(100 * under / total),
        "over_min": _round(over / 60),
        "avg_hr": _round(hr_secs / total),
    }


# Aerobic decoupling (Pa:HR, Friel): how far efficiency (speed per heartbeat)
# falls from the first half of a steady effort to the second. Under 5% is the
# usual mark of a durable aerobic engine; over 8% means the back half slipped.
_DECOUPLING_MIN_USABLE_S = 20 * 60    # shorter than this the halves are noise
_DECOUPLING_WARMUP_S = 10 * 60        # skipped only when the run is long enough to afford it
_DECOUPLING_WARMUP_MIN_TOTAL_S = 50 * 60
_DECOUPLING_MIN_SPEED = 0.5           # m/s: standing still is not running
_DECOUPLING_STEADY_CV = 0.20          # pace spread above this is intervals/hills, not a steady effort


def _aerobic_decoupling(rows):
    """Pa:HR decoupling for one activity's stream rows, or None when it can't
    be computed (no speed column, too short, nothing moving).

    Splits the moving time in half by timer time and compares efficiency
    (speed / HR, time-weighted) between the halves. `steady` is False when the
    pace varied too much for the number to mean anything (intervals, hills) —
    the figure is still returned, so the caller sees both and can say why it
    is not a durability read.
    """
    pts = [(r["t"], r["hr"], r["speed"]) for r in rows or []
           if r.get("speed") is not None and r["speed"] >= _DECOUPLING_MIN_SPEED]
    if len(pts) < 4:
        return None
    cap = _gap_cap([(t, hr) for t, hr, _ in pts])
    segs = []   # (start_t, seconds, hr, speed) — each sample weighted by the gap to the next
    for (t0, hr, sp), (t1, _, _) in zip(pts, pts[1:]):
        dt = t1 - t0
        if 0 < dt <= cap:
            segs.append((t0, dt, hr, sp))
    if not segs:
        return None
    total_span = pts[-1][0] - pts[0][0]
    skipped = _DECOUPLING_WARMUP_S if total_span >= _DECOUPLING_WARMUP_MIN_TOTAL_S else 0
    start = pts[0][0] + skipped
    segs = [s for s in segs if s[0] >= start]
    usable = sum(s[1] for s in segs)
    if usable < _DECOUPLING_MIN_USABLE_S:
        return None
    # Halve by moving time, so a long stop doesn't tilt where the middle falls.
    half, acc, split = usable / 2, 0.0, len(segs)
    for i, s in enumerate(segs):
        acc += s[1]
        if acc >= half:
            split = i + 1
            break
    first, second = segs[:split], segs[split:]
    if not first or not second:
        return None

    def _half(part):
        secs = sum(s[1] for s in part)
        hr = sum(s[2] * s[1] for s in part) / secs
        speed = sum(s[3] * s[1] for s in part) / secs
        return {
            "avg_hr": _round(hr),
            "pace_min_per_km": _round(1000 / speed / 60, 2),
            "efficiency": speed / hr,
        }

    a, b = _half(first), _half(second)
    pct = (a["efficiency"] - b["efficiency"]) / a["efficiency"] * 100
    speeds = [s[3] for s in segs]
    mean = sum(speeds) / len(speeds)
    cv = (sum((v - mean) ** 2 for v in speeds) / len(speeds)) ** 0.5 / mean
    steady = cv <= _DECOUPLING_STEADY_CV
    if not steady:
        read = "not a steady effort — pace varied too much for decoupling to mean anything"
    elif pct < 5:
        read = "under 5%: strong aerobic durability, the engine held"
    elif pct <= 8:
        read = "5-8%: moderate fade — a little hot or durability still building"
    else:
        read = "over 8%: too hard, or heat/fuel/fitness let the back half slip"
    for h in (a, b):
        h.pop("efficiency")
    return {
        "decoupling_pct": _round(pct),
        "first_half": a,
        "second_half": b,
        "steady": steady,
        "read": read,
        "warmup_skipped_min": _round(skipped / 60) if skipped else 0,
        "usable_min": _round(usable / 60),
    }


# Sleep regularity. A night belongs to its WAKE-UP date (that is how Garmin keys
# it). Times are the watch's own local wall-clock (the *Local timestamps, which
# are local time encoded as if it were UTC), so a trip across a timezone does
# not read as a shifted night.
_SLEEP_MIN_NIGHTS = 5            # fewer than this and spread/regularity are noise
_SLEEP_EVENING_ORIGIN_MIN = 18 * 60   # bedtimes are measured from 18:00 so 23:30 and 01:00 do not straddle midnight
_SOCIAL_JET_LAG_H = 1.0          # Wittmann et al. 2006's usual flag for a work/free-day shift


def _clock_min(ms):
    """Minutes since local midnight for a local-wall-clock ms timestamp."""
    t = datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc)
    return t.hour * 60 + t.minute + t.second / 60


def _sleep_day(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).date()


def _sd(values):
    if len(values) < 3:
        return None
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _hhmm(minutes):
    m = round(minutes) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _sleep_regularity(nights):
    """Timing regularity from nights of {date, start_ms, end_ms} (local clock).

    - Bedtime / wake / midpoint spread (SD, minutes).
    - Social jet lag: weekend-night midpoint minus weekday-night midpoint
      (Wittmann 2006). A night's weekday is its wake-up date's.
    - Sleep Regularity Index (Phillips 2017): the chance of being in the same
      sleep/wake state 24 hours apart, 100 = identical every day, 0 = random.
      Computed from the sleep window of each pair of consecutive nights (naps
      and mid-night wakings are not in the window), so it is the onset/offset
      version of the index, not the minute-by-minute one.
    Returns None under _SLEEP_MIN_NIGHTS usable nights.
    """
    rows = []
    for n in nights:
        s, e = n.get("start_ms"), n.get("end_ms")
        if not isinstance(s, (int, float)) or not isinstance(e, (int, float)) or e <= s:
            continue
        bed = (_clock_min(s) - _SLEEP_EVENING_ORIGIN_MIN) % 1440   # minutes after 18:00
        wake = _clock_min(e)
        rows.append({
            "date": n["date"], "start_ms": s, "end_ms": e, "bed": bed, "wake": wake,
            "mid": (bed + (e - s) / 120000) % 1440,                # (e-s)/2 in minutes
            "hours": (e - s) / 3600000,
            "weekend": datetime.date.fromisoformat(n["date"]).weekday() >= 5,
        })
    rows.sort(key=lambda r: r["date"])
    if len(rows) < _SLEEP_MIN_NIGHTS:
        return None

    def clock(minutes_after_evening):
        return _hhmm(minutes_after_evening + _SLEEP_EVENING_ORIGIN_MIN)

    out = {
        "nights_used": len(rows),
        "avg_bedtime": clock(fmean(r["bed"] for r in rows)),
        "avg_wake": _hhmm(fmean(r["wake"] for r in rows)),
        "avg_midpoint": clock(fmean(r["mid"] for r in rows)),
        "avg_sleep_h": _round(fmean(r["hours"] for r in rows), 2),
        "bedtime_sd_min": _round(_sd([r["bed"] for r in rows]), 0),
        "wake_sd_min": _round(_sd([r["wake"] for r in rows]), 0),
        "midpoint_sd_min": _round(_sd([r["mid"] for r in rows]), 0),
    }

    # Sleep Regularity Index over consecutive-date night pairs
    by_date = {r["date"]: r for r in rows}
    agree = []
    for r in rows:
        nxt = by_date.get((datetime.date.fromisoformat(r["date"]) + datetime.timedelta(days=1)).isoformat())
        if not nxt:
            continue

        def window(night):
            """The night as minutes inside its own noon-to-noon window, clipped to it."""
            wake_day = _sleep_day(night["end_ms"])
            noon0 = datetime.datetime.combine(wake_day - datetime.timedelta(days=1),
                                              datetime.time(12), datetime.timezone.utc).timestamp() * 1000
            a = min(1440.0, max(0.0, (night["start_ms"] - noon0) / 60000))
            b = min(1440.0, max(0.0, (night["end_ms"] - noon0) / 60000))
            return a, b

        (a1, b1), (a2, b2) = window(r), window(nxt)
        overlap = max(0.0, min(b1, b2) - max(a1, a2))
        union = (b1 - a1) + (b2 - a2) - overlap
        agree.append((overlap + (1440 - union)) / 1440)
    if len(agree) >= 3:
        out["sleep_regularity_index"] = _round(100 * (2 * fmean(agree) - 1), 1)
        out["sri_pairs"] = len(agree)

    wk = [r["mid"] for r in rows if not r["weekend"]]
    we = [r["mid"] for r in rows if r["weekend"]]
    if len(wk) >= 3 and len(we) >= 2:
        diff_h = (fmean(we) - fmean(wk)) / 60
        out["social_jet_lag"] = {
            "weekday_midpoint": clock(fmean(wk)),
            "weekend_midpoint": clock(fmean(we)),
            "shift_h": _round(diff_h, 2),
            "flag": abs(diff_h) > _SOCIAL_JET_LAG_H,
            "weekday_nights": len(wk),
            "weekend_nights": len(we),
        }
    out["nights"] = [
        {"date": r["date"], "bed": clock(r["bed"]), "wake": _hhmm(r["wake"]),
         "midpoint": clock(r["mid"]), "hours": _round(r["hours"], 2)}
        for r in rows
    ]
    return out


def _threshold_test(rows, effort_min=30, tail_min=20, garmin_lthr=None):
    """Read a 30-minute threshold field test out of one activity's stream.

    The protocol (Friel): warm up, run 30 minutes as hard as can be held
    evenly, cool down; LTHR = average HR over the LAST 20 minutes of the
    effort, threshold pace = its pace. The effort is located as the 30-minute
    timer window with the highest time-weighted HR, so the warm-up and
    cool-down fall outside it without being declared.
    """
    samples = [(r["t"], r["hr"]) for r in rows]
    cap = _gap_cap(samples)
    effort_s, tail_s = effort_min * 60, tail_min * 60
    if samples[-1][0] - samples[0][0] < effort_s:
        return {"not_applicable": f"activity shorter than the {effort_min}-minute effort"}

    # Prefix sums of HR*dt and dt over sample intervals, gaps over the cap
    # contributing nothing — the window average is then O(1).
    n = len(samples)
    hr_sum, t_sum = [0.0] * n, [0.0] * n
    for i in range(1, n):
        dt = samples[i][0] - samples[i - 1][0]
        ok = 0 < dt <= cap
        hr_sum[i] = hr_sum[i - 1] + (samples[i - 1][1] * dt if ok else 0.0)
        t_sum[i] = t_sum[i - 1] + (dt if ok else 0.0)

    def _avg(i, j):
        secs = t_sum[j] - t_sum[i]
        return (hr_sum[j] - hr_sum[i]) / secs if secs > 0 else None

    best = None
    j = 0
    for i in range(n):
        while j < n - 1 and samples[j][0] - samples[i][0] < effort_s:
            j += 1
        if samples[j][0] - samples[i][0] < effort_s:
            break
        a = _avg(i, j)
        if a is not None and (best is None or a > best[0]):
            best = (a, i, j)
    if best is None:
        return {"not_applicable": "no continuous 30-minute stretch in the HR stream"}
    _, i0, j0 = best
    k = i0
    while k < j0 and samples[j0][0] - samples[k][0] > tail_s:
        k += 1
    lthr = _avg(k, j0)
    first10 = _avg(i0, k)

    pace = None
    d0, d1 = rows[k].get("dist"), rows[j0].get("dist")
    tail_secs = samples[j0][0] - samples[k][0]
    if isinstance(d0, (int, float)) and isinstance(d1, (int, float)) and d1 > d0:
        pace = round(tail_secs / ((d1 - d0) / 1000))
    else:
        speeds = [r["speed"] for r in rows[k:j0 + 1] if isinstance(r.get("speed"), (int, float)) and r["speed"] > 0]
        if speeds:
            pace = round(1000 / fmean(speeds))

    out = {
        "protocol": f"{effort_min}-min effort, LTHR = avg HR of its last {tail_min} min",
        "effort_window_min": [_round(samples[i0][0] / 60), _round(samples[j0][0] / 60)],
        "lthr_estimate_bpm": round(lthr) if lthr else None,
        "threshold_pace_s_per_km": pace,
        "first_10_min_avg_hr": round(first10) if first10 else None,
        # A 30-minute effort that was paced evenly rises a few bpm across the
        # window; a big jump means a hot start or a finishing kick, and the
        # tail average reads high.
        "caveat": "valid only for a solo, evenly paced, all-out 30 minutes — "
                  "a race, a group run or a surge-and-fade reads high or low",
    }
    if out["lthr_estimate_bpm"]:
        out["lt_zone_floors_bpm"] = _lthr_floors(out["lthr_estimate_bpm"])
    if garmin_lthr:
        out["garmin_lthr_bpm"] = garmin_lthr
    return _drop_none(out)


def _shares(secs):
    """(total_seconds, easy_pct, hard_pct) over Z1-2 vs Z3+."""
    total = sum(secs.values())
    if total <= 0:
        return 0.0, None, None
    easy = sum(v for n, v in secs.items() if n <= 2)
    return total, round(100 * easy / total), round(100 * (total - easy) / total)


def _recompute_hr_zones(activity_id, model, stored=None, samples=None,
                        max_samples=_HR_STREAM_SAMPLES):
    """Time-in-zone for one activity scored against `model`, never raising.

    Always answers: either the recomputed distribution, or an explicit
    `not_recomputable` with the reason. Falling back to the frozen value
    without saying so would reintroduce the exact bug this replaces.
    """
    if samples is None:
        samples, reason = _hr_stream(activity_id, max_samples)
        if samples is None:
            return {"not_recomputable": reason}
    secs, below = _integrate_zones(samples, model["floors"])
    total, easy, hard = _shares(secs)
    if not total:
        return {"not_recomputable": "no heart-rate time inside any zone"}

    out = {
        "easy_share_pct": easy,
        "hard_share_pct": hard,
        "total_min": _round(total / 60),
        "below_z1_min": _round(below / 60),
        "by_zone": {
            f"z{n}": {
                "min": _round(v / 60),
                "pct": round(100 * v / total),
                "low_bpm": model["floors"][n - 1],
            }
            for n, v in secs.items()
        },
        "applied": _model_summary(model),
        "samples": len(samples),
    }
    verification = _verify_against_stored(samples, stored, model["floors"])
    if verification:
        out["verification"] = verification
    return _drop_none(out)


def _stored_floors(stored, boundary_key="low_bpm"):
    """The bands Garmin froze into the activity, from a _zone_distribution."""
    by_zone = (stored or {}).get("by_zone") if isinstance(stored, dict) else None
    if not isinstance(by_zone, dict):
        return None
    floors = []
    for n in range(1, 6):
        low = (by_zone.get(f"z{n}") or {}).get(boundary_key)
        if not isinstance(low, (int, float)):
            return None
        floors.append(int(low))
    return floors


def _stamp_history(stamps, current=None):
    """Group per-session frozen bands into the distinct zone models they came
    from, newest first.

    A zone-model change is invisible in any single activity — the bands are
    stamped at upload and look authoritative forever. Seen side by side across
    a block they are obvious, and this is what turns "the numbers moved" into
    "the model changed on this date, and these N sessions carry the old one".
    `stamps` is [(date, floors)] in the order sessions were analysed.
    """
    groups = []
    for date, floors in stamps:
        if not floors:
            continue
        if groups and groups[-1]["bands"] == floors:
            groups[-1]["sessions"] += 1
            groups[-1]["oldest"] = date
        else:
            groups.append({"bands": list(floors), "sessions": 1,
                           "newest": date, "oldest": date})
    if len(groups) < 2 and current is None:
        return None
    for g in groups:
        if current is not None:
            g["matches_current_model"] = g["bands"] == list(current)
    return groups or None


def _verify_against_stored(samples, stored, applied_floors):
    """Re-run the integration with the activity's OWN frozen bands.

    Reproducing Garmin's stored easy-share from the raw stream is what proves
    the integration faithful: any remaining difference against the live model
    is then a real zone-model change and not an arithmetic error.
    """
    frozen = _stored_floors(stored)
    if not frozen:
        return {}
    secs, _ = _integrate_zones(samples, frozen)
    _, easy, _ = _shares(secs)
    stored_easy = stored.get("easy_share_pct")
    return _drop_none({
        "stored_bands_bpm": frozen,
        "recomputed_with_stored_bands_pct": easy,
        "stored_easy_share_pct": stored_easy,
        # Within a point means the sample stream reproduces Garmin's own
        # integration; a mismatch means distrust the recomputed number too.
        "matches_stored": (
            abs(easy - stored_easy) <= 1
            if isinstance(easy, int) and isinstance(stored_easy, int) else None
        ),
        # The one-field answer to "is this activity affected by the zone-model
        # change?" — it replaces eyeballing every zone's low_bpm by hand.
        "stored_bands_differ": frozen != list(applied_floors),
    })


def _power_model_drift(activities):
    """Have the POWER zone bands moved across the analysed window?

    Garmin freezes powerTimeInZones at upload exactly as it freezes the HR
    ones, and the skill leans on power as the heat-proof cross-check — so the
    same silent rot applies, and there is no live power-zone configuration to
    score against (the biometric-service endpoints that exist for HR return
    404/405 for power). A drift check is therefore the available guard: sample
    the oldest and newest session in the window, two calls regardless of N,
    and say plainly whether the model moved under them.
    """
    ids = [a.get("activityId") for a in activities if isinstance(a, dict) and a.get("activityId")]
    if len(ids) < 2:
        return None
    ends = []
    for aid in (ids[0], ids[-1]):
        try:
            floors = _stored_floors(
                _zone_distribution(
                    garmin_client.get_activity_power_in_timezones(aid),
                    boundary_key="low_w",
                ),
                boundary_key="low_w",
            )
        except Exception:
            return None
        if not floors:
            return None
        ends.append(floors)
    newest, oldest = ends
    return {
        "bands_w": newest,
        "stable_across_window": newest == oldest,
        "note": (
            "power zone bands unchanged across the analysed window — "
            "power_easy_share_pct is comparable across these sessions"
            if newest == oldest else
            f"power zone bands CHANGED across the window (oldest {oldest}, "
            "newest {newest}) — power_easy_share_pct is frozen per activity "
            "just like HR, so do not compare or average across the join"
        ).replace("{newest}", str(newest)),
    }


def _pacing(lap_rows):
    """First-half vs second-half pace drift from km laps (fade detection).
    Laps are split at the distance midpoint and each half's pace is
    distance-weighted — the typed-splits feed this used before mixes
    run/walk/stand segments with whole-run aggregate rows, and an
    unweighted mean over unequal distances misreads the drift."""
    rows = [
        r for r in lap_rows
        if isinstance(r.get("km"), (int, float)) and r["km"] > 0
        and isinstance(r.get("pace_s_per_km"), (int, float))
    ]
    if len(rows) < 2:
        return None
    total_km = sum(r["km"] for r in rows)
    half, cum = total_km / 2, 0.0
    first, second = [], []
    for r in rows:
        (first if cum + r["km"] / 2 <= half else second).append(r)
        cum += r["km"]
    if not first or not second:
        return None

    def _pace(part):
        km = sum(r["km"] for r in part)
        return sum(r["pace_s_per_km"] * r["km"] for r in part) / km

    fp, sp = _pace(first), _pace(second)
    drift = round(100 * (sp - fp) / fp, 1)
    return {
        "splits": rows,
        "first_half_pace_s": round(fp),
        "second_half_pace_s": round(sp),
        "drift_pct": drift,   # + = slowed (positive split/fade), - = negative split
        "shape": "negative_split" if drift < -1 else "faded" if drift > 3 else "even",
    }


def _utc_offset_hours(bb_day):
    """Local-minus-GMT hours from a body-battery day's own paired timestamps."""
    gmt, local = bb_day.get("startTimestampGMT"), bb_day.get("startTimestampLocal")
    if not (isinstance(gmt, str) and isinstance(local, str)):
        return None
    try:
        delta = (datetime.datetime.fromisoformat(local[:19])
                 - datetime.datetime.fromisoformat(gmt[:19]))
    except ValueError:
        return None
    return delta.total_seconds() / 3600


def _utc_offset_from_events(date):
    try:
        for e in garmin_client.get_body_battery_events(date) or []:
            off = (e.get("event") or {}).get("timezoneOffset") if isinstance(e, dict) else None
            if isinstance(off, (int, float)):
                return off / 3600000
    except Exception:
        return None
    return None


def _energy_windows(curve):
    """Best and worst 2-hour windows (adjacent hours averaged), falling back
    to single hours when fewer than two adjacent pairs were recorded — one
    pair would make the peak and the trough the same window."""
    pairs = [
        (a["hour"], round((a["level"] + b["level"]) / 2))
        for a, b in zip(curve, curve[1:]) if b["hour"] == a["hour"] + 1
    ]
    if len(pairs) >= 2:
        best, worst = max(pairs, key=lambda p: p[1]), min(pairs, key=lambda p: p[1])
        return (
            {"from_hour": best[0], "to_hour": best[0] + 2, "level": best[1]},
            {"from_hour": worst[0], "to_hour": worst[0] + 2, "level": worst[1]},
        )
    if len(curve) >= 2:
        best = max(curve, key=lambda c: c["level"])
        worst = min(curve, key=lambda c: c["level"])
        return (
            {"from_hour": best["hour"], "to_hour": best["hour"] + 1, "level": best["level"]},
            {"from_hour": worst["hour"], "to_hour": worst["hour"] + 1, "level": worst["level"]},
        )
    return None, None


def _plan_summary(p, today):
    """One enrolled Garmin plan: identity, status, dates, and where today
    falls in it. The list endpoint returns the athlete's own enrolments
    (scheduled, in progress and completed), not the catalogue."""
    start = (p.get("startDate") or "")[:10] or None
    end = (p.get("endDate") or "")[:10] or None
    out = {
        "id": p.get("trainingPlanId"),
        "name": p.get("name"),
        "category": p.get("trainingPlanCategory"),
        "adaptive": "ADAPTIVE" in (p.get("trainingPlanCategory") or "") or None,
        "status": (p.get("trainingStatus") or {}).get("statusKey"),
        "sport": (p.get("trainingType") or {}).get("typeKey"),
        "version": (p.get("trainingVersion") or {}).get("versionName"),
        "weeks": p.get("durationInWeeks"),
        "avg_weekly_workouts": p.get("avgWeeklyWorkouts"),
        "start_date": start,
        "end_date": end,
    }
    try:
        if end:
            out["days_to_end"] = (datetime.date.fromisoformat(end) - today).days
        if start and datetime.date.fromisoformat(start) <= today:
            out["current_week"] = (today - datetime.date.fromisoformat(start)).days // 7 + 1
    except ValueError:
        pass
    return _drop_none(out)


def _adaptive_plan_detail(plan_id, today):
    """Phases and the upcoming sessions of an adaptive (Garmin Coach) plan.

    Garmin generates an adaptive plan only about a week ahead and drops
    sessions once they pass, so this is the whole of what exists: there are
    no future weeks to show and no history to go back to.
    """
    raw = garmin_client.get_adaptive_training_plan_by_id(plan_id) or {}
    phases = [
        _drop_none({
            "phase": ph.get("trainingPhase"),
            "start": ph.get("startDate"),
            "end": ph.get("endDate"),
            "current": ph.get("currentPhase") or None,
        })
        for ph in raw.get("adaptivePlanPhases") or [] if isinstance(ph, dict)
    ]
    upcoming = []
    for task in raw.get("taskList") or []:
        if not isinstance(task, dict) or (task.get("calendarDate") or "") < today.isoformat():
            continue
        w = task.get("taskWorkout") or {}
        upcoming.append(_drop_none({
            "date": task.get("calendarDate"),
            "sport": (w.get("sportType") or {}).get("sportTypeKey"),
            "name": w.get("workoutName"),
            "target": w.get("workoutDescription"),
            "est_min": _minutes(w.get("estimatedDurationInSecs")),
            "effect": w.get("trainingEffectLabel") if w.get("trainingEffectLabel") != "INVALID" else None,
            "long": task.get("longWkt") or None,
            "rest_day": w.get("restDay") or None,
            "status": w.get("adaptiveCoachingWorkoutStatus"),
            "workout_uuid": w.get("workoutUuid"),
        }))
    upcoming.sort(key=lambda t: t.get("date") or "")
    return _drop_none({"phases": phases or None, "upcoming": upcoming or None})


def _vo2_as_of(end, then):
    """VO2 max as of `end` and as of `then`, carrying the last measurement
    forward.

    Garmin writes a VO2 value only on days it recomputes one (after a
    qualifying activity), so a single-day lookup is usually empty and the
    4-week change came back None with no reason. One range call covers both
    dates; each takes the newest measurement on or before it.
    """
    start = (then - datetime.timedelta(days=60)).isoformat()
    supported, data = _get_max_metrics_range(garmin_client, start, end.isoformat())
    if supported:
        by_date = _extract_dated_vo2_measurements(data)
    else:
        # No range endpoint on this client: the two single days, as before.
        by_date = {}
        for day in (then, end):
            payload = garmin_client.get_max_metrics(day.isoformat()) or []
            item = payload[0] if isinstance(payload, list) and payload else {}
            generic = (item.get("generic") or {}) if isinstance(item, dict) else {}
            v = generic.get("vo2MaxPreciseValue") or generic.get("vo2MaxValue")
            if isinstance(v, (int, float)):
                by_date[day.isoformat()] = {"running": v}

    def _latest(day, sport):
        best = None
        for d, m in by_date.items():
            if d <= day.isoformat() and sport in m and (best is None or d > best[0]):
                best = (d, m[sport])
        return best

    sport = "running" if any("running" in m for m in by_date.values()) else "cycling"
    return _latest(end, sport), _latest(then, sport), sport


# Concurrent Garmin calls per trend. The trend makes three calls per session;
# run serially that is ~11 s for ten runs. Four in flight cuts it to a few
# seconds without looking like a burst to Garmin's rate limiter.
_TREND_WORKERS = 4


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
                    # The on-watch recovery clock, from the same payload.
                    "recovery_hours": (
                        _recovery_from_readiness(readiness_list, "training_readiness") or {}
                    ).get("remaining_hours"),
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
                    "sleep_need_min": (dto.get("sleepNeed") or {}).get("actual"),
                    "sleep_debt_min": (
                        round((dto.get("sleepNeed") or {}).get("actual") - duration / 60)
                        if isinstance((dto.get("sleepNeed") or {}).get("actual"), (int, float))
                        and isinstance(duration, (int, float)) else None
                    ),
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

    @app.tool()
    async def get_coach_report(end_date: str, weeks: int = 6) -> str:
        """Deep one-call training analysis for a dedicated coaching pass:
        today's readiness/sleep/HRV/body-battery, adherence over the last 14
        days (days_since_last_session, session counts by type, recent
        sessions), training-load position vs Garmin's optimal band WITH
        direction over 2 and 4 weeks (rising/falling/flat TSB and acute load),
        and fitness trajectory (VO2 max change over ~4 weeks, endurance score,
        HRV weekly average vs baseline). Heavier than get_wellness_brief —
        intended for a once-daily coach run that reasons about trends, not
        just today. Compare its numbers against the athlete's block targets.

        Args:
            end_date: Analysis date (usually today) in YYYY-MM-DD format
            weeks: Trend lookback window for the endurance/fitness range (default 6)
        """
        end = datetime.date.fromisoformat(end_date)
        d14 = (end - datetime.timedelta(days=13)).isoformat()
        d2w = (end - datetime.timedelta(days=14)).isoformat()
        d4w = (end - datetime.timedelta(days=28)).isoformat()
        window_start = (end - datetime.timedelta(weeks=weeks)).isoformat()
        report = {"as_of": end_date}
        errors = {}

        try:
            r = next((e for e in (garmin_client.get_training_readiness(end_date) or []) if isinstance(e, dict)), {})
            report["readiness"] = _drop_none(
                {"score": r.get("score"), "level": r.get("level"), "feedback": r.get("feedbackShort")}
            )
        except Exception as e:
            errors["readiness"] = str(e)

        try:
            sleep = garmin_client.get_sleep_data(end_date) or {}
            dto = sleep.get("dailySleepDTO") or {}
            dur = dto.get("sleepTimeSeconds")
            report["sleep_last_night"] = _drop_none(
                {
                    "score": ((dto.get("sleepScores") or {}).get("overall") or {}).get("value"),
                    "duration_h": _round(dur / 3600, 2) if isinstance(dur, (int, float)) else None,
                    "resting_hr": sleep.get("restingHeartRate"),
                    "overnight_hrv_ms": sleep.get("avgOvernightHrv"),
                }
            )
        except Exception as e:
            errors["sleep"] = str(e)

        try:
            hrv = (garmin_client.get_hrv_data(end_date) or {}).get("hrvSummary") or {}
            baseline = hrv.get("baseline") or {}
            report["hrv"] = _drop_none(
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
            bb = next((d for d in (garmin_client.get_body_battery(end_date, end_date) or []) if isinstance(d, dict)), {})
            report["body_battery"] = _drop_none({"current_level": _current_body_battery_level(bb)})
        except Exception as e:
            errors["body_battery"] = str(e)

        try:
            sessions = _sessions_between(d14, end_date)
            last_7 = [s for s in sessions if s.get("date", "") >= (end - datetime.timedelta(days=6)).isoformat()]
            by_type = {}
            for s in sessions:
                key = s.get("type") or "unknown"
                by_type[key] = by_type.get(key, 0) + 1
            report["adherence"] = _drop_none(
                {
                    "days_since_last_session": _days_since_last_session(sessions, end_date),
                    "last_7d_sessions": len(last_7),
                    "last_14d_sessions": len(sessions),
                    "by_type_14d": by_type or None,
                    "recent_sessions": sessions[-5:],
                    "note": None if sessions else "no sessions recorded in the last 14 days",
                }
            )
        except Exception as e:
            errors["adherence"] = str(e)

        try:
            now = _load_snapshot(end_date)
            ago2 = _load_snapshot(d2w)
            ago4 = _load_snapshot(d4w)
            report["load"] = _drop_none(
                {
                    "now": now,
                    "2w_ago": _drop_none({"acute_load": ago2.get("acute_load"), "tsb": ago2.get("tsb")}) or None,
                    "4w_ago": _drop_none({"acute_load": ago4.get("acute_load"), "tsb": ago4.get("tsb")}) or None,
                    "acute_direction_4w": _direction(now.get("acute_load"), ago4.get("acute_load")),
                    "tsb_direction_4w": _direction(now.get("tsb"), ago4.get("tsb")),
                }
            )
        except Exception as e:
            errors["load"] = str(e)

        fitness = {}
        try:
            now_m, past_m, sport = _vo2_as_of(end, end - datetime.timedelta(days=28))
            now_v, past_v = (now_m or (None, None))[1], (past_m or (None, None))[1]
            fitness.update({
                "vo2max_now": _round(now_v),
                "vo2max_measured_on": now_m[0] if now_m else None,
                "vo2max_4w_ago": _round(past_v),
                "vo2max_change": _round(now_v - past_v, 1) if isinstance(now_v, (int, float)) and isinstance(past_v, (int, float)) else None,
                "vo2max_sport": sport if now_m else None,
                "vo2max_note": None if now_m and past_m else "no VO2 max measurement in the 60 days before one of the two dates",
            })
        except Exception as e:
            errors["vo2max"] = str(e)
        try:
            score_dto = (garmin_client.get_endurance_score(window_start, end_date) or {}).get("enduranceScoreDTO") or {}
            fitness["endurance_score"] = score_dto.get("overallScore") or score_dto.get("score")
        except Exception as e:
            errors["endurance_score"] = str(e)
        if _drop_none(fitness):
            report["fitness"] = _drop_none(fitness)

        if errors:
            report["errors"] = errors
        return json.dumps(report, indent=2)

    @app.tool()
    async def get_session_analysis(
        date: str = "",
        activity_id: int = 0,
        lthr: int = 0,
        easy_ceiling_bpm: int = 0,
        threshold_test: bool = False,
    ) -> str:
        """Objective execution quality for one training session: HR time-in-zones
        with zone boundaries (easy vs hard share — did an "easy" run stay
        aerobic?), POWER time-in-zones (no cardiac lag or heat inflation — the
        cross-check when HR looks hot), the athlete's own RPE/feel rating vs
        the objective read, per-split pacing with fade/negative-split
        detection, cadence + stride length, session cost (stamina, body
        battery, sweat), terrain with grade-adjusted pace, and set/rep detail
        for strength sessions. Pass a `date` (analyses that day's longest
        activity; the others are listed in `other_activities_that_day`) or an
        explicit `activity_id`. Use to audit whether a session was executed as
        intended — the key discipline for aerobic-base building (easy runs
        must actually be easy).

        `hr_ceiling` (with `easy_ceiling_bpm`) is the zone-model-free check:
        time at or under the plan's own HR cap and the time-weighted average
        HR, straight from the raw stream. Prefer it when the plan has a cap.

        `hr_zones` carries Garmin's STORED zone times, frozen into the activity
        at upload. `hr_zones.recomputed` re-integrates the raw HR stream (timer
        time) against the CURRENT zone model for the activity's sport;
        `recomputed.applied` names that model — `floors_basis` says what the
        floors really are built on and `method_mismatch` flags a config whose
        label disagrees with its floors (Garmin has served 50–90% of max HR
        under a LACTATE_THRESHOLD label). `recomputed.verification` re-scores
        the same stream with the activity's own frozen bands; when it
        reproduces the stored share the arithmetic is trustworthy.
        Activities with no HR stream say so in `not_recomputable`.

        `aerobic_decoupling` (runs only) is Pa:HR: efficiency (speed per
        heartbeat) of the second half against the first, warm-up skipped on
        runs of 50+ minutes. Under 5% = durable aerobic engine, 5-8% = a
        moderate fade, over 8% = too hard or the back half slipped. Only a
        read on a steady effort: when the pace varied too much (intervals,
        hills) `steady` is false and the figure is not a durability read.
        Absent for runs under ~20 usable minutes or with no speed stream.

        Args:
            date: YYYY-MM-DD to analyse that day's longest activity (optional)
            activity_id: explicit Garmin activity id (optional; overrides date)
            lthr: score zones on %LTHR bands built from this threshold instead
                of Garmin's live config (optional) — e.g. a field-tested LTHR
            easy_ceiling_bpm: the coaching plan's easy HR cap (optional); adds
                `hr_ceiling`
            threshold_test: read this activity as a 30-minute LTHR field test
                (optional); adds `threshold_test` — the hardest 30-minute
                window, LTHR = average HR of its last 20 minutes, threshold
                pace, and the %LTHR zone floors that LTHR implies
        """
        def run():
            result = {}
            errors = {}
            try:
                summary, aid, note, others = _resolve_activity(activity_id, date)
                if note:
                    return {"note": note}
                result["session"] = _drop_none(_summarize_session(summary))
                result["activity_id"] = aid
                if others:
                    result["other_activities_that_day"] = others
            except Exception as e:
                return {"error": str(e)}

            form = _form_metrics(summary)
            result["form"] = _drop_none({
                "cadence_spm": form["cadence_spm"],
                "stride_length_m": form["stride_length_m"],
            }) or None

            dto = _full_activity(summary, aid)
            result["effort"] = _drop_none(_effort_rating(dto)) or None
            result["cost"] = _drop_none(_session_cost(dto)) or None
            result["terrain"] = _terrain(dto)

            rows, stream_error = _activity_stream(aid)
            samples = [(r["t"], r["hr"]) for r in rows] if rows else None
            if easy_ceiling_bpm:
                result["hr_ceiling"] = (
                    _ceiling_share(samples, easy_ceiling_bpm) if samples else None
                ) or {"not_recomputable": stream_error or "no HR time recorded"}

            stored_zones = None
            try:
                stored_zones = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
            except Exception as e:
                errors["hr_zones"] = str(e)
            # Recompute against the live zone model. Kept beside the stored
            # numbers, never on top of them: the comparison is what shows the
            # correction landed, and overwriting would hide a bad zone config.
            model, model_error = _hr_zone_model(
                lthr or None, _zone_sport(result["session"].get("type"))
            )
            if not model:
                recomputed = {"not_recomputable": model_error}
            elif samples is None:
                recomputed = {"not_recomputable": stream_error}
            else:
                recomputed = _recompute_hr_zones(aid, model, stored_zones, samples=samples)
            if stored_zones:
                stored_zones["recomputed"] = recomputed
                result["hr_zones"] = stored_zones
            else:
                result["hr_zones"] = {"recomputed": recomputed}

            if rows and "running" in (result["session"].get("type") or ""):
                result["aerobic_decoupling"] = _aerobic_decoupling(rows)

            if threshold_test:
                garmin_lthr = model and (model.get("lthr_configured") or model.get("lthr"))
                result["threshold_test"] = (
                    _threshold_test(rows, garmin_lthr=garmin_lthr) if rows
                    else {"not_applicable": stream_error}
                )

            try:
                result["power_zones"] = _zone_distribution(
                    garmin_client.get_activity_power_in_timezones(aid), boundary_key="low_w"
                )
            except Exception as e:
                errors["power_zones"] = str(e)
            if "strength" in (result["session"].get("type") or ""):
                try:
                    result["strength_sets"] = _summarize_exercise_sets(
                        garmin_client.get_activity_exercise_sets(aid)
                    )
                except Exception as e:
                    errors["strength_sets"] = str(e)
            try:
                result["pacing"] = _pacing(_lap_form_rows(garmin_client.get_activity_splits(aid)))
            except Exception as e:
                errors["pacing"] = str(e)
            try:
                w = garmin_client.get_activity_weather(aid)
                w = w if isinstance(w, dict) else {}
                temp_c = round((w["temp"] - 32) * 5 / 9) if isinstance(w.get("temp"), (int, float)) else None
                result["weather"] = _drop_none({
                    "temp_c": temp_c,
                    "humidity_pct": w.get("relativeHumidity"),
                    # heat inflates HR — flag so a warm run isn't misread as too-hard
                    "heat_note": "warm — HR runs higher, discount the zone read" if isinstance(temp_c, (int, float)) and temp_c >= 22 else None,
                }) or None
            except Exception as e:
                errors["weather"] = str(e)

            if errors:
                result["errors"] = errors
            return _drop_none(result)

        return json.dumps(await asyncio.to_thread(run), indent=2)

    @app.tool()
    async def get_running_dynamics(activity_id: int = 0, date: str = "") -> str:
        """Running-form metrics for one session: average/max cadence, stride
        length, vertical oscillation + ratio, ground contact time and L/R
        balance, running power (avg + normalized), a per-lap cadence and
        stride breakdown with a first-third vs last-third fatigue comparison
        (does stride collapse late?), the session's cost (stamina depletion,
        body-battery drain, sweat loss), terrain with grade-adjusted pace,
        the athlete's own RPE/feel rating, and Garmin's performance-condition
        curve (start vs end — a late slide is a durability signal). Fields
        the device didn't record are null — wrist-based dynamics never report
        L/R balance. Pass an explicit `activity_id`, a `date` (that day's
        main activity), or neither (most recent activity). Use alongside
        get_session_analysis when the question is form/economy rather than
        effort.

        Args:
            activity_id: explicit Garmin activity id (optional; overrides date)
            date: YYYY-MM-DD to analyse that day's main activity (optional)
        """
        result = {}
        errors = {}
        try:
            summary, aid, note, others = _resolve_activity(activity_id, date)
            if note:
                return json.dumps({"note": note}, indent=2)
            result["session"] = _drop_none(_summarize_session(summary))
            result["activity_id"] = aid
            if others:
                result["other_activities_that_day"] = others
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

        # Nulls stay visible here on purpose: "not recorded" is an answer.
        form = _form_metrics(summary)
        if form["gct_balance_pct"] is None:
            form["gct_balance_note"] = "L/R balance needs a chest strap or RD pod; wrist dynamics don't record it"
        result["form"] = form

        dto = _full_activity(summary, aid)
        result["effort"] = _drop_none(_effort_rating(dto)) or None
        result["cost"] = _drop_none(_session_cost(dto)) or None
        result["terrain"] = _terrain(dto)

        try:
            result["performance_condition"] = _performance_condition(
                garmin_client.get_activity_details(aid, maxchart=200, maxpoly=100)
            )
        except Exception as e:
            errors["performance_condition"] = str(e)

        try:
            rows = _lap_form_rows(garmin_client.get_activity_splits(aid))
            if rows:
                result["by_lap"] = rows
                result["fatigue"] = _form_fatigue(rows)
        except Exception as e:
            errors["laps"] = str(e)

        if errors:
            result["errors"] = errors
        return json.dumps(result, indent=2)

    @app.tool()
    async def get_execution_trend(
        count: int = 10,
        activity_type: str = "running",
        lthr: int = 0,
        recompute_zones: bool = True,
        easy_ceiling_bpm: int = 0,
    ) -> str:
        """Execution quality ACROSS the last N sessions of one type — surfaces
        patterns a single-session view misses: how many runs were genuinely
        easy vs 'grey zone' vs hard, the average easy-share, per-run cadence +
        stride length (form drift across weeks), per-run power easy-share and
        the athlete's own RPE (perception vs objective effort — the
        calibration gap), and monotony (are they all the same distance?). Use
        for weekly review and to diagnose training distribution (e.g. the
        classic 'every run is moderately hard' base-building failure).

        With `easy_ceiling_bpm` (the plan's HR cap) each run gets
        `under_ceiling_pct` — time at or under the cap from the raw stream —
        and the easy/grey/hard grading uses it (`execution_basis` says so).
        That grading does not depend on Garmin's zone model at all.

        Each run carries `pace_min_per_km` and `efficiency_factor` (speed in
        m/s per bpm, x1000: the same fitness gives the same number at any easy
        heart rate). Compare it only between runs of similar length and route.

        Each steady run also gets `decoupling_pct` (Pa:HR: second-half
        efficiency against the first; under 5% = durable aerobic engine) and
        the batch gets `aerobic_decoupling` (average, runs under 5%) — the
        cleanest "is the base rebuilding" signal across a block. Runs with
        varied pace or no speed stream are left out rather than scored.

        Zone easy-share is recomputed from each session's raw HR stream
        against the CURRENT zone model for the sport by default, because
        Garmin freezes zone times into an activity at upload and a stored
        series spanning a zone-model change must never be averaged across.
        `zone_model` names the model applied, including `floors_basis` and any
        `method_mismatch`; `zone_model_stamps` groups sessions by the bands
        they were stamped with; sessions with no stream are listed in
        `not_recomputable` and left out of the average.

        Args:
            count: number of recent sessions to analyse (default 10; ~3 API
                calls per session, four in flight)
            activity_type: typeKey filter, e.g. running / cycling (default running)
            lthr: score zones on %LTHR bands built from this threshold instead
                of Garmin's live config (optional)
            recompute_zones: recompute zone time from raw HR (default True).
                False uses Garmin's stored per-activity zone times — cheaper,
                not comparable across a zone-model change
            easy_ceiling_bpm: the coaching plan's easy HR cap (optional)
        """
        def run():
            scan = max(20, min(count * 4, 100))
            try:
                acts = garmin_client.get_activities(0, scan) or []
            except Exception as e:
                return {"error": str(e)}
            acts = [a for a in acts if isinstance(a, dict)
                    and activity_type in (((a.get("activityType") or {}).get("typeKey")) or "")]
            found = len(acts)
            acts = acts[:count]

            # One zone-config fetch for the whole batch — the model is
            # per-athlete and per-sport, not per-activity.
            model, model_error = (None, "recompute_zones=False")
            if recompute_zones:
                model, model_error = _hr_zone_model(lthr or None, _zone_sport(activity_type))

            def one(a):
                aid = a.get("activityId")
                dist = a.get("distance")
                form = _form_metrics(a)
                row = {
                    "date": (a.get("startTimeLocal") or "")[:10],
                    "km": round(dist / 1000, 1) if isinstance(dist, (int, float)) else None,
                    "avg_hr": a.get("averageHR"),
                    "cadence_spm": form["cadence_spm"],
                    "stride_length_m": form["stride_length_m"],
                    "power_easy_share_pct": _power_share_from_list(a),
                    "aerobic_te": round(a["aerobicTrainingEffect"], 1) if isinstance(a.get("aerobicTrainingEffect"), (int, float)) else None,
                }
                # Pace and efficiency from the list item (no extra call). The factor
                # is speed (m/s) per bpm x 1000, so the same fitness reads the same
                # number at any easy HR: the trend that shows an aerobic base
                # building, where raw pace moves with how hard the run was.
                spd, hr = a.get("averageSpeed"), a.get("averageHR")
                if isinstance(spd, (int, float)) and spd > 0:
                    row["pace_min_per_km"] = round(1000 / spd / 60, 2)
                    if isinstance(hr, (int, float)) and hr > 0:
                        row["efficiency_factor"] = round(spd / hr * 1000, 1)
                try:
                    row["rpe_10"] = _effort_rating(garmin_client.get_activity(aid) or {})["rpe_10"]
                except Exception:
                    pass
                stamp = None
                try:
                    z = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
                    row["easy_share_pct"] = z["easy_share_pct"] if z else None
                    # The bands this session was stamped with, for the drift report.
                    stamp = (row["date"], _stored_floors(z))
                except Exception:
                    pass
                samples, reason = (None, None)
                if model or easy_ceiling_bpm:
                    samples, reason = _hr_stream(aid, _HR_STREAM_SAMPLES_TREND)
                    if samples is not None:
                        # Same cached rows as _hr_stream: no extra API call. Only a
                        # steady effort gets a figure — intervals and hills would
                        # drag the trend with a number that means nothing.
                        dec = _aerobic_decoupling(_activity_stream(aid, _HR_STREAM_SAMPLES_TREND)[0])
                        if dec and dec["steady"]:
                            row["decoupling_pct"] = dec["decoupling_pct"]
                # The graded zone share: recomputed when we can, stored
                # otherwise, but never a mix — one basis is the point.
                if model:
                    zone_share = None
                    if samples is not None:
                        rec = _recompute_hr_zones(aid, model, samples=samples)
                        if "not_recomputable" in rec:
                            reason = rec["not_recomputable"]
                        else:
                            zone_share = rec["easy_share_pct"]
                            row["easy_share_recomputed_pct"] = zone_share
                else:
                    zone_share = row.get("easy_share_pct")
                if easy_ceiling_bpm and samples is not None:
                    c = _ceiling_share(samples, easy_ceiling_bpm)
                    row["under_ceiling_pct"] = c["under_pct"] if c else None
                graded = row.get("under_ceiling_pct") if easy_ceiling_bpm else zone_share
                row["execution"] = ("unknown" if graded is None
                                    else "easy" if graded >= 65
                                    else "hard" if graded < 35 else "grey")
                failed = None
                if model and "easy_share_recomputed_pct" not in row:
                    failed = reason
                elif easy_ceiling_bpm and samples is None:
                    failed = reason
                return _drop_none(row), stamp, (str(aid), failed) if failed else None

            with ThreadPoolExecutor(max_workers=_TREND_WORKERS) as pool:
                results = list(pool.map(one, acts))

            runs = [r for r, _, _ in results]
            stamps = [s for _, s, _ in results if s]
            not_recomputable = dict(f for _, _, f in results if f)
            buckets = {"easy": 0, "grey": 0, "hard": 0, "unknown": 0}
            for r in runs:
                buckets[r["execution"]] += 1
            share_key = "easy_share_recomputed_pct" if model else "easy_share_pct"
            graded = [r for r in runs if r.get(share_key) is not None]
            avg_easy = round(fmean(r[share_key] for r in graded)) if graded else None
            dists = [r["km"] for r in runs if r.get("km")]
            decs = [r["decoupling_pct"] for r in runs if r.get("decoupling_pct") is not None]
            decoupling = _drop_none({
                "avg_pct": round(fmean(decs), 1) if decs else None,
                "runs_under_5pct": f"{sum(1 for d in decs if d < 5)} of {len(decs)}" if decs else None,
                "basis": "Pa:HR on steady runs of ~20+ usable minutes; under 5% = durable aerobic engine"
                         if decs else None,
            }) or None
            ceiling = None
            if easy_ceiling_bpm:
                under = [r["under_ceiling_pct"] for r in runs if r.get("under_ceiling_pct") is not None]
                with_hr = [r for r in runs if isinstance(r.get("avg_hr"), (int, float))]
                ceiling = _drop_none({
                    "bpm": easy_ceiling_bpm,
                    "runs_avg_hr_at_or_under": f"{sum(1 for r in with_hr if r['avg_hr'] <= easy_ceiling_bpm)} of {len(with_hr)}",
                    "avg_under_pct": round(fmean(under)) if under else None,
                })
            return _drop_none({
                "analysed": len(runs),
                "note": (
                    f"only {found} {activity_type} sessions among the last {scan} activities"
                    if found < count else None
                ),
                "activity_type": activity_type,
                "distribution": buckets,
                "execution_basis": (
                    f"time at or under {easy_ceiling_bpm} bpm (the plan's cap)"
                    if easy_ceiling_bpm else "zone easy-share (Z1-2)"
                ),
                "ceiling": ceiling,
                "aerobic_decoupling": decoupling,
                "avg_easy_share_pct": avg_easy,
                "easy_share_basis": (
                    "recomputed from raw HR against the current zone model"
                    if model else
                    f"Garmin's stored per-activity zone times ({model_error}) — "
                    "NOT comparable across a zone-model change"
                ),
                "zone_model": model and _model_summary(model),
                "not_recomputable": not_recomputable or None,
                # A zone-model change is a step, invisible in any one activity.
                # Listed here it names the date it happened and how many
                # sessions still carry the old bands.
                "zone_model_stamps": _stamp_history(
                    stamps, model["floors"] if model else None
                ),
                "power_model": _power_model_drift(acts),
                "monotony": _drop_none({
                    "distinct_distances": len(set(dists)) if dists else None,
                    "distance_range_km": [min(dists), max(dists)] if dists else None,
                }),
                "sessions": runs,
            })

        return json.dumps(await asyncio.to_thread(run), indent=2)

    @app.tool()
    async def get_energy_curve(date: str) -> str:
        """Intraday body-battery trajectory for energy-aware scheduling: hourly
        energy levels, the peak window (best for demanding cognitive work) and
        the trough (protect / recover), current level, and readiness. Use to
        place deep work (e.g. LeetCode) at the energy peak and light/admin work
        in the trough.

        Args:
            date: YYYY-MM-DD (usually today)
        """
        out = {"date": date}
        errors = {}
        try:
            bb = next((d for d in (garmin_client.get_body_battery(date, date) or []) if isinstance(d, dict)), {})
            offset_h = _utc_offset_hours(bb)
            if offset_h is None:
                offset_h = _utc_offset_from_events(date)
            if offset_h is None:
                offset_h = 0
                out["timezone_note"] = "UTC offset unknown for this day — hours are UTC"
            idx = _descriptor_index(bb.get("bodyBatteryValueDescriptorDTOList"), "bodyBatteryLevel", 2)
            hourly = {}
            for row in bb.get("bodyBatteryValuesArray") or []:
                if isinstance(row, (list, tuple)) and len(row) > idx and isinstance(row[0], (int, float)) and isinstance(row[idx], (int, float)):
                    hour = int(((row[0] / 1000) + offset_h * 3600) // 3600 % 24)
                    hourly.setdefault(hour, []).append(row[idx])
            curve = [{"hour": h, "level": round(fmean(v))} for h, v in sorted(hourly.items())]
            out["current_level"] = _current_body_battery_level(bb)
            out["hourly"] = curve
            peak, trough = _energy_windows(curve)
            if peak:
                out["peak_window"] = peak
                out["trough_window"] = trough
        except Exception as e:
            errors["body_battery"] = str(e)
        try:
            r = next((e for e in (garmin_client.get_training_readiness(date) or []) if isinstance(e, dict)), {})
            out["readiness"] = _drop_none({"score": r.get("score"), "level": r.get("level")})
        except Exception as e:
            errors["readiness"] = str(e)
        if errors:
            out["errors"] = errors
        return json.dumps(out, indent=2)

    @app.tool()
    async def get_health_flags(date: str) -> str:
        """Early-warning fusion of illness/overtraining signals for one date:
        resting HR vs 7-day average, overnight HRV vs personal band, sleep debt
        (slept vs Garmin's personalized need), respiration, SpO2 (if tracked),
        and morning body battery. Returns severity green/amber/red plus which
        signals are off. Use to catch a cold or over-reach before it's felt.

        Args:
            date: YYYY-MM-DD (usually today)
        """
        signals = {}
        flags = []
        errors = {}
        try:
            hr = garmin_client.get_heart_rates(date) or {}
            rhr, avg7 = hr.get("restingHeartRate"), hr.get("lastSevenDaysAvgRestingHeartRate")
            signals["resting_hr"] = rhr
            signals["resting_hr_7d_avg"] = avg7
            if isinstance(rhr, (int, float)) and isinstance(avg7, (int, float)) and rhr - avg7 >= 5:
                flags.append(f"resting HR {rhr} is {rhr - avg7} over the 7-day average")
        except Exception as e:
            errors["heart_rate"] = str(e)
        try:
            hv = (garmin_client.get_hrv_data(date) or {}).get("hrvSummary") or {}
            base = hv.get("baseline") or {}
            last, low, wk = hv.get("lastNightAvg"), base.get("balancedLow"), hv.get("weeklyAvg")
            signals["hrv_last_night"] = last
            signals["hrv_status"] = hv.get("status")
            if isinstance(last, (int, float)) and isinstance(low, (int, float)) and last < low:
                flags.append(f"overnight HRV {last}ms below the balanced band ({low}ms)")
            elif isinstance(last, (int, float)) and isinstance(wk, (int, float)) and last < 0.85 * wk:
                flags.append(f"overnight HRV {last}ms well under the weekly average ({wk}ms)")
        except Exception as e:
            errors["hrv"] = str(e)
        try:
            sl = garmin_client.get_sleep_data(date) or {}
            dto = sl.get("dailySleepDTO") or {}
            slept = dto.get("sleepTimeSeconds")
            need = (dto.get("sleepNeed") or {}).get("actual")  # minutes
            if isinstance(slept, (int, float)) and isinstance(need, (int, float)):
                debt_min = round(need - slept / 60)
                signals["sleep_debt_min"] = debt_min
                if debt_min >= 60:
                    flags.append(f"slept {debt_min} min under your need last night")
        except Exception as e:
            errors["sleep"] = str(e)
        try:
            resp = garmin_client.get_respiration_data(date) or {}
            signals["waking_respiration"] = resp.get("avgWakingRespirationValue")
        except Exception:
            pass
        try:
            spo2 = (garmin_client.get_spo2_data(date) or {}).get("averageSpO2")
            if isinstance(spo2, (int, float)):
                signals["avg_spo2"] = spo2
                if spo2 < 90:
                    flags.append(f"average SpO2 {spo2}% is low")
        except Exception:
            pass
        signals = _drop_none(signals)
        # Don't declare "green" when nothing was evaluable (e.g. today unsynced).
        evaluated = any(k in signals for k in ("resting_hr", "hrv_last_night", "sleep_debt_min"))
        if not evaluated:
            result = {"date": date, "severity": "unknown", "flags": [],
                      "note": "today's metrics haven't synced — re-check later or use yesterday", "signals": signals}
        else:
            severity = "red" if len(flags) >= 2 else "amber" if len(flags) == 1 else "green"
            result = {"date": date, "severity": severity, "flags": flags, "signals": signals}
        if errors:
            result["errors"] = errors
        return json.dumps(result, indent=2)

    @app.tool()
    async def get_sleep_regularity(end_date: str, days: int = 14) -> str:
        """Sleep TIMING consistency over the last N nights: how much bedtime,
        wake time and mid-sleep move night to night, the weekday-vs-weekend
        shift (social jet lag), and the Sleep Regularity Index. Duration and
        score say how much and how well he slept; this says whether the clock
        was steady, which has its own link to recovery and health.

        Times are the watch's local wall-clock, so a trip across timezones is
        not misread as a shifted night, and a night belongs to its wake-up
        date. `sleep_regularity_index` (Phillips 2017) is the chance of being
        in the same sleep/wake state 24 h apart: 100 = identical every day,
        lower = less regular; it is the onset/offset version (naps and
        mid-night wakings are not in it) and is only a read against his own
        earlier fortnight, so compare calls rather than quote one. Rule of
        thumb used for the reads: mid-sleep SD under ~30 min is tight, over
        ~60 min is irregular; a weekend shift over 1 h (Wittmann 2006) is
        social jet lag. Nights without a recorded sleep window are listed in
        `missing_nights`; under 5 usable nights nothing is scored.

        One Garmin request per night (4 in flight); the default 14 is about
        14 calls.

        Args:
            end_date: YYYY-MM-DD, the last wake-up date (usually today)
            days: nights to look back, 7-30 (default 14)
        """
        def run():
            try:
                end = datetime.date.fromisoformat(end_date)
            except ValueError as e:
                return {"error": f"Invalid date: {e}. Use YYYY-MM-DD."}
            n_days = max(7, min(int(days), 30))
            dates = [(end - datetime.timedelta(days=i)).isoformat() for i in range(n_days)]

            def one(d):
                try:
                    dto = (garmin_client.get_sleep_data(d) or {}).get("dailySleepDTO") or {}
                except Exception as e:
                    return d, None, str(e)
                s, e = dto.get("sleepStartTimestampLocal"), dto.get("sleepEndTimestampLocal")
                if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                    return d, None, None
                return d, {"date": _sleep_day(e).isoformat(), "start_ms": s, "end_ms": e}, None

            with ThreadPoolExecutor(max_workers=_TREND_WORKERS) as pool:
                results = list(pool.map(one, dates))
            nights = [n for _, n, _ in results if n]
            missing = sorted(d for d, n, _ in results if n is None)
            errors = {d: err for d, n, err in results if err}
            out = {"window": {"start": dates[-1], "end": dates[0]}, "nights_requested": n_days}
            scored = _sleep_regularity(nights)
            if scored is None:
                out["note"] = (f"only {len(nights)} night(s) with a recorded sleep window — "
                               f"need {_SLEEP_MIN_NIGHTS}+ to score regularity")
            else:
                out.update(scored)
                reads = []
                sd = scored.get("midpoint_sd_min")
                if sd is not None:
                    reads.append("mid-sleep varies by ~%d min night to night — %s" % (
                        sd, "tight" if sd < 30 else "irregular" if sd > 60 else "moderate"))
                sjl = scored.get("social_jet_lag")
                if sjl:
                    reads.append("weekend sleep sits %.1f h %s weekday sleep%s" % (
                        abs(sjl["shift_h"]), "later than" if sjl["shift_h"] > 0 else "earlier than",
                        " — social jet lag" if sjl["flag"] else ""))
                out["read"] = reads
            if missing:
                out["missing_nights"] = missing
            if errors:
                out["errors"] = errors
            return out

        return json.dumps(await asyncio.to_thread(run), indent=2)

    @app.tool()
    async def get_wins(days: int = 14) -> str:
        """Recent wins for motivation/adherence: personal records set, badges
        earned, current race predictions, and the recorded-activity streak over
        the window. Use to celebrate progress in a briefing.

        Args:
            days: lookback window in days (default 14)
        """
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        out = {"window_days": days}
        errors = {}
        try:
            prs = garmin_client.get_personal_record() or []
            recent = []
            for p in prs:
                ts = p.get("activityStartDateTimeInGMT")
                d = None
                if isinstance(ts, (int, float)):
                    d = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).date()
                if d and d >= cutoff:
                    recent.append(_drop_none({"activity": p.get("activityName"), "type": p.get("activityType"), "date": str(d)}))
            out["recent_prs"] = recent
        except Exception as e:
            errors["prs"] = str(e)
        try:
            badges = [
                b for b in garmin_client.get_earned_badges() or []
                if isinstance(b, dict)
                and (b.get("badgeEarnedDate") or "")[:10] >= cutoff.isoformat()
            ]
            badges.sort(key=lambda b: b.get("badgeEarnedDate") or "", reverse=True)
            out["recent_badges"] = [
                _drop_none({"name": b.get("badgeName"), "date": (b.get("badgeEarnedDate") or "")[:10] or None})
                for b in badges[:10]
            ]
        except Exception as e:
            errors["badges"] = str(e)
        try:
            rp = garmin_client.get_race_predictions() or {}
            out["race_prediction_5k"] = rp.get("time5K") or (rp.get("predictions") or {}).get("5K")
        except Exception:
            pass
        if errors:
            out["errors"] = errors
        return json.dumps(out, indent=2)

    @app.tool()
    async def get_plan_context() -> str:
        """What structured training is prescribed/available: the athlete's
        enrolled Garmin plans that are not completed (status, dates, days to
        the race, current week), and for an adaptive Garmin Coach plan its
        phases and the ~7 days of sessions Garmin has generated; plus the saved
        workout library (name + id + type — real sessions that sync to the
        watch). Use so the coach prescribes from real available workouts and
        reconciles any enrolled Garmin plan; if none, the coach owns the
        periodization itself (see plan.md). The calendar itself comes from
        get_scheduled_workouts."""
        out = {}
        errors = {}
        try:
            plans = (garmin_client.get_training_plans() or {}).get("trainingPlanList", [])
            today = datetime.date.today()
            summarised = [_plan_summary(p, today) for p in plans if isinstance(p, dict)]
            out["active_plans"] = [p for p in summarised if p.get("status") not in ("Completed", "Cancelled")]
            out["all_plans_count"] = len(summarised)
            if not out["active_plans"]:
                out["note"] = "no active Garmin plan — the coach owns periodization (plan.md)"
            elif any(p.get("adaptive") for p in out["active_plans"]):
                out["note"] = (
                    "an adaptive Garmin plan is enrolled: it puts its own workouts on the "
                    "watch calendar — reconcile them with the coach's cards"
                )
                for p in out["active_plans"]:
                    if p.get("adaptive"):
                        try:
                            p.update(_adaptive_plan_detail(p["id"], today))
                        except Exception as e:
                            errors[f"plan_{p['id']}"] = str(e)
        except Exception as e:
            errors["plans"] = str(e)
        try:
            workouts = garmin_client.get_workouts(0, 30) or []
            out["saved_workouts"] = [
                _drop_none({
                    "id": w.get("workoutId"),
                    "name": w.get("workoutName"),
                    "type": (w.get("sportType") or {}).get("sportTypeKey"),
                })
                for w in workouts if isinstance(w, dict)
            ]
        except Exception as e:
            errors["saved_workouts"] = str(e)
        if errors:
            out["errors"] = errors
        return json.dumps(out, indent=2)

    return app
