"""Composite one-call briefs for daily planning and coaching.

These aggregate several Garmin endpoints server-side so an assistant doing a
morning briefing or a weekly training review needs one tool call instead of
five to seven. Sub-sections fail independently: a missing night of sleep data
must not take down the whole brief, so each block is fetched in isolation and
failures are reported in an "errors" map instead of raising.
"""

import datetime
import json
from statistics import fmean

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
    """Resolve one activity to (summary_payload, activity_id, note): an
    explicit id, else a date's main activity, else the most recent one.
    note is set (and the payload None) when nothing was found."""
    if activity_id:
        return garmin_client.get_activity(activity_id) or {}, activity_id, None
    if date:
        acts = garmin_client.get_activities_by_date(date, date) or []
    else:
        acts = garmin_client.get_activities(0, 1) or []
    acts = [a for a in acts if isinstance(a, dict)]
    if not acts:
        return None, None, f"no activity found for {date or 'most recent'}"
    return acts[0], acts[0].get("activityId"), None


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
# A gap longer than this is an auto-pause or a dropout, not time spent in a
# zone — counting it would credit whichever zone happened to precede it.
_HR_GAP_CAP_S = 60


def _zone_model_from_config(entries, sport="DEFAULT"):
    """Pull the zone floors for one sport out of Garmin's zone-config payload.

    The config is the live model — what the watch applies today — as opposed to
    the bands frozen into each activity at upload time.
    """
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return None
    chosen = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        if (e.get("sport") or "").upper() == sport.upper():
            chosen = e
            break
        if chosen is None:
            chosen = e
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
    }


def _hr_zone_model(lthr=None, sport="DEFAULT"):
    """The zone model to score against: Garmin's live config, optionally
    re-anchored to a caller-supplied LTHR.

    The LTHR is a parameter and never a constant because the two sources
    disagree in practice — Garmin auto-detects one value while a coaching plan
    may be written against another, and ~5 bpm moves every boundary. Whichever
    is applied is echoed back in the output so a wrong one is diagnosable
    rather than silent. Returns (model, None) or (None, reason).
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
        "floors": list(base["floors"]),
    }
    if lthr:
        configured = base["lthr"]
        if not isinstance(configured, (int, float)) or configured <= 0:
            model["lthr_override_ignored"] = (
                f"no configured LTHR to rescale from; used the zone config as-is"
            )
        else:
            # Garmin's own floors move linearly with LTHR (verified against two
            # activities stamped under different LTHRs), so rescaling by the
            # ratio reproduces the bands the watch would have applied.
            scale = lthr / configured
            model["floors"] = [int(round(f * scale)) for f in base["floors"]]
            model["lthr_configured"] = configured
            model["lthr"] = lthr
            model["lthr_source"] = "caller_override"
    model["floors_bpm"] = {f"z{i}": f for i, f in enumerate(model["floors"], start=1)}
    return model, None


def _hr_stream(activity_id, max_samples=_HR_STREAM_SAMPLES):
    """(elapsed_seconds, bpm) samples for one activity, or (None, reason).

    maxpoly=0 drops the GPS polyline: this needs two columns of a 26-column
    payload and the track is the bulk of it.
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
    t_i = index.get("sumElapsedDuration", index.get("sumDuration"))
    if hr_i is None:
        return None, "activity has no heart-rate stream (manual entry, or HR not recorded)"
    if t_i is None:
        return None, "activity stream carries no elapsed-time column"
    samples = []
    for row in details.get("activityDetailMetrics") or []:
        m = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(m, list) or len(m) <= max(hr_i, t_i):
            continue
        hr, t = m[hr_i], m[t_i]
        if isinstance(hr, (int, float)) and isinstance(t, (int, float)) and hr > 0:
            samples.append((t, hr))
    if len(samples) < 2:
        return None, "activity has no usable heart-rate samples"
    samples.sort(key=lambda s: s[0])
    return samples, None


def _integrate_zones(samples, floors):
    """Seconds per zone, time-weighting each HR sample by the gap to the next.

    Returns (secs_by_zone, below_z1_secs). HR under the zone-1 floor is counted
    separately and excluded from the shares, which is what Garmin does too —
    its stored totals omit sub-Z1 time rather than folding it into Z1.
    """
    secs = {n: 0.0 for n in range(1, len(floors) + 1)}
    below = 0.0
    for (t0, hr), (t1, _) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > _HR_GAP_CAP_S:
            continue
        zone = 0
        for n, floor in enumerate(floors, start=1):
            if hr >= floor:
                zone = n
        if zone:
            secs[zone] += dt
        else:
            below += dt
    return secs, below


def _shares(secs):
    """(total_seconds, easy_pct, hard_pct) over Z1-2 vs Z3+."""
    total = sum(secs.values())
    if total <= 0:
        return 0.0, None, None
    easy = sum(v for n, v in secs.items() if n <= 2)
    return total, round(100 * easy / total), round(100 * (total - easy) / total)


def _recompute_hr_zones(activity_id, model, stored=None, samples=None):
    """Time-in-zone for one activity scored against `model`, never raising.

    Always answers: either the recomputed distribution, or an explicit
    `not_recomputable` with the reason. Falling back to the frozen value
    without saying so would reintroduce the exact bug this replaces.
    """
    if samples is None:
        samples, reason = _hr_stream(activity_id)
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
        "applied": _drop_none({
            "lthr": model.get("lthr"),
            "lthr_source": model.get("lthr_source"),
            "lthr_configured": model.get("lthr_configured"),
            "lthr_override_ignored": model.get("lthr_override_ignored"),
            "training_method": model.get("training_method"),
            "max_hr": model.get("max_hr"),
            "floors_bpm": model.get("floors_bpm"),
        }),
        "samples": len(samples),
    }
    verification = _verify_against_stored(samples, stored, model["floors"])
    if verification:
        out["verification"] = verification
    return _drop_none(out)


def _stored_floors(stored):
    """The bands Garmin froze into the activity, from a _zone_distribution."""
    by_zone = (stored or {}).get("by_zone") if isinstance(stored, dict) else None
    if not isinstance(by_zone, dict):
        return None
    floors = []
    for n in range(1, 6):
        low = (by_zone.get(f"z{n}") or {}).get("low_bpm")
        if not isinstance(low, (int, float)):
            return None
        floors.append(int(low))
    return floors


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
                    "score": (dto.get("sleepScores") or {}).get("overall", {}).get("value"),
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

        try:
            vo2_now = ((garmin_client.get_max_metrics(end_date) or [{}])[0].get("generic") or {})
            vo2_past = ((garmin_client.get_max_metrics(d4w) or [{}])[0].get("generic") or {})
            now_v = vo2_now.get("vo2MaxPreciseValue") or vo2_now.get("vo2MaxValue")
            past_v = vo2_past.get("vo2MaxPreciseValue") or vo2_past.get("vo2MaxValue")
            fitness = {
                "vo2max_now": now_v,
                "vo2max_4w_ago": past_v,
                "vo2max_change": _round(now_v - past_v, 1) if isinstance(now_v, (int, float)) and isinstance(past_v, (int, float)) else None,
            }
            score_dto = (garmin_client.get_endurance_score(window_start, end_date) or {}).get("enduranceScoreDTO") or {}
            fitness["endurance_score"] = score_dto.get("overallScore") or score_dto.get("score")
            report["fitness"] = _drop_none(fitness)
        except Exception as e:
            errors["fitness"] = str(e)

        if errors:
            report["errors"] = errors
        return json.dumps(report, indent=2)

    @app.tool()
    async def get_session_analysis(date: str = "", activity_id: int = 0, lthr: int = 0) -> str:
        """Objective execution quality for one training session: HR time-in-zones
        with zone boundaries (easy vs hard share — did an "easy" run stay
        aerobic?), POWER time-in-zones (no cardiac lag or heat inflation — the
        cross-check when HR looks hot), the athlete's own RPE/feel rating vs
        the objective read, per-split pacing with fade/negative-split
        detection, cadence + stride length, session cost (stamina, body
        battery, sweat), terrain with grade-adjusted pace, and set/rep detail
        for strength sessions. Pass a
        `date` (analyses that day's main activity) or an explicit `activity_id`.
        Use to audit whether a session was executed as intended — the key
        discipline for aerobic-base building (easy runs must actually be easy).

        `hr_zones` carries Garmin's STORED zone times, which are frozen into the
        activity at upload and never change afterwards — an activity recorded
        under an old zone model keeps its old bands forever. `hr_zones.recomputed`
        re-integrates time-in-zone from the raw HR sample stream against the
        CURRENT zone model, so easy-share is comparable across a zone-model
        change; read it in preference to the stored share, and use
        `recomputed.stored_bands_differ` to tell whether this activity was
        affected at all. `recomputed.applied.lthr` names the threshold actually
        used, and `recomputed.verification` re-scores the same stream with the
        activity's own frozen bands — when it reproduces the stored share, the
        recomputation is trustworthy. Activities with no HR stream (manual
        entries) say so in `recomputed.not_recomputable` rather than quietly
        falling back.

        Args:
            date: YYYY-MM-DD to analyse that day's main activity (optional)
            activity_id: explicit Garmin activity id (optional; overrides date)
            lthr: override the lactate-threshold HR the zone model is anchored
                to (optional; default reads Garmin's live zone configuration).
                Every boundary scales with it, so a 5 bpm difference is worth
                several points of easy-share — pass the coaching plan's value
                when it disagrees with Garmin's auto-detected one.
        """
        result = {}
        errors = {}
        try:
            summary, aid, note = _resolve_activity(activity_id, date)
            if note:
                return json.dumps({"note": note}, indent=2)
            result["session"] = _drop_none(_summarize_session(summary))
            result["activity_id"] = aid
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

        form = _form_metrics(summary)
        result["form"] = _drop_none({
            "cadence_spm": form["cadence_spm"],
            "stride_length_m": form["stride_length_m"],
        }) or None

        dto = _full_activity(summary, aid)
        result["effort"] = _drop_none(_effort_rating(dto)) or None
        result["cost"] = _drop_none(_session_cost(dto)) or None
        result["terrain"] = _terrain(dto)

        stored_zones = None
        try:
            stored_zones = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
        except Exception as e:
            errors["hr_zones"] = str(e)
        # Recompute against the live zone model. Kept beside the stored numbers,
        # never on top of them: the comparison is what shows the correction
        # landed, and overwriting would hide a bad zone config.
        model, model_error = _hr_zone_model(lthr or None)
        recomputed = (
            _recompute_hr_zones(aid, model, stored_zones) if model
            else {"not_recomputable": model_error}
        )
        if stored_zones:
            stored_zones["recomputed"] = recomputed
            result["hr_zones"] = stored_zones
        else:
            result["hr_zones"] = {"recomputed": recomputed}
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
        return json.dumps(_drop_none(result), indent=2)

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
            summary, aid, note = _resolve_activity(activity_id, date)
            if note:
                return json.dumps({"note": note}, indent=2)
            result["session"] = _drop_none(_summarize_session(summary))
            result["activity_id"] = aid
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
    ) -> str:
        """Execution quality ACROSS the last N sessions of one type — surfaces
        patterns a single-session view misses: how many runs were genuinely
        easy (Z1-2) vs 'grey zone' (mostly Z3) vs hard (Z3+ heavy), the average
        easy-share, per-run cadence + stride length (form drift across weeks),
        per-run power easy-share and the athlete's own RPE (perception vs
        objective effort — the calibration gap), and monotony (are they all
        the same distance/effort?). Use
        for weekly review and to diagnose training distribution (e.g. the
        classic 'every run is moderately hard' base-building failure).

        Easy-share is recomputed from each session's raw HR stream against the
        CURRENT zone model by default. Garmin freezes zone times into an
        activity at upload, so a stored series spanning a zone-model change is
        a broken prefix joined to a clean suffix and must never be averaged
        across — recomputing puts every session on one model, which is what
        makes the average and the trend mean anything. `easy_share_basis` says
        which basis was used, `zone_model` names the thresholds applied, and
        sessions whose stream is missing are reported in `not_recomputable`
        and excluded from the average rather than silently mixed in.

        Args:
            count: number of recent sessions to analyse (default 10, ~2N API
                calls, ~3N with recompute_zones)
            activity_type: typeKey filter, e.g. running / cycling (default running)
            lthr: override the lactate-threshold HR the zone model is anchored
                to (optional; default reads Garmin's live zone configuration)
            recompute_zones: recompute time-in-zone from raw HR (default True).
                Set False for Garmin's stored per-activity zone times, which
                are cheaper but not comparable across a zone-model change.
        """
        try:
            acts = garmin_client.get_activities(0, max(1, min(count * 2, 40))) or []
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
        acts = [a for a in acts if isinstance(a, dict)
                and (a.get("activityType") or {}).get("typeKey", "").find(activity_type) >= 0][:count]

        # One zone-config fetch for the whole batch — the model is per-athlete,
        # not per-activity.
        model, model_error = (None, "recompute_zones=False")
        if recompute_zones:
            model, model_error = _hr_zone_model(lthr or None)

        runs = []
        not_recomputable = {}
        buckets = {"easy": 0, "grey": 0, "hard": 0, "unknown": 0}
        for a in acts:
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
            try:
                row["rpe_10"] = _effort_rating(garmin_client.get_activity(aid) or {})["rpe_10"]
            except Exception:
                pass
            try:
                z = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
                row["easy_share_pct"] = z["easy_share_pct"] if z else None
            except Exception:
                pass
            # The graded share: recomputed when we can, stored otherwise, but
            # never a mix — a single basis is the whole point of the series.
            if model:
                rec = _recompute_hr_zones(aid, model)
                if "not_recomputable" in rec:
                    not_recomputable[str(aid)] = rec["not_recomputable"]
                    graded_share = None
                else:
                    graded_share = rec["easy_share_pct"]
                    row["easy_share_recomputed_pct"] = graded_share
            else:
                graded_share = row.get("easy_share_pct")
            tag = ("unknown" if graded_share is None
                   else "easy" if graded_share >= 65
                   else "hard" if graded_share < 35 else "grey")
            row["execution"] = tag
            buckets[tag] += 1
            runs.append(_drop_none(row))
        share_key = "easy_share_recomputed_pct" if model else "easy_share_pct"
        graded = [r for r in runs if r.get(share_key) is not None]
        avg_easy = round(fmean(r[share_key] for r in graded)) if graded else None
        dists = [r["km"] for r in runs if r.get("km")]
        return json.dumps(_drop_none({
            "analysed": len(runs),
            "activity_type": activity_type,
            "distribution": buckets,
            "avg_easy_share_pct": avg_easy,
            "easy_share_basis": (
                "recomputed from raw HR against the current zone model"
                if model else
                f"Garmin's stored per-activity zone times ({model_error}) — "
                "NOT comparable across a zone-model change"
            ),
            "zone_model": model and _drop_none({
                "lthr": model.get("lthr"),
                "lthr_source": model.get("lthr_source"),
                "training_method": model.get("training_method"),
                "floors_bpm": model.get("floors_bpm"),
            }),
            "not_recomputable": not_recomputable or None,
            "monotony": _drop_none({
                "distinct_distances": len(set(dists)) if dists else None,
                "distance_range_km": [min(dists), max(dists)] if dists else None,
            }),
            "sessions": runs,
        }), indent=2)

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
        offset_h = 0
        try:
            events = garmin_client.get_body_battery_events(date) or []
            for e in events:
                off = (e.get("event") or {}).get("timezoneOffset")
                if isinstance(off, (int, float)):
                    offset_h = off / 3600000
                    break
        except Exception:
            pass
        try:
            bb = next((d for d in (garmin_client.get_body_battery(date, date) or []) if isinstance(d, dict)), {})
            idx = _descriptor_index(bb.get("bodyBatteryValueDescriptorDTOList"), "bodyBatteryLevel", 2)
            hourly = {}
            for row in bb.get("bodyBatteryValuesArray") or []:
                if isinstance(row, (list, tuple)) and len(row) > idx and isinstance(row[0], (int, float)) and isinstance(row[idx], (int, float)):
                    hour = int(((row[0] / 1000) + offset_h * 3600) // 3600 % 24)
                    hourly.setdefault(hour, []).append(row[idx])
            curve = [{"hour": h, "level": round(fmean(v))} for h, v in sorted(hourly.items())]
            out["current_level"] = _current_body_battery_level(bb)
            out["hourly"] = curve
            # best/worst 2-hour window
            if len(curve) >= 2:
                best = max(curve, key=lambda c: c["level"])
                worst = min(curve, key=lambda c: c["level"])
                out["peak_window"] = {"around_hour": best["hour"], "level": best["level"]}
                out["trough_window"] = {"around_hour": worst["hour"], "level": worst["level"]}
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
            badges = garmin_client.get_earned_badges() or []
            out["recent_badges"] = [b.get("badgeName") for b in badges[:5] if isinstance(b, dict)]
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
        """What structured training is prescribed/available: active Garmin plans
        (if any), upcoming scheduled workouts, and the athlete's saved workout
        library (name + id + type — real sessions that sync to the watch). Use
        so the coach prescribes from real available workouts and reconciles any
        active Garmin plan; if no active plan, the coach owns the periodization
        itself (see plan.md)."""
        out = {}
        errors = {}
        try:
            plans = (garmin_client.get_training_plans() or {}).get("trainingPlanList", [])
            active = [
                _drop_none({
                    "id": p.get("trainingPlanId"),
                    "name": p.get("name"),
                    "category": p.get("trainingPlanCategory"),
                    "status": (p.get("trainingStatus") or {}).get("statusKey"),
                    "weeks": p.get("durationInWeeks"),
                    "avg_weekly_workouts": p.get("avgWeeklyWorkouts"),
                })
                for p in plans if isinstance(p, dict)
            ]
            out["active_plans"] = [p for p in active if p.get("status") not in ("Completed", "Cancelled")]
            out["all_plans_count"] = len(active)
            if not out["active_plans"]:
                out["note"] = "no active Garmin plan — the coach owns periodization (plan.md)"
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
