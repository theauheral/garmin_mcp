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
        "aerobic_te": activity.get("aerobicTrainingEffect"),
        "anaerobic_te": activity.get("anaerobicTrainingEffect"),
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
    src = activity if isinstance(activity, dict) else {}
    if isinstance(src.get("summaryDTO"), dict):
        src = src["summaryDTO"]
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


def _zone_distribution(zones):
    """Summarize get_activity_hr_in_timezones output into easy vs hard share."""
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
            "low_bpm": z.get("zoneLowBoundary"),
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


def _pacing(splits):
    """First-half vs second-half pace drift from typed splits (fade detection)."""
    rows = []
    items = splits.get("splits") if isinstance(splits, dict) else splits
    if not isinstance(items, list):
        return None
    for s in items:
        if not isinstance(s, dict):
            continue
        dist, dur = s.get("distance"), s.get("duration") or s.get("movingDuration")
        if isinstance(dist, (int, float)) and dist > 0 and isinstance(dur, (int, float)):
            rows.append({"km": round(dist / 1000, 2), "pace_s_per_km": round(dur / (dist / 1000)), "avg_hr": s.get("averageHR")})
    if len(rows) < 2:
        return None
    half = len(rows) // 2
    first = fmean(r["pace_s_per_km"] for r in rows[:half])
    second = fmean(r["pace_s_per_km"] for r in rows[half:])
    drift = round(100 * (second - first) / first, 1)
    return {
        "splits": rows,
        "first_half_pace_s": round(first),
        "second_half_pace_s": round(second),
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
    async def get_session_analysis(date: str = "", activity_id: int = 0) -> str:
        """Objective execution quality for one training session: HR time-in-zones
        with zone boundaries (easy vs hard share — did an "easy" run stay
        aerobic?), per-split pacing with fade/negative-split detection, average
        cadence + stride length, and the session summary. Pass a
        `date` (analyses that day's main activity) or an explicit `activity_id`.
        Use to audit whether a session was executed as intended — the key
        discipline for aerobic-base building (easy runs must actually be easy).

        Args:
            date: YYYY-MM-DD to analyse that day's main activity (optional)
            activity_id: explicit Garmin activity id (optional; overrides date)
        """
        result = {}
        errors = {}
        try:
            if activity_id:
                summary = garmin_client.get_activity(activity_id) or {}
                aid = activity_id
            else:
                if date:
                    acts = garmin_client.get_activities_by_date(date, date) or []
                else:
                    acts = garmin_client.get_activities(0, 1) or []
                acts = [a for a in acts if isinstance(a, dict)]
                if not acts:
                    return json.dumps({"note": f"no activity found for {date or 'most recent'}"}, indent=2)
                summary = acts[0]
                aid = summary.get("activityId") or summary.get("activityId".lower())
            result["session"] = _drop_none(_summarize_session(summary))
            result["activity_id"] = aid
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

        form = _form_metrics(summary)
        result["form"] = _drop_none({
            "cadence_spm": form["cadence_spm"],
            "stride_length_m": form["stride_length_m"],
        }) or None

        try:
            result["hr_zones"] = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
        except Exception as e:
            errors["hr_zones"] = str(e)
        try:
            result["pacing"] = _pacing(garmin_client.get_activity_typed_splits(aid))
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
        balance, running power (avg + normalized), plus a per-lap cadence and
        stride breakdown with a first-third vs last-third fatigue comparison
        (does stride collapse late?). Fields the device didn't record are
        null — wrist-based dynamics never report L/R balance. Pass an explicit
        `activity_id`, a `date` (that day's main activity), or neither (most
        recent activity). Use alongside get_session_analysis when the question
        is form/economy rather than effort.

        Args:
            activity_id: explicit Garmin activity id (optional; overrides date)
            date: YYYY-MM-DD to analyse that day's main activity (optional)
        """
        result = {}
        errors = {}
        try:
            if activity_id:
                summary = garmin_client.get_activity(activity_id) or {}
                aid = activity_id
            else:
                if date:
                    acts = garmin_client.get_activities_by_date(date, date) or []
                else:
                    acts = garmin_client.get_activities(0, 1) or []
                acts = [a for a in acts if isinstance(a, dict)]
                if not acts:
                    return json.dumps({"note": f"no activity found for {date or 'most recent'}"}, indent=2)
                summary = acts[0]
                aid = summary.get("activityId")
            result["session"] = _drop_none(_summarize_session(summary))
            result["activity_id"] = aid
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

        # Nulls stay visible here on purpose: "not recorded" is an answer.
        form = _form_metrics(summary)
        if form["gct_balance_pct"] is None:
            form["gct_balance_note"] = "L/R balance needs a chest strap or RD pod; wrist dynamics don't record it"
        result["form"] = form

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
    async def get_execution_trend(count: int = 10, activity_type: str = "running") -> str:
        """Execution quality ACROSS the last N sessions of one type — surfaces
        patterns a single-session view misses: how many runs were genuinely
        easy (Z1-2) vs 'grey zone' (mostly Z3) vs hard (Z3+ heavy), the average
        easy-share, per-run cadence + stride length (form drift across weeks),
        and monotony (are they all the same distance/effort?). Use
        for weekly review and to diagnose training distribution (e.g. the
        classic 'every run is moderately hard' base-building failure).

        Args:
            count: number of recent sessions to analyse (default 10, ~N API calls)
            activity_type: typeKey filter, e.g. running / cycling (default running)
        """
        try:
            acts = garmin_client.get_activities(0, max(1, min(count * 2, 40))) or []
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
        acts = [a for a in acts if isinstance(a, dict)
                and (a.get("activityType") or {}).get("typeKey", "").find(activity_type) >= 0][:count]
        runs = []
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
                "aerobic_te": round(a["aerobicTrainingEffect"], 1) if isinstance(a.get("aerobicTrainingEffect"), (int, float)) else None,
            }
            try:
                z = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
                if z:
                    es = z["easy_share_pct"]
                    row["easy_share_pct"] = es
                    tag = "easy" if es >= 65 else "hard" if es < 35 else "grey"
                else:
                    tag = "unknown"
            except Exception:
                tag = "unknown"
            row["execution"] = tag
            buckets[tag] += 1
            runs.append(_drop_none(row))
        graded = [r for r in runs if r.get("easy_share_pct") is not None]
        avg_easy = round(fmean(r["easy_share_pct"] for r in graded)) if graded else None
        dists = [r["km"] for r in runs if r.get("km")]
        return json.dumps({
            "analysed": len(runs),
            "activity_type": activity_type,
            "distribution": buckets,
            "avg_easy_share_pct": avg_easy,
            "monotony": _drop_none({
                "distinct_distances": len(set(dists)) if dists else None,
                "distance_range_km": [min(dists), max(dists)] if dists else None,
            }),
            "sessions": runs,
        }, indent=2)

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
        import datetime as _dt
        cutoff = (_dt.date.today() - _dt.timedelta(days=days))
        out = {"window_days": days}
        errors = {}
        try:
            prs = garmin_client.get_personal_record() or []
            recent = []
            for p in prs:
                ts = p.get("activityStartDateTimeInGMT")
                d = None
                if isinstance(ts, (int, float)):
                    d = _dt.datetime.utcfromtimestamp(ts / 1000).date()
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
