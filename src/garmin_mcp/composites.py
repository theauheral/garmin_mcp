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
        by_zone[f"z{n}"] = {"min": round(secs / 60, 1), "pct": round(100 * secs / total)}
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
        (easy vs hard share — did an "easy" run stay aerobic?), per-split pacing
        with fade/negative-split detection, and the session summary. Pass a
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

        try:
            result["hr_zones"] = _zone_distribution(garmin_client.get_activity_hr_in_timezones(aid))
        except Exception as e:
            errors["hr_zones"] = str(e)
        try:
            result["pacing"] = _pacing(garmin_client.get_activity_typed_splits(aid))
        except Exception as e:
            errors["pacing"] = str(e)

        if errors:
            result["errors"] = errors
        return json.dumps(_drop_none(result), indent=2)

    @app.tool()
    async def get_execution_trend(count: int = 10, activity_type: str = "running") -> str:
        """Execution quality ACROSS the last N sessions of one type — surfaces
        patterns a single-session view misses: how many runs were genuinely
        easy (Z1-2) vs 'grey zone' (mostly Z3) vs hard (Z3+ heavy), the average
        easy-share, and monotony (are they all the same distance/effort?). Use
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
            row = {
                "date": (a.get("startTimeLocal") or "")[:10],
                "km": round(dist / 1000, 1) if isinstance(dist, (int, float)) else None,
                "avg_hr": a.get("averageHR"),
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
