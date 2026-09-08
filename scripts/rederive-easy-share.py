#!/usr/bin/env python3
"""Re-derive the easy-share KPI across a block, from raw HR rather than stored bands.

Garmin freezes ``hrTimeInZones`` into each activity at upload, so every session
recorded under an old zone model keeps the wrong bands forever. The KPI this
block was founded on -- "easy-share 19%, target 70%" -- was therefore computed
against bands that put the easy/hard cut ~20 bpm too low, and it has been
reporting failure ever since.

This walks a date range, re-integrates time-in-zone from each activity's raw HR
sample stream against the zone model in force today, and prints stored versus
recomputed side by side.

Trust rests on the self-check, not on the new number: for every activity the
server also re-integrates the same samples using the STORED bands and must
reproduce Garmin's own figure. Where that check fails, or where no sample
stream exists, the row is reported as unusable rather than quietly replaced.

Usage:
    uv run scripts/rederive-easy-share.py 2026-07-03 2026-08-28   (from the fork root)
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import garminconnect  # noqa: E402
from garmin_mcp import composites  # noqa: E402

#: The coaching cap. Deliberately tighter than Garmin's Z2 top, because LT1
#: sits inside Z2 rather than at its ceiling.
EASY_CAP_BPM = 150


def main() -> int:
    start, end = sys.argv[1], sys.argv[2]

    client = garminconnect.Garmin()
    client.login(os.environ.get("GARMINTOKENS", os.path.expanduser("~/.garminconnect")))
    composites.garmin_client = client

    # The live zone model — what the watch applies today — as opposed to the
    # bands frozen into each activity at upload.
    model, model_error = composites._hr_zone_model(None)
    if not model:
        print(f"cannot read the live zone model: {model_error}")
        return 1
    print(f"live zone model: LTHR {model.get('lthr')} "
          f"({model.get('lthr_source')})  floors {model['floors']}\n")

    activities = client.get_activities_by_date(start, end)
    rows, unusable = [], []

    for a in activities:
        kind = (a.get("activityType") or {}).get("typeKey", "")
        if kind not in {"running", "cycling", "treadmill_running", "trail_running"}:
            continue
        aid = a.get("activityId")
        date = str(a.get("startTimeLocal", ""))[:10]
        avg_hr = a.get("averageHR")

        try:
            stored = composites._zone_distribution(client.get_activity_hr_in_timezones(aid))
            rec = composites._recompute_hr_zones(aid, model, stored=stored)
        except Exception as exc:  # noqa: BLE001
            unusable.append((date, avg_hr, f"error: {exc}"))
            continue

        if not rec or rec.get("easy_share_pct") is None:
            unusable.append((date, avg_hr, (rec or {}).get("not_recomputable", "no HR stream")))
            continue
        check = rec.get("verification", {})
        if check.get("matches_stored") is False:
            unusable.append((date, avg_hr, "self-check FAILED — stored bands do not reproduce"))
            continue

        rows.append({
            "date": date,
            "kind": kind,
            "avg_hr": avg_hr,
            "stored": stored.get("easy_share_pct"),
            "recomputed": rec.get("easy_share_pct"),
            "bands_differ": check.get("stored_bands_differ"),
        })

    rows.sort(key=lambda r: r["date"])
    print(f"{'date':12s} {'type':10s} {'avgHR':>6s} {'stored':>7s} {'recomp':>7s}  stamp")
    print("-" * 60)
    for r in rows:
        stamp = "legacy" if r["bands_differ"] else "current"
        print(f"{r['date']:12s} {r['kind'][:10]:10s} {r['avg_hr'] or 0:6.0f} "
              f"{r['stored']:6}% {r['recomputed']:6}%  {stamp}")

    if unusable:
        print("\nNOT RECOMPUTABLE — excluded, never silently replaced:")
        for date, hr, why in sorted(unusable):
            print(f"  {date}  avgHR {hr}  {why}")

    stored_vals = [r["stored"] for r in rows if r["stored"] is not None]
    rec_vals = [r["recomputed"] for r in rows if r["recomputed"] is not None]
    if rec_vals:
        print(f"\nBLOCK MEAN easy-share   stored {statistics.mean(stored_vals):.0f}%"
              f"   ->   recomputed {statistics.mean(rec_vals):.0f}%   (n={len(rec_vals)})")
        by_cap = sum(1 for r in rows if (r["avg_hr"] or 999) <= EASY_CAP_BPM)
        print(f"Sessions with avg HR <= {EASY_CAP_BPM}: {by_cap} of {len(rows)}")
        print(f"Sessions >=70% easy (recomputed): "
              f"{sum(1 for v in rec_vals if v >= 70)} of {len(rec_vals)}")
    print(f"\ncoverage: {len(rows)} recomputed, {len(unusable)} unusable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
