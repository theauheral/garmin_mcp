"""Recomputing time-in-zone from the raw HR stream.

Garmin freezes hrTimeInZones into an activity at upload, so every session
recorded under an old zone model keeps the old bands forever. These cover the
replacement: integrate the raw stream against the *current* model, say so when
it can't be done, and never silently fall back to the frozen value.
"""

import pytest

from garmin_mcp import composites
from garmin_mcp.composites import (
    _hr_stream,
    _stamp_history,
    _hr_zone_model,
    _integrate_zones,
    _recompute_hr_zones,
    _shares,
    _zone_model_from_config,
)

# Garmin's live zone-config payload, %LTHR anchored (the corrected model).
ZONE_CONFIG = [{
    "trainingMethod": "LACTATE_THRESHOLD",
    "restingHeartRateUsed": 48,
    "lactateThresholdHeartRateUsed": 170,
    "zone1Floor": 110, "zone2Floor": 130, "zone3Floor": 150,
    "zone4Floor": 160, "zone5Floor": 168,
    "maxHeartRateUsed": 196,
    "sport": "DEFAULT",
}]

# The stock %max bands an older activity carries frozen into it.
FROZEN_FLOORS = [99, 118, 139, 157, 178]
LIVE_FLOORS = [110, 130, 150, 160, 168]


class FakeClient:
    """Only the two endpoints the recomputation touches."""

    def __init__(self, zones=ZONE_CONFIG, details=None, details_error=None):
        self._zones = zones
        self._details = details
        self._details_error = details_error
        self.details_calls = []

    def connectapi(self, path):
        if isinstance(self._zones, Exception):
            raise self._zones
        return self._zones

    def get_activity_details(self, activity_id, maxchart=None, maxpoly=None):
        self.details_calls.append((activity_id, maxchart, maxpoly))
        if self._details_error:
            raise self._details_error
        return self._details


def details_payload(samples, hr_key="directHeartRate", time_key="sumElapsedDuration"):
    """A details response carrying (elapsed_s, bpm) pairs in two columns."""
    return {
        "metricDescriptors": [
            {"metricsIndex": 0, "key": hr_key},
            {"metricsIndex": 1, "key": time_key},
        ],
        "activityDetailMetrics": [{"metrics": [hr, t]} for t, hr in samples],
    }


NO_HR_DETAILS = {
    "metricDescriptors": [{"metricsIndex": 0, "key": "directPower"}],
    "activityDetailMetrics": [{"metrics": [220]}, {"metrics": [230]}],
}


def steady(bpm, seconds, step=1):
    """A constant-HR stream — the clearest way to make two zone models disagree."""
    return [(t, bpm) for t in range(0, seconds + 1, step)]


@pytest.fixture
def client(monkeypatch):
    def _configure(**kwargs):
        c = FakeClient(**kwargs)
        composites.configure(c)
        return c
    return _configure


# --- zone model --------------------------------------------------------------


def test_zone_model_reads_the_live_configuration(client):
    client()
    model, error = _hr_zone_model()

    assert error is None
    assert model["floors"] == LIVE_FLOORS
    assert model["lthr"] == 170
    assert model["lthr_source"] == "garmin_zone_config"
    assert model["training_method"] == "LACTATE_THRESHOLD"
    assert model["floors_bpm"]["z3"] == 150


def test_zone_model_picks_the_requested_sport():
    entries = [
        {"sport": "CYCLING", "zone1Floor": 1, "zone2Floor": 2, "zone3Floor": 3,
         "zone4Floor": 4, "zone5Floor": 5},
        ZONE_CONFIG[0],
    ]
    assert _zone_model_from_config(entries, "DEFAULT")["floors"] == LIVE_FLOORS


def test_zone_model_falls_back_to_the_first_entry_when_sport_is_absent():
    assert _zone_model_from_config(ZONE_CONFIG, "TRIATHLON")["floors"] == LIVE_FLOORS


def test_zone_model_rejects_a_payload_missing_a_floor():
    broken = [{**ZONE_CONFIG[0], "zone4Floor": None}]
    assert _zone_model_from_config(broken) is None
    assert _zone_model_from_config("nonsense") is None


def test_lthr_override_rescales_every_boundary(client):
    """The two sources disagree in practice; ~5 bpm moves every boundary."""
    client()
    model, error = _hr_zone_model(lthr=175)

    assert error is None
    # 110 * 175/170 = 113.2, 150 * 175/170 = 154.4 ...
    assert model["floors"] == [113, 134, 154, 165, 173]
    assert model["lthr"] == 175
    assert model["lthr_configured"] == 170
    assert model["lthr_source"] == "caller_override"


def test_lthr_override_without_a_configured_threshold_says_so(client):
    client(zones=[{**ZONE_CONFIG[0], "lactateThresholdHeartRateUsed": None,
                   "trainingMethod": "PERCENT_MAX"}])
    model, _ = _hr_zone_model(lthr=175)

    assert model["floors"] == LIVE_FLOORS  # unchanged, not silently rescaled
    assert "lthr_override_ignored" in model
    assert model["lthr_source"] == "garmin_zone_config"


def test_zone_config_failure_returns_a_reason(client):
    client(zones=RuntimeError("503"))
    model, error = _hr_zone_model()

    assert model is None
    assert "zone configuration unavailable" in error


# --- stream extraction -------------------------------------------------------


def test_hr_stream_extracts_time_and_bpm(client):
    c = client(details=details_payload([(0, 120), (1, 121), (2, 122)]))
    samples, error = _hr_stream(7, max_samples=500)

    assert error is None
    assert samples == [(0, 120), (1, 121), (2, 122)]
    # maxpoly=0: the GPS track is the bulk of the payload and is not needed.
    assert c.details_calls == [(7, 500, 0)]


def test_hr_stream_sorts_and_drops_unusable_rows(client):
    payload = details_payload([(2, 130), (0, 120)])
    payload["activityDetailMetrics"].append({"metrics": [None, 3]})
    payload["activityDetailMetrics"].append({"metrics": [0, 4]})  # 0 bpm = no reading
    client(details=payload)
    samples, error = _hr_stream(7)

    assert error is None
    assert samples == [(0, 120), (2, 130)]


def test_hr_stream_falls_back_to_sum_duration(client):
    client(details=details_payload([(0, 120), (1, 121)], time_key="sumDuration"))
    samples, error = _hr_stream(7)

    assert error is None and len(samples) == 2


def test_hr_stream_without_heart_rate_says_why(client):
    client(details=details_payload([(0, 120), (1, 121)], hr_key="directPower"))
    samples, error = _hr_stream(7)

    assert samples is None
    assert "no heart-rate stream" in error


def test_hr_stream_with_too_few_samples_says_why(client):
    client(details=details_payload([(0, 120)]))
    samples, error = _hr_stream(7)

    assert samples is None
    assert "no usable heart-rate samples" in error


def test_hr_stream_api_failure_says_why(client):
    client(details_error=RuntimeError("500"))
    samples, error = _hr_stream(7)

    assert samples is None
    assert "HR stream unavailable" in error


# --- integration -------------------------------------------------------------


def test_integrate_weights_each_sample_by_the_gap_to_the_next():
    samples = [(0, 120), (60, 120), (120, 140), (180, 140),
               (240, 155), (300, 155), (360, 100), (420, 100)]
    secs, below = _integrate_zones(samples, LIVE_FLOORS)

    assert secs == {1: 120.0, 2: 120.0, 3: 120.0, 4: 0.0, 5: 0.0}
    assert below == 60.0  # sub-Z1 kept separate, as Garmin's own totals do
    assert _shares(secs) == (360.0, 67, 33)


def test_integrate_ignores_pauses():
    """A long gap is an auto-pause, not time spent in the zone before it."""
    samples = [(0, 155), (60, 155), (3600, 155), (3660, 155)]
    secs, _ = _integrate_zones(samples, LIVE_FLOORS)

    assert secs[3] == 120.0  # the 3540s gap is dropped, not credited to Z3


def test_shares_of_an_empty_distribution():
    assert _shares({1: 0.0, 2: 0.0}) == (0.0, None, None)


# --- the whole recomputation -------------------------------------------------


def test_recompute_matches_a_hand_calculation(client):
    """600 s at a steady 145 bpm, scored against the live model.

    145 sits above the Z2 floor (130) and below Z3 (150), so the whole session
    is easy: 10 minutes in Z2 and nothing anywhere else.
    """
    client(details=details_payload(steady(145, 600)))
    model, _ = _hr_zone_model()
    out = _recompute_hr_zones(11, model)

    assert out["easy_share_pct"] == 100
    assert out["hard_share_pct"] == 0
    assert out["total_min"] == 10.0
    assert out["by_zone"]["z2"]["min"] == 10.0
    assert out["by_zone"]["z2"]["low_bpm"] == 130
    assert out["by_zone"]["z3"]["min"] == 0.0
    assert out["samples"] == 601
    assert out["applied"]["lthr"] == 170


def test_the_same_stream_scores_hard_under_the_frozen_bands(client):
    """The bug in one assertion: identical HR, opposite verdict.

    145 bpm is Z3 under the stock %max bands frozen into a pre-correction
    activity, and Z2 under the corrected model. Re-pulling the activity would
    return the frozen answer forever.
    """
    secs_frozen, _ = _integrate_zones(steady(145, 600), FROZEN_FLOORS)
    secs_live, _ = _integrate_zones(steady(145, 600), LIVE_FLOORS)

    assert _shares(secs_frozen)[1] == 0
    assert _shares(secs_live)[1] == 100


def test_recompute_verifies_itself_against_the_stored_bands(client):
    """Reproducing Garmin's own number from the raw stream is the proof."""
    client(details=details_payload(steady(145, 600)))
    model, _ = _hr_zone_model()
    stored = {
        "easy_share_pct": 0,
        "by_zone": {f"z{n}": {"low_bpm": f} for n, f in enumerate(FROZEN_FLOORS, 1)},
    }
    out = _recompute_hr_zones(11, model, stored)

    verification = out["verification"]
    assert verification["recomputed_with_stored_bands_pct"] == 0
    assert verification["stored_easy_share_pct"] == 0
    assert verification["matches_stored"] is True
    assert verification["stored_bands_bpm"] == FROZEN_FLOORS
    assert verification["stored_bands_differ"] is True
    assert out["easy_share_pct"] == 100  # ... while the live model disagrees


def test_recompute_flags_a_stream_that_disagrees_with_the_stored_total(client):
    client(details=details_payload(steady(145, 600)))
    model, _ = _hr_zone_model()
    stored = {
        "easy_share_pct": 55,  # nothing like what the stream integrates to
        "by_zone": {f"z{n}": {"low_bpm": f} for n, f in enumerate(FROZEN_FLOORS, 1)},
    }
    out = _recompute_hr_zones(11, model, stored)

    assert out["verification"]["matches_stored"] is False


def test_recompute_omits_verification_without_usable_stored_bands(client):
    client(details=details_payload(steady(145, 600)))
    model, _ = _hr_zone_model()

    assert "verification" not in _recompute_hr_zones(11, model, {"easy_share_pct": 4})


def test_recompute_says_when_it_cannot(client):
    """No HR stream must be an explicit answer, never a silent fallback."""
    client(details=details_payload([(0, 120), (1, 121)], hr_key="directPower"))
    model, _ = _hr_zone_model()
    out = _recompute_hr_zones(11, model)

    assert "no heart-rate stream" in out["not_recomputable"]
    assert "easy_share_pct" not in out


def test_recompute_says_when_no_time_lands_in_any_zone(client):
    client(details=details_payload(steady(70, 600)))  # entire session below Z1
    model, _ = _hr_zone_model()
    out = _recompute_hr_zones(11, model)

    assert out["not_recomputable"] == "no heart-rate time inside any zone"


# --- zone-model drift ---------------------------------------------------------


def test_stamp_history_groups_sessions_by_the_model_they_carry():
    """The step change a single activity cannot show."""
    stamps = [
        ("2026-08-26", LIVE_FLOORS),
        ("2026-08-24", LIVE_FLOORS),
        ("2026-08-14", FROZEN_FLOORS),
        ("2026-07-31", FROZEN_FLOORS),
        ("2026-07-20", FROZEN_FLOORS),
    ]
    groups = _stamp_history(stamps, current=LIVE_FLOORS)

    assert [g["sessions"] for g in groups] == [2, 3]
    assert groups[0]["bands"] == LIVE_FLOORS
    assert groups[0]["newest"] == "2026-08-26" and groups[0]["oldest"] == "2026-08-24"
    assert groups[0]["matches_current_model"] is True
    assert groups[1]["oldest"] == "2026-07-20"      # the change landed on 08-24
    assert groups[1]["matches_current_model"] is False


def test_stamp_history_is_quiet_when_one_model_covers_everything():
    """Nothing to report is worth nothing said — unless the one model in play
    is not the current one, which is exactly worth saying."""
    stamps = [("2026-08-26", LIVE_FLOORS), ("2026-08-24", LIVE_FLOORS)]

    assert _stamp_history(stamps) is None
    only = _stamp_history(stamps, current=FROZEN_FLOORS)
    assert len(only) == 1 and only[0]["matches_current_model"] is False


def test_stamp_history_skips_sessions_with_no_bands():
    groups = _stamp_history([("2026-08-26", LIVE_FLOORS), ("2026-08-14", None)])

    assert groups is None or len(groups) == 1


# --- stream caching -----------------------------------------------------------


def test_stream_is_fetched_once_per_activity(client):
    """Samples never change after upload, so a repeat call must not re-download."""
    c = client(details=details_payload(steady(145, 60)))
    first, _ = _hr_stream(11)
    second, _ = _hr_stream(11)

    assert first == second
    assert len(c.details_calls) == 1


def test_reconfiguring_the_client_drops_cached_streams(client):
    """A different client must never be served the previous one's samples."""
    client(details=details_payload(steady(145, 60)))
    _hr_stream(11)
    c2 = client(details=details_payload(steady(100, 60)))
    samples, _ = _hr_stream(11)

    assert len(c2.details_calls) == 1
    assert samples[0][1] == 100


def test_a_failure_is_cached_too_rather_than_retried_per_session(client):
    c = client(details=NO_HR_DETAILS)
    assert _hr_stream(11)[0] is None
    assert _hr_stream(11)[0] is None
    assert len(c.details_calls) == 1
