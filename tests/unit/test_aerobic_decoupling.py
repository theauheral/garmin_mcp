"""Aerobic decoupling (Pa:HR) from the activity stream.

Efficiency is speed per heartbeat; decoupling is how far it falls from the
first half of a steady effort to the second. These pin the arithmetic, the
warm-up rule, and the cases where the number must not be offered as a read.
"""

from garmin_mcp.composites import _aerobic_decoupling


def _rows(minutes, hr_at, speed_at, step=10):
    """Stream rows every `step` s: hr_at(frac)/speed_at(frac) are functions of 0..1."""
    n = int(minutes * 60 / step)
    return [
        {"t": i * step, "hr": hr_at(i / n), "dist": None, "speed": speed_at(i / n)}
        for i in range(n + 1)
    ]


def test_flat_effort_has_no_decoupling():
    out = _aerobic_decoupling(_rows(40, lambda f: 140, lambda f: 3.0))
    assert out["decoupling_pct"] == 0
    assert out["steady"] is True
    assert out["read"].startswith("under 5%")


def test_rising_hr_at_constant_pace_decouples():
    # HR 140 -> 154 at a constant 3.0 m/s: second-half HR averages ~7% above the first
    out = _aerobic_decoupling(_rows(40, lambda f: 140 + 14 * f, lambda f: 3.0))
    assert 4.5 < out["decoupling_pct"] < 6.5
    assert out["second_half"]["avg_hr"] > out["first_half"]["avg_hr"]
    assert out["first_half"]["pace_min_per_km"] == out["second_half"]["pace_min_per_km"]


def test_large_fade_reads_over_eight():
    out = _aerobic_decoupling(_rows(40, lambda f: 140 + 30 * f, lambda f: 3.0))
    assert out["decoupling_pct"] > 8
    assert out["read"].startswith("over 8%")


def test_warmup_skipped_only_on_long_runs():
    long_run = _aerobic_decoupling(_rows(60, lambda f: 140, lambda f: 3.0))
    short_run = _aerobic_decoupling(_rows(30, lambda f: 140, lambda f: 3.0))
    assert long_run["warmup_skipped_min"] == 10
    assert short_run["warmup_skipped_min"] == 0


def test_warmup_does_not_pollute_the_first_half():
    # a slow, low-HR first 10 min would read as a huge fade if it stayed in the first half
    def hr(f):
        return 110 if f < 10 / 60 else 145

    def speed(f):
        return 2.0 if f < 10 / 60 else 3.0

    out = _aerobic_decoupling(_rows(60, hr, speed))
    assert abs(out["decoupling_pct"]) < 1


def test_too_short_is_not_computed():
    assert _aerobic_decoupling(_rows(15, lambda f: 140, lambda f: 3.0)) is None


def test_no_speed_stream_is_not_computed():
    rows = [{"t": i * 10, "hr": 140, "dist": None, "speed": None} for i in range(300)]
    assert _aerobic_decoupling(rows) is None


def test_stopped_time_is_not_running_time():
    rows = _rows(40, lambda f: 140, lambda f: 3.0)
    for r in rows[100:130]:           # a 5-minute stop
        r["speed"] = 0.0
    out = _aerobic_decoupling(rows)
    assert out["decoupling_pct"] == 0
    assert out["usable_min"] < 36


def test_intervals_are_flagged_not_steady():
    def speed(f):
        return 4.5 if int(f * 40) % 2 == 0 else 2.0   # alternating hard/easy minutes

    out = _aerobic_decoupling(_rows(40, lambda f: 150, speed))
    assert out["steady"] is False
    assert "not a steady effort" in out["read"]
