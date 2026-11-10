"""Sleep timing regularity: spread, social jet lag, Sleep Regularity Index.

Nights are keyed by WAKE-UP date and carry the watch's local wall-clock, encoded
as if it were UTC (what Garmin's *Local timestamps are).
"""

import datetime

from garmin_mcp.composites import _sleep_regularity


def local_ms(day, hour, minute=0):
    """A local wall-clock instant as the *Local timestamps encode it."""
    dt = datetime.datetime.combine(day, datetime.time(hour, minute), datetime.timezone.utc)
    return dt.timestamp() * 1000


def night(wake_day, bed_hour, bed_min, wake_hour, wake_min=0):
    """The night ending on wake_day: bed the evening before (or after midnight)."""
    bed_day = wake_day - datetime.timedelta(days=1) if bed_hour >= 12 else wake_day
    return {
        "date": wake_day.isoformat(),
        "start_ms": local_ms(bed_day, bed_hour, bed_min),
        "end_ms": local_ms(wake_day, wake_hour, wake_min),
    }


MON = datetime.date(2026, 11, 2)   # a Monday


def days(n, start=MON):
    return [start + datetime.timedelta(days=i) for i in range(n)]


def test_identical_nights_are_perfectly_regular():
    out = _sleep_regularity([night(d, 23, 0, 7) for d in days(10)])
    assert out["sleep_regularity_index"] == 100
    assert out["midpoint_sd_min"] == 0
    assert out["avg_bedtime"] == "23:00"
    assert out["avg_wake"] == "07:00"
    assert out["avg_midpoint"] == "03:00"
    assert out["avg_sleep_h"] == 8
    assert out["social_jet_lag"]["shift_h"] == 0
    assert out["social_jet_lag"]["flag"] is False


def test_later_weekend_nights_are_social_jet_lag():
    # wake days Sat/Sun (weekday>=5) go to bed at 01:00 and wake 09:00
    nights = []
    for d in days(14):
        if d.weekday() >= 5:
            nights.append(night(d, 1, 0, 9))
        else:
            nights.append(night(d, 23, 0, 7))
    out = _sleep_regularity(nights)
    sjl = out["social_jet_lag"]
    assert sjl["weekday_midpoint"] == "03:00"
    assert sjl["weekend_midpoint"] == "05:00"
    assert sjl["shift_h"] == 2.0
    assert sjl["flag"] is True
    assert sjl["weekday_nights"] == 10 and sjl["weekend_nights"] == 4
    assert out["sleep_regularity_index"] < 100


def test_bedtimes_either_side_of_midnight_do_not_wrap():
    # 23:30 and 00:30 alternate: the mean is midnight, not midday
    nights = [night(d, 23, 30, 7) if i % 2 == 0 else night(d, 0, 30, 7) for i, d in enumerate(days(10))]
    out = _sleep_regularity(nights)
    assert out["avg_bedtime"] == "00:00"
    assert 25 < out["bedtime_sd_min"] < 40       # ~31 min SD, not hundreds


def test_an_erratic_week_scores_lower_than_a_steady_one():
    steady = _sleep_regularity([night(d, 23, 0, 7) for d in days(10)])
    beds = [(22, 0, 6), (2, 30, 10), (23, 45, 7), (1, 0, 8), (21, 30, 5), (3, 0, 11), (23, 0, 7),
            (2, 0, 9), (22, 30, 6), (0, 30, 8)]
    erratic = _sleep_regularity([night(d, h, m, w) for d, (h, m, w) in zip(days(10), beds)])
    assert erratic["sleep_regularity_index"] < steady["sleep_regularity_index"] - 20
    assert erratic["midpoint_sd_min"] > 60


def test_too_few_nights_scores_nothing():
    assert _sleep_regularity([night(d, 23, 0, 7) for d in days(4)]) is None


def test_nights_without_a_valid_window_are_skipped():
    nights = [night(d, 23, 0, 7) for d in days(6)]
    nights.append({"date": days(7)[6].isoformat(), "start_ms": None, "end_ms": None})
    nights.append({"date": days(8)[7].isoformat(), "start_ms": 5.0, "end_ms": 5.0})   # zero length
    assert _sleep_regularity(nights)["nights_used"] == 6


def test_the_index_only_counts_consecutive_nights():
    """Every other night recorded: there is no pair of adjacent days, so no index
    (a missing night must not read as 'awake all day')."""
    nights = [night(d, 23, 0, 7) for d in days(14)[::2]]
    out = _sleep_regularity(nights)
    assert out["nights_used"] == 7
    assert "sleep_regularity_index" not in out
    assert out["midpoint_sd_min"] == 0            # spread still reported


def test_a_weekend_with_one_night_is_not_scored_for_jet_lag():
    nights = [night(d, 23, 0, 7) for d in days(5)]    # Mon-Fri wake days only
    out = _sleep_regularity(nights)
    assert "social_jet_lag" not in out
