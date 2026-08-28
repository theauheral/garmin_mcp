"""Unit tests for the env-var tool filter (_ToolFilter)."""

from garmin_mcp import _COACHING_PROFILE, _ToolFilter, _resolve_enabled_tools


class FakeApp:
    """Minimal stand-in for FastMCP: records which tools get registered."""

    def __init__(self):
        self.registered = []

    def tool(self, *args, **kwargs):
        explicit = kwargs.get("name") or (
            args[0] if args and isinstance(args[0], str) else None
        )

        def decorator(fn):
            self.registered.append(explicit or fn.__name__)
            return fn

        return decorator

    def run(self):
        return "ran"


def _register(filt, names):
    """Register one no-op tool per name through the filter."""
    for n in names:
        def fn():
            return None

        fn.__name__ = n
        filt.tool()(fn)


def test_no_filter_registers_all():
    app = FakeApp()
    filt = _ToolFilter(app, set(), set())
    _register(filt, ["get_a", "get_b"])
    assert app.registered == ["get_a", "get_b"]


def test_allowlist_only_registers_listed():
    app = FakeApp()
    filt = _ToolFilter(app, {"get_a"}, set())
    _register(filt, ["get_a", "get_b"])
    assert app.registered == ["get_a"]


def test_denylist_skips_listed():
    app = FakeApp()
    filt = _ToolFilter(app, set(), {"get_b"})
    _register(filt, ["get_a", "get_b"])
    assert app.registered == ["get_a"]


def test_allowlist_takes_precedence_over_denylist():
    app = FakeApp()
    filt = _ToolFilter(app, {"get_a"}, {"get_a"})
    _register(filt, ["get_a", "get_b"])
    assert app.registered == ["get_a"]


def test_matching_is_case_insensitive():
    app = FakeApp()
    filt = _ToolFilter(app, {"get_a"}, set())
    _register(filt, ["GET_A"])
    assert app.registered == ["GET_A"]


def test_unknown_filter_names_flags_typos():
    app = FakeApp()
    filt = _ToolFilter(app, {"get_a", "get_typo"}, set())
    _register(filt, ["get_a"])
    assert filt.unknown_filter_names() == ["get_typo"]


def test_explicit_name_kwarg_used_for_matching():
    app = FakeApp()
    filt = _ToolFilter(app, {"real_name"}, set())

    def fn():
        return None

    fn.__name__ = "internal_fn"
    filt.tool(name="real_name")(fn)
    assert app.registered == ["real_name"]
    assert filt.unknown_filter_names() == []


def test_passthrough_to_wrapped_app():
    app = FakeApp()
    filt = _ToolFilter(app, set(), set())
    assert filt.run() == "ran"


# --- Profile resolution (GARMIN_ENABLED_TOOLS) -------------------------------


def test_unset_falls_back_to_the_coaching_profile():
    """The default must be the curated profile, not the full 148-tool surface."""
    allowed, desc = _resolve_enabled_tools(None)
    assert allowed == set(_COACHING_PROFILE)
    assert "coaching" in desc and "default" in desc


def test_blank_value_is_treated_as_unset():
    assert _resolve_enabled_tools("   ")[0] == set(_COACHING_PROFILE)
    assert _resolve_enabled_tools(",,")[0] == set(_COACHING_PROFILE)


def test_all_disables_filtering_entirely():
    """An empty allowlist is _ToolFilter's 'register everything' signal."""
    allowed, desc = _resolve_enabled_tools("all")
    assert allowed == set()
    assert "no filter" in desc


def test_all_is_case_insensitive_and_wins_over_other_entries():
    assert _resolve_enabled_tools("ALL")[0] == set()
    assert _resolve_enabled_tools("get_stats,All")[0] == set()


def test_profile_name_expands():
    allowed, desc = _resolve_enabled_tools("coaching")
    assert allowed == set(_COACHING_PROFILE)
    assert desc == "coaching profile"


def test_profile_composes_with_explicit_names():
    allowed, desc = _resolve_enabled_tools("coaching, get_devices")
    assert allowed == set(_COACHING_PROFILE) | {"get_devices"}
    assert desc == "coaching profile + 1 named tool(s)"


def test_explicit_list_registers_only_those():
    allowed, desc = _resolve_enabled_tools("get_stats, GET_SLEEP_DATA")
    assert allowed == {"get_stats", "get_sleep_data"}
    assert desc == "explicit allowlist"


def test_coaching_profile_covers_the_skill_entry_points():
    """The composites the skill opens with, plus the workout write path.

    If a rename ever drops one of these the default session silently loses the
    tool the skill tells the model to call first.
    """
    must_have = {
        "get_wellness_brief", "get_training_week", "get_coach_report",
        "get_session_analysis", "get_execution_trend", "get_energy_curve",
        "get_health_flags", "get_plan_context", "get_running_dynamics",
        "get_wins", "upload_workout", "schedule_workouts",
        "unschedule_workout", "delete_workout", "create_manual_activity",
        "set_perceived_effort",
    }
    assert must_have <= set(_COACHING_PROFILE)


def test_coaching_profile_is_a_real_cut():
    """~72 of 148 — if this ever approaches the full surface the point is lost."""
    assert 30 <= len(_COACHING_PROFILE) <= 90
