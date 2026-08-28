"""Startup smoke tests for the packaged MCP server."""

import asyncio
from unittest.mock import Mock

import garmin_mcp
import pytest
from mcp.server.fastmcp import FastMCP


def _start_server(monkeypatch, enabled=None):
    """Run main() without real Garmin auth, stopping before the server loop.

    Returns the captured {transport, tool_count, tool_names} of that start.
    """
    run_calls = []

    monkeypatch.delenv("GARMIN_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("GARMIN_MCP_HOST", raising=False)
    monkeypatch.delenv("GARMIN_MCP_PORT", raising=False)
    monkeypatch.delenv("GARMIN_DISABLED_TOOLS", raising=False)
    monkeypatch.delenv("GARMIN_TOOL_PROFILE", raising=False)
    if enabled is None:
        monkeypatch.delenv("GARMIN_ENABLED_TOOLS", raising=False)
    else:
        monkeypatch.setenv("GARMIN_ENABLED_TOOLS", enabled)
    monkeypatch.setattr(garmin_mcp, "init_api", lambda _email, _password: Mock())

    def capture_run(self, **kwargs):
        tools = asyncio.run(self.list_tools())
        run_calls.append(
            {
                "transport": kwargs.get("transport"),
                "tool_count": len(tools),
                "tool_names": [tool.name for tool in tools],
                "tools": {tool.name: tool for tool in tools},
            }
        )

    monkeypatch.setattr(FastMCP, "run", capture_run)
    garmin_mcp.main()
    assert run_calls
    return run_calls[0]


def test_main_registers_tools_and_starts_stdio(monkeypatch):
    started = _start_server(monkeypatch)

    assert started["transport"] == "stdio"
    assert started["tool_count"] >= 10
    assert "get_workouts" in started["tool_names"]


def test_tool_annotations_are_applied(monkeypatch):
    # The full surface: get_devices etc. are outside the default coaching profile.
    tools = _start_server(monkeypatch, enabled="all")["tools"]
    assert tools["get_devices"].annotations.readOnlyHint is True
    assert tools["create_run_workout"].annotations.destructiveHint is False
    assert tools["delete_workout"].annotations.destructiveHint is True
    assert tools["schedule_workout"].annotations.idempotentHint is True
    assert tools["set_fit_download_dir"].annotations.openWorldHint is False


def test_main_rejects_malformed_allowlist_before_garmin_initialization(
    monkeypatch, capsys
):
    init_api = Mock()
    monkeypatch.setenv("GARMIN_ENABLED_TOOLS", ",,  ,")
    monkeypatch.setattr(garmin_mcp, "init_api", init_api)
    monkeypatch.setattr(FastMCP, "run", lambda _self, **_kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        garmin_mcp.main()

    assert exc_info.value.code == 1
    assert (
        "Invalid GARMIN_ENABLED_TOOLS: expected at least one tool name"
        in capsys.readouterr().err
    )
    init_api.assert_not_called()


def test_main_starts_server_before_garmin_login_completes(monkeypatch):
    """app.run() must not wait on Garmin login to finish (issue #255).

    slow_init_api() blocks until the test releases it. If main() still
    called init_api() synchronously before app.run(), this test would fail
    with the assertion message below instead of hanging forever, because
    the release only happens *after* main() has already returned.
    """
    import threading

    monkeypatch.delenv("GARMIN_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("GARMIN_MCP_HOST", raising=False)
    monkeypatch.delenv("GARMIN_MCP_PORT", raising=False)

    login_release = threading.Event()

    def slow_init_api(_email, _password):
        released = login_release.wait(2)
        assert released, "login must not need to finish before app.run() is called"
        return Mock()

    monkeypatch.setattr(garmin_mcp, "init_api", slow_init_api)

    reached_run = threading.Event()

    def capture_run(self, **kwargs):
        reached_run.set()

    monkeypatch.setattr(FastMCP, "run", capture_run)

    try:
        garmin_mcp.main()
        assert reached_run.is_set()
    finally:
        login_release.set()


def test_main_applies_tool_profile(monkeypatch, capsys):
    """GARMIN_TOOL_PROFILE=minimal registers exactly the minimal set."""
    from garmin_mcp.tool_profiles import TOOL_PROFILES

    run_calls = []
    monkeypatch.delenv("GARMIN_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("GARMIN_ENABLED_TOOLS", raising=False)
    monkeypatch.delenv("GARMIN_DISABLED_TOOLS", raising=False)
    monkeypatch.setenv("GARMIN_TOOL_PROFILE", "minimal")
    monkeypatch.setattr(garmin_mcp, "init_api", lambda _email, _password: Mock())

    def capture_run(self, **kwargs):
        tools = asyncio.run(self.list_tools())
        run_calls.append({tool.name for tool in tools})

    monkeypatch.setattr(FastMCP, "run", capture_run)

    garmin_mcp.main()

    assert run_calls[0] == TOOL_PROFILES["minimal"]
    assert "Tool profile: minimal (" in capsys.readouterr().err


def test_main_profile_composes_with_allow_and_deny_lists(monkeypatch):
    from garmin_mcp.tool_profiles import TOOL_PROFILES

    run_calls = []
    monkeypatch.delenv("GARMIN_MCP_TRANSPORT", raising=False)
    monkeypatch.setenv("GARMIN_TOOL_PROFILE", "nutrition")
    monkeypatch.setenv("GARMIN_ENABLED_TOOLS", "get_activities")
    monkeypatch.setenv("GARMIN_DISABLED_TOOLS", "delete_food_log,delete_custom_food")
    monkeypatch.setattr(garmin_mcp, "init_api", lambda _email, _password: Mock())
    monkeypatch.setattr(
        FastMCP, "run", lambda self, **kwargs: run_calls.append({t.name for t in asyncio.run(self.list_tools())})
    )

    garmin_mcp.main()

    expected = (TOOL_PROFILES["nutrition"] | {"get_activities"}) - {"delete_food_log", "delete_custom_food"}
    assert run_calls[0] == expected


def test_main_rejects_unknown_tool_profile_before_garmin_initialization(monkeypatch, capsys):
    init_api = Mock()
    monkeypatch.setenv("GARMIN_TOOL_PROFILE", "helth")
    monkeypatch.setattr(garmin_mcp, "init_api", init_api)
    monkeypatch.setattr(FastMCP, "run", lambda _self, **_kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        garmin_mcp.main()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Unknown GARMIN_TOOL_PROFILE 'helth'" in err
    assert "Known profiles:" in err
    init_api.assert_not_called()


def test_default_start_registers_only_the_coaching_profile(monkeypatch):
    """The context cost of a default session is the whole point of the filter."""
    started = _start_server(monkeypatch)

    assert set(started["tool_names"]) == set(garmin_mcp._COACHING_PROFILE)
    assert "get_wellness_brief" in started["tool_names"]
    # Registered upstream, deliberately out of the coaching profile.
    assert "get_devices" not in started["tool_names"]


def test_all_restores_the_full_surface(monkeypatch):
    started = _start_server(monkeypatch, enabled="all")

    assert started["tool_count"] > 2 * len(garmin_mcp._COACHING_PROFILE)
    assert "get_devices" in started["tool_names"]
    assert "get_workouts" in started["tool_names"]


def test_every_coaching_profile_name_is_a_real_tool(monkeypatch):
    """Drift guard: a profile entry that matches nothing is a silently missing
    tool, and the only symptom is the model not having it."""
    full = set(_start_server(monkeypatch, enabled="all")["tool_names"])

    assert set(garmin_mcp._COACHING_PROFILE) - full == set()


def test_unknown_name_warns_on_stderr(monkeypatch, capsys):
    """A typo must not fail silently — it looks identical to a missing tool."""
    _start_server(monkeypatch, enabled="get_wellness_brief,get_wellnes_brief")

    assert "get_wellnes_brief" in capsys.readouterr().err
