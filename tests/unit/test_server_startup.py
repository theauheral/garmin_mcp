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


def test_default_start_registers_only_the_coaching_profile(monkeypatch):
    """The context cost of a default session is the whole point of the filter."""
    started = _start_server(monkeypatch)

    assert set(started["tool_names"]) == set(garmin_mcp._COACHING_PROFILE)
    assert "get_wellness_brief" in started["tool_names"]
    # Registered upstream, deliberately out of the coaching profile.
    assert "get_devices" not in started["tool_names"]


def test_all_restores_the_full_surface(monkeypatch):
    started = _start_server(monkeypatch, enabled="all")
    full = set(started["tool_names"])
    profile = set(garmin_mcp._COACHING_PROFILE)

    assert profile < full                 # strict superset — nothing is removed
    assert len(full - profile) >= 40      # ... and the profile is a real cut
    assert "get_devices" in full
    assert "get_workouts" in full


def test_every_coaching_profile_name_is_a_real_tool(monkeypatch):
    """Drift guard: a profile entry that matches nothing is a silently missing
    tool, and the only symptom is the model not having it."""
    full = set(_start_server(monkeypatch, enabled="all")["tool_names"])

    assert set(garmin_mcp._COACHING_PROFILE) - full == set()


def test_unknown_name_warns_on_stderr(monkeypatch, capsys):
    """A typo must not fail silently — it looks identical to a missing tool."""
    _start_server(monkeypatch, enabled="get_wellness_brief,get_wellnes_brief")

    assert "get_wellnes_brief" in capsys.readouterr().err
