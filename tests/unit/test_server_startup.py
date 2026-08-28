"""Startup smoke tests for the packaged MCP server."""

import asyncio
from unittest.mock import Mock

import garmin_mcp
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
