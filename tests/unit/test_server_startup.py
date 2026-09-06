"""Startup smoke tests for the packaged MCP server."""

import asyncio
from unittest.mock import Mock

import garmin_mcp
from mcp.server.fastmcp import FastMCP


def _start_server(monkeypatch, enabled=None, transport=None):
    """Run main() without real Garmin auth, stopping before the server loop.

    Returns the captured {transport, tool_count, tool_names, app} of that start.
    """
    run_calls = []

    monkeypatch.delenv("GARMIN_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("GARMIN_MCP_HOST", raising=False)
    monkeypatch.delenv("GARMIN_MCP_PORT", raising=False)
    monkeypatch.delenv("GARMIN_DISABLED_TOOLS", raising=False)
    if transport is not None:
        monkeypatch.setenv("GARMIN_MCP_TRANSPORT", transport)
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
                "app": self,
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


class TestStatelessHttpTransport:
    """The server is meant to sit behind a broker that gives no session affinity.

    Session-based streamable HTTP binds a client to the process that answered
    `initialize` via an Mcp-Session-Id, and rejects everything else with
    `400 Bad Request: Missing session ID`. Behind a load balancer that routes
    the second request elsewhere, that reads as an outage rather than as a
    routing problem — so these settings are correctness, not tuning.
    """

    def test_app_is_built_stateless(self, monkeypatch):
        app = _start_server(monkeypatch)["app"]

        assert app.settings.stateless_http is True

    def test_streamable_http_is_mounted_at_slash_mcp(self, monkeypatch):
        """Remote clients hard-code the full URL; the path is a public contract."""
        app = _start_server(monkeypatch)["app"]

        assert app.settings.streamable_http_path == garmin_mcp._STREAMABLE_HTTP_PATH
        assert garmin_mcp._STREAMABLE_HTTP_PATH == "/mcp"

    def test_session_manager_really_is_stateless(self, monkeypatch):
        """Assert on the object that enforces it, not just on the setting.

        `stateless_http` reaching the session manager is what actually stops
        the 400s; a library rename would leave the setting true and the
        behaviour broken.
        """
        app = _start_server(monkeypatch, transport="streamable-http")["app"]
        app.streamable_http_app()  # builds the session manager

        assert app.session_manager.stateless is True

    def test_http_start_reports_the_url_and_mode(self, monkeypatch, capsys):
        """An operator pasting the wrong URL is the other way this looks down."""
        _start_server(monkeypatch, transport="streamable-http")

        err = capsys.readouterr().err
        assert "stateless" in err
        assert "http://127.0.0.1:8000/mcp" in err
