"""
Modular MCP Server for Garmin Connect Data
"""

import os
import sys
import base64
import threading

import requests
from mcp.server.fastmcp import FastMCP

from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError, GarminConnectTooManyRequestsError

# Import all modules
from garmin_mcp import token_utils
from garmin_mcp import activity_management
from garmin_mcp import health_wellness
from garmin_mcp import user_profile
from garmin_mcp import devices
from garmin_mcp import gear_management
from garmin_mcp import weight_management
from garmin_mcp import challenges
from garmin_mcp import training
from garmin_mcp import workouts
from garmin_mcp import workout_templates
from garmin_mcp import data_management
from garmin_mcp import womens_health
from garmin_mcp import nutrition
from garmin_mcp import workout_builders
from garmin_mcp import courses
from garmin_mcp import activity_analysis
from garmin_mcp import calendar_events
from garmin_mcp import composites


def is_interactive_terminal() -> bool:
    """Detect if running in interactive terminal vs MCP subprocess.

    Returns:
        bool: True if running in an interactive terminal, False otherwise
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def get_mfa() -> str:
    """Get MFA code from user input.

    Raises:
        RuntimeError: If running in non-interactive environment
    """
    if not is_interactive_terminal():
        print(
            "\nERROR: MFA code required but no interactive terminal available.\n"
            "Please run 'garmin-mcp-auth' in your terminal first.\n"
            "See: https://github.com/Taxuspt/garmin_mcp#mfa-setup\n",
            file=sys.stderr,
        )
        raise RuntimeError("MFA required but non-interactive environment")

    print(
        "\nGarmin Connect MFA required. Please check your email/phone for the code.",
        file=sys.stderr,
    )
    return input("Enter MFA code: ")


def _normalize_optional_user_config(value: str | None, key: str) -> str | None:
    """Treat an unresolved optional Desktop Extension value as unset."""
    unresolved_placeholder = f"${{user_config.{key}}}"
    return None if value == unresolved_placeholder else value


# Get credentials from environment
email = os.environ.get("GARMIN_EMAIL")
email_file = os.environ.get("GARMIN_EMAIL_FILE")
if email and email_file:
    raise ValueError(
        "Must only provide one of GARMIN_EMAIL and GARMIN_EMAIL_FILE, got both"
    )
elif email_file:
    with open(email_file, "r") as email_file:
        email = email_file.read().rstrip()

password = os.environ.get("GARMIN_PASSWORD")
password_file = os.environ.get("GARMIN_PASSWORD_FILE")
if password and password_file:
    raise ValueError(
        "Must only provide one of GARMIN_PASSWORD and GARMIN_PASSWORD_FILE, got both"
    )
elif password_file:
    with open(password_file, "r") as password_file:
        password = password_file.read().rstrip()

tokenstore = token_utils.get_token_path()
tokenstore_base64 = token_utils.get_token_base64_path()
is_cn = os.getenv("GARMIN_IS_CN", "false").lower() in ("true", "1", "yes")


# --- Tool filtering ---------------------------------------------------------
# This server registers 148 tools. Every tool schema is loaded into the
# consuming model's context on EVERY turn, so the full surface is a permanent
# tax on context, latency and cost — and on accuracy, since near-duplicate
# tools invite mis-selection. Tools are therefore filtered by name at
# registration time; no module is removed, the tool simply never reaches the
# model.
#   GARMIN_ENABLED_TOOLS  - comma-separated allowlist; if set, ONLY these register.
#                           Accepts profile names (see _PROFILES) and the
#                           special value "all" for the full 148-tool surface.
#                           Unset = the "coaching" profile (the default).
#   GARMIN_DISABLED_TOOLS - comma-separated denylist; ignored if an allowlist
#                           is in effect (which, by default, it is — pass
#                           GARMIN_ENABLED_TOOLS=all to use a denylist).
# Names are case-insensitive; names matching no real tool warn on stderr.
def _parse_tool_set(value):
    if not value:
        return set()
    return {name.strip().lower() for name in value.split(",") if name.strip()}


# Every tool named by the garmin-wellness skill
# (github.com/theauheral/garmin-cowork-plugin, skills/garmin-wellness/SKILL.md).
# Mechanically derived from that file, not hand-picked: the skill routes reads
# through the composites and writes through the workout/logging tools, and its
# guardrail section names each destructive tool it writes a rule for — a rule
# for a tool that isn't registered would be a rule about nothing. Regenerate
# with the plugin repo's scripts/derive-coaching-profile.py when SKILL.md moves.
_COACHING_PROFILE = frozenset({
    # activity_analysis
    "get_activity_fit_data",
    # activity_management
    "create_manual_activity", "get_activities", "get_activities_by_date",
    "get_activity_types", "set_activity_description",
    "set_activity_event_type", "set_activity_feel", "set_activity_name",
    "set_activity_type", "set_perceived_effort",
    # challenges
    "get_personal_record", "get_race_predictions",
    # composites
    "get_coach_report", "get_energy_curve", "get_execution_trend",
    "get_health_flags", "get_plan_context", "get_running_dynamics",
    "get_session_analysis", "get_training_week", "get_wellness_brief",
    "get_wins",
    # courses
    "delete_course",
    # data_management
    "add_body_composition", "add_hydration_data", "set_blood_pressure",
    # gear_management
    "remove_gear_from_activity",
    # health_wellness
    "get_all_day_stress", "get_body_battery", "get_daily_steps",
    "get_heart_rates", "get_heart_rates_summary", "get_sleep_data",
    "get_sleep_summary", "get_stats", "get_stress_summary",
    "get_training_readiness", "get_weekly_intensity_minutes",
    # nutrition
    "create_custom_food", "delete_custom_food", "delete_food_log",
    "log_custom_food", "log_food", "set_nutrition_daily_settings",
    "update_custom_food", "upsert_and_log",
    # training
    "get_endurance_score", "get_hrv_data", "get_hrv_trend",
    "get_progress_summary_between_dates", "get_respiration_trend",
    "get_training_load_balance", "get_training_load_trend",
    "get_training_status", "get_vo2max_trend", "request_reload",
    # weight_management
    "add_weigh_in", "add_weigh_in_with_timestamps", "delete_weigh_ins",
    # workout_builders
    "schedule_week",
    # workouts
    "delete_workout", "delete_workouts", "get_scheduled_workouts",
    "get_workout_by_id", "get_workouts", "schedule_workout",
    "schedule_workouts", "unschedule_workout", "unschedule_workouts",
    "upload_workout", "upload_workouts",
})

_PROFILES = {"coaching": _COACHING_PROFILE}
_DEFAULT_PROFILE = "coaching"
_ALL_TOOLS = "all"


def _resolve_enabled_tools(raw):
    """Expand GARMIN_ENABLED_TOOLS into the allowlist actually applied.

    Returns (allowlist, description). An empty allowlist means "register
    everything"; description is what gets reported on stderr at startup so the
    active filter is never invisible.

    Unset picks the default profile rather than the full surface: an unfiltered
    session costs the model 148 tool schemas per turn, which is the wrong
    default for the coaching agent this server exists to feed. "all" is the
    documented escape hatch, and profile names compose with explicit tool names
    ("coaching,get_devices").

    Blank counts as unset. A value that is set yet names nothing (",,  ,")
    raises ValueError instead.
    """
    names = _parse_tool_set(raw)
    if not names:
        if raw and raw.strip():
            # Set, yet naming nothing (",,  ,"): a broken config, not a
            # request for the default. Refuse to start rather than quietly
            # serve a profile the operator never asked for.
            raise ValueError(
                "Invalid GARMIN_ENABLED_TOOLS: expected at least one tool name"
            )
        return set(_PROFILES[_DEFAULT_PROFILE]), f"{_DEFAULT_PROFILE} profile (default)"
    if _ALL_TOOLS in names:
        return set(), "all tools (no filter)"
    allowed, used_profiles = set(), []
    for name in names:
        if name in _PROFILES:
            allowed |= _PROFILES[name]
            used_profiles.append(name)
        else:
            allowed.add(name)
    if used_profiles:
        extra = len(names) - len(used_profiles)
        desc = " + ".join(f"{p} profile" for p in sorted(used_profiles))
        return allowed, desc + (f" + {extra} named tool(s)" if extra else "")
    return allowed, "explicit allowlist"


def _resolve_tool_filters():
    """Read and validate the tool filter env vars at server startup.

    Returns (enabled, disabled, description). Resolved in main() rather than
    at import, so the env vars are read at the moment the server actually
    starts (matching the transport config below) — and validated before the
    Garmin login is even started, so a malformed allowlist (ValueError) stops
    the server on the spot instead of after a pointless login.
    """
    enabled_tools, description = _resolve_enabled_tools(
        os.getenv("GARMIN_ENABLED_TOOLS")
    )
    disabled_tools = _parse_tool_set(os.getenv("GARMIN_DISABLED_TOOLS"))
    return enabled_tools, disabled_tools, description


_VALID_TRANSPORTS = ("stdio", "streamable-http", "sse")

# Default per-call timeout (seconds). Garmin's API occasionally stalls a single
# request indefinitely; without a bound the blocking client call hangs until the
# MCP client's own timeout (~4 min) fires, reporting the whole server as
# unresponsive (see issue #248). 90s sits comfortably above a normal slow call
# yet well below that ceiling. Override with GARMIN_MCP_CALL_TIMEOUT; set 0 to
# disable the bound entirely.
_DEFAULT_CALL_TIMEOUT = 90.0


def _resolve_call_timeout() -> float:
    """Read GARMIN_MCP_CALL_TIMEOUT; fall back to the default on bad/absent input.

    A value <= 0 disables the timeout (returns 0.0).
    """
    raw = os.getenv("GARMIN_MCP_CALL_TIMEOUT")
    if raw is None or not raw.strip():
        return _DEFAULT_CALL_TIMEOUT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(
            f"Invalid GARMIN_MCP_CALL_TIMEOUT {raw!r}; using default "
            f"{_DEFAULT_CALL_TIMEOUT}s.",
            file=sys.stderr,
        )
        return _DEFAULT_CALL_TIMEOUT
    return value if value > 0 else 0.0


class _GarminProxy:
    """Wraps the Garmin client to bound call duration and clarify runtime errors.

    Two jobs:

    1. Timeout: each client call runs on a daemon worker thread and is abandoned
       if it does not return within the configured timeout (issue #248 — an
       occasional Garmin request stalls forever and the blocking call would
       otherwise hang the whole server until the MCP client gives up minutes
       later). A stalled call raises a clear, retry-able error instead; the
       abandoned daemon thread dies with the process and never blocks shutdown.
       Such stalls are rare and transient, so a fresh thread per call is cheap
       relative to the network round-trip it guards.

    2. Error translation: token expiry or rate-limiting during a tool call would
       otherwise surface a raw library traceback. Known Garmin exceptions become
       user-friendly messages instead.
    """

    _MESSAGES = {
        GarminConnectAuthenticationError: (
            "Garmin authentication expired. "
            "Re-run 'garmin-mcp-auth' to refresh your tokens and restart the server."
        ),
        GarminConnectTooManyRequestsError: (
            "Garmin rate limit hit. Wait a few minutes before retrying."
        ),
        GarminConnectConnectionError: (
            "Garmin Connect is unreachable. Check your network connection or try again later."
        ),
    }

    def __init__(self, client, timeout=None):
        self._client = client
        self._timeout = _resolve_call_timeout() if timeout is None else timeout

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _invoke(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except tuple(self._MESSAGES) as exc:
                for exc_type, msg in self._MESSAGES.items():
                    if isinstance(exc, exc_type):
                        error_details = str(exc)
                        full_msg = f"{msg} (Details: {error_details})" if error_details else msg
                        raise type(exc)(full_msg) from None
                raise

        def _call(*args, **kwargs):
            if not self._timeout:
                return _invoke(*args, **kwargs)

            # Run on a daemon thread and join with a timeout. The worker's
            # return value or exception is captured and replayed in the caller
            # so translated Garmin errors propagate unchanged.
            outcome = {}

            def _worker():
                try:
                    outcome["value"] = _invoke(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - replayed below
                    outcome["error"] = exc

            worker = threading.Thread(
                target=_worker, name=f"garmin-call:{name}", daemon=True
            )
            worker.start()
            worker.join(self._timeout)
            if worker.is_alive():
                raise TimeoutError(
                    f"Garmin request '{name}' did not return within "
                    f"{self._timeout:g}s and was abandoned. This is usually a "
                    f"transient stall on Garmin's side — please try again. "
                    f"(Adjust with GARMIN_MCP_CALL_TIMEOUT, or set it to 0 to "
                    f"disable the limit.)"
                )
            if "error" in outcome:
                raise outcome["error"]
            return outcome.get("value")

        return _call


class _ThreadFilteredStream:
    """Wraps a stream so only ``owner_thread``'s writes reach it.

    Installed as ``sys.stdout`` before the background Garmin login thread
    starts (issue #255): the login call may still emit stray progress
    output, and since ``sys.stdout`` is a single process-wide object, an
    unfiltered write from that thread would land in the middle of the
    JSON-RPC messages the main thread writes once ``app.run()`` starts,
    corrupting the stdio framing. Writes from any other thread are silently
    discarded, matching the previous (single-threaded) behavior of
    swallowing that output entirely.
    """

    def __init__(self, real_stream, owner_thread):
        self._real_stream = real_stream
        self._owner_thread = owner_thread

    def write(self, data):
        if threading.current_thread() is self._owner_thread:
            return self._real_stream.write(data)
        return len(data)

    def flush(self):
        self._real_stream.flush()

    def __getattr__(self, name):
        return getattr(self._real_stream, name)


class _PendingGarminClient:
    """Stands in for the real Garmin client until a background login finishes.

    Passed as the ``client`` argument to ``_GarminProxy``, so ``_GarminProxy``
    itself needs no changes: every attribute it looks up on ``self._client``
    comes through here first. Before login finishes, that lookup blocks (up
    to ``timeout`` seconds, ``0``/``None`` disables the bound) instead of
    returning immediately -- this is what lets ``main()`` call ``app.run()``
    right away instead of waiting on Garmin login before the MCP handshake
    can be answered (issue #255).
    """

    def __init__(self, timeout):
        self._timeout = timeout
        self._ready = threading.Event()
        self._client = None
        self._login_error = None

    def start(self, login_fn):
        """Run ``login_fn`` on a background daemon thread; store its outcome."""

        def _worker():
            try:
                client = login_fn()
            except BaseException as exc:  # noqa: BLE001 - stored, not raised here
                self._login_error = exc
            else:
                if client is None:
                    self._login_error = RuntimeError(
                        "Garmin login failed. Run 'garmin-mcp-auth' to "
                        "authenticate, then restart the server."
                    )
                    print(
                        "Garmin Connect client failed to initialize (see "
                        "errors above). Tool calls will fail until this is "
                        "fixed; run 'garmin-mcp-auth' and restart the server.",
                        file=sys.stderr,
                    )
                else:
                    self._client = client
                    print(
                        "Garmin Connect client initialized successfully.",
                        file=sys.stderr,
                    )
            finally:
                self._ready.set()

        threading.Thread(target=_worker, name="garmin-login", daemon=True).start()
        return self

    def __getattr__(self, name):
        if not self._ready.wait(self._timeout if self._timeout else None):
            raise RuntimeError(
                f"Garmin login did not finish within {self._timeout:g}s. "
                "Run 'garmin-mcp-auth' to verify your credentials, then "
                "restart the server."
            )
        if self._login_error is not None:
            raise self._login_error
        return getattr(self._client, name)


def _parse_transport_config() -> tuple[str, str, int]:
    """Read and validate HTTP transport env vars. Raises ValueError on bad input."""
    transport = os.getenv("GARMIN_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"Invalid GARMIN_MCP_TRANSPORT {transport!r}; "
            f"expected one of {', '.join(_VALID_TRANSPORTS)}"
        )
    # Bind to loopback by default: the HTTP transport performs no authentication,
    # so a 0.0.0.0 default would expose full read/write access to the user's
    # Garmin account to the whole network. Opt in explicitly with GARMIN_MCP_HOST.
    http_host = os.getenv("GARMIN_MCP_HOST", "127.0.0.1")
    http_port = int(os.getenv("GARMIN_MCP_PORT", "8000"))
    return transport, http_host, http_port


class _ToolFilter:
    """Wraps a FastMCP app to conditionally register tools by function name.

    Modules register via ``@app.tool()``; we intercept that decorator and skip
    registration for any tool not permitted by the env-var filter. All other
    attribute access (``run``, ``resource``, ...) passes through to the app.
    """

    def __init__(self, app, enabled, disabled):
        self._app = app
        self._enabled = enabled
        self._disabled = disabled
        self._seen = set()  # tool names encountered, for typo detection

    def _allowed(self, name):
        name = name.lower()
        if self._enabled:
            return name in self._enabled
        return name not in self._disabled

    def tool(self, *args, **kwargs):
        decorator = self._app.tool(*args, **kwargs)
        # Prefer the explicit registered name if given (@app.tool(name="x")),
        # so the env-var filter matches what the user actually configures.
        explicit = kwargs.get("name") or (
            args[0] if args and isinstance(args[0], str) else None
        )

        def wrapper(fn):
            name = explicit or getattr(fn, "__name__", "")
            self._seen.add(name.lower())
            if self._allowed(name):
                return decorator(fn)
            return fn  # skip registration; tool never reaches the LLM

        return wrapper

    def unknown_filter_names(self):
        """Configured names that never matched a real tool (likely typos)."""
        configured = self._enabled or self._disabled
        return sorted(configured - self._seen)

    def __getattr__(self, item):
        return getattr(self._app, item)
# ---------------------------------------------------------------------------


def init_api(email, password):
    """Initialize Garmin API with your credentials."""
    import io

    # Claude Desktop may leave blank optional user_config values as literal
    # placeholders. Do not mistake those strings for credentials and trigger a
    # rate-limited Garmin login from a non-interactive MCP process.
    email = _normalize_optional_user_config(email, "garmin_email")
    password = _normalize_optional_user_config(password, "garmin_password")

    try:
        # Using Oauth1 and OAuth2 token files from directory
        print(
            f"Trying to login to Garmin Connect using token data from directory '{tokenstore}'...\n",
            file=sys.stderr,
        )

        # Using Oauth1 and Oauth2 tokens from base64 encoded string
        # print(
        #     f"Trying to login to Garmin Connect using token data from file '{tokenstore_base64}'...\n"
        # )
        # dir_path = os.path.expanduser(tokenstore_base64)
        # with open(dir_path, "r") as token_file:
        #     tokenstore = token_file.read()

        # Suppress stderr during token validation to hide noisy library
        # warnings. stdout is deliberately NOT swapped here: by the time
        # init_api() runs, sys.stdout is a _ThreadFilteredStream installed
        # in main() before this call's background thread was started, which
        # already discards any stray write from this thread on its own. A
        # second swap here would instead risk swallowing real MCP protocol
        # output written concurrently by the server's own thread (#255).
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            garmin = Garmin(is_cn=is_cn)
            garmin.login(tokenstore)
        finally:
            sys.stderr = old_stderr

    except (FileNotFoundError, GarminConnectConnectionError, GarminConnectTooManyRequestsError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again

        # Check if we're in a non-interactive environment without credentials
        if not is_interactive_terminal() and (not email or not password):
            print(
                "ERROR: OAuth tokens not found and no interactive terminal available.\n"
                "Please authenticate first:\n"
                "  1. Run: garmin-mcp-auth\n"
                "  2. Enter your credentials and MFA code\n"
                "  3. Restart your MCP client\n"
                f"Tokens will be saved to: {tokenstore}\n",
                file=sys.stderr,
            )
            return None

        print(
            "Login tokens not present, login with your Garmin Connect credentials to generate them.\n"
            f"They will be stored in '{tokenstore}' for future use.\n",
            file=sys.stderr,
        )
        try:
            garmin = Garmin(
                email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa, return_on_mfa=True
            )
            # sys.stdout is a _ThreadFilteredStream (installed in main());
            # any stray progress output from this call is already discarded
            # without a local swap here (see the token-load path above).
            result1, result2 = garmin.login()
            if result1 == "needs_mfa":
                mfa_code = get_mfa()
                garmin.resume_login(result2, mfa_code)
            # Save Oauth1 and Oauth2 token files to directory for next login
            garmin.client.dump(tokenstore)
            # Restrict the freshly written tokens to owner-only. These are
            # ~6-month bearer credentials; the default umask would otherwise
            # leave them world-readable on multi-user hosts.
            token_utils.secure_token_dir(tokenstore)
            print(
                f"Oauth tokens stored in '{tokenstore}' directory for future use. (first method)\n",
                file=sys.stderr,
            )
            # Encode Oauth1 and Oauth2 tokens to base64 string and save to file for next login (alternative way)
            token_json_path = os.path.join(tokenstore, "garmin_tokens.json")
            with open(token_json_path, "r") as f:
                token_data = f.read()
            token_base64 = base64.b64encode(token_data.encode()).decode()
            with open(tokenstore_base64, "w") as token_file:
                token_file.write(token_base64)
            os.chmod(tokenstore_base64, 0o600)
            print(
                f"Oauth tokens encoded as base64 string and saved to '{tokenstore_base64}' file for future use. (second method)\n",
                file=sys.stderr,
            )
        except (
            FileNotFoundError,
            GarminConnectConnectionError,
            GarminConnectTooManyRequestsError,
            GarminConnectAuthenticationError,
            requests.exceptions.HTTPError,
        ) as err:
            error_msg = str(err)

            # Provide clean, actionable error messages
            print("\nAuthentication failed.", file=sys.stderr)

            if isinstance(err, GarminConnectAuthenticationError):
                if "MFA" in error_msg or "code" in error_msg.lower():
                    print("MFA code may be incorrect or expired.", file=sys.stderr)
                else:
                    print("Invalid email or password.", file=sys.stderr)
            elif isinstance(err, GarminConnectTooManyRequestsError):
                print(
                    "Too many requests. Please wait and try again.", file=sys.stderr
                )
            elif isinstance(err, GarminConnectConnectionError):
                if "401" in error_msg or "Unauthorized" in error_msg:
                    print(
                        "Invalid credentials. Please check your email and password.",
                        file=sys.stderr,
                    )
                elif "500" in error_msg or "503" in error_msg:
                    print(
                        "Garmin Connect service issue. Please try again later.",
                        file=sys.stderr,
                    )
                else:
                    print(f"Error: {error_msg.split(':')[0]}", file=sys.stderr)
            elif isinstance(err, requests.exceptions.HTTPError):
                print("Network error. Please check your connection.", file=sys.stderr)
            else:
                print(f"Error: {error_msg.split(':')[0]}", file=sys.stderr)

            print(
                f"\nTip: Run 'garmin-mcp-auth' to authenticate interactively.",
                file=sys.stderr,
            )
            return None

    return garmin


def main():
    """Initialize the MCP server and register all tools"""

    # On Windows, stdout runs in text mode and translates \n to \r\n, which
    # breaks the MCP stdio framing that Claude Desktop and other clients expect.
    # Force binary-transparent newlines so JSON messages arrive intact.
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline="\n")

    # Garmin login (init_api) runs on a background thread below so it can't
    # block the MCP handshake (issue #255). That thread may still emit
    # stray writes to stdout; route sys.stdout so only this thread's writes
    # reach the real stream, protecting the JSON-RPC framing this thread
    # writes once app.run() starts.
    sys.stdout = _ThreadFilteredStream(sys.stdout, threading.current_thread())

    # --- Transport configuration --------------------------------------------
    # By default the server speaks stdio (Claude Desktop, MCP Inspector, etc.).
    # Set GARMIN_MCP_TRANSPORT=streamable-http (or sse) to serve over HTTP.
    #   GARMIN_MCP_TRANSPORT - stdio (default) | streamable-http | sse
    #   GARMIN_MCP_HOST      - bind address for HTTP transports (default 127.0.0.1)
    #   GARMIN_MCP_PORT      - bind port for HTTP transports (default 8000)
    try:
        enabled_tools, disabled_tools, tool_filter_desc = _resolve_tool_filters()
        transport, http_host, http_port = _parse_transport_config()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # Start Garmin login in the background so it never blocks the MCP
    # handshake (issue #255). Tool calls block on it individually instead,
    # through _GarminProxy -> _PendingGarminClient, once they're actually
    # invoked. 90s comfortably covers a normal slow login while still
    # failing well before a client's own initialize timeout would matter
    # again on a later call.
    pending_client = _PendingGarminClient(timeout=90.0)
    pending_client.start(lambda: init_api(email, password))
    garmin_client = _GarminProxy(pending_client)

    # Configure all modules with the Garmin client
    activity_management.configure(garmin_client)
    health_wellness.configure(garmin_client)
    user_profile.configure(garmin_client)
    devices.configure(garmin_client)
    gear_management.configure(garmin_client)
    weight_management.configure(garmin_client)
    challenges.configure(garmin_client)
    training.configure(garmin_client)
    workouts.configure(garmin_client)
    data_management.configure(garmin_client)
    womens_health.configure(garmin_client)
    nutrition.configure(garmin_client)
    workout_builders.configure(garmin_client)
    courses.configure(garmin_client)
    activity_analysis.configure(garmin_client)
    calendar_events.configure(garmin_client)
    composites.configure(garmin_client)

    # Create the MCP app, wrapped so the env-var filter can drop tools.
    # host/port only matter for the HTTP transports; stdio ignores them.
    fastmcp = FastMCP("Garmin Connect v1.0", host=http_host, port=http_port)
    app = _ToolFilter(fastmcp, enabled_tools, disabled_tools)
    if enabled_tools:
        # Name the escape hatch on every filtered start: a tool the user
        # expected and cannot find must not look like a missing feature.
        print(
            f"Tool filter: {tool_filter_desc} — {len(enabled_tools)} tool(s). "
            f"Set GARMIN_ENABLED_TOOLS=all for the full surface.",
            file=sys.stderr,
        )
    elif disabled_tools:
        print(f"Tool filter: denylist of {len(disabled_tools)} tool(s).", file=sys.stderr)
    else:
        print(f"Tool filter: {tool_filter_desc}.", file=sys.stderr)

    # Register tools from all modules
    app = activity_management.register_tools(app)
    app = health_wellness.register_tools(app)
    app = user_profile.register_tools(app)
    app = devices.register_tools(app)
    app = gear_management.register_tools(app)
    app = weight_management.register_tools(app)
    app = challenges.register_tools(app)
    app = training.register_tools(app)
    app = workouts.register_tools(app)
    app = data_management.register_tools(app)
    app = womens_health.register_tools(app)
    app = nutrition.register_tools(app)
    app = workout_builders.register_tools(app)
    app = courses.register_tools(app)
    app = activity_analysis.register_tools(app)
    app = calendar_events.register_tools(app)
    app = composites.register_tools(app)

    # Register resources (workout templates)
    app = workout_templates.register_resources(app)

    # Warn about filter entries that matched no tool (most likely typos)
    unknown = app.unknown_filter_names()
    if unknown:
        # Either a typo in the env var, or a profile entry that no longer
        # matches a real tool after a version bump. Both are silent bugs
        # otherwise: the tool just never appears.
        print(
            f"Tool filter: warning — name(s) not found and ignored: {', '.join(unknown)}",
            file=sys.stderr,
        )

    # When serving over HTTP, expose a plain health endpoint for k8s probes.
    # The MCP endpoint itself requires a handshake and isn't probe-friendly.
    if transport != "stdio":
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse

        @fastmcp.custom_route("/healthz", methods=["GET"])
        async def healthz(_request: "Request") -> "PlainTextResponse":
            return PlainTextResponse("ok")

        print(
            f"Serving MCP over {transport} on {http_host}:{http_port}",
            file=sys.stderr,
        )

    # Run the MCP server
    app.run(transport=transport)


if __name__ == "__main__":
    main()
