"""MCP (Model Context Protocol) client integration — the chatbot as an MCP
*consumer*. Tools from every configured external MCP server (config.py's
MCP_SERVERS — a JSON array, same config-driven-registry shape as
EXTRA_LLM_PROVIDERS in llm/router.py) are pulled in at process start and
merged into the SAME tool set every track already presents to a model
(see agent/tools.py's TOOL_SCHEMAS.update(mcp_tool_specs()) and its
execute() dispatch to call_mcp_tool below) — an MCP tool is indistinguishable
from a hand-written one or a JSON plugin to perceive_intent/output_guard,
same principle as plugins.py's PLUGINS merge.

Unlike a hand-written PM tool or a JSON plugin, an MCP tool's arguments are
NOT fuzzy-resolved against live PM data (§6.1 step 7 doesn't apply — "which
timezone" isn't a name that needs matching to a real project/task id) — they
pass straight from perceive_intent's tool call to the MCP server as-is, the
same "no entity resolution needed" shape a plain literal argument (a due
date string, a status enum) already has on every built-in tool.

Real MCP servers speak stdio (a subprocess, e.g. `python -m some_server` or
`npx @some/mcp-server` — exactly how Claude Desktop/Claude Code configure
one) or a remote HTTP/SSE URL, optionally with a static `headers` entry
(e.g. `{"Authorization": "Bearer <token>"}`) for a remote server that
accepts a manually-obtained personal API token. Both transports are
supported (see _open_session below); the bundled default (config.py's
MCP_SERVERS default) is a small, safe, fully offline example server —
mcp_servers/utility_server.py — added so this integration is exercised
out of the box with zero external network access or operator setup.
Point MCP_SERVERS at any real external MCP server to replace/extend it —
that's the whole point of the config-driven registry, same as
EXTRA_LLM_PROVIDERS. A real, verified example (as of this writing):
Notion's own official server, `@notionhq/notion-mcp-server` —
    {"name": "notion", "command": "npx", "args": ["-y", "@notionhq/notion-mcp-server"],
     "env": {"NOTION_TOKEN": "ntn_..."}}
— a single static integration-secret env var, no OAuth involved.

`headers` does NOT implement OAuth (no browser authorization redirect/
callback, no per-user token storage/refresh) — several notable services'
OFFICIAL remote MCP servers (GitHub's at api.githubcopilot.com/mcp,
Google's Gmail/Calendar servers, and Linear's at mcp.linear.app) require
a full OAuth 2.0/2.1 browser-based flow, which is a genuinely different,
larger feature (a per-user "connected accounts" subsystem) than this
module takes on. `headers` covers the "I already have a static personal
API token, pass it on every request" case only — check the specific
service's own docs for whether it offers one as an alternative to OAuth.

An MCP server that's unreachable/misconfigured is skipped with a logged
warning, never a hard failure — the same "optional, additive" philosophy
malformed JSON plugins and an empty EXTRA_LLM_PROVIDERS already get.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Runs an MCP coroutine to completion on its own thread + event loop —
    same bridge adk_agent.py's _run_async uses, for the same reason: the
    calling code (agent/tools.py's execute(), called from a plain sync
    handle_message) must not collide with whatever event loop the FastAPI
    worker handling this request is already running."""
    outcome: dict[str, Any] = {}

    def _target():
        loop = asyncio.new_event_loop()
        try:
            outcome["value"] = loop.run_until_complete(coro)
        except Exception as e:  # noqa: BLE001 — re-raised on the caller's thread below
            outcome["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@dataclass
class _McpServerConfig:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None  # remote HTTP/SSE MCP server, instead of a stdio subprocess
    # Static per-request headers for a `url` server — e.g. {"Authorization":
    # "Bearer <token>"} for a remote server that accepts a manually-obtained
    # personal API token instead of (or alongside) a full OAuth flow. This
    # does NOT implement OAuth itself (no browser redirect/callback, no
    # per-user token storage/refresh) — that's a genuinely different,
    # bigger feature; this only covers "I already have a static token,
    # pass it on every request," the same way `env` already covers a
    # stdio server's credentials.
    headers: dict[str, str] | None = None


def _configured_servers() -> list[_McpServerConfig]:
    raw = settings().MCP_SERVERS
    try:
        entries = json.loads(raw) if raw and raw.strip() else []
    except (json.JSONDecodeError, TypeError):
        logger.warning("MCP_SERVERS is not valid JSON — no MCP tools loaded")
        return []
    out = []
    for e in entries:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        if not e.get("command") and not e.get("url"):
            continue
        command = e.get("command")
        # The bundled servers' config (and any operator's) can only ever
        # know the CONVENTIONAL launcher name "python" — never which exact
        # binary a given deploy target actually exposes on PATH (some hosts
        # only have "python3", not "python"). Resolve to the interpreter
        # actually running THIS process instead of trusting PATH to have
        # guessed right; every other literal command (e.g. "npx") is left
        # untouched, since that resolution only makes sense for Python.
        if command == "python":
            command = sys.executable
        out.append(_McpServerConfig(name=e["name"], command=command,
                                    args=e.get("args", []), env=e.get("env"),
                                    url=e.get("url"), headers=e.get("headers")))
    return out


def _mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


async def _open_session(server: _McpServerConfig):
    """Returns an async context manager yielding an initialized ClientSession
    for one server, over whichever transport it's configured for."""
    if server.url:
        from mcp.client.sse import sse_client
        return sse_client(server.url, headers=server.headers)
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=server.command, args=server.args, env=server.env)
    return stdio_client(params)


async def _list_one_server_tools(server: _McpServerConfig) -> list[dict]:
    from mcp import ClientSession

    transport = await _open_session(server)
    async with transport as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{"name": t.name, "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {},
                                                     "required": []},
                    "_mcp_server": server.name}
                   for t in result.tools]


async def _call_one_tool(server: _McpServerConfig, name: str, arguments: dict) -> str:
    from mcp import ClientSession

    transport = await _open_session(server)
    async with transport as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            texts = [block.text for block in result.content if getattr(block, "type", "") == "text"]
            return "\n".join(texts) if texts else json.dumps([
                block.model_dump() for block in result.content])


# Discovered once per process (mirrors plugins.py's PLUGINS/load_plugins()
# lifecycle: a new/edited MCP_SERVERS entry needs a restart to take effect,
# same as any other config-driven registry here) — a server that fails to
# start/list is logged and skipped, not a startup failure for the whole bot.
_TOOL_TO_SERVER: dict[str, _McpServerConfig] = {}
_TOOL_SPECS: list[dict] | None = None


def _discover() -> list[dict]:
    global _TOOL_SPECS
    if _TOOL_SPECS is not None:
        return _TOOL_SPECS
    specs: list[dict] = []
    if not _mcp_available():
        logger.info("`mcp` package not installed — no MCP tools loaded")
        _TOOL_SPECS = []
        return _TOOL_SPECS
    for server in _configured_servers():
        try:
            server_specs = _run_async(_list_one_server_tools(server))
        except Exception as e:
            logger.warning("MCP server %r unreachable (%s) — skipped", server.name, e)
            continue
        for spec in server_specs:
            _TOOL_TO_SERVER[spec["name"]] = server
            specs.append({k: v for k, v in spec.items() if k != "_mcp_server"})
    _TOOL_SPECS = specs
    return _TOOL_SPECS


def mcp_tool_specs() -> list[dict]:
    """Every discovered MCP tool, in the same {name, description,
    parameters} shape agent/tools.py's TOOL_SCHEMAS entries use."""
    return _discover()


def mcp_tool_names() -> frozenset[str]:
    _discover()
    return frozenset(_TOOL_TO_SERVER.keys())


def call_mcp_tool(name: str, arguments: dict) -> dict:
    """Step 11/12 for an MCP tool: dispatch to whichever configured server
    actually owns it. Raises (never returns a sentinel) on failure, same
    contract as agent/tools.execute() overall — the caller already wraps
    every tool execution in a try/except (see native_agent.handle_message)."""
    _discover()
    server = _TOOL_TO_SERVER.get(name)
    if server is None:
        raise RuntimeError(f"MCP tool {name!r} is not currently available")
    text = _run_async(_call_one_tool(server, name, arguments))
    return {"ok": True, "kind": "mcp_tool_result", "tool": name, "text": text}
