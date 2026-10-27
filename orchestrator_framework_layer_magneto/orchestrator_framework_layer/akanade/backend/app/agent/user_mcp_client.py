"""Per-user sibling of mcp_client.py — tools sourced from a user's OWN
OAuth connections (GitHub/Linear/Google — app/oauth_store.py) rather than
the single static, shared MCP_SERVERS config. Two servers, two different
trust models: mcp_client.py's tools are the SAME for every user, forever
(until a restart); this module's tools differ per user and can appear
the instant they connect (invalidate() is called by the oauth router on
connect/disconnect) or disappear the instant they disconnect or a token
can't be refreshed.

Deliberately NOT merged into mcp_client.py's module-global
_TOOL_SPECS/_TOOL_TO_SERVER — those are safe to cache forever precisely
BECAUSE they're the same for everyone; a per-user cache keyed only by
tool name would leak user A's GitHub issue tool into user B's LLM
tool-call surface. See tools.py's execute() and each of the four agent
tracks (native_agent.py/langgraph_agent.py/claude_agent.py/adk_agent.py)
for where `user_mcp_tool_specs(user["sub"])` gets appended to the
per-request tool list, and where execute() dispatches
call_user_mcp_tool() alongside the existing mcp_client.call_mcp_tool().

Reuses mcp_client.py's private transport helpers (_McpServerConfig,
_run_async, _list_one_server_tools, _call_one_tool, _mcp_available) —
same async-bridge-over-stdio/SSE mechanics, just built with a per-user
OAuth bearer token instead of a static config-file header.
"""
from __future__ import annotations

import logging
import time

from .. import oauth_store
from ..oauth.providers import get_provider
from . import mcp_client

logger = logging.getLogger(__name__)

# Tool DISCOVERY (the list_tools() round trip) is cached for a few
# minutes per (user_id, provider) — cheap to keep fresh since a stale
# entry only means a newly-connected account's tools take up to this
# long to appear BEFORE the oauth router's explicit invalidate() call,
# which is the real mechanism for "instant" (connect/disconnect always
# invalidates immediately; this TTL is just a safety net, e.g. a token
# that silently stopped working). Never used for the access TOKEN
# itself — every actual tool call re-fetches a fresh valid token via
# oauth_store.get_valid_access_token(), which handles its own refresh.
_CACHE_TTL_SECONDS = 300
_cache: dict[tuple[str, str], tuple[list[dict], float]] = {}


def invalidate(user_id: str) -> None:
    for key in [k for k in _cache if k[0] == user_id]:
        del _cache[key]


def _connected_providers_with_mcp(user_id: str) -> list[str]:
    """Every provider this user has connected AND that has real MCP
    wiring (get_provider(...).mcp_url set) — see providers.py's Google
    entry for the one currently-unwired case, which just yields no tools
    here rather than erroring."""
    out = []
    for row in oauth_store.list_connections(user_id):
        provider = row["SK"].removeprefix("PROVIDER#")
        if get_provider(provider).mcp_url:
            out.append(provider)
    return out


def _specs_for_provider(user_id: str, provider: str) -> list[dict]:
    key = (user_id, provider)
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[1] < _CACHE_TTL_SECONDS:
        return cached[0]
    if not mcp_client._mcp_available():
        return []
    token = oauth_store.get_valid_access_token(user_id, provider)
    if not token:
        return []
    p = get_provider(provider)
    server = mcp_client._McpServerConfig(name=f"user::{provider}", url=p.mcp_url,
                                         headers={"Authorization": f"Bearer {token}"})
    try:
        specs = mcp_client._run_async(mcp_client._list_one_server_tools(server))
    except Exception as e:  # noqa: BLE001 — same "skip, don't crash the turn" rule as mcp_client._discover
        logger.warning("OAuth MCP server %r unreachable for user %s (%s) — skipped", provider, user_id, e)
        specs = []
    _cache[key] = (specs, time.monotonic())
    return specs


def user_mcp_tool_specs(user_id: str) -> list[dict]:
    """Every tool available to THIS user right now, in the same
    {name, description, parameters} shape mcp_client.mcp_tool_specs()
    returns — callers append this to that list, never replace it."""
    return [{k: v for k, v in s.items() if k != "_mcp_server"}
           for provider in _connected_providers_with_mcp(user_id)
           for s in _specs_for_provider(user_id, provider)]


def user_mcp_tool_names(user_id: str) -> frozenset[str]:
    return frozenset(s["name"] for s in user_mcp_tool_specs(user_id))


def provider_for_tool(user_id: str, name: str) -> str | None:
    """Which connected provider (e.g. "github") owns tool `name`, or
    None — used both for dispatch below and by native_agent.py's
    confirm_prompt_for() to name the provider in the human-facing
    confirmation prompt."""
    for provider in _connected_providers_with_mcp(user_id):
        if any(s["name"] == name for s in _specs_for_provider(user_id, provider)):
            return provider
    return None


def call_user_mcp_tool(user_id: str, name: str, arguments: dict) -> dict:
    """Step 11/12 for a per-user OAuth MCP tool — mirrors
    mcp_client.call_mcp_tool()'s contract (raises, never a sentinel).
    Always fetches a FRESH valid token (never reuses a cached one),
    since a stale-but-not-yet-expired-by-clock token that the provider
    already invalidated would otherwise fail silently mid-call."""
    provider = provider_for_tool(user_id, name)
    if provider is None:
        raise RuntimeError(f"OAuth MCP tool {name!r} is not currently available for this user")
    token = oauth_store.get_valid_access_token(user_id, provider)
    if not token:
        raise RuntimeError(f"Your {provider} connection needs to be reconnected")
    p = get_provider(provider)
    server = mcp_client._McpServerConfig(name=f"user::{provider}", url=p.mcp_url,
                                         headers={"Authorization": f"Bearer {token}"})
    text = mcp_client._run_async(mcp_client._call_one_tool(server, name, arguments))
    return {"ok": True, "kind": "mcp_tool_result", "tool": name, "text": text}
