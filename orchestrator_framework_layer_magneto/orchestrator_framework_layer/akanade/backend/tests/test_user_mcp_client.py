"""app/agent/user_mcp_client.py — per-user OAuth-connected MCP tools, and
their wiring into tools.py (execute()/user_openai_tool_specs/
user_tool_schemas) and native_agent.py's confirmation gate
(requires_confirmation/confirm_prompt_for).

Mocks the transport layer (mcp_client._mcp_available/_run_async) since
there's no real GitHub/Linear/Google MCP server reachable in a test run
— same "test the wiring, not a live network call" boundary
test_mcp_client_config.py already draws for the static-config path.
"""
from __future__ import annotations

import pytest

from app import oauth_store
from app.agent import mcp_client, native_agent, tools, user_mcp_client


def _fake_connection(user_id: str, provider: str, token: str = "tok"):
    oauth_store.save_connection(user_id, provider, access_token=token, refresh_token=None,
                                expires_in=None, scopes=["repo"])


def _fake_list(monkeypatch, specs: list[dict]):
    monkeypatch.setattr(mcp_client, "_mcp_available", lambda: True)

    def fake_run_async(coro):
        coro.close()  # never actually awaited — avoid a "never awaited" warning
        return specs
    monkeypatch.setattr(mcp_client, "_run_async", fake_run_async)


# ── discovery ─────────────────────────────────────────────────────────

def test_no_tools_for_a_user_with_no_connections():
    user_mcp_client.invalidate("nobody")
    assert user_mcp_client.user_mcp_tool_specs("nobody") == []
    assert user_mcp_client.user_mcp_tool_names("nobody") == frozenset()


def test_no_tools_when_the_only_connected_provider_has_no_mcp_wiring():
    # google's provider entry deliberately has mcp_url=None (providers.py)
    _fake_connection("g-user", "google")
    user_mcp_client.invalidate("g-user")
    assert user_mcp_client.user_mcp_tool_specs("g-user") == []
    oauth_store.delete_connection("g-user", "google")


def test_discovers_tools_for_a_connected_provider(monkeypatch):
    _fake_connection("gh-user", "github")
    user_mcp_client.invalidate("gh-user")
    _fake_list(monkeypatch, [{"name": "open_issue", "description": "Open a GitHub issue",
                             "parameters": {"type": "object",
                                           "properties": {"title": {"type": "string"}},
                                           "required": ["title"]},
                             "_mcp_server": "user::github"}])

    specs = user_mcp_client.user_mcp_tool_specs("gh-user")
    assert len(specs) == 1
    assert specs[0]["name"] == "open_issue"
    assert "_mcp_server" not in specs[0]  # internal-only field, stripped before it leaves this module
    assert user_mcp_client.user_mcp_tool_names("gh-user") == frozenset({"open_issue"})
    assert user_mcp_client.provider_for_tool("gh-user", "open_issue") == "github"

    oauth_store.delete_connection("gh-user", "github")
    user_mcp_client.invalidate("gh-user")


def test_discovery_is_cached_within_the_ttl(monkeypatch):
    _fake_connection("cache-user", "github")
    user_mcp_client.invalidate("cache-user")
    calls = {"n": 0}

    def fake_run_async(coro):
        coro.close()
        calls["n"] += 1
        return [{"name": "open_issue", "description": "x",
                "parameters": {"properties": {}, "required": []}}]
    monkeypatch.setattr(mcp_client, "_mcp_available", lambda: True)
    monkeypatch.setattr(mcp_client, "_run_async", fake_run_async)

    user_mcp_client.user_mcp_tool_specs("cache-user")
    user_mcp_client.user_mcp_tool_specs("cache-user")
    assert calls["n"] == 1  # second call served from cache, no new discovery round trip

    oauth_store.delete_connection("cache-user", "github")
    user_mcp_client.invalidate("cache-user")


def test_invalidate_forces_rediscovery(monkeypatch):
    _fake_connection("inv-user", "github")
    user_mcp_client.invalidate("inv-user")
    calls = {"n": 0}

    def fake_run_async(coro):
        coro.close()
        calls["n"] += 1
        return []
    monkeypatch.setattr(mcp_client, "_mcp_available", lambda: True)
    monkeypatch.setattr(mcp_client, "_run_async", fake_run_async)

    user_mcp_client.user_mcp_tool_specs("inv-user")
    user_mcp_client.invalidate("inv-user")
    user_mcp_client.user_mcp_tool_specs("inv-user")
    assert calls["n"] == 2

    oauth_store.delete_connection("inv-user", "github")


def test_call_user_mcp_tool_dispatches_to_the_owning_provider(monkeypatch):
    _fake_connection("call-user", "github")
    user_mcp_client.invalidate("call-user")
    _fake_list(monkeypatch, [{"name": "open_issue", "description": "x",
                             "parameters": {"properties": {}, "required": []}}])
    user_mcp_client.user_mcp_tool_specs("call-user")  # populate the discovery cache

    def fake_run_async_call(coro):
        coro.close()
        return "issue #42 opened"
    monkeypatch.setattr(mcp_client, "_run_async", fake_run_async_call)
    result = user_mcp_client.call_user_mcp_tool("call-user", "open_issue", {"title": "bug"})
    assert result == {"ok": True, "kind": "mcp_tool_result", "tool": "open_issue",
                      "text": "issue #42 opened"}

    oauth_store.delete_connection("call-user", "github")
    user_mcp_client.invalidate("call-user")


def test_call_user_mcp_tool_raises_for_an_unknown_tool():
    with pytest.raises(RuntimeError):
        user_mcp_client.call_user_mcp_tool("nobody", "does_not_exist", {})


# ── tools.py wiring ───────────────────────────────────────────────────

def test_user_openai_tool_specs_never_leaks_across_users(monkeypatch):
    _fake_connection("wire-user", "github")
    user_mcp_client.invalidate("wire-user")
    _fake_list(monkeypatch, [{"name": "open_issue", "description": "Open an issue",
                             "parameters": {"type": "object",
                                           "properties": {"title": {"type": "string"}},
                                           "required": ["title"]}}])

    specs = tools.user_openai_tool_specs({"sub": "wire-user"})
    assert any(s["name"] == "open_issue" for s in specs)
    schemas = tools.user_tool_schemas({"sub": "wire-user"})
    assert schemas["open_issue"]["required"] == ["title"]

    # A DIFFERENT user (no connection of their own) sees nothing.
    assert tools.user_openai_tool_specs({"sub": "someone-else"}) == []

    oauth_store.delete_connection("wire-user", "github")
    user_mcp_client.invalidate("wire-user")


def test_execute_dispatches_a_connected_users_tool_through_user_mcp_client(monkeypatch):
    _fake_connection("exec-user", "github")
    user_mcp_client.invalidate("exec-user")
    _fake_list(monkeypatch, [{"name": "open_issue", "description": "x",
                             "parameters": {"properties": {}, "required": []}}])
    tools.user_openai_tool_specs({"sub": "exec-user"})  # populate cache

    def fake_run_async_call(coro):
        coro.close()
        return "done"
    monkeypatch.setattr(mcp_client, "_run_async", fake_run_async_call)
    result = tools.execute({"sub": "exec-user"}, "open_issue", {"title": "x"})
    assert result["ok"] is True
    assert result["text"] == "done"

    oauth_store.delete_connection("exec-user", "github")
    user_mcp_client.invalidate("exec-user")


# ── confirmation gate (native_agent.py) ──────────────────────────────

def test_requires_confirmation_true_for_a_connected_users_oauth_tool(monkeypatch):
    _fake_connection("confirm-user", "github")
    user_mcp_client.invalidate("confirm-user")
    _fake_list(monkeypatch, [{"name": "open_issue", "description": "x",
                             "parameters": {"properties": {}, "required": []}}])

    user = {"sub": "confirm-user"}
    assert native_agent.requires_confirmation(user, "open_issue") is True
    prompt = native_agent.confirm_prompt_for(user, "open_issue", {"title": "bug"})
    assert "github" in prompt
    assert "YOUR connected" in prompt

    oauth_store.delete_connection("confirm-user", "github")
    user_mcp_client.invalidate("confirm-user")


def test_requires_confirmation_false_for_an_unconnected_users_unrelated_tool():
    user = {"sub": "no-connections-user"}
    assert native_agent.requires_confirmation(user, "list_projects") is False
