"""Track C (claude), the MCP client integration, and the ACP peer-agent
endpoints — three additions layered on top of the existing framework/tool
stack, each covered independently of the §6.1 track-equivalence suite
(test_agent_flows.py) since none of them change that algorithm's behavior
for native/langgraph/adk.
"""
from app.agent import native_agent
from tests.conftest import auth


def say(client, tok, msg, track, session="claude-t"):
    r = client.post(f"/api/chat/message?framework={track}",
                    json={"message": msg, "session_id": f"{session}-{track}"},
                    headers=auth(tok))
    assert r.status_code == 200, r.text
    return r.json()


# ── Track C: claude ───────────────────────────────────────────────────────

def test_claude_framework_selectable_and_degrades_to_rule_based(client, exec_token):
    """CLAUDE_API_KEY is blanked for the whole test session (conftest.py) —
    same "tests must be hermetic" rule every other provider follows — so
    this track's own preferred rung is unavailable and TierClient falls
    back through the chain to the deterministic rule-based floor, exactly
    like every other track already does with no keys configured. Proves
    ?framework=claude is a real, selectable option that never hard-fails
    just because Claude itself isn't configured."""
    d = say(client, exec_token,
            "create a task called 'Claude track task' in Compass OS", "claude")
    assert "Created task" in d["reply"]
    assert d["framework"] == "claude"


def test_claude_provider_registered_but_not_in_any_chain(client):
    from app.llm.router import CHAINS, _providers, available_providers

    assert "claude" in _providers()
    # Deliberately absent from the ordinary fallback chains (config.py's
    # CLAUDE_API_KEY comment) — it only answers when a track asks for it
    # BY NAME via TierClient(preferred="claude").
    for rungs in CHAINS.values():
        assert "claude" not in rungs
    ids = [p["id"] for p in available_providers()]
    assert "claude" in ids  # still visible/selectable for the UI picker


# ── MCP client integration ──────────────────────────────────────────────

def test_mcp_tools_merged_into_shared_tool_schemas():
    from app.agent import tools

    assert "get_current_datetime" in tools.TOOL_SCHEMAS
    assert "days_between" in tools.TOOL_SCHEMAS
    # The second MCP_SERVERS default (config.py) — Anthropic's own
    # reference fetch server, a real external ability (read a web page)
    # alongside the toy offline example above. No network call made
    # here — just confirming discovery/schema-merge, same as the other
    # two; requires_confirmation() covers the "don't actually call it
    # without asking" half (see test_agent_flows.py's MCP confirm test).
    assert "fetch" in tools.TOOL_SCHEMAS
    names = [s["name"] for s in tools.openai_tool_specs()]
    assert "get_current_datetime" in names
    assert "fetch" in names


def test_mcp_tool_executes_via_the_bundled_example_server():
    """A real subprocess round-trip to mcp_servers/utility_server.py over
    stdio — not a mock — through the exact same resolve_arguments/
    tools.execute/format_result path every hand-written and JSON-plugin
    tool goes through (see native_agent.handle_message's steps 7/11/15),
    proving an MCP-sourced tool is genuinely indistinguishable from a
    built-in one once perceive_intent has named it.

    Unlike test_plugins.py's rename_project (which has its own JSON
    trigger_patterns and so IS reachable through the rule-based/no-LLM
    floor via free text), an MCP tool's whole point is to be discovered
    and called by an LLM doing real function-calling against its schema —
    it has no trigger_patterns of its own, so this is tested at the
    tool-call layer directly rather than via free text through the chat
    endpoint (which, with every LLM rung blanked for tests, bottoms out at
    RuleBased and wouldn't recognize this tool's phrasing — a real and
    expected limitation of the offline floor, not a bug in this
    integration)."""
    from app.agent import tools
    from app.agent.state import format_result

    user = {"sub": "test-user"}
    resolved, clarify = native_agent.resolve_arguments(user, "days_between", {
        "start_date": "2026-08-01", "end_date": "2026-09-15"})
    assert clarify is None
    result = tools.execute(user, "days_between", resolved)
    assert result["ok"] is True
    assert "45 day" in format_result(result)


def test_mcp_tool_dispatch_is_framework_agnostic():
    """resolve_arguments/tools.execute (shared by every track — see their
    own module docstrings) don't care which framework called them, so an
    MCP tool works identically regardless of which of native/langgraph/
    adk/claude is answering; nothing here is claude-specific."""
    from app.agent import tools

    for _ in range(2):  # calling twice exercises the cached-after-first-call path too
        result = tools.execute({"sub": "u"}, "get_current_datetime", {"timezone": "UTC"})
        assert result["ok"] is True
        assert "UTC" in result["text"]


# ── ACP (Agent Communication Protocol) ──────────────────────────────────

def test_acp_agent_discovery(client):
    r = client.get("/acp/agents")
    assert r.status_code == 200
    agents = r.json()["agents"]
    assert len(agents) == 1
    assert agents[0]["name"] == "algo-kaizen-pm"
    assert "claude" in agents[0]["metadata"]["available_frameworks"]


def test_acp_run_requires_auth(client):
    r = client.post("/acp/runs", json={
        "agent_name": "algo-kaizen-pm", "session_id": "acp-noauth",
        "input": [{"role": "user",
                  "parts": [{"content_type": "text/plain", "content": "hi"}]}]})
    assert r.status_code == 401


def test_acp_run_executes_the_same_agent_as_the_chat_widget(client, exec_token):
    r = client.post("/acp/runs", json={
        "agent_name": "algo-kaizen-pm", "session_id": "acp-run-1",
        "input": [{"role": "user",
                  "parts": [{"content_type": "text/plain",
                             "content": "create a task called 'ACP task' in Compass OS"}]}]},
        headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["agent_name"] == "algo-kaizen-pm"
    reply_text = body["output"][0]["parts"][0]["content"]
    assert "Created task" in reply_text


def test_acp_run_unknown_agent_404s(client, exec_token):
    r = client.post("/acp/runs", json={
        "agent_name": "not-a-real-agent", "session_id": "acp-404",
        "input": [{"role": "user",
                  "parts": [{"content_type": "text/plain", "content": "hi"}]}]},
        headers=auth(exec_token))
    assert r.status_code == 404
