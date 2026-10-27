"""Coverage for CodingAgent's "client tools" mode (coding_agent.py's
module docstring, "CLIENT TOOLS MODE") — the mechanism that lets a
caller (a VS Code extension, today) execute the agent's tool calls
against the user's REAL machine instead of the server's own sandbox.

No existing test in this suite exercised CodingAgent's tool-calling
loop at all before this file (test_gmaofn_api.py only checks routing:
that a code_generation task lands on coding-agent). These tests stub
TierClient.chat the same way test_agent_flows.py's MCP-confirmation
tests do, since the offline RuleBased floor these tests otherwise run
against never emits a real tool_call.
"""
from __future__ import annotations

from tests.conftest import auth


def test_client_tools_mode_advertises_list_files_and_grep(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_tools = {}

    def fake_chat(self, node, messages, tools=None):
        seen_tools["names"] = {t["name"] for t in (tools or [])}
        return LLMResult(text="no tool needed", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "just say hi",
        "session_id": "client-tools-schema-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    # list_files/grep/delete_file only exist in client_tools mode — this
    # server has no real workspace of its own to run them against
    # otherwise.
    assert {"run_code", "read_file", "write_file", "list_files", "grep", "delete_file"} <= seen_tools["names"]


def test_execute_threads_history_into_real_model_messages(client, exec_token, monkeypatch):
    """Real, observed gap this covers: ExecuteRequest never carried any
    prior-turn context at all before this — every call built exactly
    [system, this ONE user message], so a second message in the same
    chat ("now edit line 3 of it") had no idea what "it" meant."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_messages = {}

    def fake_chat(self, node, messages, tools=None):
        seen_messages["messages"] = messages
        return LLMResult(text="no tool needed", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "now edit line 3 of it",
        "session_id": "client-tools-history-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
        "history": [
            {"role": "user", "content": "read fibonacci.py"},
            {"role": "assistant", "content": "It defines a recursive fib(n)."},
            {"role": "system", "content": "should be dropped — never trusted from a caller"},
            {"role": "user", "content": 123},
        ],
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    messages = seen_messages["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "read fibonacci.py"
    assert messages[2]["content"] == "It defines a recursive fib(n)."
    assert messages[3]["content"] == "now edit line 3 of it"


def test_client_tools_mode_never_advertises_list_files_and_grep_by_default(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_tools = {}

    def fake_chat(self, node, messages, tools=None):
        seen_tools["names"] = {t["name"] for t in (tools or [])}
        return LLMResult(text="no tool needed", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "just say hi",
        "session_id": "default-mode-schema-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert seen_tools["names"] == {"run_code", "read_file", "write_file"}
    assert "list_files" not in seen_tools["names"] and "grep" not in seen_tools["names"]
    assert "delete_file" not in seen_tools["names"]


def test_client_tools_pending_call_then_resume_to_final_answer(client, exec_token, monkeypatch):
    """The real end-to-end shape: the first turn's tool call comes back
    as a `pending_tool_call` (never executed server-side), and POSTing
    a fabricated result to /tool_result resumes the SAME conversation
    to a final answer — exactly what a VS Code extension does after
    actually reading the file on the user's machine."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = {"n": 0}

    def fake_chat(self, node, messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResult(tool_call={"name": "read_file", "arguments": {"path": "app.py"}},
                             provider="fake-llm")
        # Second call (the resume) should see the tool's result folded
        # into the conversation as a system message.
        assert any("Flask" in m.get("content", "") for m in messages)
        return LLMResult(text="app.py defines a Flask app.", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "What does app.py do?",
        "session_id": "client-tools-resume-1",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    call = body["result"]["pending_tool_call"]
    assert call is not None, body
    assert call["tool"] == "read_file"
    assert call["args"] == {"path": "app.py"}
    assert body["result"]["ok"] is True
    assert body["result"]["pending_confirmation"] is False  # a distinct mechanism, unset here

    r2 = client.post("/api/v1/orchestration/tool_result", json={
        "session_id": "client-tools-resume-1",
        "call_id": call["call_id"],
        "output": "from flask import Flask\napp = Flask(__name__)",
    }, headers=auth(exec_token))
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["result"]["pending_tool_call"] is None
    assert "Flask" in body2["result"]["text"]
    assert calls["n"] == 2


def test_client_tools_resume_rejects_a_mismatched_session_id(client, exec_token, monkeypatch):
    """call_id is an unguessable uuid4, but session ownership is the
    actual access check — a resume for the wrong session must fail
    cleanly, not silently continue someone else's in-flight turn."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(tool_call={"name": "list_files", "arguments": {}}, provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "what files are here?",
        "session_id": "client-tools-resume-2",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
    }, headers=auth(exec_token))
    call_id = r.json()["result"]["pending_tool_call"]["call_id"]

    r2 = client.post("/api/v1/orchestration/tool_result", json={
        "session_id": "someone-elses-session",
        "call_id": call_id,
        "output": "irrelevant",
    }, headers=auth(exec_token))
    assert r2.status_code == 200
    assert "doesn't belong to this session" in r2.json()["result"]["text"]


def test_client_tools_resume_of_unknown_call_id_is_a_clean_failure(client, exec_token):
    r = client.post("/api/v1/orchestration/tool_result", json={
        "session_id": "whatever",
        "call_id": "not-a-real-call-id",
        "output": "irrelevant",
    }, headers=auth(exec_token))
    assert r.status_code == 200
    assert "already expired" in r.json()["result"]["text"]


def test_write_file_in_client_tools_mode_pauses_instead_of_writing_to_the_server_sandbox(
    client, exec_token, monkeypatch, tmp_path
):
    """The regression this guards: in the DEFAULT mode, write_file
    stashes into the server's own _PENDING_WRITES and asks a plain-text
    yes/no. In client_tools mode it must NEVER do that — it has to
    come back as a pending_tool_call instead, so the actual write can
    happen on the user's real machine."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(tool_call={"name": "write_file",
                                    "arguments": {"path": "out.py", "content": "print(1)"}},
                         provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write out.py",
        "session_id": "client-tools-write-1",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["pending_confirmation"] is False
    assert body["result"]["pending_tool_call"]["tool"] == "write_file"
    assert body["result"]["pending_tool_call"]["args"] == {"path": "out.py", "content": "print(1)"}


# ── escalating away from a malformed local-model tool-call attempt ──────

def test_leaked_tool_schema_from_the_auto_picked_local_model_escalates_and_recovers(
    client, exec_token, monkeypatch
):
    """Real, observed failure: qwen2.5-coder:1.5b (the auto-picked
    default — see CodingAgent._preferred_provider) sometimes returns
    genuinely malformed JSON/leaked prompt formatting instead of a real
    tool call — not just "the tool call as text" (which
    _parse_text_tool_call already recovers), but broken enough that no
    JSON parse can save it. The first call simulates exactly that; the
    escalated retry (a fresh get_llm_client("delay_ok") call, forced by
    _looks_like_leaked_tool_schema) gets a real answer instead — the
    user must see the real answer, never the raw garbage."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = []

    def fake_chat(self, node, messages, tools=None):
        calls.append(node)
        if len(calls) == 1:
            return LLMResult(
                text='<tools>\n{"type": "function", "function": {read_file Read a file}}',
                provider="ollama-coder")
        return LLMResult(text="Here is a real answer.", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a fibonacci function",
        "session_id": "client-tools-leaked-schema-1",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["ok"] is True
    assert body["result"]["text"] == "Here is a real answer."
    assert len(calls) == 2  # one real retry, not a loop


def test_leaked_tool_schema_never_escalates_past_an_explicit_provider_pick(
    client, exec_token, monkeypatch
):
    """The escalation is only for the AUTO-picked local coder default —
    a caller who explicitly picked a provider (the UI model picker)
    gets that exact provider's own answer, garbled or not, never
    silently routed to a different one behind their back."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = []

    def fake_chat(self, node, messages, tools=None):
        calls.append(node)
        return LLMResult(text='<tools>\n{"type": "function", "function": {}}', provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a fibonacci function",
        "session_id": "client-tools-leaked-schema-explicit-1",
        "preferred_provider": "ollama-coder",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "<tools>" in body["result"]["text"]  # the explicit pick's own raw answer, unaltered
    assert len(calls) == 1  # never retried


def test_hallucinated_tool_call_from_the_auto_picked_local_model_escalates_and_recovers(
    client, exec_token, monkeypatch
):
    """Real, observed failure from a 60-prompt comprehensive test — the
    DOMINANT Coding Agent failure mode, distinct from the leaked-schema
    case above: qwen2.5-coder:1.5b answers a "write me a function"
    prompt with a well-formed {"name": "<the-function-it-was-asked-to-
    write>", "arguments": {...}} object instead of real code (e.g.
    asked for a palindrome checker, it replied
    {"name": "is_palindrome", "arguments": {"string": {"type":
    "string"}}} verbatim). This parses as valid JSON shaped exactly
    like a tool call, so _parse_text_tool_call() correctly declines it
    (the name isn't a real tool) but _looks_like_leaked_tool_schema()
    ALSO misses it (no "<tools>"/{"type": "function"} marker) — without
    _looks_like_hallucinated_tool_call(), this raw JSON blob was
    silently returned to the user as the final answer."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = []

    def fake_chat(self, node, messages, tools=None):
        calls.append(node)
        if len(calls) == 1:
            return LLMResult(
                text='{"name": "is_palindrome", "arguments": {"string": {"type": "string"}}}',
                provider="ollama-coder")
        return LLMResult(text="def is_palindrome(s):\n    return s == s[::-1]", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a python function that checks if a string is a palindrome",
        "session_id": "client-tools-hallucinated-call-1",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["ok"] is True
    assert "is_palindrome" in body["result"]["text"] and "def " in body["result"]["text"]
    assert '"arguments"' not in body["result"]["text"]  # never the raw hallucinated JSON
    assert len(calls) == 2  # one real retry, not a loop


def test_hallucinated_tool_call_never_escalates_past_an_explicit_provider_pick(
    client, exec_token, monkeypatch
):
    """Same non-second-guessing rule as the leaked-schema case: a
    caller who explicitly picked a provider gets that exact provider's
    own answer, hallucinated or not, never silently routed elsewhere."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = []

    def fake_chat(self, node, messages, tools=None):
        calls.append(node)
        return LLMResult(text='{"name": "is_palindrome", "arguments": {"s": "x"}}', provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a palindrome checker",
        "session_id": "client-tools-hallucinated-call-explicit-1",
        "preferred_provider": "ollama-coder",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert '"arguments"' in body["result"]["text"]  # the explicit pick's own raw answer, unaltered
    assert len(calls) == 1  # never retried


def test_client_tools_mode_allows_more_than_the_default_mode_step_budget(client, exec_token, monkeypatch):
    """Real behavior change this covers: client_tools mode's own step
    budget (_MAX_TOOL_STEPS_CLIENT) is much larger than the default
    server-sandbox mode's _MAX_TOOL_STEPS=4 — a genuine multi-file,
    "work continuously until done" task needs more than 4 round trips.
    Chains 6 real read_file rounds (more than the default cap would
    allow) and confirms the agent is still going, not cut off early."""
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = {"n": 0}
    ROUNDS = 6

    def fake_chat(self, node, messages, tools=None):
        calls["n"] += 1
        if calls["n"] <= ROUNDS:
            return LLMResult(tool_call={"name": "read_file", "arguments": {"path": f"file{calls['n']}.py"}},
                             provider="fake-llm")
        return LLMResult(text="Done reading every file.", provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "Read every file in the project",
        "session_id": "client-tools-long-chain-1",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    body = r.json()
    call = body["result"]["pending_tool_call"]
    session_id = "client-tools-long-chain-1"

    # Resume through every remaining round — if the default mode's
    # cap of 4 applied here, this would exhaust the budget and report
    # "didn't reach a final answer" well before round 6.
    for _ in range(ROUNDS - 1):
        assert call is not None, body
        r = client.post("/api/v1/orchestration/tool_result", json={
            "session_id": session_id,
            "call_id": call["call_id"],
            "output": "file contents",
        }, headers=auth(exec_token))
        assert r.status_code == 200, r.text
        body = r.json()
        call = body["result"]["pending_tool_call"]

    # Final round: no more tool calls queued, a real final answer.
    assert call is not None, body
    r = client.post("/api/v1/orchestration/tool_result", json={
        "session_id": session_id,
        "call_id": call["call_id"],
        "output": "file contents",
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    final = r.json()
    assert final["result"]["pending_tool_call"] is None
    assert final["result"]["text"] == "Done reading every file."
    assert calls["n"] == ROUNDS + 1


# ── intent-answer cache never serves a hit to an explicit provider pick ──

def test_intent_cache_never_serves_a_cached_answer_to_an_explicit_provider_pick(
    client, exec_token, monkeypatch
):
    """Real bug found live: the cache lookup gate never checked for an
    explicit preferred_provider — a caller who deliberately asked for a
    SPECIFIC model could silently get a cached answer from a different,
    earlier run's provider, breaking the "an explicit pick is never
    second-guessed" guarantee _resolve_hybrid() already upholds
    elsewhere in this file. Primes the cache via an auto-picked (no
    preferred_provider) call, then confirms a second call with the SAME
    prompt text but an explicit preferred_provider gets a real answer
    from that provider, never the cached one."""
    from app.framework.agents import coding_agent
    from app.framework.intent_cache import CachedAnswer
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    coding_agent.coding_cache.remember("write a fibonacci function", "CACHED ANSWER", "groq")
    assert coding_agent.coding_cache.lookup("write a fibonacci function") is not None

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="a real live answer from the explicit pick", provider="ollama-coder")

    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a fibonacci function",
        "session_id": "cache-explicit-pick-check",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "preferred_provider": "ollama-coder",
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert r.json()["result"]["text"] == "a real live answer from the explicit pick"
