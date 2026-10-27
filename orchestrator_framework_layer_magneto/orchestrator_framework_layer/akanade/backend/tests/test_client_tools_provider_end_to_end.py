"""End-to-end coverage for the exact HTTP shape VS Code's api.ts actually
drives: POST /execute (client_tools=true) -> pending_tool_call -> one or
more POST /tool_result round trips -> a final plain-text answer. Real gap
this guards against: /tool_result's hand-built response dict used to
silently drop provider/tier entirely (fixed separately), and it's the
path every real multi-step client-tools turn finishes through — a test
that only exercises /execute directly would miss a regression here."""
from __future__ import annotations

from tests.conftest import auth


def test_provider_and_tier_survive_multiple_tool_result_round_trips(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    calls = {"n": 0}

    def fake_chat(self, node, messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResult(tool_call={"name": "list_files", "arguments": {}}, provider="glm-5.3-flash")
        if calls["n"] == 2:
            return LLMResult(tool_call={"name": "read_file", "arguments": {"path": "README.md"}},
                             provider="glm-5.3-flash")
        return LLMResult(text="Here's what I found and how to proceed.", provider="glm-5.3-flash")

    monkeypatch.setattr(TierClient, "chat", fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "deploy this project",
        "session_id": "client-tools-provider-e2e",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
        "client_tools": True,
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["pending_tool_call"] is not None
    call_id = result["pending_tool_call"]["call_id"]

    # Round 1: report the list_files outcome, expect the SECOND tool call back.
    r2 = client.post("/api/v1/orchestration/tool_result", json={
        "session_id": "client-tools-provider-e2e",
        "call_id": call_id,
        "output": "README.md\ndocker-compose.yml",
    }, headers=auth(exec_token))
    assert r2.status_code == 200, r2.text
    result2 = r2.json()["result"]
    assert result2["pending_tool_call"] is not None, "expected a second tool call, not a final answer yet"
    call_id_2 = result2["pending_tool_call"]["call_id"]

    # Round 2: report the read_file outcome, expect the FINAL plain-text answer.
    r3 = client.post("/api/v1/orchestration/tool_result", json={
        "session_id": "client-tools-provider-e2e",
        "call_id": call_id_2,
        "output": "# README contents...",
    }, headers=auth(exec_token))
    assert r3.status_code == 200, r3.text
    result3 = r3.json()["result"]
    assert result3["pending_tool_call"] is None, "this must be the final answer, not another pause"
    assert result3["ok"] is True
    assert "Here's what I found" in result3["text"]
    assert result3["provider"] == "glm-5.3-flash", (
        f"the final /tool_result response must carry the real answering provider, got {result3!r}")
    assert result3["tier"] == "default"
