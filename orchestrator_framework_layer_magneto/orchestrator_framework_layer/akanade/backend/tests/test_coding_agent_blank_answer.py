"""Real bug found live: a hosted "reasoning" model (confirmed with
glm-5.3-flash) can return a genuinely EMPTY visible answer with no
LLMResult.error at all — the HTTP call itself succeeded, it just produced
no usable text. Before this fix, validate() treated that as a clean pass
(zero fenced blocks -> "no syntax error") and respond() treated it as a
normal ok=True, confidence=0.85 answer — worse, it got written into
coding_cache, silently serving the SAME blank reply to every future
caller of that exact prompt. See coding_agent.py's validate()/respond()
for the actual fix: one retry (same shape as the syntax-error retry),
and if it's STILL blank, a real ok=False/confidence=0.0 outcome with an
honest message instead of a blank bubble.
"""
from __future__ import annotations

from tests.conftest import auth


def _patch_tier_client(monkeypatch, chat_fn):
    from app.llm.router import TierClient

    def fake_init(self, tier, preferred=None, preferred_timeout=None, exclude_rule_based=False,
                 explicit_cascade=None):
        self.tier = tier
        self.preferred = preferred
        self.log = []
        self.tool_selections = []

    monkeypatch.setattr(TierClient, "__init__", fake_init)
    monkeypatch.setattr(TierClient, "chat", chat_fn)


def test_blank_first_answer_retries_and_returns_the_real_one(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult

    calls = {"n": 0}

    def fake_chat(self, node, messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResult(text="", provider="glm-5.3-flash")  # the reasoning-budget-exhausted case
        return LLMResult(text="def add(a, b):\n    return a + b", provider="glm-5.3-flash")

    _patch_tier_client(monkeypatch, fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a function that adds two numbers",
        "session_id": "blank-answer-retry",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert calls["n"] == 2, "must retry exactly once after a blank first answer"
    assert result["ok"] is True
    assert "def add" in result["text"]
    assert result["provider"] == "glm-5.3-flash"


def test_still_blank_after_retry_is_a_real_failure_not_a_silent_blank_reply(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="", provider="glm-5.3-flash")  # every attempt comes back blank

    _patch_tier_client(monkeypatch, fake_chat)

    r = client.post("/api/v1/orchestration/execute", json={
        "task": "write a function that subtracts two numbers",
        "session_id": "blank-answer-persists",
        "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]},
    }, headers=auth(exec_token))
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["ok"] is False, "an empty answer must never be reported as ok=True"
    assert result["text"], "must show a real message, never a blank bubble"
    assert "empty" in result["text"].lower()
