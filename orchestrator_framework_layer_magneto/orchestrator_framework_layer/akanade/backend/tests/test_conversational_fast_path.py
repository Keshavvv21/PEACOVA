"""app.agent.native_agent.is_pure_conversational() and its wiring into
handle_message() — real, measured problem this fixes: perceive_intent
sent the FULL ~4600-token/41-tool PM+trading+MCP schema on every single
turn, including "Are you up?", plus a domain-router bandit consultation
and a RAG relevance check that are pure overhead for a message that was
never going to call a tool. See native_agent.py's own comment for the
full reasoning.
"""
from __future__ import annotations

from tests.conftest import auth


def test_greetings_and_chit_chat_are_recognized():
    from app.agent.native_agent import is_pure_conversational

    for msg in ["hi", "Hi!", "hello", "hey there", "Hello Algo Akanade",
               "are you up?", "Are you there?", "you up?",
               "how's it going", "what's up", "good morning",
               "thanks!", "thank you so much", "ok", "cool", "got it",
               "bye", "see you later", "what have you done", "what's new"]:
        assert is_pure_conversational(msg), f"expected {msg!r} to be recognized as conversational"


def test_real_requests_are_never_misclassified_as_conversational():
    from app.agent.native_agent import is_pure_conversational

    for msg in ["hi, can you list my tasks", "hello, create a task called Fix login bug",
               "what is the status of sprint 4", "backtest a 20/50 crossover on AAPL",
               "thanks, now show me overdue tasks", "what have you done to fix the deploy pipeline",
               "read fibonacci.py", "hey, what's the weather like for our deployment window"]:
        assert not is_pure_conversational(msg), f"expected {msg!r} to NOT be treated as conversational"


def test_a_greeting_skips_the_tool_schema_entirely(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_tools = {}

    def fake_chat(self, node, messages, tools=None):
        seen_tools["value"] = tools
        return LLMResult(text="Hey! I'm up and ready to help.", provider="groq")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/chat/message",
                    json={"message": "Are you up?", "session_id": "fast-path-greeting"},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert seen_tools["value"] is None
    assert r.json()["reply"] == "Hey! I'm up and ready to help."


def test_a_real_request_still_gets_the_full_tool_schema(client, exec_token, monkeypatch):
    from app.llm.providers import LLMResult
    from app.llm.router import TierClient

    seen_tools = {}

    def fake_chat(self, node, messages, tools=None):
        seen_tools["value"] = tools
        return LLMResult(text="Here are your tasks.", provider="groq")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    r = client.post("/api/chat/message",
                    json={"message": "list all my tasks", "session_id": "fast-path-real"},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert seen_tools["value"] is not None and len(seen_tools["value"]) > 0
