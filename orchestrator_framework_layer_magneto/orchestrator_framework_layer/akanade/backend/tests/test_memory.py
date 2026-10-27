"""Tests for the memory-optimization layer (app.agent.memory): the rolling
summary that sessions.save() folds trimmed-window messages into instead of
just discarding them, and the fail-soft behavior when no real LLM answers."""
from app.agent import memory, sessions
from app.llm.providers import LLMResult


class _FakeClient:
    """Stands in for router.TierClient — same chat(node, messages, tools)
    shape, fixed canned reply, no network."""

    def __init__(self, result: LLMResult):
        self._result = result
        self.log = []

    def chat(self, node, messages, tools=None):
        return self._result


def test_fold_into_summary_is_a_noop_with_nothing_dropped(monkeypatch):
    """No dropped messages means no LLM call at all — verified by making
    get_llm_client blow up if it's ever reached."""
    def boom(*a, **k):
        raise AssertionError("get_llm_client() must not be called with nothing to fold")
    monkeypatch.setattr(memory, "get_llm_client", boom)
    assert memory.fold_into_summary("previous summary", []) == "previous summary"


def test_fold_into_summary_uses_the_router_and_returns_its_text(monkeypatch):
    """A real (non-rule-based) provider's reply becomes the new summary —
    this is the "connects to an LLM through the same router as every
    other call" path."""
    fake = _FakeClient(LLMResult(text="User asked about the Website Revamp project.",
                                 provider="groq-llama-3.3-70b"))
    monkeypatch.setattr(memory, "get_llm_client", lambda tier: fake)
    dropped = [{"role": "user", "content": "tell me about Website Revamp"},
              {"role": "assistant", "content": "It's on track, 4 open tasks."}]
    result = memory.fold_into_summary("", dropped)
    assert result == "User asked about the Website Revamp project."


def test_fold_into_summary_rejects_a_rule_based_answer(monkeypatch):
    """RuleBased is a deterministic intent parser, not a summarizer — its
    output would be nonsense as a running summary, so a reply attributed
    to it is discarded in favor of keeping the previous summary."""
    fake = _FakeClient(LLMResult(text="Sorry, I didn't understand that.",
                                 provider="rule-based"))
    monkeypatch.setattr(memory, "get_llm_client", lambda tier: fake)
    result = memory.fold_into_summary("existing summary", [{"role": "user", "content": "x"}])
    assert result == "existing summary"


def test_fold_into_summary_falls_back_on_an_empty_reply(monkeypatch):
    """Every provider genuinely failed (router.py's own worst case: an
    empty LLMResult) — keep the previous summary rather than overwrite it
    with nothing."""
    fake = _FakeClient(LLMResult(text="", provider="none", error="all providers failed"))
    monkeypatch.setattr(memory, "get_llm_client", lambda tier: fake)
    result = memory.fold_into_summary("existing summary", [{"role": "user", "content": "x"}])
    assert result == "existing summary"


def test_sessions_save_folds_trimmed_messages_into_the_running_summary(monkeypatch):
    """End-to-end through sessions.save(): once the rolling window trims
    anything, that content shows up in state["summary"] instead of being
    gone, and every track reads it back via sessions.load()."""
    monkeypatch.setattr(sessions.settings(), "SESSION_WINDOW_TURNS", 2)
    fake = _FakeClient(LLMResult(text="Discussed the Q3 roadmap.", provider="groq-llama-3.3-70b"))
    monkeypatch.setattr(memory, "get_llm_client", lambda tier: fake)

    state = sessions.load("memtest-user", "memtest-session")
    assert state["summary"] == ""
    state["messages"] = [
        {"role": "user", "content": "what's the Q3 roadmap"},
        {"role": "assistant", "content": "Ship the mobile app and the new board."},
        {"role": "user", "content": "and after that"},
        {"role": "assistant", "content": "Performance review season."},
    ]
    sessions.save("memtest-user", "memtest-session", state)

    assert state["summary"] == "Discussed the Q3 roadmap."
    assert len(state["messages"]) == 2  # trimmed to the (monkeypatched) window of 2

    reloaded = sessions.load("memtest-user", "memtest-session")
    assert reloaded["summary"] == "Discussed the Q3 roadmap."
    sessions.clear("memtest-user", "memtest-session")
