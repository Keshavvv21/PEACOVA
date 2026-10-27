"""Coverage for llm/router.py's call_provider_directly() — bypasses
every tier/chain/bandit entirely, for callers (scripts/
log_complex_ground_truth.py) that need a real answer from a SPECIFIC
named provider, never a silent substitute from TierClient's own
fallback chain."""
from __future__ import annotations

from app.llm import router
from app.llm.providers import LLMResult


def test_unknown_provider_name_errors_cleanly():
    result = router.call_provider_directly("not-a-real-provider", [{"role": "user", "content": "hi"}])
    assert result.error
    assert "not-a-real-provider" in result.error


def test_unconfigured_provider_errors_without_calling_chat(monkeypatch):
    called = {"chat": False}

    class FakeProvider:
        def available(self):
            return False

        def chat(self, messages, tools=None):
            called["chat"] = True
            return LLMResult(text="should never get here", provider="fake")

    monkeypatch.setattr(router, "_providers", lambda: {"fake": FakeProvider()})
    result = router.call_provider_directly("fake", [{"role": "user", "content": "hi"}])
    assert result.error
    assert not called["chat"]


def test_a_real_available_provider_answers_directly(monkeypatch):
    class FakeProvider:
        def available(self):
            return True

        def chat(self, messages, tools=None):
            return LLMResult(text="real answer", provider="fake-strong-model")

    monkeypatch.setattr(router, "_providers", lambda: {"fake": FakeProvider()})
    result = router.call_provider_directly("fake", [{"role": "user", "content": "hi"}])
    assert not result.error
    assert result.text == "real answer"
    assert result.provider == "fake-strong-model"


def test_timeout_override_is_applied_then_restored(monkeypatch):
    seen_timeout_during_call = []

    class FakeProvider:
        timeout = 30.0

        def available(self):
            return True

        def chat(self, messages, tools=None):
            seen_timeout_during_call.append(self.timeout)
            return LLMResult(text="ok", provider="fake")

    provider = FakeProvider()
    monkeypatch.setattr(router, "_providers", lambda: {"fake": provider})
    router.call_provider_directly("fake", [{"role": "user", "content": "hi"}], timeout=120.0)
    assert seen_timeout_during_call == [120.0]
    assert provider.timeout == 30.0  # restored after the call


def test_an_exception_inside_chat_surfaces_as_a_normal_error_not_a_crash(monkeypatch):
    class FakeProvider:
        def available(self):
            return True

        def chat(self, messages, tools=None):
            raise RuntimeError("real API blew up")

    monkeypatch.setattr(router, "_providers", lambda: {"fake": FakeProvider()})
    result = router.call_provider_directly("fake", [{"role": "user", "content": "hi"}])
    assert result.error
    assert "real API blew up" in result.error
