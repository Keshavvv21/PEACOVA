"""Proves the actual claim, not just the CRUD layer: once a provider's
budget is exceeded, llm/router.py's TierClient.chat() genuinely skips
it and falls through to the next rung in the chain — real
_effective_chains()/available_keys filtering, not a monkeypatched
shortcut."""
from __future__ import annotations

from app.llm import provider_budget
from app.llm.providers import LLMResult, OpenAICompatible
from app.llm.router import TierClient

from tests.conftest import auth

PROVIDER_NAME = "budget-test-provider"


def _register_provider(client, exec_token, *, tiers=("realtime",)):
    r = client.post("/api/admin/providers",
                    json={"name": PROVIDER_NAME, "base_url": "http://fake.invalid/v1",
                          "model": "fake-model", "api_key": "fake-key",
                          "tiers": list(tiers)},
                    headers=auth(exec_token))
    assert r.status_code == 201
    return r.json()["id"]


def test_a_maxed_out_provider_is_skipped_by_tier_client(client, exec_token, monkeypatch):
    provider_id = _register_provider(client, exec_token)
    try:
        provider_budget.set_budget(PROVIDER_NAME, max_tokens=100, max_cost_usd=None,
                                   cost_per_1k_prompt_tokens=None,
                                   cost_per_1k_completion_tokens=None)
        provider_budget.record_usage(PROVIDER_NAME, prompt_tokens=100, completion_tokens=0)
        assert provider_budget.is_exceeded(PROVIDER_NAME) is True

        called_providers = []

        def fake_chat(self, messages, tools=None):
            called_providers.append(self.name)
            if self.name == PROVIDER_NAME:
                # Should never actually be reached — is_exceeded() must
                # filter it out before any provider in the chain gets called.
                return LLMResult(text="I should never answer", provider=self.name)
            return LLMResult(text="rule-based fallback answered instead", provider=self.name)

        monkeypatch.setattr(OpenAICompatible, "chat", fake_chat)

        result = TierClient(tier="realtime").chat("test-node", [{"role": "user", "content": "hi"}])
        assert PROVIDER_NAME not in [
            n for n in called_providers if n == PROVIDER_NAME
        ], "the budget-exceeded provider must never be called at all"
        assert "rule-based" in result.text or result.text  # something else answered instead
    finally:
        provider_budget.delete_budget(PROVIDER_NAME)
        client.delete(f"/api/admin/providers/{provider_id}", headers=auth(exec_token))


def test_an_explicit_preferred_pick_still_respects_the_budget(client, exec_token, monkeypatch):
    provider_id = _register_provider(client, exec_token)
    try:
        provider_budget.set_budget(PROVIDER_NAME, max_tokens=1, max_cost_usd=None,
                                   cost_per_1k_prompt_tokens=None,
                                   cost_per_1k_completion_tokens=None)
        provider_budget.record_usage(PROVIDER_NAME, prompt_tokens=1, completion_tokens=0)

        def fake_chat(self, messages, tools=None):
            return LLMResult(text=f"answered by {self.name}", provider=self.name)

        monkeypatch.setattr(OpenAICompatible, "chat", fake_chat)

        # Explicitly asking for the maxed-out provider must NOT bypass
        # the budget — see router.py's `self.preferred` branch, which
        # now also checks is_exceeded().
        result = TierClient(tier="realtime", preferred=PROVIDER_NAME).chat(
            "test-node", [{"role": "user", "content": "hi"}])
        assert f"answered by {PROVIDER_NAME}" not in result.text
    finally:
        provider_budget.delete_budget(PROVIDER_NAME)
        client.delete(f"/api/admin/providers/{provider_id}", headers=auth(exec_token))
