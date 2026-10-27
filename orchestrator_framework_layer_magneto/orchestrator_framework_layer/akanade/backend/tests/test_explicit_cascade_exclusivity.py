"""Proves the actual claim, not just the CRUD layer: TierClient.chat()'s
`explicit_cascade` (CodingAgent's own always-on cascade, see
llm/coding_pipeline.py) is EXCLUSIVE — once at least one named candidate
is available, nothing outside that list is ever tried, unlike
primary_pipeline's own routellm_cascade, which is a PREFIX (falls through
to the rest of the tier's normal chain on failure). Real gap this fixes:
before this, explicit_cascade only reordered the front of the chain, so
every OTHER configured provider (max, luna, deepseek, ...) was still
reachable once the named cascade was exhausted — the opposite of "every
other provider should be off" for the Coding Agent."""
from __future__ import annotations

from app.llm.providers import LLMResult, OpenAICompatible
from app.llm.router import TierClient

from tests.conftest import auth

_NAME_A = "cascade-excl-test-a"
_NAME_B = "cascade-excl-test-b"


def _register(client, exec_token, name):
    r = client.post("/api/admin/providers",
                    json={"name": name, "base_url": "http://fake.invalid/v1",
                          "model": "fake-model", "api_key": "fake-key",
                          "tiers": ["delay_ok"]},
                    headers=auth(exec_token))
    assert r.status_code == 201
    return r.json()["id"]


def test_explicit_cascade_never_falls_through_to_the_rest_of_the_chain(client, exec_token, monkeypatch):
    id_a = _register(client, exec_token, _NAME_A)
    id_b = _register(client, exec_token, _NAME_B)
    try:
        called = []

        def fake_chat(self, messages, tools=None):
            called.append(self.name)
            if self.name in (_NAME_A, _NAME_B):
                raise RuntimeError(f"{self.name} always fails in this test")
            return LLMResult(text="should never be reached", provider=self.name)

        monkeypatch.setattr(OpenAICompatible, "chat", fake_chat)

        result = TierClient(tier="delay_ok", exclude_rule_based=True,
                            explicit_cascade=[_NAME_A, _NAME_B]).chat(
            "test-node", [{"role": "user", "content": "hi"}])

        assert called == [_NAME_A, _NAME_B], (
            "explicit_cascade must try exactly its own two candidates, in "
            f"order, and nothing else — actually called: {called}")
        assert result.error, "every named candidate failed — this must be a real error, not a silent fallback"
    finally:
        client.delete(f"/api/admin/providers/{id_a}", headers=auth(exec_token))
        client.delete(f"/api/admin/providers/{id_b}", headers=auth(exec_token))


def test_primary_pipelines_routellm_cascade_still_falls_through(client, exec_token, monkeypatch):
    """The OTHER caller of this same splice mechanism — primary_pipeline's
    routellm_cascade — must keep its existing "prefix, then fall through
    to the rest of the chain" behavior; only CodingAgent's explicit_cascade
    became exclusive. Confirms the fix didn't overcorrect."""
    from app.llm import primary_pipeline

    id_a = _register(client, exec_token, _NAME_A)
    try:
        primary_pipeline.set_pipeline("routellm", "test")
        called = []

        def fake_chat(self, messages, tools=None):
            called.append(self.name)
            raise RuntimeError(f"{self.name} always fails in this test")

        monkeypatch.setattr(OpenAICompatible, "chat", fake_chat)
        monkeypatch.setattr(primary_pipeline, "resolve_routellm_cascade", lambda: [_NAME_A])

        # No other real provider has a key in this test environment (see
        # conftest.py's "force every rung unavailable" setup) — the
        # deterministic "rule" floor is the one always-available rung
        # that's genuinely a DIFFERENT provider than _NAME_A, so reaching
        # it (rather than a bare "all providers failed" error) is the
        # real, unambiguous proof this cascade fell through to the rest
        # of the chain instead of stopping at its own one candidate.
        result = TierClient(tier="delay_ok").chat("test-node", [{"role": "user", "content": "hi"}])

        assert called == [_NAME_A]
        assert not result.error
        assert result.provider == "rule-based", (
            f"must fall through to the rule-based floor after the cascade's own candidate fails, "
            f"got provider={result.provider!r} error={result.error!r}")
    finally:
        primary_pipeline.set_pipeline("routellm", "test")  # restore the deployment default
        client.delete(f"/api/admin/providers/{id_a}", headers=auth(exec_token))
