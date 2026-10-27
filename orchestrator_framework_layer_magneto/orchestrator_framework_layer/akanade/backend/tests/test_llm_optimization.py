"""Real code for ARCHITECTURE.md §7's model optimization strategies:
llm/optimization.py's OptimizationProfile (declarative — what a
backend IS deployed with) and llm/optimization_router.py's
OptimizationRouter (a SECOND Thompson Sampling bandit, reusing
gmaofn/bandit.py's core, that picks a REAL request-level parameter
combination on every single model call and learns from that call's own
latency/success) — plus the actual wiring into llm/router.py's
TierClient.chat(), which runs this on every provider in the chain.
"""
from __future__ import annotations

from app.framework.bandit import LinearBanditCore
from app.llm.optimization import OptimizationProfile
from app.llm.optimization_router import DEFAULT_ARMS, OptimizationArm, OptimizationRouter
from app.llm.providers import LLMResult, OpenAICompatible
from app.llm.router import TierClient


# ── LinearBanditCore (the shared primitive) ──────────────────────────────

def test_linear_bandit_core_learns_the_better_arm():
    core = LinearBanditCore(feature_dim=2, exploration_scale=0.2)
    for _ in range(50):
        core.update("good", [0.0, 0.0], 0.9)
        core.update("bad", [0.0, 0.0], -0.6)
    picks = [core.select(["good", "bad"], [0.0, 0.0])[0] for _ in range(150)]
    assert picks.count("good") / len(picks) > 0.8


# ── OptimizationProfile (declarative metadata) ───────────────────────────

def test_optimization_profile_round_trips_through_dict():
    profile = OptimizationProfile(kv_cache=True, continuous_batching=True,
                                  tensor_parallel_size=4, attention_backend="paged_attention")
    restored = OptimizationProfile.from_dict(profile.to_dict())
    assert restored == profile


def test_optimization_profile_lists_enabled_strategies():
    profile = OptimizationProfile(kv_cache=True, speculative_decoding=True,
                                  speculative_draft_model="tiny-draft", tensor_parallel_size=4)
    strategies = profile.enabled_strategies()
    assert "kv_cache" in strategies
    assert any("speculative_decoding" in s and "tiny-draft" in s for s in strategies)
    assert any("tensor_parallel" in s for s in strategies)


def test_default_optimization_profile_has_no_strategies_enabled():
    assert OptimizationProfile().enabled_strategies() == []


# ── OptimizationRouter (the per-call RL decision) ────────────────────────

def test_default_arms_include_balanced_low_latency_high_quality():
    ids = {a.arm_id for a in DEFAULT_ARMS}
    assert {"balanced", "low_latency", "high_quality"} == ids


def test_optimization_router_learns_to_prefer_the_better_arm():
    router = OptimizationRouter(exploration_scale=0.2)
    router.register_arms("test-provider", [
        OptimizationArm("great", "always works fast", extra_body={}),
        OptimizationArm("bad", "always slow/fails", extra_body={}),
    ])
    for _ in range(40):
        router.update("test-provider", OptimizationArm("great", ""), "realtime",
                      ok=True, latency_ms=50)
        router.update("test-provider", OptimizationArm("bad", ""), "realtime",
                      ok=False, latency_ms=4000)
    picks = [router.select("test-provider", "realtime").arm_id for _ in range(100)]
    assert picks.count("great") / len(picks) > 0.75


def test_optimization_router_falls_back_to_default_arms_for_unregistered_provider():
    router = OptimizationRouter()
    arm = router.select("some-brand-new-provider", "realtime")
    assert arm.arm_id in {a.arm_id for a in DEFAULT_ARMS}


# ── Real request-payload pass-through (extra_body) ───────────────────────

def test_openai_compatible_merges_extra_body_into_request_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("app.llm.providers.httpx.post", fake_post)

    provider = OpenAICompatible("test", "http://fake", "key", "test-model",
                                extra_body={"temperature": 0.0, "top_k": 40})
    result = provider.chat([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["top_k"] == 40
    assert captured["payload"]["model"] == "test-model"


def test_tier_client_applies_and_restores_optimization_arm(monkeypatch):
    """The real wiring in TierClient.chat(): a provider's extra_body is
    temporarily overridden with the RL-selected arm's extra_body for
    exactly one call, then restored — proven by checking the payload
    seen mid-call reflects the arm, and the provider's own extra_body
    is unchanged afterward."""
    seen_payloads = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout):
        seen_payloads.append(dict(json))
        return FakeResponse()

    monkeypatch.setattr("app.llm.providers.httpx.post", fake_post)

    provider = OpenAICompatible("fake-rung", "http://fake", "key", "test-model",
                                extra_body={"original": "value"})
    from app.llm import optimization_router as opt_mod
    opt_mod.optimization_router.register_arms("fake-rung", [
        OptimizationArm("forced", "test arm", extra_body={"temperature": 0.0}),
    ])

    client = TierClient(tier="realtime")
    # Bypass the normal chain-building (which reads real settings) —
    # exercise the exact same per-provider block chat() runs by calling
    # it against a chain of just this one fake provider.
    monkeypatch.setattr("app.llm.router._effective_chains", lambda: {"realtime": ["fake-rung"]})
    monkeypatch.setattr("app.llm.router._providers", lambda: {"fake-rung": provider})

    result = client.chat("test_node", [{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert seen_payloads[0]["temperature"] == 0.0        # the arm's override was applied
    assert seen_payloads[0]["original"] == "value"       # merged with, not replacing, the provider's own
    assert provider.extra_body == {"original": "value"}  # restored after the call
    assert client.log[0]["optimizationArm"] == "forced"
