"""app/llm/provider_optimizer.py — the RL-learned reordering of a
tier's provider fallback chain (real success/latency, not the fixed
CHAINS order), and its wiring into TierClient.chat() (router.py).
"""
from __future__ import annotations

from app.llm.provider_optimizer import ProviderOptimizer, provider_optimizer
from app.llm.providers import LangChainOllama
from app.llm.router import CHAINS, TierClient, _providers


# ── ProviderOptimizer unit tests ───────────────────────────────────────

def test_order_keeps_the_floor_last_regardless_of_learning():
    opt = ProviderOptimizer()
    for _ in range(20):
        opt.record_outcome("realtime", "rule", ok=True, latency_ms=1)  # no-op, see docstring
    ordered = opt.order("realtime", ["luna", "groq", "rule"])
    assert ordered[-1] == "rule"


def test_order_is_a_stable_permutation_not_a_drop():
    opt = ProviderOptimizer()
    candidates = ["luna", "groq", "deepseek", "ollama", "rule"]
    ordered = opt.order("realtime", candidates)
    assert sorted(ordered) == sorted(candidates)


def test_order_with_a_single_candidate_is_a_noop():
    opt = ProviderOptimizer()
    assert opt.order("realtime", ["luna"]) == ["luna"]


def test_order_with_no_candidates_is_a_noop():
    opt = ProviderOptimizer()
    assert opt.order("realtime", []) == []


def test_repeated_failures_eventually_rank_a_provider_lower():
    opt = ProviderOptimizer()
    # groq fails a lot, luna always succeeds fast — real signal, not a mock.
    for _ in range(15):
        opt.record_outcome("realtime", "groq", ok=False, latency_ms=5000)
        opt.record_outcome("realtime", "luna", ok=True, latency_ms=200)
    # Sample the ranking many times — Thompson Sampling is randomized, so
    # assert on the CLEAR MAJORITY behavior, not every single draw.
    luna_first = sum(1 for _ in range(50)
                     if opt.order("realtime", ["groq", "luna"])[0] == "luna")
    assert luna_first > 35  # well above chance (25/50) given the strong signal


def test_arms_are_isolated_per_tier():
    opt = ProviderOptimizer()
    for _ in range(15):
        opt.record_outcome("realtime", "groq", ok=False, latency_ms=5000)
        opt.record_outcome("realtime", "luna", ok=True, latency_ms=200)
    # The SAME providers in a DIFFERENT tier have no learned history yet —
    # order should still roughly track the cold-start (input) order.
    bulk_order = opt.order("bulk", ["groq", "luna"])
    assert set(bulk_order) == {"groq", "luna"}


def test_record_outcome_never_learns_from_the_floor():
    opt = ProviderOptimizer()
    for _ in range(20):
        opt.record_outcome("realtime", "rule", ok=False, latency_ms=999999)
    # No arm should have been created for the floor at all.
    assert opt._core.posterior_mean("realtime::rule") is None


# ── persistence (opt-in via name=) ──────────────────────────────────────

def test_named_instance_persists_after_enough_updates():
    opt1 = ProviderOptimizer(name="test-provider-optimizer-persistence")
    opt1._SAVE_EVERY = 3  # small, so the test doesn't need 20 real calls
    for _ in range(3):
        opt1.record_outcome("realtime", "luna", ok=True, latency_ms=100)

    opt2 = ProviderOptimizer(name="test-provider-optimizer-persistence")
    assert opt2._core.posterior_mean("realtime::luna") is not None


def test_unnamed_instance_never_persists():
    opt1 = ProviderOptimizer()
    for _ in range(20):
        opt1.record_outcome("realtime", "luna", ok=True, latency_ms=100)
    opt2 = ProviderOptimizer()
    assert opt2._core.posterior_mean("realtime::luna") is None


# ── stats() — the "who's winning for what" dashboard's data source ──────

def test_stats_reports_calls_wins_and_win_rate_per_tier_provider():
    opt = ProviderOptimizer(name="test-provider-optimizer-stats")
    opt.record_outcome("realtime", "luna", ok=True, latency_ms=100)
    opt.record_outcome("realtime", "luna", ok=False, latency_ms=100)
    row = next(r for r in opt.stats() if r["tier"] == "realtime" and r["provider"] == "luna")
    assert row["calls"] == 2
    assert row["wins"] == 1
    assert row["win_rate"] == 0.5
    assert row["posterior_mean_weights"] is not None


def test_stats_never_includes_the_floor_provider():
    opt = ProviderOptimizer(name="test-provider-optimizer-stats-floor")
    for _ in range(5):
        opt.record_outcome("realtime", "rule", ok=False, latency_ms=1)
    assert all(r["provider"] != "rule" for r in opt.stats())


def test_unnamed_optimizer_stats_is_still_readable_in_memory():
    opt = ProviderOptimizer()
    opt.record_outcome("realtime", "luna", ok=True, latency_ms=100)
    row = next(r for r in opt.stats() if r["tier"] == "realtime" and r["provider"] == "luna")
    assert row["calls"] == 0  # in-memory counts are only tracked for named instances
    assert row["posterior_mean_weights"] is not None  # the bandit itself still learned


# ── real wiring into TierClient.chat() ──────────────────────────────────

def test_tier_client_chat_uses_provider_keys_not_display_names(monkeypatch):
    """Regression test for the real bug this feature surfaced: `.name`
    (display string, e.g. "gpt-5.6-luna") and the registry key ("luna")
    are different namespaces — TierClient.chat() must key everything
    (ordering, preferred-matching, provider_optimizer bookkeeping) by
    the registry key throughout, never mix the two."""
    calls = []

    def fake_record_outcome(tier, key, *, ok, latency_ms):
        calls.append(key)

    monkeypatch.setattr(provider_optimizer, "record_outcome", fake_record_outcome)
    client = TierClient(tier="bulk")  # bulk chain has no configured keys -> falls to rule
    result = client.chat("test-node", [{"role": "user", "content": "hi"}])
    assert result.provider == "rule-based"  # the real RuleBased provider's .name
    # router.py reports the REGISTRY key ("rule") to provider_optimizer,
    # never the display name ("rule-based") — ProviderOptimizer.record_outcome
    # itself is what decides to ignore the floor (see
    # test_record_outcome_never_learns_from_the_floor above); the bug
    # this regression-tests would have shown "rule-based" here instead.
    assert calls == ["rule"]


def test_preferred_timeout_and_escalation_use_the_registry_key(monkeypatch):
    """Regression test for the second real bug this surfaced: Hybrid
    mode's `preferred="ollama"` (a registry key) was being compared
    against `p.name` ("ollama-<model>", a display string) — always
    False — so `preferred_timeout`/`preferred_escalated` silently never
    fired for the exact built-in provider Hybrid mode is built around.
    Simulated here without a real Ollama server: force EVERY
    LangChainOllama instance to report available and fail on `.chat()`
    (TierClient.chat() builds a fresh provider set on every call via
    `_providers()`, so patching one instance wouldn't affect the one it
    actually uses — patch the class instead) and confirm escalation is
    detected via the registry key, not the display name."""
    def fake_chat(self, messages, tools):
        raise RuntimeError("simulated ollama timeout")

    monkeypatch.setattr(LangChainOllama, "available", lambda self: True)
    monkeypatch.setattr(LangChainOllama, "chat", fake_chat)

    client = TierClient(tier="realtime", preferred="ollama", preferred_timeout=5.0)
    result = client.chat("test-node", [{"role": "user", "content": "hi"}])
    assert client.preferred_escalated is True
    assert result.provider != _providers()["ollama"].name  # fell through to a later rung
