"""app/llm/cost_optimizer.py — the RL-learned, cost-aware re-ranking of
a tier's provider chain (composed with provider_optimizer.py's own
success/latency ordering), pre-execution cost prediction, and real
tracked cost/token/latency/quality stats. Mirrors
tests/test_provider_optimizer.py's structure and conventions.
"""
from __future__ import annotations

from app.llm.cost_optimizer import CostOptimizer, CostEstimate, cost_and_savings_for_call
from app.llm.providers import LangChainOllama
from app.llm.router import TierClient
from app.llm import cost_optimizer as cost_optimizer_module


# ── estimate() — pre-execution cost prediction ───────────────────────────

def test_estimate_returns_one_forecast_per_candidate():
    opt = CostOptimizer()
    messages = [{"role": "user", "content": "Backtest AAPL for the last 10 years"}]
    estimates = opt.estimate(messages, ["luna", "ollama", "rule"])
    assert {e.provider for e in estimates} == {"luna", "ollama", "rule"}


def test_estimate_reports_zero_cost_for_free_providers():
    opt = CostOptimizer()
    estimates = opt.estimate([{"role": "user", "content": "hello"}], ["ollama", "rule"])
    assert all(e.predicted_cost_usd == 0.0 for e in estimates)


def test_estimate_reports_nonzero_cost_for_a_hosted_provider():
    opt = CostOptimizer()
    estimates = opt.estimate([{"role": "user", "content": "hello " * 200}], ["luna"])
    assert estimates[0].predicted_cost_usd > 0.0
    assert estimates[0].predicted_prompt_tokens > 0
    assert estimates[0].predicted_completion_tokens > 0


def test_a_longer_message_predicts_a_higher_cost():
    opt = CostOptimizer()
    short = opt.estimate([{"role": "user", "content": "hi"}], ["luna"])[0]
    long = opt.estimate([{"role": "user", "content": "hello " * 500}], ["luna"])[0]
    assert long.predicted_cost_usd > short.predicted_cost_usd


# ── order() — cost-aware ranking ──────────────────────────────────────────

def test_order_keeps_the_floor_last_regardless_of_cost():
    opt = CostOptimizer()
    estimates = opt.estimate([{"role": "user", "content": "hi"}], ["luna", "max", "rule"])
    ordered = opt.order("realtime", estimates)
    assert ordered[-1] == "rule"


def test_order_is_a_stable_permutation_not_a_drop():
    opt = CostOptimizer()
    candidates = ["luna", "groq", "deepseek", "ollama", "rule"]
    estimates = opt.estimate([{"role": "user", "content": "hi"}], candidates)
    ordered = opt.order("realtime", estimates)
    assert sorted(ordered) == sorted(candidates)


def test_order_with_a_single_candidate_is_a_noop():
    opt = CostOptimizer()
    estimates = opt.estimate([{"role": "user", "content": "hi"}], ["luna"])
    assert opt.order("realtime", estimates) == ["luna"]


def test_order_with_no_candidates_is_a_noop():
    opt = CostOptimizer()
    assert opt.order("realtime", []) == []


def test_a_free_local_provider_is_cold_start_disfavored_against_a_hosted_one():
    """Real bug found live: a genuinely free candidate (ollama, $0) used
    to be COLD-START FAVORED over a paid hosted one purely on cost, with
    zero latency evidence behind the pick — on this deployment, $0
    means "CPU-only local model," repeatedly measured at 15-37 seconds
    for an ordinary tool-calling turn, next to 1-2s hosted answers. That
    cost-only seed alone was enough to override provider_optimizer.py's
    own (correctly hosted-first) cold-start ordering on every fresh
    process start. A free candidate must now be the LESS likely
    cold-start pick against any real paid option — see
    _FREE_PROVIDER_COLD_START_SEED — while remaining genuinely reachable
    (never hard-disabled) so real evidence can still favor it later."""
    opt = CostOptimizer()
    ollama_first = 0
    for _ in range(50):
        estimates = opt.estimate([{"role": "user", "content": "hello " * 200}], ["max", "ollama"])
        if opt.order("realtime", estimates)[0] == "ollama":
            ollama_first += 1
    assert 0 < ollama_first < 20  # reachable, but no longer the cold-start favorite


# ── record_outcome() — real reward from real outcomes ────────────────────

def test_record_outcome_never_learns_from_the_floor():
    opt = CostOptimizer()
    for _ in range(20):
        opt.record_outcome("realtime", "rule", ok=False, latency_ms=999999,
                          truncated=False, prompt_tokens=0, completion_tokens=0)
    assert opt._core.posterior_mean("realtime::rule") is None


def test_record_outcome_returns_the_real_dollar_cost_of_this_call():
    opt = CostOptimizer()
    cost = opt.record_outcome("realtime", "luna", ok=True, latency_ms=100,
                              truncated=False, prompt_tokens=1000, completion_tokens=1000)
    assert cost > 0.0


def test_record_outcome_for_a_free_provider_returns_zero_cost():
    opt = CostOptimizer()
    cost = opt.record_outcome("realtime", "ollama", ok=True, latency_ms=100,
                              truncated=False, prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.0


def test_repeated_expensive_failures_eventually_rank_a_provider_lower():
    opt = CostOptimizer()
    for _ in range(15):
        opt.record_outcome("realtime", "max", ok=False, latency_ms=5000,
                          truncated=False, prompt_tokens=2000, completion_tokens=2000)
        opt.record_outcome("realtime", "ollama", ok=True, latency_ms=200,
                          truncated=False, prompt_tokens=2000, completion_tokens=2000)
    ollama_first = 0
    for _ in range(50):
        estimates = opt.estimate([{"role": "user", "content": "hi"}], ["max", "ollama"])
        if opt.order("realtime", estimates)[0] == "ollama":
            ollama_first += 1
    assert ollama_first > 35


def test_a_truncated_reply_is_penalized_even_though_ok_is_true():
    opt = CostOptimizer()
    for _ in range(15):
        opt.record_outcome("realtime", "luna", ok=True, latency_ms=100,
                          truncated=True, prompt_tokens=500, completion_tokens=500)
        opt.record_outcome("realtime", "max", ok=True, latency_ms=100,
                          truncated=False, prompt_tokens=500, completion_tokens=500)
    mean_luna = opt._core.posterior_mean("realtime::luna")
    mean_max = opt._core.posterior_mean("realtime::max")
    assert mean_luna[-1] < mean_max[-1]  # bias term — luna's truncated calls score lower


# ── persistence (opt-in via name=) ────────────────────────────────────────

def test_named_instance_persists_after_enough_updates():
    opt1 = CostOptimizer(name="test-cost-optimizer-persistence")
    opt1._SAVE_EVERY = 3
    for _ in range(3):
        opt1.record_outcome("realtime", "luna", ok=True, latency_ms=100,
                           truncated=False, prompt_tokens=100, completion_tokens=100)
    opt2 = CostOptimizer(name="test-cost-optimizer-persistence")
    assert opt2._core.posterior_mean("realtime::luna") is not None


def test_unnamed_instance_never_persists():
    opt1 = CostOptimizer()
    for _ in range(20):
        opt1.record_outcome("realtime", "luna", ok=True, latency_ms=100,
                           truncated=False, prompt_tokens=100, completion_tokens=100)
    opt2 = CostOptimizer()
    assert opt2._core.posterior_mean("realtime::luna") is None


# ── stats() and savings_summary() — the dashboard's real data source ─────

def test_stats_reports_real_tracked_totals_per_tier_provider():
    opt = CostOptimizer(name="test-cost-optimizer-stats")
    opt.record_outcome("realtime", "luna", ok=True, latency_ms=100,
                      truncated=False, prompt_tokens=1000, completion_tokens=500)
    opt.record_outcome("realtime", "luna", ok=False, latency_ms=200,
                      truncated=False, prompt_tokens=0, completion_tokens=0)
    row = next(r for r in opt.stats() if r["tier"] == "realtime" and r["provider"] == "luna")
    assert row["calls"] == 2
    assert row["win_rate"] == 0.5
    assert row["total_prompt_tokens"] == 1000
    assert row["total_completion_tokens"] == 500
    assert row["total_cost_usd"] > 0.0
    assert row["avg_latency_ms"] == 150.0
    assert row["posterior_mean_weights"] is not None


def test_stats_never_includes_the_floor_provider():
    opt = CostOptimizer(name="test-cost-optimizer-stats-floor")
    for _ in range(5):
        opt.record_outcome("realtime", "rule", ok=False, latency_ms=1,
                          truncated=False, prompt_tokens=0, completion_tokens=0)
    assert all(r["provider"] != "rule" for r in opt.stats())


def test_savings_summary_is_zero_with_no_calls_yet():
    opt = CostOptimizer(name="test-cost-optimizer-savings-empty")
    summary = opt.savings_summary()
    assert summary["total_cost_usd"] == 0.0
    assert summary["estimated_savings_usd"] == 0.0


def test_savings_summary_shows_real_savings_when_cheaper_providers_win():
    opt = CostOptimizer(name="test-cost-optimizer-savings-real")
    # All traffic went to the free local model instead of the priciest
    # hosted tier — real, computable savings vs. the counterfactual.
    for _ in range(10):
        opt.record_outcome("realtime", "ollama", ok=True, latency_ms=100,
                          truncated=False, prompt_tokens=1000, completion_tokens=1000)
    summary = opt.savings_summary()
    assert summary["total_cost_usd"] == 0.0
    assert summary["counterfactual_cost_usd"] > 0.0
    assert summary["estimated_savings_usd"] == summary["counterfactual_cost_usd"]
    assert summary["estimated_savings_pct"] == 100.0


# ── cost_and_savings_for_call() — the per-call helper router.py uses ────────

def test_cost_and_savings_for_call_is_zero_savings_at_the_priciest_rate():
    from app.llm import pricing_table
    priciest = max(pricing_table._DEFAULT_RATES_USD_PER_1K,
                   key=lambda k: sum(pricing_table._DEFAULT_RATES_USD_PER_1K[k]))
    result = cost_and_savings_for_call(priciest, 1000, 1000)
    assert result["savings_usd"] == 0.0
    assert result["savings_pct"] == 0.0
    assert result["cost_usd"] == result["counterfactual_cost_usd"]


def test_cost_and_savings_for_call_shows_real_savings_for_a_free_provider():
    result = cost_and_savings_for_call("ollama", 1000, 1000)
    assert result["cost_usd"] == 0.0
    assert result["counterfactual_cost_usd"] > 0.0
    assert result["savings_usd"] == result["counterfactual_cost_usd"]
    assert result["savings_pct"] == 100.0


def test_cost_and_savings_for_call_is_zero_for_zero_tokens():
    result = cost_and_savings_for_call("ollama", 0, 0)
    assert result == {"cost_usd": 0.0, "counterfactual_cost_usd": 0.0,
                      "savings_usd": 0.0, "savings_pct": 0.0}


# ── real wiring into TierClient.chat() ────────────────────────────────────

def test_tier_client_chat_records_a_cost_outcome_keyed_by_registry_key(monkeypatch):
    """Same regression shape as test_provider_optimizer.py's own wiring
    test — cost_optimizer must be keyed by the registry key ("rule"),
    never a display name, and must actually fire on a real chat() call."""
    calls = []

    def fake_record_outcome(tier, key, *, ok, latency_ms, truncated, prompt_tokens, completion_tokens):
        calls.append(key)
        return 0.0

    monkeypatch.setattr(cost_optimizer_module.cost_optimizer, "record_outcome", fake_record_outcome)
    client = TierClient(tier="bulk")  # bulk chain has no configured keys -> falls to rule
    result = client.chat("test-node", [{"role": "user", "content": "hi"}])
    assert result.provider == "rule-based"
    assert calls == ["rule"]


def test_tier_client_chat_still_works_when_a_provider_fails(monkeypatch):
    """cost_optimizer's failure-path record_outcome must not itself
    raise or otherwise break the existing fallback loop."""
    def fake_chat(self, messages, tools):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(LangChainOllama, "available", lambda self: True)
    monkeypatch.setattr(LangChainOllama, "chat", fake_chat)

    client = TierClient(tier="realtime", preferred="ollama")
    result = client.chat("test-node", [{"role": "user", "content": "hi"}])
    assert result is not None  # fell through to a later rung without crashing


def test_tier_log_carries_cost_and_savings_after_a_real_call():
    """"Show the token savings after each API/model call" — the SAME
    tier_log entry every platform (web, mobile, VS Code) already reads
    per turn must carry the real per-call costUsd/savingsUsd, not just
    the aggregate dashboard. bulk's chain has no configured keys, so
    this exercises the real (non-monkeypatched) rule-floor path."""
    client = TierClient(tier="bulk")
    client.chat("test-node", [{"role": "user", "content": "hi"}])
    entry = next(e for e in client.log if e.get("provider") == "rule-based")
    assert entry["costUsd"] == 0.0
    assert entry["savingsUsd"] == 0.0


def test_tier_log_records_zero_cost_and_savings_on_a_failed_call(monkeypatch):
    """The failure branch must also carry costUsd/savingsUsd (both 0.0
    — no usage was ever billed for a call that never returned) so
    clients rendering tier_log don't need to special-case a missing
    field on an error entry."""
    def fake_chat(self, messages, tools):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(LangChainOllama, "available", lambda self: True)
    monkeypatch.setattr(LangChainOllama, "chat", fake_chat)

    client = TierClient(tier="realtime", preferred="ollama")
    client.chat("test-node", [{"role": "user", "content": "hi"}])
    failed_entry = next(e for e in client.log if e.get("error"))
    assert failed_entry["costUsd"] == 0.0
    assert failed_entry["savingsUsd"] == 0.0
