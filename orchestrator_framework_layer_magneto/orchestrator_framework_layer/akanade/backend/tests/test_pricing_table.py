"""app/llm/pricing_table.py — the $/1k-token rate table underneath the
Token & Cost Optimizer layer, and its composition with an admin-
configured provider_budget.py rate (which must always win when set).
"""
from __future__ import annotations

from app.llm import pricing_table, provider_budget


def test_estimate_tokens_is_a_positive_chars_over_four_heuristic():
    assert pricing_table.estimate_tokens("") == 1  # never zero — max(1, ...)
    assert pricing_table.estimate_tokens("a" * 400) == 100


def test_estimate_prompt_tokens_joins_every_message_content():
    messages = [{"role": "user", "content": "a" * 40}, {"role": "assistant", "content": "b" * 40}]
    # 80 chars of content plus the joining space -> roughly 81/4
    assert pricing_table.estimate_prompt_tokens(messages) == pricing_table.estimate_tokens(
        "a" * 40 + " " + "b" * 40)


def test_ollama_and_rule_are_always_free():
    assert pricing_table.rates_for("ollama") == (0.0, 0.0)
    assert pricing_table.rates_for("rule") == (0.0, 0.0)
    assert pricing_table.estimate_cost_usd("ollama", 100_000, 100_000) == 0.0


def test_a_known_hosted_provider_has_a_nonzero_default_rate():
    prompt_rate, completion_rate = pricing_table.rates_for("luna")
    assert prompt_rate > 0.0
    assert completion_rate > 0.0


def test_an_unknown_provider_gets_the_conservative_fallback_rate():
    assert pricing_table.rates_for("some-brand-new-provider") == pricing_table._FALLBACK_RATE_USD_PER_1K


def test_estimate_cost_usd_scales_linearly_with_tokens():
    small = pricing_table.estimate_cost_usd("luna", 1000, 1000)
    large = pricing_table.estimate_cost_usd("luna", 2000, 2000)
    assert large == small * 2


def test_admin_configured_budget_rate_overrides_the_default_table(monkeypatch):
    """The whole point of checking provider_budget first — a real
    operator-entered rate must always win over this file's best-effort
    public list price."""
    monkeypatch.setattr(provider_budget, "get_budget",
                        lambda key: {"costPer1kPromptTokens": 9.0, "costPer1kCompletionTokens": 18.0}
                        if key == "luna" else None)
    assert pricing_table.rates_for("luna") == (9.0, 18.0)


def test_admin_budget_with_only_one_rate_set_falls_back_for_the_other(monkeypatch):
    monkeypatch.setattr(provider_budget, "get_budget",
                        lambda key: {"costPer1kPromptTokens": 9.0, "costPer1kCompletionTokens": None}
                        if key == "luna" else None)
    default_prompt, default_completion = pricing_table._DEFAULT_RATES_USD_PER_1K["luna"]
    prompt_rate, completion_rate = pricing_table.rates_for("luna")
    assert prompt_rate == 9.0
    assert completion_rate == default_completion


def test_a_provider_with_no_budget_configured_at_all_uses_the_default_table(monkeypatch):
    monkeypatch.setattr(provider_budget, "get_budget", lambda key: None)
    assert pricing_table.rates_for("luna") == pricing_table._DEFAULT_RATES_USD_PER_1K["luna"]


def test_max_known_rates_is_the_priciest_rate_in_the_default_table():
    max_prompt, max_completion = pricing_table.max_known_rates()
    all_prompts = [p for p, _ in pricing_table._DEFAULT_RATES_USD_PER_1K.values()]
    all_completions = [c for _, c in pricing_table._DEFAULT_RATES_USD_PER_1K.values()]
    assert max_prompt == max(all_prompts)
    assert max_completion == max(all_completions)


def test_counterfactual_cost_usd_uses_the_priciest_known_rate():
    max_prompt, max_completion = pricing_table.max_known_rates()
    assert pricing_table.counterfactual_cost_usd(1000, 1000) == max_prompt + max_completion


def test_counterfactual_cost_usd_is_zero_for_zero_tokens():
    assert pricing_table.counterfactual_cost_usd(0, 0) == 0.0
