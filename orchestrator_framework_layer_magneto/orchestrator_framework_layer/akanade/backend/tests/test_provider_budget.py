"""llm/provider_budget.py — per-provider daily token/cost caps and the
availability-filtering they feed into TierClient.chat()."""
from __future__ import annotations

import pytest

from app.llm import provider_budget


def test_no_budget_configured_never_exceeds():
    assert provider_budget.is_exceeded("claude") is False


def test_max_cost_usd_without_a_rate_is_rejected():
    with pytest.raises(ValueError):
        provider_budget.set_budget("claude", max_tokens=None, max_cost_usd=5.0,
                                   cost_per_1k_prompt_tokens=None,
                                   cost_per_1k_completion_tokens=None)


def test_token_budget_trips_once_reached():
    provider_budget.set_budget("claude-token-test", max_tokens=100, max_cost_usd=None,
                               cost_per_1k_prompt_tokens=None, cost_per_1k_completion_tokens=None)
    assert provider_budget.is_exceeded("claude-token-test") is False
    provider_budget.record_usage("claude-token-test", prompt_tokens=60, completion_tokens=50)
    assert provider_budget.is_exceeded("claude-token-test") is True


def test_cost_budget_trips_once_reached():
    provider_budget.set_budget("claude-cost-test", max_tokens=None, max_cost_usd=0.01,
                               cost_per_1k_prompt_tokens=1.0, cost_per_1k_completion_tokens=2.0)
    # 5 prompt + 3 completion tokens -> 5/1000*1.0 + 3/1000*2.0 = 0.011 >= 0.01
    provider_budget.record_usage("claude-cost-test", prompt_tokens=5, completion_tokens=3)
    assert provider_budget.is_exceeded("claude-cost-test") is True


def test_record_usage_is_a_noop_without_a_configured_budget():
    # No exception, no tracking — the common "nobody set a budget" case.
    provider_budget.record_usage("no-budget-provider", prompt_tokens=999999, completion_tokens=0)
    assert provider_budget.today_usage("no-budget-provider") == {"tokens": 0, "costUsd": 0.0}


def test_delete_budget_removes_the_cap():
    provider_budget.set_budget("claude-delete-test", max_tokens=1, max_cost_usd=None,
                               cost_per_1k_prompt_tokens=None, cost_per_1k_completion_tokens=None)
    provider_budget.record_usage("claude-delete-test", prompt_tokens=1, completion_tokens=0)
    assert provider_budget.is_exceeded("claude-delete-test") is True
    provider_budget.delete_budget("claude-delete-test")
    assert provider_budget.is_exceeded("claude-delete-test") is False
