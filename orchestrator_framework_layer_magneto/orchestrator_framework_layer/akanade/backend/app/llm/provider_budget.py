"""Per-provider daily token/cost budgets — an admin sets a cap (a token
count and/or a $ amount) for any provider, built-in (e.g. "claude",
configured via CLAUDE_API_KEY) or admin-added
(routers/admin_providers.py). Once EITHER cap is hit today,
TierClient.chat() (llm/router.py) automatically excludes that provider
from both the normal chain order AND an explicit `preferred` pick — the
exact same "just try the next rung" mechanism a provider outage already
takes, so a maxed-out budget degrades gracefully (falls through to the
next provider) instead of erroring or silently overspending.

Deliberately NOT provider-specific despite the "if I add a Claude key"
motivating case — the same risk (a real per-token cost, no built-in
spending cap) applies to any hosted provider an admin points a real API
key at, so this is generic by provider key from the start.

A day resets at UTC midnight (same convention as security_report.py's
own daily digest) — usage is stored per (provider_key, date), so
yesterday's spend never carries into today's cap check, and a provider
becomes available again automatically the moment the day rolls over,
with no cron job or manual reset needed.

Cost tracking needs an admin-supplied $/1k-token rate (there's no
universal live pricing feed for every possible provider) — a budget
with only `max_tokens` set works with zero extra configuration, but
`set_budget()` refuses to accept `max_cost_usd` without at least one
rate configured, rather than silently accepting a cap that would
accumulate $0.00 forever and never trip.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..db import store


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _budget_key(provider_key: str) -> str:
    return f"PROVIDERBUDGET#{provider_key}"


def _usage_key(provider_key: str) -> str:
    return f"PROVIDERBUDGETUSAGE#{provider_key}"


def get_budget(provider_key: str) -> dict | None:
    return store("ProviderBudgets").get(_budget_key(provider_key), "META")


def list_budgets() -> list[dict]:
    return store("ProviderBudgets").scan("PROVIDERBUDGET#")


def set_budget(provider_key: str, *, max_tokens: int | None, max_cost_usd: float | None,
              cost_per_1k_prompt_tokens: float | None,
              cost_per_1k_completion_tokens: float | None) -> dict:
    if max_cost_usd is not None and cost_per_1k_prompt_tokens is None \
            and cost_per_1k_completion_tokens is None:
        raise ValueError("max_cost_usd needs at least one of cost_per_1k_prompt_tokens/"
                         "cost_per_1k_completion_tokens set — otherwise cost can never be "
                         "computed from real usage and the cap would never trip.")
    row = {
        "providerKey": provider_key,
        "maxTokens": max_tokens,
        "maxCostUsd": max_cost_usd,
        "costPer1kPromptTokens": cost_per_1k_prompt_tokens,
        "costPer1kCompletionTokens": cost_per_1k_completion_tokens,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    store("ProviderBudgets").put(_budget_key(provider_key), "META", row)
    return row


def delete_budget(provider_key: str) -> None:
    store("ProviderBudgets").delete(_budget_key(provider_key), "META")


def today_usage(provider_key: str) -> dict:
    row = store("ProviderBudgetUsage").get(_usage_key(provider_key), _today())
    return row or {"tokens": 0, "costUsd": 0.0}


def record_usage(provider_key: str, *, prompt_tokens: int, completion_tokens: int) -> None:
    """Fire-and-forget from TierClient.chat()'s success path — a no-op
    (not even a store read) when no budget is configured for this
    provider, so this costs nothing extra for the common "no budget
    set" case. Never raises — a bookkeeping failure must not break a
    reply that already succeeded."""
    budget = get_budget(provider_key)
    if not budget:
        return
    try:
        today = _today()
        row = store("ProviderBudgetUsage").get(_usage_key(provider_key), today) \
            or {"tokens": 0, "costUsd": 0.0}
        row["tokens"] = row.get("tokens", 0) + prompt_tokens + completion_tokens
        cost_delta = 0.0
        if budget.get("costPer1kPromptTokens") is not None:
            cost_delta += prompt_tokens / 1000 * budget["costPer1kPromptTokens"]
        if budget.get("costPer1kCompletionTokens") is not None:
            cost_delta += completion_tokens / 1000 * budget["costPer1kCompletionTokens"]
        row["costUsd"] = row.get("costUsd", 0.0) + cost_delta
        store("ProviderBudgetUsage").put(_usage_key(provider_key), today, row)
    except Exception:
        pass


def is_exceeded(provider_key: str) -> bool:
    """The ONLY function llm/router.py's availability filtering calls —
    False (never blocks) when no budget is configured, matching every
    other provider's default "just works" behavior."""
    budget = get_budget(provider_key)
    if not budget:
        return False
    usage = today_usage(provider_key)
    if budget.get("maxTokens") is not None and usage.get("tokens", 0) >= budget["maxTokens"]:
        return True
    if budget.get("maxCostUsd") is not None and usage.get("costUsd", 0.0) >= budget["maxCostUsd"]:
        return True
    return False
