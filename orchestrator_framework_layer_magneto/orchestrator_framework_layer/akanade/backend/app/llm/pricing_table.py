"""Real, best-effort $/1k-token pricing for every provider this codebase
can route to — the one piece llm/cost_optimizer.py and
llm/provider_budget.py both need and neither had on its own.
provider_budget.py's own docstring is explicit that "there's no
universal live pricing feed for every possible provider," so its cost
tracking only ever activates once an admin manually sets a rate for
that specific provider. This table is the DEFAULT for a provider an
admin never configured — checked SECOND, after provider_budget's own
admin-supplied rate, so a real operator-entered bill rate always wins
over this file's best-effort public list-price estimate.

Rates are approximate, public list prices as of this file's last
update (2026-08) for the model class each tier's hosted rung is wired
to — a provider changing its pricing, or a specific account's
negotiated/volume rate, will drift from these numbers over time. This
is good enough for RELATIVE cost-aware ranking ("which of these
candidates is cheaper right now") and directional cost/savings
tracking, not a claim of billing-accurate reconciliation — set a real
rate via llm/provider_budget.py's set_budget() for anything that needs
to match an actual invoice.
"""
from __future__ import annotations

from . import provider_budget

# provider registry key -> (cost per 1k prompt tokens USD, cost per 1k
# completion tokens USD). Public list prices, best-effort — see module
# docstring. Ollama and the deterministic "rule" floor are exactly
# $0.00: no metered API call happens for either.
_DEFAULT_RATES_USD_PER_1K: dict[str, tuple[float, float]] = {
    "ollama": (0.0, 0.0),
    "rule": (0.0, 0.0),
    "groq": (0.00059, 0.00079),  # Llama-3.3-70B-class pricing on Groq's paid API
    "deepseek": (0.00027, 0.0011),  # DeepSeek-V3-class API pricing
    "luna": (0.0025, 0.01),  # this deployment's GPT-4o-class hosted tier
    "max": (0.003, 0.015),  # this deployment's Claude-Sonnet-class hosted tier
    "algo-engine": (0.0005, 0.0015),  # internal/self-hosted engine — nominal placeholder rate
}
# An unlisted or admin-added provider with no configured rate at all —
# a conservative mid-range guess so it's never silently priced as free.
_FALLBACK_RATE_USD_PER_1K = (0.002, 0.008)


def estimate_tokens(text: str) -> int:
    """No real tokenizer at this layer either (same reasoning as
    llm/token_optimizer.py's own _estimate_tokens) — chars/4 is the
    common rough ratio for English text, good enough for a BEFORE-the-
    call estimate and for directional cost comparison across providers,
    not exact tokenization."""
    return max(1, len(text) // 4)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    return estimate_tokens(" ".join(str(m.get("content", "")) for m in messages))


def rates_for(provider_key: str) -> tuple[float, float]:
    """(cost_per_1k_prompt, cost_per_1k_completion) for `provider_key`.
    An admin-configured budget rate (llm/provider_budget.py) always wins
    when set; falls back to this table's best-effort default, then to a
    conservative fallback rate for a provider neither knows about."""
    default_prompt, default_completion = _DEFAULT_RATES_USD_PER_1K.get(
        provider_key, _FALLBACK_RATE_USD_PER_1K)
    budget = provider_budget.get_budget(provider_key)
    if not budget:
        return (default_prompt, default_completion)
    prompt_rate = budget.get("costPer1kPromptTokens")
    completion_rate = budget.get("costPer1kCompletionTokens")
    return (prompt_rate if prompt_rate is not None else default_prompt,
           completion_rate if completion_rate is not None else default_completion)


def estimate_cost_usd(provider_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = rates_for(provider_key)
    return prompt_tokens / 1000 * prompt_rate + completion_tokens / 1000 * completion_rate


def max_known_rates() -> tuple[float, float]:
    """The priciest (prompt, completion) rate among every provider this
    table knows about — the "what if this had gone to the most
    expensive real alternative instead" baseline every savings
    calculation in llm/cost_optimizer.py is built from (whole-system
    savings_summary() and the per-call savings shown after each model
    call alike), so both use the exact same baseline."""
    all_known_rates = list(_DEFAULT_RATES_USD_PER_1K.values())
    max_prompt_rate = max((p for p, _ in all_known_rates), default=0.0)
    max_completion_rate = max((c for _, c in all_known_rates), default=0.0)
    return (max_prompt_rate, max_completion_rate)


def counterfactual_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """What `prompt_tokens`/`completion_tokens` would have cost at the
    priciest rate this table knows about — the shared baseline for
    every savings number in this layer."""
    max_prompt_rate, max_completion_rate = max_known_rates()
    return prompt_tokens / 1000 * max_prompt_rate + completion_tokens / 1000 * max_completion_rate
