"""The v1 default strategy — CEO memo: "rules + cheapest eligible model
above threshold," made concrete. Deterministic on purpose: identical
input always produces identical output, no sampling, no randomness — an
operator can explain exactly why a given candidate was ranked first.

Hard eligibility (policy_gate, provider_budget, .available()) already
happened in Akanade BEFORE a candidate ever reaches this service — this
module only orders what's already eligible.
"""
from __future__ import annotations

from .models import Candidate, RouteContext

# A cheap model that fails still costs something (the failed attempt
# itself, plus whatever it takes to escalate and retry) — this is the
# CEO memo's own "first approximation": E[cost] = C_cheap + (1 -
# p_accept) * C_escalation. p_accept is approximated here by
# capability_tier (0..1) rather than a trained probability (see the
# plan's "hand-designed heuristic first" decision) — a model tagged 0.9
# is treated as ~90% likely to produce an acceptable answer for an
# average task at its declared min-capability bar.
_ESCALATION_COST_USD = 0.01  # a conservative flat estimate for "try again
                              # on the next candidate" — deliberately not
                              # modeled per-candidate yet; see plan Phase 1.


def _expected_cost(c: Candidate, est_tokens: int) -> float:
    direct_cost = c.cost_per_1k_tokens * (est_tokens / 1000.0)
    p_accept = max(0.0, min(1.0, c.capability_tier))
    return direct_cost + (1.0 - p_accept) * _ESCALATION_COST_USD


def order(candidates: list[Candidate], context: RouteContext) -> list[str]:
    """Returns candidate ids ordered cheapest-expected-cost-first among
    those meeting task_min_capability — never drops a candidate outright
    (an ineligible one is just pushed to the back), so ak-chatbot's own
    fallback walk still has every real option to try if the top pick's
    actual call fails."""
    eligible = [c for c in candidates if c.capability_tier >= context.task_min_capability]
    ineligible = [c for c in candidates if c.capability_tier < context.task_min_capability]
    eligible.sort(key=lambda c: _expected_cost(c, context.est_tokens))
    return [c.id for c in eligible] + [c.id for c in ineligible]
