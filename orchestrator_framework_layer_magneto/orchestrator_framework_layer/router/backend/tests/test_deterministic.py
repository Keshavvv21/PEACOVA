from app.deterministic import order
from app.models import Candidate, RouteContext


def _candidates():
    return [
        Candidate(id="luna", cost_per_1k_tokens=0.02, capability_tier=0.9),
        Candidate(id="groq", cost_per_1k_tokens=0.001, capability_tier=0.7),
        Candidate(id="algo-akanade-slm", cost_per_1k_tokens=0.0, capability_tier=0.5),
    ]


def test_deterministic_order_is_repeatable():
    context = RouteContext(tier="realtime", est_tokens=500, task_min_capability=0.0)
    first = order(_candidates(), context)
    second = order(_candidates(), context)
    assert first == second, "identical input must produce identical order — no randomness"


def test_cheapest_eligible_wins():
    context = RouteContext(tier="realtime", est_tokens=500, task_min_capability=0.0)
    result = order(_candidates(), context)
    # groq/algo-akanade-slm are far cheaper than luna at this token count —
    # one of them should lead, never the expensive rung.
    assert result[0] in ("groq", "algo-akanade-slm")


def test_ineligible_candidate_pushed_back_not_dropped():
    context = RouteContext(tier="realtime", est_tokens=500, task_min_capability=0.8)
    result = order(_candidates(), context)
    # Only "luna" (0.9) meets the 0.8 bar — it must still lead, and every
    # candidate must still be present (fallback walk needs them all).
    assert result[0] == "luna"
    assert set(result) == {"luna", "groq", "algo-akanade-slm"}
