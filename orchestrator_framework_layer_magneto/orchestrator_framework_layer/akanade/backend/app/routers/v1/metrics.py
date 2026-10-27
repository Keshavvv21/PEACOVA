"""§15 — observability's REST surface: the router's learned posterior per
agent (not just its LAST decision), the agent network's current
nodes/edges, and the adaptive reward weights' current values — "why is
routing shifting" answerable from outside the process, not just from a
single task's trace.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...auth import require_role
from ...framework import metrics_history
from ...framework.network import network
from ...framework.orchestrator import orchestrator
from ...framework.registry import registry
from ...llm.cost_optimizer import cost_optimizer
from ...llm.domain_router import domain_router
from ...llm.provider_optimizer import provider_optimizer

router = APIRouter(prefix="/api/v1/metrics", tags=["gmaofn-metrics"])


@router.get("")
def get_metrics(_: dict = Depends(require_role("Admin"))):
    agent_router = orchestrator.router
    posteriors = {}
    for agent in registry.list_agents(include_disabled=True):
        mean = agent_router._bandit.posterior_mean(agent.metadata.agent_id)
        posteriors[agent.metadata.agent_id] = (
            {"posterior_mean_weights": [round(float(x), 4) for x in mean]}
            if mean is not None else {"posterior_mean_weights": None, "note": "no data yet"})
    return {
        "agents": posteriors,
        "reward_weights": orchestrator.feedback_loop.weights.weights.snapshot(),
        "network": network.to_dict(),
    }


@router.get("/history")
def get_metrics_history(limit: int = 200, _: dict = Depends(require_role("Admin"))):
    """One row per FeedbackLoop.close()/submit_user_feedback() call —
    see metrics_history.py's own docstring for why this is the correct
    place to answer "is routing improving over time," which the
    point-in-time GET above genuinely cannot (it's always CURRENT state
    only, with no history behind it)."""
    return {"snapshots": metrics_history.history(limit=limit)}


@router.get("/routing")
def get_routing_metrics(_: dict = Depends(require_role("Admin"))):
    """Who's actually winning which kind of request today — the
    per-domain fine-tuned-SLM bandit (llm/domain_router.py, now also
    consulted on plain "auto" turns, not just an explicit "domain"
    pick) and the per-(tier, provider) ordering bandit
    (llm/provider_optimizer.py), both real-usage-trained and persisted
    across restarts. Same per-arm win-rate/posterior shape
    GET /api/v1/rag/metrics already established for retrieval."""
    return {
        "domain_routing": domain_router.stats(),
        "provider_routing": provider_optimizer.stats(),
    }


@router.get("/cost")
def get_cost_metrics(_: dict = Depends(require_role("Admin"))):
    """Real, tracked token usage, latency, and $ cost per (tier,
    provider) plus a whole-system savings estimate — the Token & Cost
    Optimizer layer's dashboard data source. See llm/cost_optimizer.py
    and routers/v1/cost.py (the pre-execution cost-predict endpoint)."""
    return {
        "per_provider": cost_optimizer.stats(),
        "summary": cost_optimizer.savings_summary(),
    }
