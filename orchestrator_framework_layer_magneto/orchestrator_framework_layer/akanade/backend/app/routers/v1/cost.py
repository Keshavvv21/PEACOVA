"""Predict-before-you-spend — the one piece of the Token & Cost
Optimizer layer genuinely worth its own endpoint rather than folding
into routers/v1/metrics.py's Admin-only observability surface: any
authenticated caller (a UI, a script, the chat widget itself before it
sends a message) can ask "what would this cost, on which model" BEFORE
committing to a real LLM call. Real token estimate, real (or admin-
configured) $/1k rates — see llm/cost_optimizer.py and
llm/pricing_table.py for exactly how; this router is just the REST
surface over both.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...auth import current_user
from ...llm.cost_optimizer import cost_optimizer
from ...llm.router import Tier, available_provider_keys_for_tier

router = APIRouter(prefix="/api/v1/cost", tags=["cost-optimizer"])


class CostPredictRequest(BaseModel):
    message: str
    tier: Tier = "realtime"


@router.post("/predict")
def predict_cost(payload: CostPredictRequest, _: dict = Depends(current_user)):
    """Estimates this message's token count and predicted $ cost against
    every provider actually available for `tier` right now — the exact
    candidate set TierClient.chat() itself would consider, so this
    preview never shows a provider that isn't a real option this turn.
    Cheapest-first, matching the ordering cost_optimizer.order() applies
    to a real call."""
    candidates = available_provider_keys_for_tier(payload.tier)
    estimates = cost_optimizer.estimate([{"role": "user", "content": payload.message}], candidates)
    estimates.sort(key=lambda e: e.predicted_cost_usd)
    return {"tier": payload.tier, "estimates": [asdict(e) for e in estimates]}
