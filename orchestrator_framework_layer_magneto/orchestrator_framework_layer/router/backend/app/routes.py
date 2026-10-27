"""ak-router's endpoints. Deliberately no auth here — this service is
never exposed publicly (only reachable on the internal ak-net Docker
network, same trust boundary as, say, ak-dynamo). Admin role-gating for
the strategy toggle lives on ak-chatbot's side (a small proxy endpoint
there, using Akanade's existing require_role — see the plan's write-up);
this service just does what it's told by a trusted caller.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from . import bert_shadow, deterministic, thompson
from .db import store
from .models import (BertShadowConfigIn, BertShadowConfigOut, OutcomeRequest, RouteContext,
                     RouteRequest, RouteResponse, RouterConfigIn, RouterConfigOut)

router = APIRouter()


@router.post("/route", response_model=RouteResponse)
def route(payload: RouteRequest) -> RouteResponse:
    strategy = thompson.get_strategy()
    deterministic_order = deterministic.order(payload.candidates, payload.context)

    # The shadow policy runs on EVERY call regardless of the live
    # strategy — this is how RoutingOutcomes accumulates "what TS would
    # have done" evidence even while deterministic is live. See plan
    # Phase 2 step 9.
    shadow_pick = thompson.thompson_policy.pick(payload.candidates, payload.context)

    # Second shadow-only comparison signal, same posterior as shadow_pick
    # above but a UCB arm-selection rule instead of a posterior sample —
    # see thompson.py's pick_ucb(). Never affects ordered_ids/strategy_used.
    ucb_pick = thompson.thompson_policy.pick_ucb(payload.candidates, payload.context)

    # Same shadow-mode-first shape as TS above, OFF by default (see
    # bert_shadow.is_enabled()) — a real cost (a BERT forward pass) so
    # this only runs at all once an admin has explicitly opted in via
    # PATCH /admin/bert-shadow, and even then never touches ordered_ids.
    bert_hint = bert_shadow.hint(payload.context.prompt)

    if strategy == "thompson_sampling" and shadow_pick is not None:
        # Live TS: its own pick goes first, the deterministic order fills
        # in the rest (including the OTHER two-arm candidate, if any, and
        # every non-arm candidate) — ak-chatbot's fallback walk still has
        # every real option behind it if the top pick's call fails.
        ordered = [shadow_pick] + [i for i in deterministic_order if i != shadow_pick]
        return RouteResponse(ordered_ids=ordered, strategy_used="thompson_sampling",
                             shadow_pick=shadow_pick, bert_hint=bert_hint, ucb_pick=ucb_pick)

    return RouteResponse(ordered_ids=deterministic_order, strategy_used="deterministic",
                         shadow_pick=shadow_pick, bert_hint=bert_hint, ucb_pick=ucb_pick)


@router.post("/outcome", status_code=204)
def outcome(payload: OutcomeRequest) -> None:
    # Only the two designated arms feed the live bandit's posterior —
    # see thompson.update()'s own guard (a no-op for any other
    # candidate_id). Every outcome, regardless of strategy, still gets
    # logged to RoutingOutcomes below so the comparison dashboard has a
    # complete picture, not just TS-arm calls.
    thompson.thompson_policy.update(
        payload.candidate_id, RouteContext(tier=payload.tier),
        ok=payload.ok, latency_ms=payload.latency_ms)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": payload.tier,
        "chosen": payload.candidate_id,
        "strategyUsed": payload.strategy_used,
        "shadowPick": payload.shadow_pick,
        "bertHint": payload.bert_hint,
        "ucbPick": payload.ucb_pick,
        "ok": payload.ok,
        "costUsd": payload.cost_usd,
        "latencyMs": payload.latency_ms,
    }
    store("RoutingOutcomes").put(f"OUTCOME#{uuid.uuid4().hex[:12]}", "META", row)


@router.get("/admin/routing-outcomes")
def list_routing_outcomes(since: str | None = None, until: str | None = None,
                          limit: int = 2000) -> list[dict]:
    """Raw feed for the Routing Comparison dashboard (see plan): every
    logged request's actual outcome plus its Thompson/UCB shadow picks.
    No auth here — same internal-ak-net-only trust boundary as every other
    route in this file; RBAC happens on ak-chatbot's proxy side."""
    rows = store("RoutingOutcomes").scan(pk_prefix="OUTCOME#")
    if since:
        rows = [r for r in rows if r.get("timestamp", "") >= since]
    if until:
        rows = [r for r in rows if r.get("timestamp", "") <= until]
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows[:limit]


@router.get("/admin/strategy", response_model=RouterConfigOut)
def get_strategy() -> RouterConfigOut:
    return RouterConfigOut(**thompson.get_strategy_row())


@router.patch("/admin/strategy", response_model=RouterConfigOut)
def patch_strategy(payload: RouterConfigIn) -> RouterConfigOut:
    row = thompson.set_strategy(payload.strategy, payload.updated_by)
    return RouterConfigOut(**row)


@router.get("/admin/bert-shadow", response_model=BertShadowConfigOut)
def get_bert_shadow() -> BertShadowConfigOut:
    return BertShadowConfigOut(**bert_shadow.get_row())


@router.patch("/admin/bert-shadow", response_model=BertShadowConfigOut)
def patch_bert_shadow(payload: BertShadowConfigIn) -> BertShadowConfigOut:
    row = bert_shadow.set_enabled(payload.enabled, payload.updated_by)
    return BertShadowConfigOut(**row)
