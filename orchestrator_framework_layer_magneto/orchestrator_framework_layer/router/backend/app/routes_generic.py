"""Endpoints for the "GENERIC ARCHITECTURE" comparison arm
(generic_architecture.py) — kept in a separate router file from
routes.py's Algo aKANADE endpoints so it's obvious at a glance that
nothing here is live-traffic routing.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from . import generic_architecture
from .db import store

router = APIRouter(prefix="/generic-architecture")


class GenericRouteRequest(BaseModel):
    # The real prompt text — RouteLLM's classifier scores THIS to decide
    # default vs escalation tier. Empty (the default) still works, just
    # reflects the router's own cold-start prior rather than a real
    # difficulty judgment.
    prompt: str = ""


class GenericPick(BaseModel):
    model_name: str
    tier: str


class GenericOutcome(BaseModel):
    model_name: str
    tier: str
    ok: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    quality_score: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    task_family: str = "general"


@router.post("/route", response_model=GenericPick)
def generic_route(payload: GenericRouteRequest = GenericRouteRequest()) -> GenericPick:
    return GenericPick(**generic_architecture.pick(payload.prompt))


@router.post("/escalate", response_model=GenericPick)
def generic_escalate(payload: GenericRouteRequest = GenericRouteRequest()) -> GenericPick:
    return GenericPick(**generic_architecture.escalate(payload.prompt))


@router.post("/outcome", status_code=204)
def generic_outcome(payload: GenericOutcome) -> None:
    """Writes into the SAME ModelBenchmarks table Phase 5's eval-harness
    runs write into (see the plan's model_source tagging) — "candidate"
    here, "ours" for Algo aKANADE's own SLMs/models — so the Model
    Evaluation dashboard's three-way comparison (Generic Architecture
    Models / Algo aKANADE SLMs / All Models) reads one table, not two."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": payload.model_name,
        "modelSource": "candidate",
        "tier": payload.tier,
        "taskFamily": payload.task_family,
        "ok": payload.ok,
        "latencyMs": payload.latency_ms,
        "costUsd": payload.cost_usd,
        "qualityScore": payload.quality_score,
        "promptTokens": payload.prompt_tokens,
        "completionTokens": payload.completion_tokens,
    }
    store("ModelBenchmarks").put(f"BENCHMARK#{uuid.uuid4().hex[:12]}", "META", row)
