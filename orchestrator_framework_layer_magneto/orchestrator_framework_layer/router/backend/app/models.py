"""Request/response shapes for ak-router's /route and /outcome — the
ONLY contract ak-chatbot's TierClient.chat() needs to know about. No
session, no auth, no provider credentials cross this boundary; see the
plan's "Router-service is stateless about Akanade's own session/policy
state."
"""
from __future__ import annotations

from pydantic import BaseModel, Field

Strategy = str  # "deterministic" | "thompson_sampling"


class Candidate(BaseModel):
    id: str  # registry key — "groq", "algo-akanade-slm", etc., matched
              # against Akanade's own provs dict, never a display name.
    cost_per_1k_tokens: float = 0.0
    p95_latency_ms: float = 2000.0
    capability_tier: float = 0.5  # 0..1, see admin_providers.py's new field


class RouteContext(BaseModel):
    tier: str = "realtime"  # "realtime" | "delay_ok" | "bulk"
    est_tokens: int = 500
    task_min_capability: float = 0.0
    # The actual latest user message text — OPTIONAL, and the one field
    # here that's genuinely new: every other field on this model is
    # lightweight metadata, deliberately never the prompt itself (see
    # this file's own docstring on why almost nothing crosses this
    # boundary). Only populated by the caller when the admin-opt-in
    # BERT difficulty shadow signal is enabled (see bert_shadow.py) —
    # unset, this endpoint behaves exactly as it always has. Still only
    # ever reaches this service over the internal-only ak-net network,
    # the same trust boundary ak-dynamo (which already stores this
    # exact content permanently) sits on.
    prompt: str | None = None


class RouteRequest(BaseModel):
    candidates: list[Candidate]
    context: RouteContext


class RouteResponse(BaseModel):
    ordered_ids: list[str]
    strategy_used: Strategy
    # What the shadow Thompson-Sampling policy would have picked, even
    # while it isn't the live decision — see RoutingOutcomes' shadow_pick
    # column and the plan's Phase 2 step 9.
    shadow_pick: str | None = None
    # RouteLLM's "bert" classifier's own tier verdict for this prompt
    # ("default"/"escalation"), or None if the shadow signal is off,
    # "bert" isn't available, or no `context.prompt` was given — same
    # shadow-only shape as shadow_pick: never affects ordered_ids on
    # its own. See bert_shadow.py.
    bert_hint: str | None = None
    # What the SAME two-arm posterior's UCB rule would have picked —
    # comparison-only, same shadow shape as shadow_pick. See
    # thompson.py's pick_ucb()/bandit.py's select_ucb().
    ucb_pick: str | None = None


class OutcomeRequest(BaseModel):
    candidate_id: str
    ok: bool
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    strategy_used: Strategy
    shadow_pick: str | None = None
    bert_hint: str | None = None
    ucb_pick: str | None = None
    tier: str = "realtime"
    context_vector: list[float] = Field(default_factory=list)


class RouterConfigOut(BaseModel):
    strategy: Strategy
    updated_by: str | None = None
    updated_at: str | None = None


class RouterConfigIn(BaseModel):
    strategy: Strategy
    updated_by: str


class BertShadowConfigOut(BaseModel):
    enabled: bool
    updated_by: str | None = None
    updated_at: str | None = None


class BertShadowConfigIn(BaseModel):
    enabled: bool
    updated_by: str
