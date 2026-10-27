"""Which pipeline answers real, live chat traffic — three admin-selectable
modes, not just a binary blue/green:
  - "akanade"          SECONDARY ONLY — RouteLLM never consulted at all.
                      Algo aKANADE's own TierClient chain, what
                      was live before this switch existed ("akanade" here
                      names the infrastructure/codebase only, never a
                      user-facing product name — the product is Algo
                      aKANADE).
  - "routellm"        PRIMARY + SECONDARY (default) — the CEO memo's own
                      generic architecture reference design (RouteLLM's
                      difficulty classifier, Default vs Escalation tier)
                      is preferred, with automatic failover to Algo
                      aKANADE's own chain treated as an expected,
                      routine part of this mode.
  - "routellm_only"   PRIMARY ONLY (intended) — same RouteLLM
                      preference and the SAME automatic-failover safety
                      net (a chat turn must never hard-fail — see
                      below), but a fallback to Algo aKANADE's chain in
                      this mode is logged as an unintended "leak" rather
                      than routine behavior, since the whole point of
                      selecting this mode is to measure RouteLLM's own,
                      unassisted coverage for the board/CEO comparison.
                      See router.py's own primary_pipeline_used vs
                      primary_pipeline_mode fields on TierClient — the
                      dashboard can compute a leak rate from the two.

Automatic failover is built in, not bolted on, for BOTH "routellm" modes:
resolve_routellm_preferred() below returns None whenever ak-router is
unreachable, the router service is disabled, or RouteLLM's pick has no
real registered provider on this deployment — TierClient.chat() treats
that exactly like "no preference," and falls straight through to Akanade's
own normal chain for that one call. A prompt never goes unanswered
because the primary side had a bad moment; it silently lands on green
instead of blue for that turn, same call, no user-visible failure — this
holds even in "routellm_only," per an explicit admin decision: a stricter
"let the turn fail" mode was considered and rejected, since it violates
the "a chat turn must never hard-fail" guarantee every other part of
this system relies on.

Same DB-backed admin-toggle pattern as router_service_config.py — a
single row, read fresh on every call (live on the next turn, no
redeploy), Executive-write/Admin-read via admin_router_config.py.

Applying RouteLLM's pick UNIFORMLY here, inside TierClient.chat()
itself (see llm/router.py's own call site), rather than only on one
specific endpoint, is the fix for a real gap the first live A/B test
surfaced: routing RouteLLM's pick through the general PM chat endpoint
only (the only one that accepted a provider override at the time) gave
it a structurally worse shot on coding prompts than Algo aKANADE's own
coding-agent path got — an apples-to-oranges comparison, not a real
capability gap. Wiring the preference in at TierClient itself means
every track (PM, coding, trading) gets the identical treatment,
regardless of which endpoint a given prompt happens to arrive through.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..db import store
from . import router_client

logger = logging.getLogger(__name__)

_PK = "PRIMARY_PIPELINE_CONFIG"
_SK = "SINGLETON"
_DEFAULT = "routellm"
_VALID_PIPELINES = ("akanade", "routellm", "routellm_only")
# Modes that consult RouteLLM at all — "akanade" alone skips the call
# entirely (see the module docstring's three-mode breakdown).
ROUTELLM_MODES = ("routellm", "routellm_only")

# RouteLLM's decision names (generic_architecture.py's DEFAULT_TIER_MODELS/
# ESCALATION_TIER_MODELS, on the ak-router side) don't exactly match
# this deployment's registered provider keys (llm/router.py's
# _providers()/EXTRA_LLM_PROVIDERS) — map to the real, callable
# provider. generic_architecture.py's own escalation tier is now pared down
# to exactly these three (Muse Spark 1.3 xHigh / Gemini 3.8 Flash High
# were dropped after the first live A/B test surfaced them as
# zero-answer picks — no real provider, or only a loose capability
# match), so every key below always resolves to a real provider today;
# kept as an explicit map (not just passthrough) so a future tier
# change that reintroduces an unmapped candidate fails safely (a
# missing dict key resolves to None via .get() in
# resolve_routellm_preferred() below, not a KeyError).
REGISTERED_MODEL_MAP: dict[str, str | None] = {
    "glm-5.3-flash": "glm-5.3-flash",
    "qwen3.8-flash": "qwen3.8-flash",
    "glm-5.3-full-max": "glm-5.3-full",
    "gpt-5.6-sol-xhigh": "sol-xhigh",
    "claude-opus-5-medium": "claude",
}


def get_pipeline() -> str:
    row = store("PrimaryPipelineConfig").get(_PK, _SK)
    return (row or {}).get("pipeline", _DEFAULT)


def set_pipeline(pipeline: str, updated_by: str) -> dict:
    if pipeline not in _VALID_PIPELINES:
        raise ValueError(f"unknown pipeline: {pipeline!r}")
    row = {"pipeline": pipeline, "updated_by": updated_by,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    store("PrimaryPipelineConfig").put(_PK, _SK, row)
    return row


def get_pipeline_row() -> dict:
    row = store("PrimaryPipelineConfig").get(_PK, _SK)
    return row or {"pipeline": _DEFAULT, "updated_by": None, "updated_at": None}


def resolve_routellm_preferred(prompt: str) -> str | None:
    """Called from TierClient.chat() only when get_pipeline() is one of
    ROUTELLM_MODES ("routellm" or "routellm_only"). Returns a real,
    registered provider key to prefer, or
    None (fall through to Algo aKANADE's own normal chain) when
    ak-router is unreachable/disabled (router_client.py's own circuit
    breaker keeps this cheap even then), or — should
    generic_architecture.py's tier lists ever grow a candidate this map
    doesn't know about — RouteLLM's pick has no real provider configured
    on this deployment. Never raises."""
    decision = router_client.call_generic_architecture_route(prompt)
    if not decision:
        return None
    model_name = decision.get("model_name")
    registered = REGISTERED_MODEL_MAP.get(model_name)
    if registered is None:
        logger.info("RouteLLM picked %r (tier=%s) — no registered provider, falling through to Akanade's own chain",
                   model_name, decision.get("tier"))
    return registered


# The real, deployed live-traffic behavior for both ROUTELLM_MODES — NOT
# RouteLLM's own per-prompt classifier pick (resolve_routellm_preferred()
# above still exists for logging/shadow comparison, but no longer decides
# the real call order). Found live, 2026-09-03: with no funded OpenAI key,
# RouteLLM's "mf" router falls back to "random" for every single decision
# — a coin flip, not a real difficulty judgment, so "escalate this prompt
# or not" was never actually informed by the prompt's own content. A real
# deterministic, cost-aware ACCEPTANCE cascade (the CEO memo's own
# original ask) replaces it: try the cheapest tier first, escalate ONLY
# when a rung genuinely fails (exception/timeout/unavailable — the SAME
# failure signal TierClient.chat()'s own chain-walk already uses for
# every other rung), never on a predicted-but-unverified difficulty
# score. Order matches the CEO's own tier diagram exactly: Default tier
# (GLM-5.3-Flash, then Qwen3.8-Flash) first, Escalation tier (GLM-5.3
# full, then Sol xHigh, then Opus 5) only once Default is exhausted —
# after this cascade, TierClient's own normal chain (ending in groq)
# still applies as the final, always-on safety net.
CASCADE_ORDER: tuple[str, ...] = tuple(
    v for v in REGISTERED_MODEL_MAP.values() if v is not None)


def resolve_routellm_cascade() -> list[str]:
    """The ordered candidate list for both ROUTELLM_MODES — spliced
    into TierClient.chat()'s own chain_keys (router.py), which already
    walks a list of provider keys and moves to the next on ANY failure.
    No network call to ak-router needed for the real routing decision
    itself anymore (see the module comment above) — this is a pure,
    static, always-available list, so it can never itself be a source
    of "ak-router is down" failover the way resolve_routellm_preferred()
    was."""
    return list(CASCADE_ORDER)
