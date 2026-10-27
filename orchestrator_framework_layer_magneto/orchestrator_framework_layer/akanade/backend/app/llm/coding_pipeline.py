"""The Secondary Pipeline — the VS Code coding agent's own model cascade.
The MODEL never used to be admin- or user-toggleable ("the user never
sees or picks a model" still holds — a caller can't name a specific
model), but WHICH TIERS are even in play now is: an admin can turn the
Escalation tier off entirely, same DB-backed, read-fresh-every-call
toggle pattern primary_pipeline.py's own pipeline switch uses (see
get_escalation_enabled/set_escalation_enabled below), gated at
Admin-or-above (routers/admin_router_config.py's coding-escalation
endpoints) rather than primary_pipeline's Executive-only bar — this is a
single scoped, per-agent cost lever, not the biggest switch in the admin
surface.

Default tier: GLM-5.3-Flash — routine/medium-difficulty/high-volume coding
turns, tried first on every call, no exception, regardless of the
escalation toggle below.

Escalation tier: GLM-5.3 Full -> Sol xHigh -> Opus 5 -> Groq -> the local
coding SLM (ollama-coder), tried in that exact sequential order ONLY when
a rung actually fails (a real exception/timeout/unavailable —
TierClient.chat()'s own per-rung failure handling, the SAME
failure-driven-not-predicted-difficulty philosophy primary_pipeline.py's
own module docstring already commits to and explains the reasoning for)
— AND only when an admin has switched escalation on (effective_cascade()
below). Off by default (_DEFAULT_ESCALATION_ENABLED): every coding turn
was previously eligible for this real-spend escalation with no admin
control over it at all, so a fresh/unconfigured deployment now stays on
Default-tier-only traffic until an admin opts in.

This cascade is EXCLUSIVE, not just a preference — see router.py's
`is_exclusive_cascade` at TierClient.chat()'s own explicit_cascade splice
site. Real gap found live: Groq and the local coding SLM used to be left
out of CODING_CASCADE_ORDER entirely on the theory that they were "already
the delay_ok tier's own normal chain tail, reached automatically" — true,
but that same normal chain tail ALSO contains max/luna/deepseek/hy4/
algo-engine, none of which this agent was ever supposed to reach ("every
OTHER provider should be off" for this track). Naming Groq/the local SLM
explicitly, in the exact position asked for, and making the whole cascade
exclusive, is what actually enforces that — not an accident of chain
ordering. If every named candidate here is genuinely unavailable, TierClient
still degrades to whatever else is configured rather than hard-failing
(see router.py's own comment on that one edge case) — the "everything
else is off" guarantee holds for the normal case, not as a way to make a
misconfigured deployment refuse to answer at all.
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..db import store

# OpenRouter-gatewayed hosted models (glm-5.3-flash/glm-5.3-full/sol-xhigh
# are configured via EXTRA_LLM_PROVIDERS on the deployment, base_url
# https://openrouter.ai/api/v1 or the model's own native API — an ops
# concern, invisible here); "claude" is Algo aKANADE's own AnthropicNative
# provider (model claude-opus-5, i.e. "Opus 5"); "groq"/"ollama-coder" are
# router.py's own built-in registry keys (the hosted Groq rung and the
# local code-specialized SLM on the same Ollama server, respectively).
# Provider keys, not display names — see llm/providers.py's registry.
CODING_CASCADE_ORDER: tuple[str, ...] = (
    "glm-5.3-flash",
    "glm-5.3-full",
    "sol-xhigh",
    "claude",
    "groq",
    "ollama-coder",
)

# CODING_CASCADE_ORDER[1:] — everything past the Default tier's own first
# (and only) rung. A plain slice, not a duplicated literal, so growing
# either tier in CODING_CASCADE_ORDER never needs a second edit here.
_ESCALATION_TIER: tuple[str, ...] = CODING_CASCADE_ORDER[1:]

_PK = "CODING_PIPELINE_CONFIG"
_SK = "SINGLETON"
_DEFAULT_ESCALATION_ENABLED = False


def get_escalation_enabled() -> bool:
    row = store("CodingPipelineConfig").get(_PK, _SK)
    return (row or {}).get("escalation_enabled", _DEFAULT_ESCALATION_ENABLED)


def set_escalation_enabled(enabled: bool, updated_by: str) -> dict:
    row = {"escalation_enabled": enabled, "updated_by": updated_by,
          "updated_at": datetime.now(timezone.utc).isoformat()}
    store("CodingPipelineConfig").put(_PK, _SK, row)
    return row


def get_escalation_config_row() -> dict:
    row = store("CodingPipelineConfig").get(_PK, _SK)
    return row or {"escalation_enabled": _DEFAULT_ESCALATION_ENABLED,
                   "updated_by": None, "updated_at": None}


def effective_cascade() -> list[str]:
    """What coding_agent.py's execute() actually splices into
    TierClient.chat() as `explicit_cascade` this call — just the Default
    tier while an admin has escalation switched off (today's default),
    or the full Default+Escalation cascade once they turn it on. Read
    fresh on every call (store() read, no caching) so a toggle in the
    admin UI takes effect on the very next coding turn — no redeploy, no
    process restart, same live-immediately convention every other
    DB-backed admin toggle in this codebase already follows."""
    if get_escalation_enabled():
        return list(CODING_CASCADE_ORDER)
    return [k for k in CODING_CASCADE_ORDER if k not in _ESCALATION_TIER]
