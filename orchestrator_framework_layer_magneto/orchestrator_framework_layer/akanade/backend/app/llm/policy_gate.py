"""Policy gate — the piece "policy always overrides the classifier" from
the AA internal-routing architecture note names explicitly, and ALGO AKANADE
didn't have as a first-class layer before this: a check that runs BEFORE
any per-call routing preference (Hybrid mode's local/hosted bandit pick
in llm/hybrid_router.py, an explicit `preferred=` override, provider_
optimizer.py's learned reordering) and can override all of them
outright. Wired into the ONE choke point every LLM call already passes
through — TierClient.chat() (llm/router.py) — so every existing caller
gets this automatically, with zero call-site changes, including agents
that never ran llm/guardrails.py's input_guard() at all (CodingAgent,
CodeSecurityAgent's own LLM-phrasing call, SupplyChainAgent — none of
these are reachable through native_agent.py's chat turn where input_
guard() already runs).

Three concrete, testable rules — deliberately NOT a general data-
classification taxonomy or an ML classifier, same "targeted checks, not
NeMo-style dialog rails" scope decision guardrails.py's own docstring
already makes for this codebase:

  1. Prompt-injection block — reuses guardrails.looks_like_injection()
     verbatim (not a second, divergent pattern list). Pure defense in
     depth for callers that never ran native_agent.py's input_guard()
     — that path still gets first crack and usually catches this
     before a call ever reaches TierClient at all.

  2. Confidentiality — a message whose content looks like it contains a
     real secret (AWS access key, PEM private key block, a JWT, or a
     common vendor secret-token prefix) is routed local-only: every
     provider registry key backed by hardware THIS deployment controls
     (LOCAL_PROVIDER_KEYS below), never a third-party hosted API. If no
     local rung is available the call still proceeds through whatever's
     left (graceful degradation, same philosophy provider_budget.py's
     own exhausted-budget handling already uses — this is a data-
     exfiltration guard, not an uptime guard), but the decision is
     still returned so a caller that wants to refuse outright in that
     case can.

  3. Destructive-intent — wording matching the SAME verb vocabulary
     agent/native_agent.py's own confirm_prompt_for() already keys its
     destructive-action confirmation prompts on (delete/remove/archive/
     cancel/revoke/deactivate/...) is treated as high-stakes: routing
     prefers a HOSTED rung over a local one for this call, overriding
     whatever Hybrid mode's bandit or an explicit local `preferred` pick
     would otherwise do. This is the note's own principle made concrete
     — "false routing to Flash is the expensive error" is truest
     exactly when the request can change real data.

None of this replaces the RBAC/confirmation gate that decides WHETHER a
destructive action actually executes (native_agent.py's
requires_confirmation(), or a coding-agent tool's own confirmation
round-trip) — this only ever decides WHICH MODEL answers, never whether
the action runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .guardrails import looks_like_injection

Action = Literal["allow", "force_local", "force_hosted", "block"]

# Provider registry keys (llm/router.py's `provs` dict) that run on
# hardware this deployment controls — never a third-party API call, so
# rule 2's confidentiality routing can send to any of these freely.
# "rule" (the deterministic floor) is neither local nor hosted in any
# meaningful sense but is always safe to route to either way, so it's
# left out of both this set and hosted-preference reordering.
LOCAL_PROVIDER_KEYS = frozenset({"ollama", "ollama-coder", "algo-engine"})

_SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                                     # AWS access key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),                   # PEM private key
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT shape
    re.compile(r"\b(?:sk|xox[baprs]|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"),  # common vendor secret prefixes
]

_DESTRUCTIVE_WORDS = re.compile(
    r"\b(delete|remove|archive|cancel|permanently|revoke|deactivate|"
    r"drop|destroy|wipe|purge)\b", re.I)


@dataclass
class PolicyDecision:
    action: Action
    reason: str = ""


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def _all_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))


def evaluate(messages: list[dict]) -> PolicyDecision:
    """The single entry point TierClient.chat() calls on every turn.
    Order matters: injection block first (cheapest, highest-severity),
    then confidentiality (a secret in ANY message, not just the latest
    user turn — it could be echoed back from an earlier tool result),
    then destructive-intent (only the latest user turn — that's the
    actual request being routed right now)."""
    last_user = _last_user_text(messages)
    if last_user and looks_like_injection(last_user):
        return PolicyDecision("block", "prompt-injection pattern matched")

    full_text = _all_text(messages)
    for pat in _SECRET_PATTERNS:
        if pat.search(full_text):
            return PolicyDecision(
                "force_local",
                "message content looks like it contains a real secret — "
                "kept on local-only providers, never sent to a hosted API")

    if last_user and _DESTRUCTIVE_WORDS.search(last_user):
        return PolicyDecision(
            "force_hosted",
            "destructive-intent wording detected — routing prefers the "
            "highest-quality tier for this call")

    return PolicyDecision("allow")
