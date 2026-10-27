"""RL layer for the security gate — continuous learning on top of
security_guard.py's deterministic rules, the same "reuse LinearBanditCore
for a new per-call decision, reward deferred until the real outcome is
known" pattern as llm/token_optimizer.py, framework/rag_optimizer.py, and
llm/optimization_router.py.

Scope, deliberately narrow and safety-first — read this before touching
either function below:

- This NEVER decides whether a rule fires, and NEVER touches the 🔴 RISK
  tier at all. security_guard.py's `_RULES` are the SOLE, deterministic
  authority on that — RISK still always blocks unconditionally, and a
  NEEDS_REVIEW finding is never silently promoted or demoted to a
  different flag by this module. Same "RL augments, never gates, never
  substitutes for the real check" boundary rag_optimizer.py's own
  docstring draws for retrieval quality.
- What it DOES learn: for each 🟡 NEEDS_REVIEW rule, split by which
  SURFACE it fired on (source="user_prompt"/"email_content"/
  "calendar_event"/"chat_message"/"tool_response"/"generated_reply" —
  security_guard.classify()'s own `source` values) — arm_id =
  "{rule_id}::{source}" — a running estimate of how often THAT SPECIFIC
  rule, on THAT SPECIFIC surface, turned out to matter versus turned out
  to be a nuisance the user waved through. Surfaced as an optional,
  additive note (confidence_note()) alongside the flag — the flag itself
  is untouched.
- Reward signal is REAL, not synthetic: the only thing this learns from
  is an actual user confirmation outcome on a write action whose
  confirmation prompt carried NEEDS_REVIEW findings — see
  native_agent.record_security_feedback(), called from every track's
  yes/no confirmation handling, the SAME real signal every other
  optimizer in this codebase already defers to (token_optimizer's
  retried/out_of_scope, rag_optimizer's task success). A flagged reply
  with no confirmation step (a plain read/answer) produces no signal —
  there's nothing to learn from silence, and this module doesn't invent
  one rather than leave the gap honest.
"""
from __future__ import annotations

import re
import threading

from .framework import bandit_persistence
from .framework.bandit import LinearBanditCore

# One constant-context arm per (rule_id, source) — there's no per-call
# feature to condition on yet (unlike rag_optimizer's query-length
# signal), just "has this specific pattern, on this specific surface,
# tended to be real." The bias term IS the whole learned estimate.
_FEATURE_DIM = 1
_CONTEXT = [1.0]

_RULE_ID_RE = re.compile(r"^(SEC-[A-Z-]+):")


def _rule_id(finding: str) -> str | None:
    m = _RULE_ID_RE.match(finding)
    return m.group(1) if m else None


class SecurityOptimizer:
    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.3,
                name: str | None = None):
        """`name` opts into durable persistence (bandit_persistence.py —
        app/db.py's store(), the SAME PK/SK document store
        OAuthConnections/ChatSessions/Providers already use): learned
        state loads on construction and saves after every
        record_feedback, so what this has learned survives a restart
        instead of resetting to a blank prior. None (the default) is a
        pure in-memory instance — what every test in
        test_security_optimizer.py uses, so each test starts from a
        known-blank slate rather than whatever the shared singleton
        below has learned."""
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._lock = threading.Lock()
        self._name = name
        if name:
            bandit_persistence.load(name, self._core)

    def confidence_note(self, rule_id: str, source: str) -> str | None:
        """A short, human-readable note IF this (rule, source) pair has
        been through at least one real confirmation outcome — None for a
        brand-new pair (no learned opinion yet) or a middling one (not
        enough signal either way), never a misleadingly confident guess
        from zero data."""
        arm_id = f"{rule_id}::{source}"
        with self._lock:
            mean = self._core.posterior_mean(arm_id)
        if mean is None:
            return None
        score = float(mean[-1])  # the bias term, since _CONTEXT is constant
        # +1.0/-1.0 rewards (see record_feedback) converge asymptotically
        # TOWARD their reward value but never reach it exactly (standard
        # Bayesian-linear-regression shrinkage) — 0.3 in magnitude is
        # comfortably crossed within single-digit real confirmations
        # (empirically ~0.33 after just one "declined" outcome, ~0.47
        # after eight) while still requiring more than a single data
        # point's worth of noise to flip a fresh arm either way.
        if score > 0.3:
            return "this pattern has usually been confirmed as a real concern"
        if score < -0.3:
            return "this pattern has usually turned out fine when reviewed"
        return None

    def record_feedback(self, findings: list[str], *, source: str, approved: bool) -> None:
        """Called once a write action flagged 🟡 NEEDS_REVIEW at
        confirmation time gets its real yes/no answer. Symmetric reward
        magnitude for both outcomes (deliberately simple, not weighting
        "declined" more heavily than "approved despite the flag" — either
        real outcome is treated as equally informative): `approved=False`
        (user declined specifically after seeing the flagged action) is
        +1.0, evidence the rule(s) that fired actually mattered;
        `approved=True` (proceeded anyway) is -1.0, evidence they were a
        nuisance for this case. Never called for a RISK-flagged turn —
        RISK never reaches a confirmation prompt to begin with
        (native_agent.security_precheck blocks it outright, earlier)."""
        reward = -1.0 if approved else 1.0
        with self._lock:
            for finding in findings:
                rid = _rule_id(finding)
                if rid:
                    self._core.update(f"{rid}::{source}", _CONTEXT, reward)
            if self._name:
                bandit_persistence.save(self._name, self._core)


# Process-wide singleton, same lifecycle as token_optimizer/rag_optimizer/
# optimization_router's module-level instances — `name=` opts it into
# durable persistence (see __init__'s docstring): this is the ONE real,
# wired-up example of the pattern bandit_persistence.py documents as
# reusable for the OTHER bandit-backed optimizers in this codebase.
security_optimizer = SecurityOptimizer(name="security_optimizer")
