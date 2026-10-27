"""Hybrid mode — an RL-selected top-level choice: is this message
complicated enough to warrant a hosted "outer" model, or simple enough
for the free local SLM to handle well? Sits ABOVE llm/router.py's
TierClient fallback chains — when Hybrid mode is selected
(ChatMessageIn.provider == "hybrid"), this decides which provider should
be first-preferred for THIS specific message, then hands off to the
exact same TierClient.preferred mechanism the "pick one specific model"
feature already uses — a bad pick still degrades gracefully through the
normal fallback chain right behind it, never a hard failure.

Two arms: "local" (prefer the actual Ollama rung — free, fast for
simple asks) and "hosted" (no override — let the tier's normal chain,
which already tries hosted rungs before Ollama, decide). Context is a
small, cheap, non-LLM complexity estimate (message length, clause
count, whether the tier's own cheap classifier already called this
delay_ok) — same "no extra model round trip" philosophy as
token_optimizer.py/rag_optimizer.py's heuristics.

Reward is asymmetric on purpose: a local success is the best outcome
(free AND it worked) and scores higher than a hosted success (worked,
but cost money); a local failure (the message needed a retry or went
out of scope) is penalized harder than a hosted failure, since routing
something local couldn't actually handle wastes a whole turn that a
hosted model would likely have gotten right the first time.

Escalation-on-slowness: the "local" arm is passed to llm/router.py's
TierClient as `preferred_timeout=LOCAL_TIMEOUT_SECONDS` (10s), NOT the
provider's own longer default — if the SLM hasn't answered within that
window, TierClient's existing fallback-on-exception logic (same path a
network error already takes) moves straight to a hosted rung, so the
user is never stuck waiting out a genuinely slow local answer just
because the bandit guessed "local" was worth trying. When that happens,
the hosted model's answer is logged as an extra nightly-fine-tune
example (finetune_log.log_hybrid_escalation_turn(), called from each
track's finish() when TierClient.preferred_escalated is True) — a real
worked example of what the SLM should have been able to answer itself,
distilled from the model that actually did.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from ..framework.bandit import LinearBanditCore
from .router import available_providers

_FEATURE_DIM = 3  # [word_count/50 clamped, clause_count/4 clamped, tier_is_delay_ok]

_LOCAL_PROVIDER_KEY = "ollama"

# How long the local SLM gets before Hybrid mode gives up and escalates to
# a hosted rung this turn — see module docstring's "Escalation-on-slowness".
LOCAL_TIMEOUT_SECONDS = 10.0


def _complexity_features(message: str, tier: str) -> list[float]:
    words = message.split()
    clauses = len(re.split(r"[,;]| and | or ", message))
    return [
        min(len(words) / 50.0, 1.0),
        min(clauses / 4.0, 1.0),
        1.0 if tier == "delay_ok" else 0.0,
    ]


def _local_provider_available(local_provider: str) -> bool:
    return any(p["id"] == local_provider and p["available"] for p in available_providers())


@dataclass
class HybridSelection:
    arm: str  # "local" | "hosted"
    context: list[float]


class HybridRouter:
    """One LinearBanditCore, two arms — "local" vs "hosted".

    exploration_scale is deliberately LOWER than this codebase's other
    bandits (domain_router/cost_optimizer/provider_optimizer all use
    0.4) — real, repeated, consistent measurement on this deployment
    found local (CPU-only Ollama) taking 8-100+ seconds for ordinary
    tool-calling turns, next to 1-2s hosted answers, EVERY time it was
    tried; that isn't a subtle, close-call tradeoff worth heavy ongoing
    exploration for, it's a severe and consistent latency cliff. Less
    exploration means the bandit converges on — and stays on — whichever
    arm real evidence favors, instead of re-sampling a demonstrably
    10-50x-slower option indefinitely just because Thompson Sampling
    never fully stops exploring by design."""

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.15):
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._lock = threading.Lock()

    def choose(self, message: str, tier: str, local_provider: str = _LOCAL_PROVIDER_KEY) -> HybridSelection:
        context = _complexity_features(message, tier)
        if not _local_provider_available(local_provider):
            return HybridSelection(arm="hosted", context=context)
        with self._lock:
            # A real cold-start seed against "local" — this bandit's own
            # persistence-free state (see class docstring: no
            # bandit_persistence here, so every process restart is a
            # genuine cold start) means a fresh process would otherwise
            # pick between "local" and "hosted" as a near coin-flip
            # until real outcomes accumulate. -2.0 (not a token nudge)
            # because the dot product this seed feeds into is divided
            # across _FEATURE_DIM+1 dimensions (see LinearBanditCore.
            # ensure()) — a small seed_utility barely moves the actual
            # sampled score at all; this value was chosen empirically to
            # produce a real, measurable cold-start preference for
            # "hosted" given the severity confirmed live (local took
            # 8-100+ seconds repeatedly, next to 1-2s hosted answers).
            # `ensure()`'s own docstring is explicit this only sets the
            # posterior MEAN while leaving it at its WIDEST — the very
            # first real "local" outcome already starts overriding it,
            # never a hard-coded preference that can't be beaten by
            # actual evidence.
            winner, _ = self._core.select(["local", "hosted"], context,
                                          seed_utilities={"local": -2.0})
        return HybridSelection(arm=winner, context=context)

    def preferred_provider_for(self, selection: HybridSelection, local_provider: str = _LOCAL_PROVIDER_KEY) -> str | None:
        """None means "don't override" — let the tier's normal chain
        (which always tries hosted rungs before Ollama) decide.
        `local_provider` lets a different call site's own "local" rung
        stand in for plain "ollama" — CodingAgent's is "ollama-coder"
        (a real, code-specialized model on the same Ollama server), not
        the generic one this module defaults to for the PM/Trading
        chat pipeline it was originally written for."""
        return local_provider if selection.arm == "local" else None

    def record_outcome(self, selection: HybridSelection, *, success: bool,
                       latency_ms: float = 0.0) -> None:
        if selection.arm == "local":
            reward = 1.0 if success else -0.6
        else:
            reward = 0.5 if success else -0.3
        # Real gap found live: this had NO latency term at all, so a
        # "local" success always outscored a "hosted" success by a flat
        # 0.5 regardless of how slow it actually was — on THIS
        # deployment (CPU-only Ollama), an ordinary tool-calling turn
        # measured at 20-30+ seconds, next to a hosted answer in 1-5s.
        # That flat bias taught the bandit to keep preferring "local"
        # even when it's 5-20x slower every single time. /30_000 (full
        # penalty by 30s, aggressive enough to flip a slow local win
        # below a fast hosted one) is a steeper scale than provider_
        # optimizer.py/cost_optimizer.py's /100_000 fix earlier today —
        # deliberately so, since this bandit's built-in success-reward
        # gap (1.0 vs 0.5) is itself larger and needs a stronger
        # counterweight to actually move the ranking. A genuinely fast
        # local win (sub-2s) still scores highest of all, unchanged.
        reward -= min(latency_ms / 30_000, 0.9)
        with self._lock:
            self._core.update(selection.arm, selection.context, max(-1.0, min(1.0, reward)))


# Process-wide singleton, same lifecycle as optimization_router/token_optimizer/rag_optimizer.
hybrid_router = HybridRouter()
