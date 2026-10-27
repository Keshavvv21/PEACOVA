"""RL-selected model-call optimization — the piece that runs on EVERY
single model call, not just once per completed task like the agent-level
router (gmaofn/router.py). Reuses the exact same math (gmaofn/bandit.py's
LinearBanditCore — Thompson Sampling over a linear reward model) for a
DIFFERENT decision: which "optimization arm" (a real, meaningful
combination of request-level generation parameters) to send THIS call.

This is deliberately scoped to what a single HTTP call to an
already-running inference server can actually influence. KV cache,
continuous batching, speculative decoding, attention backend, and
tensor/pipeline/data parallelism (llm/optimization.py's
OptimizationProfile) are fixed at that server's own process startup —
no per-request field changes them, so an RL policy "choosing" among
them per call would be theatre, not a real decision. What genuinely
DOES vary per request on virtually every OpenAI-compatible endpoint —
`temperature`, `max_tokens`, and (where the server supports it, most
vLLM/TGI deployments do) `top_p`/`top_k`/`repetition_penalty` — is real,
and IS what this router picks between, learned from each call's actual
latency/success outcome. Combined with a genuinely-optimized backend
(declared via OptimizationProfile, configured once at deploy time),
every call is served through BOTH: a server already running continuous
batching + PagedAttention + tensor parallelism etc., AND a request-level
parameter choice this process itself keeps improving.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..framework.bandit import LinearBanditCore

_FEATURE_DIM = 2  # [tier_is_realtime, tier_is_bulk] — see _tier_features()


@dataclass
class OptimizationArm:
    """One named, real combination of request-level generation
    parameters. `extra_body` is merged into the actual request payload
    (see providers.py's OpenAICompatible/HuggingFaceProvider) — this is
    genuinely different generation behavior per arm, not a label."""

    arm_id: str
    description: str
    extra_body: dict = field(default_factory=dict)


# The default arm set every provider gets unless a deployment registers
# its own (see OptimizationRouter.register_arms) — three real, distinct
# generation-parameter profiles trading determinism/speed against
# diversity/quality, which is a genuine, request-level decision on
# virtually every OpenAI-compatible endpoint.
DEFAULT_ARMS: list[OptimizationArm] = [
    OptimizationArm("balanced", "Provider's own default sampling — no override.",
                    extra_body={}),
    OptimizationArm("low_latency", "Greedy-leaning, capped output — fastest wall-clock "
                                   "response, favored for realtime/interactive turns.",
                    extra_body={"temperature": 0.0, "max_tokens": 512}),
    OptimizationArm("high_quality", "Warmer sampling, longer budget — favored for "
                                    "delay_ok/bulk turns where quality matters more "
                                    "than shaving milliseconds.",
                    extra_body={"temperature": 0.7, "top_p": 0.9, "max_tokens": 2048}),
]


def _tier_features(tier: str) -> list[float]:
    return [1.0 if tier == "realtime" else 0.0, 1.0 if tier == "bulk" else 0.0]


class OptimizationRouter:
    """One LinearBanditCore PER PROVIDER (a "low_latency" arm that's
    great for Groq may be wrong for a slow local vLLM box) — arm ids are
    namespaced as f"{provider_name}::{arm_id}" internally so one shared
    core still keeps every provider's learning independent without a
    dict-of-cores.
    """

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.4):
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._arms: dict[str, list[OptimizationArm]] = {}
        self._lock = threading.Lock()

    def register_arms(self, provider_name: str, arms: list[OptimizationArm]) -> None:
        with self._lock:
            self._arms[provider_name] = arms

    def _arms_for(self, provider_name: str) -> list[OptimizationArm]:
        with self._lock:
            return self._arms.get(provider_name, DEFAULT_ARMS)

    def select(self, provider_name: str, tier: str) -> OptimizationArm:
        arms = self._arms_for(provider_name)
        if len(arms) == 1:
            return arms[0]
        arm_ids = [f"{provider_name}::{a.arm_id}" for a in arms]
        winner_key, _ = self._core.select(arm_ids, _tier_features(tier))
        winner_id = winner_key.split("::", 1)[1]
        return next(a for a in arms if a.arm_id == winner_id)

    def update(self, provider_name: str, arm: OptimizationArm, tier: str, *,
              ok: bool, latency_ms: float, latency_budget_ms: float = 5000.0) -> None:
        """Reward here is a fast, self-contained proxy (this call's own
        success + how much of the latency budget it used) — deliberately
        NOT the task-level Shapley-attributed reward gmaofn/feedback.py
        computes (that closes once per whole TASK, seconds to minutes
        later; this needs to learn within the same request cycle)."""
        norm_latency = min(latency_ms / max(latency_budget_ms, 1.0), 1.0)
        reward = (1.0 if ok else -0.5) - 0.5 * norm_latency
        key = f"{provider_name}::{arm.arm_id}"
        self._core.update(key, _tier_features(tier), reward)


# Process-wide singleton, same lifecycle as gmaofn.router.router.
optimization_router = OptimizationRouter()


def timed_call(fn, *args, **kwargs):
    """Small helper so callers (llm/router.py's TierClient) get both the
    result AND a real wall-clock latency in one line, for update()'s
    latency_ms — avoids every call site re-deriving time.time() deltas."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, (time.time() - t0) * 1000.0
