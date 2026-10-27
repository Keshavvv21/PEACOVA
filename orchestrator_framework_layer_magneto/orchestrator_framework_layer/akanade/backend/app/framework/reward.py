"""The reward function every executed task produces, and the adaptive
coefficients it's computed from — the closed loop's actual learning
signal (see feedback.py for how a reward becomes a router update).

    R = α·Quality + β·Success + γ·UserFeedback − δ·Cost − ε·Latency − ζ·Failure

All six coefficients are configurable (RewardWeights) and adaptively
updated from observed outcomes (AdaptiveWeights.update) rather than fixed
forever — a REINFORCE-style nudge: each coefficient drifts toward
whichever term most consistently co-occurred with a HIGH reward, and away
from whichever term co-occurred with a LOW one, renormalized so the six
weights always sum to 1 (a probability simplex over "what matters right
now"). This is deliberately simple and auditable (an operator can read
weights.snapshot() and see exactly why routing shifted) rather than a
second black-box model layered on top of the router's own bandit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """α, β, γ, δ, ε, ζ — always kept normalized to sum to 1 so R stays
    bounded in roughly [-1, 1] regardless of how they've adapted."""

    quality: float = 0.30        # α
    success: float = 0.25        # β
    user_feedback: float = 0.20  # γ
    cost: float = 0.10           # δ
    latency: float = 0.10        # ε
    failure: float = 0.05        # ζ

    def normalize(self) -> None:
        total = (self.quality + self.success + self.user_feedback
                 + self.cost + self.latency + self.failure)
        if total <= 0:
            return
        self.quality /= total
        self.success /= total
        self.user_feedback /= total
        self.cost /= total
        self.latency /= total
        self.failure /= total

    def snapshot(self) -> dict[str, float]:
        return {"quality": self.quality, "success": self.success,
                "user_feedback": self.user_feedback, "cost": self.cost,
                "latency": self.latency, "failure": self.failure}


@dataclass
class ExecutionOutcome:
    """What one agent execution actually produced — the raw material
    both the reward calculation and the adaptive-weight update read."""

    quality: float                      # 0..1, self- or externally-evaluated
    success: bool
    cost_usd: float
    latency_s: float
    failed: bool = False                # a hard failure (exception), not just low quality
    user_feedback: float | None = None  # 0..1, only when a user actually rated this


def compute_reward(outcome: ExecutionOutcome, weights: RewardWeights, *,
                   cost_budget_usd: float = 1.0, latency_budget_s: float = 30.0) -> float:
    """R = α·Quality + β·Success + γ·UserFeedback − δ·Cost − ε·Latency − ζ·Failure,
    with Cost/Latency normalized against the caller's own budget/latency
    requirement (TaskContext.budget_usd / latency_requirement_s) rather
    than an arbitrary fixed scale — a $0.02 call is expensive against a
    $0.01 budget and cheap against a $5 one; the reward should say so."""
    success_term = 1.0 if outcome.success else 0.0
    feedback_term = outcome.user_feedback if outcome.user_feedback is not None else outcome.quality
    failure_term = 1.0 if outcome.failed else 0.0
    norm_cost = min(outcome.cost_usd / max(cost_budget_usd, 1e-6), 1.0)
    norm_latency = min(outcome.latency_s / max(latency_budget_s, 1e-6), 1.0)

    r = (weights.quality * outcome.quality
         + weights.success * success_term
         + weights.user_feedback * feedback_term
         - weights.cost * norm_cost
         - weights.latency * norm_latency
         - weights.failure * failure_term)
    return max(-1.0, min(1.0, r))


@dataclass
class AdaptiveWeights:
    """Wraps a RewardWeights and nudges it after every observed
    (outcome, reward) pair — the "implement adaptive weights" requirement.

    Update rule: for each term, compare its own value on this outcome
    against the outcome's overall reward. A term that was HIGH when the
    reward was HIGH (or LOW when the reward was LOW) — i.e. positively
    correlated with the signal it's supposed to predict — gets nudged up;
    one that moved opposite to the reward gets nudged down. This is a
    single-sample, online stand-in for a proper multi-sample correlation
    (computing a running Pearson correlation per term is the natural
    next step for a persistent deployment) but is enough to make the
    weights genuinely responsive to what's actually driving outcomes,
    not fixed forever.
    """

    weights: RewardWeights = field(default_factory=RewardWeights)
    learning_rate: float = 0.02
    _min_weight: float = 0.02  # never let a term vanish to exactly 0 — always explorable

    def update(self, outcome: ExecutionOutcome, reward: float) -> None:
        success_term = 1.0 if outcome.success else 0.0
        feedback_term = outcome.user_feedback if outcome.user_feedback is not None else outcome.quality
        failure_term = 1.0 if outcome.failed else 0.0
        # Cost/latency are COSTS (their coefficient is subtracted), so a
        # term "moving with" a high reward means the cost was LOW, i.e.
        # (1 - normalized_value) tracks reward the same direction quality/
        # success/feedback do.
        norm_cost = min(outcome.cost_usd, 1.0)
        norm_latency = min(outcome.latency_s / 30.0, 1.0)

        signals = {
            "quality": outcome.quality, "success": success_term,
            "user_feedback": feedback_term, "cost": 1.0 - norm_cost,
            "latency": 1.0 - norm_latency, "failure": 1.0 - failure_term,
        }
        centered_reward = reward  # already in [-1, 1], 0 = neutral
        for field_name, signal_value in signals.items():
            current = getattr(self.weights, field_name)
            # signal_value in [0,1]; center at 0.5 so "average" signal ×
            # any reward contributes ~0 net drift.
            delta = self.learning_rate * centered_reward * (signal_value - 0.5)
            setattr(self.weights, field_name, max(self._min_weight, current + delta))
        self.weights.normalize()
