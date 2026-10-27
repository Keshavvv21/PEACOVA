"""The Adaptive Agent Assignment Engine — §2/§3's routing layer, and
ALGO AKANADE's actual research contribution: agent selection modeled as a
LINEAR CONTEXTUAL BANDIT solved with Thompson Sampling (Agrawal & Goyal,
"Thompson Sampling for Contextual Bandits with Linear Payoffs", ICML
2013), warm-started from a hand-designed utility prior and continuously
corrected by real observed rewards (whose credit, when multiple agents
jointly worked a task, is split via shapley.py's Nobel-Prize-linked
Shapley value before it ever reaches this module).

Per candidate agent a, ALGO AKANADE maintains a Bayesian linear regression
posterior over "reward given context": N(μ_a, ν² B_a⁻¹), where
B_a = Σ x xᵗ + λI (λ = ridge prior strength) and the posterior mean is
μ_a = B_a⁻¹ f_a, f_a = Σ r·x accumulated from every past (context,
reward) pair this agent has actually been given. Selecting an agent means
SAMPLING θ_a ~ N(μ_a, ν² B_a⁻¹) for every candidate and picking
argmax_a θ_a·x — sampling (rather than always taking the posterior mean)
is what gives Thompson Sampling its exploration: an agent with a wide,
uncertain posterior occasionally gets picked even if its mean looks
worse, so it keeps getting data instead of being starved by an early bad
run.

The formal objective this approximates online is exactly §4's

    π* = argmax_π E[Σ_t γ^t R_t]

— a bandit is the γ→0 (single-step, no long-horizon planning) special
case of that same RL objective; Q-Learning/PPO over a full MDP
(sequencing an entire multi-turn task rather than a single agent-pick) is
the natural next step and is deliberately left as a pluggable strategy
(see AdaptiveRouter.strategy) rather than implemented here twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .agent_base import AgentMetadata
from .bandit import LinearBanditCore
from .task import TaskContext
from .task_memory import TaskMemory

Strategy = Literal["thompson_sampling"]  # extension point — see module docstring


@dataclass
class UtilityWeights:
    """§4's `U(a|T) = w1·Q + w2·R + w3·C + w4·F − w5·Cost − w6·Latency` —
    the COLD-START prior every agent's bandit posterior is warm-started
    from (see AdaptiveRouter._ensure), before any real feedback exists.
    Once feedback arrives, the bandit's own learned posterior mean
    increasingly dominates this hand-designed heuristic — U(a|T) matters
    most on agent #1's very first call, least by agent #100."""

    w_quality: float = 0.30       # w1 — Q(a,T): predicted task quality
    w_reliability: float = 0.20   # w2 — R(a): historical reliability
    w_capability: float = 0.25    # w3 — C(a,T): capability match
    w_feedback: float = 0.10      # w4 — F(a,T): feedback-derived performance
    w_cost: float = 0.10          # w5 — Cost(a)
    w_latency: float = 0.05       # w6 — Latency(a)


def utility_prior(agent: AgentMetadata, context: TaskContext,
                  weights: UtilityWeights | None = None) -> float:
    """U(a|T) computed from the agent's own registry metadata — the
    heuristic that stands in for Q/C/F before this agent has any of its
    own observed rewards. Returns a value roughly in [-1, 1], same scale
    as reward.compute_reward's output, so it composes as a prior on the
    bandit's own reward scale rather than needing a separate unit."""
    w = weights or UtilityWeights()
    capability_match = 1.0 if agent.matches_capabilities(context.required_capabilities) else 0.0
    if context.task_type in agent.supported_tasks:
        capability_match = min(1.0, capability_match + 0.5)
    elif not context.required_capabilities:
        # No explicit capability requirement AND this agent doesn't even
        # claim the task's type — don't let it look like a full
        # capability match just because the empty set trivially
        # "subsets" everything.
        capability_match = 0.0
    quality_guess = agent.reliability_score  # no per-task quality history yet — reliability is the best guess
    feedback_guess = agent.reliability_score
    norm_cost = min(agent.cost_per_call_usd / max(context.budget_usd, 1e-6), 1.0)
    norm_latency = min((agent.latency_profile.p95_ms / 1000.0) / max(context.latency_requirement_s, 1e-6), 1.0)

    u = (w.w_quality * quality_guess
         + w.w_reliability * agent.reliability_score
         + w.w_capability * capability_match
         + w.w_feedback * feedback_guess
         - w.w_cost * norm_cost
         - w.w_latency * norm_latency)
    return max(-1.0, min(1.0, u))


@dataclass
class RoutingDecision:
    agent_id: str
    sampled_utility: float
    posterior_mean_utility: float
    all_candidate_scores: dict[str, float] = field(default_factory=dict)
    strategy: Strategy = "thompson_sampling"


class ContextualThompsonRouter:
    """One Bayesian linear regression posterior per agent_id, via the
    shared `LinearBanditCore` (bandit.py) — the agent-selection-specific
    part of this class is purely: turn a TaskContext into a feature
    vector, and seed each agent's posterior from `utility_prior` instead
    of a bare 0.0. `ridge`/`exploration_scale` pass straight through to
    the core; see its module docstring and this file's own for the math.
    """

    def __init__(self, feature_dim: int, *, ridge: float = 1.0,
                exploration_scale: float = 0.35, rng: np.random.Generator | None = None):
        self._core = LinearBanditCore(feature_dim, ridge=ridge,
                                      exploration_scale=exploration_scale, rng=rng)
        # Per-instance (not a module singleton) — same reasoning as every
        # other embedding-memory layer in this framework: a fresh
        # ContextualThompsonRouter() in a test must never inherit another
        # instance's remembered tasks.
        self.task_memory = TaskMemory()

    def score(self, agent: AgentMetadata, context: TaskContext,
             utility_weights: UtilityWeights | None = None, task_text: str | None = None) -> RoutingDecision:
        """Score ONE agent (no argmax across candidates) — exposed on its
        own so observability.py can report every candidate's score, not
        just the winner's. `task_text`, when given, adds TaskMemory's
        embedding-based bonus/penalty on top of the hand-designed prior —
        see this module's docstring and task_memory.py's."""
        seed = utility_prior(agent, context, utility_weights)
        if task_text:
            seed = max(-1.0, min(1.0, seed + self.task_memory.bonus_for(task_text, agent.agent_id)))
        sampled, mean = self._core.score(agent.agent_id, context.as_vector(), seed_utility=seed)
        return RoutingDecision(agent_id=agent.agent_id, sampled_utility=sampled,
                               posterior_mean_utility=mean)

    def select(self, candidates: list[AgentMetadata], context: TaskContext,
              utility_weights: UtilityWeights | None = None, task_text: str | None = None) -> RoutingDecision:
        """Thompson Sampling's actual decision rule: score every
        candidate (each with its OWN fresh posterior sample) and take
        argmax — an agent with high uncertainty occasionally wins even
        with a middling mean, which is what keeps exploring it."""
        if not candidates:
            raise ValueError("select() needs at least one candidate agent")
        scored = [self.score(a, context, utility_weights, task_text) for a in candidates]
        winner = max(scored, key=lambda d: d.sampled_utility)
        winner.all_candidate_scores = {d.agent_id: d.sampled_utility for d in scored}
        return winner

    def update(self, agent_id: str, context: TaskContext, reward: float,
              task_text: str | None = None) -> None:
        """The bandit's learning step: Bayesian linear regression's
        standard sufficient-statistics update, B += xxᵗ, f += r·x — pulls
        this agent's posterior mean toward actually predicting `reward`
        for contexts that look like `context`, and narrows its posterior
        (more data ⇒ more confidence ⇒ less exploration for this agent
        going forward, until a NEW kind of context or a NEW low reward
        widens it again). `task_text`, when given, ALSO remembers this
        (request, agent, reward) triple in TaskMemory for future lookups
        — the framework's continuous-learning half, distinct from the
        bandit's own numeric-feature learning above."""
        self._core.update(agent_id, context.as_vector(), reward)
        if task_text:
            self.task_memory.remember(task_text, agent_id, reward)

    def posterior_mean(self, agent_id: str) -> np.ndarray | None:
        return self._core.posterior_mean(agent_id)


class AdaptiveRouter:
    """The public entry point orchestrator.py calls — wraps
    ContextualThompsonRouter behind a `strategy` name so a future
    Q-Learning/PPO router can be swapped in via configuration (§3's
    "support algorithms such as Contextual Bandits, ... Q-Learning,
    Policy Gradient, PPO") without changing this class's callers."""

    def __init__(self, feature_dim: int = 6, *, strategy: Strategy = "thompson_sampling",
                utility_weights: UtilityWeights | None = None, **kwargs):
        if strategy != "thompson_sampling":
            raise ValueError(f"Unknown routing strategy: {strategy!r}")
        self.strategy: Strategy = strategy
        self.utility_weights = utility_weights or UtilityWeights()
        self._bandit = ContextualThompsonRouter(feature_dim, **kwargs)

    def select(self, candidates: list[AgentMetadata], context: TaskContext,
              task_text: str | None = None) -> RoutingDecision:
        return self._bandit.select(candidates, context, self.utility_weights, task_text)

    def update(self, agent_id: str, context: TaskContext, reward: float,
              task_text: str | None = None) -> None:
        self._bandit.update(agent_id, context, reward, task_text)

    def score_all(self, candidates: list[AgentMetadata], context: TaskContext) -> dict[str, RoutingDecision]:
        return {a.agent_id: self._bandit.score(a, context, self.utility_weights) for a in candidates}


# Process-wide singleton, same lifecycle/rationale as registry.registry.
router = AdaptiveRouter()
