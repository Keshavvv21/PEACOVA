"""Shapley values — Lloyd Shapley's 1953 cooperative-game-theory solution
concept, for which he shared the 2012 Nobel Memorial Prize in Economic
Sciences (with Alvin Roth, "for the theory of stable allocations and the
practice of market design") — used here to fairly split a workflow's
single observed reward across however many agents jointly produced it,
BEFORE that credit feeds into each agent's own Thompson Sampling posterior
(router.py). Without this, a naive scheme has to either dump the whole
reward on the last agent in a pipeline (crediting the Reviewer for the
Coder's + Tester's work) or split it evenly (ignoring that one agent may
have barely contributed) — both of which teach the router the wrong
lesson about which agent actually deserves more of that traffic.

For a coalition of agents N with a characteristic (value) function v(S)
— "how good is the outcome if only the agents in subset S had
participated" — agent i's Shapley value is the average of its marginal
contribution v(S ∪ {i}) − v(S) over every possible arrival order:

    φ_i(v) = Σ_{S ⊆ N\\{i}}  [|S|! (|N|-|S|-1)! / |N|!] · (v(S ∪ {i}) − v(S))

This module's characteristic function models each agent's standalone
quality as an independent chance of "covering" the task well — a
noisy-OR: a coalition's value is the probability at least one of its
members produced a good-enough result, v(S) = 1 − Π_{i∈S}(1 − quality_i).
This is a genuine, non-trivial (superadditive, diminishing-returns)
characteristic function — two mediocre agents together are worth more
than either alone, but not simply their sum — not a toy whose Shapley
value trivially reduces to the input.

Exact computation is used up to _EXACT_LIMIT agents (2^n subsets is fine
at that size); beyond that, the standard Monte Carlo permutation estimator
(shuffle the agent order, accumulate each agent's marginal contribution as
it's "added" to the running coalition, average over many shuffles) is
used instead — the same approximation SHAP and other applied-Shapley
tools use for large player counts.
"""
from __future__ import annotations

import itertools
import math
import random

_EXACT_LIMIT = 8


def _noisy_or(qualities: list[float]) -> float:
    product = 1.0
    for q in qualities:
        product *= (1.0 - max(0.0, min(1.0, q)))
    return 1.0 - product


def _exact_shapley(agent_ids: list[str], quality: dict[str, float]) -> dict[str, float]:
    n = len(agent_ids)
    others = {a: [x for x in agent_ids if x != a] for a in agent_ids}
    phi = {a: 0.0 for a in agent_ids}
    for a in agent_ids:
        rest = others[a]
        for r in range(len(rest) + 1):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for subset in itertools.combinations(rest, r):
                v_without = _noisy_or([quality[x] for x in subset])
                v_with = _noisy_or([quality[x] for x in subset] + [quality[a]])
                phi[a] += weight * (v_with - v_without)
    return phi


def _monte_carlo_shapley(agent_ids: list[str], quality: dict[str, float],
                         samples: int = 500, seed: int | None = None) -> dict[str, float]:
    rng = random.Random(seed)
    phi = {a: 0.0 for a in agent_ids}
    for _ in range(samples):
        order = list(agent_ids)
        rng.shuffle(order)
        coalition: list[str] = []
        v_prev = 0.0
        for a in order:
            coalition.append(a)
            v_now = _noisy_or([quality[x] for x in coalition])
            phi[a] += (v_now - v_prev)
            v_prev = v_now
    return {a: total / samples for a, total in phi.items()}


def shapley_values(quality_by_agent: dict[str, float], *, seed: int | None = None) -> dict[str, float]:
    """Each participating agent's fair marginal-contribution share, as a
    fraction of v(N) — i.e. `sum(shapley_values(...).values()) == v(N)`
    (the efficiency axiom), so multiplying each share by the workflow's
    actual observed reward (reward.py's compute_reward) gives each
    agent's attributed reward directly."""
    agent_ids = list(quality_by_agent.keys())
    if len(agent_ids) == 1:
        only = agent_ids[0]
        return {only: _noisy_or([quality_by_agent[only]])}
    if not agent_ids:
        return {}
    if len(agent_ids) <= _EXACT_LIMIT:
        return _exact_shapley(agent_ids, quality_by_agent)
    return _monte_carlo_shapley(agent_ids, quality_by_agent, seed=seed)


def attribute_reward(quality_by_agent: dict[str, float], total_reward: float, *,
                     seed: int | None = None) -> dict[str, float]:
    """Splits `total_reward` (the workflow's single observed reward,
    reward.compute_reward's output) across participating agents in
    proportion to their Shapley share of v(N) — the number every agent's
    own router.ContextualThompsonRouter.update() call actually uses."""
    shares = shapley_values(quality_by_agent, seed=seed)
    v_total = sum(shares.values())
    if v_total <= 1e-9:
        # Every agent contributed ~nothing measurable — split evenly
        # rather than divide by ~0, so nobody's update blows up.
        n = len(quality_by_agent) or 1
        return {a: total_reward / n for a in quality_by_agent}
    return {a: total_reward * (share / v_total) for a, share in shares.items()}
