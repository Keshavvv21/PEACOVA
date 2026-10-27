"""The raw Bayesian linear regression Thompson Sampling primitive — pulled
out of router.py's ContextualThompsonRouter so it's reusable for ANY
"pick the best arm for this context, learn from what actually happened"
decision, not just agent selection. llm/optimization_router.py is the
second real user: which generation-parameter combination ("optimization
arm") to send on THIS model call, learned from that call's own
latency/success, is exactly the same mathematical shape as which AGENT
to route a task to — same math, applied one level lower and updated on a
much faster cycle (every model call, not every completed task).

See router.py's module docstring for the full derivation (Agrawal &
Goyal, ICML 2013) — this module is just the arm-id-keyed mechanics,
with no notion of "agent" or "task" at all.
"""
from __future__ import annotations

import numpy as np


class LinearBanditCore:
    """One Bayesian linear regression posterior per arm_id (any hashable
    string — an agent_id in router.py, an optimization-arm id in
    optimization_router.py, or anything else shaped like "pick the best
    of a few options for this context, using feedback").
    """

    def __init__(self, feature_dim: int, *, ridge: float = 1.0,
                exploration_scale: float = 0.35, rng: np.random.Generator | None = None):
        self._d = feature_dim + 1  # +1 bias term
        self._ridge = ridge
        self._nu = exploration_scale
        self._B: dict[str, np.ndarray] = {}
        self._f: dict[str, np.ndarray] = {}
        self._rng = rng or np.random.default_rng()

    def ensure(self, arm_id: str, seed_utility: float = 0.0) -> None:
        if arm_id in self._B:
            return
        self._B[arm_id] = self._ridge * np.eye(self._d)
        # See router.py's module docstring on why this seeds the MEAN at
        # seed_utility while leaving the posterior at its WIDEST — a
        # prior easily overridden by the first few real outcomes, not a
        # claim of real evidence.
        self._f[arm_id] = np.full(self._d, seed_utility * self._ridge / self._d)

    @staticmethod
    def features(context_vector: list[float]) -> np.ndarray:
        return np.array(context_vector + [1.0])

    def score(self, arm_id: str, context_vector: list[float], *,
             seed_utility: float = 0.0) -> tuple[float, float]:
        """Returns (sampled_value, posterior_mean_value)."""
        self.ensure(arm_id, seed_utility)
        x = self.features(context_vector)
        B = self._B[arm_id]
        f = self._f[arm_id]
        B_inv = np.linalg.inv(B)
        mu = B_inv @ f
        cov = (self._nu ** 2) * B_inv
        theta = self._rng.multivariate_normal(mu, cov)
        return float(theta @ x), float(mu @ x)

    def select(self, arm_ids: list[str], context_vector: list[float], *,
              seed_utilities: dict[str, float] | None = None) -> tuple[str, dict[str, float]]:
        """Thompson Sampling's decision rule: score every arm with its
        OWN fresh posterior sample, argmax. Returns (winner, {arm: sampled score})."""
        if not arm_ids:
            raise ValueError("select() needs at least one arm")
        seeds = seed_utilities or {}
        scored = {a: self.score(a, context_vector, seed_utility=seeds.get(a, 0.0))[0] for a in arm_ids}
        winner = max(scored, key=scored.get)
        return winner, scored

    def update(self, arm_id: str, context_vector: list[float], reward: float) -> None:
        self.ensure(arm_id)
        x = self.features(context_vector)
        self._B[arm_id] += np.outer(x, x)
        self._f[arm_id] += reward * x

    def posterior_mean(self, arm_id: str) -> np.ndarray | None:
        if arm_id not in self._B:
            return None
        return np.linalg.inv(self._B[arm_id]) @ self._f[arm_id]

    def to_state(self) -> dict:
        """A JSON-serializable snapshot of every arm's posterior — see
        bandit_persistence.py, the durable save/load pair built on top
        of this. Plain nested lists, not numpy arrays, so this round-trips
        through any of app/db.py's stores (LocalStore's json.dump,
        DynamoStore's boto3 Item) with no custom (de)serializer."""
        return {arm_id: {"B": self._B[arm_id].tolist(), "f": self._f[arm_id].tolist()}
               for arm_id in self._B}

    def load_state(self, state: dict) -> None:
        """Restores every arm's posterior from to_state()'s shape —
        REPLACES this core's in-memory state, doesn't merge with
        whatever it already had (call this once, right after
        construction, before any real select()/update() this process
        makes — see bandit_persistence.load())."""
        for arm_id, snap in state.items():
            self._B[arm_id] = np.array(snap["B"])
            self._f[arm_id] = np.array(snap["f"])
