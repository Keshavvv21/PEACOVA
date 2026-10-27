"""The raw Bayesian linear regression Thompson Sampling primitive — a
direct copy of Akanade's app/framework/bandit.py's LinearBanditCore (see
that file for the full derivation, Agrawal & Goyal, ICML 2013). Copied
rather than imported for the same reason as db.py: this service has no
dependency on the rest of akanade/backend/app.
"""
from __future__ import annotations

import numpy as np


class LinearBanditCore:
    """One Bayesian linear regression posterior per arm_id (any hashable
    string)."""

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
        if not arm_ids:
            raise ValueError("select() needs at least one arm")
        seeds = seed_utilities or {}
        scored = {a: self.score(a, context_vector, seed_utility=seeds.get(a, 0.0))[0] for a in arm_ids}
        winner = max(scored, key=scored.get)
        return winner, scored

    def select_ucb(self, arm_ids: list[str], context_vector: list[float], *,
                   seed_utilities: dict[str, float] | None = None) -> tuple[str, dict[str, float]]:
        """Same posterior as select() (Thompson Sampling), but picks by upper
        confidence bound instead of a posterior sample: mean + exploration_scale
        * sqrt(x^T B^-1 x), the standard LinUCB rule (Li et al., WWW 2010).
        Comparing this against select()'s picks on the SAME learned B/f per arm
        isolates the exploration STRATEGY as the only variable."""
        if not arm_ids:
            raise ValueError("select_ucb() needs at least one arm")
        seeds = seed_utilities or {}
        x = self.features(context_vector)
        scored: dict[str, float] = {}
        for a in arm_ids:
            self.ensure(a, seeds.get(a, 0.0))
            B_inv = np.linalg.inv(self._B[a])
            mu = B_inv @ self._f[a]
            radius = self._nu * float(np.sqrt(x @ B_inv @ x))
            scored[a] = float(mu @ x) + radius
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
        return {arm_id: {"B": self._B[arm_id].tolist(), "f": self._f[arm_id].tolist()}
               for arm_id in self._B}

    def load_state(self, state: dict) -> None:
        for arm_id, snap in state.items():
            self._B[arm_id] = np.array(snap["B"])
            self._f[arm_id] = np.array(snap["f"])
