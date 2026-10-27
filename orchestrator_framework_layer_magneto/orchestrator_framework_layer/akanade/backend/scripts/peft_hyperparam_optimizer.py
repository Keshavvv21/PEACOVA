"""The 10th real optimizer in this codebase, and the only one that
picks nightly PEFT fine-tune hyperparameters: which LoRA rank/alpha and
reward threshold to train with TONIGHT, learned from whether last
night's choice actually passed the regression gate — not a fixed
config picked once and never revisited.

Every OTHER "which option to pick, learn from what happened" decision
here (agent routing, framework track, RAG store×algorithm, Hybrid
local/hosted, token-schema size, provider ordering, the security-
confidence layer) is the exact same Thompson Sampling contextual bandit
— app/framework/bandit.py's LinearBanditCore, reused, never
reimplemented, everywhere else in this repo. This module can't import
that one directly: scripts/finetune_local_slm.py (this optimizer's only
caller) is a deliberately STANDALONE script with its own venv, never
importing the main `app` package (see that file's own module
docstring) — the same isolation reason data_engine.py's HashEmbedder is
a local reimplementation instead of an import. So the identical
Bayesian-linear-regression Thompson Sampling math is reproduced here,
faithfully, rather than approximated with a weaker heuristic — "same
algorithm everywhere, isolated where it has to be," not "a different,
simpler algorithm just for this one."

Arms are named hyperparameter PRESETS (the same "named preset, not a
raw continuous knob" shape token_optimizer.py's full/top12/top6/top3
and hybrid_router.py's local/hosted already use) rather than tuning
lora_r/lora_alpha/min_reward as three independent continuous bandits —
these three interact (a higher rank with a low reward threshold trains
on noisier data with more capacity to overfit it), so picking a
JOINTLY sane combination each time is the safer shape than three
uncoordinated single-value choices.

Persistence: a plain JSON file (--state-path), not
bandit_persistence.py's DynamoDB/LocalStore-backed store() — that
module reaches into the live app's `app.db`, unavailable to this
standalone script's own process. Same to_state()/load_state() shape as
the real LinearBanditCore, just written straight to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Each arm bundles the three hyperparameters that actually interact —
# see module docstring. "balanced" is finetune_local_slm.py's long-
# standing hand-picked default, kept as the cold-start-friendly middle
# option rather than assuming this bandit already knows better on day one.
ARMS: dict[str, dict] = {
    "conservative": {"lora_r": 4, "lora_alpha": 8, "min_reward": 0.8},
    "balanced": {"lora_r": 8, "lora_alpha": 16, "min_reward": 0.7},
    "aggressive": {"lora_r": 16, "lora_alpha": 32, "min_reward": 0.6},
}

_FEATURE_DIM = 2  # [curated_example_count / 500, avg_reward] + bias, added below


class _LinearBanditCore:
    """A faithful, standalone reimplementation of
    app/framework/bandit.py's LinearBanditCore — identical math (see
    that file's own docstring for the full Agrawal & Goyal derivation),
    duplicated rather than imported for the isolation reason in this
    module's own docstring above."""

    def __init__(self, feature_dim: int, *, ridge: float = 1.0, exploration_scale: float = 0.35,
                rng: np.random.Generator | None = None):
        self._d = feature_dim + 1
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

    def score(self, arm_id: str, context_vector: list[float], *, seed_utility: float = 0.0) -> float:
        self.ensure(arm_id, seed_utility)
        x = self.features(context_vector)
        B_inv = np.linalg.inv(self._B[arm_id])
        mu = B_inv @ self._f[arm_id]
        theta = self._rng.multivariate_normal(mu, (self._nu ** 2) * B_inv)
        return float(theta @ x)

    def select(self, arm_ids: list[str], context_vector: list[float]) -> tuple[str, dict[str, float]]:
        scored = {a: self.score(a, context_vector) for a in arm_ids}
        return max(scored, key=scored.get), scored

    def update(self, arm_id: str, context_vector: list[float], reward: float) -> None:
        self.ensure(arm_id)
        x = self.features(context_vector)
        self._B[arm_id] += np.outer(x, x)
        self._f[arm_id] += reward * x

    def to_state(self) -> dict:
        return {a: {"B": self._B[a].tolist(), "f": self._f[a].tolist()} for a in self._B}

    def load_state(self, state: dict) -> None:
        for arm_id, snap in state.items():
            self._B[arm_id] = np.array(snap["B"])
            self._f[arm_id] = np.array(snap["f"])


class PeftHyperparamOptimizer:
    """The one entry point finetune_local_slm.py (--auto-hyperparams)
    and regression_gate.py (--update-peft-bandit) both use. Context is
    cheap and non-LLM, same as every other optimizer here: how much
    curated data exists tonight, and how clean it is on average — more/
    cleaner data can support a bigger, more aggressive LoRA rank without
    overfitting; a thin, noisy corpus is safer trained conservatively."""

    def __init__(self, state_path: str | Path):
        self.path = Path(state_path)
        self._core = _LinearBanditCore(_FEATURE_DIM)
        if self.path.exists():
            try:
                self._core.load_state(json.loads(self.path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt/partial state file — start fresh rather than crash a nightly job

    @staticmethod
    def _context(curated_count: int, avg_reward: float) -> list[float]:
        return [min(curated_count, 500) / 500.0, avg_reward]

    def select_arm(self, *, curated_count: int, avg_reward: float) -> tuple[str, dict]:
        """Returns (arm_name, hyperparams) — hyperparams is one of ARMS'
        own dicts, ready to splat straight into argparse-equivalent
        overrides. Also returns the context used, via select_arm_full(),
        for a caller that needs to persist it for record_outcome() later
        in a SEPARATE process (regression_gate.py runs well after this
        script exits)."""
        return self.select_arm_full(curated_count=curated_count, avg_reward=avg_reward)[:2]

    def select_arm_full(self, *, curated_count: int, avg_reward: float) -> tuple[str, dict, list[float]]:
        context = self._context(curated_count, avg_reward)
        arm, _ = self._core.select(list(ARMS.keys()), context)
        return arm, ARMS[arm], context

    def record_outcome(self, arm: str, context: list[float], *, score_margin: float) -> None:
        """`score_margin` is regression_gate.py's `new_score - old_score`
        — positive means tonight's fine-tune genuinely beat the
        currently-served model, negative means it regressed. Mapped to a
        reward via tanh so one outlier night's huge margin can't swamp
        the posterior the way an unbounded reward would (the same
        "clamp to a sane range" instinct finetune_log.py's own
        compute_reward already applies, just continuous here instead of
        the three fixed penalty tiers that fits)."""
        reward = float(np.tanh(score_margin * 10))  # tanh(±0.1 margin) ≈ ±0.76 — a real fail/pass
        # already dominates the signal without needing a huge margin to saturate
        self._core.update(arm, context, reward)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._core.to_state()), encoding="utf-8")
