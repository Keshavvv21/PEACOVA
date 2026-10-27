"""Thompson Sampling — opt-in only, restricted to exactly two arms
("groq", "algo-akanade-slm"), modeled directly on Akanade's
llm/hybrid_router.py (one LinearBanditCore, two arms, an asymmetric
reward). Deliberately NOT a copy of provider_optimizer.py/
cost_optimizer.py, which span the full provider chain — the CEO memo is
explicit this should stay narrow until it's proven, never the thing that
picks among Luna/Max/GLM/Hy4/Deepseek/etc.

Runs in TWO modes depending on RouterConfig.strategy:
  - shadow (default): computed on every /route call and returned as
    shadow_pick, but never influences ordered_ids. This is what lets the
    comparison dashboard accumulate "what TS would have done" evidence
    before anyone flips it live — the CEO memo's own promotion gate.
  - live: only reachable when an admin has explicitly set strategy to
    "thompson_sampling"; even then, only ever picks between the two
    fixed arms and only when at least one of them is actually a
    candidate for this call — otherwise falls back to the deterministic
    order untouched.
"""
from __future__ import annotations

import threading

from .bandit import LinearBanditCore
from .db import store
from .models import Candidate, RouteContext

ARMS = ("groq", "algo-akanade-slm")
_FEATURE_DIM = 2  # [est_tokens/2000 clamped, tier_is_delay_ok]

_BANDIT_PK = "BANDIT#thompson_2arm"
_BANDIT_SK = "STATE"


def _context_vector(context: RouteContext) -> list[float]:
    return [
        min(context.est_tokens / 2000.0, 1.0),
        1.0 if context.tier == "delay_ok" else 0.0,
    ]


class TwoArmThompsonPolicy:
    def __init__(self) -> None:
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=1.0, exploration_scale=0.3)
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        row = store("RouterBanditState").get(_BANDIT_PK, _BANDIT_SK)
        if row and row.get("state"):
            self._core.load_state(row["state"])

    def _save(self) -> None:
        store("RouterBanditState").put(_BANDIT_PK, _BANDIT_SK, {"state": self._core.to_state()})

    def eligible_arms(self, candidates: list[Candidate]) -> list[str]:
        ids = {c.id for c in candidates}
        return [a for a in ARMS if a in ids]

    def pick(self, candidates: list[Candidate], context: RouteContext) -> str | None:
        """Returns the arm TS would pick, or None if neither ARMS
        candidate is present for this call (shadow_pick stays null
        rather than a value that couldn't actually apply)."""
        arms = self.eligible_arms(candidates)
        if not arms:
            return None
        if len(arms) == 1:
            return arms[0]  # nothing to sample between — the one real option
        with self._lock:
            winner, _ = self._core.select(list(arms), _context_vector(context))
        return winner

    def pick_ucb(self, candidates: list[Candidate], context: RouteContext) -> str | None:
        """Same eligibility rules as pick(), but selects by UCB against the
        SAME posterior Thompson Sampling above uses — see bandit.py's
        select_ucb()'s own docstring for why sharing the core matters."""
        arms = self.eligible_arms(candidates)
        if not arms:
            return None
        if len(arms) == 1:
            return arms[0]
        with self._lock:
            winner, _ = self._core.select_ucb(list(arms), _context_vector(context))
        return winner

    def update(self, arm_id: str, context: RouteContext, *, ok: bool, latency_ms: float) -> None:
        if arm_id not in ARMS:
            return  # only the two designated arms ever get real feedback
        # Same reward asymmetry shape as hybrid_router.py's own local/
        # hosted split — a slow success is worth less than a fast one,
        # a failure always costs more than it could have gained.
        reward = 1.0 if ok else -0.6
        reward -= min(latency_ms / 30_000, 0.9)
        with self._lock:
            self._core.update(arm_id, _context_vector(context), max(-1.0, min(1.0, reward)))
            self._save()


# Process-wide singleton — same lifecycle as every other bandit
# singleton in this codebase (hybrid_router, provider_optimizer, ...).
thompson_policy = TwoArmThompsonPolicy()


# --- RouterConfig — the admin-toggled live/shadow switch ---

_CONFIG_PK = "ROUTER_CONFIG"
_CONFIG_SK = "SINGLETON"
_DEFAULT_STRATEGY = "deterministic"


def get_strategy() -> str:
    row = store("RouterConfig").get(_CONFIG_PK, _CONFIG_SK)
    return (row or {}).get("strategy", _DEFAULT_STRATEGY)


def set_strategy(strategy: str, updated_by: str) -> dict:
    from datetime import datetime, timezone
    if strategy not in ("deterministic", "thompson_sampling"):
        raise ValueError(f"unknown strategy: {strategy!r}")
    row = {"strategy": strategy, "updated_by": updated_by,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    store("RouterConfig").put(_CONFIG_PK, _CONFIG_SK, row)
    return row


def get_strategy_row() -> dict:
    row = store("RouterConfig").get(_CONFIG_PK, _CONFIG_SK)
    return row or {"strategy": _DEFAULT_STRATEGY, "updated_by": None, "updated_at": None}
