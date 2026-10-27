"""RL-learned provider ORDERING within a tier's fallback chain — which
of the (usually several) available rungs to try FIRST for a given tier,
learned from real success + latency. Closes a real gap `router.py`'s
own module docstring documented as a fixed, hand-authored order
(`CHAINS`): outside an explicit `preferred` override (a user's own
model pick, or Hybrid mode's local/hosted choice — see
`llm/hybrid_router.py`), every call for a given tier tried the exact
same provider first, forever, regardless of which one has actually been
fastest/most reliable in practice. Same `LinearBanditCore` primitive as
every other optimizer here (`rag_optimizer.py`, `token_optimizer.py`,
`security_optimizer.py`) — arms = `"{tier}::{provider}"`, one posterior
per (tier, provider) pair.

Deliberately narrow, safety-preserving scope — read before touching:
this ONLY reorders which AVAILABLE, non-floor provider `TierClient.chat()`
tries first. It never removes a rung, never lets the deterministic
`rule` floor move ahead of a real provider (`order()` always keeps it
last), and never overrides an explicit `preferred` — `TierClient.chat()`
applies that override AFTER this reorder, so it still wins outright,
unchanged. If the RL-picked provider fails, the existing fallback loop
tries the rest of the (now RL-ordered) chain exactly as it always has —
this optimizer only ever changes step one of a decision that already
has a full safety net behind it, the same "augments, never replaces the
guarantee" rule `rag_optimizer.py`'s own docstring establishes for
retrieval.

Uses Thompson Sampling for a full RANKING, not just a single winner:
every candidate gets one fresh sample from its own posterior, and the
whole list is sorted by that sample — the same exploration/exploitation
behavior `LinearBanditCore.select()` gives a single pick, extended to
"which order should we try ALL of these in."
"""
from __future__ import annotations

import threading

from ..db import store as db_store
from ..framework import bandit_persistence
from ..framework.bandit import LinearBanditCore

# One constant-context arm per (tier, provider) — no per-call features
# yet (unlike rag_optimizer's query-length signal); the bias term is the
# whole learned estimate, same shape security_optimizer.py uses.
_FEATURE_DIM = 1
_CONTEXT = [1.0]

_FLOOR_PROVIDER = "rule"  # the deterministic rule-based floor — never reordered ahead of a real provider
_COUNTS_TABLE = "ProviderOptimizerCounts"


class ProviderOptimizer:
    # This optimizer updates on EVERY LLM call (every chat turn, every
    # retry) — much higher frequency than security_optimizer.py's
    # confirmation-driven updates. Persisting on every single one would
    # add a real DB write to the hot path of every model call, including
    # against a real DynamoDB endpoint's network latency. Throttled
    # instead: save every _SAVE_EVERY updates, an honest bounded-
    # staleness trade-off (worst case on a crash: lose the last few
    # updates' learning, never the whole posterior).
    _SAVE_EVERY = 20

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.4,
                name: str | None = None):
        """`name` opts into durable persistence (bandit_persistence.py),
        same pattern security_optimizer.py established — learned
        provider preferences survive a restart instead of resetting to
        the original CHAINS order every redeploy. None (the default) is
        pure in-memory, what every unit test uses."""
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._lock = threading.Lock()
        self._name = name
        self._updates_since_save = 0
        if name:
            bandit_persistence.load(name, self._core)
        # Human-readable call/win counts alongside the bandit's own
        # Bayesian sufficient stats — same rag_optimizer.py rationale
        # (LinearBanditCore has no notion of "how many calls"). Kept
        # in-memory only when unnamed (matches bandit persistence itself
        # being skipped for every unit test's unnamed instance).
        self._counts: dict[str, dict[str, int]] = self._load_counts() if name else {}

    def _load_counts(self) -> dict[str, dict[str, int]]:
        row = db_store(_COUNTS_TABLE).get(self._name, "COUNTS")
        return (row or {}).get("arms", {})

    def _save_counts(self) -> None:
        db_store(_COUNTS_TABLE).put(self._name, "COUNTS", {"arms": self._counts})

    def order(self, tier: str, provider_names: list[str]) -> list[str]:
        """Returns `provider_names` reordered by a fresh Thompson-Sampled
        draw from each one's own (tier, provider) posterior — highest
        sample first. The `rule` floor, if present, is always moved back
        to the end regardless of its sample, so a cold or unlucky
        posterior can never rank the deterministic fallback ahead of a
        real model. Cold-start (`seed_utility`) biases toward the
        ORIGINAL `CHAINS` order — position 0 in the input list seeds
        highest — so a freshly-deployed system's behavior matches
        today's fixed order exactly, and only drifts from it once real
        evidence justifies it, same "prior easily overridden, never a
        claim of real evidence" rule `bandit.py`'s own `ensure()` uses."""
        real = [n for n in provider_names if n != _FLOOR_PROVIDER]
        if len(real) > 1:
            with self._lock:
                scored = [(self._core.score(f"{tier}::{name}", _CONTEXT,
                                            seed_utility=len(real) - i)[0], i, name)
                         for i, name in enumerate(real)]
            real = [name for _, _, name in sorted(scored, key=lambda t: (-t[0], t[1]))]
        return real + ([_FLOOR_PROVIDER] if _FLOOR_PROVIDER in provider_names else [])

    def record_outcome(self, tier: str, provider: str, *, ok: bool, latency_ms: float) -> None:
        """Real signal only — called from TierClient.chat() right where
        it already records the SAME call's outcome for
        optimization_router.py (generation params) and the tier_log.
        Reward mirrors hybrid_router.py's local/hosted shape: success
        matters most, a real (capped) latency penalty on top so a
        technically-successful-but-slow rung doesn't rank as highly as
        an equally-successful fast one."""
        if provider == _FLOOR_PROVIDER:
            return  # never learned — the floor's rank is fixed, see order()
        reward = 1.0 if ok else -1.0
        # Real gap found live, twice: at the old /10_000 scale + 0.3 cap,
        # ANY call slower than 3s already hit the cap — a 5s call and a
        # 60s CPU-only local-SLM call scored identically. Rescaled to
        # /100_000 first; that alone still wasn't enough once real
        # persisted data was inspected live: "groq" had picked up a
        # handful of genuine failures (rate-limited during unusually
        # heavy same-day testing), dragging its learned score down to a
        # statistical tie with "ollama"'s 100%-successful-but-34s-average
        # record — a 0.3 cap only costs a consistently-slow rung ~0.3 of
        # reward, not enough to separate it from an occasionally-failing
        # but fast one. 0.85 (near the full [-1,1] reward range) ensures
        # a rung that is ALWAYS slow scores well below one that mostly
        # succeeds fast, even accounting for a real, non-trivial recent
        # failure rate on the fast side.
        reward -= min(latency_ms / 100_000, 0.85)
        reward = max(-1.0, min(1.0, reward))
        arm_id = f"{tier}::{provider}"
        with self._lock:
            self._core.update(arm_id, _CONTEXT, reward)
            if self._name:
                counts = self._counts.setdefault(arm_id, {"calls": 0, "wins": 0})
                counts["calls"] += 1
                if ok:
                    counts["wins"] += 1
                # Counts persist on the SAME throttle as the bandit itself
                # (in-memory _counts is always accurate for stats() within
                # this process; only the DB write — the actual hot-path
                # cost this class's docstring warns about — is bounded).
                self._updates_since_save += 1
                if self._updates_since_save >= self._SAVE_EVERY:
                    self._updates_since_save = 0
                    bandit_persistence.save(self._name, self._core)
                    self._save_counts()

    def stats(self) -> list[dict]:
        """Current per-(tier, provider) performance — the "who's winning
        for what" dashboard's real data source. Same shape discipline as
        rag_optimizer.stats()/domain_router.stats()."""
        with self._lock:
            arm_ids = set(self._counts) | set(self._core._B)
            rows = []
            for arm_id in sorted(arm_ids):
                tier, provider = arm_id.split("::", 1)
                counts = self._counts.get(arm_id, {"calls": 0, "wins": 0})
                mean = self._core.posterior_mean(arm_id)
                rows.append({
                    "tier": tier,
                    "provider": provider,
                    "calls": counts["calls"],
                    "wins": counts["wins"],
                    "win_rate": round(counts["wins"] / counts["calls"], 4) if counts["calls"] else None,
                    "posterior_mean_weights": [round(float(x), 4) for x in mean] if mean is not None else None,
                })
            return rows


# Process-wide singleton, durable — the first provider-selection layer to
# opt into bandit_persistence.py alongside security_optimizer.py.
provider_optimizer = ProviderOptimizer(name="provider_optimizer")
