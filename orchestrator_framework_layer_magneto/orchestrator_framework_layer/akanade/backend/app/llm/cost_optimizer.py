"""Cost-aware model selection — Thompson Sampling over (tier, provider)
arms, same LinearBanditCore/persistence shape as llm/provider_optimizer.py,
with real dollar cost added to the reward. provider_optimizer.py's own
docstring is explicit that its reward is "pure success+latency... no
cost term at all today" — this module is that missing term, COMPOSED
with (not replacing) provider_optimizer's own ordering: see
llm/router.py's TierClient.chat() for exactly where the two combine.
Same "augments, never replaces the safety net" rule every optimizer in
this codebase follows — this never removes a candidate, never lets the
"rule" floor move ahead of a real provider, and is applied before the
policy-gate's own force_local/force_hosted resorts so a safety override
always wins as the outer sort key (Python's stable sort keeps this
module's cost-aware order as the tie-break within each group).

Also the "predict cost before execution" half of the ask — nothing
else in this backend estimates a prompt's token count or dollar cost
before a provider is actually called (confirmed by investigation: zero
hits for any such function anywhere else in this codebase).
estimate() gives a per-candidate-provider forecast using
pricing_table.py's estimate_prompt_tokens()/rates_for() — real inputs,
one honest assumption (a completion-length ratio, since the real
completion doesn't exist yet to measure).

Quality is tracked via the same signal already available synchronously
inside TierClient.chat(): whether the call actually succeeded, and
whether the provider's own reply was truncated (LLMResult.truncated) —
an HTTP-successful-but-cut-off reply is real evidence of lower quality
even though `ok` is True. This deliberately does NOT need the deferred-
finalize wiring llm/token_optimizer.py requires (that module has to
wait for retried/out_of_scope because a trimmed tool schema can silently
"succeed" at the HTTP level while still picking the wrong tool) — a
provider's own success/truncation signal has no such ambiguity, so this
stays fully synchronous, the same complexity tier as provider_optimizer.py.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from ..db import store as db_store
from ..framework import bandit_persistence
from ..framework.bandit import LinearBanditCore
from . import pricing_table

_FEATURE_DIM = 1
_CONTEXT = [1.0]
_FLOOR_PROVIDER = "rule"
_COUNTS_TABLE = "CostOptimizerCounts"

# How strongly a candidate's PREDICTED cost pulls the RL score toward it
# before any real outcome is known — a pure cold-start prior, same
# "easily overridden by real evidence" rule bandit.py's own ensure()
# uses for every seed_utility elsewhere in this codebase. Cost is
# normalized against the priciest candidate in the SAME call so this
# number means the same thing regardless of which providers happen to
# be available this turn.
#
# Real bug found live: at the old weight (1.5), this seed alone was
# enough to override provider_optimizer.py's own cold-start ordering
# (which already correctly seeds hosted rungs ahead of "ollama" per
# CHAINS' fixed order) — order() runs AFTER provider_optimizer's, so
# this module's free-provider bias silently won on every fresh process
# start, before either bandit had a single real outcome to learn from.
# Confirmed live: a fresh restart's first "list my tasks"/"fetch price
# history" calls landed on local (CPU-only) Ollama and took 15-37
# seconds, purely from this cold-start seed, with zero real evidence
# behind the pick. Lowered so cost is a gentle tie-break among
# similarly-fast candidates, never something that overrides a real,
# already-correct hosted-first default before real data exists —
# matching the reward function's own "cost matters, but never more than
# whether the call actually worked" rule, now true for the seed too.
_COST_SEED_WEIGHT = 0.3

# A genuinely free candidate (predicted_cost_usd == 0 — Ollama, a $0-
# configured admin rate) gets THIS seed instead of the cost-ratio
# formula above — see order()'s own comment for why. Empirically
# calibrated (not a token nudge): at 0.0 (neutral), a free-vs-hosted
# cold start was still a near coin-flip (Thompson Sampling's own
# exploration variance dominates two arms with equal seeded means);
# -1.0 was the smallest magnitude that reliably favored the (real,
# repeatedly measured) faster hosted option roughly 80%+ of cold-start
# draws while still leaving real room for local to win and start
# accumulating genuine evidence, rather than being hard-disabled.
_FREE_PROVIDER_COLD_START_SEED = -1.0
# Real per-outcome reward weights — cost matters, but never more than
# whether the call actually worked, matching every other optimizer's
# "success is the dominant term" convention in this codebase.
_COST_PENALTY_CAP = 0.5
# Same real-data finding as provider_optimizer.py's identical fix: a
# 0.2 cap only costs a consistently-slow rung ~0.2 of reward, not
# enough to separate it from a fast rung that has picked up a real
# (same-day, testing-induced) failure or two. 0.85 ensures an ALWAYS-
# slow rung scores well below a mostly-fast one even accounting for a
# non-trivial recent failure rate on the fast side.
_LATENCY_PENALTY_CAP = 0.85
_TRUNCATED_PENALTY = 0.25
# A completion is typically shorter than the prompt for most chat-style
# turns in this codebase (PM/task lookups, short trading answers) — a
# fixed ratio, not a claim of precision; only used to turn a prompt-
# token estimate into a full request's predicted cost BEFORE any real
# completion exists to measure.
_ASSUMED_COMPLETION_RATIO = 0.5


@dataclass
class CostEstimate:
    """One candidate provider's pre-execution forecast — estimate()'s
    return shape."""

    provider: str
    predicted_prompt_tokens: int
    predicted_completion_tokens: int
    predicted_cost_usd: float


class CostOptimizer:
    # Same throttled-persistence reasoning as provider_optimizer.py:
    # this updates on every LLM call, so saving every single update
    # would add a real DB write to the hot path of every model call.
    _SAVE_EVERY = 20

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.4,
                name: str | None = None):
        """`name` opts into durable persistence (bandit_persistence.py) —
        None (the default) is pure in-memory, what every unit test uses."""
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._lock = threading.Lock()
        self._name = name
        self._updates_since_save = 0
        if name:
            bandit_persistence.load(name, self._core)
        self._counts: dict[str, dict] = self._load_counts() if name else {}

    def _load_counts(self) -> dict[str, dict]:
        row = db_store(_COUNTS_TABLE).get(self._name, "COUNTS")
        return (row or {}).get("arms", {})

    def _save_counts(self) -> None:
        db_store(_COUNTS_TABLE).put(self._name, "COUNTS", {"arms": self._counts})

    def estimate(self, messages: list[dict], provider_names: list[str]) -> list[CostEstimate]:
        """One forecast per candidate provider, BEFORE any of them is
        actually called. Real inputs (a chars/4 token estimate, each
        provider's own real-or-admin-configured $/1k rate), one honest
        assumption for the thing that genuinely can't be known yet
        (completion length)."""
        prompt_tokens = pricing_table.estimate_prompt_tokens(messages)
        completion_tokens = max(1, int(prompt_tokens * _ASSUMED_COMPLETION_RATIO))
        return [
            CostEstimate(
                provider=name,
                predicted_prompt_tokens=prompt_tokens,
                predicted_completion_tokens=completion_tokens,
                predicted_cost_usd=pricing_table.estimate_cost_usd(name, prompt_tokens, completion_tokens),
            )
            for name in provider_names
        ]

    def order(self, tier: str, estimates: list[CostEstimate]) -> list[str]:
        """Thompson-sampled ranking, seeded toward the CHEAPER
        candidates — never removes one, only orders them, so a
        genuinely-better-but-pricier provider can still win once real
        outcomes justify it. The "rule" floor, if present, always moves
        to the end regardless of cost (it's never a real model call to
        prefer over)."""
        real = [e for e in estimates if e.provider != _FLOOR_PROVIDER]
        if len(real) > 1:
            max_cost = max((e.predicted_cost_usd for e in real), default=0.0)
            with self._lock:
                scored = [
                    (
                        self._core.score(
                            f"{tier}::{e.provider}", _CONTEXT,
                            # Real bug found live: a genuinely FREE candidate
                            # (predicted_cost_usd == 0 — Ollama, a $0-configured
                            # admin rate) used to get the SAME maximally-
                            # favorable seed as merely "the cheapest of the
                            # paid options" would. On this deployment $0 means
                            # "CPU-only local model, repeatedly measured at
                            # 15-37 seconds for an ordinary tool-calling turn,"
                            # not "a slightly cheaper hosted model" — treating
                            # both the same let this cold-start seed alone
                            # override provider_optimizer.py's own (correctly
                            # hosted-first) cold-start ordering on every fresh
                            # process start, before either bandit had a single
                            # real outcome. A free candidate now seeds NEUTRAL
                            # (0.0) instead of maximally favorable — a real
                            # non-zero-cost saving still nudges an otherwise-
                            # similar hosted option ahead of a pricier one,
                            # but "free" alone no longer manufactures a
                            # preference this reward function has no latency
                            # evidence yet to justify.
                            seed_utility=(_FREE_PROVIDER_COLD_START_SEED if e.predicted_cost_usd == 0
                                         else (1.0 - (e.predicted_cost_usd / max_cost if max_cost else 0.0))
                                         * _COST_SEED_WEIGHT),
                        )[0],
                        e.provider,
                    )
                    for e in real
                ]
            real_names = [name for _, name in sorted(scored, key=lambda t: -t[0])]
        else:
            real_names = [e.provider for e in real]
        floor = [_FLOOR_PROVIDER] if any(e.provider == _FLOOR_PROVIDER for e in estimates) else []
        return real_names + floor

    def record_outcome(self, tier: str, provider: str, *, ok: bool, latency_ms: float,
                       truncated: bool, prompt_tokens: int, completion_tokens: int) -> float:
        """Real per-call reward: success is dominant (matches every
        other optimizer here), then a capped latency penalty (same
        formula as provider_optimizer.py), a truncation penalty (an
        HTTP-successful-but-cut-off reply is lower quality even though
        `ok` is True), then a real $-based cost penalty — this module's
        actual new contribution. Returns the real dollar cost this call
        incurred (0.0 for the floor, or when no usage was reported)."""
        if provider == _FLOOR_PROVIDER:
            return 0.0
        actual_cost = pricing_table.estimate_cost_usd(provider, prompt_tokens, completion_tokens)
        reward = 1.0 if ok else -1.0
        # Same real gap, same fix as provider_optimizer.py's identical
        # formula: at /10_000 the 0.2 cap saturated by 2s, so a 34s-
        # average local SLM and a sub-1s hosted call scored almost
        # identically — confirmed live (ollama 34.2s avg vs groq 0.9s avg,
        # posteriors only ~0.04 apart), letting a much slower rung win
        # roughly half the time on Thompson Sampling noise alone.
        # /100_000 pushes the cap out to 20s so real latency differences
        # across the observed 1s-60s+ range actually move the reward.
        reward -= min(latency_ms / 100_000, _LATENCY_PENALTY_CAP)
        if truncated:
            reward -= _TRUNCATED_PENALTY
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            cost_per_1k_this_call = actual_cost / (total_tokens / 1000)
            # A provider whose own rate is 0 (Ollama, a $0-configured
            # admin rate) never incurs this penalty — genuinely free,
            # not merely "cheap this call."
            if cost_per_1k_this_call > 0:
                reward -= min(cost_per_1k_this_call / 20.0, _COST_PENALTY_CAP)
        reward = max(-1.0, min(1.0, reward))
        arm_id = f"{tier}::{provider}"
        with self._lock:
            self._core.update(arm_id, _CONTEXT, reward)
            if self._name:
                counts = self._counts.setdefault(arm_id, {
                    "calls": 0, "wins": 0, "total_cost_usd": 0.0, "total_prompt_tokens": 0,
                    "total_completion_tokens": 0, "total_latency_ms": 0.0, "truncated_count": 0,
                })
                counts["calls"] += 1
                if ok:
                    counts["wins"] += 1
                counts["total_cost_usd"] += actual_cost
                counts["total_prompt_tokens"] += prompt_tokens
                counts["total_completion_tokens"] += completion_tokens
                counts["total_latency_ms"] += latency_ms
                if truncated:
                    counts["truncated_count"] += 1
                self._updates_since_save += 1
                if self._updates_since_save >= self._SAVE_EVERY:
                    self._updates_since_save = 0
                    bandit_persistence.save(self._name, self._core)
                    self._save_counts()
        return actual_cost

    def stats(self) -> list[dict]:
        """Per-(tier, provider): real tracked cost/tokens/latency/quality
        — the actual data source for a cost dashboard. Same shape
        discipline as provider_optimizer.stats()/domain_router.stats()."""
        with self._lock:
            arm_ids = set(self._counts) | set(self._core._B)
            rows = []
            for arm_id in sorted(arm_ids):
                tier, provider = arm_id.split("::", 1)
                c = self._counts.get(arm_id, {
                    "calls": 0, "wins": 0, "total_cost_usd": 0.0, "total_prompt_tokens": 0,
                    "total_completion_tokens": 0, "total_latency_ms": 0.0, "truncated_count": 0,
                })
                mean = self._core.posterior_mean(arm_id)
                calls = c["calls"]
                rows.append({
                    "tier": tier,
                    "provider": provider,
                    "calls": calls,
                    "win_rate": round(c["wins"] / calls, 4) if calls else None,
                    "total_cost_usd": round(c["total_cost_usd"], 6),
                    "avg_cost_usd_per_call": round(c["total_cost_usd"] / calls, 6) if calls else 0.0,
                    "total_prompt_tokens": c["total_prompt_tokens"],
                    "total_completion_tokens": c["total_completion_tokens"],
                    "avg_latency_ms": round(c["total_latency_ms"] / calls, 1) if calls else 0.0,
                    "truncated_rate": round(c["truncated_count"] / calls, 4) if calls else None,
                    "posterior_mean_weights": [round(float(x), 4) for x in mean] if mean is not None else None,
                })
            return rows

    def savings_summary(self) -> dict:
        """The whole-system "how much has this layer actually saved"
        number — real total spend so far vs. a real counterfactual (what
        the SAME total token volume would have cost if every call had
        gone to the single priciest provider this system KNOWS about,
        instead). Compares against every rate in pricing_table's own
        table, not just providers that happened to get called — a
        deployment that has been routing everything to a free local
        model still has a real, honest "here's what that would have
        cost on the priciest real alternative" number, not an
        artificial zero just because the pricier option was never
        actually used."""
        rows = self.stats()
        total_cost = sum(r["total_cost_usd"] for r in rows)
        total_prompt_tokens = sum(r["total_prompt_tokens"] for r in rows)
        total_completion_tokens = sum(r["total_completion_tokens"] for r in rows)
        total_tokens = total_prompt_tokens + total_completion_tokens
        if not rows or total_tokens == 0:
            return {"total_cost_usd": 0.0, "total_tokens": 0, "counterfactual_cost_usd": 0.0,
                   "estimated_savings_usd": 0.0, "estimated_savings_pct": 0.0}
        counterfactual = pricing_table.counterfactual_cost_usd(total_prompt_tokens, total_completion_tokens)
        savings = max(0.0, counterfactual - total_cost)
        return {
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "counterfactual_cost_usd": round(counterfactual, 6),
            "estimated_savings_usd": round(savings, 6),
            "estimated_savings_pct": round(savings / counterfactual * 100, 2) if counterfactual else 0.0,
        }


def cost_and_savings_for_call(provider: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """The "show savings after each model call" number, for exactly ONE
    real call — same baseline (pricing_table.counterfactual_cost_usd)
    savings_summary() uses for the whole-system total, just applied to
    a single call's own token counts. Called once per real LLM call,
    right where TierClient.chat() already computes that call's actual
    cost (see llm/router.py) — every platform that already renders
    tier_log (web, mobile, VS Code) gets this for free, no new endpoint
    or client-side pricing logic needed."""
    actual_cost = pricing_table.estimate_cost_usd(provider, prompt_tokens, completion_tokens)
    counterfactual = pricing_table.counterfactual_cost_usd(prompt_tokens, completion_tokens)
    savings = max(0.0, counterfactual - actual_cost)
    return {
        "cost_usd": round(actual_cost, 6),
        "counterfactual_cost_usd": round(counterfactual, 6),
        "savings_usd": round(savings, 6),
        "savings_pct": round(savings / counterfactual * 100, 2) if counterfactual else 0.0,
    }


# Process-wide singleton, durable — same lifecycle as provider_optimizer.
cost_optimizer = CostOptimizer(name="cost_optimizer")
