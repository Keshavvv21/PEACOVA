"""A tiny latency/error tracker feeding the auto framework picker
(chat.py's default — no ?framework= or ?framework=auto), mirroring
llm/router.py's TierClient fallback-chain philosophy one level up: at the
FRAMEWORK level instead of the model-provider level. Whichever of the
three tracks (native, langgraph, adk) has been fastest/most reliable so
far is MOST LIKELY to be picked; if the picked track errors, the
auto-picker's own fallback in chat.py tries the next distinct track in
TRACKS before giving up.

choose() below is a softmax (Boltzmann) multi-armed-bandit sampler, not a
deterministic argmax: every track's per-call utility (reliability ÷
latency) is turned into a probability via softmax, and the actual pick is
a weighted random draw over that distribution — see choose()'s docstring
for the exact formula. That replaces the previous scheme (always take the
single best track, except a flat 10% chance of trying "the" runner-up)
with something properly continuous: a track's share of traffic scales
smoothly with how much better or worse it's measured, nothing is ever
locked out at exactly 0%, and nothing needs a hardcoded exploration rate.

adk_agent.py's Phase 2 scaffold is finished (real ADK Runner/
SessionService/LlmAgent machinery, see that module's own docstring) —
"adk" earns its place in rotation the same way it would in a real
bake-off: it's placed LAST in TRACKS below, so a cold start's
fewest-calls-first rule (see choose()) still tries native/langgraph
first when all three are equally unproven, and adk only pulls ahead of
them on a genuine measured edge, never by tie-breaking.

In-memory only (per-process, reset on restart) — the right shape for a
reference deployment; a shared cache/table would be the production
version if this needed to survive restarts or run behind multiple
workers.
"""
from __future__ import annotations

import math
import random
import threading

TRACKS = ("native", "langgraph", "adk")
_MIN_SAMPLES = 3  # don't let one lucky/unlucky first call decide everything
# Softmax temperature: how sharply probability mass concentrates on the
# best-scoring track, applied to utilities AFTER min-max normalization to
# [0, 1] (see probabilities()) — deliberately not applied to the raw
# reliability/ms utility directly, since that value's absolute scale
# depends entirely on how many milliseconds a call happens to take (a
# 50ms-vs-600ms gap and a 5ms-vs-60ms gap are the same RELATIVE gap, and
# should produce the same-shaped distribution either way). Lower
# temperature = sharper (closer to "always pick the best"); higher =
# closer to uniform. 0.25 keeps the best track favored (comfortably the
# majority of traffic) without collapsing the others to a vanishing
# probability — see this module's tests for the actual numbers.
_TEMPERATURE = 0.25
_lock = threading.Lock()
_stats: dict[str, dict] = {t: {"calls": 0, "errors": 0, "ms_total": 0} for t in TRACKS}


def record(track: str, ms: int, error: bool) -> None:
    if track not in _stats:
        return
    with _lock:
        s = _stats[track]
        s["calls"] += 1
        s["ms_total"] += ms
        if error:
            s["errors"] += 1


def _utility(s: dict) -> float:
    """reliability ÷ latency — higher is better. reliability is
    (1 - error_rate) in [0, 1]; the +1ms in the denominator keeps this
    finite if avg latency is ever ~0 (an instant/mocked provider)."""
    calls = s["calls"]
    reliability = 1 - s["errors"] / calls
    avg_ms = s["ms_total"] / calls
    return reliability / (avg_ms + 1.0)


def probabilities() -> dict[str, float]:
    """The softmax distribution choose() samples from, for any TRACK with
    enough samples to have a trustworthy utility. Exposed on its own (also
    used by choose()) so an admin can see WHY the picker is favoring one
    track over another, not just which one it happens to have picked."""
    with _lock:
        snap = {t: dict(v) for t, v in _stats.items()}
    scored = [t for t in TRACKS if snap[t]["calls"] > 0]
    if not scored:
        return {t: 1 / len(TRACKS) for t in TRACKS}  # true cold start: uniform

    utilities = {t: _utility(snap[t]) for t in scored}
    # Min-max normalize to [0, 1] before applying temperature — see
    # _TEMPERATURE's comment on why: this makes the softmax's sharpness
    # depend only on each track's RELATIVE standing (worst=0, best=1),
    # not on whatever absolute ms scale the raw utilities happen to sit
    # at. A single scored track (or a dead tie) has zero spread, in which
    # case it's just a fair uniform split among the scored tracks.
    lo, hi = min(utilities.values()), max(utilities.values())
    spread = hi - lo
    normalized = {t: ((u - lo) / spread if spread > 0 else 0.0) for t, u in utilities.items()}
    weights = {t: math.exp(n / _TEMPERATURE) for t, n in normalized.items()}
    total = sum(weights.values())
    probs = {t: w / total for t, w in weights.items()}
    for t in TRACKS:
        probs.setdefault(t, 0.0)  # an unscored (0-call) track gets 0% until it has data
    return probs


def choose(default: str = "native") -> str:
    """A weighted random draw over probabilities() — NOT "always the
    best": every track's share of the draw scales continuously with how
    much better or worse it's measured (via softmax over reliability ÷
    latency), so a clearly-better track gets most of the traffic while a
    weaker one still gets picked occasionally and keeps being measured,
    with no fixed exploration rate to tune.

    Below _MIN_SAMPLES for any track, this instead returns whichever
    under-sampled track has seen the FEWEST calls so far (`default` only
    breaking a true cold-start tie, 0 calls each) — so every track gets a
    fair, deliberate warm-up on real traffic before the probability model
    trusts its numbers at all."""
    with _lock:
        snap = {t: dict(v) for t, v in _stats.items()}

    under_sampled = [t for t in TRACKS if snap[t]["calls"] < _MIN_SAMPLES]
    if under_sampled:
        fewest = min(snap[t]["calls"] for t in under_sampled)
        tied = [t for t in under_sampled if snap[t]["calls"] == fewest]
        return default if default in tied else tied[0]

    probs = probabilities()
    draw = random.random()
    cumulative = 0.0
    for t in TRACKS:
        cumulative += probs[t]
        if draw <= cumulative:
            return t
    return TRACKS[-1]  # floating-point safety net (cumulative should hit 1.0)


def snapshot() -> dict:
    with _lock:
        snap = {t: {**v, "avg_ms": round(v["ms_total"] / max(v["calls"], 1), 1)}
                for t, v in _stats.items()}
    probs = probabilities()
    for t in TRACKS:
        snap[t]["pick_probability"] = round(probs[t], 4)
    return snap
