"""Reads Algo Akanade's durable optimization-metrics files directly off disk —
the SAME app/db.py LocalStore JSON files Algo Akanade's own backend writes
(no shared Python import between the two processes/services; just the
one on-disk contract: `{"PK||SK": {...row..., "PK":..., "SK":...}}`, the
same shape LocalStore.put() has always stamped). This is what lets the
dashboard show real numbers even when Algo Akanade's own process isn't
running right now — see README_OPTIMIZATION_DASHBOARD.md § Architecture.

Two tables it reads:
  OptimizationEvents — one row per LLM call (app/optimization_events.py),
                       PK = "YYYY-MM-DD", SK = "EVENT#<iso-ts>#<id>".
  BanditState        — one row per named optimizer's persisted RL
                       posterior (framework/bandit_persistence.py).
                       Only provider_optimizer opts into this today —
                       optimization_router/token_optimizer are in-memory-
                       only in this Algo Akanade version, so their live
                       arm/reward numbers come from the event log above,
                       not from this file (see rl_state.py's docstring).
"""
from __future__ import annotations

import datetime as dt
import json
import os

from .config import settings


def _load_table(table: str) -> dict:
    path = os.path.join(settings().ALGO_AKANADE_DATA_DIR, f"{table}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def list_events_for_date(date: str) -> list[dict]:
    """`date` is "YYYY-MM-DD". Chronological order."""
    rows = _load_table("OptimizationEvents")
    events = [v for v in rows.values() if v.get("PK") == date and v.get("SK", "").startswith("EVENT#")]
    return sorted(events, key=lambda r: r["occurred_at"])


def list_recent_dates(n: int = 14) -> list[str]:
    """Every distinct date with at least one event, most recent first,
    capped at `n` — what the daily-trend view iterates over."""
    rows = _load_table("OptimizationEvents")
    dates = sorted({v["PK"] for v in rows.values() if v.get("SK", "").startswith("EVENT#")}, reverse=True)
    return dates[:n]


def summarize(events: list[dict]) -> dict:
    """Same aggregation Algo Akanade's own optimization_events.summarize()
    computes — duplicated here (not imported) because this is a
    genuinely separate service/deployment unit; kept in lockstep by
    both reading from the identical row shape logged by
    app/optimization_events.py.log_call()."""
    n = len(events)
    if n == 0:
        return {"n_calls": 0, "success_rate": None, "avg_latency_ms": None,
                "fallback_rate": None, "by_provider": {}, "by_arm": {}}
    n_ok = sum(1 for e in events if e["ok"])
    n_fallback = sum(1 for e in events if e["fallback_used"])
    avg_latency = sum(e["latency_ms"] for e in events) / n

    by_provider: dict[str, dict] = {}
    by_arm: dict[str, dict] = {}
    for e in events:
        p = by_provider.setdefault(e["provider"], {"n_calls": 0, "n_ok": 0, "latency_sum": 0.0})
        p["n_calls"] += 1
        p["n_ok"] += int(e["ok"])
        p["latency_sum"] += e["latency_ms"]
        a = by_arm.setdefault(e["arm_id"], {"n_calls": 0, "n_ok": 0, "latency_sum": 0.0})
        a["n_calls"] += 1
        a["n_ok"] += int(e["ok"])
        a["latency_sum"] += e["latency_ms"]

    def _finalize(d: dict) -> dict:
        return {k: {"n_calls": v["n_calls"],
                    "success_rate": round(v["n_ok"] / v["n_calls"], 3),
                    "avg_latency_ms": round(v["latency_sum"] / v["n_calls"], 1)}
               for k, v in d.items()}

    return {
        "n_calls": n,
        "success_rate": round(n_ok / n, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "fallback_rate": round(n_fallback / n, 3),
        "by_provider": _finalize(by_provider),
        "by_arm": _finalize(by_arm),
    }


def daily_report(date: str) -> dict:
    """This day's summary plus the delta against the previous calendar
    day — the "daily improvement / feedback loop" view: a falling
    avg_latency_ms or a rising success_rate day-over-day is the RL
    optimizers' own learning showing up in real outcomes, not a
    self-reported claim."""
    today_events = list_events_for_date(date)
    today = summarize(today_events)

    prev_date = (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()
    prev_events = list_events_for_date(prev_date)
    prev = summarize(prev_events)

    def _delta(key: str) -> float | None:
        if today[key] is None or prev[key] is None:
            return None
        return round(today[key] - prev[key], 3)

    return {
        "date": date,
        "today": today,
        "previous_date": prev_date,
        "previous": prev,
        "delta": {
            "success_rate": _delta("success_rate"),
            "avg_latency_ms": _delta("avg_latency_ms"),
            "fallback_rate": _delta("fallback_rate"),
            "n_calls": (today["n_calls"] - prev["n_calls"]) if prev["n_calls"] is not None else None,
        },
    }


def provider_optimizer_posteriors() -> dict[str, dict[str, float]] | None:
    """Reads provider_optimizer's persisted Thompson Sampling state
    (BanditState.json, arm ids "{tier}::{provider}") and returns the
    posterior mean per (tier, provider) — grouped by tier, highest mean
    first, matching what provider_optimizer.order() actually acts on.
    Returns None if nothing has been persisted yet (a fresh deployment
    that hasn't made 20 real calls — see provider_optimizer.py's
    _SAVE_EVERY). The context vector this optimizer uses is 1-dimensional
    ([1.0]), so `B` is a 1x1 matrix and posterior_mean = f[0] / B[0][0] —
    plain arithmetic, no numpy dependency needed just to read it back."""
    rows = _load_table("BanditState")
    row = rows.get("OPTIMIZER#provider_optimizer||STATE")
    if not row or not row.get("arms"):
        return None
    by_tier: dict[str, dict[str, float]] = {}
    for arm_id, snap in row["arms"].items():
        if "::" not in arm_id:
            continue
        tier, provider = arm_id.split("::", 1)
        b = snap["B"][0][0]
        f = snap["f"][0]
        mean = f / b if b else 0.0
        by_tier.setdefault(tier, {})[provider] = round(mean, 4)
    return {tier: dict(sorted(providers.items(), key=lambda kv: -kv[1]))
           for tier, providers in by_tier.items()}
