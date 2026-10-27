"""Durable log of every LLM-call optimization decision — the source of
truth for the performance_optimizer dashboard (see
orchestrator_framework_layer/performance_optimizer/dashboard/). Every
call `llm/router.py`'s TierClient.chat() makes (success OR failure) gets
ONE row here: which provider/tier, which RL-selected optimization arm
(llm/optimization_router.py), whether it succeeded, how long it took,
and whether it was a fallback (not the tier's first-tried rung this
call). Reuses app/db.py's existing PK/SK store — same mechanism
SecurityEvents/OAuthConnections/ChatSessions already use, no new
persistence layer, so this dashboard reads from the same DATA_DIR any
other admin surface in this deployment already does.

Deliberately narrow schema — this is a metrics feed, not a transcript.
No message content is ever stored here (that already lives in
ChatSessions under the user's own access controls); this table only
ever holds routing/timing/outcome facts an admin needs to see "is the
optimizer actually helping," never anything a user typed or the model
replied.
"""
from __future__ import annotations

import datetime as dt
import uuid

from .db import store

_TABLE = "OptimizationEvents"


def _date_key(occurred_at: dt.datetime) -> str:
    return occurred_at.strftime("%Y-%m-%d")


def log_call(*, node: str, tier: str, provider: str, arm_id: str, ok: bool,
            latency_ms: float, fallback_used: bool) -> None:
    """Never raises on a storage hiccup — a failed metrics write must
    not break the chat reply it's describing, same fail-soft rule
    security_events.log_event()/bandit_persistence.save() already
    follow."""
    now = dt.datetime.now(dt.timezone.utc)
    event_id = uuid.uuid4().hex[:12]
    row = {
        "node": node,
        "tier": tier,
        "provider": provider,
        "arm_id": arm_id,
        "ok": ok,
        "latency_ms": round(latency_ms, 1),
        "fallback_used": fallback_used,
        "occurred_at": now.isoformat(),
    }
    try:
        store(_TABLE).put(_date_key(now), f"EVENT#{now.isoformat()}#{event_id}", row)
    except Exception:  # noqa: BLE001
        pass


def _strip_storage_keys(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("PK", "SK")}


def list_events_for_date(date: str) -> list[dict]:
    """`date` is "YYYY-MM-DD". Chronological order."""
    events = store(_TABLE).scan(date, "EVENT#")
    return sorted((_strip_storage_keys(e) for e in events), key=lambda r: r["occurred_at"])


def summarize(events: list[dict]) -> dict:
    """Aggregate a day's (or any) event list into the numbers the
    dashboard actually plots — one place both the live summary endpoint
    and the daily-report endpoint call, so the two views can never
    silently disagree on how a rate is computed."""
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
