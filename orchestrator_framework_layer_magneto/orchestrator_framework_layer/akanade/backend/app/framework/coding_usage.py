"""Usage/billing tracking for the Secondary Pipeline (see
llm/coding_pipeline.py) — one persisted row per completed VS Code
coding-agent turn: which user, which model actually answered, real
tokens/cost, whether this turn escalated past the default tier, and a
credit figure derived from that real cost.

Deliberately NOT a new cost-computation path: every number here is
aggregated straight from a turn's own tier_log entries (see
llm/router.py's TierClient.chat(), which already appends
{provider, costUsd, usage, ms, ...} per real model call, the exact same
figures the Cost/Token Optimizer dashboards already read) — this module
only sums what already ran, never recomputes or estimates anything.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..db import store
from ..llm.coding_pipeline import CODING_CASCADE_ORDER

# 1 credit = $0.01 — an arbitrary but fixed, documented conversion. Change
# this ONE constant to re-price every FUTURE row; existing rows keep
# whatever credits value was computed at the time, same as a real invoice
# isn't retroactively rewritten when a price list changes.
USD_PER_CREDIT = 0.01

# The Secondary Pipeline's own Default tier (see coding_pipeline.py) —
# whichever provider actually answers is compared against this ONE key to
# decide "default" vs "escalation," not against admin/global pipeline state.
DEFAULT_TIER_PROVIDER = CODING_CASCADE_ORDER[0]


def record_usage(*, user_id: str, session_id: str, task_category: str,
                 tier_log: list[dict]) -> dict | None:
    """Aggregates ONE completed turn's tier_log into one CodingUsage row.
    Returns None (records nothing billable) for a turn with no real model
    call in it — a pure cache hit, or a confirmed-write turn that made no
    new call this time around."""
    real_calls = [e for e in (tier_log or []) if e.get("provider") and not e.get("cacheHit")]
    if not real_calls:
        return None

    input_tokens = sum((e.get("usage") or {}).get("prompt_tokens", 0) or 0 for e in real_calls)
    output_tokens = sum((e.get("usage") or {}).get("completion_tokens", 0) or 0 for e in real_calls)
    cost_usd = sum(e.get("costUsd", 0.0) or 0.0 for e in real_calls)

    final_provider = real_calls[-1]["provider"]
    escalated = final_provider != DEFAULT_TIER_PROVIDER
    escalation_count = sum(1 for e in real_calls if e["provider"] != DEFAULT_TIER_PROVIDER)

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "user_id": user_id,
        "session_id": session_id,
        "task_category": task_category,
        "model": final_provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "credits": round(cost_usd / USD_PER_CREDIT, 2),
        "cost_usd": round(cost_usd, 6),
        "escalated": escalated,
        "escalation_count": escalation_count,
        "tier": "escalation" if escalated else "default",
        "requests": 1,
        "timestamp": timestamp,
    }
    store("CodingUsage").put(f"USAGE#{user_id}", f"{timestamp}#{uuid.uuid4().hex[:8]}", row)
    return row


def list_usage_for_user(user_id: str, limit: int = 2000) -> list[dict]:
    rows = store("CodingUsage").scan(pk_prefix=f"USAGE#{user_id}")
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows[:limit]


def summarize(rows: list[dict]) -> dict:
    """Summary cards + the analysis figures the dashboard needs — pure
    aggregation over already-recorded rows, no new tracking."""
    total_requests = len(rows)
    total_credits = round(sum(r.get("credits", 0.0) for r in rows), 2)
    total_cost = round(sum(r.get("cost_usd", 0.0) for r in rows), 6)
    total_tokens = sum(r.get("total_tokens", 0) for r in rows)
    escalations = sum(1 for r in rows if r.get("escalated"))
    default_count = total_requests - escalations

    by_model: dict[str, int] = {}
    for r in rows:
        by_model[r.get("model", "unknown")] = by_model.get(r.get("model", "unknown"), 0) + 1

    return {
        "totalCreditsUsed": total_credits,
        "totalTokens": total_tokens,
        "totalRequests": total_requests,
        "totalCostUsd": total_cost,
        "escalationCount": escalations,
        "defaultCount": default_count,
        "escalationPct": round(100 * escalations / total_requests, 1) if total_requests else 0.0,
        "avgCostPerRequest": round(total_cost / total_requests, 6) if total_requests else 0.0,
        "avgTokensPerRequest": round(total_tokens / total_requests, 1) if total_requests else 0.0,
        "requestsByModel": by_model,
    }
