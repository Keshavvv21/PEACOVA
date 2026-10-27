"""The Secondary Pipeline's usage/billing dashboard — GET
/api/coding-usage, one call returning everything
CodingUsageDashboard.tsx needs: summary cards, a daily trend series, a
model-distribution breakdown, and the detailed per-request table. Always
scoped to the CALLING user's own usage (current_user's `sub`) — this is
a self-service "my usage" view, not an admin-wide report across every
user; see framework/coding_usage.py for how each row was recorded.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from ..auth import current_user
from ..framework import coding_usage

router = APIRouter(prefix="/api/coding-usage", tags=["coding-usage"])

_RANGES = {"today": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "all": None}


def _day_bucket(iso_timestamp: str) -> str:
    return iso_timestamp[:10]  # "YYYY-MM-DD" — ISO timestamps sort/slice correctly as strings


@router.get("")
def get_coding_usage(range: str = "30d", user: dict = Depends(current_user)) -> dict:
    if range not in _RANGES:
        range = "30d"
    all_rows = coding_usage.list_usage_for_user(user["sub"])
    delta = _RANGES[range]
    if delta is not None:
        since = (datetime.now(timezone.utc) - delta).isoformat()
        rows = [r for r in all_rows if r.get("timestamp", "") >= since]
    else:
        rows = all_rows

    summary = coding_usage.summarize(rows)

    # Daily trend — credits/cost/tokens/requests per day, oldest first,
    # so TrendChart-style consumers can plot left-to-right chronologically.
    by_day: dict[str, dict] = defaultdict(lambda: {"credits": 0.0, "costUsd": 0.0, "tokens": 0, "requests": 0,
                                                    "escalations": 0})
    for r in rows:
        bucket = by_day[_day_bucket(r.get("timestamp", ""))]
        bucket["credits"] += r.get("credits", 0.0)
        bucket["costUsd"] += r.get("cost_usd", 0.0)
        bucket["tokens"] += r.get("total_tokens", 0)
        bucket["requests"] += 1
        if r.get("escalated"):
            bucket["escalations"] += 1
    trend = [
        {"date": day, "credits": round(v["credits"], 2), "costUsd": round(v["costUsd"], 6),
         "tokens": v["tokens"], "requests": v["requests"], "escalations": v["escalations"]}
        for day, v in sorted(by_day.items())
    ]

    table = [
        {"timestamp": r.get("timestamp"), "taskCategory": r.get("task_category"), "model": r.get("model"),
         "inputTokens": r.get("input_tokens"), "outputTokens": r.get("output_tokens"),
         "credits": r.get("credits"), "costUsd": r.get("cost_usd"), "escalated": r.get("escalated")}
        for r in rows
    ]

    return {"range": range, "summary": summary, "trend": trend, "table": table}
