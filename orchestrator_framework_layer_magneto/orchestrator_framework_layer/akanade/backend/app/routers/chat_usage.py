"""The general PM/chat track's usage dashboard — same shape as
routers/coding_usage.py (which serves the VS Code coding agent), for the
"my usage" self-service view every user gets (GET /api/chat-usage), plus
an Admin-only, all-users view for the admin dashboard (GET
/api/admin/chat-usage), which adds a per-user breakdown ("usage per
user") on top of the same summary/trend/table shape. See
framework/chat_usage.py for how each row was recorded.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from ..auth import current_user, require_role
from ..framework import chat_usage

router = APIRouter(tags=["chat-usage"])

_RANGES = {"today": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30), "all": None}


def _day_bucket(iso_timestamp: str) -> str:
    return iso_timestamp[:10]  # "YYYY-MM-DD" — ISO timestamps sort/slice correctly as strings


def _filter_range(rows: list[dict], range_key: str) -> list[dict]:
    delta = _RANGES[range_key]
    if delta is None:
        return rows
    since = (datetime.now(timezone.utc) - delta).isoformat()
    return [r for r in rows if r.get("timestamp", "") >= since]


def _build_response(rows: list[dict], range_key: str) -> dict:
    summary = chat_usage.summarize(rows)

    by_day: dict[str, dict] = defaultdict(lambda: {"credits": 0.0, "costUsd": 0.0, "tokens": 0,
                                                    "requests": 0, "escalations": 0})
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
        {"timestamp": r.get("timestamp"), "framework": r.get("framework"), "model": r.get("model"),
         "userId": r.get("user_id"), "inputTokens": r.get("input_tokens"), "outputTokens": r.get("output_tokens"),
         "credits": r.get("credits"), "costUsd": r.get("cost_usd"), "escalated": r.get("escalated")}
        for r in rows
    ]

    return {"range": range_key, "summary": summary, "trend": trend, "table": table}


@router.get("/api/chat-usage")
def get_chat_usage(range: str = "30d", user: dict = Depends(current_user)) -> dict:
    """Self-service 'my usage' — always scoped to the calling user's own
    rows, same convention as GET /api/coding-usage."""
    if range not in _RANGES:
        range = "30d"
    rows = _filter_range(chat_usage.list_usage_for_user(user["sub"]), range)
    return _build_response(rows, range)


@router.get("/api/admin/chat-usage")
def get_admin_chat_usage(range: str = "30d", _: dict = Depends(require_role("Admin"))) -> dict:
    """Admin-wide view across every user — feeds the admin dashboard's
    'usage per user' breakdown, on top of the same summary/trend/table
    shape the self-service view already uses."""
    if range not in _RANGES:
        range = "30d"
    rows = _filter_range(chat_usage.list_usage_all(), range)
    response = _build_response(rows, range)
    response["byUser"] = chat_usage.summarize_by_user(rows)
    return response
