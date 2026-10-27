"""Usage tracking for the general PM/chat track (POST /api/chat/message)
— one persisted row per completed turn: which user, which model actually
answered, real tokens/cost, and whether that turn needed a fallback past
its first real attempt. Same "aggregate straight from tier_log, never
recompute or estimate" shape as coding_usage.py (the VS Code coding
agent's own equivalent) — the difference is this track has no single
fixed "Default tier" model to compare against (PM/chat can land on any
of luna/max/deepseek/groq/glm/... depending on the deterministic chain
or ak-router's own live-learned order), so "escalated" here means
"needed more than one real attempt this turn" (tier_log's own
fallbackUsed on the final real call), not "isn't one specific named
provider."
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..db import store

# 1 credit = $0.01 — same fixed, documented conversion coding_usage.py uses.
USD_PER_CREDIT = 0.01


def record_usage(*, user_id: str, session_id: str, framework: str,
                 tier_log: list[dict]) -> dict | None:
    """Aggregates ONE completed chat turn's tier_log into one ChatUsage
    row. Returns None (records nothing billable) for a turn with no real
    model call in it — a pure conversational-cache hit, or a resumed
    confirmation turn with nothing new to bill."""
    real_calls = [e for e in (tier_log or []) if e.get("provider") and not e.get("cacheHit")]
    if not real_calls:
        return None

    input_tokens = sum((e.get("usage") or {}).get("prompt_tokens", 0) or 0 for e in real_calls)
    output_tokens = sum((e.get("usage") or {}).get("completion_tokens", 0) or 0 for e in real_calls)
    cost_usd = sum(e.get("costUsd", 0.0) or 0.0 for e in real_calls)

    final = real_calls[-1]
    final_provider = final["provider"]
    escalated = bool(final.get("fallbackUsed")) or len(real_calls) > 1

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "user_id": user_id,
        "session_id": session_id,
        "framework": framework,
        "model": final_provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "credits": round(cost_usd / USD_PER_CREDIT, 2),
        "cost_usd": round(cost_usd, 6),
        "escalated": escalated,
        "requests": 1,
        "timestamp": timestamp,
    }
    store("ChatUsage").put(f"USAGE#{user_id}", f"{timestamp}#{uuid.uuid4().hex[:8]}", row)
    return row


def list_usage_for_user(user_id: str, limit: int = 2000) -> list[dict]:
    rows = store("ChatUsage").scan(pk_prefix=f"USAGE#{user_id}")
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows[:limit]


def list_usage_all(limit: int = 5000) -> list[dict]:
    """Every user's rows — the admin-wide view (routers/chat_usage.py's
    own /api/admin/chat-usage), since a regular user only ever sees
    their own via list_usage_for_user()."""
    rows = store("ChatUsage").scan(pk_prefix="USAGE#")
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

    by_model: dict[str, int] = {}
    for r in rows:
        by_model[r.get("model", "unknown")] = by_model.get(r.get("model", "unknown"), 0) + 1

    return {
        "totalCreditsUsed": total_credits,
        "totalTokens": total_tokens,
        "totalRequests": total_requests,
        "totalCostUsd": total_cost,
        "escalationCount": escalations,
        "escalationPct": round(100 * escalations / total_requests, 1) if total_requests else 0.0,
        "avgCostPerRequest": round(total_cost / total_requests, 6) if total_requests else 0.0,
        "avgTokensPerRequest": round(total_tokens / total_requests, 1) if total_requests else 0.0,
        "requestsByModel": by_model,
    }


def summarize_by_user(rows: list[dict]) -> list[dict]:
    """Per-user breakdown for the admin dashboard — 'usage per user',
    one row per user_id, sorted by total spend (highest first) so the
    admin sees the heaviest users up top."""
    by_user: dict[str, dict] = {}
    for r in rows:
        uid = r.get("user_id", "unknown")
        u = by_user.setdefault(uid, {"userId": uid, "requests": 0, "credits": 0.0,
                                     "costUsd": 0.0, "tokens": 0})
        u["requests"] += 1
        u["credits"] += r.get("credits", 0.0)
        u["costUsd"] += r.get("cost_usd", 0.0)
        u["tokens"] += r.get("total_tokens", 0)
    out = list(by_user.values())
    for u in out:
        u["credits"] = round(u["credits"], 2)
        u["costUsd"] = round(u["costUsd"], 6)
    out.sort(key=lambda u: u["costUsd"], reverse=True)
    return out
