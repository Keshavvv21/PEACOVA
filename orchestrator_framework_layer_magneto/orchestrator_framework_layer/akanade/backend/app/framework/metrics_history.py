"""A small time-series log of what FeedbackLoop.close() actually decided,
one row per closed task — the only way to answer "is routing actually
getting better over time" from outside the process. GET /api/v1/metrics
(routers/v1/metrics.py) is a point-in-time-only snapshot of the router's
CURRENT posterior; nothing before this module kept any history of it at
all (confirmed: bandit_persistence.py overwrites the same key on every
save, task_memory.py's deque isn't persisted or queryable, and
FeedbackLoop._records is in-process/lost on restart).

Stored via the SAME app/db.py LocalStore/DynamoStore abstraction every
other admin table already uses — no new storage engine, no separate DB.

One row per FeedbackLoop.close() call — a real "a task's outcome was
just scored" event, since reward_weights only actually drift when an
outcome is scored — not a periodic poll tied to whether anyone happens
to have the dashboard open.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..db import store

_TABLE = "MetricsSnapshots"
_PK = "METRICS"
# In-process reference-deployment simplicity, same convention
# bandit_persistence.py/task_memory.py already use: bounded history,
# not unbounded growth, rather than a real TTL/archival policy.
_MAX_ROWS = 500


def record(task_id: str, total_reward: float, reward_weights: dict[str, float],
          attributed_rewards: dict[str, float]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    row_id = uuid.uuid4().hex[:8]
    store(_TABLE).put(_PK, f"SNAPSHOT#{now}#{row_id}", {
        "id": row_id,
        "task_id": task_id,
        "timestamp": now,
        "total_reward": round(float(total_reward), 4),
        "reward_weights": {k: round(float(v), 4) for k, v in reward_weights.items()},
        "attributed_rewards": {k: round(float(v), 4) for k, v in attributed_rewards.items()},
    })
    _prune()


def _prune() -> None:
    rows = store(_TABLE).scan(_PK, "SNAPSHOT#")
    if len(rows) <= _MAX_ROWS:
        return
    # SK is timestamp-prefixed, so lexicographic sort is chronological —
    # drop the oldest rows over the cap.
    overflow = sorted(rows, key=lambda r: r["SK"])[: len(rows) - _MAX_ROWS]
    for row in overflow:
        store(_TABLE).delete(row["PK"], row["SK"])


def history(limit: int = 200) -> list[dict]:
    """Ascending by time (oldest first) — the natural order for a chart's
    x-axis; the caller (the API route) trims to the most recent `limit`
    for a caller that doesn't want the full 500-row cap."""
    rows = store(_TABLE).scan(_PK, "SNAPSHOT#")
    rows.sort(key=lambda r: r["SK"])
    if limit:
        rows = rows[-limit:]
    return [
        {
            "id": r["id"],
            "task_id": r["task_id"],
            "timestamp": r["timestamp"],
            "total_reward": r["total_reward"],
            "reward_weights": r["reward_weights"],
            "attributed_rewards": r["attributed_rewards"],
        }
        for r in rows
    ]
