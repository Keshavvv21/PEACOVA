"""Durable log of non-safe security_guard.py flags — the source of
truth for the admin Security dashboard (routers/admin_security.py) and
the daily report (security_report.py). Every 🟡/🔴 flagged reply gets
ONE row here; 🟢 safe replies are never logged, so this table stays
real signal an admin can actually act on, not a copy of every chat
turn. Reuses app/db.py's existing PK/SK store — same mechanism
OAuthConnections/ChatSessions/BanditState already use, no new
persistence layer.
"""
from __future__ import annotations

import datetime as dt
import uuid

from .db import store

_TABLE = "SecurityEvents"


def _date_key(occurred_at: dt.datetime) -> str:
    return occurred_at.strftime("%Y-%m-%d")


def log_event(user_id: str, *, flag: str, findings: list[str], excerpt: str) -> None:
    """No-op for flag == "safe" — see module docstring. Never raises on
    a storage hiccup: a failed security-event log must not break the
    chat reply it's describing, same fail-soft rule
    bandit_persistence.save() already follows for RL state."""
    if flag == "safe":
        return
    now = dt.datetime.now(dt.timezone.utc)
    event_id = uuid.uuid4().hex[:12]
    row = {
        "user_id": user_id,
        "flag": flag,
        "findings": findings,
        # Truncated — this is an admin-visible excerpt for triage, not
        # the full turn; the real transcript still lives in ChatSessions
        # under the user's own normal access controls.
        "excerpt": (excerpt or "")[:280],
        "occurred_at": now.isoformat(),
    }
    try:
        store(_TABLE).put(_date_key(now), f"EVENT#{now.isoformat()}#{event_id}", row)
    except Exception:  # noqa: BLE001
        pass


def _strip_storage_keys(row: dict) -> dict:
    """store()'s put() stamps PK/SK onto every row (db.py) — internal
    storage keys, not part of this module's real return shape."""
    return {k: v for k, v in row.items() if k not in ("PK", "SK")}


def list_events_for_date(date: str) -> list[dict]:
    """`date` is "YYYY-MM-DD". Chronological order."""
    events = store(_TABLE).scan(date, "EVENT#")
    return sorted((_strip_storage_keys(e) for e in events), key=lambda r: r["occurred_at"])


def list_recent_events(days: int = 7) -> list[dict]:
    """Every event across the last `days` calendar days (UTC),
    newest-first — the admin dashboard's default view. A real,
    if inefficient for a huge table, full scan filtered by date; see
    db.py's own scan() — every other admin-facing listing in this
    codebase (Providers, OAuthConnections) makes the identical
    trade-off for a table this size."""
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    events = [_strip_storage_keys(e) for e in store(_TABLE).scan("", "EVENT#") if e["PK"] >= cutoff]
    return sorted(events, key=lambda r: r["occurred_at"], reverse=True)
