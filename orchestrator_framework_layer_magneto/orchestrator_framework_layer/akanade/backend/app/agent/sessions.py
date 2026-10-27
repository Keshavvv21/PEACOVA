"""§9 — the ChatSessions table: PK USER#{userId}, SK SESSION#{sessionId}.

Same schema for every track (the checkpointer/SessionService equivalence
the design demands): rolling message window, a running `summary` of
whatever's scrolled out of that window (see memory.fold_into_summary,
the memory-optimization layer this feeds), pendingConfirmation,
pendingClarification, activeFramework, modelTierLog, updatedAt.
Backed by db.store(), so it's local JSON in dev and DynamoDB in prod
without this file changing.
"""
from __future__ import annotations

import datetime as dt

from ..config import settings
from ..db import store
from . import memory


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _is_stale(row: dict) -> bool:
    updated = row.get("updatedAt")
    if not updated:
        return False
    try:
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(updated)
    except ValueError:
        return False
    return age.days >= settings().SESSION_TTL_DAYS


def load(user_id: str, session_id: str) -> dict:
    row = store("ChatSessions").get(f"USER#{user_id}", f"SESSION#{session_id}")
    # SESSION_TTL_DAYS enforced HERE rather than via a separate scheduled
    # job — the frontend already rotates session_id daily (see
    # useChatBot.ts), so this is a safety net for anything left idle past
    # the TTL (a browser tab kept open, a session nobody revisited): the
    # next time it's touched, it's treated as gone and the stale row is
    # cleaned up on the way out, no cron needed.
    if row and _is_stale(row):
        clear(user_id, session_id)
        row = None
    return row or {
        "messages": [], "summary": "", "pendingConfirmation": None,
        "pendingClarification": None, "activeFramework": None,
        "modelTierLog": [], "updatedAt": _now(),
    }


def save(user_id: str, session_id: str, state: dict):
    window = settings().SESSION_WINDOW_TURNS
    messages = state["messages"]
    # Fold whatever's about to scroll out of the window into the running
    # summary BEFORE trimming, instead of just discarding it — see
    # memory.fold_into_summary. A no-op (returns the messages unchanged)
    # when there's nothing to drop yet.
    dropped, kept = messages[:-window], messages[-window:]
    state["summary"] = memory.fold_into_summary(state.get("summary", ""), dropped)
    state["messages"] = kept
    state["modelTierLog"] = state["modelTierLog"][-200:]
    state["updatedAt"] = _now()
    store("ChatSessions").put(f"USER#{user_id}", f"SESSION#{session_id}", state)


def clear(user_id: str, session_id: str) -> None:
    """Wipes one session back to fresh — the "Clear chat" button's backend
    half (see routers/chat.py's DELETE /api/chat/history) and load()'s own
    stale-session cleanup above."""
    store("ChatSessions").delete(f"USER#{user_id}", f"SESSION#{session_id}")


def all_tier_logs() -> list[dict]:
    """Feeds GET /api/chat/metrics (§10) and the Deepseek evaluation."""
    logs = []
    for row in store("ChatSessions").scan("USER#", "SESSION#"):
        logs.extend(row.get("modelTierLog", []))
    return logs
