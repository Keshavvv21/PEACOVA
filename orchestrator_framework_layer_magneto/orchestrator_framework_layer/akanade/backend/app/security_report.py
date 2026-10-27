"""Daily security digest — real logged events (security_events.py),
summarized in plain English via ONE more `get_llm_client()` call, same
"bulk tier, background housekeeping, reject a rule-based non-answer"
pattern agent/memory.py's `fold_into_summary()` already established for
session-summary generation. Cached per-day (a report is regenerated
only if one for that date doesn't already exist) so repeated dashboard
loads don't re-summarize the same day over and over.
"""
from __future__ import annotations

import datetime as dt

from . import security_events
from .db import store
from .llm.router import get_llm_client

_TABLE = "SecurityDailyReports"

_REPORT_SYSTEM = {
    "role": "system",
    "content": (
        "You are a security analyst writing a short daily digest for an "
        "engineering admin. You are given a fixed list of real, already-"
        "detected security events from a chat assistant's guardrail layer "
        "— never invent, add, or omit an event from that list; never claim "
        "the day was clean unless the list is genuinely empty. Summarize in "
        "3-6 sentences: what kinds of things were flagged, whether any "
        "single user had a concerning pattern (repeated risk-tier events), "
        "and one concrete recommendation if anything stands out. Plain "
        "prose only, no headers, no bullet list."
    ),
}


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _fallback_summary(events: list[dict]) -> str:
    """Used only if every LLM provider is unreachable (bottoms out at
    RuleBased) — a real, honest, non-LLM summary rather than nothing."""
    risk = sum(1 for e in events if e["flag"] == "risk")
    review = len(events) - risk
    users = sorted({e["user_id"] for e in events})
    return (f"{len(events)} security-flagged event(s) today: {risk} risk-tier, "
           f"{review} needs-review. Affected user(s): {', '.join(users)}. "
           f"(Automated summary unavailable — every LLM provider was "
           f"unreachable when this report was generated; the raw event list "
           f"above is still complete and accurate.)")


def generate_daily_report(date: str | None = None, *, force: bool = False) -> dict:
    """Returns the cached report for `date` (default: today, UTC) if one
    already exists, unless `force=True`. Never raises — a summarization
    failure degrades to `_fallback_summary`, never a 500 to the admin
    dashboard calling this."""
    target = date or _today()
    if not force:
        cached = store(_TABLE).get(target, "REPORT")
        if cached:
            # store()'s put() stamps PK/SK onto every row (db.py) — strip
            # them back off so a cached-hit response is byte-identical in
            # shape to a freshly-generated one, not an internal-storage
            # leak into the admin API response.
            return {k: v for k, v in cached.items() if k not in ("PK", "SK")}

    events = security_events.list_events_for_date(target)
    if not events:
        report = {"date": target, "event_count": 0, "risk_count": 0,
                  "needs_review_count": 0, "affected_users": [],
                  "summary": "No security-flagged events today.", "events": []}
        store(_TABLE).put(target, "REPORT", report)
        return report

    risk_events = [e for e in events if e["flag"] == "risk"]
    review_events = [e for e in events if e["flag"] == "needs_review"]
    affected_users = sorted({e["user_id"] for e in events})

    lines = [f"- [{e['flag'].upper()}] user={e['user_id']}: {'; '.join(e['findings'][:2])}"
            for e in events]
    prompt = (f"Events for {target} ({len(events)} total, {len(risk_events)} risk, "
             f"{len(review_events)} needs-review):\n" + "\n".join(lines))
    client = get_llm_client("bulk")
    result = client.chat("security_daily_report", [_REPORT_SYSTEM,
                                                    {"role": "user", "content": prompt}])
    summary = result.text.strip() if result.provider != "rule-based" and result.text else None

    report = {
        "date": target, "event_count": len(events), "risk_count": len(risk_events),
        "needs_review_count": len(review_events), "affected_users": affected_users,
        "summary": summary or _fallback_summary(events), "events": events,
    }
    store(_TABLE).put(target, "REPORT", report)
    return report
