"""Real Gmail / Google Calendar / Google Chat REST API calls — the
service layer agent/google_tools.py's tool functions call into, same
"thin layer around the real remote API, raises on real failure" shape
real_pm_service.py already follows for the PM app's own REST API.

Every function takes an already-valid `access_token` (agent/
google_tools.py fetches one fresh per call via
oauth_store.get_valid_access_token — this module never touches
oauth_store itself, keeping it a pure "call Google" layer, mockable in
tests without any OAuth-store setup). Raises httpx.HTTPStatusError (or
ValueError for a bad recipient shape) on failure — the caller
(google_tools.execute_tool, then tools.execute()'s existing try/except
in every agent track) turns that into "Sorry — that failed: {e}", same
convention every other tool already follows.

Endpoints verified against Google's own current API docs (not assumed
from training data) — see README.md's Google integrations section for
the exact OAuth scopes each one needs.
"""
from __future__ import annotations

import base64
from email.mime.text import MIMEText

import httpx

_TIMEOUT = 15


def _auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _get(url: str, access_token: str, **params) -> dict:
    r = httpx.get(url, headers=_auth_headers(access_token), params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(url: str, access_token: str, body: dict) -> dict:
    r = httpx.post(url, headers=_auth_headers(access_token), json=body, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Gmail (gmail.readonly / gmail.compose) ────────────────────────────

_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


def list_recent_emails(access_token: str, *, max_results: int = 10,
                       query: str | None = None) -> list[dict]:
    """Real Gmail messages.list + one messages.get per hit (format=metadata,
    which also returns `snippet`) — enough for the assistant to summarize
    without pulling full bodies. Returns [{id, from, subject, date, snippet}]."""
    listing = _get(f"{_GMAIL_BASE}/messages", access_token,
                   maxResults=max_results, **({"q": query} if query else {}))
    out = []
    for item in listing.get("messages", []):
        msg = _get(f"{_GMAIL_BASE}/messages/{item['id']}", access_token,
                  format="metadata", metadataHeaders=["From", "Subject", "Date"])
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        out.append({"id": msg["id"], "from": headers.get("From", ""),
                   "subject": headers.get("Subject", ""), "date": headers.get("Date", ""),
                   "snippet": msg.get("snippet", "")})
    return out


def _build_raw_message(to: str, subject: str, body: str) -> str:
    mime = MIMEText(body)
    mime["to"] = to
    mime["subject"] = subject
    return base64.urlsafe_b64encode(mime.as_bytes()).decode()


def create_draft(access_token: str, *, to: str, subject: str, body: str) -> dict:
    raw = _build_raw_message(to, subject, body)
    result = _post(f"{_GMAIL_BASE}/drafts", access_token, {"message": {"raw": raw}})
    return {"draft_id": result.get("id"), "to": to, "subject": subject}


def send_email(access_token: str, *, to: str, subject: str, body: str) -> dict:
    raw = _build_raw_message(to, subject, body)
    result = _post(f"{_GMAIL_BASE}/messages/send", access_token, {"raw": raw})
    return {"message_id": result.get("id"), "to": to, "subject": subject}


# ── Google Calendar (calendar.events) ──────────────────────────────────

_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3/calendars/primary/events"


def list_calendar_events(access_token: str, *, time_min: str | None = None,
                         time_max: str | None = None, max_results: int = 10) -> list[dict]:
    """time_min/time_max are RFC3339 timestamps (e.g. "2026-08-26T00:00:00Z");
    omitted means Google's own default window. Returns each event's
    id/summary/start/end/htmlLink — enough to list/describe, not the
    full resource."""
    params: dict = {"maxResults": max_results, "singleEvents": "true", "orderBy": "startTime"}
    if time_min:
        params["timeMin"] = time_min
    if time_max:
        params["timeMax"] = time_max
    result = _get(_CALENDAR_BASE, access_token, **params)
    return [{"id": e.get("id"), "summary": e.get("summary", "(no title)"),
            "start": e.get("start", {}), "end": e.get("end", {}),
            "html_link": e.get("htmlLink")}
           for e in result.get("items", [])]


def create_calendar_event(access_token: str, *, summary: str, start: str, end: str,
                          attendees: list[str] | None = None,
                          description: str | None = None) -> dict:
    """start/end are RFC3339 timestamps. `attendees` is a list of email
    addresses — Google sends them the real invite."""
    body: dict = {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}
    if description:
        body["description"] = description
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]
    result = _post(_CALENDAR_BASE, access_token, body)
    return {"event_id": result.get("id"), "summary": summary,
           "html_link": result.get("htmlLink")}


# ── Google Chat (chat.messages.create / chat.spaces.readonly) ─────────

_CHAT_BASE = "https://chat.googleapis.com/v1"


def _resolve_chat_space(access_token: str, recipient: str) -> str:
    """`recipient` must already be a real Chat resource — a space
    ("spaces/AAAAxxxxxxx", e.g. a room) or a Directory/People API user
    resource ("users/108...", NOT a plain email address). Google's Chat
    API has no endpoint that resolves an arbitrary email to a Chat user
    without a separate People/Directory API lookup and its own scope —
    out of scope here; see README.md's Google Chat section for exactly
    why a plain email can't be auto-resolved, and how to get a real
    users/{id} or spaces/{id} to pass instead."""
    if recipient.startswith("spaces/"):
        return recipient
    if recipient.startswith("users/"):
        result = _get(f"{_CHAT_BASE}/spaces:findDirectMessage", access_token, name=recipient)
        return result["name"]
    raise ValueError(
        f"{recipient!r} isn't a Chat space or user resource id — Google Chat "
        f"can't resolve a plain email address on its own. Pass a space id "
        f"(\"spaces/...\") or a Directory/People API user resource id "
        f"(\"users/...\") instead.")


def send_chat_message(access_token: str, *, recipient: str, text: str) -> dict:
    space = _resolve_chat_space(access_token, recipient)
    result = _post(f"{_CHAT_BASE}/{space}/messages", access_token, {"text": text})
    return {"message_name": result.get("name"), "space": space}
