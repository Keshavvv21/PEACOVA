"""Gmail / Google Calendar / Google Chat as per-user tools — the
hand-written sibling to user_mcp_client.py's per-user OAuth MCP tools.
Hand-written rather than MCP-based because no single official MCP
server covers all three Google products together (see
app/oauth/providers.py's `_google()` docstring) — these call Google's
real REST APIs directly (services/google_api.py) using the user's own
refreshed OAuth token, exactly the same "per-request, per-user, never
merged into the process-global tools.TOOL_SCHEMAS" shape
user_mcp_client.py already established, reusing that same principle
rather than inventing a third pattern.

WRITE_TOOLS (draft_email, send_email, create_calendar_event,
send_chat_message) require the same confirmation gate as any other
write-capable tool (native_agent.requires_confirmation/
confirm_prompt_for) — read tools (read_emails, list_calendar_events)
don't, matching the spec this was built against: only actions that
modify external Google data need an explicit yes/no.
"""
from __future__ import annotations

from .. import oauth_store
from ..services import google_api

WRITE_TOOLS: frozenset[str] = frozenset(
    {"draft_email", "send_email", "create_calendar_event", "send_chat_message"})

_TOOL_SCHEMAS: dict[str, dict] = {
    "read_emails": {
        "description": "Read recent Gmail messages (sender, subject, date, and a snippet of "
                       "the body) for the connected Google account — use this to answer "
                       "questions about or summarize the user's inbox.",
        "required": [],
        "properties": {
            "query": {"type": "string", "description": "Optional Gmail search query "
                                                        "(e.g. \"from:boss@work.com\", "
                                                        "\"is:unread\")"},
            "max_results": {"type": "integer", "description": "How many messages to fetch "
                                                               "(default 10)"},
        },
    },
    "draft_email": {
        "description": "Create a draft email in the connected Gmail account (does NOT send "
                       "it) — requires user confirmation before it runs.",
        "required": ["to", "subject", "body"],
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
    "send_email": {
        "description": "Send an email from the connected Gmail account — requires user "
                       "confirmation before it runs.",
        "required": ["to", "subject", "body"],
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
    "list_calendar_events": {
        "description": "List upcoming events on the connected Google Calendar.",
        "required": [],
        "properties": {
            "time_min": {"type": "string", "description": "RFC3339 start of the window, "
                                                           "e.g. 2026-08-26T00:00:00Z"},
            "time_max": {"type": "string", "description": "RFC3339 end of the window"},
            "max_results": {"type": "integer"},
        },
    },
    "create_calendar_event": {
        "description": "Create/schedule a new event on the connected Google Calendar — "
                       "requires user confirmation before it runs.",
        "required": ["summary", "start", "end"],
        "properties": {
            "summary": {"type": "string", "description": "Event title"},
            "start": {"type": "string", "description": "RFC3339 start time"},
            "end": {"type": "string", "description": "RFC3339 end time"},
            "description": {"type": "string"},
            "attendees": {"type": "array", "items": {"type": "string"},
                         "description": "Attendee email addresses"},
        },
    },
    "send_chat_message": {
        "description": "Send a text message in Google Chat, on behalf of the connected "
                       "account — requires user confirmation before it runs. `recipient` "
                       "must be a real Chat space id (\"spaces/...\") or Directory/People "
                       "API user resource id (\"users/...\"); a plain email address can't "
                       "be resolved automatically.",
        "required": ["recipient", "text"],
        "properties": {
            "recipient": {"type": "string"},
            "text": {"type": "string"},
        },
    },
}


def _is_connected(user_id: str) -> bool:
    return oauth_store.get_connection(user_id, "google") is not None


def tool_specs_for_user(user_id: str) -> list[dict]:
    """Same {name, description, parameters} shape mcp_client.mcp_tool_specs()/
    user_mcp_client.user_mcp_tool_specs() return, so tools.py's merge logic
    treats every per-user tool source identically."""
    if not _is_connected(user_id):
        return []
    return [{"name": name, "description": schema["description"],
            "parameters": {"type": "object", "properties": schema["properties"],
                           "required": schema["required"]}}
           for name, schema in _TOOL_SCHEMAS.items()]


def tool_names_for_user(user_id: str) -> frozenset[str]:
    return frozenset(_TOOL_SCHEMAS) if _is_connected(user_id) else frozenset()


def _require_token(user_id: str) -> str:
    token = oauth_store.get_valid_access_token(user_id, "google")
    if not token:
        raise RuntimeError("Your Google connection needs to be reconnected — "
                           "visit Account → Integrations.")
    return token


def execute_tool(user_id: str, name: str, args: dict) -> dict:
    """Step 11/12 for a Google tool — mirrors mcp_client.call_mcp_tool()'s
    contract (raises, never a sentinel; formats a compact human-readable
    `text` the same shape every other tool result already has)."""
    token = _require_token(user_id)

    if name == "read_emails":
        emails = google_api.list_recent_emails(
            token, max_results=args.get("max_results") or 10, query=args.get("query"))
        if not emails:
            text = "No emails found."
        else:
            text = "\n".join(f"- From {e['from']} — \"{e['subject']}\" ({e['date']}): "
                            f"{e['snippet']}" for e in emails)
        return {"ok": True, "kind": "google_tool_result", "tool": name, "text": text,
               "raw": emails}

    if name == "draft_email":
        result = google_api.create_draft(token, to=args["to"], subject=args["subject"],
                                         body=args["body"])
        return {"ok": True, "kind": "google_tool_result", "tool": name,
               "text": f"Draft created to {result['to']} — \"{result['subject']}\".",
               "raw": result}

    if name == "send_email":
        result = google_api.send_email(token, to=args["to"], subject=args["subject"],
                                       body=args["body"])
        return {"ok": True, "kind": "google_tool_result", "tool": name,
               "text": f"Email sent to {result['to']} — \"{result['subject']}\".",
               "raw": result}

    if name == "list_calendar_events":
        events = google_api.list_calendar_events(
            token, time_min=args.get("time_min"), time_max=args.get("time_max"),
            max_results=args.get("max_results") or 10)
        if not events:
            text = "No upcoming events found."
        else:
            text = "\n".join(f"- {e['summary']} ({e['start'].get('dateTime') or e['start'].get('date')})"
                            for e in events)
        return {"ok": True, "kind": "google_tool_result", "tool": name, "text": text,
               "raw": events}

    if name == "create_calendar_event":
        result = google_api.create_calendar_event(
            token, summary=args["summary"], start=args["start"], end=args["end"],
            attendees=args.get("attendees"), description=args.get("description"))
        return {"ok": True, "kind": "google_tool_result", "tool": name,
               "text": f"Event \"{result['summary']}\" created.", "raw": result}

    if name == "send_chat_message":
        result = google_api.send_chat_message(token, recipient=args["recipient"],
                                              text=args["text"])
        return {"ok": True, "kind": "google_tool_result", "tool": name,
               "text": f"Chat message sent to {result['space']}.", "raw": result}

    raise RuntimeError(f"Google tool {name!r} is not currently available")
