"""services/google_api.py (real Gmail/Calendar/Chat REST calls, mocked
httpx — no live Google account needed to test) and agent/google_tools.py
(the per-user tool layer + confirmation-gate wiring built on top of it).
"""
from __future__ import annotations

import pytest

from app import oauth_store
from app.agent import google_tools, native_agent, tools
from app.services import google_api


def _fake_google_connection(user_id: str, token: str = "tok"):
    oauth_store.save_connection(user_id, "google", access_token=token, refresh_token=None,
                                expires_in=None, scopes=["gmail.readonly"])


class _FakeResponse:
    def __init__(self, json_body):
        self._json = json_body

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


# ── services/google_api.py (mocked httpx) ──────────────────────────────

def test_list_recent_emails_calls_list_then_get_per_message(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/messages"):
            return _FakeResponse({"messages": [{"id": "m1"}, {"id": "m2"}]})
        return _FakeResponse({
            "id": url.rsplit("/", 1)[-1],
            "snippet": "hello there",
            "payload": {"headers": [{"name": "From", "value": "a@b.com"},
                                   {"name": "Subject", "value": "Hi"},
                                   {"name": "Date", "value": "2026-01-01"}]},
        })

    monkeypatch.setattr(google_api.httpx, "get", fake_get)
    emails = google_api.list_recent_emails("tok", max_results=2)
    assert len(emails) == 2
    assert emails[0]["from"] == "a@b.com"
    assert emails[0]["subject"] == "Hi"
    assert emails[0]["snippet"] == "hello there"


def test_create_draft_builds_a_base64_mime_message(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"id": "draft-1"})

    monkeypatch.setattr(google_api.httpx, "post", fake_post)
    result = google_api.create_draft("tok", to="x@y.com", subject="Hi", body="body text")
    assert result["draft_id"] == "draft-1"
    assert captured["url"].endswith("/drafts")
    assert "raw" in captured["json"]["message"]


def test_send_email_posts_to_messages_send(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/messages/send")
        assert "raw" in json
        return _FakeResponse({"id": "msg-1"})

    monkeypatch.setattr(google_api.httpx, "post", fake_post)
    result = google_api.send_email("tok", to="x@y.com", subject="Hi", body="body")
    assert result["message_id"] == "msg-1"


def test_list_calendar_events(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeResponse({"items": [{"id": "e1", "summary": "Standup",
                                        "start": {"dateTime": "2026-01-01T09:00:00Z"},
                                        "end": {"dateTime": "2026-01-01T09:30:00Z"}}]})

    monkeypatch.setattr(google_api.httpx, "get", fake_get)
    events = google_api.list_calendar_events("tok")
    assert events[0]["summary"] == "Standup"


def test_create_calendar_event_with_attendees(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse({"id": "evt-1", "htmlLink": "https://calendar.google.com/x"})

    monkeypatch.setattr(google_api.httpx, "post", fake_post)
    result = google_api.create_calendar_event(
        "tok", summary="Sync", start="2026-01-01T09:00:00Z", end="2026-01-01T09:30:00Z",
        attendees=["a@b.com"])
    assert result["event_id"] == "evt-1"
    assert captured["json"]["attendees"] == [{"email": "a@b.com"}]


def test_send_chat_message_to_a_space_id_directly(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("spaces/AAA/messages")
        return _FakeResponse({"name": "spaces/AAA/messages/1"})

    monkeypatch.setattr(google_api.httpx, "post", fake_post)
    result = google_api.send_chat_message("tok", recipient="spaces/AAA", text="hi")
    assert result["space"] == "spaces/AAA"


def test_send_chat_message_resolves_a_user_resource_via_find_direct_message(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        assert url.endswith("spaces:findDirectMessage")
        return _FakeResponse({"name": "spaces/DM123"})

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("spaces/DM123/messages")
        return _FakeResponse({"name": "spaces/DM123/messages/1"})

    monkeypatch.setattr(google_api.httpx, "get", fake_get)
    monkeypatch.setattr(google_api.httpx, "post", fake_post)
    result = google_api.send_chat_message("tok", recipient="users/12345", text="hi")
    assert result["space"] == "spaces/DM123"


def test_send_chat_message_rejects_a_plain_email_recipient():
    with pytest.raises(ValueError):
        google_api.send_chat_message("tok", recipient="someone@example.com", text="hi")


# ── agent/google_tools.py ────────────────────────────────────────────

def test_tools_unavailable_without_a_google_connection():
    assert google_tools.tool_specs_for_user("nobody") == []
    assert google_tools.tool_names_for_user("nobody") == frozenset()


def test_tools_available_once_connected():
    _fake_google_connection("gu1")
    names = google_tools.tool_names_for_user("gu1")
    assert names == frozenset(google_tools._TOOL_SCHEMAS)
    oauth_store.delete_connection("gu1", "google")


def test_read_tools_are_not_in_write_tools():
    assert "read_emails" not in google_tools.WRITE_TOOLS
    assert "list_calendar_events" not in google_tools.WRITE_TOOLS


def test_write_tools_are_exactly_the_four_mutating_actions():
    assert google_tools.WRITE_TOOLS == {"draft_email", "send_email",
                                        "create_calendar_event", "send_chat_message"}


def test_execute_tool_without_a_connection_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="reconnected"):
        google_tools.execute_tool("nobody", "read_emails", {})


def test_execute_tool_read_emails_dispatches_to_google_api(monkeypatch):
    _fake_google_connection("gu2")
    monkeypatch.setattr(google_api, "list_recent_emails",
                        lambda token, max_results=10, query=None: [
                            {"from": "a@b.com", "subject": "Hi", "date": "d", "snippet": "s"}])
    result = google_tools.execute_tool("gu2", "read_emails", {})
    assert result["ok"] is True
    assert "a@b.com" in result["text"]
    oauth_store.delete_connection("gu2", "google")


def test_execute_tool_send_email_dispatches_to_google_api(monkeypatch):
    _fake_google_connection("gu3")
    monkeypatch.setattr(google_api, "send_email",
                        lambda token, to, subject, body: {"message_id": "1", "to": to,
                                                          "subject": subject})
    result = google_tools.execute_tool("gu3", "send_email",
                                       {"to": "x@y.com", "subject": "Hi", "body": "b"})
    assert "x@y.com" in result["text"]
    oauth_store.delete_connection("gu3", "google")


def test_execute_tool_unknown_name_raises():
    _fake_google_connection("gu4")
    with pytest.raises(RuntimeError):
        google_tools.execute_tool("gu4", "delete_everything", {})
    oauth_store.delete_connection("gu4", "google")


# ── tools.py wiring ───────────────────────────────────────────────────

def test_user_openai_tool_specs_includes_google_tools_once_connected():
    _fake_google_connection("gu5")
    specs = tools.user_openai_tool_specs({"sub": "gu5"})
    names = {s["name"] for s in specs}
    assert "read_emails" in names
    assert "send_email" in names
    oauth_store.delete_connection("gu5", "google")


def test_execute_dispatches_google_tool_through_tools_execute(monkeypatch):
    _fake_google_connection("gu6")
    monkeypatch.setattr(google_api, "list_calendar_events",
                        lambda token, time_min=None, time_max=None, max_results=10: [])
    result = tools.execute({"sub": "gu6"}, "list_calendar_events", {})
    assert result["ok"] is True
    assert "No upcoming events" in result["text"]
    oauth_store.delete_connection("gu6", "google")


# ── native_agent.py confirmation gate ─────────────────────────────────

def test_read_google_tool_does_not_require_confirmation():
    _fake_google_connection("gu7")
    assert native_agent.requires_confirmation({"sub": "gu7"}, "read_emails") is False
    oauth_store.delete_connection("gu7", "google")


def test_write_google_tool_requires_confirmation():
    _fake_google_connection("gu8")
    assert native_agent.requires_confirmation({"sub": "gu8"}, "send_email") is True
    oauth_store.delete_connection("gu8", "google")


def test_confirm_prompt_for_send_email_names_the_google_account_and_shows_the_exact_action():
    user = {"sub": "gu9"}
    prompt = native_agent.confirm_prompt_for(
        user, "send_email", {"to": "x@y.com", "subject": "Offer", "body": "..."})
    assert "Google" in prompt
    assert "x@y.com" in prompt
    assert "SEND an email" in prompt


def test_confirm_prompt_for_create_calendar_event():
    user = {"sub": "gu10"}
    prompt = native_agent.confirm_prompt_for(
        user, "create_calendar_event",
        {"summary": "1:1", "start": "2026-01-01T09:00:00Z", "end": "2026-01-01T09:30:00Z"})
    assert "calendar event" in prompt
    assert "1:1" in prompt


def test_confirm_prompt_for_send_chat_message():
    user = {"sub": "gu11"}
    prompt = native_agent.confirm_prompt_for(user, "send_chat_message",
                                             {"recipient": "spaces/AAA", "text": "hey"})
    assert "Google Chat" in prompt
