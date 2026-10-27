"""Coverage for the general PM/chat track's own usage tracking
(framework/chat_usage.py + routers/chat_usage.py) — the same
"one persisted row per real turn" idea as coding_usage.py's VS Code
equivalent, now for POST /api/chat/message."""
from __future__ import annotations

from tests.conftest import auth


def test_a_real_chat_turn_shows_up_in_my_own_usage(client, exec_token):
    r = client.post("/api/chat/message?framework=native",
                    json={"message": "hello, what can you do?", "session_id": "usage-check-1"},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text

    r2 = client.get("/api/chat-usage", headers=auth(exec_token))
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert data["summary"]["totalRequests"] >= 1
    assert len(data["table"]) >= 1
    assert data["table"][0]["model"]  # a real provider name, never blank


def test_admin_chat_usage_requires_admin_role(client, member_token):
    r = client.get("/api/admin/chat-usage", headers=auth(member_token))
    assert r.status_code == 403


def test_admin_chat_usage_shows_a_per_user_breakdown(client, exec_token):
    client.post("/api/chat/message?framework=native",
               json={"message": "what tasks do I have?", "session_id": "usage-check-2"},
               headers=auth(exec_token))

    r = client.get("/api/admin/chat-usage", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    data = r.json()
    assert "byUser" in data
    assert any(u["requests"] >= 1 for u in data["byUser"])
