"""Tests for /api/admin/providers (app.routers.admin_providers) — the
admin-managed LLM provider registry, and its live wiring into
llm/router.py's provider chains."""

from tests.conftest import auth


def test_member_cannot_manage_providers(client, member_token):
    r = client.get("/api/admin/providers", headers=auth(member_token))
    assert r.status_code == 403

    r = client.post("/api/admin/providers",
                    json={"name": "Gemini Pro", "base_url": "https://x", "model": "gemini-pro",
                          "api_key": "secret"},
                    headers=auth(member_token))
    assert r.status_code == 403


def test_create_list_update_delete_provider(client, exec_token):
    r = client.post("/api/admin/providers",
                    json={"name": "Gemini Pro", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                          "model": "gemini-pro", "api_key": "sk-test-123", "tiers": ["realtime"]},
                    headers=auth(exec_token))
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Gemini Pro"
    assert body["enabled"] is True
    # the real key is never echoed back — only whether one's set
    assert body["api_key_set"] is True
    assert "api_key" not in body
    provider_id = body["id"]

    r = client.get("/api/admin/providers", headers=auth(exec_token))
    assert r.status_code == 200
    assert any(p["id"] == provider_id for p in r.json())

    r = client.patch(f"/api/admin/providers/{provider_id}", json={"enabled": False},
                     headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    r = client.delete(f"/api/admin/providers/{provider_id}", headers=auth(exec_token))
    assert r.status_code == 204

    r = client.get("/api/admin/providers", headers=auth(exec_token))
    assert not any(p["id"] == provider_id for p in r.json())


def test_update_404s_for_unknown_provider(client, exec_token):
    r = client.patch("/api/admin/providers/does-not-exist", json={"enabled": False},
                     headers=auth(exec_token))
    assert r.status_code == 404


def test_enabled_provider_shows_up_in_the_model_picker(client, exec_token):
    r = client.post("/api/admin/providers",
                    json={"name": "Custom Model", "base_url": "https://example.invalid/v1",
                          "model": "custom-1", "api_key": "sk-test", "tiers": ["realtime"]},
                    headers=auth(exec_token))
    provider_id = r.json()["id"]

    r = client.get("/api/chat/providers", headers=auth(exec_token))
    names = [p["id"] for p in r.json()["providers"]]
    assert "Custom Model" in names

    client.patch(f"/api/admin/providers/{provider_id}", json={"enabled": False},
                headers=auth(exec_token))
    r = client.get("/api/chat/providers", headers=auth(exec_token))
    names = [p["id"] for p in r.json()["providers"]]
    assert "Custom Model" not in names

    client.delete(f"/api/admin/providers/{provider_id}", headers=auth(exec_token))
