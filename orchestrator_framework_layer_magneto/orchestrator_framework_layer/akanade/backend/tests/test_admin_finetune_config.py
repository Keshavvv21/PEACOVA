"""Tests for /api/admin/finetune-config (app.routers.admin_finetune_config)
— what SHOULD get trained tonight, editable via the Admin UI instead of
shell-script env vars. One row per domain, upserted by PUT."""

from tests.conftest import auth


def test_member_cannot_manage_finetune_config(client, member_token):
    r = client.get("/api/admin/finetune-config", headers=auth(member_token))
    assert r.status_code == 403

    r = client.put("/api/admin/finetune-config/general",
                   json={"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "ollama_base_tag": "qwen2.5:1.5b"},
                   headers=auth(member_token))
    assert r.status_code == 403


def test_put_general_creates_and_updates_in_place(client, exec_token):
    r = client.put("/api/admin/finetune-config/general",
                   json={"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "ollama_base_tag": "qwen2.5:1.5b",
                         "max_minutes": 90},
                   headers=auth(exec_token))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "general"
    assert body["domain"] is None
    assert body["max_minutes"] == 90
    assert body["serve_backend"] == "ollama"  # the default, unchanged behavior
    created_at = body["created_at"]

    r = client.put("/api/admin/finetune-config/general",
                   json={"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "ollama_base_tag": "qwen2.5:1.5b",
                         "max_minutes": 150},
                   headers=auth(exec_token))
    assert r.status_code == 200
    body2 = r.json()
    assert body2["max_minutes"] == 150
    assert body2["created_at"] == created_at  # upsert, not a new row

    r = client.get("/api/admin/finetune-config", headers=auth(exec_token))
    assert len([c for c in r.json() if c["id"] == "general"]) == 1


def test_put_a_domain_config_uses_the_domain_as_the_key(client, exec_token):
    r = client.put("/api/admin/finetune-config/legal",
                   json={"domain": "legal", "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                         "ollama_base_tag": "qwen2.5:1.5b"},
                   headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["domain"] == "legal"
    assert r.json()["id"] == "legal"


def test_serve_backend_can_be_set_to_llama_server(client, exec_token):
    r = client.put("/api/admin/finetune-config/coding",
                   json={"domain": "coding", "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                         "ollama_base_tag": "qwen2.5:1.5b", "serve_backend": "llama_server"},
                   headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["serve_backend"] == "llama_server"


def test_serve_backend_rejects_an_unknown_value(client, exec_token):
    r = client.put("/api/admin/finetune-config/general",
                   json={"base_model": "a", "ollama_base_tag": "b", "serve_backend": "vllm"},
                   headers=auth(exec_token))
    assert r.status_code == 422


def test_put_rejects_a_mismatched_key_and_domain(client, exec_token):
    r = client.put("/api/admin/finetune-config/support",
                   json={"domain": "legal", "base_model": "x", "ollama_base_tag": "y"},
                   headers=auth(exec_token))
    assert r.status_code == 400


def test_list_enabled_excludes_disabled_configs(client, exec_token):
    client.put("/api/admin/finetune-config/general",
              json={"base_model": "a", "ollama_base_tag": "b", "enabled": True},
              headers=auth(exec_token))
    client.put("/api/admin/finetune-config/coding",
              json={"domain": "coding", "base_model": "a", "ollama_base_tag": "b", "enabled": False},
              headers=auth(exec_token))
    r = client.get("/api/admin/finetune-config/enabled", headers=auth(exec_token))
    ids = [c["id"] for c in r.json()]
    assert "general" in ids
    assert "coding" not in ids


def test_delete_config(client, exec_token):
    client.put("/api/admin/finetune-config/temp",
              json={"domain": "temp", "base_model": "a", "ollama_base_tag": "b"},
              headers=auth(exec_token))
    r = client.delete("/api/admin/finetune-config/temp", headers=auth(exec_token))
    assert r.status_code == 204

    r = client.delete("/api/admin/finetune-config/temp", headers=auth(exec_token))
    assert r.status_code == 404
