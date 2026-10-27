"""Tests for /api/admin/slm-versions (app.routers.admin_slm) — the local
version-history log the nightly convert_and_register_ollama_model.sh
run reports to, and the Admin dashboard reads."""

from tests.conftest import auth


def test_member_cannot_manage_slm_versions(client, member_token):
    r = client.get("/api/admin/slm-versions", headers=auth(member_token))
    assert r.status_code == 403

    r = client.post("/api/admin/slm-versions",
                    json={"tag": "algo-akanade-slm:20260827", "base_tag": "qwen2.5:1.5b",
                          "quant_type": "Q4_K_M", "sanity_check": "passed"},
                    headers=auth(member_token))
    assert r.status_code == 403


def test_record_and_list_a_version(client, exec_token):
    r = client.post("/api/admin/slm-versions",
                    json={"tag": "algo-akanade-slm:20260827", "base_tag": "qwen2.5:1.5b",
                          "quant_type": "Q4_K_M", "sanity_check": "passed",
                          "regression_gate": "passed", "registered": True},
                    headers=auth(exec_token))
    assert r.status_code == 201
    body = r.json()
    assert body["tag"] == "algo-akanade-slm:20260827"
    assert body["registered"] is True
    assert "id" in body and "created_at" in body

    r = client.get("/api/admin/slm-versions", headers=auth(exec_token))
    assert r.status_code == 200
    assert any(v["id"] == body["id"] for v in r.json())


def test_a_failed_night_is_still_recorded(client, exec_token):
    r = client.post("/api/admin/slm-versions",
                    json={"tag": "algo-akanade-slm:20260826", "base_tag": "qwen2.5:1.5b",
                          "quant_type": "Q4_K_M", "sanity_check": "passed",
                          "regression_gate": "failed", "registered": False},
                    headers=auth(exec_token))
    assert r.status_code == 201
    assert r.json()["registered"] is False


def test_domain_field_defaults_to_none_and_round_trips(client, exec_token):
    r = client.post("/api/admin/slm-versions",
                    json={"tag": "algo-akanade-slm:20260827", "base_tag": "qwen2.5:1.5b",
                          "quant_type": "Q4_K_M", "sanity_check": "passed"},
                    headers=auth(exec_token))
    assert r.json()["domain"] is None

    r = client.post("/api/admin/slm-versions",
                    json={"tag": "algo-akanade-slm-legal:20260827", "base_tag": "qwen2.5:1.5b",
                          "quant_type": "Q4_K_M", "sanity_check": "passed", "domain": "legal"},
                    headers=auth(exec_token))
    assert r.json()["domain"] == "legal"


def test_newest_version_listed_first(client, exec_token):
    for tag in ["algo-akanade-slm:20260101", "algo-akanade-slm:20260102"]:
        client.post("/api/admin/slm-versions",
                    json={"tag": tag, "base_tag": "qwen2.5:1.5b", "quant_type": "Q4_K_M",
                          "sanity_check": "passed", "regression_gate": "passed", "registered": True},
                    headers=auth(exec_token))
    r = client.get("/api/admin/slm-versions", headers=auth(exec_token))
    tags = [v["tag"] for v in r.json()]
    assert tags.index("algo-akanade-slm:20260102") < tags.index("algo-akanade-slm:20260101")
