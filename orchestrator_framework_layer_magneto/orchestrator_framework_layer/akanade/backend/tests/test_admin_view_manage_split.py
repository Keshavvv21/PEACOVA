"""Coverage for the "Admins manage, Admins + Senior Managers view" RBAC
split (admin_providers.py/admin_finetune_config.py/admin_eval_reports.py/
admin_provider_budgets.py/admin_slm.py): every GET/list endpoint on these
five routers was Executive-role-only (real app's Admin ONLY) before this
change; now it's Admin-role (which auth.py's _REAL_ROLE_MAP already maps
the real app's Manager2/"Senior Manager" onto) while every mutating
endpoint (POST/PUT/PATCH/DELETE) stays Executive-only, unchanged.

No prior test in this suite ever exercised `admin_token` (conftest.py's
bot-rank-"Admin" user, "meera" — the demo-store stand-in for a real
Manager2/Senior Manager) against ANY of these five routers at all — every
existing test here only used `exec_token` (bot rank "Executive") or
`member_token` (bot rank "Member"). This file is the first coverage of
the middle rank specifically.
"""
from __future__ import annotations

from tests.conftest import auth


def test_admin_rank_can_view_but_not_manage_providers(client, admin_token):
    r = client.get("/api/admin/providers", headers=auth(admin_token))
    assert r.status_code == 200, r.text

    create = client.post("/api/admin/providers", json={
        "name": "should-be-blocked", "base_url": "http://example.com", "model": "x",
        "api_key": "k", "tiers": ["bulk"],
    }, headers=auth(admin_token))
    assert create.status_code == 403


def test_admin_rank_can_view_but_not_manage_finetune_config(client, admin_token):
    r = client.get("/api/admin/finetune-config", headers=auth(admin_token))
    assert r.status_code == 200, r.text
    r2 = client.get("/api/admin/finetune-config/enabled", headers=auth(admin_token))
    assert r2.status_code == 200, r2.text

    put = client.put("/api/admin/finetune-config/general", json={
        "base_model": "should-be-blocked", "ollama_base_tag": "x",
    }, headers=auth(admin_token))
    assert put.status_code == 403


def test_admin_rank_can_view_but_not_manage_eval_reports(client, admin_token):
    r = client.get("/api/admin/slm-eval-reports", headers=auth(admin_token))
    assert r.status_code == 200, r.text

    post = client.post("/api/admin/slm-eval-reports", json={
        "tag": "x", "base_tag": "x", "tolerance": 0.05, "old_score": 0.8,
        "new_score": 0.81, "passed": True, "items": [],
    }, headers=auth(admin_token))
    assert post.status_code == 403


def test_admin_rank_can_view_but_not_manage_provider_budgets(client, admin_token):
    r = client.get("/api/admin/provider-budgets", headers=auth(admin_token))
    assert r.status_code == 200, r.text

    put = client.put("/api/admin/provider-budgets/some-provider",
                     json={"max_tokens": 1000}, headers=auth(admin_token))
    assert put.status_code == 403


def test_admin_rank_can_view_but_not_manage_slm_versions(client, admin_token):
    r = client.get("/api/admin/slm-versions", headers=auth(admin_token))
    assert r.status_code == 200, r.text

    post = client.post("/api/admin/slm-versions", json={
        "tag": "x", "base_tag": "x", "quant_type": "q4", "sanity_check": True,
    }, headers=auth(admin_token))
    assert post.status_code == 403


def test_member_rank_still_cannot_even_view(client, member_token):
    """Unchanged behavior — the view/manage split lowers the bar from
    Executive to Admin, not all the way down to Member."""
    for path in ("/api/admin/providers", "/api/admin/finetune-config",
                "/api/admin/slm-eval-reports", "/api/admin/provider-budgets",
                "/api/admin/slm-versions"):
        r = client.get(path, headers=auth(member_token))
        assert r.status_code == 403, f"{path} should still 403 for Member"
