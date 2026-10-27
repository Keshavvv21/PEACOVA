"""Tests for /api/admin/provider-budgets (app.routers.admin_provider_budgets)."""

from tests.conftest import auth


def test_member_cannot_manage_provider_budgets(client, member_token):
    r = client.get("/api/admin/provider-budgets", headers=auth(member_token))
    assert r.status_code == 403

    r = client.put("/api/admin/provider-budgets/claude",
                   json={"max_tokens": 1000}, headers=auth(member_token))
    assert r.status_code == 403


def test_set_a_token_budget_and_list_it(client, exec_token):
    r = client.put("/api/admin/provider-budgets/claude",
                   json={"max_tokens": 100000}, headers=auth(exec_token))
    assert r.status_code == 200
    body = r.json()
    assert body["provider_key"] == "claude"
    assert body["max_tokens"] == 100000
    assert body["today_tokens"] == 0
    assert body["exceeded"] is False

    r = client.get("/api/admin/provider-budgets", headers=auth(exec_token))
    assert any(b["provider_key"] == "claude" for b in r.json())


def test_max_cost_without_a_rate_is_rejected(client, exec_token):
    r = client.put("/api/admin/provider-budgets/claude-2",
                   json={"max_cost_usd": 5.0}, headers=auth(exec_token))
    assert r.status_code == 400


def test_max_cost_with_a_rate_is_accepted(client, exec_token):
    r = client.put("/api/admin/provider-budgets/claude-3",
                   json={"max_cost_usd": 5.0, "cost_per_1k_prompt_tokens": 0.003,
                         "cost_per_1k_completion_tokens": 0.015},
                   headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["max_cost_usd"] == 5.0


def test_delete_budget(client, exec_token):
    client.put("/api/admin/provider-budgets/temp-provider",
              json={"max_tokens": 1}, headers=auth(exec_token))
    r = client.delete("/api/admin/provider-budgets/temp-provider", headers=auth(exec_token))
    assert r.status_code == 204

    r = client.delete("/api/admin/provider-budgets/temp-provider", headers=auth(exec_token))
    assert r.status_code == 404
