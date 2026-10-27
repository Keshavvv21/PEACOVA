"""REST surface for the Token & Cost Optimizer layer:
- POST /api/v1/cost/predict — any authenticated user, predicts token
  count and $ cost for a hypothetical message BEFORE it's ever sent to
  a real provider.
- GET /api/v1/metrics/cost — Admin-only, real tracked cost/token/
  latency/quality stats plus a whole-system savings summary.
"""
from __future__ import annotations

from tests.conftest import auth


def test_predict_cost_returns_one_estimate_per_available_provider(client, member_token):
    r = client.post("/api/v1/cost/predict",
                    json={"message": "Backtest AAPL for the last 10 years", "tier": "realtime"},
                    headers=auth(member_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tier"] == "realtime"
    assert isinstance(body["estimates"], list)
    for est in body["estimates"]:
        assert set(est) == {"provider", "predicted_prompt_tokens",
                            "predicted_completion_tokens", "predicted_cost_usd"}
        assert est["predicted_prompt_tokens"] > 0
        assert est["predicted_cost_usd"] >= 0.0


def test_predict_cost_estimates_are_sorted_cheapest_first(client, member_token):
    r = client.post("/api/v1/cost/predict",
                    json={"message": "hello", "tier": "realtime"},
                    headers=auth(member_token))
    assert r.status_code == 200, r.text
    costs = [e["predicted_cost_usd"] for e in r.json()["estimates"]]
    assert costs == sorted(costs)


def test_predict_cost_requires_authentication(client):
    r = client.post("/api/v1/cost/predict", json={"message": "hi", "tier": "realtime"})
    assert r.status_code in (401, 403)


def test_predict_cost_defaults_to_realtime_tier(client, member_token):
    r = client.post("/api/v1/cost/predict", json={"message": "hi"}, headers=auth(member_token))
    assert r.status_code == 200, r.text
    assert r.json()["tier"] == "realtime"


def test_cost_metrics_requires_admin(client, member_token):
    r = client.get("/api/v1/metrics/cost", headers=auth(member_token))
    assert r.status_code == 403


def test_cost_metrics_returns_per_provider_and_summary(client, admin_token):
    r = client.get("/api/v1/metrics/cost", headers=auth(admin_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "per_provider" in body and isinstance(body["per_provider"], list)
    assert "summary" in body
    for key in ("total_cost_usd", "total_tokens", "counterfactual_cost_usd",
               "estimated_savings_usd", "estimated_savings_pct"):
        assert key in body["summary"]
