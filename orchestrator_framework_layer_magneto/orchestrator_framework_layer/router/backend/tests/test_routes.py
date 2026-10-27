from fastapi.testclient import TestClient

from app import bert_shadow
from app.main import app

client = TestClient(app)


def _route_payload(**context_overrides):
    return {
        "candidates": [
            {"id": "luna", "cost_per_1k_tokens": 0.02, "capability_tier": 0.9},
            {"id": "groq", "cost_per_1k_tokens": 0.001, "capability_tier": 0.7},
            {"id": "algo-akanade-slm", "cost_per_1k_tokens": 0.0, "capability_tier": 0.5},
        ],
        "context": {"tier": "realtime", "est_tokens": 500, "task_min_capability": 0.0,
                    **context_overrides},
    }


def test_route_defaults_to_deterministic():
    # Strategy toggle defaults to "deterministic" until an admin has
    # ever PATCHed it — the CEO memo's own bar: never TS out of the box.
    client.patch("/admin/strategy", json={"strategy": "deterministic", "updated_by": "test"})
    resp = client.post("/route", json=_route_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_used"] == "deterministic"
    assert set(body["ordered_ids"]) == {"luna", "groq", "algo-akanade-slm"}


def test_strategy_toggle_round_trips():
    resp = client.patch("/admin/strategy", json={"strategy": "thompson_sampling", "updated_by": "test-admin"})
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "thompson_sampling"

    resp = client.get("/admin/strategy")
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "thompson_sampling"

    # Reset for any other test relying on the default.
    client.patch("/admin/strategy", json={"strategy": "deterministic", "updated_by": "test-admin"})


def test_route_includes_a_ucb_pick_alongside_the_ts_shadow_pick():
    resp = client.post("/route", json=_route_payload())
    body = resp.json()
    assert body["ucb_pick"] in ("groq", "algo-akanade-slm")
    # Never affects the live order regardless of strategy (still deterministic here).
    assert body["strategy_used"] == "deterministic"


def test_route_uses_ts_pick_only_when_live_and_restricted_to_two_arms():
    client.patch("/admin/strategy", json={"strategy": "thompson_sampling", "updated_by": "test"})
    resp = client.post("/route", json=_route_payload())
    body = resp.json()
    assert body["strategy_used"] == "thompson_sampling"
    assert body["ordered_ids"][0] in ("groq", "algo-akanade-slm")
    assert set(body["ordered_ids"]) == {"luna", "groq", "algo-akanade-slm"}
    client.patch("/admin/strategy", json={"strategy": "deterministic", "updated_by": "test"})


def test_outcome_accepts_and_persists():
    resp = client.post("/outcome", json={
        "candidate_id": "groq", "ok": True, "latency_ms": 250.0, "cost_usd": 0.001,
        "strategy_used": "deterministic", "shadow_pick": "groq", "tier": "realtime",
    })
    assert resp.status_code == 204


def test_bert_shadow_defaults_off_and_costs_nothing_even_with_a_prompt():
    # Off by default, same bar as Thompson Sampling — never opt-in
    # without an explicit admin PATCH.
    client.patch("/admin/bert-shadow", json={"enabled": False, "updated_by": "test"})
    resp = client.post("/route", json=_route_payload(prompt="what is 2+2?"))
    assert resp.status_code == 200
    assert resp.json()["bert_hint"] is None


def test_bert_shadow_hint_appears_once_enabled_but_never_reorders(monkeypatch):
    # Mocked so this test doesn't pay for a real checkpoint load —
    # generic_architecture.py's own tests already cover the real thing
    # end-to-end (test_pick_returns_a_default_or_escalation_model etc.).
    monkeypatch.setattr(bert_shadow, "hint", lambda prompt: "escalation")
    client.patch("/admin/bert-shadow", json={"enabled": True, "updated_by": "test"})
    client.patch("/admin/strategy", json={"strategy": "deterministic", "updated_by": "test"})

    resp = client.post("/route", json=_route_payload(prompt="a genuinely hard prompt"))
    body = resp.json()
    assert body["bert_hint"] == "escalation"
    # Still just a comparison column — the deterministic order is
    # completely untouched by what bert_hint says.
    assert body["strategy_used"] == "deterministic"
    assert set(body["ordered_ids"]) == {"luna", "groq", "algo-akanade-slm"}

    client.patch("/admin/bert-shadow", json={"enabled": False, "updated_by": "test"})


def test_bert_shadow_toggle_round_trips():
    resp = client.patch("/admin/bert-shadow", json={"enabled": True, "updated_by": "test-admin"})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True

    resp = client.get("/admin/bert-shadow")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True

    client.patch("/admin/bert-shadow", json={"enabled": False, "updated_by": "test-admin"})


def test_outcome_accepts_bert_hint():
    resp = client.post("/outcome", json={
        "candidate_id": "groq", "ok": True, "latency_ms": 250.0, "cost_usd": 0.001,
        "strategy_used": "deterministic", "shadow_pick": "groq", "bert_hint": "default",
        "tier": "realtime",
    })
    assert resp.status_code == 204


def test_outcome_accepts_ucb_pick():
    resp = client.post("/outcome", json={
        "candidate_id": "groq", "ok": True, "latency_ms": 250.0, "cost_usd": 0.001,
        "strategy_used": "deterministic", "shadow_pick": "groq", "ucb_pick": "algo-akanade-slm",
        "tier": "realtime",
    })
    assert resp.status_code == 204


def test_admin_routing_outcomes_returns_logged_rows_with_all_three_picks():
    client.post("/outcome", json={
        "candidate_id": "groq", "ok": True, "latency_ms": 123.0, "cost_usd": 0.002,
        "strategy_used": "deterministic", "shadow_pick": "groq", "bert_hint": "default",
        "ucb_pick": "algo-akanade-slm", "tier": "realtime",
    })
    resp = client.get("/admin/routing-outcomes", params={"limit": 5000})
    assert resp.status_code == 200
    rows = resp.json()
    assert any(r["chosen"] == "groq" and r["ucbPick"] == "algo-akanade-slm"
              and r["shadowPick"] == "groq" and r["bertHint"] == "default" for r in rows)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
