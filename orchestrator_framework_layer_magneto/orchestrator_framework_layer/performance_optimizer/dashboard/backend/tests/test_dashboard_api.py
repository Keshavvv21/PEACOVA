"""End-to-end tests against a REAL temp data directory shaped exactly
like app/db.py's LocalStore writes it (this dashboard reads Algo Akanade's
files directly, see events_store.py's docstring) — no mocking of the
storage layer, so a real format drift between the two services would
actually fail this suite.
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("ALGO_AKANADE_DATA_DIR", str(d))
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "admin")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "admin")
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "test-secret")
    from app.config import settings
    settings.cache_clear()
    yield d
    settings.cache_clear()


def _write_events(data_dir, date: str, rows: list[dict]):
    path = data_dir / "OptimizationEvents.json"
    table = {}
    if path.exists():
        with open(path) as f:
            table = json.load(f)
    for i, row in enumerate(rows):
        pk, sk = date, f"EVENT#{date}T00:00:{i:02d}#e{i}"
        table[f"{pk}||{sk}"] = {**row, "PK": pk, "SK": sk, "occurred_at": f"{date}T00:00:{i:02d}"}
    with open(path, "w") as f:
        json.dump(table, f)


def _client():
    from app.main import app
    return TestClient(app)


def test_login_rejects_wrong_password(data_dir):
    c = _client()
    r = c.post("/api/login", json={"username": "admin", "password": "nope"})
    assert r.status_code == 401


def test_login_accepts_configured_admin_and_summary_requires_it(data_dir):
    c = _client()
    r = c.post("/api/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200
    token = r.json()["access_token"]

    unauth = c.get("/api/summary")
    assert unauth.status_code == 401

    auth = c.get("/api/summary", headers={"Authorization": f"Bearer {token}"})
    assert auth.status_code == 200
    body = auth.json()
    assert "gpu" in body and "kernel_tiers" in body["gpu"]
    assert len(body["gpu"]["kernel_tiers"]) == 3


def test_daily_report_computes_real_delta_between_two_days(data_dir):
    _write_events(data_dir, "2026-08-27", [
        {"node": "n1", "tier": "realtime", "provider": "groq", "arm_id": "groq::balanced",
         "ok": True, "latency_ms": 800.0, "fallback_used": False},
        {"node": "n1", "tier": "realtime", "provider": "groq", "arm_id": "groq::balanced",
         "ok": False, "latency_ms": 400.0, "fallback_used": False},
    ])
    _write_events(data_dir, "2026-08-28", [
        {"node": "n1", "tier": "realtime", "provider": "groq", "arm_id": "groq::low_latency",
         "ok": True, "latency_ms": 300.0, "fallback_used": False},
        {"node": "n1", "tier": "realtime", "provider": "groq", "arm_id": "groq::low_latency",
         "ok": True, "latency_ms": 300.0, "fallback_used": False},
    ])

    c = _client()
    token = c.post("/api/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
    r = c.get("/api/daily", params={"date": "2026-08-28"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()

    assert body["today"]["n_calls"] == 2
    assert body["today"]["success_rate"] == 1.0
    assert body["today"]["avg_latency_ms"] == 300.0
    assert body["previous"]["success_rate"] == 0.5
    # real improvement: success rate up, latency down, day over day
    assert body["delta"]["success_rate"] == pytest.approx(0.5)
    assert body["delta"]["avg_latency_ms"] == pytest.approx(-300.0)


def test_events_endpoint_returns_chronological_order(data_dir):
    _write_events(data_dir, "2026-08-28", [
        {"node": "a", "tier": "bulk", "provider": "deepseek", "arm_id": "deepseek::balanced",
         "ok": True, "latency_ms": 100.0, "fallback_used": False},
        {"node": "b", "tier": "bulk", "provider": "deepseek", "arm_id": "deepseek::balanced",
         "ok": True, "latency_ms": 150.0, "fallback_used": True},
    ])
    c = _client()
    token = c.post("/api/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
    r = c.get("/api/events", params={"date": "2026-08-28"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    events = r.json()["events"]
    assert [e["node"] for e in events] == ["a", "b"]


def test_provider_rl_posteriors_read_real_bandit_state_shape(data_dir):
    # Exactly the shape framework/bandit_persistence.save() writes:
    # {"arms": {"{tier}::{provider}": {"B": [[x]], "f": [y]}}}
    state = {
        "OPTIMIZER#provider_optimizer||STATE": {
            "PK": "OPTIMIZER#provider_optimizer", "SK": "STATE",
            "arms": {
                "realtime::luna": {"B": [[2.0]], "f": [1.6]},
                "realtime::groq": {"B": [[2.0]], "f": [0.8]},
            },
        }
    }
    with open(data_dir / "BanditState.json", "w") as f:
        json.dump(state, f)

    c = _client()
    token = c.post("/api/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
    r = c.get("/api/summary", headers={"Authorization": f"Bearer {token}"})
    posteriors = r.json()["provider_rl_posteriors"]
    assert posteriors["realtime"]["luna"] == pytest.approx(0.8)
    assert posteriors["realtime"]["groq"] == pytest.approx(0.4)
    # sorted highest posterior mean first, matching what order() acts on
    assert list(posteriors["realtime"].keys()) == ["luna", "groq"]
