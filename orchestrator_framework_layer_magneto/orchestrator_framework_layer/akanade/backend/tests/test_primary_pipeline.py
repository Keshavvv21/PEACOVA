"""app/llm/primary_pipeline.py — the three-way switch between RouteLLM
("routellm", primary+secondary/default, or "routellm_only", primary
only) and Akanade's own chain ("akanade", secondary only) for real live chat
responses, plus /api/admin/router/primary-pipeline.
"""
from __future__ import annotations

from app.llm import primary_pipeline
from tests.conftest import auth


def test_defaults_to_routellm_as_primary():
    # A fresh DB row (never PATCHed) must default to "routellm" — the
    # explicit "primary is his [RouteLLM], secondary is akanade" instruction.
    assert primary_pipeline.get_pipeline() == "routellm"


def test_set_and_get_pipeline():
    primary_pipeline.set_pipeline("akanade", "test")
    assert primary_pipeline.get_pipeline() == "akanade"
    primary_pipeline.set_pipeline("routellm_only", "test")
    assert primary_pipeline.get_pipeline() == "routellm_only"
    primary_pipeline.set_pipeline("routellm", "test")
    assert primary_pipeline.get_pipeline() == "routellm"


def test_routellm_modes_constant_covers_both_routellm_settings():
    # Both "routellm" (primary+secondary) and "routellm_only" (primary
    # only, still auto-failover per the admin's own decision) must
    # consult RouteLLM — only "akanade" skips the call entirely.
    assert set(primary_pipeline.ROUTELLM_MODES) == {"routellm", "routellm_only"}
    assert "akanade" not in primary_pipeline.ROUTELLM_MODES


def test_set_pipeline_rejects_unknown_value():
    try:
        primary_pipeline.set_pipeline("something-else", "test")
        assert False, "should have raised"
    except ValueError:
        pass


def test_registered_model_map_has_no_dangling_unmapped_candidates():
    # generic_architecture.py's tier lists were pared down (after the first
    # A/B test) to exactly the candidates with a real provider here —
    # every entry should resolve to a real key, never None, today.
    assert all(v is not None for v in primary_pipeline.REGISTERED_MODEL_MAP.values())


def test_registered_model_map_returns_none_for_an_unknown_pick():
    # .get() semantics, not [] — a future tier candidate this map
    # doesn't know about must fail safe (fall through), not KeyError.
    assert primary_pipeline.REGISTERED_MODEL_MAP.get("some-future-model") is None


def test_primary_pipeline_endpoint_requires_admin_to_view(client, member_token):
    r = client.get("/api/admin/router/primary-pipeline", headers=auth(member_token))
    assert r.status_code == 403


def test_primary_pipeline_endpoint_requires_executive_to_flip(client, admin_token):
    r = client.patch("/api/admin/router/primary-pipeline", json={"pipeline": "akanade"},
                     headers=auth(admin_token))
    assert r.status_code == 403


def test_primary_pipeline_endpoint_round_trips(client, exec_token):
    r = client.get("/api/admin/router/primary-pipeline", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert r.json()["pipeline"] == "routellm"

    r2 = client.patch("/api/admin/router/primary-pipeline", json={"pipeline": "akanade"},
                      headers=auth(exec_token))
    assert r2.status_code == 200, r2.text
    assert r2.json()["pipeline"] == "akanade"

    r3 = client.get("/api/admin/router/primary-pipeline", headers=auth(exec_token))
    assert r3.json()["pipeline"] == "akanade"

    r4 = client.patch("/api/admin/router/primary-pipeline", json={"pipeline": "routellm_only"},
                      headers=auth(exec_token))
    assert r4.status_code == 200, r4.text
    assert r4.json()["pipeline"] == "routellm_only"

    # Reset for other tests.
    client.patch("/api/admin/router/primary-pipeline", json={"pipeline": "routellm"},
                 headers=auth(exec_token))


def test_primary_pipeline_endpoint_rejects_unknown_value(client, exec_token):
    r = client.patch("/api/admin/router/primary-pipeline", json={"pipeline": "bogus"},
                     headers=auth(exec_token))
    assert r.status_code == 422


def test_resolve_routellm_preferred_returns_none_when_ak_router_unreachable():
    # No ak-router running in this test process — must degrade to None
    # (Akanade's own chain), never raise.
    assert primary_pipeline.resolve_routellm_preferred("write a function") is None
