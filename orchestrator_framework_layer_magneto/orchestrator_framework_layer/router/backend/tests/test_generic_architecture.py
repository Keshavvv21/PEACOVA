from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import generic_architecture as ga
from app.generic_architecture import DEFAULT_TIER_MODELS, ESCALATION_TIER_MODELS, escalate, pick
from app.main import app

client = TestClient(app)


def test_pick_returns_a_default_or_escalation_model():
    result = pick("what is 2+2?")
    assert result["tier"] in ("default", "escalation")
    if result["tier"] == "default":
        assert result["model_name"] in DEFAULT_TIER_MODELS
    else:
        assert result["model_name"] in ESCALATION_TIER_MODELS


def test_escalate_always_returns_an_escalation_tier_model():
    result = escalate("anything")
    assert result["tier"] == "escalation"
    assert result["model_name"] in ESCALATION_TIER_MODELS


def test_pick_never_raises_even_without_a_real_openai_key():
    # The whole point of the per-call "mf" -> "random" fallback in
    # generic_architecture.py: no valid OPENAI_API_KEY is configured in
    # this test environment (or most real deployments of this
    # codebase, which use custom OpenAI-COMPATIBLE endpoints, never
    # OpenAI itself) — pick() must degrade gracefully, never raise.
    for _ in range(5):
        result = pick("a reasonably complex prompt to score")
        assert result["tier"] in ("default", "escalation")


def test_decide_tier_falls_through_mf_to_bert_when_mf_call_fails(monkeypatch):
    # "mf" constructed fine but fails at CALL time (no real OpenAI key,
    # the real-world case) — bert should decide instead, and mf should
    # never be asked again on this call.
    fake_controller = MagicMock()
    fake_controller.routers = {"mf": object(), "bert": object(), "random": object()}

    def fake_route(prompt, router, threshold):
        if router == "mf":
            raise RuntimeError("401 unauthorized")
        if router == "bert":
            return ga._STRONG_LABEL
        raise AssertionError(f"should not reach router {router!r}")

    fake_controller.route.side_effect = fake_route
    monkeypatch.setattr(ga, "_get_controller", lambda: fake_controller)

    assert ga._decide_tier("anything") == ga._STRONG_LABEL


def test_decide_tier_skips_a_router_that_never_constructed(monkeypatch):
    # sw_ranking failed at CONSTRUCTION time (see _get_controller) and
    # so is simply absent from controller.routers — _decide_tier must
    # skip it without even attempting a call, going straight to
    # whichever candidate actually exists.
    fake_controller = MagicMock()
    fake_controller.routers = {"bert": object(), "random": object()}  # no "mf", no "sw_ranking"
    fake_controller.route.return_value = ga._WEAK_LABEL
    monkeypatch.setattr(ga, "_get_controller", lambda: fake_controller)

    assert ga._decide_tier("anything") == ga._WEAK_LABEL
    called_routers = [call.kwargs["router"] for call in fake_controller.route.call_args_list]
    assert called_routers == ["bert"]


def test_decide_tier_falls_back_to_random_when_every_candidate_fails(monkeypatch):
    fake_controller = MagicMock()
    fake_controller.routers = {"mf": object(), "bert": object(), "random": object()}

    def fake_route(prompt, router, threshold):
        if router == "random":
            return ga._WEAK_LABEL
        raise RuntimeError(f"{router} failed")

    fake_controller.route.side_effect = fake_route
    monkeypatch.setattr(ga, "_get_controller", lambda: fake_controller)

    assert ga._decide_tier("anything") == ga._WEAK_LABEL


def test_route_endpoint_accepts_a_prompt():
    resp = client.post("/generic-architecture/route", json={"prompt": "hello"})
    assert resp.status_code == 200
    assert resp.json()["tier"] in ("default", "escalation")


def test_route_endpoint_works_with_no_body():
    resp = client.post("/generic-architecture/route")
    assert resp.status_code == 200


def test_escalate_endpoint():
    resp = client.post("/generic-architecture/escalate", json={"prompt": "anything"})
    assert resp.status_code == 200
    assert resp.json()["tier"] == "escalation"


def test_outcome_endpoint_persists_with_candidate_model_source():
    resp = client.post("/generic-architecture/outcome", json={
        "model_name": "glm-5.3-flash", "tier": "default", "ok": True,
        "latency_ms": 400.0, "cost_usd": 0.002, "quality_score": 0.8,
        "prompt_tokens": 120, "completion_tokens": 80, "task_family": "reasoning",
    })
    assert resp.status_code == 204
