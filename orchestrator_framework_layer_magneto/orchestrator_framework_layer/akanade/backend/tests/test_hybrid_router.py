"""llm/hybrid_router.py — the "Hybrid" mode RL router: complicated
requests toward hosted models, easy ones toward the free local SLM.
Mirrors test_token_optimizer.py's / test_rag_optimizer.py's conventions.
"""
from __future__ import annotations

import pytest

from app.llm.hybrid_router import HybridRouter, HybridSelection, _complexity_features
from tests.conftest import auth


def test_complexity_features_reward_longer_multi_clause_messages_with_higher_values():
    simple = _complexity_features("what is overdue", "realtime")
    complex_msg = _complexity_features(
        "create a task, then assign it to Priya, and set the due date, and notify the team",
        "delay_ok")
    assert complex_msg[0] > simple[0]   # more words
    assert complex_msg[1] > simple[1]   # more clauses
    assert complex_msg[2] == 1.0 and simple[2] == 0.0  # tier flag


def test_choose_returns_hosted_when_local_provider_unavailable(monkeypatch):
    monkeypatch.setattr("app.llm.hybrid_router._local_provider_available", lambda local_provider: False)
    router = HybridRouter()
    selection = router.choose("anything", "realtime")
    assert selection.arm == "hosted"


def test_choose_can_pick_local_once_it_has_real_fast_evidence(monkeypatch):
    """"local" is deliberately biased AGAINST at cold start (see
    choose()'s own seed_utilities={"local": -2.0} and the class
    docstring's reasoning — real, repeated measurement found it
    severely slower on this deployment), never hard-disabled: enough
    real, genuinely FAST local successes still make it win. This
    replaces the old "blind 50-draw cold start" version of this test,
    which asserted the exact coin-flip behavior this fix intentionally
    removed."""
    monkeypatch.setattr("app.llm.hybrid_router._local_provider_available", lambda local_provider: True)
    router = HybridRouter()
    fast_local_win = HybridSelection("local", [0.0, 0.0, 0.0])
    for _ in range(15):
        router.record_outcome(fast_local_win, success=True, latency_ms=200)
    picks = {router.choose("hi", "realtime").arm for _ in range(50)}
    assert "local" in picks


def test_cold_start_strongly_favors_hosted_over_a_blind_coin_flip(monkeypatch):
    """The actual bug this locks in: a fresh process (no persistence —
    see HybridRouter's own docstring) used to pick "local" vs "hosted"
    as a near coin-flip on the very first few calls, before any real
    evidence existed — confirmed live to repeatedly cost 8-100+ seconds
    per turn on this deployment. Cold start must now clearly favor
    "hosted" by default, never a 50/50 gamble on an option already
    known (from every other call site's own real measurements) to be
    severely slower here."""
    monkeypatch.setattr("app.llm.hybrid_router._local_provider_available", lambda local_provider: True)
    router = HybridRouter()
    picks = [router.choose(f"read file number {i}", "realtime").arm for i in range(30)]
    assert picks.count("hosted") > picks.count("local")


def test_preferred_provider_for_maps_arms_to_router_keys():
    router = HybridRouter()
    assert router.preferred_provider_for(HybridSelection("local", [0, 0, 0])) == "ollama"
    assert router.preferred_provider_for(HybridSelection("hosted", [0, 0, 0])) is None


def test_router_learns_to_prefer_the_arm_with_the_higher_reward(monkeypatch):
    monkeypatch.setattr("app.llm.hybrid_router._local_provider_available", lambda local_provider: True)
    router = HybridRouter(exploration_scale=0.2)
    context = [0.2, 0.2, 0.0]
    for _ in range(40):
        router._core.update("local", context, 1.0)   # local keeps succeeding
        router._core.update("hosted", context, -0.3)  # hosted keeps "failing" here
    picks = [router._core.select(["local", "hosted"], context)[0] for _ in range(100)]
    assert picks.count("local") / len(picks) > 0.75


def test_record_outcome_penalizes_a_local_failure_harder_than_a_hosted_one():
    router = HybridRouter()
    local_fail = HybridSelection("local", [0.0, 0.0, 0.0])
    hosted_fail = HybridSelection("hosted", [0.0, 0.0, 0.0])
    for _ in range(30):
        router.record_outcome(local_fail, success=False)
        router.record_outcome(hosted_fail, success=False)
    mean_local = router._core.posterior_mean("local") @ router._core.features([0.0, 0.0, 0.0])
    mean_hosted = router._core.posterior_mean("hosted") @ router._core.features([0.0, 0.0, 0.0])
    assert mean_local < mean_hosted  # local failure scored worse


def test_record_outcome_rewards_a_local_success_more_than_a_hosted_one():
    router = HybridRouter()
    local_ok = HybridSelection("local", [0.0, 0.0, 0.0])
    hosted_ok = HybridSelection("hosted", [0.0, 0.0, 0.0])
    for _ in range(30):
        router.record_outcome(local_ok, success=True)
        router.record_outcome(hosted_ok, success=True)
    mean_local = router._core.posterior_mean("local") @ router._core.features([0.0, 0.0, 0.0])
    mean_hosted = router._core.posterior_mean("hosted") @ router._core.features([0.0, 0.0, 0.0])
    assert mean_local > mean_hosted  # free success scored better than paid success


def test_a_slow_local_success_scores_worse_than_a_fast_hosted_success():
    """Real gap found live: record_outcome() used to have NO latency
    term at all, so a "local" success always scored 0.5 higher than a
    "hosted" success regardless of how slow it actually was. On the real
    deployment, an ordinary local tool-calling turn measured 20-30+
    seconds next to a 1-5s hosted answer — this confirms a genuinely
    slow local win no longer systematically outranks a fast hosted one,
    the actual behavior change needed to stop the bandit from
    preferring a rung that's 10-20x slower just because it's free."""
    router = HybridRouter()
    local_slow_ok = HybridSelection("local", [0.0, 0.0, 0.0])
    hosted_fast_ok = HybridSelection("hosted", [0.0, 0.0, 0.0])
    for _ in range(30):
        router.record_outcome(local_slow_ok, success=True, latency_ms=25_000)
        router.record_outcome(hosted_fast_ok, success=True, latency_ms=1_500)
    mean_local = router._core.posterior_mean("local") @ router._core.features([0.0, 0.0, 0.0])
    mean_hosted = router._core.posterior_mean("hosted") @ router._core.features([0.0, 0.0, 0.0])
    assert mean_hosted > mean_local


# ── real end-to-end: "hybrid" as a provider value through /api/chat/message ──

TRACKS = ["native", "langgraph", "adk"]


@pytest.mark.parametrize("track", TRACKS)
def test_hybrid_provider_value_answers_normally_through_the_real_chat_endpoint(
        client, exec_token, track):
    """"hybrid" isn't a real provider name (get_llm_client would reject
    it as `preferred`) — this proves the interception in each track
    actually resolves it to a real provider (or None) before that call,
    rather than the sentinel leaking through and breaking the turn."""
    r = client.post(f"/api/chat/message?framework={track}",
                    json={"message": "what is overdue in Compass OS",
                          "session_id": f"hybrid-{track}", "provider": "hybrid"},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provider"]  # a REAL provider answered, not the literal "hybrid"
    assert d["provider"] != "hybrid"
