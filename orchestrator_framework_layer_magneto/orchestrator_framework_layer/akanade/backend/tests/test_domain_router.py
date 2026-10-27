"""llm/domain_router.py — "Domain" mode: RL-selected choice among
however many domain-specific Algo-Akanade-SLM variants are currently
registered, by content (embedding memory), not just complexity. Mirrors
test_hybrid_router.py's conventions.
"""
from __future__ import annotations

from app.llm.domain_router import (DOMAIN_PROVIDER_PREFIX, GENERAL_ARM, DomainRouter,
                                   DomainSelection, registered_domains)


def _fake_providers(*domains: str):
    return [{"id": f"{DOMAIN_PROVIDER_PREFIX}{d}", "name": f"algo-akanade-slm-{d}", "available": True}
           for d in domains] + [{"id": "ollama", "name": "ollama-qwen2.5:1.5b", "available": True}]


def test_registered_domains_reads_the_naming_convention(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers",
                        lambda: _fake_providers("legal", "support"))
    domains = registered_domains()
    assert domains == {"legal": "algo-akanade-slm-legal", "support": "algo-akanade-slm-support"}


def test_registered_domains_excludes_unavailable_and_non_domain_providers(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers", lambda: [
        {"id": "algo-akanade-slm-legal", "name": "x", "available": False},
        {"id": "ollama", "name": "ollama-qwen2.5:1.5b", "available": True},
    ])
    assert registered_domains() == {}


def test_choose_falls_back_to_general_when_no_domains_registered(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers", lambda: [])
    router = DomainRouter()
    selection = router.choose("draft a contract clause", "realtime")
    assert selection.arm == GENERAL_ARM


def test_choose_can_pick_a_registered_domain(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers",
                        lambda: _fake_providers("legal", "support"))
    router = DomainRouter()
    picks = {router.choose("some query", "realtime").arm for _ in range(50)}
    # Not asserting WHICH arm wins pre-training (Thompson Sampling is
    # randomized) — just that a real domain is a reachable outcome.
    assert picks & {"legal", "support", GENERAL_ARM}


def test_preferred_provider_for_maps_arms_to_router_keys():
    router = DomainRouter()
    assert router.preferred_provider_for(
        DomainSelection("legal", [0, 0, 0], "text")) == "algo-akanade-slm-legal"
    assert router.preferred_provider_for(
        DomainSelection(GENERAL_ARM, [0, 0, 0], "text")) is None


def test_is_confidently_relevant_rejects_general_arm():
    router = DomainRouter()
    assert router.is_confidently_relevant(
        DomainSelection(GENERAL_ARM, [0, 0, 0], "anything")) is False


def test_is_confidently_relevant_rejects_a_cold_domain_with_no_memory_yet():
    # Regression test for a real bug found in production: choose()'s core
    # bandit score is fit purely on message complexity (word/clause count,
    # tier), which has no notion of topic — a short trading question and a
    # short coding question look identical to it, so choose() could pick
    # e.g. "coding" for a message that has nothing to do with coding, with
    # zero prior evidence for or against. is_confidently_relevant() must
    # reject that pick (no memory match at all yet) even though choose()
    # itself returned a non-GENERAL_ARM winner.
    router = DomainRouter()
    selection = DomainSelection("coding", [0.1, 0.0, 0.0], "Fetch AAPL price history for 2024.")
    assert router.is_confidently_relevant(selection) is False


def test_is_confidently_relevant_accepts_a_genuine_content_match():
    router = DomainRouter()
    router.record_outcome(
        DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA for this vendor"), success=True)
    selection = DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA for this new vendor")
    assert router.is_confidently_relevant(selection) is True


def test_is_confidently_relevant_rejects_a_past_failure_on_a_similar_query():
    router = DomainRouter()
    router.record_outcome(
        DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA for this vendor"), success=False)
    selection = DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA for this new vendor")
    assert router.is_confidently_relevant(selection) is False


def test_record_outcome_updates_both_the_bandit_and_the_memory(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers",
                        lambda: _fake_providers("legal"))
    router = DomainRouter()
    selection = DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA for this vendor")
    router.record_outcome(selection, success=True)
    # The bandit learned something for "legal" ...
    assert router._core.posterior_mean("legal") is not None
    # ... AND the embedding memory remembers this exact kind of query
    # helped "legal" — a near-identical future query gets a positive bonus.
    bonus = router._memory.bonus_for("draft an NDA for this new vendor", "legal")
    assert bonus > 0


def test_a_past_failure_on_a_similar_query_lowers_the_memory_bonus(monkeypatch):
    router = DomainRouter()
    selection = DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA for this vendor")
    router.record_outcome(selection, success=False)
    bonus = router._memory.bonus_for("draft an NDA for this new vendor", "legal")
    assert bonus < 0


def test_record_outcome_persists_across_a_fresh_instance(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers",
                        lambda: _fake_providers("legal"))
    router1 = DomainRouter()
    selection = DomainSelection("legal", [0.1, 0.0, 0.0], "draft an NDA")
    router1.record_outcome(selection, success=True)

    router2 = DomainRouter()  # a FRESH instance, same persist name
    assert router2._core.posterior_mean("legal") is not None
    stats = {row["domain"]: row for row in router2.stats()}
    assert stats["legal"]["calls"] >= 1
    assert stats["legal"]["wins"] >= 1


def test_stats_reports_calls_wins_and_win_rate():
    router = DomainRouter()
    selection = DomainSelection("support", [0.1, 0.0, 0.0], "how do I reset my password")
    router.record_outcome(selection, success=True)
    router.record_outcome(selection, success=False)
    row = next(r for r in router.stats() if r["domain"] == "support")
    assert row["calls"] == 2
    assert row["wins"] == 1
    assert row["win_rate"] == 0.5
    assert row["posterior_mean_weights"] is not None


def test_router_learns_to_prefer_the_domain_with_the_higher_reward(monkeypatch):
    monkeypatch.setattr("app.llm.domain_router.available_providers",
                        lambda: _fake_providers("legal", "support"))
    router = DomainRouter(exploration_scale=0.2)
    context = [0.2, 0.2, 0.0]
    for _ in range(40):
        router._core.update("legal", context, 1.0)
        router._core.update("support", context, -0.5)
    picks = [router._core.select(["legal", "support"], context)[0] for _ in range(100)]
    assert picks.count("legal") / len(picks) > 0.75
