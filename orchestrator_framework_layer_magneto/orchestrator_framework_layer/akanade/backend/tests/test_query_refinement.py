"""gmaofn/query_refinement.py — the query refinement layer sitting above
every chat track. Mirrors test_hybrid_router.py's conventions.
"""
from __future__ import annotations

import json

import pytest

from app.framework.query_refinement import (DriftAuditLog, QueryRefinementPipeline,
                                         RefinementMemory, RefinementSelection,
                                         _complexity_features, embedding_similarity)
from app.llm.providers import LLMResult
from app.llm.router import TierClient
from tests.conftest import auth


def test_complexity_features_reward_longer_filler_heavy_messages_with_higher_values():
    simple = _complexity_features("what is overdue")
    fillerish = _complexity_features("could you please tell me about the overdue tasks and status")
    assert fillerish[0] > simple[0]  # more words
    assert fillerish[1] > simple[1]  # more filler phrases


def test_complexity_features_flags_missing_terminal_punctuation():
    no_punct = _complexity_features("show me all the overdue tasks assigned to the whole team today")
    with_punct = no_punct  # baseline itself has no punctuation
    punctuated = _complexity_features("Show me all the overdue tasks assigned to the whole team today.")
    assert with_punct[2] == 1.0
    assert punctuated[2] == 0.0


def test_embedding_similarity_is_high_for_near_identical_text_low_for_unrelated():
    same_ish = embedding_similarity("show me overdue tasks", "display the overdue tasks")
    unrelated = embedding_similarity("show me overdue tasks", "delete the production database now")
    assert same_ish > unrelated


def test_refine_skips_when_gate_picks_skip(monkeypatch):
    pipeline = QueryRefinementPipeline()
    monkeypatch.setattr(pipeline._core, "select", lambda arms, ctx: ("skip", {}))
    monkeypatch.setattr(TierClient, "chat", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("skip must never call the LLM")))
    result = pipeline.refine("what is overdue")
    assert result.refined_text == "what is overdue"
    assert result.selection.arm == "skip"


def test_refine_returns_refined_text_when_llm_agrees_and_similarity_is_high(monkeypatch):
    pipeline = QueryRefinementPipeline()
    monkeypatch.setattr(pipeline._core, "select", lambda arms, ctx: ("refine", {}))
    payload = json.dumps({"refined": "Show me the overdue tasks.",
                          "missing_clarifications": [], "suggested_queries": ["Who owns them?"]})
    monkeypatch.setattr(TierClient, "chat",
                        lambda self, node, messages, tools=None: LLMResult(text=payload, provider="fake-llm"))
    result = pipeline.refine("show me overdue tasks")
    assert result.refined_text == "Show me the overdue tasks."
    assert result.was_repaired is False
    assert result.suggested_queries == ["Who owns them?"]


def test_refine_reverts_to_original_when_llm_rewrite_drifts_too_far(monkeypatch):
    pipeline = QueryRefinementPipeline()
    monkeypatch.setattr(pipeline._core, "select", lambda arms, ctx: ("refine", {}))
    payload = json.dumps({"refined": "Delete every project in the workspace immediately.",
                          "missing_clarifications": [], "suggested_queries": []})
    monkeypatch.setattr(TierClient, "chat",
                        lambda self, node, messages, tools=None: LLMResult(text=payload, provider="fake-llm"))
    result = pipeline.refine("what is overdue on my board")
    assert result.was_repaired is True
    assert result.refined_text == "what is overdue on my board"  # reverted, not the drifted rewrite
    assert result.needs_confirmation is True


def test_refine_falls_back_to_original_on_llm_error(monkeypatch):
    pipeline = QueryRefinementPipeline()
    monkeypatch.setattr(pipeline._core, "select", lambda arms, ctx: ("refine", {}))
    monkeypatch.setattr(TierClient, "chat",
                        lambda self, node, messages, tools=None: LLMResult(error="boom", provider="none"))
    result = pipeline.refine("show me overdue tasks")
    assert result.refined_text == "show me overdue tasks"
    assert result.selection.llm_error is True


def test_record_outcome_penalizes_a_skip_failure_harder_than_a_wasted_refine():
    pipeline = QueryRefinementPipeline()
    skip_fail = RefinementSelection(arm="skip", context=[0.0, 0.0, 0.0])
    refine_noop = RefinementSelection(arm="refine", context=[0.0, 0.0, 0.0], text_changed=False)
    for _ in range(30):
        pipeline.record_outcome(skip_fail, success=False)
        pipeline.record_outcome(refine_noop, success=True)
    mean_skip = pipeline._core.posterior_mean("skip") @ pipeline._core.features([0.0, 0.0, 0.0])
    mean_refine = pipeline._core.posterior_mean("refine") @ pipeline._core.features([0.0, 0.0, 0.0])
    assert mean_skip < mean_refine


def test_drift_audit_log_records_only_when_called_and_uses_quantized_storage():
    log = DriftAuditLog(maxlen=5)
    assert log.snapshot() == []
    log.record("show me overdue tasks", "delete the database", 0.1)
    snap = log.snapshot()
    assert len(snap) == 1
    assert snap[0]["original"] == "show me overdue tasks"
    assert snap[0]["similarity"] == 0.1


def test_drift_audit_log_is_bounded():
    log = DriftAuditLog(maxlen=3)
    for i in range(10):
        log.record(f"query {i}", f"drifted {i}", 0.1)
    assert len(log.snapshot()) == 3


def test_refinement_memory_hits_on_near_duplicate_and_misses_on_unrelated():
    memory = RefinementMemory()
    memory.remember("show me overdue tasks", "Show me the overdue tasks.")
    assert memory.lookup("show me overdue tasks") == "Show me the overdue tasks."
    assert memory.lookup("delete every project immediately") is None


def test_refinement_memory_is_bounded():
    memory = RefinementMemory(maxlen=3)
    for i in range(10):
        memory.remember(f"query number {i}", f"refined query number {i}")
    assert len(memory) == 3


def test_refine_reuses_memory_on_repeat_query_without_calling_the_llm(monkeypatch):
    pipeline = QueryRefinementPipeline()
    monkeypatch.setattr(pipeline._core, "select", lambda arms, ctx: ("refine", {}))
    payload = json.dumps({"refined": "Show me the overdue tasks.",
                          "missing_clarifications": [], "suggested_queries": []})
    calls = {"n": 0}

    def fake_chat(self, node, messages, tools=None):
        calls["n"] += 1
        return LLMResult(text=payload, provider="fake-llm")

    monkeypatch.setattr(TierClient, "chat", fake_chat)
    first = pipeline.refine("show me overdue tasks")
    assert calls["n"] == 1
    assert first.selection.from_memory is False

    second = pipeline.refine("show me overdue tasks")
    assert calls["n"] == 1  # memory hit — no second LLM call
    assert second.refined_text == "Show me the overdue tasks."
    assert second.selection.from_memory is True


def test_record_outcome_barely_penalizes_a_memory_hit_even_on_failure(monkeypatch):
    pipeline = QueryRefinementPipeline()
    rewards = []
    monkeypatch.setattr(pipeline._core, "update", lambda arm_id, ctx, reward: rewards.append(reward))
    memory_fail = RefinementSelection(arm="refine", context=[0.0, 0.0, 0.0], from_memory=True)
    llm_error_fail = RefinementSelection(arm="refine", context=[0.0, 0.0, 0.0], llm_error=True)
    pipeline.record_outcome(memory_fail, success=False)
    pipeline.record_outcome(llm_error_fail, success=False)
    assert rewards[0] > rewards[1]  # memory-hit failure penalized far less than an LLM-error failure



# ── real end-to-end: refinement runs transparently through /api/chat/message ──

def test_refinement_field_present_when_gate_chooses_refine(client, exec_token, monkeypatch):
    from app.framework.query_refinement import query_refinement
    monkeypatch.setattr(query_refinement._core, "select", lambda arms, ctx: ("refine", {}))
    payload = json.dumps({"refined": "What tasks are overdue?",
                          "missing_clarifications": ["Which project?"], "suggested_queries": []})
    monkeypatch.setattr(TierClient, "chat",
                        lambda self, node, messages, tools=None:
                        LLMResult(text=payload, provider="fake-llm") if node == "query_refinement"
                        else LLMResult(text="Nothing is overdue.", provider="fake-llm"))
    r = client.post("/api/chat/message",
                    json={"message": "what tasks are overdue", "session_id": "refine-e2e"},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["refinement"] is not None
    assert d["refinement"]["missing_clarifications"] == ["Which project?"]


def test_refinement_field_absent_when_gate_chooses_skip(client, exec_token, monkeypatch):
    from app.framework.query_refinement import query_refinement
    monkeypatch.setattr(query_refinement._core, "select", lambda arms, ctx: ("skip", {}))
    r = client.post("/api/chat/message",
                    json={"message": "what is overdue", "session_id": "skip-e2e"},
                    headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert r.json()["refinement"] is None


def test_refinement_audit_endpoint_is_admin_only(client, exec_token):
    r = client.get("/api/chat/refinement-audit", headers=auth(exec_token))
    assert r.status_code == 200, r.text
    assert "drifted" in r.json()
