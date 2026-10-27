"""app/gmaofn/rag_optimizer.py — RL-selected (store, algorithm) retrieval,
intent-aware querying, and citation surfacing on top of the generic RAG
layer (rag.py). Mirrors test_token_optimizer.py's conventions.
"""
from __future__ import annotations

from app.framework.rag import RagDocument, InMemoryVectorStoreAdapter, rag_registry
from app.framework.rag_optimizer import (RagOptimizer, RagSelection, format_sources_note,
                                      is_rag_relevant, refine_query_for_intent,
                                      retrieve_with_optimizer)


def _store_with(*texts_and_meta: tuple[str, dict]) -> InMemoryVectorStoreAdapter:
    store = InMemoryVectorStoreAdapter(name="test-store")
    store.add_documents([RagDocument(text=t, metadata=m) for t, m in texts_and_meta])
    return store


# ── query refinement / relevance heuristics ──────────────────────────────

def test_refine_query_for_intent_strips_conversational_filler():
    refined = refine_query_for_intent("could you please tell me about the sprint delay")
    assert refined == "sprint delay"


def test_refine_query_for_intent_falls_back_to_original_when_all_filler():
    assert refine_query_for_intent("please could you") == "please could you"


def test_is_rag_relevant_true_for_question_shaped_messages():
    assert is_rag_relevant("what is causing the sprint delay?")
    assert is_rag_relevant("How do I set up the reporting dashboard")


def test_is_rag_relevant_false_for_a_plain_command():
    assert not is_rag_relevant("delete task 5")
    assert not is_rag_relevant("mark T2 as done")


# ── RagSelection / arm namespacing ───────────────────────────────────────

def test_rag_selection_arm_id_namespaces_store_and_algorithm():
    sel = RagSelection(store_name="pm-index", algorithm="hybrid", context=[0.0, 0.0])
    assert sel.arm_id == "pm-index::hybrid"


def test_select_with_a_single_store_and_single_algorithm_skips_the_bandit():
    opt = RagOptimizer(algorithms=("vector",))
    sel = opt.select("any query", ["only-store"])
    assert sel.store_name == "only-store"
    assert sel.algorithm == "vector"


# ── record_outcome / learning ────────────────────────────────────────────

def test_record_outcome_penalizes_a_pair_that_returns_nothing():
    opt = RagOptimizer()
    empty_result = RagSelection("s", "vector", [0.0, 0.0])
    got_hits = RagSelection("s", "hybrid", [0.0, 0.0])
    for _ in range(30):
        opt.record_outcome(empty_result, success=True, n_hits=0)   # "success" but no hits — still bad
        opt.record_outcome(got_hits, success=True, n_hits=3)
    mean_empty = opt._core.posterior_mean("s::vector") @ opt._core.features([0.0, 0.0])
    mean_hits = opt._core.posterior_mean("s::hybrid") @ opt._core.features([0.0, 0.0])
    assert mean_hits > mean_empty


def test_optimizer_learns_to_prefer_the_algorithm_that_actually_succeeds():
    opt = RagOptimizer(exploration_scale=0.2)
    context = [0.3, 0.3]
    for _ in range(40):
        opt._core.update("s::vector", context, -0.2)
        opt._core.update("s::hybrid", context, 1.0)
    picks = [opt._core.select(["s::vector", "s::hybrid"], context)[0] for _ in range(100)]
    assert picks.count("s::hybrid") / len(picks) > 0.75


# ── retrieve_with_optimizer() — the real end-to-end entry point ─────────

def test_retrieve_with_optimizer_ranks_the_relevant_document_first(monkeypatch):
    rag_registry.register("rag-opt-test", _store_with(
        ("The sprint velocity dropped due to unresolved blockers on task T2.",
         {"title": "sprint-notes"}),
        ("Unrelated document about cooking pasta recipes.", {"title": "recipes"}),
    ))
    result = retrieve_with_optimizer("what is causing the sprint delay?",
                                     store_names=["rag-opt-test"])
    assert result.n_hits == 2
    assert result.sources_used[0]["metadata"]["title"] == "sprint-notes"
    assert result.store_name == "rag-opt-test"
    assert result.algorithm in {"vector", "lexical_rerank", "hybrid"}


def test_retrieve_with_optimizer_reports_only_hits_that_fit_the_context_budget():
    rag_registry.register("rag-opt-budget-test",
                          _store_with(("x" * 3000, {"title": "big"}),
                                     ("y" * 3000, {"title": "also-big"})))
    result = retrieve_with_optimizer("x query", store_names=["rag-opt-budget-test"],
                                     max_context_chars=100)
    assert result.n_hits == 2                 # both retrieved
    assert len(result.sources_used) <= 1       # but the budget only fits one


def test_retrieve_with_optimizer_handles_an_empty_store_without_crashing():
    rag_registry.register("rag-opt-empty-test", InMemoryVectorStoreAdapter(name="empty"))
    result = retrieve_with_optimizer("anything", store_names=["rag-opt-empty-test"])
    assert result.n_hits == 0
    assert result.sources_used == []
    assert result.context == ""


# ── citation formatting ──────────────────────────────────────────────────

def test_format_sources_note_lists_titles():
    note = format_sources_note([{"index": 1, "score": 0.9, "metadata": {"title": "sprint-notes"}}])
    assert "sprint-notes" in note


def test_format_sources_note_empty_when_nothing_used():
    assert format_sources_note([]) == ""
