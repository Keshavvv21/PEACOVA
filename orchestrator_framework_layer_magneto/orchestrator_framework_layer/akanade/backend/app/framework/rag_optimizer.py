"""RAG optimizer — RL-selected best knowledge source AND retrieval
algorithm, intent-aware querying, and citation surfacing on top of §9's
generic RAG layer (rag.py's RagRegistry/VectorStoreAdapter/build_context).

Four real, additive pieces, same "reuse LinearBanditCore for a new
per-call decision" pattern as llm/optimization_router.py (generation
parameters) and llm/token_optimizer.py (tool-schema size):

1. **Best (source, algorithm) selection** — when more than one
   VectorStoreAdapter is registered (rag_registry.list_names()) AND more
   than one real retrieval algorithm exists (_ALGORITHMS below), which
   COMBINATION to use for THIS request is a decision worth learning, not
   a hardcoded choice — arms are namespaced "{store}::{algorithm}",
   exactly optimization_router.py's "{provider}::{arm_id}" convention, so
   one bandit core learns every combination's own posterior
   independently. A store/algorithm pair that comes back empty or
   off-topic for a query family should stop getting picked for that
   family over time.
2. **Real, distinct retrieval algorithms** (_ALGORITHMS) — not variations
   on a theme:
     "vector"         — pure embedding-similarity ranking, as the store
                         returns it (rag.py's ChromaVectorStoreAdapter/
                         InMemoryVectorStoreAdapter's own cosine ranking).
     "lexical_rerank" — casts a wider net (k*2 candidates), then reranks
                         by REAL keyword/lexical overlap with the query
                         (Jaccard over word sets) — catches a topically-
                         related-but-keyword-poor chunk's opposite
                         failure mode: a keyword-exact chunk vector
                         similarity alone ranked too low.
     "hybrid"          — widest net (k*3), final score is a weighted
                         blend of normalized vector similarity AND
                         lexical overlap — the two independent signals
                         combined instead of either alone.
   (rag.py's own `rerank()` is a separate, narrower function for its one
   REST caller, routers/v1/rag.py — it now shares this module's real
   `lexical_overlap()`/`words()` helpers, imported from rag.py rather
   than duplicated, so both call sites use ONE lexical-overlap
   implementation instead of two independently-maintained ones.)
3. **Intent-aware retrieval** — the raw user message is rarely the best
   search query (greetings, filler, "please tell me about..."
   wrapping); refine_query_for_intent() strips that down to the
   keyword-dense core before it ever reaches a vector store, and
   is_rag_relevant() decides whether retrieval is even worth attempting
   for this message at all (no point querying a knowledge base for
   "delete task 5").
4. **Citation surfacing** — retrieve_with_optimizer() returns not just a
   context string but exactly which source chunks made it into that
   context (via rag.py's build_context_with_included_hits(), the one
   real source of truth for "used" vs "considered but truncated") — so
   a caller can literally tell the user which sources were used,
   instead of answering as if retrieval were invisible.

Reward is deferred, same shape as token_optimizer.py: record_outcome()
is meant to be called once the real task outcome (retried/out_of_scope,
or any other completion signal a caller already computes) is known —
this module has no opinion on WHEN that is, only on what to do with it.

Guardrail note (consistent with token_optimizer.py's hierarchy): this
optimizer only ever changes WHICH knowledge source/algorithm retrieves
supporting context — it never gates, approves, or substitutes for a
tool call. input_guard/output_guard/requires_confirmation run
identically regardless of which RAG combination (or none) was used;
retrieved context is additional information handed to the model, not a
bypass of anything the model still has to be validated against.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from . import bandit_persistence
from .bandit import LinearBanditCore
from .rag import (RagHit, build_context_with_included_hits, rag_registry,
                  lexical_overlap as _lexical_overlap, words as _words)
from ..db import store as db_store

# Aliased to db_store (not `store`) — this module already has a local
# variable named `store` (a VectorStoreAdapter instance, inside
# retrieve_with_optimizer()) that would otherwise silently shadow the
# app/db.py document-store factory.
_PERSIST_NAME = "rag_optimizer"
_COUNTS_TABLE = "RagOptimizerCounts"
_QUERY_STATS_TABLE = "RagQueryStats"

_FEATURE_DIM = 2  # [query_len_words/30 clamped, num_candidate_arms/10 clamped]

_ALGORITHMS = ("vector", "lexical_rerank", "hybrid")


def _apply_algorithm(algorithm: str, query: str, hits: list[RagHit], k: int) -> list[RagHit]:
    """hits is already sorted by the store's own vector-similarity score
    (both bundled adapters return search results pre-sorted) — each
    algorithm decides how (or whether) to re-rank that before truncating
    to k. Deliberately reads `hits`, never re-queries the store — the
    caller already fetched a wide-enough candidate pool up front."""
    if algorithm == "vector":
        return hits[:k]
    query_words = _words(query)
    if algorithm == "lexical_rerank":
        reranked = sorted(hits, key=lambda h: _lexical_overlap(query_words, h.text), reverse=True)
        return reranked[:k]
    if algorithm == "hybrid":
        max_score = max((h.score for h in hits), default=1.0) or 1.0
        blended = sorted(
            hits,
            key=lambda h: 0.5 * (h.score / max_score) + 0.5 * _lexical_overlap(query_words, h.text),
            reverse=True,
        )
        return blended[:k]
    raise ValueError(f"unknown retrieval algorithm {algorithm!r}")

_FILLER_WORDS = frozenset({
    "please", "could", "you", "tell", "me", "about", "the", "a", "an", "i", "want", "to",
    "know", "can", "would", "like", "hey", "hi", "hello", "so", "um", "just", "it", "is",
})

_QUESTION_MARKERS = re.compile(
    r"^(what|who|when|where|why|how|which|does|do|is|are|can|could|show|tell|find|list|"
    r"explain|summarize|describe)\b", re.I)


def refine_query_for_intent(message: str) -> str:
    """Cheap, LLM-free query refinement — strips conversational filler so
    the retriever sees the keyword-dense core of what's actually being
    asked, not the whole sentence verbatim. A real query-rewriting LLM
    call is the obvious upgrade path, deliberately avoided here for the
    same reason token_optimizer.py avoids one for tool selection: an
    extra model round trip would defeat the point of a FAST layer.
    Falls back to the original message if refining would leave nothing
    (e.g. an all-filler message) — an empty query is never useful."""
    words = re.findall(r"[a-zA-Z0-9]+", message.lower())
    kept = [w for w in words if w not in _FILLER_WORDS]
    return " ".join(kept) if kept else message


def is_rag_relevant(message: str) -> bool:
    """Whether this message looks like it's actually asking for
    information a knowledge base might hold, vs a command/statement
    retrieval has nothing to add to (e.g. "delete task 5", "mark T2
    done"). Cheap heuristic — question-shaped phrasing or a literal '?'
    — good enough to avoid a wasted retrieval call on every single turn;
    not a claim of real intent-classification precision. False negatives
    just mean "no retrieval attempted," never a wrong answer."""
    text = message.strip()
    return text.endswith("?") or bool(_QUESTION_MARKERS.match(text))


@dataclass
class RagSelection:
    """What select() picked, carried forward to record_outcome() once
    the real task outcome is known — same deferred-finalize shape as
    llm/token_optimizer.py's Selection."""

    store_name: str
    algorithm: str
    context: list[float]

    @property
    def arm_id(self) -> str:
        return f"{self.store_name}::{self.algorithm}"


class RagOptimizer:
    """One LinearBanditCore, arms = "{store}::{algorithm}" — every real
    combination of knowledge source and retrieval algorithm gets its own
    independent posterior, exactly optimization_router.py's per-provider
    namespacing applied to a different pair of axes."""

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.4,
                algorithms: tuple[str, ...] = _ALGORITHMS):
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._algorithms = algorithms
        self._lock = threading.Lock()
        # Real learning signal has been flowing into this bandit since
        # day one (native_agent.py/langgraph_agent.py/adk_agent.py/
        # document_pipeline.py/ground_truth.py all already call
        # record_outcome()) — it just evaporated on every restart until
        # now. bandit_persistence.py's own docstring names this module
        # as an intended (not-yet-done) adopter of its load/save pair;
        # this is that adoption, unchanged elsewhere.
        bandit_persistence.load(_PERSIST_NAME, self._core)
        # LinearBanditCore only keeps the Bayesian sufficient statistics
        # (B/f) needed for posterior_mean/select — no raw call/hit/
        # success counts, which a human reading a dashboard actually
        # wants ("how many queries has this combination even seen").
        # Small, additive, persisted the same way (app/db.py's store()),
        # never touching the bandit math itself.
        self._counts: dict[str, dict[str, int]] = self._load_counts()

    @staticmethod
    def _load_counts() -> dict[str, dict[str, int]]:
        row = db_store(_COUNTS_TABLE).get("RAG_OPTIMIZER", "COUNTS")
        return (row or {}).get("arms", {})

    def _save_counts(self) -> None:
        db_store(_COUNTS_TABLE).put("RAG_OPTIMIZER", "COUNTS", {"arms": self._counts})

    def select(self, query: str, store_names: list[str]) -> RagSelection:
        if not store_names:
            raise ValueError("select() needs at least one candidate store name")
        arm_ids = [f"{s}::{a}" for s in store_names for a in self._algorithms]
        context = [min(len(query.split()) / 30.0, 1.0), min(len(arm_ids) / 10.0, 1.0)]
        if len(arm_ids) == 1:
            winner = arm_ids[0]
        else:
            with self._lock:
                winner, _ = self._core.select(arm_ids, context)
        store_name, algorithm = winner.split("::", 1)
        return RagSelection(store_name, algorithm, context)

    def record_outcome(self, selection: RagSelection, *, success: bool, n_hits: int) -> None:
        """A (store, algorithm) pair that returned nothing is a bad pick
        regardless of whatever else happened this turn — that penalty
        dominates. Given it DID return hits, the eventual task outcome
        (success, from the caller's own retried/out_of_scope-equivalent
        signal) is what actually matters."""
        if n_hits == 0:
            reward = -0.5
        else:
            reward = 1.0 if success else -0.2
        with self._lock:
            self._core.update(selection.arm_id, selection.context, max(-1.0, min(1.0, reward)))
            counts = self._counts.setdefault(selection.arm_id, {"calls": 0, "empty_hits": 0, "successes": 0})
            counts["calls"] += 1
            if n_hits == 0:
                counts["empty_hits"] += 1
            elif success:
                counts["successes"] += 1
            self._save_counts()
        bandit_persistence.save(_PERSIST_NAME, self._core)

    def stats(self) -> list[dict]:
        """Current per-(source, algorithm) performance — the RAG
        dashboard's real data source. posterior_mean_weights mirrors
        routers/v1/metrics.py's own agents field shape (a raw weight
        vector, not collapsed to one number) for the same reason: this
        is a transparent linear model, and a dashboard can chart
        whichever component it wants rather than this module deciding
        for it."""
        with self._lock:
            arm_ids = set(self._counts) | set(self._core._B)
            rows = []
            for arm_id in sorted(arm_ids):
                store_name, algorithm = arm_id.split("::", 1)
                counts = self._counts.get(arm_id, {"calls": 0, "empty_hits": 0, "successes": 0})
                mean = self._core.posterior_mean(arm_id)
                rows.append({
                    "store": store_name,
                    "algorithm": algorithm,
                    "calls": counts["calls"],
                    "empty_hits": counts["empty_hits"],
                    "successes": counts["successes"],
                    "hit_rate": round(1 - counts["empty_hits"] / counts["calls"], 4) if counts["calls"] else None,
                    "success_rate": round(counts["successes"] / counts["calls"], 4) if counts["calls"] else None,
                    "posterior_mean_weights": [round(float(x), 4) for x in mean] if mean is not None else None,
                })
            return rows


# Process-wide singleton, same lifecycle as optimization_router/token_optimizer.
rag_optimizer = RagOptimizer()


@dataclass
class RetrievalResult:
    context: str
    # [{"index": int, "score": float, "metadata": dict}] — exactly the
    # chunks that made it into `context` (see
    # rag.build_context_with_included_hits), not merely "considered."
    sources_used: list[dict] = field(default_factory=list)
    store_name: str = ""
    algorithm: str = ""
    selection: RagSelection | None = None
    n_hits: int = 0


def retrieve_with_optimizer(query: str, *, store_names: list[str] | None = None, k: int = 5,
                            filters: dict | None = None, max_context_chars: int = 4000,
                            refine_query: bool = True) -> RetrievalResult:
    """The real, callable entry point: picks the best-scoring (source,
    algorithm) combination for THIS query (RL, see RagOptimizer above),
    retrieves + applies that algorithm + builds a context string, and
    reports exactly which sources actually made it in. Callers should
    hold onto `.selection` and call rag_optimizer.record_outcome(...)
    once the real turn outcome is known (mirrors llm/token_optimizer.py's
    deferred-finalize pattern — see that module for why immediate
    "it didn't error" isn't the right success signal to record against)."""
    names = store_names or rag_registry.list_names()
    refined_query = refine_query_for_intent(query) if refine_query else query
    selection = rag_optimizer.select(refined_query, names)
    store = rag_registry.get(selection.store_name)

    # Wider net than k so lexical_rerank/hybrid have real candidates to
    # re-order rather than just re-sorting an already-truncated top-k.
    candidates = store.search(refined_query, k=k * 3, filters=filters)
    hits = _apply_algorithm(selection.algorithm, refined_query, candidates, k)
    context_text, included = build_context_with_included_hits(hits, max_chars=max_context_chars)
    sources_used = [{"index": i + 1, "score": h.score, "metadata": h.metadata}
                    for i, h in enumerate(included)]
    _record_query_stats(has_sources=bool(sources_used), n_sources=len(sources_used))
    return RetrievalResult(context=context_text, sources_used=sources_used,
                           store_name=selection.store_name, algorithm=selection.algorithm,
                           selection=selection, n_hits=len(hits))


# A real, honest proxy for "did RAG actually contribute to this answer" —
# every real chat turn across all three agent frameworks (native/
# langgraph/adk) plus document_pipeline/ground_truth calls THIS function
# (confirmed: it's the one retrieval entry point they all share), so
# counting here — not at the raw /api/v1/rag/query dev/test REST
# endpoint, which never goes through the optimizer at all — is what
# actually reflects real usage.
_query_stats_lock = threading.Lock()


def _record_query_stats(*, has_sources: bool, n_sources: int) -> None:
    with _query_stats_lock:
        row = db_store(_QUERY_STATS_TABLE).get("RAG_QUERIES", "STATS") or {
            "total_queries": 0, "queries_with_sources": 0, "total_sources": 0,
        }
        row["total_queries"] += 1
        if has_sources:
            row["queries_with_sources"] += 1
        row["total_sources"] += n_sources
        db_store(_QUERY_STATS_TABLE).put("RAG_QUERIES", "STATS", row)


def query_stats() -> dict:
    row = db_store(_QUERY_STATS_TABLE).get("RAG_QUERIES", "STATS") or {
        "total_queries": 0, "queries_with_sources": 0, "total_sources": 0,
    }
    total = row["total_queries"]
    return {
        "total_queries": total,
        "queries_with_sources": row["queries_with_sources"],
        "coverage_rate": round(row["queries_with_sources"] / total, 4) if total else None,
        "avg_sources_per_query": round(row["total_sources"] / total, 4) if total else None,
    }


def format_sources_note(sources_used: list[dict]) -> str:
    """A short, human-readable line an agent's reply can append — the
    "mention which ones used" part. Falls back to nothing (empty string)
    when there's nothing to cite, so callers can always safely do
    `reply + format_sources_note(result.sources_used)` without an `if`."""
    if not sources_used:
        return ""
    labels = []
    for s in sources_used:
        meta = s.get("metadata") or {}
        label = meta.get("title") or meta.get("doc_id") or meta.get("id") or f"source {s['index']}"
        labels.append(str(label))
    return f"\n\n(Used: {', '.join(labels)})"
