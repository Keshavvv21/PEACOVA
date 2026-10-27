"""§9 — the RAG abstraction layer's REST surface: index/search/query
against any named VectorStoreAdapter (gmaofn/rag.py's RagRegistry) —
default ships with "pm-index" (the PM domain's real Chroma-backed store)
and "default" (a runtime-writable in-memory store); register more via
`rag_registry.register(name, MyVectorStoreAdapter())`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...auth import current_user, require_role
from ...framework.document_pipeline import (UnsafePathError, ingest_path, summarize_document,
                                         suggest_prompts)
from ...framework.ground_truth import create_ground_truth
from ...framework.rag import RagDocument, build_context, chunk_text, rag_registry, rerank
from ...framework.rag_optimizer import query_stats, rag_optimizer

router = APIRouter(prefix="/api/v1/rag", tags=["gmaofn-rag"])


class IndexRequest(BaseModel):
    store: str = "default"
    text: str
    metadata: dict = Field(default_factory=dict)
    chunk_size: int = 800


@router.post("/index")
def index_document(body: IndexRequest, _: dict = Depends(current_user)):
    try:
        store = rag_registry.get(body.store)
    except KeyError as e:
        raise HTTPException(404, str(e))
    chunks = chunk_text(body.text, chunk_size=body.chunk_size)
    docs = [RagDocument(text=c, metadata=body.metadata) for c in chunks]
    store.add_documents(docs)
    return {"store": body.store, "chunks_indexed": len(docs)}


class SearchRequest(BaseModel):
    store: str = "default"
    query: str
    k: int = 5


@router.post("/search")
def search(body: SearchRequest, user: dict = Depends(current_user)):
    try:
        store = rag_registry.get(body.store)
    except KeyError as e:
        raise HTTPException(404, str(e))
    hits = store.search(body.query, k=body.k, filters={"user": user})
    return {"hits": [{"text": h.text, "score": h.score, "metadata": h.metadata} for h in hits]}


class QueryRequest(BaseModel):
    store: str = "default"
    query: str
    k: int = 5
    max_context_chars: int = 4000


@router.post("/query")
def query(body: QueryRequest, user: dict = Depends(current_user)):
    """search + rerank + build_context in one call — the full pipeline's
    last three stages, returning exactly what an agent/LLM prompt would
    be given as retrieved context."""
    try:
        store = rag_registry.get(body.store)
    except KeyError as e:
        raise HTTPException(404, str(e))
    hits = store.search(body.query, k=body.k, filters={"user": user})
    ranked = rerank(hits, query=body.query)
    return {"context": build_context(ranked, max_chars=body.max_context_chars),
           "sources": [{"text": h.text, "score": h.score, "metadata": h.metadata} for h in ranked]}


class IngestPathRequest(BaseModel):
    # A path RELATIVE to settings().RAG_INGEST_DIR — never an absolute
    # server path; document_pipeline.ingest_path() enforces this again
    # itself (belt and suspenders), see its UnsafePathError.
    path: str
    store_name: str | None = None
    chunk_size: int = 800


@router.post("/ingest-path")
def ingest_path_endpoint(body: IngestPathRequest, _: dict = Depends(require_role("Admin"))):
    """Folder/document ingestion (gmaofn/document_pipeline.py) — Admin-only
    since it reads server-side files, even though they're confined to
    RAG_INGEST_DIR. The resulting store is immediately usable by
    /summarize, /suggest-prompts below, this router's own
    /search|/query, AND the live PM chat (rag_optimizer.py checks every
    registered store). recommended_k/recommended_max_context_chars are
    what /summarize and /suggest-prompts will use by default for this
    store — scaled to how much was actually ingested (a 3-chunk note
    gets a narrower, cheaper default than a 500-chunk corpus)."""
    try:
        result = ingest_path(body.path, store_name=body.store_name, chunk_size=body.chunk_size)
    except UnsafePathError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"store": result.store_name, "files_indexed": result.files_indexed,
           "chunks_indexed": result.chunks_indexed, "skipped_files": result.skipped_files,
           "recommended_k": result.recommended_k,
           "recommended_max_context_chars": result.recommended_max_context_chars}


class SummarizeRequest(BaseModel):
    store: str
    query: str = "summarize the main points"
    # None (the default) auto-scales to the store's actual size — see
    # document_pipeline._recommended_retrieval_params(); pass a real
    # value to override.
    k: int | None = None
    max_context_chars: int | None = None


@router.post("/summarize")
def summarize(body: SummarizeRequest, _: dict = Depends(current_user)):
    """RL-picked retrieval (rag_optimizer.py, scoped to `store`) + a real
    LLM call (the "bulk" tier) for a grounded summary — see
    document_pipeline.summarize_document()."""
    if body.store not in rag_registry.list_names():
        raise HTTPException(404, f"No RAG store registered as {body.store!r}")
    return summarize_document(body.store, query=body.query, k=body.k,
                              max_context_chars=body.max_context_chars)


class SuggestPromptsRequest(BaseModel):
    store: str
    query: str = "what is this document about"
    n: int = 3
    k: int | None = None
    max_context_chars: int | None = None


@router.post("/suggest-prompts")
def suggest_prompts_endpoint(body: SuggestPromptsRequest, _: dict = Depends(current_user)):
    """Same retrieval as /summarize, a different ask — concrete follow-up
    questions grounded in the ingested content."""
    if body.store not in rag_registry.list_names():
        raise HTTPException(404, f"No RAG store registered as {body.store!r}")
    return suggest_prompts(body.store, query=body.query, n=body.n, k=body.k,
                           max_context_chars=body.max_context_chars)


class GroundTruthRequest(BaseModel):
    prompts: list[str]
    store: str | None = None  # None = answer from general knowledge, ungrounded
    k: int = 6
    max_context_chars: int = 4000
    log_for_training: bool = True


@router.post("/ground-truth")
def ground_truth_endpoint(body: GroundTruthRequest, _: dict = Depends(require_role("Admin"))):
    """The enterprise pipeline's "Create ground truth" stage —
    Admin-only since it triggers a real LLM call per prompt (real cost)
    and, by default, writes to the fine-tune corpus. Every result is
    returned for review regardless of success; only calls that actually
    succeeded get logged."""
    if body.store is not None and body.store not in rag_registry.list_names():
        raise HTTPException(404, f"No RAG store registered as {body.store!r}")
    if not body.prompts:
        raise HTTPException(400, "prompts must be non-empty")
    results = create_ground_truth(body.prompts, store_name=body.store, k=body.k,
                                  max_context_chars=body.max_context_chars,
                                  log_for_training=body.log_for_training)
    return {"results": [{"prompt": r.prompt, "answer": r.answer, "provider": r.provider,
                        "grounded": r.grounded, "sources_used": r.sources_used, "ok": r.ok}
                       for r in results],
           "logged_for_training": sum(1 for r in results if r.ok) if body.log_for_training else 0}


@router.get("/metrics")
def rag_metrics(_: dict = Depends(require_role("Admin"))):
    """How RAG is actually performing, for the RAG dashboard — per-
    (source, algorithm) reward/hit/success stats from RagOptimizer (real
    signal: every real chat turn across all three agent frameworks plus
    document_pipeline/ground_truth already calls record_outcome()), plus
    a query-coverage rollup (how often retrieval actually surfaced
    something) as the honest, measurable proxy for "how RAG improves
    responses" — there's no attempt here to score whether a specific
    answer got BETTER because of retrieval, which would need real
    before/after evaluation this module doesn't do."""
    return {"arms": rag_optimizer.stats(), "query_coverage": query_stats()}
