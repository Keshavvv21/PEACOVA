"""§9 — the generic RAG abstraction layer:

    Document Loader → Chunking → Embedding → Vector DB → Retriever
    → Reranker → Context Builder → Agent/LLM

`VectorStoreAdapter` is the one interface every backend implements
(structural Protocol, same LSP/ISP shape llm/providers.py's LLMProvider
already uses in this codebase) — no vector database is hard-coded.
Ships with two real, independent adapters proving that:

  ChromaVectorStoreAdapter — wraps the EXISTING PM-domain vector_index
      (retrieval/vectorstore.py, the real chromadb-backed store the
      `semantic_search` tool already uses) so ALGO AKANADE's generic RAG API
      and the PM chatbot's own tool share ONE index, not two competing
      ones.
  InMemoryVectorStoreAdapter — a second, genuinely independent adapter
      (pure Python, cosine similarity, the SAME HashEmbedder the PM
      domain falls back to when sentence-transformers isn't installed)
      for any non-PM document collection an application registers at
      runtime — proof the abstraction isn't secretly Chroma-only.

Adding FAISS/Qdrant/Weaviate/Pinecone/Elasticsearch/pgvector is one more
class satisfying VectorStoreAdapter — nothing here or in
routers/gmaofn_rag.py needs to change.
"""
from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..retrieval.vectorstore import HashEmbedder, vector_index


@dataclass
class RagDocument:
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RagHit:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class VectorStoreAdapter(Protocol):
    name: str

    def add_documents(self, documents: list[RagDocument]) -> None: ...

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[RagHit]: ...


_DEFAULT_RAG_USER = {"sub": "gmaofn", "role": "Admin"}


@dataclass
class ChromaVectorStoreAdapter:
    """Read-through wrapper around the PM domain's own `vector_index` —
    documents there are populated by the PM service layer's own
    on_mutation hooks (task/note creation), not through this adapter's
    add_documents (there is deliberately no generic "index arbitrary
    text into the PM index" path — that store's RBAC filtering assumes
    every entry IS a real task/note with a real owning project)."""

    name: str = "chroma-pm-index"

    def add_documents(self, documents: list[RagDocument]) -> None:
        raise NotImplementedError(
            "The PM domain's Chroma index is populated by pm_service's own "
            "mutation hooks (real tasks/notes), not generic documents — use "
            "InMemoryVectorStoreAdapter (or your own VectorStoreAdapter) for "
            "an application-managed document collection instead.")

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[RagHit]:
        user = (filters or {}).get("user") or _DEFAULT_RAG_USER
        hits = vector_index.search(query, user, k)
        return [RagHit(text=h["text"], score=h["score"], metadata=h.get("meta", {})) for h in hits]


# Below this many vectors, IVF clustering's own overhead (building it,
# scanning every centroid) costs more than it saves — brute force stays
# faster AND exact. See vector_storage_optimizer.py's module docstring
# for the O(sqrt(n)) complexity analysis this threshold is based on.
IVF_MIN_VECTORS = 50


@dataclass
class InMemoryVectorStoreAdapter:
    """A real, independent second backend — pure-Python cosine similarity
    over HashEmbedder vectors (the exact fallback embedder
    retrieval/vectorstore.py itself uses when sentence-transformers isn't
    installed), so this needs zero extra dependencies and genuinely
    supports runtime `add_documents` for any application's own document
    collection — the "knowledge_sources" an agent declares (see
    agent_base.AgentMetadata.rag_enabled).

    Once the corpus crosses IVF_MIN_VECTORS, search() automatically
    builds and uses an IVF index (vector_storage_optimizer.py) instead
    of a brute-force scan — same "optimize once, scaled to size" idea as
    document_pipeline.py's retrieval-breadth scaling. Below that
    threshold, brute force stays exact and is already fast enough that
    clustering overhead wouldn't pay for itself."""

    name: str = "in-memory-cosine"
    _embedder: HashEmbedder = field(default_factory=HashEmbedder)
    _docs: list[RagDocument] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ivf_index: object = field(default=None, repr=False, compare=False)
    _ivf_stale: bool = field(default=True, repr=False, compare=False)

    def add_documents(self, documents: list[RagDocument]) -> None:
        with self._lock:
            for doc in documents:
                self._docs.append(doc)
                self._vectors.append(self._embedder.embed(doc.text))
            self._ivf_stale = True  # rebuild lazily on the next search()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))  # both already unit-normalized by HashEmbedder

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[RagHit]:
        with self._lock:
            docs, vectors = list(self._docs), list(self._vectors)
            if not docs:
                return []
            q = self._embedder.embed(query)
            if len(docs) >= IVF_MIN_VECTORS:
                from .vector_storage_optimizer import IVFIndex
                if self._ivf_stale or self._ivf_index is None:
                    self._ivf_index = IVFIndex(vectors)
                    self._ivf_stale = False
                hits = self._ivf_index.search(q, k, n_probe=max(1, self._ivf_index.n_clusters // 4))
                return [RagHit(text=docs[i].text, score=round(score, 4), metadata=docs[i].metadata)
                       for i, score in hits]
        scored = sorted(((self._cosine(q, v), d) for v, d in zip(vectors, docs)),
                        key=lambda t: t[0], reverse=True)
        return [RagHit(text=d.text, score=round(score, 4), metadata=d.metadata)
               for score, d in scored[:k]]


# ── chunking (the pipeline's first real transform before embedding) ─────

def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """A simple, dependency-free sliding-window chunker over sentences —
    good enough for the bundled adapters above; swap for a
    token-aware/semantic chunker in a real deployment without touching
    anything downstream (chunks are just strings either way)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > chunk_size and current:
            chunks.append(current.strip())
            # A bare character slice can land mid-word (e.g. "...or
            # request a new one" instead of the intended "...to reset
            # or request a new one"), corrupting both this chunk's
            # embedding (a garbage leading token) and what the LLM
            # reads as its opening words. Snap to the next word
            # boundary instead — drop any partial leading fragment.
            tail = current[-overlap:]
            if " " in tail:
                tail = tail.split(" ", 1)[1]
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


# ── reranker + context builder ───────────────────────────────────────────

def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_overlap(query_words: set[str], text: str) -> float:
    """Jaccard overlap between a query's words and a chunk's words — the
    real lexical signal rerank() below blends in, and the same function
    rag_optimizer.py's "lexical_rerank"/"hybrid" algorithms use (imported
    from here, not duplicated) — one implementation, every caller."""
    text_words = words(text)
    if not query_words or not text_words:
        return 0.0
    return len(query_words & text_words) / len(query_words | text_words)


def rerank(hits: list[RagHit], *, query: str | None = None,
          top_n: int | None = None) -> list[RagHit]:
    """The pipeline's reranker stage. With `query` given: a real (if
    simple) second pass — blends the vector store's own normalized score
    with lexical (Jaccard) overlap against the query, since a pure
    embedding-similarity ranking can float a topically-related but
    keyword-poor chunk above a more directly relevant one, or vice versa.
    Without `query` (backward-compatible default): a plain re-sort by
    existing score — hits typically arrive pre-sorted already, so this
    is a no-op in practice, which is why every caller that wants real
    reranking should pass `query`. Swap in a real cross-encoder reranker
    here without touching build_context/search."""
    if query:
        query_words = words(query)
        max_score = max((h.score for h in hits), default=0.0) or 1.0
        ranked = sorted(
            hits,
            key=lambda h: 0.5 * (h.score / max_score) + 0.5 * lexical_overlap(query_words, h.text),
            reverse=True)
    else:
        ranked = sorted(hits, key=lambda h: h.score, reverse=True)
    return ranked[:top_n] if top_n else ranked


def build_context_with_included_hits(hits: list[RagHit], *,
                                     max_chars: int = 4000) -> tuple[str, list[RagHit]]:
    """Same pipeline stage as build_context() below, but also returns
    exactly WHICH hits made it into the final string — build_context()
    truncates by char budget, so a hit near the end of a long list may
    never actually reach the agent/LLM. rag_optimizer.py's citation
    surfacing ("which sources were actually used") needs that distinction;
    a naive re-derivation via string search on the output would be
    fragile, so this is the one real source of truth both build_context()
    and the optimizer read from."""
    parts: list[str] = []
    included: list[RagHit] = []
    used = 0
    for i, hit in enumerate(hits):
        block = f"[source {i + 1}, score={hit.score}] {hit.text}"
        if used + len(block) > max_chars:
            # Skip THIS hit, don't stop entirely — hits are score-sorted,
            # so an oversized higher-scored chunk used to zero out every
            # smaller, still-relevant hit behind it (break instead of
            # continue): the single worst case was the top hit alone
            # exceeding max_chars and the whole turn getting NO context
            # at all, even with perfectly good smaller hits available.
            continue
        parts.append(block)
        included.append(hit)
        used += len(block)
    return "\n\n".join(parts), included


def build_context(hits: list[RagHit], *, max_chars: int = 4000) -> str:
    """The pipeline's final stage before the agent/LLM ever sees anything
    — plain concatenation with a source label per chunk, truncated to a
    caller-chosen budget rather than an arbitrary hard cap."""
    return build_context_with_included_hits(hits, max_chars=max_chars)[0]


class RagRegistry:
    """Named VectorStoreAdapters an application/agent can address by
    name (an agent's `knowledge_sources`, §9) — the registry
    routers/gmaofn_rag.py's REST endpoints and any RAG-enabled agent
    both read from."""

    def __init__(self) -> None:
        self._stores: dict[str, VectorStoreAdapter] = {
            "pm-index": ChromaVectorStoreAdapter(),
            "default": InMemoryVectorStoreAdapter(),
        }

    def register(self, name: str, store: VectorStoreAdapter) -> None:
        self._stores[name] = store

    def get(self, name: str) -> VectorStoreAdapter:
        if name not in self._stores:
            raise KeyError(f"No RAG store registered as {name!r}")
        return self._stores[name]

    def list_names(self) -> list[str]:
        return list(self._stores.keys())


rag_registry = RagRegistry()
