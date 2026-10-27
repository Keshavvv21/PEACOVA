"""§5 Vector Store — semantic search over task/note text.

Store: Chroma (embedded, persisted to disk) exactly as specified, with a
pure-Python cosine store as the zero-dependency fallback so tests and
air-gapped runs never break.

Embedder: sentence-transformers all-MiniLM-L6-v2 when installed (the
design's choice — local, CPU, ~80MB); otherwise a deterministic hashed
bag-of-words embedder (384-dim, same interface) so the pipeline is
always real even before the model file lands on the box.

§5.2's RBAC caveat is implemented here as post_filter(): a vector index
has no concept of "who can see this", so results are filtered against
the caller's project visibility BEFORE anything reaches the LLM or user.
"""
from __future__ import annotations

import hashlib
import math
import re

from ..config import settings
from ..services import pm_service

DIM = 384

# A real, empirically-confirmed failure mode of the plain hashed
# bag-of-words scheme below: two topically UNRELATED queries/chunks that
# happen to share common function words ("how", "do", "my", "is", ...)
# scored as similar as genuinely related ones — e.g. "how do I reset my
# password" vs "how do I delete my account" shared enough stopword
# overlap to rank a password-reset chunk above every actually-relevant
# result for an account-deletion query. Filtering these out before
# hashing is a real, targeted fix for that specific failure mode (not a
# claim that this makes HashEmbedder a semantic embedder — it's still a
# bag-of-words scheme with no synonym understanding; see MiniLMEmbedder
# below for real semantic embeddings when that dependency is available).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "so", "of", "on", "in",
    "at", "to", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "doing", "done",
    "have", "has", "had", "having", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "i", "me", "my", "mine",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "ours", "they", "them", "their",
    "theirs", "this", "that", "these", "those", "what", "which", "who",
    "whom", "whose", "how", "when", "where", "why", "not", "no", "yes",
    "please", "just", "about", "into", "out", "up", "down", "over",
    "under", "again", "then", "than", "too", "very", "s", "t", "don",
})


# ── embedders ────────────────────────────────────────────────────────────

class HashEmbedder:
    """Deterministic hashed bag-of-words: no downloads, real similarity —
    with stopwords filtered out before hashing (see _STOPWORDS above),
    so two chunks sharing only common function words no longer score as
    falsely similar as two chunks that actually share topical content."""

    name = "hash-bow-384"

    def embed(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in _STOPWORDS:
                continue
            # code-security-agent flags this (PY-WEAK-HASH) — MD5 is used
            # here purely as a token -> bucket-index hash for a bag-of-words
            # vector, not for anything security-sensitive (no secret/password
            # ever passes through it); intentional, not a finding to fix.
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


class MiniLMEmbedder:
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def make_embedder():
    choice = settings().EMBEDDER
    if choice in ("auto", "minilm"):
        try:
            return MiniLMEmbedder()
        except Exception:
            if choice == "minilm":
                raise
    return HashEmbedder()


# ── stores ───────────────────────────────────────────────────────────────

class ChromaStore:
    # supply-chain-agent flagged 5 real HIGH-severity CVEs against the
    # installed chromadb==1.5.9 (GHSA-2wm9-hf6c-p5cr, GHSA-36p7-vc44-83pf,
    # GHSA-f4j7-r4q5-qw2c, GHSA-xph7-9rjv-w5fr, PYSEC-2026-311) — checked
    # against OSV.dev directly, confirmed still the latest PyPI release (no
    # fixed version to upgrade to yet). All five are in Chroma's networked
    # HTTP server API (multi-tenant auth/RBAC, `/api/v2/tenants/...`
    # endpoints) per the advisories, which PersistentClient below never
    # runs — there is no server process, no network listener, no tenant/auth
    # layer here at all. Re-check OSV.dev before ever switching this to
    # chromadb's HttpClient/`chroma run` server mode; that would put these
    # exact CVEs back in scope.
    def __init__(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=settings().CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            "algopm", metadata={"hnsw:space": "cosine"})

    def upsert(self, doc_id: str, vector: list[float], text: str, meta: dict):
        self.col.upsert(ids=[doc_id], embeddings=[vector],
                        documents=[text], metadatas=[meta])

    def query(self, vector: list[float], k: int,
             allowed_project_ids: set[str] | None = None) -> list[dict]:
        # RBAC enforced HERE, before retrieval — Chroma's own `where`
        # metadata filter (its ANN search itself only ever considers
        # matching rows) rather than the old "fetch extra, discard
        # unauthorized ones after scoring" approach VectorIndex.search()
        # used to rely on. `allowed_project_ids=None` means "no RBAC
        # scoping requested" (the caller — Executive's own unrestricted
        # view — already decided everything is visible); an EMPTY set is
        # never passed in here (VectorIndex.search() short-circuits to
        # [] before calling this at all — Chroma's `$in` with an empty
        # list isn't guaranteed to mean "match nothing" the way an empty
        # Python set naturally does).
        where = ({"project_id": {"$in": list(allowed_project_ids)}}
                if allowed_project_ids is not None else None)
        r = self.col.query(query_embeddings=[vector], n_results=k, where=where)
        out = []
        for i, doc_id in enumerate(r["ids"][0]):
            out.append({"id": doc_id, "text": r["documents"][0][i],
                        "meta": r["metadatas"][0][i],
                        "score": 1 - r["distances"][0][i]})
        return out


class MemoryStore:
    """Cosine over an in-process dict — the fallback and the test double."""

    def __init__(self):
        self.docs: dict[str, tuple[list[float], str, dict]] = {}

    def upsert(self, doc_id, vector, text, meta):
        self.docs[doc_id] = (vector, text, meta)

    def query(self, vector, k, allowed_project_ids: set[str] | None = None):
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))
        # Same pre-retrieval RBAC scoping as ChromaStore.query() above —
        # filter the candidate set BEFORE scoring, not after, so the
        # zero-dependency fallback store enforces the identical
        # visibility rule the real Chroma-backed store does.
        items = self.docs.items() if allowed_project_ids is None else (
            (i, doc) for i, doc in self.docs.items()
            if doc[2].get("project_id") in allowed_project_ids)
        scored = [{"id": i, "text": t, "meta": m, "score": cos(vector, v)}
                  for i, (v, t, m) in items]
        return sorted(scored, key=lambda d: -d["score"])[:k]


class VectorIndex:
    def __init__(self):
        self.embedder = make_embedder()
        try:
            self.store = ChromaStore()
            self.backend = "chroma"
        except Exception:
            self.store = MemoryStore()
            self.backend = "memory"
        pm_service.on_mutation(self._on_mutation)

    # §6.1 step 13: enqueue changed text for re-embedding (sync here;
    # async task queue is the production refinement)
    def _on_mutation(self, kind: str, entity: dict):
        if kind == "task" and not entity.get("deleted"):
            self.index_task(entity)

    def index_task(self, t: dict):
        text = f"{t['title']}. {t.get('description','')}"
        self.store.upsert(
            t["id"], self.embedder.embed(text), text,
            {"type": "task", "project_id": t["project_id"], "task_id": t["id"]})

    def reindex_all(self):
        for t in pm_service.list_tasks():
            self.index_task(t)

    def search(self, query: str, user: dict, k: int = 5) -> list[dict]:
        # RBAC is now enforced BEFORE retrieval (the store's own query(),
        # via `allowed_project_ids`) rather than by over-fetching k*3 and
        # discarding unauthorized hits afterward — a real, previously-
        # open gap: nothing stopped an unauthorized project's content
        # from being scored, ranked, and briefly held in memory before
        # being thrown away. post_filter() below still runs as a cheap
        # defense-in-depth check, not the primary enforcement anymore.
        if user["role"] == "Executive":
            hits = self.store.query(self.embedder.embed(query), k)
        else:
            visible = {p["id"] for p in pm_service.list_projects(user)}
            if not visible:
                return []
            hits = self.store.query(self.embedder.embed(query), k,
                                    allowed_project_ids=visible)
        return self.post_filter(hits, user)[:k]

    @staticmethod
    def post_filter(hits: list[dict], user: dict) -> list[dict]:
        """§5.2 RBAC — defense-in-depth double-check, kept as a safety net
        in case a store's own pre-retrieval filtering (search() above)
        ever has a bug or a new store implementation forgets to honor
        `allowed_project_ids`: only Executive (the top demo rank, mapped
        from the real app's Admin — see auth._REAL_ROLE_MAP) sees every
        project's results; every other role, Admin (mapped from Senior
        Manager) included, is scoped down to projects they're a member
        of (see pm_service.list_projects) — an Admin here does NOT mean
        org-wide reach, unlike the real app's Admin role of the same
        name."""
        if user["role"] == "Executive":
            return hits
        visible = {p["id"] for p in pm_service.list_projects(user)}
        return [h for h in hits if h["meta"].get("project_id") in visible]


vector_index = VectorIndex()
