"""Intent-answer cache — sits in front of the PM/Trading chat pipeline
(app/routers/chat.py) and the Coding Agent (agents/coding_agent.py),
returning a previously-generated FINAL ANSWER directly on a hit, skipping
the SLM/LLM call entirely. This is deliberately a different thing from
query_refinement.py's RefinementMemory, which only caches a rewritten
query string (still pays for the real answer-generating call afterwards)
— here, a hit means zero model calls for the whole turn.

Reuses query_refinement.py's exact mechanism for "is this message close
enough to one we've seen": the same zero-cost HashEmbedder
(retrieval/vectorstore.py) plus vector_storage_optimizer.py's scalar
quantization, bounded deque + lock. The one deliberate difference is the
similarity threshold: RefinementMemory reuses a rewrite it computed
itself (worst case, a slightly-off rewrite), while a miss here would
silently hand the user an outright wrong ANSWER — so this cache matches
closer (0.96 vs RefinementMemory's 0.93) before trusting a hit.

Safety is enforced by CALLERS, not by this module: this cache has no
notion of "is this intent safe to reuse" (freshness, tool calls,
pending confirmations) — chat.py only ever calls it for messages
is_pure_conversational() already judged as pure small talk, and
coding_agent.py only for generate/general_coding turns that never
touched a tool. Two separate module-level instances (conversational vs
coding) keep those namespaces from ever mixing.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from ..config import settings
from ..retrieval.vectorstore import HashEmbedder
from .vector_storage_optimizer import dequantize_vector, quantize_vector

_embedder = HashEmbedder()

_PUNCT_TAIL = re.compile(r"[\s.!?,]+$")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Exact-match key. HashEmbedder filters stopwords, so a message made
    ENTIRELY of them ("are you up?", "how are you") embeds to a zero
    vector and can never match by cosine similarity — which is exactly
    the pure-conversational class the chat side caches. Normalized
    string equality is both the correct and the zero-false-hit mechanism
    for that closed set of fixed phrases; similarity below is what
    generalizes to near-duplicate content-bearing prompts."""
    return _PUNCT_TAIL.sub("", _WHITESPACE.sub(" ", text.strip().lower()))


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # HashEmbedder output is already unit-normalized


@dataclass
class _Entry:
    text: str
    key: str
    answer: str
    provider: str | None
    created_at: float
    ttl_s: float
    q_bytes: bytes
    q_min: float
    q_scale: float


@dataclass
class CachedAnswer:
    text: str
    provider: str | None


class IntentCache:
    """Bounded, in-memory, process-lifetime cache — same persistence
    story as RefinementMemory (query_refinement.py): lost on restart,
    no new disk/Redis layer added for this. TTL bounds staleness on top
    of the caller-side intent gating, which is the primary safety net."""

    def __init__(self, maxlen: int, ttl_s: float) -> None:
        self._entries: deque[_Entry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._ttl_s = ttl_s

    def lookup(self, text: str) -> CachedAnswer | None:
        threshold = settings().INTENT_CACHE_SIMILARITY_THRESHOLD
        key = _normalize(text)
        q = _embedder.embed(text)
        degenerate = not any(q)  # all-stopword message: similarity is meaningless, key match only
        now = time.time()
        with self._lock:
            entries = list(self._entries)
        best_sim, best_entry = 0.0, None
        for e in entries:
            if now - e.created_at > e.ttl_s:
                continue
            if e.key == key:
                return CachedAnswer(text=e.answer, provider=e.provider)
            if degenerate:
                continue
            sim = _cosine(q, dequantize_vector(e.q_bytes, e.q_min, e.q_scale))
            if sim > best_sim:
                best_sim, best_entry = sim, e
        if best_entry is None or best_sim < threshold:
            return None
        return CachedAnswer(text=best_entry.answer, provider=best_entry.provider)

    def remember(self, text: str, answer: str, provider: str | None) -> None:
        q_bytes, q_min, q_scale = quantize_vector(_embedder.embed(text))
        with self._lock:
            self._entries.append(_Entry(text=text, key=_normalize(text), answer=answer,
                                        provider=provider, created_at=time.time(),
                                        ttl_s=self._ttl_s,
                                        q_bytes=q_bytes, q_min=q_min, q_scale=q_scale))

    def __len__(self) -> int:
        return len(self._entries)


# Process-wide singletons, same lifecycle as query_refinement/hybrid_router.
# Separate namespaces: a cached greeting must never be considered for a
# coding prompt, and vice versa.
conversational_cache = IntentCache(
    maxlen=settings().INTENT_CACHE_MAX_ENTRIES,
    ttl_s=settings().CONVERSATIONAL_CACHE_TTL_SECONDS)
coding_cache = IntentCache(
    maxlen=settings().INTENT_CACHE_MAX_ENTRIES,
    ttl_s=settings().CODING_CACHE_TTL_SECONDS)
