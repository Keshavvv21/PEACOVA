"""Query refinement — a layer that sits ABOVE the rest of the chat
pipeline (called once per turn from routers/chat.py, before ANY agent
track sees the message): rewrite the user's raw text for grammar/
clarity while strictly preserving its meaning, constraints, and
ambiguity, catch cases where the rewrite actually drifted from what the
user said, and revert those — never let a "cleaned up" query silently
mean something different than what was typed.

Modeled on a 6-step catalog (Query Refiner -> Structured Converter ->
Validation Agent -> Repair Agent -> Confirmation -> Orchestrator), but
collapsed to what a generic (non-domain-specific) chat pipeline actually
needs, and made cheap enough to run on every live turn:

  1. Gate (LinearBanditCore, "refine" vs "skip"): most messages are
     already clear — paying an extra LLM round trip on EVERY turn would
     be a real latency/cost tax, especially painful on a local SLM tier.
     Cheap, non-LLM heuristics (length, filler phrases, missing
     punctuation) decide whether refinement is even worth attempting;
     same "no extra model round trip for the common case" philosophy as
     token_optimizer.py/rag_optimizer.py/hybrid_router.py.
  2. Memory lookup (RefinementMemory, checked BEFORE any LLM call): every
     accepted past refinement is remembered by its raw query's quantized
     embedding; a new query close enough to one already seen reuses that
     cached refinement directly. This is what makes the layer genuinely
     "refine as it goes" rather than a static prompt: its effective LLM
     round-trip rate on repeat/near-duplicate phrasing drops the longer
     the process runs, at zero extra cost for a hit.
  3. Refine + structure (ONE LLM call, on a memory miss): rewrite for
     clarity and extract a short structured summary + missing
     clarifications + suggested rephrasings, in a single response — the
     catalog's separate Query Refiner/Structured Converter prompts
     collapsed into one call, since a generic chat turn doesn't need two
     round trips to get both.
  4. Drift check (embedding similarity, NOT a second LLM call): cosine
     similarity between the raw and refined text via HashEmbedder — the
     exact zero-cost, dependency-free embedder rag.py already uses as
     its SLM-safe fallback. This is the real "optimize embeddings for
     SLMs" piece: catching meaning drift costs nothing extra to compute,
     regardless of which model tier is actually running.
  5. Repair: below the similarity threshold, the rewrite is judged too
     risky to trust silently — revert to the original raw text rather
     than call yet another LLM (there's nothing an LLM call would add
     here that reverting doesn't already give you for free). An
     accepted (non-repaired) refinement is remembered (step 2's
     RefinementMemory); a repaired one never is, so a bad rewrite can't
     poison future lookups.
  6. Confirmation packaging: nothing here ever BLOCKS the turn (a chat
     API has no synchronous confirm/edit round trip mid-message) —
     instead `needs_confirmation` + `missing_clarifications` +
     `suggested_queries` are surfaced to the UI as a chip, so a user can
     see and act on ambiguity without every turn stopping to ask first.

Two independent RL/embedding mechanisms compound here: the LinearBanditCore
gate (step 1) learns whether refining is worth it AT ALL for a given kind
of message, while RefinementMemory (step 2) learns WHICH specific queries
already have a known-good refinement — one improves the skip/refine
policy, the other improves the refine path's own cost over time. Every
stored embedding (RefinementMemory's cache and DriftAuditLog's audit
trail below) uses vector_storage_optimizer.py's scalar quantization — a
real ~4x memory reduction vs raw floats, not just mentioned in docs.
"""
from __future__ import annotations

import json
import re
import threading
from collections import deque
from dataclasses import dataclass, field

from ..llm.router import get_llm_client
from ..retrieval.vectorstore import HashEmbedder
from .bandit import LinearBanditCore
from .vector_storage_optimizer import dequantize_vector, quantize_vector

_FEATURE_DIM = 3  # [word_count/30 clamped, filler_score, punctuation_issue_score]
_SIMILARITY_THRESHOLD = 0.6  # below this, the rewrite is judged too risky to trust silently

_FILLER_PATTERNS = re.compile(
    r"\b(could you|can you|please|i want to know|i would like|show me|tell me about|kindly)\b",
    re.IGNORECASE)

_embedder = HashEmbedder()

_SYSTEM_PROMPT = {"role": "system", "content": (
    "You rewrite the user's message for grammar and clarity ONLY. Never narrow, "
    "substitute, upgrade, or add precision to what they actually said — if they "
    "said \"good growth\", keep \"good\", do not change it to \"strong\". Preserve "
    "every constraint, metric, and ambiguity exactly as given. Respond with ONLY a "
    "JSON object: {\"refined\": \"<rewritten text>\", \"missing_clarifications\": "
    "[<up to 2 short strings, or empty>], \"suggested_queries\": [<up to 3 short "
    "alternative phrasings, or empty>]}"
)}


def _parse_json_dict(raw: str) -> dict:
    try:
        parsed = json.loads(raw) if raw and raw.strip() else {}
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _complexity_features(message: str) -> list[float]:
    words = message.split()
    filler_hits = len(_FILLER_PATTERNS.findall(message))
    punctuation_issue = 1.0 if (len(message) > 40 and message == message.lower()
                                and not message.rstrip().endswith(("?", ".", "!"))) else 0.0
    return [min(len(words) / 30.0, 1.0), min(filler_hits / 3.0, 1.0), punctuation_issue]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # HashEmbedder output is already unit-normalized


def embedding_similarity(a: str, b: str) -> float:
    """Zero-LLM-cost semantic-drift signal — same embedder rag.py falls
    back to for SLM-safe, dependency-free retrieval."""
    return _cosine(_embedder.embed(a), _embedder.embed(b))


@dataclass
class DriftRecord:
    original: str
    refined: str
    similarity: float


class DriftAuditLog:
    """Bounded in-memory trail of drifted refinements, storing each
    pair's ORIGINAL embedding quantized (vector_storage_optimizer.py's
    scalar quantization — a real 4x memory reduction vs raw floats) so
    a longer history can be kept without floats piling up. Text is kept
    as-is (it's already short chat messages, not the bottleneck)."""

    def __init__(self, maxlen: int = 200) -> None:
        self._log: deque[DriftRecord] = deque(maxlen=maxlen)
        self._quantized: deque[tuple[bytes, float, float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, original: str, refined: str, similarity: float) -> None:
        q_bytes, q_min, q_scale = quantize_vector(_embedder.embed(original))
        with self._lock:
            self._log.append(DriftRecord(original=original, refined=refined, similarity=similarity))
            self._quantized.append((q_bytes, q_min, q_scale))

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [{"original": r.original, "refined": r.refined, "similarity": round(r.similarity, 3)}
                    for r in self._log]


_MEMORY_HIT_THRESHOLD = 0.93  # near-duplicate query: reuse the cached refinement, skip the LLM entirely


@dataclass
class MemoryEntry:
    raw: str
    refined: str
    q_bytes: bytes
    q_min: float
    q_scale: float


class RefinementMemory:
    """Grows as the pipeline runs — every ACCEPTED refinement (passed the
    drift check, actually changed the text) is remembered here, its raw
    query's embedding stored via vector_storage_optimizer.py's scalar
    quantization (~4x less memory than raw floats). A later query similar
    enough to one already seen reuses its cached refinement directly
    instead of paying for another LLM round trip — the literal meaning of
    "refines as it goes": this isn't a static prompt, its effective hit
    rate on repeat/near-duplicate phrasing genuinely improves the longer
    the process runs, at zero extra LLM cost for a hit."""

    def __init__(self, maxlen: int = 500) -> None:
        self._entries: deque[MemoryEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def remember(self, raw: str, refined: str) -> None:
        q_bytes, q_min, q_scale = quantize_vector(_embedder.embed(raw))
        with self._lock:
            self._entries.append(MemoryEntry(raw=raw, refined=refined,
                                             q_bytes=q_bytes, q_min=q_min, q_scale=q_scale))

    def lookup(self, raw: str) -> str | None:
        q = _embedder.embed(raw)
        with self._lock:
            entries = list(self._entries)
        best_sim, best_refined = 0.0, None
        for e in entries:
            sim = _cosine(q, dequantize_vector(e.q_bytes, e.q_min, e.q_scale))
            if sim > best_sim:
                best_sim, best_refined = sim, e.refined
        return best_refined if best_sim >= _MEMORY_HIT_THRESHOLD else None

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class RefinementSelection:
    arm: str  # "refine" | "skip"
    context: list[float]
    text_changed: bool = False
    llm_error: bool = False
    from_memory: bool = False


@dataclass
class RefinementResult:
    original_text: str
    refined_text: str
    was_repaired: bool = False
    similarity: float | None = None
    missing_clarifications: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    selection: RefinementSelection | None = None


class QueryRefinementPipeline:
    """One LinearBanditCore gating "is refining this message worth an
    LLM round trip" — same math as every other optimizer layer, applied
    one step earlier than any of them (before intent/RAG/hybrid routing
    even see the message)."""

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.35):
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._lock = threading.Lock()
        self.memory = RefinementMemory()
        self.drift_log = DriftAuditLog()

    def refine(self, raw_query: str) -> RefinementResult:
        context = _complexity_features(raw_query)
        with self._lock:
            arm, _ = self._core.select(["refine", "skip"], context)
        if arm == "skip" or not raw_query.strip():
            return RefinementResult(original_text=raw_query, refined_text=raw_query,
                                    selection=RefinementSelection(arm="skip", context=context))

        cached = self.memory.lookup(raw_query)
        if cached is not None:
            selection = RefinementSelection(arm="refine", context=context,
                                            text_changed=(cached != raw_query), from_memory=True)
            return RefinementResult(original_text=raw_query, refined_text=cached, selection=selection)

        client = get_llm_client("realtime")
        res = client.chat("query_refinement", [_SYSTEM_PROMPT, {"role": "user", "content": raw_query}])
        parsed = _parse_json_dict(res.text or "")
        refined_text = (parsed.get("refined") or "").strip()
        if res.error or not refined_text:
            selection = RefinementSelection(arm="refine", context=context, llm_error=True)
            return RefinementResult(original_text=raw_query, refined_text=raw_query, selection=selection)

        similarity = embedding_similarity(raw_query, refined_text)
        was_repaired = similarity < _SIMILARITY_THRESHOLD
        final_text = raw_query if was_repaired else refined_text
        if was_repaired:
            self.drift_log.record(raw_query, refined_text, similarity)
        elif final_text != raw_query:
            self.memory.remember(raw_query, final_text)

        missing = [m for m in (parsed.get("missing_clarifications") or []) if isinstance(m, str)][:2]
        suggested = [s for s in (parsed.get("suggested_queries") or []) if isinstance(s, str)][:3]
        selection = RefinementSelection(arm="refine", context=context,
                                        text_changed=(final_text != raw_query))
        return RefinementResult(
            original_text=raw_query, refined_text=final_text, was_repaired=was_repaired,
            similarity=similarity, missing_clarifications=missing, suggested_queries=suggested,
            needs_confirmation=was_repaired or bool(missing), selection=selection)

    def record_outcome(self, selection: RefinementSelection | None, *, success: bool) -> None:
        """Reward mirrors hybrid_router.py's asymmetric design: "skip" is
        the free/fast choice (rewarded well on success, penalized hard on
        failure — an unclear message that WASN'T refined and then failed
        is exactly the case this gate exists to catch), "refine" is the
        costly/safer choice (a refinement that changed nothing wasted a
        real LLM call regardless of outcome; an outright LLM failure is
        worse still). A memory hit (RefinementMemory) is refine's BEST
        outcome — the clarity benefit of a full refine, at skip's zero
        LLM cost — so it's barely penalized even on failure."""
        if selection is None:
            return
        if selection.arm == "skip":
            reward = 1.0 if success else -0.6
        elif selection.from_memory:
            reward = 1.0 if success else -0.1
        elif selection.llm_error:
            reward = -0.5
        elif not selection.text_changed:
            reward = -0.3
        else:
            reward = 1.0 if success else -0.2
        with self._lock:
            self._core.update(selection.arm, selection.context, max(-1.0, min(1.0, reward)))


# Process-wide singleton, same lifecycle as hybrid_router/rag_optimizer/token_optimizer.
query_refinement = QueryRefinementPipeline()
