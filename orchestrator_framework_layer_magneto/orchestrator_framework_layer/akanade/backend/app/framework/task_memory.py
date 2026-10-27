"""Agent-routing memory — the "learnings stored in a vector store as
embeddings" piece router.py's bandit was missing on its own:
`LinearBanditCore` learns purely from ENGINEERED features
(`TaskContext.as_vector()`), with no sense of "have I seen a request
like THIS one before." `TaskMemory` closes that gap the same way
`query_refinement.py`'s `RefinementMemory` already does for message
refinement — quantized embeddings (`vector_storage_optimizer.py`) of
each task's raw text, paired with which agent actually won and how well
it did, looked up by cosine similarity the next time a similar request
arrives.

This is a WARM-START signal, not a bypass: a close match nudges the
matching agent's seed_utility up or down before Thompson Sampling still
runs its own exploration — the bandit's own learned posterior remains
the real decision-maker once enough data accumulates for a given
context. Exactly the role `utility_prior()` already plays for a
brand-new agent, extended here to "a brand-new KIND of request this
agent hasn't specifically seen yet." Every stored embedding uses the
same scalar quantization as the rest of this framework's memory layers
— real reuse, not a separate implementation.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from ..retrieval.vectorstore import HashEmbedder
from .vector_storage_optimizer import dequantize_vector, quantize_vector

# Looser than query_refinement.py's 0.93 near-duplicate bar — routing
# wants "the same KIND of request," not "the identical string."
_MATCH_THRESHOLD = 0.75
# Cap on how far a memory match can nudge an agent's seed_utility — a
# STRONG hint, never enough on its own to override a real bandit
# posterior once one exists (see ContextualThompsonRouter.score).
_MAX_BONUS = 0.5

_embedder = HashEmbedder()


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # HashEmbedder output is already unit-normalized


@dataclass
class _MemoryEntry:
    agent_id: str
    reward: float
    q_bytes: bytes
    q_min: float
    q_scale: float


class TaskMemory:
    """Grows as the router runs — every task an agent was actually
    scored against is remembered by its text's quantized embedding.
    Reward-weighted: a past WIN nudges that agent up for a similar
    future request; a past bad outcome nudges it down, same reward
    scale the bandit itself already uses."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._entries: deque[_MemoryEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def remember(self, task_text: str, agent_id: str, reward: float) -> None:
        if not task_text or not task_text.strip():
            return
        q_bytes, q_min, q_scale = quantize_vector(_embedder.embed(task_text))
        with self._lock:
            self._entries.append(_MemoryEntry(agent_id=agent_id, reward=reward,
                                              q_bytes=q_bytes, q_min=q_min, q_scale=q_scale))

    def bonus_for(self, task_text: str, agent_id: str) -> float:
        """The best-matching past outcome's reward, scaled into
        [-MAX_BONUS, MAX_BONUS] — 0.0 when there's no memory yet for a
        request this similar and THIS agent, positive after a past win,
        negative after a past failure on the same kind of request."""
        if not task_text or not task_text.strip():
            return 0.0
        q = _embedder.embed(task_text)
        with self._lock:
            entries = list(self._entries)
        best_sim, best_reward = 0.0, 0.0
        for e in entries:
            if e.agent_id != agent_id:
                continue
            sim = _cosine(q, dequantize_vector(e.q_bytes, e.q_min, e.q_scale))
            if sim >= _MATCH_THRESHOLD and sim > best_sim:
                best_sim, best_reward = sim, e.reward
        return best_reward * _MAX_BONUS if best_sim > 0 else 0.0

    def __len__(self) -> int:
        return len(self._entries)
