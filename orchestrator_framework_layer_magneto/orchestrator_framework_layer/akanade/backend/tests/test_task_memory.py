"""app/gmaofn/task_memory.py — the embedding-based routing memory that
closes router.py's "learnings stored in a vector store" gap. Mirrors
test_query_refinement.py's RefinementMemory test conventions.
"""
from __future__ import annotations

from app.framework.task_memory import TaskMemory


def test_bonus_is_zero_with_no_memory_yet():
    memory = TaskMemory()
    assert memory.bonus_for("fix the failing tests", "coder") == 0.0


def test_remember_then_bonus_for_a_near_duplicate_request():
    memory = TaskMemory()
    memory.remember("fix the failing unit tests in the payments module", "coder", 0.9)
    # One word swapped ("payments" → "billing") — a genuine near-
    # duplicate; changing TWO content words at once (the old example
    # here) drops cosine similarity below the match threshold once
    # stopwords no longer pad the shared-word count — see
    # vectorstore.py's HashEmbedder docstring on that fix.
    bonus = memory.bonus_for("fix the failing unit tests in the billing module", "coder")
    assert bonus > 0.0


def test_bonus_is_agent_specific():
    memory = TaskMemory()
    memory.remember("fix the failing unit tests in the payments module", "coder", 0.9)
    # A DIFFERENT agent never won anything like this — no bonus just
    # because SOME agent did.
    assert memory.bonus_for("fix the failing unit tests in the payments module", "writer") == 0.0


def test_bonus_reflects_a_past_bad_outcome_not_just_a_good_one():
    memory = TaskMemory()
    memory.remember("fix the failing unit tests in the payments module", "coder", -0.8)
    bonus = memory.bonus_for("fix the broken unit tests in the payments module", "coder")
    assert bonus < 0.0


def test_bonus_ignores_an_unrelated_request():
    memory = TaskMemory()
    memory.remember("fix the failing unit tests in the payments module", "coder", 1.0)
    assert memory.bonus_for("what's the weather like on Mars", "coder") == 0.0


def test_remember_and_bonus_for_ignore_blank_text():
    memory = TaskMemory()
    memory.remember("", "coder", 1.0)
    memory.remember("   ", "coder", 1.0)
    assert len(memory) == 0
    assert memory.bonus_for("", "coder") == 0.0


def test_memory_is_bounded():
    memory = TaskMemory(maxlen=3)
    for i in range(10):
        memory.remember(f"request number {i} about something specific", "coder", 1.0)
    assert len(memory) == 3
