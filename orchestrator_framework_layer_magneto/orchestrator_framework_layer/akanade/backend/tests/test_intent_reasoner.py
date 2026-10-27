"""Coverage for llm/intent_reasoner.py — the Coding Agent's intent-
reasoning layer. See README_SLM_REASONING_LAYER.md for the architecture.
"""
from __future__ import annotations

import pytest

from app.llm.intent_reasoner import (
    INTENT_DEBUG,
    INTENT_GENERAL,
    INTENT_GENERATE,
    INTENT_REFACTOR,
    INTENT_REPO_ANALYSIS,
    INTENT_TEST,
    classify_intent,
    guidance_for,
)


@pytest.mark.parametrize("text,expected", [
    ("Debug why this Kafka consumer keeps rebalancing every few seconds.", INTENT_DEBUG),
    ("Find and fix a race condition in a Python asyncio producer-consumer system.", INTENT_DEBUG),
    ("My script crashes with a KeyError, why?", INTENT_DEBUG),
    ("Refactor this 400-line function into smaller modules without changing behavior.", INTENT_REFACTOR),
    ("Clean up this class, it's hard to read.", INTENT_REFACTOR),
    ("Write comprehensive pytest tests for this FastAPI service.", INTENT_TEST),
    ("Add unit tests covering the edge cases.", INTENT_TEST),
    ("Review this entire codebase for security issues.", INTENT_REPO_ANALYSIS),
    ("Analyze the whole project's dependency structure.", INTENT_REPO_ANALYSIS),
    ("Write a Python function that reverses a string.", INTENT_GENERATE),
    ("Implement a thread-safe LRU cache.", INTENT_GENERATE),
    ("What is a Kafka consumer group?", INTENT_GENERAL),
])
def test_classify_intent_matches_the_obvious_human_reading(text, expected):
    assert classify_intent(text) == expected


def test_debug_wins_over_generate_when_both_verbs_present():
    # "write a test that reproduces this bug" mentions both "write" (a
    # generate-shaped verb) and "bug" (debug) — debug is the more
    # specific, more actionable intent and must win (see _PATTERNS'
    # own ordering comment).
    assert classify_intent("Write a test that reproduces this bug in the parser.") == INTENT_DEBUG


def test_general_guidance_is_empty_string():
    # The general/fallback bucket adds NO extra instruction — a
    # plain-knowledge question shouldn't get debug/refactor/test framing
    # bolted on just because it fell through every specific pattern.
    assert guidance_for(INTENT_GENERAL) == ""


def test_every_specific_intent_has_real_non_empty_guidance():
    for intent in (INTENT_DEBUG, INTENT_REFACTOR, INTENT_TEST, INTENT_REPO_ANALYSIS, INTENT_GENERATE):
        assert guidance_for(intent).strip(), f"{intent} has no guidance text"
