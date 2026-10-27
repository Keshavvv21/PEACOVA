"""The Coding Agent's intent-reasoning layer — a cheap, deterministic
pre-generation step, not another LLM round trip. See README_SLM_
REASONING_LAYER.md for the full architecture and why this is a
regex/keyword heuristic rather than a second model call.

Deliberately NOT a second LLM call: the whole point of an intent layer
here is to make the SLM's SINGLE generation better-targeted, not to add
a round trip in front of it — an SLM is already the latency-sensitive
path (see providers.py's OLLAMA_KEEP_ALIVE comment, hybrid_router.py's
LOCAL_TIMEOUT_SECONDS), so a design that "reasons about intent" via yet
another model call would directly fight the exact latency goal this
layer exists to support. A regex/keyword classifier runs in
microseconds and costs nothing.

classify_intent() maps a task's raw text to one of a small, fixed set
of coding-task intents — the same categories
scripts/run_slm_test_suite.py's own 100-prompt catalog is organized by,
so "does the reasoning layer route this prompt into the category a
human would file it under" is itself a testable claim, not just an
assertion.
"""
from __future__ import annotations

import re

INTENT_DEBUG = "debug"
INTENT_REFACTOR = "refactor"
INTENT_TEST = "test"
INTENT_REPO_ANALYSIS = "repo_analysis"
INTENT_GENERATE = "generate"
INTENT_GENERAL = "general_coding"

# Order matters — checked top to bottom, first match wins. Debug/refactor/
# test/repo-analysis are checked BEFORE the generic "generate" bucket
# because a prompt like "write a test that reproduces this bug" should
# resolve to debug (the more SPECIFIC, more actionable intent), not the
# generic "write" verb that also appears in it.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (INTENT_DEBUG, re.compile(
        r"\b(debug|diagnos\w*|root cause|fix (this|the|a)|why (is|does|did)|"
        r"error|bug|crash(es|ed|ing)?|exception|traceback|stack trace|"
        r"fails?|failing|broken|not working|doesn'?t work|race condition|"
        r"deadlock|memory leak|hangs?|stuck|regression)\b", re.I)),
    (INTENT_REFACTOR, re.compile(
        r"\b(refactor|clean ?up|simplify|reorgani[sz]e|extract\w* (a |the )?"
        r"(function|method|class|module)|rename|restructure|de-?duplicate|"
        r"split (this|the) (function|file|class)|improve readability)\b", re.I)),
    (INTENT_TEST, re.compile(
        r"\b(unit tests?|write tests?|pytest|test coverage|test suite|"
        r"assert\w*|mock\w*|edge cases?)\b", re.I)),
    (INTENT_REPO_ANALYSIS, re.compile(
        r"\b(repository|repo-?level|codebase|across (the|this) (repo|"
        r"codebase|project)|entire project|every file|whole project|"
        r"this project'?s)\b", re.I)),
    (INTENT_GENERATE, re.compile(
        r"\b(write|create|implement|generate|build|add (a|an|the)|"
        r"design (a|an))\b", re.I)),
]

# Appended to the system prompt when non-empty (see coding_agent.py's
# plan()) — short, imperative, and specific to what actually goes wrong
# for THIS intent, mirroring SYSTEM_PROMPT's own "imperative, not
# descriptive" style (its own docstring explains why: a softer,
# descriptive phrasing lets models narrate instead of act).
_GUIDANCE: dict[str, str] = {
    INTENT_DEBUG: (
        " The user is asking you to DEBUG something real. Find the actual "
        "root cause before proposing a fix — read the file or run the code "
        "to see the real error first; do not guess at a fix from the "
        "description alone if you can verify it directly instead."
    ),
    INTENT_REFACTOR: (
        " The user is asking you to REFACTOR. Preserve existing behavior "
        "exactly — read the current code first, then change structure or "
        "clarity only. Never change behavior unless they explicitly asked "
        "for a behavior change too."
    ),
    INTENT_TEST: (
        " The user is asking you to write or run TESTS. Cover the real "
        "edge cases (empty input, boundary values, error paths) — not just "
        "the happy path."
    ),
    INTENT_REPO_ANALYSIS: (
        " The user is asking about the REPOSITORY as a whole, not an "
        "isolated snippet. Use list_files/grep to actually look at real "
        "files before answering — never answer from assumptions about "
        "what the codebase probably contains."
    ),
    INTENT_GENERATE: (
        " The user wants NEW code written. Make sure what you return is "
        "syntactically complete and directly runnable, not a sketch or "
        "partial fragment."
    ),
    INTENT_GENERAL: "",
}


def classify_intent(text: str) -> str:
    for intent, pattern in _PATTERNS:
        if pattern.search(text):
            return intent
    return INTENT_GENERAL


def guidance_for(intent: str) -> str:
    return _GUIDANCE.get(intent, "")
