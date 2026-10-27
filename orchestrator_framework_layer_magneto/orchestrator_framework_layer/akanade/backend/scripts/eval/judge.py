"""LLM-as-Judge (§D) — Claude as referee, never a benchmarked candidate.

Stubbed for now (per explicit choice — no Anthropic key wired into this
pipeline yet): judge_score() returns a deterministic word-overlap
heuristic against the gold answer, clearly flagged is_mock=True in every
record it produces (harness.py propagates that flag into the results
file, and aggregate.py surfaces it in the report) so a real run's numbers
are never silently confused with mock ones.

To wire in the real judge once you have an ANTHROPIC_API_KEY: replace the
body of _real_judge_score() below with an actual API call. The function
signature already enforces the bias guard — it only ever receives
`answer`, `gold_answer`, and `rubric`, never a candidate id or model
name, so the real implementation can't be swayed by knowing which model
produced the answer. Ask for a structured JSON reply
({"score": 0.0-1.0, "reasoning": str}) and parse it the same way
score_deterministic() does in harness.py — fail closed (score 0, don't
crash the run) on a malformed judge reply.
"""
from __future__ import annotations

import re


def _mock_judge_score(answer: str, gold_answer: str, rubric: str) -> float:
    """Heuristic stand-in: word-overlap between the answer and the gold
    answer, as a rough (and clearly-labeled-mock) proxy for what a real
    judge would score. Good enough to smoke-test the pipeline's plumbing
    end-to-end; not a substitute for real judgment."""
    def words(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    gold_words, answer_words = words(gold_answer), words(answer)
    if not gold_words:
        return 0.0
    overlap = len(gold_words & answer_words) / len(gold_words)
    return round(min(1.0, overlap), 2)


def _real_judge_score(answer: str, gold_answer: str, rubric: str) -> float:
    raise NotImplementedError(
        "Real Claude-as-judge isn't wired in yet — set up an ANTHROPIC_API_KEY and "
        "implement this function (see this module's docstring), or drop --real-judge."
    )


def judge_score(*, answer: str, gold_answer: str, rubric: str, use_real_judge: bool) -> dict:
    if use_real_judge:
        score = _real_judge_score(answer, gold_answer, rubric)
        return {"score": score, "is_mock": False}
    return {"score": _mock_judge_score(answer, gold_answer, rubric), "is_mock": True}
