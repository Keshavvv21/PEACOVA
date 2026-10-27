"""Coverage for scripts/run_slm_test_suite.py's own logic — the quality
heuristic and report math — independent of any real provider call
(those are exercised live, not in the unit suite; see
README_SLM_TEST_SUITE.md for how to actually run the 100-prompt suite).
"""
from __future__ import annotations

import json

from scripts.run_slm_test_suite import PROMPTS, PromptResult, _quality_check, write_report


def test_prompt_catalog_has_at_least_100_unique_ids():
    assert len(PROMPTS) >= 100
    ids = [p[0] for p in PROMPTS]
    assert len(ids) == len(set(ids))


def test_prompt_catalog_covers_every_requested_category():
    categories = {c for _, c, _ in PROMPTS}
    for required in ("generate", "debug", "refactor", "test", "repo_analysis"):
        assert required in categories


def test_quality_check_flags_missing_code_block_for_a_generate_prompt():
    notes = _quality_check("generate", "write a function", "Sure, here's how you'd think about it.")
    assert any("No fenced code block" in n for n in notes)


def test_quality_check_accepts_a_real_valid_code_block():
    text = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```"
    notes = _quality_check("generate", "write a function", text)
    assert notes == []


def test_quality_check_flags_a_syntactically_broken_code_block():
    text = "```python\ndef f(x)\n    return x + 1\n```"  # missing colon
    notes = _quality_check("generate", "write a function", text)
    assert any("did not parse" in n for n in notes)


def test_quality_check_flags_suspiciously_short_responses():
    notes = _quality_check("general_coding", "explain X", "ok")
    assert any("short" in n.lower() for n in notes)


def test_write_report_computes_correct_pass_rate_and_averages(tmp_path):
    results = [
        PromptResult(id="A", category="generate", prompt="p1", passed=True, latency_ms=100.0,
                    prompt_tokens=10, completion_tokens=20, error=None, response_preview="ok",
                    intent_classified="generate", intent_matches_category=True),
        PromptResult(id="B", category="generate", prompt="p2", passed=False, latency_ms=200.0,
                    prompt_tokens=10, completion_tokens=0, error="timeout", response_preview="",
                    intent_classified="generate", intent_matches_category=True),
        PromptResult(id="C", category="debug", prompt="p3", passed=True, latency_ms=300.0,
                    prompt_tokens=15, completion_tokens=40, error=None, response_preview="ok",
                    intent_classified="debug", intent_matches_category=True),
    ]
    out = tmp_path / "report.json"
    write_report(results, out, "ollama-coder", "coding")
    report = json.loads(out.read_text())
    assert report["summary"]["total_prompts"] == 3
    assert report["summary"]["passed"] == 2
    assert report["summary"]["failed"] == 1
    assert report["summary"]["pass_rate"] == round(2 / 3, 3)
    # B errored, so only A and C's latencies count toward the average.
    assert report["summary"]["avg_latency_ms"] == 200.0
    assert report["summary"]["intent_reasoner_accuracy"] == 1.0
    assert report["by_category"]["generate"]["total"] == 2
    assert report["by_category"]["generate"]["passed"] == 1
    assert report["by_category"]["debug"]["passed"] == 1
