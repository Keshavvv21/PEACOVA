"""Tests for the candidate-selection eval pipeline's pure logic
(scripts/eval/) — deterministic scoring, aggregation math, weight
validation, and the human-sample weighting. No network calls (those are
exercised by hand against the real OpenRouter API — see deploy/README.md's
"Validated in-sandbox" note); this file covers the parts that don't need
one.
"""
import random

import pytest

from scripts.eval import regression_gate
from scripts.eval.aggregate import aggregate, load_weights, percentile
from scripts.eval.decide import rank
from scripts.eval.human_sample import sample
from scripts.eval.judge import judge_score
from scripts.eval.harness import score_deterministic


def test_score_deterministic_exact_match_is_case_and_whitespace_insensitive():
    item = {"match": "exact", "gold_answer": "  Beta  "}
    assert score_deterministic(item, "beta") == 1.0
    assert score_deterministic(item, "Gamma") == 0.0


def test_score_deterministic_regex_searches_not_fullmatches():
    item = {"match": "regex", "gold_pattern": r"(?i)\b5\b"}
    assert score_deterministic(item, "there are 5 todo tasks") == 1.0
    assert score_deterministic(item, "there are 6 todo tasks") == 0.0


def test_score_deterministic_schema_validates_json_and_strips_markdown_fences():
    item = {"match": "schema", "gold_schema": {
        "type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}},
    }}
    assert score_deterministic(item, '{"n": 3}') == 1.0
    assert score_deterministic(item, '```json\n{"n": 3}\n```') == 1.0
    assert score_deterministic(item, '{"n": "not an int"}') == 0.0
    assert score_deterministic(item, "not json at all") == 0.0


def test_load_weights_rejects_a_set_that_does_not_sum_to_one(tmp_path):
    bad = tmp_path / "weights.yaml"
    bad.write_text("a: 0.5\nb: 0.6\n")
    with pytest.raises(ValueError):
        load_weights(bad)


def test_load_weights_accepts_a_set_that_sums_to_one(tmp_path):
    good = tmp_path / "weights.yaml"
    good.write_text("a: 0.4\nb: 0.6\n")
    assert load_weights(good) == {"a": 0.4, "b": 0.6}


def test_percentile_p50_and_p95_on_a_known_list():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 50) == 5.0
    assert percentile(values, 95) == 10.0
    assert percentile([], 50) == 0.0


def test_aggregate_computes_weighted_score_and_cost_per_1k():
    weights = {"a": 0.6, "b": 0.4}
    records = [
        {"candidate_id": "m1", "category": "a", "score": 1.0, "latency_s": 1.0,
         "cost_usd": 0.001, "error": None},
        {"candidate_id": "m1", "category": "b", "score": 0.5, "latency_s": 2.0,
         "cost_usd": 0.002, "error": None},
    ]
    report = aggregate(records, weights)
    assert report["m1"]["by_category"] == {"a": 1.0, "b": 0.5}
    assert report["m1"]["weighted_score"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)
    assert report["m1"]["cost_per_1k_requests_usd"] == pytest.approx((0.003 / 2) * 1000)
    assert report["m1"]["latency_p50_s"] in (1.0, 2.0)


def test_aggregate_excludes_errored_calls_from_latency_but_keeps_them_in_score():
    weights = {"a": 1.0}
    records = [
        {"candidate_id": "m1", "category": "a", "score": 0.0, "latency_s": 0.0,
         "cost_usd": 0.0, "error": "boom"},
    ]
    report = aggregate(records, weights)
    assert report["m1"]["n_errors"] == 1
    assert report["m1"]["latency_p50_s"] == 0.0  # no successful calls to measure


def test_rank_orders_by_score_penalized_by_cost_and_latency():
    report = {
        "cheap_slightly_worse": {"weighted_score": 0.80, "cost_per_1k_requests_usd": 0.0,
                                 "latency_p95_s": 0.5},
        "expensive_slightly_better": {"weighted_score": 0.82, "cost_per_1k_requests_usd": 50.0,
                                      "latency_p95_s": 20.0},
    }
    ranked = rank(report)
    assert ranked[0][0] == "cheap_slightly_worse"  # cost/latency penalty flips the order


def test_human_sample_weights_toward_reasoning_planning_complex_reasoning():
    records = ([{"item_id": f"extract_{i}", "category": "extraction", "needs_human_review": True}
               for i in range(20)]
              + [{"item_id": f"reason_{i}", "category": "reasoning", "needs_human_review": True}
                 for i in range(20)])
    chosen = sample(records, frac=0.5, weight_multiplier=5.0, rng=random.Random(42))
    reasoning_count = sum(1 for r in chosen if r["category"] == "reasoning")
    assert reasoning_count > len(chosen) / 2  # weighted category picked more than its 50% share


def test_judge_score_mock_rewards_word_overlap_with_gold_answer():
    result = judge_score(answer="the sky is blue", gold_answer="the sky is blue",
                         rubric="exact match expected", use_real_judge=False)
    assert result["is_mock"] is True
    assert result["score"] == 1.0

    result = judge_score(answer="completely unrelated text", gold_answer="the sky is blue",
                         rubric="exact match expected", use_real_judge=False)
    assert result["score"] < 1.0


def test_judge_score_real_judge_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        judge_score(answer="x", gold_answer="y", rubric="z", use_real_judge=True)


def test_regression_gate_exits_2_not_1_when_test_assets_are_missing(tmp_path, monkeypatch):
    """A missing/empty test-assets dir must be diagnosed as exit 2 ("couldn't
    run the gate") — not crash into Python's default exit 1, which the
    caller (convert_and_register_ollama_model.sh) would misread as "ran
    fine, genuinely regressed" instead of "setup problem, go fix it"."""
    empty_dir = tmp_path / "empty_test_assets"
    empty_dir.mkdir()
    monkeypatch.setattr("sys.argv", [
        "regression_gate.py", "--base-tag", "old:1b", "--new-tag", "new:1b",
        "--test-assets-dir", str(empty_dir), "--weights", str(tmp_path / "weights.yaml"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        regression_gate.main()
    assert exc_info.value.code == 2
