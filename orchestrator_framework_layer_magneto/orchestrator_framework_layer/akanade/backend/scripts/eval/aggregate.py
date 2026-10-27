"""Aggregation & Reporting (§E) — per-model x per-category score matrix,
latency p50/p95, cost per 1k requests, weighted by eval/weights.yaml
(agreed BEFORE scoring — this script refuses to run if the weights don't
sum to 1.0, so a run can't quietly rebalance categories after seeing
results).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import yaml

from .human_sample import merge_human_scores


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, round(p / 100 * (len(s) - 1)))
    return s[idx]


def load_weights(path: Path) -> dict[str, float]:
    weights = yaml.safe_load(path.open("r", encoding="utf-8"))
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{path} weights sum to {total}, not 1.0 — fix before aggregating, "
                         f"don't rebalance after seeing scores.")
    return weights


def aggregate(records: list[dict], weights: dict[str, float]) -> dict:
    by_candidate: dict[str, list[dict]] = {}
    for r in records:
        by_candidate.setdefault(r["candidate_id"], []).append(r)

    report = {}
    for candidate_id, rows in by_candidate.items():
        by_category = {}
        for category in weights:
            cat_rows = [r for r in rows if r["category"] == category]
            by_category[category] = round(mean(r["score"] for r in cat_rows), 3) if cat_rows else None

        weighted_score = sum((by_category[c] or 0.0) * w for c, w in weights.items())
        latencies = [r["latency_s"] for r in rows if r.get("error") is None]
        total_cost = sum(r["cost_usd"] for r in rows)
        n_items = len(rows)
        human_reviewed = sum(1 for r in rows if r.get("scored_by_human"))
        mock_judged = sum(1 for r in rows if r.get("judge_is_mock"))

        report[candidate_id] = {
            "by_category": by_category,
            "weighted_score": round(weighted_score, 3),
            "latency_p50_s": round(percentile(latencies, 50), 3),
            "latency_p95_s": round(percentile(latencies, 95), 3),
            "cost_per_1k_requests_usd": round(total_cost / n_items * 1000, 4) if n_items else 0.0,
            "n_items": n_items,
            "n_errors": sum(1 for r in rows if r.get("error")),
            "n_human_reviewed": human_reviewed,
            "n_mock_judged": mock_judged,
        }
    return report


def render_markdown(report: dict) -> str:
    lines = ["| Candidate | Weighted score | p50 latency | p95 latency | $/1k req | Errors | Mock-judged |",
            "|---|---|---|---|---|---|---|"]
    for candidate_id, r in sorted(report.items(), key=lambda kv: -kv[1]["weighted_score"]):
        lines.append(f"| {candidate_id} | {r['weighted_score']} | {r['latency_p50_s']}s | "
                     f"{r['latency_p95_s']}s | ${r['cost_per_1k_requests_usd']} | "
                     f"{r['n_errors']}/{r['n_items']} | {r['n_mock_judged']} |")
    if any(r["n_mock_judged"] for r in report.values()):
        lines.append("\n**Note:** scores marked mock-judged used the heuristic stand-in judge "
                     "(judge.py), not a real referee — treat this run as a plumbing smoke test, "
                     "not a real candidate-selection result, until a real judge is wired in.")
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    eval_dir = here.parent.parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=eval_dir / "results" / "raw.jsonl")
    parser.add_argument("--human-review-csv", type=Path,
                        default=eval_dir / "results" / "human_review.csv")
    parser.add_argument("--weights", type=Path, default=eval_dir / "weights.yaml")
    parser.add_argument("--out", type=Path, default=eval_dir / "results" / "report.json")
    parser.add_argument("--out-md", type=Path, default=eval_dir / "results" / "report.md")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.results.open("r", encoding="utf-8")]
    records = merge_human_scores(records, args.human_review_csv)
    weights = load_weights(args.weights)
    report = aggregate(records, weights)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"\nWrote {args.out} and {args.out_md}")


if __name__ == "__main__":
    main()
