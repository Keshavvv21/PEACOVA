"""Persists aggregate.py's report.json into the ModelBenchmarks table —
the "model evaluation database" the Model Evaluation dashboard reads,
alongside (not instead of) the human-readable report.md/decision.md
aggregate.py already writes.

One row per (candidate, task-category) pair — finer-grained than one
row per candidate, matching the CEO memo's own "overall + by task
family" ask; a candidate with no scored items in a given category is
skipped (aggregate.py's own `by_category[c]` is None there).

Deliberately a SEPARATE script from aggregate.py, not a flag added to
it — aggregate.py stays dependency-free of app.db (its own venv is the
lightweight eval_requirements.txt, no boto3), and this script imports
the SAME app.db.store() everything else in this codebase uses, run
against report.json's OUTPUT rather than the raw per-item records, so
it can run in a totally different Python environment (the main
backend's, not eval-venv) at a different point in the pipeline (see
run_eval.sh / a future daily_model_benchmark.sh, which runs this right
after aggregate.py, in the backend's own venv, not eval-venv).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import store  # noqa: E402


def write_benchmarks(report: dict, *, model_source: str, run_started_at: str) -> int:
    """Returns the number of rows written."""
    n = 0
    for candidate_id, r in report.items():
        for category, score in r["by_category"].items():
            if score is None:
                continue  # this candidate had no scored items in this category
            row = {
                "timestamp": run_started_at,
                "model": candidate_id,
                "modelSource": model_source,
                "tier": None,  # not tracked at the harness level — see generic_architecture.py's own outcome rows for tier-tagged data
                "taskFamily": category,
                "ok": True,
                "latencyMs": r["latency_p50_s"] * 1000,
                "latencyP95Ms": r["latency_p95_s"] * 1000,
                "costUsd": r["cost_per_1k_requests_usd"],
                "qualityScore": score,
                "weightedScore": r["weighted_score"],
                # Not tracked per-category by aggregate.py today — 0 rather
                # than a guessed number, same "honest zero, not a fabricated
                # placeholder" rule pricing_table.py's own fallback rate
                # follows for an unlisted provider.
                "promptTokens": 0,
                "completionTokens": 0,
                "successRate": round((r["n_items"] - r["n_errors"]) / r["n_items"], 3) if r["n_items"] else 0.0,
                "nItems": r["n_items"],
                "mockJudged": bool(r["n_mock_judged"]),
            }
            store("ModelBenchmarks").put(f"BENCHMARK#{uuid.uuid4().hex[:12]}", "META", row)
            n += 1
    return n


def main() -> None:
    here = Path(__file__).resolve().parent
    eval_dir = here.parent.parent / "eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=eval_dir / "results" / "report.json")
    parser.add_argument("--model-source", choices=["ours", "candidate"], default="ours")
    # Passed by admin_model_benchmarks.py's run_benchmark_sweep() so its
    # own _update_capability_tiers() step can match rows from THIS run
    # by exact timestamp — defaults to "now" for a standalone manual
    # invocation, where nothing downstream needs to correlate rows back
    # to a specific triggering call.
    parser.add_argument("--run-started-at", default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    run_started_at = args.run_started_at or datetime.now(timezone.utc).isoformat()
    n = write_benchmarks(report, model_source=args.model_source, run_started_at=run_started_at)
    print(f"Wrote {n} ModelBenchmarks rows from {args.report} (model_source={args.model_source}).")


if __name__ == "__main__":
    main()
