"""Model Evaluation — "automatically tests new models against our
workload" (the CEO memo's own "suggested first experiment," see
scripts/eval/harness.py) exposed as an admin-triggerable, background-
job-tracked action, plus the read side the Model Evaluation dashboard
(RouterMetricsDashboard.tsx's sibling) queries.

Same RBAC shape as every other admin action here: triggering a real run
is Executive (same tier as record_slm_version/record_eval_report's own
convention — it costs real API spend once real candidate/provider keys
are configured), viewing results is Admin.

The actual harness needs its own venv (scripts/eval_requirements.txt —
deliberately NOT a dependency of this always-running image, see
harness.py's own module docstring) — run_benchmark_sweep() below shells
out via subprocess rather than importing harness.py directly, exactly
like deploy/nightly_finetune.sh already does for finetune_local_slm.py.
On a deployment where that venv was never set up, the subprocess exits
non-zero and finish_job() records the real error — a missing optional
tool, not a crash of this endpoint or the whole chatbot process.
"""
from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import require_role
from ..db import store
from ..framework import model_benchmark_status

router = APIRouter(prefix="/api/admin/model-benchmarks", tags=["admin-model-benchmarks"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # .../akanade/backend
_EVAL_PYTHON = os.getenv("EVAL_PYTHON", "python3")


class RunBenchmarkRequest(BaseModel):
    model_source: str = "ours"  # "ours" | "candidate"


def _update_capability_tiers(run_started_at: str) -> None:
    """Closes the "Router flag/update" loop from the plan's architecture
    diagram: a model's capability_tier (ak-router's deterministic
    strategy input, see llm/capability_tiers.py) starts as a
    conservative registration-time guess and gets PATCHed here to the
    real measured weighted_score once a benchmark run actually has one
    — so ranking stays evidence-based, not frozen at a one-time guess.
    Best-effort and silent on any mismatch: a benchmarked candidate_id
    that doesn't match any admin-added Provider's own `name` (e.g. a
    built-in provider, or a "his architecture" candidate with no
    Providers row at all) just has nothing to update."""
    rows = [r for r in store("ModelBenchmarks").scan("BENCHMARK#", "")
           if r.get("timestamp") == run_started_at]
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r.get("weightedScore", r.get("qualityScore", 0.0)))
    if not by_model:
        return
    provider_rows = store("Providers").scan("PROVIDER#")
    for p in provider_rows:
        scores = by_model.get(p["name"])
        if not scores:
            continue
        avg = max(0.0, min(1.0, sum(scores) / len(scores)))
        p["capabilityTier"] = round(avg, 3)
        store("Providers").put(p["PK"], p["SK"], p)


def _run_benchmark_sweep(job_id: str, model_source: str) -> None:
    """The actual chain: harness -> aggregate -> write_benchmarks_to_db,
    same three steps run_eval.sh already documents, minus human_sample/
    decide (this is an automated background run, not the guided "which
    model should we self-host" workflow those two serve).

    Real, observed gap this closes: harness.py makes a real HTTP call
    per (candidate, item) with its OWN 180s timeout — a handful of
    candidates times a handful of items can genuinely take many
    minutes, sometimes exceeding a single subprocess's own timeout
    below. Launched on a plain daemon thread (see run_benchmark_sweep()
    below), NOT FastAPI's BackgroundTasks — Starlette's TestClient
    (and, worth knowing, some ASGI setups) waits for BackgroundTasks to
    finish before the request/response cycle completes; a real thread
    this function's caller doesn't join is genuinely fire-and-forget,
    so triggering a sweep never makes the /run call (or, via the
    provider-creation auto-trigger, POST /api/admin/providers itself)
    hang waiting on a live, multi-minute, real-network harness run."""
    run_started_at = datetime.now(timezone.utc).isoformat()
    try:
        steps = [
            # Same "${OPENROUTER_API_KEY:-}" pattern run_eval.sh already
            # uses — real, load-bearing gap this closes: without it,
            # every OpenRouter-backed candidates.yaml entry fails
            # outright (local-only Ollama candidates still score fine
            # either way, per harness.py's own graceful per-candidate
            # error handling).
            [_EVAL_PYTHON, "-m", "scripts.eval.harness",
             "--openrouter-api-key", os.getenv("OPENROUTER_API_KEY", "")],
            [_EVAL_PYTHON, "-m", "scripts.eval.aggregate"],
            [_EVAL_PYTHON, "-m", "scripts.eval.write_benchmarks_to_db",
             "--model-source", model_source, "--run-started-at", run_started_at],
        ]
        for cmd in steps:
            result = subprocess.run(cmd, cwd=_BACKEND_DIR, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                model_benchmark_status.finish_job(
                    job_id, ok=False,
                    summary=f"{' '.join(cmd)} failed (exit {result.returncode}): {result.stderr[-400:]}")
                return
        try:
            _update_capability_tiers(run_started_at)
        except Exception as e:  # never fail the whole job over the closed-loop step
            model_benchmark_status.finish_job(
                job_id, ok=True,
                summary=f"Benchmark sweep ({model_source}) completed; capability_tier update failed: {e}")
            return
        model_benchmark_status.finish_job(job_id, ok=True, summary=f"Benchmark sweep ({model_source}) completed.")
    except Exception as e:
        # Belt-and-suspenders: a job must never stay "running" forever
        # (subprocess.TimeoutExpired, a missing eval venv raising before
        # returncode is even meaningful, ...) — always resolve the job.
        model_benchmark_status.finish_job(job_id, ok=False, summary=f"Benchmark sweep crashed: {e}")


def run_benchmark_sweep(job_id: str, model_source: str) -> None:
    """Public entry point — starts _run_benchmark_sweep on a detached
    daemon thread and returns immediately. Callers (trigger_run() below,
    admin_providers.py's auto-trigger) never await this and never pass
    it to FastAPI's BackgroundTasks — see _run_benchmark_sweep's own
    docstring for why."""
    threading.Thread(target=_run_benchmark_sweep, args=(job_id, model_source), daemon=True).start()


@router.post("/run")
def trigger_run(payload: RunBenchmarkRequest, user: dict = Depends(require_role("Executive"))) -> dict:
    if payload.model_source not in ("ours", "candidate"):
        raise HTTPException(422, "model_source must be \"ours\" or \"candidate\"")
    job_id = model_benchmark_status.start_job(model_source=payload.model_source, triggered_by=user["sub"])
    run_benchmark_sweep(job_id, payload.model_source)
    return {"job_id": job_id}


@router.get("/jobs")
def list_jobs(_: dict = Depends(require_role("Admin"))) -> dict:
    jobs = model_benchmark_status.list_jobs()
    return {"jobs": jobs, "running": [j for j in jobs if j["status"] == "running"]}


@router.get("")
def list_benchmarks(model_source: str | None = None, limit: int = 200,
                    _: dict = Depends(require_role("Admin"))) -> dict:
    rows = store("ModelBenchmarks").scan("BENCHMARK#", "")
    if model_source:
        rows = [r for r in rows if r.get("modelSource") == model_source]
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return {"benchmarks": [{k: v for k, v in r.items() if k not in ("PK", "SK")} for r in rows[:limit]]}
