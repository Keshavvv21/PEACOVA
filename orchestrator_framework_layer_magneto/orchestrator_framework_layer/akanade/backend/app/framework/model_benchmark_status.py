"""Live status for a model-benchmark run — the "background job
execution with a live status bar" pattern, same shape as
peft_job_status.py, applied to `scripts/eval/harness.py` runs instead
of nightly fine-tune jobs. `admin_model_benchmarks.py`'s `/run`
endpoint calls `start_job()` right before launching the harness
subprocess, and `finish_job()` once it (and aggregate.py +
write_benchmarks_to_db.py, chained after it) actually exits — so
`GET /api/admin/model-benchmark-jobs` can show "is a benchmark sweep
running right now" the same way the existing PEFT status bar does for
fine-tune jobs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..db import store as db_store

_TABLE = "ModelBenchmarkJobs"
_PARTITION_KEY = "BENCHMARK_JOBS"  # one flat partition — same low-volume reasoning as PeftJobStatus


def start_job(*, model_source: str, triggered_by: str) -> str:
    """`model_source` — "ours" (a newly admin-added provider, or the
    daily cron sweeping everything) or "candidate" (a "his architecture"
    benchmark run). Returns a job_id the caller reports back to
    finish_job() once the harness+aggregate+write-to-db chain exits."""
    job_id = uuid.uuid4().hex[:12]
    row = {
        "job_id": job_id,
        "model_source": model_source,
        "triggered_by": triggered_by,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "summary": None,
    }
    db_store(_TABLE).put(_PARTITION_KEY, f"JOB#{job_id}", row)
    return job_id


def finish_job(job_id: str, *, ok: bool, summary: str) -> None:
    row = db_store(_TABLE).get(_PARTITION_KEY, f"JOB#{job_id}")
    if not row:
        return  # a status report for a job we never saw start — never crash the caller over this
    row["status"] = "done" if ok else "failed"
    row["finished_at"] = datetime.now(timezone.utc).isoformat()
    row["summary"] = summary[:500]
    db_store(_TABLE).put(_PARTITION_KEY, f"JOB#{job_id}", row)


def _strip_storage_keys(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("PK", "SK")}


def list_jobs(limit: int = 50) -> list[dict]:
    rows = db_store(_TABLE).scan(_PARTITION_KEY, "JOB#")
    ordered = sorted((_strip_storage_keys(r) for r in rows), key=lambda r: r["started_at"], reverse=True)
    return ordered[:limit]


def running_jobs() -> list[dict]:
    return [j for j in list_jobs(limit=200) if j["status"] == "running"]
