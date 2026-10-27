"""Live status for the nightly PEFT pipeline's actual training jobs —
the "background job execution with a live status bar" surface.
`deploy/nightly_finetune.sh` reports into this (via the two REST calls
below) right before and right after each domain's real `docker run
... scripts.finetune_local_slm` call, so a status bar/dashboard can
show what's genuinely happening RIGHT NOW, not just last night's
finished result (that's PeftMetricsDashboard.tsx's job — this is the
"is something training THIS SECOND" complement to it).

Deliberately "mini," per the ask — this is a status BOARD, not a job
QUEUE: nightly_finetune.sh still decides scheduling entirely on its own
(FINETUNE_MAX_PARALLEL — sequential on CPU, one job per GPU when GPUs
are detected, already real before this file existed). This module only
ever records what that scheduler already decided to do, and lets
anyone ask "is domain X training right now" before submitting more
data for it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..db import store as db_store

_TABLE = "PeftJobStatus"
_DATE_KEY = "PEFT_JOBS"  # one flat partition — job volume is low (a handful a night), no need to shard by day


def start_job(*, domain: str, mode: str, slot: int) -> str:
    """`mode` is "cpu" or "gpu" (nightly_finetune.sh already knows this —
    GPU_COUNT>0 for this slot — and passes it straight through, no
    guessing here). Returns a job_id the caller reports back to
    finish_job() when the training subprocess actually exits."""
    job_id = uuid.uuid4().hex[:12]
    row = {
        "job_id": job_id,
        "domain": domain or "general",
        "mode": mode,
        "slot": slot,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "summary": None,
    }
    db_store(_TABLE).put(_DATE_KEY, f"JOB#{job_id}", row)
    return job_id


def finish_job(job_id: str, *, ok: bool, summary: str) -> None:
    row = db_store(_TABLE).get(_DATE_KEY, f"JOB#{job_id}")
    if not row:
        return  # a status report for a job we never saw start — never crash the caller over this
    row["status"] = "done" if ok else "failed"
    row["finished_at"] = datetime.now(timezone.utc).isoformat()
    row["summary"] = summary[:500]
    db_store(_TABLE).put(_DATE_KEY, f"JOB#{job_id}", row)


def _strip_storage_keys(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("PK", "SK")}


def list_jobs(limit: int = 50) -> list[dict]:
    """Newest-started first — the status bar/dashboard's real data
    source. Includes running, done, and failed jobs together (the
    caller filters/labels by `status`, this just returns the recent
    history)."""
    rows = db_store(_TABLE).scan(_DATE_KEY, "JOB#")
    ordered = sorted((_strip_storage_keys(r) for r in rows), key=lambda r: r["started_at"], reverse=True)
    return ordered[:limit]


def running_jobs() -> list[dict]:
    return [j for j in list_jobs(limit=200) if j["status"] == "running"]


def is_domain_running(domain: str) -> bool:
    domain = domain or "general"
    return any(j["domain"] == domain for j in running_jobs())
