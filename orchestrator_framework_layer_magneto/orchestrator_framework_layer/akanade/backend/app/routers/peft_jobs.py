"""Live PEFT job status — the REST half of the "background job
execution with a status bar" feature. `deploy/nightly_finetune.sh`
reports into `/start` and `/finish` (using the same admin token it
already authenticates every other nightly call with — see
fetch_configs()'s FINETUNE_ADMIN_EMAIL/PASSWORD); any dashboard or
status-bar client polls `GET /api/admin/peft-jobs`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..auth import require_role
from ..framework import peft_job_status

router = APIRouter(prefix="/api/admin/peft-jobs", tags=["peft-jobs"])


class StartJobRequest(BaseModel):
    domain: str
    mode: str  # "cpu" or "gpu"
    slot: int = 0


class FinishJobRequest(BaseModel):
    ok: bool
    summary: str = ""


@router.get("")
def list_jobs(_: dict = Depends(require_role("Admin"))):
    jobs = peft_job_status.list_jobs()
    return {
        "jobs": jobs,
        "running": [j for j in jobs if j["status"] == "running"],
    }


@router.post("/start")
def start_job(payload: StartJobRequest, _: dict = Depends(require_role("Admin"))):
    job_id = peft_job_status.start_job(domain=payload.domain, mode=payload.mode, slot=payload.slot)
    return {"job_id": job_id}


@router.post("/{job_id}/finish")
def finish_job(job_id: str, payload: FinishJobRequest, _: dict = Depends(require_role("Admin"))):
    peft_job_status.finish_job(job_id, ok=payload.ok, summary=payload.summary)
    return {"ok": True}
