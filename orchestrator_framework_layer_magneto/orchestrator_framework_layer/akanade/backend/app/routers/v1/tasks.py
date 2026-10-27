"""§14's `/api/v1/tasks` and `/api/v1/evaluations` — looking back at a
task ALGO AKANADE already orchestrated: what it was evaluated as, and what
reward each contributing agent was attributed (before/after any human
feedback — see routers/v1/feedback.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...auth import current_user
from ...framework.orchestrator import orchestrator

router = APIRouter(prefix="/api/v1", tags=["gmaofn-tasks"])


@router.get("/tasks/{task_id}")
def get_task(task_id: str, _: dict = Depends(current_user)):
    record = orchestrator.feedback_loop.get_record(task_id)
    if record is None:
        raise HTTPException(404, f"No task record for {task_id!r}")
    return {"task_id": record.task_id,
           "quality_by_agent": record.quality_by_agent,
           "outcome": {"quality": record.outcome.quality, "success": record.outcome.success,
                      "cost_usd": record.outcome.cost_usd, "latency_s": record.outcome.latency_s,
                      "failed": record.outcome.failed, "user_feedback": record.outcome.user_feedback},
           "attributed_rewards": record.attributed_rewards}


@router.get("/evaluations/{task_id}")
def get_evaluation(task_id: str, user: dict = Depends(current_user)):
    """Alias of GET /api/v1/tasks/{task_id} under the name §14 lists this
    concept by — "evaluation" (the outcome ALGO AKANADE scored a task's
    execution as) and "task" (the record of that execution) are the same
    stored object here, not two different stores."""
    return get_task(task_id, user)
