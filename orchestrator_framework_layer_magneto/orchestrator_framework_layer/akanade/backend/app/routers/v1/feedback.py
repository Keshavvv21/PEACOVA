"""§5/§14 — closing the loop from the outside: a human (or an evaluator
service) rates a task ALGO AKANADE already answered, and that rating becomes a
second, human-grounded reward applied to every agent that contributed to
it (via Shapley-attributed credit, same as the automatic self-evaluated
close() that already ran right after execution — see
gmaofn/feedback.py's module docstring)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...auth import current_user
from ...framework.orchestrator import orchestrator

router = APIRouter(prefix="/api/v1/feedback", tags=["gmaofn-feedback"])


class FeedbackRequest(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="0 (bad) .. 1 (great)")
    comment: str | None = None


@router.post("")
def submit_feedback(body: FeedbackRequest, _: dict = Depends(current_user)):
    try:
        attributed = orchestrator.feedback_loop.submit_user_feedback(body.task_id, body.score)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"task_id": body.task_id, "attributed_rewards": attributed,
           "reward_weights": orchestrator.feedback_loop.weights.weights.snapshot()}
