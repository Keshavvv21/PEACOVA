"""§12/§14's `/api/v1/workflows` — the execution-mode primitives
orchestrator.py composes dynamically per task (see workflow.py), exposed
read-only: ALGO AKANADE doesn't store named workflow templates — the shape is
DERIVED from each task's own capability requirements every time (that's
the "dynamically change this workflow depending on task requirements"
property working, not a missing feature) — this endpoint documents which
primitives exist for a caller building their OWN explicit workflow
directly against the gmaofn.workflow module."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...auth import current_user
from ...framework import workflow as workflow_module

router = APIRouter(prefix="/api/v1/workflows", tags=["gmaofn-workflows"])

_STEP_TYPES = [workflow_module.SequentialStep, workflow_module.ParallelStep,
              workflow_module.ConsensusStep, workflow_module.FallbackStep,
              workflow_module.RetryStep, workflow_module.HierarchicalStep]


@router.get("")
def list_workflow_step_kinds(_: dict = Depends(current_user)):
    return {"step_kinds": [
        {"kind": cls.kind, "class": cls.__name__,
         "description": (cls.__doc__ or "").strip().split("\n")[0]}
        for cls in _STEP_TYPES
    ]}
