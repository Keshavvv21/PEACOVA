"""ALGO AKANADE's generic REST API surface — /api/v1/* — additive to the PM
domain's own existing routers (routers/pm.py, routers/chat.py,
routers/acp.py), never replacing them. See app/gmaofn/README.md."""
from __future__ import annotations

from . import agents, cost, feedback, metrics, models, orchestration, rag, tasks, tools, workflows

ALL_ROUTERS = [agents.router, orchestration.router, feedback.router, models.router,
              tools.router, rag.router, metrics.router, tasks.router, workflows.router,
              cost.router]
