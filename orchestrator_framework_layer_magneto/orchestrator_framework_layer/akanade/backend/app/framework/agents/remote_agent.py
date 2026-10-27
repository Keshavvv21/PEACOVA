"""RemoteHttpAgent — what `POST /api/v1/agents/register` actually
registers when the caller isn't a Python module already imported into
this process: a thin proxy that POSTs the task to `webhook_url` and
expects `{"text": str, "ok": bool, "confidence": float}` back. This is
what makes "register a new agent via API" genuinely mean something for
an external service/team, not just "call registry.register() from
Python" — a contributor can stand up any HTTP endpoint, in any language,
and register it into ALGO AKANADE's routing/workflow/feedback machinery without
touching this repo at all.
"""
from __future__ import annotations

import httpx

from ..agent_base import Agent, AgentMetadata, AgentResult
from ..task import Task


class RemoteHttpAgent(Agent):
    def __init__(self, metadata: AgentMetadata, webhook_url: str, *, timeout: float = 30.0):
        self.metadata = metadata
        self.webhook_url = webhook_url
        self.timeout = timeout

    def understand(self, task: Task) -> dict:
        return {"task": task}

    def plan(self, task: Task, understanding: dict) -> dict:
        return {"task": task}

    def execute(self, task: Task, plan: dict) -> dict:
        payload = {"task_id": task.task_id, "text": task.text, "session_id": task.session_id,
                  "metadata": {k: v for k, v in task.metadata.items() if k != "user"}}
        response = httpx.post(self.webhook_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def validate(self, task: Task, result: dict) -> dict:
        return result

    def respond(self, task: Task, validated: dict) -> AgentResult:
        return AgentResult(
            text=validated.get("text", ""), ok=bool(validated.get("ok", True)),
            confidence=float(validated.get("confidence", 0.6)), raw=validated)
