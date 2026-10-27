"""ACP (Agent Communication Protocol) — the chatbot as a *peer* agent other
agents/apps can talk to "sideways", complementing the MCP client
integration (agent/mcp_client.py, which lets THIS chatbot consume tools
FROM other systems). Where MCP connects an agent downward to tools/data,
ACP standardizes how a client application — another agent, an orchestrator,
a multi-agent network — talks to THIS agent as one node in that network.

A minimal, honest subset of the real spec (agentcommunicationprotocol.dev /
the Linux Foundation BeeAI project's ACP): agent discovery (GET /acp/agents)
and a synchronous run (POST /acp/runs), using the spec's own manifest/run
message shapes (role + content-typed parts) rather than an invented ad-hoc
format — a genuine ACP client can point at this app's /acp/* base URL and
get back wire-compatible responses, even though only the sync-run subset
(no async/streaming runs, no multi-agent manifests) is implemented.

Auth: same JWT current_user dependency every other endpoint in this app
uses — the calling agent authenticates as an already-known Algo Kaizen
user, and the run executes exactly as if that user had typed the message
into the chat widget (same handle_message() every track already exposes,
same role/visibility rules, nothing new granted). ACP itself doesn't
mandate a specific auth scheme; this is this app's own reasonable choice,
not something the spec requires.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..agent.adapter import get_agent
from ..auth import current_user
from ..config import settings

router = APIRouter(prefix="/acp", tags=["acp"])

AGENT_NAME = "algo-kaizen-pm"


class AcpMessagePart(BaseModel):
    content_type: str = "text/plain"
    content: str


class AcpMessage(BaseModel):
    role: str = "user"
    parts: list[AcpMessagePart]

    def text(self) -> str:
        return "\n".join(p.content for p in self.parts if p.content_type == "text/plain")


class AcpRunRequest(BaseModel):
    agent_name: str = AGENT_NAME
    session_id: str = "default"
    input: list[AcpMessage]
    # An explicit framework choice for this one run (native/langgraph/adk/
    # claude) — omit to use this deployment's configured default, same
    # ?framework= semantics as POST /api/chat/message.
    framework: str | None = None


class AcpRunResult(BaseModel):
    run_id: str
    agent_name: str
    session_id: str
    status: str
    output: list[AcpMessage]


@router.get("/agents")
def list_agents():
    """ACP agent discovery — this deployment exposes exactly one agent:
    whichever chat framework is configured, presented uniformly regardless
    of which track (native/langgraph/adk/claude) actually answers a run."""
    return {"agents": [{
        "name": AGENT_NAME,
        "description": "Algo Kaizen's PM assistant — create/query projects, "
                       "tasks, sprints, and more via natural language.",
        "input_content_types": ["text/plain"],
        "output_content_types": ["text/plain"],
        "metadata": {"default_framework": settings().CHAT_AGENT_FRAMEWORK,
                     "available_frameworks": ["native", "langgraph", "adk", "claude"]},
    }]}


@router.post("/runs", response_model=AcpRunResult)
def create_run(body: AcpRunRequest, user: dict = Depends(current_user)):
    """A synchronous ACP run: the input message(s) are joined into one chat
    turn (this deployment's agents are single-turn-per-request, same as the
    chat widget) and answered through the normal get_agent(...).handle_message
    path — a peer agent calling this sees the exact same behavior a human
    typing into the widget would, just wrapped in ACP's own message shape."""
    if body.agent_name != AGENT_NAME:
        raise HTTPException(404, f"Unknown agent {body.agent_name!r}")
    if not body.input:
        raise HTTPException(400, "input must contain at least one message")
    user_msg = "\n".join(m.text() for m in body.input if m.role == "user") or body.input[-1].text()

    agent = get_agent(body.framework)
    reply = agent.handle_message(user, body.session_id, user_msg)

    return AcpRunResult(
        run_id=str(uuid.uuid4()),
        agent_name=AGENT_NAME,
        session_id=body.session_id,
        status="completed",
        output=[AcpMessage(role="agent", parts=[
            AcpMessagePart(content_type="text/plain", content=reply.reply)])],
    )
