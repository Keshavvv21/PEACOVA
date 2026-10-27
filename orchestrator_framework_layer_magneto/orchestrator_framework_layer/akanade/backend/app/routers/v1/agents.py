"""§6/§14 — the Agent Registry's REST surface:

    POST   /api/v1/agents/register
    GET    /api/v1/agents
    GET    /api/v1/agents/{agent_id}
    PUT    /api/v1/agents/{agent_id}
    DELETE /api/v1/agents/{agent_id}

`register` accepts a `webhook_url` and builds a RemoteHttpAgent proxy —
the path for registering an agent that ISN'T Python already imported
into this process (see agents/remote_agent.py's docstring). A native
Python agent (PMAgent, CodingAgent, a new domain's own Agent subclass)
registers itself via `gmaofn.registry.registry.register(MyAgent())`
directly — no REST round-trip needed, same "installable without
modifying the orchestration engine" property either way.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...auth import require_role
from ...framework.agent_base import AgentMetadata, AgentStatus, LatencyProfile
from ...framework.agents.remote_agent import RemoteHttpAgent
from ...framework.network import network
from ...framework.registry import AgentAlreadyRegisteredError, AgentNotFoundError, registry

router = APIRouter(prefix="/api/v1/agents", tags=["gmaofn-agents"])


class AgentRegisterRequest(BaseModel):
    agent_id: str
    name: str
    description: str
    webhook_url: str = Field(..., description="Where ALGO AKANADE POSTs {task_id, text, session_id, "
                                              "metadata} and expects back {text, ok, confidence}.")
    capabilities: list[str] = Field(default_factory=list)
    model: str = ""
    cost_per_call_usd: float = 0.0
    latency_p50_ms: float = 500.0
    latency_p95_ms: float = 2000.0
    reliability_score: float = 0.6


class AgentOut(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    model: str
    tools: list[str]
    rag_enabled: bool
    cost_per_call_usd: float
    version: str
    reliability_score: float
    status: str


def _to_out(metadata: AgentMetadata) -> AgentOut:
    return AgentOut(agent_id=metadata.agent_id, name=metadata.name, description=metadata.description,
                    capabilities=sorted(metadata.capabilities), model=metadata.model,
                    tools=list(metadata.tools), rag_enabled=metadata.rag_enabled,
                    cost_per_call_usd=metadata.cost_per_call_usd, version=metadata.version,
                    reliability_score=metadata.reliability_score, status=metadata.status.value)


@router.post("/register", response_model=AgentOut, status_code=201)
def register_agent(body: AgentRegisterRequest, _: dict = Depends(require_role("Admin"))):
    metadata = AgentMetadata(
        agent_id=body.agent_id, name=body.name, description=body.description,
        capabilities=frozenset(body.capabilities), model=body.model,
        cost_per_call_usd=body.cost_per_call_usd,
        latency_profile=LatencyProfile(p50_ms=body.latency_p50_ms, p95_ms=body.latency_p95_ms),
        reliability_score=body.reliability_score, status=AgentStatus.ACTIVE)
    agent = RemoteHttpAgent(metadata, body.webhook_url)
    try:
        registry.register(agent)
    except AgentAlreadyRegisteredError:
        raise HTTPException(409, f"Agent {body.agent_id!r} is already registered")
    network.add_agent(metadata)
    return _to_out(metadata)


@router.get("", response_model=list[AgentOut])
def list_agents(include_disabled: bool = False, _: dict = Depends(require_role("Member"))):
    return [_to_out(a.metadata) for a in registry.list_agents(include_disabled=include_disabled)]


@router.get("/{agent_id}", response_model=AgentOut)
def get_agent(agent_id: str, _: dict = Depends(require_role("Member"))):
    try:
        agent = registry.get(agent_id)
    except AgentNotFoundError:
        raise HTTPException(404, f"No agent {agent_id!r}")
    return _to_out(agent.metadata)


class AgentUpdateRequest(BaseModel):
    status: str | None = None  # "active" | "degraded" | "disabled"


@router.put("/{agent_id}", response_model=AgentOut)
def update_agent(agent_id: str, body: AgentUpdateRequest, _: dict = Depends(require_role("Admin"))):
    try:
        agent = registry.get(agent_id)
    except AgentNotFoundError:
        raise HTTPException(404, f"No agent {agent_id!r}")
    if body.status is not None:
        try:
            agent.metadata.status = AgentStatus(body.status)
        except ValueError:
            raise HTTPException(422, f"Invalid status {body.status!r}")
    return _to_out(agent.metadata)


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: str, _: dict = Depends(require_role("Admin"))):
    try:
        registry.unregister(agent_id)
    except AgentNotFoundError:
        raise HTTPException(404, f"No agent {agent_id!r}")
    network.remove_agent(agent_id)
