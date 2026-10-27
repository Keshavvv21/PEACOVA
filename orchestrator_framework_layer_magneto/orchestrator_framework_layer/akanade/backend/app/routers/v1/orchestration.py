"""§14's headline endpoint:

    POST /api/v1/orchestration/execute
    {"task": "...", "context": {}, "constraints": {"max_cost": 1.0, "max_latency": 30}}
    → {"task_id", "selected_agents", "execution_graph", "result", "metrics", "confidence"}

This is the SAME orchestrator (gmaofn.orchestrator.orchestrator) that
routes to PMAgent, OpenClawAgent, CodingAgent, or any agent registered at
runtime — a caller here never names a framework/agent, only describes
the task; Algo aKANADE decides who answers.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...auth import current_user
from ...framework.orchestrator import NoAgentsAvailableError, orchestrator
from ...framework.task import Task, TaskContext, TaskType

router = APIRouter(prefix="/api/v1/orchestration", tags=["gmaofn-orchestration"])

# TEMPORARY diagnostic logging — see execute()/tool_result()'s own log
# lines below. Added to settle, with real evidence instead of guessing,
# whether a live VS Code session that reports no model/tier badge is a
# server-side gap (this endpoint genuinely sending provider=None for a
# real final answer) or a client-side one (the extension not rendering
# data the server did send). Remove once that's confirmed either way.
logger = logging.getLogger("orchestration.diag")


class ContextIn(BaseModel):
    task_type: str = "generic"
    complexity: float = 0.5
    required_capabilities: list[str] = Field(default_factory=list)
    context_size: int = 0
    requires_rag: bool = False
    knowledge_sources: list[str] = Field(default_factory=list)


class ConstraintsIn(BaseModel):
    max_cost: float = 1.0
    max_latency: float = 30.0


class ExecuteRequest(BaseModel):
    task: str
    session_id: str = "default"
    context: ContextIn = Field(default_factory=ContextIn)
    constraints: ConstraintsIn = Field(default_factory=ConstraintsIn)
    preferred_provider: str | None = None
    # Opt-in "local autonomy" mode (see coding_agent.py's module
    # docstring, "CLIENT TOOLS MODE"): when true, CodingAgent never
    # touches its own server-side sandbox — every tool call it wants to
    # make comes back as `result.pending_tool_call` instead, for the
    # caller (a VS Code extension, today — anything implementing the
    # same protocol tomorrow) to execute for real against the user's
    # own machine and report back via POST .../tool_result. Ignored by
    # every other agent (PM tool, etc.) — CodingAgent is the only one
    # that reads task.metadata["client_tools"] at all.
    client_tools: bool = False
    # Real, observed gap this closes: this endpoint has never carried any
    # prior-turn context at all — every call built exactly [system, this
    # ONE user message], so a second message in the same chat ("now edit
    # line 3 of it") had zero idea what "it" even referred to. session_id
    # only ever gated pending-write/pending-tool-call bookkeeping
    # (_PENDING_WRITES/_PENDING_TOOL_CALLS), never actual conversation
    # memory. The caller (VS Code extension) already keeps its own
    # visible chat log for display — this just lets it hand that SAME
    # log back as real model context. Ignored by every agent except
    # CodingAgent today (see coding_agent.py's plan()); each entry is
    # {"role": "user"|"assistant", "content": "..."}.
    history: list[dict] | None = None


class ToolResultRequest(BaseModel):
    """Resumes a turn CodingAgent paused via `pending_tool_call` (see
    ExecuteRequest.client_tools) once the caller has actually executed
    that tool against the user's real machine."""

    session_id: str
    call_id: str
    output: str = ""
    error: str | None = None


@router.post("/execute")
def execute(body: ExecuteRequest, user: dict = Depends(current_user)):
    try:
        task_type = TaskType(body.context.task_type)
    except ValueError:
        task_type = TaskType.GENERIC
    context = TaskContext(
        task_type=task_type, complexity=body.context.complexity,
        required_capabilities=frozenset(body.context.required_capabilities),
        context_size=body.context.context_size,
        latency_requirement_s=body.constraints.max_latency,
        budget_usd=body.constraints.max_cost,
        requires_rag=body.context.requires_rag,
        knowledge_sources=tuple(body.context.knowledge_sources),
    )
    task = Task(text=body.task, context=context, session_id=body.session_id,
               metadata={"user": user, "preferred_provider": body.preferred_provider,
                        "client_tools": body.client_tools, "history": body.history or []})
    try:
        result = orchestrator.execute(task)
    except NoAgentsAvailableError as e:
        from fastapi import HTTPException
        raise HTTPException(503, str(e))
    payload = result.to_dict()
    payload["trace"] = result.trace.to_dict()
    logger.warning(
        "DIAG execute session=%r client_tools=%r selected_agents=%r ok=%r "
        "pending_tool_call=%r provider=%r tier=%r",
        body.session_id, body.client_tools, payload.get("selected_agents"),
        payload["result"]["ok"], payload["result"]["pending_tool_call"] is not None,
        payload["result"].get("provider"), payload["result"].get("tier"))
    return payload


@router.post("/tool_result")
def tool_result(body: ToolResultRequest, user: dict = Depends(current_user)):
    """The other half of client_tools mode. Bypasses the orchestrator's
    own routing entirely — there's no "which agent should handle this"
    decision to make here, only "continue the specific in-flight
    CodingAgent turn this call_id belongs to" — so this calls
    CodingAgent.resume_tool_call() directly via the registry, then runs
    the SAME validate()/respond() tail every agent's run() would, to
    produce a normally-shaped AgentResult."""
    from fastapi import HTTPException

    from ...framework.registry import AgentNotFoundError

    try:
        agent = orchestrator.registry.get("coding-agent")
    except AgentNotFoundError:
        raise HTTPException(503, "coding-agent is not registered")
    task = Task(text="", session_id=body.session_id,
               metadata={"user": user, "client_tools": True})
    try:
        raw = agent.resume_tool_call(task, body.call_id, body.output, body.error)  # type: ignore[attr-defined]
    except AttributeError:
        raise HTTPException(500, "coding-agent does not support client tools resume")
    validated = agent.validate(task, raw)
    result = agent.respond(task, validated)
    payload = {
        "task_id": task.task_id,
        "selected_agents": ["coding-agent"],
        "result": {
            "text": result.text,
            "ok": result.ok,
            "pending_confirmation": result.pending_confirmation,
            "confirm_prompt": result.confirm_prompt,
            "pending_tool_call": result.raw.get("pending_tool_call"),
            # Real gap this closes: unlike /execute's OrchestrationResult.
            # to_dict(), this hand-built response dropped every one of
            # these — a client-tools turn (the common VS Code path, since
            # every tool call routes back through THIS endpoint via
            # /tool_result) never learned which model actually answered
            # once a tool call was involved, even though respond() always
            # computes it. Same field names/semantics as to_dict() so a
            # caller (api.ts) can treat both endpoints' replies uniformly.
            "provider": result.raw.get("provider"),
            "tier": result.raw.get("tier"),
            "tierLog": result.raw.get("tier_log"),
        },
        "confidence": round(result.confidence, 4),
    }
    logger.warning(
        "DIAG tool_result session=%r call_id=%r ok=%r pending_tool_call=%r "
        "provider=%r tier=%r",
        body.session_id, body.call_id, payload["result"]["ok"],
        payload["result"]["pending_tool_call"] is not None,
        payload["result"].get("provider"), payload["result"].get("tier"))
    return payload
