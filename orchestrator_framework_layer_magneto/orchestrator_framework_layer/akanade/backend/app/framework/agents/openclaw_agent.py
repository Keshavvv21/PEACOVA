"""OpenClawAgent — a general-purpose ALGO AKANADE agent built directly on the
public Anthropic Claude API (llm/providers.py's AnthropicNative,
registered as the "claude" provider in llm/router.py), with NO PM tools
and no domain-specific logic — proof that ALGO AKANADE's model abstraction
layer is reusable by a plain, tool-free agent just as easily as by the
PM domain's tool-calling one.

"Open" because it talks to Claude's genuinely public API (no special
access), and gracefully degrades — like every other rung in this
codebase — through llm/router.py's normal provider chain if
CLAUDE_API_KEY isn't configured, rather than hard-failing.
"""
from __future__ import annotations

from ...llm.router import get_llm_client
from ..agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from ..task import Task, TaskType

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful, concise general-purpose assistant. Answer directly; "
              "you have no tools available, so never claim to have taken an action.",
}


class OpenClawAgent(Agent):
    """A tool-free, model-only agent — plan()/validate() are near
    identity since there's nothing to resolve or confirm; the only real
    work happens in execute()'s single model call."""

    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="openclaw-agent",
            name="OpenClaw Agent",
            description="General-purpose reasoning/Q&A agent backed directly by the "
                       "public Anthropic Claude API — no PM tools, no domain logic. "
                       "Falls back through the normal provider chain if CLAUDE_API_KEY "
                       "is unset, same as every other rung in this codebase.",
            capabilities=frozenset({"general_reasoning", "question_answering", "summarization"}),
            supported_tasks=frozenset({TaskType.GENERIC, TaskType.QUESTION_ANSWERING,
                                       TaskType.RESEARCH}),
            model="claude-sonnet-5 (falls back through the normal chain if unconfigured)",
            tools=(),
            rag_enabled=False,
            cost_per_call_usd=0.003,
            latency_profile=LatencyProfile(p50_ms=800, p95_ms=3000),
            version="1.0.0",
            reliability_score=0.75,
            status=AgentStatus.ACTIVE,
        )

    def understand(self, task: Task) -> dict:
        return {"text": task.text}

    def plan(self, task: Task, understanding: dict) -> dict:
        return {"messages": [SYSTEM_PROMPT, {"role": "user", "content": understanding["text"]}]}

    def execute(self, task: Task, plan: dict) -> dict:
        client = get_llm_client("realtime", preferred="claude")
        result = client.chat("openclaw_respond", plan["messages"])
        return {"result": result, "tier_log": client.log}

    def validate(self, task: Task, result: dict) -> dict:
        return result

    def respond(self, task: Task, validated: dict) -> AgentResult:
        result = validated["result"]
        ok = not bool(result.error)
        text = result.text if ok else f"Sorry — every model rung failed: {result.error}"
        return AgentResult(text=text, ok=ok, confidence=0.8 if ok else 0.0,
                           raw={"provider": result.provider, "tier_log": validated["tier_log"]})
