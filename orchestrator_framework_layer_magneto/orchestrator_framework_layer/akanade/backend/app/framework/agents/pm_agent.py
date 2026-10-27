"""PMAgent — the pre-existing PM chatbot (agent/native_agent.py,
langgraph_agent.py, adk_agent.py, claude_agent.py, tools.py, ...),
wrapped as ONE ALGO AKANADE agent. Zero behavior change: every prompt in
CHAT_PROMPTS.md, every test in tests/test_agent_flows.py, still goes
through the exact same `get_agent(framework).handle_message(...)` call
this file simply delegates to — proving ALGO AKANADE is additive, not a
rewrite, of the domain logic it wraps.

`understand`/`plan`/`validate` are near-identity here because the PM
chatbot's OWN internal pipeline (native_agent.handle_message's 17 numbered
steps) already does perceive/resolve/confirm/execute/format in one call
— this wrapper's real job is just translating ALGO AKANADE's generic
Task/AgentResult shape at the boundary, not reimplementing anything.
"""
from __future__ import annotations

from ...agent.adapter import get_agent
from ...agent.state import AgentReply
from ..agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from ..task import Task, TaskType

_DEMO_USER = {"sub": "gmaofn-demo-user", "role": "Admin"}


class PMAgent(Agent):
    def __init__(self, framework: str | None = None, *,
                agent_id: str = "pm-agent", name: str = "PM Agent",
                description: str | None = None):
        self.framework = framework  # None = this deployment's configured default
        self.metadata = AgentMetadata(
            agent_id=agent_id,
            name=name,
            description=description or
                "Project-management assistant — create/query projects, tasks, "
                "sprints, spaces, users, and more via natural language. Wraps the "
                "pre-existing Algo PM chatbot (native/langgraph/adk/claude tracks).",
            capabilities=frozenset({"project_management", "task_management", "reporting",
                                    "semantic_search"}),
            supported_tasks=frozenset({TaskType.PROJECT_MANAGEMENT, TaskType.GENERIC}),
            model="configurable (native/langgraph/adk/claude — see llm/router.py)",
            tools=("create_task", "query_tasks", "graph_query", "semantic_search", "..."),
            rag_enabled=True,
            cost_per_call_usd=0.002,
            latency_profile=LatencyProfile(p50_ms=600, p95_ms=2500),
            version="2.0.0",
            reliability_score=0.85,
            status=AgentStatus.ACTIVE,
        )

    def _user_and_session(self, task: Task) -> tuple[dict, str]:
        user = task.metadata.get("user", _DEMO_USER)
        return user, task.session_id

    def understand(self, task: Task) -> dict:
        return {"task": task}

    def plan(self, task: Task, understanding: dict) -> dict:
        return {"task": task}

    def execute(self, task: Task, plan: dict) -> dict:
        user, session_id = self._user_and_session(task)
        agent_handle = get_agent(self.framework)
        reply: AgentReply = agent_handle.handle_message(
            user, session_id, task.text,
            preferred_provider=task.metadata.get("preferred_provider"))
        return {"reply": reply, "framework": agent_handle.name}

    def validate(self, task: Task, result: dict) -> dict:
        return result  # native_agent's own output/input guards already ran

    def respond(self, task: Task, validated: dict) -> AgentResult:
        reply: AgentReply = validated["reply"]
        confidence = 0.4 if reply.pending_confirmation else 0.85
        return AgentResult(
            text=reply.reply, ok=True, confidence=confidence,
            pending_confirmation=reply.pending_confirmation,
            confirm_prompt=reply.confirm_prompt,
            raw={"framework": validated["framework"], "tier_log": reply.tier_log},
        )
