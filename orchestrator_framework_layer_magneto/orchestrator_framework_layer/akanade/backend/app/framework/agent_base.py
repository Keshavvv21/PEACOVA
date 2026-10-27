"""The one interface every ALGO AKANADE agent implements, and the metadata every
agent publishes to the registry (registry.py) so the router can score it
without knowing anything about its internals.

A concrete agent (agents/pm_agent.py, agents/coding_agent.py,
agents/openclaw_agent.py, ...) is free to implement understand/plan/
execute/validate/respond however it wants internally — wrap an existing
system (PMAgent wraps the whole pre-existing PM chatbot verbatim), call a
model directly (OpenClawAgent), or run a fixed pipeline (CodingAgent). The
orchestrator only ever calls these six methods and reads `metadata`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from .task import Task, TaskType


class AgentStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"       # available but recently unreliable
    DISABLED = "disabled"       # registered but excluded from routing


@dataclass
class LatencyProfile:
    p50_ms: float = 500.0
    p95_ms: float = 2000.0


@dataclass
class AgentMetadata:
    """§6's Agent Registry entry — every field the spec lists, plus the
    reliability_score the router's prior is seeded from before any real
    feedback exists for this agent."""

    agent_id: str
    name: str
    description: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    supported_tasks: frozenset[TaskType] = field(default_factory=lambda: frozenset({TaskType.GENERIC}))
    model: str = ""
    tools: tuple[str, ...] = ()
    rag_enabled: bool = False
    cost_per_call_usd: float = 0.0
    latency_profile: LatencyProfile = field(default_factory=LatencyProfile)
    version: str = "1.0.0"
    reliability_score: float = 0.7   # prior belief, before any observed feedback
    status: AgentStatus = AgentStatus.ACTIVE

    def matches_capabilities(self, required: frozenset[str]) -> bool:
        return required.issubset(self.capabilities)


@dataclass
class AgentResult:
    """What every agent's `respond()` returns — the orchestrator wraps
    this with routing/reward metadata (see orchestrator.py) before it
    reaches the caller."""

    text: str
    ok: bool = True
    confidence: float = 0.7
    pending_confirmation: bool = False
    confirm_prompt: str | None = None
    tokens_used: int = 0
    tool_calls: tuple[str, ...] = ()
    raw: dict = field(default_factory=dict)


class Agent(ABC):
    """§7's generic Agent interface. `evaluate()` has a sane default
    (confidence * ok, i.e. "how much do I trust my own answer") so a
    simple agent isn't forced to implement self-critique — override it
    for anything that can meaningfully judge its own output (e.g.
    CodingAgent's syntax-check-based validate/evaluate)."""

    metadata: AgentMetadata

    @abstractmethod
    def understand(self, task: Task) -> dict:
        """Parse/interpret the task into whatever internal representation
        this agent needs (an intent, a plan spec, ...). Returned dict is
        passed to plan()."""

    @abstractmethod
    def plan(self, task: Task, understanding: dict) -> dict:
        """Decide the concrete steps/tool calls this agent will take.
        Returned dict is passed to execute()."""

    @abstractmethod
    def execute(self, task: Task, plan: dict) -> dict:
        """Actually do the work (call a model, call a tool, mutate state).
        Returned dict is passed to validate()."""

    @abstractmethod
    def validate(self, task: Task, result: dict) -> dict:
        """Check the raw execution result for correctness/safety before
        it's phrased as a reply. Returned dict is passed to respond()."""

    @abstractmethod
    def respond(self, task: Task, validated: dict) -> AgentResult:
        """Render the final natural-language reply."""

    def evaluate(self, task: Task, result: AgentResult) -> float:
        """Self-assessed quality in [0, 1] — the Q(a,T) term of the
        utility function (see gmaofn/reward.py) when no external
        evaluator/user feedback is available yet for this call. Default:
        the agent's own confidence, zeroed out on failure."""
        return result.confidence if result.ok else 0.0

    def run(self, task: Task) -> AgentResult:
        """The fixed pipeline every agent shares — understand → plan →
        execute → validate → respond — so the orchestrator never has to
        know which of the six steps a given agent actually does
        interesting work in."""
        understanding = self.understand(task)
        plan = self.plan(task, understanding)
        raw = self.execute(task, plan)
        validated = self.validate(task, raw)
        return self.respond(task, validated)
