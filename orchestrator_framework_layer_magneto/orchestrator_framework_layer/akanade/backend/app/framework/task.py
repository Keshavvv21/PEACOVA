"""The unit of work ALGO AKANADE routes and executes: a Task, plus the context
vector the adaptive router scores candidate agents against.

Deliberately domain-agnostic — nothing here mentions projects, tasks (the
PM kind), code, or documents. A "task" in ALGO AKANADE is any natural-language
request an application hands to the orchestrator.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    """A coarse task-type label — one axis of the context vector (§3's
    `x = [task_type, complexity, ...]`). Domains are free to only ever use
    GENERIC plus their own convention in `Task.metadata["domain_task_type"]`
    for anything finer-grained; this enum stays small on purpose."""

    GENERIC = "generic"
    PROJECT_MANAGEMENT = "project_management"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    QUESTION_ANSWERING = "question_answering"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENTATION = "documentation"


@dataclass
class TaskContext:
    """The feature vector U(a|T)'s Q/C terms and the router's contextual
    bandit are conditioned on — §3's `x = [task_type, complexity,
    required_capabilities, context_size, latency_requirement, budget,
    historical_context, ...]`, made concrete.

    `as_vector()` is the actual numeric encoding the router's linear
    utility model reads — kept separate from the human-readable fields
    above so a new feature can be added to the vector without every
    call site that only cares about e.g. `required_capabilities`
    needing to change.
    """

    task_type: TaskType = TaskType.GENERIC
    complexity: float = 0.5          # 0 (trivial) .. 1 (very hard)
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    context_size: int = 0            # tokens/characters of context attached
    latency_requirement_s: float = 30.0   # caller's max acceptable latency
    budget_usd: float = 1.0               # caller's max acceptable cost
    requires_rag: bool = False
    knowledge_sources: tuple[str, ...] = ()

    def as_vector(self) -> list[float]:
        """A fixed-length numeric encoding for the linear utility model
        (router.py's ContextualThompsonRouter) — complexity and three
        normalized/clamped scalars, plus the capability-count and RAG
        flag. Kept simple and interpretable on purpose: this is a
        transparent linear model, not a black box, so an operator can
        read off which feature drove a routing decision."""
        return [
            self.complexity,
            min(self.context_size / 8000.0, 1.0),
            min(self.latency_requirement_s / 60.0, 1.0),
            min(self.budget_usd / 5.0, 1.0),
            1.0 if self.requires_rag else 0.0,
            min(len(self.required_capabilities) / 5.0, 1.0),
        ]


@dataclass
class Task:
    """One unit of work handed to ALGO AKANADE's orchestrator — the `task`
    field of `POST /api/v1/orchestration/execute`'s request body."""

    text: str
    context: TaskContext = field(default_factory=TaskContext)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = "default"
    metadata: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
