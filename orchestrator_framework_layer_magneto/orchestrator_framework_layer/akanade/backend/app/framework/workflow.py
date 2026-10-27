"""§1's execution modes — sequential, parallel, hierarchical, consensus/
voting, fallback, retry — as small composable step types the orchestrator
strings into a `Workflow`, rather than one hard-coded control-flow shape.

Every step type takes the SAME two inputs (a Task and the AgentRegistry)
and returns a WorkflowStepResult, so orchestrator.py can execute an
arbitrary list of these without caring which kind each one is — the
"dynamically change this workflow depending on task requirements" (§12)
requirement is a property of composing these primitives at orchestration
time, not a separate code path per shape.
"""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .agent_base import Agent, AgentResult
from .registry import AgentRegistry
from .task import Task


@dataclass
class StepExecution:
    """One agent's actual run inside a step — kept even for agents that
    failed/were skipped, so the orchestration trace (observability.py)
    shows the full picture, not just whoever "won"."""

    agent_id: str
    result: AgentResult | None
    latency_s: float
    error: str | None = None


@dataclass
class WorkflowStepResult:
    step_kind: str
    executions: list[StepExecution]
    final: AgentResult
    winning_agent_id: str


def _run_one(agent: Agent, task: Task) -> StepExecution:
    t0 = time.time()
    try:
        result = agent.run(task)
        return StepExecution(agent_id=agent.metadata.agent_id, result=result,
                             latency_s=time.time() - t0)
    except Exception as e:  # a single agent's failure never crashes the workflow
        return StepExecution(agent_id=agent.metadata.agent_id, result=None,
                             latency_s=time.time() - t0, error=str(e))


class SequentialStep:
    """Run agents one after another, each seeing the PREVIOUS agent's
    reply appended to the task text (a simple, explicit hand-off — a
    Planner's output literally becomes part of what the Coding agent
    reads) — §12's Planner → Research → Coding → Testing → Reviewer
    pipeline. The step's `final` is the LAST agent's response; every
    intermediate agent's execution is still recorded."""

    kind = "sequential"

    def __init__(self, agents: list[Agent]):
        if not agents:
            raise ValueError("SequentialStep needs at least one agent")
        self.agents = agents

    def run(self, task: Task) -> WorkflowStepResult:
        executions: list[StepExecution] = []
        current_task = task
        last_ok: AgentResult | None = None
        for agent in self.agents:
            execution = _run_one(agent, current_task)
            executions.append(execution)
            if execution.result is not None:
                last_ok = execution.result
                current_task = Task(
                    text=f"{current_task.text}\n\n[{agent.metadata.name} said]: {execution.result.text}",
                    context=current_task.context, task_id=current_task.task_id,
                    session_id=current_task.session_id, metadata=current_task.metadata,
                    constraints=current_task.constraints)
        final = last_ok or AgentResult(text="Every agent in the sequence failed.", ok=False, confidence=0.0)
        return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                  final=final, winning_agent_id=executions[-1].agent_id)


class ParallelStep:
    """Run every agent independently on the SAME task, concurrently — for
    "get N independent takes", feeding into either ConsensusStep (below)
    or a caller-supplied `select` picking the best one by confidence."""

    kind = "parallel"

    def __init__(self, agents: list[Agent], *, max_workers: int = 8):
        if not agents:
            raise ValueError("ParallelStep needs at least one agent")
        self.agents = agents
        self.max_workers = max_workers

    def run(self, task: Task) -> WorkflowStepResult:
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.agents))) as pool:
            executions = list(pool.map(lambda a: _run_one(a, task), self.agents))
        ok = [e for e in executions if e.result is not None]
        winner = max(ok, key=lambda e: e.result.confidence) if ok else None
        final = winner.result if winner else AgentResult(text="Every agent failed.", ok=False, confidence=0.0)
        return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                  final=final,
                                  winning_agent_id=winner.agent_id if winner else executions[0].agent_id)


class ConsensusStep(ParallelStep):
    """§1's "agent voting/consensus": run every agent in parallel, then
    pick the reply that the MOST agents effectively agree with — approximated
    here by exact text match after normalization (a real deployment would
    cluster semantically-similar replies; exact-match consensus is the
    honest, no-extra-model-call baseline). Ties fall back to ParallelStep's
    highest-confidence rule."""

    kind = "consensus"

    def run(self, task: Task) -> WorkflowStepResult:
        base = super().run(task)
        ok = [e for e in base.executions if e.result is not None]
        if len(ok) < 2:
            return WorkflowStepResult(step_kind=self.kind, executions=base.executions,
                                      final=base.final, winning_agent_id=base.winning_agent_id)
        votes = Counter(e.result.text.strip().lower() for e in ok)
        top_text, top_count = votes.most_common(1)[0]
        if top_count > 1:
            winner = next(e for e in ok if e.result.text.strip().lower() == top_text)
            return WorkflowStepResult(step_kind=self.kind, executions=base.executions,
                                      final=winner.result, winning_agent_id=winner.agent_id)
        return WorkflowStepResult(step_kind=self.kind, executions=base.executions,
                                  final=base.final, winning_agent_id=base.winning_agent_id)


class FallbackStep:
    """Try agents in priority order; the first one that succeeds wins.
    Every earlier failure is still recorded in the trace — this is the
    agent-level analogue of llm/router.py's TierClient chain-walk in the
    PM domain, generalized to any registered agent."""

    kind = "fallback"

    def __init__(self, agents_in_priority_order: list[Agent]):
        if not agents_in_priority_order:
            raise ValueError("FallbackStep needs at least one agent")
        self.agents = agents_in_priority_order

    def run(self, task: Task) -> WorkflowStepResult:
        executions: list[StepExecution] = []
        for agent in self.agents:
            execution = _run_one(agent, task)
            executions.append(execution)
            if execution.result is not None and execution.result.ok:
                return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                          final=execution.result, winning_agent_id=execution.agent_id)
        final = AgentResult(text="Every fallback agent failed.", ok=False, confidence=0.0)
        return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                  final=final, winning_agent_id=executions[-1].agent_id)


class RetryStep:
    """Retry ONE agent up to `max_attempts` times (e.g. for a
    transiently-flaky tool call) before giving up — distinct from
    FallbackStep, which moves to a DIFFERENT agent instead of retrying
    the same one."""

    kind = "retry"

    def __init__(self, agent: Agent, *, max_attempts: int = 3):
        self.agent = agent
        self.max_attempts = max_attempts

    def run(self, task: Task) -> WorkflowStepResult:
        executions: list[StepExecution] = []
        for _ in range(self.max_attempts):
            execution = _run_one(self.agent, task)
            executions.append(execution)
            if execution.result is not None and execution.result.ok:
                return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                          final=execution.result, winning_agent_id=execution.agent_id)
        final = AgentResult(text=f"{self.agent.metadata.name} failed after "
                                 f"{self.max_attempts} attempts.", ok=False, confidence=0.0)
        return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                  final=final, winning_agent_id=executions[-1].agent_id)


class HierarchicalStep:
    """A lead agent delegates to worker agents, then synthesizes their
    replies itself — §1's "agent delegation": the lead's own `plan()` isn't
    reused here (workers are chosen by the caller, not inferred by the
    lead), but the lead's `execute()`/`respond()` gets the workers' output
    appended to its task text, same hand-off shape SequentialStep uses,
    so a HierarchicalStep is really "ParallelStep, then feed the results
    to one more agent" — kept as its own named type since that's a
    distinct, common shape (§12's "Planner → Multiple Specialists →
    Validator")."""

    kind = "hierarchical"

    def __init__(self, lead: Agent, workers: list[Agent]):
        self.lead = lead
        self.workers = workers

    def run(self, task: Task) -> WorkflowStepResult:
        worker_step = ParallelStep(self.workers).run(task) if self.workers else None
        executions = list(worker_step.executions) if worker_step else []
        worker_summary = "\n".join(
            f"- {e.agent_id}: {e.result.text}" for e in executions if e.result is not None)
        lead_task = Task(text=f"{task.text}\n\n[Worker outputs]\n{worker_summary}",
                         context=task.context, task_id=task.task_id,
                         session_id=task.session_id, metadata=task.metadata,
                         constraints=task.constraints)
        lead_execution = _run_one(self.lead, lead_task)
        executions.append(lead_execution)
        final = lead_execution.result or AgentResult(text="Lead agent failed to synthesize.",
                                                      ok=False, confidence=0.0)
        return WorkflowStepResult(step_kind=self.kind, executions=executions,
                                  final=final, winning_agent_id=self.lead.metadata.agent_id)


WorkflowStep = (SequentialStep | ParallelStep | ConsensusStep | FallbackStep
               | RetryStep | HierarchicalStep)


@dataclass
class Workflow:
    """An ordered list of steps — each step's `final` text is threaded
    into the NEXT step's task, same hand-off SequentialStep uses
    internally, so a Workflow of [ParallelStep, HierarchicalStep,
    RetryStep] composes exactly like §12's dynamically-varying pipelines."""

    steps: list[WorkflowStep]
    name: str = "workflow"

    def run(self, task: Task, registry: AgentRegistry | None = None) -> list[WorkflowStepResult]:
        results: list[WorkflowStepResult] = []
        current_task = task
        for step in self.steps:
            result = step.run(current_task)
            results.append(result)
            current_task = Task(
                text=f"{current_task.text}\n\n[{step.kind} step result]: {result.final.text}",
                context=current_task.context, task_id=current_task.task_id,
                session_id=current_task.session_id, metadata=current_task.metadata,
                constraints=current_task.constraints)
        return results
