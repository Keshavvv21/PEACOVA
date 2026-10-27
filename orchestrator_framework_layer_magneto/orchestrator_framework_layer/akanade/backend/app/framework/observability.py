"""§15 — the orchestration trace: every routing decision, execution, and
score, in one JSON-serializable object per task, so "why did agent X
answer this" is always answerable after the fact — not just the final
reply.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .router import RoutingDecision
from .workflow import WorkflowStepResult


@dataclass
class TraceEvent:
    kind: str            # "routing_decision" | "step_result" | "reward" | "error"
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


@dataclass
class OrchestrationTrace:
    task_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def record_routing(self, decision: RoutingDecision) -> None:
        self.events.append(TraceEvent(kind="routing_decision", data={
            "selected_agent": decision.agent_id,
            "sampled_utility": round(decision.sampled_utility, 4),
            "posterior_mean_utility": round(decision.posterior_mean_utility, 4),
            "all_candidate_scores": {k: round(v, 4) for k, v in decision.all_candidate_scores.items()},
            "strategy": decision.strategy,
        }))

    def record_step(self, result: WorkflowStepResult) -> None:
        self.events.append(TraceEvent(kind="step_result", data={
            "step_kind": result.step_kind,
            "winning_agent": result.winning_agent_id,
            "executions": [
                {"agent_id": e.agent_id, "latency_s": round(e.latency_s, 4),
                 "ok": e.result.ok if e.result else False,
                 "confidence": round(e.result.confidence, 4) if e.result else 0.0,
                 "error": e.error}
                for e in result.executions
            ],
        }))

    def record_reward(self, agent_id: str, reward: float, weights_snapshot: dict) -> None:
        self.events.append(TraceEvent(kind="reward", data={
            "agent_id": agent_id, "reward": round(reward, 4),
            "reward_weights": {k: round(v, 4) for k, v in weights_snapshot.items()},
        }))

    def record_error(self, message: str) -> None:
        self.events.append(TraceEvent(kind="error", data={"message": message}))

    def to_dict(self) -> dict:
        return {"task_id": self.task_id,
                "events": [{"kind": e.kind, "timestamp": e.timestamp, "data": e.data}
                          for e in self.events]}
