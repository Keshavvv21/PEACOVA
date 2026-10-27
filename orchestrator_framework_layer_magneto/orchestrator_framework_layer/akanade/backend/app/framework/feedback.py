"""§5 — the closed loop:

    Task → Agent Selection → Agent Execution → Result → Evaluation
    → Feedback → Reward Calculation → Policy Update → Improved Future Routing

`FeedbackLoop.close()` runs this immediately after every orchestrated
task using self/validator-evaluated quality as the feedback signal (so
routing keeps improving even when no human ever rates anything — the
same "the router must still work with zero external signal" philosophy
the rest of this codebase applies to LLM keys). `submit_user_feedback()`
is the explicit human-in-the-loop path (§14's `/api/v1/feedback`):
when a real rating arrives later for a task that's already been closed
once, it recomputes that task's reward with the human's score standing in
for `user_feedback` and applies a SECOND router update — real feedback
outranks the agents' own self-assessment without discarding it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..background import run_in_background
from . import metrics_history, shapley
from .reward import AdaptiveWeights, ExecutionOutcome, compute_reward
from .router import AdaptiveRouter
from .task import TaskContext


@dataclass
class TaskRecord:
    task_id: str
    context: TaskContext
    quality_by_agent: dict[str, float]
    outcome: ExecutionOutcome
    attributed_rewards: dict[str, float]
    # The task's raw text — kept so submit_user_feedback() can also
    # re-remember it in TaskMemory (router.py's ContextualThompsonRouter),
    # not just re-run the numeric-feature bandit update.
    task_text: str = ""


class FeedbackLoop:
    def __init__(self, router: AdaptiveRouter, weights: AdaptiveWeights | None = None):
        self.router = router
        self.weights = weights or AdaptiveWeights()
        self._records: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def close(self, task_id: str, context: TaskContext, quality_by_agent: dict[str, float],
              outcome: ExecutionOutcome, task_text: str = "") -> dict[str, float]:
        """Evaluation → Feedback → Reward → Policy Update, in one call.
        Returns each participating agent's attributed reward (for the
        orchestration trace). `task_text`, when given, also feeds
        router.py's TaskMemory — the framework's continuous-learning
        embedding memory, not just the numeric-feature bandit update."""
        total_reward = compute_reward(outcome, self.weights.weights,
                                      cost_budget_usd=context.budget_usd,
                                      latency_budget_s=context.latency_requirement_s)
        attributed = shapley.attribute_reward(quality_by_agent, total_reward)
        for agent_id, agent_reward in attributed.items():
            self.router.update(agent_id, context, agent_reward, task_text)
        self.weights.update(outcome, total_reward)
        with self._lock:
            self._records[task_id] = TaskRecord(task_id=task_id, context=context,
                                                quality_by_agent=quality_by_agent,
                                                outcome=outcome, attributed_rewards=attributed,
                                                task_text=task_text)
        # The one real "an outcome was just scored" event — see
        # metrics_history.py's own docstring for why this, and not a
        # periodic poll, is the correct place to log a snapshot. Deferred:
        # `record()` ends in a real store().put() plus a full-table
        # scan()-based prune() for overflow (see that module), and
        # `attributed` (already computed above) is everything the caller
        # of close() actually needs back.
        run_in_background(metrics_history.record, task_id, total_reward,
                          self.weights.weights.snapshot(), attributed)
        return attributed

    def submit_user_feedback(self, task_id: str, feedback_score: float) -> dict[str, float]:
        """A human's 0..1 rating for a task already closed once — recomputes
        that task's reward with THIS as `user_feedback` (overriding the
        self-evaluated proxy compute_reward would otherwise fall back to)
        and applies a second, human-grounded router update."""
        with self._lock:
            record = self._records.get(task_id)
        if record is None:
            raise KeyError(f"No task record for {task_id!r} — was it ever closed via close()?")
        feedback_score = max(0.0, min(1.0, feedback_score))
        outcome = ExecutionOutcome(quality=record.outcome.quality, success=record.outcome.success,
                                   cost_usd=record.outcome.cost_usd, latency_s=record.outcome.latency_s,
                                   failed=record.outcome.failed, user_feedback=feedback_score)
        total_reward = compute_reward(outcome, self.weights.weights,
                                      cost_budget_usd=record.context.budget_usd,
                                      latency_budget_s=record.context.latency_requirement_s)
        attributed = shapley.attribute_reward(record.quality_by_agent, total_reward)
        for agent_id, agent_reward in attributed.items():
            self.router.update(agent_id, record.context, agent_reward, record.task_text)
        self.weights.update(outcome, total_reward)
        with self._lock:
            record.outcome = outcome
            record.attributed_rewards = attributed
        metrics_history.record(task_id, total_reward, self.weights.weights.snapshot(), attributed)
        return attributed

    def get_record(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._records.get(task_id)
