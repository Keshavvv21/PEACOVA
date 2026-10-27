"""ALGO AKANADE core — the generic multi-agent orchestration framework layered
on top of the PM domain (see tests/test_claude_mcp_acp.py for the PM
domain's own agent-framework tests, and tests/test_gmaofn_api.py for the
REST surface). This file tests the framework's OWN math and mechanics
with small, fully deterministic fake agents — never the real PM
chatbot — so these tests say nothing about PM behavior specifically
(that's what the rest of this suite already covers) and everything about
whether ALGO AKANADE's routing/workflow/feedback machinery is correct in
general.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.framework import shapley
from app.framework.agent_base import Agent, AgentMetadata, AgentResult, AgentStatus
from app.framework.feedback import FeedbackLoop
from app.framework.network import AgentNetwork
from app.framework.orchestrator import NoAgentsAvailableError, Orchestrator
from app.framework.registry import AgentAlreadyRegisteredError, AgentNotFoundError, AgentRegistry
from app.framework.reward import AdaptiveWeights, ExecutionOutcome, RewardWeights, compute_reward
from app.framework.router import AdaptiveRouter, utility_prior
from app.framework.task import Task, TaskContext, TaskType
from app.framework.workflow import (ConsensusStep, FallbackStep, HierarchicalStep,
                                 ParallelStep, RetryStep, SequentialStep, Workflow)


class FakeAgent(Agent):
    """A fully deterministic Agent for testing the framework's mechanics
    without depending on any real LLM/tool call."""

    def __init__(self, agent_id: str, *, reply: str = "ok", ok: bool = True,
                confidence: float = 0.8, capabilities: frozenset[str] = frozenset(),
                fail_first_n: int = 0):
        self.metadata = AgentMetadata(agent_id=agent_id, name=agent_id, description="test agent",
                                      capabilities=capabilities, status=AgentStatus.ACTIVE)
        self.reply, self.ok, self.confidence = reply, ok, confidence
        self._calls = 0
        self._fail_first_n = fail_first_n

    def understand(self, task):
        return {}

    def plan(self, task, understanding):
        return {}

    def execute(self, task, plan):
        self._calls += 1
        if self._calls <= self._fail_first_n:
            raise RuntimeError(f"{self.metadata.agent_id} transient failure #{self._calls}")
        return {}

    def validate(self, task, result):
        return result

    def respond(self, task, validated) -> AgentResult:
        return AgentResult(text=self.reply, ok=self.ok, confidence=self.confidence)


def make_task(**kwargs) -> Task:
    return Task(text="do something", context=TaskContext(**kwargs))


# ── Shapley value (shapley.py) — the Nobel-Prize-linked credit assignment ──

def test_shapley_single_agent_gets_everything():
    shares = shapley.shapley_values({"a": 0.9})
    assert shares["a"] == pytest.approx(0.9)


def test_shapley_efficiency_axiom():
    """Sum of Shapley values == v(N) (the grand coalition's value) —
    every unit of value is attributed to SOMEONE, none lost or invented."""
    qualities = {"a": 0.6, "b": 0.4, "c": 0.8}
    shares = shapley.shapley_values(qualities)
    v_n = shapley._noisy_or(list(qualities.values()))
    assert sum(shares.values()) == pytest.approx(v_n, abs=1e-9)


def test_shapley_symmetry_axiom():
    """Two agents with IDENTICAL quality get IDENTICAL Shapley value —
    the characteristic function only depends on quality, not identity."""
    shares = shapley.shapley_values({"a": 0.5, "b": 0.5, "c": 0.9})
    assert shares["a"] == pytest.approx(shares["b"], abs=1e-9)


def test_shapley_higher_quality_gets_more_credit():
    shares = shapley.shapley_values({"a": 0.9, "b": 0.1})
    assert shares["a"] > shares["b"]


def test_shapley_monte_carlo_matches_exact_reasonably_for_larger_n():
    qualities = {f"agent{i}": 0.3 + 0.05 * i for i in range(9)}  # 9 > _EXACT_LIMIT (8)
    mc = shapley.shapley_values(qualities, seed=42)
    assert set(mc.keys()) == set(qualities.keys())
    assert sum(mc.values()) == pytest.approx(shapley._noisy_or(list(qualities.values())), abs=0.05)


def test_attribute_reward_splits_total_reward_proportionally():
    attributed = shapley.attribute_reward({"a": 0.9, "b": 0.1}, total_reward=0.5)
    assert sum(attributed.values()) == pytest.approx(0.5, abs=1e-6)
    assert attributed["a"] > attributed["b"]


# ── Reward function (reward.py) ──────────────────────────────────────────

def test_reward_increases_with_quality():
    weights = RewardWeights()
    low = compute_reward(ExecutionOutcome(quality=0.1, success=True, cost_usd=0.01, latency_s=1), weights)
    high = compute_reward(ExecutionOutcome(quality=0.9, success=True, cost_usd=0.01, latency_s=1), weights)
    assert high > low


def test_reward_decreases_with_cost_and_latency():
    weights = RewardWeights()
    cheap = compute_reward(ExecutionOutcome(quality=0.5, success=True, cost_usd=0.01, latency_s=1), weights,
                           cost_budget_usd=1.0)
    expensive = compute_reward(ExecutionOutcome(quality=0.5, success=True, cost_usd=0.99, latency_s=1), weights,
                               cost_budget_usd=1.0)
    assert cheap > expensive


def test_reward_failure_is_penalized():
    weights = RewardWeights()
    ok = compute_reward(ExecutionOutcome(quality=0.5, success=True, cost_usd=0.1, latency_s=1, failed=False), weights)
    failed = compute_reward(ExecutionOutcome(quality=0.5, success=False, cost_usd=0.1, latency_s=1, failed=True), weights)
    assert ok > failed


def test_adaptive_weights_stay_normalized_after_updates():
    adaptive = AdaptiveWeights()
    outcome = ExecutionOutcome(quality=0.9, success=True, cost_usd=0.01, latency_s=1)
    for _ in range(50):
        reward = compute_reward(outcome, adaptive.weights)
        adaptive.update(outcome, reward)
    total = sum(adaptive.weights.snapshot().values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert all(v > 0 for v in adaptive.weights.snapshot().values())


# ── Adaptive router (router.py) — Thompson Sampling contextual bandit ───

def test_utility_prior_rewards_capability_match():
    good = AgentMetadata(agent_id="good", name="good", description="",
                         capabilities=frozenset({"coding"}), reliability_score=0.5)
    bad = AgentMetadata(agent_id="bad", name="bad", description="",
                        capabilities=frozenset(), reliability_score=0.5)
    context = TaskContext(required_capabilities=frozenset({"coding"}))
    assert utility_prior(good, context) > utility_prior(bad, context)


def test_router_learns_to_prefer_the_higher_reward_agent():
    """The actual RL claim: after enough (context, reward) feedback,
    Thompson Sampling should pick the consistently-better agent
    substantially more often than the consistently-worse one — not
    100% of the time (that would mean no exploration at all, which is
    itself a red flag for a bandit), but clearly more than half."""
    router = AdaptiveRouter(feature_dim=6, ridge=1.0, exploration_scale=0.2,
                            rng=np.random.default_rng(7))
    good = AgentMetadata(agent_id="good", name="good", description="", reliability_score=0.5)
    bad = AgentMetadata(agent_id="bad", name="bad", description="", reliability_score=0.5)
    context = TaskContext()

    for _ in range(60):
        router.update("good", context, 0.9)
        router.update("bad", context, -0.4)

    picks = [router.select([good, bad], context).agent_id for _ in range(200)]
    good_rate = picks.count("good") / len(picks)
    assert good_rate > 0.8, f"expected the router to strongly prefer 'good', got {good_rate:.2f}"


def test_router_score_all_returns_every_candidate():
    router = AdaptiveRouter(feature_dim=6)
    a = AgentMetadata(agent_id="a", name="a", description="")
    b = AgentMetadata(agent_id="b", name="b", description="")
    scores = router.score_all([a, b], TaskContext())
    assert set(scores.keys()) == {"a", "b"}


def test_router_task_memory_biases_a_brand_new_context_toward_the_past_winner():
    """The embedding-memory gap this session closed: two agents with
    IDENTICAL cold-start priors and zero bandit history — nothing in
    utility_prior or the bandit posterior can tell them apart yet — but
    TaskMemory remembers "coder" won on requests like this before, so a
    near-duplicate request should now favor it even before any bandit
    data of its own exists."""
    router = AdaptiveRouter(feature_dim=6, ridge=1.0, exploration_scale=0.05,
                            rng=np.random.default_rng(3))
    coder = AgentMetadata(agent_id="coder", name="coder", description="", reliability_score=0.5)
    writer = AgentMetadata(agent_id="writer", name="writer", description="", reliability_score=0.5)
    context = TaskContext()

    router.update("coder", context, 0.9, task_text="fix the failing unit tests in the payments module")
    router.update("writer", context, -0.3, task_text="fix the failing unit tests in the payments module")

    picks = [router.select([coder, writer], context,
                          task_text="fix the broken unit tests in the billing module").agent_id
            for _ in range(100)]
    assert picks.count("coder") > picks.count("writer")


def test_router_task_memory_has_no_effect_on_an_unrelated_request():
    """A memory match requires real similarity (task_memory.py's
    _MATCH_THRESHOLD) — an unrelated request must not inherit a bonus
    just because SOME past task exists in memory."""
    router = AdaptiveRouter(feature_dim=6)
    a = AgentMetadata(agent_id="a", name="a", description="", reliability_score=0.5)
    context = TaskContext()
    router.update("a", context, 1.0, task_text="fix the failing unit tests in the payments module")
    bonus = router._bandit.task_memory.bonus_for("what's the weather like on Mars", "a")
    assert bonus == 0.0


# ── Workflow engine (workflow.py) ────────────────────────────────────────

def test_sequential_step_chains_agents_and_hands_off_text():
    a = FakeAgent("a", reply="first")
    b = FakeAgent("b", reply="second")
    result = SequentialStep([a, b]).run(make_task())
    assert result.final.text == "second"
    assert len(result.executions) == 2


def test_parallel_step_picks_highest_confidence():
    a = FakeAgent("a", reply="low", confidence=0.3)
    b = FakeAgent("b", reply="high", confidence=0.9)
    result = ParallelStep([a, b]).run(make_task())
    assert result.final.text == "high"
    assert result.winning_agent_id == "b"


def test_fallback_step_tries_in_order_and_stops_at_first_success():
    failing = FakeAgent("failing", ok=False)
    working = FakeAgent("working", reply="rescued")
    result = FallbackStep([failing, working]).run(make_task())
    assert result.final.text == "rescued"
    assert len(result.executions) == 2


def test_retry_step_retries_the_same_agent_and_eventually_succeeds():
    flaky = FakeAgent("flaky", reply="finally", fail_first_n=2)
    result = RetryStep(flaky, max_attempts=3).run(make_task())
    assert result.final.text == "finally"
    assert len(result.executions) == 3


def test_retry_step_gives_up_after_max_attempts():
    always_fails = FakeAgent("always_fails", fail_first_n=99)
    result = RetryStep(always_fails, max_attempts=2).run(make_task())
    assert result.final.ok is False
    assert len(result.executions) == 2


def test_consensus_step_picks_the_majority_reply():
    a = FakeAgent("a", reply="Paris", confidence=0.5)
    b = FakeAgent("b", reply="Paris", confidence=0.5)
    c = FakeAgent("c", reply="London", confidence=0.9)
    result = ConsensusStep([a, b, c]).run(make_task())
    assert result.final.text == "Paris"  # 2 votes beats 1, even though c had higher confidence


def test_hierarchical_step_synthesizes_worker_output():
    worker = FakeAgent("worker", reply="worker output")
    lead = FakeAgent("lead", reply="synthesized")
    result = HierarchicalStep(lead, [worker]).run(make_task())
    assert result.final.text == "synthesized"
    assert len(result.executions) == 2  # worker + lead


def test_workflow_chains_multiple_steps():
    step1 = SequentialStep([FakeAgent("a", reply="step one done")])
    step2 = SequentialStep([FakeAgent("b", reply="step two done")])
    results = Workflow([step1, step2]).run(make_task())
    assert len(results) == 2
    assert results[-1].final.text == "step two done"


# ── Agent registry (registry.py) ─────────────────────────────────────────

def test_registry_register_get_list_unregister():
    reg = AgentRegistry()
    agent = FakeAgent("x")
    reg.register(agent)
    assert "x" in reg
    assert reg.get("x") is agent
    assert len(reg.list_agents()) == 1
    reg.unregister("x")
    assert "x" not in reg


def test_registry_duplicate_registration_raises():
    reg = AgentRegistry()
    reg.register(FakeAgent("dup"))
    with pytest.raises(AgentAlreadyRegisteredError):
        reg.register(FakeAgent("dup"))


def test_registry_missing_agent_raises():
    reg = AgentRegistry()
    with pytest.raises(AgentNotFoundError):
        reg.get("nope")


# ── Agent network (network.py) ───────────────────────────────────────────

def test_network_capability_neighbors_finds_covering_agents():
    net = AgentNetwork()
    a = AgentMetadata(agent_id="a", name="a", description="", capabilities=frozenset())
    b = AgentMetadata(agent_id="b", name="b", description="", capabilities=frozenset({"coding"}))
    net.add_agent(a)
    net.add_agent(b)
    covering = net.capability_neighbors("a", "coding", [a, b])
    assert [m.agent_id for m in covering] == ["b"]


def test_network_record_handoff_and_most_frequent_next():
    net = AgentNetwork()
    net.add_agent(AgentMetadata(agent_id="a", name="a", description=""))
    net.add_agent(AgentMetadata(agent_id="b", name="b", description=""))
    for _ in range(3):
        net.record_handoff("a", "b")
    top = net.most_frequent_next("a")
    assert top[0][0] == "b"
    assert top[0][1] == 3


def test_network_dependency_chain():
    net = AgentNetwork()
    for agent_id in ("planner", "coder", "reviewer"):
        net.add_agent(AgentMetadata(agent_id=agent_id, name=agent_id, description=""))
    net.declare_dependency("planner", "coder")
    net.declare_dependency("coder", "reviewer")
    assert net.dependency_chain_from("planner") == ["planner", "coder", "reviewer"]


# ── Feedback loop (feedback.py) — the closed loop ────────────────────────

def test_feedback_loop_close_updates_router_and_records_task():
    router = AdaptiveRouter(feature_dim=6)
    loop = FeedbackLoop(router)
    context = TaskContext()
    outcome = ExecutionOutcome(quality=0.8, success=True, cost_usd=0.01, latency_s=1)
    attributed = loop.close("task-1", context, {"agent-a": 0.8}, outcome)
    assert "agent-a" in attributed
    record = loop.get_record("task-1")
    assert record is not None
    assert record.outcome.quality == 0.8


def test_feedback_loop_user_feedback_overrides_and_re_updates():
    router = AdaptiveRouter(feature_dim=6)
    loop = FeedbackLoop(router)
    context = TaskContext()
    outcome = ExecutionOutcome(quality=0.5, success=True, cost_usd=0.01, latency_s=1)
    loop.close("task-2", context, {"agent-a": 0.5}, outcome)
    updated = loop.submit_user_feedback("task-2", 1.0)  # a glowing human rating
    record = loop.get_record("task-2")
    assert record.outcome.user_feedback == 1.0
    assert updated["agent-a"] != pytest.approx(0.0)


def test_feedback_loop_unknown_task_raises():
    loop = FeedbackLoop(AdaptiveRouter(feature_dim=6))
    with pytest.raises(KeyError):
        loop.submit_user_feedback("no-such-task", 0.5)


# ── Orchestrator (orchestrator.py) — the end-to-end call ─────────────────

def test_orchestrator_no_agents_raises():
    from app.framework.registry import AgentRegistry as FreshRegistry
    orch = Orchestrator(registry=FreshRegistry())
    with pytest.raises(NoAgentsAvailableError):
        orch.execute(make_task())


def test_orchestrator_single_capable_agent_runs_directly():
    from app.framework.network import AgentNetwork as FreshNetwork
    from app.framework.registry import AgentRegistry as FreshRegistry

    reg = FreshRegistry()
    reg.register(FakeAgent("solo", reply="handled it", capabilities=frozenset({"x"})))
    orch = Orchestrator(registry=reg, router=AdaptiveRouter(feature_dim=6), network=FreshNetwork())
    result = orch.execute(make_task(required_capabilities=frozenset({"x"})))
    assert result.selected_agents == ["solo"]
    assert result.result.text == "handled it"
    assert result.metrics["step_kind"] == "sequential"
    assert "solo" in result.metrics["reward_attributed"]


def test_orchestrator_delegates_to_workers_when_lead_only_partially_qualifies():
    """The task needs BOTH "a" and "b". "lead" only has "a" and "worker"
    only has "b" — neither alone fully qualifies, so the capability
    pre-filter (which only requires SOME overlap, not full coverage)
    keeps both candidates in play; once the router picks "lead", its
    still-missing "b" triggers hierarchical delegation to "worker"."""
    from app.framework.network import AgentNetwork as FreshNetwork
    from app.framework.registry import AgentRegistry as FreshRegistry

    reg = FreshRegistry()
    lead = FakeAgent("lead", reply="synthesized", capabilities=frozenset({"a"}))
    worker = FakeAgent("worker", reply="worker did it", capabilities=frozenset({"b"}))
    reg.register(lead)
    reg.register(worker)
    router = AdaptiveRouter(feature_dim=6)
    context = TaskContext(required_capabilities=frozenset({"a", "b"}))
    for _ in range(20):
        router.update("lead", context, 0.9)
        router.update("worker", context, -0.9)
    orch = Orchestrator(registry=reg, router=router, network=FreshNetwork())
    result = orch.execute(Task(text="do something", context=context))
    assert result.metrics["step_kind"] == "hierarchical"
    assert set(result.selected_agents) == {"lead", "worker"}
    assert result.result.text == "synthesized"
