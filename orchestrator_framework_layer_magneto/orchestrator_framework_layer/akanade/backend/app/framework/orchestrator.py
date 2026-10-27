"""The Orchestrator — ties registry + router + workflow + network +
feedback + observability into the one call every application actually
makes: `orchestrator.execute(task)`, the engine behind
`POST /api/v1/orchestration/execute`.

Workflow shape is chosen dynamically from the task's OWN declared
requirements (§12), not hard-coded: if one agent's capabilities already
cover everything the task needs, it runs alone (the common "simple
question" case); if not, the router's top pick becomes a lead that
delegates to whichever OTHER registered agents cover the missing
capabilities (network.capability_neighbors) and synthesizes their
output — §12's "Planner → Multiple Specialists → Validator" shape,
arrived at generically rather than special-cased per domain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agent_base import AgentResult
from .feedback import FeedbackLoop
from .network import AgentNetwork, network as default_network
from .observability import OrchestrationTrace
from .registry import AgentRegistry, registry as default_registry
from .reward import ExecutionOutcome
from .router import AdaptiveRouter, router as default_router
from .task import Task
from .workflow import HierarchicalStep, SequentialStep


class NoAgentsAvailableError(Exception):
    pass


@dataclass
class OrchestrationResult:
    task_id: str
    selected_agents: list[str]
    result: AgentResult
    confidence: float
    trace: OrchestrationTrace
    metrics: dict = field(default_factory=dict)
    execution_graph: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "selected_agents": self.selected_agents,
            "execution_graph": self.execution_graph,
            "result": {"text": self.result.text, "ok": self.result.ok,
                      "pending_confirmation": self.result.pending_confirmation,
                      "confirm_prompt": self.result.confirm_prompt,
                      # Present only for CodingAgent turns run in "client
                      # tools" mode (coding_agent.py) — {call_id, tool,
                      # args} the caller must execute itself (on the
                      # user's real machine) and report back to
                      # POST /api/v1/orchestration/tool_result. None for
                      # every other agent/turn, so existing callers that
                      # don't know this field exists see nothing new.
                      "pending_tool_call": self.result.raw.get("pending_tool_call"),
                      # The real underlying model that answered (e.g.
                      # "groq-llama-3.3-70b", "ollama-qwen2.5-coder:1.5b") —
                      # real gap found live: without this, the Coding
                      # Agent's own UI meta line had nothing to show but
                      # the agent's name ("coding-agent"), unlike PM/
                      # Trading's chat turns, which already surface the
                      # real provider via AgentReply.provider. None for
                      # any agent that doesn't set this in its own
                      # AgentResult.raw (e.g. a non-LLM agent).
                      "provider": self.result.raw.get("provider"),
                      # "default" or "escalation" — which of the Coding
                      # Agent's own tiers (llm/coding_pipeline.py)
                      # actually answered, admin-controlled via
                      # /api/admin/router/coding-escalation. None for any
                      # agent/turn that doesn't set this (e.g. a non-
                      # coding agent, or a cache hit/error).
                      "tier": self.result.raw.get("tier"),
                      # Every rung this turn actually attempted, in
                      # order, each with its own real outcome (ok/error/
                      # latency) — CodingAgent's own TierClient.log
                      # (see router.py's chat()). Added for the same
                      # reason `provider` above was: without it, the
                      # only visible signal for "why did the primary-
                      # pipeline cascade end up on groq" was the FINAL
                      # provider, hiding every real per-rung failure
                      # (a 402/429/insufficient-credit account issue
                      # looks identical to "never tried" without this).
                      # None for any agent that doesn't set it.
                      "tierLog": self.result.raw.get("tier_log")},
            "metrics": self.metrics,
            "confidence": round(self.confidence, 4),
        }


class Orchestrator:
    def __init__(self, *, registry: AgentRegistry | None = None,
                router: AdaptiveRouter | None = None,
                network: AgentNetwork | None = None,
                feedback_loop: FeedbackLoop | None = None):
        # Deliberately `is not None`, never a bare `x or default` — an
        # intentionally EMPTY AgentRegistry is falsy (it defines __len__),
        # so `registry or default_registry` would silently discard a
        # caller's genuinely-empty-on-purpose registry and substitute the
        # global singleton instead.
        self.registry = registry if registry is not None else default_registry
        self.router = router if router is not None else default_router
        self.network = network if network is not None else default_network
        self.feedback_loop = feedback_loop if feedback_loop is not None else FeedbackLoop(self.router)

    def execute(self, task: Task) -> OrchestrationResult:
        candidates = self.registry.list_agents()
        if not candidates:
            raise NoAgentsAvailableError("No agents registered with ALGO AKANADE")
        # Self-healing: ensure every candidate has a network node with real
        # metadata BEFORE any capability/communication edge involving it is
        # added below — independent of whether bootstrap.py (or a caller
        # building their own Orchestrator) remembered to call
        # network.add_agent() first. add_agent() is idempotent (NetworkX
        # add_node just refreshes the attrs), so this is always safe.
        for candidate in candidates:
            self.network.add_agent(candidate.metadata)

        # A hard capability pre-filter BEFORE the bandit scores anyone —
        # without this, an agent with no relevant capability at all can
        # still win on reliability/cost/latency alone (those terms are
        # only PART of utility_prior's sum), which both wastes a
        # hierarchical delegation hop and teaches the bandit the wrong
        # lesson about who's actually good at this task type. Falls back
        # to the full candidate list when literally nobody declares the
        # capability, so an under-described task still gets an answer via
        # the existing hierarchical-delegation path below rather than a
        # hard failure.
        required = task.context.required_capabilities
        if required:
            capable = [a for a in candidates if required & a.metadata.capabilities]
            if capable:
                candidates = capable

        trace = OrchestrationTrace(task_id=task.task_id)
        decision = self.router.select([a.metadata for a in candidates], task.context, task_text=task.text)
        trace.record_routing(decision)
        lead = self.registry.get(decision.agent_id)

        missing = task.context.required_capabilities - lead.metadata.capabilities
        t0 = time.time()
        if missing:
            workers = self.network.capability_neighbors(lead.metadata.agent_id, next(iter(missing)),
                                                         [a.metadata for a in candidates])
            worker_agents = [self.registry.get(m.agent_id) for m in workers]
            step_result = HierarchicalStep(lead, worker_agents).run(task)
            for e in step_result.executions[:-1]:
                self.network.record_handoff(e.agent_id, lead.metadata.agent_id)
        else:
            step_result = SequentialStep([lead]).run(task)
        latency_s = time.time() - t0
        trace.record_step(step_result)

        quality_by_agent = {}
        for execution in step_result.executions:
            if execution.result is None:
                continue
            agent = self.registry.get(execution.agent_id)
            quality_by_agent[execution.agent_id] = agent.evaluate(task, execution.result)
        if not quality_by_agent:
            quality_by_agent = {lead.metadata.agent_id: 0.0}

        total_cost = sum(self.registry.get(a).metadata.cost_per_call_usd for a in quality_by_agent)
        outcome = ExecutionOutcome(
            quality=quality_by_agent.get(step_result.winning_agent_id, 0.0),
            success=step_result.final.ok, cost_usd=total_cost, latency_s=latency_s,
            failed=not step_result.final.ok and any(e.error for e in step_result.executions))
        attributed = self.feedback_loop.close(task.task_id, task.context, quality_by_agent, outcome,
                                              task_text=task.text)
        for agent_id, agent_reward in attributed.items():
            trace.record_reward(agent_id, agent_reward, self.feedback_loop.weights.weights.snapshot())

        return OrchestrationResult(
            task_id=task.task_id,
            selected_agents=list(quality_by_agent.keys()),
            result=step_result.final,
            confidence=step_result.final.confidence,
            trace=trace,
            metrics={"latency_s": round(latency_s, 4), "cost_usd": round(total_cost, 4),
                    "reward_attributed": {k: round(v, 4) for k, v in attributed.items()},
                    "step_kind": step_result.step_kind},
            execution_graph=self.network.to_dict(),
        )


# Process-wide singleton — the object app/routers/orchestration.py's
# endpoints actually call.
orchestrator = Orchestrator()
