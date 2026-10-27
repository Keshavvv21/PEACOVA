"""§6 — the Agent Registry. In-memory, process-lifetime (same "the right
shape for a reference deployment" tradeoff agent/framework_stats.py's own
docstring makes) — swap `_agents`'s dict for a real table behind the same
four methods to persist across restarts; nothing above this module needs
to change.

A new agent is "installable without modifying the orchestration engine"
(§6) by construction: `register()` takes an already-built `Agent`
instance, and everything downstream (router.py, orchestrator.py) only
ever iterates `list_agents()`/looks up `get()` — neither has an
if/elif naming a specific agent_id anywhere.
"""
from __future__ import annotations

import threading

from .agent_base import Agent, AgentStatus


class AgentAlreadyRegisteredError(Exception):
    pass


class AgentNotFoundError(Exception):
    pass


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = threading.Lock()

    def register(self, agent: Agent, *, replace: bool = False) -> None:
        agent_id = agent.metadata.agent_id
        with self._lock:
            if agent_id in self._agents and not replace:
                raise AgentAlreadyRegisteredError(agent_id)
            self._agents[agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        with self._lock:
            if agent_id not in self._agents:
                raise AgentNotFoundError(agent_id)
            del self._agents[agent_id]

    def get(self, agent_id: str) -> Agent:
        with self._lock:
            agent = self._agents.get(agent_id)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        return agent

    def list_agents(self, *, include_disabled: bool = False) -> list[Agent]:
        with self._lock:
            agents = list(self._agents.values())
        if include_disabled:
            return agents
        return [a for a in agents if a.metadata.status != AgentStatus.DISABLED]

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        self.get(agent_id).metadata.status = status

    def __contains__(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._agents

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)


# Process-wide singleton — every router/orchestrator/API route shares this
# one registry, same lifecycle as agent/framework_stats.py's module-level
# _stats dict elsewhere in this codebase.
registry = AgentRegistry()
