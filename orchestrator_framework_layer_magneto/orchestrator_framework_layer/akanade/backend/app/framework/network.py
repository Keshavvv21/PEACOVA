"""§13 — Agent Network: agents as nodes in a graph, not merely a flat
list, connected by three distinct edge kinds so "which agent should be
called next" is a graph query, not a hard-coded pipeline:

  capability  — a → b when b can do something a's own capability set is
                missing, i.e. b is a legitimate next hop when a task
                needs a capability a doesn't have
  dependency  — a → b when a's output is typically consumed BY b (Coding
                → Testing → Reviewer), a declared production pipeline
  communication — a ↔ b when the two have actually exchanged
                messages/hand-offs during real executions (built up
                live, not declared up front — see record_handoff)

Built on `networkx` (already a dependency here — see retrieval/graph.py's
own use of it for the PM domain's knowledge graph), so this reuses a
real, well-tested graph library rather than hand-rolling adjacency lists.
"""
from __future__ import annotations

import networkx as nx

from .agent_base import AgentMetadata


class AgentNetwork:
    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def add_agent(self, metadata: AgentMetadata) -> None:
        self._graph.add_node(metadata.agent_id, metadata=metadata)

    def remove_agent(self, agent_id: str) -> None:
        if self._graph.has_node(agent_id):
            self._graph.remove_node(agent_id)

    def declare_dependency(self, upstream_id: str, downstream_id: str, *, weight: float = 1.0) -> None:
        """upstream_id's output is typically consumed by downstream_id —
        §12's fixed pipelines (Planner → Coding → Testing → Reviewer)
        declared as edges instead of literal code paths."""
        self._graph.add_edge(upstream_id, downstream_id, kind="dependency", weight=weight)

    def record_handoff(self, from_id: str, to_id: str) -> None:
        """Called by the orchestrator every time one agent's output
        actually fed into another (SequentialStep/HierarchicalStep) —
        builds the communication graph up from REAL executions rather
        than a declared topology, incrementing the edge weight each time
        the same hand-off recurs so `most_frequent_next()` reflects
        actual usage, not just what was configured."""
        existing = [d for _, _, d in self._graph.out_edges(from_id, data=True)
                   if d.get("kind") == "communication" and self._graph.has_edge(from_id, to_id)]
        if self._graph.has_edge(from_id, to_id, key="communication"):
            self._graph[from_id][to_id]["communication"]["weight"] += 1
        else:
            self._graph.add_edge(from_id, to_id, key="communication", kind="communication", weight=1)
        del existing  # only used for the has_edge probe above; keeps intent readable

    def capability_neighbors(self, agent_id: str, missing_capability: str,
                            candidates: list[AgentMetadata]) -> list[AgentMetadata]:
        """Which OTHER registered agents can cover a capability `agent_id`
        itself lacks — recorded as capability edges so this is queryable
        rather than re-derived by scanning every agent's metadata each
        time (network.py is the place that answers "who else can do
        this", registry.py just holds the agents)."""
        covering = [a for a in candidates if missing_capability in a.capabilities and a.agent_id != agent_id]
        for a in covering:
            self._graph.add_edge(agent_id, a.agent_id, kind="capability", weight=1.0,
                                 capability=missing_capability)
        return covering

    def most_frequent_next(self, agent_id: str, top_n: int = 3) -> list[tuple[str, float]]:
        """Ranked by observed communication-edge weight — "given this
        agent just ran, who does it usually hand off to next" — used by
        orchestrator.py to suggest (not force) the next workflow step."""
        if not self._graph.has_node(agent_id):
            return []
        edges = [(v, d.get("weight", 1.0)) for _, v, d in self._graph.out_edges(agent_id, data=True)
                if d.get("kind") == "communication"]
        edges.sort(key=lambda t: t[1], reverse=True)
        return edges[:top_n]

    def dependency_chain_from(self, agent_id: str) -> list[str]:
        """The declared downstream pipeline starting at agent_id, e.g.
        planner → coding → testing → reviewer, read straight off the
        dependency edges rather than hard-coded anywhere."""
        chain = [agent_id]
        current = agent_id
        seen = {agent_id}
        while True:
            nexts = [v for _, v, d in self._graph.out_edges(current, data=True)
                    if d.get("kind") == "dependency" and v not in seen]
            if not nexts:
                break
            current = nexts[0]
            chain.append(current)
            seen.add(current)
        return chain

    def to_dict(self) -> dict:
        """A JSON-serializable snapshot for the observability/metrics API
        — nodes with their key metadata, edges with kind+weight. A node
        can exist with no "metadata" attribute if it was only ever
        touched via add_edge (e.g. NetworkX auto-creating an endpoint
        for a capability/communication edge whose add_agent() call
        hasn't happened yet) — reported with empty fields rather than
        crashing, since a hole in observability is far less harmful than
        the metrics endpoint 500ing."""
        nodes = [{"agent_id": n, "name": d["metadata"].name,
                 "capabilities": sorted(d["metadata"].capabilities)}
                if "metadata" in d else {"agent_id": n, "name": n, "capabilities": []}
                for n, d in self._graph.nodes(data=True)]
        edges = [{"from": u, "to": v, "kind": d.get("kind"), "weight": d.get("weight", 1.0)}
                for u, v, d in self._graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}


# Process-wide singleton, same lifecycle as registry.registry/router.router.
network = AgentNetwork()
