"""§4 Graph-Based Retrieval — the in-memory NetworkX knowledge graph.

Nodes: users, spaces, projects, tasks, sprints. Edges: containment and
assignment. Rebuilt on a short TTL (§4.1) plus immediate invalidation
from the service layer's mutation hooks. Strictly read-only (§4.2):
graph_query returns already-correct result sets; the LLM only phrases
them — it never joins data itself.
"""
from __future__ import annotations

import datetime as dt
import threading
import time

import networkx as nx

from ..config import settings
from ..services import pm_service


class KnowledgeGraph:
    def __init__(self):
        self._g = nx.MultiDiGraph()
        self._built_at = 0.0
        self._lock = threading.Lock()
        pm_service.on_mutation(lambda kind, e: self.invalidate())

    # ── lifecycle ────────────────────────────────────────────────────

    def invalidate(self):
        self._built_at = 0.0

    def graph(self) -> nx.MultiDiGraph:
        with self._lock:
            if time.time() - self._built_at > settings().GRAPH_TTL_SECONDS:
                self._rebuild()
            return self._g

    def _rebuild(self):
        g = nx.MultiDiGraph()
        for u in pm_service.list_users():
            g.add_node(u["id"], kind="user", name=u["name"], role=u["role"])
        for s in pm_service.list_spaces():
            g.add_node(s["id"], kind="space", name=s["name"])
        for p in pm_service.list_projects():
            g.add_node(p["id"], kind="project", name=p["name"], board=p["board"])
            g.add_edge(p["space_id"], p["id"], rel="contains")
            for sp in pm_service.list_sprints(p["id"]):
                g.add_node(sp["id"], kind="sprint", name=sp["name"], state=sp["state"])
                g.add_edge(p["id"], sp["id"], rel="has_sprint")
        for t in pm_service.list_tasks():
            g.add_node(t["id"], kind="task", name=t["title"], status=t["status"],
                       due=t.get("due_date"), priority=t["priority"])
            g.add_edge(t["project_id"], t["id"], rel="contains")
            if t.get("assignee_id"):
                g.add_edge(t["id"], t["assignee_id"], rel="assigned_to")
            if t.get("sprint_id"):
                g.add_edge(t["sprint_id"], t["id"], rel="in_sprint")
        self._g = g
        self._built_at = time.time()

    # ── multi-hop queries (§4.2's examples, implemented) ─────────────

    def _tasks_under(self, node_id: str) -> list[str]:
        """All task nodes reachable below a space/project/sprint node."""
        g = self.graph()
        if node_id not in g:
            return []
        out = []
        for n in nx.descendants(g, node_id) | {node_id}:
            if g.nodes[n].get("kind") == "task":
                out.append(n)
        return out

    def _visible_task_ids(self, user: dict | None) -> set[str] | None:
        """None means "no restriction" (Executive, the top demo rank mapped
        from the real app's Admin — see auth._REAL_ROLE_MAP — or no `user`
        given at all, e.g. this graph's own invalidation/rebuild machinery);
        otherwise the confidentiality-scoped set of task node ids filed
        under a project this user can see (pm_service.list_projects). Every
        query below intersects its candidate id set against this so
        graph_query can't be used to see across projects a chat tool's own
        list/get calls already refuse to show."""
        if user is None or user["role"] == "Executive":
            return None
        ids: set[str] = set()
        for p in pm_service.list_projects(user):
            ids.update(self._tasks_under(p["id"]))
        return ids

    def overdue_tasks(self, scope_id: str | None = None,
                      assignee_id: str | None = None, user: dict | None = None) -> list[dict]:
        g = self.graph()
        today = dt.date.today().isoformat()
        ids = self._tasks_under(scope_id) if scope_id else [
            n for n in g.nodes if g.nodes[n].get("kind") == "task"]
        visible = self._visible_task_ids(user)
        if visible is not None:
            ids = [n for n in ids if n in visible]
        out = []
        for n in ids:
            nd = g.nodes[n]
            if nd.get("due") and nd["due"] < today and nd["status"] != "Done":
                if assignee_id and not g.has_edge(n, assignee_id):
                    continue
                out.append({"id": n, **nd})
        return out

    def tasks_for(self, assignee_id: str, scope_id: str | None = None,
                 user: dict | None = None) -> list[dict]:
        g = self.graph()
        ids = self._tasks_under(scope_id) if scope_id else [
            n for n in g.nodes if g.nodes[n].get("kind") == "task"]
        visible = self._visible_task_ids(user)
        if visible is not None:
            ids = [n for n in ids if n in visible]
        return [{"id": n, **g.nodes[n]} for n in ids if g.has_edge(n, assignee_id)]

    def workload(self, scope_id: str | None = None, user: dict | None = None) -> list[dict]:
        """'who has the most tasks' — counts per assignee, descending."""
        g = self.graph()
        candidates = self._tasks_under(scope_id) if scope_id else list(g.nodes)
        visible = self._visible_task_ids(user)
        counts: dict[str, int] = {}
        for n in candidates:
            if g.nodes[n].get("kind") != "task" or g.nodes[n]["status"] == "Done":
                continue
            if visible is not None and n not in visible:
                continue
            for _, u, d in g.out_edges(n, data=True):
                if d.get("rel") == "assigned_to":
                    counts[u] = counts.get(u, 0) + 1
        return sorted(
            [{"user_id": u, "name": g.nodes[u]["name"], "open_tasks": c}
             for u, c in counts.items()],
            key=lambda r: -r["open_tasks"])

    def overdue_by_person(self, scope_id: str | None = None, user: dict | None = None) -> list[dict]:
        """'summary of people who are overdue' — counts of OVERDUE (not just
        open) tasks per assignee, descending. Same shape as workload() but
        filtered to the overdue_tasks() definition of late."""
        g = self.graph()
        today = dt.date.today().isoformat()
        candidates = self._tasks_under(scope_id) if scope_id else list(g.nodes)
        visible = self._visible_task_ids(user)
        counts: dict[str, int] = {}
        for n in candidates:
            nd = g.nodes[n]
            if nd.get("kind") != "task" or nd["status"] == "Done":
                continue
            if visible is not None and n not in visible:
                continue
            if not nd.get("due") or nd["due"] >= today:
                continue
            for _, u, d in g.out_edges(n, data=True):
                if d.get("rel") == "assigned_to":
                    counts[u] = counts.get(u, 0) + 1
        return sorted(
            [{"user_id": u, "name": g.nodes[u]["name"], "overdue_tasks": c}
             for u, c in counts.items()],
            key=lambda r: -r["overdue_tasks"])


knowledge_graph = KnowledgeGraph()
