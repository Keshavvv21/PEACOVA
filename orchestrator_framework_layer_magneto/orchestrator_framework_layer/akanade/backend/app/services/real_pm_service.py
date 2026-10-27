"""Client for the REAL Algo-Kaizen app's own REST API — used instead of
pm_service's demo LocalStore when REAL_APP_API_BASE_URL is configured (see
config.py, auth.py). Every call forwards the CALLING USER's own real
token (never a bot credential), so results and permissions are exactly
what that person would see clicking around the real app — same rule
pm_service follows for the demo store, just over HTTP instead of an
in-process call.

Covers both reads (list/overdue/workload) and writes (create/patch/delete
task, project, space, sprint, subtask, personal task, task message) — see
tools.py's execute()/resolve_arguments() for which tools branch here vs.
the demo pm_service, and app/agent/tools.py's resolve_*_real() functions
for how a name in a chat message becomes one of these real ids.

Several single-task/subtask/message endpoints below require project_id
as a QUERY PARAM, not just the resource id in the URL — the real app's
own authorization check (require_project_membership) needs to know which
project to check membership against, and a task/subtask/message thread
id alone doesn't carry that (see the real app's own docstrings on
app/routers/tasks.py, subtasks.py, messages.py). tools.py threads this
through via Resolution.project_id, populated when a task is resolved.
"""
from __future__ import annotations

import datetime as dt

import httpx

from ..config import settings


def _url(path: str) -> str:
    return f"{settings().REAL_APP_API_BASE_URL}{path}"


def _raise_readable(r: httpx.Response) -> None:
    """httpx's own HTTPStatusError message is a wall of text (the URL, a
    docs link) with no room for the actual reason — FastAPI's error body
    (`{"detail": "..."}`, or a list of validation errors) is the part
    worth showing in a "Sorry — that failed: ..." chat reply."""
    try:
        body = r.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
    except ValueError:
        detail = r.text
    raise RuntimeError(f"{r.status_code}: {detail}")


def _get(token: str, path: str, params: dict | None = None) -> list[dict] | dict:
    r = httpx.get(_url(path), params=params,
                  headers={"Authorization": f"Bearer {token}"}, timeout=10)
    if r.is_error:
        _raise_readable(r)
    return r.json()


def _post(token: str, path: str, body: dict, params: dict | None = None) -> list[dict] | dict:
    r = httpx.post(_url(path), json=body, params=params,
                   headers={"Authorization": f"Bearer {token}"}, timeout=10)
    if r.is_error:
        _raise_readable(r)
    return r.json() if r.content else {}


def _patch(token: str, path: str, body: dict, params: dict | None = None) -> list[dict] | dict:
    r = httpx.patch(_url(path), json=body, params=params,
                    headers={"Authorization": f"Bearer {token}"}, timeout=10)
    if r.is_error:
        _raise_readable(r)
    return r.json() if r.content else {}


def _delete(token: str, path: str, params: dict | None = None) -> None:
    r = httpx.delete(_url(path), params=params,
                     headers={"Authorization": f"Bearer {token}"}, timeout=10)
    if r.is_error:
        _raise_readable(r)


# ── reads ──────────────────────────────────────────────────────────────

def list_projects(token: str) -> list[dict]:
    """Every project the real, signed-in user can see."""
    return _get(token, "/api/projects")


def get_project(token: str, project_id: str) -> dict:
    return _get(token, f"/api/projects/{project_id}")


def list_users(token: str) -> list[dict]:
    return _get(token, "/api/users")


def get_user_name(token: str, user_id: str) -> str | None:
    """No single GET /api/users/{id} exists on the real app (see
    app/routers/users.py) — only for phrasing a reply, so a list+find is
    simpler than adding an endpoint that exists solely for chat wording."""
    for u in list_users(token):
        if u["user_id"] == user_id:
            return u["name"]
    return None


def list_spaces(token: str) -> list[dict]:
    return _get(token, "/api/spaces")


def list_roles(token: str) -> list[dict]:
    """The org's roles — the 4 built-ins plus any custom roles an Admin has
    created (see the real app's app/routers/roles.py). require_permission
    ("manage_roles") on the real side means this 403s for anyone without
    that grant, same as any other admin-only real_pm_service call."""
    return _get(token, "/api/roles")


def delete_user(token: str, user_id: str) -> None:
    """Permanently delete a real account — Admin-only on the real backend
    (enforced there, a 403 surfaces as this call raising). The real app's
    own UserService cascades the cleanup (unassigns their tasks rather
    than leaving them dangling, drops project memberships, deletes their
    personal tasks) — this is just the same DELETE /api/users/{id} the
    real app's own UI hits, nothing bot-specific."""
    _delete(token, f"/api/users/{user_id}")


def list_tasks_for_project(token: str, project_id: str) -> list[dict]:
    return _get(token, f"/api/projects/{project_id}/tasks")


def list_all_tasks(token: str) -> list[dict]:
    """No single real-app endpoint lists every task across every project
    (by design — see app/routers/tasks.py), so this fans out: one call to
    list projects, one per project for its tasks. Fine at this app's
    real-world project counts; would need pagination/concurrency to scale
    much further, which isn't a problem worth solving speculatively here.

    /api/projects and /api/projects/{id}/tasks use different
    authorization rules — an Admin can LIST every org project but isn't
    necessarily a MEMBER of each one, and the tasks sub-resource 403s for
    a non-member. Skip those rather than letting one inaccessible project
    crash every task lookup across the whole org."""
    tasks: list[dict] = []
    for p in list_projects(token):
        try:
            tasks.extend(list_tasks_for_project(token, p["project_id"]))
        except RuntimeError as e:
            if not str(e).startswith("403"):
                raise
    return tasks


def list_my_tasks(token: str) -> list[dict]:
    return _get(token, "/api/tasks/mine")


def get_task(token: str, task_id: str, project_id: str) -> dict:
    return _get(token, f"/api/tasks/{task_id}", params={"project_id": project_id})


def list_sprints_for_project(token: str, project_id: str) -> list[dict]:
    return _get(token, f"/api/projects/{project_id}/sprints")


def find_sprint_project_id(token: str, sprint_id: str) -> str | None:
    """No single real endpoint returns one sprint by id (see
    app/routers/sprints.py) — needed for start/complete's project_id URL
    segment when the user didn't name a project, so this scans once.
    Same non-member-project skip as list_all_tasks above."""
    for p in list_projects(token):
        try:
            sprints = list_sprints_for_project(token, p["project_id"])
        except RuntimeError as e:
            if str(e).startswith("403"):
                continue
            raise
        for sp in sprints:
            if sp["sprint_id"] == sprint_id:
                return p["project_id"]
    return None


def sprint_name(token: str, project_id: str, sprint_id: str) -> str | None:
    """Same "no single-item endpoint" shape as find_sprint_project_id —
    this one's only for phrasing a reply ("moved to sprint X")."""
    for sp in list_sprints_for_project(token, project_id):
        if sp["sprint_id"] == sprint_id:
            return sp["name"]
    return None


def overdue_tasks(token: str) -> list[dict]:
    """Unscoped only (every project) — same boundary as list_all_tasks: a
    scoped ("overdue in <project>") ask still needs real entity resolution
    for the project name, not yet converted, so it stays on the demo
    knowledge_graph path (see tools.py's graph_query branch)."""
    today = dt.date.today().isoformat()
    return [t for t in list_all_tasks(token)
            if t.get("due_date") and t["due_date"] < today and t.get("status") != "done"]


def workload(token: str) -> list[dict]:
    """Open (not done) task counts per assignee, across every project —
    same unscoped-only boundary as overdue_tasks above."""
    counts: dict[str, int] = {}
    for t in list_all_tasks(token):
        if t.get("status") == "done" or not t.get("assignee_id"):
            continue
        who = t.get("assignee_name") or t["assignee_id"]
        counts[who] = counts.get(who, 0) + 1
    return sorted(({"name": name, "open_tasks": n} for name, n in counts.items()),
                  key=lambda r: -r["open_tasks"])


def list_personal_tasks(token: str) -> list[dict]:
    """A user's own private to-do list — never tied to a project, never
    visible to anyone else. Genuinely a different feature from "my tasks"
    (list_my_tasks above, project-assigned work), not a synonym for it —
    see app/schemas/personal_task_schemas.py's own docstring on the real
    app's side."""
    return _get(token, "/api/personal-tasks")


def create_personal_note(token: str, title: str | None, body: str) -> dict:
    """A user's own private freeform note — see
    app/schemas/personal_note_schemas.py on the real app's side. Gated by
    that app's org-wide "personalNotes" feature flag; a 403 here (an Admin
    turned it off) surfaces as this call raising."""
    return _post(token, "/api/personal-notes", {"title": title, "body": body})


# ── writes ─────────────────────────────────────────────────────────────

def create_space(token: str, name: str, description: str | None = None,
                 project_owner_id: str | None = None) -> dict:
    body = {"name": name, "description": description}
    if project_owner_id:
        body["project_owner_id"] = project_owner_id
    return _post(token, "/api/spaces", body)


def create_project(token: str, name: str, space_id: str | None = None,
                   methodology: str = "kanban", member_ids: list[str] | None = None,
                   project_owner_id: str | None = None,
                   team_lead_id: str | None = None) -> dict:
    """project_owner_id/team_lead_id are required by the real API depending
    on the CALLER's own role (an Admin must name a Senior Manager, an Admin
    or Manager2 must name a Manager) — see ProjectCreate's docstring on the
    real app's side. Not resolved here: if the caller didn't name one and
    the real API needs it, the real 400 and its message come straight back
    to the chat reply ("Sorry — that failed: ...") rather than being
    guessed at."""
    body = {"name": name, "space_id": space_id, "methodology": methodology,
            "member_ids": member_ids or []}
    if project_owner_id:
        body["project_owner_id"] = project_owner_id
    if team_lead_id:
        body["team_lead_id"] = team_lead_id
    return _post(token, "/api/projects", body)


def add_member(token: str, project_id: str, user_id: str, role: str = "Member") -> dict:
    return _post(token, f"/api/projects/{project_id}/members",
                {"user_id": user_id, "role": role})


def create_task(token: str, project_id: str, title: str, description: str | None = None,
                due_date: str | None = None, assignee_id: str | None = None,
                sprint_id: str | None = None) -> dict:
    body = {"title": title, "description": description, "due_date": due_date,
            "assignee_id": assignee_id, "sprint_id": sprint_id}
    return _post(token, f"/api/projects/{project_id}/tasks", body)


def patch_task(token: str, task_id: str, project_id: str, **changes) -> dict:
    """Covers assign (assignee_id), due-date change, and status change —
    the same one PATCH endpoint the real app's own task board uses for
    all three; None values are dropped so this only touches the fields
    actually being changed (see TaskUpdate's "unset means leave alone"
    docstring on the real app's side)."""
    body = {k: v for k, v in changes.items() if v is not None}
    return _patch(token, f"/api/tasks/{task_id}", body, params={"project_id": project_id})


def delete_task(token: str, task_id: str, project_id: str) -> None:
    _delete(token, f"/api/tasks/{task_id}", params={"project_id": project_id})


def set_task_sprint(token: str, task_id: str, project_id: str, sprint_id: str | None) -> dict:
    """Sprint reassignment is its own dedicated endpoint on the real
    app (not part of the generic task PATCH) specifically so it can
    express "move back to the Backlog" as an explicit null — see
    TaskUpdate's own docstring on the real app's side for why."""
    return _patch(token, f"/api/tasks/{task_id}/sprint", {"sprint_id": sprint_id},
                 params={"project_id": project_id})


def create_subtask(token: str, task_id: str, project_id: str, title: str,
                   due_date: str | None = None) -> dict:
    return _post(token, f"/api/tasks/{task_id}/subtasks",
                {"title": title, "due_date": due_date}, params={"project_id": project_id})


def list_subtasks(token: str, task_id: str, project_id: str) -> list[dict]:
    return _get(token, f"/api/tasks/{task_id}/subtasks", params={"project_id": project_id})


def patch_subtask(token: str, task_id: str, project_id: str, subtask_id: str, **changes) -> dict:
    body = {k: v for k, v in changes.items() if v is not None}
    return _patch(token, f"/api/tasks/{task_id}/subtasks/{subtask_id}", body,
                 params={"project_id": project_id})


def archive_project(token: str, project_id: str) -> dict:
    return _patch(token, f"/api/projects/{project_id}", {"status": "archived"})


def create_sprint(token: str, project_id: str, name: str) -> dict:
    return _post(token, f"/api/projects/{project_id}/sprints", {"name": name})


def start_sprint(token: str, project_id: str, sprint_id: str) -> dict:
    return _post(token, f"/api/projects/{project_id}/sprints/{sprint_id}/start", {})


def complete_sprint(token: str, project_id: str, sprint_id: str) -> dict:
    return _post(token, f"/api/projects/{project_id}/sprints/{sprint_id}/complete", {})


def patch_personal_task(token: str, personal_task_id: str, **changes) -> dict:
    body = {k: v for k, v in changes.items() if v is not None}
    return _patch(token, f"/api/personal-tasks/{personal_task_id}", body)


def create_task_message(token: str, task_id: str, project_id: str, text: str,
                        mentions: list[str] | None = None) -> dict:
    return _post(token, f"/api/tasks/{task_id}/messages",
                {"text": text, "mentions": mentions or []}, params={"project_id": project_id})


def create_project_message(token: str, project_id: str, text: str,
                           mentions: list[str] | None = None) -> dict:
    """Unlike create_task_message, no project_id query param is needed —
    the real app's POST /api/projects/{id}/messages already has the
    project id in the path, that IS what require_project_membership
    checks against."""
    return _post(token, f"/api/projects/{project_id}/messages",
                {"text": text, "mentions": mentions or []})
