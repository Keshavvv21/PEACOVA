"""The mini Aksha PM service layer — the 'existing backend' the bot rides
on. Spaces → Projects → Tasks (+ Sprints), role checks, soft delete
(trash), a due-date audit trail, and invalidation hooks (§6.1 step 13):
every mutation refreshes the graph and re-embeds changed text.

The bot NEVER bypasses this layer (§1): chat tools call these functions
with the caller's identity, exactly as the REST router does.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Optional

from fastapi import HTTPException

from ..auth import hash_pw
from ..db import store
from ..models import Role

# invalidation hooks are registered by retrieval layers at startup
_invalidation_hooks: list = []


def on_mutation(fn):
    _invalidation_hooks.append(fn)
    return fn


def _fire_hooks(kind: str, entity: dict):
    for fn in _invalidation_hooks:
        try:
            fn(kind, entity)
        except Exception:
            pass  # retrieval refresh must never break a mutation


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


ROLE_ORDER: dict[str, int] = {"Member": 0, "Manager": 1, "Admin": 2, "Executive": 3}


def _need(user: dict, minimum: Role):
    if ROLE_ORDER[user["role"]] < ROLE_ORDER[minimum]:
        raise HTTPException(403, f"requires {minimum}+ role")


# ── users ────────────────────────────────────────────────────────────────

def list_users() -> list[dict]:
    return store("Users").scan("USER#")


def get_user(user_id: str) -> Optional[dict]:
    return store("Users").get(f"USER#{user_id}", "PROFILE")


def delete_user(user: dict, user_id: str) -> None:
    """Demo-store counterpart to real_pm_service.delete_user — Admin+
    only. Unlike the real app's UserService this demo store doesn't
    reach across tasks/projects to clean up references (the demo dataset
    has no such cross-entity cascade wired up); it just removes the
    profile row, matching the scope of this store's other demo mutations."""
    _need(user, "Admin")
    if not store("Users").get(f"USER#{user_id}", "PROFILE"):
        raise HTTPException(404, f"user {user_id!r} not found")
    store("Users").delete(f"USER#{user_id}", "PROFILE")


def create_user(user: dict, name: str, role: str, password: str = "demo123",
                email: str = "") -> dict:
    """Provision a new demo account — Admin+ only, mirrors the real app's
    "no public self-registration" rule. username is derived from name
    (lowercased, spaces → dots) since the bot only ever gets a display
    name out of a chat sentence, never a chosen login handle. email is
    optional and purely informational here — this demo realm still logs
    in by username (see LoginIn), not email, so it's stored on the
    profile but never used as a credential."""
    _need(user, "Admin")
    if role not in ROLE_ORDER:
        raise HTTPException(400, f"invalid role {role!r}")
    username = name.lower().replace(" ", ".")
    if store("Users").get(f"USER#{username}", "PROFILE"):
        raise HTTPException(409, f"user {username!r} already exists")
    u = {"id": username, "name": name, "role": role, "email": email,
         "password_hash": hash_pw(password)}
    store("Users").put(f"USER#{username}", "PROFILE", u)
    _fire_hooks("user", u)
    return u


# ── spaces ───────────────────────────────────────────────────────────────

def create_space(user: dict, name: str) -> dict:
    _need(user, "Manager")
    sp = {"id": _id("space"), "name": name, "created_by": user["sub"],
          "created_at": _now(), "deleted": False}
    store("Spaces").put(f"SPACE#{sp['id']}", "META", sp)
    _fire_hooks("space", sp)
    return sp


def _member_of(user: dict, project: dict) -> bool:
    return user["sub"] in project.get("members", [])


def list_spaces(user: dict | None = None) -> list[dict]:
    """Spaces visible to `user`: every one for Executive (this demo store's
    top rank, mapped from the real app's Admin — see auth._REAL_ROLE_MAP) or
    when called with no user (background indexing, see app.retrieval); for
    everyone else, only a space they created or that has a project they
    belong to filed under it — mirrors the real backend's confidentiality
    rule (SpaceService.list_spaces), demo-store version."""
    spaces = [s for s in store("Spaces").scan("SPACE#") if not s.get("deleted")]
    if user is None or user["role"] == "Executive":
        return spaces
    visible_space_ids = {p["space_id"] for p in list_projects(user) if p.get("space_id")}
    return [s for s in spaces if s["created_by"] == user["sub"] or s["id"] in visible_space_ids]


# ── projects ─────────────────────────────────────────────────────────────

def create_project(user: dict, name: str, space_id: str, board: str) -> dict:
    _need(user, "Manager")
    if not store("Spaces").get(f"SPACE#{space_id}", "META"):
        raise HTTPException(404, f"space {space_id} not found")
    p = {"id": _id("proj"), "name": name, "space_id": space_id, "board": board,
         "created_by": user["sub"], "created_at": _now(),
         "deleted": False, "archived": False, "members": [user["sub"]],
         # member_roles is project-scoped (Owner/CoOwner/Manager/Member),
         # distinct from the global Role every user also carries — see
         # models.ProjectRole. The creator is always Owner.
         "member_roles": {user["sub"]: "Owner"}}
    store("Projects").put(f"PROJECT#{p['id']}", "META", p)
    _fire_hooks("project", p)
    return p


def add_member(user: dict, project_id: str, member_user_id: str,
               role: str = "Member") -> dict:
    """Add a user to a project with a project-scoped role — Manager+ (global)
    only, same bar as creating the project itself. Re-adding an existing
    member just updates their role (promote/demote), doesn't duplicate them."""
    _need(user, "Manager")
    if role not in ("Owner", "CoOwner", "Manager", "Member"):
        raise HTTPException(400, f"invalid project role {role!r}")
    if not get_user(member_user_id):
        raise HTTPException(404, f"user {member_user_id} not found")
    p = get_project(project_id)
    if member_user_id not in p["members"]:
        p["members"].append(member_user_id)
    p.setdefault("member_roles", {})[member_user_id] = role
    store("Projects").put(f"PROJECT#{project_id}", "META", p)
    _fire_hooks("project", p)
    return p


def list_projects(user: dict | None = None) -> list[dict]:
    """Projects visible to `user`: every one for Executive, or when called
    with no user (background indexing); everyone else only sees projects
    they're a member of — which already covers "created it" too, since
    create_project always adds the creator as a Member/Owner."""
    projects = [p for p in store("Projects").scan("PROJECT#")
                if not p.get("deleted") and not p.get("archived")]
    if user is None or user["role"] == "Executive":
        return projects
    return [p for p in projects if _member_of(user, p)]


def get_project(project_id: str, user: dict | None = None) -> dict:
    """Fetch a project, 404ing if missing and 403ing if `user` was passed
    and isn't allowed to see it (Executive, or an actual member — same rule
    as list_projects). `user` is omitted by internal callers that already
    have their own permission gate or just need an existence check (e.g.
    create_task), so they're unaffected by this visibility rule."""
    p = store("Projects").get(f"PROJECT#{project_id}", "META")
    if not p or p.get("deleted"):
        raise HTTPException(404, f"project {project_id} not found")
    if user is not None and user["role"] != "Executive" and not _member_of(user, p):
        raise HTTPException(403, "not a member of this project")
    return p


def archive_project(user: dict, project_id: str) -> dict:
    _need(user, "Admin")
    p = get_project(project_id)
    p["archived"] = True
    p["archived_at"] = _now()
    store("Projects").put(f"PROJECT#{project_id}", "META", p)
    _fire_hooks("project", p)
    return p


def rename_project(user: dict, project_id: str, new_name: str) -> dict:
    """Same bar as creating one. The demo target for the JSON tool-plugin
    framework (see ../plugins.py) — a genuinely new chat capability wired
    up without touching agent/tools.py or llm/providers.py."""
    _need(user, "Manager")
    p = get_project(project_id)
    p["name"] = new_name
    store("Projects").put(f"PROJECT#{project_id}", "META", p)
    _fire_hooks("project", p)
    return p


# ── tasks ────────────────────────────────────────────────────────────────

def create_task(user: dict, project_id: str, title: str, description: str = "",
                assignee_id: Optional[str] = None, due_date: Optional[str] = None,
                priority: str = "Medium", labels: Optional[list] = None,
                start_date: Optional[str] = None) -> dict:
    get_project(project_id)  # 404 if missing
    t = {"id": _id("task"), "project_id": project_id, "title": title,
         "description": description, "assignee_id": assignee_id,
         "start_date": start_date, "due_date": due_date, "priority": priority,
         "labels": labels or [], "status": "Backlog", "sprint_id": None,
         "created_by": user["sub"], "created_at": _now(),
         "deleted": False, "audit": []}
    store("Tasks").put(f"TASK#{t['id']}", "META", t)
    _fire_hooks("task", t)
    return t


def get_task(task_id: str) -> dict:
    t = store("Tasks").get(f"TASK#{task_id}", "META")
    if not t or t.get("deleted"):
        raise HTTPException(404, f"task {task_id} not found")
    return t


def list_tasks(user: Optional[dict] = None, project_id: Optional[str] = None,
               assignee_id: Optional[str] = None) -> list[dict]:
    """Tasks visible to `user`: filtered to a project (403s via get_project
    if `user` can't see that project), to a specific assignee (someone's own
    "my tasks" is always visible to them, same as the real backend's
    /tasks/mine having no membership gate), or — with neither given —
    every task in every project `user` can see (again, everyone but
    Executive, or a caller with no `user`, is scoped down from "every task
    in every project" to just their own projects')."""
    out = [t for t in store("Tasks").scan("TASK#") if not t.get("deleted")]
    if project_id:
        if user is not None:
            get_project(project_id, user)
        out = [t for t in out if t["project_id"] == project_id]
    elif user is not None and not assignee_id:
        visible_ids = {p["id"] for p in list_projects(user)}
        out = [t for t in out if t["project_id"] in visible_ids]
    if assignee_id:
        out = [t for t in out if t.get("assignee_id") == assignee_id]
    return out


def patch_task(user: dict, task_id: str, **changes) -> dict:
    t = get_task(task_id)
    for k, v in changes.items():
        if v is None:
            continue
        if k == "due_date" and v != t.get("due_date"):
            # the due-date audit trail the design says the bot can't skip
            t["audit"].append({"field": "due_date", "from": t.get("due_date"),
                               "to": v, "by": user["sub"], "at": _now()})
        t[k] = v
    store("Tasks").put(f"TASK#{task_id}", "META", t)
    _fire_hooks("task", t)
    return t


def move_task_to_sprint(user: dict, task_id: str, sprint_id: Optional[str]) -> dict:
    t = get_task(task_id)
    if sprint_id is not None:
        get_sprint(sprint_id)  # validate
        if t["status"] == "Backlog":
            t["status"] = "Todo"
    t["sprint_id"] = sprint_id
    store("Tasks").put(f"TASK#{task_id}", "META", t)
    _fire_hooks("task", t)
    return t


def delete_task(user: dict, task_id: str) -> dict:
    """Soft delete → trash, per the design's existing behavior."""
    t = get_task(task_id)
    t["deleted"] = True
    t["deleted_at"] = _now()
    t["deleted_by"] = user["sub"]
    store("Tasks").put(f"TASK#{task_id}", "META", t)
    _fire_hooks("task", t)
    return t


# ── subtasks ─────────────────────────────────────────────────────────────

def create_subtask(user: dict, task_id: str, title: str) -> dict:
    get_task(task_id)  # 404 if missing/deleted
    st = {"id": _id("subtask"), "task_id": task_id, "title": title,
          "done": False, "created_by": user["sub"], "created_at": _now()}
    store("Subtasks").put(f"SUBTASK#{st['id']}", "META", st)
    _fire_hooks("subtask", st)
    return st


def list_subtasks(task_id: str) -> list[dict]:
    return [s for s in store("Subtasks").scan("SUBTASK#") if s["task_id"] == task_id]


def get_subtask(subtask_id: str) -> dict | None:
    return store("Subtasks").get(f"SUBTASK#{subtask_id}", "META")


def patch_subtask(user: dict, subtask_id: str, **changes) -> dict:
    st = get_subtask(subtask_id)
    if not st:
        raise HTTPException(404, f"subtask {subtask_id} not found")
    st.update({k: v for k, v in changes.items() if v is not None})
    store("Subtasks").put(f"SUBTASK#{subtask_id}", "META", st)
    _fire_hooks("subtask", st)
    return st


# ── sprints ──────────────────────────────────────────────────────────────

def create_sprint(user: dict, project_id: str, name: str) -> dict:
    _need(user, "Manager")
    get_project(project_id)
    s = {"id": _id("sprint"), "project_id": project_id, "name": name,
         "state": "Planned", "created_at": _now()}
    store("Sprints").put(f"SPRINT#{s['id']}", "META", s)
    _fire_hooks("sprint", s)
    return s


def get_sprint(sprint_id: str) -> dict:
    s = store("Sprints").get(f"SPRINT#{sprint_id}", "META")
    if not s:
        raise HTTPException(404, f"sprint {sprint_id} not found")
    return s


def list_sprints(project_id: str) -> list[dict]:
    return [s for s in store("Sprints").scan("SPRINT#")
            if s["project_id"] == project_id]


def start_sprint(user: dict, sprint_id: str) -> dict:
    _need(user, "Manager")
    s = get_sprint(sprint_id)
    s["state"] = "Active"
    s["started_at"] = _now()
    store("Sprints").put(f"SPRINT#{s['id']}", "META", s)
    _fire_hooks("sprint", s)
    return s


def complete_sprint(user: dict, sprint_id: str) -> dict:
    _need(user, "Manager")
    s = get_sprint(sprint_id)
    s["state"] = "Completed"
    s["completed_at"] = _now()
    store("Sprints").put(f"SPRINT#{s['id']}", "META", s)
    # incomplete tasks fall back to backlog — a real business rule the
    # bot must not be able to skip (why complete_sprint is destructive)
    for t in list_tasks():
        if t.get("sprint_id") == sprint_id and t["status"] != "Done":
            t["sprint_id"] = None
            t["status"] = "Backlog"
            store("Tasks").put(f"TASK#{t['id']}", "META", t)
            _fire_hooks("task", t)
    _fire_hooks("sprint", s)
    return s
