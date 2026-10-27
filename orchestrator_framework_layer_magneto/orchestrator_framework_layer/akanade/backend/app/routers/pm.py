"""The mini Aksha PM REST API — 'the same API the React frontend already
uses' (§1). The chat tools call the service layer these routes call;
the routes exist so the frontend board and curl can act as the UI would.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import current_user, hash_pw, make_token, require_role
from ..db import store
from ..models import (LoginIn, MemberAddIn, ProjectIn, SpaceIn, SprintIn,
                      SubtaskIn, TaskIn, TaskPatch, TokenOut, UserCreateIn)
from ..services import pm_service
from ..services.ws import manager

router = APIRouter(prefix="/api")


@router.post("/auth/login", response_model=TokenOut)
def login(body: LoginIn):
    username = body.username or body.email
    if not username:
        raise HTTPException(422, "username or email required")
    u = store("Users").get(f"USER#{username}", "PROFILE")
    if not u or u["password_hash"] != hash_pw(body.password):
        raise HTTPException(401, "bad credentials")
    return TokenOut(access_token=make_token(u["id"], u["name"], u["role"]),
                    role=u["role"], user_id=u["id"], name=u["name"])


@router.get("/users")
def users(_: dict = Depends(current_user)):
    return [{"id": u["id"], "name": u["name"], "role": u["role"]}
            for u in pm_service.list_users()]


@router.post("/users")
async def create_user(body: UserCreateIn, user: dict = Depends(require_role("Admin"))):
    u = pm_service.create_user(user, body.name, body.role, body.password, body.email)
    await manager.broadcast({"type": "user.created",
                             "user": {"id": u["id"], "name": u["name"], "role": u["role"]}})
    return {"id": u["id"], "name": u["name"], "role": u["role"], "email": u["email"]}


@router.post("/spaces")
async def create_space(body: SpaceIn, user: dict = Depends(require_role("Manager"))):
    s = pm_service.create_space(user, body.name)
    await manager.broadcast({"type": "space.created", "space": s})
    return s


@router.get("/spaces")
def spaces(user: dict = Depends(current_user)):
    return pm_service.list_spaces(user)


@router.post("/projects")
async def create_project(body: ProjectIn,
                         user: dict = Depends(require_role("Manager"))):
    p = pm_service.create_project(user, body.name, body.space_id, body.board)
    await manager.broadcast({"type": "project.created", "project": p})
    return p


@router.get("/projects")
def projects(user: dict = Depends(current_user)):
    return pm_service.list_projects(user)


@router.post("/projects/{project_id}/members")
async def add_member(project_id: str, body: MemberAddIn,
                     user: dict = Depends(require_role("Manager"))):
    p = pm_service.add_member(user, project_id, body.user_id, body.role)
    await manager.broadcast({"type": "project.member_added", "project": p})
    return p


@router.post("/projects/{project_id}/tasks")
async def create_task(project_id: str, body: TaskIn,
                      user: dict = Depends(current_user)):
    t = pm_service.create_task(user, project_id, body.title, body.description,
                               body.assignee_id, body.due_date, body.priority,
                               body.labels, body.start_date)
    await manager.broadcast({"type": "task.created", "task": t})
    return t


@router.get("/projects/{project_id}/tasks")
def project_tasks(project_id: str, user: dict = Depends(current_user)):
    return pm_service.list_tasks(user, project_id=project_id)


@router.get("/tasks/mine")
def my_tasks(user: dict = Depends(current_user)):
    return pm_service.list_tasks(assignee_id=user["sub"])


@router.patch("/tasks/{task_id}")
async def patch_task(task_id: str, body: TaskPatch,
                     user: dict = Depends(current_user)):
    t = pm_service.patch_task(user, task_id, **body.model_dump())
    await manager.broadcast({"type": "task.updated", "task": t})
    return t


@router.patch("/tasks/{task_id}/sprint")
async def move_task(task_id: str, sprint_id: str | None = None,
                    user: dict = Depends(current_user)):
    t = pm_service.move_task_to_sprint(user, task_id, sprint_id)
    await manager.broadcast({"type": "task.updated", "task": t})
    return t


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, user: dict = Depends(current_user)):
    t = pm_service.delete_task(user, task_id)
    await manager.broadcast({"type": "task.deleted", "task": t})
    return t


@router.post("/tasks/{task_id}/subtasks")
async def create_subtask(task_id: str, body: SubtaskIn,
                         user: dict = Depends(current_user)):
    st = pm_service.create_subtask(user, task_id, body.title)
    await manager.broadcast({"type": "subtask.created", "subtask": st})
    return st


@router.get("/tasks/{task_id}/subtasks")
def task_subtasks(task_id: str, _: dict = Depends(current_user)):
    return pm_service.list_subtasks(task_id)


@router.post("/projects/{project_id}/sprints")
async def create_sprint(project_id: str, body: SprintIn,
                        user: dict = Depends(require_role("Manager"))):
    s = pm_service.create_sprint(user, project_id, body.name)
    await manager.broadcast({"type": "sprint.created", "sprint": s})
    return s


@router.get("/projects/{project_id}/sprints")
def sprints(project_id: str, _: dict = Depends(current_user)):
    return pm_service.list_sprints(project_id)


@router.post("/sprints/{sprint_id}/start")
async def start_sprint(sprint_id: str,
                       user: dict = Depends(require_role("Manager"))):
    s = pm_service.start_sprint(user, sprint_id)
    await manager.broadcast({"type": "sprint.updated", "sprint": s})
    return s


@router.post("/sprints/{sprint_id}/complete")
async def complete_sprint(sprint_id: str,
                          user: dict = Depends(require_role("Manager"))):
    s = pm_service.complete_sprint(user, sprint_id)
    await manager.broadcast({"type": "sprint.updated", "sprint": s})
    return s


@router.delete("/projects/{project_id}")
async def archive_project(project_id: str,
                          user: dict = Depends(require_role("Admin"))):
    p = pm_service.archive_project(user, project_id)
    await manager.broadcast({"type": "project.archived", "project": p})
    return p
