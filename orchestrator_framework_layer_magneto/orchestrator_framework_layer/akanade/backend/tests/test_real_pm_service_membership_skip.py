"""Regression tests for the "not a member of this project" crash found by
a live end-to-end run of every prompt in CHAT_PROMPTS.md against a real
seeded backend: an Admin can LIST every org project (GET /api/projects)
but isn't necessarily a MEMBER of each one, and the real app 403s
/api/projects/{id}/tasks and /api/projects/{id}/sprints for a non-member
(see tool_layer/backend/app/core/deps.py's require_project_membership).

Before this fix, list_all_tasks/find_sprint_project_id/resolve_sprint_real
fanned out over every listed project and let the FIRST 403 raise
unhandled — crashing "Move X to in progress", "Why is X taking long?",
task-duration checks, "Delete the task X", and any org-wide sprint
summary as a real 500, never reaching execute()'s own try/except (name
resolution happens earlier, in resolve_arguments). The fix: skip a 403'd
project rather than let it poison every other project's data.
"""
import pytest

from app.agent import tools
from app.services import real_pm_service


PROJECTS = [{"project_id": "p-ok", "name": "OK Project", "methodology": "kanban"},
            {"project_id": "p-forbidden", "name": "Forbidden Project", "methodology": "scrum"}]


def test_list_all_tasks_skips_a_403_project_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(real_pm_service, "list_projects", lambda token: PROJECTS)

    def fake_list_tasks(token, project_id):
        if project_id == "p-forbidden":
            raise RuntimeError("403: Not a member of this project")
        return [{"task_id": "t1", "title": "Fix login bug", "project_id": project_id}]

    monkeypatch.setattr(real_pm_service, "list_tasks_for_project", fake_list_tasks)

    tasks = real_pm_service.list_all_tasks("tok")
    assert [t["task_id"] for t in tasks] == ["t1"]


def test_list_all_tasks_still_raises_a_non_403_error(monkeypatch):
    monkeypatch.setattr(real_pm_service, "list_projects", lambda token: PROJECTS)

    def fake_list_tasks(token, project_id):
        if project_id == "p-forbidden":
            raise RuntimeError("500: Internal Server Error")
        return []

    monkeypatch.setattr(real_pm_service, "list_tasks_for_project", fake_list_tasks)

    with pytest.raises(RuntimeError, match="500"):
        real_pm_service.list_all_tasks("tok")


def test_find_sprint_project_id_skips_a_403_project(monkeypatch):
    monkeypatch.setattr(real_pm_service, "list_projects", lambda token: PROJECTS)

    def fake_list_sprints(token, project_id):
        if project_id == "p-forbidden":
            raise RuntimeError("403: Not a member of this project")
        return [{"sprint_id": "s1", "project_id": project_id}]

    monkeypatch.setattr(real_pm_service, "list_sprints_for_project", fake_list_sprints)

    assert real_pm_service.find_sprint_project_id("tok", "s1") == "p-ok"


def test_resolve_sprint_real_skips_a_403_project(monkeypatch):
    monkeypatch.setattr(real_pm_service, "list_projects", lambda token: PROJECTS)

    def fake_list_sprints(token, project_id):
        if project_id == "p-forbidden":
            raise RuntimeError("403: Not a member of this project")
        return [{"sprint_id": "s1", "name": "Sprint 1", "project_id": project_id}]

    monkeypatch.setattr(real_pm_service, "list_sprints_for_project", fake_list_sprints)

    res = tools.resolve_sprint_real("tok", "Sprint 1")
    assert res.id == "s1"
