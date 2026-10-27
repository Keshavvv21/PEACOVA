"""Regression test for real-mode create_project silently 400ing.

The real app's own ProjectService requires an Admin creator to name a
Senior Manager (Manager2) as project_owner, and an Admin or Manager2
creator to additionally name a Team Lead (Manager1) — see
backend/app/services/project_service.py's _ROLES_REQUIRING_PROJECT_OWNER/
_ROLES_REQUIRING_TEAM_LEAD on the real app's side. Before this fix,
resolve_arguments's create_project branch never checked this, so a real
Admin/Manager2 user just saying "create a project called X in space Y"
sailed past resolution only to 400 against the real API — surfacing to
the chat as a flat "Sorry — that failed", not a follow-up question.

This can't be exercised through the normal say()-over-HTTP tests in
test_agent_flows.py: those run in demo mode (REAL_APP_API_BASE_URL is
forced empty for the whole test session — see conftest.py, "Real-key
testing is a deliberate, separate, by-hand step"). Real mode is
therefore tested here at the resolve_arguments unit level instead,
faking is_real_mode/resolve_space_real/resolve_user_real so no real
HTTP call is made.
"""
from app.agent import native_agent, tools


def _user(role: str) -> dict:
    # role is already mapped through auth._REAL_ROLE_MAP: real Admin ->
    # "Executive", real Manager2 -> "Admin", real Manager1 -> "Manager".
    return {"sub": "u1", "name": "Test User", "role": role, "_real_token": "tok"}


def test_admin_creator_without_manager_is_asked_for_one(monkeypatch):
    monkeypatch.setattr(tools, "is_real_mode", lambda user: True)
    monkeypatch.setattr(tools, "resolve_space_real", lambda token, name: tools.Resolution(id="sp1"))

    resolved, clarify = native_agent.resolve_arguments(
        _user("Executive"), "create_project", {"name": "New Site", "space": "Engineering"})

    assert resolved is None
    assert clarify["field"] == "manager"
    assert "senior manager" in clarify["question"].lower()


def test_manager2_creator_without_team_lead_is_asked_for_one(monkeypatch):
    monkeypatch.setattr(tools, "is_real_mode", lambda user: True)
    monkeypatch.setattr(tools, "resolve_space_real", lambda token, name: tools.Resolution(id="sp1"))

    resolved, clarify = native_agent.resolve_arguments(
        _user("Admin"), "create_project", {"name": "New Site", "space": "Engineering"})

    assert resolved is None
    assert clarify["field"] == "team_lead"
    assert "team lead" in clarify["question"].lower()


def test_manager1_creator_needs_neither(monkeypatch):
    monkeypatch.setattr(tools, "is_real_mode", lambda user: True)
    monkeypatch.setattr(tools, "resolve_space_real", lambda token, name: tools.Resolution(id="sp1"))

    resolved, clarify = native_agent.resolve_arguments(
        _user("Manager"), "create_project", {"name": "New Site", "space": "Engineering"})

    assert clarify is None
    assert resolved == {"name": "New Site", "space_id": "sp1", "board": "Kanban"}


def test_admin_creator_who_named_a_manager_is_then_asked_for_team_lead(monkeypatch):
    # Naming a Senior Manager satisfies the first check, but an Admin
    # creator still owes the real API a Team Lead too — the two required
    # fields are independent, not one-and-done.
    monkeypatch.setattr(tools, "is_real_mode", lambda user: True)
    monkeypatch.setattr(tools, "resolve_space_real", lambda token, name: tools.Resolution(id="sp1"))
    monkeypatch.setattr(tools, "resolve_user_real", lambda token, name: tools.Resolution(id="mgr1"))

    resolved, clarify = native_agent.resolve_arguments(
        _user("Executive"), "create_project",
        {"name": "New Site", "space": "Engineering", "manager": "Priya"})

    assert resolved is None
    assert clarify["field"] == "team_lead"


def test_admin_creator_who_named_both_proceeds(monkeypatch):
    monkeypatch.setattr(tools, "is_real_mode", lambda user: True)
    monkeypatch.setattr(tools, "resolve_space_real", lambda token, name: tools.Resolution(id="sp1"))
    monkeypatch.setattr(tools, "resolve_user_real", lambda token, name: tools.Resolution(id=f"id-{name}"))

    resolved, clarify = native_agent.resolve_arguments(
        _user("Executive"), "create_project",
        {"name": "New Site", "space": "Engineering", "manager": "Priya", "team_lead": "Sam"})

    assert clarify is None
    assert resolved == {
        "name": "New Site", "space_id": "sp1", "board": "Kanban",
        "manager_id": "id-Priya", "team_lead_id": "id-Sam",
    }


def test_demo_mode_never_requires_manager_or_team_lead(monkeypatch):
    # is_real_mode defaults to False here (no _real_token) — the demo
    # store's own create_project has no such rule, so an Executive-role
    # demo user must NOT be blocked. resolve_space is faked so this test
    # doesn't depend on the demo store having been seeded yet (session
    # ordering with the other test modules) or on any particular space
    # existing in it.
    monkeypatch.setattr(tools, "resolve_space", lambda name, user=None: tools.Resolution(id="sp1"))
    resolved, clarify = native_agent.resolve_arguments(
        {"sub": "u1", "name": "Demo Exec", "role": "Executive"},
        "create_project", {"name": "New Site", "space": "Product"})

    assert clarify is None
    assert resolved is not None
    assert "manager_id" not in resolved
    assert "team_lead_id" not in resolved
