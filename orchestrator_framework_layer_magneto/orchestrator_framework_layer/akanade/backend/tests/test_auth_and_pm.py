from tests.conftest import auth


def test_login_rejects_bad_password(client):
    r = client.post("/api/auth/login",
                    json={"username": "aniruddha", "password": "wrong"})
    assert r.status_code == 401


def test_member_cannot_create_space(client, member_token):
    r = client.post("/api/spaces", json={"name": "Nope"},
                    headers=auth(member_token))
    assert r.status_code == 403


def test_task_crud_and_due_date_audit(client, exec_token):
    pid = client.get("/api/projects", headers=auth(exec_token)).json()[0]["id"]
    t = client.post(f"/api/projects/{pid}/tasks",
                    json={"title": "Audit me"}, headers=auth(exec_token)).json()
    r = client.patch(f"/api/tasks/{t['id']}",
                     json={"due_date": "2026-09-01"}, headers=auth(exec_token))
    audit = r.json()["audit"]
    assert audit and audit[-1]["field"] == "due_date"
    assert audit[-1]["to"] == "2026-09-01"


def test_subtask_create_and_list(client, exec_token):
    pid = client.get("/api/projects", headers=auth(exec_token)).json()[0]["id"]
    t = client.post(f"/api/projects/{pid}/tasks",
                    json={"title": "Has subtasks"}, headers=auth(exec_token)).json()
    st = client.post(f"/api/tasks/{t['id']}/subtasks",
                     json={"title": "Check logs"}, headers=auth(exec_token))
    assert st.status_code == 200 and st.json()["title"] == "Check logs"
    listed = client.get(f"/api/tasks/{t['id']}/subtasks", headers=auth(exec_token)).json()
    assert any(s["title"] == "Check logs" for s in listed)


def test_create_user_admin_only(client, exec_token, member_token):
    r = client.post("/api/users", json={"name": "Rest User", "role": "Manager"},
                    headers=auth(member_token))
    assert r.status_code == 403

    r = client.post("/api/users", json={"name": "Rest User", "role": "Manager",
                                        "email": "rest.user@example.com"},
                    headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["role"] == "Manager"
    assert r.json()["email"] == "rest.user@example.com"
    login = client.post("/api/auth/login",
                        json={"username": "rest.user", "password": "demo123"})
    assert login.status_code == 200 and login.json()["role"] == "Manager"


def test_admin_without_membership_cannot_see_projects_spaces_or_tasks(client, admin_token, exec_token):
    # Confidentiality fix: the demo store's "Admin" role (mapped from the
    # real app's Manager2/Senior Manager, NOT its own top rank "Executive")
    # is not automatically a member of every project — meera isn't in the
    # seeded Compass OS project, so she shouldn't see it, its space, or its
    # tasks, even though she can still see the overall (empty-to-her)
    # projects/spaces LIST endpoints without a 403.
    projects = client.get("/api/projects", headers=auth(admin_token)).json()
    assert all(p["name"] != "Compass OS" for p in projects)

    spaces = client.get("/api/spaces", headers=auth(admin_token)).json()
    assert all(s["name"] != "Product" for s in spaces)

    pid = client.get("/api/projects", headers=auth(exec_token)).json()[0]["id"]
    r = client.get(f"/api/projects/{pid}/tasks", headers=auth(admin_token))
    assert r.status_code == 403


def test_member_of_project_still_sees_its_project_space_and_tasks(client, member_token):
    projects = client.get("/api/projects", headers=auth(member_token)).json()
    assert any(p["name"] == "Compass OS" for p in projects)

    spaces = client.get("/api/spaces", headers=auth(member_token)).json()
    assert any(s["name"] == "Product" for s in spaces)

    pid = next(p for p in projects if p["name"] == "Compass OS")["id"]
    r = client.get(f"/api/projects/{pid}/tasks", headers=auth(member_token))
    assert r.status_code == 200


def test_complete_sprint_moves_incomplete_to_backlog(client, exec_token):
    pid = client.get("/api/projects", headers=auth(exec_token)).json()[0]["id"]
    s = client.post(f"/api/projects/{pid}/sprints", json={"name": "Sprint T"},
                    headers=auth(exec_token)).json()
    t = client.post(f"/api/projects/{pid}/tasks", json={"title": "Unfinished"},
                    headers=auth(exec_token)).json()
    client.patch(f"/api/tasks/{t['id']}/sprint?sprint_id={s['id']}",
                 headers=auth(exec_token))
    client.post(f"/api/sprints/{s['id']}/complete", headers=auth(exec_token))
    t2 = [x for x in client.get(f"/api/projects/{pid}/tasks",
                                headers=auth(exec_token)).json()
          if x["id"] == t["id"]][0]
    assert t2["sprint_id"] is None and t2["status"] == "Backlog"
