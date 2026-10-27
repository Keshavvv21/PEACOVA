"""Demo data: users across all four roles, the Compass OS space/project,
Sprint 1, and tasks (one overdue) — enough for every worked example in
the design's §7 and the manual's checkpoints."""
from __future__ import annotations

import datetime as dt

from .auth import hash_pw
from .db import store
from .retrieval.vectorstore import vector_index
from .services import pm_service

USERS = [
    ("aniruddha", "Aniruddha", "Executive"),
    ("meera", "Meera", "Admin"),
    ("rahul", "Rahul", "Manager"),
    ("priya", "Priya", "Member"),
]


def seed(force: bool = False):
    if store("Users").get("USER#aniruddha", "PROFILE") and not force:
        return False
    for uid, name, role in USERS:
        store("Users").put(f"USER#{uid}", "PROFILE", {
            "id": uid, "name": name, "role": role,
            "password_hash": hash_pw("demo123")})
    exec_user = {"sub": "aniruddha", "name": "Aniruddha", "role": "Executive"}
    sp = pm_service.create_space(exec_user, "Product")
    proj = pm_service.create_project(exec_user, "Compass OS", sp["id"], "Scrum")
    proj["members"] += ["rahul", "priya"]
    store("Projects").put(f"PROJECT#{proj['id']}", "META", proj)
    s1 = pm_service.create_sprint(exec_user, proj["id"], "Sprint 1")
    pm_service.start_sprint(exec_user, s1["id"])

    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    week_ago = (dt.date.today() - dt.timedelta(days=8)).isoformat()
    next_week = (dt.date.today() + dt.timedelta(days=7)).isoformat()
    # Planned start->due is 7 days; it's now 8 days in and still open — a
    # deliberate overrun so task_duration_check has something to report on
    # out of the box, same reasoning as due_date=yesterday for "overdue".
    t1 = pm_service.create_task(exec_user, proj["id"], "Android login crash",
                                "App crashes on login screen on Android 14",
                                assignee_id="priya", due_date=yesterday,
                                start_date=week_ago)
    pm_service.move_task_to_sprint(exec_user, t1["id"], s1["id"])
    pm_service.create_task(exec_user, proj["id"], "Payment gateway timeout",
                           "Checkout intermittently times out on UPI",
                           assignee_id="priya", due_date=next_week)
    pm_service.create_task(exec_user, proj["id"], "Dark mode for dashboard",
                           "Design and implement dark theme",
                           assignee_id="rahul", due_date=next_week)
    vector_index.reindex_all()
    return True
