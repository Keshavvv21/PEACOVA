"""The complete, 24-step (+2 setup) chat bot user journey — one project,
start to finish — run for real against the native track's demo store,
exactly as a person typing into the widget would experience it in one
continuous session. This is deliberately NOT parametrized across tracks
(test_agent_flows.py already proves native/langgraph/adk answer
identically for each individual capability) — its job is different:
proving a long, REALISTIC, stateful conversation holds together end to
end, including the parts that only fail when state leaks between turns
(sprint-name collisions across projects, a declined feature swallowing
the next unrelated message) — bugs that a suite of independent
single-turn tests, however thorough, structurally cannot catch.

Both bugs below were found by actually running this exact journey
before this test existed:

  1. move_task_to_sprint resolved a sprint NAME globally instead of
     scoped to the task's own project — two same-named sprints in
     different projects ("Sprint 1" is a common name) made an
     unambiguous move ambiguous. Fixed in native_agent.py by passing
     the task's own project_id into resolve_sprint.
  2. A tool declined as "not available in this demo" (task/project
     messages, personal notes, roles) returned a "clarify" shape with
     no real follow-up field to fill in — which left pendingClarification
     set, silently swallowing the user's NEXT, completely unrelated
     message as if it were answering a question nobody actually asked.
     Fixed across all four tracks by treating a None clarify field as a
     terminal decline (end the turn, no pending state) rather than a
     real follow-up.

Setup creates "Priya Nair" and "Rohan Sharma" as real users first (the
seeded demo persona is only "Priya"/"Rahul" — first names — so the
journey's own full names need to exist before space/project creation
references them).
"""
from __future__ import annotations

from tests.conftest import auth


def say(client, tok, msg, session):
    r = client.post("/api/chat/message?framework=native",
                    json={"message": msg, "session_id": session},
                    headers=auth(tok))
    assert r.status_code == 200, r.text
    return r.json()


def test_complete_user_journey_one_project_start_to_finish(client, exec_token):
    session = "full-journey"

    def step(msg):
        return say(client, exec_token, msg, session)

    # ── setup: the two named people the journey references ──────────────
    d = step("Create a new user named Priya Nair as a Manager")
    assert "Created user" in d["reply"] and "Priya Nair" in d["reply"]
    d = step("Create a new user named Rohan Sharma as a Member")
    assert "Created user" in d["reply"] and "Rohan Sharma" in d["reply"]

    # ── space, project, membership ───────────────────────────────────────
    d = step("Create a space called Product Launch Q1 with Priya Nair as senior manager")
    assert d["reply"] == "Created space “Product Launch Q1”."

    # NOTE: create_project/create_task don't quote-preserve original
    # casing the way create_space does (resolve_arguments applies
    # .title() to a new SPACE's name specifically) — when the offline
    # rule-based floor answers (no LLM configured, exactly this test
    # suite's hermetic setup), an UNQUOTED project/task name is captured
    # from the pattern-matcher's lowercased copy of the message. A real
    # LLM preserves the user's original casing natively (verified live
    # against a real provider before this test was written) — this is a
    # rule-based-floor-only quirk, so assertions below are
    # case-insensitive for names that aren't quoted in the prompt.
    d = step("Create a scrum project called Website Revamp in space Product Launch Q1 "
            "with Priya Nair as senior manager and Rohan Sharma as team lead")
    assert d["reply"].lower() == "created project “website revamp”."

    d = step("Add Rohan Sharma as member to project Website Revamp")
    assert "Added Rohan Sharma to" in d["reply"] and "Member" in d["reply"]

    # ── tasks & subtasks ──────────────────────────────────────────────────
    # Rule-based-floor-compatible phrasing here (title + project + due
    # date via "with due date") — the richer one-sentence phrasing
    # ("assign it to X, priority Y, due Z, starting W" all together)
    # needs a real LLM to parse and is verified separately in
    # test_create_task_supports_assignee_priority_and_start_date below,
    # which exercises resolve_arguments/tools.execute directly rather
    # than depending on rule-based regex coverage for a sentence shape
    # it was never designed to understand.
    d = step("Create a task called Fix login bug in Website Revamp with due date 2026-08-28")
    assert "created task" in d["reply"].lower() and "fix login bug" in d["reply"].lower()
    assert "2026-08-28" in d["reply"]
    d = step("Assign Fix login bug to Rohan Sharma")
    assert "assigned" in d["reply"].lower() and "rohan sharma" in d["reply"].lower()

    # Rule-based-floor phrasing is "subtask CALLED x TO y" (title before
    # the task name) — same real-LLM-vs-offline-floor distinction as above.
    d = step("Add a subtask called Write unit tests covering token refresh and "
             "expiry on the login screen to Fix login bug")
    assert "added subtask" in d["reply"].lower()
    assert "write unit tests covering token refresh and expiry" in d["reply"].lower()

    d = step("Create a task called Doomed cleanup demo in Website Revamp with due date 2026-08-22")
    assert "created task" in d["reply"].lower() and "doomed cleanup demo" in d["reply"].lower()

    # ── sprint ────────────────────────────────────────────────────────────
    d = step("Create a sprint called Sprint 1 in Website Revamp")
    assert d["reply"].lower() == "created sprint “sprint 1”."

    d = step("Start Sprint 1 in Website Revamp")
    assert "started" in d["reply"].lower() and "sprint 1" in d["reply"].lower()

    d = step("Move Fix login bug to Sprint 1")
    # Regression guard for bug #1 above: this must resolve UNAMBIGUOUSLY
    # even though the seeded Compass OS project also has its own
    # "Sprint 1" — scoped resolution means this task's own project's
    # sprint is the only candidate.
    assert d["pending_confirmation"] is False
    assert "couldn't" not in d["reply"].lower()
    assert "more than one" not in d["reply"].lower()
    assert "moved" in d["reply"].lower() and "sprint 1" in d["reply"].lower()

    # ── messaging & notes (demo-mode: real-app-only, decline gracefully) ─
    # Rule-based-floor phrasing again ("message TO/ON task X saying Y" —
    # not "message THE task X saying Y", which a real LLM parses fine
    # but the offline pattern-matcher doesn't); the decline fires
    # regardless of the exact captured text since it happens before any
    # entity resolution.
    d = step("Send a message to task Fix login bug saying I've pushed a fix, please "
             "review by end of day 2026-08-27, and mention Priya Nair")
    assert "real-app-only" in d["reply"]
    assert d["pending_confirmation"] is False

    d = step("Send a message to the Website Revamp project chat saying the deploy is "
             "scheduled for Friday 2026-08-28, please have your PRs merged by "
             "Thursday 2026-08-27, and mention Rohan Sharma")
    assert "real-app-only" in d["reply"]

    d = step("Save a note: root cause of the login bug was a stale JWT cache that "
             "wasn't invalidated on token refresh, discovered 2026-08-21")
    assert "real-app-only" in d["reply"]

    # ── regression guard for bug #2: the declines above must NOT have left
    # a pendingClarification behind — this next, totally unrelated
    # question must be answered on its own merits, not misread as a
    # follow-up answer to any of the three declines just above. ─────────
    d = step("What tasks are overdue in Website Revamp?")
    assert "couldn't find a space" not in d["reply"].lower()
    assert "nothing is overdue" in d["reply"].lower() or "overdue" in d["reply"].lower()

    d = step("Who has the most tasks right now?")
    assert "couldn't find a space" not in d["reply"].lower()

    d = step("Why is Fix login bug taking so long?")
    assert "couldn't find a space" not in d["reply"].lower()
    # As of the journey's reference date the task hasn't started yet —
    # never a crash, never a guess.
    assert d["pending_confirmation"] is False

    d = step("Summary of Website Revamp")
    assert "couldn't find a space" not in d["reply"].lower()
    assert "website revamp" in d["reply"].lower()

    # ── wrapping the work up ─────────────────────────────────────────────
    d = step("Mark Fix login bug as done")
    assert "done" in d["reply"].lower()

    d = step("Delete the task Doomed cleanup demo")
    assert d["pending_confirmation"] is True
    assert "sure" in d["reply"].lower()
    d = step("yes")
    assert "deleted" in d["reply"].lower()

    # Qualified with "in Website Revamp" — this test suite's seeded demo
    # data (seed.py) ALSO creates a sprint literally named "Sprint 1" in
    # the Compass OS project, so an unqualified "Complete Sprint 1" is
    # genuinely ambiguous here (correct behavior: it asks which one,
    # exactly like move_task_to_sprint would if it weren't scoped to a
    # task's own project — start_sprint/complete_sprint's scoping is
    # via an explicit, optional "project" argument instead, so naming it
    # is this test's job, not a bug to fix).
    d = step("Complete Sprint 1 in Website Revamp")
    assert d["pending_confirmation"] is True
    d = step("yes")
    assert "completed" in d["reply"].lower() and "sprint 1" in d["reply"].lower()

    # ── end: delete everything, back to pre-walkthrough state ───────────
    # KNOWN GAP (found by this test, not fixed here): there is no
    # delete_space tool in TOOL_SCHEMAS at all — space deletion (with its
    # cascading remove-everything-underneath behavior) isn't exposed to
    # chat yet, only via the real app's own UI, which requires typing the
    # exact space name to confirm — a materially different, riskier
    # confirmation shape than every other destructive tool's plain
    # yes/no, and not something to add in a rush. The bot declines this
    # gracefully (no crash, no hallucinated tool call) rather than
    # pretending to have deleted anything.
    d = step("Delete the space Product Launch Q1")
    assert d["pending_confirmation"] is False
    # Never misread as an attempt to delete a TASK named "the space
    # Product Launch Q1" (the regression this test guards against —
    # see the delete_task catch-all's negative lookahead fix) — the
    # exact decline wording differs between a real LLM (which explains
    # itself in its own words) and the offline rule-based floor (a
    # fixed capability list), so only the ABSENCE of a wrong task-shaped
    # reply is asserted, not one specific phrasing.
    assert "which task" not in d["reply"].lower()
    assert "the space" not in d["reply"].lower()

    d = step("Delete the user account for Priya Nair")
    assert d["pending_confirmation"] is True
    d = step("yes")
    assert "deleted" in d["reply"].lower()

    d = step("Delete the user account for Rohan Sharma")
    assert d["pending_confirmation"] is True
    d = step("yes")
    assert "deleted" in d["reply"].lower()


def test_create_task_supports_assignee_priority_and_start_date():
    """The richer one-sentence phrasing the full journey above uses
    against a real LLM ("assign it to X, priority Y, due Z, starting W",
    all in the SAME create_task call) needs real natural-language
    understanding the offline rule-based floor was never designed to
    provide — so this exercises the actual code path (resolve_arguments
    → tools.execute) directly with already-parsed arguments, the same
    shape a real LLM's tool call produces, proving the NEW create_task
    fields (this session's own extension — previously only title/
    project/description/due_date were wired through) actually work."""
    from app import seed
    from app.agent import native_agent, tools
    from app.services import pm_service

    seed.seed()
    exec_user = {"sub": "aniruddha", "name": "Aniruddha", "role": "Executive"}

    resolved, clarify = native_agent.resolve_arguments(exec_user, "create_task", {
        "title": "Fix login bug", "project": "Compass OS", "priority": "high",
        "assignee": "rahul", "due_date": "2026-08-28", "start_date": "2026-08-21",
    })
    assert clarify is None, clarify
    assert resolved["priority"] == "High"
    assert resolved["due_date"] == "2026-08-28"
    assert resolved["start_date"] == "2026-08-21"
    assert "assignee_id" in resolved

    result = tools.execute(exec_user, "create_task", resolved)
    assert result["ok"] is True
    stored = pm_service.get_task(result["id"])
    assert stored["priority"] == "High"
    assert stored["start_date"] == "2026-08-21"
    assert stored["due_date"] == "2026-08-28"
    assert stored["assignee_id"] == resolved["assignee_id"]
