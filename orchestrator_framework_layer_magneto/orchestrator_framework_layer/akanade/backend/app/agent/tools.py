"""§8 — the shared tool layer: one Python function per tool, used by every
track (native reference, LangGraph, ADK) through thin adapters.

Each tool is a translation layer only: resolve names → real IDs (never
LLM-invented, §6.1 step 7), call the SAME service functions the REST
router calls, as the CALLING USER (§11), return a compact result.

DESTRUCTIVE is the fixed, hard-coded set (§6.1 step 9) — a constant,
never the LLM's judgement.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process

from ..config import settings
from ..plugins import PLUGINS, SERVICE_METHOD_REGISTRY
from ..retrieval.graph import knowledge_graph
from ..retrieval.vectorstore import vector_index
from ..services import real_pm_service
from . import google_tools, mcp_client, peft_training_tool, trading_tools, user_mcp_client
from ..services import pm_service

DESTRUCTIVE: frozenset[str] = frozenset(
    {"delete_task", "archive_project", "complete_sprint", "delete_user"}
    | {name for name, p in PLUGINS.items() if p.destructive}
)

# change_task_status's enum (this design's PascalCase, matching the demo
# store) -> the real app's own TaskStatus values (lowercase snake_case,
# see backend/app/models/task.py). "blocked"/"hold" are real states this
# demo-shaped enum has no equivalent for — not reachable from chat today.
_STATUS_TO_REAL: dict[str, str] = {
    "Todo": "todo", "InProgress": "in_progress", "Backlog": "backlog", "Done": "done",
}

# OpenAI-function-style schemas — bound to perceive_intent's LLM call and
# used by the output guard. One source of truth for every track.
TOOL_SCHEMAS: dict[str, dict] = {
    "create_space": {"description": "Create a Space",
        "required": ["name"],
        "properties": {"name": {"type": "string"},
                       "manager": {"type": "string",
                                  "description": "The Senior Manager who supervises "
                                                 "this space, only if the user named "
                                                 "one — some creators are required to "
                                                 "name one, the app will ask if so."}}},
    "create_project": {"description": "Create a Project in a Space",
        "required": ["name", "space"],
        "properties": {"name": {"type": "string"}, "space": {"type": "string"},
                       "board": {"type": "string"},
                       "manager": {"type": "string",
                                  "description": "The Senior Manager who supervises "
                                                 "this project, only if named."},
                       "team_lead": {"type": "string",
                                    "description": "The Manager who runs this "
                                                   "project day-to-day, only if named."}}},
    "create_task": {"description": "Create a Task in a Project — optionally "
                                   "assigning it, setting its priority, and/or "
                                   "its start date in the SAME call, all in one turn.",
        "required": ["title", "project"],
        "properties": {
            "title": {"type": "string",
                     "description": "The task's full title verbatim, exactly as "
                                    "the user phrased it — never shorten or drop "
                                    "words from it, even if a due date or project "
                                    "name follows it in the same sentence."},
            "project": {"type": "string"},
            "description": {"type": "string"},
            "due_date": {"type": "string",
                        "description": "ISO format YYYY-MM-DD, only if the user "
                                       "gave a due date — omit otherwise."},
            "start_date": {"type": "string",
                          "description": "ISO format YYYY-MM-DD, only if the user "
                                         "gave a start date (e.g. 'starting "
                                         "2026-08-21') — omit otherwise."},
            "assignee": {"type": "string",
                        "description": "Who to assign the new task to, only if "
                                       "the user named someone in the same "
                                       "request — omit otherwise (use the "
                                       "separate assign_task tool to assign an "
                                       "EXISTING task later)."},
            "priority": {"type": "string", "enum": ["Low", "Medium", "High"],
                        "description": "Only if the user named one — omit to "
                                       "leave it at the default (Medium)."}},
    },
    "create_subtask": {"description": "Create a subtask under a task",
        "required": ["title", "task"],
        "properties": {"title": {"type": "string"}, "task": {"type": "string"},
                       "due_date": {"type": "string",
                                   "description": "ISO format YYYY-MM-DD, only "
                                                  "if the user gave one."}}},
    "list_subtasks": {"description": "List a task's subtasks (checklist items) "
                                     "and whether each is done",
        "required": ["task"],
        "properties": {"task": {"type": "string"}}},
    "complete_subtask": {"description": "Mark a task's subtask done, or reopen "
                                        "it as not done",
        "required": ["task", "subtask"],
        "properties": {"task": {"type": "string"}, "subtask": {"type": "string"},
                       "done": {"type": "boolean",
                                "description": "true to mark it done (the default "
                                               "— use this whenever the user says "
                                               "\"mark done\"/\"complete\"/\"finished\" "
                                               "without saying otherwise), false to "
                                               "reopen/undo it."}}},
    "create_user": {"description": "Provision a new user account — Admin only",
        "required": ["name", "role"],
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "email": {"type": "string",
                     "description": "Optional — only if the user gave one. "
                                    "Stored on the profile; this demo realm "
                                    "still logs in by username, not email."},
            "password": {"type": "string",
                        "description": "Optional — only if the user explicitly "
                                       "gave one. Defaults to demo123 otherwise."}},
    },
    "add_member": {"description": "Add a user to a project with a project role "
                                  "(Owner/CoOwner/Manager/Member)",
        "required": ["user", "project"],
        "properties": {"user": {"type": "string"}, "project": {"type": "string"},
                       "role": {"type": "string"}}},
    "delete_user": {"description": "Permanently delete a user account — "
                                   "Admin only, DESTRUCTIVE. Their tasks are "
                                   "unassigned (not deleted), project "
                                   "memberships removed, personal tasks "
                                   "deleted.",
        "required": ["user"], "properties": {"user": {"type": "string"}}},
    "assign_task": {"description": "Assign a task to a user",
        "required": ["task", "assignee"],
        "properties": {"task": {"type": "string"}, "assignee": {"type": "string"}}},
    "set_due_date": {"description": "Set/change a task's due date (audited)",
        "required": ["task", "due_date"],
        "properties": {"task": {"type": "string"}, "due_date": {"type": "string"}}},
    "change_task_status": {"description": "Change a task's status — use this "
                                          "for 'mark X as done/in progress/todo', "
                                          "'move X to backlog', etc.",
        "required": ["task", "status"],
        "properties": {"task": {"type": "string"},
                       "status": {"type": "string",
                                  "enum": ["Todo", "InProgress", "Backlog", "Done"]}}},
    "move_task_to_sprint": {"description": "Move a task into a sprint",
        "required": ["task", "sprint"],
        "properties": {"task": {"type": "string"}, "sprint": {"type": "string"}}},
    "create_sprint": {"description": "Create a sprint in a project",
        "required": ["name", "project"],
        "properties": {"name": {"type": "string"}, "project": {"type": "string"}}},
    "start_sprint": {"description": "Start a sprint",
        "required": ["sprint"],
        "properties": {"sprint": {"type": "string"}, "project": {"type": "string"}}},
    "complete_sprint": {"description": "Complete (end) a sprint — DESTRUCTIVE",
        "required": ["sprint"],
        "properties": {"sprint": {"type": "string"}, "project": {"type": "string"}}},
    "delete_task": {"description": "Delete (trash) a task — DESTRUCTIVE",
        "required": ["task"], "properties": {"task": {"type": "string"}}},
    "archive_project": {"description": "Archive a project — DESTRUCTIVE",
        "required": ["project"], "properties": {"project": {"type": "string"}}},
    "query_tasks": {"description": "List tasks — in one project, just the "
                                   "user's own ('my tasks', 'where are my "
                                   "tasks', 'where can I find my tasks'), "
                                   "or omit both for every task across "
                                   "every project ('list all tasks'). Also "
                                   "handles 'tasks tagged Python', 'tasks "
                                   "with the AWS tag'. A 'where' question "
                                   "about tasks is asking WHICH tasks/"
                                   "projects they're in, not a physical "
                                   "location — always resolve it with this "
                                   "tool, never as a location lookup.",
        "required": [], "properties": {"project": {"type": "string"},
                                       "mine": {"type": "boolean"},
                                       "technology": {"type": "string",
                                           "description": "Filter to tasks "
                                               "with this technology tag, "
                                               "only if the user named one "
                                               "— omit otherwise."}}},
    "list_projects": {"description": "List every project the user can see — use "
                                     "for 'list/show all projects', 'what projects "
                                     "are there'. Takes no arguments.",
        "required": [], "properties": {}},
    "project_summary": {"description": "Task-status breakdown, overdue count, "
                                       "and timeline (start/due date) for ONE "
                                       "named project — use for 'summary of "
                                       "project X', 'how's project X doing'. "
                                       "Omit `project` for a condensed summary "
                                       "across EVERY project the user can see — "
                                       "use for 'summary of all projects', "
                                       "'project timelines', 'give me an "
                                       "overview of everything'.",
        "required": [], "properties": {"project": {"type": "string"}}},
    "sprint_summary": {"description": "Per-sprint breakdown (status, dates, "
                                      "tasks done/total) for one project's "
                                      "sprints — use for 'summary of sprints "
                                      "in project X', 'summary of the last "
                                      "sprint', 'summary of the last 3 "
                                      "sprints'. `count` limits it to the N "
                                      "most recent sprints (1 for just 'the "
                                      "last sprint') — omit for every sprint. "
                                      "Omit `project` too for a summary across "
                                      "every scrum project's sprints.",
        "required": [], "properties": {
            "project": {"type": "string"},
            "count": {"type": "integer",
                     "description": "How many of the MOST RECENT sprints to "
                                    "include, only if the user asked for a "
                                    "specific number (e.g. 'last 3') or just "
                                    "'the last sprint' (count=1) — omit for "
                                    "all sprints."}}},
    "list_spaces": {"description": "List every space the user can see — use "
                                   "for 'list/show all spaces', 'what spaces "
                                   "are there'. Takes no arguments.",
        "required": [], "properties": {}},
    "list_roles": {"description": "List the org's roles — the 4 built-in "
                                  "roles (Member/Manager/Senior Manager/"
                                  "Admin) plus any custom roles an Admin "
                                  "has created, with their permissions. Use "
                                  "for 'what roles exist', 'list roles', "
                                  "'what custom roles are there'. Admin-"
                                  "only in real mode. Takes no arguments.",
        "required": [], "properties": {}},
    "graph_query": {"description": "Multi-hop question over the knowledge graph. "
                                   "Use this for 'what's overdue', 'who has the most "
                                   "tasks/workload/busiest', 'what tasks does X have', "
                                   "or 'summary of people who are overdue' — map the "
                                   "user's phrasing onto ONE of the four fixed "
                                   "question values below, never pass the user's raw "
                                   "wording as `question`.",
        "required": ["question"],
        "properties": {
            "question": {"type": "string",
                        "enum": ["overdue", "workload", "tasks_for", "overdue_by_person"],
                        "description": "overdue = the overdue tasks themselves; "
                                       "workload = who has the most OPEN tasks/busiest; "
                                       "overdue_by_person = counts of OVERDUE tasks per "
                                       "person (not all open tasks); tasks_for = tasks "
                                       "assigned to one named person (requires assignee)"},
            "scope": {"type": "string", "description": "project/space name to scope to, if any"},
            "assignee": {"type": "string", "description": "person's name, required for tasks_for"}},
    },
    "semantic_search": {"description": "Find tasks by meaning (vector search)",
        "required": ["query"], "properties": {"query": {"type": "string"}}},
    "list_users": {"description": "List people in the org, or look up one "
                                  "person — use for 'who is X', 'is X an "
                                  "admin', 'list users/people', 'who knows "
                                  "Python', 'people tagged AWS'.",
        "required": [], "properties": {
            "name": {"type": "string",
                    "description": "A specific person's name to look up, "
                                   "only if the user named one — omit to "
                                   "list everyone."},
            "technology": {"type": "string",
                          "description": "Filter to people with this skill/"
                                         "technology tag, only if the user "
                                         "named one (e.g. 'Python', "
                                         "'AWS') — omit otherwise. Never "
                                         "combine with `name`."}}},
    "task_info": {"description": "Look up one task's current details — use "
                                 "for 'who is working on X', 'X's assignee', "
                                 "'status of task X', 'details on task X'.",
        "required": ["task"], "properties": {"task": {"type": "string"}}},
    "task_duration_check": {"description": "Check a task's planned duration "
                                           "(due date minus start date) "
                                           "against how long it's actually "
                                           "taken/been running — use for "
                                           "'why is X taking so long', 'is X "
                                           "behind schedule', 'is X running "
                                           "over'. Omit `task` to list every "
                                           "task currently running longer "
                                           "than planned, optionally within "
                                           "one project — use for 'which "
                                           "tasks are taking too long', "
                                           "'what's running behind in "
                                           "project X'.",
        "required": [], "properties": {"task": {"type": "string"},
                                       "project": {"type": "string"}}},
    "message_task": {"description": "Post a message/comment on a task's "
                                    "discussion thread — use for 'tell X "
                                    "about task Y', 'message X on task Y', "
                                    "'update the thread on task Y'.",
        "required": ["task", "text"],
        "properties": {
            "task": {"type": "string"},
            "text": {"type": "string",
                    "description": "The message's full text verbatim, "
                                   "exactly as the user phrased it."},
            "mention": {"type": "string",
                       "description": "A person's name to @mention in the "
                                      "message, only if the user named one."}},
    },
    "message_project": {"description": "Post a message to a PROJECT's chat "
                                       "thread (not one specific task's) — "
                                       "use for 'post to the X project "
                                       "chat', 'send a message to the X "
                                       "project', 'tell the X team...', "
                                       "'update the X project chat'.",
        "required": ["project", "text"],
        "properties": {
            "project": {"type": "string"},
            "text": {"type": "string",
                    "description": "The message's full text verbatim, "
                                   "exactly as the user phrased it."},
            "mention": {"type": "string",
                       "description": "A person's name to @mention in the "
                                      "message, only if the user named one."}},
    },
    "list_personal_tasks": {"description": "List the user's own private "
                                           "to-do list — a personal task is "
                                           "never tied to a project. Use for "
                                           "'my personal tasks', 'what's on "
                                           "my personal to-do list'.",
        "required": [], "properties": {}},
    "save_note": {"description": "Save a private personal note for the "
                                 "user — never tied to a project. Use for "
                                 "'save this to my notes', 'note that X', "
                                 "'remember this: X', 'jot this down'. When "
                                 "the user says 'save THIS'/'save that', "
                                 "pull the actual content from what was just "
                                 "discussed earlier in this conversation — "
                                 "never pass the literal word 'this'.",
        "required": ["body"],
        "properties": {
            "title": {"type": "string",
                     "description": "Short title, only if the user gave "
                                    "one or one is obvious from context — "
                                    "omit otherwise."},
            "body": {"type": "string",
                    "description": "The note's full text, verbatim or "
                                   "summarized from context — never the "
                                   "literal word 'this'/'that'."}}},
    "move_personal_task": {"description": "Move a personal task to a "
                                          "different Eisenhower Matrix "
                                          "quadrant (urgent/important).",
        "required": ["task", "quadrant"],
        "properties": {
            "task": {"type": "string"},
            "quadrant": {"type": "string",
                        "enum": ["do_first", "schedule", "delegate", "eliminate"],
                        "description": "do_first = urgent+important; schedule = "
                                       "important, not urgent; delegate = urgent, "
                                       "not important; eliminate = neither"}},
    },
    # Trading Agent tools (agent/trading_tools.py) — real historical
    # price data and a backtest, not PM actions, but wired through this
    # same tool-calling pipeline (native/langgraph/adk) so they're
    # reachable from the exact same chat UI already deployed everywhere,
    # no separate agent-picker needed.
    "fetch_price_history": {
        "description": "Fetch real daily historical price data (open/high/low/close/volume) "
                       "for a stock or crypto ticker from a free public data source.",
        "required": ["symbol", "start", "end"],
        "properties": {
            "symbol": {"type": "string",
                      "description": "A ticker as Yahoo Finance recognizes it, e.g. 'AAPL' "
                                     "for Apple, 'BTC-USD' for Bitcoin."},
            "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
            "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
        },
    },
    "run_backtest": {
        "description": "Backtest a moving-average-crossover trading strategy against real "
                       "historical price data for a symbol, with an equity-curve chart and "
                       "trader metrics (CAGR, volatility, Sharpe, max drawdown). Omit start/end "
                       "for the last 10 years by default.",
        "required": ["symbol"],
        "properties": {
            "symbol": {"type": "string",
                      "description": "A ticker as Yahoo Finance recognizes it, e.g. 'AAPL'."},
            "start": {"type": "string",
                     "description": "Start date, YYYY-MM-DD. Omit for 10 years ago."},
            "end": {"type": "string", "description": "End date, YYYY-MM-DD. Omit for today."},
            "fast_window": {"type": "integer",
                            "description": "Fast moving-average window in days (default 20)."},
            "slow_window": {"type": "integer",
                            "description": "Slow moving-average window in days (default 50)."},
        },
    },
    # PEFT training request (agent/peft_training_tool.py) — "train the
    # local model on this file/link/text." `domain` is deliberately
    # REQUIRED here so native_agent.py's resolve_arguments always gets a
    # chance to ask which domain when the user didn't already say
    # (trading/coding/general/...) — see that function's own handling
    # of this tool name.
    "request_peft_training": {
        "description": "Log this content as a real training example for the nightly "
                       "fine-tune pipeline, for a specific domain's local model "
                       "(e.g. trading, coding, general).",
        "required": ["domain", "topic"],
        "properties": {
            "domain": {"type": "string",
                      "description": "Which domain's local model this should train "
                                     "(e.g. 'trading', 'coding', 'general')."},
            "topic": {"type": "string", "description": "A short label for what this is about."},
            "content": {"type": "string", "description": "The text to train on, if pasted directly."},
            "url": {"type": "string", "description": "A URL to fetch and train on, if given instead of content."},
        },
    },
}

# Every JSON tool plugin (../plugins.py) is indistinguishable from a
# hand-written tool from here on — perceive_intent's LLM call and
# output_guard both just see another entry in TOOL_SCHEMAS.
TOOL_SCHEMAS.update({name: p.schema for name, p in PLUGINS.items()})

# Every discovered MCP tool (mcp_client.py — config.py's MCP_SERVERS) is
# merged in the same way: perceive_intent and output_guard see them as just
# more entries in TOOL_SCHEMAS, on every track (native/langgraph/adk/claude
# all share this module). {"description","required","properties"} is
# already the shape mcp_tool_specs() returns (MCP's own inputSchema is JSON
# Schema, same as every other tool here).
TOOL_SCHEMAS.update({s["name"]: {"description": s["description"],
                                 "required": s["parameters"].get("required", []),
                                 "properties": s["parameters"].get("properties", {})}
                     for s in mcp_client.mcp_tool_specs()})


def openai_tool_specs() -> list[dict]:
    return [{"name": n, "description": s["description"],
             "parameters": {"type": "object",
                            "properties": s["properties"],
                            "required": s["required"]}}
            for n, s in TOOL_SCHEMAS.items()]


def _user_tool_specs_raw(user: dict) -> list[dict]:
    """Every per-user tool source, in mcp_tool_specs()'s common
    {name, description, parameters} shape — user_mcp_client.py's OAuth
    MCP tools (GitHub/Linear) and google_tools.py's hand-written Gmail/
    Calendar/Chat tools. New per-user sources are added here, once."""
    return (user_mcp_client.user_mcp_tool_specs(user["sub"])
           + google_tools.tool_specs_for_user(user["sub"]))


def user_openai_tool_specs(user: dict) -> list[dict]:
    """Per-REQUEST extension of openai_tool_specs() — this user's own
    OAuth-connected tools (GitHub/Linear via user_mcp_client.py, Gmail/
    Calendar/Chat via google_tools.py — see app/oauth_store.py).
    Deliberately never merged into the process-global TOOL_SCHEMAS/
    openai_tool_specs() above: those are shared by every user/request,
    and a per-user tool merged in there would leak user A's connected
    tools into user B's LLM tool-call surface. Callers add this list to
    openai_tool_specs()'s output at the point they build a chat turn's
    LLM-facing tool list — see each of the four agent tracks'
    handle_message/perceive_intent."""
    return [{"name": s["name"], "description": s["description"],
             "parameters": {"type": "object",
                            "properties": s["parameters"].get("properties", {}),
                            "required": s["parameters"].get("required", [])}}
            for s in _user_tool_specs_raw(user)]


def user_tool_schemas(user: dict) -> dict:
    """Same per-user extension as user_openai_tool_specs, but in
    TOOL_SCHEMAS' {name: {description, required, properties}} shape —
    for output_guard(), which validates a model's tool call against
    this shape, not openai_tool_specs()'s. Callers merge this with the
    global tools.TOOL_SCHEMAS (`{**tools.TOOL_SCHEMAS, **tools.user_tool_schemas(user)}`)
    right before calling output_guard, mirroring how the tool list itself
    is combined."""
    return {s["name"]: {"description": s["description"],
                        "required": s["parameters"].get("required", []),
                        "properties": s["parameters"].get("properties", {})}
           for s in _user_tool_specs_raw(user)}


# ── entity resolution (§6.1 step 7): never guess ─────────────────────────

@dataclass
class Resolution:
    id: Optional[str] = None
    ambiguous: list[dict] | None = None      # >1 candidate → clarify
    missing: bool = False
    # Only populated for a resolved TASK: real mode's task PATCH/DELETE
    # endpoints require project_id as a query param (an authorization
    # check — see app/routers/tasks.py's require_project_membership),
    # which the resolved task_id alone doesn't carry. Unused/None for
    # every other entity kind and for demo mode.
    project_id: Optional[str] = None


def _resolve(name: str, candidates: list[dict], key: str = "name",
             cutoff: int = 72) -> Resolution:
    if not name or not candidates:
        return Resolution(missing=True)
    # exact (case-insensitive) beats fuzzy
    exact = [c for c in candidates if c[key].lower() == name.lower()]
    if len(exact) == 1:
        return Resolution(id=exact[0]["id"], project_id=exact[0].get("project_id"))
    if len(exact) > 1:
        return Resolution(ambiguous=exact)
    scored = process.extract(name, {c["id"]: c[key] for c in candidates},
                             scorer=fuzz.WRatio, score_cutoff=cutoff, limit=3)
    if not scored:
        return Resolution(missing=True)
    top = scored[0][1]
    hits = [s for s in scored if s[1] >= top - 5]   # near-ties are ambiguous
    by_id = {c["id"]: c for c in candidates}
    if len(hits) > 1:
        return Resolution(ambiguous=[by_id[h[2]] for h in hits])
    return Resolution(id=scored[0][2], project_id=by_id[scored[0][2]].get("project_id"))


def resolve_project(name: str, user: dict | None = None) -> Resolution:
    return _resolve(name, _dedupe_by_name(pm_service.list_projects(user), "id", "name", "created_at"))


def _dedupe_by_name(items: list[dict], id_key: str, title_key: str,
                    date_key: str) -> list[dict]:
    """The space-level counterpart to _task_candidates' duplicate-only
    qualification — a space has no natural parent to qualify by the way a
    task has a project, so a repeat name (easy to end up with from
    testing the same "create a space called X" phrasing more than once)
    is disambiguated by creation time instead, and only when it actually
    collides — a uniquely-named space's candidate stays exactly as typed."""
    from collections import Counter
    titles = Counter(t[title_key] for t in items)
    out = []
    for t in items:
        title = t[title_key]
        if titles[title] > 1 and t.get(date_key):
            when = str(t[date_key])[:16].replace("T", " ")
            name = f"{title} (created {when})"
        else:
            name = title
        out.append({"id": t[id_key], "name": name})
    return out


def resolve_space(name: str, user: dict | None = None) -> Resolution:
    return _resolve(name, _dedupe_by_name(pm_service.list_spaces(user), "id", "name", "created_at"))


def _task_candidates(tasks: list[dict], id_key: str, title_key: str, project_key: str,
                     proj_names: dict[str, str] | None) -> list[dict]:
    """Builds _resolve()'s candidate list for a task search. Only titles
    that actually COLLIDE get project-qualified ("Set up CI pipeline (Aksha
    PM Mobile)") — qualifying every candidate unconditionally was tried
    first and made things worse: in a dataset with mostly one project
    (the demo store), every candidate would share the identical "(Compass
    OS)" suffix, which is pure noise to the fuzzy matcher and measurably
    degraded ordinary, unambiguous single-task lookups. Qualifying only
    real duplicates keeps every unique title's match exact/clean while
    still making a genuine collision (the actual problem this solves)
    both fuzzy-matchable by project name and legible in the resulting
    "which one did you mean?" list (see native_agent.ambiguous)."""
    from collections import Counter
    titles = Counter(t[title_key] for t in tasks)
    out = []
    for t in tasks:
        title = t[title_key]
        if titles[title] > 1 and proj_names:
            name = f"{title} ({proj_names.get(t[project_key], '?')})"
        else:
            name = title
        out.append({"id": t[id_key], "name": name, "project_id": t[project_key]})
    return out


def resolve_task(name: str, project_id: str | None = None, user: dict | None = None) -> Resolution:
    tasks = pm_service.list_tasks(user, project_id=project_id)
    proj_names = None if project_id else {p["id"]: p["name"] for p in pm_service.list_projects(user)}
    return _resolve(name, _task_candidates(tasks, "id", "title", "project_id", proj_names))


def resolve_subtask(name: str, task_id: str, user: dict | None = None) -> Resolution:
    subs = pm_service.list_subtasks(task_id)
    return _resolve(name, [{"id": s["id"], "name": s["title"]} for s in subs])


def resolve_user(name: str) -> Resolution:
    return _resolve(name, [{"id": u["id"], "name": u["name"]}
                           for u in pm_service.list_users()], cutoff=60)


def resolve_sprint(name: str, project_id: str | None = None, user: dict | None = None) -> Resolution:
    sprints = []
    projects = ([pm_service.get_project(project_id, user)] if project_id
                else pm_service.list_projects(user))
    for p in projects:
        sprints += pm_service.list_sprints(p["id"])
    return _resolve(name, sprints)


def _duration_status(t: dict) -> dict | None:
    """Planned (due_date minus start_date) vs actual duration for one task
    dict — real mode's snake_case fields and demo mode's (see
    pm_service.create_task's start_date param) both land here unchanged.
    None means there isn't enough data to judge: no start_date/due_date, or
    a done task whose completion timestamp isn't known (the demo store
    doesn't stamp completed_at the way the real backend does, so a demo
    task that's already Done is left unjudged rather than guessed at)."""
    start, due = t.get("start_date"), t.get("due_date")
    if not start or not due:
        return None
    done = (t.get("status") or "").lower() == "done"
    reference = t.get("completed_at") if done else None
    if done and not reference:
        return None
    try:
        start_d = dt.date.fromisoformat(start[:10])
        due_d = dt.date.fromisoformat(due[:10])
        ref_d = dt.date.fromisoformat(reference[:10]) if reference else dt.date.today()
    except ValueError:
        return None
    expected_days = (due_d - start_d).days
    elapsed_days = (ref_d - start_d).days
    return {"expected_days": expected_days, "elapsed_days": elapsed_days,
           "overrun_days": elapsed_days - expected_days, "done": done}


def _assignee_name(t: dict, real: bool) -> str | None:
    if real:
        return t.get("assignee_name")
    if not t.get("assignee_id"):
        return None
    who = pm_service.get_user(t["assignee_id"])
    return who["name"] if who else None


def is_real_mode(user: dict) -> bool:
    """True when THIS user's turn should hit the real app's own API
    instead of the demo store — REAL_APP_API_BASE_URL configured AND
    auth.py's real-token path actually ran for this user (see
    auth.py's _real_app_user, which stamps "_real_token")."""
    return bool(settings().REAL_APP_API_BASE_URL) and "_real_token" in user


# The real-data counterparts to resolve_project/_task/_user/_space/_sprint
# above — same _resolve() fuzzy-match core, real API candidate lists
# instead of the demo store's. Field names differ (project_id vs id,
# title vs name, ...) only in the dict comprehension feeding _resolve();
# the matching logic itself isn't duplicated.

def resolve_project_real(token: str, name: str) -> Resolution:
    return _resolve(name, _dedupe_by_name(real_pm_service.list_projects(token),
                                          "project_id", "name", "created_at"))


def resolve_space_real(token: str, name: str) -> Resolution:
    return _resolve(name, _dedupe_by_name(real_pm_service.list_spaces(token),
                                          "space_id", "name", "created_at"))


def resolve_subtask_real(token: str, name: str, task_id: str, project_id: str) -> Resolution:
    subs = real_pm_service.list_subtasks(token, task_id, project_id)
    return _resolve(name, [{"id": s["subtask_id"], "name": s["title"]} for s in subs])


def resolve_user_real(token: str, name: str) -> Resolution:
    return _resolve(name, [{"id": u["user_id"], "name": u["name"]}
                           for u in real_pm_service.list_users(token)], cutoff=60)


def resolve_task_real(token: str, name: str, project_id: str | None = None) -> Resolution:
    # Same duplicate-only qualification as resolve_task above — see
    # _task_candidates' docstring for why unconditional qualification
    # was tried first and made ordinary lookups worse.
    if project_id:
        tasks = real_pm_service.list_tasks_for_project(token, project_id)
        proj_names = None
    else:
        tasks = real_pm_service.list_all_tasks(token)
        proj_names = {p["project_id"]: p["name"]
                      for p in real_pm_service.list_projects(token)}
    return _resolve(name, _task_candidates(tasks, "task_id", "title", "project_id", proj_names))


def resolve_sprint_real(token: str, name: str, project_id: str | None = None) -> Resolution:
    sprints = []
    projects = ([{"project_id": project_id}] if project_id
               else real_pm_service.list_projects(token))
    for p in projects:
        # Same non-member-project skip as list_all_tasks/find_sprint_project_id
        # in real_pm_service.py — an Admin can list every org project without
        # being a member of each one.
        try:
            proj_sprints = real_pm_service.list_sprints_for_project(token, p["project_id"])
        except RuntimeError as e:
            if str(e).startswith("403"):
                continue
            raise
        for sp in proj_sprints:
            sprints.append({"id": sp["sprint_id"], "name": sp["name"],
                           "project_id": sp["project_id"]})
    return _resolve(name, sprints)


# What a JSON plugin's "resolve" entry can name — the same five entity
# kinds every hand-written tool above already resolves against, reused as
# a lookup table rather than duplicated.
_ENTITY_RESOLVERS = {
    "project": resolve_project, "space": resolve_space, "task": resolve_task,
    "user": resolve_user, "sprint": resolve_sprint,
}


def resolve_plugin_arguments(plugin, args: dict) -> tuple[dict | None, dict | None]:
    """The plugin-generic counterpart to native_agent.resolve_arguments's
    hardcoded per-tool branches — steps 7/8 for any JSON tool, dispatched
    purely from its "resolve" mapping. Entity args are fuzzy-matched to
    real ids exactly like a built-in tool (never LLM-invented); anything
    not named in "resolve" passes through unchanged (e.g. a plain string
    like a new name) so service_call.args can reference it too.

    Returns (resolved, clarify) — clarify, when present, is
    {"question": str, "field": str}, "field" being arg_name (the plugin's
    OWN args key, not the entity kind) so a resumed clarification (see
    native_agent.resume_clarification) overwrites the exact field that
    was actually ambiguous/missing, never a guess."""
    resolved: dict = {}
    for arg_name, spec in plugin.resolve.items():
        resolver = _ENTITY_RESOLVERS[spec["entity"]]
        res = resolver(args.get(arg_name, ""))
        if res.ambiguous:
            names = ", ".join(f"“{c['name']}”" for c in res.ambiguous[:4])
            return None, {"question": f"I found more than one {spec['entity']} "
                                      f"matching that: {names}. Which one did "
                                      f"you mean?", "field": arg_name,
                          "candidates": res.ambiguous}
        if res.missing:
            return None, {"question": f"I couldn't find a {spec['entity']} "
                                      f"matching “{args.get(arg_name)}”.",
                          "field": arg_name}
        resolved[spec["as"]] = res.id
    for k, v in args.items():
        if k not in plugin.resolve:
            resolved[k] = v
    return resolved, None


def execute_plugin(user: dict, plugin, resolved: dict) -> dict:
    """Step 11/12 for a JSON tool: call its one whitelisted service method
    (plugins.SERVICE_METHOD_REGISTRY — never an arbitrary dynamic call) with
    resolved's values mapped onto that method's kwargs, then render
    response_template — plain str.format, never eval, over resolved's
    fields plus whatever the service call itself returned."""
    method = SERVICE_METHOD_REGISTRY[plugin.service_call["method"]]
    kwargs = {kwarg: resolved[key] for kwarg, key in plugin.service_call["args"].items()}
    result = method(user, **kwargs)
    fields = {**resolved, **(result if isinstance(result, dict) else {})}
    return {"ok": True, "kind": "plugin_result",
           "text": plugin.response_template.format(**fields)}


# ── tool execution (§6.1 step 11 — the only mutating path) ───────────────

def execute(user: dict, name: str, resolved: dict) -> dict:
    """Runs the named tool with fully-resolved arguments AS the calling
    user. Returns a compact dict the formatter can phrase.

    Real mode (is_real_mode) branches BEFORE the demo pm_service call in
    every write below, hitting real_pm_service (real REST API) instead —
    resolve_arguments already resolved every name against real ids for
    this same turn (see native_agent.py), so the id space always matches
    whichever store this function is about to call."""
    if name in PLUGINS:
        return execute_plugin(user, PLUGINS[name], resolved)
    if name in mcp_client.mcp_tool_names():
        return mcp_client.call_mcp_tool(name, resolved)
    if name in user_mcp_client.user_mcp_tool_names(user["sub"]):
        return user_mcp_client.call_user_mcp_tool(user["sub"], name, resolved)
    if name in google_tools.tool_names_for_user(user["sub"]):
        return google_tools.execute_tool(user["sub"], name, resolved)
    if name == "fetch_price_history":
        try:
            points = trading_tools.fetch_price_history(
                resolved["symbol"], resolved["start"], resolved["end"])
        except trading_tools.PriceDataError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "kind": "price_history", "symbol": resolved["symbol"],
                "n_points": len(points),
                "points": [vars(p) for p in points[-30:]]}  # last 30 — enough context, not the whole history
    if name == "run_backtest":
        try:
            result = trading_tools.run_backtest(
                resolved["symbol"], resolved.get("start"), resolved.get("end"),
                fast_window=resolved.get("fast_window") or 20,
                slow_window=resolved.get("slow_window") or 50)
        except trading_tools.PriceDataError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "kind": "backtest", **result}
    if name == "request_peft_training":
        try:
            result = peft_training_tool.request_training(
                domain=resolved["domain"], topic=resolved["topic"],
                content=resolved.get("content"), url=resolved.get("url"))
        except peft_training_tool.TrainingRequestError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "kind": "peft_training_request", **result}
    real = is_real_mode(user)
    token = user.get("_real_token")

    if name == "create_space":
        if real:
            s = real_pm_service.create_space(token, resolved["name"],
                                             project_owner_id=resolved.get("manager_id"))
            return {"ok": True, "kind": "space", "name": s["name"], "id": s["space_id"]}
        s = pm_service.create_space(user, resolved["name"])
        return {"ok": True, "kind": "space", "name": s["name"], "id": s["id"]}
    if name == "create_project":
        if real:
            p = real_pm_service.create_project(
                token, resolved["name"], resolved["space_id"],
                resolved.get("board", "Kanban").lower(),
                project_owner_id=resolved.get("manager_id"),
                team_lead_id=resolved.get("team_lead_id"))
            return {"ok": True, "kind": "project", "name": p["name"], "id": p["project_id"]}
        p = pm_service.create_project(user, resolved["name"],
                                      resolved["space_id"],
                                      resolved.get("board", "Kanban"))
        return {"ok": True, "kind": "project", "name": p["name"], "id": p["id"]}
    if name == "create_task":
        if real:
            # real_pm_service.create_task's REST endpoint doesn't accept
            # priority/start_date at creation time (only description,
            # due_date, assignee_id, sprint_id) — assignee passes through;
            # priority/start_date are demo-mode-only until the real app's
            # own POST /api/projects/{id}/tasks grows those fields too.
            t = real_pm_service.create_task(
                token, resolved["project_id"], resolved["title"],
                resolved.get("description") or None, due_date=resolved.get("due_date"),
                assignee_id=resolved.get("assignee_id"))
            out = {"ok": True, "kind": "task", "title": t["title"], "id": t["task_id"]}
            if t.get("due_date"):
                out["created_due_date"] = t["due_date"]
            return out
        t = pm_service.create_task(user, resolved["project_id"],
                                   resolved["title"],
                                   resolved.get("description", ""),
                                   assignee_id=resolved.get("assignee_id"),
                                   due_date=resolved.get("due_date"),
                                   priority=resolved.get("priority", "Medium"),
                                   start_date=resolved.get("start_date"))
        out = {"ok": True, "kind": "task", "title": t["title"], "id": t["id"]}
        # a separate key from the set_due_date tool's "due_date" (below) —
        # that one means "the due date was just changed", this means "the
        # task happened to be created with one"; format_result phrases
        # each differently.
        if t.get("due_date"):
            out["created_due_date"] = t["due_date"]
        return out
    if name == "create_subtask":
        if real:
            st = real_pm_service.create_subtask(
                token, resolved["task_id"], resolved["task_project_id"],
                resolved["title"], resolved.get("due_date"))
            return {"ok": True, "kind": "subtask", "title": st["title"], "id": st["subtask_id"]}
        st = pm_service.create_subtask(user, resolved["task_id"], resolved["title"])
        return {"ok": True, "kind": "subtask", "title": st["title"], "id": st["id"]}
    if name == "list_subtasks":
        if real:
            subs = real_pm_service.list_subtasks(token, resolved["task_id"], resolved["task_project_id"])
            return {"ok": True, "kind": "subtask_list",
                    "subtasks": [{"title": s["title"], "done": s["done"]} for s in subs]}
        subs = pm_service.list_subtasks(resolved["task_id"])
        return {"ok": True, "kind": "subtask_list",
                "subtasks": [{"title": s["title"], "done": s["done"]} for s in subs]}
    if name == "complete_subtask":
        if real:
            st = real_pm_service.patch_subtask(token, resolved["task_id"], resolved["task_project_id"],
                                               resolved["subtask_id"], done=resolved["done"])
            return {"ok": True, "kind": "subtask", "title": st["title"], "done": st["done"]}
        st = pm_service.patch_subtask(user, resolved["subtask_id"], done=resolved["done"])
        return {"ok": True, "kind": "subtask", "title": st["title"], "done": st["done"]}
    if name == "create_user":
        # No real-mode branch: this always provisions a DEMO account (see
        # tools.TOOL_SCHEMAS' own description, "Admin only" — the real
        # app's own account provisioning is the "Invite someone new" flow
        # in CreateProjectModal, a UI action, not a chat one; wiring a
        # third way to create real user accounts wasn't asked for and
        # would duplicate that existing, already-audited path).
        u = pm_service.create_user(user, resolved["name"], resolved["role"],
                                   resolved.get("password", "demo123"),
                                   resolved.get("email", ""))
        return {"ok": True, "kind": "user", "name": u["name"], "role": u["role"],
                "id": u["id"], "email": u.get("email", "")}
    if name == "delete_user":
        if real:
            who = real_pm_service.get_user_name(token, resolved["user_id"])
            real_pm_service.delete_user(token, resolved["user_id"])
            return {"ok": True, "kind": "user_deleted", "name": who or resolved["user_id"]}
        who = pm_service.get_user(resolved["user_id"])
        pm_service.delete_user(user, resolved["user_id"])
        return {"ok": True, "kind": "user_deleted",
                "name": who["name"] if who else resolved["user_id"]}
    if name == "add_member":
        if real:
            real_pm_service.add_member(token, resolved["project_id"],
                                       resolved["member_id"], resolved["role"])
            proj = real_pm_service.get_project(token, resolved["project_id"])
            who = real_pm_service.get_user_name(token, resolved["member_id"])
            return {"ok": True, "kind": "membership", "project": proj["name"],
                    "member": who or resolved["member_id"], "role": resolved["role"]}
        p = pm_service.add_member(user, resolved["project_id"],
                                  resolved["member_id"], resolved["role"])
        who = pm_service.get_user(resolved["member_id"])
        return {"ok": True, "kind": "membership", "project": p["name"],
                "member": who["name"] if who else resolved["member_id"],
                "role": resolved["role"]}
    if name == "assign_task":
        if real:
            t = real_pm_service.patch_task(token, resolved["task_id"],
                                           resolved["task_project_id"],
                                           assignee_id=resolved["assignee_id"])
            who = real_pm_service.get_user_name(token, resolved["assignee_id"])
            return {"ok": True, "kind": "task", "title": t["title"],
                    "assignee": who or resolved["assignee_id"]}
        t = pm_service.patch_task(user, resolved["task_id"],
                                  assignee_id=resolved["assignee_id"])
        who = pm_service.get_user(resolved["assignee_id"])
        return {"ok": True, "kind": "task", "title": t["title"],
                "assignee": who["name"] if who else resolved["assignee_id"]}
    if name == "set_due_date":
        if real:
            t = real_pm_service.patch_task(token, resolved["task_id"],
                                           resolved["task_project_id"],
                                           due_date=resolved["due_date"])
            return {"ok": True, "kind": "task", "title": t["title"], "due_date": t["due_date"]}
        t = pm_service.patch_task(user, resolved["task_id"],
                                  due_date=resolved["due_date"])
        return {"ok": True, "kind": "task", "title": t["title"],
                "due_date": t["due_date"]}
    if name == "change_task_status":
        if real:
            t = real_pm_service.patch_task(token, resolved["task_id"],
                                           resolved["task_project_id"],
                                           status=_STATUS_TO_REAL[resolved["status"]])
            return {"ok": True, "kind": "task", "title": t["title"], "status": t["status"]}
        t = pm_service.patch_task(user, resolved["task_id"],
                                  status=resolved["status"])
        return {"ok": True, "kind": "task", "title": t["title"],
                "status": t["status"]}
    if name == "move_task_to_sprint":
        if real:
            t = real_pm_service.set_task_sprint(token, resolved["task_id"],
                                                resolved["task_project_id"],
                                                resolved["sprint_id"])
            sp_name = real_pm_service.sprint_name(token, resolved["task_project_id"],
                                                  resolved["sprint_id"])
            return {"ok": True, "kind": "task", "title": t["title"],
                    "sprint": sp_name or "that sprint", "due_date": t.get("due_date")}
        t = pm_service.move_task_to_sprint(user, resolved["task_id"],
                                           resolved["sprint_id"])
        sp = pm_service.get_sprint(resolved["sprint_id"])
        return {"ok": True, "kind": "task", "title": t["title"],
                "sprint": sp["name"], "due_date": t.get("due_date")}
    if name == "create_sprint":
        if real:
            s = real_pm_service.create_sprint(token, resolved["project_id"], resolved["name"])
            return {"ok": True, "kind": "sprint", "name": s["name"], "id": s["sprint_id"]}
        s = pm_service.create_sprint(user, resolved["project_id"], resolved["name"])
        return {"ok": True, "kind": "sprint", "name": s["name"], "id": s["id"]}
    if name == "start_sprint":
        if real:
            pid = resolved.get("project_id") or real_pm_service.find_sprint_project_id(token, resolved["sprint_id"])
            s = real_pm_service.start_sprint(token, pid, resolved["sprint_id"])
            return {"ok": True, "kind": "sprint", "name": s["name"], "state": s["status"]}
        s = pm_service.start_sprint(user, resolved["sprint_id"])
        return {"ok": True, "kind": "sprint", "name": s["name"], "state": s["state"]}
    if name == "complete_sprint":
        if real:
            pid = resolved.get("project_id") or real_pm_service.find_sprint_project_id(token, resolved["sprint_id"])
            s = real_pm_service.complete_sprint(token, pid, resolved["sprint_id"])
            return {"ok": True, "kind": "sprint", "name": s["name"], "state": s["status"]}
        s = pm_service.complete_sprint(user, resolved["sprint_id"])
        return {"ok": True, "kind": "sprint", "name": s["name"], "state": s["state"]}
    if name == "delete_task":
        if real:
            t = real_pm_service.get_task(token, resolved["task_id"], resolved["task_project_id"])
            real_pm_service.delete_task(token, resolved["task_id"], resolved["task_project_id"])
            return {"ok": True, "kind": "task", "title": t["title"], "deleted": True}
        t = pm_service.delete_task(user, resolved["task_id"])
        return {"ok": True, "kind": "task", "title": t["title"], "deleted": True}
    if name == "archive_project":
        if real:
            p = real_pm_service.archive_project(token, resolved["project_id"])
            return {"ok": True, "kind": "project", "name": p["name"], "archived": True}
        p = pm_service.archive_project(user, resolved["project_id"])
        return {"ok": True, "kind": "project", "name": p["name"], "archived": True}

    # ── read-only tools ─────────────────────────────────────────────
    if name == "query_tasks":
        # project_id, when present, is now real-resolved too (resolve_
        # arguments' rproject) — every case below can hit the real API.
        if real and resolved.get("mine"):
            tasks = real_pm_service.list_my_tasks(token)
        elif real and resolved.get("project_id"):
            tasks = real_pm_service.list_tasks_for_project(token, resolved["project_id"])
        elif real:
            tasks = real_pm_service.list_all_tasks(token)
        elif resolved.get("mine"):
            tasks = pm_service.list_tasks(assignee_id=user["sub"])
        else:
            tasks = pm_service.list_tasks(user, project_id=resolved.get("project_id"))
        if resolved.get("technology"):
            # Client-side filter — neither the real backend's list endpoints
            # nor the demo store take a technology query param (they just
            # return the full tag list per task), so this is the same
            # fan-out-then-filter shape list_all_tasks itself already uses.
            tech = resolved["technology"].lower()
            tasks = [t for t in tasks if tech in [x.lower() for x in t.get("technologies", [])]]
        if real:
            out_tasks = [{"title": t["title"], "status": t["status"],
                         "due": t.get("due_date"), "assignee": t.get("assignee_name")}
                        for t in tasks]
        else:
            out_tasks = []
            for t in tasks:
                who = pm_service.get_user(t["assignee_id"]) if t.get("assignee_id") else None
                out_tasks.append({"title": t["title"], "status": t["status"],
                                 "due": t.get("due_date"),
                                 "assignee": who["name"] if who else None})
        return {"ok": True, "kind": "task_list", "tasks": out_tasks}
    if name == "project_summary":
        today = dt.date.today().isoformat()
        if resolved.get("project_id"):
            if real:
                proj = real_pm_service.get_project(token, resolved["project_id"])
                proj_tasks = real_pm_service.list_tasks_for_project(token, resolved["project_id"])
                sprints = (real_pm_service.list_sprints_for_project(token, resolved["project_id"])
                          if proj.get("methodology") == "scrum" else [])
                overdue = sum(1 for t in proj_tasks if t.get("due_date")
                             and t["due_date"] < today and t.get("status") != "done")
                return {"ok": True, "kind": "project_summary", "name": proj["name"],
                        "status": proj.get("status"), "methodology": proj.get("methodology"),
                        "start_date": proj.get("start_date"), "due_date": proj.get("due_date"),
                        "total_tasks": len(proj_tasks),
                        "task_counts": dict(Counter(t.get("status", "unknown") for t in proj_tasks)),
                        "overdue_count": overdue, "sprint_count": len(sprints)}
            proj = pm_service.get_project(resolved["project_id"], user)
            proj_tasks = pm_service.list_tasks(user, project_id=resolved["project_id"])
            overdue = sum(1 for t in proj_tasks if t.get("due_date")
                         and t["due_date"] < today and t.get("status") != "Done")
            return {"ok": True, "kind": "project_summary", "name": proj["name"],
                    "status": "archived" if proj.get("archived") else "active",
                    "methodology": proj.get("board"), "start_date": None, "due_date": None,
                    "total_tasks": len(proj_tasks),
                    "task_counts": dict(Counter(t.get("status", "Unknown") for t in proj_tasks)),
                    "overdue_count": overdue, "sprint_count": len(pm_service.list_sprints(resolved["project_id"]))}
        # No project named — condensed row per project, across everything the
        # caller can see.
        if real:
            rows = []
            for p in real_pm_service.list_projects(token):
                # An Admin can LIST every org project but isn't necessarily a
                # MEMBER of each one (the tasks sub-resource 403s for a
                # non-member) — skip those rather than failing the whole
                # org-wide summary over one inaccessible project.
                try:
                    p_tasks = real_pm_service.list_tasks_for_project(token, p["project_id"])
                except RuntimeError as e:
                    if str(e).startswith("403"):
                        continue
                    raise
                overdue = sum(1 for t in p_tasks if t.get("due_date")
                             and t["due_date"] < today and t.get("status") != "done")
                done = sum(1 for t in p_tasks if t.get("status") == "done")
                rows.append({"name": p["name"], "status": p.get("status"),
                            "start_date": p.get("start_date"), "due_date": p.get("due_date"),
                            "total_tasks": len(p_tasks), "done_tasks": done, "overdue_count": overdue})
            return {"ok": True, "kind": "project_summary", "projects": rows}
        rows = []
        for p in pm_service.list_projects(user):
            p_tasks = pm_service.list_tasks(user, project_id=p["id"])
            overdue = sum(1 for t in p_tasks if t.get("due_date")
                         and t["due_date"] < today and t.get("status") != "Done")
            done = sum(1 for t in p_tasks if t.get("status") == "Done")
            rows.append({"name": p["name"], "status": "archived" if p.get("archived") else "active",
                        "start_date": None, "due_date": None,
                        "total_tasks": len(p_tasks), "done_tasks": done, "overdue_count": overdue})
        return {"ok": True, "kind": "project_summary", "projects": rows}
    if name == "sprint_summary":
        count = resolved.get("count")
        if resolved.get("project_id"):
            if real:
                proj = real_pm_service.get_project(token, resolved["project_id"])
                sprints = real_pm_service.list_sprints_for_project(token, resolved["project_id"])
                sprints = sorted(sprints, key=lambda s: s.get("start_date") or s.get("created_at") or "",
                                 reverse=True)
                if count:
                    sprints = sprints[:count]
                rows = [{"name": s["name"], "project": proj["name"], "status": s.get("status"),
                        "start_date": s.get("start_date"), "end_date": s.get("end_date"),
                        "total_tasks": s.get("total_tasks", 0), "done_tasks": s.get("done_tasks", 0)}
                       for s in sprints]
                return {"ok": True, "kind": "sprint_summary", "sprints": rows}
            proj = pm_service.get_project(resolved["project_id"], user)
            p_tasks = pm_service.list_tasks(user, project_id=resolved["project_id"])
            sprints = sorted(pm_service.list_sprints(resolved["project_id"]),
                             key=lambda s: s.get("created_at") or "", reverse=True)
            if count:
                sprints = sprints[:count]
            rows = []
            for s in sprints:
                s_tasks = [t for t in p_tasks if t.get("sprint_id") == s["id"]]
                rows.append({"name": s["name"], "project": proj["name"], "status": s.get("state"),
                            "start_date": None, "end_date": None, "total_tasks": len(s_tasks),
                            "done_tasks": sum(1 for t in s_tasks if t.get("status") == "Done")})
            return {"ok": True, "kind": "sprint_summary", "sprints": rows}
        # No project named — fan out across every scrum project's sprints.
        if real:
            rows = []
            for p in real_pm_service.list_projects(token):
                if p.get("methodology") != "scrum":
                    continue
                # Same non-member-project skip as the project_summary branch above.
                try:
                    sprints = real_pm_service.list_sprints_for_project(token, p["project_id"])
                except RuntimeError as e:
                    if str(e).startswith("403"):
                        continue
                    raise
                for s in sprints:
                    rows.append({"name": s["name"], "project": p["name"], "status": s.get("status"),
                                "start_date": s.get("start_date"), "end_date": s.get("end_date"),
                                "total_tasks": s.get("total_tasks", 0), "done_tasks": s.get("done_tasks", 0)})
            rows.sort(key=lambda r: r.get("start_date") or "", reverse=True)
            if count:
                rows = rows[:count]
            return {"ok": True, "kind": "sprint_summary", "sprints": rows}
        rows = []
        for p in pm_service.list_projects(user):
            p_tasks = pm_service.list_tasks(user, project_id=p["id"])
            for s in sorted(pm_service.list_sprints(p["id"]), key=lambda s: s.get("created_at") or "",
                            reverse=True):
                s_tasks = [t for t in p_tasks if t.get("sprint_id") == s["id"]]
                rows.append({"name": s["name"], "project": p["name"], "status": s.get("state"),
                            "start_date": None, "end_date": None, "total_tasks": len(s_tasks),
                            "done_tasks": sum(1 for t in s_tasks if t.get("status") == "Done"),
                            "created_at": s.get("created_at") or ""})
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        for r in rows:
            del r["created_at"]
        if count:
            rows = rows[:count]
        return {"ok": True, "kind": "sprint_summary", "sprints": rows}
    if name == "list_projects":
        if settings().REAL_APP_API_BASE_URL and "_real_token" in user:
            projects = real_pm_service.list_projects(user["_real_token"])
            return {"ok": True, "kind": "project_list",
                    "projects": [{"name": p["name"], "board": p.get("methodology", "")}
                                 for p in projects]}
        projects = pm_service.list_projects(user)
        return {"ok": True, "kind": "project_list",
                "projects": [{"name": p["name"], "board": p.get("board", "")}
                             for p in projects]}
    if name == "list_spaces":
        if settings().REAL_APP_API_BASE_URL and "_real_token" in user:
            spaces = real_pm_service.list_spaces(user["_real_token"])
            return {"ok": True, "kind": "space_list",
                    "spaces": [{"name": s["name"], "description": s.get("description", "")}
                              for s in spaces]}
        spaces = pm_service.list_spaces(user)
        return {"ok": True, "kind": "space_list",
                "spaces": [{"name": s["name"], "description": s.get("description", "")}
                          for s in spaces]}
    if name == "list_roles":
        roles = real_pm_service.list_roles(token)
        return {"ok": True, "kind": "role_list",
                "roles": [{"name": r["name"], "base_rank": r["base_rank"],
                          "extra_permissions": r.get("extra_permissions", []),
                          "is_system": r.get("is_system", False)} for r in roles]}
    if name == "list_users":
        if real:
            users = real_pm_service.list_users(token)
            if resolved.get("user_id"):
                u = next(u for u in users if u["user_id"] == resolved["user_id"])
                return {"ok": True, "kind": "user_info", "name": u["name"],
                        "role": u["role"], "email": u.get("email", "")}
            if resolved.get("technology"):
                tech = resolved["technology"].lower()
                users = [u for u in users if tech in [x.lower() for x in u.get("technologies", [])]]
            return {"ok": True, "kind": "user_list",
                    "users": [{"name": u["name"], "role": u["role"]} for u in users]}
        users = pm_service.list_users()
        if resolved.get("user_id"):
            u = pm_service.get_user(resolved["user_id"])
            return {"ok": True, "kind": "user_info", "name": u["name"],
                    "role": u["role"], "email": u.get("email", "")}
        if resolved.get("technology"):
            tech = resolved["technology"].lower()
            users = [u for u in users if tech in [x.lower() for x in u.get("technologies", [])]]
        return {"ok": True, "kind": "user_list",
                "users": [{"name": u["name"], "role": u["role"]} for u in users]}
    if name == "task_info":
        if real:
            t = real_pm_service.get_task(token, resolved["task_id"], resolved["task_project_id"])
            return {"ok": True, "kind": "task_info", "title": t["title"],
                    "status": t["status"], "assignee": t.get("assignee_name"),
                    "due": t.get("due_date"), "priority": t.get("priority")}
        tasks = pm_service.list_tasks(user, project_id=resolved["task_project_id"])
        t = next(t for t in tasks if t["id"] == resolved["task_id"])
        who = pm_service.get_user(t["assignee_id"]) if t.get("assignee_id") else None
        return {"ok": True, "kind": "task_info", "title": t["title"],
                "status": t["status"], "assignee": who["name"] if who else None,
                "due": t.get("due_date"), "priority": t.get("priority")}
    if name == "task_duration_check":
        if resolved.get("task_id"):
            if real:
                t = real_pm_service.get_task(token, resolved["task_id"], resolved["task_project_id"])
            else:
                tasks = pm_service.list_tasks(user, project_id=resolved["task_project_id"])
                t = next(t for t in tasks if t["id"] == resolved["task_id"])
            info = _duration_status(t)
            if info is None:
                return {"ok": True, "kind": "duration_check", "tasks": [],
                        "note": (f"“{t['title']}” doesn't have both a start and "
                                f"due date set, so I can't judge its duration.")}
            return {"ok": True, "kind": "duration_check",
                    "tasks": [{"title": t["title"], "assignee": _assignee_name(t, real), **info}]}
        # No specific task named — every task currently running longer than
        # planned, optionally within one project.
        if real:
            tasks = (real_pm_service.list_tasks_for_project(token, resolved["project_id"])
                    if resolved.get("project_id") else real_pm_service.list_all_tasks(token))
        else:
            tasks = pm_service.list_tasks(user, project_id=resolved.get("project_id"))
        rows = []
        for t in tasks:
            info = _duration_status(t)
            if info and info["overrun_days"] > 0:
                rows.append({"title": t["title"], "assignee": _assignee_name(t, real), **info})
        rows.sort(key=lambda r: -r["overrun_days"])
        return {"ok": True, "kind": "duration_check", "tasks": rows}
    if name == "message_task":
        mentions = [resolved["mention_id"]] if resolved.get("mention_id") else []
        m = real_pm_service.create_task_message(
            token, resolved["task_id"], resolved["task_project_id"],
            resolved["text"], mentions)
        t = real_pm_service.get_task(token, resolved["task_id"], resolved["task_project_id"])
        return {"ok": True, "kind": "task_message", "task": t["title"],
                "text": m["text"], "mention": resolved.get("mention_name")}
    if name == "message_project":
        mentions = [resolved["mention_id"]] if resolved.get("mention_id") else []
        m = real_pm_service.create_project_message(
            token, resolved["project_id"], resolved["text"], mentions)
        proj = real_pm_service.get_project(token, resolved["project_id"])
        return {"ok": True, "kind": "project_message", "project": proj["name"],
                "text": m["text"], "mention": resolved.get("mention_name")}
    if name == "list_personal_tasks":
        pts = real_pm_service.list_personal_tasks(token)
        return {"ok": True, "kind": "personal_task_list",
                "tasks": [{"title": t["title"], "done": t["done"],
                          "urgent": t["urgent"], "important": t["important"],
                          "due": t.get("end_date")} for t in pts]}
    if name == "save_note":
        note = real_pm_service.create_personal_note(token, resolved.get("title"), resolved["body"])
        return {"ok": True, "kind": "note_saved", "title": note.get("title"), "body": note["body"]}
    if name == "move_personal_task":
        pt = real_pm_service.patch_personal_task(
            token, resolved["personal_task_id"],
            urgent=resolved["urgent"], important=resolved["important"])
        return {"ok": True, "kind": "personal_task", "title": pt["title"],
                "urgent": pt["urgent"], "important": pt["important"]}
    if name == "graph_query":
        q = resolved["question"]
        # scope/assignee ARE real-resolved now (rproject/rspace/rsprint/
        # ruser, see resolve_arguments) — what's NOT yet built is the real
        # equivalent of a SCOPED overdue/workload query (real_pm_service's
        # overdue_tasks/workload don't take a scope param), so only those
        # two fall back to demo when a scope was named. tasks_for never
        # uses scope_id at all, so it has no such caveat.
        unscoped_real = (real and not resolved.get("scope_id"))
        if q == "overdue":
            if unscoped_real:
                rows = real_pm_service.overdue_tasks(token)
                return {"ok": True, "kind": "overdue",
                        "tasks": [{"name": t["title"], "due": t["due_date"],
                                  "status": t["status"]} for t in rows]}
            rows = knowledge_graph.overdue_tasks(
                resolved.get("scope_id"), resolved.get("assignee_id"), user)
            return {"ok": True, "kind": "overdue", "tasks": rows}
        if q == "workload":
            if unscoped_real:
                return {"ok": True, "kind": "workload",
                        "rows": real_pm_service.workload(token)}
            return {"ok": True, "kind": "workload",
                    "rows": knowledge_graph.workload(resolved.get("scope_id"), user)}
        if q == "overdue_by_person":
            if unscoped_real:
                counts = Counter(t["assignee_name"] for t in real_pm_service.overdue_tasks(token)
                                if t.get("assignee_name"))
                rows = sorted(({"name": n, "overdue_tasks": c} for n, c in counts.items()),
                             key=lambda r: -r["overdue_tasks"])
                return {"ok": True, "kind": "overdue_by_person", "rows": rows}
            return {"ok": True, "kind": "overdue_by_person",
                    "rows": knowledge_graph.overdue_by_person(resolved.get("scope_id"), user)}
        if q == "tasks_for":
            if real:
                rows = [{"name": t["title"], "status": t["status"]}
                       for t in real_pm_service.list_all_tasks(token)
                       if t.get("assignee_id") == resolved["assignee_id"]]
            else:
                rows = knowledge_graph.tasks_for(resolved["assignee_id"],
                                                 resolved.get("scope_id"), user)
            return {"ok": True, "kind": "tasks_for", "tasks": rows,
                    "assignee": resolved.get("assignee_name", "")}
        return {"ok": False, "error": f"unknown graph question {q!r}"}
    if name == "semantic_search":
        hits = vector_index.search(resolved["query"], user, k=5)
        return {"ok": True, "kind": "semantic",
                "hits": [{"text": h["text"], "score": round(h["score"], 3)}
                         for h in hits]}
    return {"ok": False, "error": f"unknown tool {name!r}"}
