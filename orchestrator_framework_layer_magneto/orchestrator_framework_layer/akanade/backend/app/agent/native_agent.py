"""The reference implementation of the §6.1 seventeen-step algorithm.

This file IS the numbered spec, in order — each step is commented with
its number so code review can check implementation against the list,
exactly as the design demands. The LangGraph track (langgraph_agent.py)
implements the same spec as a StateGraph; both must trace to the same
numbered path (§6.5's acceptance test).
"""
from __future__ import annotations

import re

from .. import security_events, security_guard
from ..background import run_in_background
from ..security_optimizer import security_optimizer
from . import finetune_log, google_tools, mcp_client, sessions, user_mcp_client
from ..framework.rag_optimizer import is_rag_relevant, rag_optimizer, retrieve_with_optimizer
from ..framework.rag import rag_registry
from ..llm.domain_router import domain_router
from ..llm.guardrails import input_guard, output_guard
from ..llm.hybrid_router import LOCAL_TIMEOUT_SECONDS, hybrid_router
from ..llm.providers import RuleBased
from ..llm.router import get_llm_client
from ..llm.token_optimizer import token_optimizer
from ..plugins import PLUGINS
from ..services import real_pm_service
from . import tools
from .state import AgentReply, format_result

YES = re.compile(r"^\s*(y|yes|yeah|yep|confirm|do it|go ahead)\s*[.!]?\s*$", re.I)
NO = re.compile(r"^\s*(n|no|nope|cancel|stop|don'?t)\s*[.!]?\s*$", re.I)

# Step 6's system message: nudges the model to prefer calling a tool with
# its best guess at each entity over asking a plain-text question — the
# app's own resolve_entities (step 7/8) already asks a PROPER clarifying
# question when something's missing or ambiguous, AND (via
# resume_clarification above) remembers the in-progress call across the
# next turn. A free-text question from the model instead skips both: no
# tool_call means nothing is recorded to resume, so the next turn has to
# reconstruct intent from raw chat history alone.
PERCEIVE_INTENT_SYSTEM = {
    "role": "system",
    "content": ("If the user's request is missing a detail an action needs (like "
               "which project a task belongs to), still call the matching tool "
               "using your best interpretation of whatever entity name they "
               "mentioned — do not ask for it in plain text. The system will ask "
               "a precise follow-up question itself if that guess doesn't "
               "resolve to anything, and it remembers what you were doing when "
               "the user answers."),
}

# Real bug found live: a genuine "Backtest a 20/50-day moving average
# crossover on AAPL for 2024" request on the Trading Agent tab got
# answered in free text (no tool call) and was shown to the user as
# "not grounded in your PM data" — the model had no signal it was even
# in a trading context, since the Trading tab sends the exact same
# request shape as the PM tab (see README_TRADING_AGENT.md). Injected
# ONLY when ChatMessageIn.target == "trading" (see that field's own
# docstring), never for the default PM tab.
TRADING_TARGET_SYSTEM = {
    "role": "system",
    "content": ("The user is using the Trading Agent. If they ask about a stock/crypto "
               "price, historical data, a backtest, or a trading strategy, call "
               "fetch_price_history or run_backtest with your best interpretation of "
               "the symbol/dates/strategy mentioned — do not answer from general "
               "knowledge instead."),
}


def looks_like_a_fresh_command(msg: str) -> bool:
    """True if `msg` matches one of RuleBased's own hand-written command
    patterns — a strong, free (no LLM call, no network) signal that this
    is a brand-new, structured request rather than an answer to whatever
    question is still pending. Guards resume_clarification's own default
    (merge the message into the stuck field, see its docstring): without
    this, a user who gives up on a stuck "which project?" and types an
    unrelated fresh command instead — "list all projects", "show my
    tasks" — gets it silently swallowed as a failed attempt to name a
    project, and the SAME question gets re-asked forever, since every
    later message hits the identical dead end. Deliberately checked
    against RuleBased specifically (not "did an LLM understand this") —
    its patterns are precise, deliberate command shapes, so a match here
    is a much safer signal than "the field just doesn't look like a
    project name", which would misfire on real project names too.
    """
    return RuleBased().chat([{"role": "user", "content": msg}]).tool_call is not None


def resume_clarification(user: dict, pending_clarify: dict,
                         user_msg: str) -> tuple[str, dict | None, dict | None]:
    """Resumes a paused step 8 (§6.1): the follow-up answer to a clarifying
    question is merged into the ORIGINAL tool call — into the SPECIFIC
    field resolve_arguments was actually asking about (pending_clarify's
    "field", set when the clarify was first raised; see Clarify below),
    not a guess. Without this, a session that just asked "which project?"
    has no memory of the task title it was already given — each turn
    would restart perceive_intent from scratch with only raw chat history
    as a hint, which an LLM isn't guaranteed to reconstruct correctly.

    A tool schema's ambiguous/missing field isn't always the LAST required
    one (message_task's is "task", first of ["task","text"]) — guessing
    "last required field" instead of tracking the real one would silently
    overwrite the WRONG field (here, stuffing the disambiguating answer
    into "text") and re-ask the same question forever. field falls back
    to the old last-required-field guess only if genuinely absent (e.g. a
    plugin's older clarify shape), never as the primary mechanism.

    Shared by both tracks (§6.5's equivalence rule) rather than duplicated,
    since neither track's flow (a straight-line function here, a
    StateGraph in langgraph_agent.py) is restructured just to reopen one
    paused turn — both call this the same way, from outside their normal
    perceive_intent path.

    Returns (reply, pending_confirmation_or_None, pending_clarification_or_None)
    — the same three-way shape handle_message already persists per turn.
    """
    name = pending_clarify["tool"]
    args = dict(pending_clarify.get("args") or {})
    field = pending_clarify.get("field")
    if not field:
        required = tools.TOOL_SCHEMAS[name]["required"]
        missing = [f for f in required if not args.get(f)]
        field = missing[0] if missing else required[-1]

    candidates = pending_clarify.get("candidates")
    if candidates:
        # The question was "which one" among a SPECIFIC set (see
        # native_agent.ambiguous) — match the answer against just those
        # candidates first, not a fresh unconstrained search. A fresh
        # search over the full universe would let a reply like "the one
        # in Aksha PM Mobile" drift onto some UNRELATED task that also
        # mentions that project, instead of the one actually being
        # disambiguated.
        pick = tools._resolve(user_msg.strip(), candidates)
        if pick.id and not pick.ambiguous:
            args[field] = next(c["name"] for c in candidates if c["id"] == pick.id)
        elif pick.ambiguous:
            # Narrowed but still tied (e.g. "Aksha PM Mobile" vs "Aksha PM
            # Web" both mention "Aksha PM") — re-ask with the SMALLER set
            # rather than silently falling through to the broad search
            # that caused the original drift bug.
            names = ", ".join(f"“{c['name']}”" for c in pick.ambiguous[:4])
            return (f"Still not sure which one — {names}. Which one did you mean?",
                   None, {"tool": name, "args": args, "field": field,
                         "candidates": pick.ambiguous})
        else:
            # Genuinely didn't match any of the offered candidates — the
            # user may be answering a different question entirely (e.g.
            # naming the task by a different word than any candidate
            # uses); an unconstrained search is the right fallback here.
            args[field] = user_msg.strip()
    else:
        args[field] = user_msg.strip()
    resolved, clarify = resolve_arguments(user, name, args)
    if clarify:
        if clarify["field"] is None:
            # A terminal decline ("that's not available in this demo"),
            # not a real follow-up question — end the turn cleanly rather
            # than leaving a pendingClarification that would silently
            # swallow the user's NEXT, unrelated message as if it were
            # answering this one.
            return clarify["question"], None, None
        return clarify["question"], None, {"tool": name, "args": args,
                                           "field": clarify["field"],
                                           "candidates": clarify.get("candidates")}
    if requires_confirmation(user, name):
        prompt = confirm_prompt_for(user, name, args)
        return prompt, {"tool": name, "args": resolved, "prompt": prompt}, None
    try:
        result = tools.execute(user, name, resolved)
    except Exception as e:
        return f"Sorry — that failed: {e}", None, None
    return format_result(result), None, None


def classify_task_complexity(msg: str) -> str:
    """Step 4 — cheap, LOCAL (non-LLM) classifier: realtime vs delay-ok.
    Chat turns are realtime; explicit report/digest asks are delay-ok."""
    if re.search(r"\b(report|digest|weekly summary|rca|analysis)\b", msg.lower()):
        return "delay_ok"
    return "realtime"


# Pure greeting/chit-chat, checked against the WHOLE message (anchored,
# not a substring search) — deliberately narrow: a false NEGATIVE just
# means a genuine greeting goes through the normal (slower, but correct)
# tool-calling path same as before this existed; a false POSITIVE would
# mean a real request gets treated as small talk and never gets its tool
# call, which is the actually dangerous direction to be wrong in. Real,
# measured problem this fixes: perceive_intent sends the FULL tool
# schema (41 tools, ~4600 tokens) on every single turn including "Are
# you up?" — real added prompt-processing cost a small model pays for
# every single message regardless of content, plus a domain-router
# bandit consultation and a RAG relevance check that are pure overhead
# for a message that was never going to call a tool or need retrieved
# context. Matched messages skip the tool schema (see handle_message's
# `tools=None if is_pure_conversational... else user_tool_specs`), RAG,
# and domain-routing entirely — the actual LLM call and its own latency
# is untouched, since that's real, necessary work, never faked.
_PURE_CONVERSATIONAL = re.compile(
    r"^\s*(hi|hey|hello|yo|sup)[.!?]*\s*$"
    r"|^\s*(hi|hey|hello)\s+(there|akanade|algo)[.!?]*\s*$"
    r"|^\s*(are you (up|there|awake|online))[.!?]*\s*$"
    r"|^\s*(you (up|there)\??)[.!?]*\s*$"
    r"|^\s*(how('?s| is| are) it going|how are you( doing)?|what'?s up|"
    r"good (morning|afternoon|evening|night))[.!?]*\s*$"
    r"|^\s*(thanks?( you)?|thank you( so much)?|ty|cheers|ok(ay)?|cool|great|nice|got it|"
    r"sounds good|makes sense)[.!?]*\s*$"
    r"|^\s*(bye|goodbye|see ya|see you( later| soon)?|later|good ?night)[.!?]*\s*$"
    r"|^\s*(what have you (done|been up to)|what did you do|what'?s new)[.!?]*\s*$",
    re.IGNORECASE)


def is_pure_conversational(msg: str) -> bool:
    """True for a message that is, on its own, obviously not asking for
    any PM/trading/MCP tool action or retrieved context — a greeting,
    an acknowledgement, small talk. Anchored to the FULL message (after
    stripping whitespace) so "hi, can you list my tasks" does NOT match
    (a real request tacked onto a greeting still needs the real path)."""
    return bool(_PURE_CONVERSATIONAL.match(msg or ""))


def resolve_arguments(user: dict, name: str, args: dict) -> tuple[dict | None, dict | None]:
    """Step 7 — fuzzy-match every raw entity against live data to obtain
    real IDs; no ID is ever invented by the LLM. Returns (resolved,
    clarify): exactly one is non-None. clarify, when present, is
    {"question": str, "field": str} — "field" is the args key resume_
    clarification (above) should overwrite with the next turn's answer,
    tracked explicitly rather than guessed.

    In real mode (tools.is_real_mode) every resolver below hits the real
    app's own API for candidates instead of the demo store — see
    tools.py's resolve_*_real() functions. The surrounding logic
    (ambiguous/missing handling, field names) is identical either way:
    Resolution's shape doesn't change based on which store answered it."""
    r: dict = {}
    real = tools.is_real_mode(user)
    token = user.get("_real_token")

    def rproject(nm: str) -> tools.Resolution:
        return tools.resolve_project_real(token, nm) if real else tools.resolve_project(nm, user)

    def rspace(nm: str) -> tools.Resolution:
        return tools.resolve_space_real(token, nm) if real else tools.resolve_space(nm, user)

    def ruser(nm: str) -> tools.Resolution:
        return tools.resolve_user_real(token, nm) if real else tools.resolve_user(nm)

    def rtask(nm: str, project_id: str | None = None) -> tools.Resolution:
        return (tools.resolve_task_real(token, nm, project_id) if real
               else tools.resolve_task(nm, project_id, user))

    def rsprint(nm: str, project_id: str | None = None) -> tools.Resolution:
        return (tools.resolve_sprint_real(token, nm, project_id) if real
               else tools.resolve_sprint(nm, project_id, user))

    def clarify(question: str, field: str, candidates: list[dict] | None = None) -> dict:
        return {"question": question, "field": field, "candidates": candidates}

    def ambiguous(label: str, res: tools.Resolution, field: str) -> dict:
        # For tasks, candidate "name" is already project-qualified (see
        # resolve_task/resolve_task_real) — same-titled tasks in different
        # projects are common ("Set up CI pipeline" isn't a rare title),
        # and a bare title list would be genuinely unanswerable otherwise.
        # The candidates themselves are carried in the clarify dict too
        # (see resume_clarification) so a follow-up like "the one in
        # Aksha PM Mobile" is matched against just THESE candidates, not
        # a fresh unconstrained search that could drift onto an unrelated
        # task that also happens to mention that project.
        names = ", ".join(f"“{c['name']}”" for c in res.ambiguous[:4])
        return clarify(f"I found more than one {label} matching that: {names}. "
                       f"Which one did you mean?", field, res.ambiguous)

    if name in PLUGINS:
        resolved, clarify_q = tools.resolve_plugin_arguments(PLUGINS[name], args)
        if clarify_q:
            return None, clarify_q  # already {"question","field"} — see tools.py
        return resolved, None

    if name in mcp_client.mcp_tool_names():
        # No entity resolution for an MCP tool — its arguments (a timezone
        # name, an ISO date, ...) are literal values, not PM entity names
        # that need fuzzy-matching to a real id (step 7 doesn't apply here,
        # same as e.g. change_task_status's "status" enum above).
        return args, None
    if name in ("fetch_price_history", "run_backtest"):
        # Trading Agent tools (agent/trading_tools.py) — symbol/date
        # arguments are literal values too, same reasoning as the MCP
        # bypass just above.
        return args, None
    if name == "request_peft_training":
        # "Confirm the domain with the user" happens right here — the
        # exact same clarify()/resume_clarification round trip every
        # other ambiguous PM request already uses, not a special-cased
        # flow of its own. A model that already extracted a domain from
        # the user's own words (e.g. "train the trading model on this")
        # skips straight through; only a genuinely unnamed domain stops
        # to ask.
        domain = (args.get("domain") or "").strip()
        if not domain:
            return None, clarify(
                "Which domain's local model should this train — for example "
                "'trading', 'coding', or 'general'?", "domain")
        if not args.get("content") and not args.get("url"):
            return None, clarify(
                "What should I train it on — paste the text, or give me a URL?", "content")
        return args, None

    if name in ("create_space",):
        r["name"] = args["name"].strip().title()
        if args.get("manager"):
            mg = ruser(args["manager"])
            if mg.ambiguous:
                return None, ambiguous("person", mg, "manager")
            if mg.missing:
                return None, clarify(f"I couldn't find anyone named "
                                     f"“{args['manager']}” to be the Senior "
                                     f"Manager. Who should it be?", "manager")
            r["manager_id"] = mg.id
        return r, None

    if name == "create_project":
        sp = rspace(args["space"])
        if sp.ambiguous:
            return None, ambiguous("space", sp, "space")
        if sp.missing:
            return None, clarify(f"I couldn't find a space called "
                                 f"“{args['space']}”. Which space should "
                                 f"this project go in?", "space")
        r.update(name=args["name"].strip(), space_id=sp.id,
                 board=args.get("board", "Kanban"))
        # The real app's own ProjectService requires an Admin creator to
        # name a Senior Manager (Manager2) as project_owner, and an Admin
        # or Manager2 creator to name a Team Lead (Manager1) — see
        # project_service.py's _ROLES_REQUIRING_PROJECT_OWNER/
        # _ROLES_REQUIRING_TEAM_LEAD on the real app's side. Without this
        # check, create_project sailed past resolve_arguments only to 400
        # against the real API on a missing field it was always going to
        # need — surfacing as a flat "Sorry — that failed" instead of the
        # same follow-up question the real app's own creation form would
        # ask. user["role"] is already mapped through auth._REAL_ROLE_MAP
        # (real Admin -> "Executive", real Manager2 -> "Admin" here), so
        # the checks below use the mapped equivalents, not the real names.
        if real and not args.get("manager") and user["role"] == "Executive":
            return None, clarify("Every project needs a Senior Manager "
                                 "(Manager2) to supervise it — who should "
                                 "that be?", "manager")
        if real and not args.get("team_lead") and user["role"] in ("Executive", "Admin"):
            return None, clarify("Every project needs a Team Lead (Manager1) "
                                 "to run it day-to-day — who should that "
                                 "be?", "team_lead")
        if args.get("manager"):
            mg = ruser(args["manager"])
            if mg.ambiguous:
                return None, ambiguous("person", mg, "manager")
            if mg.missing:
                return None, clarify(f"I couldn't find anyone named "
                                     f"“{args['manager']}” to be the Senior "
                                     f"Manager. Who should it be?", "manager")
            r["manager_id"] = mg.id
        if args.get("team_lead"):
            tl = ruser(args["team_lead"])
            if tl.ambiguous:
                return None, ambiguous("person", tl, "team_lead")
            if tl.missing:
                return None, clarify(f"I couldn't find anyone named "
                                     f"“{args['team_lead']}” to be the Team "
                                     f"Lead. Who should it be?", "team_lead")
            r["team_lead_id"] = tl.id
        return r, None

    if name in ("create_task", "create_sprint"):
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify(f"I couldn't find a project matching "
                                 f"“{args['project']}”. Which project?", "project")
        r["project_id"] = pr.id
        r["name" if name == "create_sprint" else "title"] = \
            args.get("name", args.get("title", "")).strip()
        if name == "create_task":
            r["description"] = args.get("description", "")
            if args.get("due_date"):
                import datetime as _dt
                try:
                    r["due_date"] = _dt.date.fromisoformat(args["due_date"]).isoformat()
                except ValueError:
                    return None, clarify(f"“{args['due_date']}” isn't a valid date "
                                         f"(use YYYY-MM-DD) — what should the due "
                                         f"date be? Or say the task without one.", "due_date")
            if args.get("start_date"):
                import datetime as _dt
                try:
                    r["start_date"] = _dt.date.fromisoformat(args["start_date"]).isoformat()
                except ValueError:
                    return None, clarify(f"“{args['start_date']}” isn't a valid start "
                                         f"date (use YYYY-MM-DD).", "start_date")
            if args.get("assignee"):
                asg = ruser(args["assignee"])
                if asg.ambiguous:
                    return None, ambiguous("person", asg, "assignee")
                if asg.missing:
                    return None, clarify(f"I couldn't find anyone named "
                                         f"“{args['assignee']}” to assign it to.", "assignee")
                r["assignee_id"] = asg.id
            if args.get("priority"):
                priority = args["priority"].strip().title()
                if priority not in ("Low", "Medium", "High"):
                    return None, clarify(f"What priority should “{args.get('title', 'this task')}” "
                                         f"have — Low, Medium, or High?", "priority")
                r["priority"] = priority
        return r, None

    if name == "create_subtask":
        tk = rtask(args["task"])
        if tk.ambiguous:
            return None, ambiguous("task", tk, "task")
        if tk.missing:
            return None, clarify(f"I couldn't find a task matching "
                                 f"“{args['task']}”. Which task should this "
                                 f"subtask go under?", "task")
        r["task_id"] = tk.id
        r["task_project_id"] = tk.project_id
        r["title"] = args["title"].strip()
        if args.get("due_date"):
            import datetime as _dt
            try:
                r["due_date"] = _dt.date.fromisoformat(args["due_date"]).isoformat()
            except ValueError:
                return None, clarify(f"“{args['due_date']}” isn't a valid date "
                                     f"(use YYYY-MM-DD).", "due_date")
        return r, None

    if name == "list_subtasks":
        tk = rtask(args["task"])
        if tk.ambiguous:
            return None, ambiguous("task", tk, "task")
        if tk.missing:
            return None, clarify(f"I couldn't find a task matching "
                                 f"“{args['task']}”. Which task?", "task")
        r["task_id"] = tk.id
        r["task_project_id"] = tk.project_id
        return r, None

    if name == "complete_subtask":
        tk = rtask(args["task"])
        if tk.ambiguous:
            return None, ambiguous("task", tk, "task")
        if tk.missing:
            return None, clarify(f"I couldn't find a task matching "
                                 f"“{args['task']}”. Which task is that "
                                 f"subtask under?", "task")
        sb = (tools.resolve_subtask_real(token, args["subtask"], tk.id, tk.project_id) if real
             else tools.resolve_subtask(args["subtask"], tk.id, user))
        if sb.ambiguous:
            return None, ambiguous("subtask", sb, "subtask")
        if sb.missing:
            return None, clarify(f"I couldn't find a subtask matching "
                                 f"“{args['subtask']}” under that task. "
                                 f"Which one?", "subtask")
        r["task_id"] = tk.id
        r["task_project_id"] = tk.project_id
        r["subtask_id"] = sb.id
        r["done"] = args["done"] if isinstance(args.get("done"), bool) else True
        return r, None

    if name == "create_user":
        role = args["role"].strip().title()
        if role not in ("Member", "Manager", "Admin", "Executive"):
            return None, clarify(f"What role should {args.get('name', 'they')} have — "
                                 f"Member, Manager, Admin, or Executive?", "role")
        r["name"] = args["name"].strip().title()
        r["role"] = role
        if args.get("email"):
            r["email"] = args["email"].strip()
        if args.get("password"):
            r["password"] = args["password"]
        return r, None

    if name == "add_member":
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify(f"I couldn't find a project matching "
                                 f"“{args['project']}”. Which project?", "project")
        us = ruser(args["user"])
        if us.ambiguous:
            return None, ambiguous("person", us, "user")
        if us.missing:
            return None, clarify(f"Who should I add? I couldn't match "
                                 f"“{args['user']}”.", "user")
        role = (args.get("role") or "Member").strip().title()
        if role == "Co-owner" or role == "Coowner":
            role = "CoOwner"
        if role not in ("Owner", "CoOwner", "Manager", "Member"):
            return None, clarify(f"What role should {args['user']} have on "
                                 f"{args['project']} — Owner, CoOwner, Manager, "
                                 f"or Member?", "role")
        r["project_id"] = pr.id
        r["member_id"] = us.id
        r["role"] = role
        return r, None

    if name in ("assign_task", "set_due_date", "change_task_status",
                "delete_task", "move_task_to_sprint"):
        tk = rtask(args["task"])
        if tk.ambiguous:
            return None, ambiguous("task", tk, "task")
        if tk.missing:
            return None, clarify(f"I couldn't find a task matching "
                                 f"“{args['task']}”. Which task did you mean?", "task")
        r["task_id"] = tk.id
        r["task_project_id"] = tk.project_id
        if name == "assign_task":
            us = ruser(args["assignee"])
            if us.ambiguous:
                return None, ambiguous("person", us, "assignee")
            if us.missing:
                return None, clarify(f"Who should I assign it to? I couldn't match "
                                     f"“{args['assignee']}”.", "assignee")
            r["assignee_id"] = us.id
        if name == "set_due_date":
            r["due_date"] = args["due_date"]
        if name == "change_task_status":
            r["status"] = args["status"]
        if name == "move_task_to_sprint":
            # Scoped to the TASK's own project (tk.project_id, just
            # resolved above) — an unscoped search would treat two
            # same-named sprints in different projects ("Sprint 1" is a
            # common name) as ambiguous even though only one of them is
            # actually reachable from this task.
            spr = rsprint(args["sprint"], tk.project_id)
            if spr.ambiguous:
                return None, ambiguous("sprint", spr, "sprint")
            if spr.missing:
                return None, clarify(f"Which sprint? I couldn't match "
                                     f"“{args['sprint']}”.", "sprint")
            r["sprint_id"] = spr.id
        return r, None

    if name in ("start_sprint", "complete_sprint"):
        pid = None
        if args.get("project"):
            pr = rproject(args["project"])
            pid = pr.id
        spr = rsprint(args["sprint"], pid)
        if spr.ambiguous:
            return None, ambiguous("sprint", spr, "sprint")
        if spr.missing:
            return None, clarify(f"Which sprint? I couldn't match "
                                 f"“{args['sprint']}”.", "sprint")
        return {"sprint_id": spr.id}, None

    if name == "archive_project":
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify("Which project should I archive?", "project")
        return {"project_id": pr.id}, None

    if name == "query_tasks":
        tech = args.get("technology")
        if args.get("mine"):
            return {"mine": True, "technology": tech}, None
        if not args.get("project"):
            # No project named at all — "list all tasks" means every task,
            # every project, not "please pick one" (that clarify only makes
            # sense once a project WAS named but didn't resolve, below).
            return {"technology": tech}, None
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify(f"I couldn't find a project matching "
                                 f"“{args['project']}”. Which project?", "project")
        return {"project_id": pr.id, "technology": tech}, None

    if name in ("project_summary", "sprint_summary"):
        r: dict = {}
        if args.get("count") is not None:
            r["count"] = args["count"]
        if not args.get("project"):
            # No project named — summarize across everything, same
            # "omit means every project" convention as query_tasks above.
            return r, None
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify(f"I couldn't find a project matching "
                                 f"“{args['project']}”. Which project?", "project")
        r["project_id"] = pr.id
        return r, None

    if name == "list_projects":
        return {}, None

    if name == "list_spaces":
        return {}, None

    if name == "list_roles":
        if not real:
            return None, clarify("Role management isn't available in this "
                                 "demo — it's a real-app-only feature.", None)  # terminal decline, not a follow-up question
        return {}, None

    if name == "list_users":
        if not args.get("name"):
            return {"technology": args.get("technology")}, None
        us = ruser(args["name"])
        if us.ambiguous:
            return None, ambiguous("person", us, "name")
        if us.missing:
            return None, clarify(f"I couldn't find anyone matching "
                                 f"“{args['name']}”.", "name")
        return {"user_id": us.id}, None

    if name == "delete_user":
        us = ruser(args["user"])
        if us.ambiguous:
            return None, ambiguous("person", us, "user")
        if us.missing:
            return None, clarify(f"I couldn't find anyone matching "
                                 f"“{args['user']}”. Who should I delete?", "user")
        return {"user_id": us.id}, None

    if name == "task_info":
        tk = rtask(args["task"])
        if tk.ambiguous:
            return None, ambiguous("task", tk, "task")
        if tk.missing:
            return None, clarify(f"I couldn't find a task matching "
                                 f"“{args['task']}”. Which task?", "task")
        return {"task_id": tk.id, "task_project_id": tk.project_id}, None

    if name == "task_duration_check":
        if args.get("task"):
            tk = rtask(args["task"])
            if tk.ambiguous:
                return None, ambiguous("task", tk, "task")
            if tk.missing:
                return None, clarify(f"I couldn't find a task matching "
                                     f"“{args['task']}”. Which task?", "task")
            return {"task_id": tk.id, "task_project_id": tk.project_id}, None
        r: dict = {}
        if not args.get("project"):
            # No task, no project — every overrunning task the user can see.
            return r, None
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify(f"I couldn't find a project matching "
                                 f"“{args['project']}”. Which project?", "project")
        r["project_id"] = pr.id
        return r, None

    if name == "message_task":
        if not real:
            return None, clarify("Task messages aren't available in this demo — "
                                 "they're a real-app-only feature.", None)  # terminal decline, not a follow-up question
        tk = rtask(args["task"])
        if tk.ambiguous:
            return None, ambiguous("task", tk, "task")
        if tk.missing:
            return None, clarify(f"I couldn't find a task matching "
                                 f"“{args['task']}”. Which task?", "task")
        r["task_id"] = tk.id
        r["task_project_id"] = tk.project_id
        r["text"] = args["text"].strip()
        if args.get("mention"):
            mu = ruser(args["mention"])
            if mu.ambiguous:
                return None, ambiguous("person", mu, "mention")
            if mu.id:
                r["mention_id"] = mu.id
                r["mention_name"] = args["mention"]
        return r, None

    if name == "message_project":
        if not real:
            return None, clarify("Project messages aren't available in this "
                                 "demo — they're a real-app-only feature.", None)  # terminal decline, not a follow-up question
        pr = rproject(args["project"])
        if pr.ambiguous:
            return None, ambiguous("project", pr, "project")
        if pr.missing:
            return None, clarify(f"I couldn't find a project matching "
                                 f"“{args['project']}”. Which project?", "project")
        r["project_id"] = pr.id
        r["text"] = args["text"].strip()
        if args.get("mention"):
            mu = ruser(args["mention"])
            if mu.ambiguous:
                return None, ambiguous("person", mu, "mention")
            if mu.id:
                r["mention_id"] = mu.id
                r["mention_name"] = args["mention"]
        return r, None

    if name == "list_personal_tasks":
        if not real:
            return None, clarify("Personal to-dos aren't available in this demo — "
                                 "they're a real-app-only feature.", None)  # terminal decline, not a follow-up question
        return {}, None

    if name == "save_note":
        if not real:
            return None, clarify("Personal notes aren't available in this demo — "
                                 "they're a real-app-only feature.", None)  # terminal decline, not a follow-up question
        return {"title": args.get("title"), "body": args["body"]}, None

    if name == "move_personal_task":
        if not real:
            return None, clarify("Personal to-dos aren't available in this demo — "
                                 "they're a real-app-only feature.", None)  # terminal decline, not a follow-up question
        pt = tools._resolve(args["task"], [{"id": t["personal_task_id"], "name": t["title"]}
                                           for t in real_pm_service.list_personal_tasks(token)])
        if pt.ambiguous:
            return None, ambiguous("personal task", pt, "task")
        if pt.missing:
            return None, clarify(f"I couldn't find a personal task matching "
                                 f"“{args['task']}”. Which one?", "task")
        quadrant = args["quadrant"]
        urgent, important = {
            "do_first": (True, True), "schedule": (False, True),
            "delegate": (True, False), "eliminate": (False, False),
        }[quadrant]
        return {"personal_task_id": pt.id, "urgent": urgent, "important": important}, None

    if name == "graph_query":
        r["question"] = args["question"]
        if args.get("scope"):
            sc = rproject(args["scope"])
            if sc.id is None:
                sc = rspace(args["scope"])
            if sc.id is None:
                sp = rsprint(args["scope"])
                sc = sp
            if sc.id:
                r["scope_id"] = sc.id
        if args.get("assignee"):
            us = ruser(args["assignee"])
            if us.ambiguous:
                return None, ambiguous("person", us, "assignee")
            if us.id:
                r["assignee_id"] = us.id
                r["assignee_name"] = args["assignee"]
            elif args["question"] == "tasks_for":
                return None, clarify(f"I couldn't match the person "
                                     f"“{args['assignee']}”.", "assignee")
        if args["question"] == "tasks_for" and "assignee_id" not in r:
            return None, clarify("Whose tasks would you like to see?", "assignee")
        return r, None

    if name == "semantic_search":
        return {"query": args["query"]}, None

    return None, clarify(f"I don't know how to do “{name}” yet.", "name")


def requires_confirmation(user: dict, name: str) -> bool:
    """True if `name` needs an explicit yes/no before tools.execute() runs
    it — the fixed PM destructive whitelist (§11), any MCP-sourced tool
    from the shared static config (mcp_client.py), any per-user
    OAuth-connected MCP tool (user_mcp_client.py — GitHub/Linear), OR a
    write-capable Google tool (google_tools.py — draft/send email,
    create a calendar event, send a Chat message; the two READ-only
    Google tools deliberately do NOT require confirmation, same as any
    other read). An MCP tool (shared or per-user) calls an EXTERNAL
    server/account this codebase doesn't control and can't audit the
    same way a hand-written PM action or a JSON plugin (whitelisted to
    one fixed pm_service method) already is — so every MCP tool call
    gets the same "ask permission first" gate a destructive PM action
    gets, not just the ones that happen to also be destructive; a
    per-user OAuth tool doubly so, since it can write to the CALLING
    USER'S OWN real external account. Centralized here so all four
    tracks apply the identical rule at their identical gating point,
    §6.5's equivalence rule."""
    return (name in tools.DESTRUCTIVE or name in mcp_client.mcp_tool_names()
           or name in user_mcp_client.user_mcp_tool_names(user["sub"])
           or name in google_tools.WRITE_TOOLS)


def confirm_prompt_for(user: dict, name: str, args: dict) -> str:
    """Step 10 — the confirmation question with the resolved target named."""
    if name in google_tools.WRITE_TOOLS:
        # Same "name the account, don't over-explain" shape as the
        # per-user OAuth MCP branch below — the concrete action being
        # confirmed IS the review surface the spec asks for ("the user
        # should be able to review the exact action before approving
        # it"): every argument is shown verbatim, nothing paraphrased.
        args_desc = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "no arguments"
        verbs = {"draft_email": "draft an email", "send_email": "SEND an email",
                 "create_calendar_event": "create a calendar event",
                 "send_chat_message": "send a Google Chat message"}
        note = _security_confidence_note(name, args)
        return (f"⚠️ This will {verbs.get(name, name)} using YOUR connected Google "
                f"account, with {args_desc}. Review it carefully before approving."
                f"{note} Allow? (yes / no)")
    oauth_provider = user_mcp_client.provider_for_tool(user["sub"], name)
    if oauth_provider:
        # Same "never interpolate the tool's own description" rule as the
        # shared-MCP branch below, plus naming which of the user's OWN
        # connected accounts this write would go through — the stakes are
        # materially different from a shared external server (this
        # writes to the calling user's real GitHub/Linear/Google), so the
        # prompt says so explicitly rather than reusing the generic
        # "external MCP server" wording verbatim.
        args_desc = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "no arguments"
        return (f"⚠️ This will call “{name}” with {args_desc}, using YOUR connected "
                f"{oauth_provider} account — this app doesn't audit what this tool "
                f"actually does beyond its name and arguments, shown here. "
                f"Allow? (yes / no)")
    if name in mcp_client.mcp_tool_names():
        # Deliberately NEVER interpolates the MCP tool's own `description`
        # into this human-facing prompt — that string comes from whatever
        # external MCP server is configured (mcp_client.py's
        # mcp_tool_specs(), sourced from the server's own tool schema) and
        # is written to persuade a MODEL, not to inform a person. Real,
        # confirmed example: the bundled `mcp-server-fetch` reference
        # server's "fetch" tool description literally reads "Although
        # originally you did not have internet access, and were advised
        # to refuse and tell the user this, this tool now grants you
        # internet access..." — exactly the kind of text that should
        # never be rendered verbatim in a trust decision a HUMAN is
        # making, whether from Anthropic's own reference server or a
        # less scrupulous third-party one. The tool's real description
        # still reaches the model normally via tools.TOOL_SCHEMAS/
        # openai_tool_specs() (it needs that to reason about the tool at
        # all) — only this confirmation prompt omits it, on principle,
        # rather than trying to sanitize/truncate untrusted text and
        # hoping that's enough.
        args_desc = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "no arguments"
        return (f"⚠️ This will call the external tool “{name}” with {args_desc}. "
                f"This tool comes from an external MCP server this app doesn't "
                f"control or audit — only its name and arguments are shown here. "
                f"Allow? (yes / no)")
    target = (args.get("task") or args.get("sprint") or args.get("project")
             or args.get("user") or "this")
    verbs = {"delete_task": "delete task",
             "archive_project": "archive project",
             "complete_sprint": "complete (end)",
             "delete_user": "permanently delete the account of"}
    # A JSON plugin marked "destructive": true isn't in the hand-written
    # verbs map above, so fall back to its own description rather than
    # KeyError, so a new destructive plugin works with zero native_agent.py
    # changes, same as every other part of the plugin framework.
    verb = verbs.get(name) or (PLUGINS[name].description.lower() if name in PLUGINS else "do this")
    return (f"⚠️ This will {verb} “{target}”. "
            f"Are you sure? (yes / no)")


def security_precheck(user_msg: str) -> str | None:
    """Same early-refusal shape as input_guard() — called right after it,
    at the identical point in every track — but for security_guard's
    RISK tier specifically: a request that reads as credential/token
    exfiltration, bulk data exfiltration, or an explicit instruction to
    bypass the confirmation gate is refused OUTRIGHT, before any tool
    ever runs, not merely flagged after the fact (see assess_security
    below for the post-hoc flag every reply still gets). Returns the
    refusal text, or None to proceed normally."""
    assessment = security_guard.classify(user_msg, source="user_prompt")
    if assessment.flag == security_guard.RISK:
        return ("⚠️ I can't help with that as written — it looks like a request that "
                "could expose credentials, exfiltrate data, or bypass a safety "
                "control. If this was a false positive, please rephrase.")
    return None


def assess_security(user_id: str, user_msg: str, reply_text: str) -> tuple[str, list[str]]:
    """The 🟢/🟡/🔴 flag (security_guard.SAFE/NEEDS_REVIEW/RISK) + findings
    for THIS reply — aggregated across the two text surfaces every reply
    path has: the user's own prompt, and the final reply going back out
    (which, once a tool ran, already contains any email/calendar/chat
    content that was read — see google_tools.py/user_mcp_client.py's
    tool results, rendered into `reply` by format_result()). Called once
    per reply, at every track's single exit point (finish()/AgentReply
    construction) — §6.5 equivalence. Never raises/blocks on its own —
    security_precheck above is the (separate, earlier) gate that can
    actually refuse a turn; this is the flag EVERY reply gets, safe or not.

    Any non-safe flag is also logged to security_events.py — the source
    of truth for the admin Security dashboard and daily report
    (routers/admin_security.py, security_report.py)."""
    result = security_guard.aggregate([
        security_guard.classify(user_msg, source="user_prompt"),
        security_guard.classify(reply_text or "", source="generated_reply"),
    ])
    # log_event() itself no-ops for the common "safe" case (see its own
    # docstring) — for the rare non-safe case it's a real synchronous
    # store().put(), deferred here since the CALLER only needs
    # result.flag/findings (already computed above), never the fact that
    # this write has completed.
    run_in_background(security_events.log_event, user_id, flag=result.flag,
                      findings=result.findings, excerpt=reply_text or user_msg)
    return result.flag, result.findings


def _security_confidence_note(name: str, args: dict) -> str:
    """security_optimizer.py's learned note (if any) for a pending
    write-action's own NEEDS_REVIEW findings — additive text appended to
    the confirmation prompt, never changing WHETHER confirmation is
    required. Empty string (not None) so callers can always interpolate
    it directly without a None check."""
    assessment = security_guard.classify(f"{name} {args}", source="user_prompt")
    notes = [n for f in assessment.findings
            if (n := security_optimizer.confidence_note(f.split(":", 1)[0], "user_prompt"))]
    return f" (Note: {'; '.join(notes)}.)" if notes else ""


def record_security_feedback(findings: list[str], *, approved: bool) -> None:
    """The real reward signal security_optimizer.py's RL layer learns
    from — called from every track's yes/no confirmation handling, right
    where a destructive/MCP/Google-write confirmation is resolved. A
    no-op when `findings` is empty (most confirmations carry none —
    security_guard.py has nothing to say about a plain "delete this
    task" confirmation, only about content it actually flagged)."""
    if findings:
        # Ends in a real bandit_persistence.save() (disk/DynamoDB write) —
        # deferred since nothing reads this call's completion back.
        run_in_background(security_optimizer.record_feedback, findings,
                          source="user_prompt", approved=approved)


def handle_message(user: dict, session_id: str, user_msg: str,
                   framework_label: str = "native",
                   preferred_provider: str | None = None,
                   target: str | None = None) -> AgentReply:
    """Steps 1–17, in order. `target` — see ChatMessageIn.target's own
    docstring: "trading" nudges perceive_intent toward the trading
    tools for a finance-shaped question that isn't obviously a tool
    call otherwise, real fix for a genuine live bug (a backtest request
    on the Trading tab got the PM-specific out-of-scope treatment)."""
    # 1. receive  ── 2. load_session_state
    state = sessions.load(user["sub"], session_id)
    state["activeFramework"] = framework_label
    tier_log: list[dict] = []
    client = None      # bound below at step 5 — None here so finish() (called
                       # from paths BEFORE step 5 too, e.g. an input_guard
                       # refusal) can safely check it without UnboundLocalError.
    hybrid_selection = None  # set below (llm/hybrid_router.py) only when
                       # Hybrid mode is actually selected — same reason as client.
    domain_selection = None  # set below (llm/domain_router.py) only when
                       # Domain mode is actually selected — same reason as client.
    rag_result = None  # set below (gmaofn/rag_optimizer.py) if retrieval was
                       # actually attempted this turn — same reason as client.

    def finish(reply: str, *, pending: dict | None = None,
               clarify: dict | None = None, out_of_scope: bool = False) -> AgentReply:
        # security_guard's flag for this reply, computed FIRST — pending
        # (if any) needs it attached before sessions.save() below
        # persists it, so security_optimizer.py's RL layer can read it
        # back once the real yes/no answer to THIS confirmation is known
        # (see record_security_feedback). A store's put() flushes
        # synchronously — mutating `pending` after save() would silently
        # never reach disk/Dynamo.
        security_flag, security_findings = assess_security(user["sub"], user_msg, reply)
        if pending is not None:
            pending["security_findings"] = security_findings
        # 16. persist_session_state (append turn, clear resolved, trim, TTL)
        prior_messages = list(state["messages"])
        state["messages"].append({"role": "user", "content": user_msg})
        state["messages"].append({"role": "assistant", "content": reply})
        state["pendingConfirmation"] = pending
        state["pendingClarification"] = clarify
        state["modelTierLog"] = state.get("modelTierLog", []) + tier_log
        sessions.save(user["sub"], session_id, state)
        # Nightly-fine-tuning data collection (finetune_log.py) — only for
        # a genuine completed answer (nothing still pending/unresolved),
        # and only when a real LLM call actually happened (tier_log is
        # empty when a guard refused before ever reaching one, e.g.
        # input_guard) — see that module's docstring for the reward
        # heuristic "retried"/"out_of_scope" feed into.
        if pending is None and clarify is None and tier_log:
            # None of this is needed for the reply already computed above,
            # and nothing later in THIS request (or a follow-up request on
            # this session) reads any of it back synchronously — it's pure
            # learning signal (fine-tune log rows, bandit posteriors) that
            # only matters by the NEXT time a routing decision is made,
            # already an eventually-consistent story throughout this
            # codebase (see e.g. domain_router.py's own "loses to
            # GENERAL_ARM until it has real evidence" framing). Several of
            # these calls end in a real synchronous disk/DynamoDB write
            # (bandit_persistence.save() inside rag_optimizer/domain_router,
            # finetune_log's file append) — deferred to a background thread
            # (app/background.py) so the user's reply is never held up
            # waiting for them, per the "cached weights + fast routing +
            # fast reply, heavy bookkeeping after" architecture.
            retried = any(entry.get("node") == "perceive_intent(retry)" for entry in tier_log)

            def _record_learning_signals() -> None:
                finetune_log.log_ollama_turn(
                    messages=prior_messages + [{"role": "user", "content": user_msg}],
                    reply=reply, provider=tier_log[-1]["provider"],
                    retried=retried, out_of_scope=out_of_scope,
                    routing_strategy=tier_log[-1].get("routingStrategy"),
                    primary_pipeline=tier_log[-1].get("primaryPipeline"),
                    primary_pipeline_mode=tier_log[-1].get("primaryPipelineMode"))
                # Hybrid mode escalated away from the local SLM this turn
                # (llm/hybrid_router.py's LOCAL_TIMEOUT_SECONDS, or a genuine
                # error) — log the hosted model's answer too, as a training
                # example of what the local model should learn to do itself.
                if (client is not None and client.preferred_escalated
                       and hybrid_selection is not None and hybrid_selection.arm == "local"):
                    finetune_log.log_hybrid_escalation_turn(
                        messages=prior_messages + [{"role": "user", "content": user_msg}],
                        reply=reply, hosted_provider=tier_log[-1]["provider"],
                        retried=retried, out_of_scope=out_of_scope)
                # Token optimization layer (llm/token_optimizer.py) — finalize
                # here rather than at the HTTP-call site: only now is it known
                # whether the (possibly-trimmed) tool schema still led to a
                # genuinely completed answer, not just an HTTP-level success.
                success = not retried and not out_of_scope
                if client is not None:
                    for selection in client.tool_selections:
                        token_optimizer.record_outcome(selection, success=success)
                # RAG optimizer (gmaofn/rag_optimizer.py) — same deferred-
                # finalize timing as the token optimizer above.
                if rag_result is not None and rag_result.selection is not None:
                    rag_optimizer.record_outcome(rag_result.selection, success=success,
                                                 n_hits=rag_result.n_hits)
                # Hybrid mode (llm/hybrid_router.py) — same deferred-finalize
                # timing: only now is it known whether "local" actually held
                # up, or should have gone to a hosted model instead.
                if hybrid_selection is not None:
                    hybrid_router.record_outcome(
                        hybrid_selection, success=success,
                        latency_ms=sum(e.get("ms", 0) or 0 for e in tier_log))
                # Domain mode (llm/domain_router.py) — same deferred-finalize
                # timing: only now is it known whether the domain SLM this
                # bandit picked actually held up.
                if domain_selection is not None:
                    domain_router.record_outcome(domain_selection, success=success)

            run_in_background(_record_learning_signals)
        # Per-call token estimate for THIS turn (not the running average) —
        # surfaced to the UI so a user can see the local model's own saving.
        token_stats = None
        if client is not None and client.tool_selections:
            sel = client.tool_selections[-1]
            saved_pct = (round((sel.tokens_before - sel.tokens_after) / sel.tokens_before * 100, 1)
                        if sel.tokens_before else 0.0)
            token_stats = {"tokens_before": sel.tokens_before, "tokens_after": sel.tokens_after,
                          "tokens_saved_pct": saved_pct}
        # 17. return to widget (WS broadcast already fired from the service)
        return AgentReply(reply=reply, pending_confirmation=pending is not None,
                          confirm_prompt=pending and pending.get("prompt"),
                          tier_log=tier_log, out_of_scope=out_of_scope,
                          provider=tier_log[-1]["provider"] if tier_log else None,
                          token_stats=token_stats,
                          sources_used=rag_result.sources_used if rag_result else [],
                          usage=tier_log[-1].get("usage") if tier_log else None,
                          context_limit_switch=any(e.get("truncated") for e in tier_log),
                          security_flag=security_flag, security_findings=security_findings)

    # 3. pendingConfirmation? → treat purely as yes/no (step 9b/11)
    if state.get("pendingConfirmation"):
        pc = state["pendingConfirmation"]
        if YES.match(user_msg):
            record_security_feedback(pc.get("security_findings") or [], approved=True)
            try:
                result = tools.execute(user, pc["tool"], pc["args"])   # 12
            except Exception as e:                                     # 13
                return finish(f"Sorry — that failed: {e}")
            return finish(format_result(result))                       # 15
        if NO.match(user_msg):
            record_security_feedback(pc.get("security_findings") or [], approved=False)
            return finish("Okay — no changes made.")                   # 11 'no'
        return finish(pc["prompt"] + " (please answer yes or no)",
                      pending=pc)

    # 3b. pendingClarification? → this turn IS the answer to step 8's
    # question, not a fresh request (see resume_clarification above) —
    # UNLESS it looks like a brand-new structured command instead (see
    # looks_like_a_fresh_command), in which case fall through to the
    # normal flow below rather than swallowing it as a failed answer.
    if state.get("pendingClarification") and not looks_like_a_fresh_command(user_msg):
        reply, pending, clarify = resume_clarification(
            user, state["pendingClarification"], user_msg)
        return finish(reply, pending=pending, clarify=clarify)

    # input guard (§12.3 — just before perceive_intent)
    ok, refusal = input_guard(user_msg)
    if not ok:
        return finish(refusal)

    # security_guard's RISK-tier precheck — a second, broader gate than
    # input_guard's narrow injection-pattern refusal above (credential/
    # data exfiltration requests, "skip confirmation" instructions).
    blocked = security_precheck(user_msg)
    if blocked:
        return finish(blocked)

    # 4. classify_task_complexity (local, non-LLM) ── 5. select_model
    tier = classify_task_complexity(user_msg)
    # Fast conversational path (see is_pure_conversational's own
    # docstring) — a plain greeting/chit-chat never needed the
    # domain-routing bandit consultation OR a RAG relevance check below,
    # both skipped for this class of message; the tool schema itself is
    # skipped further down (still the SAME tier/provider selection as
    # every other turn — this never changes WHICH model answers, only
    # how much irrelevant schema/context it's asked to reason over).
    is_conversational = is_pure_conversational(user_msg)
    # Captured BEFORE the hybrid/domain rewrites below — a hybrid pick
    # that resolves to "hosted" also sets preferred_provider back to
    # None, and that must NOT be mistaken for "the caller sent nothing"
    # by the auto-domain branch further down.
    _caller_sent_nothing = preferred_provider is None
    # Hybrid mode (llm/hybrid_router.py) — a UI-selectable third option
    # alongside "Auto" and picking one specific model: RL decides, per
    # message, whether this looks simple enough for the free local SLM
    # or complicated enough to prefer a hosted model. Only intercepts the
    # literal "hybrid" sentinel; any real provider name or None passes
    # through unchanged (§3.4's normal preferred-provider behavior).
    if preferred_provider == "hybrid":
        hybrid_selection = hybrid_router.choose(user_msg, tier)
        preferred_provider = hybrid_router.preferred_provider_for(hybrid_selection)
    # Domain mode (llm/domain_router.py) — a fourth UI-selectable option:
    # RL decides, per message, WHICH of possibly several registered
    # domain-specific Algo-Akanade-SLM variants (or none, falling back to
    # the normal chain) this query is most likely to be answered well by.
    # Only intercepts the literal "domain" sentinel — mutually exclusive
    # with "hybrid" above (a caller sends one or the other).
    if preferred_provider == "domain":
        domain_selection = domain_router.choose(user_msg, tier)
        preferred_provider = domain_router.preferred_provider_for(domain_selection)
    # True "auto" (no explicit mode/model chosen at all) — since the RL
    # domain bandit above is only ever consulted when a caller literally
    # sends "domain", a registered, genuinely-winning fine-tuned SLM
    # never got a chance to answer a plain "auto" turn. Same choose()
    # call as the explicit "domain" branch, just also applied here; a
    # cold/untrained domain naturally loses to GENERAL_ARM's own seeded
    # prior (see DomainRouter's docstring) so this is a no-op until a
    # domain SLM has real evidence of being the better pick — gated by
    # is_confidently_relevant(), NOT just choose()'s raw winner: choose()'s
    # bandit score is fit purely on message complexity (word/clause count),
    # which has no notion of topic, so it can favor e.g. the "coding" arm
    # for a short trading question just as readily as a short coding one.
    # Confirmed live: this cost ~10s (LOCAL_TIMEOUT_SECONDS) per turn on a
    # Trading Agent conversation before the real answer even started.
    elif _caller_sent_nothing and not is_conversational:
        domain_selection = domain_router.choose(user_msg, tier)
        if not domain_router.is_confidently_relevant(domain_selection):
            domain_selection = None
        else:
            preferred_provider = domain_router.preferred_provider_for(domain_selection)
    # Only "local" (hybrid) or an actual domain pick ever gets the shorter
    # escalation timeout — both are real Ollama-backed local rungs that
    # can be genuinely slow on CPU; an explicit "hosted"/"general" pick
    # (None override, normal chain order) has no local rung to escalate
    # away from.
    preferred_timeout = (LOCAL_TIMEOUT_SECONDS
                         if (hybrid_selection is not None and hybrid_selection.arm == "local")
                         or (domain_selection is not None and preferred_provider is not None)
                         else None)
    # Real bug found live: the "rule" floor's reply is a generic PM-tool
    # capability menu — right for a plain PM chat that exhausted every
    # real rung, but wrong and confusing for a Trading Agent conversation
    # (target=="trading"), same reasoning CodingAgent's own calls already
    # apply (see llm/router.py's TierClient.exclude_rule_based).
    client = get_llm_client(tier, preferred=preferred_provider,
                            preferred_timeout=preferred_timeout,
                            exclude_rule_based=(target == "trading"))       # §3.4

    # RAG optimizer (gmaofn/rag_optimizer.py) — only attempted for
    # info-seeking messages (is_rag_relevant), and only injected as extra
    # context if it actually found something; never gates or replaces the
    # tool-calling flow below, and never bypasses input_guard/output_guard/
    # requires_confirmation — see that module's own guardrail note.
    # Best-effort: a broken knowledge store must never break core PM chat.
    rag_context_messages: list[dict] = []
    if not is_conversational and is_rag_relevant(user_msg):
        try:
            rag_result = retrieve_with_optimizer(user_msg, store_names=rag_registry.list_names())
            if rag_result.context:
                rag_context_messages = [{"role": "system",
                                        "content": f"Potentially relevant context, if useful:\n"
                                                   f"{rag_result.context}"}]
        except Exception:
            rag_result = None

    # 6. perceive_intent — one LLM call with the full tool schema attached.
    # memory_context (empty until the rolling window has ever trimmed
    # anything — see sessions.save/memory.fold_into_summary) keeps whatever
    # scrolled out of state["messages"][-6:] present in gist form instead
    # of just gone.
    memory_context = [{"role": "system",
                       "content": f"Summary of earlier conversation: {state['summary']}"}] \
        if state.get("summary") else []
    target_hint = [TRADING_TARGET_SYSTEM] if target == "trading" else []
    messages = [PERCEIVE_INTENT_SYSTEM] + target_hint + memory_context + rag_context_messages + \
        state["messages"][-6:] + [{"role": "user", "content": user_msg}]
    # user_openai_tool_specs (user_mcp_client.py) adds THIS user's own
    # OAuth-connected tools (GitHub/Linear/Google) on top of the shared
    # global set — never merged into tools.TOOL_SCHEMAS itself, since
    # that dict is shared by every user/request (see tools.py's
    # user_openai_tool_specs docstring).
    user_tool_specs = tools.openai_tool_specs() + tools.user_openai_tool_specs(user)
    user_tool_schemas = {**tools.TOOL_SCHEMAS, **tools.user_tool_schemas(user)}
    # Real, measured cost: the full tool schema above is ~4600 tokens
    # (41 tools) — real prompt-processing latency a small model pays on
    # EVERY turn, even "Are you up?". None here for a pure-conversational
    # message (see is_pure_conversational) — the "no tool call" path
    # below (output_guard(None, ...) passes trivially, "# small talk")
    # already handles a plain-text reply correctly with no tool schema
    # attached at all.
    res = client.chat("perceive_intent", messages, None if is_conversational else user_tool_specs)
    tier_log.extend(client.log)

    # output guard (§12.3): schema-validate before resolve_entities;
    # on failure re-ask ONCE on the same tier (§12.2)
    ok, why = output_guard(res.tool_call, user_tool_schemas)
    if not ok:
        res = client.chat("perceive_intent(retry)", messages +
                          [{"role": "system",
                            "content": f"Previous tool call invalid: {why}. Retry."}],
                          user_tool_specs)
        tier_log.extend(client.log[len(tier_log):] or client.log[-1:])
        ok, why = output_guard(res.tool_call, user_tool_schemas)
        if not ok:
            return finish("I couldn't turn that into a valid action "
                          f"({why}). Could you rephrase?")

    if not res.tool_call:                                              # small talk
        return finish(res.text or "How can I help with your projects?",
                      out_of_scope=bool(res.text) and res.provider != "rule-based")

    name, args = res.tool_call["name"], res.tool_call["arguments"]

    # 7. resolve_entities ── 8. ambiguous/missing → clarify & end turn
    resolved, clarify = resolve_arguments(user, name, args)
    if clarify:
        if clarify["field"] is None:
            # Terminal decline — see resume_clarification's identical check above.
            return finish(clarify["question"])
        return finish(clarify["question"],
                      clarify={"tool": name, "args": args, "field": clarify["field"],
                              "candidates": clarify.get("candidates")})

    # 9. is_destructive? — constant lookup, never the LLM's judgement
    if requires_confirmation(user, name):
        # 10. confirm_with_user: persist {tool,args}, ask, end turn
        prompt = confirm_prompt_for(user, name, args)
        return finish(prompt, pending={"tool": name, "args": resolved,
                                       "prompt": prompt})

    # 11–12. execute_tool (sole mutating path, caller's own identity)
    try:
        result = tools.execute(user, name, resolved)
    except Exception as e:                                             # 13
        return finish(f"Sorry — that failed: {e}")

    # 14. invalidation hooks already fired inside the service layer
    # 15. format_response — natural language, never a JSON dump
    return finish(format_result(result))
