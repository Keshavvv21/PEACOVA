"""Shared shapes every track produces/consumes."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentReply:
    """§2.2 — what AgentHandle.handle_message returns, identically for
    native, LangGraph, and ADK tracks."""
    reply: str
    pending_confirmation: bool = False
    confirm_prompt: str | None = None
    tier_log: list[dict] = field(default_factory=list)
    # True when perceive_intent answered in free text instead of calling a
    # PM tool — i.e. the request wasn't something any registered PM action
    # covers. The widget gates reveal of an out-of-scope answer behind an
    # explicit accept, rather than presenting a general-knowledge/LLM-
    # opinion answer with the same trust as a real PM lookup. Left False
    # for the plain "how can I help" fallback (empty model text, e.g. a
    # bare greeting) — that's not an answer to gate, just a prompt for
    # more detail.
    out_of_scope: bool = False
    # Which provider actually answered this turn (e.g. "ollama-qwen2.5:1.5b",
    # "groq-llama-3.3-70b") — tier_log[-1]["provider"] surfaced for the UI,
    # so a user can see which model (local vs hosted) handled their request.
    provider: str | None = None
    # llm/token_optimizer.py's per-call estimate for THIS turn specifically
    # (not the running average) — {"tokens_before", "tokens_after",
    # "tokens_saved_pct"} when the local model's tool schema was reduced
    # this turn, else None (every other provider, or no tools were sent).
    token_stats: dict | None = None
    # gmaofn/rag_optimizer.py's citation list for THIS turn — [] when RAG
    # wasn't attempted (is_rag_relevant() said no) or returned nothing.
    sources_used: list[dict] = field(default_factory=list)
    # The provider's OWN reported token usage for THIS turn's actual
    # answering call — tier_log[-1]["usage"] surfaced for the UI, same
    # {"prompt_tokens", "completion_tokens", "total_tokens"} shape every
    # provider normalizes to (see llm/providers.py). None for a provider
    # that doesn't report usage (e.g. LangChainOllama) — distinct from
    # token_stats above, which is the LOCAL tool-schema trim's estimate,
    # not a real usage figure from any provider's response.
    usage: dict | None = None
    # True when an EARLIER rung in this turn's chain got cut off by its
    # own context/output limit (LLMResult.truncated) and TierClient
    # escalated to a later provider instead of returning the incomplete
    # answer — surfaced so the UI can say WHY the model changed mid-turn,
    # distinct from a plain error-triggered fallback.
    context_limit_switch: bool = False
    # security_guard.py's aggregate flag for every real text surface this
    # turn touched (the user's prompt plus the final reply text going
    # out — see native_agent.assess_security) — "safe"/"needs_review"/
    # "risk", surfaced to the UI as a 🟢/🟡/🔴 badge. None only for reply
    # paths that predate this field ever being set (shouldn't happen in
    # practice — finish() always computes it — but kept optional rather
    # than defaulting to "safe" so a missed call site is visibly None,
    # not silently misreported as safe).
    security_flag: str | None = None
    security_findings: list[str] = field(default_factory=list)


def format_result(result: dict) -> str:
    """§6.1 step 15 — short natural-language confirmation, never a JSON
    dump. Deterministic templates; a Luna call can replace this wording
    layer without touching anything upstream."""
    k = result.get("kind")
    if not result.get("ok"):
        return f"Sorry — that didn't work: {result.get('error', 'unknown error')}."
    if k == "space":
        return f"Created space \u201c{result['name']}\u201d."
    if k == "project":
        if result.get("archived"):
            return f"Archived project \u201c{result['name']}\u201d."
        return f"Created project \u201c{result['name']}\u201d."
    if k == "task":
        if result.get("deleted"):
            return f"Deleted \u201c{result['title']}\u201d (moved to trash)."
        if "sprint" in result:
            due = f" Due {result['due_date']}." if result.get("due_date") else ""
            return f"Moved \u201c{result['title']}\u201d to {result['sprint']}.{due}"
        if "assignee" in result:
            return f"Assigned \u201c{result['title']}\u201d to {result['assignee']}."
        if "due_date" in result:
            return f"Due date for \u201c{result['title']}\u201d set to {result['due_date']}."
        if "status" in result:
            # Demo mode's PascalCase and real mode's snake_case TaskStatus
            # (see app/models/task.py on the real app's side) both land
            # here \u2014 readable either way rather than a raw "in_progress".
            labels = {"todo": "Todo", "inprogress": "In Progress", "backlog": "Backlog",
                     "done": "Done", "blocked": "Blocked", "hold": "On Hold"}
            status = labels.get(result["status"].lower().replace("_", ""), result["status"])
            return f"\u201c{result['title']}\u201d is now {status}."
        if "created_due_date" in result:
            return (f"Created task \u201c{result['title']}\u201d, "
                    f"due {result['created_due_date']}.")
        return f"Created task \u201c{result['title']}\u201d."
    if k == "subtask":
        if "done" in result:
            return f"Marked “{result['title']}” as {'done' if result['done'] else 'not done'}."
        return f"Added subtask “{result['title']}”."
    if k == "subtask_list":
        subs = result["subtasks"]
        if not subs:
            return "That task has no subtasks yet."
        lines = [f"• {'✅' if s['done'] else '⬜'} {s['title']}" for s in subs]
        done_n = sum(1 for s in subs if s["done"])
        return f"{done_n}/{len(subs)} subtask(s) done:\n" + "\n".join(lines)
    if k == "user":
        email_note = f" — {result['email']}" if result.get("email") else ""
        return f"Created user “{result['name']}” ({result['role']}){email_note}."
    if k == "user_deleted":
        return f"Deleted “{result['name']}”'s account."
    if k == "note_saved":
        title_note = f" “{result['title']}”" if result.get("title") else ""
        return f"Saved to your notes{title_note}."
    if k == "membership":
        return (f"Added {result['member']} to “{result['project']}” "
                f"as {result['role']}.")
    if k == "plugin_result":
        # A JSON tool plugin (../plugins.py) already rendered its own text
        # via response_template — format_result's only job here is to not
        # need to know anything about the specific plugin.
        return result["text"]
    if k == "mcp_tool_result":
        # An external MCP tool's own text content block(s) — same "already
        # rendered, nothing to reformat" shape as plugin_result above.
        return result["text"]
    if k == "sprint":
        # Demo mode's PascalCase ("Active"/"Completed") and real mode's
        # lowercase SprintStatus ("active"/"completed" \u2014 see
        # app/schemas/sprint_schemas.py on the real app's side) both land
        # here, so both spellings are checked rather than just one.
        state = (result.get("state") or "").lower()
        if state == "active":
            return f"Started {result['name']}."
        if state == "completed":
            return (f"Completed {result['name']}. Incomplete tasks moved back "
                    f"to the backlog.")
        return f"Created sprint \u201c{result['name']}\u201d."
    if k == "task_list":
        ts = result["tasks"]
        if not ts:
            return "No tasks found."
        lines = [f"• {t['title']} — {t['status']}"
                 + (f", assigned to {t['assignee']}" if t.get("assignee") else "")
                 + (f", due {t['due']}" if t.get("due") else "") for t in ts[:10]]
        more = f"\n(+{len(ts)-10} more)" if len(ts) > 10 else ""
        return f"{len(ts)} task(s):\n" + "\n".join(lines) + more
    if k == "project_summary":
        status_labels = {"todo": "Todo", "inprogress": "In Progress", "backlog": "Backlog",
                        "done": "Done", "blocked": "Blocked", "hold": "On Hold",
                        "unknown": "Unknown"}
        if "projects" in result:
            ps = result["projects"]
            if not ps:
                return "No projects found."
            lines = []
            for p in ps:
                timeline = (f", {p['start_date']} → {p['due_date']}"
                           if p.get("start_date") or p.get("due_date") else "")
                lines.append(f"• {p['name']} ({p.get('status', 'active')}) — "
                            f"{p['done_tasks']}/{p['total_tasks']} done, "
                            f"{p['overdue_count']} overdue{timeline}")
            return f"{len(ps)} project(s):\n" + "\n".join(lines)
        counts = ", ".join(f"{status_labels.get(s.lower().replace('_', ''), s)}: {n}"
                           for s, n in result["task_counts"].items())
        timeline = (f"\nTimeline: {result['start_date'] or '?'} → {result['due_date'] or '?'}"
                   if result.get("start_date") or result.get("due_date") else "")
        sprint_note = f"\n{result['sprint_count']} sprint(s)." if result.get("sprint_count") else ""
        return (f"“{result['name']}” ({result.get('status', 'active')}, "
                f"{result.get('methodology', 'kanban')}) — {result['total_tasks']} task(s): "
                f"{counts or 'none'}. {result['overdue_count']} overdue.{timeline}{sprint_note}")
    if k == "sprint_summary":
        ss = result["sprints"]
        if not ss:
            return "No sprints found."
        lines = []
        for s in ss:
            dates = (f", {s['start_date']} → {s['end_date']}"
                    if s.get("start_date") or s.get("end_date") else "")
            lines.append(f"• {s['name']} ({s['project']}) — {s.get('status', '?')}, "
                        f"{s['done_tasks']}/{s['total_tasks']} done{dates}")
        return f"{len(ss)} sprint(s):\n" + "\n".join(lines)
    if k == "project_list":
        ps = result["projects"]
        if not ps:
            return "No projects found."
        lines = [f"• {p['name']}" + (f" ({p['board']})" if p.get("board") else "") for p in ps]
        return f"{len(ps)} project(s):\n" + "\n".join(lines)
    if k == "space_list":
        ss = result["spaces"]
        if not ss:
            return "No spaces found."
        lines = [f"• {s['name']}" + (f" — {s['description']}" if s.get("description") else "") for s in ss]
        return f"{len(ss)} space(s):\n" + "\n".join(lines)
    if k == "user_list":
        us = result["users"]
        if not us:
            return "No users found."
        lines = [f"• {u['name']} — {u['role']}" for u in us]
        return f"{len(us)} user(s):\n" + "\n".join(lines)
    if k == "user_info":
        email_note = f" — {result['email']}" if result.get("email") else ""
        return f"{result['name']} is {result['role']}{email_note}."
    if k == "task_info":
        bits = [f"status {result['status']}"]
        if result.get("assignee"):
            bits.append(f"assigned to {result['assignee']}")
        if result.get("priority"):
            bits.append(f"{result['priority']} priority")
        if result.get("due"):
            bits.append(f"due {result['due']}")
        return f"“{result['title']}” — " + ", ".join(bits) + "."
    if k == "task_message":
        mention_note = f" (mentioned {result['mention']})" if result.get("mention") else ""
        return f"Posted on “{result['task']}”{mention_note}: {result['text']}"
    if k == "project_message":
        mention_note = f" (mentioned {result['mention']})" if result.get("mention") else ""
        return f"Posted on “{result['project']}”{mention_note}: {result['text']}"
    if k == "role_list":
        rs = result["roles"]
        if not rs:
            return "No roles found."
        lines = []
        for r in rs:
            extra = f", extra: {', '.join(r['extra_permissions'])}" if r.get("extra_permissions") else ""
            tag = " (built-in)" if r.get("is_system") else ""
            lines.append(f"• {r['name']} — base {r['base_rank']}{extra}{tag}")
        return f"{len(rs)} role(s):\n" + "\n".join(lines)
    if k == "personal_task_list":
        ts = result["tasks"]
        if not ts:
            return "Your personal to-do list is empty."
        quadrant = {(True, True): "Do First", (False, True): "Schedule",
                   (True, False): "Delegate", (False, False): "Eliminate"}
        lines = [f"• {t['title']} — {quadrant[(t['urgent'], t['important'])]}"
                + (" (done)" if t["done"] else "")
                + (f", due {t['due']}" if t.get("due") else "") for t in ts]
        return f"{len(ts)} personal task(s):\n" + "\n".join(lines)
    if k == "personal_task":
        quadrant = {(True, True): "Do First", (False, True): "Schedule",
                   (True, False): "Delegate", (False, False): "Eliminate"}
        return (f"Moved “{result['title']}” to "
               f"{quadrant[(result['urgent'], result['important'])]}.")
    if k == "overdue":
        ts = result["tasks"]
        if not ts:
            return "Nothing is overdue. 🎉"
        lines = [f"• {t['name']} — due {t['due']} ({t['status']})" for t in ts]
        return f"{len(ts)} overdue task(s):\n" + "\n".join(lines)
    if k == "workload":
        rows = result["rows"]
        if not rows:
            return "No open assigned tasks."
        lines = [f"• {r['name']}: {r['open_tasks']} open" for r in rows[:8]]
        return "Open tasks by person:\n" + "\n".join(lines)
    if k == "overdue_by_person":
        rows = result["rows"]
        if not rows:
            return "No one has overdue tasks."
        lines = [f"• {r['name']}: {r['overdue_tasks']} overdue" for r in rows[:8]]
        return "Overdue tasks by person:\n" + "\n".join(lines)
    if k == "tasks_for":
        ts = result["tasks"]
        who = result.get("assignee") or "them"
        if not ts:
            return f"No tasks assigned to {who}."
        lines = [f"• {t['name']} — {t['status']}" for t in ts[:10]]
        return f"{who} has {len(ts)} task(s):\n" + "\n".join(lines)
    if k == "duration_check":
        ts = result["tasks"]
        if not ts:
            return result.get("note") or "Nothing is running longer than planned. 🎉"

        def _line(t: dict) -> str:
            who = f" ({t['assignee']})" if t.get("assignee") else ""
            over = t["overrun_days"]
            if over > 0:
                state_txt = (f"finished {over} day(s) late"
                            if t["done"] else
                            f"running {over} day(s) over its planned {t['expected_days']}")
            elif t["done"]:
                state_txt = f"finished on time (planned {t['expected_days']} day(s))"
            else:
                state_txt = f"on track — {t['elapsed_days']}/{t['expected_days']} day(s) elapsed"
            return f"“{t['title']}”{who} — {state_txt}"

        if len(ts) == 1:
            return _line(ts[0]) + "."
        lines = [f"• {_line(t)}" for t in ts]
        return f"{len(ts)} task(s) running over their planned duration:\n" + "\n".join(lines)
    if k == "semantic":
        hits = result["hits"]
        if not hits:
            return "No similar tasks found."
        lines = [f"• {h['text'][:80]} (score {h['score']})" for h in hits]
        return "Closest matches:\n" + "\n".join(lines)
    if k == "price_history":
        last = result["points"][-1] if result["points"] else None
        last_line = f" Latest close: {last['close']} on {last['date']}." if last else ""
        return f"{result['n_points']} trading day(s) of price data for {result['symbol']}.{last_line}"
    if k == "peft_training_request":
        return (f"Logged this as a training example for the '{result['domain']}' domain "
                f"({result['chars_logged']} characters, from {result['source']}). "
                f"It'll be included in tonight's scheduled fine-tune run — training doesn't "
                f"happen immediately.")
    if k == "backtest":
        sharpe = f", Sharpe {result['sharpe_ratio']}" if result["sharpe_ratio"] is not None else ""
        cagr = f", CAGR {result['cagr_pct']}%" if result.get("cagr_pct") is not None else ""
        bh_cagr = (f" (buy-and-hold CAGR {result['buy_and_hold_cagr_pct']}%)"
                  if result.get("buy_and_hold_cagr_pct") is not None else "")
        chart_md = (f"\n\n![{result['symbol']} equity curve](data:image/png;base64,{result['chart_base64']})"
                   if result.get("chart_base64") else "")
        return (f"Backtest of {result['symbol']} ({result['fast_window']}/{result['slow_window']}-day "
                f"crossover, {result['start']} to {result['end']}): strategy return "
                f"{result['total_return_pct']}%{cagr} vs buy-and-hold {result['buy_and_hold_return_pct']}%"
                f"{bh_cagr}, {result['n_trades']} trade(s), annualized volatility "
                f"{result['annualized_volatility_pct']}%, max drawdown {result['max_drawdown_pct']}%"
                f"{sharpe}.{chart_md}")
    return "Done."
