"""Track A (§6.3) — the SAME §6.1 algorithm as a real LangGraph StateGraph.

Node names match the design's table one-for-one: load_state →
perceive_intent → resolve_entities → route → (clarify | confirm | execute)
→ format_response → persist_state. The confirmation gate uses LangGraph's
native interrupt(): the graph genuinely pauses at confirm_with_user, its
checkpoint persists, and the next invoke resumes it with Command(resume=…).

Shared with the native track (deliberately — §6.5's equivalence demand):
tools.py, sessions.py, guardrails, the tier router, and format_result.
The checkpointer is LangGraph's MemorySaver, with the durable session
window mirrored into the same ChatSessions table the native track uses.
"""
from __future__ import annotations

import re
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from . import finetune_log, sessions
from ..framework.rag import rag_registry
from ..framework.rag_optimizer import is_rag_relevant, rag_optimizer, retrieve_with_optimizer
from ..llm.domain_router import domain_router
from ..llm.guardrails import input_guard, output_guard
from ..llm.hybrid_router import LOCAL_TIMEOUT_SECONDS, hybrid_router
from ..llm.router import get_llm_client
from ..llm.token_optimizer import token_optimizer
from . import tools
from .. import security_guard
from .native_agent import (PERCEIVE_INTENT_SYSTEM, TRADING_TARGET_SYSTEM, YES, assess_security,
                           classify_task_complexity, confirm_prompt_for, is_pure_conversational,
                           looks_like_a_fresh_command, record_security_feedback,
                           requires_confirmation, resolve_arguments, resume_clarification,
                           security_precheck)
from .state import AgentReply, format_result


class BotState(TypedDict, total=False):
    user: dict
    session_id: str
    user_msg: str
    preferred_provider: Optional[str]
    # See ChatMessageIn.target's own docstring / native_agent.py's
    # identical addition (§6.5 equivalence) — "trading" nudges
    # perceive_intent toward the trading tools.
    target: Optional[str]
    # The rolling summary of whatever's already scrolled out of
    # state["messages"]'s window (see sessions.py/memory.fold_into_summary)
    # — the memory-optimization layer's read side; handle_message passes
    # it in from sessions.load(), perceive_intent prepends it if non-empty.
    summary: str
    tier: str
    tool_name: Optional[str]
    tool_args: Optional[dict]
    resolved: Optional[dict]
    # The specific args key resolve_arguments was asking about, when it
    # returns a clarify — see native_agent.resolve_arguments's own
    # docstring on why "field" is tracked rather than guessed on resume.
    clarify_field: Optional[str]
    # The specific candidates that WERE ambiguous, when clarify_field is
    # set due to ambiguity rather than a plain miss — see native_agent's
    # ambiguous()/resume_clarification.
    clarify_candidates: Optional[list]
    reply: str
    tier_log: list
    out_of_scope: bool
    # In-memory only, never persisted (unlike tier_log) — carries
    # llm/token_optimizer.py's Selection objects out of perceive_intent so
    # handle_message can finalize their reward once retried/out_of_scope
    # are known; see that module's docstring.
    tool_selections: list
    # gmaofn/rag_optimizer.py's selection + citation info for THIS turn —
    # same in-memory-only, not-persisted convention as tool_selections.
    rag_selection: object
    rag_sources_used: list
    rag_n_hits: int
    # llm/hybrid_router.py's selection for THIS turn, when Hybrid mode
    # was actually chosen (preferred_provider == "hybrid") — same
    # in-memory-only convention as tool_selections/rag_selection.
    hybrid_selection: object
    # llm/domain_router.py's selection for THIS turn, when Domain mode
    # was actually chosen (preferred_provider == "domain") — same
    # in-memory-only convention as hybrid_selection above.
    domain_selection: object
    # TierClient.preferred_escalated for THIS turn — same in-memory-only
    # convention; see hybrid_selection above and native_agent.py's
    # identical field on its own `client`.
    preferred_escalated: bool


# ── nodes (names per §6.3's table) ───────────────────────────────────────

def perceive_intent(s: BotState) -> BotState:                 # steps 4-6
    ok, refusal = input_guard(s["user_msg"])                  # input guard
    if not ok:
        return {**s, "reply": refusal, "tool_name": None}
    blocked = security_precheck(s["user_msg"])                # security_guard RISK gate
    if blocked:
        return {**s, "reply": blocked, "tool_name": None}
    tier = classify_task_complexity(s["user_msg"])
    # Fast conversational path — same native_agent.py addition (§6.5
    # equivalence): see is_pure_conversational's own docstring.
    is_conversational = is_pure_conversational(s["user_msg"])
    # Hybrid mode (llm/hybrid_router.py) — same interception as
    # native_agent.py's identical block (§6.5 equivalence).
    preferred_provider = s.get("preferred_provider")
    _caller_sent_nothing = preferred_provider is None
    hybrid_selection = None
    if preferred_provider == "hybrid":
        hybrid_selection = hybrid_router.choose(s["user_msg"], tier)
        preferred_provider = hybrid_router.preferred_provider_for(hybrid_selection)
    # Domain mode (llm/domain_router.py) — same interception as
    # native_agent.py's identical block (§6.5 equivalence).
    domain_selection = None
    if preferred_provider == "domain":
        domain_selection = domain_router.choose(s["user_msg"], tier)
        preferred_provider = domain_router.preferred_provider_for(domain_selection)
    # True "auto" — same native_agent.py addition (§6.5 equivalence):
    # let a genuinely-winning fine-tuned domain SLM answer a plain auto
    # turn too, not just an explicit "domain" pick. Gated by
    # is_confidently_relevant() — see native_agent.py's identical block
    # for why choose()'s raw winner alone isn't a safe signal here.
    elif _caller_sent_nothing and not is_conversational:
        domain_selection = domain_router.choose(s["user_msg"], tier)
        if not domain_router.is_confidently_relevant(domain_selection):
            domain_selection = None
        else:
            preferred_provider = domain_router.preferred_provider_for(domain_selection)
    preferred_timeout = (LOCAL_TIMEOUT_SECONDS
                         if (hybrid_selection is not None and hybrid_selection.arm == "local")
                         or (domain_selection is not None and preferred_provider is not None)
                         else None)
    # Real bug found live (native_agent.py's identical fix, §6.5
    # equivalence): the "rule" floor's generic PM-tool capability menu is
    # wrong for a Trading Agent conversation — never let this tier's
    # chain fall through to it for target=="trading".
    client = get_llm_client(tier, preferred=preferred_provider, preferred_timeout=preferred_timeout,
                            exclude_rule_based=(s.get("target") == "trading"))
    memory_context = [{"role": "system",
                       "content": f"Summary of earlier conversation: {s['summary']}"}] \
        if s.get("summary") else []

    # RAG optimizer (gmaofn/rag_optimizer.py) — same best-effort,
    # never-gates-anything convention as native_agent.py's identical
    # block (§6.5 equivalence).
    rag_context_messages: list[dict] = []
    rag_selection, rag_sources_used, rag_n_hits = None, [], 0
    if not is_conversational and is_rag_relevant(s["user_msg"]):
        try:
            rag_result = retrieve_with_optimizer(s["user_msg"], store_names=rag_registry.list_names())
            rag_selection, rag_sources_used, rag_n_hits = (
                rag_result.selection, rag_result.sources_used, rag_result.n_hits)
            if rag_result.context:
                rag_context_messages = [{"role": "system",
                                        "content": f"Potentially relevant context, if useful:\n"
                                                   f"{rag_result.context}"}]
        except Exception:
            pass

    # user_openai_tool_specs (user_mcp_client.py) adds THIS user's own
    # OAuth-connected tools (GitHub/Linear/Google) on top of the shared
    # global set — same per-request merge native_agent.py does (§6.5
    # equivalence), never into the process-global tools.TOOL_SCHEMAS.
    user_tool_specs = tools.openai_tool_specs() + tools.user_openai_tool_specs(s["user"])
    user_tool_schemas = {**tools.TOOL_SCHEMAS, **tools.user_tool_schemas(s["user"])}
    # Same real, measured fast-path as native_agent.py: None here skips
    # the ~4600-token/41-tool schema for a pure greeting/chit-chat turn.
    target_hint = [TRADING_TARGET_SYSTEM] if s.get("target") == "trading" else []
    res = client.chat("perceive_intent",
                      [PERCEIVE_INTENT_SYSTEM] + target_hint + memory_context + rag_context_messages +
                      [{"role": "user", "content": s["user_msg"]}],
                      None if is_conversational else user_tool_specs)
    log = list(client.log)
    ok, why = output_guard(res.tool_call, user_tool_schemas)  # output guard
    if not ok:
        res = client.chat("perceive_intent(retry)",
                          [{"role": "user", "content": s["user_msg"]},
                           {"role": "system", "content": f"Invalid: {why}. Retry."}],
                          user_tool_specs)
        log = list(client.log)
        ok, why = output_guard(res.tool_call, user_tool_schemas)
        if not ok:
            return {**s, "tier": tier, "tier_log": log, "tool_name": None,
                    "tool_selections": list(client.tool_selections),
                    "rag_selection": rag_selection, "rag_sources_used": rag_sources_used,
                    "rag_n_hits": rag_n_hits, "hybrid_selection": hybrid_selection,
                    "domain_selection": domain_selection,
                    "preferred_escalated": client.preferred_escalated,
                    "reply": f"I couldn't turn that into a valid action ({why})."}
    if not res.tool_call:
        return {**s, "tier": tier, "tier_log": log, "tool_name": None,
                "tool_selections": list(client.tool_selections),
                "rag_selection": rag_selection, "rag_sources_used": rag_sources_used,
                "rag_n_hits": rag_n_hits, "hybrid_selection": hybrid_selection,
                "domain_selection": domain_selection,
                "preferred_escalated": client.preferred_escalated,
                "reply": res.text or "How can I help with your projects?",
                "out_of_scope": bool(res.text) and res.provider != "rule-based"}
    return {**s, "tier": tier, "tier_log": log,
            "tool_selections": list(client.tool_selections),
            "rag_selection": rag_selection, "rag_sources_used": rag_sources_used,
            "rag_n_hits": rag_n_hits, "hybrid_selection": hybrid_selection,
            "domain_selection": domain_selection,
            "preferred_escalated": client.preferred_escalated,
            "tool_name": res.tool_call["name"],
            "tool_args": res.tool_call["arguments"]}


def resolve_entities(s: BotState) -> BotState:                # step 7
    if not s.get("tool_name"):
        return s
    resolved, clarify = resolve_arguments(s["user"], s["tool_name"],
                                          s["tool_args"])
    if clarify:
        return {**s, "resolved": None, "reply": clarify["question"],
                "clarify_field": clarify["field"],
                "clarify_candidates": clarify.get("candidates")}
    return {**s, "resolved": resolved}


def route(s: BotState) -> str:                                # steps 8-9
    if not s.get("tool_name") or s.get("resolved") is None:
        return "clarify"                                      # or small talk
    if requires_confirmation(s["user"], s["tool_name"]):
        return "destructive"
    return "proceed"


def ask_clarifying_question(s: BotState) -> BotState:         # step 8
    return s  # reply already set by perceive/resolve


def confirm_with_user(s: BotState) -> BotState:               # steps 10-11
    prompt = confirm_prompt_for(s["user"], s["tool_name"], s["tool_args"])
    # Unlike the other three tracks (which persist security_findings onto
    # a plain `pending` dict across two separate handle_message calls),
    # this track's confirmation is a single LangGraph interrupt()/resume
    # — no intermediate dict to carry the assessment through, so it's
    # simplest to just recompute it fresh, right here, once `answer` is
    # known (§6.5 equivalence: same real signal, different mechanics).
    assessment = security_guard.classify(f"{s['tool_name']} {s['tool_args']}", source="user_prompt")
    answer = interrupt({"prompt": prompt})   # ← graph PAUSES here; the
    # checkpoint persists; the next invoke resumes with the user's answer
    if isinstance(answer, str) and YES.match(answer):
        record_security_feedback(assessment.findings, approved=True)
        return s                                              # → execute
    record_security_feedback(assessment.findings, approved=False)
    return {**s, "tool_name": None, "reply": "Okay — no changes made."}


def execute_tool(s: BotState) -> BotState:                    # steps 12-14
    if not s.get("tool_name"):
        return s
    try:
        result = tools.execute(s["user"], s["tool_name"], s["resolved"])
    except Exception as e:
        return {**s, "reply": f"Sorry — that failed: {e}"}
    return {**s, "reply": format_result(result)}              # step 15 inline


def format_response(s: BotState) -> BotState:                 # step 15
    return s


# ── graph wiring (mirrors the design's §6.3 sketch) ──────────────────────

def build_graph():
    g = StateGraph(BotState)
    g.add_node("perceive_intent", perceive_intent)
    g.add_node("resolve_entities", resolve_entities)
    g.add_node("ask_clarifying_question", ask_clarifying_question)
    g.add_node("confirm_with_user", confirm_with_user)
    g.add_node("execute_tool", execute_tool)
    g.add_node("format_response", format_response)

    g.set_entry_point("perceive_intent")
    g.add_edge("perceive_intent", "resolve_entities")
    g.add_conditional_edges("resolve_entities", route, {
        "clarify": "ask_clarifying_question",
        "destructive": "confirm_with_user",
        "proceed": "execute_tool",
    })
    g.add_edge("confirm_with_user", "execute_tool")
    g.add_edge("ask_clarifying_question", "format_response")
    g.add_edge("execute_tool", "format_response")
    g.add_edge("format_response", END)
    return g.compile(checkpointer=MemorySaver())


_app = build_graph()


def handle_message(user: dict, session_id: str, user_msg: str,
                   framework_label: str = "langgraph",
                   preferred_provider: str | None = None,
                   target: str | None = None) -> AgentReply:
    """Steps 1-3 + 16-17 wrap the graph; the graph is steps 4-15."""
    state = sessions.load(user["sub"], session_id)             # step 2
    state["activeFramework"] = framework_label
    cfg = {"configurable": {"thread_id": f"{user['sub']}:{session_id}"}}

    # step 3b: this turn IS the answer to a paused step 8 clarifying
    # question, not a fresh request — resolved outside the graph (see
    # resume_clarification's docstring for why neither track restructures
    # its flow just to reopen one paused turn) — UNLESS it looks like a
    # brand-new structured command instead (see looks_like_a_fresh_command).
    if (not state.get("pendingConfirmation") and state.get("pendingClarification")
            and not looks_like_a_fresh_command(user_msg)):
        reply, pending, clarify = resume_clarification(
            user, state["pendingClarification"], user_msg)
        state["messages"].append({"role": "user", "content": user_msg})
        state["messages"].append({"role": "assistant", "content": reply})
        state["pendingConfirmation"] = pending
        state["pendingClarification"] = clarify
        sessions.save(user["sub"], session_id, state)
        security_flag, security_findings = assess_security(user["sub"], user_msg, reply)
        return AgentReply(reply=reply, pending_confirmation=pending is not None,
                          confirm_prompt=pending and pending.get("prompt"), tier_log=[],
                          security_flag=security_flag, security_findings=security_findings)

    if state.get("pendingConfirmation"):                       # step 3 → resume
        out = _app.invoke(Command(resume=user_msg), cfg)
    else:
        out = _app.invoke({"user": user, "session_id": session_id,
                           "user_msg": user_msg, "tier_log": [],
                           "preferred_provider": preferred_provider, "target": target,
                           "summary": state.get("summary", "")}, cfg)

    # a fresh interrupt? LangGraph surfaces it in the state snapshot
    snap = _app.get_state(cfg)
    pending = None
    if snap.next and "confirm_with_user" in snap.next or snap.tasks:
        for t in snap.tasks:
            for intr in getattr(t, "interrupts", []) or []:
                pending = {"prompt": intr.value["prompt"]}
    reply = (pending["prompt"] if pending
             else out.get("reply", "Done."))

    # step 8's clarify route (not small talk — tool_name present but
    # resolve_entities couldn't finish) needs the same resumable memory
    # pendingConfirmation already gets, just persisted rather than paused
    # via interrupt() (§6.1 step 8 ends the turn, it doesn't suspend it).
    clarify = None
    # A None clarify_field means resolve_entities returned a TERMINAL
    # decline ("that's not available in this demo"), not a real
    # follow-up question — leave pendingClarification unset so the
    # user's NEXT, unrelated message isn't silently swallowed as if it
    # were answering this one (native_agent.py's identical check, §6.5
    # equivalence).
    if (not pending and out.get("tool_name") and out.get("resolved") is None
            and out.get("clarify_field") is not None):
        clarify = {"tool": out["tool_name"], "args": out.get("tool_args") or {},
                  "field": out.get("clarify_field"),
                  "candidates": out.get("clarify_candidates")}

    # step 16 — persist into the SAME ChatSessions table as native
    prior_messages = list(state["messages"])
    state["messages"].append({"role": "user", "content": user_msg})
    state["messages"].append({"role": "assistant", "content": reply})
    state["pendingConfirmation"] = pending
    state["pendingClarification"] = clarify
    turn_tier_log = out.get("tier_log") or []
    state["modelTierLog"] = state.get("modelTierLog", []) + turn_tier_log
    sessions.save(user["sub"], session_id, state)              # step 17 returns

    # Nightly-fine-tuning data collection — same gate/heuristic as
    # native_agent.py's identical call (§6.5 equivalence): only a
    # genuine completed answer, only when a real LLM call happened.
    if pending is None and clarify is None and turn_tier_log:
        retried = any(entry.get("node") == "perceive_intent(retry)" for entry in turn_tier_log)
        out_of_scope_flag = bool(not pending and out.get("out_of_scope"))
        finetune_log.log_ollama_turn(
            messages=prior_messages + [{"role": "user", "content": user_msg}],
            reply=reply, provider=turn_tier_log[-1]["provider"],
            retried=retried, out_of_scope=out_of_scope_flag)
        # Hybrid mode escalated away from the local SLM this turn — same
        # "log the hosted answer as SLM training data" as native_agent.py's
        # identical block (§6.5 equivalence).
        hs = out.get("hybrid_selection")
        if out.get("preferred_escalated") and hs is not None and hs.arm == "local":
            finetune_log.log_hybrid_escalation_turn(
                messages=prior_messages + [{"role": "user", "content": user_msg}],
                reply=reply, hosted_provider=turn_tier_log[-1]["provider"],
                retried=retried, out_of_scope=out_of_scope_flag)
        # Token optimization layer (llm/token_optimizer.py) — same
        # finalize-with-the-real-outcome timing as native_agent.py's
        # identical block (§6.5 equivalence).
        success = not retried and not out_of_scope_flag
        for selection in out.get("tool_selections") or []:
            token_optimizer.record_outcome(selection, success=success)
        # RAG optimizer (gmaofn/rag_optimizer.py) — same deferred-finalize
        # timing as the token optimizer above.
        rag_selection = out.get("rag_selection")
        if rag_selection is not None:
            rag_optimizer.record_outcome(rag_selection, success=success,
                                         n_hits=out.get("rag_n_hits") or 0)
        # Hybrid mode (llm/hybrid_router.py) — same deferred-finalize
        # timing as the token/RAG optimizers above.
        hybrid_selection = out.get("hybrid_selection")
        if hybrid_selection is not None:
            hybrid_router.record_outcome(
                hybrid_selection, success=success,
                latency_ms=sum(e.get("ms", 0) or 0 for e in turn_tier_log))
        # Domain mode (llm/domain_router.py) — same deferred-finalize
        # timing as the token/RAG optimizers above.
        domain_selection = out.get("domain_selection")
        if domain_selection is not None:
            domain_router.record_outcome(domain_selection, success=success)

    token_stats = None
    tool_selections = out.get("tool_selections") or []
    if tool_selections:
        sel = tool_selections[-1]
        saved_pct = (round((sel.tokens_before - sel.tokens_after) / sel.tokens_before * 100, 1)
                    if sel.tokens_before else 0.0)
        token_stats = {"tokens_before": sel.tokens_before, "tokens_after": sel.tokens_after,
                      "tokens_saved_pct": saved_pct}

    security_flag, security_findings = assess_security(user["sub"], user_msg, reply)
    return AgentReply(reply=reply, pending_confirmation=pending is not None,
                      confirm_prompt=pending and pending["prompt"],
                      tier_log=out.get("tier_log") or [],
                      out_of_scope=bool(not pending and out.get("out_of_scope")),
                      provider=turn_tier_log[-1]["provider"] if turn_tier_log else None,
                      token_stats=token_stats, sources_used=out.get("rag_sources_used") or [],
                      usage=turn_tier_log[-1].get("usage") if turn_tier_log else None,
                      context_limit_switch=any(e.get("truncated") for e in turn_tier_log),
                      security_flag=security_flag, security_findings=security_findings)
