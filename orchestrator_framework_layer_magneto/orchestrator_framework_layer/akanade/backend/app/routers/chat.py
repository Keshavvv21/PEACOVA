"""§10 — the chat endpoints: POST /api/chat/message, GET /api/chat/history,
WS /ws/chat, GET /api/chat/metrics (admin-only, feeds the bake-off)."""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from ..agent import sessions
from ..agent import framework_stats
from ..agent.adapter import get_agent
from ..agent.native_agent import is_pure_conversational
from ..auth import current_user, decode_token, require_role
from ..config import settings
from ..framework import chat_usage
from ..framework.intent_cache import conversational_cache
from ..framework.query_refinement import query_refinement
from ..llm.router import available_providers
from ..models import ChatMessageIn, ChatReply
from ..services.ws import manager

router = APIRouter()


@router.post("/api/chat/message", response_model=ChatReply)
async def chat_message(body: ChatMessageIn,
                       framework: str | None = Query(default=None),
                       user: dict = Depends(current_user)):
    # No explicit ?framework= (or the literal "auto"): let
    # framework_stats.choose() pick whichever of native/langgraph has
    # been faster and more reliable so far, and fall back to the other
    # one if it errors — the framework-level equivalent of the model
    # router's tier fallback chain. An explicit ?framework= (including
    # "adk") is always honored exactly as asked, no auto-picking or
    # fallback layered on top of it.
    #
    # EXCEPT mid-pause: a destructive confirmation on the LangGraph track
    # is a real interrupt() living in ITS OWN checkpoint (see
    # langgraph_agent.py), which native has no access to and stores a
    # differently-shaped pendingConfirmation for ({"prompt"} only vs
    # native's {"tool","args","prompt"}) — auto-switching tracks for that
    # one resume turn 500s with a bare KeyError. Pinning to whichever
    # track asked the question (persisted as the session's own
    # activeFramework) is the ONLY safe choice for that turn, so the
    # picker — and its own fallback-on-error below — are both bypassed
    # while a confirmation or clarification is pending.
    auto = framework is None or framework.lower() == "auto"
    pinned = None
    if auto:
        state = sessions.load(user["sub"], body.session_id)
        if state.get("pendingConfirmation") or state.get("pendingClarification"):
            pinned = state.get("activeFramework") or "native"
    # Query refinement (gmaofn/query_refinement.py) — runs ONCE here, above
    # every agent track, so native/langgraph/adk all see the same refined
    # text without each needing their own copy of this logic (unlike
    # hybrid_router.py, which genuinely differs per track's LLM call site).
    # Skipped entirely on a pinned turn (a "yes"/edit reply resuming a
    # pending confirmation/clarification) — rewriting that reply for
    # "clarity" would risk mangling the exact word the resume logic expects.
    # Intent-answer cache (framework/intent_cache.py) — only ever
    # consulted for messages is_pure_conversational() already judges as
    # pure small talk (greetings/thanks/chit-chat): the one class of
    # message the pipeline already treats as safe to answer without any
    # tool schema or domain routing (see native_agent.py's own use of
    # this same check). Checked on the RAW message, BEFORE refinement:
    # refinement's own bandit gate can itself choose to spend an LLM
    # round trip, and paying that on a hit would defeat the entire point
    # of the cache (measured live: 7.5s on a "hit" when the gate chose
    # to refine). Keyed on the raw text for the same reason, so lookup
    # and write always agree. A hit skips refinement, agent selection,
    # and handle_message entirely — zero model calls for this turn.
    # `not body.provider` is real, not defensive: a caller who explicitly
    # picked a model (or "hybrid"/"domain" mode) is asking for THAT
    # decision to actually run this turn, never a cached answer from
    # a prior, possibly-different pick — same "an explicit choice is
    # never second-guessed" rule this codebase applies everywhere else
    # (see coding_agent.py's identical fix, found via the same real bug).
    cacheable = (not pinned) and settings().INTENT_CACHE_ENABLED and not body.provider \
        and is_pure_conversational(body.message)
    if cacheable:
        cache_t0 = time.time()
        cached = conversational_cache.lookup(body.message)
        if cached is not None:
            cache_tier_log = [{"node": "intent_cache", "tier": "cache",
                               "provider": cached.provider, "fallbackUsed": False,
                               "ms": round((time.time() - cache_t0) * 1000, 1),
                               "cacheHit": True}]
            # handle_message's own finish() is what normally appends the
            # turn and flushes it — a hit skips that entirely, so the
            # exchange has to be persisted here or it would be missing
            # from /api/chat/history and from the next turn's rolling
            # context window.
            state = sessions.load(user["sub"], body.session_id)
            state["messages"].append({"role": "user", "content": body.message})
            state["messages"].append({"role": "assistant", "content": cached.text})
            state["modelTierLog"] = state.get("modelTierLog", []) + cache_tier_log
            sessions.save(user["sub"], body.session_id, state)
            # No-op here (cache_tier_log's own cacheHit=True entry is
            # filtered out by chat_usage.record_usage's real_calls check)
            # — called anyway for the same reason coding_agent.py always
            # calls its own record_usage(): one call site, correct
            # either way, never two copies of "should this turn bill?"
            # logic to keep in sync.
            chat_usage.record_usage(user_id=user["sub"], session_id=body.session_id,
                                    framework="cache", tier_log=cache_tier_log)
            await manager.broadcast({"type": "chat.reply", "user_id": user["sub"],
                                     "session_id": body.session_id, "reply": cached.text})
            return ChatReply(reply=cached.text, session_id=body.session_id, framework="cache",
                             tier_log=cache_tier_log,
                             provider=f"cache:{cached.provider}", security_flag="safe")
    # Query refinement (gmaofn/query_refinement.py) — runs ONCE here, above
    # every agent track, so native/langgraph/adk all see the same refined
    # text without each needing their own copy of this logic (unlike
    # hybrid_router.py, which genuinely differs per track's LLM call site).
    # Skipped entirely on a pinned turn (a "yes"/edit reply resuming a
    # pending confirmation/clarification) — rewriting that reply for
    # "clarity" would risk mangling the exact word the resume logic expects.
    refinement = None if pinned else query_refinement.refine(body.message)
    message_for_agent = refinement.refined_text if refinement else body.message
    agent = get_agent(pinned or (framework_stats.choose() if auto else framework))
    t0 = time.time()
    try:
        reply = agent.handle_message(user, body.session_id, message_for_agent,
                                     preferred_provider=body.provider, target=body.target)
    except Exception:
        if pinned or not (auto and agent.name in framework_stats.TRACKS):
            if refinement is not None:
                query_refinement.record_outcome(refinement.selection, success=False)
            raise
        framework_stats.record(agent.name, int((time.time() - t0) * 1000), error=True)
        agent = get_agent(next(t for t in framework_stats.TRACKS if t != agent.name))
        t0 = time.time()
        reply = agent.handle_message(user, body.session_id, message_for_agent,
                                     preferred_provider=body.provider, target=body.target)
    if agent.name in framework_stats.TRACKS:
        framework_stats.record(agent.name, int((time.time() - t0) * 1000), error=False)
    if refinement is not None:
        query_refinement.record_outcome(refinement.selection, success=not reply.out_of_scope)
    # out_of_scope is deliberately NOT a guard here: small talk is ALWAYS
    # flagged out_of_scope (it's free text, not grounded in PM data) —
    # that's the expected state for exactly the class of message this
    # cache accepts, not a failure signal. The rule-based floor IS
    # excluded: its generic capability-menu reply is what every failed
    # chain degrades to, and caching that would pin a degraded answer
    # for the whole TTL.
    if (cacheable and not reply.pending_confirmation
            and reply.provider and reply.provider != "rule-based"):
        conversational_cache.remember(body.message, reply.reply, reply.provider)
    # mutations already broadcast from the service layer; push the chat
    # reply too so other open widgets of the same user can mirror it
    await manager.broadcast({"type": "chat.reply", "user_id": user["sub"],
                             "session_id": body.session_id,
                             "reply": reply.reply})
    refinement_out = None
    if refinement is not None and refinement.selection is not None and refinement.selection.arm == "refine":
        refinement_out = {
            "original": refinement.original_text, "refined": refinement.refined_text,
            "was_repaired": refinement.was_repaired,
            "similarity": round(refinement.similarity, 3) if refinement.similarity is not None else None,
            "missing_clarifications": refinement.missing_clarifications,
            "suggested_queries": refinement.suggested_queries,
            "needs_confirmation": refinement.needs_confirmation,
        }
    chat_usage.record_usage(user_id=user["sub"], session_id=body.session_id,
                            framework=agent.name, tier_log=reply.tier_log)
    return ChatReply(reply=reply.reply,
                     pending_confirmation=reply.pending_confirmation,
                     confirm_prompt=reply.confirm_prompt,
                     session_id=body.session_id, framework=agent.name,
                     tier_log=reply.tier_log, out_of_scope=reply.out_of_scope,
                     provider=reply.provider, token_stats=reply.token_stats,
                     sources_used=reply.sources_used, refinement=refinement_out,
                     usage=reply.usage, context_limit_switch=reply.context_limit_switch,
                     security_flag=reply.security_flag, security_findings=reply.security_findings)


@router.get("/api/chat/framework-stats")
def chat_framework_stats(_: dict = Depends(require_role("Admin"))):
    """What the auto-picker above is actually seeing — per-track calls,
    errors, and average latency, live since this process started."""
    return framework_stats.snapshot()


@router.get("/api/chat/refinement-audit")
def chat_refinement_audit(_: dict = Depends(require_role("Admin"))):
    """Every drifted query-refinement case since this process started —
    where the LLM's "clarified" rewrite diverged too far (embedding
    similarity below threshold) from what the user actually typed, and
    was reverted. See gmaofn/query_refinement.py's DriftAuditLog."""
    return {"drifted": query_refinement.drift_log.snapshot()}


@router.get("/api/chat/providers")
def chat_providers(_: dict = Depends(current_user)):
    """Feeds the UI's model picker — every provider in any tier's chain,
    with whether it's actually configured/reachable right now."""
    return {"providers": available_providers()}


@router.get("/api/chat/history")
def chat_history(session_id: str = "default",
                 user: dict = Depends(current_user)):
    state = sessions.load(user["sub"], session_id)
    return {"messages": state["messages"],
            "pending_confirmation": bool(state.get("pendingConfirmation"))}


@router.delete("/api/chat/history", status_code=204)
def clear_chat_history(session_id: str = "default",
                       user: dict = Depends(current_user)):
    """The "Clear chat" button — wipes THIS user's own session back to
    fresh (their own token, their own session_id — never another user's,
    same rule every other chat endpoint follows)."""
    sessions.clear(user["sub"], session_id)


@router.get("/api/chat/metrics")
def chat_metrics(_: dict = Depends(require_role("Admin"))):
    """Aggregated per-tier latency/error/fallback stats from modelTierLog
    — the §3.4 Deepseek evaluation and framework bake-off feed."""
    logs = sessions.all_tier_logs()
    agg: dict = defaultdict(lambda: {"calls": 0, "errors": 0,
                                     "fallbacks": 0, "ms_total": 0})
    for row in logs:
        key = f"{row['tier']}/{row['provider']}"
        a = agg[key]
        a["calls"] += 1
        a["errors"] += 1 if row.get("error") else 0
        a["fallbacks"] += 1 if row.get("fallbackUsed") else 0
        a["ms_total"] += row.get("ms", 0)
    return {k: {**v, "avg_ms": round(v["ms_total"] / max(v["calls"], 1), 1)}
            for k, v in sorted(agg.items())}


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, token: str = Query(default="")):
    try:
        decode_token(token)                   # token on connect (§10)
    except Exception:
        await ws.close(code=4401)
        return
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()           # keepalive/pings; events flow out
    except WebSocketDisconnect:
        manager.disconnect(ws)
