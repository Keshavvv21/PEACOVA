"""Track B (§6.4) — Google ADK, built to the same §6.1 spec.

OPTIONAL: imported only when CHAT_AGENT_FRAMEWORK=adk (or ?framework=adk)
and google-adk is installed (per §14, both tracks may be installed; only
the active one is instantiated).

Real ADK machinery drives intent perception: an `LlmAgent` with one
`BaseTool` per entry in `tools.TOOL_SCHEMAS` (the exact same schema every
other track presents to a model — no drift), run through a real `Runner`
+ `InMemorySessionService`, exactly like a genuine ADK deployment would.
The one deliberate departure from a "normal" ADK app is `RouterLlm`
(below): instead of wiring in Gemini or LiteLLM, it's a `BaseLlm` that
forwards straight into `llm/router.py`'s `get_llm_client()`/`TierClient`
— the same fallback chain (Luna → Groq → Deepseek → Ollama → rule-based,
plus any admin-added/EXTRA_LLM_PROVIDERS rung) every other track walks.
That keeps the design's §16 lint rule ("get_llm_client() is the only
sanctioned entry to any LLM call") true for this track too, and means
adding a provider via the Admin AI Models UI lights up ADK exactly the
same way it lights up native/langgraph — no separate config surface.

Steps 7-15 (resolve_entities, destructive-confirm, execute, format) are
NOT reimplemented — they're imported verbatim from native_agent.py,
§6.5's equivalence rule enforced by imports, not discipline, same as
langgraph_agent.py. The graph is only ever driven for ONE turn (through
perceive_intent) and then explicitly closed — this track never lets ADK
execute a tool or ask the model to write a second, free-text "final
answer" turn, since step 15 (format_response) is deliberately
deterministic (agent/state.py's format_result) on every other track too;
letting ADK's own agent loop run to completion would silently make this
track's replies come from a different code path than native/langgraph's.

`handle_message()` stays a plain synchronous function — the same
Callable[..., AgentReply] contract adapter.py's AgentHandle expects for
every track — even though ADK's own APIs (Runner, SessionService) are
async-only. `_run_async()` bridges that by running the coroutine to
completion on a fresh event loop in its own thread, never touching
whatever loop the calling FastAPI worker is already running on.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from . import finetune_log, sessions, user_mcp_client
from ..framework.rag import rag_registry
from ..framework.rag_optimizer import is_rag_relevant, rag_optimizer, retrieve_with_optimizer
from ..llm.guardrails import input_guard, output_guard
from ..llm.domain_router import domain_router
from ..llm.hybrid_router import LOCAL_TIMEOUT_SECONDS, hybrid_router
from ..llm.providers import LLMResult
from ..llm.router import TierClient, get_llm_client
from ..llm.token_optimizer import token_optimizer
from . import tools
from .native_agent import (NO, PERCEIVE_INTENT_SYSTEM, TRADING_TARGET_SYSTEM, YES,
                          assess_security, classify_task_complexity, confirm_prompt_for,
                          is_pure_conversational, looks_like_a_fresh_command,
                          record_security_feedback, requires_confirmation, resolve_arguments,
                          resume_clarification, security_precheck)
from .state import AgentReply, format_result


def _run_async(coro):
    """Runs an ADK coroutine to completion on its own thread + event loop,
    so a sync handle_message (the contract every track shares) can call
    into ADK's async-only Runner/SessionService without colliding with
    the event loop the calling FastAPI worker is already running on
    (asyncio.run() would raise "cannot be called from a running event
    loop" if called directly from inside chat.py's async endpoint)."""
    outcome: dict[str, Any] = {}

    def _target():
        loop = asyncio.new_event_loop()
        try:
            outcome["value"] = loop.run_until_complete(coro)
        except Exception as e:  # noqa: BLE001 — re-raised on the caller's thread below
            outcome["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class RouterLlm(BaseLlm):
    """A BaseLlm that never talks to Gemini/LiteLLM itself — every call is
    forwarded to an already-constructed TierClient (see module docstring).
    `node` is the same tier_log label native_agent uses ("perceive_intent"
    / "perceive_intent(retry)"), so ADK-track metrics aggregate under the
    same keys as every other track's in GET /api/chat/metrics."""

    client: TierClient
    node: str = "perceive_intent"

    async def generate_content_async(self, llm_request, stream: bool = False):
        messages: list[dict] = []
        si = llm_request.config.system_instruction if llm_request.config else None
        if isinstance(si, str) and si:
            messages.append({"role": "system", "content": si})
        for content in (llm_request.contents or []):
            role = "assistant" if content.role == "model" else "user"
            text = "".join(p.text or "" for p in (content.parts or []) if p.text)
            if text:
                messages.append({"role": role, "content": text})
        tool_specs: list[dict] = []
        if llm_request.config and llm_request.config.tools:
            for t in llm_request.config.tools:
                for fd in (t.function_declarations or []):
                    tool_specs.append({
                        "name": fd.name, "description": fd.description or "",
                        "parameters": fd.parameters_json_schema
                                     or {"type": "object", "properties": {}, "required": []},
                    })
        res = self.client.chat(self.node, messages, tool_specs or None)
        if res.tool_call:
            part = types.Part(function_call=types.FunctionCall(
                name=res.tool_call["name"], args=res.tool_call["arguments"]))
        else:
            part = types.Part(text=res.text or "")
        yield LlmResponse(content=types.Content(role="model", parts=[part]), turn_complete=True)


class _SchemaTool(BaseTool):
    """One per TOOL_SCHEMAS entry, built straight from the SAME schema dict
    tools.openai_tool_specs() derives for every other track — never ADK's
    usual type-hint auto-derivation, so the model sees byte-identical tool
    definitions regardless of which track answers (§6.5). run_async is
    never actually invoked: _perceive() below always closes the run after
    the first event, before ADK's own loop would ever execute a tool."""

    def __init__(self, name: str, schema: dict):
        super().__init__(name=name, description=schema["description"])
        self._schema = schema

    def _get_declaration(self):
        return types.FunctionDeclaration(
            name=self.name, description=self.description,
            parameters_json_schema={"type": "object",
                                    "properties": self._schema["properties"],
                                    "required": self._schema["required"]})

    async def run_async(self, *, args, tool_context):
        raise RuntimeError(f"_SchemaTool({self.name!r}).run_async should never run — "
                           f"adk_agent._perceive() always closes the run after the "
                           f"first function-call event.")


# The SHARED tool set only — safe to build once at import time since it's
# identical for every user/request. Per-user OAuth-connected tools
# (user_mcp_client.py) are NOT included here (a module-global list can't
# safely carry per-user entries across concurrent requests) — _perceive()
# below takes an `extra_tools` param built fresh per call instead, the
# ADK-track equivalent of the other three tracks' per-request
# `tools.openai_tool_specs() + tools.user_openai_tool_specs(user)` merge.
_TOOLS = [_SchemaTool(name, schema) for name, schema in tools.TOOL_SCHEMAS.items()]


async def _perceive(client: TierClient, user_msg: str, history: list[dict],
                    node: str, extra_instruction: str | None = None,
                    summary: str | None = None,
                    extra_tools: list[_SchemaTool] | None = None,
                    tools_enabled: bool = True) -> LLMResult:
    """One perceive_intent turn through real ADK machinery: builds an
    LlmAgent + Runner + a fresh InMemorySessionService (seeded with the
    same rolling window native_agent's own perceive_intent call sees),
    runs it, and stops at the FIRST function-call-or-text event — ADK
    never gets to execute a tool or ask the model for a second, free-text
    reply (see module docstring on why that matters for §6.5 parity).

    `summary` is the memory-optimization layer's read side (see
    sessions.py/memory.fold_into_summary) — whatever's already scrolled
    out of `history`, folded down to a few sentences instead of gone.

    `extra_tools` — this turn's caller-specific tools (per-user OAuth
    MCP), appended to the shared `_TOOLS` for this call only.

    `tools_enabled=False` (see native_agent.py's is_pure_conversational)
    — same real, measured fast-path as the other two tracks: a pure
    greeting/chit-chat turn gets NO tools at all, not even the shared
    `_TOOLS` set, skipping the real prompt-processing cost of the full
    schema for a message that was never going to call one.
    """
    instruction = PERCEIVE_INTENT_SYSTEM["content"]
    if summary:
        instruction += f"\n\nSummary of earlier conversation: {summary}"
    if extra_instruction:
        instruction += f"\n\n{extra_instruction}"
    agent = LlmAgent(name="algopm_intent",
                     model=RouterLlm(model=f"router/{client.tier}", client=client, node=node),
                     instruction=instruction,
                     tools=(_TOOLS + (extra_tools or [])) if tools_enabled else [])
    session_service = InMemorySessionService()
    app_name, user_id = "algopm", "adk-track"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    for m in history:
        role = "model" if m["role"] == "assistant" else "user"
        await session_service.append_event(session=session, event=Event(
            author=role, content=types.Content(role=role, parts=[types.Part(text=m["content"])])))
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    new_message = types.Content(role="user", parts=[types.Part(text=user_msg)])
    gen = runner.run_async(user_id=user_id, session_id=session.id, new_message=new_message)
    tool_call: dict | None = None
    text = ""
    try:
        async for event in gen:
            calls = event.get_function_calls()
            if calls:
                tool_call = {"name": calls[0].name, "arguments": dict(calls[0].args or {})}
                break
            if event.content and event.content.parts:
                t = "".join(p.text or "" for p in event.content.parts if p.text)
                if t:
                    text = t
                    break
    finally:
        # Abandoning the generator early (rather than draining it to
        # completion) is what keeps ADK from ever reaching tool execution
        # or a second model turn — but it also leaves the generator
        # suspended mid-span under ADK's OpenTelemetry instrumentation.
        # Closing it explicitly, rather than letting GC do it later (on a
        # different thread/task context than the one the span was opened
        # on), avoids a noisy-but-harmless context-detach traceback on
        # every single call — confirmed via a standalone probe script.
        try:
            await gen.aclose()
        except Exception:
            pass
    # RouterLlm.generate_content_async's own client.chat() call already
    # appended {node, tier, provider, ...} to client.log — the ADK event
    # stream reconstructed above carries the text/tool_call but drops
    # which provider actually answered, so recover it from there rather
    # than defaulting to LLMResult's own provider="" (which every other
    # track never sees, since they use client.chat()'s return value
    # directly instead of rebuilding one from ADK events).
    provider = client.log[-1]["provider"] if client.log else ""
    return LLMResult(tool_call=tool_call, text=text, provider=provider)


def handle_message(user: dict, session_id: str, user_msg: str,
                   framework_label: str = "adk",
                   preferred_provider: str | None = None,
                   target: str | None = None) -> AgentReply:
    """Same steps as native_agent.handle_message (1-3b, input guard,
    resolve/confirm/execute/format identical by import) — only step 4-6
    (tier selection + perceive_intent) is genuinely ADK-specific."""
    state = sessions.load(user["sub"], session_id)
    state["activeFramework"] = framework_label
    tier_log: list[dict] = []
    client = None      # bound below at step 5 — None here so finish() (called
                       # from paths BEFORE step 5 too, e.g. an input_guard
                       # refusal) can safely check it without UnboundLocalError.
    rag_result = None  # set below (gmaofn/rag_optimizer.py) if retrieval was
                       # actually attempted this turn — same reason as client.
    hybrid_selection = None  # set below (llm/hybrid_router.py) only when
                       # Hybrid mode is actually selected — same reason as client.
    domain_selection = None  # set below (llm/domain_router.py) only when
                       # Domain mode is actually selected — same reason as client.

    def finish(reply: str, *, pending: dict | None = None,
              clarify: dict | None = None, out_of_scope: bool = False) -> AgentReply:
        # Computed first — a store's put() flushes synchronously, so
        # attaching security_findings to `pending` AFTER sessions.save()
        # below would silently never persist (see native_agent.py's
        # identical ordering note, §6.5 equivalence).
        security_flag, security_findings = assess_security(user["sub"], user_msg, reply)
        if pending is not None:
            pending["security_findings"] = security_findings
        prior_messages = list(state["messages"])
        state["messages"].append({"role": "user", "content": user_msg})
        state["messages"].append({"role": "assistant", "content": reply})
        state["pendingConfirmation"] = pending
        state["pendingClarification"] = clarify
        state["modelTierLog"] = state.get("modelTierLog", []) + tier_log
        sessions.save(user["sub"], session_id, state)
        # Nightly-fine-tuning data collection — same gate/heuristic as
        # native_agent.py's identical call (§6.5 equivalence).
        token_stats = None
        if pending is None and clarify is None and tier_log:
            retried = any(entry.get("node") == "perceive_intent(retry)" for entry in tier_log)
            finetune_log.log_ollama_turn(
                messages=prior_messages + [{"role": "user", "content": user_msg}],
                reply=reply, provider=tier_log[-1]["provider"],
                retried=retried, out_of_scope=out_of_scope)
            # Hybrid mode escalated away from the local SLM this turn — same
            # "log the hosted answer as SLM training data" as
            # native_agent.py's identical block (§6.5 equivalence).
            if (client is not None and client.preferred_escalated
                   and hybrid_selection is not None and hybrid_selection.arm == "local"):
                finetune_log.log_hybrid_escalation_turn(
                    messages=prior_messages + [{"role": "user", "content": user_msg}],
                    reply=reply, hosted_provider=tier_log[-1]["provider"],
                    retried=retried, out_of_scope=out_of_scope)
            # Token optimization layer (llm/token_optimizer.py) — same
            # finalize-with-the-real-outcome timing as native_agent.py's
            # identical block (§6.5 equivalence).
            if client is not None:
                success = not retried and not out_of_scope
                for selection in client.tool_selections:
                    token_optimizer.record_outcome(selection, success=success)
                if client.tool_selections:
                    sel = client.tool_selections[-1]
                    saved_pct = (round((sel.tokens_before - sel.tokens_after)
                                       / sel.tokens_before * 100, 1) if sel.tokens_before else 0.0)
                    token_stats = {"tokens_before": sel.tokens_before,
                                  "tokens_after": sel.tokens_after, "tokens_saved_pct": saved_pct}
            # RAG optimizer (gmaofn/rag_optimizer.py) — same deferred-
            # finalize timing as the token optimizer above.
            if rag_result is not None and rag_result.selection is not None:
                rag_optimizer.record_outcome(rag_result.selection, success=success,
                                             n_hits=rag_result.n_hits)
            # Hybrid mode (llm/hybrid_router.py) — same deferred-finalize
            # timing as the token/RAG optimizers above.
            if hybrid_selection is not None:
                hybrid_router.record_outcome(
                    hybrid_selection, success=success,
                    latency_ms=sum(e.get("ms", 0) or 0 for e in tier_log))
            # Domain mode (llm/domain_router.py) — same deferred-finalize
            # timing as the token/RAG optimizers above.
            if domain_selection is not None:
                domain_router.record_outcome(domain_selection, success=success)
        return AgentReply(reply=reply, pending_confirmation=pending is not None,
                          confirm_prompt=pending and pending.get("prompt"), tier_log=tier_log,
                          provider=tier_log[-1]["provider"] if tier_log else None,
                          token_stats=token_stats,
                          sources_used=rag_result.sources_used if rag_result else [],
                          out_of_scope=out_of_scope,
                          usage=tier_log[-1].get("usage") if tier_log else None,
                          context_limit_switch=any(e.get("truncated") for e in tier_log),
                          security_flag=security_flag, security_findings=security_findings)

    if state.get("pendingConfirmation"):
        pc = state["pendingConfirmation"]
        if YES.match(user_msg):
            record_security_feedback(pc.get("security_findings") or [], approved=True)
            try:
                result = tools.execute(user, pc["tool"], pc["args"])
            except Exception as e:
                return finish(f"Sorry — that failed: {e}")
            return finish(format_result(result))
        if NO.match(user_msg):
            record_security_feedback(pc.get("security_findings") or [], approved=False)
            return finish("Okay — no changes made.")
        return finish(pc["prompt"] + " (please answer yes or no)", pending=pc)

    if state.get("pendingClarification") and not looks_like_a_fresh_command(user_msg):
        reply, pending, clarify = resume_clarification(
            user, state["pendingClarification"], user_msg)
        return finish(reply, pending=pending, clarify=clarify)

    ok, refusal = input_guard(user_msg)
    if not ok:
        return finish(refusal)

    blocked = security_precheck(user_msg)
    if blocked:
        return finish(blocked)

    tier = classify_task_complexity(user_msg)
    # Fast conversational path — same native_agent.py addition (§6.5
    # equivalence): see is_pure_conversational's own docstring.
    is_conversational = is_pure_conversational(user_msg)
    _caller_sent_nothing = preferred_provider is None
    # Hybrid mode (llm/hybrid_router.py) — same interception as
    # native_agent.py's identical block (§6.5 equivalence).
    if preferred_provider == "hybrid":
        hybrid_selection = hybrid_router.choose(user_msg, tier)
        preferred_provider = hybrid_router.preferred_provider_for(hybrid_selection)
    # Domain mode (llm/domain_router.py) — same interception as
    # native_agent.py's identical block (§6.5 equivalence).
    if preferred_provider == "domain":
        domain_selection = domain_router.choose(user_msg, tier)
        preferred_provider = domain_router.preferred_provider_for(domain_selection)
    # True "auto" — same native_agent.py addition (§6.5 equivalence).
    # Gated by is_confidently_relevant() — see native_agent.py's identical
    # block for why choose()'s raw winner alone isn't a safe signal here.
    elif _caller_sent_nothing and not is_conversational:
        domain_selection = domain_router.choose(user_msg, tier)
        if not domain_router.is_confidently_relevant(domain_selection):
            domain_selection = None
        else:
            preferred_provider = domain_router.preferred_provider_for(domain_selection)
    preferred_timeout = (LOCAL_TIMEOUT_SECONDS
                         if (hybrid_selection is not None and hybrid_selection.arm == "local")
                         or (domain_selection is not None and preferred_provider is not None)
                         else None)
    # Real bug found live (native_agent.py's identical fix, §6.5
    # equivalence): never let this tier's chain fall through to the
    # "rule" floor's generic PM-tool capability menu for a Trading
    # Agent conversation.
    client = get_llm_client(tier, preferred=preferred_provider, preferred_timeout=preferred_timeout,
                            exclude_rule_based=(target == "trading"))
    history = state["messages"][-6:]

    # RAG optimizer (gmaofn/rag_optimizer.py) — same best-effort,
    # never-gates-anything convention as native_agent.py's identical
    # block (§6.5 equivalence). _perceive's extra_instruction is the ADK
    # equivalent of the extra system message the other two tracks prepend.
    rag_extra = None
    if not is_conversational and is_rag_relevant(user_msg):
        try:
            rag_result = retrieve_with_optimizer(user_msg, store_names=rag_registry.list_names())
            if rag_result.context:
                rag_extra = f"Potentially relevant context, if useful:\n{rag_result.context}"
        except Exception:
            rag_result = None

    # user_openai_tool_specs/user_tool_schemas (user_mcp_client.py) add
    # THIS user's own OAuth-connected tools on top of the shared _TOOLS
    # set — same per-request merge every other track does (§6.5
    # equivalence); ADK needs the extra _SchemaTool wrapping since it
    # builds real ADK BaseTool objects rather than a plain dict list.
    extra_tools = [_SchemaTool(s["name"], {"description": s["description"],
                                          "required": s["parameters"].get("required", []),
                                          "properties": s["parameters"].get("properties", {})})
                  for s in user_mcp_client.user_mcp_tool_specs(user["sub"])]
    user_tool_schemas = {**tools.TOOL_SCHEMAS, **tools.user_tool_schemas(user)}
    # Same real bug fix as native_agent.py/langgraph_agent.py's identical
    # addition — see TRADING_TARGET_SYSTEM's own docstring. ADK's
    # _perceive() takes one combined extra-instruction string rather than
    # a separate system message list, so the two hints are concatenated.
    extra_instruction = "\n\n".join(
        [m for m in (TRADING_TARGET_SYSTEM["content"] if target == "trading" else None, rag_extra) if m])
    res = _run_async(_perceive(client, user_msg, history, "perceive_intent",
                               extra_instruction=extra_instruction or None, summary=state.get("summary"),
                               extra_tools=extra_tools, tools_enabled=not is_conversational))
    tier_log.extend(client.log)
    ok, why = output_guard(res.tool_call, user_tool_schemas)
    if not ok:
        retry_instruction = f"Previous tool call invalid: {why}. Retry."
        if rag_extra:
            retry_instruction = f"{rag_extra}\n\n{retry_instruction}"
        res = _run_async(_perceive(client, user_msg, history, "perceive_intent(retry)",
                                   extra_instruction=retry_instruction,
                                   summary=state.get("summary"), extra_tools=extra_tools,
                                   tools_enabled=not is_conversational))
        tier_log.extend(client.log[len(tier_log):] or client.log[-1:])
        ok, why = output_guard(res.tool_call, user_tool_schemas)
        if not ok:
            return finish(f"I couldn't turn that into a valid action ({why}). Could you rephrase?")

    if not res.tool_call:
        return finish(res.text or "How can I help with your projects?",
                      out_of_scope=bool(res.text) and res.provider != "rule-based")

    name, args = res.tool_call["name"], res.tool_call["arguments"]
    resolved, clarify = resolve_arguments(user, name, args)
    if clarify:
        if clarify["field"] is None:
            # Terminal decline, not a follow-up question — see
            # native_agent.py's identical check (§6.5 equivalence).
            return finish(clarify["question"])
        return finish(clarify["question"],
                      clarify={"tool": name, "args": args, "field": clarify["field"],
                              "candidates": clarify.get("candidates")})

    if requires_confirmation(user, name):
        prompt = confirm_prompt_for(user, name, args)
        return finish(prompt, pending={"tool": name, "args": resolved, "prompt": prompt})

    try:
        result = tools.execute(user, name, resolved)
    except Exception as e:
        return finish(f"Sorry — that failed: {e}")

    return finish(format_result(result))
