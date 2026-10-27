"""Track C (§6.4-style bake-off entry, alongside langgraph/adk) — the real
Anthropic Claude API as its own agent framework option.

OPTIONAL: imported only when CHAT_AGENT_FRAMEWORK=claude (or
?framework=claude). Works with zero extra setup — CLAUDE_API_KEY unset just
means this track's own provider (llm/providers.py's AnthropicNative) is
unavailable, and get_llm_client's normal tier chain (Luna → Groq → ... →
rule-based) answers instead, the same graceful degradation every other rung
in this stack already has. Configure CLAUDE_API_KEY (and optionally
CLAUDE_MODEL, default "claude-sonnet-5") to actually have Claude answer.

This track's one genuine difference from native_agent.py (step 4-6: tier
selection + perceive_intent) is that it pins "claude" as TierClient's
preferred rung — tried first, falling back through the tier's normal chain
exactly like every other preferred-provider request already does (see
llm/router.py's TierClient.chat). Steps 7-15 (resolve_entities,
destructive-confirm, execute, format) are NOT reimplemented — imported
verbatim from native_agent.py, §6.5's equivalence rule enforced by imports,
same as langgraph_agent.py and adk_agent.py.

The MCP-sourced tools (agent/mcp_client.py) are not special-cased here —
they're already merged into agent/tools.py's TOOL_SCHEMAS, which
tools.openai_tool_specs() (passed to every client.chat() call below, same
as every other track) already includes. Claude, Luna, Groq, or the
rule-based floor all see the exact same tool set either way. Per-user
OAuth-connected tools (user_mcp_client.py) are added per-request on top
of that, same as every other track (§6.5 equivalence) — see
tools.user_openai_tool_specs()/user_tool_schemas().

This track previously gated confirmation on the raw `name in
tools.DESTRUCTIVE` check rather than calling requires_confirmation() like
the other three tracks — meaning it never asked before running an MCP
tool (shared or per-user), silently out of step with §6.5's equivalence
rule. Fixed to call requires_confirmation() like every other track,
discovered while wiring in per-user OAuth tools (a write-capable
per-user MCP tool reaching this track unconfirmed would have been a real
gap, not a hypothetical one).
"""
from __future__ import annotations

from ..llm.guardrails import input_guard, output_guard
from ..llm.router import get_llm_client
from . import sessions, tools
from .native_agent import (NO, PERCEIVE_INTENT_SYSTEM, TRADING_TARGET_SYSTEM, YES,
                           assess_security, classify_task_complexity, confirm_prompt_for,
                           record_security_feedback, requires_confirmation, resolve_arguments,
                           resume_clarification, security_precheck)
from .state import AgentReply, format_result


def handle_message(user: dict, session_id: str, user_msg: str,
                   framework_label: str = "claude",
                   preferred_provider: str | None = None,
                   target: str | None = None) -> AgentReply:
    """Same steps as native_agent.handle_message (1-3b, input guard,
    resolve/confirm/execute/format identical by import) — only step 4-6
    (tier selection + perceive_intent) pins Claude as the preferred rung."""
    state = sessions.load(user["sub"], session_id)
    state["activeFramework"] = framework_label
    tier_log: list[dict] = []

    def finish(reply: str, *, pending: dict | None = None,
              clarify: dict | None = None) -> AgentReply:
        security_flag, security_findings = assess_security(user["sub"], user_msg, reply)
        if pending is not None:
            pending["security_findings"] = security_findings
        state["messages"].append({"role": "user", "content": user_msg})
        state["messages"].append({"role": "assistant", "content": reply})
        state["pendingConfirmation"] = pending
        state["pendingClarification"] = clarify
        state["modelTierLog"] = state.get("modelTierLog", []) + tier_log
        sessions.save(user["sub"], session_id, state)
        return AgentReply(reply=reply, pending_confirmation=pending is not None,
                          confirm_prompt=pending and pending.get("prompt"), tier_log=tier_log,
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

    if state.get("pendingClarification"):
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
    # An explicit ?provider= from the caller still wins (same precedence
    # every other track gives it) — "claude" is only the DEFAULT preferred
    # rung for this track, not a hard pin, so a user/admin picking a
    # different model in the UI is still honored.
    # Real bug found live (native_agent.py's identical fix, §6.5
    # equivalence): never let this tier's chain fall through to the
    # "rule" floor's generic PM-tool capability menu for a Trading
    # Agent conversation.
    client = get_llm_client(tier, preferred=preferred_provider or "claude",
                            exclude_rule_based=(target == "trading"))

    memory_context = [{"role": "system",
                       "content": f"Summary of earlier conversation: {state['summary']}"}] \
        if state.get("summary") else []
    target_hint = [TRADING_TARGET_SYSTEM] if target == "trading" else []
    messages = [PERCEIVE_INTENT_SYSTEM] + target_hint + memory_context + state["messages"][-6:] + \
        [{"role": "user", "content": user_msg}]
    user_tool_specs = tools.openai_tool_specs() + tools.user_openai_tool_specs(user)
    user_tool_schemas = {**tools.TOOL_SCHEMAS, **tools.user_tool_schemas(user)}
    res = client.chat("perceive_intent", messages, user_tool_specs)
    tier_log.extend(client.log)

    ok, why = output_guard(res.tool_call, user_tool_schemas)
    if not ok:
        res = client.chat("perceive_intent(retry)", messages +
                          [{"role": "system",
                            "content": f"Previous tool call invalid: {why}. Retry."}],
                          user_tool_specs)
        tier_log.extend(client.log[len(tier_log):] or client.log[-1:])
        ok, why = output_guard(res.tool_call, user_tool_schemas)
        if not ok:
            return finish(f"I couldn't turn that into a valid action ({why}). Could you rephrase?")

    if not res.tool_call:
        return finish(res.text or "How can I help with your projects?")

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
