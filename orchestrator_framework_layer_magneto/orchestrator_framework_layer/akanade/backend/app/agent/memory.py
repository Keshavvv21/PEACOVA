"""Memory optimization for long-running chats — a rolling summary that
replaces "discard" with "compress".

sessions.save() trims state["messages"] to the last SESSION_WINDOW_TURNS
turns before persisting (see sessions.py) — until now, anything older
than that was simply gone. Fine for a short chat, but a session kept open
for days lost its entire earlier conversation the instant it scrolled out
of the window, and every track re-paid the same token cost for the same
handful of recent turns on every call regardless of how long the
conversation actually was.

fold_into_summary() is called from sessions.save() with whatever's about
to be trimmed off the front of the window, and folds it into one running
summary string via an LLM call — through get_llm_client(), the SAME
router every other call in this codebase goes through (§16's lint rule:
that's the only sanctioned entry to any provider), not a hardcoded model.
That's what makes this "connect to LLMs" the same way the rest of the
stack does: whichever tier/provider is actually available answers it,
with the same tier-chain fallback and rule-based floor as any other call.

Tier is "bulk": summarizing old turns is background housekeeping, not
something the user is waiting on — the same reasoning
classify_task_complexity uses to route report/digest asks to delay_ok
rather than realtime.

Each track (native/langgraph/adk) reads state["summary"] back out and
prepends it as one system message alongside its normal rolling window —
see native_agent.handle_message, langgraph_agent.perceive_intent, and
adk_agent._perceive's `summary` param.
"""
from __future__ import annotations

from ..llm.router import get_llm_client

_SUMMARY_SYSTEM = {
    "role": "system",
    "content": (
        "You maintain a running summary of an ongoing chat between a user "
        "and a project-management assistant. Given the PREVIOUS SUMMARY "
        "(may be empty) and a batch of OLDER MESSAGES that are about to "
        "scroll out of the visible window, write an UPDATED SUMMARY: a "
        "short paragraph (about 100 words) capturing what the user cares "
        "about, decisions made, and any open threads. Plain prose only — "
        "no headers, no bullet list, nothing but the summary text itself."
    ),
}

# Max words/chars aren't enforced numerically here — a natural-language
# instruction is enough for this use (an over-length reply just gets
# compressed further on the NEXT fold, since this always summarizes
# against the previous summary rather than the raw transcript).


def fold_into_summary(previous_summary: str, dropped_messages: list[dict]) -> str:
    """Compress `dropped_messages` — the turns sessions.save() is about to
    discard from the rolling window — into an updated running summary,
    merged with whatever was already summarized.

    Never raises, and never regresses to something worse than what was
    already there: TierClient.chat() itself already walks its whole
    fallback chain and never throws (router.py's own contract), and if
    every hosted provider is down it bottoms out at RuleBased — which
    isn't actually trying to summarize (it pattern-matches for task
    intents), so its output is deliberately rejected here rather than
    saved as a bogus "summary". Either way this function returns
    `previous_summary` unchanged rather than losing what was already
    captured, matching every other LLM-backed feature's fail-soft
    behavior in this codebase.
    """
    if not dropped_messages:
        return previous_summary

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in dropped_messages)
    prompt = (
        f"PREVIOUS SUMMARY: {previous_summary or '(none yet)'}\n\n"
        f"OLDER MESSAGES:\n{transcript}\n\n"
        "UPDATED SUMMARY:"
    )
    client = get_llm_client("bulk")
    result = client.chat("memory_fold_summary",
                         [_SUMMARY_SYSTEM, {"role": "user", "content": prompt}])
    if result.provider == "rule-based":
        return previous_summary
    text = (result.text or "").strip()
    return text or previous_summary
