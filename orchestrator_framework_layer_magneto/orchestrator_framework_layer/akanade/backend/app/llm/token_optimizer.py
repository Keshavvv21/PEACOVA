"""Token optimization layer — the local-model-specific lever gmaofn/README.md's
"Optimizing for Ollama specifically" section previously listed as
"proposed, not implemented": the full tool-call JSON schema is otherwise
sent identically to every provider, but a CPU-only local model is
bottlenecked by PROMPT-token count, and the tool schema is a real,
non-trivial chunk of every request's prompt. This module trims that
schema for the local (Ollama) rung specifically — every other provider's
call is completely unaffected, see optimize_tools_for_provider() below.

RL algorithm: the same LinearBanditCore (Thompson Sampling over a linear
reward model) already used by llm/optimization_router.py for generation-
parameter selection, applied here to a different per-call decision — how
many tool schemas to keep. Arms are reduction levels ("full", "top12",
"top6", "top3"); context is [how many optional tools exist, how long the
user's message is]; reward combines tokens actually saved with whether
the call still succeeded, so a level that saves tokens but starts
breaking tool calls gets learned against, not just rewarded for speed.

Guardrail hierarchy (the non-negotiable floor — this is the part that
makes trimming safe rather than just fast): every tool
native_agent.requires_confirmation() already treats as needing explicit
permission — the fixed DESTRUCTIVE whitelist (delete_task,
archive_project, complete_sprint, delete_user, and any destructive JSON
plugin) plus every MCP-sourced tool — is NEVER trimmed, at ANY reduction
level. The RL policy only ever narrows down the REMAINING, non-critical
tools, ranked by relevance to the current message so the ones most
likely to actually get called survive the cut. A user's ability to even
ask for a destructive or external action must never quietly disappear
just because the local model is slow — that would be trading safety for
speed, which is not the trade this layer is allowed to make.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

from ..framework.bandit import LinearBanditCore

_FEATURE_DIM = 2  # [optional_tool_count/40 clamped, message_word_count/60 clamped]

# Ordered loosest -> tightest. "full" (keep_n=None) is always arm 0, so a
# context where trimming genuinely isn't worth it can still legitimately
# win the bandit rather than every call being forced through a cut.
_REDUCTION_LEVELS: list[tuple[str, int | None]] = [
    ("full", None),
    ("top12", 12),
    ("top6", 6),
    ("top3", 3),
]
_ARM_IDS = [level for level, _ in _REDUCTION_LEVELS]
_KEEP_N_BY_ARM = dict(_REDUCTION_LEVELS)


def _required_tool_names() -> frozenset[str]:
    """The guardrail hierarchy's floor — see module docstring. Imported
    lazily: this module sits in llm/ alongside providers.py, which
    tools.py already imports indirectly via router.py; importing
    agent.tools eagerly here would risk a circular import at module load
    time, so it's deferred to call time instead, same pattern
    native_agent.py itself uses for tools/mcp_client."""
    from ..agent import mcp_client
    from ..agent import tools as agent_tools
    return agent_tools.DESTRUCTIVE | mcp_client.mcp_tool_names()


def _relevance(message_words: set[str], tool_name: str, description: str) -> float:
    """Cheap, LLM-free keyword-overlap heuristic — no extra model round
    trip, which would defeat the entire point of trying to go FASTER.
    A learned/embedding-based ranker is a reasonable future upgrade;
    this is deliberately simple, fast, and dependency-free."""
    haystack_words = set(re.findall(r"[a-z0-9]+", f"{tool_name} {description}".lower()))
    if not message_words or not haystack_words:
        return 0.0
    return len(message_words & haystack_words) / len(haystack_words)


def _estimate_tokens(schemas: dict[str, dict]) -> int:
    """No real tokenizer at this layer (it has to stay provider-agnostic
    — the same reduced list is handed to whichever LangChain client is
    calling it) — a chars/4 estimate, the common rough ratio for English
    text. This is an ESTIMATE, not exact tokenization; it only needs to
    be directionally consistent before-vs-after for the reward signal,
    not exact — see tests/test_token_optimizer.py."""
    return len(json.dumps(schemas)) // 4


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


@dataclass
class Selection:
    """What select() picked, carried forward so record_outcome() can
    update the bandit once this call's real success/failure is known —
    mirrors optimization_router's OptimizationArm-then-update() shape."""

    arm_id: str
    context: list[float]
    tokens_before: int
    tokens_after: int


class TokenOptimizer:
    """One LinearBanditCore, arms = reduction levels — the same shape as
    llm/optimization_router.py's OptimizationRouter, applied to a
    different per-call decision (how many tool schemas to send, not
    which generation parameters)."""

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.4):
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._lock = threading.Lock()
        self._tokens_before_total = 0
        self._tokens_after_total = 0
        self._n_calls = 0

    def select(self, messages: list[dict], all_tool_schemas: dict[str, dict]) -> tuple[dict, Selection]:
        required = _required_tool_names() & set(all_tool_schemas)
        optional = {k: v for k, v in all_tool_schemas.items() if k not in required}

        message_words = set(re.findall(r"[a-z0-9]+", _last_user_text(messages).lower()))
        ranked_optional = sorted(
            optional.items(),
            key=lambda kv: -_relevance(message_words, kv[0], kv[1].get("description", "")),
        )

        context = [min(len(optional) / 40.0, 1.0), min(len(message_words) / 60.0, 1.0)]
        arm_id, _ = self._core.select(_ARM_IDS, context)
        keep_n = _KEEP_N_BY_ARM[arm_id]
        kept_optional = dict(ranked_optional) if keep_n is None else dict(ranked_optional[:keep_n])

        reduced = {**{k: all_tool_schemas[k] for k in required}, **kept_optional}
        tokens_before, tokens_after = _estimate_tokens(all_tool_schemas), _estimate_tokens(reduced)

        with self._lock:
            self._tokens_before_total += tokens_before
            self._tokens_after_total += tokens_after
            self._n_calls += 1
        return reduced, Selection(arm_id, context, tokens_before, tokens_after)

    def record_outcome(self, selection: Selection, *, success: bool) -> None:
        """Reward = this call's own success (dominant term, matching
        optimization_router's convention — a faster call that broke the
        tool call is a loss, not a win) plus a bonus scaled by the
        fraction of tokens actually saved, so a level that saves nothing
        real never out-scores one that saves more for the same success
        rate."""
        saved_frac = 0.0
        if selection.tokens_before:
            saved_frac = max(0.0, (selection.tokens_before - selection.tokens_after)
                             / selection.tokens_before)
        reward = (1.0 if success else -0.5) + 0.3 * saved_frac
        self._core.update(selection.arm_id, selection.context, max(-1.0, min(1.0, reward)))

    def average_tokens_stats(self) -> dict:
        """The observable half of "average token optimization" — running
        average estimated prompt tokens the tool schema alone cost,
        before vs after this layer, over every call it's actually run
        for (local-model calls only; see optimize_tools_for_provider)."""
        with self._lock:
            n = self._n_calls
            if n == 0:
                return {"n_calls": 0, "avg_tokens_before": 0.0, "avg_tokens_after": 0.0,
                        "avg_tokens_saved_pct": 0.0}
            before, after = self._tokens_before_total / n, self._tokens_after_total / n
            saved_pct = round((before - after) / before * 100, 1) if before else 0.0
            return {"n_calls": n, "avg_tokens_before": round(before, 1),
                    "avg_tokens_after": round(after, 1), "avg_tokens_saved_pct": saved_pct}


# Process-wide singleton, same lifecycle as optimization_router.
token_optimizer = TokenOptimizer()


def optimize_tools_for_provider(provider_name: str, messages: list[dict],
                                tools: list[dict] | None) -> tuple[list[dict] | None, Selection | None]:
    """The actual hook llm/router.py's TierClient.chat calls, right
    before p.chat(messages, tools). Only a provider whose name starts
    with "ollama" (the convention finetune_log.py and router.py's own
    provider-registration already use) gets its tool list reduced —
    every other provider gets the exact same `tools` list as always,
    byte for byte. Returns (possibly-reduced tools, a Selection to pass
    to record_outcome() once the call's outcome is known — or None if
    no reduction happened this call, e.g. a non-Ollama provider or no
    tools were passed at all)."""
    if not tools or not provider_name.startswith("ollama"):
        return tools, None

    schemas = {t["name"]: {"description": t.get("description", ""),
                           "required": t.get("parameters", {}).get("required", []),
                           "properties": t.get("parameters", {}).get("properties", {})}
              for t in tools}
    reduced_schemas, selection = token_optimizer.select(messages, schemas)
    reduced_tools = [{"name": n, "description": s["description"],
                      "parameters": {"type": "object", "properties": s["properties"],
                                     "required": s["required"]}}
                     for n, s in reduced_schemas.items()]
    return reduced_tools, selection
