"""§12 Guardrails — the two lightweight checks the design adopts for v2.0.

Input guard (before perceive_intent): a targeted injection-pattern scan
over the raw user message and any free text about to enter a prompt.
Output guard (after perceive_intent): schema-validate the extracted
{intent, arguments} against the named tool's expected shape before
resolve_entities may run; on failure the agent re-asks once on the same
tier (§12.2) before falling through the chain.

Deliberately NOT NeMo-style dialog rails — deferred per §12.2.
"""
from __future__ import annotations

import re
from typing import Callable

# Pre-compiled once at import time, not per-call — input_guard runs on every
# single turn (§12.3), so this is on the hot path.
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|previous\s+|above\s+|prior\s+)*(instructions|prompts)",
    r"disregard .{0,30}(instructions|system prompt)",
    r"you are now .{0,40}(unrestricted|dan|jailbroken)",
    r"reveal .{0,30}(system prompt|api key|secret|jwt)",
    r"</?(system|assistant)>",
]
_COMPILED_INJECTION_PATTERNS = [re.compile(p) for p in INJECTION_PATTERNS]

REFUSAL_MESSAGE = ("That request looks like an attempt to change my "
                   "instructions, which I can't act on. I can help "
                   "with Aksha PM tasks — try asking about projects, "
                   "tasks, or sprints.")


def looks_like_injection(text: str) -> bool:
    """The actual pattern check, split out of input_guard() so
    llm/policy_gate.py can apply the SAME rule to every TierClient.chat()
    call — not just the native_agent.py chat turn that calls input_guard()
    directly — without a second, divergent pattern list to keep in sync."""
    low = text.lower()
    return any(pat.search(low) for pat in _COMPILED_INJECTION_PATTERNS)


def input_guard(text: str) -> tuple[bool, str]:
    """Returns (ok, reason). Blocking is a refusal message, not an error."""
    if looks_like_injection(text):
        return False, REFUSAL_MESSAGE
    return True, ""


# JSON-schema "type" → validator, keyed the same way tool_schemas' own
# properties[x]["type"] values are spelled. Add a new schema type by adding
# one entry here — output_guard's loop below never needs to change (Open/
# Closed); today's tool_schemas only use string/boolean, so that's all
# that's registered, but a "number" or "array" tool argument in the future
# is a one-line addition, not a new branch in the validation logic.
_TYPE_VALIDATORS: dict[str, Callable[[object], bool]] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
}


def output_guard(tool_call: dict | None, tool_schemas: dict) -> tuple[bool, str]:
    """Schema check: named tool exists (fixed whitelist, §11) and required
    arguments are present with sane types. A few lines, as promised."""
    if not tool_call:
        return True, ""  # plain-text replies pass through
    name = tool_call.get("name")
    if name not in tool_schemas:
        return False, f"unknown tool {name!r}"
    schema = tool_schemas[name]
    args = tool_call.get("arguments") or {}
    for req in schema.get("required", []):
        if req not in args or args[req] in (None, ""):
            return False, f"missing required argument {req!r} for {name}"
    for k, v in args.items():
        if v is None:
            continue  # absent optional argument, not a type violation
        prop = schema.get("properties", {}).get(k, {})
        expect = prop.get("type")
        validate = _TYPE_VALIDATORS.get(expect)
        if validate is not None and not validate(v):
            return False, f"argument {k!r} must be a {expect}"
        # A model isn't actually bound by a JSON-schema "enum" the way a
        # required field or a type is — nothing stops it from putting the
        # user's raw wording into a field like graph_query's "question"
        # even with an enum listed (confirmed: this is exactly how "what's
        # overdue" produced question="what's overdue" instead of
        # question="overdue"). Checking membership here, on the SAME path
        # a bad type already retries through, turns that into one
        # self-correcting retry instead of a late, confusing runtime
        # "unknown X" error from inside execute().
        enum = prop.get("enum")
        if enum is not None and v not in enum:
            return False, f"argument {k!r} must be one of {enum}, not {v!r}"
    return True, ""
