"""CodingAgent — §7's example second domain (code_generation, debugging,
code_review, testing, refactoring), proving Algo aKANADE generalizes past the
PM domain it started in without touching the framework core: this file
is the ENTIRE integration surface for a brand-new domain — it's just
another Agent, registered the same way PMAgent/OpenClawAgent are (see
app/gmaofn_bootstrap.py).

Originally this agent only ever generated text and ran `ast.parse` on
any fenced Python block — a real syntax check, but never actually
RUNNING anything, and no file access at all. It now has three real
LLM-callable tools, the same flat {name, description, parameters} shape
agent/tools.py's PM tools already use, passed straight into
TierClient.chat()'s `tools=` argument:

  run_code(code)          — actually EXECUTES the snippet in a sandboxed
                            subprocess (framework/sandbox.py) and hands
                            the model real stdout/stderr/exit-code back,
                            so it can catch a runtime error ast.parse
                            can't (a syntax-valid script that raises at
                            line 1 used to be reported as fine).
  read_file(path)          — reads a text file, scoped to
                            settings().CODING_WORKSPACE_DIR via the SAME
                            resolve_within() path-traversal guard
                            code_security_agent.py already uses for its
                            own file scanning (reused, not reimplemented).
  write_file(path, content) — DESTRUCTIVE. Never executes on the first
                            call: it stashes the proposed write and
                            returns AgentResult.pending_confirmation
                            instead, the same field every other agent's
                            result carries but this one is the first to
                            actually populate. Only a follow-up Task for
                            the SAME session_id with
                            task.metadata["confirm_write"]=True performs
                            the write — mirrors agent/native_agent.py's
                            own "persist {tool,args}, ask, act on yes"
                            shape for its own destructive PM tools, just
                            keyed by session_id since this agent has no
                            native_agent-style session store of its own.

`validate()` still does the same real syntax check as before (extract
every fenced ```python``` block, `ast.parse` it, retry once on a
failure) for the plain-generation path — unchanged, since a tool-call
turn and a plain-generation turn are different shapes and shouldn't be
conflated.

CLIENT TOOLS MODE (`task.metadata["client_tools"] = True`, set by
`POST /api/v1/orchestration/execute`'s `client_tools` field): this
server has real local-filesystem access ONLY to its own disposable
sandbox (`CODING_WORKSPACE_DIR`) — never to a user's actual project.
Getting Claude-Code-style autonomy (grep the real repo, open/edit real
files, run a real command, all on the model's own initiative) means
the ACTUAL read/write/exec has to happen somewhere that has that
access: the caller (a VS Code extension, today). In this mode, EVERY
tool call — including write_file — short-circuits the loop the same
way write_file's confirmation already did: instead of executing
anything, `_handle_tool_call` returns a `client_tool_call` sentinel,
`_run_agentic_loop` stashes the in-flight conversation state in
`_PENDING_TOOL_CALLS` keyed by a fresh `call_id`, and the API response
carries `{call_id, tool, args}` back to the caller instead of a
prompt. The caller executes it for real (against the user's own
machine, with whatever confirmation UX it wants for a destructive one)
and calls `POST /api/v1/orchestration/tool_result` with the outcome,
which resumes this SAME loop via `resume_tool_call()` exactly where it
left off — same `_MAX_TOOL_STEPS` budget, same "propose the next tool
call or give a final answer" prompt shape as the fully-server-side
path. `list_files`/`grep` (CLIENT_ONLY_TOOL_SCHEMAS) only exist in this
mode, since only the caller has a real workspace to list or search.
The DEFAULT mode (`client_tools` absent/False) is completely
unaffected — `_PENDING_WRITES` and the plain-text yes/no confirmation
flow below are only ever reached when `client_tools` is off.
"""
from __future__ import annotations

import ast
import json
import re
import time
import uuid
from pathlib import Path

from ...config import settings
from ...llm.hybrid_router import LOCAL_TIMEOUT_SECONDS, HybridSelection, hybrid_router
from ...llm.intent_reasoner import classify_intent, guidance_for
from ...llm import coding_pipeline
from ...llm.providers import LLMResult
from ...llm.router import get_llm_client
from ..agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from ..document_pipeline import UnsafePathError, resolve_within
from .. import coding_usage
from ..intent_cache import coding_cache
from ..sandbox import run_python
from ..task import Task, TaskType

# Intents inherently grounded in a specific file/error/codebase state —
# never safe to cache-serve or cache-write regardless of whether a tool
# actually fired this turn (framework/intent_cache.py's gating is a
# CALLER responsibility; this is that gate for the Coding Agent side).
_CACHEABLE_INTENTS = frozenset({"generate", "general_coding"})

SYSTEM_PROMPT = {
    "role": "system",
    # Directive, not descriptive — real, observed failure mode: models
    # (including hosted ones, not just the small local coder) will
    # happily WRITE OUT what a tool call would look like, or narrate
    # "I'm ready to create the file...", instead of actually invoking
    # the tool, when the instructions only describe what the tools do
    # rather than telling the model to use them. This phrasing exists
    # specifically to stop that: every sentence about a tool ends in an
    # imperative ("call X"), and the "do NOT just describe/show" line is
    # there because a softer version of this prompt still let a real,
    # otherwise-compliant hosted model narrate instead of act.
    "content": "You are a coding assistant with real tools — use them, don't just "
              "describe what you would do. If the user asks you to run/execute/test "
              "code, you MUST call run_code — do not just print what the output "
              "would be. If the user asks you to save/write/create/modify a file, "
              "you MUST call write_file — do not just show its contents in a code "
              "block and describe writing it. If the user asks you to remove/delete "
              "a file, call delete_file (only available with a real workspace — see "
              "your tool list). If the user asks about an existing file, call "
              "read_file. Only skip the tools when the user is just asking you to "
              "write or explain a plain code snippet, not to save or run anything. "
              "A destructive call (write_file, delete_file) needs the user's "
              "confirmation first — that confirmation step happens automatically "
              "after you call it; you do not need to ask permission yourself, just "
              "call the tool. When asked to review/debug/refactor code the user "
              "pasted, do that directly instead of writing new code from scratch. "
              "When a task needs several steps (read a file, then edit it, then run "
              "it, then fix what failed), keep calling tools across as many turns "
              "as it genuinely takes — don't stop and ask what to do next just "
              "because the first step succeeded; only stop once the task is "
              "actually done or you hit a real blocker you can't resolve yourself.",
}

# Appended to SYSTEM_PROMPT's content ONLY in client_tools mode (see
# plan() below) — real, observed failure mode: asked to "edit
# fibonacci.py" with no file attached/mentioned, the model would guess
# a bare relative path (or the wrong one, in a repo with several files
# of that name in different folders) and either silently write a NEW
# file next to a real one it never found, or read_file/write_file would
# fail against a nonexistent guessed path — from the user's side this
# looks exactly like "the agent doesn't code in the workspace at all,"
# even though list_files/grep (CLIENT_ONLY_TOOL_SCHEMAS) already exist
# and already resolve against the user's REAL, currently-open folder
# with zero extra configuration (see localTools.ts's
# resolveWithinWorkspace). The tools existing was never the gap — the
# model never being told to actually use them before guessing was.
_CLIENT_TOOLS_ADDENDUM = (
    " You are working directly in the user's real, currently-open project folder — "
    "not a sandbox, not a hypothetical one. Never assume or guess a file's exact path. "
    "Before you read, edit, or delete a file whose exact path you are not already "
    "certain of (e.g. the user just named it, like \"fibonacci.py\", without giving you "
    "the full path), call list_files or grep FIRST to find where it actually is — search "
    "by filename (e.g. glob '**/fibonacci.py') if a plain top-level guess doesn't turn it "
    "up. If it doesn't exist anywhere, say so before offering to create it, rather than "
    "creating a new file the user didn't ask for. This applies every single time a file is "
    "named without its full path already being known from earlier in this conversation."
)

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# Same shape (and same intent) as agent/native_agent.py's own YES/NO —
# duplicated rather than imported so this generic example agent doesn't
# take a dependency on the PM-specific chat module. The orchestration
# API's ExecuteRequest (routers/v1/orchestration.py) has no dedicated
# "confirm" field — a caller can only ever send plain task text — so
# confirming/declining a pending write is read from the SAME plain-text
# reply a human would actually type, not a metadata flag nothing in the
# real request schema could ever populate.
_YES = re.compile(r"^\s*(y|yes|yeah|yep|confirm|do it|go ahead)\s*[.!]?\s*$", re.I)
_NO = re.compile(r"^\s*(n|no|nope|cancel|stop|don'?t)\s*[.!]?\s*$", re.I)

# OpenAI-function-style schemas, the same flat {description, required,
# properties} shape agent/tools.py's TOOL_SCHEMAS uses — TierClient.chat()
# wraps each into {"type": "function", "function": ...} itself (see
# llm/providers.py's OpenAICompatible.chat()), so this file never needs
# to know the wire format.
CODING_TOOL_SCHEMAS: dict[str, dict] = {
    "run_code": {
        "description": "Execute a Python snippet in a sandboxed subprocess and "
                       "return its real stdout, stderr, and exit code. Use this "
                       "to verify generated code actually runs, not just parses.",
        "required": ["code"],
        "properties": {"code": {"type": "string",
                                "description": "The full Python source to run."}},
    },
    "read_file": {
        "description": "Read a text file from the coding workspace.",
        "required": ["path"],
        "properties": {"path": {"type": "string",
                                "description": "Path relative to the workspace root."}},
    },
    "write_file": {
        "description": "Propose writing (creating or overwriting) a text file in "
                       "the coding workspace. This is DESTRUCTIVE — it always "
                       "requires the user's explicit yes/no confirmation before "
                       "anything is actually written.",
        "required": ["path", "content"],
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    },
}

# Advertised ONLY when this turn is running in "client tools" mode (see
# _coding_tool_specs) — this server has no workspace of its own to list
# or search in that mode; a VS Code extension (or any other client
# implementing the same protocol) does, on the user's real machine.
# Offering these unconditionally would let a model call them in the
# DEFAULT (server-sandbox) mode too, where nothing could ever answer
# them.
CLIENT_ONLY_TOOL_SCHEMAS: dict[str, dict] = {
    "list_files": {
        "description": "List files in the user's real local workspace matching a "
                       "glob pattern, executed on their machine — NOT the server "
                       "sandbox. Use this to discover what exists before reading "
                       "or editing it.",
        "required": [],
        "properties": {"glob": {"type": "string",
                                "description": "Glob pattern, e.g. '**/*.py'. Defaults to everything."}},
    },
    "grep": {
        "description": "Search file CONTENTS across the user's real local "
                       "workspace for a regex pattern, executed on their machine "
                       "— NOT the server sandbox. Returns matching lines as "
                       "'path:line: text'. Use this the way you'd grep a repo: to "
                       "find where something is defined or used before opening it.",
        "required": ["pattern"],
        "properties": {"pattern": {"type": "string", "description": "A regex pattern."},
                       "glob": {"type": "string",
                                "description": "Restrict the search to files matching this glob. "
                                              "Defaults to everything."}},
    },
    "delete_file": {
        "description": "Permanently delete a file from the user's real local "
                       "workspace, executed on their machine — NOT the server "
                       "sandbox. This is DESTRUCTIVE; the caller decides how (or "
                       "whether) to confirm it before reporting back what "
                       "happened, the same as write_file in this mode.",
        "required": ["path"],
        "properties": {"path": {"type": "string",
                                "description": "Path relative to the workspace root."}},
    },
}

# In-process only, keyed by session_id — same "resets on restart, never
# the only copy of anything a user would notice missing" convention
# framework/task_memory.py's own bandit memory already uses (see
# README_VECTOR_EMBEDDING_TABLES.md §2.3-2.5): a pending write lost to a
# restart just needs to be re-proposed, nothing silently happens instead.
# Only ever populated in the DEFAULT (non-client-tools) mode — see
# _handle_tool_call's client_tools branch, which never touches this.
_PENDING_WRITES: dict[str, dict] = {}

# "Client tools" mode's equivalent of _PENDING_WRITES above — same
# in-process-only, same-restart-just-re-ask convention, but keyed by a
# fresh call_id (not session_id) rather than the write-specific one,
# because in this mode EVERY tool call (not just write_file) pauses the
# loop and waits for the caller to execute it and report back. Holds
# everything _run_agentic_loop needs to resume exactly where it left
# off: the conversation so far, which tools already ran, and which step
# of the _MAX_TOOL_STEPS budget comes next. Deliberately safe under the
# single-process deployment this backend actually runs as today
# (bare `uvicorn app.main:app`, no multi-worker) — the same assumption
# _PENDING_WRITES already relies on.
_PENDING_TOOL_CALLS: dict[str, dict] = {}

# The agentic-loop bound (execute()) — how many NON-destructive tool
# calls (run_code/read_file) can chain within one turn before this
# agent gives up and reports what it did instead of looping forever.
# write_file always short-circuits the loop immediately via
# pending_confirmation regardless of this bound — a destructive action
# is never chained without the user seeing it. 4 is enough for a real
# "write, run, see it fail, fix, run again" cycle without letting a
# model that never settles on a final answer burn unbounded turns/cost.
_MAX_TOOL_STEPS = 4
# Client tools mode's own, much bigger budget — see module docstring's
# "work continuously until the task is completed" (a VS Code-extension-
# style caller running against the user's REAL workspace, with its own
# confirmation UX per call — see localTools.ts's auto-approve mode).
# Every tool call here (including write_file/delete_file) already
# short-circuits through the caller's own confirmation before it does
# anything real, so a much larger step budget is safe: it bounds how
# many genuinely useful actions one task can take, not how many
# destructive ones slip through unconfirmed. Still finite — a model
# that genuinely never settles on a final answer still gets cut off
# and told to keep going explicitly, never an unbounded loop.
_MAX_TOOL_STEPS_CLIENT = 25


def _coding_tool_specs(client_tools: bool = False) -> list[dict]:
    schemas = CODING_TOOL_SCHEMAS
    if client_tools:
        schemas = {**CODING_TOOL_SCHEMAS, **CLIENT_ONLY_TOOL_SCHEMAS}
    return [{"name": n, "description": s["description"],
             "parameters": {"type": "object", "properties": s["properties"],
                            "required": s["required"]}}
            for n, s in schemas.items()]


# Sanitizes ExecuteRequest.history (routers/v1/orchestration.py) before
# it ever reaches an actual model call — a caller-supplied list is
# untrusted input, not something to splice into `messages` verbatim.
# Only "user"/"assistant" roles are accepted (never "system" — a client
# should never be able to inject its own system instructions this way)
# and only string content; anything else is dropped rather than raising,
# since a malformed one entry shouldn't break an otherwise-fine turn.
# Capped at the most recent 40 entries — this is real conversation
# grounding, not an unbounded transcript dump into every single call.
_MAX_HISTORY_ENTRIES = 40


def _sanitize_history(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if (isinstance(entry, dict) and entry.get("role") in ("user", "assistant")
                and isinstance(entry.get("content"), str) and entry["content"]):
            out.append({"role": entry["role"], "content": entry["content"]})
    return out[-_MAX_HISTORY_ENTRIES:]


def _extract_python_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _CODE_BLOCK.finditer(text)]


def _first_syntax_error(blocks: list[str]) -> str | None:
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as e:
            return f"{e.msg} at line {e.lineno}"
    return None


# Matches a CLOSED fenced block (```json ... ``` or a bare ``` ... ```,
# any language tag) — a weak model printing a tool call as text picks
# whichever fence language tag it likes, or none at all.
_ANY_CLOSED_FENCE = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)
# An OPEN fence with no closing ``` at all — real, observed behavior:
# generation can stop (max_tokens, or the model just trailing off)
# before the model ever emits the closing marker, so requiring one
# would silently miss an otherwise-complete, otherwise-valid JSON body.
_OPEN_FENCE = re.compile(r"```(?:\w+)?\s*\n?(.*)$", re.DOTALL)


def _parse_text_tool_call(text: str, client_tools: bool = False) -> dict | None:
    """See execute()'s own comment for why this exists. Tries, in order:
    every CLOSED fenced block (the common case), then the tail of an
    OPEN/truncated fence with no closing marker, then the whole reply
    verbatim. json.loads(), not a brace-counting regex — a "{...}"
    pattern breaks the moment the JSON has ITS OWN nested braces (e.g.
    "arguments": {"code": "..."}), which every real tool call here
    does."""
    valid_names = CODING_TOOL_SCHEMAS.keys() | (CLIENT_ONLY_TOOL_SCHEMAS.keys() if client_tools else set())
    candidates = [m.group(1).strip() for m in _ANY_CLOSED_FENCE.finditer(text)]
    if m := _OPEN_FENCE.search(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if (isinstance(obj, dict) and obj.get("name") in valid_names
                and isinstance(obj.get("arguments"), dict)):
            return {"name": obj["name"], "arguments": obj["arguments"]}
    return None


# Real, observed failure mode qwen2.5-coder:1.5b (ollama-coder) hits
# under load that _parse_text_tool_call above can't recover from: not
# "the tool call as valid JSON, just printed as text" (what that
# function targets), but genuinely MALFORMED JSON — an unquoted key, a
# stray "<tools>" tag leaked from its own prompt formatting — that no
# amount of JSON-repair should be expected to reliably fix. Detecting
# this class specifically (never a false-positive on real prose/code,
# which essentially never contains a literal "<tools>" tag or this
# exact schema-shaped prefix) is what lets _run_agentic_loop tell
# "the local model is failing at tool-calling" apart from "the model
# gave a real plain-text answer" and escalate instead of showing the
# user raw garbage.
_LEAKED_TOOL_SCHEMA_MARKERS = ("<tools>", '{"type": "function"', "{'type': 'function'")


def _looks_like_leaked_tool_schema(text: str) -> bool:
    head = text.lstrip()[:200]
    return any(marker in head for marker in _LEAKED_TOOL_SCHEMA_MARKERS)


# A SECOND, distinct failure mode confirmed via live testing (60-prompt
# comprehensive test), the dominant one in practice — more common than
# the leaked-schema case above: qwen2.5-coder:1.5b repeatedly answers a
# "write me a function" prompt with a well-formed
# {"name": "<the-function-it-was-asked-to-write>", "arguments": {...}}
# object instead of the actual code — e.g. asked to write a palindrome
# checker, it replies `{"name": "is_palindrome", "arguments":
# {"string": {"type": "string"}}}` verbatim. This is real, valid JSON
# shaped exactly like a tool call, so _parse_text_tool_call() above
# correctly declines to treat it as one (the name isn't a real tool),
# but _looks_like_leaked_tool_schema() ALSO misses it (no "<tools>" tag,
# no {"type": "function"} prefix — a different shape entirely), so
# without this check the raw JSON blob was silently returned to the
# user as if it were the final answer. Confirmed live: this exact
# pattern accounted for 6 of 10 Coding Agent failures in one test run.
def _looks_like_hallucinated_tool_call(text: str) -> bool:
    """True when the reply (or a fenced block within it) parses as a
    JSON object shaped like a tool call ("name" + "arguments" dict),
    REGARDLESS of whether "name" is a real tool — real prose/code
    answers essentially never parse as that exact top-level shape, so
    this doesn't false-positive on a genuine answer. Reuses the same
    candidate extraction as _parse_text_tool_call (fenced block, open
    fence, or the whole reply) so detection matches what was actually
    (unsuccessfully) tried as a real tool call."""
    candidates = [m.group(1).strip() for m in _ANY_CLOSED_FENCE.finditer(text)]
    if m := _OPEN_FENCE.search(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments"), dict)):
            return True
    return False


class CodingAgent(Agent):
    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="coding-agent",
            name="Coding Agent",
            description="Generates, reviews, debugs, and refactors code. Can "
                       "actually RUN a generated snippet in a sandbox, and "
                       "read/write files in a scoped workspace (writes always "
                       "require explicit user confirmation first). Every "
                       "generated Python code block is also syntax-checked "
                       "(ast.parse) before the reply is returned, with one "
                       "automatic retry on a syntax error.",
            capabilities=frozenset({"code_generation", "debugging", "code_review",
                                    "testing", "refactoring", "code_execution",
                                    "workspace_file_io"}),
            supported_tasks=frozenset({TaskType.CODE_GENERATION, TaskType.CODE_REVIEW}),
            model="configurable (via llm/router.py's provider chain)",
            tools=("ast_syntax_check", "run_code", "read_file", "write_file",
                  "list_files (client_tools mode)", "grep (client_tools mode)"),
            rag_enabled=False,
            cost_per_call_usd=0.004,
            latency_profile=LatencyProfile(p50_ms=1200, p95_ms=4000),
            version="1.1.0",
            reliability_score=0.7,
            status=AgentStatus.ACTIVE,
        )

    def understand(self, task: Task) -> dict:
        has_pending = task.session_id in _PENDING_WRITES
        text = task.text.strip()
        return {"text": task.text,
               "confirming_write": has_pending and bool(_YES.match(text)),
               "canceling_write": has_pending and bool(_NO.match(text))}

    def plan(self, task: Task, understanding: dict) -> dict:
        if understanding["confirming_write"]:
            return {"confirm_write": True}
        if understanding["canceling_write"]:
            return {"cancel_write": True}
        content = SYSTEM_PROMPT["content"]
        if task.metadata.get("client_tools"):
            content += _CLIENT_TOOLS_ADDENDUM
        # The intent-reasoning layer (llm/intent_reasoner.py) — a cheap,
        # deterministic classification of what KIND of coding task this
        # is (debug/refactor/test/repo-analysis/generate), appended as
        # intent-specific guidance rather than run as a second model
        # call, so it costs nothing in latency. See
        # README_SLM_REASONING_LAYER.md for the full architecture.
        intent = classify_intent(understanding["text"])
        content += guidance_for(intent)
        system = {**SYSTEM_PROMPT, "content": content}
        history = _sanitize_history(task.metadata.get("history"))
        return {"messages": [system, *history, {"role": "user", "content": understanding["text"]}],
               "intent": intent}

    @staticmethod
    def _require_str(args: dict, key: str) -> str | None:
        """Same shape as llm/guardrails.py's output_guard() type check for
        PM tools — a malformed argument gets a clear "must be a string"
        result, never an unhandled crash. Real, observed failure: a small
        local model (qwen2.5-coder:1.5b) occasionally emits {"code": {...
        nested object ...}} instead of a plain string for a tool
        argument — sandbox.run_python()'s subprocess.run() has no idea
        what to do with a dict argv element and raises a bare TypeError,
        which SequentialStep.run() (framework/workflow.py) then turns
        into an opaque "every agent in the sequence failed" for the
        user. Catching it HERE gives a real, specific, recoverable
        message instead."""
        val = args.get(key, "")
        return val if isinstance(val, str) else None

    def _handle_tool_call(self, task: Task, tool_call: dict, client_tools: bool = False) -> dict:
        name, args = tool_call["name"], tool_call.get("arguments") or {}
        if client_tools:
            # Every tool — including write_file — pauses the loop here
            # instead of executing anything server-side: this server has
            # no access to the user's real files at all in this mode.
            # _run_agentic_loop is the one place that turns this sentinel
            # into an actual _PENDING_TOOL_CALLS entry and the API
            # response the client acts on; _PENDING_WRITES/the plain-
            # text yes/no confirmation flow below is deliberately never
            # reached in this mode — the CLIENT decides how (or whether)
            # to confirm a write/run with its own user before reporting
            # back what happened.
            return {"client_tool_call": True}
        if name == "run_code":
            code = self._require_str(args, "code")
            if code is None:
                return {"summary": f"Malformed run_code call — 'code' must be a string, "
                                   f"got {type(args.get('code')).__name__}. Try again with a plain "
                                   f"string of Python source.",
                        "raw": {"error": "malformed arguments"}}
            res = run_python(code)
            status = "timed out" if res.timed_out else f"exited {res.exit_code}"
            return {"summary": f"run_code {status}.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}",
                   "raw": res.to_dict()}
        if name in ("read_file", "write_file"):
            path = self._require_str(args, "path")
            if path is None:
                return {"summary": f"Malformed {name} call — 'path' must be a string, "
                                   f"got {type(args.get('path')).__name__}.",
                        "raw": {"error": "malformed arguments"}}
            try:
                resolved = resolve_within(path, settings().CODING_WORKSPACE_DIR)
            except UnsafePathError as e:
                return {"summary": f"Refused: {e}", "raw": {"error": str(e)}}
            if name == "read_file":
                if not resolved.exists():
                    return {"summary": f"{path!r} does not exist in the workspace.",
                            "raw": {"found": False}}
                text = resolved.read_text(encoding="utf-8", errors="ignore")
                return {"summary": text, "raw": {"found": True, "path": str(resolved)}}
            # write_file — never executes here; stash and ask.
            content = self._require_str(args, "content")
            if content is None:
                return {"summary": f"Malformed write_file call — 'content' must be a string, "
                                   f"got {type(args.get('content')).__name__}.",
                        "raw": {"error": "malformed arguments"}}
            _PENDING_WRITES[task.session_id] = {"path": str(resolved), "content": content}
            rel = resolved.relative_to(Path(settings().CODING_WORKSPACE_DIR).resolve())
            overwrite_note = " (this OVERWRITES an existing file)" if resolved.exists() else ""
            return {"pending_confirmation": True,
                    "confirm_prompt": f"⚠️ This will write {len(content)} characters to "
                                      f"“{rel}” in the coding workspace{overwrite_note}. "
                                      f"Allow? (yes / no)"}
        return {"summary": f"Unknown tool {name!r}."}

    @staticmethod
    def _preferred_provider(task: Task) -> str:
        """A caller-specified provider (task.metadata["preferred_provider"],
        e.g. a UI model picker) always wins outright. Otherwise this agent
        opportunistically tries "ollama-coder" (config.py's
        OLLAMA_CODER_MODEL, e.g. Qwen2.5-Coder — a model actually trained
        on code) FIRST — TierClient.chat()'s own `preferred` mechanism
        means an unpulled/unavailable coder model just falls through to
        the normal delay_ok chain right behind it, never a hard failure,
        same graceful-degradation story as every other optional rung.
        "hybrid" (the "Auto Hybrid" picker option) is resolved BEFORE this
        ever runs — see execute()'s own hybrid_router.choose() call — so
        this never actually sees that literal sentinel on the primary
        call path; it's only reachable here via validate()'s retry (a
        second, unrelated call after a syntax-error already occurred),
        where "hybrid" resolving to nothing/no-op via the normal chain is
        an acceptable, already-established degradation, not a fresh
        decision worth re-running the bandit for."""
        return task.metadata.get("preferred_provider") or "ollama-coder"

    # "Auto Hybrid" (task.metadata["preferred_provider"] == "hybrid") —
    # reuses llm/hybrid_router.py's existing local-vs-hosted RL bandit
    # (previously only ever wired into the PM/Trading chat pipeline's
    # three agent implementations — native_agent.py/langgraph_agent.py/
    # adk_agent.py) instead of building a parallel mechanism. "local"
    # here means ollama-coder specifically (a real, code-specialized
    # model on the same Ollama server), not hybrid_router's own generic
    # "ollama" default — see preferred_provider_for()'s own comment on
    # why that's overridable per call site.
    _HYBRID_LOCAL_PROVIDER = "ollama-coder"

    def _resolve_hybrid(self, task: Task) -> tuple[str | None, HybridSelection | None]:
        explicit = task.metadata.get("preferred_provider")
        if explicit and explicit != "hybrid":
            return explicit, None  # a caller's own real pick — never second-guessed
        # The Secondary Pipeline (llm/coding_pipeline.py) — the coding
        # agent's own fixed, always-on model cascade. Deliberately NOT
        # admin- or user-toggleable, unlike the PM/trading tracks'
        # primary_pipeline.py switch: "the user must never choose the
        # model" is an explicit architecture rule for this agent
        # specifically (see coding_pipeline.py's own docstring). Deferring
        # here (None, None) lets execute()'s explicit_cascade= argument to
        # get_llm_client drive routing instead of hybrid_router's
        # local/hosted bandit below, UNLESS a caller explicitly opted into
        # "hybrid" by name (a deliberate ask for THIS mechanism
        # specifically — never second-guessed, same as any other real
        # override above).
        if explicit != "hybrid":
            return None, None
        # Real gap found live: a plain default turn (explicit is falsy)
        # used to hard-prefer "ollama-coder" unconditionally via
        # _preferred_provider() below, bypassing hybrid_router's
        # complexity-aware local-vs-hosted choice entirely UNLESS a
        # caller opted in with the literal "hybrid" string — the exact
        # same bandit/timeout/escalation machinery just never applied to
        # the common case. Now applied by default too (same "auto"
        # extension native_agent.py's own domain/hybrid handling
        # already does for its pipeline). Confirmed live: a plain
        # "read this file" tool-call decision on ollama-coder (CPU) can
        # take 100+ seconds for what should be a trivial, near-instant
        # turn; this lets hybrid_router route it to a fast hosted model
        # instead when the message doesn't look complex, while a
        # genuinely code-heavy request can still land on ollama-coder —
        # now with LOCAL_TIMEOUT_SECONDS's escalate-away-from-slow
        # safety net (see execute()'s own preferred_timeout), which a
        # hard preferred= pick never had at all.
        selection = hybrid_router.choose(task.text, "delay_ok", local_provider=self._HYBRID_LOCAL_PROVIDER)
        preferred = hybrid_router.preferred_provider_for(selection, local_provider=self._HYBRID_LOCAL_PROVIDER)
        # None means "don't override" (hybrid_router picked "hosted") —
        # get_llm_client's own preferred=None is exactly "use the normal
        # chain," so this is never a special case to handle separately.
        return preferred, selection

    def _perform_confirmed_write(self, task: Task) -> tuple[str, bool]:
        pending = _PENDING_WRITES.pop(task.session_id, None)
        if not pending:
            return "Nothing pending to confirm — the write may have already timed out.", False
        path = Path(pending["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pending["content"], encoding="utf-8")
        return f"Written {len(pending['content'])} characters to {path}.", True

    def _run_agentic_loop(self, task: Task, client, messages: list, tools_used: list,
                          start_step: int, client_tools: bool,
                          hybrid_selection: HybridSelection | None = None,
                          last_provider: str | None = None) -> dict:
        """The bounded agentic loop shared by execute() (a fresh turn,
        start_step=0) and resume_tool_call() (continuing after a client
        tool call's result comes back, start_step=wherever it left off)
        — write → run → observe a failure → fix → run again → finish,
        all inside _MAX_TOOL_STEPS (client_tools mode gets the much
        bigger _MAX_TOOL_STEPS_CLIENT — see its own docstring).
        write_file (default mode) or ANY tool (client_tools mode)
        always short-circuits this immediately via pending_confirmation
        / pending_tool_call — a destructive or off-server action is
        never chained without the caller seeing it first.

        `last_provider` — the most recent real provider that actually
        answered a step of THIS client-tools turn, carried in from
        resume_tool_call() (via _PENDING_TOOL_CALLS, see below). Real gap
        found live: a long client-tools turn that genuinely exhausts
        _MAX_TOOL_STEPS_CLIENT (confirmed live: an exploratory task that
        kept calling list_files/grep/run_code past the 25-step budget)
        used to fall back to the "coding-agent" sentinel with NO real
        model attribution at all — the VS Code meta line showed nothing
        useful even though a real model (e.g. glm-5.3-flash) drove every
        one of those steps. Threading the last real provider through
        means that final "ran out of steps" message can still honestly
        say which model was doing the work, instead of looking like no
        model ever answered.
        """
        escalated = False
        max_steps = _MAX_TOOL_STEPS_CLIENT if client_tools else _MAX_TOOL_STEPS
        for step in range(start_step, max_steps):
            node = "coding_generate" if step == 0 else f"coding_generate(step {step + 1})"
            result = client.chat(node, messages, tools=_coding_tool_specs(client_tools))
            last_provider = result.provider or last_provider
            # Real, observed gap: a weak local model (confirmed with
            # qwen2.5-coder:1.5b via Ollama, comparing the SAME prompt
            # against Groq, which DOES use Ollama's native tool-calling
            # protocol correctly) sometimes just prints the tool-call
            # JSON as plain text instead of emitting a real provider-
            # level tool call — LLMResult.tool_call stays None, and
            # result.text is the JSON blob verbatim. Falling back to
            # parsing it out of the text is what makes coding-agent's
            # tools still usable on a model that unreliable.
            tool_call = result.tool_call or (
                _parse_text_tool_call(result.text, client_tools) if result.text else None)

            # A step further than the fallback parse above: when even
            # THAT can't recover a tool call because the text is
            # genuinely malformed (not just plain-text-JSON but broken
            # JSON/leaked prompt formatting — see
            # _looks_like_leaked_tool_schema's own docstring) OR is a
            # well-formed but HALLUCINATED tool call naming a function
            # that isn't a real tool (_looks_like_hallucinated_tool_call
            # — confirmed live as the dominant Coding Agent failure
            # mode), and this was the auto-picked local coder model
            # (never a caller's own explicit choice — that pick is never
            # second-guessed), retry ONCE against whatever this tier's
            # normal chain actually has available — a fresh
            # client.chat() call inline, not a `continue` to the next
            # `step`, so this doesn't cost this turn one of its
            # _MAX_TOOL_STEPS budget. Same "detect a real failure,
            # escalate to a better rung" shape already used for hybrid
            # mode's local-timeout escalation and validate()'s own
            # syntax-error retry below.
            if (not tool_call and not escalated and result.text
                    and (_looks_like_leaked_tool_schema(result.text)
                        or _looks_like_hallucinated_tool_call(result.text))
                    and not task.metadata.get("preferred_provider")):
                escalated = True
                client = get_llm_client("delay_ok", exclude_rule_based=True)
                result = client.chat(node, messages, tools=_coding_tool_specs(client_tools))
                last_provider = result.provider or last_provider
                tool_call = result.tool_call or (
                    _parse_text_tool_call(result.text, client_tools) if result.text else None)

            if not tool_call:
                return {"result": result, "messages": messages, "tier_log": list(client.log),
                       "retried": escalated, "tool_used": tools_used[-1] if tools_used else None,
                       "tool_steps": tools_used, "hybrid_selection": hybrid_selection}

            tool_result = self._handle_tool_call(task, tool_call, client_tools)
            if tool_result.get("pending_confirmation"):
                return {"pending_confirmation": True, "confirm_prompt": tool_result["confirm_prompt"],
                        "tier_log": list(client.log)}
            if tool_result.get("client_tool_call"):
                call_id = uuid.uuid4().hex
                _PENDING_TOOL_CALLS[call_id] = {
                    "session_id": task.session_id,
                    # The SAME resolved provider this whole turn already
                    # committed to — never re-run hybrid_router.choose()
                    # on resume, which would re-roll the bandit mid-turn
                    # and could switch horses (local vs hosted) partway
                    # through one logical task.
                    "preferred_provider": client.preferred,
                    "hybrid_selection": hybrid_selection,
                    "last_provider": last_provider,
                    "messages": messages,
                    "tools_used": tools_used,
                    "next_step": step + 1,
                    "tool_name": tool_call["name"],
                    "tool_args": tool_call["arguments"],
                }
                return {"pending_tool_call": {"call_id": call_id, "tool": tool_call["name"],
                                              "args": tool_call["arguments"]},
                        "tier_log": list(client.log)}

            tools_used.append(tool_call["name"])
            # Same "append and ask again" shape validate()'s syntax-
            # error retry uses below, generalized to loop — the prompt
            # explicitly offers BOTH options (finish now, or call
            # another tool) rather than forcing a final answer, which
            # is what makes a fix-and-retry sequence possible at all.
            messages = messages + [
                {"role": "system", "content": f"You called {tool_call['name']} with "
                                              f"{tool_call['arguments']!r}. Result:\n"
                                              f"{tool_result['summary']}\n"
                                              f"If this fully answers the request, give your final "
                                              f"answer to the user now. Only call another tool if you "
                                              f"genuinely need to (e.g. the code you ran failed and you "
                                              f"want to fix and retry it)."}]

        # Exhausted _MAX_TOOL_STEPS without the model ever giving a
        # plain-text final answer — real tool work still happened
        # (tools_used is non-empty), so report that honestly rather
        # than silently discarding it as if nothing occurred.
        exhausted = LLMResult(
            text=f"I made {len(tools_used)} tool call(s) ({', '.join(tools_used)}) but didn't reach a "
                f"final answer within this turn's limit — ask me to continue if you'd like me to keep "
                f"going.",
            provider=last_provider or "coding-agent")
        return {"result": exhausted, "messages": messages, "tier_log": list(client.log),
               "retried": False, "tool_used": tools_used[-1] if tools_used else None,
               "tool_steps": tools_used, "hybrid_selection": hybrid_selection}

    def execute(self, task: Task, plan: dict) -> dict:
        if plan.get("confirm_write"):
            text, ok = self._perform_confirmed_write(task)
            return {"write_performed": ok, "write_text": text}
        if plan.get("cancel_write"):
            _PENDING_WRITES.pop(task.session_id, None)
            return {"write_performed": False, "write_cancelled": True,
                    "write_text": "Okay, not writing that file."}

        client_tools = bool(task.metadata.get("client_tools"))
        # Intent-answer cache (framework/intent_cache.py) — only for
        # self-contained generate/general_coding prompts with no client
        # tools in play; debug/refactor/test/repo_analysis are excluded
        # outright since they're inherently grounded in a specific file/
        # error/codebase state. A hit skips _resolve_hybrid/get_llm_client/
        # _run_agentic_loop entirely — zero model calls for this turn.
        # Real bug found live: this gate never checked for an explicit
        # preferred_provider pick — a caller who deliberately asked for
        # a SPECIFIC model (e.g. testing "ollama-coder" directly, or a
        # real user picking one in the UI) could silently get a cached
        # answer from a DIFFERENT, earlier run's provider instead,
        # breaking the "an explicit pick is never second-guessed"
        # guarantee _resolve_hybrid() already upholds elsewhere in this
        # same file. Also fixes a real test failure this caused: two
        # tests sharing the same prompt text, one auto-picked (writes
        # the cache) and one explicit-provider (must never read it back).
        if (settings().INTENT_CACHE_ENABLED and not client_tools
                and not task.metadata.get("preferred_provider")
                and plan.get("intent") in _CACHEABLE_INTENTS):
            cache_t0 = time.time()
            cached = coding_cache.lookup(task.text)
            if cached is not None:
                return {"result": LLMResult(text=cached.text, provider=f"cache:{cached.provider}"),
                       "messages": plan["messages"],
                       "tier_log": [{"node": "intent_cache", "tier": "cache",
                                     "provider": cached.provider, "fallbackUsed": False,
                                     "ms": round((time.time() - cache_t0) * 1000, 1),
                                     "cacheHit": True}],
                       "retried": False, "tool_used": None, "tool_steps": [],
                       "hybrid_selection": None, "intent": plan.get("intent")}
        preferred, hybrid_selection = self._resolve_hybrid(task)
        # Real gap found live: this call never passed preferred_timeout
        # at all, unlike native_agent.py's identical hybrid-local case
        # (LOCAL_TIMEOUT_SECONDS) — a "local" pick here had no
        # escalate-away-from-slow safety net whatsoever. Only applied
        # when hybrid_router genuinely picked "local" (never for an
        # explicit caller-named provider, which keeps its own normal
        # timeout — see _resolve_hybrid's own comment on never second-
        # guessing an explicit pick).
        preferred_timeout = (LOCAL_TIMEOUT_SECONDS
                             if hybrid_selection is not None and hybrid_selection.arm == "local"
                             else None)
        # The Secondary Pipeline's fixed cascade is only meaningful when
        # nothing more specific already won above (an explicit
        # preferred_provider, or hybrid_router's own local/hosted pick) —
        # `preferred` is None in exactly that case (see _resolve_hybrid).
        client = get_llm_client("delay_ok", preferred=preferred, preferred_timeout=preferred_timeout,
                                exclude_rule_based=True,
                                explicit_cascade=coding_pipeline.effective_cascade()
                                if preferred is None else None)
        result = self._run_agentic_loop(task, client, plan["messages"], [], 0, client_tools, hybrid_selection)
        # Surfaced through to AgentResult.raw (see respond()) — lets a
        # caller (scripts/run_slm_test_suite.py) verify the intent layer
        # actually classified a prompt the way a human would, not just
        # that it ran.
        result["intent"] = plan.get("intent")
        return result

    def resume_tool_call(self, task: Task, call_id: str, output: str, error: str | None) -> dict:
        """Continues a turn paused by a client_tool_call (see
        _run_agentic_loop) once the caller — a VS Code extension
        executing the tool on the user's real machine, or any other
        client implementing the same protocol — reports back what
        happened. `task.session_id` must match the one the pending call
        was stashed under; this is the ONLY cross-request identity check
        an unauthenticated call_id would otherwise lack (the call_id
        itself is still an unguessable uuid4, but session ownership is
        what actually gates *using* it)."""
        pending = _PENDING_TOOL_CALLS.pop(call_id, None)
        if not pending or pending["session_id"] != task.session_id:
            return {"result": LLMResult(
                        text="That tool call already expired, was already resumed, or doesn't belong "
                            "to this session — please try again.",
                        provider="coding-agent"),
                    "messages": [], "tier_log": [], "retried": False, "tool_used": None, "tool_steps": []}
        tools_used = pending["tools_used"] + [pending["tool_name"]]
        observation = f"ERROR: {error}" if error else output
        messages = pending["messages"] + [
            {"role": "system", "content": f"You called {pending['tool_name']} with "
                                          f"{pending['tool_args']!r}. Result:\n{observation}\n"
                                          f"If this fully answers the request, give your final answer "
                                          f"to the user now. Only call another tool if you genuinely "
                                          f"need to (e.g. the code you ran failed and you want to fix "
                                          f"and retry it)."}]
        client = get_llm_client("delay_ok", preferred=pending["preferred_provider"],
                                exclude_rule_based=True)
        return self._run_agentic_loop(task, client, messages, tools_used, pending["next_step"], True,
                                      pending.get("hybrid_selection"), pending.get("last_provider"))

    def validate(self, task: Task, result: dict) -> dict:
        if result.get("pending_confirmation") or result.get("pending_tool_call") or "write_performed" in result:
            return result
        # Every return path below is a genuine FINAL outcome for this
        # task (never another pause) — exactly once per real hybrid pick,
        # regardless of how many client-tools rounds/resumes it took to
        # get here (hybrid_selection is threaded through every one of
        # those unchanged, see _run_agentic_loop/resume_tool_call).
        hybrid_selection = result.get("hybrid_selection")
        hybrid_latency_ms = sum(e.get("ms", 0) or 0 for e in result.get("tier_log") or [])

        def _record_hybrid(success: bool) -> None:
            if hybrid_selection is not None:
                hybrid_router.record_outcome(hybrid_selection, success=success,
                                             latency_ms=hybrid_latency_ms)

        # A turn that already used a tool (execute()'s agentic loop) has
        # its final text grounded by that tool call already — run_code
        # itself already proved the code works, and the reply here is a
        # natural-language explanation of that result, not literally
        # "generate code" output. ast.parse-checking prose that happens
        # to quote or reference a snippet is the wrong check to run on
        # it, and a REAL, observed bug when it does: an unnecessary
        # retry here burns an extra provider call, and if THAT call
        # falls through to the rule-based floor (e.g. a Groq per-minute
        # rate limit, more likely now that the loop itself can already
        # make several calls in one turn), it silently overwrites an
        # already-correct, already-tool-verified answer with the
        # generic PM-domain help text. Skipped entirely for any tool-
        # call turn; still runs for the plain "just generate code" path.
        if result.get("tool_used"):
            result["syntax_ok"] = None
            _record_hybrid(not result["result"].error)
            return result
        raw = result["result"]
        if raw.error:
            _record_hybrid(False)
            return result
        # Real bug found live: a hosted "reasoning" model (confirmed with
        # glm-5.3-flash) can spend its whole completion budget on internal
        # reasoning tokens and return a genuinely EMPTY visible answer —
        # LLMResult.error stays unset (the HTTP call itself succeeded), so
        # falling straight into the syntax check below saw zero fenced
        # blocks, treated that as "no syntax error" (syntax_ok=True), and
        # _record_hybrid(True) fed the hybrid bandit a false SUCCESS signal
        # for a turn that produced nothing — worse, respond() then cached
        # that blank text for every future caller of the same prompt (see
        # its own fix). One retry here, same shape as the syntax-error
        # retry below, gives this a real chance at a usable answer instead
        # of just relabeling the failure.
        if not raw.text or not raw.text.strip():
            client = get_llm_client("delay_ok", preferred=self._preferred_provider(task),
                                    exclude_rule_based=True)
            retry_messages = result["messages"] + [
                {"role": "system", "content": "Your last response was empty. Answer the request "
                                              "directly this time."}]
            retried = client.chat("coding_generate(retry)", retry_messages)
            result["tier_log"] += client.log
            result["result"] = retried
            result["retried"] = True
            retried_blank = not retried.error and not (retried.text and retried.text.strip())
            retried_blocks = (_extract_python_blocks(retried.text)
                              if not retried.error and not retried_blank else [])
            result["syntax_ok"] = (not retried.error and not retried_blank
                                   and _first_syntax_error(retried_blocks) is None)
            _record_hybrid(bool(result["syntax_ok"]))
            return result
        blocks = _extract_python_blocks(raw.text)
        error = _first_syntax_error(blocks) if blocks else None
        if error is None:
            result["syntax_ok"] = True
            _record_hybrid(True)
            return result
        # One retry, same "append the failure, ask again" shape
        # native_agent.py's output_guard retry uses for tool calls.
        client = get_llm_client("delay_ok", preferred=self._preferred_provider(task),
                                exclude_rule_based=True)
        retry_messages = result["messages"] + [
            {"role": "system", "content": f"Your Python code had a syntax error: {error}. "
                                          f"Regenerate it, fixed."}]
        retried = client.chat("coding_generate(retry)", retry_messages)
        result["tier_log"] += client.log
        retried_blocks = _extract_python_blocks(retried.text) if not retried.error else []
        result["result"] = retried
        result["retried"] = True
        result["syntax_ok"] = not retried.error and _first_syntax_error(retried_blocks) is None
        _record_hybrid(bool(result["syntax_ok"]))
        return result

    def respond(self, task: Task, validated: dict) -> AgentResult:
        # Secondary Pipeline usage/billing (llm/coding_pipeline.py +
        # framework/coding_usage.py) — every branch below reads from the
        # SAME validated["tier_log"], so recording once here, before the
        # branch-specific logic, covers every real-model-call path
        # (pending_tool_call, pending_confirmation, and a final answer)
        # without duplicating this call three times. record_usage() itself
        # no-ops for a turn with no real call in it (a pure cache hit, or
        # the "write_performed" confirmation branch, which has no
        # tier_log key at all).
        user = task.metadata.get("user") or {}
        if user.get("sub"):
            coding_usage.record_usage(
                user_id=user["sub"], session_id=task.session_id,
                task_category=str(task.context.task_type.value if hasattr(task.context.task_type, "value")
                                  else task.context.task_type),
                tier_log=validated.get("tier_log") or [])
        if validated.get("pending_tool_call"):
            call = validated["pending_tool_call"]
            # `text` here is a plain-language fallback status line for
            # any caller that doesn't specially handle pending_tool_call
            # (raw, below, is where a real client actually reads the
            # structured {call_id, tool, args} it needs to execute this
            # itself and call /tool_result). ok=True — this isn't a
            # failure, it's a normal mid-turn pause.
            return AgentResult(text=f"🔧 Calling {call['tool']} on your machine…",
                               ok=True, confidence=1.0,
                               raw={"pending_tool_call": call, "tier_log": validated.get("tier_log", [])})
        if validated.get("pending_confirmation"):
            prompt = validated["confirm_prompt"]
            return AgentResult(text=prompt, ok=True, confidence=1.0,
                               pending_confirmation=True, confirm_prompt=prompt,
                               raw={"tier_log": validated.get("tier_log", [])})
        if "write_performed" in validated:
            # A cancellation is a successful "did what was asked" outcome
            # (ok=True) even though no file was written — only a write
            # that was CONFIRMED but still failed to land (e.g. the
            # pending entry expired) counts as ok=False.
            cancelled = validated.get("write_cancelled", False)
            ok = validated["write_performed"] or cancelled
            confidence = 0.9 if validated["write_performed"] else (0.8 if cancelled else 0.0)
            return AgentResult(text=validated["write_text"], ok=ok, confidence=confidence,
                               raw={"write_performed": validated["write_performed"], "cancelled": cancelled})
        raw = validated["result"]
        # Real bug found live: a hosted "reasoning" model (confirmed with
        # glm-5.3-flash) occasionally spends its entire completion budget
        # on internal reasoning tokens and returns a genuinely EMPTY
        # visible answer — LLMResult.error stays unset (the HTTP call
        # itself succeeded), so `ok = not bool(raw.error)` alone treated
        # this as a normal success: confidence 0.85, a blank bubble shown
        # to the user, and — worse — that blank answer got written into
        # coding_cache.remember() below, silently serving the SAME blank
        # reply to every future caller of that exact prompt until the
        # cache entry's TTL expired. An empty answer is exactly as much
        # of a failure as a raised exception from this agent's own
        # caller's point of view, so it's treated as one here.
        blank_answer = not raw.error and not (raw.text and raw.text.strip())
        ok = not bool(raw.error) and not blank_answer
        # syntax_ok is a real three-way signal, not a boolean: True (the
        # plain-generation path's ast.parse check passed), False (that
        # check failed, even after the retry), or None (a tool-call
        # turn — validate() deliberately skips the check entirely, see
        # its own comment, since the tool call already grounds this
        # answer at least as well as syntax-checking would). Only False
        # should be penalized; None is "not applicable," not "failed."
        syntax_ok = validated.get("syntax_ok")
        confidence = 0.0 if not ok else (0.4 if syntax_ok is False else 0.85)
        if raw.error:
            text = f"Sorry — every model rung failed: {raw.error}"
        elif blank_answer:
            text = (f"Sorry — {raw.provider or 'the model'} returned an empty answer that time. "
                    f"Please try again or rephrase your request.")
        else:
            text = raw.text
        # Cache write — mirrors execute()'s read-side gate exactly (same
        # intent whitelist, no client tools, no tool call fired this
        # turn) plus ok/syntax_ok, and skips a cache HIT re-caching
        # itself (raw.provider already prefixed "cache:"). `ok` already
        # excludes a blank_answer turn, so a genuinely empty reply is
        # never written into the cache for a future caller to inherit.
        if (ok and syntax_ok is not False and not validated.get("tool_used")
                and validated.get("intent") in _CACHEABLE_INTENTS
                and not str(raw.provider).startswith("cache:")
                and raw.provider != "rule-based"
                and settings().INTENT_CACHE_ENABLED and not task.metadata.get("client_tools")):
            coding_cache.remember(task.text, text, raw.provider)
        # Which tier actually answered — "default" (glm-5.3-flash, tried
        # first on every call) vs "escalation" (any other real rung,
        # whether that's coding_pipeline's own GLM-full/Sol-xHigh/Opus-5
        # cascade or the normal delay_ok chain it falls through to —
        # coding_usage.py's own escalated/tier bookkeeping already treats
        # "not the default provider" as the one meaningful line, reused
        # here rather than re-deriving it). None for a non-answer (error,
        # cache hit, or the rule-based/pending branches, which never
        # reach this line at all).
        #
        # Real gap found live: _run_agentic_loop's own step-budget-
        # exhausted message and resume_tool_call's expired-call-id
        # message both set provider="coding-agent" — a sentinel for "this
        # agent produced this text itself, no model call backs it,"
        # never a real registered provider key. Without this guard,
        # raw.provider != DEFAULT_TIER_PROVIDER above would mislabel that
        # synthetic status text as "escalation tier," actively wrong
        # rather than just uninformative — exactly the "shown info
        # doesn't match what really happened" failure mode. Neither
        # provider nor tier is shown for it; VS Code's meta line simply
        # omits the "via ..." line for these, same as any other
        # non-answer.
        displayed_provider = None if raw.provider == "coding-agent" else raw.provider
        tier = None
        if ok and displayed_provider and not str(displayed_provider).startswith("cache:"):
            tier = "default" if displayed_provider == coding_usage.DEFAULT_TIER_PROVIDER else "escalation"
        return AgentResult(text=text, ok=ok, confidence=confidence,
                           raw={"provider": displayed_provider, "tier": tier,
                               "retried": validated.get("retried", False),
                               "syntax_ok": validated.get("syntax_ok"),
                               "tool_used": validated.get("tool_used"),
                               "tool_raw": validated.get("tool_raw"),
                               "tool_steps": validated.get("tool_steps"),
                               "intent": validated.get("intent"),
                               "tier_log": validated["tier_log"]})

    def evaluate(self, task: Task, result: AgentResult) -> float:
        # A syntactically valid code block is a stronger self-assessed
        # quality signal than plain confidence alone — override the base
        # class's confidence-only default with that extra information.
        # A confirmation question or a completed write isn't a "code
        # quality" judgement at all, so neither is scored against
        # syntax_ok — plain confidence stands for those. syntax_ok is
        # a three-way signal (see respond()'s own comment): only an
        # EXPLICIT False (the check ran and failed) is penalized here —
        # None (not applicable: a tool-call turn, or the confirmation/
        # write branches above, which never set this key at all) must
        # NOT be treated the same as a failed check.
        if not result.ok:
            return 0.0
        syntax_ok = result.raw.get("syntax_ok")
        if syntax_ok is False:
            return result.confidence * 0.5
        return result.confidence
