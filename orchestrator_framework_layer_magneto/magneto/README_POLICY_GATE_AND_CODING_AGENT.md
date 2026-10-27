# Closing three real gaps — policy gate, pre-retrieval RBAC, a coding agent with actual tools

This documents three concrete fixes made after comparing ALGO AKANADE against
an internal architecture note describing a two-tier (Flash/Frontier)
LLM harness with a policy-gated router, a LiteLLM-style gateway, and
RAG permissions enforced before retrieval. Most of that note's
substance was already real in this codebase (`llm/hybrid_router.py` is
already a binary local/hosted bandit with the exact asymmetric-reward
principle the note describes; `llm/router.py`'s fallback chains +
`llm/provider_budget.py` already do the gateway's job). Three specific,
narrower gaps did not exist yet and are fixed here:

1. RAG permissions were enforced **after** retrieval, not before.
2. There was no policy layer that could override the routing
   classifier's pick outright — "policy always overrides the
   classifier" wasn't a real, enforced rule anywhere.
3. `coding-agent` (`app/framework/agents/coding_agent.py`) only ever
   generated text and syntax-checked it — no file access, no real code
   execution, nothing a "coding agent" usually implies.

## 1. Pre-retrieval RBAC (`app/retrieval/vectorstore.py`)

**Before**: `VectorIndex.search()` fetched `k * 3` results from the
vector store, unfiltered, then discarded whichever ones the caller
wasn't allowed to see (`post_filter()`). An unauthorized project's task
text was still embedded-compared, scored, ranked, and briefly held in
memory before being thrown away — never leaked over the wire, but a
real, avoidable gap between "retrieved" and "allowed to be retrieved."

**Now**: `ChromaStore.query()` and `MemoryStore.query()` both take an
`allowed_project_ids` set and filter BEFORE scoring — Chroma via its
own `where={"project_id": {"$in": [...]}}` metadata filter (so its ANN
search itself only ever considers matching rows), the in-memory
fallback by filtering the candidate dict before computing cosine
similarity at all. `VectorIndex.search()` computes the caller's visible
project set first and passes it straight into `store.query()`; a
member of zero projects returns `[]` immediately, no query at all.
`post_filter()` still runs afterward as a cheap defense-in-depth
double-check, not the primary enforcement mechanism anymore.

`framework/rag.py`'s `ChromaVectorStoreAdapter` (the `"pm-index"` RAG
store `POST /api/v1/rag/search` uses) is a read-through wrapper around
this exact same `vector_index.search()` call, so it gets the fix
transparently — no changes needed there.

## 2. The policy gate (`app/llm/policy_gate.py`, new)

The architecture note's "Policy gate first... Policy always overrides
the classifier" wasn't a real, first-class layer before — the closest
things were `llm/guardrails.py`'s injection scan (only ever run inside
`native_agent.py`'s own chat turn) and per-tool RBAC checks (about
whether an action executes, never about which model answers it).

`policy_gate.evaluate(messages)` is now called at the top of
`TierClient.chat()` (`llm/router.py`) — the one choke point every LLM
call in the system already passes through, tier or track or agent
regardless — so every existing caller gets this automatically, with
zero call-site changes. Three rules, all evaluated from the outgoing
message content alone:

| Rule | Trigger | Effect |
|---|---|---|
| **Injection block** | `guardrails.looks_like_injection()` matches the latest user turn | Call is refused outright — no provider is ever contacted. Reuses the exact same pattern list and refusal text `input_guard()` uses, not a second divergent copy. This is real defense-in-depth: `CodingAgent`, `CodeSecurityAgent`'s own LLM-phrasing call, and `SupplyChainAgent` never ran `input_guard()` at all before this — they call `TierClient.chat()` directly, bypassing native_agent.py's chat turn entirely. |
| **Confidentiality** | Message content matches an AWS access key, a PEM private-key block, a JWT shape, or a common vendor secret-token prefix (`sk_`/`ghp_`/`xoxb_`/...) | Routing is restricted to `LOCAL_PROVIDER_KEYS` (`ollama`, ``, `algo-engine` — providers this deployment itself runs) plus the deterministic `rule` floor. A secret-shaped string is never sent to a third-party hosted API. Degrades gracefully if no local rung is up, rather than hard-failing. |
| **Destructive-intent** | Latest user turn matches the same verb vocabulary `native_agent.py`'s own `confirm_prompt_for()` already keys its confirmation prompts on (delete/remove/archive/cancel/revoke/deactivate/drop/destroy/wipe/purge) | Local rungs are de-prioritized (moved after hosted ones, never removed — a destructive-sounding request still degrades to local if every hosted rung is down) and an explicit local `preferred` pick (e.g. Hybrid mode's own "local" arm) is ignored for this call. This is the note's "false routing to Flash is the expensive error" principle applied concretely: the highest-stakes turns get the best available model, not whatever the bandit's cheap word-count heuristic happened to pick. |

None of this decides **whether** a destructive action actually runs —
that's still `native_agent.py`'s `requires_confirmation()`/RBAC gate,
untouched. The policy gate only ever decides **which model answers**.

A blocked call is logged into `TierClient.log` with
`{"provider": "policy-gate", "policyAction": "block", "reason": ...}` —
visible in `tier_log` the same way a real provider attempt is, so an
audit of "why did this route the way it did" always has an answer, per
the note's own audit/feedback recommendation.

## 3. CodingAgent gets real tools (`app/framework/agents/coding_agent.py`)

**Before**: `execute()` called `client.chat(node, messages)` with no
`tools` argument at all — pure text generation. `validate()` ran
`ast.parse` on any fenced code block, a genuine syntax check, but
"parses" and "runs" are very different bars, and there was no file
access of any kind.

**Now**, three real LLM-callable tools, passed into
`client.chat(node, messages, tools=...)` in the same flat
`{name, description, parameters}` shape `agent/tools.py`'s PM tools
already use:

- **`run_code(code)`** — actually executes the snippet
  (`app/framework/sandbox.py`, new) in a subprocess: Python's own `-I`
  isolated mode (no `PYTHONPATH`, no user site-packages — a generated
  snippet can't accidentally or deliberately import this app's own
  modules off `sys.path`), a stripped environment, a fresh throwaway
  temp directory per call, and a hard wall-clock timeout (5s default).
  Real stdout/stderr/exit-code come back to the model, so a
  syntactically-valid script that raises at line 1 is no longer
  reported as fine. **This is explicitly not a container/VM sandbox** —
  no seccomp, no network namespace, no filesystem jail beyond the
  throwaway cwd. It's a real, honestly-scoped improvement over "never
  ran it at all," not a defense against a genuinely adversarial actor.
- **`read_file(path)`** — reads a text file scoped to
  `settings().CODING_WORKSPACE_DIR` (new config, defaults to
  `<DATA_DIR>/coding_workspace`) via the exact same
  `document_pipeline.resolve_within()` path-traversal guard
  `code_security_agent.py` already reuses for its own file scanning —
  a path that resolves outside the workspace is refused, never read.
- **`write_file(path, content)`** — **destructive, never executes on
  the first call.** The proposed write is stashed in an in-process dict
  keyed by `task.session_id` (resets on restart — same "never the only
  copy of anything a user would notice missing" convention
  `framework/task_memory.py`'s own bandit memory already uses) and the
  agent returns `AgentResult(pending_confirmation=True, confirm_prompt=...)`
  — a field that already existed on `AgentResult` but that nothing
  actually populated before this. Only a follow-up `Task` for the SAME
  `session_id` whose plain text matches a yes/no pattern (`_YES`/`_NO`,
  the same shape `agent/native_agent.py`'s own `YES`/`NO` use —
  duplicated rather than imported so this generic example agent isn't
  coupled to the PM-specific chat module) resolves it: "yes" performs
  the write, "no" cancels it and drops the pending entry. Read from
  plain task text, not a metadata flag — `routers/v1/orchestration.py`'s
  `ExecuteRequest` has no dedicated "confirm" field, so a metadata flag
  would have been unreachable through the real API a caller actually
  uses. This mirrors `native_agent.py`'s own "persist `{tool, args}`,
  ask, act on the reply" shape for its destructive PM tools, adapted
  since this agent has no native_agent-style session store of its own.

When the model calls `run_code` or `read_file`, `execute()` runs one
follow-up `client.chat()` round with the tool's real output appended as
a system message, so the final reply is grounded in what actually
happened — not what the model assumed would happen. `validate()`'s
`ast.parse` retry is unchanged for the plain-generation path (no tool
call at all); it's simply skipped for the pending-confirmation and
write-performed branches, which aren't "generate code" turns.

### Trying it

```
POST /api/v1/orchestration/execute
{"task": "Write a function that computes the nth Fibonacci number, then run it on n=10 to check it's right",
 "session_id": "demo-1",
 "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]}}
```

A model that calls `run_code` here will get back real output (`55`) to
confirm against, rather than trusting its own arithmetic. To try a
write:

```
POST /api/v1/orchestration/execute
{"task": "Save that function to fib.py in the workspace",
 "session_id": "demo-1", ...}
```

→ comes back with `pending_confirmation: true` and a `confirm_prompt`
naming the exact path and byte count. Confirming — plain text, the same
way a human would actually reply, no special field:

```
POST /api/v1/orchestration/execute
{"task": "yes", "session_id": "demo-1", ...}
```

→ actually writes `<CODING_WORKSPACE_DIR>/fib.py`. Replying `"no"`
instead cancels it — the pending write is dropped, nothing is written.

## What this deliberately does NOT fix

Being upfront about scope, same as this codebase's existing "known
limitation" callouts elsewhere (e.g. `retrieval/vectorstore.py`'s own
CVE-exposure note):

- **No container/VM sandbox.** `run_code` is a real subprocess with
  reduced privileges, not an isolated environment — see its own
  docstring. A genuinely adversarial user could still open a socket or
  spawn a process from inside it.
- **No codebase-aware RAG.** `read_file`/`write_file` are scoped to a
  flat workspace directory, not an indexed, symbol-searchable codebase
  — there's still no code-specific chunking/embedding anywhere in this
  system (`README_VECTOR_EMBEDDING_TABLES.md`'s five tables are all
  PM-domain or generic-document, unchanged by this work).
  `CodingAgent` still runs on the general `delay_ok` tier, not a
  dedicated coding-only tier.
  `code_security_agent.py`/`supply_chain_agent.py` remain static-analysis
  only — nothing in this pass wires them into `CodingAgent`'s own
  execute/validate pipeline.
- **No SSO/OIDC.** Auth is still the existing custom JWT against the
  real backend's own `/api/auth/login`, not corporate SSO — unrelated
  to routing/RBAC and out of scope for this pass.
- **No per-user/team rate limiting.** `llm/provider_budget.py`'s daily
  token/cost caps are per-provider, not per-caller — still true after
  this work.
- **The confidentiality/destructive-intent rules are regex/keyword
  heuristics**, not a trained classifier or a real data-classification
  taxonomy — same "targeted checks, not a general dialog-rail system"
  scope decision `guardrails.py`'s own docstring already makes for this
  codebase. False positives (a benign message that happens to contain
  the word "cancel") cost a needlessly-preferred hosted call, never a
  wrong answer or a blocked one — the same conservative-by-design bias
  the note itself recommends.

## Files touched

| File | Change |
|---|---|
| `app/llm/policy_gate.py` | New — the three-rule policy gate. |
| `app/llm/router.py` | `TierClient.chat()` calls `policy_gate.evaluate()` first; `force_local`/`force_hosted` filter and reorder the chain. |
| `app/llm/guardrails.py` | `looks_like_injection()` extracted from `input_guard()` so `policy_gate.py` can reuse it; `REFUSAL_MESSAGE` exported as a constant. |
| `app/retrieval/vectorstore.py` | `ChromaStore.query()`/`MemoryStore.query()` take `allowed_project_ids` and filter before scoring; `VectorIndex.search()` computes the visible set up front. |
| `app/framework/sandbox.py` | New — `run_python()`, the real subprocess execution `run_code` uses. |
| `app/framework/agents/coding_agent.py` | `run_code`/`read_file`/`write_file` tools, tool-call handling, confirmation-gated writes. |
| `app/config.py` | New `CODING_WORKSPACE_DIR` setting. |

All existing tests pass unchanged (627 passed / 1 pre-existing flaky
failure confirmed present before this work too — a Thompson-sampling
agent-variant selection test, unrelated to any of the above).

## Follow-up: Qwen2.5-Coder as coding-agent's preferred local model

`coding-agent` now has a second local Ollama model of its own
(`OLLAMA_CODER_MODEL`, default `qwen2.5-coder:1.5b`), registered as its
own `"ollama-coder"` provider key and explicitly preferred by
`CodingAgent._preferred_provider()` for every coding task — see
[`tool_layer_coding/README.md`](../../tool_layer_coding/README.md) for
the full picture, including why Algo Engine can't run it yet. Two real
bugs surfaced while testing this against the live deployment and are
fixed here too:

1. **Tool-call-as-text.** `qwen2.5-coder:1.5b` sometimes doesn't use
   Ollama's native tool-calling protocol reliably — it prints the tool
   call's JSON shape as plain reply text instead (confirmed by
   comparing the identical prompt against Groq, which DOES call tools
   correctly through the exact same code path). `coding_agent.py` now
   has `_parse_text_tool_call()`: a fallback that recovers a real tool
   call from the reply text when the provider-level `tool_call` field
   comes back empty — tried against every closed fenced block, an
   open/truncated fence (generation can stop before the closing
   marker), then the whole reply verbatim, via `json.loads()` (not a
   brace-counting regex, which breaks the moment the JSON has its own
   nested braces — every real tool call here does). This is a
   real-but-partial mitigation, not a full fix: a 1.5B model is still
   measurably less reliable at this than a hosted model, and some
   replies still show the raw JSON when even this parser can't make
   sense of it. If tool-call reliability matters more than free/local
   inference for your use case, a larger `OLLAMA_CODER_MODEL` (7B+) or
   an explicit `preferred_provider` override is the real fix, not a
   further-clever parser.
2. **A malformed tool argument crashed the whole agent.** Real,
   observed: the same small model occasionally emitted `{"code": {...
   nested object ...}}` instead of a plain string for `run_code`'s
   `code` argument. `sandbox.run_python()` has no way to hand a `dict`
   to `subprocess.run()`, so it raised a bare `TypeError` — caught by
   `framework/workflow.py`'s `SequentialStep`, which turned it into an
   opaque "Every agent in the sequence failed" for the user, with the
   real cause buried in a swallowed exception. `_handle_tool_call()`
   now validates every string argument (`code`/`path`/`content`) before
   using it — the same "argument must be a {type}" pattern
   `llm/guardrails.py`'s `output_guard()` already uses for PM tools —
   and returns a clear, recoverable message instead of ever reaching
   `subprocess.run()` with the wrong type. Confirmed fixed against the
   real deployment: 0 crashes across a 10-iteration stress-test loop
   that reproduced the bug reliably before this fix.

Files added on top of the table above: none — both fixes are inside
`app/framework/agents/coding_agent.py`. New config:
`app/config.py`'s `OLLAMA_CODER_MODEL`; new provider key
`"ollama-coder"` in `app/llm/router.py`'s `_providers()`/
`available_providers()`; `"ollama-coder"` added to
`app/llm/policy_gate.py`'s `LOCAL_PROVIDER_KEYS`.
