# CodingAgent "client tools" mode — real local autonomy for the VS Code extension

**Status: deployed and live.** The "NOT YET DEPLOYED" note below is from
this feature's original build session — long since shipped and iterated
on since (see the "Full auto-approve" section right below for the
latest addition: v0.9.10+).

## Full auto-approve (v0.9.10+) — "work like Claude Code until told to stop"

The confirmation-per-call behavior described throughout this doc is
still the *default*. On top of it, the header's 🔒/🔓 button
(`AUTO_APPROVE_STATE_KEY`, off by default, remembered per workspace)
lets the user explicitly grant full auto-approve — turning it ON shows
a real modal acknowledgement first (never a silent toggle, since this
is "the user granting permission," not an ambient default):

- **What changes**: `write_file`, `delete_file` (new — see below), and
  `run_code` in `localTools.ts` skip their per-call confirmation
  entirely while this is on, so a multi-step task (read a file, edit
  it, run it, fix a failure, run again) proceeds without stopping at
  every step — the actual "smooth, Claude-Code-like" behavior this was
  asked for.
- **What never changes, on or off**: `resolveWithinWorkspace()`'s
  path-jail (nothing outside the open workspace folder is ever
  reachable) and a new `looksDangerous()`/`isSensitivePath()` check —
  a `run_code` snippet matching a known-destructive pattern (`eval`/
  `exec`, `shell=True`, `os.system`, `rm -rf`, `shutil.rmtree`, `sudo`,
  `os.remove`/`os.unlink`) or a `write_file`/`delete_file` path that
  looks like a credentials file (`.env`, `.ssh/`, `id_rsa`,
  `.aws/credentials`, `.npmrc`, anything under `.git/`) ALWAYS shows
  the strongest modal confirmation, auto-approve or not — the "only
  stopping for a genuine security-sensitive action or scope violation"
  half of the ask. `delete_file` also uses `useTrash: true` (OS
  trash/recycle bin, not a permanent unlink) as an extra recovery
  margin under auto-approve.
- **New tool**: `delete_file(path)` — `CLIENT_ONLY_TOOL_SCHEMAS`
  (client_tools mode only, same reasoning as `list_files`/`grep`: the
  server has no real workspace of its own to delete from). Dispatches
  through `_handle_tool_call`'s existing client_tools branch with zero
  backend changes needed (it's tool-name-agnostic — every tool call
  already short-circuits to `client_tool_call` in this mode).
- **Bigger step budget for real continuous work**: the shared
  `_run_agentic_loop`'s cap was a flat `_MAX_TOOL_STEPS = 4` for both
  modes — plenty for the disposable server sandbox, far too small for
  "keep working across a real multi-file task." Client tools mode now
  gets its own `_MAX_TOOL_STEPS_CLIENT = 25` (each step there is
  already gated by the caller's own confirmation UX, so a bigger
  ceiling only bounds how much *useful* work one task can do, not how
  many destructive actions slip through unconfirmed); default mode's
  cap is unchanged. `MAX_CLIENT_TOOL_ROUNDS` (panel.ts's own client-
  side loop bound) was raised to match.
- **Tests**: `tests/test_coding_agent_client_tools.py` — `delete_file`
  is advertised in client_tools mode and never in default mode; a real
  6-round chained tool-call sequence proves the bigger client-mode
  budget applies (would have been cut short at round 4 under the old
  shared cap).

## The problem this solves

Algo Kodo (the VS Code extension) looked like Claude Code but
didn't act like it: the coding agent's `read_file`/`write_file`/`run_code`
tools only ever touched `CODING_WORKSPACE_DIR`, a disposable sandbox
*on the server* — never the user's actual open project. Getting real
files meant manually `@mention`-ing them one at a time; a `write_file`
landed on the server and had to be read back and copied into the local
workspace as a separate trick. There was no grep, no autonomous
multi-file browsing, nothing resembling Claude Code's actual behavior of
acting on your repo on its own initiative.

## The design

The model still runs server-side (same routing, same policy gate, same
provider chain, same ALGO AKANADE account/login) — only *where a tool call
actually executes* moves. When a turn opts into `client_tools` mode:

1. `CodingAgent` decides it wants to call a tool (same as always), but
   instead of executing it against the sandbox, it pauses the turn and
   returns `{call_id, tool, args}` to the caller.
2. The caller — the VS Code extension, or anything else implementing the
   same small protocol — executes that tool for real, against the user's
   actual machine.
3. The caller reports back what happened
   (`POST /api/v1/orchestration/tool_result`), and the SAME
   conversation/tool-budget resumes exactly where it left off.

This is the same shape as MCP / Claude Code's own client-executed-tools
loop, built directly into the existing orchestration API rather than as
a separate protocol.

## Backend changes

- **`app/framework/agents/coding_agent.py`** — see its module docstring,
  "CLIENT TOOLS MODE", for the full design writeup in context. Summary:
  - `task.metadata["client_tools"] = True` (set by the router below) is
    the switch. Every existing default-mode behavior (the sandbox,
    `_PENDING_WRITES`, the plain-text yes/no write confirmation) is
    **completely untouched** when this is absent/False — zero risk to
    any existing caller (the web PM chat widget, any other integration).
  - Two NEW tools, advertised **only** in this mode (the server has
    nothing to run them against otherwise): `list_files(glob)`,
    `grep(pattern, glob)`.
  - `_run_agentic_loop()` — the old `execute()` loop body, extracted so
    both a fresh turn and a resumed one share it.
  - `_PENDING_TOOL_CALLS: dict[str, dict]` — in-process, keyed by a
    fresh `uuid4` `call_id`, same "resets on restart, just re-propose"
    convention `_PENDING_WRITES` already uses. **This assumes the
    single-process deployment this backend already runs as today** (bare
    `uvicorn app.main:app`, confirmed via `deploy/update_containers.sh` —
    no gunicorn/multi-worker). If that ever changes, this needs a real
    shared store (Redis) instead — flagged, not built, since it isn't
    needed for the current deployment.
  - `resume_tool_call(task, call_id, output, error)` — the resume entry
    point; checks `task.session_id` matches the one the call was stashed
    under (the actual access control — `call_id` alone is unguessable
    but not itself an ownership check).

- **`app/framework/orchestrator.py`** — `OrchestrationResult.to_dict()`
  now also emits `result.pending_tool_call` (from `AgentResult.raw`),
  always present, `null` for every non-client-tools turn and every other
  agent — existing callers see nothing new.

- **`app/routers/v1/orchestration.py`**:
  - `ExecuteRequest.client_tools: bool = False` — new, defaults off.
  - `POST /api/v1/orchestration/tool_result` — new endpoint,
    `{session_id, call_id, output, error}` → same response shape as
    `/execute`. Bypasses the orchestrator's routing (there's no "which
    agent" decision on a resume) and calls `CodingAgent.resume_tool_call`
    directly via the registry.

### Tests

`tests/test_coding_agent_client_tools.py` (new, 6 tests, all passing) —
the first real coverage of CodingAgent's tool-calling loop at all (the
existing suite only checked routing). Covers: the two client-only tools
are advertised in client_tools mode and NEVER in default mode; a full
pending-call → tool_result → final-answer round trip; a resume rejected
for a mismatched `session_id`; a resume of an unknown `call_id` fails
cleanly; `write_file` pauses instead of touching the server sandbox in
this mode. Full suite: **634 passed** (one `test_orchestration_then_...`
test in `test_gmaofn_api.py` is pre-existing flaky — confirmed via
repeated runs *without* any of these changes present — Thompson-sampling
routing randomness occasionally picks `pm-agent-adk` over `pm-agent`;
unrelated to this feature).

## VS Code extension changes (v0.7.0)

- **`src/localTools.ts`** (new) — the other half. Real
  `read_file`/`write_file`/`run_code`/`list_files`/`grep`, executed
  against `vscode.workspace` and a real `python3` subprocess:
  - Every path is resolved against the open workspace folder and
    **refused** if it would escape it — the same containment
    `resolve_within()` already enforces server-side, just applied here
    since this is where real disk access now happens.
  - `write_file` and `run_code` (the two that touch the user's real
    machine destructively) each require a blocking **modal**
    confirmation (`vscode.window.showWarningMessage(..., {modal:true})`)
    before doing anything — deliberately a stronger gate than the
    existing inline chat Yes/No buttons, since this is real local
    disk/process access, not a server sandbox.
  - `run_code` shells out to `python3` on PATH; if it's missing, that
    comes back as a normal tool error the agent can react to, not a
    crash.

- **`src/api.ts`** — `send()` takes a `clientTools` flag (sets
  `client_tools` in the request body); new `sendToolResult()` posts to
  `/tool_result`. Both return the same `SendResult` shape, including the
  new `pendingToolCall`.

- **`src/panel.ts`** — `handleSend()` now drives the loop: while a
  response carries `pendingToolCall`, it posts a visible `tool`-type
  message (`🔧 tool(args)…`), calls `executeLocalTool()`, posts the
  outcome (`✅ .../❌ ...`), and calls `sendToolResult()` — repeating
  (bounded defensively at `MAX_CLIENT_TOOL_ROUNDS = 8` client-side, on
  top of the server's own `_MAX_TOOL_STEPS = 4` per turn) until a final
  answer arrives. A new header toggle (🤖/☁️, default **on**) lets the
  user turn this off per workspace and fall back to the old
  sandbox-only behavior; the old `mirrorServerWriteIfAny()` trick is
  skipped entirely in this mode since there's nothing left to mirror —
  the write already happened directly in the real workspace.

### Forward/backward compatibility

The extension is now built to send `client_tools: true` by default. Sent
to the **currently-deployed** (pre-this-change) backend, that field is
simply an unrecognized extra key on a Pydantic model — ignored, not
rejected — so until the backend below is actually deployed, the
extension behaves EXACTLY like v0.6.0 (no `pending_tool_call` ever comes
back, so the loop never triggers). Nothing breaks in the meantime; the
new capability is simply inert until both halves are live.

## Deploying the backend change

Nothing here needs a new dependency, a Dockerfile change, or a new
environment variable — pure application code.

```bash
./deploy/build_and_push_chatbot.sh
```

builds `dockerhubalgo/algo-kaizen-chatbot` (both `:latest` and a
`:<git-sha>` rollback tag) and pushes it. Auth: either `docker login`
yourself first, or `export DOCKERHUB_PASSWORD=...` before running it —
it does **not** hardcode a password the way
`deploy/update_containers.sh`/`deploy/deploy_all_in_one.sh` currently do
(a real, separate, pre-existing issue worth rotating/fixing — that
plaintext DockerHub password is committed to this repo in cleartext;
flagged here, not touched, since fixing it wasn't part of this task).

Then, **on the server**:

```bash
./deploy/update_containers.sh
```

— already pulls `dockerhubalgo/algo-kaizen-chatbot:latest` and restarts
`ak-chatbot` with the same env/volumes; no edits needed there. Verify
with `curl <chatbot base url>/api/health` afterward.

Rollback, if needed: re-run `update_containers.sh` after re-tagging a
previous `:<git-sha>` image as `:latest` and pushing that instead.
