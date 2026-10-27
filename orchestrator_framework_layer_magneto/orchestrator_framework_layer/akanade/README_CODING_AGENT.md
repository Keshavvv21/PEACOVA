# Coding Agent — real file/execution access, two ways

The third of ALGO AKANADE's three chat-reachable capabilities (alongside the
PM Tool and the Trading Agent), and the only one with real read/write/
execute access to a filesystem — either a server-side sandbox, or (via
the VS Code extension) the user's own real workspace.

This is a focused summary; the full design rationale and diffs live in
`README_POLICY_GATE_AND_CODING_AGENT.md` (§3).

## Two execution modes

**Server sandbox (default)** — `app/framework/agents/coding_agent.py`,
reached via `POST /api/v1/orchestration/execute`:
- `run_code(code)` — executes in a real subprocess (`app/framework/
  sandbox.py`): Python's `-I` isolated mode, a stripped environment, a
  throwaway temp cwd, a 5s wall-clock timeout. Real stdout/stderr/exit
  code come back to the model — not just a syntax check.
- `read_file(path)` / `write_file(path, content)` — scoped to
  `settings().CODING_WORKSPACE_DIR` (a server-side sandbox directory,
  path-traversal-guarded via `resolve_within()`). `write_file` is
  **destructive and never executes on the first call** — it stashes the
  proposed write and returns `pending_confirmation=True`; only a
  follow-up "yes" in the same session performs it.

**Client tools mode (`ExecuteRequest.client_tools: true`)** — the model
still decides *what* to do, but every tool call comes back as a
`pending_tool_call` instead of executing server-side; the caller
executes it for real and reports the result via
`POST /api/v1/orchestration/tool_result`. This is what the **VS Code
extension** (`tool_layer_coding/vscode-extension`) uses: `localTools.ts`
runs `read_file`/`write_file`/`run_code`/`list_files`/`grep` against
the user's *actual open workspace* (`vscode.workspace.workspaceFolders`),
with a real modal confirmation before any write — genuinely useful
local-autonomy, not a sandboxed toy. This mode is why the VS Code
extension can write a file you actually see in your own editor, instead
of a server path you'd have to go find.

## Combining with the Trading Agent

The Coding Agent's own framework (`app/framework/workflow.py`) already
supports a literal `SequentialStep([agent_a, agent_b])` hand-off — but
there's no REST endpoint that exposes picking two specific agents
explicitly today (`ExecuteRequest` has no such field). In practice,
"Trading Agent designs a strategy, Coding Agent writes the code" is
satisfied today through the **other** pipeline the Trading Agent
actually lives in (`app/agent/tools.py`, the same one the PM Tool
uses) — see `README_TRADING_AGENT.md`'s "Combining with the Coding
Agent" section for how that reply-in-one-turn approach works, and what
it does and doesn't cover.

## Verification

`tests/test_coding_agent_client_tools.py` (client-tools round trip,
`pending_tool_call`/`tool_result`), `tests/test_agent_flows.py` (server
sandbox `run_code`/`read_file`/`write_file` including the confirmation
gate). Live: the VS Code extension's own README
(`tool_layer_coding/vscode-extension/README.md`) documents a real,
manual end-to-end check against an actual open folder.
