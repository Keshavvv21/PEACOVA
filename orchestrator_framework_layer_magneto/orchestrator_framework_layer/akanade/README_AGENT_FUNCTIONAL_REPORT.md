# Agent Functional Report

Status of all 9 registered agents (`framework/bootstrap.py`), combining
the automated test suite (run fresh this session) with what was
actually live-verified against production versus what still needs a
live check once the deploy server (unreachable for the second half of
this session — SSH timing out) is back.

## Automated test suite

`.venv/bin/python3 -m pytest -q` — **791 passed, 1 failed** (fresh run).
The one failure, `test_orchestration_execute_real_pm_prompt_end_to_end`,
is a pre-existing, already-documented flake (Thompson Sampling
occasionally routes to `pm-agent-adk` instead of `pm-agent` — confirmed
via `git stash` earlier this session to fail 2-3/4 times on completely
unmodified code, unrelated to anything changed). Not a regression.

## Per-agent status

| Agent | Test files | Automated tests | Live-verified this session | Notes |
|---|---|---|---|---|
| `pm-agent` | 2 | ✅ pass | ✅ (multiple real chat turns, PEFT dashboard, provider routing checks) | Core, most-exercised agent this session |
| `pm-agent-adk` | shares pm-agent's tests (framework param) | ✅ pass | Indirectly (same code path, ADK track) | The occasional-routing-flake target above — not a defect in the agent itself, a bandit-exploration artifact |
| `openclaw-agent` | 2 | ✅ pass | ⬜ Not exercised this session | No code touched this session; last known-good state is whatever the test suite covers |
| `coding-agent` | 3 (+ `test_coding_agent_hybrid.py`, `test_call_provider_directly.py`, `test_intent_reasoner.py` added this session) | ✅ pass | ✅ Extensively — real workspace edits, conversation memory, Auto Hybrid, intent reasoning, 105-prompt suite prepared | The single most-changed and most-tested agent this session |
| `mcp-utility-agent` | 1 | ✅ pass | ⬜ Not exercised this session | No code touched |
| `langchain-research-agent` | 1 | ✅ pass | ⬜ Not exercised this session | No code touched |
| `code-security-agent` | 3 | ✅ pass | ⬜ Not exercised this session | No code touched |
| `supply-chain-agent` | 2 | ✅ pass | ⬜ Not exercised this session | No code touched |
| `cloud-config-agent` | 2 | ✅ pass | ⬜ Not exercised this session | No code touched |

## What "live-verified" actually means here

Not a guess — real HTTP calls against the live production deployment,
this session, with logged results:

- Real conversation-memory round trip ("remember the number 42" → "you
  asked me to remember the number 42" in a follow-up call).
- Real Auto Hybrid pick for CodingAgent (`ok: true` against production).
- Real GLM/Hy4/Gemini/Luna provider visibility check via
  `GET /api/chat/providers`.
- Real latency measurements (13.8s cold / 0.2-5.4s warm for
  `ollama-coder`; 161ms → 9ms for the auth cache).
- Real nightly-PEFT pipeline run through GGUF conversion, quantization,
  Ollama tag creation, sanity check, and the regression gate — the gate
  correctly refused to promote a worse candidate.

## What's NOT yet verified, and why

- **The 105-prompt SLM test suite** (`scripts/run_slm_test_suite.py`)
  has not been run for real — needs the deploy server (unreachable this
  session after a point). Ready to run the moment it's back:
  `docker exec ak-chatbot python -m scripts.run_slm_test_suite --provider ollama-coder`.
- **5 of 9 agents** (`openclaw-agent`, `mcp-utility-agent`,
  `langchain-research-agent`, `code-security-agent`,
  `supply-chain-agent`, `cloud-config-agent`) were not touched by any
  change this session and were not re-exercised live — their status
  here is "automated tests pass," not "confirmed working against
  production this session." No code in this session's diff affects
  them, so there's no specific reason to suspect a regression, but that
  is an inference from "nothing changed nearby," not a live check.
- **The auto-approval bar UI** (Pause/Approve/Reject/Stop) compiles
  clean and is wired end-to-end, but wasn't click-tested in a real VS
  Code window this session (no GUI access from this environment) — see
  the demo runbook artifact from earlier in this session for the
  "verify once yourself" flag on this specific feature.
- **Mobile app**: code changes verified via `flutter analyze` (clean)
  and a release APK builds successfully, but wasn't run on a real
  device/emulator this session.

## Recommendation

Before treating this as fully verified end-to-end: reconnect to the
deploy server, run the 105-prompt suite for real, and do one real
click-through of the VS Code approval bar and one real mobile app
session. Everything else in this table already has a real, specific,
checkable basis for its status — this section exists so "all agents
verified" isn't overstated for the parts that weren't actually
re-checked this session.
