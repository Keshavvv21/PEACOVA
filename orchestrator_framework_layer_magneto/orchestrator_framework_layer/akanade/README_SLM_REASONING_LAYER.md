# SLM Reasoning & Intent Layer

## Why this exists

Live-tested this session: the local SLM (`ollama-coder`, Qwen2.5-Coder
1.5B) gave a structurally-correct-but-shallow answer to a genuinely
complex prompt — it named the right concepts but missed the actual hard
parts a senior engineer would flag. Small models are more sensitive
than large ones to HOW a prompt is framed: an explicit statement of
what *kind* of task this is, and what specifically goes wrong for that
kind of task, measurably improves how well a small model follows
through, without touching the model's own weights.

That's what this layer is. It is **not**:

- A second model call. An SLM is already the latency-sensitive path
  (see `providers.py`'s `OLLAMA_KEEP_ALIVE` comment,
  `hybrid_router.py`'s `LOCAL_TIMEOUT_SECONDS`) — adding a "let me
  think about your intent first" round trip in front of every SLM
  generation would directly fight the whole reason this layer exists.
  It runs in the same single generation call, as prompt conditioning.
- A neural intent classifier. It's a regex/keyword heuristic — the same
  order of technique `llm/hybrid_router.py`'s own
  `_complexity_features()` already uses for its cheap, non-LLM
  complexity estimate. Cheap, fast (microseconds), and fully
  deterministic/testable, at the cost of being a heuristic rather than
  a learned classifier — a real, disclosed tradeoff, not a limitation
  hidden behind an impressive-sounding name.
- A substitute for capability. It does not make a 1.5B model reason
  like a 70B one. See `README_COMPLEX_EXAMPLE_TRAINING.md` for the
  companion honesty note on training's own limits here — the two
  documents make the same point from different angles (prompting vs.
  fine-tuning), because it's true from both angles.

## Architecture

```
User message
     │
     ▼
classify_intent(text)   ← app/llm/intent_reasoner.py
     │                     regex/keyword match against a fixed set of
     │                     coding-task intents, first match wins
     ▼
one of: debug | refactor | test | repo_analysis | generate | general_coding
     │
     ▼
guidance_for(intent)    ← a short, imperative instruction specific to
     │                     what actually goes wrong for THIS intent
     ▼
appended to CodingAgent.SYSTEM_PROMPT's content
(coding_agent.py's plan())
     │
     ▼
ONE generation call to the SLM (or whichever provider Auto Hybrid/an
explicit pick resolved to) — the intent guidance is baked into the
SAME system prompt that call already sends, not a separate call
     │
     ▼
AgentResult.raw["intent"] — the classified intent is surfaced back to
the caller (used by scripts/run_slm_test_suite.py to verify the
classifier agrees with the test suite's own human-assigned category)
```

## The six intents

Deliberately the same six categories the 100-prompt test suite
(`scripts/run_slm_test_suite.py`) is organized by — the reasoning layer
and the eval harness use one shared taxonomy, not two that could drift
apart.

| Intent | Triggers on | Guidance injected |
|---|---|---|
| `debug` | fix/error/bug/crash/exception/traceback/race condition/deadlock/memory leak/hangs/regression | Find the real root cause before proposing a fix — read the file or run the code rather than guessing from the description alone. |
| `refactor` | refactor/clean up/simplify/reorganize/extract/rename/restructure/de-duplicate | Preserve existing behavior exactly; change structure/clarity only unless a behavior change was explicitly asked for. |
| `test` | unit tests/pytest/test coverage/assert/mock/edge cases | Cover real edge cases (empty input, boundaries, error paths), not just the happy path. |
| `repo_analysis` | repository/codebase/across the repo/entire project/every file | Use `list_files`/`grep` to look at real files — never answer from assumptions about what the codebase probably contains. |
| `generate` | write/create/implement/generate/build/add/design | Return code that's syntactically complete and directly runnable, not a sketch. |
| `general_coding` | none of the above matched | No extra guidance — a plain knowledge question doesn't need debug/refactor/test framing bolted on. |

`debug`/`refactor`/`test`/`repo_analysis` are checked **before**
`generate`, because a prompt like *"write a test that reproduces this
bug"* mentions a generate-shaped verb ("write") but the more specific,
more actionable intent is `debug` — order matters and is covered by
`test_debug_wins_over_generate_when_both_verbs_present` in
`tests/test_intent_reasoner.py`.

## Where it plugs in

`app/framework/agents/coding_agent.py`'s `plan()`:

```python
intent = classify_intent(understanding["text"])
content += guidance_for(intent)
system = {**SYSTEM_PROMPT, "content": content}
```

This runs for every CodingAgent turn — client-tools mode (VS Code) and
the default server-sandbox mode alike — before the first model call is
ever made, and before Auto Hybrid's own local-vs-hosted routing
decision (`hybrid_router.choose()`), which is a separate, complementary
signal (complexity, not intent — see `README_COMPLEX_EXAMPLE_TRAINING.md`'s
discussion of what training actually changes about complexity handling).
The intent classification does not currently feed the routing decision
itself; it only conditions the prompt. Feeding intent into routing too
(e.g. always escalating `repo_analysis` to a hosted model, since that
needs more context-window headroom than a 1.5B model's SLM context
typically affords) is a natural next step, not yet built.

## Testing it

`tests/test_intent_reasoner.py` — 15 cases covering all six intents
individually, the debug-wins-over-generate ordering rule, and the
general-bucket empty-guidance invariant.

`scripts/run_slm_test_suite.py`'s 100-prompt run additionally reports,
per prompt, whether the intent layer's classification matches the
category the prompt was written under — a live check that the
heuristic actually agrees with a human's own categorization, not just
that it runs without crashing.
