# How the plugin framework +  tier were built

A build log, not a reference — `README.md`, `backend/app/llm/README.md`, and
`backend/app/plugins.d/README.md` document *what* exists and *how to use
it*; this is *why it's shaped the way it is* and the order it was built in,
for anyone extending it later.

## The ask

Two things, bundled: "the underlying [model] layer, like , can run
this" (wire a local MoE inference engine in as a model provider) and "make
it editable framework using JSONs" (the bot's PM actions shouldn't require
a Python change to extend). Both landed as one commit because they turned
out to validate each other — see "Why  was easy" below.

## Step 0 — read before writing

Before touching anything, `github.com/JustVugg/`'s README was
fetched and read. It turned out to be an inference *runtime* (weight
streaming across VRAM/RAM/NVMe for MoE models too big for a single GPU),
not a chatbot framework — and critically, `./coli serve` exposes an
**OpenAI-compatible endpoint**. That one fact removed an entire class of
work: no new HTTP client, no new auth handling, no new response parsing.
It's the existing `OpenAICompatible` class (already used for Luna/Max/
Deepseek/Groq) pointed at a different `base_url`. Confirming this *before*
writing code is why the  half of this commit is ~15 lines across
`config.py` and `router.py`, not a new provider class.

## Step 1 — figure out where the plugin registry can live

The interesting design problem wasn't the JSON schema — it was **import
direction**. The plugin registry needs to be visible from two places that
don't currently know about each other:

- `agent/tools.py` — needs it to merge plugin schemas into `TOOL_SCHEMAS`
  and to resolve/execute plugin calls.
- `llm/providers.py` (`RuleBased`) — needs it so the offline floor can
  *trigger* a plugin from a typed phrase, not just a real LLM's
  tool-calling.

Checking the existing import graph first: `agent/native_agent.py` already
imports `from ..llm.router import get_llm_client` and
`from ..llm.guardrails import input_guard`. So **agent depends on llm**,
established and one-directional. If the plugin registry lived inside
`agent/` (the obvious first instinct — it's about tools), `llm/
providers.py` importing it would create `llm → agent → llm`, a cycle.

Resolution: put the registry in a new **sibling** module,
`app/plugins.py`, at the same level as `agent/` and `llm/`, depending only
on `services/pm_service.py` (already confirmed to be a true leaf — it
imports `db`, `models`, `auth`, nothing from `agent/` or `llm/`). Both
`agent/tools.py` and `llm/providers.py` import *down* into `plugins.py`;
neither imports the other any more than before. No cycle, and it was
verified by literally running the test suite immediately after wiring the
imports (see Step 5) rather than reasoning about it and hoping.

The JSON files themselves went in `app/plugins.d/` — a `.d`-directory
(the `/etc/foo.d/` convention) rather than reusing the module name, since
`plugins.py` (module) and `plugins/` (package) can't coexist.

## Step 2 — design the JSON shape around what already exists, not from scratch

The temptation with "make it JSON-configurable" is to invent a generic
scripting mini-language. That was deliberately avoided. Instead, the JSON
schema was designed to be a thin, safe *wiring* layer over machinery that
already existed and was already tested:

- **Entity resolution** — `resolve_project`/`resolve_task`/`resolve_user`/
  `resolve_sprint` already existed in `agent/tools.py`, already fuzzy-match
  via `rapidfuzz`, already return `Resolution(id=..., ambiguous=..., 
  missing=...)`. A plugin's `"resolve"` field just says *which* of these to
  call and what to name the result — `resolve_plugin_arguments()` in
  `tools.py` is ~15 lines because it's dispatch, not new resolution logic.
- **Execution safety** — rather than let a JSON file name any Python
  callable (a real code-execution risk — a JSON file could then call
  `os.system` if the wrong attribute lookup snuck in), `service_call.method`
  is checked against `SERVICE_METHOD_REGISTRY`, a fixed dict populated by
  hand in `plugins.py`. This mirrors `agent/tools.py`'s own
  `DESTRUCTIVE: frozenset[str]` pattern one file over — "a fixed, hard-coded
  set, never the LLM's judgement" (§6.1 step 9's comment) applied to *which
  functions chat can reach at all*, not just which ones need confirmation.
- **Response text** — `str.format(**fields)`, not an expression language.
  `fields` is a plain dict (resolved values + whatever the service call
  returned); `.format()` can only substitute named placeholders, it can't
  execute anything.

Net effect: a plugin author can wire a *new phrase* to an *already-reviewed,
already-role-gated* action. They cannot make the bot do anything a human
engineer hasn't already decided is safe to expose to chat.

## Step 3 — one worked example, not a spec with no implementation

`rename_project` was picked as the demo because it needed a genuinely new
`pm_service` method (proving the whole path, not just re-wiring an
existing tool) while being low-risk (rename isn't destructive, no
cascading side effects like `complete_sprint`'s backlog-reassignment
logic). `pm_service.rename_project()` was written first, by hand, using
the exact same pattern as its neighbor `archive_project()` — same
`_need(user, "Manager")` role gate, same `get_project` / mutate / `put` /
`_fire_hooks` shape. Only *after* that existed did `rename_project.json`
get written to call it.

## Step 4 — wire the dispatch points, smallest diff at each

In order, each verified to still pass the existing test suite before
moving to the next:

1. `agent/tools.py`: import `PLUGINS`, merge `{name: p.schema for ...}`
   into `TOOL_SCHEMAS`, add `resolve_plugin_arguments()` +
   `execute_plugin()`, add one `if name in PLUGINS:` line at the top of
   `execute()`. `DESTRUCTIVE` became a set union so a plugin marking
   itself `"destructive": true` gets the confirm gate for free.
2. `agent/native_agent.py`: one `if name in PLUGINS:` line at the top of
   `resolve_arguments()`, delegating to the function from step 1. Also
   fixed `confirm_prompt_for()`'s hard-coded `verbs[name]` dict lookup —
   it would `KeyError` for a plugin-declared destructive tool not in that
   dict — to fall back to the plugin's own `description`.
   `langgraph_agent.py` needed **zero changes**: it already imports
   `resolve_arguments`/`confirm_prompt_for`/`format_result` from
   `native_agent.py` rather than duplicating them (that sharing was already
   the point of §6.5's "both tracks trace the same path" design), so
   fixing the shared functions once fixed both tracks.
3. `llm/providers.py` (`RuleBased.chat()`): one loop over
   `PLUGINS.values()`, trying each plugin's compiled `trigger_patterns`
   *after* every hand-written pattern (so a plugin regex can never shadow
   a built-in phrasing) and *before* the final "here's what I can do"
   fallback text. Named regex groups (`(?P<project>...)`) map straight to
   `m.groupdict()` — no positional-group bookkeeping to keep in sync with
   the JSON, unlike the hand-written patterns' `m.group(1)`/`m.group(2)`.
4. `agent/state.py` (`format_result()`): one `if k == "plugin_result":
   return result["text"]` branch — the plugin already rendered its own
   text via `response_template`, so the formatter's only job is to not
   need to know anything about the specific plugin.

## Step 5 — verify, don't assume

Each claim above was checked, not asserted:

- **No import cycle**: `python -m pytest tests/ -q` immediately after
  adding the three new `from ..plugins import ...` lines — an
  `ImportError` would have surfaced instantly at collection time. It
  didn't; 49→58 tests passed as functionality was added.
- **Both agent tracks actually behave identically**: rather than trust
  the "langgraph reuses native_agent's functions" claim, both were hit
  directly —
  `POST /api/chat/message?framework=native` and the default `langgraph` —
  with the same phrase, same result.
- **The offline floor really triggers it, no API keys**: live `curl`
  against a running `uvicorn` process (no `LUNA_API_KEY`/etc. set) —
  `tier_log` in the response confirmed `"provider": "rule-based"` served
  the rename, not a fallback pretending to.
- **The clarify path doesn't silently guess**: tried `rename project
  zzznonexistentzzz to Whatever` — got "I couldn't find a project
  matching..." back, not a crash or a wrong rename.
- **The role gate holds**: a `member_token` user got a 403-shaped refusal
  from the same phrase an `exec_token`/Admin user succeeded with —
  `pm_service.rename_project`'s own `_need(user, "Manager")` fired exactly
  as it does for the REST route, because the plugin calls the *same*
  function, not a copy.
- **Malformed config doesn't take the bot down**: `tests/test_plugins.py`
  writes intentionally-broken JSON (invalid syntax, an unregistered
  `service_call.method`, missing required keys) to a temp dir and confirms
  `load_plugins()` returns nothing for those files rather than raising —
  mirroring `EXTRA_LLM_PROVIDERS`' existing "optional config, a typo isn't
  an outage" behavior.

All of the above became permanent tests in `tests/test_plugins.py` and the
new cases in `tests/test_retrieval_and_router.py` ('s `available()`
gating, the `LLMProvider` Protocol still holding for every registered
provider) — the live `curl` checks were how the design was *proven* during
development; the test suite is what keeps it proven afterward.

## What this pattern is for, and what it isn't

**Good fit**: a new PM action that's fundamentally "resolve some entities,
call one already-safe service function, say what happened" — renames,
status flips, simple CRUD-shaped actions. **Not a fit**: anything needing
multi-step logic, a new confirmation UX, or a new kind of side effect
(a genuinely new mutation *shape*, not just a new *name* for an existing
one) — that still means writing a `pm_service` function and registering it
by hand in `SERVICE_METHOD_REGISTRY`, the one deliberate, reviewed,
Python-side step the framework never tries to eliminate.
