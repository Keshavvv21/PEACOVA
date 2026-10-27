# LLM Agent Architectures

An "agent," in the LLM context, is not a new kind of model. It's an LLM wired into a
control loop that lets it look at the state of the world, decide what to do next, take
an action, observe the result, and repeat — as opposed to a single forward pass that
maps one prompt to one response and stops. This document covers the two dominant
control-loop patterns (ReAct and plan-and-execute), how tool calling actually works at
the wire-protocol level, how agents manage memory across a loop that can run for many
turns, multi-agent composition, common failure modes, and — grounded in real code
rather than abstraction — how this repository's own `CodingAgent` implements a variant
of these patterns today.

---

## 1. Single-Shot vs. Agentic: What "Agent" Actually Means

A single-shot LLM call is a pure function: `response = LLM(prompt)`. The model reads
context once, generates text once, and the interaction is over. It has no way to check
whether what it said is true, no way to act on the world, and no way to recover from
being wrong — whatever comes out of the first (and only) generation is final.

An **agent** wraps that same underlying call in a loop with three added capabilities:

1. **Observe** — read state: prior output, a tool result, a file, an error message.
2. **Decide** — the LLM call itself, now conditioned on that observed state, decides
   the next action (call a tool, ask a clarifying question, or produce a final answer).
3. **Act** — execute the decided action against the real world (a shell command, an
   API call, a file write, a search query) and feed the result back into step 1.

The loop terminates when the model decides it has enough information to answer, or
when an external bound (a step limit, a timeout, a budget) cuts it off. Crucially, the
LLM itself is *not* changed by this — it's the same next-token predictor. What changes
is that it is now embedded in a harness that gives it turns, tools, and a reason to
keep going instead of stopping after one reply.

```
Single-shot:            prompt ──► LLM ──► response                (done)

Agentic:            ┌─────────────────────────────────────┐
                     │                                     │
             observe ▼          decide            act      │
        state ──► LLM(context) ──► action ──► execute ──────┘
                     ▲                                (new observation)
                     └── loop until: final answer | step limit | budget
```

This reframing is why "agent" is a spectrum, not a binary. A model that classifies
intent and picks a canned response is not agentic. A model that reads a tool's output
and decides whether to call another tool based on that output is. The defining feature
is **conditioning the next decision on an observation the model didn't have when the
loop started** — that's what a single forward pass structurally cannot do.

---

## 2. The ReAct Pattern (Reason + Act)

ReAct — from Yao et al., *"ReAct: Synergizing Reasoning and Acting in Language
Models"* (2022) — interleaves free-text reasoning ("thoughts") with concrete tool
calls ("actions") and their results ("observations"), all inside the same generation
stream. The model is prompted (or trained) to emit a repeating `Thought → Action →
Observation` triple until it has enough to answer.

The key insight: reasoning-only prompting (chain-of-thought) lets a model *think*
step by step, but can't ground itself in real information — it can only hallucinate
plausible-sounding facts. Acting-only prompting (plain tool use) lets a model *act*,
but without an explicit reasoning trace it's harder to steer or debug why it chose a
given action. ReAct interleaves both so the reasoning trace conditions the next
action, and each action's real-world result conditions the next reasoning step.

### Concrete Trace

A user asks: *"What's the exit code when I run this script, and is there a syntax
error in it?"* attaching `broken.py`.

```
Thought: I need to see the actual contents of broken.py before I can check for a
         syntax error or predict what running it would do. I don't have the file
         contents yet, so I should read it first.
Action:  read_file(path="broken.py")
Observation: "def add(a, b)\n    return a + b\n"

Thought: There's a missing colon after the function signature — that's a syntax
         error. I should confirm this by actually trying to run it, rather than
         just asserting it from a visual read.
Action:  run_code(code="def add(a, b)\n    return a + b\n")
Observation: "run_code exited 1.\nstdout:\n\nstderr:\nSyntaxError: expected ':' 
              (broken.py, line 1)"

Thought: The run confirms it — SyntaxError, exit code 1, caused by the missing
         colon after the parameter list. I now have everything needed to answer
         without guessing.
Final Answer: The script has a syntax error: a missing colon after
              `def add(a, b)`. Running it exits with code 1 and raises
              `SyntaxError: expected ':'` at line 1.
```

Notice what each observation *did*: the first one gave the model information it
structurally could not have (file contents); the second one converted a guess
("that's probably a syntax error") into a verified fact (an actual traceback with an
exit code). This is the core value proposition of ReAct over plain chain-of-thought —
each loop iteration is a chance to replace an assumption with a measurement.

### ReAct Loop Diagram

```
                         ┌────────────────────────────────┐
                         │            LLM CONTEXT          │
                         │  (system prompt + growing       │
                         │   Thought/Action/Observation     │
                         │   transcript)                    │
                         └───────────────┬──────────────────┘
                                         │
                         ┌───────────────▼──────────────────┐
                         │  1. THOUGHT                       │
                         │  Free-text reasoning about what   │
                         │  is known and what to do next.    │
                         └───────────────┬──────────────────┘
                                         │
                         ┌───────────────▼──────────────────┐
                         │  2. ACTION                        │
                         │  Emit a structured tool call:      │
                         │  {name, arguments}                 │
                         └───────────────┬──────────────────┘
                                         │
                              (harness executes the tool)
                                         │
                         ┌───────────────▼──────────────────┐
                         │  3. OBSERVATION                   │
                         │  Real tool output appended to      │
                         │  context as the next message.      │
                         └───────────────┬──────────────────┘
                                         │
                         is the task done? ──── no ──► back to 1
                                         │
                                        yes
                                         │
                         ┌───────────────▼──────────────────┐
                         │        FINAL ANSWER               │
                         └────────────────────────────────────┘
```

ReAct is **reactive**: there is no separate up-front plan. Each step is decided with
only the information available at that point, which makes it naturally adaptive to
surprises (a file doesn't exist, a command errors out) but also means the model can
wander, retry redundantly, or take more turns than a human would to reach the same
answer, since it's replanning from scratch after every single observation.

---

## 3. Plan-and-Execute Agents

Plan-and-execute agents (see also "plan-and-solve prompting," and frameworks like
LLMCompiler and early LangChain `PlanAndExecute`) split the loop into two distinct
phases instead of interleaving them:

1. **Planning phase** — one LLM call decomposes the task into an ordered list of
   steps *up front*, before any tool has been invoked. The plan is a static artifact:
   `["Read config.yaml", "Extract the database URL", "Test the connection",
   "Report success or failure"]`.
2. **Execution phase** — each step is executed in sequence (each may itself invoke a
   tool, or a sub-LLM-call), and the results are collected. Only after execution does
   a final synthesis step (sometimes another LLM call) turn the step results into an
   answer.

```
              ┌────────────────────────────────┐
              │         PLANNING PHASE          │
              │                                  │
              │  task ──► LLM ──► ordered plan:   │
              │       [step 1, step 2, ..., N]    │
              └───────────────┬──────────────────┘
                              │  (plan is now FIXED)
              ┌───────────────▼──────────────────┐
              │         EXECUTION PHASE           │
              │                                    │
              │   step 1 ──► execute ──► result 1   │
              │       │                              │
              │   step 2 ──► execute ──► result 2   │
              │       │        (optionally uses      │
              │       │         result 1 as input)   │
              │       │                              │
              │      ...                             │
              │       │                              │
              │   step N ──► execute ──► result N   │
              └───────────────┬──────────────────┘
                              │
              ┌───────────────▼──────────────────┐
              │       SYNTHESIS / RESPONSE         │
              │  combine result 1..N ──► answer     │
              └────────────────────────────────────┘

     (optional: a re-plan edge from EXECUTION back to PLANNING if a step
      fails badly enough that the original plan is no longer viable)
```

### ReAct vs. Plan-and-Execute: Tradeoffs

| Dimension | ReAct | Plan-and-Execute |
|---|---|---|
| Planning cost | Amortized: replans (implicitly) every step | One upfront planning call, cheap after that |
| Token efficiency | Lower — full reasoning trace repeated/extended every step | Higher — plan is compact and referenced, not regenerated |
| Predictability | Lower — number of steps and path are not known in advance | Higher — step count and shape are known before execution starts |
| Adaptivity to surprises | High — every step sees the latest real observation before deciding the next one | Low — a step failing (a file missing, an API erroring) may invalidate steps 3-7 of a plan made before that was known |
| Parallelism | Hard — inherently sequential, each action depends on the last observation | Easier — independent plan steps can execute concurrently since dependencies are explicit up front |
| Debuggability | Harder — the "plan" is implicit in a long, interleaved transcript | Easier — the plan is a first-class artifact you can inspect, log, and diff |

The practical rule of thumb: plan-and-execute wins when the task is well-scoped and
the steps to solve it are knowable in advance (e.g., a fixed data pipeline: "fetch →
transform → validate → load"). ReAct wins when the task is exploratory or the
environment is likely to surprise the model (debugging an unfamiliar codebase, where
step 2 genuinely cannot be decided until step 1's actual output is seen). Many
production agents are **hybrids**: a lightweight upfront classification or plan
narrows the space of likely next actions, but the actual step-by-step execution still
follows a ReAct-style observe/decide/act loop rather than a rigid, un-revisable script.
Section 9 below shows exactly this hybrid in this repository's own code.

---

## 4. Tool Use / Function Calling Mechanics

Tool use turns an LLM from a text generator into something that can take real actions
by giving it a fixed menu of functions it can *request* — the model itself never
executes anything; it only asks for something to be executed by the harness that
calls it.

### The Schema

Each tool is described to the model as a structured object with a name, a
natural-language description (this is what the model actually reads to decide *when*
to use the tool — vague descriptions produce misuse), and a JSON Schema for its
parameters:

```json
{
  "name": "run_code",
  "description": "Execute a Python snippet in a sandboxed subprocess and return its
                   real stdout, stderr, and exit code. Use this to verify generated
                   code actually runs, not just parses.",
  "parameters": {
    "type": "object",
    "properties": {
      "code": {"type": "string", "description": "The full Python source to run."}
    },
    "required": ["code"]
  }
}
```

This exact shape — `{name, description, parameters}` — is what every major provider's
function-calling API expects (OpenAI, Anthropic, and OpenAI-compatible local servers
all converge on essentially this schema, sometimes wrapped in a thin envelope like
`{"type": "function", "function": {...}}`).

### The Request/Response Cycle

1. **Advertise**: the harness sends the model's normal messages *plus* the list of
   available tool schemas on every call. The model doesn't have tools "installed" —
   it's simply told, per-request, what's on the menu this turn.
2. **Emit**: instead of (or alongside) plain text, the model can emit a structured
   tool-call object: `{"name": "run_code", "arguments": {"code": "print(1+1)"}}`. This
   is produced by the model's own token generation — modern tool-calling models are
   fine-tuned to emit this in a reserved format the provider's API then parses out of
   the raw generation and returns as a distinct, structured field (as opposed to free
   text the caller would have to parse itself).
3. **Execute**: the harness — never the model — receives that structured request,
   validates the arguments against the schema, and actually runs the corresponding
   function (a subprocess call, an HTTP request, a file read). The model has no direct
   access to the outside world; it can only ask the harness to act on its behalf.
4. **Observe**: the harness appends the tool's real output back into the conversation
   — typically as a new message the model will see on its *next* call — and the loop
   returns to step 1 with an expanded context.

```
   ┌──────────┐   1. messages + tool schemas    ┌─────────┐
   │  Harness │ ───────────────────────────────► │   LLM   │
   │          │                                  │         │
   │          │   2. {"name": "...",             │         │
   │          │       "arguments": {...}}        │         │
   │          │ ◄─────────────────────────────── │         │
   │          │                                  └─────────┘
   │  3. EXECUTE the real function                          
   │     (subprocess / API / file I/O)                      
   │                                                         
   │  4. append tool_result to messages                      
   └──────────┘
        │
        └── next call includes the growing transcript, LLM decides again
```

### The Fragile Part: When the Model Doesn't Emit a Real Tool Call

Not every model reliably uses the provider's native tool-calling protocol. Weaker or
smaller models sometimes *print the JSON as plain text* instead of emitting it through
the structured tool-call channel — syntactically it looks identical to a real tool
call, but the API returns it as ordinary generated text rather than as a parsed
`tool_call` field, so a harness that only checks the structured field will think the
model gave a plain-text answer and silently fail to act on an intended tool call. A
robust harness has to defend against this by attempting to parse a tool call out of
plain text as a fallback (fenced-code-block JSON is the common shape), and further
distinguish "the model printed valid tool-call JSON as text" from "the model's JSON is
outright malformed" (an unquoted key, a leaked prompt-formatting tag) — the former is
recoverable by parsing, the latter isn't and should be escalated rather than silently
producing garbage. This exact defensive pattern is implemented in this repo's own
agent — see Section 9.

---

## 5. Memory in Agents

An agentic loop that runs many turns accumulates state the model needs to reason
over, and that state doesn't fit cleanly into a single category. Agent memory is
typically split into three tiers:

### Short-term / Working Memory (the Context Window)

The simplest form: everything the agent has seen this turn — the system prompt, the
conversation history, and every tool call and observation so far — lives directly in
the prompt sent to the model on each loop iteration. This is exact-recall (nothing is
summarized or approximated) but is bounded by the model's context window, and every
tool result appended to it costs tokens on *every subsequent call* in the loop, not
just once. A loop that reads a 10,000-line file early on pays for that file's tokens
on every remaining step of the same turn.

### Scratchpad Memory

A specific pattern within short-term memory: an explicit, structured "notes so far"
section the agent writes to and reads from as it works, distinct from the raw
transcript. This is the literal origin of the term "scratchpad" in agent literature —
the ReAct transcript itself (the running Thought/Action/Observation log) *is* a
scratchpad: a working memory the model both writes to (each new thought) and reads
from (all prior thoughts/observations) within the boundaries of one task. It's
memory, but it is deliberately never intended to survive past the current task.

### Long-term Memory (Retrieval-Augmented Memory)

Anything that must persist *across* turns or sessions — past conversations, learned
facts, prior task outcomes — generally can't live in the context window indefinitely
(it would eventually exceed the window and would waste tokens on turns that don't need
it). The standard pattern is retrieval-augmented: past interactions are embedded and
stored in a vector store (or hybrid keyword+vector index), and at the start of a new
task the agent (or the harness on its behalf) issues a similarity search against that
store and injects only the top-k relevant results into the current context. This is
architecturally identical to RAG for documents — the difference is that the corpus
being retrieved from is the agent's own history rather than a static document
collection. It trades perfect recall (short-term memory has *everything*) for
scalability (long-term memory can hold arbitrarily much, at the cost of only surfacing
what a similarity search judges relevant, which can miss something a human would
consider obviously relevant if it's phrased differently than the query).

### Summary Table

| Tier | Lives in | Scope | Failure mode |
|---|---|---|---|
| Short-term / context window | The prompt itself | One turn/loop | Exhausts the context window (Section 7) |
| Scratchpad | The prompt itself, structured | One task | Same exhaustion risk, mitigated by only keeping structured notes instead of raw transcript |
| Long-term / RAG memory | External vector store | Across sessions | Retrieval miss — relevant memory not surfaced because the similarity search didn't rank it highly enough |

---

## 6. Multi-Agent Patterns (Brief)

Single-agent loops don't scale well to tasks that need genuinely different skills,
independent verification, or work broken across independent workers. Three common
compositions:

- **Orchestrator-worker**: one agent (the orchestrator) decomposes a task and
  dispatches sub-tasks to specialized worker agents (e.g., a "coding" worker, a
  "research" worker), then collects and synthesizes their results. The orchestrator
  never does the domain work itself — it only routes and combines. This is
  structurally plan-and-execute at the multi-agent level, where each "step" in the
  plan is delegated to a different agent instead of a different tool.
- **Debate**: two or more agents independently answer the same question, then see and
  critique each other's answers over one or more rounds, with a final round producing
  a converged (or judged) answer. The premise is that an error one agent makes
  independently is less likely to be replicated by a second agent reasoning from a
  different context, so cross-examination surfaces mistakes plain self-consistency
  (asking the same agent to check its own work) tends to miss.
- **Hierarchical delegation**: a manager agent breaks a large goal into sub-goals and
  assigns each to a subordinate agent, which may itself further decompose its
  sub-goal to agents below it. This differs from orchestrator-worker mainly in depth —
  workers in orchestrator-worker are typically leaf-level specialists with no further
  delegation, whereas hierarchical delegation allows arbitrarily many levels.

All three add coordination overhead (more LLM calls, more latency, more places for
context to get lost between agents) in exchange for either genuine parallelism,
specialization, or error-correction that a single agent's self-review can't reliably
provide.

---

## 7. Failure Modes

Agentic loops fail in ways a single-shot call structurally cannot, because they
compound across turns:

- **Tool-call hallucination**: the model invents a tool that doesn't exist, or calls
  a real tool with plausible-looking but fabricated arguments (a file path that was
  never mentioned, an API parameter it wasn't given). Guarding against this means
  validating every tool call's name and arguments against the actual schema before
  execution — never assuming a model-emitted call is well-formed just because it's
  syntactically valid JSON.
- **Infinite / unproductive loops**: a model that never reaches a state it considers
  "done" — endlessly re-reading the same file, retrying an already-failed action
  identically, or oscillating between two tool calls. Because nothing in the model
  itself enforces termination, the harness must impose an external step budget as a
  hard backstop.
- **Context window exhaustion from accumulating tool outputs**: every observation
  appended to the transcript is carried into every subsequent call in the same loop.
  A handful of large tool outputs (a big file read, a verbose stack trace, a large
  search result set) can consume the context budget faster than the reasoning itself
  does, degrading quality (the model attends less well to a longer context and may
  drop earlier constraints) well before hitting a hard token limit, and eventually
  hitting that hard limit outright.
- **Error compounding**: an agent that gets one step wrong and doesn't (or can't)
  detect it will condition every subsequent step on that error, compounding it rather
  than allowing it to self-correct. This is especially severe in plan-and-execute
  systems where an early plan step was based on a wrong assumption — later steps
  execute the rest of the (now-invalid) plan faithfully rather than noticing the
  premise broke, unless the system explicitly supports re-planning.
- **Escalating cost silently**: each additional loop iteration is a full additional
  LLM call (and, in multi-agent setups, potentially several). A task that should cost
  one call can silently cost the price of the step-budget ceiling if the model keeps
  finding reasons to call another tool instead of answering — real money and latency
  a single-shot design never risks.

The common thread: these are all consequences of the loop itself — none of them can
happen to a system that only ever makes one LLM call — which is exactly why bounding
the loop (step limits, timeouts, validation before execution) is not an optional
hardening step but a structural requirement of building an agent at all.

---

## 8. Diagram Recap

```
 ReAct (interleaved, reactive)              Plan-and-Execute (staged, upfront)
 ────────────────────────────              ─────────────────────────────────
                                             
  ┌─────────┐                                ┌──────────┐
  │ Thought │◄───────┐                       │   Plan   │  (one LLM call,
  └────┬────┘        │                       │  (fixed  │   produces the
       ▼             │                       │   list)  │   full step list)
  ┌─────────┐        │                       └────┬─────┘
  │ Action  │        │                             ▼
  └────┬────┘        │                       ┌──────────┐
       ▼             │                       │ step 1   │──► result
  ┌─────────────┐    │                       ├──────────┤
  │ Observation │────┘                       │ step 2   │──► result
  └─────┬───────┘  (loop until done)         ├──────────┤
        ▼                                     │  ...     │
   Final Answer                              ├──────────┤
                                              │ step N   │──► result
                                              └────┬─────┘
                                                    ▼
                                              ┌──────────┐
                                              │ Synthesis│──► Final Answer
                                              └──────────┘
```

---

## 9. How This Repo Does It: `CodingAgent`

The clearest way to see these patterns applied for real, rather than in the abstract,
is `orchestrator_framework_layer/akanade/backend/app/framework/agents/coding_agent.py`.
It doesn't implement a textbook ReAct loop or a textbook plan-and-execute pipeline —
it implements a pragmatic hybrid, and reading it closely shows exactly where each
pattern's ideas were borrowed and why.

### 9.1 The Planning Step Is a Regex Classifier, Not an LLM Call

`CodingAgent.plan()` (`coding_agent.py:415-434`) runs before any LLM call is made for
the actual task. Its job is to build the system prompt for the upcoming generation,
and the interesting part is *how* it decides what to put in that prompt:

```python
# coding_agent.py:423-430
# The intent-reasoning layer (llm/intent_reasoner.py) — a cheap,
# deterministic classification of what KIND of coding task this
# is (debug/refactor/test/repo-analysis/generate), appended as
# intent-specific guidance rather than run as a second model
# call, so it costs nothing in latency.
intent = classify_intent(understanding["text"])
content += guidance_for(intent)
```

`classify_intent()` itself (`intent_reasoner.py:98-102`) is a pure regex matcher, not
a model call:

```python
# intent_reasoner.py:98-102
def classify_intent(text: str) -> str:
    for intent, pattern in _PATTERNS:
        if pattern.search(text):
            return intent
    return INTENT_GENERAL
```

`_PATTERNS` (`intent_reasoner.py:38-57`) is an ordered list of five regexes —
`debug`, `refactor`, `test`, `repo_analysis`, `generate` — checked **in that specific
order**, with a comment (`intent_reasoner.py:33-37`) explaining why the order matters:
a prompt like "write a test that reproduces this bug" should classify as `debug`
(more specific, more actionable) rather than `generate` (the generic "write" verb
that also happens to appear in the sentence), so the more specific intents are checked
first and the first regex match wins. If nothing matches, it falls back to
`INTENT_GENERAL` (`intent_reasoner.py:31`).

Each intent maps to a short, imperative guidance string (`intent_reasoner.py:65-95`)
that gets appended to the system prompt — for example, the `debug` guidance
(`intent_reasoner.py:66-71`):

```python
INTENT_DEBUG: (
    " The user is asking you to DEBUG something real. Find the actual "
    "root cause before proposing a fix — read the file or run the code "
    "to see the real error first; do not guess at a fix from the "
    "description alone if you can verify it directly instead."
),
```

This "plan" is *not* a decomposed list of steps the way Section 3's plan-and-execute
description defines a plan — there's no `["step 1", "step 2", ...]` array, and no
separate execution phase that walks a fixed sequence. It is better understood as an
extremely cheap **pre-generation routing/prompting decision**: classify the task once,
deterministically, and use that classification to steer a single upcoming generation
toward the right behavior (verify before fixing, preserve behavior while refactoring,
cover edge cases when testing, etc.) — all without spending a second LLM round trip to
get there. The file's own module docstring is explicit about this design goal
(`intent_reasoner.py:6-13`): an SLM (small local model) is the latency-sensitive path,
so "reasoning about intent via yet another model call would directly fight the exact
latency goal this layer exists to support." This is a deliberate divergence from
"real" plan-and-execute, which typically *does* spend an LLM call on planning — here,
planning is made free by making it deterministic instead of generative.

### 9.2 The Execution Loop Is ReAct, With a Hard Step Bound

Once `plan()` has built the system+user messages, `execute()` (`coding_agent.py:670-688`)
hands off to `_run_agentic_loop()` (`coding_agent.py:558-668`), and *this* is where the
ReAct shape shows up clearly. Each iteration of the `for step in range(start_step,
max_steps)` loop (`coding_agent.py:574`):

1. Calls the LLM with the running `messages` transcript and the tool specs
   (`coding_agent.py:576`) — this is the "Thought" step, implicit in the model's own
   generation rather than a separately labeled field.
2. Extracts a tool call, with a fallback chain for models that don't use native
   provider-level tool calling — first the structured `result.tool_call`, then a
   regex/JSON-based extraction from plain text if the model just printed the call as
   text (`coding_agent.py:586-587`, backed by `_parse_text_tool_call()` at
   `coding_agent.py:335-358`), then an escalation to a different model rung if the
   text looks like genuinely malformed/leaked tool-schema output rather than a
   parseable call (`coding_agent.py:603-610`, using `_looks_like_leaked_tool_schema()`
   at `coding_agent.py:376-378`). This is exactly the "fragile part" described in
   Section 4 — a real, working defense against a real, observed failure mode, per the
   comment at `coding_agent.py:361-372` describing it as confirmed against
   `qwen2.5-coder:1.5b` under load.
3. If a tool call is found, executes it via `_handle_tool_call()`
   (`coding_agent.py:452-506`) — the "Action" step — and appends the real result back
   into `messages` as a new system message (`coding_agent.py:648-655`) — the
   "Observation" step, worded to explicitly invite the model to either answer now or
   call another tool: *"If this fully answers the request, give your final answer to
   the user now. Only call another tool if you genuinely need to..."*
4. If no tool call is found, the loop ends and whatever text the model produced is
   treated as the final answer (`coding_agent.py:612-615`).

This is a textbook ReAct loop, just without a labeled "Thought:" field in the prompt
— the model's reasoning is implicit in how it chooses its next tool call rather than
narrated as separate text, and the Observation is fed back exactly the way Section 2's
diagram shows. The loop is capped at `_MAX_TOOL_STEPS = 4` in the default (server
sandbox) mode, or `_MAX_TOOL_STEPS_CLIENT = 25` in "client tools" mode where a VS Code
extension executes tools against the user's real workspace (`coding_agent.py:263-275`)
— a concrete instance of Section 7's "bound the loop" requirement: the comment at
`coding_agent.py:256-262` is explicit that this exists to stop "a model that never
settles on a final answer" from "burn[ing] unbounded turns/cost."

A destructive tool call (`write_file`, or *any* tool at all in client-tools mode)
never executes inline — it short-circuits the loop immediately via
`pending_confirmation` (`coding_agent.py:502-505`) or a `client_tool_call` sentinel
(`coding_agent.py:465`, `coding_agent.py:617-640`), pausing the whole conversation
state in `_PENDING_WRITES` or `_PENDING_TOOL_CALLS` until a human (or the calling
client) confirms or reports back a result via `resume_tool_call()`
(`coding_agent.py:690-718`). This is a safety gate layered on top of the ReAct loop,
not a departure from it — the loop's Observation step simply gets its result from a
separate later request instead of the same one.

### 9.3 Which Pattern Does This Most Resemble?

**A hybrid, weighted heavily toward ReAct**, with the "planning" phase replaced by
something cheaper and narrower than the plan-and-execute literature means by the
term:

- It is **not** ReAct in the strictest sense, because there's no explicit, narrated
  "Thought:" text in the transcript — the reasoning is implicit in tool-call choice.
- It is **not** plan-and-execute in the strictest sense either, because
  `intent_reasoner.py`'s classification isn't a decomposition of the task into
  ordered steps with independent execution and a synthesis phase — it's a single
  routing decision (which of five guidance strings to prepend) made once, upfront,
  deterministically, purely to steer the *style* of the subsequent generation/loop,
  not to lay out what the loop will do.
- What actually drives task completion — `_run_agentic_loop()` — is unambiguously
  ReAct: observe (tool result) → decide (call another tool or answer) → act (execute)
  → observe again, bounded by a step limit, exactly matching Section 2's loop
  diagram, including the "give a final answer or call another tool" framing baked
  into every observation message.

So: **cheap deterministic pre-classification (a plan-and-execute-flavored idea, but
free instead of a real planning LLM call) feeding a bounded ReAct execution loop**.
This is a genuinely common real-world pattern precisely because it captures the best
of both tradeoffs from Section 3's table: the classification gives predictable,
consistent prompting behavior for a given intent type (plan-and-execute's strength)
at zero token/latency cost, while the actual multi-step work retains full ReAct-style
adaptivity to whatever a tool call actually returns (reading a file that turns out
empty, code that fails for an unexpected reason) — which a rigid, un-revisable
upfront plan would handle poorly.
