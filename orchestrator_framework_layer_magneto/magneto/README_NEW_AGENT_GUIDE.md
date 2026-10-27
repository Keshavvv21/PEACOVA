# Building a new agent with ALGO AKANADE — example by example

This is the whole point of the framework: a new domain is one file, not
a fork of the PM logic. `app/framework/agents/coding_agent.py` is a
real, already-registered example — walk through it to build your own.
This file is the standalone, expanded version of the walkthrough in the
main `README.md`; start here for the full worked path from "empty file"
to "replaces `pm-agent` as the primary service." For what already
exists — a catalog of all 9 default agents and exactly what each one
does in every pipeline step — see `README_AGENTS.md` instead; this file
is the "how to add your own" companion to that "what's already here" one.

## Example 1 — the minimal agent (five lines that actually run)

Before the real Coding Agent example below, the smallest possible agent
— proves the six-step interface (`app/framework/agent_base.py`) doesn't
require any real logic to be valid and registrable:

```python
from app.framework.agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from app.framework.task import Task, TaskType

class EchoAgent(Agent):
    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="echo-agent", name="Echo Agent",
            description="Echoes the task text back — a template, not a real capability.",
            capabilities=frozenset({"echo"}), supported_tasks=frozenset({TaskType.GENERIC}),
            cost_per_call_usd=0.0, latency_profile=LatencyProfile(p50_ms=1, p95_ms=5),
            reliability_score=0.5, status=AgentStatus.ACTIVE,
        )

    def understand(self, task: Task) -> dict:
        return {"text": task.text}

    def plan(self, task: Task, understanding: dict) -> dict:
        return understanding

    def execute(self, task: Task, plan: dict) -> dict:
        return {"echo": plan["text"]}

    def validate(self, task: Task, result: dict) -> dict:
        return result

    def respond(self, task: Task, validated: dict) -> AgentResult:
        return AgentResult(text=validated["echo"], ok=True, confidence=1.0)
```

Register it (`app/framework/bootstrap.py`), and it's a real, callable
agent — `POST /api/v1/orchestration/execute` with
`required_capabilities: ["echo"]` reaches it, gets scored by the same
Thompson Sampling bandit as every other agent, and starts collecting
real feedback from its first call. Nothing here talks to an LLM — a
capability doesn't have to.

## Example 2 — a real one: the Coding Agent

**1. Subclass `Agent` and fill in the six-step interface.** Only
`execute()` and `validate()` need to do interesting work for most
agents — `understand()`/`plan()` can be near-identity if there's no
multi-stage reasoning to capture:

```python
from app.framework.agent_base import Agent, AgentMetadata, AgentResult, AgentStatus, LatencyProfile
from app.framework.task import Task, TaskType
from app.llm.router import get_llm_client

class MyCodingAgent(Agent):
    def __init__(self):
        self.metadata = AgentMetadata(
            agent_id="my-coding-agent",
            name="My Coding Agent",
            description="Generates and reviews code in <language>.",
            capabilities=frozenset({"code_generation", "code_review"}),
            supported_tasks=frozenset({TaskType.CODE_GENERATION, TaskType.CODE_REVIEW}),
            cost_per_call_usd=0.004,
            latency_profile=LatencyProfile(p50_ms=1200, p95_ms=4000),
            reliability_score=0.7,  # the router's cold-start prior, before any real feedback
            status=AgentStatus.ACTIVE,
        )

    def understand(self, task: Task) -> dict:
        return {"text": task.text}

    def plan(self, task: Task, understanding: dict) -> dict:
        return {"messages": [{"role": "system", "content": "You write correct code."},
                             {"role": "user", "content": understanding["text"]}]}

    def execute(self, task: Task, plan: dict) -> dict:
        client = get_llm_client("delay_ok", preferred=task.metadata.get("preferred_provider"))
        return {"result": client.chat("my_agent_generate", plan["messages"])}

    def validate(self, task: Task, result: dict) -> dict:
        return result  # coding_agent.py's real version runs ast.parse here — see below

    def respond(self, task: Task, validated: dict) -> AgentResult:
        raw = validated["result"]
        return AgentResult(text=raw.text, ok=not raw.error, confidence=0.8 if not raw.error else 0.0)
```

Note `execute()` goes through `get_llm_client()` (`llm/router.py`) — the
SAME model abstraction layer/fallback chain every other agent uses. Your
agent never talks to an SDK directly, so it inherits every provider
(Luna/Groq/Claude/HuggingFace/a custom local model) and the per-call
generation-params bandit for free — see `README_REINFORCEMENT_LEARNING.md`.

**2. Do REAL validation, not a stub.** `coding_agent.py`'s actual
`validate()` extracts every fenced ` ```python ` block from the reply and
runs it through `ast.parse` — a genuine syntax check — retrying the
model once with the parse error appended if it fails:

```python
def validate(self, task: Task, result: dict) -> dict:
    blocks = _extract_python_blocks(result["result"].text)
    error = _first_syntax_error(blocks)
    if error is None:
        result["syntax_ok"] = True
        return result
    retried = get_llm_client("delay_ok").chat(
        "coding_generate(retry)",
        result["messages"] + [{"role": "system",
                               "content": f"Syntax error: {error}. Regenerate it, fixed."}])
    result["result"], result["syntax_ok"] = retried, not retried.error
    return result
```

Whatever your domain is, put its REAL correctness check here — a test
run, a schema validation, a linter — not just a confidence number the
model made up about itself.

The real `coding_agent.py` now goes further than this template shows —
it has actual LLM-callable tools (`run_code` executes a snippet for
real in a sandboxed subprocess, `read_file`/`write_file` are scoped to
a workspace directory, and `write_file` requires an explicit
confirmation round-trip before anything is written) instead of pure
text generation. See
[`README_POLICY_GATE_AND_CODING_AGENT.md`](README_POLICY_GATE_AND_CODING_AGENT.md)
for the full walkthrough — the same tool-calling shape (`tools=` on
`client.chat()`, reading `result.tool_call`) is the pattern to copy for
any agent that needs to do more than talk.

**3. Register it.** `app/framework/bootstrap.py` is the only place that
needs to change, and only to ADD a line — nothing existing is touched:

```python
from .agents.my_coding_agent import MyCodingAgent
# inside bootstrap_default_agents():
registry.register(MyCodingAgent())
network.add_agent(MyCodingAgent().metadata)
```

Or register it at runtime, no code deploy at all, via
`POST /api/v1/agents/register` with a `webhook_url` — ALGO AKANADE builds a
`RemoteHttpAgent` proxy that POSTs `{task_id, text, session_id,
metadata}` to your service and expects `{text, ok, confidence}` back.
Your agent can be written in any language, hosted anywhere; ALGO AKANADE
never needs your source.

**4. That's it — the router finds it on its own.** No if/elif anywhere
names `"my-coding-agent"`. The next task whose `required_capabilities`
includes `"code_generation"` will have your agent in the candidate pool
(`orchestrator.py`'s capability pre-filter), scored by the same Thompson
Sampling bandit as `pm-agent`/`coding-agent`/`openclaw-agent`, and it
starts collecting its own feedback-driven posterior from its very first
real task:

```
POST /api/v1/orchestration/execute
{"task": "Write a function that validates an email address",
 "context": {"task_type": "code_generation", "required_capabilities": ["code_generation"]}}
```

If both `coding-agent` (the bundled example) and your new agent declare
`code_generation`, they compete for that traffic exactly as intended —
whichever one actually produces higher quality/lower cost/lower latency,
per ALGO AKANADE's own measured feedback, wins more of it over time. You
don't pick a winner; the bandit does.

## Example 3 — replacing `pm-agent` entirely

Everything above ADDS an agent alongside `pm-agent`. To make a different
agent — `coding-agent`, or your own — the *primary* service instead, two
integration surfaces exist, and which one you touch depends on whether
you're keeping the PM chat widget's UI or not:

**Path 1 — use the generic API, touch nothing PM-related (recommended).**
`POST /api/v1/orchestration/execute` (`routers/v1/orchestration.py`) never
mentions `pm-agent` anywhere — it routes purely by declared capabilities
through `orchestrator.py`'s Thompson Sampling bandit. Point a new client
at it with `required_capabilities: ["code_generation"]` and it's already
a fully working "coding service": `pm-agent` simply never enters the
candidate pool for that task type, because it doesn't declare
`code_generation` in its `supported_tasks`. No file changes at all — this
is what "generic and extensible" means in practice.

**Path 2 — repoint the existing chat widget/mobile app at your agent.**
The widget's own endpoint, `POST /api/chat/message`
(`routers/chat.py::chat_message`), is older than ALGO AKANADE and still calls
`agent/adapter.py`'s `get_agent(framework)` directly — the PM chatbot's
own native/langgraph/adk/claude tracks, bypassing ALGO AKANADE's
registry/orchestrator entirely. That's the one call site to change if you
want your new agent to power that *same* UI (same auth, same session/
history plumbing, same React/Flutter screens) instead of PM:

```python
# routers/chat.py — replace the get_agent(...) call with ALGO AKANADE's registry
from ..framework.registry import registry

agent = registry.get("my-coding-agent")   # instead of agent.adapter.get_agent(...)
result = agent.run(Task(text=body.message, context=TaskContext(),
                        metadata={"user": user, "preferred_provider": ...}))
reply = ChatReply(reply=result.text, ...)
```

`pm-agent` keeps working (it's still in the registry, still reachable via
Path 1's generic API) — this only changes what the *widget* talks to. In
all three examples above, nothing about PM's own code —
`native_agent.py`, `tools.py`, `pm_service.py` — is touched or forked;
ALGO AKANADE was built so the domain being served is a routing/config
choice, not a source-tree fork.

## Adding a security-relevant capability the same way

If your new agent (or a new tool on an existing one) touches an
external system — sends anything, reads anything sensitive — follow the
pattern `google_tools.py`/`user_mcp_client.py` already establish rather
than inventing a new one:

1. Every WRITE action needs an entry in `native_agent.requires_confirmation()`'s
   check (or reuse `tools.DESTRUCTIVE` if it's PM-domain-shaped).
2. `confirm_prompt_for()` should show the exact arguments, never a
   paraphrase.
3. If the action reads content FROM the external system, run it through
   `security_guard.classify(..., source=...)` before it reaches the
   user — see `README_CYBERSECURITY.md`.

This is what makes a new agent's write capability get the SAME "review
before approving" guarantee every existing write action already has,
for free, rather than a bespoke and possibly-forgotten check.
