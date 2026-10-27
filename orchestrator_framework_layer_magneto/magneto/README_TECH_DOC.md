# ALGO AKANADE — Tech Doc: One-Liner Architecture + Glossary

Every subsystem in this framework, compressed to a single arrow-chain
line each, plus a glossary that defines every term used anywhere in
this repo's docs. For the full multi-box diagrams see
`README_ARROW_DIAGRAMS.md`; for the math see `README_RL_ALGORITHMS.md`;
for prose explanations see `README_EXPLAINED.md`; for optimizers
specifically see `README_OPTIMIZERS.md`. This doc is the fast index —
one line to place any subsystem, one glossary entry to define any term.

## The entire architecture, one line

**User message → query refinement (RL-gated: bandit decides refine vs
skip, an embedding-memory cache reuses prior refinements for free, a
drift check reverts any rewrite that strayed too far) → input security
precheck (🔴-tier prompts — credential exfiltration, bulk-exfil asks,
"skip confirmation" — refused outright, before any tool runs) → agent
framework (native/langgraph/adk, picked by a bandit) → guardrails →
Hybrid-mode-or-manual model tier routing (local SLM vs hosted, RL-picked)
→ intent classification against a fixed PM
tool schema (trimmed for local models by another bandit) → optional
RAG-optimizer-picked retrieval (RL-picked store × algorithm, from live
PM data or an ingested folder, itself IVF/quantization-optimized) →
entity resolution against real data → guardrail-gated confirmation for
destructive/external actions → tool execution as the calling user →
output security flag (🟢/🟡/🔴, scored on the reply + any external
content it quotes) → templated reply → session persisted + every RL
layer's reward finalized + nightly PEFT data logged → reply broadcast to
the widget — with a nightly, regression-gated LoRA fine-tune of the
local model, an on-demand candidate-selection pipeline (the same
test-asset harness), and on-demand code/supply-chain/cloud-config
security scans, all run entirely outside that hot path.**

## The layers, one arrow line each — every layer, what it does, security included

The same flow as above, broken into one line per layer instead of one
run-on paragraph — copy/paste-able on its own:

- **Client** → PM chat widget, Flutter mobile app, or another agent over ACP — one message/task in.
- **Query refinement** → bandit decides refine-vs-skip → memory cache reuses a prior accepted rewrite for free → drift check reverts a rewrite that strayed too far from the original meaning.
- **Input security precheck** (`security_guard.py` / `native_agent.security_precheck()`) → a 🔴-tier prompt (credential/token exfiltration, bulk data-exfil requests, explicit "skip confirmation" instructions) is refused OUTRIGHT here, before any tool ever runs.
- **Framework picker** (pm-agent only) → native / langgraph / adk / claude, chosen by a bandit on measured latency + reliability, not a fixed default.
- **Guardrails** (`llm/guardrails.py`) → `input_guard()` injection scan on the way in, `output_guard()` schema-validates the model's tool call on the way out — identical on every framework track.
- **Model tier + Hybrid routing** → `classify_task_complexity()` picks realtime/delay_ok/bulk → either the normal provider fallback chain, or (Hybrid mode) a 2-arm bandit choosing local Ollama SLM vs hosted, with a hard timeout that escalates mid-turn if local is too slow.
- **Token optimizer** (local models only) → when the answering provider is Ollama, a bandit trims the tool schema to full/top12/top6/top3 based on keyword overlap with the message.
- **Intent classification** (`perceive_intent`) → one LLM call maps the message to a PM tool + arguments, against the (possibly trimmed) schema above.
- **RAG optimizer** (only if the message needs it) → bandit picks `{store}::{algorithm}`, retrieves and reranks context, injects it as an extra system message before intent classification finalizes.
- **Entity resolution** → tool arguments resolved against REAL data (a real task/project/user id), never an id the model invented.
- **Guardrail-gated confirmation** → every DESTRUCTIVE PM tool and every MCP tool call stops and asks "are you sure?" — the model's own judgment is never trusted alone for these.
- **Tool execution** → runs as the calling user, against the real PM service / MCP server / Google API — never an elevated bot credential.
- **Output security flag** (`security_guard.assess_security()`) → the final reply, plus any external content it quotes (email/calendar/chat/tool output), is scored 🟢 Safe / 🟡 Needs Review / 🔴 Security Risk and rendered as a badge next to the model/tier chip.
- **Templated reply** → `response_template.format(**fields)` — plain string formatting, never `eval`.
- **Session + reward persistence** → the turn is saved to `ChatSessions`; every RL layer that fired this turn (refinement, tier/hybrid, token optimizer, RAG optimizer, agent router) finalizes its reward; a nightly-PEFT-eligible Ollama turn is logged.
- **Reply broadcast** → back to the widget/mobile app over the same HTTP response (or WebSocket, for the live board).
- **ALGO AKANADE's own agent-level RL** (only via `/api/v1/orchestration/execute`, not the pm-agent path above) → capability pre-filter → Thompson Sampling agent pick → Workflow Engine shape → Shapley credit split across every agent that participated → reward → policy update on every participant's posterior.
- **Security agents** (on demand, not per chat turn — `code-security-agent`/`supply-chain-agent`/`cloud-config-agent`) → deterministic regex scans of real source/config + a real OSV.dev dependency-vulnerability lookup; an LLM only summarizes the findings list that already exists, never invents one.
- **Nightly PEFT fine-tune** (batch, cron, entirely outside the request path above) → collect reward-tagged Ollama turns → reward-weighted LoRA → GGUF-quantize → regression-gate against the currently-served model → promote or discard.

## Every use case, optimizer, pipeline, and algorithm — one arrow line each

### Query refinement (runs above every agent track, in routers/chat.py)
- **Gate**: `raw message → cheap heuristics (word count, filler phrases, missing punctuation) → bandit picks "refine" vs "skip" → skip: message passed through untouched, zero LLM cost`
- **Memory lookup (learns as it goes)**: `arm=="refine" → embed raw query → cosine vs quantized embeddings of past ACCEPTED refinements → similarity ≥ 0.93 → reuse cached refined text, zero LLM cost this time too`
- **Refine + structure (memory miss only)**: `ONE LLM call → {refined, missing_clarifications, suggested_queries} JSON → embedding similarity(raw, refined) as the drift signal, NOT a second LLM call`
- **Repair**: `similarity < 0.6 → rewrite judged too risky → revert to the original raw text, never remembered`
- **Remember**: `similarity ≥ 0.6 and text actually changed → quantize(embed(raw)) stored in RefinementMemory → future near-duplicates hit the cache above`
- **Reward finalize**: `turn completes → out_of_scope known → skip: 1.0/-0.6 · memory hit: 1.0/-0.1 · llm error: -0.5 · no-op refine: -0.3 · real refine: 1.0/-0.2 → bandit updates`
- **Confirmation surfacing**: `was_repaired or missing_clarifications → needs_confirmation=true → ChatReply.refinement chip → user sees it, turn never blocks on it`

### Core chat flow
- **A chat turn**: `receive → load_session → guards → tier+hybrid routing → perceive_intent (+ optional RAG context) → output_guard → resolve_entities → confirm-if-destructive → execute → format_result → persist + log + finalize rewards → reply`
- **Destructive action**: `tool call → requires_confirmation() true (DESTRUCTIVE ∪ every MCP tool) → "are you sure?" → yes/no next turn → execute-or-cancel`
- **Out-of-scope message**: `perceive_intent returns free text, no tool call → out_of_scope=true → UI gate ("show me anyway") → user accepts → shown`

### Model routing
- **Tier fallback chain**: `TierClient.chat → try rung 1 → fail → rung 2 → ... → rule-based floor (never errors)`
- **Hybrid mode**: `provider=="hybrid" → hybrid_router.choose(msg, tier) → arm local→preferred="ollama" | arm hosted→preferred=None → get_llm_client → normal chain`
- **Hybrid escalation on slowness**: `arm=="local" → preferred_timeout=LOCAL_TIMEOUT_SECONDS (10s) → local rung exceeds it → same exception path as a hard failure → hosted rung answers → log_hybrid_escalation_turn() feeds nightly PEFT`
- **Context-limit escalation**: `provider.chat() returns truncated=True (finish_reason=="length"/stop_reason=="max_tokens") → more rungs left? → keep same messages, try next rung → ChatReply.context_limit_switch=true`
- **Real token usage**: `every provider response → usage normalized to {prompt_tokens,completion_tokens,total_tokens} → ChatReply.usage → UI chip, every turn (not just local-trim estimates)`
- **Admin-added provider**: `POST /api/admin/providers → EXTRA_LLM_PROVIDERS registry entry → live in the picker, no redeploy`

### Token optimizer (local model only)
- **Schema trimming**: `provider startswith "ollama" → required=DESTRUCTIVE∪MCP tools (never cut) + optional ranked by keyword overlap → bandit picks full/top12/top6/top3 → reduced schema sent`
- **Reward finalize**: `finish() computes retried/out_of_scope → record_outcome(success, tokens_saved) → bandit updates`

### RAG optimizer
- **Retrieval decision**: `is_rag_relevant(msg) → refine_query → bandit picks "{store}::{algorithm}" → store.search(k×3) → apply algorithm → truncate to k → context injected as extra system message`
- **vector algorithm**: `query → embed → cosine similarity vs every vector → top k, store's own order`
- **lexical_rerank algorithm**: `wide candidate pool (k×3, vector-ranked) → re-rank ALL by keyword/Jaccard overlap → top k`
- **hybrid algorithm**: `wide candidate pool → blended score = 0.5·(vector/max_vector) + 0.5·lexical_overlap → top k`
- **Citation surfacing**: `build_context_with_included_hits() → only the chunks that survived the char budget → sources_used in the reply`

### Document ingestion
- **Ingest a folder**: `path (inside RAG_INGEST_DIR, traversal-guarded) → walk recursively → chunk every .txt/.md → fresh dedicated store → registered into rag_registry (now live for chat too)`
- **Size-based auto-scaling**: `chunks_indexed → _recommended_retrieval_params() → (k, max_context_chars) scaled min→max with corpus size → default for summarize()/suggest_prompts() unless overridden`
- **Summarize**: `store → retrieve_with_optimizer(scoped to that one store) → excerpts → LLM call on the "bulk" tier → grounded summary + sources_used`
- **Suggest prompts**: same retrieval → different ask → N concrete follow-up questions

### Vector storage + retrieval optimization
- **IVF clustering**: `corpus ≥ 50 vectors → Lloyd's k-means, K=round(sqrt(n)) clusters → query scores centroids first → scan only n_probe nearest clusters' members → O(sqrt(n)) instead of O(n)`
- **Scalar quantization**: `v_min,v_max per vector → scale=(v_max-v_min)/255 → uint8 code per component → 4-8x smaller, error bounded by scale/2`
- **Size gate**: `add_documents() → mark IVF stale → next search() rebuilds lazily → below IVF_MIN_VECTORS stays exact brute force`

### Agent-level RL (multi-agent workflows)
- **Agent routing**: `task arrives → cold-start prior U(a|T) (hand-designed) or learned posterior → Thompson-sample θ per agent → argmax θ·x → agent runs`
- **Multi-agent credit**: `task completes → per-agent Shapley value φ_i (marginal contribution, noisy-OR characteristic function) → reward split fairly → each agent's bandit updated with ITS share`
- **Adaptive reward weights**: `outcome → R = α·Quality+β·Success+γ·Feedback−δ·Cost−ε·Latency−ζ·Failure → weights (α..ζ) drift toward whatever correlated with high R, renormalized to sum to 1`

### Nightly PEFT fine-tune
- **Data collection**: `every ollama-* turn → finetune_log.py → JSONL {messages, response, provider, reward = 1 − 0.3·retried − 0.5·out_of_scope}`
- **Training**: `filter reward≥0.7, most recent 500 → render through the model's own chat template, mask loss to the assistant's turn → LoRA (r=8,α=16) → reward-weighted loss → merge → save`
- **Promotion pipeline**: `merged model → GGUF (f16) → quantize (Q4_K_M) → new Ollama tag (base untouched) → sanity check → regression gate vs base tag → pass → register with live chatbot, no restart`
- **Cron timeline**: `02:00 DB dump → 03:30 nightly_finetune.sh → data → train → convert → gate → register → prune old checkpoints`

### Candidate-selection eval pipeline (on demand)
- **Full sweep**: `test_assets (32 gold+rubric items) × candidates (local + OpenRouter, hosting:local|cloud) → harness scores deterministic (exact/regex/schema) or llm_judge (bias-guarded) → human-sample 12% → aggregate (weighted score, latency p50/p95, $/1k) → decide (ranked top 1-2 + manual license/hosting checklist)`
- **Regression gate (reused nightly)**: `same harness, just base-tag vs new-tag, local Ollama only, zero cost → pass/fail/couldn't-run`

### Voice
- **Voice turn**: `mic audio → STT → text → EXACT same handle_message() as typed chat → reply → TTS → spoken audio`

### MCP (external tools)
- **MCP tool call**: `config.py's MCP_SERVERS → mcp_client.mcp_tool_specs() → merged into TOOL_SCHEMAS → model picks one → requires_confirmation()=true (always, every MCP tool) → confirm_prompt_for() shows name+args ONLY, never the server's own description → user confirms → tools.execute() dispatches to the real external server`

### External SDK
- **Calling the API from outside**: `pip install gmaofn-client → MAGNETClient(base_url).login() → send_message(provider="hybrid"|None|real-id) / rag_ingest_path() / rag_summarize() → same REST endpoints, zero repo-code dependency`

---

## Glossary

**Agent** — one registered unit of work-doing logic (e.g. `pm-agent`,
`coding-agent`) implementing `understand → plan → execute → validate`;
ALGO AKANADE's agent registry can route a task to any of several.

**Arm** — one candidate option a bandit chooses between (e.g. "local"
vs "hosted" in Hybrid mode, or one tool-schema reduction level).

**Bandit / multi-armed bandit** — the reinforcement-learning setting of
repeatedly choosing among fixed options ("arms") with no state carried
between choices (γ→0 special case of full RL), learning purely from
each choice's own observed reward. Every RL layer in this repo is one.

**Bayesian linear regression** — the statistical model
`LinearBanditCore` maintains per arm: a probability distribution over
"how much each context feature predicts reward," updated from data
rather than fixed.

**Bias guard (LLM-as-judge)** — never telling a judge model which
candidate produced an answer, so it can't be swayed by knowing the
model's identity/reputation.

**Brute-force search** — comparing a query against every single stored
vector, O(n) — correct and simple, but slow once n is large; what IVF
clustering exists to avoid.

**Candidate-selection pipeline** — the benchmarking harness (§A-G) used
to decide which LLM is actually worth self-hosting, reused nightly in
miniature as the regression gate.

**Chunking** — splitting a long document into smaller passages before
embedding/retrieval, since retrieving a whole document is both less
precise and more expensive than retrieving the specific relevant
paragraph.

**Cold start** — the state of a bandit arm with no observed data yet;
seeded from a hand-designed prior (e.g. `utility_prior`) rather than
starting from nothing.

**Context vector** — the small, numeric description of "what's true
about this decision right now" a bandit uses to predict reward (e.g.
`[message_word_count, tier_is_delay_ok]`).

**Confirmation gate** — the "are you sure?" step before any destructive
PM action or any MCP tool call executes; the model's judgement is never
trusted alone for those.

**Cosine similarity** — a measure of how similar two vectors' directions
are (dot product of unit-normalized vectors); the standard scoring
function for embedding-based retrieval.

**Query refinement layer** — the RL-gated rewrite-for-clarity step
(`gmaofn/query_refinement.py`) that runs BEFORE any agent track sees a
message: a bandit decides whether refining is worth an LLM round trip,
an embedding-based memory cache reuses past refinements for free, and a
drift check reverts any rewrite that changed the message's actual
meaning. See `README_OPTIMIZERS.md` §7.

**Refinement memory** — the quantized-embedding cache a repeat/near-
duplicate query hits to reuse a prior accepted refinement without
another LLM call; what makes the refinement layer "refine as it goes"
rather than a static prompt run cold every time.

**Task memory** — `gmaofn/task_memory.py`'s quantized-embedding memory
for agent ROUTING: a request similar to one an agent already won (or
lost) on before nudges that agent's cold-start seed for this new
request. Same mechanism as Refinement memory above, applied to "which
agent" instead of "how to phrase this." See `README_OPTIMIZERS.md` §8.

**Context-limit escalation** — when a provider's own response says it
got cut off by its context/output limit (OpenAI-shape
`finish_reason=="length"`, Anthropic `stop_reason=="max_tokens"`),
`TierClient` treats it like a failure and tries the next, bigger rung
with the SAME conversation — the same fallback path a hard error
already takes, so nothing the user typed is lost. See
`README_OPTIMIZERS.md` §9.

**DESTRUCTIVE** — the fixed, hard-coded set of PM tool names
(`delete_task`, `archive_project`, `complete_sprint`, `delete_user`, +
destructive plugins) that always require confirmation — never the
model's own judgement.

**Description-leak fix** — `confirm_prompt_for()`'s deliberate choice to
show only an MCP tool's NAME and ARGUMENTS in the human-facing "Allow?"
confirmation, never its own `description` string — that text is written
to persuade a MODEL (the bundled `mcp-server-fetch` reference server's
real description tries to talk a model into believing it now has
internet access despite being told otherwise), not to inform a human
making a trust decision. See `README.md`'s MCP section.

**Embedding** — a numeric vector representation of text such that
semantically similar text produces nearby vectors; what makes
similarity search possible at all.

**Fallback chain** — the ordered list of providers a tier tries in
sequence until one succeeds, ending in a deterministic rule-based
floor that never errors.

**GGUF** — the file format llama.cpp/Ollama load quantized model
weights from; the target format any fine-tuned HF model must be
converted to before Ollama can serve it.

**Guardrail** — any check (input_guard, output_guard,
requires_confirmation) that runs identically regardless of which
model/provider/optimizer path answered a turn — optimizers change
speed/cost, never whether guardrails apply.

**Hosting: local vs cloud** — local = weights and computation both on
your own hardware (Ollama), free, fully controlled; cloud = an API call
to someone else's GPUs (OpenRouter et al.), billed per token, using
THEIR key with the upstream provider, not yours.

**Hybrid mode** — the RL-selected choice, per message, between routing
to the free local SLM or a hosted model, based on a cheap non-LLM
complexity estimate.

**Intent** — in this system specifically: which single PM tool (if any)
a message maps to, plus its extracted arguments — a narrow
classification+extraction task, not open-ended reasoning.

**IVF (Inverted File Index)** — the standard first-level approximate-
nearest-neighbor technique: partition vectors into clusters, and at
query time only search the nearest few clusters instead of everything.

**k (retrieval)** — how many chunks a retrieval call asks for; scaled
to corpus size in this repo rather than left fixed.

**k-means / Lloyd's algorithm** — the clustering algorithm IVF uses:
alternately assign each point to its nearest centroid, then recompute
each centroid as the mean of its assigned points, until stable.

**LinearBanditCore** — the ~60-line shared class (`gmaofn/bandit.py`)
implementing Thompson Sampling over a linear reward model; reused,
never reimplemented, by every RL layer in this repo.

**LoRA (Low-Rank Adaptation)** — a parameter-efficient fine-tuning
technique: freeze the base model, train small "adapter" matrices
instead — a few million trainable parameters instead of the full
model's billions.

**MCP (Model Context Protocol)** — a standard for connecting an agent
to external tool servers; every MCP tool call in this repo requires
explicit user confirmation, the same gate a destructive PM action gets.

**Modelfile** — Ollama's config format describing how to build a model
tag (weights + chat template + parameters); this repo reuses the base
model's own Modelfile when creating a fine-tuned tag, swapping only the
weights reference.

**Noisy-OR** — the characteristic function this repo's Shapley
computation uses for a multi-agent coalition's value: the probability
at least one member produced a good result, given each member's own
independent quality.

**PEFT (Parameter-Efficient Fine-Tuning)** — fine-tuning approaches
(of which LoRA is one) that update a small fraction of a model's
parameters instead of all of them — practical on CPU where full
fine-tuning isn't.

**Perceive_intent** — the one LLM call per turn that performs intent
classification (see Intent above) with the full/reduced tool schema
attached.

**Posterior (Bayesian)** — the updated belief about an arm's reward
behavior after folding in observed data — narrower/more confident than
the prior, but never certain.

**PPA (Personal Package Archive)** — a third-party Ubuntu package
repository (e.g. deadsnakes, used to get newer Python versions); not
always reliably indexed on every release, which is exactly why this
repo's fine-tuning environment runs in Docker instead of depending on
one.

**Quantization (vector)** — compressing a float vector into fewer bits
per component (e.g. uint8 instead of float32/64) for a smaller memory
footprint, at the cost of a small, bounded approximation error.

**Regression gate** — the check that a nightly fine-tuned model
actually scores at least as well as the currently-served model,
before ever being promoted/registered — using the same harness as the
full candidate-selection pipeline, just two candidates instead of many.

**Reward** — the single number every bandit tries to maximize;
computed differently per layer (see `README_RL_ALGORITHMS.md`'s table)
but always representing "was this choice actually good."

**RL (Reinforcement Learning)** — learning what to do by trying things
and observing a reward signal, rather than being told the right answer
directly; every optimizer in this repo is a bandit-shaped special case
of it, except the nightly PEFT loop, which is closer to full RL's
weight-updating spirit (via a reward-weighted loss, not true policy
gradient).

**Shapley value** — the cooperative-game-theory solution (Lloyd
Shapley, 1953) for fairly splitting one shared outcome's credit across
however many participants contributed to it, based on each one's
average marginal contribution across every possible arrival order.

**SLM (Small Language Model)** — a language model small enough to run
acceptably on CPU (e.g. Qwen2.5 1.5B/3B here), as opposed to a large
hosted model needing real GPU infrastructure.

**Thompson Sampling** — the bandit decision rule: sample a random guess
from each arm's own current belief distribution, then pick whichever
arm's sample scored highest — exploration emerges naturally from
sampling real uncertainty, no separate "explore X% of the time" knob
needed.

**Tier** (`realtime`/`delay_ok`/`bulk`) — a coarse, non-LLM
classification of how time-sensitive a request is, determining which
fallback chain of providers it's tried against.

**Tool schema** — the JSON description (name, description, argument
shape) of one callable action, given to the model so it can request
it; the full set is what the token optimizer trims for local models.

**UnsafePathError** — raised when a requested ingest path resolves
outside the allowed directory — the guard against path traversal on
the document-ingestion REST endpoint.

**Utility prior** — the hand-designed formula seeding a brand-new
bandit arm's belief before any real feedback exists, so routing isn't
arbitrary on day one.

**VectorStoreAdapter** — the one interface (Protocol) every RAG backend
implements (`add_documents`, `search`), so a new vector database is one
new class, nothing else in the RAG layer changes.
