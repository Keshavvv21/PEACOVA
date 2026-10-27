# ALGO AKANADE Architecture — layers, algorithms, and the math behind them

This document goes one level deeper than `README.md`'s tour: every layer
ALGO AKANADE is built from, the exact per-task algorithm, and the formulas
behind adaptive routing (Thompson Sampling), fair multi-agent credit
assignment (the Shapley value), the reward/feedback loop, and how the
pre-existing PM chatbot's own framework picker (native/langgraph/adk)
composes with ALGO AKANADE's agent-level router instead of being replaced by
it. Plain-text arrow diagrams throughout — no rendering step needed to
read this.

## 1. The full stack, top to bottom

ALGO AKANADE adds THREE layers above what already existed (Adaptive Router,
Agent Network, Workflow Engine) and reframes the PM chatbot's own
machinery (framework picker, model tier chain, session memory) as what
sits *underneath* one specific registered agent (`pm-agent`), not the
whole system:

```
                              ┌───────────────────────────────────┐
   Application / User  ─────►│ POST /api/v1/orchestration/execute│
                              └────────────────┬──────────────────┘
                                               ▼
                              ┌───────────────────────────────────┐
                              │  TASK UNDERSTANDING (task.py)      │
                              │  text + TaskContext:                │
                              │  task_type, complexity,             │
                              │  required_capabilities, budget,     │
                              │  latency_requirement, requires_rag  │
                              └────────────────┬──────────────────┘
                                               ▼
                    ┌──────────────────────────────────────────────┐
                    │   CAPABILITY PRE-FILTER (orchestrator.py)      │
                    │   candidates := agents whose capabilities       │
                    │   overlap task.required_capabilities             │
                    │   (falls back to ALL agents if none match)        │
                    └──────────────────────┬───────────────────────┘
                                           ▼
          ┌────────────────────────────────────────────────────────────┐
          │      ADAPTIVE AGENT ASSIGNMENT ENGINE (router.py)            │
          │                                                              │
          │   utility_prior(a, T)  ──seeds──►  Thompson Sampling          │
          │   (§3's U(a|T) heuristic,           contextual bandit          │
          │    cold-start only)                 (learns from real          │
          │                                       feedback over time)      │
          │                                                              │
          │   per candidate agent: sample θ_a, score = θ_a · x(T)          │
          │   argmax(score)  ──►  SELECTED AGENT (the "lead")               │
          └──────────────────────────────┬───────────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │   AGENT NETWORK (network.py)                   │
                    │   lead missing a required capability?           │
                    │     yes ──► capability_neighbors() finds workers  │
                    │     no  ──► lead runs alone                        │
                    └───────────────────┬─────────────────────────┘
                                        ▼
              ┌───────────────────────────────────────────────────────┐
              │           WORKFLOW ENGINE (workflow.py)                  │
              │  SequentialStep │ ParallelStep │ ConsensusStep │           │
              │  FallbackStep │ RetryStep │ HierarchicalStep                │
              │  (which shape runs is DECIDED above, not hard-coded)         │
              └───────────────────────────┬───────────────────────────────┘
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                       AGENT EXECUTION                                  │
   │                                                                        │
   │   pm-agent ──────────► (wraps the ENTIRE existing PM chatbot,            │
   │                          see §2 below — its own framework picker,        │
   │                          model tier chain, and session memory            │
   │                          all still run exactly as before)                │
   │   pm-agent-adk ───────► same PM chatbot, pinned to Google ADK's           │
   │                          Runner/LlmAgent track specifically               │
   │   openclaw-agent ─────► direct Anthropic Claude call, no PM tools          │
   │   coding-agent ───────► code generation + real ast.parse validation        │
   │   mcp-utility-agent ──► tool set is ENTIRELY MCP-sourced                   │
   │                          (agent/mcp_client.py) — proves MCP is a real     │
   │                          capability source, not PM-only                   │
   │   langchain-research-agent ► a real langchain_core Runnable + a real      │
   │                          BaseRetriever over ALGO AKANADE's own RAG store        │
   │   code-security-agent, supply-chain-agent, cloud-config-agent ──►         │
   │                          real static-pattern / OSV.dev / IaC-config       │
   │                          security scans — deterministic, never an         │
   │                          LLM guessing (see README_CYBERSECURITY.md)       │
   │   <your agent> ───────► native Python (Agent subclass) OR a               │
   │                          RemoteHttpAgent proxying to any webhook            │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │      MODEL ABSTRACTION LAYER (llm/router.py)          │
              │  Luna → Groq → DeepSeek(shadow) → Ollama → rule-based   │
              │  (realtime tier; delay_ok/bulk have their own chains)    │
              │  + claude, huggingface, custom — NOT in a chain,          │
              │    answer only when explicitly preferred, same             │
              │    graceful fallback if unconfigured                        │
              └────────────────────────┬───────────────────────────────┘
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │     RAG (rag.py)  +  TOOLS (tools.py)  +  MCP           │
              │  ChromaVectorStoreAdapter / InMemoryVectorStoreAdapter    │
              │  PM tools + JSON plugins + MCP-discovered tools             │
              └────────────────────────┬───────────────────────────────┘
                                       ▼
                              Execution result
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │   EVALUATION & FEEDBACK (feedback.py)                  │
              │   agent.evaluate() self-score, OR a later human          │
              │   POST /api/v1/feedback rating                             │
              └────────────────────────┬───────────────────────────────┘
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │   SHAPLEY CREDIT ATTRIBUTION (shapley.py)               │
              │   splits ONE observed reward across every agent           │
              │   that actually participated, fairly (§4 below)             │
              └────────────────────────┬───────────────────────────────┘
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │   REWARD CALCULATION (reward.py)                        │
              │   R = α·Quality + β·Success + γ·Feedback                   │
              │       − δ·Cost − ε·Latency − ζ·Failure                      │
              └────────────────────────┬───────────────────────────────┘
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │   POLICY UPDATE                                          │
              │   • each agent's Thompson Sampling posterior updated       │
              │     with ITS Shapley-attributed reward                      │
              │   • α..ζ above nudged toward whichever term actually        │
              │     tracked the outcome (AdaptiveWeights.update)              │
              └────────────────────────┬───────────────────────────────┘
                                       │
                                       └──────────────────► back into the
                                                             Adaptive Router,
                                                             improving the
                                                             NEXT task's pick
```

## 2. Where the pre-existing PM chatbot fits

Nothing about the PM domain's own machinery changed — `pm-agent` is a
thin wrapper (`framework/agents/pm_agent.py`) around the exact same
`get_agent(framework).handle_message(...)` call `POST /api/chat/message`
already used. Its own internal per-turn algorithm — unchanged — still
runs entirely INSIDE the "AGENT EXECUTION → pm-agent" box above:

```
load_session(user, session_id)
  ▼
framework_stats.choose() ──► pick native | langgraph | adk   (a SEPARATE,
  ▼                                                            lower-level
classify_task_complexity(msg) → tier                          bandit than
  ▼                                                            ALGO AKANADE's own
get_llm_client(tier, preferred) → provider chain               agent router
  ▼                                                            — see §5)
perceive_intent → resolve_entities → confirm-if-destructive → execute → format
  ▼
persist_session + record framework stats
  ▼
reply ──► back up through pm_agent.respond() ──► ALGO AKANADE's own
                                                   evaluate()/feedback loop
```

**Two bandits, two different jobs, cleanly stacked:**

```
  ALGO AKANADE's AdaptiveRouter          "Which AGENT should answer this task —
  (router.py)                      pm-agent, coding-agent, openclaw-agent,
                                    or a new one someone just registered?"
         │
         ▼ (if pm-agent was picked)
  the PM domain's framework_stats  "Now that pm-agent is answering, which
  (agent/framework_stats.py)        TRACK should run it — native,
                                     langgraph, or adk?"
```

The PM domain's `native`/`langgraph`/`adk` picker never disappears or
gets subsumed — it's a strategy decision happening one level BELOW
ALGO AKANADE's own, for exactly one agent, exactly as it always has.

## 3. The Adaptive Agent Assignment Engine — Thompson Sampling

Modeled as a **linear contextual bandit** (Agrawal & Goyal, "Thompson
Sampling for Contextual Bandits with Linear Payoffs", ICML 2013). Per
agent `a`, ALGO AKANADE keeps a Bayesian linear regression posterior over
*reward given context*:

```
observe (context x, reward r) pairs over time
        │
        ▼
  B_a  =  Σ x·xᵗ  +  λI          (posterior precision matrix, λ = ridge prior)
  f_a  =  Σ r·x                  (posterior weighted-reward sum)
        │
        ▼
  μ_a  =  B_a⁻¹ f_a              (posterior MEAN weight vector)
  Σ_a  =  ν² · B_a⁻¹             (posterior COVARIANCE, ν = exploration scale)
```

Selecting an agent for a NEW task with context vector `x(T)`:

```
for each candidate agent a:
        θ_a  ~  N(μ_a, Σ_a)              ← SAMPLE a weight vector (the "Thompson" step)
        score(a)  =  θ_a · x(T)          ← predicted reward under this sample
        ▼
selected  =  argmax_a  score(a)
```

Sampling a *fresh* `θ_a` every time — rather than always using `μ_a`
directly — is what gives Thompson Sampling its exploration: an agent
with a wide, uncertain posterior occasionally wins even with a
middling mean, so it keeps collecting data instead of being starved by
one early bad run. This is the online, γ→0 (single-step) special case of
the general RL objective:

```
π*  =  argmax_π  E[ Σ_t γ^t R_t ]
```

— sequencing a whole multi-turn task (rather than one agent-pick) is the
natural next step, and is deliberately left pluggable
(`AdaptiveRouter.strategy`) for a future Q-Learning/policy-gradient/PPO
router without changing any caller.

**Cold start** — before an agent has ANY real feedback, its posterior is
seeded from a hand-designed heuristic instead of zero:

```
U(a|T)  =  w1·Q(a,T)  +  w2·R(a)  +  w3·C(a,T)  +  w4·F(a,T)  −  w5·Cost(a)  −  w6·Latency(a)
             │            │            │            │              │              │
             │            │            │            │              │              └─ agent.latency_profile
             │            │            │            │              └─ agent.cost_per_call_usd
             │            │            │            └─ reliability_score (no feedback yet, so reused)
             │            │            └─ 1.0 if required_capabilities ⊆ agent.capabilities
             │            └─ agent.reliability_score (registry metadata)
             └─ reliability_score (no per-task quality history yet, so reused)
        │
        ▼
  seed f_a so that  B_a⁻¹ f_a ≈ U(a|T)  on every feature direction
  (B_a = λI initially ⇒ B_a⁻¹ = I/λ ⇒ posterior MEAN starts at U(a|T),
   posterior VARIANCE starts at its WIDEST — easily overridden by the
   first few real outcomes)
```

**Learning step**, after every task closes:

```
observe reward r for agent a on context x
        ▼
  B_a  +=  x·xᵗ
  f_a  +=  r · x
        ▼
  (posterior mean shifts toward actually predicting r for
   contexts shaped like x; posterior narrows ⇒ less exploration
   for THIS agent on THIS kind of task going forward)
```

## 4. Fair multi-agent credit — the Shapley value

When more than one agent contributed to a single task (a
`HierarchicalStep`'s lead + workers, a `SequentialStep`'s pipeline), only
ONE reward is ever observed for the whole task. Dumping all of it on the
last agent, or splitting it evenly, both teach the router the wrong
lesson. ALGO AKANADE instead uses the **Shapley value** — Lloyd Shapley's 1953
cooperative-game-theory solution concept, for which he shared the 2012
Nobel Memorial Prize in Economic Sciences (with Alvin Roth, "for the
theory of stable allocations and the practice of market design").

Characteristic function — a coalition's value is the probability at
least one member did a good-enough job (a noisy-OR over standalone
quality scores):

```
v(S)  =  1  −  Π_{i ∈ S} (1 − quality_i)
```

Agent `i`'s Shapley value averages its marginal contribution over EVERY
possible arrival order:

```
                        |S|! · (|N|-|S|-1)!
φ_i(v)  =  Σ_{S ⊆ N\{i}} ───────────────────  ·  ( v(S ∪ {i}) − v(S) )
                               |N|!
```

```
  every subset S of "the others"
        │
        ▼
  weight(S)  =  |S|! (n-|S|-1)! / n!     ← how likely S is exactly who
        │                                  arrived before i, over ALL orderings
        ▼
  marginal(S)  =  v(S ∪ {i}) − v(S)      ← what i ADDED given S already there
        │
        ▼
  φ_i  =  Σ_S  weight(S) · marginal(S)
```

Exact for ≤8 agents (2⁸ = 256 subsets, trivial); a Monte Carlo
permutation estimator (shuffle order, accumulate each agent's marginal
as it's "added" to a running coalition, average over many shuffles) for
more, the same approximation SHAP-style tools use at scale.

**Efficiency axiom** (verified in `tests/test_gmaofn_core.py`):
`Σ_i φ_i = v(N)` — every unit of value is attributed to someone, none
invented or lost — so multiplying each `φ_i / v(N)` share by the task's
ACTUAL observed reward gives every agent its fair, attributed reward
directly:

```
attributed_reward(i)  =  total_reward  ×  φ_i / Σ_j φ_j
```

## 5. Reward, adaptive weights, and the closed loop

```
R  =  α·Quality  +  β·Success  +  γ·UserFeedback  −  δ·Cost  −  ε·Latency  −  ζ·Failure
```

`UserFeedback` defaults to the agent's own self-evaluated quality until a
real human rating arrives (`POST /api/v1/feedback`), at which point that
task's reward is RECOMPUTED with the human score standing in, and a
SECOND router update is applied — real feedback outranks self-assessment
without discarding it.

`α..ζ` adapt online (not fixed forever): after every task, each
coefficient nudges toward whichever term actually correlated with a HIGH
observed reward, and away from whichever moved opposite it — a
transparent, single-line-per-term update, not a second hidden model:

```
for each term (quality, success, user_feedback, cost, latency, failure):
        signal  =  that term's own value on THIS outcome, in [0,1]
                   (cost/latency inverted: LOW cost/latency ⇒ HIGH signal)
        ▼
        delta  =  learning_rate × reward × (signal − 0.5)
        ▼
        weight  +=  delta         (never below a small floor — always explorable)
        ▼
renormalize all six weights to sum to 1
```

**The full closed loop, end to end:**

```
Task
  ▼
Adaptive Router selects agent(s)                 (§3)
  ▼
Workflow Engine executes them                     (SequentialStep / ParallelStep / ...)
  ▼
Result  +  self-evaluated quality per agent
  ▼
Shapley credit attribution                         (§4)
  ▼
Reward calculation                                  R = α·Q + β·S + γ·F − δ·C − ε·L − ζ·Fail
  ▼
Router posterior update PER AGENT                    (its own Shapley-attributed share)
  ▼
Adaptive weight update                                  (α..ζ nudge toward what tracked reward)
  │
  └──────────────────────────────────────────────────────► improved routing on the NEXT task
                              ▲
                              │
              (optional, later)  POST /api/v1/feedback
                              │       a human rates this task
                              └── recompute reward with that rating,
                                  re-attribute via Shapley, update AGAIN
```

## 6. LangChain, RAG, and the model layer — optional, not foundational

```
   ALGO AKANADE core (task.py, agent_base.py, registry.py, router.py,
                workflow.py, network.py, feedback.py, orchestrator.py)
        │
        │   imports NOTHING LangChain-specific — deletable adapter,
        │   not a foundation (§16's vendor-neutral separation)
        ▼
   langchain_adapter.py
        │
        ├──► gmaofn_tool_to_langchain()     any ALGO AKANADE/MCP tool → a real
        │                                    langchain_core.tools.StructuredTool
        ├──► vector_store_to_retriever()    any rag.VectorStoreAdapter → a real
        │                                    langchain_core.retrievers.BaseRetriever
        └──► LangChainRunnableAgent          any LangChain Runnable/AgentExecutor/
                                              compiled LangGraph graph → a ALGO AKANADE Agent,
                                              registerable exactly like a native one
```

RAG pipeline (`rag.py`) — no vector database hard-coded:

```
Document  ──►  chunk_text()  ──►  VectorStoreAdapter.add_documents()
                                        │
                                   (Chroma-backed pm-index, or the
                                    in-memory adapter, or your own
                                    FAISS/Qdrant/Weaviate/Pinecone/
                                    pgvector class — same Protocol)
                                        ▼
query  ──►  VectorStoreAdapter.search()  ──►  rerank()  ──►  build_context()  ──►  Agent/LLM
```

Model layer (`llm/router.py` + `llm/providers.py`) — one call site
(`get_llm_client`), arbitrarily many backends, added via config alone:

```
TierClient.chat()
   │
   ├─ realtime chain: Luna → Groq → Deepseek(shadow) → Ollama → rule-based
   ├─ delay_ok chain: Max → Luna → Groq →  → rule-based
   ├─ bulk chain:     Deepseek → Groq → Ollama →  → rule-based
   │
   └─ preferred="X" (any of the above, or claude / huggingface / custom)
        tried FIRST, falls back to its tier's chain if unavailable
```

## 7. Model optimization strategies — where they actually live

**None of these six strategies run inside ALGO AKANADE's Python process** —
they're properties of the **inference server** behind whichever
model-layer provider answers (`llm/providers.py`'s
`CustomProvider`/`HuggingFaceProvider`/`OpenAICompatible`), fixed at that
server's own startup, not something a per-request HTTP call can add
after the fact. What DOES run in-process, on every single model call, is
a second, faster-cycling Thompson Sampling bandit
(`llm/optimization_router.py`'s `OptimizationRouter`, reusing
`framework/bandit.py`'s exact same math as the agent-level router) choosing
which REAL request-level parameter combination — temperature, max
tokens, top_p/top_k — to send, learned from that call's own
latency/success. Two honest, separate layers, composed:

```
  ALGO AKANADE (this process)                              Inference server (vLLM / TGI / etc.)
  ──────────────────────                              ─────────────────────────────────────
  OptimizationRouter.select(provider, tier)
        │  Thompson Sampling over a FEW real arms
        │  ("balanced" / "low_latency" / "high_quality" —
        │   llm/optimization_router.py's DEFAULT_ARMS)
        ▼
  provider.extra_body  ←  merged with the winning arm
        │  (temporarily, restored right after this ONE call —
        │   see TierClient.chat()'s try/finally)
        ▼
  CustomProvider.chat(messages, tools)
        │  HTTP POST /chat/completions  (temperature=…, top_p=…, …)
        └────────────────────────────────────────────►  request enters the server's
                                                          scheduler/batching/attention/
                                                          parallelism stack — the SIX
                                                          strategies below this line are
                                                          all fixed server config,
                                                          invisible to the HTTP contract
        ◄────────────────────────────────────────────  response
        ▼
  OptimizationRouter.update(provider, arm, tier,
                            ok=..., latency_ms=...)
        │  this call's own outcome feeds the bandit —
        └──────────────────────────────────────────► improves the NEXT call's arm choice
```

`OptimizationProfile` (`llm/optimization.py`) is the OTHER half — purely
declarative metadata an operator sets once (`CUSTOM_MODEL_OPTIMIZATION`/
`HUGGINGFACE_OPTIMIZATION` env vars, or an `EXTRA_LLM_PROVIDERS` entry's
`"optimization"` field) describing which of the six strategies THAT
server is actually deployed with — surfaced via `GET /api/v1/models`,
never consulted to decide how a request is made. Setting
`tensor_parallel_size=4` there does not itself shard anything; it
records that the server behind that URL already is sharded, for
observability.

**KV cache** — every generated token's attention key/value tensors are
cached so each new token only computes attention against the cache
instead of recomputing the whole sequence:

```
prompt tokens ──► compute K,V for each ──► CACHE
        ▼
generate token 1 ──► attend against CACHE ──► append token 1's K,V to CACHE
        ▼
generate token 2 ──► attend against CACHE (now longer) ──► append ...
        ▼
        ⋯ (cost per token stays ~constant instead of growing with sequence length)
```
vLLM's PagedAttention manages this cache in fixed-size non-contiguous
"pages" (like OS virtual memory) instead of one giant pre-allocated
buffer per request — the mechanism that also makes continuous batching
below practical at high concurrency.

**Continuous batching** — requests join and leave a running batch at the
token-generation granularity, not the request granularity:

```
classic (static) batching:            continuous batching:
┌──────┬──────┬──────┐                ┌──────┬──────┬──────┐
│ req A│ req B│ req C│  ← whole        │ req A│ req D│ req C│  ← req B finished
│  gen │  gen │  gen │    batch          │  gen │ (new)│  gen │    early, req D
│ step │ step │ step │    waits           │ step │ join!│ step │    slots in
└──────┴──────┴──────┘    for the        └──────┴──────┴──────┘    IMMEDIATELY,
   (B finishes early,     SLOWEST                                   no idle GPU
    GPU idles until       request                                   time wasted
    A and C finish)       to finish
```

**Speculative decoding** — a small, fast "draft" model proposes several
tokens ahead; the big model verifies them all in ONE forward pass instead
of one pass per token:

```
draft model proposes: [tok1, tok2, tok3, tok4]   (fast, cheap, maybe wrong)
        ▼
target model verifies ALL FOUR in one batched forward pass
        ▼
accept the longest correct PREFIX (say tok1, tok2 match; tok3 doesn't)
        ▼
keep [tok1, tok2], target model generates the real tok3 itself, repeat
        │
        └─ net effect: multiple tokens' worth of output for close to the
           cost of ONE target-model forward pass, whenever the draft
           model guesses right (draft/target agreement rate governs the
           actual speedup)
```

**Optimized attention mechanisms** — FlashAttention (fuses the
softmax(QKᵗ/√d)V computation into one GPU kernel, avoiding materializing
the full attention matrix in memory) and PagedAttention (KV cache paging,
above) are the two most widely deployed; both are transparent to the
API — same request/response shape, different GPU kernels underneath.

**Model parallelism** — splitting a model too large for one GPU/node
across several, three different ways:

```
Tensor parallelism:            Pipeline parallelism:          Data parallelism:
split EACH LAYER's             split the model's LAYERS       full model copy on
matrices across GPUs             across GPUs — GPU 1 has        EACH GPU — different
(GPU 1 has half the weight        layers 1-10, GPU 2 has         REQUESTS routed to
matrix, GPU 2 the other            layers 11-20, ...                different replicas,
half) — every layer's                token flows through            not one request
forward pass is itself             GPU 1 then GPU 2 then...          split across GPUs
split, GPUs sync per layer
```

**Combining approaches** — a production vLLM/TGI deployment typically
runs ALL of these together, not one at a time:

```
incoming requests
        ▼
continuous batching scheduler ── admits/evicts requests per token step
        ▼
PagedAttention KV cache ── shared, paged cache across the whole batch
        ▼
tensor parallelism ── each layer's matmul split across N GPUs in this node
        ▼
pipeline parallelism ── (if the model doesn't fit tensor-parallel alone)
                          different layer ranges on different nodes
        ▼
speculative decoding ── a small draft model proposes, the (already-
                          parallel, already-batched) target model verifies
        ▼
data parallelism ── multiple full replicas of the ENTIRE stack above,
                      load-balanced across requests, for raw throughput
                      beyond what one replica's GPUs can serve
        ▼
response ──► back through CustomProvider.chat() ──► ALGO AKANADE, none the wiser
              about which of the above actually ran
```

For the six SERVER-side strategies, ALGO AKANADE's own job stops at "know the
endpoint, send the request, read the response" — configuring `vllm
serve --tensor-parallel-size 4 --speculative-model <draft>` (or TGI's
equivalent flags) is a deployment decision made once, outside this
codebase, for whichever `CUSTOM_MODEL_BASE_URL`/`HUGGINGFACE_ENDPOINT_URL`
a ALGO AKANADE deployment points at. What ALGO AKANADE DOES actively decide, every
single call, on top of whatever server sits behind that URL, is which
real request-level parameter arm to send — the `OptimizationRouter`
described above.

## 8. The SLM Generation Layer — hybrid routing, token trimming, nightly self-improvement

§7's six strategies are all things ALGO AKANADE points AT but never touches —
config baked into somebody else's inference server. The **SLM
Generation Layer** (Ollama, `OLLAMA_BASE_URL`/`OLLAMA_MODEL` in
`config.py`, plus everything in `README_SLM_GENERATION_LAYER.md`) is
the one model in the whole tier chain ALGO AKANADE's own process actively
manages end to end: collecting the enterprise prompts and grounded
ground truth it should learn from (§8.3 step 0 below), whether to even
use the local model this turn, how to fit the PM tool schema into its
smaller context window, and — the one loop in this entire document that
isn't a bandit — a nightly job that rewrites its actual weights and
produces a real, standalone, branded artifact: **Algo-Akanade-SLM**.

How it connects back into everything above: the artifact this layer
produces is registered via `POST /api/admin/providers` with a `tiers`
field (§8.3 step 3f), which `llm/router.py`'s `_effective_chains()`
splices directly into the SAME realtime/bulk/delay_ok fallback chains
§1's MODEL ABSTRACTION LAYER box shows — so every one of ALGO AKANADE's 9
default agents (§1's AGENT EXECUTION box) can be routed to Algo-
ALGO AKANADE-SLM through the ordinary `get_llm_client(tier, ...)` call, with
no orchestrator-side code change and no restart. This layer only ever
produces a candidate for that shared model abstraction layer — it never
bypasses the Adaptive Router, the Workflow Engine, or Shapley credit
above it.

### 8.1 Hybrid mode — local vs. hosted, decided per message (`llm/hybrid_router.py`)

A third selectable mode in the model picker (alongside "Auto" and
picking one specific provider), both on web and in the mobile app's
model-picker menu:

```
 ChatMessageIn.provider == "hybrid" ?
        │ yes                              │ no → pass through unchanged,
        ▼                                        normal chain/preferred logic applies
 hybrid_router.choose(message, tier)
    context = [word_count/50, clause_count/4, tier_is_delay_ok]
    (cheap, non-LLM complexity estimate — no extra round trip)
    LinearBanditCore over TWO arms
        │
   ┌────┴─────────────────────────┐
   ▼                               ▼
 arm = "local"                arm = "hosted"
 preferred = "ollama"         preferred = None (normal chain already
 (free, fast for simple asks)  tries hosted rungs before Ollama)
   └─────────────┬────────────────┘
                ▼
    get_llm_client(tier, preferred=...) — identical from here to picking
    one specific provider: a bad pick still degrades through the SAME
    fallback chain right behind it, never a hard failure
                │
                ▼
    "local" arm gets LOCAL_TIMEOUT_SECONDS (10s) as preferred_timeout,
    not Ollama's own default — a slow local turn escalates to the next
    (hosted) rung on the SAME exception-fallback path a hard error takes
                │
                ▼
    finish() once retried/out_of_scope known — reward is ASYMMETRIC,
    not a plain success/fail bandit, because the actual cost/reliability
    tradeoff isn't symmetric either:
       local success  → +1.0        hosted success → +0.5
       local failure   → −0.6        hosted failure → −0.3
                │
                ▼
    an escalation (local timed out → hosted answered) is ALSO logged as
    an extra nightly-fine-tune example (finetune_log.log_hybrid_
    escalation_turn()) — a real, worked example of what the local model
    should have answered itself, distilled from whichever model did
```

### 8.1b Domain mode — choosing AMONG local SLMs, not just local-vs-hosted (`llm/domain_router.py`)

A fourth selectable mode, alongside Auto / one specific model / Hybrid:
where §8.1 decides local-vs-hosted by CHEAP COMPLEXITY features, Domain
mode decides WHICH of possibly several registered domain-specific
Algo-Akanade-SLM variants (README_SLM_GENERATION_LAYER.md §5) a message
is about, by CONTENT:

```
 domain_router.registered_domains()  ── live discovery, by naming
        │                               convention "algo-akanade-slm-<domain>"
        ▼
 arm_ids = ["general"] + every registered domain   (dynamic — a new
        │                                            domain SLM needs
        ▼                                            zero code changes)
 per arm: LinearBanditCore.score(arm, [word_count, clause_count, tier])
     + TaskMemory.bonus_for(message, arm)  ── embedding similarity to
        │                                     past (query, arm, reward)
        ▼                                     triples — folded into
 argmax → winner                              EVERY decision here, not
        │                                     just used as a one-time
        ▼                                     cold-start seed the way
 preferred = "algo-akanade-slm-<winner>"      §3's utility_prior is
        │        (or None for "general")
        ▼
 TierClient.preferred, same LOCAL_TIMEOUT_SECONDS escalation as §8.1
        │
        ▼
 record_outcome(success) → updates BOTH the bandit AND the memory
```

Why TaskMemory carries the real weight here (unlike §3's agent
selection, where it's a cold-start-only nudge on top of TaskContext's
already-structured `task_type` signal): a domain SLM has no such
structured per-arm feature available — the accumulated embedding
memory of which past queries worked well on which domain IS the
signal, so it has to inform every decision, not just the first one.

### 8.2 Trimming the tool schema to fit the local model's context (`llm/token_optimizer.py`)

Runs only when the answering provider's name starts with `"ollama"` —
hosted models get the full PM tool schema unconditionally:

```
 TierClient.chat() reaches a provider whose name starts with "ollama"
        │
        ▼
 optimize_tools_for_provider(provider_name, messages, tools)
        │
        ▼
 required = DESTRUCTIVE tools ∪ every MCP tool name    (guardrail floor —
        │                                               NEVER trimmed, any level)
        ▼
 optional = every other tool, ranked by keyword overlap
            with the user's message (cheap, no LLM call)
        │
        ▼
 TokenOptimizer.select(): LinearBanditCore picks a reduction level
    arms = "full" | "top12" | "top6" | "top3"
    context = [optional_tool_count/40, message_word_count/60]
        │
        ▼
 reduced = required ∪ optional[:keep_n]
        │
        ▼
 provider.chat(messages, reduced) — stashed in client.tool_selections
 (in-memory only, never in the persisted tier_log)
        │
        ▼
 finish(): reward = (1.0 if success else −0.5) + 0.3 × (tokens_saved / tokens_before)
        │
        ▼
 surfaced in the UI: "~N% fewer tokens" chip (ChatReply.token_stats)
```

### 8.3 Nightly PEFT fine-tune + regression gate — the model actually gets better while everyone sleeps

The one loop in this document that is not a bandit: a real,
reward-weighted LoRA update to the local model's own weights, run by
cron, gated by a regression check before it's ever allowed to answer a
real user.

```
 0. (anytime, DELIBERATE — not on the nightly clock) Collect enterprise
    prompts (a curated list[str] someone actually wrote down) ──▶
    POST /api/v1/rag/ground-truth (framework/ground_truth.py) ──▶
    optionally RAG-grounded, LLM-verified answer per prompt, reviewed,
    then logged via finetune_log.log_ground_truth_turn() — reward 1.0,
    the highest of any source, feeding the SAME JSONL step 2 reads below
        │
        ▼
 03:30 cron ──▶ nightly_finetune.sh
        │
        ▼
 1. docker cp ak-chatbot:.../finetune/ollama_turns.jsonl
        │  (no file yet? → exit 0, done — nothing to learn from tonight)
        ▼
 2. scripts/finetune_local_slm.py
    finetune_log.py's JSONL (every Ollama turn + every hybrid-escalation
    turn from §8.1 + every ground-truth turn from step 0: messages,
    response, provider,
    reward = 1.0 − 0.3·retried − 0.5·out_of_scope, or a fixed 1.0/0.9
    for ground-truth/document sources respectively)
        │
        ▼
    filter: reward ≥ 0.7, most-recent 500
        │
        ▼
    render through the TARGET model's own chat template;
    mask loss to −100 everywhere except the assistant's final turn
        │
        ▼
    LoRA adapters (r=8, alpha=16) on attention+MLP projections
        │
        ▼
    RewardWeightedTrainer.compute_loss():
       weighted_loss = Σ(per_example_loss × reward) / Σ(reward)
        │
        ▼
    train, bounded by WallClockStop (--max-minutes) AND --max-steps
        │  (fewer than min_examples at reward≥0.7? → exit 0, done)
        ▼
    model.merge_and_unload() → one plain HF model directory
        │
        ▼ merged HF model dir
 3. convert_and_register_ollama_model.sh
        │
        ├─ 3a convert_hf_to_gguf.py → f16 GGUF
        ├─ 3b llama-quantize → Q4_K_M GGUF
        ├─ 3c EXPORT: GGUF + Modelfile + metadata.json copied to
        │      exported_models/<tag>/ — a real, standalone, portable
        │      artifact OUTSIDE Ollama's own blob store, kept even if
        │      the sanity check or gate below fails, so a bad
        │      candidate stays inspectable, not just discarded
        ├─ 3d ollama create <new-tag>          (base tag untouched —
        │                                       a bad night never
        │                                       regresses production)
        ├─ 3e sanity check: answers "Say OK."?  no → stop, don't register
        ├─ 3f scripts/eval/regression_gate.py: new-tag vs base-tag
        │      (§8.4's candidate-eval harness, LOCAL ONLY, zero API cost)
        │      worse? → stop, don't register
        │      couldn't run? → warn, register anyway this one time
        └─ 3g upsert-by-name: PATCH the existing "algo-akanade-slm"
               provider's model field (or POST once, the first night) →
               new tag selectable NOW, no restart, no redeploy — this
               IS the connection back into the orchestrator's shared
               MODEL ABSTRACTION LAYER (§1), not a separate side channel.
               Every outcome (this one included) is also POSTed to
               /api/admin/slm-versions for the Admin dashboard's history.
        ▼
 4. prune out/ to the last 3 nights (exported_models/ is NOT pruned —
    it's the deliberate long-lived export archive)
```

### 8.4 The same harness, two jobs: picking a model AND gating its nightly update

`scripts/eval/`'s candidate-selection pipeline is built once and reused
twice — deciding which local model to START with, and (§8.3 step 3e)
gating whether a fine-tuned version is actually an improvement before
it ever reaches a real user:

```
 eval/test_assets/*.yaml (gold + rubric, 32 items)
        +
 eval/candidates.yaml (local Ollama tags + OpenRouter candidates,
                        hosting: local|cloud, explicit)
        │
        ▼
 harness.py — for each (candidate, item):
    deterministic (exact/regex/schema)  ──or──  llm_judge (bias guard:
                                                  never told which model answered)
        │
        ▼
 human_sample.py — ~12% sample, weighted to reasoning/planning/complex_reasoning
        │
        ▼
 aggregate.py — score matrix × eval/weights.yaml (locked in BEFORE
    scoring), latency p50/p95, $/1k requests
        │
        ▼
 decide.py — ranked top 1-2, cost/latency-adjusted, + manual
    license/vLLM-TGI checklist (deliberately not automated)
        │
        ▼
 regression_gate.py — same harness, reused NIGHTLY (§8.3 step 3e) for
    the cheap two-candidate (base-tag vs new-tag) PEFT promotion check
```

**Capacity planning, not a runtime optimizer:** `llm/memory_estimator.py`
answers "will this fit on my hardware" for both real local-compute paths
above — Ollama inference and the nightly LoRA job itself — using the
standard transformer memory-sizing formulas (full derivation in
`README_MEMORY_MANAGEMENT.md`). An operator tool run before choosing
hardware or a model size, not something in the request-time path.

**Why none of this touches the hosted rungs:** every one of §8.1-8.3 is
gated on either an explicit `provider="hybrid"` selection or a provider
name starting with `"ollama"` — a deployment with no `OLLAMA_BASE_URL`
configured never routes through any of this, and a hosted-only turn's
tool schema, timeout, and fine-tune eligibility are completely
unaffected by whether Ollama is configured at all.

## 9. PEFT Analyzer, RL-weighted routing, and the Trading Agent

Three additions on top of §8's pipeline, full detail in their own docs
(`README_PEFT_ANALYZER.md`, `README_TRADING_AGENT.md`,
`README_CODING_AGENT.md`):

- **PEFT Analyzer** (`app/agent/peft_analyzer.py`) — scrapes
  admin-configured Wikipedia topics into real ground-truth training
  pairs (via the existing `finetune_log.log_ground_truth_turn()`, no
  new data format), run by `deploy/nightly_finetune.sh` before the
  fine-tune loop. Local blobs capped (default 40GB), oldest-evicted.
- **"Auto" now benefits from domain-SLM routing too**: `llm/domain_
  router.py`'s bandit (previously only consulted on an explicit
  `provider="domain"` pick) is now ALSO consulted on a plain auto turn
  in `native_agent.py`/`langgraph_agent.py`/`adk_agent.py` — a
  genuinely-winning fine-tuned SLM gets a real chance to answer without
  anyone asking for Domain mode by name. Both `domain_router.py` and
  `llm/provider_optimizer.py` now persist (`bandit_persistence.py`) and
  expose `stats()` via `GET /api/v1/metrics/routing` — "who's winning
  for what," visible in `PeftMetricsDashboard.tsx`.
- **"Train this" from chat** (`app/agent/peft_training_tool.py`) — a
  user can say "train the trading model on this" (pasted text or a
  URL); `native_agent.resolve_arguments` asks which domain via the same
  `clarify()`/`resume_clarification` round trip every ambiguous PM
  request already uses if it wasn't named, then logs a real
  ground-truth example to that domain's own log file
  (`finetune_log.py`'s `domain=` param, matching
  `deploy/nightly_finetune.sh`'s existing per-domain `<domain>_
  turns.jsonl` convention). Actual training still only runs on the
  nightly schedule. Shared across all three chat tracks since they all
  call the same `native_agent.resolve_arguments`/`requires_confirmation`.
- **Trading Agent** (`app/agent/trading_tools.py`) — `fetch_price_
  history`/`run_backtest` (real Yahoo Finance data, a vectorized
  moving-average-crossover backtest, no lookahead bias) added as tools
  in the SAME shared tool-calling pipeline (`agent/tools.py`) the PM
  Tool already uses — reachable from web/mobile/VS Code with zero new
  endpoint or UI wiring, since none of them call the separate
  `app/framework/` multi-agent registry `POST /api/v1/orchestration/
  execute` reaches (investigation found neither web nor mobile ever
  call that endpoint — only the VS Code extension does).
- **Security Dashboard, made real**: the three static-analysis security
  agents (`code_security_agent.py`/`supply_chain_agent.py`/
  `cloud_config_agent.py`) now persist every scan (`security_findings.
  log_scan()`, including clean ones) instead of the findings existing
  only inside that one call's `AgentResult` — `GET /api/admin/security/
  findings` and `AdminSecurity.tsx`'s new "Threats & vulnerabilities
  identified" table are the dashboard view.
