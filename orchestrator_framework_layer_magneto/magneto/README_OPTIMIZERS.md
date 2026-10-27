# ALGO AKANADE — All Optimizers, One Document

Every optimization/RL layer in this system, each as its own section with
its own arrow diagram(s) and technique summary. For the shared math
behind all of them, see `README_RL_ALGORITHMS.md`; for the code-level
walkthrough of the bandit itself, see `README_EXPLAINED.md`.

---

## 1. Token optimizer — `llm/token_optimizer.py`

**Technique:** RL-selected tool-schema reduction, local model only.
Every other provider's request is completely untouched — this exists
because the local CPU-bound model is bottlenecked by prompt-token count,
and the full tool-call schema is a real, non-trivial chunk of every
request's prompt.

```
 TierClient.chat() reaches a provider whose name starts with "ollama"
        │
        ▼
 optimize_tools_for_provider(provider_name, messages, tools)
        │
        ▼
 required = DESTRUCTIVE ∪ every MCP tool name        ◄── guardrail floor,
        │                                                NEVER trimmed
        ▼
 optional = every other tool, ranked by keyword overlap
            with the user's message (no LLM call — cheap and fast)
        │
        ▼
 TokenOptimizer.select(): bandit picks a reduction level
    arms = "full" | "top12" | "top6" | "top3"
    context = [optional_tool_count/40, message_word_count/60]
        │
        ▼
 reduced = required ∪ optional[:keep_n]
        │
        ▼
 provider.chat(messages, reduced)
 (Selection stashed in-memory only — never in the persisted tier_log)
        │
        ▼
 finish() — once retried/out_of_scope are known:
    reward = (1.0 if success else −0.5) + 0.3 × tokens_saved_fraction
    record_outcome(selection, success)
        │
        ▼
 UI: "~N% fewer tokens" chip (ChatReply.token_stats)
```

**Guardrail hierarchy** (the part that makes this safe, not just fast):
`tools.DESTRUCTIVE ∪ mcp_client.mcp_tool_names()` is the non-negotiable
floor. A user's ability to even ask for a destructive or external
action must never quietly disappear because the local model is slow.

---

## 2. RAG optimizer — `gmaofn/rag_optimizer.py`

**Technique:** RL picks BOTH the knowledge source AND the retrieval
algorithm per query — not a single fixed pipeline. Three real,
independent algorithms exist; which one wins is learned, not hardcoded.

### 2a. Top-level selection flow

```
 user message
        │
        ▼
 is_rag_relevant(message)? ── no ──▶ skip retrieval entirely
        │ yes
        ▼
 refine_query_for_intent(message)  — strip filler ("could you please
        │                             tell me about...") to the keyword core
        ▼
 RagOptimizer.select(): bandit over arms "{store}::{algorithm}"
    stores:      pm-index (Chroma, real PM data) | default | any
                 doc-ingested store (see §3 below)
    algorithms:  vector | lexical_rerank | hybrid   (§2b-2d below)
    context = [candidate_arm_count/10, query_word_count/30]
        │
        ▼
 store.search(query, k×3)  — wide net
        │
        ▼
 apply the CHOSEN algorithm, truncate to k
        │
        ▼
 build_context_with_included_hits() — the one source of truth for
    "used" vs "considered but truncated by the char budget"
        │
        ▼
 context string → extra system message BEFORE perceive_intent's tool
 call (never replaces or gates it — same guardrails run regardless)
        │
        ▼
 finish() — once retried/out_of_scope are known:
    n_hits == 0 → reward −0.5 (dominant, regardless of anything else)
    else        → reward = 1.0 if success else −0.2
        │
        ▼
 UI: "used N sources" chip, hover for titles (ChatReply.sources_used)
```

### 2b. Algorithm — `vector`

```
 query ──▶ embed ──▶ cosine similarity vs every stored chunk's vector
        ──▶ take top k, AS THE STORE RETURNED THEM
```
Pure embedding-similarity ranking — no re-ranking pass at all. Best
when the store's own vector search is already well-calibrated for the
query family.

### 2c. Algorithm — `lexical_rerank`

```
 query ──▶ store.search(k×3)  (wide candidate pool, vector-ranked)
        ──▶ re-rank ALL candidates by real keyword/Jaccard overlap
            with the query — ignores the original vector score entirely
        ──▶ take top k
```
Catches the failure mode pure vector similarity misses: a chunk that's
an exact keyword match for the query but scored lower on embedding
similarity alone (a real, common RAG failure mode).

### 2d. Algorithm — `hybrid`

```
 query ──▶ store.search(k×3)
        ──▶ for each candidate:
              blended_score = 0.5 × (vector_score / max_vector_score)
                            + 0.5 × lexical_overlap(query, chunk)
        ──▶ sort by blended_score, take top k
```
Combines both signals instead of picking one — the algorithm most
likely to win once the bandit has enough data for a query family where
neither pure-vector nor pure-lexical is clearly superior.

### 2e. Retrieval/answer-quality fixes (from a real empirical audit)

Four real bugs found by actually constructing test corpora and checking
whether retrieval returned the RIGHT chunks, not just "some" chunks:

- **HashEmbedder false positives** (`app/retrieval/vectorstore.py`) —
  the fallback (no-`sentence-transformers`) embedder had no stopword
  filtering: two queries about completely different topics that shared
  only common function words ("how", "do", "my", "is") scored as
  similar as genuinely related ones. Confirmed empirically: "how do I
  delete my account" ranked a password-reset chunk as its ONLY non-zero
  match, purely from stopword overlap, with zero real topical content
  shared. Fixed by filtering a standard stopword list before hashing —
  still a bag-of-words scheme (no synonym understanding), but no longer
  fooled by shared filler words alone.
- **Chunk overlap corrupting text mid-word** (`rag.py`'s `chunk_text`) —
  `current[-overlap:]` sliced raw characters, sometimes landing mid-word
  (a chunk starting `"r request a new one"` instead of `"...or request a
  new one"`). That garbled fragment got embedded AND shown to the LLM as
  real text. Fixed by snapping the overlap tail to the next word
  boundary.
- **build_context dropping ALL context over one oversized hit**
  (`build_context_with_included_hits`) — the truncation loop used
  `break` the moment any hit didn't fit the remaining char budget: if
  the FIRST (highest-scored) hit alone exceeded `max_chars`, the whole
  turn got ZERO context, even with perfectly good smaller hits right
  behind it. Fixed by `continue`-ing past the oversized hit instead.
- **`rerank()` docstring/implementation mismatch** — claimed a "lexical
  overlap boost" but was a plain re-sort by already-sorted score (a
  no-op) for its one real caller, `POST /api/v1/rag/query`. Now shares
  this file's `lexical_overlap()`/`words()` helpers (imported, not
  duplicated) and does real query-aware reranking when `query` is
  passed — matching what `lexical_rerank`/`hybrid` above already did
  correctly.

**Known remaining gap, not yet fixed:** the bandit's reward signal
(`record_outcome()`) only measures whether the TURN went fine
(retried/out_of_scope), never whether the retrieved chunks were
actually topically correct — a store/algorithm combination could return
confidently-wrong context and still get rewarded `1.0` if the LLM
happened to produce an acceptable answer anyway. Downstream of the
embedder fix above (bad retrieval quality was the bigger, more direct
cause), but a real structural gap worth closing with a genuine
relevance signal later.

---

## 3. Document ingestion → RAG-backed summaries/prompts — `gmaofn/document_pipeline.py`

**Technique:** feeds the RAG optimizer above with a document you point
it at, then makes ONE real LLM call to turn the retrieved excerpts into
a grounded summary or follow-up questions.

```
 folder or file (under RAG_INGEST_DIR, path-traversal guarded)
        │
        ▼
 ingest_path(): walk recursively, chunk every .txt/.md file,
    index into a FRESH dedicated store, register into rag_registry
        │
        ├──▶ immediately queryable by the LIVE PM chat too (§2 above
        │    checks every registered store)
        ▼
 summarize_document(store) / suggest_prompts(store)
        │
        ▼
 retrieve_with_optimizer(store_names=[store])
    — bandit picks the best ALGORITHM for this one store (scoped to a
      single store: no "which store" cold-start cost for content you
      just ingested on purpose)
        │
        ▼
 real LLM call, "bulk" tier (batch/background workload, not live chat)
        │
        ▼
 {"summary"|"prompts", "sources_used", "provider"}
```

---

## 4. Hybrid optimizer — `llm/hybrid_router.py`

**Technique:** a THIRD selectable mode in the model picker (alongside
"Auto" and picking one specific provider) — RL decides, per message,
whether to prefer the free local SLM or a hosted model.

```
 user selects "Hybrid" ──▶ ChatMessageIn.provider = "hybrid"
        │
        ▼
 agent's handle_message() intercepts the LITERAL "hybrid" sentinel
 BEFORE calling get_llm_client() — any other value (None, or a real
 provider name) passes through completely unchanged
        │
        ▼
 hybrid_router.choose(message, tier)
    context = [word_count/50, clause_count/4, tier_is_delay_ok]
    (cheap, non-LLM complexity estimate — no extra round trip)
        │
   ┌─────┴──────────────────────────┐
   ▼                                 ▼
 arm = "local"                  arm = "hosted"
 preferred = "ollama"           preferred = None
 (free, fast for simple asks)   (don't override — normal chain
                                  already tries hosted rungs first)
   └─────────────┬──────────────────┘
                ▼
    get_llm_client(tier, preferred=...)
    — from here on IDENTICAL to picking a specific model: a bad
      pick still degrades gracefully through the normal fallback
      chain right behind it, never a hard failure
                │
                ▼
    finish() — once retried/out_of_scope are known:
       local success  → +1.0        hosted success → +0.5
       local failure   → −0.6        hosted failure → −0.3
       (a local failure is penalized harder — it wastes a whole
        turn a hosted model would likely have gotten right first try)
```

**Technique note on the asymmetric reward:** this is deliberately NOT a
simple "did it succeed" bandit. Success on the free path is worth more
than success on the paid path, and failure on the free path costs more
than failure on the paid path — the reward function itself encodes the
actual business tradeoff (cost vs. reliability), not just task success.

**Where it shows up in the UI:** web's chat model dropdown and the
mobile app's model-picker menu both list "Hybrid" right next to "Auto."

**Escalation on slowness:** the "local" arm doesn't get the local
provider's own default timeout — it gets `LOCAL_TIMEOUT_SECONDS` (10s,
`llm/hybrid_router.py`) passed as `TierClient.preferred_timeout`. If the
SLM hasn't answered within that window, the SAME fallback-on-exception
path a hard error already takes kicks in and the next (hosted) rung
answers instead — a user who happened to get an unusually slow local
turn is never stuck waiting it out just because the bandit guessed
"local" was worth trying. When that happens, the hosted model's answer
is logged as an EXTRA nightly-fine-tune example
(`finetune_log.log_hybrid_escalation_turn()`) — a real, worked example
of what the local SLM should have been able to answer itself, distilled
from the model that actually did. See §9 below for the same
context-limit-triggered escalation applied to EVERY tier, not just
Hybrid mode's local arm.

---

## 5. Any other optimizer in this system

Three more RL layers exist, covered in full (math + diagrams) in
`README_ARROW_DIAGRAMS.md` §3-4 and `README_RL_ALGORITHMS.md` §2-4 —
summarized here for completeness:

| Layer | File | One-line technique |
|---|---|---|
| Agent routing | `gmaofn/router.py` | Bandit over registered agents, cold-started from a hand-designed utility prior, corrected by Shapley-attributed task reward |
| Framework picker | `agent/framework_stats.py` | Bandit over native/langgraph/adk — three *identical-behavior* implementations, picked purely on measured latency/reliability |
| Generation params | `llm/optimization_router.py` | Bandit over `balanced`/`low_latency`/`high_quality` sampling-parameter presets, per provider, on every single model call |
| **PEFT hyperparameters** | `scripts/peft_hyperparam_optimizer.py` | Bandit over `conservative`/`balanced`/`aggressive` LoRA rank+alpha+reward-threshold presets for the NIGHTLY fine-tune, rewarded by `regression_gate.py`'s own promote/reject margin — see `README_SLM_PIPELINE.md` §4 |

Plus the one loop that isn't a bandit at all: the **nightly PEFT
fine-tune** (`scripts/finetune_local_slm.py`) actually updates the local
model's weights via reward-weighted LoRA, gated by a regression check
(`scripts/eval/regression_gate.py`) before promotion — see
`README_ARROW_DIAGRAMS.md` §9-11 for the full pipeline. Which
hyperparameters THAT fine-tune runs with is, as of the row above, no
longer fixed either — the one remaining "not a bandit" piece is the
actual gradient-descent training step itself, not any decision around it.

---

## 6. Vector storage + retrieval optimization — `gmaofn/vector_storage_optimizer.py`

**Not** an RL/bandit layer like §1-4 above — this one is deterministic
math, gated purely by corpus size, under `InMemoryVectorStoreAdapter`:

**Technique — IVF clustering (retrieval speed):** brute-force search
is O(n·d); once a store crosses `IVF_MIN_VECTORS` (50), Lloyd's k-means
partitions it into K = round(sqrt(n)) clusters, and a query only scans
the n_probe nearest clusters' members instead of every stored vector —
O(K·d + (n/K)·n_probe·d), minimized at K=sqrt(n). Proven exact-equivalent
to brute force when probing exhaustively (n_probe=K); the fast path
(n_probe=K/4) trades a little recall for the speedup.

```
 kmeans assignment:  cluster(x_i) = argmin_k ||x_i - μ_k||²
 kmeans update:      μ_k = (1/|C_k|) Σ_{x_i ∈ C_k} x_i
```

**Technique — scalar quantization (storage):** asymmetric per-vector
uint8 quantization —

```
scale = (max(v) - min(v)) / 255
q_i   = round((v_i - min(v)) / scale)     # 1 byte vs 4-8 for float
```

a real 4-8x memory reduction per vector, error bounded by `scale/2`.
Built and tested as a standalone utility; not yet wired into the
adapter's default storage path (changes the accuracy/memory tradeoff,
so it stays opt-in rather than silently applied everywhere).

---

## 7. Query refinement layer — `gmaofn/query_refinement.py`

**Technique:** a THIRD RL/embedding mechanism, running BEFORE every
other layer above — `routers/chat.py` calls it once, above all three
agent tracks, so native/langgraph/adk all see the already-refined text.
Rewrites the user's raw message for grammar/clarity while strictly
preserving its actual meaning, constraints, and ambiguity — and, unlike
a static prompt, gets cheaper to run the longer the process is alive.

```
 raw message
        │
        ▼
 LinearBanditCore gate: "refine" vs "skip"
    context = [word_count/30, filler_phrase_count/3, missing_punctuation]
        │
   ┌────┴─────────────────────────────┐
   ▼ skip                             ▼ refine
 pass through unchanged      RefinementMemory.lookup(raw)
 (zero LLM cost)                      │
                              ┌───────┴────────────────┐
                              ▼ hit (sim ≥ 0.93)        ▼ miss
                        reuse cached refined text   ONE LLM call →
                        (zero LLM cost — this IS    {refined, missing_
                        "refine as it goes": the    clarifications,
                        cache grows with use)        suggested_queries}
                                                      │
                                                      ▼
                                        embedding_similarity(raw, refined)
                                        — cosine via HashEmbedder, NOT
                                          a second LLM call
                                                      │
                                        ┌─────────────┴──────────────┐
                                        ▼ sim < 0.6                  ▼ sim ≥ 0.6
                                  revert to raw               keep refined text
                                  (never remembered)           RefinementMemory.remember()
                                                                (quantized embedding stored)
        │
        ▼
 finish() — once out_of_scope is known:
    skip:          1.0 / −0.6      memory hit: 1.0 / −0.1
    llm error:     −0.5            no-op refine (text unchanged): −0.3
    real refine:   1.0 / −0.2
        │
        ▼
 UI: ChatReply.refinement chip — original/refined text, missing
 clarifications, suggested rephrasings; never blocks the turn
```

**Why two mechanisms, not one:** the bandit gate learns whether refining
is worth it AT ALL for a given kind of message (a policy decision);
`RefinementMemory` separately learns WHICH SPECIFIC queries already have
a known-good refinement (a cache). The gate converges on "is this class
of message worth an LLM call"; the memory converges on "have I already
paid for this exact answer" — together they mean the layer's real LLM
cost on a repetitive workload trends toward zero over the life of the
process, not just toward whatever the gate alone would settle on.

**Embeddings, optimized (not just used):** both `RefinementMemory`'s
cache and `DriftAuditLog`'s audit trail store embeddings via
`vector_storage_optimizer.py`'s scalar quantization (§6 above) — the
same real ~4x memory reduction, reused here rather than re-implemented.
The drift check itself costs nothing extra regardless of which model
tier is running, since `HashEmbedder` is the same zero-dependency,
zero-LLM-call embedder `rag.py` falls back to for retrieval — refinement
never adds a model round trip an SLM tier would feel.

**Diagnostics:** `GET /api/chat/refinement-audit` (Admin-only) lists
every drifted case since the process started — a transparency trail for
"what almost went wrong," not a hidden internal choice.

---

## 8. Agent-routing memory — `gmaofn/task_memory.py`

**Technique:** closes a real gap §5's agent-routing bandit had on its
own — `ContextualThompsonRouter` learned purely from ENGINEERED
features (`TaskContext.as_vector()`: complexity, context size, latency
budget, ...), with no sense of "have I seen a request LIKE this one
before." `TaskMemory` adds that, the same way `query_refinement.py`'s
`RefinementMemory` (§7) already does for message refinement: quantized
embeddings of each task's raw text, paired with which agent won and how
well it did, looked up by cosine similarity on the next similar request.

```
 new task arrives (task.text, TaskContext)
        │
        ▼
 for each candidate agent:
    seed = utility_prior(agent, context)           — §5's cold-start prior
         + task_memory.bonus_for(task.text, agent.agent_id)
           — cosine sim ≥ 0.75 vs past (task_text, agent, reward) triples,
             reward-weighted (a past WIN nudges up, a past FAILURE nudges
             down), capped at ±0.5 — never enough alone to override a
             real bandit posterior once one exists
        │
        ▼
 LinearBanditCore.select() — Thompson Sampling still runs its own
 exploration on top of this warm-started seed, same as always
        │
        ▼
 task completes → FeedbackLoop.close() → router.update(..., task_text)
    — updates the bandit's numeric posterior AND remembers
      (task.text, winning_agent, reward) in TaskMemory for next time
```

**Why a warm-start bonus, not a bypass:** unlike `RefinementMemory`'s
cache (a near-duplicate query reuses the exact past answer, skipping the
LLM entirely), agent selection isn't a "do the same thing again"
decision — a close match should make an already-good candidate MORE
likely to win this round, not force a specific winner regardless of
what the bandit itself has since learned. Same reward-weighted logic as
`utility_prior()`, extended from "brand-new agent" to "brand-new KIND of
request this agent hasn't specifically handled yet."

---

## 9. Token usage + context-limit escalation — `llm/router.py`, `llm/providers.py`

**Not** a bandit layer — real signal every OpenAI-compatible/Anthropic/
HuggingFace provider already returns in its response, previously
discarded. Two pieces:

**Real token usage, every turn:** each provider's response usage field
(`usage.prompt_tokens`/`completion_tokens` for OpenAI-shape,
`usage.input_tokens`/`output_tokens` for Anthropic) is normalized into
`LLMResult.usage` and surfaced as `ChatReply.usage` — distinct from
`token_stats` (§1's LOCAL tool-schema-trim ESTIMATE), this is a real
count from whichever provider actually answered, local or hosted.

**Context-limit escalation:** a provider's own `finish_reason=="length"`
(OpenAI-shape) or `stop_reason=="max_tokens"` (Anthropic) means the
reply was cut off by hitting the model's context/output limit — a
"successful" HTTP call that still isn't a complete answer.

```
 provider.chat() returns a result with truncated=True
        │
        ▼
 more rungs left in the chain?
    │ yes                              │ no
    ▼                                  ▼
 log it, keep the SAME messages,   return the truncated result anyway
 try the next (bigger) provider    (best available answer, nothing
 — same fallback path a hard       left to try)
 error already takes, so
 conversation context is
 preserved automatically
        │
        ▼
 ChatReply.context_limit_switch = true → UI: "switched model
 (context limit)" — explains WHY the model changed mid-turn
```

---

## Quick-reference: which file to read for what

| Question | File |
|---|---|
| "What does this optimizer actually DO, step by step?" | this document |
| "What's the exact reward formula / context vector?" | `README_RL_ALGORITHMS.md` |
| "How does the bandit itself work, in code?" | `README_EXPLAINED.md` |
| "How does this fit into the REST of the system?" | `README_ARROW_DIAGRAMS.md` |
| "What changed and why, in plain prose?" | `README.md` (main) |
