# Algo-Akanade-SLM — the complete deep dive

One document, everything: what Algo-Akanade-SLM actually is, the full
pipeline that produces it, exactly what PEFT/LoRA/QLoRA are doing and
why, how it's actually used at inference time, how it differs from the
alternatives, and what's real vs. still-unverified today. Companion
docs (`README_SLM_PIPELINE.md`, `README_SLM_GENERATION_LAYER.md`,
`ARCHITECTURE.md` §8) cover pieces of this in their own context; this
is the one place that walks the whole thing start to finish.

## 1. What Algo-Akanade-SLM actually is

A small (0.6B–4B parameter) language model — Qwen, SmolLM, or Llama,
your choice of base — that:

1. **Starts** as a generic, off-the-shelf instruction-tuned checkpoint.
2. **Learns your organization's actual usage** every night, from real
   chat turns, deliberately-curated ground truth, and ingested
   documents — not a one-time fine-tune that goes stale.
3. **Gets smaller and faster**, not just smarter — 4-bit quantized
   (Q4_K_M) so it runs on CPU, no GPU required to serve it.
4. **Never ships a regression** — every candidate is gated against the
   currently-live model on a real eval harness before it's ever shown
   to a user.
5. **Is a real, exportable file** — a GGUF you can copy off the box,
   not just an opaque row in a database or a hosted API's black box.
6. **Can specialize by domain** — run several variants (legal,
   support, coding, ...) side by side, RL-routed per query.

It is the "local, continuously self-improving, verifiably-safe, your
own branded model" answer to the tier chain's other rungs (Groq,
Claude, DeepSeek, ...), which are all "someone else's model, over the
network, that never changes because of what happens in this specific
deployment."

## 2. The complete pipeline, one diagram

```
        03:30 UTC ── host crontab line, OR the no-crontab
        ak-finetune-cron container (deploy/nightly_cron_daemon.py +
        deploy/Dockerfile.finetune-cron) — either way, fires the SAME
        deploy/nightly_finetune.sh
                       │
                       ▼
        GET /api/admin/finetune-config/enabled  ── one row per domain,
        UI-editable (Admin > AI Models > "nightly config") — no rows
        yet? falls back to ONE run using this script's own env-var
        defaults, never nothing
                       │
                       ▼  (loop: once per enabled config/domain)
                  Enterprise Data                    Collect enterprise
           (real chat usage + ingested                    prompts
            CSV/PDF/TXT/MD files)                   (YOUR curated list —
                       │                              questions the business
                       │                              actually cares about)
                       │                                       │
                       │                                       ▼
                       │                          Create ground truth
                       │                     (optionally RAG-grounded, LLM-
                       │                      verified, reviewed before it's
                       │                      trusted — framework/ground_truth.py,
                       │                      POST /api/v1/rag/ground-truth)
                       │                                       │
                       └───────────────────┬───────────────────┘
                                           ▼
                                     Data Engine
                        (scripts/data_engine.py — PII scrub, cross-
                         night dedup, curation; every source above
                         funnels through the SAME finetune_log.py
                         JSONL, scored by a per-source reward: 1.0
                         organic success, 1.0 ground truth, 0.9
                         document Q&A, penalized for retries/out-
                         of-scope — see §3 below for the full formula)
                                           │
                                           ▼
                              Qwen / SmolLM / Llama, 0.6B–4B
                                     (your choice of base)
                                           │
                                           ▼
                       PEFT bandit picks LoRA hyperparameters
                    (scripts/peft_hyperparam_optimizer.py — §4 below)
                                           │
                                           ▼
                            LoRA / QLoRA fine-tune
                  (scripts/finetune_local_slm.py — reward-weighted
                   loss, §4 below; optional magnitude pruning)
                                           │
                                           ▼
                                merge_and_unload()
                          (one plain HF model directory)
                                           │
                                           ▼
                    scripts/convert_and_register_ollama_model.sh
              ┌────────────────────────────┴────────────────────────────┐
              ▼                            ▼                            ▼
        GGUF conversion              Quantize Q4_K_M              EXPORT to
        (f16, via llama.cpp)      (matches base model's own      exported_models/<tag>/
                                        quantization)          — GGUF + Modelfile +
                                                                metadata.json, a REAL
                                                                standalone artifact,
                                                                kept even on failure
                                           │
                                           ▼
                          ollama create <tag>  (base tag NEVER touched)
                                           │
                                           ▼
                    Sanity check: does it answer "Say OK." at all?
                                           │ yes
                                           ▼
              Regression gate (scripts/eval/regression_gate.py) —
              same eval harness that picked the base model in the
              first place — is this candidate BETTER than what's
              currently live?  ── no ──▶ stays in Ollama for
                                          inspection, never registered
                                           │ yes
                                           ▼
             Upsert-by-name: PATCH the existing "algo-akanade-slm[-<domain>]"
             provider's model field (or POST once, the very first night) —
             ONE row per domain, ever, deterministically pointing at the
             latest verified tag
                                           │
                                           ▼
              POST /api/admin/slm-versions (version history, success
              or failure — the Admin dashboard's "which version is
              live" table)
                                           │
                                           ▼
                llm/router.py's _effective_chains() splices the
                provider into the SAME realtime/bulk fallback chains
                every one of ALGO AKANADE's 9 default agents already calls
                through via get_llm_client(tier, ...) — no orchestrator
                code change, no restart
                                           │
                                           ▼
                    llm/domain_router.py / llm/hybrid_router.py
              (§6 below — WHICH local model, or local-vs-hosted,
               decided per message by two small RL orchestrators)
                                           │
                                           ▼
                          Real user's chat turn, answered
```

The top block (cron/scheduler → config fetch) is §7's "how tonight's
run gets triggered and configured." Everything left of "PEFT bandit" is
data collection (§3). Everything from "LoRA/QLoRA fine-tune" through
"regression gate" is training +
safety (§4-§5). Everything from "upsert-by-name" onward is how the
result becomes a real, selectable, monitored part of the running
system (§6-§7).

## 3. Where the training data actually comes from

Four sources, ALL funneling into the same `finetune_log.py` JSONL,
each with its own reward — the training script only ever reads the
`"reward"` field, never how it was computed, so adding a fifth source
later needs zero changes downstream:

| Source | Trigger | Reward | Deliberate or incidental? |
|---|---|---|---|
| Organic chat (`log_ollama_turn`) | Every real Ollama-answered turn | `1.0 − 0.3·retried − 0.5·out_of_scope` | Incidental — whatever happened to get asked |
| Hybrid escalation (`log_hybrid_escalation_turn`) | Hybrid mode's local pick timed out/failed, a hosted model answered instead | Same formula, logged under the HOSTED provider | Incidental, but a genuinely useful "what the local model should have done" example |
| Document Q&A (`log_document_turn`) | `summarize_document()`/`suggest_prompts()` grounded in an ingested CSV/PDF/TXT/MD | Fixed `0.9` | Incidental — whatever got ingested |
| **Ground truth** (`log_ground_truth_turn`) | `POST /api/v1/rag/ground-truth` — YOU hand it prompts you specifically want answered well | Fixed `1.0`, the highest of any source | **Deliberate** — this is the one source that exists because someone decided "the model needs to get THIS right," not because a user happened to ask |

Ground truth is the newest and most important piece for quality: an
admin submits a curated `list[str]` of prompts (a support team's top
tickets, a sales team's recurring objections), optionally grounded in
an already-registered RAG store, and gets back one precise LLM-
generated answer per prompt — reviewed BEFORE it's trusted, exactly
like a document summary. Only calls that actually succeeded get
logged, and deliberately from ANY provider (not gated to
`provider.startswith("ollama")` the way organic/document sources are)
— the whole point is distilling from whichever model produced the
best verified answer, often a stronger hosted tier, the same "teach
the local model from what actually worked" reasoning hybrid escalation
already established.

### 3a. How ground truth is actually created, step by step

`create_ground_truth()` (`app/framework/ground_truth.py`), called from
`POST /api/v1/rag/ground-truth`, does this for EACH prompt in the
submitted list:

1. **Optional retrieval.** If a `store_name` was given (any already-
   registered RAG store — an ingested CSV/PDF corpus, or the PM
   domain's own `"pm-index"`), it calls the SAME
   `rag_optimizer.retrieve_with_optimizer()` pipeline
   `summarize_document()`/`suggest_prompts()` already use to pull real
   excerpts for this prompt, and records the retrieval outcome back
   into `rag_optimizer` either way (§ RAG's own RL layer keeps learning
   from this call exactly like any other retrieval). No `store_name`
   given → skip straight to step 2 with no context at all.
2. **Build the prompt.** If retrieval found something:
   `"Excerpts:\n\n{context}\n\nQuestion: {prompt}"`; otherwise just
   `"Question: {prompt}"`. This, plus a fixed system instruction, is
   the ENTIRE message sent to the model:
   > "You produce a single, precise, verified reference answer to a
   > real enterprise question — this answer will be used as TRAINING
   > DATA for a smaller model, so accuracy matters more than style. If
   > excerpts are provided, answer ONLY from them and say so plainly if
   > they don't actually cover the question; never invent a fact not
   > present in the excerpts. If no excerpts are provided, answer from
   > general knowledge but be direct and unhedged — no filler, no 'as
   > an AI'."
3. **One real LLM call**, on the `"bulk"` tier (`get_llm_client("bulk")`)
   — the same tier chain everything else in this document routes
   through, so it degrades through the normal fallback chain like any
   other call, never a special code path.
4. **Success check**: `ok = bool(res.text) and not res.error`. A
   failed or empty answer is returned to the caller for visibility
   (`"ok": false`) but is NEVER logged as training data — a bad
   "ground truth" would be worse than not logging anything at all.
5. **Log, only on success, only if `log_for_training` (the default)**:
   `finetune_log.log_ground_truth_turn()` writes `(messages, reply,
   provider)` at a fixed reward of `1.0`. The LOGGED messages
   deliberately EXCLUDE step 2's own "produce verified training data"
   system instruction — that instruction only exists for THIS
   generation step and is never present at real inference time, so
   training on it would teach the model to expect a prompt it will
   never actually see. If grounded, the retrieved context IS included
   as a system message in what's logged (matching what real RAG-
   augmented inference actually looks like); if ungrounded, only the
   raw question is logged.
6. **Return, regardless of success or failure** — a
   `GroundTruthResult(prompt, answer, provider, grounded, sources_used,
   ok)` per prompt, so the caller (an admin reviewing the batch) sees
   every outcome, not just the ones that got logged.

Nothing here is auto-trusted: the whole endpoint exists to produce
REVIEWABLE candidates, and only `scripts/data_engine.py`'s later
scrub/dedup/curation pass — followed by the regression gate (§5) —
decides whether any of it actually survives into a shipped model.

`scripts/data_engine.py` sits between this combined JSONL and the
actual training run: strips PII, deduplicates against a PERSISTENT
cross-night store (`DataEngineStore` — so re-running training tomorrow
doesn't re-learn the exact same examples you already trained on
tonight), and curates the final corpus before it ever reaches the LoRA
trainer.

## 4. PEFT in depth — what's actually happening, and why not full fine-tuning

**Why PEFT (Parameter-Efficient Fine-Tuning) at all.** Full fine-tuning
a 1.5B model means updating all 1.5 billion weights, which needs
enough VRAM/RAM to hold the full model PLUS its gradients PLUS
optimizer state (Adam alone doubles the memory again) — infeasible on
a CPU-only deploy box, and massively overkill for teaching a model
"here's how OUR support tickets/PM tool calls actually look," which is
a much smaller amount of new information than the base model's entire
knowledge. PEFT freezes the base model entirely and trains a tiny set
of new parameters bolted on top.

**LoRA (Low-Rank Adaptation), concretely.** For each targeted weight
matrix `W` (attention `q_proj/k_proj/v_proj/o_proj` and MLP
`gate_proj/up_proj/down_proj` — `finetune_local_slm.py`'s
`target_modules`), LoRA doesn't touch `W` at all. It adds two small
matrices `A` (shape `d × r`) and `B` (shape `r × d`), and the effective
forward pass becomes `Wx + BAx` — `r` (the "rank") is chosen far
smaller than `d`, so `A` and `B` together have orders of magnitude
fewer parameters than `W` itself. Only `A` and `B` are trained; `W`
stays frozen and untouched. `lora_alpha` scales how much `BAx`'s
contribution counts relative to the frozen `Wx` — a bigger
`alpha/r` ratio means the adaptation has more influence over the final
output.

**QLoRA** is the same LoRA math, but the FROZEN base weights are loaded
in 4-bit (via `bitsandbytes`' `BitsAndBytesConfig`) instead of the
default fp16/bf16 — the trainable LoRA matrices themselves stay in
higher precision. This is what makes fine-tuning a bigger base model
possible on a single consumer GPU (4x less memory for the frozen
weights), but it genuinely needs a CUDA GPU (`bitsandbytes`' 4-bit
kernels don't run on CPU) — `--qlora` fails loudly with a clear message
rather than silently falling back if the box has none. The default,
CPU-friendly path skips 4-bit loading entirely and trains LoRA on top
of the full-precision frozen base.

**CPU-only means CPU-bound, not unbounded.** By default PyTorch sizes
its own intra-op thread pools (and the native BLAS libraries under it —
OpenMP/MKL/OpenBLAS) to every visible core, so an unthrottled LoRA run
on a CPU-only deploy box will genuinely pin 300-400%+ CPU in `top` —
real, expected default behavior, not a bug, but a real problem on a box
also running the live `ak-backend`/`ak-chatbot` containers overnight.
Two independent caps close this: `finetune_local_slm.py` sets
`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` and
`torch.set_num_threads()` from `FINETUNE_CPU_THREADS` (default: every
core but one) before `import torch` even runs, and
`deploy/nightly_finetune.sh`'s own `docker run --cpus="$FINETUNE_CPU_LIMIT"`
is the OS-level backstop — a hard cgroup limit Docker enforces
regardless of what the process inside thinks it's configured for, so a
misconfigured or future training path can't silently regress this.

**Reward-weighted loss** — the one piece that's specific to THIS
pipeline, not standard LoRA: a plain cross-entropy loss treats every
training example equally, but the corpus here isn't uniformly good — a
turn that needed a retry, or went out-of-scope, is real but LOWER
QUALITY data than a clean, one-shot, grounded ground-truth answer.
`RewardWeightedTrainer.compute_loss()` computes
`weighted_loss = Σ(per_example_loss × reward) / Σ(reward)` — an example
with `reward=1.0` (ground truth) pulls the model harder than one with
`reward=0.4` (a retried, borderline turn), and only examples above
`--min-reward` are included at all. Loss is masked to `-100` everywhere
except the assistant's own final turn (via each example's own chat
template), so the model is never penalized for "predicting" the
system prompt or the user's own message.

**The PEFT bandit** (`scripts/peft_hyperparam_optimizer.py`) — even the
LoRA rank/alpha and the reward-inclusion threshold aren't fixed
constants. Three arms, a genuine tradeoff (bigger rank = more capacity
to learn, but more overfitting risk on a thin/noisy corpus):

| Arm | `lora_r` | `lora_alpha` | `min_reward` |
|---|---|---|---|
| conservative | 4 | 8 | 0.8 |
| balanced (the old fixed default) | 8 | 16 | 0.7 |
| aggressive | 16 | 32 | 0.6 |

A `LinearBanditCore` (the SAME contextual-bandit primitive behind
agent routing, RAG store/algorithm selection, and every other RL
decision in this codebase — see `ARCHITECTURE.md` §3's derivation)
picks an arm from context `[curated_example_count/500, avg_reward]` —
more, cleaner data can support a more aggressive rank without
overfitting; a thin, noisy corpus is safer trained conservatively. The
regression gate's real pass/fail outcome (§5) becomes that arm's
reward, closing the loop: a rank choice that produces a model the gate
actually approves gets reinforced, one that doesn't gets suppressed.

**Optional pruning** — `--prune-sparsity` applies real global
L1-unstructured magnitude pruning (`torch.nn.utils.prune`, no new
dependency) to the MERGED model after LoRA training, zeroing out the
smallest-magnitude weights up to the requested fraction, then
`prune.remove()` makes it permanent so the zeros survive
`save_pretrained()`. Honestly documented as NOT yet exploited by the
GGUF/Q4_K_M quantizer for a smaller/faster result today — a real,
verifiable transform without a proven inference-time win through this
serving path yet, which is why it defaults off.

## 5. Safety: the regression gate, and why a bad night can't reach a user

Three independent guarantees, all real code, none of them a promise
in a docstring:

1. **The base Ollama tag is never touched or removed.** Every
   candidate gets its OWN new tag (`algo-akanade-slm[-<domain>]:<date>`)
   — a catastrophic training run just produces a tag nobody ever
   registers, while the currently-serving model keeps answering
   exactly as before.
2. **A sanity check runs first** — does the new tag answer a trivial
   "Say OK." at all? A model that doesn't even do that never reaches
   the regression gate, let alone registration.
3. **The regression gate** (`scripts/eval/regression_gate.py`) reuses
   the EXACT SAME eval harness (`eval/test_assets/*.yaml`, 32 gold +
   rubric items) that picked the base model in the first place (§8.4
   of `ARCHITECTURE.md`) to score the new tag against the current base
   tag. Worse → stays in Ollama for manual inspection, never
   registered. Couldn't run at all (setup problem) → registers
   unconditionally THIS ONE TIME with a loud warning, rather than
   silently blocking every future night until someone notices.

Every outcome — pass, fail, or "couldn't run" — is recorded, both
locally (`exported_models/<tag>/metadata.json`) and centrally
(`POST /api/admin/slm-versions`, rendered in the Admin dashboard), so
nothing about a bad night is silent.

## 6. How it's actually used at inference — two small RL orchestrators

Registering a model is necessary but not sufficient — something has
to decide WHEN to actually prefer it. Two independent, composable
mechanisms, both UI-selectable modes in the chat widget's model picker
(alongside "Auto" and picking one specific provider):

**Hybrid mode** (`llm/hybrid_router.py`) — local vs. hosted, decided
per message from CHEAP COMPLEXITY features (word count, clause count,
current tier), no extra LLM call. A short message gets a fast chance
at the free local model with a tight `LOCAL_TIMEOUT_SECONDS` (10s)
escalation to a hosted rung if it's genuinely slow this turn — reward
is asymmetric (`local success +1.0 > hosted success +0.5`, `local
failure −0.6 < hosted failure −0.3`) since the cost/reliability
tradeoff isn't symmetric either.

**Domain mode** (`llm/domain_router.py`) — a DIFFERENT axis: not
"local vs. hosted" but "WHICH of possibly several registered
domain-specific Algo-Akanade-SLM variants is this query actually
about." Domains are discovered live by naming convention
(`algo-akanade-slm-<domain>`, produced by running the SAME pipeline
above with a domain-filtered training log and `DOMAIN=<name>` set on
`convert_and_register_ollama_model.sh` — zero code changes to add a
new one). Since "which domain" is a question about CONTENT that a
word count can't answer, this reuses `framework/task_memory.py`'s
embedding-similarity memory — the same mechanism the core agent-
routing bandit uses to warm-start a brand-new agent — but as the LIVE,
every-call signal rather than a one-time cold-start seed:

```
 registered_domains()  ── live discovery: "algo-akanade-slm-legal",
        │                  "algo-akanade-slm-support", ... + "general"
        ▼
 for each arm: LinearBanditCore.score(arm, [word_count, clause_count, tier])
     + TaskMemory.bonus_for(message, arm)   ← embedding similarity to
        │                                     past (query, arm, reward)
        ▼                                     triples: THE real signal
 argmax → winner (e.g. "legal") → preferred = "algo-akanade-slm-legal"
        │
        ▼
 record_outcome(success) → updates BOTH the bandit's posterior for
                            that arm AND the embedding memory
```

Both modes degrade gracefully through the SAME `TierClient.preferred`
mechanism a manual "pick one specific model" choice already uses — a
bad RL pick this turn still falls through the normal fallback chain
right behind it, never a hard failure.

## 7. Visibility and configuration: the Admin dashboard

`POST`/`GET /api/admin/slm-versions` (`app/routers/admin_slm.py`) is a
local, inspectable version log — every nightly run, success or
failure, records its tag, base tag, sanity-check result, regression-
gate result, whether it was actually registered, and which domain (or
`null` for the general-purpose model). The Admin > AI Models page
renders this as a table underneath the provider list, grouped by
domain, so "which version is live right now, and what happened the
last N nights" is one page load, not an SSH session into the deploy
box to read `ak-finetune.log` or list `exported_models/` by hand.

Right above it, `POST`/`GET`/`PUT`/`DELETE /api/admin/finetune-config`
(`app/routers/admin_finetune_config.py`) is what SHOULD get trained
TONIGHT — one editable row per domain: any HuggingFace `base_model`
id ("choose any SLM"), the `ollama_base_tag` it's gated against, a
training-time budget, and enabled/disabled. `deploy/nightly_finetune.sh`
fetches every enabled row at the start of each run and loops over
them, so adding a second or third domain-specific model, or switching
which base model the general-purpose one trains from, is a form
submission, not an SSH session and a shell-script edit.

And the run itself no longer needs a host crontab line at all:
`deploy/nightly_cron_daemon.py` + `deploy/Dockerfile.finetune-cron`
package a tiny, dependency-free Python scheduler into its own
always-on `ak-finetune-cron` container that fires the exact same
`nightly_finetune.sh` at the configured time every day — see
`README_SLM_GENERATION_LAYER.md` §6 for the full setup, including what
this does and doesn't remove (the llama.cpp checkout/build and
`finetune.env` are still real, one-time host prerequisites; only the
"add a crontab line" step goes away).

Each config also picks a `serve_backend`: **"ollama"** (the default —
everything above, unchanged) or **"llama_server"** — skip Ollama
entirely and serve the exported GGUF directly with llama.cpp's own
OpenAI-compatible server (`deploy/Dockerfile.llama-server`), genuinely
"any SLM" since GGUF conversion is already model-family-agnostic, not
limited to whatever Ollama ships a Modelfile template for. Two
containers per domain (a stable live one, a throwaway candidate) give
this the same "old never touched until new is proven better" safety
property the Ollama tag path gets for free — see
`README_SLM_GENERATION_LAYER.md` §7 for the full mechanics, including a
real bug (a missing shared library) this surfaced and fixed, and a real
end-to-end verification run against two actual llama-server containers
serving a real GGUF.

## 8. How this differs from the alternatives

| Approach | What you get | What Algo-Akanade-SLM adds |
|---|---|---|
| Just call a hosted LLM (Groq/Claude/GPT) | Strong general answers, real per-token cost, zero domain adaptation, data leaves your infra | Free after the first fine-tune, runs entirely local, learns YOUR usage specifically |
| RAG only, no fine-tuning | Grounds answers in your documents, but the MODEL itself never gets better at your specific phrasing/tool-calling patterns | RAG still runs on top (§8 of `ARCHITECTURE.md`'s RAG+Agents layer) — this isn't instead of RAG, it's the model underneath RAG also improving |
| A one-time fine-tune, shipped once | A snapshot, frozen at however good the data was on day one, no safety net if it was actually worse | Nightly, continuous, reward-weighted, PEFT-bandit-tuned, and REGRESSION-GATED before anything ships — a bad night literally cannot reach a user |
| A generic small open-source model, used as-is | Fast and free, but has no idea about your PM tool schema, your support team's phrasing, or your domain's jargon | Same base model family, but trained nightly on YOUR real (or deliberately curated) usage |
| One fine-tuned model for everything | Reasonable if your usage is homogeneous | Domain mode lets you run several specialized variants and RL-routes each query to whichever one's embedding memory says handles that KIND of question best |

## 9. What's verified vs. what still needs the real deploy box

- **Real infrastructure, not mocks**: `serve_backend="llama_server"`'s
  underlying mechanism was run for real — two actual llama-server
  containers, an actual (tiny, public) GGUF, `regression_gate.py
  --new-base-url` scoring one against the other, a genuine PASS with a
  real per-item report. This is also where a real bug got caught and
  fixed: the first `Dockerfile.llama-server` only copied the
  `llama-server` binary, not the separate `.so` files it dynamically
  links against, so it failed to even start. See
  `README_SLM_GENERATION_LAYER.md` §7 for the full story.
- **End-to-end, across real modules, not just one file's unit tests**
  (`tests/test_slm_pipeline_e2e.py`): configure a domain via
  `PUT /api/admin/finetune-config/legal` → confirm
  `GET .../enabled` returns the exact shape
  `deploy/nightly_finetune.sh`'s own parser reads → register a
  domain provider (simulating a successful nightly run) →
  `POST /api/admin/slm-versions` → confirm the dashboard shows it live
  → confirm `domain_router.registered_domains()` discovers it for
  real (a genuine `admin_providers.py` row, no monkeypatched
  `available_providers()`) → force a deterministic Domain-mode pick
  and confirm a REAL `POST /api/chat/message` call actually reaches
  that specific provider (only the provider's own outbound HTTP call
  is faked; `native_agent.py`'s full step 1-17 flow, `TierClient`'s
  real chain-building, and `domain_router.py`'s real provider-key
  mapping all run unmocked) → confirm the outcome feeds back into the
  bandit's posterior and the embedding memory with the correct sign.
  One thing this surfaced worth knowing: a domain SLM's free-text
  answer (no PM tool call) is classified `out_of_scope=True` by
  `native_agent.py`'s existing tool-vs-chat heuristic, so its reward is
  genuinely negative — that heuristic was built for "did this produce a
  grounded PM action," not "was this a good domain answer," and Domain
  mode inherits it unmodified for now.
- **Fully test-verified** (pure Python, real unit tests in the main
  suite, no live network dependency): `data_engine.py`'s scrub/dedup/
  persistent store, `document_pipeline.py`'s CSV/PDF extraction,
  `ground_truth.py`'s grounded/ungrounded generation and logging
  (`tests/test_ground_truth.py`, 12 tests), the version-history API
  (`tests/test_admin_slm.py`, 5 tests), the nightly-config API
  (`tests/test_admin_finetune_config.py`, 6 tests), and `domain_router.py`'s
  RL orchestrator (`tests/test_domain_router.py`, 8 tests: naming-
  convention discovery, general-arm fallback, a registered domain is a
  reachable pick, the bandit AND the embedding memory both update on
  `record_outcome()`, a past failure lowers a similar future query's
  bonus, and the bandit measurably learns to prefer whichever arm keeps
  winning).
- **Docker-build-verified**: `deploy/Dockerfile.finetune-cron` builds
  clean; `python3 /app/nightly_cron_daemon.py --now` inside it
  correctly invokes `nightly_finetune.sh`, which correctly no-ops when
  `ak-chatbot`/the socket aren't reachable — not yet run as its own
  scheduled container against a real deploy box.
- **Syntax-verified, smoke-tested on a tiny model, NOT yet run end-to-
  end on the real deploy box**: the Data-Engine-integrated training
  loop, `--prune-sparsity`, `--qlora` (needs a CUDA GPU never validated
  against here at all), the export step + upsert-by-name registration
  in `convert_and_register_ollama_model.sh` (needs a real llama.cpp
  checkout + at least two real nightly runs to confirm the SECOND
  night actually PATCHes the first night's row), and the `DOMAIN` env
  var end to end (needs two actually-registered domain SLMs on a real
  deploy box to confirm Domain mode routes between them live, not just
  in monkeypatched unit tests).
- **Validated end-to-end on a real deploy box** (worker-node,
  2026-08-24): training smoke-tested; the shell script's Ollama-tag-
  creation, regression-gate, and admin-API-registration stages ran for
  real against a real llama.cpp checkout + build. GGUF conversion and
  a real multi-hour run against the actual production 1.5B/3B model
  are the two pieces still awaiting the first real nightly cron run —
  check `ak-finetune.log` after it.

## 10. Trying it locally

```bash
cd backend
pip install -r requirements.txt          # CSV/PDF ingestion works out of the box (pypdf)
python -m pytest tests/test_data_engine.py tests/test_document_pipeline.py \
  tests/test_ground_truth.py tests/test_admin_slm.py tests/test_domain_router.py \
  tests/test_admin_finetune_config.py tests/test_admin_eval_reports.py \
  tests/test_slm_pipeline_e2e.py -q

# The heavy training path — separate venv, see finetune_requirements.txt:
pip install -r scripts/finetune_requirements.txt
python scripts/finetune_local_slm.py --help

# A domain-specific variant — same script, just point --log-path at a
# domain-filtered JSONL and set DOMAIN when registering:
python scripts/finetune_local_slm.py --log-path legal_turns.jsonl --output-dir out/legal
DOMAIN=legal scripts/convert_and_register_ollama_model.sh out/legal/merged

# The lightest possible hands-on version — no Docker, no llama.cpp,
# just a LoRA adapter on a laptop GPU or a free notebook:
cd scripts/quickstart_lora_example && pip install -r requirements.txt && python train.py

# Run the nightly job as an always-on container instead of a host
# crontab line (deploy/nightly_finetune.sh's own docstring covers the
# one-time llama.cpp/finetune.env prerequisites this still needs):
docker build -t algo-kaizen-finetune-cron -f deploy/Dockerfile.finetune-cron .
sudo docker run -d --name ak-finetune-cron --network ak-net --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock -v /opt/algo-kaizen:/opt/algo-kaizen \
  algo-kaizen-finetune-cron

# Serve directly with llama.cpp instead of Ollama for a given domain —
# build the server image once, then set serve_backend="llama_server" on
# that domain's config (UI or PUT /api/admin/finetune-config/<domain>):
docker build -t algo-akanade-server -f deploy/Dockerfile.llama-server .
```
