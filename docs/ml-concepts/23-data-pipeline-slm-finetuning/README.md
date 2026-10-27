# Data Pipeline for SLM Fine-Tuning — Collection, Scraping, Curation

[08-lora-qlora](../08-lora-qlora/README.md) covers what `finetune_local_slm.py` trains *with*. This
document covers what it trains *on*: where the (prompt, response, reward) examples actually come from,
how a cold-start problem (no organic traffic yet) is solved without inventing synthetic data, and how
raw production logs get turned into a corpus safe and clean enough to bake into model weights. None of
this is a single script — it's three real, separately-runnable stages, each independently testable, that
only add up to a training set once composed together.

## Table of contents

1. [Why fine-tuning needs a data pipeline, not just a training script](#1-why-fine-tuning-needs-a-data-pipeline-not-just-a-training-script)
2. [Stage 1 — collection: four sources, four reward semantics](#2-stage-1--collection-four-sources-four-reward-semantics)
3. [The cold-start problem, and the PEFT Analyzer scraper](#3-the-cold-start-problem-and-the-peft-analyzer-scraper)
4. [Stage 2 — curation: PII/credential scrubbing](#4-stage-2--curation-piicredential-scrubbing)
5. [Stage 2 continued — near-duplicate detection](#5-stage-2-continued--near-duplicate-detection)
6. [The persistent store: an ever-growing corpus, not a nightly recompute](#6-the-persistent-store-an-ever-growing-corpus-not-a-nightly-recompute)
7. [End-to-end pipeline diagram](#7-end-to-end-pipeline-diagram)
8. [Orchestration: the nightly run order](#8-orchestration-the-nightly-run-order)
9. [How this repo actually uses it](#9-how-this-repo-actually-uses-it)
10. [Practical notes and caveats](#10-practical-notes-and-caveats)
11. [References](#11-references)

---

## 1. Why fine-tuning needs a data pipeline, not just a training script

Feeding a fine-tuning script whatever a chatbot happened to log last night has three concrete failure
modes if done naively: (1) a real user's email, phone number, or an API key typed into a chat window
gets baked permanently into model weights, unrecoverable after the fact, unlike a logging-layer
redaction that can be patched later; (2) ten near-identical copies of the single most common question
dominate the gradient over the genuinely varied examples that actually teach the model something new;
and (3) on day one — or any early day — there may simply not be enough organic traffic yet to train on
at all. A real data pipeline exists specifically to close all three gaps before a single gradient step
runs, and this repo implements one as three composable stages rather than folding ad-hoc fixes into the
training script itself.

## 2. Stage 1 — collection: four sources, four reward semantics

`app/agent/finetune_log.py` is the one place every candidate training example enters the pipeline,
regardless of source — a single dependency-free module (`json`/`pathlib`/`threading` only) safe to run
inside the always-on chatbot image, distinct from the separate, heavy-dependency training script it
feeds. Four logging functions, each called from a different real code path, each assigning a different
reward:

| Function                     | Real trigger                                                                 | Reward                                    |
|-------------------------------|-------------------------------------------------------------------------------|--------------------------------------------|
| `log_ollama_turn`             | An organic chat turn the local Ollama SLM actually answered                  | `1.0`, `−0.3` if retried, `−0.5` if out-of-scope (clamped to `[0, 1]`) |
| `log_ground_truth_turn`       | A deliberately curated (prompt, verified-answer) pair — not organic usage    | Fixed `1.0`, plus a separate `training_weight` multiplier |
| `log_document_turn`           | A grounded document-summary answer from real ingested content (CSV/PDF/etc.) | Fixed `0.9`                                |
| `log_hybrid_escalation_turn`  | Hybrid mode's local SLM was preferred but a hosted model answered instead     | Same formula as `log_ollama_turn`, logged under the hosted provider |

Reward is intentionally a simple, transparent heuristic here — not a reuse of the richer
Shapley-attributed multi-agent reward formula elsewhere in this codebase, which needs signals (a
multi-agent split, real human feedback) that don't exist on a single PM-chatbot turn today. It is a
day-one heuristic by explicit design, swappable for a richer signal later without changing anything
downstream — the training script only ever reads the `"reward"` field, never how it was computed.

The `training_weight` field on `log_ground_truth_turn` is a deliberately separate knob from reward:
reward answers "was this correct," `training_weight` answers "how much should a correct example of this
kind count relative to others of equal reward" — the concrete lever for training harder on a genuinely
hard/complex example without hand-duplicating rows in the log to get the same gradient effect. The two
are multiplied together into a single per-example loss scale inside `finetune_local_slm.py`'s
`RewardWeightedTrainer` (see [08-lora-qlora §8.1](../08-lora-qlora/README.md#81-finetune_local_slmpy--the-nightly-production-pipeline)).

`log_hybrid_escalation_turn` is worth calling out specifically: it is written under the *hosted*
provider's name even though this dataset trains the *local* model, precisely because the point of the
record is "teach the local model to answer like the bigger one just did" — a real, worked distillation
example rather than an invented one.

## 3. The cold-start problem, and the PEFT Analyzer scraper

A brand-new domain, or a system with too little organic chat traffic yet, has nothing in `finetune_log`
for `finetune_local_slm.py` to train on — and the training script itself refuses to train on too little
data (`--min-examples`, default 8) rather than overfit on scraps. Rather than fabricating synthetic
examples, `app/agent/peft_analyzer.py` solves this by fetching *real* content:

1. For every enabled fine-tune config's own admin-configured `scrape_topics` list (never a hardcoded
   topic), it calls Wikipedia's free, keyless REST summary endpoint —
   `https://en.wikipedia.org/api/rest_v1/page/summary/{title}` — the same external domain the chatbot's
   own "fetch" MCP tool already reaches in production, so this introduces no new external dependency.
2. Each successful fetch becomes one `log_ground_truth_turn` call: `(f"What is {topic}?", extract)`,
   reward `1.0` — flowing into the exact same JSONL format and downstream curation as every other
   source, never a second, divergent data format.
3. The raw scrape response is also written to a local blob file, purely for inspection/audit, under a
   directory capped at 40GB (`PEFT_ANALYZER_MAX_BYTES`) with oldest-file-first eviction run *before*
   each write — the same "capped, not unbounded" convention `CHAT_LOG_STATE_KEY` and `MetricsSnapshots`
   already use elsewhere in this codebase.
4. Idempotent per topic per day — a blob filename encodes the UTC date, so re-running the scraper
   mid-day (or a retried nightly cron) never double-scrapes the same topic or duplicates training rows.
5. A single topic's failure (unknown title, network hiccup, a disambiguation page with no extract)
   degrades that one topic to `"skipped_no_data"` — it never aborts the rest of the run or crashes the
   nightly job.

## 4. Stage 2 — curation: PII/credential scrubbing

`scripts/data_engine.py` is the real transform between the raw log and what actually gets trained on,
applied in a fixed order: scrub, then deduplicate. Scrubbing runs first specifically so that two
examples differing only in a redacted email or phone number are still recognized as near-duplicates
afterward — a raw PII substring shouldn't make two otherwise-identical examples look different to the
dedup step.

Five ordered regex patterns — more specific before more general, so a key embedded in a longer string is
caught by its own rule first — replace each match with a `[REDACTED_<KIND>]` placeholder:

```
EMAIL       \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b
API_KEY     \b(?:sk|pk|api|key|token)[-_][A-Za-z0-9]{16,}\b   (case-insensitive)
CREDIT_CARD \b(?:\d[ -]?){13,16}\b
SSN         \b\d{3}-\d{2}-\d{4}\b
PHONE       \b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b
```

Each pattern is deliberately conservative — favoring false positives over false negatives. Over-redacting
a training example costs a little signal; under-redacting bakes a real secret into model weights
permanently, and those are not the same class of mistake. This scrubbing has to happen *before* the
gradient step specifically because, unlike a display-layer redaction, a model that has already trained
on a real credential cannot be made to un-learn it after the fact.

## 5. Stage 2 continued — near-duplicate detection

Ten near-identical copies of the single most common logged question would otherwise dominate a
500-example training batch over the genuinely varied examples that actually teach the model something.
Deduplication here is a cheap, dependency-free (`re`/`hashlib`/`math` only) hashed bag-of-words cosine
similarity, deliberately reimplemented rather than importing `app/retrieval/vectorstore.py`'s
`HashEmbedder` — this script has its own isolated venv (§9) and cannot depend on the main app package.

For each example, only its most recent user turn (or the response, if there's no user turn) is embedded:

```
v[h(token) mod 256] += 1   for every non-stopword token, h = md5-based hash
v ← v / ‖v‖₂                                    (L2-normalize to unit length)
```

Two examples' similarity is then just the dot product of their two unit vectors — cosine similarity
without a separate division, since both are already normalized:

```
cosine(a, b) = a · b
```

An example is dropped as a near-duplicate if its cosine similarity to *any* already-kept example is
`≥ 0.95` (the default `dedup_threshold`). This is an `O(n²)` brute-force comparison — deliberately: at
the few-hundred-example scale this pipeline caps training batches at, brute force is simpler and cheap
enough, the identical tradeoff `vector_storage_optimizer.py` documents for its own brute-force floor
before an index becomes worth the complexity.

## 6. The persistent store: an ever-growing corpus, not a nightly recompute

`DataEngineStore` is the actual production path, not the simpler stateless `curate()` helper it's built
on top of. The distinction matters: `curate()` only deduplicates *within* one night's batch, while
`DataEngineStore.ingest()` checks every incoming example against **every example this store has ever
kept**, loaded once at construction time and updated in memory as new examples are added. A prompt
logged Monday still blocks its near-duplicate from being re-added Thursday — the corpus only ever grows
with genuinely new material, never resets. `finetune_local_slm.py` trains on `load_all()`'s full
accumulated set (capped to the most recent `--max-examples`), not a single night's raw log in isolation.

File format is one JSON object per line — the same shape as `finetune_log.py`'s own raw records, just
post-scrub — so a curated example remains inspectable with the exact same tools used on the raw log.

## 7. End-to-end pipeline diagram

```
COLLECTION (always-on chatbot image, app/agent/finetune_log.py)
 ┌────────────────┐ ┌──────────────────┐ ┌───────────────────┐ ┌────────────────────────┐
 │ log_ollama_turn │ │log_ground_truth_ │ │ log_document_turn │ │log_hybrid_escalation_  │
 │ (organic chat)  │ │turn (curated /   │ │ (real doc         │ │turn (hosted model      │
 │                 │ │ PEFT Analyzer)   │ │  ingestion)       │ │ answered instead)       │
 └────────┬────────┘ └────────┬─────────┘ └─────────┬──────────┘ └───────────┬────────────┘
          └────────────────────┴──────────┬──────────┴────────────────────────┘
                                           ▼
                              data/finetune/ollama_turns.jsonl (raw log)
                                           │
                        CURATION (scripts/data_engine.py, standalone venv)
                                           ▼
                              scrub_pii()  →  PII/credential placeholders
                                           ▼
                    DataEngineStore.ingest()  →  cosine-dedup against ALL history
                                           ▼
                     data/finetune/curated_examples.jsonl (ever-growing, durable)
                                           │
                          TRAINING (scripts/finetune_local_slm.py)
                                           ▼
                   LoRA/QLoRA fine-tune (see 08-lora-qlora, 22-peft-hyperparameter-optimization)
                                           ▼
                          merge_and_unload()  →  merged HF model directory
                                           │
                convert_and_register_ollama_model.sh  →  GGUF  →  regression_gate.py
                                           ▼
                        PASS: registered with the live chatbot (Ollama tag swap)
                        FAIL: kept in Ollama for inspection, never served
```

## 8. Orchestration: the nightly run order

`deploy/nightly_finetune.sh` runs the PEFT Analyzer scraper (`docker exec ak-chatbot python -m
app.agent.peft_analyzer`, best-effort) once, *before* the per-domain fine-tune loop — so freshly scraped
ground-truth rows for a brand-new or thin domain are already sitting in the log by the time
`finetune_local_slm.py` reads it that same night, not one night behind.

## 9. How this repo actually uses it

- **Collection**: `orchestrator_framework_layer/akanade/backend/app/agent/finetune_log.py` — all four
  logging functions and `compute_reward()`.
- **Cold-start scraping**: `orchestrator_framework_layer/akanade/backend/app/agent/peft_analyzer.py` —
  `fetch_topic_summary`, `scrape_topic`, `run()`, the 40GB cap and oldest-first eviction
  (`_evict_oldest_until_under_cap`). Runs inside the always-on chatbot image (reuses
  `log_ground_truth_turn`'s exact write path) rather than as a second standalone script/venv, since it
  needs no heavy ML dependency — only `httpx`.
- **Curation**: `orchestrator_framework_layer/akanade/backend/scripts/data_engine.py` — `scrub_pii`,
  `_PII_PATTERNS`, `_hash_embed`/`_cosine`/`deduplicate`, and the production `DataEngineStore` class.
  Deliberately dependency-free and never imports the main `app` package, since it's imported by
  `finetune_local_slm.py`, which has its own separate, heavy (`torch`/`transformers`/`peft`) venv.
- **Consumption**: `finetune_local_slm.py:365-372` — constructs a `DataEngineStore`, calls `ingest()`
  with tonight's raw log, then trains on `load_all()`, unless `--no-data-engine` is passed (a
  quick-local-test-only escape hatch, never set by the nightly cron).
- **Orchestration**: `deploy/nightly_finetune.sh` (lines around 136–145) — the PEFT Analyzer invocation,
  before the fine-tune loop.

## 10. Practical notes and caveats

- **Redaction is best-effort, not a compliance guarantee.** The regex patterns are deliberately
  conservative and will miss PII shapes they don't recognize (a non-US phone format, a company-specific
  internal ID) — this is a real risk-reduction measure, not a substitute for reviewing what actually
  flows into `finetune_log` in a genuinely regulated environment.
- **Dedup threshold (0.95) trades recall for corpus diversity.** A lower threshold drops more marginal
  near-duplicates (safer against a saturated common question) at the cost of occasionally dropping two
  examples that are actually meaningfully different but textually similar; this is a tunable, not a
  fixed constant.
- **The scraper only ever adds ground-truth-style examples, never organic chat.** It solves "there's no
  data yet," not "the existing data is bad" — Stage 2's curation is what handles data quality, and the
  two problems are solved by two different, independently testable pieces of code.
- **`O(n²)` dedup is a scale ceiling, documented as such.** Fine at the few-hundred-example corpora this
  pipeline currently targets; a much larger corpus would need an actual similarity index, the same
  tradeoff line `vector_storage_optimizer.py` already draws for its own brute-force floor.

## 11. References

- In-repo: `orchestrator_framework_layer/akanade/backend/app/agent/finetune_log.py`,
  `orchestrator_framework_layer/akanade/backend/app/agent/peft_analyzer.py`,
  `orchestrator_framework_layer/akanade/backend/scripts/data_engine.py`,
  `orchestrator_framework_layer/akanade/backend/scripts/finetune_local_slm.py`,
  `deploy/nightly_finetune.sh`.
- See also: [08-lora-qlora](../08-lora-qlora/README.md) for what the resulting corpus is trained with,
  and [22-peft-hyperparameter-optimization](../22-peft-hyperparameter-optimization/README.md) for how
  this pipeline's own `curated_count`/`avg_reward` output feeds that night's hyperparameter choice.
