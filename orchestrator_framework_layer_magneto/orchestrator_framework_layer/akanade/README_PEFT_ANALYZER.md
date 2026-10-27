# PEFT Analyzer — scraping real training data, and RL-weighted routing

Two things, shipped together because they close the same gap: the
nightly fine-tune pipeline had no real training data on a fresh
deployment, and "Auto" routing never got a chance to prefer a
fine-tuned local SLM even when one existed and was winning.

## 1. The scraper (`app/agent/peft_analyzer.py`)

**Why**: live testing found the nightly fine-tune (`deploy/nightly_
finetune.sh`) genuinely had nothing to train on — no chat traffic had
accumulated reward-scored turns yet. Waiting for organic usage to
accumulate is fine long-term, but a fresh deployment (or a low-traffic
domain) needs *something* to learn from tonight.

**What it does**: for every enabled finetune-config's own
`scrape_topics` (a list an Admin sets — see below, empty by default),
fetches that topic's summary from Wikipedia's free, keyless REST API
(`https://en.wikipedia.org/api/rest_v1/page/summary/<title>`), and logs
it as a real (`"What is <topic>?"`, verified-answer) ground-truth pair
via the existing `app/agent/finetune_log.py:log_ground_truth_turn()` —
the exact same JSONL file `scripts/finetune_local_slm.py` already
trains from. No new training format, no new data pipeline — this only
adds a new *source* of input to a pipeline that already existed.

**Storage**: raw scrape responses are kept locally, purely for
inspection/audit, under `PEFT_ANALYZER_BLOB_DIR` (env var). **On the
worker-node server this must be set to a path under `/home`** —
confirmed `/dev/sda5`, `667G` total / `42G` used / `591G` available —
not the default (which falls back to `settings().DATA_DIR`, wherever
that is on a given box). Capped at `PEFT_ANALYZER_MAX_BYTES` (default
40GB), oldest file evicted first once the cap would be exceeded by the
next write — same "capped, not unbounded" convention the rest of this
codebase already uses (chat log, metrics snapshots).

**Configuring what to scrape**: `PUT /api/admin/finetune-config/<key>`
(Executive role) now accepts a `scrape_topics: string[]` field, e.g.:

```json
{"domain": "trading", "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
 "ollama_base_tag": "qwen2.5:1.5b",
 "scrape_topics": ["Moving average", "Algorithmic trading", "Backtesting"]}
```

An empty list (the default) means "scrape nothing for this config" —
existing deployments are unaffected until an Admin actually adds
topics.

**When it runs**: `deploy/nightly_finetune.sh` runs it once, via
`docker exec ak-chatbot python -m app.agent.peft_analyzer`, *before*
the per-domain fine-tune loop — so freshly scraped rows are available
to that same night's training. Best-effort: a scrape failure (bad
topic, network hiccup) never blocks the fine-tune step that follows.

**Idempotent per topic per day** — re-running doesn't duplicate blobs
or training rows for a topic already scraped today.

## 2. Persisted, dashboard-visible RL routing

**Why**: `app/llm/domain_router.py` already RL-selects among registered
fine-tuned domain SLMs — but only when a caller explicitly sends
`provider: "domain"`, its learned state reset on every restart, and
nothing exposed *which* domain/provider was actually winning.

**What changed**:
- `domain_router.py` now persists its bandit across restarts
  (`app/framework/bandit_persistence.py`, the same mechanism
  `rag_optimizer.py` already uses) and gained a `stats()` method
  reporting per-domain `calls`/`wins`/`win_rate`/`posterior_mean_weights`.
- `app/llm/provider_optimizer.py` (already persisted) gained the same
  `stats()` view, per `(tier, provider)`.
- **"Auto" now also benefits**: `native_agent.py`/`langgraph_agent.py`/
  `adk_agent.py` consult the domain bandit on a *plain* auto turn too
  (previously only the explicit `"domain"` sentinel did) — a
  genuinely-winning fine-tuned SLM gets a real chance to answer even
  when nobody asked for Domain mode by name. A cold/untrained domain
  naturally loses to the general arm's own seeded prior, so this is a
  no-op until real evidence justifies the switch. An explicit provider
  pick (a user's own model choice, or Hybrid mode's "hosted" result)
  still always wins outright — this is a strict addition, never an
  override.
- New `GET /api/v1/metrics/routing` (Admin role):
  `{"domain_routing": [...], "provider_routing": [...]}` — the
  "who's winning for what" view. Same per-arm shape
  `GET /api/v1/rag/metrics` already established for retrieval.

## Verification

Backend: `.venv/bin/python3 -m pytest tests/test_peft_analyzer.py
tests/test_domain_router.py tests/test_provider_optimizer.py
tests/test_metrics_history_and_rag_metrics.py -q`.

Live: send any chat message with no explicit `provider` set, then
`GET /api/v1/metrics/routing` as an Admin — `provider_routing` should
show real call counts growing.
