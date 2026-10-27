# Future roadmap — possible additions, not yet built

Everything in this document is a PROPOSAL, grounded in what actually
exists today, not a claim about what's implemented. Every other
README in this repo documents real, tested code; this one is
deliberately the exception, so a reader never confuses "here's an idea
worth exploring" with "here's what ships." Each section says exactly
which existing file/module the idea would extend, so it's a concrete
next step, not a buzzword list.

## 1. A real enterprise data lake + Airflow-orchestrated pipeline

**What exists today**: `deploy/nightly_finetune.sh` does one
`docker cp` of a single JSONL file (`finetune_log.py`'s output) out of
the live `ak-chatbot` container, and `scripts/data_engine.py` scrubs/
dedups/curates it in one Python process, entirely in memory. This
works — it's real, tested, and it's the honest MVP shape for one small
local model — but it's a single-purpose pipe: one file, one consumer,
no schema versioning, no lineage, no way for a SECOND model (a
classifier, a different LLM, an analytics model) to reuse the same
cleaned data without re-implementing the scrub/dedup logic itself.

**What a real data lake adds**, concretely:

- **Storage**: land every event as an immutable, append-only record in
  object storage (S3, or MinIO if self-hosted) using a lakehouse table
  format — Delta Lake or Apache Iceberg — instead of a single JSONL
  file. Table formats give you ACID writes (no partial/corrupt reads
  mid-write, a real risk today with a flat file being appended to
  while cron reads it), schema evolution (add a new field to
  `finetune_log.py`'s records without breaking every old row), and time
  travel (reproduce EXACTLY what training data existed the night a
  given `algo-akanade-slm` tag was produced — real lineage, not just
  the `metadata.json` snapshot `convert_and_register_ollama_model.sh`
  already writes).
- **Medallion layering** (the standard lakehouse pattern):
  - *Bronze* — raw, unmodified dumps of every source `finetune_log.py`
    already writes (organic turns, hybrid escalations, document Q&A,
    ground truth) PLUS sources not currently captured at all
    (`security_events.py`'s flagged events, real human thumbs-up/down
    once that exists — see §4).
  - *Silver* — `data_engine.py`'s existing `scrub_pii()`/
    `deduplicate()` logic, unchanged in substance, just running as an
    Airflow task against lake tables instead of an in-process call
    against one file.
  - *Gold* — curated, reward-scored, training-ready tables, one per
    domain (mirroring `FinetuneConfig`'s own domain concept) — this is
    what `finetune_local_slm.py` would actually read from, and what
    ANY OTHER model could read from too.
- **Orchestration with Airflow** (or Dagster/Prefect — same idea,
  different tool) instead of a single bash script on a cron line:
  a real DAG — `extract_chat_logs` → `extract_security_events` →
  `extract_ground_truth` → `scrub_and_dedup` → `curate_by_domain` →
  `train_lora` (containerized, the same `algo-kaizen-finetune` image,
  just invoked as a `DockerOperator`/`KubernetesPodOperator` task
  instead of a raw `docker run` in bash) → `regression_gate` →
  `register_provider`. What this buys over the current script: retries
  with backoff per task (not "the whole night fails"), backfills (re-run
  just `scrub_and_dedup` for a past date without re-training), a real
  UI showing which stage failed and why, and alerting (Slack/email/
  PagerDuty on a failed gate) instead of only `ak-finetune.log`.
- **Data quality gates**: Great Expectations or Soda checks between
  Bronze→Silver→Gold (e.g., "reward is always in [0,1]", "no row has a
  raw email address after PII scrub") — a real, automated check on top
  of `data_engine.py`'s existing scrub logic, catching a regression IN
  the scrubber itself before it reaches training data.
- **How a company would build the lake FROM this repo specifically**:
  the shape is already right — `finetune_log.py`'s four logger
  functions are already "one function per Bronze source"; the work is
  pointing each one at a lake sink (a Kafka topic + a lake sink
  connector, or direct batched Parquet writes) instead of one shared
  JSONL file, and moving `data_engine.py`'s transform logic into
  Airflow tasks that read/write lake tables. Nothing about the actual
  scrub/dedup/curation ALGORITHM needs to change — only where its
  input and output physically live.
- **Training other models off the same lake**: once Gold tables exist
  with a stable schema (prompt, response, reward, domain, timestamp,
  source), any other model — a support-ticket classifier, a churn
  predictor, an internal search ranker — can be trained from the exact
  same enterprise data this SLM pipeline curates, without re-solving
  PII scrubbing or dedup. A shared feature store (Feast) on top would
  let multiple models share engineered features too, not just raw
  curated text.

## 2. Other RL / self-learning algorithms beyond Thompson Sampling

**What exists today**: every RL decision in this codebase — agent
routing, RAG store/algorithm choice, token trimming, provider
ordering, Hybrid mode, Domain mode, security confidence, PEFT
hyperparameters — runs through the SAME `LinearBanditCore`
(`framework/bandit.py`): Bayesian linear regression with Thompson
Sampling, one posterior per arm, over a small hand-engineered context
vector. This is a genuinely good default (real exploration/
exploitation tradeoff, cold-start-friendly via seeded priors,
lightweight enough to run on every chat turn) — but it's a SINGLE
algorithm reused everywhere, and a few of its assumptions (linear
reward in the context features, single-step decisions) don't fit every
decision equally well. Concretely:

- **UCB (Upper Confidence Bound)** — deterministic, not sampled:
  score each arm by `mean + c·sqrt(log(t)/n_arm)`. Simpler than
  Thompson Sampling (no posterior to sample from), and in stationary
  settings often converges with fewer wasted exploratory pulls. Worth
  A/B-testing against `provider_optimizer.py`'s ordering specifically,
  since provider latency/success rates are close to stationary
  (a provider doesn't usually get randomly better or worse turn to
  turn the way "which domain is this query about" might).
- **Epsilon-greedy (with decay)** — the simplest possible real
  baseline. Useful less as a production replacement and more as a
  CONTROL arm: running it side by side with the real Thompson Sampling
  bandit on a held-out slice of traffic would give a real, measured
  answer to "is the extra complexity of Bayesian linear regression
  actually buying anything over the dumbest possible thing that still
  explores."
- **Q-Learning / a small DQN** — for MULTI-STEP decisions, which
  nothing here currently does. Every existing bandit is single-shot:
  one context in, one arm out, one reward back. The Workflow Engine
  (`workflow.py`, `SequentialStep`/`ParallelStep`/`ConsensusStep`/
  `FallbackStep`/`RetryStep`/`HierarchicalStep`) already has the right
  SHAPE for a genuinely sequential decision (which step type to use,
  then what happens next depends on how that step went) — a tabular or
  small-network Q-learner could learn "for this kind of task, this
  workflow SHAPE" the way `router.py`'s bandit learns "for this kind of
  task, this AGENT," with credit properly assigned across the shape's
  own steps rather than only across agents within one step.
- **PPO (Proximal Policy Optimization)** — for a CONTINUOUS action
  space, which today gets approximated by picking from a small fixed
  set of presets (`optimization_router.py`'s "optimization arms" —
  fixed temperature/top_p bundles, `peft_hyperparam_optimizer.py`'s 3
  fixed LoRA-rank arms). A policy-gradient method could output a real-
  valued generation-parameter vector directly instead of choosing among
  a handful of pre-baked bundles — a genuinely bigger design change
  (needs a differentiable or at least a black-box-optimizable objective,
  more training data per decision point than a bandit needs), but
  removes the "did we pick the right small set of presets" question
  entirely.
- **Neural contextual bandits (NeuralLinear / NeuralUCB)** — swap
  `LinearBanditCore`'s linear reward model for a small neural net
  feeding into the same Bayesian-linear-on-top-of-features trick
  (NeuralLinear specifically: a shared neural feature extractor, linear
  Bayesian regression on ITS output) — worth considering if a future
  context vector stops being a handful of hand-picked numbers (like
  `[word_count/50, clause_count/4, tier_is_delay_ok]`) and starts being
  a real embedding (e.g., feeding `domain_router.py`'s query embedding
  directly into the bandit instead of only through `TaskMemory`'s
  separate similarity lookup — see that module's own docstring for why
  it currently does the latter, not the former).
- **Genuine multi-agent RL (QMIX, MADDPG, or similar)** — ALGO AKANADE
  already computes a REAL Shapley-attributed shared reward across every
  agent that participated in a task (`shapley.py`), then updates each
  agent's OWN independent bandit posterior with its share. That's
  correct for "who deserves credit," but each agent still learns
  independently — a genuine MARL algorithm would let agents learn a
  SHARED value function that accounts for how agents' policies interact
  (useful once agent-to-agent handoffs become more strategic than
  today's "lead agent calls a capability-matched worker" pattern in
  `network.py`).

## 3. Game theory beyond the Shapley value

**What exists today**: `shapley.py` computes the real Shapley value —
the unique fair-division solution satisfying efficiency, symmetry,
additivity, and the null-player axiom — to split one observed task
reward across every agent that actually participated
(`ARCHITECTURE.md` §4). This is already a real, non-trivial piece of
game theory in production, not a superficial name-drop. Other concepts
from the same field that could extend it:

- **Nash / correlated equilibrium for resource contention** — right
  now, `llm/provider_budget.py`'s daily caps are enforced
  first-come-first-served: whichever call happens to land first gets
  the budget, with no notion of fairness across competing callers
  (different domains, different agents) who might all want the same
  capped provider. Modeling budget allocation as a game — each
  task-type/domain "bids" for a share of a shared daily cap — and
  computing an equilibrium allocation would give a principled
  alternative to "first request wins," directly extending the budget
  feature `README_PROVIDER_BUDGETS.md` documents today.
- **Auction theory / VCG mechanisms** — a Vickrey-Clarke-Groves
  mechanism specifically makes TRUTHFUL bidding the dominant strategy —
  relevant if agents or domains were ever allowed to self-report how
  much budget/compute they expect to need (a VCG-style allocation means
  an agent can't gain by inflating its own reported need, unlike a
  naive priority system).
- **The Core / the Nucleolus** — alternative cooperative-game solution
  concepts to the Shapley value, with a different fairness guarantee:
  the Core specifically ensures no SUBSET of agents would do better by
  "defecting" and handling a task as their own smaller coalition
  instead. Relevant only if agents ever gain enough autonomy to
  actually refuse a task allocation — not true today (the Adaptive
  Router assigns, agents don't bid), but worth knowing the Shapley
  value doesn't guarantee coalition-stability the way the Core does, if
  that autonomy is ever added.
- **Stackelberg (leader-follower) games** — model the Adaptive Router
  as a leader that COMMITS to a routing policy first, with agents as
  followers that optimize their own behavior in response to that
  committed policy — a real reframing only worth it if agents ever get
  enough independence to have their own optimization objective distinct
  from "do the task well" (e.g., an agent that could choose to look
  busy to avoid low-value tasks) — again, not a real risk in the
  current design where agents are cooperative, not self-interested, but
  the right lens if that ever changes.

## 4. Other possible additions

- **Real human feedback (a genuine RLHF-lite loop)** — every reward
  signal today (`finetune_log.py`'s `compute_reward()`,
  `log_document_turn()`'s fixed `0.9`, `log_ground_truth_turn()`'s
  fixed `1.0`) is a HEURISTIC proxy for quality, honestly documented as
  such in that module's own docstring. A real thumbs-up/down on actual
  chat replies, fed directly into the SAME reward field
  `RewardWeightedTrainer.compute_loss()` already reads, would replace a
  proxy with a real signal without changing anything downstream.
- **A model registry (MLflow or similar)** — today, only a candidate
  that PASSES the regression gate gets a persisted record
  (`/api/admin/slm-versions`); a failed candidate's `metadata.json` and
  `eval_report.json` sit in `exported_models/` but aren't queryable the
  way a real registry would make them, with full lineage back to the
  exact data snapshot + hyperparameters + git commit that produced it.
- **Streaming ingestion (Kafka/Kinesis)** instead of the nightly
  `docker cp` batch pull — would make near-real-time fine-tuning
  possible instead of strictly once-a-night, and is the natural
  Bronze-layer entry point for the data lake in §1.
- **A/B / canary traffic splitting** — today a candidate that passes
  the gate is promoted to 100% of traffic for its domain immediately
  (`convert_and_register_ollama_model.sh`'s upsert-by-name). A gradual
  rollout (route 5% of a domain's traffic to the new candidate, watch
  its REAL production reward for a day, then promote to 100%) would
  catch a regression the offline eval harness's 32 test-asset items
  missed, before it affects every user of that domain.
- **Distributed/multi-GPU training (DeepSpeed, FSDP)** — once a base
  model outgrows what `finetune_local_slm.py`'s single-box CPU/GPU
  training can handle overnight, the reward-weighted LoRA training loop
  itself doesn't need to change, only how it's parallelized across
  devices.
