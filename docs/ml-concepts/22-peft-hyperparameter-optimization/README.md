# PEFT Hyperparameter Optimization — Bandit-Tuned LoRA Configs

[08-lora-qlora](../08-lora-qlora/README.md) covers what LoRA rank, alpha, dropout, and target modules
*mean*. This document covers a narrower, separate question this repo actually answers with code: which
values of those hyperparameters should tonight's fine-tune actually use, given that a fixed choice
picked once and never revisited can't adapt as the training corpus grows and changes? The answer here
is the same Thompson Sampling contextual bandit covered in
[12-thompson-sampling](../12-thompson-sampling/README.md), reapplied to a new kind of arm — not "which
model/provider to route to," but "which joint LoRA-rank/alpha/reward-threshold preset to train with,"
learned from whether last night's pick actually produced a better model.

## Table of contents

1. [Why a fixed PEFT config isn't enough](#1-why-a-fixed-peft-config-isnt-enough)
2. [Recap: Thompson Sampling in one page](#2-recap-thompson-sampling-in-one-page)
3. [Framing hyperparameter choice as a bandit problem](#3-framing-hyperparameter-choice-as-a-bandit-problem)
4. [The context vector: how much data, how clean](#4-the-context-vector-how-much-data-how-clean)
5. [The reward signal: the regression gate's score margin](#5-the-reward-signal-the-regression-gates-score-margin)
6. [The full nightly loop](#6-the-full-nightly-loop)
7. [Block diagram](#7-block-diagram)
8. [How this repo actually uses it](#8-how-this-repo-actually-uses-it)
9. [Practical notes and caveats](#9-practical-notes-and-caveats)
10. [References](#10-references)

---

## 1. Why a fixed PEFT config isn't enough

A LoRA rank of 8 with a reward threshold of 0.7 is a reasonable hand-picked default when the training
corpus is thin and noisy — a small rank has less capacity to overfit a handful of examples, and a
stricter reward filter keeps only the cleanest of them. But as
[the data pipeline](../23-data-pipeline-slm-finetuning/README.md) accumulates a larger, better-curated
corpus over weeks, that same conservative choice under-uses the model's capacity to actually learn from
it — a higher rank and a looser reward threshold (more data, still net-positive on average) would learn
more per night, without necessarily overfitting.

The problem is structurally identical to picking which arm of a slot machine to pull: there's no
formula that says "500 curated examples at average reward 0.82 implies rank 12" — the only way to know
whether a bigger rank actually helps *this* corpus is to try it and observe whether the resulting model
is genuinely better. That is exactly the explore/exploit problem a bandit is built to solve, so rather
than inventing a new tuning heuristic, this repo reuses the one already proven out for routing decisions
elsewhere in the stack.

## 2. Recap: Thompson Sampling in one page

(Full derivation in [12-thompson-sampling](../12-thompson-sampling/README.md) — this is only enough to
follow §3–§6.) A **contextual linear bandit** models each arm `a`'s expected reward as linear in a
context vector `x`: `E[r | x, a] = θ_a · x`, with `θ_a` unknown and given a Gaussian posterior via
Bayesian linear regression. Each round:

1. For each arm, sample `θ̃_a` from its current posterior `N(μ_a, Σ_a)`.
2. Pick `argmax_a θ̃_a · x` — the arm whose *sampled* estimate is highest, not necessarily its true mean.
3. Observe the real reward `r`, and update that arm's posterior: `B_a += x·xᵀ`, `f_a += r·x`
   (`μ_a = B_a⁻¹f_a`), where `B_a` starts at `ridge · I` and `f_a` starts at zero.

The genuine uncertainty baked into the posterior is what makes an under-tried arm occasionally get
picked even when another arm's *current* estimate looks better — exploration falls out of the sampling
step itself, with no separate epsilon or UCB bonus term needed.

## 3. Framing hyperparameter choice as a bandit problem

The three hyperparameters that matter here — `lora_r`, `lora_alpha`, and `min_reward` (the reward
threshold below which a training example is dropped) — are not tuned as three independent bandits. They
interact: a higher rank paired with a low reward threshold means more trainable capacity applied to
noisier data, a combination that is more prone to overfitting than either change made alone. Tuning them
independently could easily land on a "high rank + low threshold" combination that no one would pick on
purpose.

Instead, each **arm** is a named, jointly-sane preset — the same "named preset, not a raw continuous
knob" shape `token_optimizer.py`'s `full`/`top12`/`top6`/`top3` and `hybrid_router.py`'s `local`/`hosted`
choices already use elsewhere in this codebase:

| Arm            | `lora_r` | `lora_alpha` | `min_reward` |
|----------------|---------:|-------------:|-------------:|
| `conservative` |        4 |            8 |          0.8 |
| `balanced`     |        8 |           16 |          0.7 |
| `aggressive`   |       16 |           32 |          0.6 |

`balanced` is `finetune_local_slm.py`'s original, long-standing hand-picked default — kept as the
cold-start-friendly middle option on day one, rather than assuming the bandit already knows better
before it has ever seen an outcome.

## 4. The context vector: how much data, how clean

The context is two cheap, non-LLM numbers — consistent with every other optimizer in this codebase,
none of which spend a model call just to decide how to route or tune:

```
context = [ min(curated_count, 500) / 500,  avg_reward ]
```

- `curated_count` — how many raw examples exist tonight (clipped at 500 and normalized to `[0, 1]`,
  matching `--max-examples`'s own default cap), a proxy for "is there enough data to support a bigger,
  more expressive adapter without overfitting."
- `avg_reward` — the mean reward across those same examples, a proxy for how clean the corpus is
  tonight — a corpus dominated by out-of-scope or retried turns is a worse candidate for an aggressive,
  loose-filter preset than one dominated by clean, high-reward turns.

More/cleaner data → the posterior should, over time, learn to favor `aggressive`; a thin or noisy corpus
→ favor `conservative`. Nothing forces that mapping in code — it is exactly what Thompson Sampling is
expected to discover from real outcomes.

## 5. The reward signal: the regression gate's score margin

The bandit's reward isn't observed the same night the arm is chosen — the model has to actually be
trained, converted, and evaluated first. `scripts/eval/regression_gate.py` reruns the same evaluation
harness against last night's fine-tuned candidate and the currently-served base tag, producing
`old_score` and `new_score`. The margin between them —

```
score_margin = new_score - old_score
reward = tanh(score_margin * 10)
```

— becomes the bandit's reward for whichever arm was used. The `tanh` squash keeps one unusually large
margin from swamping the posterior the way an unbounded reward would (the same "clamp to a sane range"
instinct `finetune_log.py`'s `compute_reward()` already applies elsewhere in this stack, just continuous
here rather than three fixed penalty tiers). `tanh(±0.1 margin) ≈ ±0.76` — already close to saturating,
so a real pass/fail dominates the signal without needing an enormous margin to register.

## 6. The full nightly loop

The bandit's `select_arm()` and `record_outcome()` calls happen in two *separate processes*, on two
different nights of the pipeline's timeline, connected only by a small JSON sidecar file:

1. **Tonight, `finetune_local_slm.py --auto-hyperparams`**: computes tonight's context (raw example
   count + average reward from the log), calls `select_arm_full()`, overrides `--lora-r`/
   `--lora-alpha`/`--min-reward` with the chosen arm's values, trains, and writes
   `{"arm", "context", "state_path"}` to a sidecar file (`peft_arm_used.json`) — because this process
   exits long before the resulting model is ever evaluated.
2. **Later, once the new Ollama tag exists, `regression_gate.py --update-peft-bandit`**: reads that
   sidecar, computes `score_margin = new_score - old_score` from its own old-vs-new evaluation, and
   calls `record_outcome(arm, context, score_margin=...)` — updating the *same* persisted bandit state
   the next `--auto-hyperparams` run will load.

A missing or corrupt sidecar (the very first night, or a night `--auto-hyperparams` wasn't used) is
treated as "nothing to update yet," never a hard failure of the promotion decision itself.

## 7. Block diagram

```
 NIGHT N                                            NIGHT N (later) / promotion step
 ┌──────────────────────────────┐                   ┌────────────────────────────────────┐
 │ finetune_local_slm.py         │                   │ regression_gate.py                  │
 │  --auto-hyperparams           │                   │  --update-peft-bandit               │
 │                                │                   │                                      │
 │  context = [n/500, avg_reward]│                   │  old_score = eval(base_tag)         │
 │        │                      │                   │  new_score = eval(new_tag)          │
 │        ▼                      │                   │        │                             │
 │  bandit.select_arm_full()     │                   │  margin = new_score - old_score     │
 │        │                      │                   │        │                             │
 │        ▼                      │  writes sidecar    │        ▼                             │
 │  arm, hp = {lora_r, ...} ─────┼──►peft_arm_used.json──►  bandit.record_outcome(          │
 │        │                      │                   │      arm, context,                  │
 │        ▼                      │                   │      score_margin=margin)           │
 │  train with hp ...            │                   │        │                             │
 │        ▼                      │                   │        ▼                             │
 │  merged model  ───────────────┼──► convert_and_register_ollama_model.sh (gated on PASS)  │
 └──────────────────────────────┘                   │        │                             │
                 ▲                                   │        ▼                             │
                 └───────────────────────────────────┤  peft_bandit_state.json (updated)    │
                    read again next night             └────────────────────────────────────┘
```

## 8. How this repo actually uses it

- **`_LinearBanditCore`**
  (`orchestrator_framework_layer/akanade/backend/scripts/peft_hyperparam_optimizer.py:59-102`) — a
  standalone reimplementation of `app/framework/bandit.py`'s `LinearBanditCore`, duplicated rather than
  imported because `finetune_local_slm.py` is a deliberately standalone script with its own venv that
  never imports the main `app` package — the same isolation reason `data_engine.py`'s hashed-embedding
  function is a local reimplementation rather than a shared import.
- **`ARMS`** (`peft_hyperparam_optimizer.py:41-45`) — the three named presets from §3.
- **`PeftHyperparamOptimizer.select_arm_full`/`record_outcome`**
  (`peft_hyperparam_optimizer.py:118-159`) — the two entry points, persisting state as plain JSON
  (`--state-path`) rather than `bandit_persistence.py`'s DynamoDB/LocalStore-backed store, since that
  module reaches into the live app's database, unavailable to this standalone script's own process.
- **The `--auto-hyperparams` wiring and sidecar write** (`finetune_local_slm.py:334-361`) — computes the
  night's context from `load_examples(..., min_reward=0.0)`, calls the bandit, overrides the three CLI
  args, and writes `peft_arm_used.json`.
- **`--update-peft-bandit`** (`scripts/eval/regression_gate.py:36-56, 87-92, 135-136`) — reads the
  sidecar, computes the margin, and closes the loop via `record_outcome`.

## 9. Practical notes and caveats

- **Off by default, and correctly so.** `--auto-hyperparams` only produces a genuinely learning bandit
  if `regression_gate.py --update-peft-bandit` is also running every night to feed it real outcomes —
  turning on the picker without the feedback half just adds unexplained variance to nightly training
  runs, per both scripts' own docstrings.
- **Cold start is `balanced`.** With an untrained, ridge-regularized-to-zero posterior, the first several
  nights' picks are close to arbitrary among the three arms until enough outcomes accumulate — this is
  expected bandit behavior, not a bug, and mirrors the cold-start behavior already documented for
  `hybrid_router.py` in [12-thompson-sampling §7.2](../12-thompson-sampling/README.md#72-hybridrouter-two-arms-a-3-feature-context-asymmetric-rewards).
- **Exit code 2 (couldn't run at all) never updates the bandit** — `regression_gate.py` only calls
  `_update_peft_bandit` after `aggregate()` has produced real `old_score`/`new_score` numbers; a missing
  dependency or test-asset directory skips the gate entirely rather than recording a false reward.
- **A worse fine-tune is never promoted, regardless of what the bandit picked.** The bandit's reward
  reflects the outcome so tomorrow's pick is better-informed; it never overrides the separate,
  independent pass/fail decision (`new_score >= old_score - tolerance`) that gates whether tonight's
  model actually reaches the live chatbot.

## 10. References

- Agrawal, S., & Goyal, N. (2013). *Thompson Sampling for Contextual Bandits with Linear Payoffs.*
  ICML. (Full derivation in [12-thompson-sampling](../12-thompson-sampling/README.md).)
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- In-repo:
  `orchestrator_framework_layer/akanade/backend/scripts/peft_hyperparam_optimizer.py`,
  `orchestrator_framework_layer/akanade/backend/scripts/finetune_local_slm.py`,
  `orchestrator_framework_layer/akanade/backend/scripts/eval/regression_gate.py`.
- See also: [08-lora-qlora](../08-lora-qlora/README.md) for what these hyperparameters mean, and
  [23-data-pipeline-slm-finetuning](../23-data-pipeline-slm-finetuning/README.md) for where the
  `curated_count`/`avg_reward` context actually comes from.
