# ak-router — Primary/Secondary Routing Architecture

Standalone provider/model-ordering service (`backend/`) plus the admin-controlled
blue/green switch inside `akanade/backend` that decides which pipeline answers real
chat traffic. Built per the CEO memo's own instruction: a deterministic, cost-aware
baseline first, RouteLLM evaluated as a genuine peer to Algo aKANADE's own routing —
not a competitor's product — with Thompson Sampling demoted to an admin-opt-in,
shadow-mode-first exploration policy, never the default.

## Flow

```
                         SAME USER QUERY
                                │
                                ▼
                    ┌──────────────────────┐
                    │   COMMON QUERY       │
                    │       ROUTER         │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
             PRIMARY                     Algo aKANADE
                 │                           │
                 ▼                           ▼
        RouteLLM / Policy Layer        aKANADE Router
                 │                    Routing / Policy
        ┌────────┴────────┐            ┌─────┼─────┐
        ▼                 ▼            ▼     ▼     ▼
   DEFAULT TIER      ESCALATION TIER  SLM 1 SLM 2 MODELS
        │                 │            │     │     │
   ┌────┴────┐       ┌────┼─────┐      └─────┼─────┘
   ▼         ▼       ▼    ▼     ▼            │
 GLM-5.3  Qwen3.8  GLM-5.3 Sol  Claude       ▼
 Flash    Flash    Full   xHigh Opus 5    RESPONSE
                    Max                       │
   │                 │                        │
   └────────┬────────┘                        │
            │                                 │
            ▼                                 ▼
        RESPONSE                           RESPONSE
            │                                 │
            └──────────────┬──────────────────┘
                           │
                           ▼
                    ┌───────────────┐
                    │  ROUTER SWITCH │
                    │   / DECISION   │
                    └───────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
       RETURN TO USER   USER FEEDBACK   EVALUATION
             │              │              │
             │              ▼              ▼
             │       PEFT TRAINING DATA  PEFT TRAINING
             │              │              │
             │              ▼              ▼
             │     PEFT TRAINING      IMPROVED SLM /
             │        STORAGE             MODEL
             │              │              │
             │              ▼              ▼
             │       FUTURE SLM        ROUTER FLAG /
             │          TRAINING        ROUTER UPDATE
             │              │              │
             │              ▼              │
             │        IMPROVED SLM          │
             │              │              │
             └──────────────┴───────┬──────┘
                                    │
                                    ▼
                         MODEL PERFORMANCE DB
                                    │
                                    ▼
                         COMPARISON DASHBOARD
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        PRIMARY MODELS         aKANADE MODELS          ALL MODELS
        RouteLLM Path          SLMs + Models           Comparison
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                              DAILY REPORT
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
          QUALITY                LATENCY                  COST
          ACCURACY              TOKENS               SUCCESS RATE
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                            PEFT IMPROVEMENT
                                    │
                                    ▼
                       THOMPSON SAMPLING DATA
                              / WEIGHTS
                                    │
                                    ▼
                         BEST MODEL / BEST PATH
```

## Components

| Piece | Where | What |
|---|---|---|
| ak-router service | `router/backend/app/` | Deterministic cost-aware ordering (`deterministic.py`), shadow-only Thompson Sampling (`thompson.py`, Groq + local SLM arms only, never live by default), an admin-opt-in BERT difficulty shadow signal for the secondary pipeline (`bert_shadow.py`), and the Generic Architecture comparison arm (`generic_architecture.py`, RouteLLM). |
| Blue/green switch | `akanade/backend/app/llm/primary_pipeline.py` | Which pipeline answers real traffic — `"routellm"` (primary+secondary, default), `"routellm_only"` (primary only, same failover safety net but every fallback logged as a leak), or `"akanade"` (secondary only, RouteLLM never consulted). |
| TierClient wiring | `akanade/backend/app/llm/router.py` | Applies the switch UNIFORMLY inside `TierClient.chat()` — every track (PM chat, coding, trading) gets the same treatment regardless of entry point. |
| Admin controls | `akanade/backend/app/routers/admin_router_config.py`, `AdminAIModels.tsx` | Executive-write / Admin-read RBAC (the real app's Manager2/"Senior Manager" role satisfies Admin-read). Three-way switch in the web UI. |
| Model evaluation | `admin_model_benchmarks.py`, `ModelEvaluationDashboard.tsx` | Runs the eval harness against registered models (`ours` = Algo aKANADE's own SLMs, `candidate` = Generic Architecture models), writes `ModelBenchmarks` rows, closes the loop by PATCHing each provider's measured `capability_tier`. |

## Why deterministic-first, bandit-deferred

Both pipelines follow the same rule, straight from AlgoAnalytics' own Sep 2026
architecture recommendation ("LLM Routing: Thompson Sampling vs. a Robust
Baseline"): **build a deterministic, calibrated baseline first; a bandit is a
plug-in exploration policy promoted later, by evidence, never the default.**
The memo's own words: "In production systems we first do tried and tested,
simple and boring things. Get it working, find where it needs improvement and
only then look for advanced things to replace the vanilla stuff" — and
Thompson Sampling specifically should not be live "until someone can explain
to me exactly how it works on the board."

**Primary pipeline (`generic_architecture.py`) — a static classifier, never a
bandit.** RouteLLM's routers are all *pretrained, supervised* difficulty
classifiers (the memo's own "Layer B — supervised acceptance prediction") —
none of them learn from this deployment's live traffic or explore uncertain
options; they give the same answer for the same prompt every time, which is
exactly the "reproducible from logged inputs" bar the memo sets. RouteLLM
ships five router algorithms; only two are actually wired into the
`Controller` end-to-end without extra credentials:

- **`mf`** (matrix factorization) — needs a genuinely funded OpenAI key to
  embed each prompt at call time. This deployment has none, so it constructs
  fine but fails on every real call, falling through to the next rung — kept
  first so it activates automatically, with zero code changes, the moment a
  real key is configured.
- **`bert`** — a fine-tuned DeBERTa/XLM-RoBERTa classifier
  (`routellm/bert_gpt4_augmented`, not gated). Self-contained: no external API
  call at inference time, runs on CPU with zero extra config — **the one
  candidate that actually decides most calls today**, verified live to
  discriminate correctly (a trivial prompt → default tier, a genuinely hard
  multi-step proof → escalation tier).
- **`sw_ranking`** (similarity-weighted ranking) — same real-OpenAI-key gap as
  `mf`, plus its default Arena dataset pair has drifted upstream (confirmed
  live: a schema-alignment error at construction) — gracefully excluded, kept
  in the candidate list for the same "activates automatically once fixed"
  reason as `mf`.
- **`causal_llm`** is **not** attempted at all, not just deprioritized:
  RouteLLM's own `causal_llm/model.py` hardcodes `.to("cuda")` with no CPU
  fallback (unlike `matrix_factorization`, which checks device), so it throws
  immediately at construction on this CPU-only deployment — and its default
  checkpoint's tokenizer/base model (`meta-llama/Meta-Llama-3-8B`) is gated on
  Hugging Face, needing a separate license approval this deployment doesn't
  have. Neither gap is fixable without forking RouteLLM's own source.
- **`random`** — the guaranteed floor every candidate above degrades to if it
  fails; zero external dependencies, always constructs.

Each candidate is probed as its own single-router `Controller` first (see
`_get_controller()`), so one router's construction failure (confirmed live for
`sw_ranking`) can never take down a different router (`bert`) that works fine
on its own — a genuine robustness property, not a hypothetical one.

Picking a *specific* model within whichever tier the classifier chose is
still plain round-robin (`itertools.cycle`), not cost-ranked — deliberately:
no real per-token pricing exists anywhere in this codebase yet for
`glm-5.3-flash`/`qwen3.8-flash`/`glm-5.3-full`/`sol-xhigh`/`claude`
specifically (`pricing_table.py` only has real rates for a different provider
set). Reordering by invented costs would be the opposite of "boring and
honest about what it doesn't know" — round-robin among tier-mates the
classifier already judged roughly equivalent stays the correct, explainable
choice until real pricing exists to rank them by.

**Secondary pipeline (`deterministic.py` + `thompson.py`) — the memo's exact
staged rollout, already built.** `deterministic.py`'s cost-aware ordering
*is* the memo's "Layer C — deterministic economic decision" baseline, live by
default. Thompson Sampling matches the memo's Phase 2/3 exactly: computed and
logged on every call as a shadow comparison signal regardless of the live
strategy, restricted to exactly two arms (`groq`/`algo-akanade-slm`, never
the full provider chain), and only ever controls real traffic if an admin
explicitly flips `RouterConfig.strategy` to `"thompson_sampling"` — no
promotion has happened; it stays in shadow mode until evidence says otherwise.

`bert_shadow.py` adds one more comparison column to that same secondary
pipeline, in the same shape: OFF by default (`BertShadowConfig`, its own
admin toggle at `/admin/bert-shadow`), reuses the primary pipeline's own
already-loaded `bert` classifier (no second checkpoint), and — like
`shadow_pick` — is logged into `RoutingOutcomes` for comparison but never
reorders a real `/route` response on its own. The full 5-router cascade was
deliberately **not** duplicated into the secondary pipeline: `mf`/`sw_ranking`
would be exactly as non-functional there (same missing OpenAI key) and
`causal_llm` exactly as infeasible (same CUDA/license gap) — adding
non-working rungs for symmetry is complexity the memo argues against, not
robustness.

## Naming

`akanade` (lowercase) names the infrastructure/codebase only — folder paths, Docker
image internals, the wire-level pipeline enum value — never a user-facing label.
**Algo aKANADE** is the product name shown everywhere else (UI, mobile app, VS Code
extension, docs). "Generic Architecture" is RouteLLM's own arm — the CEO's reference
design, evaluated as a peer, not a vendor product.

## Known limitation (found in live testing, 2026-09-03)

A live 15-prompt coding test found that the coding-agent's own internal call chain
does not yet consistently propagate an explicit provider preference through to its
real answering call — neither RouteLLM's tier decision nor a direct API-level
override reliably changed which model answered a coding-agent turn in that test; it
landed on the same fallback model regardless. The blue/green switch is fully live
and correct for general chat traffic; the coding-agent track needs its own
preference-plumbing fixed before RouteLLM's tier decisions change coding-agent
model selection in practice. See the live test reports (Primary/Secondary path
comparisons) for the full per-prompt evidence.
