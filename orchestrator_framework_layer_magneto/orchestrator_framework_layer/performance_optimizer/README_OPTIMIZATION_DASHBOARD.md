# The Optimization Dashboard — RL optimizers, GPU tier, daily feedback loop

A small, standalone admin app that answers one question Algo Akanade's own API
never surfaced anywhere: **is the optimization layer actually helping?**
Every RL-driven decision Algo Akanade's `llm/router.py` makes on every LLM
call — which provider to try first, which generation-parameter arm to
send, whether the local Algo Engine's GPU tier is actually the one
answering — now has a durable trail and a dashboard reading it.

Two admin-only ways to log in are unrelated by design: the main app's
own Users table, and this dashboard's own separate admin/admin login
(configurable). This is intentionally its own service — it keeps working
to show yesterday's numbers even when Algo Akanade's own process is down.

## Architecture

```
 Real LLM call in Algo Akanade
 ──────────────────────────
 llm/router.py: TierClient.chat()
   │
   ├─▶ optimization_router.select()   RL arm choice   (in-memory bandit)
   ├─▶ provider_optimizer.order()     RL provider order (bandit_persistence
   │                                   → BanditState.json, "provider_optimizer" only)
   ├─▶ p.chat(...)                    the real HTTP call
   │
   └─▶ optimization_events.log_call() ── durable row ──▶ OptimizationEvents.json
                                                           (app/db.py LocalStore,
                                                            DATA_DIR-scoped)

 Algo Engine (Go), independently running
 ─────────────────────────────────────────
 GET /stats  ──▶  {"kernel_tier": "pure-go" | "cpu-accel" | "cuda", ...}
                   (quant.Kernel.Name(), see internal/quant/quant.go)

 Optimization Dashboard (this app) — a SEPARATE FastAPI + React service
 ────────────────────────────────────────────────────────────────────
 dashboard/backend/app/
   config.py         ALGO_AKANADE_DATA_DIR, ALGO_ENGINE_BASE_URL, its own admin
                      credential + JWT secret — all independent of Algo Akanade's
   auth.py            its OWN login (admin/admin by default) — NOT the
                      Users table, NOT Algo Akanade's JWT secret
   events_store.py    reads OptimizationEvents.json + BanditState.json
                      DIRECTLY off disk (no HTTP call to Algo Akanade needed —
                      works even if that process isn't running right now)
   engine_client.py   best-effort GET {ALGO_ENGINE_BASE_URL}/stats, plus
                      the static 3-tier kernel description (always shown)
   main.py            POST /api/login, GET /api/summary, /api/daily,
                      /api/events — all Admin-JWT-gated (this app's own JWT)

 dashboard/frontend/  React + TypeScript + Vite, no chart library — plain
                      CSS bars/tables, dark technical theme matching the
                      Algo Engine's own embedded dashboard's palette.
   Login.tsx           this dashboard's own sign-in
   Dashboard.tsx        stat cards, daily feedback-loop deltas, GPU/kernel
                        panel, RL provider-ordering posteriors, per-
                        provider/per-arm tables, recent events
```

Why read files instead of calling Algo Akanade's HTTP API: `OptimizationEvents`
and `BanditState` are both `app/db.py` `LocalStore` JSON tables — the same
on-disk shape every other admin surface in this codebase already reads
(`security_events.py`, `bandit_persistence.py`). Reading them directly
means this dashboard shows real numbers even when Algo Akanade's own process
happens to be restarting or down, which is exactly the moment "what
happened yesterday" needs to still work. In a container deployment this
means mounting the same volume Algo Akanade's `DATA_DIR` uses, read-only, into
this service — `ALGO_AKANADE_DATA_DIR` is the one setting that has to agree
with Algo Akanade's own `DATA_DIR`.

## What gets recorded, and by what

`app/optimization_events.py` (Algo Akanade side) — one row per `TierClient.chat()`
attempt, success OR failure: `node, tier, provider, arm_id, ok, latency_ms,
fallback_used, occurred_at`. No message content — this is a metrics feed,
not a transcript (the real transcript already lives in `ChatSessions`
under the user's own access controls). Hooked into the exact two places
`llm/router.py` already logs to `self.log` (the in-memory, per-session
`modelTierLog`) — this durable log is the same facts, kept across
sessions and restarts instead of thrown away when the chat turn ends.

`app/framework/bandit_persistence.py` (Algo Akanade side, pre-existing) —
`provider_optimizer`'s Thompson Sampling posterior, saved every 20 calls.
The ONE optimizer among optimization_router/token_optimizer/
provider_optimizer that opts into durable persistence today — the other
two are in-memory-only in this Algo Akanade version, so their live per-call
choices show up in the event log above (arm distribution, success/latency
by arm), not as raw posterior weights on this dashboard. That's actually
the more useful view for "is the RL policy working": a rising success
rate for the arm the bandit increasingly prefers is the evidence, not the
regression coefficients behind it.

## GPU / kernel tier panel — what "deep" means here

Three real tiers, not a GPU on/off toggle (`performance_optimizer/
algo_optimzer_engine/algo-engine-source/internal/quant/`):

| Tier | Build | Mechanism |
|---|---|---|
| `pure-go` | `go build` (no tags) | Plain Go loop — the correctness oracle every other tier is equivalence-tested against. |
| `cpu-accel` | `make accel` (`-tags asmkernel`) | cgo C loop, `-O3 -march=native` — real AVX2/NEON autovectorization. Builds and passes its equivalence test on a normal dev box (no GPU needed). |
| `cuda` | `make cuda` (`-tags cuda`) | `internal/quant/cuda/kernels.cu`, cgo-linked. Weights uploaded to device memory ONCE and cached by host pointer (residency, not per-call PCIe shuttling); one thread block row-parallelizes the matvec. Needs a real NVIDIA GPU + `nvcc` to build — code-complete, unverified on a box without one. |

`GET /stats` (`internal/api/api.go`) now reports `kernel_tier` from
`Engine.KernelTier()` → `quant.Kernel.Name()` — which tier is ACTUALLY
running in a given binary, not which build tag was requested. The
dashboard's GPU panel shows this live when `ALGO_ENGINE_BASE_URL` is
configured and the engine is reachable, and shows the three tiers'
capabilities either way — useful even with no engine currently running,
since "what would `make cuda` get me" is itself a real question an admin
asks before provisioning a GPU box.

## Daily metrics and the feedback loop

`GET /api/daily?date=YYYY-MM-DD` computes that day's rollup
(`events_store.summarize()`) and the delta against the day before —
success rate, avg latency, fallback rate, call volume. A rising success
rate and falling latency day-over-day is `optimization_router`'s and
`provider_optimizer`'s bandits actually having learned something between
those two days, visible in real outcomes rather than asserted. Every
number here comes from `OptimizationEvents.json` rows real chat turns
wrote — nothing is synthesized for the dashboard.

## Running it

```bash
# 1. Algo Akanade's backend (the event producer) — from orchestrator_framework_layer/akanade/backend
uvicorn app.main:app --port 8011

# 2. (optional) a local Algo Engine, for the live GPU panel — from
#    orchestrator_framework_layer/performance_optimizer/algo_optimzer_engine/algo-engine-source
go run ./cmd/algo-engine gen -model ./toy-model      # one-time: synthetic weights
go run ./cmd/algo-engine serve -model ./toy-model -addr :8081

# 3. This dashboard's backend — from performance_optimizer/dashboard/backend
export ALGO_AKANADE_DATA_DIR=../../../akanade/backend/data   # must match step 1's DATA_DIR
export ALGO_ENGINE_BASE_URL=http://localhost:8081        # optional, step 2
export DASHBOARD_ADMIN_USER=admin DASHBOARD_ADMIN_PASSWORD=admin
uvicorn app.main:app --port 8020

# 4. This dashboard's frontend — from performance_optimizer/dashboard/frontend
npm install && npm run dev   # http://localhost:5175, proxies /api to :8020
```

Sign in with `DASHBOARD_ADMIN_USER`/`DASHBOARD_ADMIN_PASSWORD` (default
`admin`/`admin` — change both via env vars for anything beyond a laptop
demo). Chat with the real app through Algo Akanade's own widget/API and the
dashboard's numbers update on its own 15-second poll.

## Verified, end to end, on this machine

Not just unit-tested — actually run: Algo Akanade's backend was started
against a fresh `DATA_DIR`, logged into as a real seeded user, and sent
five real chat prompts (`groq-llama-3.3-70b`/`gemini-flash` both actually
answered — real provider calls, real `optimizationArm` selections, real
latencies). `OptimizationEvents.json` picked up all five rows. This
dashboard's backend, pointed at that same data dir, correctly computed
the per-provider/per-arm breakdown over `GET /api/summary`. A real
`toy-model` Algo Engine instance was generated and served
(`go run ./cmd/algo-engine gen`/`serve`), and the dashboard's GPU panel
correctly showed `kernel_tier: "pure-go"` live from its `/stats`. The
frontend's dev server was started and its `/api` proxy confirmed reaching
the backend (login round-tripped a real JWT through it). Both the Go
engine's full test suite (`go build`/`go vet`/`go test ./...`, plus the
`-tags asmkernel` CPU-accelerated build) and Algo Akanade's full Python suite
(628 tests) pass after these changes — the router.py instrumentation
didn't change any existing behavior, only added a durable log write next
to logging that already happened.

**Not verified here** (this dev box's real limits, not code gaps): the
`cuda` kernel tier needs an actual NVIDIA GPU + `nvcc`, neither present
on this machine — see `README_OPTIMIZATION_DASHBOARD.md`'s GPU panel
section above for what's code-complete versus what's build-and-run
proven. Ollama wasn't installed in this environment either, so
`token_optimizer.py`'s tool-schema trimming path (Ollama-only) rode
along in every test/import but wasn't exercised by a live local-model
call in this particular run — its own unit tests (`tests/
test_...optimizer...py` in Algo Akanade's suite, part of the 628 passing)
cover that logic directly.
