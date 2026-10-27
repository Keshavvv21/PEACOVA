# performance_optimizer

The model execution and optimization layer. `algo_optimzer_engine/algo-engine-source`
is a real, standalone Go program — **Algo Engine**, ALGO AKANADE's self-hosted
model execution runtime (tokenizer, safetensors loader, router/gate, tiered
expert store, concurrent per-token pipeline, KV-cached attention, CLI, and
an OpenAI-compatible HTTP API) — verified building and passing its full
test suite (`go build ./...`, `go vet ./...`, `go test ./...`, all green,
including a real `models/olmoe` package for OLMoE-class frontier weights,
not just the toy teaching model). See
[`Arhitecture_Opitimization.md`](Arhitecture_Opitimization.md) for the full
positioning: Algo Engine is not an MoE-only engine and not a GPU-only
accelerator — those are capabilities inside a general SLM/LLM runtime whose
job is deciding *how* to execute a model ALGO AKANADE has already decided *what*
to run.

Its matmul kernel is a three-tier `quant.Kernel` interface, selected at
build time and never mixed:

| Tier | Build | What it is |
|---|---|---|
| Default | `go build` (no tags) | Pure Go, CGO_ENABLED=0, a static binary — the correctness oracle every other tier is equivalence-tested against. |
| CPU-accelerated | `make accel` (`-tags asmkernel`) | A cgo C loop compiled `-O3 -march=native`, autovectorized (AVX2/NEON) — verified building and passing `equiv_test.go` on this box. |
| GPU | `make cuda` (`-tags cuda`) | Real CUDA (`internal/quant/cuda/kernels.cu`, cgo-linked via `libaekernels.a`) with an upload-once device-residency cache — genuine code, but needs an NVIDIA GPU + `nvcc` to build/run, which this dev box doesn't have, so it's unverified here. |

**Wired into Algo Akanade**: `orchestrator_framework_layer/akanade/backend/app/llm/router.py`
registers `"algo-engine"` as a real provider (`OpenAICompatible`, pointed
at `ALGO_ENGINE_BASE_URL`) and includes it in the `delay_ok` and `bulk`
`CHAINS` — a real fallback rung, not just a pin-only extra. Point
`ALGO_ENGINE_BASE_URL` at a running `algo-engine serve` (any kernel tier)
to put it in rotation; empty (the default) means it's silently skipped,
same as every other unconfigured provider (`.available()` gates it out).

**Which kernel tier is ACTUALLY running** is now a real, queryable fact,
not just a build-time choice: `quant.Kernel.Name()` reports
`"pure-go"`/`"cpu-accel"`/`"cuda"`, surfaced on `GET /stats`
(`internal/api/api.go`) as `kernel_tier` and shown live in the engine's
own embedded dashboard (`internal/api/web/`).

## See also

- **[README_OPTIMIZATION_DASHBOARD.md](README_OPTIMIZATION_DASHBOARD.md)** —
  the admin dashboard (its own login, separate from every other account
  in this codebase) for RL-optimizer call metrics, the GPU/kernel-tier
  panel above, and day-over-day feedback-loop trends. Reads Algo Akanade's
  durable `OptimizationEvents`/`BanditState` tables directly off disk, so
  it keeps working even when Algo Akanade's own process isn't running.
