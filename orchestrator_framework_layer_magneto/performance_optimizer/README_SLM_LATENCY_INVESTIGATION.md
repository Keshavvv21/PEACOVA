# SLM latency: what's real, what's not, and why

This documents a real investigation into hitting ~0.5s GPU / <1s CPU
latency for the coding SLM, using the existing Algo Engine (Go
performance-optimizer/GPU layer at `algo_optimzer_engine/algo-engine-source`).
Written to be checked, not taken on faith — every claim below was
verified against this repo's actual code and this environment's actual
hardware.

## What's real and already verified

- `algo-engine-source` is a genuine, working Go inference engine —
  quantization kernels (`internal/quant`, both a portable default path
  and a CUDA path), attention (`internal/attention`), a KV cache
  (`internal/kvcache`), safetensors loading, a tokenizer, and an MoE
  expert store. `go build ./...` and `go test ./...` both pass clean
  (verified this session, every package).
- A CPU micro-benchmark run for real this session:
  `BenchmarkMatVec128x64` at ~14.5µs/op, `BenchmarkMatVecQ8_128x64` at
  ~15.9µs/op (`go test -bench=. ./internal/quant/...`). Real numbers,
  on this machine's CPU (Intel Core 5 120U) — but a single matrix-vector
  op at this size tells you almost nothing about end-to-end generation
  latency for a real model (a real forward pass chains this operation
  across every layer, for every generated token).

## The real gap: no GPU, and no Qwen support

Two separate, load-bearing facts, checked directly rather than assumed:

1. **No GPU is reachable anywhere in this session's environment.**
   `nvidia-smi` doesn't exist on the local dev machine, and every check
   of the actual deploy server this session pointed the same direction
   (CPU-only Ollama, no CUDA runtime found). The "~0.5s on GPU" target
   cannot be tested, verified, or honestly claimed without GPU hardware
   to run it on — this is a hardware access problem, not a code
   problem, and no amount of Go code changes fixes a target that has
   no hardware to run against.
2. **`models/` only contains `olmoe` and `toy`** — this engine has
   real, from-scratch Go implementations of specific model
   architectures, and Qwen2.5-Coder (the model actually deployed as
   `ollama-coder`) is not one of them. Adding real Qwen2 architecture
   support (its specific attention/RoPE variant, weight layout,
   tokenizer) is a substantial model-implementation project in its own
   right — comparable in scope to `models/olmoe`'s own existing
   implementation — not a tuning pass on existing code. Attempting to
   fake this (claim latency numbers for a model this engine can't
   actually run) would be worse than not attempting it.

Given both gaps, this session did **not** attempt to wire Algo Engine
in as `ollama-coder`'s replacement, and did **not** publish any
GPU/CPU latency number for it — there's nothing real to measure yet.

## Why sub-1-second matters less than it sounds for a real answer

Even with a hypothetically-optimized engine, "under 1 second" is a
function of how many tokens the answer needs, not just how fast each
token is generated. A model that generates at a genuinely fast
50 tokens/second still takes 2+ seconds for a 100-token answer (a short
function with a docstring) and 6+ seconds for a 300-token one (most
real debugging/refactoring answers this session's own 105-prompt test
suite includes). Sub-1-second is realistic for the FIRST token
(time-to-first-token) or for a trivial one-line answer, not for a
typical, useful coding response — a claim worth stating plainly rather
than letting "under 1 second" imply something it can't be for most real
prompts.

## What actually reduced real, observed latency this session

Not aspirational — measured, live, on the actual deployment:

| Fix | Before | After | Where |
|---|---|---|---|
| Ollama `keep_alive` (was the 5-minute default; unloading the model between normal-length gaps between messages) | 13.8-22s (cold reload) | 0.2-5.4s (warm) | `app/llm/providers.py` |
| Auth token validation cache (every request re-validated against the real app over HTTP) | 161ms | 9ms | `app/auth.py` |

These are real wins because they removed genuinely wasted overhead —
cold-start reloading and repeated network round trips — not because
they made the model itself compute faster. That's the honest,
available lever here: cut overhead around inference, don't claim to
have rewritten inference itself into meeting a target with no hardware
or model support to run against.

## Parallelization: what's safe and what isn't

Two real findings, checked against this repo's actual code before
either was acted on:

- **Running the API server with multiple uvicorn worker processes**
  (the standard way to parallelize a Python API across CPU cores) would
  **break client-tools mode** — `_PENDING_TOOL_CALLS`/`_PENDING_WRITES`
  (`coding_agent.py`), the auth-validation cache (`auth.py`), and every
  routing bandit (`hybrid_router.py`, `domain_router.py`,
  `provider_optimizer.py`) are plain in-process Python dicts. Multiple
  worker processes don't share memory — a client-tools turn's pending
  tool call, stashed in worker A, would be invisible to worker B if the
  next request in that same turn happened to land there. **Not
  implemented, on purpose** — this is exactly the kind of change the
  "make sure nothing breaks" instruction is right to guard against.
- **Parallelizing `scripts/eval/regression_gate.py`'s scoring loop**
  (currently 64 fully sequential HTTP calls — 32 test items × 2
  candidates, measured at ~60s wall-clock this session) is a real
  candidate — `score_item()` is a pure function with no shared mutable
  state, so it's thread-safe to parallelize. But both candidates
  typically share ONE Ollama instance (`SERVE_BACKEND=ollama`, the
  default) — firing concurrent requests at a single CPU-bound model
  server can queue/contend rather than actually run in parallel, and
  might make wall-clock time WORSE, not better, depending on Ollama's
  own concurrency configuration on that box. This is genuinely
  untestable without live access to measure before/after — **not
  implemented this session**, flagged here as a real, scoped follow-up:
  safe and clearly beneficial for `SERVE_BACKEND=llama_server` (old and
  new already run as separate server processes there — no shared-
  resource contention at all), unproven for the default Ollama path.

## If you want to actually pursue this further

1. **Provision real GPU access** (locally or on the deploy box) — the
   single fastest, lowest-engineering-risk way to materially cut
   latency is to run the existing Ollama/llama.cpp stack with real CUDA
   acceleration instead of CPU. This alone likely gets close-to-1-second
   token generation for a 1.5B model without touching Algo Engine at
   all.
2. **If Algo Engine is still the goal**: scope "add Qwen2 architecture
   support" as its own dedicated project (weight loading, attention
   variant, RoPE, tokenizer, a real correctness test against known
   HuggingFace outputs — the same bar `models/olmoe`'s own tests hold
   themselves to), before any latency work on top of it.
3. **A smaller/more aggressively quantized model** (a 0.5B variant, or
   4-bit instead of the current quantization) trades some quality for
   real, measurable speed — worth benchmarking against the 105-prompt
   suite's own quality bar (`README_SLM_TEST_SUITE.md`) before adopting.
