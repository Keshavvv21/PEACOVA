<p align="center"><b>🐦 algo-engine</b><br>
<i>ALGO AKANADE's self-hosted model execution layer, written entirely in Go.</i></p>

**Algo Engine** is a laptop-runnable, single-binary model execution runtime.
It is **not** an MoE-only engine and **not** a GPU-only accelerator — those
are capabilities inside a broader runtime whose job is: given a model, a
workload, and the hardware actually available, determine (and keep
learning) the most efficient way to execute that model while preserving
exact correctness. ALGO AKANADE decides *what* to run; Algo Engine decides *how*
to run it — across CPU, GPU, or hybrid execution, and across VRAM, RAM, and
disk as one tiered memory hierarchy, streaming only the weights a given
token actually needs rather than requiring the whole model resident.

This repository is the reference implementation for the **Algo Engine Build
Manual** and the companion of **Algo Engine Technical Design v0.1**
(AlgoAnalytics internal) and
[`Arhitecture_Opitimization.md`](../Arhitecture_Opitimization.md) (the
system-level architecture and generalization roadmap beyond MoE). Today's
shipped model family is a deliberately tiny *toy* MoE model, so the full
pipeline — routing, int8 quantization, safetensors streaming, LRU +
learned-pin caching, KV-cache persistence, the overlap pipeline — runs
anywhere in milliseconds and is fully real, fully testable, with no real
weights required; dense and hybrid model families are the next model-seam
targets, not a rewrite of anything below.

## Highlights

- **Zero dependencies.** Standard library only; `CGO_ENABLED=0` produces a
  fully static binary. The one sanctioned seam for native acceleration (the
  matmul kernel) is behind a build tag and off by default.
- **A five-step per-token path** — route → union → place → overlap →
  learn — implemented as goroutines with a semaphore-bounded disk tier.
- **The invariant, under test:** placement decides speed, never semantics.
  `TestTinyCacheStillCorrect` proves a capacity-1 expert cache produces
  bit-identical output to a capacity-64 cache.
- **Learned caching:** `usage.json` records your workload's routing; the
  hottest experts are pinned into RAM at startup. The engine gets faster the
  more you use it.
- **Warm sessions:** per-session KV caches persist to disk; conversations
  reopen with zero re-prefill.
- **OpenAI-compatible API** (`/v1/chat/completions`, normal + SSE streaming)
  and an **embedded web dashboard** with live tier stats — one binary, no
  extra runtime.

## Quick start

Requires Go ≥ 1.22. From the repository root:

```bash
# 1. build the binary
go build -o bin/algo-engine ./cmd/algo-engine

# 2. generate the toy model (~1.4 MB, deterministic from the seed)
./bin/algo-engine gen --model toy-model

# 3. verify everything is wired (config, weights, one full forward pass)
./bin/algo-engine doctor --model toy-model

# 4. chat in the terminal (untrained weights → structured gibberish;
#    watch tok/s, hit-rate and resident experts instead)
./bin/algo-engine chat --model toy-model

# 5. or serve the API + web dashboard and open http://localhost:8080
./bin/algo-engine web --model toy-model
```

Talk to the API like any OpenAI-style endpoint:

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":24}'

curl -N -s localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

curl -s localhost:8080/stats   # hit rate, resident experts, evictions
```

## Testing

```bash
go test ./...          # full suite
go test -race ./...    # REQUIRED before every commit — the store and heat
                       # map are shared mutable state; races are build failures
go vet ./...

# the optional cgo kernel (needs a C compiler; default build never does):
CGO_ENABLED=1 go test -tags asmkernel ./...        # incl. the equivalence guard
CGO_ENABLED=1 go build -tags asmkernel -o bin/algo-engine-accel ./cmd/algo-engine

# benchmark both kernels before trusting either:
go test -bench=. ./internal/quant/
CGO_ENABLED=1 go test -tags asmkernel -bench=. ./internal/quant/
```

Measured on the reference machine at the toy's 128×64 shapes, the cgo kernel
is *slower* than pure Go (cgo call overhead dominates at small matrices) —
the exact negative result the Technical Design anticipated. The tag stays
optional; earn it with numbers at your model's real shapes.

### GPU tier (Phase 4, `-tags cuda`)

The CUDA scaffold ships tag-guarded: `internal/quant/cuda/kernels.cu`
(upload-once resident weights, naive row-per-thread kernels) and
`kernel_cuda.go`. Requires the NVIDIA toolkit — never the default build:

```bash
make cuda                                          # nvcc → static lib → go link
CGO_ENABLED=1 go test -tags cuda ./internal/quant/ # equivalence guard vs the oracle
./bin/algo-engine-cuda doctor --model toy-model
```

The default and `asmkernel` builds are verified in CI; the `cuda` build's
checkpoint requires GPU hardware (its Go side compiles; the equivalence
guard covers it the moment it links). Acceptance bar, in order: identical
output vs the CPU binary first, speed second.

Key tests worth knowing by name:

| Test | Proves |
|---|---|
| `engine.TestTinyCacheStillCorrect` | cache size never changes output (the project invariant) |
| `pipeline.TestForwardDeterministic` | 20 runs under `-race` → bit-identical sums |
| `engine.TestSameSeedSameModelSameOutput` | reproducible model generation end to end |
| `expertstore.TestPinsSurviveEviction` | learned pins outlive LRU churn |
| `quant.TestQ8MatVecMatchesDequantPath` | the quantized kernel equals the float oracle |

## Commands

| Command | Purpose |
|---|---|
| `gen --model DIR [--seed N]` | write a complete toy model directory |
| `chat --model DIR [--session S] [--temp T]` | terminal chat with live tier stats; sessions persist |
| `serve --model DIR [--addr :8080]` | headless OpenAI-compatible API + dashboard |
| `web --model DIR [--addr :8080]` | serve and open the dashboard |
| `doctor --model DIR` | config / weights / forward-pass readiness check |

## Running a real SLM (the `models/` seam)

The engine is family-agnostic behind the `ModelEngine` interface
(`models/models.go`): the toy is one family; **OLMoE** (7B total / 1B
active) is the first real target, with its adapter shipped — verified HF
tensor map, config reader, OLMoE-Instruct chat template, and a planner:

```bash
./bin/algo-engine families            #   olmoe / toy
./bin/algo-engine plan --model /path/to/OLMoE-1B-7B   # needs only config.json
# → dense 0.44 GB RAM-resident · experts 6 GB streamed (6 MB/expert)
#   active set 768 MB/token · suggested RAM tier ~1.5 GB for >90% hits
```

Running OLMoE end-to-end is gated on four bounded engine deltas
(multi-head attention + RoPE with q/k norms, SwiGLU experts, the real
tokenizer, per-group scales) plus executing the converter — Build Manual
Step 18 specifies each; the tiering, pipeline, and surfaces don't change.

## Layout

| Go package | Role |
|---|---|
| `internal/tokenizer` | byte-level BPE |
| `internal/safetensors` | tensor container, ranged reads |
| `internal/quant` | int8 quantization + the kernel/compute-backend seam (Pure Go / cgo / CUDA) |
| `internal/router` | MoE gate (top-k + softmax) |
| `internal/heatmap` | usage counts → `usage.json`, drives the next load's RAM pin set |
| `internal/expertstore` | RAM (LRU+pins) / disk tiers |
| `internal/pipeline` | route→union→place→overlap→learn, as goroutines |
| `internal/attention` + `kvcache` | attention, persisted KV cache |
| `internal/engine` | forward pass, sampling, generate |
| `internal/api` (+ `web/`) | OpenAI-compatible API + embedded dashboard |
| `cmd/algo-engine` | gen · chat · serve · web · doctor · families · plan |

## License

Proposed license: Apache-2.0. Algo Engine is an original implementation —
not a fork, and does not vendor any third-party engine's code.
