# Algo Engine — architecture overview and block diagrams

This is the system-level companion to [`Algo Engine.md`](Algo%20Engine.md)
(the build manual: quick start, commands, testing, license). That file
tells you how to run it; this one explains how it actually works
internally, and how it plugs into the rest of Algo-Kaizen as a real rung
in ALGO AKANADE's model router. No build manual duplicated here — where the
two overlap, this file points at the source file that's the real answer.

## 1. What this is, in one paragraph

**Algo Engine is not an MoE-only engine and not a GPU-only accelerator.**
Those are capabilities inside a broader runtime. Algo Engine is
ALGO AKANADE's self-hosted **model execution and optimization layer**: given a
model, a workload, and the hardware actually available, its job is to
determine — and keep learning — the most efficient way to execute that
model while preserving exact correctness. **ALGO AKANADE decides WHAT to
run; Algo Engine decides HOW to run it** — across CPU, GPU, or hybrid
execution, and across VRAM, RAM, and disk as one tiered memory
hierarchy, streaming only the weights a given token actually needs
rather than requiring the whole model resident. Today's shipped model
family (MoE, via a tiered expert store) is the first, fully real
instance of that general idea, not the ceiling of it — see §10 for the
generalization already underway. It ships an OpenAI-compatible HTTP API
(`/v1/chat/completions`), which is the only thing that matters to the
rest of this repo — it means ALGO AKANADE's model router
(`orchestrator_framework_layer/akanade/backend/app/llm/router.py`) can
treat a running Algo Engine exactly like any other OpenAI-compatible
provider, just pointed at a different `base_url`, with no protocol
translation layer of its own.

## 2. Where this fits in Algo-Kaizen

```
 ALGO AKANADE backend (orchestrator_framework_layer/akanade/backend)
        │
        │  llm/router.py — TierClient.chat(), one of many chain rungs
        ▼
 provs["algo-engine"] = OpenAICompatible("algo-engine-local",
                          settings().ALGO_ENGINE_BASE_URL, "", "algo-engine",
                          requires_key=False)
        │
        │  present in CHAINS["delay_ok"] / CHAINS["bulk"], right after
        │   — .available() gates it out entirely when
        │  ALGO_ENGINE_BASE_URL is unset, same as every other optional
        │  rung; TierClient's fallback chain skips it silently
        ▼
        │  HTTP, OpenAI-compatible — no auth, no SDK, no gateway of its
        │  own; requires_key=False because this binary runs on
        │  infrastructure THIS deployment controls, same trust tier as
        │  Ollama, never a third-party API call
        ▼
 Algo Engine binary — THIS directory, running as `algo-engine serve
 --model <dir> --addr :<port>` (see §4 below for what happens inside)
```

Two things follow directly from `router.py`'s own comments (worth
reading there for the full context, not duplicated here):

- **No API key, ever.** `requires_key=False` — the same reason Ollama
  doesn't need one: this is infrastructure the deployment itself runs,
  not a hosted vendor.
- **Present in the registry even when unconfigured.** `ALGO_ENGINE_BASE_URL`
  defaults to empty; `.available()` returns `False` until an operator
  points it at a real running instance, so this provider costs nothing
  and breaks nothing on a deployment that never uses it — the exact
  same "unavailable rung, gated out, never blocks the chain" story
  every other optional provider in `router.py` follows.
- **No Dockerfile ships this yet.** Unlike Ollama (`ollama/ollama` image)
  or the llama.cpp-based `algo-akanade-server`
  (`deploy/Dockerfile.llama-server`, a DIFFERENT binary for GGUF
  models), this engine has no containerization in `deploy/` today — it
  is meant to be built and run directly (`go build`, then `serve`) on
  whatever host `ALGO_ENGINE_BASE_URL` points at. Containerizing it is
  a straightforward follow-up (static binary, `CGO_ENABLED=0`), not
  something this repo has done yet.

## 3. The GPU story: why this exists at all

`router.py`'s own comment calls this "the local GPU-tier engine (§19)"
— the default build is CPU-only (`CGO_ENABLED=0`, pure Go), but the
same source tree has a real, tag-gated CUDA kernel
(`internal/quant/cuda/kernels.cu` + `kernel_cuda.go`, built via
`make cuda`) that uploads expert weights to the GPU ONCE and reuses the
device pointer on every later call — the beginning of a real VRAM tier,
not a toy. This is the honest, incremental path the engine itself takes
(CPU always works; GPU is an accelerant behind a build tag, verified
against the CPU oracle before it's trusted) — see §6 below for the full
three-tier kernel story.

## 4. Block diagram — one chat request, end to end

```
 POST /v1/chat/completions  (internal/api/api.go)
        │  {model, messages[], max_tokens, temperature, stream}
        ▼
 handleChat() ── stream? ──yes──▶ streamChat() (SSE, same generate loop,
        │no                        chunks pushed as tokens are produced)
        ▼
 Engine.Generate(ctx, messages, maxTokens, temp)     (internal/engine)
        │
        │  render prompt via the model family's ChatTemplate
        │  (models/toy or models/olmoe — family-agnostic above this line)
        ▼
 ┌─────────────────────── per generated token ────────────────────────┐
 │                                                                     │
 │  embed(token) + position                                           │
 │        │                                                           │
 │        ▼                                                           │
 │  for each layer:                                                   │
 │    x += Attention(RMSNorm(x))    ── dense, always RAM-resident      │
 │        │        (internal/attention + internal/kvcache:             │
 │        │         q=Wq·h k=Wk·h v=Wv·h, append to per-session cache, │
 │        │         causal softmax over ALL cached positions, Wo·ctx)  │
 │        ▼                                                           │
 │    x += MoE(RMSNorm(x))          ── routed, tiered store            │
 │        │                                                           │
 │        │  1. ROUTE  (internal/router) — gate scores every expert,   │
 │        │     top-k softmax over just the winners                    │
 │        │  2. UNION  (internal/pipeline) — dedupe requested experts   │
 │        │     across the batch                                       │
 │        │  3. PLACE  (internal/expertstore) — RAM (LRU + learned      │
 │        │     pins) hit, or disk (safetensors ranged read) miss       │
 │        │  4. OVERLAP (internal/pipeline) — one goroutine per expert, │
 │        │     disk streaming bounded by a semaphore, overlapped with  │
 │        │     RAM-hit compute; results summed in FIXED expert-ID      │
 │        │     order regardless of arrival order (bit-identical reruns)│
 │        │  5. LEARN  (internal/heatmap) — every Get() records a       │
 │        │     hit/miss; hottest experts earn permanent RAM residency  │
 │        │     at the NEXT load, persisted to usage.json               │
 │        ▼                                                           │
 │  logits = RMSNorm(x) · embedᵀ   (weight-tied LM head)                │
 │        ▼                                                           │
 │  sample next token (temperature)                                    │
 │        │                                                           │
 │        └──▶ append to KV cache, emit token (SSE chunk if streaming) │
 └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
 chatResponse{choices[0].message.content, finish_reason}
        │  (or the final SSE chunk with finish_reason set, if streaming)
        ▼
 back to ALGO AKANADE's TierClient — indistinguishable from any other
 OpenAI-compatible rung in the chain
```

`GET /stats` reads the SAME heat map and expert-store state live —
hit rate, resident-expert count, evictions — the numbers the embedded
web dashboard (`internal/api/web/index.html`) polls.

## 5. Block diagram — the tiered expert store (the core memory-tiering idea)

This is the part that makes "run a 7B-total-parameter MoE model on a
laptop" real rather than aspirational: never load every expert, only
the ones a token actually routes to, and remember which ones matter.

```
                     Get(layer, expertID)
                            │
                            ▼
                  ┌─────────────────────┐
                  │   RAM tier (LRU +    │◀── learned pins from
                  │   learned pin set)   │    heatmap.Hottest(N),
                  │                      │    applied at model LOAD,
                  └──────────┬───────────┘    survive normal LRU churn
                        hit  │  miss
                  ┌──────────┴───────────┐
                  ▼                      ▼
          dequantize int8→f32    stream from disk
          (identical scale,      (safetensors ranged
           identical bytes           ReadAt — the whole
           as the disk copy)         point of the container
                  │                  format: any tensor is
                  │                  one seek away)
                  │                      │
                  │                bounded by a semaphore
                  │                (disk concurrency capped,
                  │                 not one goroutine per
                  │                 expert hammering disk)
                  │                      │
                  └──────────┬───────────┘
                             ▼
                    heatmap.Record(hit/miss)
                    — read on every Get, written to
                      usage.json, drives NEXT load's
                      pin set
                             │
                             ▼
                  identical Expert{W1, W2} either way

  THE INVARIANT, under test (engine.TestTinyCacheStillCorrect):
  a capacity-1 cache and a capacity-64 cache produce BIT-IDENTICAL
  output. Placement — RAM vs disk, pinned vs evicted — may only ever
  change SPEED. The numbers a token sees never depend on which tier
  answered.
```

## 6. Block diagram — the kernel seam (where C/CUDA is allowed in)

The "One Rule" this codebase names explicitly: Go everywhere, and
exactly one sanctioned seam for something faster, gated by build tags,
and never trusted until it's proven bit-identical to the pure-Go
answer.

```
 quant.Active() — the one function every hot-path matmul calls
        │
        ├── default build (no tags, CGO_ENABLED=0)
        │     └─▶ Pure{} — pure Go matvec. Fully static binary.
        │         Always correct by construction; the oracle every
        │         other tier is checked against.
        │
        ├── -tags asmkernel (needs a C compiler)
        │     └─▶ cgo matvec, `-O3 -march=native` (autovectorized:
        │         AVX2/NEON). Gated by equiv_test.go — CI fails the
        │         merge if this doesn't match Pure{} within 1e-4.
        │         Measured result on the toy's shapes: SLOWER than
        │         pure Go (cgo call overhead dominates at small
        │         matrices) — kept anyway as the honest negative
        │         result the design doc predicted, not deleted to
        │         hide it.
        │
        └── -tags cuda (needs the NVIDIA toolkit + `make cuda`)
              └─▶ kernels.cu, upload-once resident weights (first
                  MatVec against a tensor uploads it to the GPU and
                  caches the device pointer; every later call reuses
                  it — real tier residency, not a shuttle). Same
                  equivalence guard as asmkernel; verified in CI on
                  the Go side, the GPU checkpoint itself needs real
                  hardware. Acceptance order: identical output vs
                  the CPU binary FIRST, speed second — never the
                  other way around.

 Exactly one Active() exists in any given build — the three files
 above are mutually exclusive at compile time via their //go:build
 tags. Deleting kernel_accel.go/kernel_cuda.go and rebuilding
 default still produces a fully working engine — that's the point
 of a seam, not a hard dependency.
```

## 7. Package layout

| Go package | Role | Real mechanism (not a stub) |
|---|---|---|
| `internal/tokenizer` | byte-level BPE | 256 byte tokens + learned merge table; degrades to plain byte tokenization with an empty table |
| `internal/safetensors` | tensor container | `[8B length][JSON header][raw data]`; ranged `ReadAt` per tensor — the format streaming is built on |
| `internal/quant` | int8 quantization + kernel seam | symmetric per-tensor int8, scale = max\|w\|/127; see §6 for the three kernel tiers |
| `internal/router` | MoE gate | linear scorer → top-k → softmax over just the winners; weights double as the expert blend ratio |
| `internal/heatmap` | usage → learned pins | per-(layer,expert) hit/miss counts, persisted to `usage.json`, drives next load's RAM pin set |
| `internal/expertstore` | tiered cache | RAM (LRU + pins) / disk (safetensors), semaphore-bounded disk concurrency — see §5 |
| `internal/pipeline` | the 5-step per-token path | route → union → place → overlap → learn, as goroutines; fixed summation order for bit-identical reruns |
| `internal/attention` + `internal/kvcache` | attention + memory | single-head causal attention; KV cache persists per-session to disk for warm reopen |
| `internal/engine` | forward pass + generation | wires every package above into `Generate()`; owns `ModelConfig`/model loading |
| `internal/api` (+ `api/web/`) | HTTP surface | `/v1/chat/completions` (normal + SSE), `/stats`, embedded dashboard — see §4 |
| `models/` (`toy`, `olmoe`) | the multi-family seam | `ModelEngine` interface — everything above this line is family-agnostic; see §8 |
| `cmd/algo-engine` | the one binary | `gen · chat · serve · web · doctor · families · plan` |

## 8. Model families: toy today, OLMoE next

The engine is family-agnostic behind `models.ModelEngine`
(`models/models.go`) — the toy model exists purely so the full pipeline
(routing, quantization, streaming, caching, the API) runs in
milliseconds and is fully testable without needing real weights
anywhere. `models/olmoe` is the adapter for the first REAL target
(AllenAI's OLMoE, 7B total / 1B active) — its own file header states
its status plainly, worth quoting rather than paraphrasing:

> Honest status: the mapping, template, and planner are complete and
> tested. Executing the conversion and running the result needs four
> engine deltas: multi-head attention + RoPE with q/k norms, SwiGLU
> experts (three matrices), the real tokenizer via `LoadMerges`,
> per-group scales. `Generate()` says so rather than pretending.

`./bin/algo-engine plan --model /path/to/OLMoE-1B-7B` already runs
today against just a checkpoint's `config.json` — no weights needed —
and prints a real memory plan (dense RAM-resident size, streamed expert
total, active-set-per-token size, suggested RAM tier for >90% hits).
That planner, the tensor map, and the chat template are the reusable,
already-correct 80% of the work; the four deltas above are what's left
before OLMoE actually generates real text through this engine.

## 9. Testing philosophy (why the test names matter)

Every tier in §5 and §6 exists to change SPEED, never OUTPUT — the test
suite is built to catch a violation of that specific invariant, not
just "does it crash":

| Test | What it actually proves |
|---|---|
| `engine.TestTinyCacheStillCorrect` | a capacity-1 expert cache and a capacity-64 cache produce bit-identical output |
| `pipeline.TestForwardDeterministic` | 20 runs under `-race` → bit-identical sums (goroutine arrival order doesn't leak into the answer) |
| `quant.TestQ8MatVecMatchesDequantPath` | the quantized kernel equals the float oracle |
| `quant/equiv_test.go` (asmkernel/cuda only) | an accelerated kernel matches the pure-Go oracle within 1e-4 — CI blocks a merge that fails this, regardless of benchmark numbers |
| `expertstore.TestPinsSurviveEviction` | a learned pin outlives normal LRU churn |
| `engine.TestSameSeedSameModelSameOutput` | model generation is reproducible end to end from a seed |

`go test -race ./...` is called out in the build manual as REQUIRED
before every commit specifically because the expert store and heat map
are shared mutable state accessed from per-expert goroutines — a data
race here wouldn't just be a bug, it would be a silent violation of the
one invariant this whole design exists to guarantee.

## 10. Beyond MoE — the generalization roadmap

§1's positioning ("ALGO AKANADE decides WHAT to run; Algo Engine decides HOW")
is the target architecture, not yet fully built — per the project's own
non-negotiable rule, that means extending what's here, not rebuilding
it. Concretely, in order:

1. **`ComputeBackend`/`DeviceBackend` seam** — today's `quant.Kernel`
   (§6) already IS this seam in miniature (one interface, three mutually-
   exclusive build-tagged implementations, an equivalence guard gating
   every accelerated tier). Generalizing it means widening the interface
   from `MatVec`/`MatVecQ8` to the fuller op set a dense or hybrid model
   needs (attention, RMSNorm, RoPE, activation) — same pattern, more
   methods, same "Go is the default, C/CUDA earn their place through
   benchmarks" rule §6 already enforces.
2. **Dense and hybrid model families**, alongside MoE — `models.ModelEngine`
   (§8) is already the family-agnostic seam; a dense family (no router,
   no expert store — every layer just always resident or always tiered,
   the same RAM/disk placement machinery §5 already has minus the
   routing step) is the next family to add there, not a new engine.
3. **A generic tensor/model memory manager** — §5's expert store
   generalizes from "MoE experts, RAM vs. disk" to "any model's tensors,
   VRAM vs. RAM vs. disk," so a dense model's layers can be placed the
   same way an MoE model's experts are today.
4. **An execution planner** — estimate compute/transfer/sync/cache-miss
   cost per candidate strategy (CPU/GPU/hybrid, which layers/experts go
   where) and pick the cheapest one that fits the hardware, instead of
   today's fixed "whichever kernel tier this binary was built with."
5. **RL over execution strategy** — the same `LinearBanditCore`
   Thompson-Sampling primitive ALGO AKANADE's own optimizers already use
   (`framework/bandit.py`), applied to a different decision: not which
   provider or generation-parameter arm, but which execution strategy
   (device placement, prefetch, eviction, batch size) actually performed
   best for a given model+hardware+workload, learned from real telemetry.

Every one of these is additive to the code already described in §4–§9,
verified as it lands the same way — build clean, `go test -race ./...`
clean, equivalence-tested against the existing pure-Go oracle before any
new accelerated path is trusted.

## 11. See also

- [`Algo Engine.md`](Algo%20Engine.md) — quick
  start, every CLI command, build/test commands including the cgo and
  CUDA tags, license.
- `orchestrator_framework_layer/akanade/backend/app/llm/router.py` — the
  `algo-engine` provider entry and `ALGO_ENGINE_BASE_URL` (§2 above).
- `orchestrator_framework_layer/akanade/README_ARROW_DIAGRAMS.md` — the
  equivalent diagram set for ALGO AKANADE's own request path, same visual
  convention used in §4–§6 above.
