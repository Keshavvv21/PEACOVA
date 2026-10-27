# CUDA/GPU Optimization for Training and Serving LLMs

Every other README in this series — quantization, attention optimizations, memory optimization, distributed training, inference serving — is ultimately about the same underlying resource: getting a GPU to spend as much of its time as possible doing useful floating-point math, instead of waiting on memory or sitting idle between kernel launches. This document is the hardware/execution-model foundation those documents build on. It explains *why* GPUs are the right tool for LLMs in the first place, how a GPU actually executes a program, why the memory hierarchy — not raw FLOPs — is usually the real bottleneck, and how techniques like kernel fusion and FlashAttention exploit that hierarchy.

Nothing here requires you to write CUDA. The goal is to build the mental model that explains why `torch.compile` helps, why one attention kernel is 3x faster than another with identical math, why batch size 1 wastes 95% of an H100, and why NVLink matters for multi-GPU training.

## Table of Contents

1. [Why GPUs, Not CPUs, for Deep Learning](#1-why-gpus-not-cpus-for-deep-learning)
2. [The CUDA Execution Model](#2-the-cuda-execution-model)
3. [The GPU Memory Hierarchy](#3-the-gpu-memory-hierarchy)
4. [Compute-Bound vs Memory-Bound: The Roofline Model](#4-compute-bound-vs-memory-bound-the-roofline-model)
5. [Kernel Fusion](#5-kernel-fusion)
6. [FlashAttention: Fusion Applied to Attention](#6-flashattention-fusion-applied-to-attention)
7. [Mixed Precision and Tensor Cores](#7-mixed-precision-and-tensor-cores)
8. [Batching and GPU Utilization](#8-batching-and-gpu-utilization)
9. [Multi-GPU Communication](#9-multi-gpu-communication)
10. [What an Application Engineer Actually Needs to Know](#10-what-an-application-engineer-actually-needs-to-know)

---

## 1. Why GPUs, Not CPUs, for Deep Learning

### CPUs are latency-optimized; GPUs are throughput-optimized

A modern server CPU has somewhere between 8 and 128 cores. Each core is enormous by silicon standards: deep out-of-order execution pipelines, branch predictors, speculative execution, large per-core L1/L2 caches, and sophisticated prefetchers. All of that machinery exists to make a *single sequential stream of instructions* run as fast as possible, including code full of unpredictable branches, pointer chasing, and irregular memory access. This is the right design for an OS kernel, a web server, or a database query planner — code where each instruction is different from the last and depends on the result of the previous one.

A GPU inverts the trade-off entirely. An NVIDIA H100 has on the order of 16,896 CUDA cores. Each individual core is primitive compared to a CPU core — no branch prediction worth mentioning, no out-of-order execution, a fraction of the control logic. What the GPU spends its transistor budget on instead is raw arithmetic units and the scheduling hardware to keep thousands of them busy simultaneously, executing *the same instruction* on different pieces of data. This is SIMT: **Single Instruction, Multiple Threads**. One instruction fetch/decode drives many parallel execution lanes.

| | CPU (e.g., server Xeon/EPYC) | GPU (e.g., H100) |
|---|---|---|
| Core count | 8–128 | ~16,900 CUDA cores + 528 Tensor Cores |
| Core design | Complex, out-of-order, branch-predicting | Simple, in-order, minimal control logic |
| Optimized for | Low latency on one instruction stream | High throughput across many identical streams |
| Cache per core | Large (MBs of L2/L3 per core) | Small, shared across many threads |
| Peak fp32 throughput | ~1–3 TFLOPS | ~60–67 TFLOPS (non-Tensor-Core) |
| Peak bf16 (Tensor Core) | N/A | ~990 TFLOPS (dense), ~1979 with sparsity |
| Best at | Branchy, sequential, irregular-memory code | Regular, data-parallel, arithmetic-heavy code |

### Why matrix multiplication is the perfect fit

The dominant operation inside a transformer — attention projections, the QKV computation, the feed-forward layers, the output head — is matrix multiplication. A matmul `C = A @ B` where `A` is `(M, K)` and `B` is `(K, N)` requires computing `M * N` output elements, and each output element `C[i,j] = sum_k A[i,k] * B[k,j]` is an independent dot product. Every one of those `M*N` dot products:

- Reads from the same two input matrices (no dependency between output elements)
- Performs the exact same sequence of operations (multiply-accumulate, repeated `K` times)
- Writes to a distinct location with no data dependency on any other output element

This is "embarrassingly parallel": there is no sequential chain of logic, no branching that differs between elements, and the same instruction (fused multiply-add) is issued over and over on different data. That is exactly what SIMT hardware wants. A GPU can assign different output elements (or tiles of output elements) to different threads and have thousands of them compute in lockstep. A CPU, by contrast, has to time-slice this same work across a much smaller number of cores, and most of its per-core sophistication (branch prediction, speculation) goes unused because there are no branches to predict.

This is why an A100/H100 can outperform a top-end CPU on a large matmul by two to three orders of magnitude, and why every major deep learning framework (PyTorch, TensorFlow, JAX) dispatches the compute-heavy parts of a model to GPU kernels while the CPU is relegated to orchestration: reading data, launching kernels, moving tensors, running Python control flow.

---

## 2. The CUDA Execution Model

To understand *why* GPU code performs the way it does — and why certain code patterns are disastrous — you need the execution hierarchy CUDA exposes.

### Threads → Warps → Thread Blocks → Grids

```
Grid
 └─ Thread Block 0        Thread Block 1        Thread Block 2   ...
     └─ Warp 0 (32 threads, lockstep)
     └─ Warp 1 (32 threads, lockstep)
     └─ Warp 2 (32 threads, lockstep)
     ...
```

- **Thread**: the smallest unit of execution. A CUDA kernel is written as the code for *one* thread; you then launch millions of them.
- **Warp**: a group of exactly **32 threads** that the hardware schedules and executes together, physically, in lockstep — one instruction fetched and issued for all 32 lanes at once. The warp is the real atomic unit of scheduling on NVIDIA GPUs; threads don't run independently, warps do.
- **Thread block**: a programmer-defined group of threads (up to 1024) that is guaranteed to run on the *same* Streaming Multiprocessor (SM) and can cooperate through fast on-chip shared memory and barrier synchronization (`__syncthreads()`). A block is made up of some whole number of warps.
- **Grid**: the full set of thread blocks launched by one kernel invocation, covering the entire problem (e.g., every output element of a matmul).

### The Streaming Multiprocessor (SM)

The SM is the physical execution unit on the chip — an H100 has 132 of them. Each SM contains:

- Multiple warp schedulers (able to issue instructions from several warps per cycle)
- A pool of CUDA cores (fp32/int32 ALUs) and Tensor Cores
- A register file, partitioned across all resident threads
- A block of fast on-chip memory split between L1 cache and programmer-managed shared memory

When you launch a kernel, the GPU's scheduler distributes thread blocks across the available SMs. Each SM can hold multiple thread blocks (and therefore many warps) *resident* simultaneously, up to limits set by register and shared-memory usage. Crucially, the SM doesn't need all its resident warps to be doing useful work at every cycle — while one warp is stalled waiting on a slow memory load, the warp scheduler switches, for free (no context-switch overhead like a CPU), to another resident warp that's ready to issue an instruction. This is **latency hiding**: the GPU tolerates the hundreds-of-cycles latency of a global memory access by having enough other warps in flight to keep the arithmetic units busy. This is also why occupancy (how many warps are resident per SM relative to the hardware maximum) is a first-order performance knob — too few resident warps and the SM has nothing to switch to during a stall.

### Warp divergence: why branches are expensive

Because a warp is *one instruction stream* physically fanned out across 32 lanes, all 32 threads execute the same instruction every cycle. If your kernel contains a data-dependent branch —

```cuda
if (threadIdx.x % 2 == 0) {
    do_a();
} else {
    do_b();
}
```

— and different threads *within the same warp* take different branches, the hardware cannot actually run both paths concurrently on the same 32-wide unit. Instead it serializes: it executes `do_a()` for the threads that took that path while masking off (disabling) the others, then executes `do_b()` for the remaining threads while masking off the first group. The warp's effective throughput is cut in proportion to how many divergent paths exist — a two-way divergence roughly halves throughput for that warp, a four-way divergence quarters it, and so on. This is **warp divergence**, and it is one of the most common ways naive CUDA code (or naive kernel-writing in Triton) loses performance.

The practical implication for LLM workloads: this is part of why the core transformer math (dense matmuls, elementwise activations, layer norm) maps so well to GPUs — it's branch-free, uniform work applied identically across every element. It's also why ragged/variable-length sequence handling, top-k sampling, or MoE (mixture-of-experts) routing — where different tokens in the same batch take genuinely different computational paths — are much harder to make GPU-efficient, and why serving frameworks invest real engineering (padding strategies, expert-parallel routing, custom kernels) specifically to avoid divergence and wasted lanes in those cases.

---

## 3. The GPU Memory Hierarchy

Modern GPUs are, for almost all real LLM workloads, not limited by how many FLOPs the arithmetic units can issue — they're limited by how fast data can be moved to where those arithmetic units can use it. Understanding the memory hierarchy is the single most important thing for reasoning about GPU performance.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         GPU MEMORY HIERARCHY                         │
│                                                                        │
│   REGISTERS                                                          │
│   ┌────┐  per-thread, ~256 KB total per SM (H100)                    │
│   │ ▓▓ │  Bandwidth: ~effectively unlimited (part of the ALU)        │
│   └────┘  Latency:   ~1 cycle                                        │
│      │    Scope:     single thread, not shared                       │
│      ▼                                                                │
│   SHARED MEMORY / L1 CACHE                                           │
│   ┌──────┐  per-SM, ~228 KB configurable split (H100), on-chip SRAM  │
│   │ ▓▓▓▓ │  Bandwidth:  ~19–33 TB/s aggregate across all SMs         │
│   └──────┘  Latency:    ~20–30 cycles                                │
│      │      Scope:      threads within one block, programmer-managed│
│      ▼                                                                │
│   L2 CACHE                                                            │
│   ┌────────┐  chip-wide, ~50 MB (H100), on-chip but off-SM           │
│   │ ▓▓▓▓▓▓ │  Bandwidth:  a few TB/s effective, shared by all SMs    │
│   └────────┘  Latency:    ~200 cycles                                │
│      │        Scope:      all SMs, hardware-managed cache             │
│      ▼                                                                │
│   GLOBAL MEMORY (HBM)                                                 │
│   ┌────────────┐  off-chip, 80 GB (H100 SXM)                         │
│   │ ▓▓▓▓▓▓▓▓▓▓ │  Bandwidth:  ~3.35 TB/s (H100 SXM, HBM3)            │
│   └────────────┘  Latency:    ~400–800 cycles                        │
│                    Scope:      entire GPU, all kernels, persists     │
│                                between kernel launches                │
│                                                                        │
│   size:      registers < shared mem/L1 << L2 << HBM                  │
│   speed:     registers > shared mem/L1 >> L2 >> HBM                  │
└──────────────────────────────────────────────────────────────────────┘
```

(Numbers above are order-of-magnitude, using H100 SXM as a representative reference point; exact figures vary by GPU model and generation, but the *shape* of the hierarchy — several orders of magnitude of size difference paired with roughly inverse orders of magnitude of speed difference — is a structural property of all modern GPUs, not a spec particular to one chip.)

Walking down the hierarchy:

- **Registers** are the fastest possible storage — private to a single thread, accessed essentially at the speed of computation itself. Every intermediate value a thread works with ideally lives here. There are a limited number per thread; use too many and the compiler "spills" to slower memory, or the SM can host fewer concurrent threads (hurting occupancy).
- **Shared memory** is on-chip SRAM local to one SM, explicitly managed by the programmer (in raw CUDA, via `__shared__` arrays). All threads in a block can read and write it, making it the mechanism for cooperation — e.g., one thread loads a value from global memory into shared memory once, and 31 other threads in the block reuse it instead of each re-fetching it from HBM. The same physical SRAM also backs the L1 cache; a portion of it (a "carveout") is often configurable between the two. It is roughly two orders of magnitude smaller than global memory but roughly an order of magnitude (or more) faster to access.
- **L2 cache** is a single cache shared by all SMs on the chip. It automatically (no programmer control) holds recently-accessed global memory data, and can serve as a fast path for data reused across different SMs or between successive kernel launches, without going all the way to HBM.
- **Global memory (HBM — High Bandwidth Memory)** is the large, off-chip DRAM that holds model weights, activations, the KV cache, and any tensor too big for on-chip memory. It's "large but slow" only *relative to on-chip memory* — 3+ TB/s is enormous compared to a CPU's DRAM (tens of GB/s), but it is still 10-20x slower than shared memory/L1 and vastly slower than registers, and every access costs hundreds of cycles of latency (hidden, when possible, by having other warps ready to execute).

The key practical consequence: any algorithm that has to repeatedly round-trip data through global memory — write an intermediate result out, then immediately read it back in for the next step — pays the HBM bandwidth/latency tax over and over, even though the GPU's arithmetic units are sitting there capable of doing far more work per second than the memory system can feed them. This single fact motivates nearly every "kernel-level" optimization technique used in modern deep learning, including FlashAttention (Section 6).

---

## 4. Compute-Bound vs Memory-Bound: The Roofline Model

Whether a given operation is limited by the GPU's arithmetic throughput or by its memory bandwidth is not a matter of opinion — it's determined by a single ratio: **arithmetic intensity**, defined as

```
Arithmetic Intensity (FLOPs/byte) = Total floating-point operations performed
                                     ─────────────────────────────────────────
                                     Total bytes moved to/from global memory
```

The **roofline model** plots this conceptually: achievable performance (FLOPs/s) on the y-axis against arithmetic intensity on the x-axis. There are two regimes:

```
 Achievable
 FLOPs/s
    │                              ______________________  ← compute roof
    │                          ⟋                            (peak TFLOPS,
    │                      ⟋                                 fixed by silicon)
    │                  ⟋
    │              ⟋      "compute-bound" region
    │          ⟋          (arithmetic intensity high enough
    │      ⟋               to saturate the ALUs)
    │  ⟋
    │⟋  "memory-bound" region
    │   (performance limited by
    │    memory bandwidth; the line's
    │    slope = peak bandwidth, GB/s)
    └──────────────────────────────────────────────────
                Arithmetic Intensity (FLOPs/byte)
```

- In the **memory-bound** region (low arithmetic intensity — few FLOPs performed per byte fetched), performance is capped by how fast data can be streamed from HBM, regardless of how many idle ALUs are sitting on the chip. Making the ALUs faster does nothing here; the only way to go faster is to move less data or move it faster.
- In the **compute-bound** region (high arithmetic intensity — many FLOPs reused per byte fetched), performance is capped by the chip's peak arithmetic throughput, and the memory system has slack. Here, using faster math units (e.g., Tensor Cores, lower precision) directly buys speed.

**Where LLM operations fall:**

- A large, well-tiled matmul (e.g., the feed-forward layers, `Q @ K^T` for long enough sequences) has high arithmetic intensity — each element loaded from memory is reused many times across the `K` reduction dimension — and tends to be **compute-bound**. This is where Tensor Cores and mixed precision give the most direct benefit.
- Elementwise operations (activation functions, bias adds, layer normalization, dropout, softmax by itself) do a small, fixed number of FLOPs per element but still have to read that element in and write a result out. Arithmetic intensity is very low, so these are almost always **memory-bound** — the GPU spends most of its time waiting on HBM traffic, not computing.
- Naive (unfused) attention is a particularly bad case: it involves several of these low-intensity elementwise/reduction steps (masking, softmax, dropout) chained together, each one reading and writing the *entire* `n × n` attention matrix to global memory. Even though the underlying math (two matmuls plus a softmax) is not enormous in FLOP count for moderate sequence lengths, the memory traffic explodes as `O(n^2)`, which is exactly why attention becomes the dominant cost as sequence length grows, and why it was the first place framework authors invested in a fused, memory-aware kernel (Section 6).

This is the concept that ties together most of the other optimization techniques discussed elsewhere in this series: quantization (see `docs/ml-concepts/15-quantization/README.md`) increases arithmetic intensity by shrinking the bytes-moved term; the KV cache and continuous batching (see `docs/ml-concepts/17-inference-serving-optimization/README.md`) exist because decoding one token at a time is a severely memory-bound regime (you reload the entire model's weights from HBM per token generated) and batching amortizes that weight traffic across more useful compute.

---

## 5. Kernel Fusion

A **CUDA kernel** is a single GPU function launched from the host (CPU). Every kernel launch has fixed overhead — typically a few microseconds for the launch itself, plus, critically, every kernel that isn't the very first or last step in a pipeline has to **read its inputs from global memory and write its outputs back to global memory**, because global memory is the only storage that persists and is visible across separate kernel launches (registers and shared memory are scoped to the threads/blocks of a single kernel invocation and are gone the instant it ends).

Consider a simple sequence: elementwise multiply, then add a bias, then apply a GELU activation — three logically separate operations. An **unfused** implementation (e.g., naive eager-mode PyTorch, one op per line) launches three kernels:

```
Unfused (3 kernels, 3 launches, 4 full round-trips to HBM):

  Kernel 1: multiply
    read  x from HBM ──┐
    read  y from HBM ──┼─► compute x*y ──► write tmp1 to HBM
                        ┘
  Kernel 2: add bias
    read  tmp1 from HBM ──┐
    read  bias from HBM ──┼─► compute tmp1+bias ──► write tmp2 to HBM
                           ┘
  Kernel 3: GELU
    read  tmp2 from HBM ──► compute gelu(tmp2) ──► write out to HBM

  Total global memory traffic: reads(x,y,tmp1,bias,tmp2) + writes(tmp1,tmp2,out)
                                = 5 reads + 3 writes of full tensors, plus
                                  3 separate kernel launch overheads
```

```
Fused (1 kernel, 1 launch, 1 round-trip to HBM):

  Kernel: fused multiply-add-gelu
    read  x    from HBM ──┐
    read  y    from HBM ──┤
    read  bias from HBM ──┼─► compute x*y in registers
                           │   ──► add bias in registers
                           │   ──► apply gelu in registers
                           ┘   ──► write out to HBM (once)

  Total global memory traffic: 3 reads + 1 write, 1 kernel launch overhead
```

The fused version does the *exact same arithmetic* — it hasn't reduced FLOPs at all — but it has eliminated the two intermediate tensors (`tmp1`, `tmp2`) ever touching HBM. Each intermediate value is produced in a register, immediately consumed by the next operation while still in the register, and only the final result is written back out. Since these operations are exactly the low-arithmetic-intensity, memory-bound kind described in Section 4, and their cost is dominated by HBM traffic rather than compute, cutting the memory traffic by roughly 2-3x (in this example) translates almost directly into a proportional wall-clock speedup. This is precisely the mechanism behind `torch.compile` / `torch.jit` graph fusion, XLA's fusion passes, and hand-written fused kernels like NVIDIA's fused LayerNorm or fused AdamW: the compiler or kernel author identifies a chain of elementwise/memory-bound ops and rewrites them as one kernel so intermediates never leave on-chip storage.

### The attention example, informally

The clearest large-scale illustration of this idea is attention. A naive implementation computes it as three separate steps, each its own kernel (or several):

1. `S = Q @ K^T` — write the full `(seq_len, seq_len)` score matrix `S` to HBM.
2. `P = softmax(S)` (with masking) — read `S` back from HBM, write `P` (same shape) back to HBM.
3. `O = P @ V` — read `P` back from HBM, compute the output.

For a sequence length `n`, materializing `S` and `P` costs `O(n^2)` memory — for `n = 8192` and fp16, that's a 128M-element matrix, over 250 MB, written and then re-read *twice* per attention head, per layer. This is why naive attention becomes catastrophically memory-bound at long context lengths even though the underlying FLOP count (`O(n^2 * d)`) is comparatively modest — the GPU spends most of its cycles waiting on HBM traffic for a matrix that gets used once and thrown away. Section 6 covers the fused solution.

---

## 6. FlashAttention: Fusion Applied to Attention

FlashAttention is the canonical, most-cited real-world proof that the fusion principle in Section 5 matters enormously at scale. This section covers it at the conceptual level needed to understand *why* it works as a memory-hierarchy optimization; the full algorithmic derivation, the backward-pass recomputation trick, and its variants (FlashAttention-2/3, sliding-window and grouped-query interactions) are covered in `docs/ml-concepts/18-attention-optimizations/README.md` — this section is intentionally the "GPU architecture" half of that story.

### The core idea: never materialize the full attention matrix

Instead of computing the full `Q @ K^T` score matrix, softmax-ing it, and then multiplying by `V` as three separate global-memory round-trips, FlashAttention restructures the computation as a single fused kernel using two techniques:

**1. Tiling.** `Q`, `K`, and `V` are split into blocks small enough that a block of each fits into the SM's fast on-chip shared memory (SRAM) at once, alongside the accumulating output block. The kernel loops over `K`/`V` blocks; for each one, it loads that block into shared memory, computes the partial score matrix for just that tile (a small `(block_size, block_size)` piece, not the full `n × n` matrix), and immediately consumes it to update a running output accumulator — all without that partial score tile ever being written to HBM. Only `Q`, `K`, `V` (inputs) and `O` (the final output) ever touch global memory; the `n × n` intermediate is never materialized there at all.

**2. Online softmax.** The obstacle to tiling naively is that softmax normalization needs the sum (and, for numerical stability, the max) across the *entire* row of the score matrix before you can produce a correctly normalized output — information you don't have yet if you're only looking at one tile of `K`/`V` at a time. FlashAttention solves this with an online (running/streaming) softmax: it keeps a running maximum and a running sum for each query row as it processes `K`/`V` blocks one at a time, and *rescales* the partial output accumulated so far whenever a new block reveals a larger max than seen previously. This is mathematically exact (not an approximation) — by the end of the loop over all `K`/`V` blocks, the accumulated, rescaled output equals exactly what you'd get from computing the full softmax over the whole row at once. The trick is that it only ever needs `O(block_size)` memory for the running statistics, never the full row.

Conceptually:

```
Naive attention:
  for each layer/head:
     S = Q @ K^T                    → write O(n²) to HBM
     P = softmax(S)                 → read + write O(n²) to HBM
     O = P @ V                      → read O(n²) from HBM
  Total HBM traffic ~ O(n²)  (dominant cost, even though FLOPs ~ O(n²·d))

FlashAttention:
  for each layer/head:
     for each block of K, V (loaded into SRAM):
        compute S_block = Q_block @ K_block^T   (stays in SRAM/registers)
        update running max, running sum          (online softmax, in SRAM)
        rescale + accumulate O_block += ...       (stays in SRAM/registers)
     write final O_block to HBM once
  Total HBM traffic ~ O(n)   (only Q, K, V, O — never the n×n matrix)
```

By turning an `O(n^2)`-memory-traffic operation into an `O(n)`-memory-traffic one (while keeping the FLOP count essentially unchanged), FlashAttention converts attention from a severely memory-bandwidth-bound operation into one that's much closer to compute-bound — and because it also avoids ever allocating the full `n × n` matrix, it dramatically cuts peak memory usage, which is what allows much longer context lengths to fit at all. This is the reason FlashAttention is not a "nice speedup" but close to a strict requirement for any modern long-context LLM: the naive quadratic-memory-traffic version simply doesn't scale to the sequence lengths current models support, both in wall-clock time and in GPU memory capacity.

---

## 7. Mixed Precision and Tensor Cores

### Why dedicated hardware for lower precision

Since Volta (V100, 2017), NVIDIA GPUs have included **Tensor Cores**: specialized execution units, separate from the general-purpose CUDA cores, that perform small fused matrix-multiply-accumulate operations (e.g., a 4x4 or larger tile MMA) directly in hardware for reduced-precision formats — originally fp16, and now also bf16, tf32, fp8, and int8. Because these formats use fewer bits per value, more of them fit into the same register/cache footprint and more multiply-accumulate lanes can be packed into the same silicon area, so Tensor Cores deliver substantially higher throughput for matmuls than the general fp32 CUDA cores — on an H100, dense bf16 Tensor Core throughput (~990 TFLOPS) is roughly 15x the fp32 CUDA-core throughput (~67 TFLOPS). Since matmuls dominate both training and inference compute (Section 1), routing them through Tensor Cores instead of standard fp32 ALUs is one of the single largest available speedups, and it directly increases arithmetic intensity's payoff too — lower-precision data also means fewer bytes moved per element from HBM, which helps the memory-bound side of the equation as well.

### fp16/bf16 in training: the underflow problem and how it's handled

Naively running an entire training loop in fp16 breaks because fp16 has a very narrow dynamic range (its exponent has only 5 bits, versus 8 for fp32). Gradients in deep networks, especially early in training or in shallow layers, are frequently very small — smaller than fp16 can represent at all — and fp16 gradients will silently **underflow to zero**, at which point that part of the network stops learning.

**Mixed precision training** solves this with three cooperating techniques:

1. **fp32 master weights.** A full-precision (fp32) copy of every parameter is kept as the "true" weight. The forward and backward passes run in fp16/bf16 (fast, Tensor-Core-accelerated), but the optimizer step (computing the actual weight update) is applied to the fp32 master copy, then a fp16/bf16 copy is cast out for the next forward pass. This avoids the update itself being lost to rounding error, since a fp16 update added to a fp16 weight can easily round away to nothing when the update is much smaller than the weight's own magnitude.
2. **Loss scaling.** Before backpropagation, the loss value is multiplied by a scale factor (commonly starting somewhere in the thousands, e.g., 2^16, and adjusted dynamically). Since gradients scale linearly with the loss through the chain rule, this shifts the entire gradient distribution upward into fp16's representable range, well clear of the underflow threshold, without changing the mathematical result. After the backward pass computes scaled gradients, they're divided back down by the same scale factor before the optimizer step (working in fp32 at that point). Dynamic loss scaling monitors for `inf`/`NaN` gradients (a sign the scale is too aggressive), and backs the scale factor off when they occur, then cautiously increases it again over time to stay as close to the representable ceiling as possible.
3. **bf16 as a simpler alternative.** bf16 (bfloat16) uses the same 8-bit exponent width as fp32 (just a shorter, 7-bit mantissa), so it has fp32's dynamic range but fp16's memory/throughput footprint. This mostly eliminates the underflow problem that motivates loss scaling, at the cost of less numerical precision per value (more rounding error, but not underflow-to-zero). Most large-model training today (LLaMA, GPT-family, etc.) uses bf16 for exactly this reason — it needs less careful numerical babysitting than fp16 while still getting full Tensor Core throughput.

### int8/fp8 in inference

Serving doesn't backpropagate, so the failure modes differ (no vanishing-gradient-style underflow to worry about), but the goal is the same: push data into a format Tensor Cores execute fastest while also shrinking memory traffic and footprint. int8 and, on Hopper/Blackwell-class GPUs, fp8 are increasingly used for weights and/or activations, with calibration or per-channel scaling factors keeping the reduced range from clipping outliers. The mechanics of *how* weights get quantized to these formats (GPTQ, AWQ, per-channel scales, calibration datasets) are covered in `docs/ml-concepts/15-quantization/README.md`; the point here is that it's the same Tensor Core hardware and the same "fewer bits → more throughput, less HBM traffic" argument, applied at inference time instead of training time.

---

## 8. Batching and GPU Utilization

Section 1 established that a GPU's advantage comes from having thousands of parallel lanes available. Processing a single request — one sequence, batch size 1 — uses only a tiny fraction of those lanes for anything close to their capacity, for a simple reason: at batch size 1, a matmul like `x @ W` has `x` as a single vector (or a short sequence of vectors for prefill), so the amount of independent, parallel work available to spread across the SMs is small relative to the number of SMs and threads the chip has. During single-token autoregressive decoding in particular, generating each new token requires reading the *entire* set of model weights from HBM to perform a matmul against a batch of size 1 — an extremely low arithmetic-intensity, severely memory-bound operation (Section 4), since a huge amount of weight data is moved per token but very little compute is done with each byte before it's discarded.

**Batching** — running multiple independent sequences (either multiple separate requests, or multiple sequences in a training minibatch) through the model together — fixes this by turning `x` from a single vector into a matrix with one row per sequence in the batch. The exact same weight matrix `W`, once loaded into fast on-chip memory (or even just brought in from HBM once), is now reused across every row of the batch before being discarded. This directly increases arithmetic intensity (the same bytes of `W` now amortize over many more FLOPs) and directly increases the amount of independent, parallel work available for the SMs to chew on simultaneously — filling far more of the GPU's lanes per launch and per byte of weight traffic. This is why serving throughput (tokens/sec across all users) scales up dramatically with batch size until some other resource (HBM capacity for the KV cache, typically) becomes the limiting factor, and why every serious inference server is built around keeping the GPU fed with as large and well-formed a batch as possible at all times, rather than handling requests strictly one at a time.

Doing this well for LLM inference specifically — where requests arrive at different times and generate different numbers of tokens, so a naive fixed batch would force short requests to wait for long ones — is the subject of **continuous batching** (iteration-level scheduling that adds/removes sequences from the batch every decoding step) and **paged attention** (managing the KV cache in fixed-size, non-contiguous blocks so memory isn't wasted on padding for variable-length sequences). Both are covered in full in `docs/ml-concepts/17-inference-serving-optimization/README.md`; the point to take from this section is *why* batching matters at the hardware level — it's a direct lever on GPU occupancy and arithmetic intensity, not just a scheduling convenience.

---

## 9. Multi-GPU Communication

Training (and sometimes serving) large models requires splitting work across multiple GPUs, which means those GPUs need to exchange data — gradients, activations, or parameter shards — over some physical interconnect. The bandwidth of that interconnect matters enormously, because communication that isn't hidden behind computation becomes pure added wall-clock time.

**PCIe** (the general-purpose bus connecting a GPU to the rest of the system, and often used GPU-to-GPU as well) tops out around 32-64 GB/s per direction for PCIe Gen4/Gen5 x16 — fast for a general peripheral bus, but a bottleneck compared to what GPUs can compute and compared to HBM bandwidth.

**NVLink** is NVIDIA's purpose-built GPU-to-GPU interconnect, offering far higher bandwidth — on the order of 900 GB/s of total bandwidth per GPU on an H100 (NVLink 4), roughly 15-20x what PCIe provides. On systems with **NVSwitch**, this high-bandwidth connectivity extends to *all* GPUs in a node simultaneously (all-to-all), not just adjacent pairs, which matters a great deal for collective operations like all-reduce (used to sum gradients across data-parallel replicas) and all-to-all (used in tensor/expert-parallel communication patterns).

Why this matters concretely: in data-parallel training, after each GPU computes gradients on its local batch shard, those gradients must be summed (all-reduced) across every GPU before the optimizer step — an operation whose cost scales with model size and repeats every training step. If that traffic is forced over PCIe instead of NVLink, communication time can dominate the step time no matter how fast each GPU's local compute is. This is a large part of why multi-GPU training performance depends heavily on the physical topology of the machine (NVLink-connected GPUs within a node vs. a slower fabric across nodes), not just on GPU count.

The other lever, alongside raw bandwidth, is **overlapping computation with communication**: rather than waiting for backprop to finish entirely before starting the all-reduce, frameworks (PyTorch DDP, FSDP/ZeRO) kick off communication for early layers' gradients *while* backprop is still computing gradients for later layers, hiding the communication cost behind compute that would be happening anyway. Done well, this makes communication nearly free in wall-clock terms, as long as bandwidth keeps up with the rate compute produces new data to send. The specific parallelism strategies that generate this communication — data, tensor, and pipeline parallelism — are covered in `docs/ml-concepts/19-distributed-training/README.md`.

---

## 10. What an Application Engineer Actually Needs to Know

Here is the honest version, without pretending everyone needs to be a CUDA kernel author:

**You will almost never write raw CUDA.** The overwhelming majority of engineers building or serving LLM-based applications never write a `__global__` kernel by hand. Instead, you work through layers that have already implemented the optimizations in this document:

- **PyTorch** (with `torch.compile`) applies automatic kernel fusion, picks efficient matmul kernels via cuBLAS/cuDNN, and handles mixed precision through `torch.autocast` and `GradScaler`.
- **Triton** is a Python-embedded language for writing custom GPU kernels at a much higher level than raw CUDA — you describe block-level tiling and let the compiler handle warp/thread-level details. Most modern FlashAttention implementations, and much of what's inside `torch.compile`'s generated code, are Triton kernels.
- **vLLM**, **TensorRT-LLM**, **SGLang**, and similar serving frameworks implement paged attention, continuous batching, and highly-tuned fused kernels for attention and the KV cache, so that using them correctly gets you most of the throughput benefits described in Sections 5, 6, and 8 without touching a kernel yourself.
- **cuBLAS/cuDNN/cuBLASLt** are NVIDIA's own hand-tuned libraries for matmul and standard NN primitives, and they're what PyTorch/TensorFlow call under the hood for most dense linear algebra.

**What understanding this model actually buys you, day to day:**

- You can read a benchmark or a PR description ("fused the RMSNorm + residual add into one kernel", "switched to FlashAttention-2", "enabled bf16 autocast") and know *why* it should be faster, instead of taking the speedup on faith — it's fusion reducing HBM round-trips, or Tensor Cores raising throughput, or batching raising arithmetic intensity.
- You know why increasing batch size helps throughput up to a point and then stops (you've hit a compute-bound ceiling, or run out of HBM for the KV cache) — this directly informs how you size instances and set serving concurrency limits.
- You know why a model that looks "small" in parameter count can still be memory-bandwidth-limited at inference (autoregressive decoding rereads all its weights from HBM every single token), which explains why quantization (fewer bytes per weight) often speeds up serving even on GPUs with plenty of spare compute.
- You can tell the difference between a problem that a bigger/faster GPU actually fixes (a genuinely compute-bound workload) and one it won't (a memory-bandwidth-bound workload, where you need a GPU with more HBM bandwidth specifically, or an algorithmic fix like fusion, not just "more FLOPs").
- When choosing between libraries or configs — eager PyTorch vs. `torch.compile`, vLLM vs. a hand-rolled generation loop, fp16 vs. bf16 vs. int8 — you can reason from first principles about which one is likely faster and why, rather than only trusting a benchmark you can't explain.

In short: you don't need to be able to write the kernel, but you do need to be able to look at a slow model and ask the right diagnostic question — "is this memory-bound or compute-bound, and does the code in front of me actually exploit the GPU the way Sections 1-9 describe?" That question is almost always answerable, and it's almost always where the real speedup is hiding.

---

## See Also

- `docs/ml-concepts/18-attention-optimizations/README.md` — Full FlashAttention algorithm/derivation, FlashAttention-2/3, multi-query and grouped-query attention, sliding-window attention.
- `docs/ml-concepts/17-inference-serving-optimization/README.md` — KV cache mechanics, continuous batching, paged attention (vLLM), speculative decoding.
- `docs/ml-concepts/19-distributed-training/README.md` — Data/tensor/pipeline parallelism and how they map onto multi-GPU communication patterns.
- `docs/ml-concepts/15-quantization/README.md` — INT8/INT4/fp8 quantization mechanics: PTQ vs QAT, GPTQ, AWQ, GGUF.
- `docs/ml-concepts/16-memory-optimization-training/README.md` — Mixed precision, gradient checkpointing, ZeRO/FSDP, activation memory accounting in training.
