# Performance Optimization Methodology for LLM Systems

Every other document in this series — [CUDA/GPU optimization](../14-cuda-gpu-optimization/README.md), [quantization](../15-quantization/README.md), [memory optimization for training](../16-memory-optimization-training/README.md), [inference serving optimization](../17-inference-serving-optimization/README.md), [attention optimizations](../18-attention-optimizations/README.md), [distributed training](../19-distributed-training/README.md) — describes a specific, powerful technique and the mechanism that makes it work. None of them tell you *when to reach for which one*, or how to figure out that you have a problem those techniques would even solve. This document is the missing piece: the methodology for finding a real performance problem before you fix it, and for choosing the right fix once you've found it.

The central failure mode this document exists to prevent is applying an optimization technique because it's well-known, impressive, or the thing you read about last, rather than because you measured a bottleneck that technique actually addresses. Someone spends a week hand-writing a fused Triton kernel for an operation that turns out to be 2% of wall-clock time. Someone enables FlashAttention on a workload that was never attention-bound in the first place, because the real bottleneck was a `DataLoader` with zero worker processes. Someone doubles batch size to "improve GPU utilization" and makes p99 latency for interactive users worse. Every one of these is a real, common mistake, and every one of them is prevented by the same discipline: measure first, form a hypothesis, change one thing, measure again.

This document is not a substitute for the technique-specific docs. It is the decision procedure that tells you which one of them to open.

## Table of Contents

1. [The Optimization Mindset](#1-the-optimization-mindset)
2. [The Three Fundamental Bottleneck Categories](#2-the-three-fundamental-bottleneck-categories)
3. [Profiling Tools and How to Use Them](#3-profiling-tools-and-how-to-use-them)
4. [Using the Roofline Model in Practice](#4-using-the-roofline-model-in-practice)
5. [Benchmarking Methodology Done Correctly](#5-benchmarking-methodology-done-correctly)
6. [The Decision Framework: From Symptom to Fix](#6-the-decision-framework-from-symptom-to-fix)
7. [Profiling Tool Hierarchy Diagram](#7-profiling-tool-hierarchy-diagram)
8. [Closing: Where the Real Wins Come From](#8-closing-where-the-real-wins-come-from)

---

## 1. The Optimization Mindset

### "Premature optimization is the root of all evil" — and it's worse for ML systems

Donald Knuth's famous line is usually quoted to justify not micro-optimizing a loop before you know it matters. In LLM training and serving systems, the same principle applies but the stakes and the traps are bigger, for three reasons specific to this domain:

1. **The bottleneck is frequently the opposite of what intuition suggests.** Engineers coming from general software backgrounds default to assuming a GPU workload is compute-bound — "it's doing matrix multiplication on a GPU, so the GPU's math is the limit." In reality, a large fraction of LLM workloads, especially autoregressive decode, are memory-bandwidth-bound (Section 2), and a nontrivial fraction of "GPU is slow" complaints turn out to be the GPU sitting nearly idle waiting on the CPU (data loading, tokenization, Python-level orchestration, host-to-device transfer). If you don't measure, your prior about where the time is going is a coin flip at best, and the base rate favors the counterintuitive answer.
2. **The cost of guessing wrong is higher.** A CUDA kernel rewrite, a change to model parallelism strategy, or a new quantization scheme can be days to weeks of engineering effort, plus a nontrivial risk of introducing numerical bugs or regressions elsewhere. Compare that to a 10-minute profiler pass first — the asymmetry between "cost to measure" and "cost to guess wrong and rewrite" is enormous, and it grows the deeper into the stack the candidate fix lives (a kernel rewrite costs more than a config flag).
3. **Optimizations don't compose for free, and some actively conflict.** Techniques tuned for one bottleneck can worsen another — the throughput/latency tension (Section 5) is the clearest example: increasing batch size relieves a memory-bandwidth bottleneck during decode but can directly hurt tail latency for a request queued behind a larger batch. Applying a technique without knowing what you're trading against it can make the system worse on the metric that actually mattered, even while an aggregate number (e.g. average throughput) looks better.

### The loop

There is one correct loop, and every section below is in service of executing it well:

```
   ┌─────────────┐
   │   MEASURE   │  Get a real number for wall-clock time (or the
   │             │  metric that matters — TTFT, TPOT, throughput,
   └──────┬──────┘  step time). Not a guess. Not "it feels slow."
          │
          ▼
   ┌─────────────┐
   │   IDENTIFY  │  Use profiling tools (Section 3) to find WHERE
   │  BOTTLENECK │  the time actually goes, and WHICH category
   │             │  (compute / memory / host) it falls into.
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  HYPOTHESIZE│  Form a specific, falsifiable claim: "this is
   │             │  memory-bound because attention materializes an
   │             │  n×n matrix; FlashAttention should fix it."
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ APPLY ONE   │  Change exactly one thing. Not "enable flash
   │   CHANGE    │  attention AND bump batch size AND switch to
   │             │  bf16" all at once — you won't know which one
   └──────┬──────┘  did what, or which one broke something.
          │
          ▼
   ┌─────────────┐
   │ RE-MEASURE  │  Same metric, same methodology (Section 5),
   │             │  same hardware/load conditions. Confirm the
   │             │  hypothesis was right and quantify the gain.
   └──────┬──────┘
          │
          └────────────► back to MEASURE (there is always a new
                          bottleneck once the old one is fixed —
                          Amdahl's Law guarantees it)
```

Two properties of this loop are easy to state and easy to skip under time pressure:

- **"Apply one change" is the single most-violated step.** It is tempting, once you've identified a slow region, to throw every plausible fix at it in one pass to save iteration time. This destroys your ability to attribute the result: if three changes together produce a 1.4x speedup, you don't know if one was free, one did nothing, and one caused a regression the others masked. Batch changes only once you understand each one's individual effect in isolation, and re-verify important combinations even then.
- **The loop doesn't terminate in one pass.** Fixing the first bottleneck deterministically exposes the next one — this is Amdahl's Law in practice: eliminate 90% of the time a component was taking, and whatever was previously "the other 10%, not worth looking at" is now the majority of a smaller total. Expect to run the loop several times on a serious optimization effort, with diminishing but real returns each pass, until you hit a bottleneck category you've decided isn't worth chasing further (often the host-side one, once it drops below the GPU's own floor).

---

## 2. The Three Fundamental Bottleneck Categories

Every operation, kernel, training step, or inference request falls into (or spends time split across) exactly three categories. Correctly classifying which one you're in is 80% of the diagnostic work — it tells you which family of fixes can possibly help, and which ones are guaranteed to do nothing.

### (a) Compute-bound

The GPU's arithmetic units (CUDA cores / Tensor Cores) are the limiting resource. Data is arriving from memory fast enough to keep the ALUs continuously fed; the ceiling is how many floating-point operations per second the silicon can physically issue. See the [roofline model introduction](../14-cuda-gpu-optimization/README.md#4-compute-bound-vs-memory-bound-the-roofline-model) for the underlying arithmetic-intensity argument.

**Diagnostic signature:**
- High SM (Streaming Multiprocessor) utilization, often 80-100%, sustained.
- Achieved FLOPs/s close to the hardware's advertised peak for the precision in use (e.g., a matmul running near an H100's ~990 TFLOPS bf16 Tensor Core peak).
- Memory controller / HBM bandwidth utilization comparatively low — data is arriving faster than it's consumed, or is being reused heavily once loaded.
- Increasing batch size further gives diminishing or no throughput improvement (you've saturated the ALUs; more parallel work has nowhere to go).

**Typical LLM examples:** large, well-tiled feed-forward matmuls with long sequences or large batches; prefill on a long prompt (many tokens processed in one forward pass gives the matmuls a large, dense shape); training step compute for large hidden dimensions.

### (b) Memory-bandwidth-bound

The arithmetic units are frequently idle, not because there's no work queued but because the data those units need hasn't arrived from HBM yet. This is **the most common bottleneck for LLM inference decode**, and it is far more common across LLM workloads in general than most engineers expect coming in.

**Diagnostic signature:**
- Low SM utilization / low achieved FLOPs relative to peak, *despite* the GPU showing as "busy" (`nvidia-smi` utilization can be misleadingly high here — see Section 3's caveat).
- High memory controller / DRAM (HBM) utilization, often close to 100% of the GPU's rated bandwidth.
- The operation's measured throughput tracks the GPU's memory bandwidth spec, not its FLOPs spec — e.g., doubling precision from fp16 to fp32 (doubling bytes moved, same FLOP count) roughly halves throughput; halving precision (fp16 → int8) roughly doubles it.
- Increasing batch size *does* help here, up to a point, because it amortizes the fixed cost of streaming weights from HBM across more useful compute per byte moved (see the [batching and GPU utilization discussion](../14-cuda-gpu-optimization/README.md#8-batching-and-gpu-utilization)).

**Typical LLM examples:** single-token autoregressive decode (the entire weight matrix must be re-streamed from HBM per generated token, against a tiny amount of per-token compute); elementwise operations (layer norm, activations, masking, dropout, softmax in isolation); unfused attention that materializes the full `n × n` score matrix.

### (c) I/O- or host-bound

The GPU is doing nothing productive — often sitting fully idle — because it's waiting on something that isn't GPU compute or GPU memory bandwidth at all: reading data off disk or a network filesystem, CPU-side preprocessing (tokenization, augmentation, collation), Python interpreter overhead (object creation, `.item()` calls that force a device sync, dynamic control flow), or a slow host-to-device transfer.

**Diagnostic signature:**
- Low overall GPU utilization (`nvidia-smi` shows the GPU near 0% for stretches), while CPU utilization on one or more cores is high, or `iowait` is elevated, or the process is visibly blocked on disk/network reads.
- A timeline view (Nsight Systems, or `torch.profiler`'s trace) shows visible gaps between GPU kernels — the GPU finishes a batch of work and then sits idle until the next batch of inputs is ready.
- Adding more `DataLoader` workers, prefetching, or pinning memory measurably closes the gap (a strong confirming signal, and the fix itself).
- This bottleneck is invisible to naive wall-clock-only benchmarking of "the model" — if you only time `model(x)` and not the full loop including data loading, you can miss it entirely and wrongly conclude the model itself is slow.

**Typical LLM examples:** training with too few dataloader workers or no prefetching against a large sharded dataset; loading model weights from slow network storage at server startup; a request-handling loop in a serving framework that does per-request Python-level tokenization/detokenization or JSON (de)serialization synchronously on the critical path; excessive CPU-GPU synchronization points (e.g., calling `.item()` or `.cpu()` inside a hot loop, which forces the GPU to finish everything queued before Python can proceed, destroying the asynchronous overlap that makes GPUs fast in the first place).

### Why classification matters more than any specific fix

Each category has a disjoint set of effective fixes. A faster GPU (more FLOPs) does nothing for a memory-bandwidth-bound workload — you'll pay for more silicon and see the same wall-clock time, because the bottleneck was never the arithmetic units. A GPU with more HBM bandwidth does nothing for a host-bound workload — the GPU was never the resource in short supply. Getting the category wrong doesn't just fail to help; it can consume a purchasing decision, a migration, or a rewrite for zero gain. This is why Sections 3 and 4 exist: to give you tools that answer "which category am I in" with evidence, not intuition.

---

## 3. Profiling Tools and How to Use Them

The tools below form a hierarchy from coarse to fine-grained. **Always start at the top.** Reaching for Nsight Compute's per-kernel warp-stall analysis before you've established, at the system level, that a specific kernel is even worth that attention is a common way to waste hours — you'll get an extremely precise answer to a question that didn't need asking.

### `nvidia-smi` / `nvtop` — coarse GPU health, first look, always

The first command to run, always, before anything else. `nvidia-smi` (or the friendlier terminal UI `nvtop`, or `nvidia-smi dmon` for a streaming view) gives you, at a glance:

- **GPU utilization %** — but read this with the caveat below.
- **Memory used / memory total** — is the model + activations + KV cache close to exhausting device memory? (An OOM or near-OOM condition can itself force suboptimal behavior, like smaller batches than you intended.)
- **Power draw and temperature** — sustained near the power/thermal limit for long stretches can indicate throttling, which silently degrades performance in a way that looks like "the code is slow" but is actually "the chip backed off its clocks."
- **Per-process memory and utilization** — useful on a shared/multi-tenant box to confirm nothing else is competing for the GPU.

**The critical caveat:** `nvidia-smi`'s "GPU-Util" percentage means "the fraction of the last sampling window during which at least one kernel was running on the GPU," not "the fraction of the GPU's arithmetic capacity being used." A GPU can show 100% utilization while running a severely memory-bound, low-arithmetic-intensity kernel (Section 2b) that's using a small fraction of available FLOPs — the ALUs are "busy" only in the sense that *something* is scheduled, not in the sense that they're doing much per cycle. Do not conclude "compute-bound" from a high `nvidia-smi` percentage alone; it rules out the host-bound category (GPU isn't idle) but says nothing about compute- vs. memory-bound. Use it as a fast idle/busy/OOM check, then move down the hierarchy for anything more specific.

### PyTorch Profiler (`torch.profiler`) — operator-level time breakdown

Once `nvidia-smi` confirms the GPU is actually busy (ruling out a purely host-bound problem) or shows suspicious idle gaps (suggesting one), `torch.profiler` is the next stop for anyone working at the PyTorch level. It instruments your actual training or inference loop and produces a table (and/or a Chrome-trace-format timeline) of:

- Wall-clock time spent per named operator (`aten::matmul`, `aten::layer_norm`, your own custom ops), both CPU-side and CUDA-side, self time vs. total including children.
- Which operators dominate — usually the single most actionable output. If 60% of step time is inside a handful of named ops, you now know exactly where to focus the next level of investigation, instead of guessing.
- Memory allocation events (`profile_memory=True`), useful for catching unexpected activation memory growth.
- Whether time is being lost to host-side overhead — a large gap between "CPU time" and actual CUDA kernel time for the same op, or a long tail of tiny ops each incurring Python/dispatch overhead, points back at the host-bound category (Section 2c) even inside an otherwise GPU-heavy trace.

Practically: wrap the region of interest (a handful of training steps, or several inference requests, never a single one — see warm-up in Section 5) with `torch.profiler.profile(...)`, export the trace, and either read the summary table or load it in Chrome's `chrome://tracing` (or `tensorboard`) viewer for a visual timeline. This is the right level of resolution for "which module/operator is the problem," and for most application-level engineers, it's as deep as you need to go before you can already act (e.g., you find that a hand-written attention implementation dominates time, and that's your cue to check whether FlashAttention is enabled — see [attention optimizations](../18-attention-optimizations/README.md)).

### NVIDIA Nsight Systems — the timeline view, CPU/GPU overlap, and idle gaps

`torch.profiler` tells you which named operators are slow in aggregate; Nsight Systems (`nsys`) shows you the actual **timeline**: CPU threads, CUDA API calls, kernel launches, memory copies, and NCCL/communication events, all laid out on a synchronized time axis so you can visually see overlap (or its absence).

This is the right tool for a specific, common class of question that operator-level tables answer poorly: *"is the GPU actually busy the whole time, or is it launching a kernel, finishing quickly, and then waiting for the CPU to queue up the next one?"* A Nsight Systems trace makes idle gaps between kernels immediately visible as literal empty space on the GPU row of the timeline — exactly the signature of host-bound stalls (Section 2c), Python overhead, or insufficiently asynchronous kernel launch patterns (e.g., inserting a synchronizing `.item()` call between GPU-bound steps). It's also the tool of choice for diagnosing multi-GPU communication problems — you can see whether NCCL all-reduce calls overlap with backward-pass compute (as they should) or whether compute stalls waiting for communication to finish (see [distributed training](../19-distributed-training/README.md) for what that communication is doing and why overlap matters).

Practically: run your training/inference process under `nsys profile -o report python train.py ...`, then open the resulting `.nsys-rep` file in the Nsight Systems GUI. Look first for gaps in the GPU row — that's host-bound time, in effect for free, before you even look at what's inside the busy segments.

### NVIDIA Nsight Compute — deep per-kernel analysis, once you know which kernel

Nsight Compute (`ncu`) is the most detailed tool in this stack, and correspondingly the narrowest in scope: it profiles **one kernel at a time** in exhaustive depth — occupancy (resident warps vs. hardware maximum), achieved memory throughput as a percentage of the GPU's rated HBM bandwidth, achieved compute throughput as a percentage of peak FLOPs, warp stall reason breakdowns (stalled on memory dependency, stalled on execution dependency, stalled waiting for a barrier, etc.), and register/shared-memory usage per thread block.

This is exactly the roofline-model data made concrete for a single kernel (Section 4): Nsight Compute will tell you, for the specific kernel you point it at, both its achieved FLOPs/s and its achieved bytes/s, letting you directly compute where it sits on the roofline and which roof it's up against.

**Use this only after** `torch.profiler` or Nsight Systems has already told you which specific kernel is worth this level of scrutiny. Running Nsight Compute across an entire training step by default is slow (its instrumentation overhead is much higher than Nsight Systems' or the PyTorch profiler's) and produces an overwhelming amount of data if you haven't already narrowed the question to "why is *this* kernel slow." Point it at one kernel name (`ncu --kernel-name <name> ...`) once you know which one matters.

### The right order

```
nvidia-smi/nvtop  →  torch.profiler  →  Nsight Systems  →  Nsight Compute
(is GPU idle/OOM?)  (which OP dominates?)  (where are the gaps,     (why is THIS
                                             does compute/comm        kernel slow,
                                             overlap?)                 exactly?)
```

Each tool answers a narrower question than the one before it, and each one is more expensive (in setup time, run time, and mental effort to interpret) than the one before it. Skipping straight to the bottom because it's the most "serious"-sounding tool is a common way to spend an afternoon getting a very precise answer to the wrong question.

---

## 4. Using the Roofline Model in Practice

The [roofline model itself](../14-cuda-gpu-optimization/README.md#4-compute-bound-vs-memory-bound-the-roofline-model) — the plot of achievable FLOPs/s against arithmetic intensity, with a memory-bandwidth-limited slope on the left and a compute-limited ceiling on the right — is introduced in the CUDA/GPU optimization document. This section is about actually using it as a working tool, not just understanding the picture.

### Step 1: Compute your operation's arithmetic intensity

```
Arithmetic Intensity (FLOPs/byte) = Total FLOPs performed / Total bytes moved to/from HBM
```

For a dense matmul `C = A @ B` with `A: (M, K)`, `B: (K, N)`:
- FLOPs ≈ `2 * M * N * K` (one multiply and one add per output-element-per-reduction-step)
- Bytes moved (naively, no reuse) ≈ `bytes_per_element * (M*K + K*N + M*N)` for reading both inputs once and writing the output once — but well-tiled implementations reuse tiles of `A` and `B` from shared memory/registers across many output elements, so the *effective* bytes moved is much lower than this naive figure. This is precisely why matmul kernel quality (tile size, how much reuse the implementation achieves) determines how close you get to the compute roof — the formula is a starting point, not the answer; a profiler (Nsight Compute) reports the *achieved* bytes moved for a real kernel, which is what you actually want.

For a memory-bound elementwise op (e.g., a standalone layer norm over a tensor of `N` elements): FLOPs is small and fixed per element (a handful of arithmetic ops), while bytes moved is one read and one write per element at minimum — arithmetic intensity is a small constant, independent of `N`, and it doesn't improve no matter how large the tensor gets. This is the signature of a permanently memory-bound operation; no amount of batching or tiling changes its intensity, because there's no reduction dimension to reuse data across (contrast with the matmul above, where reuse across the `K` dimension is exactly what tiling exploits).

### Step 2: Compute the hardware's two roofs

You need two numbers from the GPU's spec sheet (or, better, from a microbenchmark of your actual hardware, which is sometimes meaningfully below the marketed peak):

- **Peak compute** (FLOPs/s) for the precision you're using — this differs hugely by precision (fp32 CUDA core peak vs. bf16 Tensor Core peak vs. int8/fp8 peak are all different numbers on the same chip; using the wrong one for your actual dtype gives a wrong roofline).
- **Peak memory bandwidth** (bytes/s) — the HBM bandwidth spec (e.g., ~3.35 TB/s for an H100 SXM).

The **ridge point** — the arithmetic intensity at which the two roofs intersect — is `peak_compute / peak_bandwidth`. Below that intensity, you're memory-bound; above it, you're compute-bound, *for that hardware*. This number is worth computing once per GPU you care about and keeping on hand, because it turns "is my operation memory- or compute-bound" from a profiling exercise into a single division once you know the operation's arithmetic intensity.

### Step 3: Plot (or just compare) and let it tell you where to invest

You don't need to literally draw the chart every time — the decision rule it encodes is:

- If your operation's arithmetic intensity is **below the ridge point**, you are memory-bound, and the *only* effective fixes reduce bytes moved: kernel fusion (eliminate intermediate round-trips to HBM), quantization (fewer bytes per element), better data layout (contiguous access patterns, avoiding strided/scattered reads), or reducing the volume of data that needs to move at all (e.g., a smaller KV cache). **Making the arithmetic faster — a bigger GPU, more Tensor Cores, lower-latency math — does nothing here**, because the arithmetic units are already waiting on data more than they're waiting on their own throughput. This is the single most common way engineering effort gets wasted: optimizing the compute side of an operation that was never compute-limited.
- If your operation's arithmetic intensity is **above the ridge point**, you are compute-bound, and the effective fixes target FLOPs and math throughput: using Tensor Cores via lower precision, reducing the actual FLOP count (algorithmic changes, sparsity, a smaller model), or better kernel scheduling to keep the ALUs fed at their rated issue rate. Reducing bytes moved further does essentially nothing here, because memory was never the constraint.

The practical payoff of doing this arithmetic explicitly, even approximately, before committing to an optimization direction: it tells you in advance which of the technique-specific documents in this series is even capable of helping. If your bottleneck's arithmetic intensity sits well below the ridge point, a note that says "try `torch.compile` for better kernel fusion" or "switch to FlashAttention" is on the right side of the roofline; a suggestion to "use a bigger GPU" or "enable Tensor Cores" is not, no matter how reasonable it sounds in isolation.

---

## 5. Benchmarking Methodology Done Correctly

A profiler tells you where time goes *within* a measurement. Benchmarking methodology determines whether the measurement itself is trustworthy in the first place. Both matter, but a beautifully precise profile of a badly-designed benchmark just tells you, precisely, the wrong number.

### Warm-up runs

The first invocation of almost anything in this stack is slower than subsequent ones, for reasons that have nothing to do with steady-state performance:

- **JIT compilation** — `torch.compile`, XLA, and Triton's autotuning all compile kernels (and, for autotuning, benchmark several candidate implementations) on first call for a given input shape. This can add seconds to the first call and is completely absent from call two onward for the same shape.
- **CUDA context initialization and cuDNN/cuBLAS algorithm selection** — the first call to a given operation shape often triggers a one-time search over candidate algorithms (`cudnn.benchmark=True` in PyTorch explicitly does this) that isn't repeated afterward.
- **Cold caches** — L2 cache, page cache for data files, and the OS/filesystem cache for model weights are all cold on a fresh process and warm after a few iterations, so early iterations pay real I/O and cache-miss costs that steady-state iterations don't.
- **Memory allocator warm-up** — PyTorch's caching CUDA allocator grows its pool over the first several iterations of a training loop; early steps trigger `cudaMalloc` calls that later steps, reusing the pool, don't.

The fix is mechanical and non-negotiable: run a fixed number of warm-up iterations (enough to exercise every code path and shape you'll measure — for variable-shape workloads, warm up multiple representative shapes), discard their timings entirely, and only start the timed region afterward. Skipping this is one of the most common ways a benchmark produces a number nobody can reproduce.

### Percentiles, not averages

Report **p50, p90, and p99** (and, for latency-critical serving systems, often p99.9), not a single mean, across many trials. This matters enormously for two related reasons:

1. **The mean hides the tail, and the tail is what users feel.** A system with a mean latency of 100ms and a p99 of 800ms *feels* like an 800ms system to the unlucky 1% of requests, every time they hit it — and at scale, "1% of requests" is not a rounding error, it's thousands of bad experiences a day. Averaging across trials that include occasional stalls (a garbage collection pause, a scheduler preemption, a batch that happened to include one unusually long sequence) actively conceals the problem an average is supposed to summarize.
2. **LLM-serving tail latency has systematic, non-random causes**, not just noise — a request that lands behind a large batch, or that triggers a KV-cache eviction/preemption, or that shares a GPU with a much longer concurrent request, will reliably show up in the tail rather than being a one-off fluke. Reporting only the mean makes these systematic effects invisible, when they are usually exactly the thing worth fixing.

Run enough trials to get a stable estimate of p99 specifically — a distribution's tail needs more samples to characterize reliably than its center does; a handful of runs is enough to trust a mean, but not enough to trust a p99.

### Controlling for confounds

A single machine, run once, tells you very little. Sources of variance that will corrupt a comparison between "before" and "after" if left uncontrolled:

- **Thermal throttling.** A GPU run at sustained high load for long enough can hit its thermal limit and clock down, meaning a benchmark run late in a long session can look slower than an identical one run earlier — not because anything about the code changed, but because the chip is running at a lower clock. Watch `nvidia-smi`'s clock and temperature readings during long benchmark sessions, and prefer benchmarking after a fixed warm-up period at the *steady-state* thermal condition you actually expect in production, rather than the first few (cool, high-clock) iterations.
- **Other processes on the same GPU/machine.** A second job sharing the GPU, another team's process contending for CPU cores or memory bandwidth, or background system activity (log rotation, other users on a shared node) can each shift results by a meaningful margin. Confirm you have exclusive access, or explicitly note contention if you don't.
- **Batch composition.** For LLM serving specifically, a batch's latency and throughput depend on the *distribution* of sequence lengths in it, not just the count of requests — a batch of ten short requests behaves very differently from a batch of one very long request and nine short ones. Comparing "before" and "after" runs against different, unpinned request distributions can attribute a workload-composition difference to your code change. Use a fixed, representative, ideally realistic (traced from production, if possible) request distribution for both sides of any comparison.
- **Order effects.** Alternate "before" and "after" runs, or randomize their order, rather than running all of one condition and then all of the other back to back — this catches slow environmental drift (thermal, memory fragmentation, a competing process that started partway through) that would otherwise masquerade as a real difference between conditions.

### The LLM-serving-specific metrics

General latency/throughput numbers undersell what actually matters for a generative model, because a single "request" has two very different phases with opposite performance characteristics (this split is covered in depth in [inference serving optimization](../17-inference-serving-optimization/README.md#1-why-llm-inference-is-uniquely-awkward-to-serve)):

- **Time To First Token (TTFT)** — the latency from request arrival to the first generated token being available. Dominated by **prefill**: processing the entire input prompt in one (typically compute-bound, large-batch-shaped) forward pass. TTFT is what a user perceives as "did this thing start responding," and it's the metric most sensitive to prompt length and to how busy the server's prefill queue is.
- **Time Per Output Token (TPOT)**, sometimes called inter-token latency — the average time between successive generated tokens after the first. Dominated by **decode**: the autoregressive, typically memory-bandwidth-bound (Section 2b) loop of generating one token at a time. TPOT is what a user perceives as "is this streaming smoothly," and it's the metric most sensitive to KV cache size, batch size, and memory bandwidth.
- **Throughput** (tokens/sec, aggregated across all concurrent requests being served) — the metric an operator cares about for cost-per-token and capacity planning. It's maximized by keeping the GPU as busy and as large-batched as possible.

**Why optimizing for throughput and optimizing for latency can directly conflict:** increasing batch size raises arithmetic intensity during decode (more useful compute gets amortized per byte of weight streamed from HBM per step — see Section 2b and the [batching discussion](../14-cuda-gpu-optimization/README.md#8-batching-and-gpu-utilization)), which increases aggregate throughput. But a larger batch means each individual step (and therefore each individual token, for every request in that batch) takes longer to compute, because the GPU has more total work to do per step even though it's doing it more efficiently in aggregate. A request that would have had a TPOT of 20ms in a batch of 4 might see 35ms in a batch of 32 — worse for that individual user, even though the server's total tokens/sec went up substantially. Similarly, a new request arriving needs its prefill scheduled somehow — running it immediately (good for that request's TTFT) can mean pausing or interleaving with in-flight decode steps for other requests (bad for their TPOT), while queuing it behind the current batch (good for other requests' TPOT) directly hurts its own TTFT. This is a real resource-allocation trade-off, not a bug to be engineered away, and it's the reason serving frameworks expose scheduling knobs (max batch size, prioritization policies, chunked prefill) rather than a single "make it fast" setting — because "fast" means different, sometimes opposed, things depending on whether you're optimizing for the operator's cost-per-token or an individual user's perceived responsiveness. Continuous batching and PagedAttention (both in [inference serving optimization](../17-inference-serving-optimization/README.md)) are specifically about managing this trade-off well rather than eliminating it — no scheduling policy makes it disappear.

Any benchmark or optimization claim for a serving system should report TTFT, TPOT, and throughput separately (with percentiles on the first two), at a stated, realistic concurrency level — a single "requests per second" or "average latency" number without these broken out is close to meaningless for deciding whether a change actually helped the thing you care about.

---

## 6. The Decision Framework: From Symptom to Fix

Once you know your bottleneck category (Section 2) and have a hypothesis about the specific mechanism (Section 3/4's tools got you there), the fix should follow directly. This is the connective tissue between "what's slow" and "which document in this series to open."

- **Symptom: attention dominates profiled time, and it's memory-bandwidth-bound (materializing an `n × n` score matrix).**
  → Fused attention kernels. See [FlashAttention](../18-attention-optimizations/README.md) (and the GPU-mechanism half of the story in [CUDA/GPU optimization §6](../14-cuda-gpu-optimization/README.md#6-flashattention-fusion-applied-to-attention)).

- **Symptom: the KV cache is consuming most of GPU memory, capping batch size / concurrent requests / max context length.**
  → Quantized KV cache, or reduce the number of KV heads via **MQA/GQA** (see [attention optimizations](../18-attention-optimizations/README.md)); or manage cache memory more efficiently with **PagedAttention** (see [inference serving optimization](../17-inference-serving-optimization/README.md)) so fragmentation/padding waste stops being the limiting factor before raw cache size is.

- **Symptom: GPU sits idle between requests, or a batch drains to near-empty before new work arrives (visible as gaps in a Nsight Systems trace, or low `nvidia-smi` utilization despite requests being in flight).**
  → **Continuous batching** / iteration-level scheduling (see [inference serving optimization](../17-inference-serving-optimization/README.md)) so new requests join an in-flight batch instead of waiting for it to fully complete.

- **Symptom: the model (or model + activations + KV cache + optimizer states) doesn't fit in available GPU memory.**
  → **Quantization** to shrink weight footprint (see [quantization](../15-quantization/README.md)); **ZeRO/FSDP** sharding to spread optimizer/gradient/parameter state across GPUs during training (see [memory optimization for training](../16-memory-optimization-training/README.md)); or **multi-GPU parallelism** (tensor/pipeline/data parallelism) if the model or batch is fundamentally too large for one device regardless of sharding tricks (see [distributed training](../19-distributed-training/README.md)).

- **Symptom: GPU utilization is low overall, CPU usage is high, and/or profiling shows the process blocked on disk, network, or Python-level work (host-bound, Section 2c).**
  → More `DataLoader` worker processes, prefetching (`prefetch_factor`, overlapping the next batch's load with the current step's compute), pinned host memory for faster host→device transfer, and eliminating unnecessary CPU-GPU synchronization points in the hot loop (stray `.item()`/`.cpu()` calls). This is squarely an engineering-practice fix, not a GPU-side algorithmic one — no technique elsewhere in this series addresses it, because the GPU was never the resource in short supply.

- **Symptom: compute-bound (high SM utilization, achieved FLOPs near the hardware peak for the current precision), and you're already using an efficient, fused kernel implementation.**
  → You've run out of "free" wins from better kernels; the remaining levers are reducing the actual amount of arithmetic needed (fewer layers, a smaller model, [speculative decoding](../17-inference-serving-optimization/README.md) to get more tokens per expensive forward pass) or moving to a faster numeric precision so the same FLOP count executes faster (int8/fp8 via [quantization](../15-quantization/README.md), or bf16/fp16 mixed precision via [memory optimization for training](../16-memory-optimization-training/README.md)).

### Decision-tree diagram

```
                              ┌─────────────────────────────┐
                              │   Where does wall-clock       │
                              │   time actually go?           │
                              │   (nvidia-smi → profiler)      │
                              └───────────────┬───────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        │                                     │                                     │
        ▼                                     ▼                                     ▼
 GPU near-idle,                    GPU busy, SM util LOW,             GPU busy, SM util HIGH,
 CPU/iowait HIGH                   HBM/memory-ctrl util HIGH          achieved FLOPs near peak
        │                                     │                                     │
        ▼                                     ▼                                     ▼
 ┌─────────────┐                 ┌─────────────────────────┐         ┌─────────────────────────┐
 │ HOST-BOUND  │                 │   MEMORY-BANDWIDTH-BOUND │         │      COMPUTE-BOUND       │
 └──────┬──────┘                 └────────────┬─────────────┘         └────────────┬─────────────┘
        │                                     │                                     │
        ▼                                     ▼                                     ▼
 More DataLoader          ┌────────────────────┴────────────────────┐     Already using a fused/
 workers, prefetch,       │                                         │     efficient kernel?
 pinned memory, kill      ▼                                         ▼          │
 stray .item()/.cpu()  Attention op is        KV cache is the           ┌──────┴──────┐
 sync points            the hot spot          memory hog                │             │
                            │                     │                    NO            YES
                            ▼                     ▼                     │             │
                    FlashAttention        MQA/GQA, quantized KV    Enable      Reduce FLOPs needed:
                    (18-attention-        cache, PagedAttention    fusion /    fewer layers, smaller
                    optimizations)        (18-attention-opt,       torch.      model, speculative
                                          17-inference-serving)    compile     decoding (17-inference-
                                                                                serving-optimization);
                            ┌─────────────────────┐                            or faster precision:
                            │ Model/state doesn't  │                           int8/fp8 quantization
                            │ fit in memory at all │                           (15-quantization),
                            └──────────┬───────────┘                          mixed precision
                                       │                                      (16-memory-optimization-
                                       ▼                                      training)
                          Quantization (15-quantization) /
                          ZeRO / FSDP sharding (16-memory-
                          optimization-training) / multi-GPU
                          parallelism (19-distributed-training)
```

---

## 7. Profiling Tool Hierarchy Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 0 — SYSTEM HEALTH, SECONDS TO CHECK                              │
│  nvidia-smi / nvtop                                                     │
│  Answers: is the GPU idle, OOM, throttling, or contended?               │
│  Rules out: host-bound (if GPU shows busy) — does NOT distinguish       │
│  compute-bound from memory-bound (util% doesn't measure that).         │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  GPU is busy but wall-clock is still
                                 │  slower than expected → go one level down
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 1 — OPERATOR-LEVEL BREAKDOWN, MINUTES TO RUN                     │
│  torch.profiler                                                         │
│  Answers: which named op/module dominates wall-clock time?             │
│  Also flags host-side overhead (large CPU time vs. CUDA time gap).      │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  need to see overlap/gaps across
                                 │  CPU threads, kernels, and comm ops
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 2 — TIMELINE / OVERLAP VIEW, MINUTES TO RUN + GUI INSPECTION     │
│  NVIDIA Nsight Systems (nsys)                                           │
│  Answers: is the GPU idle between kernels? Does compute overlap        │
│  communication? Where exactly are the gaps in the timeline?            │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  a SPECIFIC kernel is now the known
                                 │  culprit — need to know exactly why
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  LEVEL 3 — SINGLE-KERNEL DEEP ANALYSIS, HIGH OVERHEAD, TARGETED USE     │
│  NVIDIA Nsight Compute (ncu)                                            │
│  Answers: occupancy, achieved memory throughput vs. peak, achieved     │
│  compute throughput vs. peak, warp stall reasons — for ONE kernel.     │
│  This is the roofline model made concrete for the exact kernel you     │
│  point it at.                                                          │
└────────────────────────────────────────────────────────────────────────┘

   Rule of thumb: never start below Level 0. Never jump straight to
   Level 3 without a Level 1/2 result telling you which kernel deserves
   that much attention — Level 3's overhead and information volume make
   it a poor tool for open-ended "what's slow" questions.
```

---

## 8. Closing: Where the Real Wins Come From

An honest note to end on, because it's easy for a document like this — full of profilers and roofline arithmetic and per-kernel stall reasons — to leave the impression that performance is primarily won at that level of granularity. It usually isn't.

The largest performance deltas in most real LLM systems come from decisions made **before** any profiler is ever run: the choice of model size (a smaller model that meets the quality bar is a bigger win than any kernel optimization applied to a larger one), the maximum context length actually needed (since attention and KV-cache costs scale with sequence length, and a system architected for 4K context when 100K was never needed pays a real, avoidable tax), the batching/serving architecture chosen at design time (an architecture built around continuous batching and PagedAttention from day one doesn't need heroic later optimization to claw back what naive request-at-a-time serving gave away), and whether the training/data pipeline was built with overlap and prefetching in mind from the start rather than retrofitted after someone finally profiled a bored GPU.

This doesn't make the methodology in this document unimportant — the loop in Section 1 is exactly how you find the highest-leverage of the *remaining* opportunities once the big architectural bets are made, and it's how you avoid genuinely wasting effort on a change that can't help. But it does mean the highest-leverage moment to apply performance thinking is at design time, not exclusively in a post-hoc optimization pass — profile early, even on a prototype, rather than only once something in production is unambiguously too slow.

The other half of the honest note: every technique named in Sections 6 and 7, and in every other document in this series, is a tool to reach for **once you have measured which bottleneck you actually have** — not a checklist to apply by default because it appears in a paper, a conference talk, or another team's stack. FlashAttention doesn't help a workload that was never attention-bound. Quantization doesn't help a workload that was already comfortably within memory. ZeRO sharding doesn't help a model that already fits on one GPU. Applying any of them without the measurement in Sections 1-5 first is not free — it costs engineering time, adds complexity and surface area for bugs, and in some cases (aggressive quantization, larger batches) trades away something else (quality, latency) that you may not have intended to spend. The methodology is the point; the techniques are what it points you at.

---

## See Also

- [`docs/ml-concepts/14-cuda-gpu-optimization/README.md`](../14-cuda-gpu-optimization/README.md) — CUDA execution model, GPU memory hierarchy, the roofline model's introduction, kernel fusion mechanics.
- [`docs/ml-concepts/15-quantization/README.md`](../15-quantization/README.md) — INT8/INT4/GPTQ/AWQ: shrinking bytes moved and memory footprint.
- [`docs/ml-concepts/16-memory-optimization-training/README.md`](../16-memory-optimization-training/README.md) — Mixed precision, gradient checkpointing, ZeRO/FSDP.
- [`docs/ml-concepts/17-inference-serving-optimization/README.md`](../17-inference-serving-optimization/README.md) — KV cache, continuous batching, PagedAttention, speculative decoding, TTFT/TPOT mechanics.
- [`docs/ml-concepts/18-attention-optimizations/README.md`](../18-attention-optimizations/README.md) — FlashAttention, MQA/GQA, sliding-window attention.
- [`docs/ml-concepts/19-distributed-training/README.md`](../19-distributed-training/README.md) — Data/tensor/pipeline parallelism and multi-GPU communication.
