# Distributed Training Strategies for Large Models

Training a modern large language model is not a "make the loop run longer" problem —
past a certain scale, it is structurally impossible on one accelerator, for two
independent reasons: the model does not fit in the memory of a single GPU, and even
when it does fit, a single GPU cannot execute enough floating-point operations per
second to finish training in a useful amount of time. Distributed training is the set
of techniques for splitting a single logical training job — one model, one optimizer,
one loss — across many GPUs so that both the memory footprint and the compute are
divided up, while the mathematical result stays equivalent (or acceptably close) to
what a single infinitely large, infinitely fast GPU would have produced.

This document covers the three orthogonal axes along which a training job can be
split — **data parallelism**, **tensor parallelism**, and **pipeline parallelism** —
how each one actually moves bytes across a network, why they get combined into "3D
parallelism" for frontier-scale runs, and where memory-optimized data parallelism
(ZeRO / FSDP) fits into the picture. The companion document
[`../16-memory-optimization-training/README.md`](../16-memory-optimization-training/README.md)
covers the full per-parameter memory breakdown (weights, gradients, optimizer states,
activations) and the ZeRO partitioning stages in detail — this document treats that
material as a black box and focuses on how work and communication are distributed
*across devices*, not how memory is reduced *on* a device.

---

## 1. Why a Single GPU Isn't Enough

There are two separate failure modes, and it matters which one you're up against
because they call for different fixes.

### 1.1 The model doesn't fit in memory

The most basic constraint: a GPU has a fixed amount of high-bandwidth memory (HBM) —
40GB or 80GB on an A100, 80GB or 141GB on an H100/H200 — and the model's parameters
alone have to fit inside that budget before you've even loaded a single training
example. Just storing the weights of a few well-known models in half precision
(2 bytes/parameter) illustrates the problem:

| Model              | Parameters | Weights alone (fp16/bf16) |
|--------------------|-----------:|---------------------------:|
| GPT-2 XL           |     1.5 B  |                     ~3 GB  |
| GPT-3              |   175 B    |                   ~350 GB  |
| PaLM               |   540 B    |                  ~1,080 GB |
| GPT-4 (est., MoE)  | ~1.8 T     |                  ~3,600 GB |

GPT-3's weights alone (350GB) already exceed the 80GB of memory on a single H100 by a
factor of 4.4x — and that's *before* accounting for gradients (another 350GB at
fp16), optimizer states (for Adam, typically 8 bytes/parameter more for fp32 momentum
and variance — another 1,400GB), and activation memory that grows with batch size and
sequence length. The full breakdown of that stack — weights + gradients + optimizer
states + activations, and exactly how ZeRO/FSDP, activation checkpointing, and mixed
precision each attack a different term in that sum — is the subject of the
memory-optimization README; the point to take away here is simply that **the
parameter count alone can exceed a single device's entire memory budget**, which means
there is no batch size, no precision trick, and no clever kernel that lets the model
run on one GPU. The model has to be *physically split across multiple devices* before
training can start at all. This is a **capacity** problem, and it's what tensor
parallelism and pipeline parallelism exist to solve — they each place a different
slice of the model's parameters on a different GPU so that no single device ever has
to hold the whole thing.

### 1.2 Even a model that fits is too slow to train alone

The second failure mode is independent of the first: suppose the model *does* fit
(e.g., a 7B or 13B model comfortably fits on a single 80GB GPU in bf16 with a
reasonable batch size). Modern LLM pretraining runs still consume on the order of
hundreds of billions to tens of trillions of tokens. A single GPU has a fixed, finite
FLOPs/second throughput — an H100 delivers roughly 989 TFLOPs of dense bf16 compute at
peak (far less in practice once you account for kernel efficiency, memory-bandwidth
stalls, and non-matmul work). Using the standard approximation that training a
transformer costs roughly `6 * N * D` FLOPs (N = parameters, D = training tokens),
training a 13B-parameter model on 300B tokens costs approximately:

```
6 * 13e9 * 300e9 ≈ 2.34e22 FLOPs
```

At a realistic sustained throughput of ~40% of peak (roughly 400 TFLOPs/s effective on
one H100), that single GPU would need:

```
2.34e22 / 4e14 ≈ 58,500,000 seconds ≈ 677 days ≈ ~1.9 years
```

— for a model that's small by current frontier standards. Nobody trains a
production model over a two-year single-GPU run: capital sits idle, competitors ship
first, and the hardware itself will be obsolete before the run finishes. The fix here
is not to place different *parts* of the model on different GPUs — the whole model
already fits on one — it's to run *more copies of the same computation in parallel*
so that N GPUs divide the wall-clock time by roughly N. This is a **throughput**
problem, and it's exactly what data parallelism solves.

These two problems — capacity and throughput — are why "distributed training" isn't
one technique but a family of them: DP alone fixes throughput but not capacity; TP and
PP fix capacity but, used alone across many devices, leave throughput scaling on the
table. Frontier training runs need all of it at once.

---

## 2. Data Parallelism (DP)

Data parallelism is the simplest and oldest strategy, and it directly targets the
throughput problem from section 1.2.

### 2.1 Mechanics

1. The **full model** — every parameter, in full — is replicated identically onto
   every participating GPU. If you have 8 GPUs, you have 8 complete, byte-identical
   copies of the model.
2. Each global training batch is **split (sharded) across GPUs**: if the global batch
   size is 512 and there are 8 GPUs, each GPU receives a disjoint micro-shard of 64
   examples.
3. Each GPU independently runs a full **forward pass** and **backward pass** on its
   own shard, using its own local copy of the weights. At this point, each GPU has
   computed a full set of gradients — but those gradients only reflect the 64 examples
   *that particular GPU* saw, not the full 512-example batch.
4. Before the optimizer is allowed to update any weights, the gradients from all 8
   GPUs are **synchronized** via an **all-reduce**: every GPU's gradient tensor for
   every parameter is summed (or averaged) across all GPUs, and the result is written
   back so that *all 8 GPUs end up holding the identical, fully-averaged gradient*.
5. Every GPU then runs the **same optimizer step** using the same averaged gradient
   against its own (identical) copy of the weights. Because the inputs to the
   optimizer step are now identical everywhere, all 8 replicas remain bit-for-bit in
   sync after the step, ready to repeat the cycle on the next batch.

```
GPU 0: shard_0 ──► fwd/bwd ──► grad_0 ┐
GPU 1: shard_1 ──► fwd/bwd ──► grad_1 ┤
GPU 2: shard_2 ──► fwd/bwd ──► grad_2 ├──► ALL-REDUCE (sum/avg) ──► same grad on all
GPU 3: shard_3 ──► fwd/bwd ──► grad_3 ┘                              │
                                                                       ▼
                                              every GPU: optimizer.step() (identical)
```

Mathematically, this is *exact*: averaging gradients computed on disjoint shards of a
batch and applying them once is identical to computing the gradient on the full batch
in one shot (loss is a sum/mean over examples, and differentiation is linear). Data
parallelism does not approximate anything — it reproduces single-GPU large-batch
training exactly, just computed concurrently.

### 2.2 Communication cost

The synchronization point is the all-reduce over every gradient tensor, once per
training step. This is the *only* required communication in pure DP, but it is not
free: the volume of data moved is proportional to the full size of the model's
gradients (which is the same size as the model's parameters). For a ring all-reduce
across `N` GPUs, each GPU sends and receives approximately `2 * (N-1)/N` times the
size of the gradient tensor — for large `N` this converges to roughly 2x the model
size in bytes moved per GPU, per step, regardless of how many GPUs are added (this is
the appeal of ring all-reduce: its cost is nearly independent of `N`). In practice,
frameworks like PyTorch DDP overlap this communication with the backward pass itself —
gradients for the last-computed layers are bucketed and all-reduced *while* earlier
layers are still computing their gradients — so the synchronization cost is partially
hidden rather than fully serial.

Because the communication volume scales with **model size**, not with the number of
GPUs, DP communication cost is essentially flat as you scale out GPU count (for a
fixed model) — which is precisely why DP scales so well for small-to-medium models:
adding more GPUs adds more compute (linear speedup) without proportionally increasing
communication.

### 2.3 What DP does *not* solve

Despite scaling throughput cleanly, pure data parallelism does nothing for the
capacity problem from section 1.1 — **every single GPU still holds a full,
byte-identical copy of the entire model, its gradients, and its optimizer states.**
If the model doesn't fit on one GPU, replicating it doesn't help; it fails on every
replica in exactly the same way it failed on the one GPU you started with. This is the
key limitation that motivates tensor and pipeline parallelism: DP scales *how many
examples you process per second*, not *how big a model you can hold at all*. (ZeRO/FSDP,
covered in section 6, is precisely an answer to "can we keep DP's communication
pattern and throughput scaling while *not* requiring every replica to hold a full
copy of everything" — but that's a memory optimization layered on top of DP, not a
different parallelism axis.)

---

## 3. Tensor Parallelism (TP)

Tensor parallelism (also called intra-layer model parallelism) attacks the capacity
problem directly, by splitting the *weight matrices themselves* — not the data, not
the layers — across GPUs, so that a single layer's computation is jointly performed by
multiple devices, none of which holds the full weight matrix for that layer.

### 3.1 Mechanics

Consider a single large linear layer `Y = X @ W`, where `W` is an `[in, out]` weight
matrix too large to comfortably fit (or too slow to compute on) one GPU. Megatron-LM's
approach (the canonical reference implementation of TP for transformers) splits it one
of two ways:

**Column-parallel**: split `W` along its output (column) dimension into `W = [W_1 |
W_2]` across 2 GPUs. Each GPU computes `Y_i = X @ W_i` independently — no
communication needed for this half, since each GPU has the *full* input `X` (typically
broadcast or already replicated) and only a *slice* of the weight. The two GPUs now
each hold half of the output activation, `Y_1` and `Y_2`, which together concatenate
to the full `Y`.

**Row-parallel**: split `W` along its input (row) dimension into `W = [W_1; W_2]`. Each
GPU needs only its slice of the input `X_i` and computes a *partial* sum `X_i @ W_i`.
Because the true output is `X @ W = X_1 @ W_1 + X_2 @ W_2`, the partial results must be
**summed across GPUs** — an all-reduce — before the result is a valid, complete
output.

Real transformer TP implementations chain these two patterns to minimize
communication: for the MLP block (`Linear → GELU → Linear`), the first linear is
column-parallel (no communication needed to produce its output, since GELU is
elementwise and can be applied independently on each GPU's column-slice) and the
second linear is row-parallel (which requires exactly *one* all-reduce, at the very
end of the block, to combine the partial sums into the full block output). For
multi-head attention, each GPU is simply assigned a disjoint subset of attention heads
end-to-end (since heads are independent), with a row-parallel-style all-reduce over
the output projection at the end. The net result: **one all-reduce in the forward
pass and one in the backward pass, per transformer block**, rather than one per
individual matrix multiplication.

```
Column-parallel linear (no comm needed to produce partial outputs):
   X (full, on both GPUs) ──► GPU0: X @ W_1 = Y_1 ┐
                          └─► GPU1: X @ W_2 = Y_2 ┘  (Y_1, Y_2 are both "final" —
                                                       concatenation is implicit)

Row-parallel linear (needs all-reduce to combine partial sums):
   X_1 (GPU0's slice) ──► GPU0: X_1 @ W_1 = partial_1 ┐
   X_2 (GPU1's slice) ──► GPU1: X_2 @ W_2 = partial_2 ┴─► ALL-REDUCE ──► Y (complete, on both)
```

### 3.2 Why TP needs very fast interconnect

The critical property of TP is *where* the communication happens: not once per
training step (like DP), but **inside the forward and backward pass of every single
layer**. A 96-layer transformer with TP performs on the order of ~200 all-reduce (or
all-gather) operations per training step (roughly one per block in forward, one per
block in backward), each one sitting directly on the critical path — the next layer's
matrix multiply cannot start until the communication finishes, because it needs the
combined activation as its input. There is no way to overlap this communication with
independent compute the way DP overlaps gradient sync with backward computation,
because the very next operation *depends* on the result of the communication.

This makes TP extremely latency- and bandwidth-sensitive. It is only practical over
**NVLink / NVSwitch** — the direct GPU-to-GPU interconnect within a single server node,
delivering on the order of 900 GB/s bidirectional bandwidth per GPU (NVLink 4 on
H100) with microsecond-scale latency. Attempting the same communication pattern over a
standard inter-node network fabric (InfiniBand or Ethernet, typically 200–400 Gb/s per
NIC — roughly an order of magnitude less bandwidth, and with far higher latency due to
the network stack and switch hops) would stall every single layer waiting on the
network, destroying throughput. This is why **tensor parallelism is, in practice,
confined to GPUs within a single node** — you split the model as finely as the
node's NVLink domain allows (e.g., 8-way TP across the 8 GPUs of one DGX/HGX server),
and use a different, communication-cheaper strategy to go beyond that.

---

## 4. Pipeline Parallelism (PP)

Pipeline parallelism also attacks the capacity problem, but splits the model along a
different axis than TP: instead of splitting individual weight matrices, it splits the
model's **sequence of layers** into contiguous **stages**, and assigns each stage to a
different GPU (or group of GPUs).

### 4.1 Mechanics

For a 32-layer transformer split across 4 GPUs, a naive assignment might be: GPU 0
holds layers 1–8, GPU 1 holds layers 9–16, GPU 2 holds layers 17–24, GPU 3 holds
layers 25–32. During the forward pass, GPU 0 computes layers 1–8 on the input batch
and sends the resulting activations to GPU 1, which computes layers 9–16 and sends its
output to GPU 2, and so on, until GPU 3 produces the final output and the loss.
During the backward pass, the same relationship runs in reverse: GPU 3 computes
gradients for layers 25–32 and sends the gradient of the activation *at its input
boundary* back to GPU 2, which continues backpropagation through layers 17–24 and
passes the boundary gradient back to GPU 1, and so on back to GPU 0.

The communication here is qualitatively different from TP: it is **point-to-point**
(GPU `i` only ever talks to GPU `i-1` and `i+1`), it happens only **at stage
boundaries** (a handful of times per micro-batch, not once per matrix multiply inside
a layer), and the volume transferred is just the activation tensor at that one
boundary (batch × sequence × hidden_dim), not the full parameter or gradient volume.
This makes PP communication far cheaper and far more tolerant of higher-latency,
lower-bandwidth links than TP — which is exactly why PP is the strategy of choice for
crossing **node boundaries**, where NVLink is no longer available and traffic has to
go over the (comparatively slow) InfiniBand/Ethernet fabric between servers.

### 4.2 The pipeline bubble problem

The naive scheme above has a serious efficiency flaw. If you feed one full batch
through the pipeline as a single unit, then at the very start, GPU 1, GPU 2, and GPU 3
have *nothing to do* — they are waiting for GPU 0 to finish the entire forward pass on
layers 1–8 before they receive any input at all. Symmetrically, at the end of the
backward pass, GPU 0, GPU 1, and GPU 2 sit idle waiting for gradients to arrive back
from the later stages. With `p` pipeline stages and only one batch in flight at a
time, at any given instant only *one* of the `p` GPUs is doing useful work — the other
`p-1` are idle. This wasted time is called the **pipeline bubble**, and in the naive
scheme the bubble consumes the overwhelming majority of the total wall-clock time as
`p` grows — pipeline parallelism alone, done this way, barely beats the throughput of
a single GPU processing everything serially, because you've paid the memory-capacity
benefit of splitting layers across devices but thrown away almost all the parallel
compute you were hoping to gain.

### 4.3 Micro-batching fixes it (GPipe-style)

The fix, introduced by GPipe (Huang et al., 2018) and used by essentially every
modern pipeline-parallel framework, is to split each training batch into many smaller
**micro-batches** and stream them through the pipeline back-to-back, rather than
sending one large batch through as an indivisible unit. As soon as GPU 0 finishes the
forward pass on micro-batch 1 and hands it to GPU 1, GPU 0 immediately starts on
micro-batch 2 — it does not wait for micro-batch 1 to reach the end of the pipeline.
This means that after a brief initial "fill" period (while the first micro-batch
propagates through all `p` stages), *every* GPU has useful work queued up almost all
of the time, because there's a different micro-batch at each stage simultaneously.

The bubble doesn't disappear entirely — there's still an unavoidable "fill" delay at
the start (waiting for the first micro-batch to reach the last stage) and a "drain"
delay at the end (waiting for the last micro-batch to finish flowing back through the
backward pass) — but its *relative* cost shrinks as the number of micro-batches `m`
grows. The standard bubble-fraction approximation for GPipe's schedule is:

```
bubble fraction ≈ (p - 1) / m
```

where `p` is the number of pipeline stages and `m` is the number of micro-batches per
batch. With `p = 4` stages and only `m = 1` micro-batch (the naive case), the bubble
fraction is `3/1 = 300%` of useful work time — catastrophic. With `m = 32`
micro-batches, it drops to `3/32 ≈ 9%` — a large improvement, at the cost of needing
to keep `m` sets of intermediate activations alive in memory simultaneously (GPipe
specifically flushes all micro-batch forward activations before running any backward
passes, which is memory-expensive).

**1F1B (one-forward-one-backward) scheduling**, introduced by PipeDream and refined by
Megatron-LM and DeepSpeed, improves on this further by interleaving forward and
backward micro-batch computation as early as possible rather than running all
forwards before any backwards: as soon as a GPU has produced a forward result and
handed it downstream, it looks for a backward micro-batch to process before moving
to the next forward, keeping the number of "in-flight" activations that need to stay
resident in memory much smaller than GPipe's all-forward-then-all-backward schedule,
while achieving a similar (or slightly better) bubble fraction. This memory
efficiency is what makes 1F1B (and its interleaved-stage variant, where each GPU owns
multiple *non-contiguous* chunks of layers to shrink the bubble further at the cost of
more communication) the schedule of choice in production frameworks like Megatron-LM
and DeepSpeed today, rather than plain GPipe.

---

## 5. Combining All Three: 3D Parallelism

None of the three strategies above is sufficient alone at frontier scale:

- **DP alone** scales throughput but every GPU still needs to hold the entire model —
  impossible once the model exceeds single-GPU memory.
- **TP alone**, scaled out to enough GPUs to hold a trillion-parameter model, would
  require TP's per-layer all-reduce to cross many nodes — and section 3.2 established
  that TP's communication pattern falls apart the moment it has to leave the NVLink
  domain of a single node.
- **PP alone**, scaled out to enough stages to hold the same model, suffers a bubble
  fraction that grows with the number of stages (`(p-1)/m`) — at some point you need
  an impractically large number of micro-batches (and impractically large activation
  memory) to keep the bubble small, and PP does nothing to speed up the *raw compute
  throughput* the way DP does.

Frontier training frameworks (Megatron-LM, DeepSpeed, and the training stacks behind
GPT-3/4-scale and PaLM-scale models) therefore combine all three simultaneously —
commonly called **3D parallelism** — and the specific way they're nested is not
arbitrary: it is a direct mapping of each strategy's communication pattern onto the
matching level of the hardware network hierarchy.

- **Tensor parallelism is used *within* a node**, across the GPUs connected by NVLink
  (typically 8-way on a single DGX/HGX server), because TP demands the highest
  bandwidth and lowest latency of the three (an all-reduce inside every layer, on the
  critical path).
- **Pipeline parallelism is used *across* groups of nodes**, because its
  point-to-point, boundary-only communication tolerates the lower bandwidth and
  higher latency of inter-node InfiniBand/Ethernet fabric far better than TP would.
- **Data parallelism is the outermost layer**, replicating the *entire (TP × PP)
  group* — not a single GPU — across as many such groups as there are GPUs available,
  because DP's all-reduce happens only once per step and its volume is
  bandwidth-bound rather than latency-bound, making it tolerant of running over
  whatever fabric connects separate racks or pods of the cluster.

```
Example: 64 GPUs, TP=4 (within a node), PP=4 (across 4 node-groups), DP=4 (replicas)

Node-group A (DP replica 0)      Node-group B (DP replica 1)   ... (x4 replicas)
┌─────────────────────────┐      ┌─────────────────────────┐
│ PP stage 0 (layers 1-8)  │      │ PP stage 0 (layers 1-8)  │
│  [GPU0 GPU1 GPU2 GPU3]   │◄──┐  │  [GPU0 GPU1 GPU2 GPU3]   │
│   \_______TP=4_______/   │   │  │   \_______TP=4_______/   │
├─────────────────────────┤   │  ├─────────────────────────┤
│ PP stage 1 (layers 9-16) │   │  │ PP stage 1 (layers 9-16) │
│  [GPU4 GPU5 GPU6 GPU7]   │  ...
├─────────────────────────┤        (PP stages chained across
│ PP stage 2 (layers17-24) │         node groups via slower
├─────────────────────────┤         inter-node network)
│ PP stage 3 (layers25-32) │
└─────────────────────────┘
        ▲                                      ▲
        └────────── DP all-reduce ─────────────┘
        (gradients synced across the 4 replicas, once/step,
         over the cluster's cross-pod fabric)
```

Each GPU's *global rank* is therefore addressed by a 3-tuple `(dp_rank, pp_rank,
tp_rank)`. A concrete example close to real published configurations: training a
~530B-parameter model (comparable to Megatron-Turing NLG) used 8-way tensor
parallelism (matching the 8 GPUs per DGX A100 node, all on NVLink), 35-way pipeline
parallelism (spanning groups of nodes), and roughly 12-way to 14-way data parallelism
on top, for a total of on the order of 3,000+ GPUs acting as one coordinated training
job. The specific split point between TP and PP degree is itself a tuning decision:
more TP shrinks per-GPU memory further but adds more per-layer communication; more PP
shrinks memory with cheaper communication but adds more bubble overhead; the DP degree
absorbs whatever GPU count is left over to hit the desired global batch size and
overall throughput target.

The unifying principle: **match the communication pattern to the fastest link that
can sustain it, and use the cheapest strategy to cross the slowest links.** NVLink
inside a node is reserved for TP's frequent, latency-critical, per-layer traffic;
the inter-node fabric carries PP's infrequent, point-to-point, boundary-only traffic
and DP's infrequent, once-per-step, bandwidth-only traffic.

---

## 6. Where ZeRO / FSDP Fits In

It's tempting to think of ZeRO (Zero Redundancy Optimizer, from Microsoft's DeepSpeed)
and PyTorch's FSDP (Fully Sharded Data Parallel) as a "fourth" parallelism strategy
alongside DP, TP, and PP. That framing is incorrect and worth correcting explicitly:
**ZeRO/FSDP is a memory-optimized re-implementation of data parallelism itself, not a
new axis along which the model or the batch is split.**

Recall from section 2.3 that plain DP's core limitation is that every replica holds a
*full, redundant copy* of the model's weights, gradients, and optimizer states. ZeRO's
insight is that this redundancy is unnecessary: instead of every one of the `N` DP
ranks storing 100% of the optimizer states (and, in later stages, gradients and even
parameters), each rank stores only a `1/N` shard of them, and ranks exchange
just-in-time slices of the full tensors via all-gather/reduce-scatter operations
exactly when a particular shard is needed for computation, then release it. The
*data-parallel structure* — full model participates in forward/backward on a batch
shard, gradients get synchronized across ranks — is completely preserved; what changes
is *how much of the model's state each rank has to physically hold in memory at
rest*, which is precisely a memory question, not a "how is the model/batch split"
question. The full breakdown of ZeRO's three stages (partitioning optimizer states,
then gradients, then parameters themselves — and FSDP's equivalent parameter-sharding
model) — along with the other memory levers (activation checkpointing, mixed
precision, CPU/NVMe offload) — is covered in depth in
[`../16-memory-optimization-training/README.md`](../16-memory-optimization-training/README.md);
this document's point is narrower: ZeRO-DP is *still data parallelism* in the sense
that matters for this document (it splits the *batch*, not the *model's layers or
weight matrices*), and it composes with TP and PP exactly the way plain DP does —
sitting in the outermost, replica-level position of the 3D-parallel hierarchy
described in section 5, just with a cheaper memory footprint per replica. Production
stacks frequently run "ZeRO-3D" or FSDP-plus-TP/PP hybrids for exactly this reason:
you get 3D parallelism's communication-topology-aware split of work *and* ZeRO's
reduction of redundant state within the outer DP dimension, simultaneously, because
the two are solving different problems (how work is distributed vs. how much memory
each participant needs) and don't conflict.

---

## 7. Side-by-Side: DP vs. TP vs. PP on 4 GPUs

The three strategies split fundamentally different things. The diagram below shows
all three applied to the same abstract model — 4 sequential layers, each containing
one large matrix multiply — distributed across 4 GPUs, so the contrast is direct:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA PARALLELISM: 4 full model replicas, each sees 1/4 of the data       │
├──────────────────────────────────────────────────────────────────────────┤
│  GPU0:[L1 L2 L3 L4] ◄─ data shard A     each GPU: full fwd+bwd on its    │
│  GPU1:[L1 L2 L3 L4] ◄─ data shard B     own data shard, independently,  │
│  GPU2:[L1 L2 L3 L4] ◄─ data shard C     then ALL-REDUCE gradients       │
│  GPU3:[L1 L2 L3 L4] ◄─ data shard D     across all 4 before opt.step()  │
│                                                                          │
│  comm: 1 all-reduce/step, sized to full model (~2x model size in bytes) │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ TENSOR PARALLELISM: 1 model, each layer's weights split 4 ways          │
├──────────────────────────────────────────────────────────────────────────┤
│  full batch X ──► [GPU0: 1/4 of L1,L2,L3,L4] ┐                          │
│                ──► [GPU1: 1/4 of L1,L2,L3,L4] ├─► combined via          │
│                ──► [GPU2: 1/4 of L1,L2,L3,L4] │   all-reduce/all-gather │
│                ──► [GPU3: 1/4 of L1,L2,L3,L4] ┘   INSIDE every layer    │
│                                                                          │
│  comm: ~2 all-reduces per layer (fwd+bwd), on the critical path —       │
│  requires NVLink; practical only within one node                       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ PIPELINE PARALLELISM: 1 model, layers split into 4 sequential stages    │
├──────────────────────────────────────────────────────────────────────────┤
│  batch X ──► GPU0:[L1] ──► GPU1:[L2] ──► GPU2:[L3] ──► GPU3:[L4] ──► out │
│              (fwd: activations flow left → right)                       │
│              (bwd: gradients flow right → left, GPU3 → GPU0)            │
│                                                                          │
│  comm: point-to-point activation/gradient hand-off at 3 boundaries only,│
│  cheap enough to cross node boundaries over InfiniBand/Ethernet          │
└──────────────────────────────────────────────────────────────────────────┘
```

The contrast to hold onto: DP duplicates the *whole model*, N times, and splits the
*data*. TP and PP both split the *model itself* (so no single GPU ever needs the full
thing) but along different axes — TP slices *inside* each layer (fine-grained, needs
every layer to talk, needs the fastest possible link), while PP slices *between*
layers (coarse-grained, only adjacent stages talk, tolerates a much slower link).

---

## 8. The Pipeline Bubble, Visualized

The diagram below shows a 4-stage pipeline (GPU0–GPU3), comparing the naive
single-batch schedule against a micro-batched schedule. `F` = forward pass on a
micro-batch, `B` = backward pass on a micro-batch, `.` = idle GPU (bubble). Time
advances left to right; each column is one time-step.

```
NAIVE (1 micro-batch = the whole batch, m = 1, p = 4 stages):

           t0   t1   t2   t3   t4   t5   t6   t7
  GPU0:  [ F  ] [ .  ] [ .  ] [ .  ] [ .  ] [ .  ] [ .  ] [ B  ]
  GPU1:  [ .  ] [ F  ] [ .  ] [ .  ] [ .  ] [ .  ] [ B  ] [ .  ]
  GPU2:  [ .  ] [ .  ] [ F  ] [ .  ] [ .  ] [ B  ] [ .  ] [ .  ]
  GPU3:  [ .  ] [ .  ] [ .  ] [ F  ] [ B  ] [ .  ] [ .  ] [ .  ]
                                     ▲
                              only ONE GPU is ever doing useful
                              work at any given time-step —
                              bubble fraction = (p-1)/m = 3/1 = 300%


MICRO-BATCHED (GPipe-style, m = 4 micro-batches, p = 4 stages):
  F_k / B_k = forward/backward on micro-batch k

           t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10
  GPU0: [ F1 ][ F2 ][ F3 ][ F4 ][ .  ][ .  ][ .  ][ B4 ][ B3 ][ B2 ][ B1 ]
  GPU1: [ .  ][ F1 ][ F2 ][ F3 ][ F4 ][ .  ][ B4 ][ B3 ][ B2 ][ B1 ][ .  ]
  GPU2: [ .  ][ .  ][ F1 ][ F2 ][ F3 ][ F4 ][ B4 ][ B3 ][ B2 ][ B1 ][ .  ] (shifted)
  GPU3: [ .  ][ .  ][ .  ][ F1 ][ F2 ][ F3 ][ F4 ][ B4 ][ B3 ][ B2 ][ .. ]
                    ▲                                              ▲
              fill: waiting for              drain: waiting for the
              micro-batch 1 to               last micro-batch's
              reach every stage              backward to flow all the
                                             way back to GPU0
        ── once "full", every GPU has a different micro-batch in flight
           simultaneously — bubble is now confined to the fill/drain
           edges, not the entire steady-state middle


1F1B (interleaves fwd/bwd per stage to cap in-flight activations,
similar bubble fraction to GPipe but far less peak activation memory):

  GPU0: [ F1 ][ F2 ][ F3 ][B1/F4][ B2 ][ B3 ][ B4 ]
  GPU1: [ .  ][ F1 ][ F2 ][B1/F3][B2/F4][ B3 ][ B4 ]
  GPU2: [ .  ][ .  ][ F1 ][B1/F2][B2/F3][B3/F4][ B4 ]
  GPU3: [ .  ][ .  ][ .  ][ F1  ][ B1  ][ ... ][ ... ]
        each GPU alternates forward/backward as soon as a backward
        becomes available, instead of finishing ALL forwards first —
        bounds the number of live activation buffers to ~p instead of ~m
```

As `m` (the number of micro-batches) grows relative to `p` (the number of pipeline
stages), the fill/drain edges shrink to a smaller and smaller fraction of total
run time, and the steady-state middle — where every GPU is continuously busy —
dominates. This is the entire reason production pipeline-parallel training always
splits batches into many small micro-batches rather than sending one batch through
the pipeline as a single unit.

---

## 9. Summary Comparison

| | Data Parallelism | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|---|
| **What is split** | The batch (data) | Individual weight matrices, within each layer | The model's layers, into sequential stages |
| **What each GPU holds** | Full model, full optimizer state | A shard of every layer's weights | A subset of contiguous layers, in full |
| **Solves capacity problem?** | No — full model still on every GPU | Yes | Yes |
| **Solves throughput problem?** | Yes | Partially (more GPUs compute in parallel, but per-layer comm overhead grows) | Partially (bubble limits ideal speedup) |
| **Communication pattern** | All-reduce over full gradient, once/step | All-reduce/all-gather, every layer, fwd + bwd | Point-to-point activation/gradient at stage boundaries only |
| **Communication frequency** | Low (once per step) | Very high (every layer, every step) | Moderate (a few hand-offs per micro-batch) |
| **Bandwidth/latency sensitivity** | Bandwidth-bound, latency-tolerant | Extremely latency- and bandwidth-sensitive | Latency-tolerant, low bandwidth needed |
| **Typical hardware scope** | Across nodes/pods, whole cluster | Within one node (NVLink only) | Across groups of nodes |
| **Key failure mode if misused** | None (just doesn't fix capacity) | Crossing nodes stalls every layer on network | Naive scheduling wastes most GPU-time in bubbles |
| **Memory-optimized variant** | ZeRO / FSDP (shards optimizer state/gradients/params across DP ranks) | — | 1F1B / interleaved scheduling (bounds live-activation memory) |

---

## 10. Practical Takeaways

- If your model **fits on one GPU** and you just need more throughput to finish
  training in reasonable wall-clock time, **use data parallelism** (optionally with
  ZeRO/FSDP if you also want to reduce per-GPU memory pressure to run a larger batch
  or model without adding TP/PP complexity).
- If your model **does not fit on one GPU**, you must use tensor parallelism and/or
  pipeline parallelism to physically split it — DP cannot help with this by itself.
- Keep **tensor parallelism inside a single node**, sized to the number of
  NVLink-connected GPUs available (commonly 4 or 8) — never stretch TP across the
  slower inter-node fabric.
- Use **pipeline parallelism to scale beyond one node's GPU count**, and always pair
  it with enough **micro-batches** (or an interleaved/1F1B schedule) to keep the
  bubble fraction `(p-1)/m` small — a pipeline with too few micro-batches per batch
  will underperform even a naive data-parallel setup.
- At frontier scale, expect all three to be active simultaneously in a 3D-parallel
  configuration, with **data parallelism as the outermost replication layer**, and
  ZeRO/FSDP applied within that outer DP dimension to further shrink the memory each
  replica needs to carry — see the memory-optimization README for exactly how that
  sharding works at the byte level.
