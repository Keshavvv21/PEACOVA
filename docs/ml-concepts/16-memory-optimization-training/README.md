# Training-Time Memory Optimization for Large Models

Training a transformer runs out of GPU memory long before it runs out of
compute. A GPU that can *execute* the matrix multiplications for a 7B-parameter
model in milliseconds still can't *fit* the model, because training needs far
more memory than the weights alone: a full copy of the gradients, several
copies of optimizer state, and a snapshot of every intermediate activation
needed to run backpropagation. This doc works through where every byte goes,
with real arithmetic, and then covers the techniques used in practice to fit
bigger models into the same GPU: mixed precision, gradient checkpointing,
gradient accumulation, ZeRO/FSDP sharding, CPU/NVMe offload, and — the most
effective lever of all for fine-tuning — not training most of the parameters
in the first place (LoRA/QLoRA).

Convention used throughout: 1 GB = 10⁹ bytes (decimal), matching how GPU specs
and most papers report memory. Where this doc cross-references this
repo's `memory_estimator.py` (which uses binary GiB = 2³⁰ bytes internally),
the GiB figure will read about 7% smaller than the equivalent GB figure for
the same byte count — flagged explicitly where it comes up.

## 1. Where training memory actually goes

Training memory splits into four categories. Three of them (weights,
gradients, optimizer states) are usually lumped together as "model states" —
they scale only with the number of parameters, not with batch size or
sequence length. The fourth (activations) scales with parameters *and* with
batch size, sequence length, and depth, which is what makes it the part that
blows up unpredictably.

### (a) Model weights

The parameters themselves: every attention projection matrix, every
feed-forward matrix, every embedding row. Memory cost is simply
`num_params × bytes_per_element`, where the element size depends on the
storage dtype:

| dtype | bytes/param |
|---|---|
| fp32 | 4 |
| fp16 / bf16 | 2 |
| int8 | 1 |
| int4 | 0.5 |

### (b) Gradients

Backpropagation produces one gradient value per parameter — `∂loss/∂w` for
every weight `w`. Structurally this is a tensor exactly the same shape as the
weight tensor, so it costs exactly the same amount of memory as the weights,
at whatever dtype the gradients are stored in.

### (c) Optimizer states

This is where SGD and Adam diverge sharply. Plain SGD with momentum needs one
extra buffer (the momentum accumulator) — 1× the weight memory. Adam (and
AdamW, which is what nearly all transformer training uses) tracks **two**
running statistics per parameter:

- **first moment estimate `m`** — an exponential moving average of the
  gradient (momentum)
- **second moment estimate `v`** — an exponential moving average of the
  squared gradient (used to adapt the per-parameter learning rate)

Each of `m` and `v` is a full tensor the same shape as the weights. So Adam's
optimizer state alone is **2× the weight memory**. Add that to the weights
(1×) and gradients (1×) already accounted for, and a naive Adam training
setup — everything stored at one uniform precision — costs **4× the weight
memory** before a single activation is stored:

```
weights (1x) + gradients (1x) + optimizer state m,v (2x) = 4x weight memory
```

### (d) Activation memory

Every layer's forward pass produces intermediate tensors — attention
projections, softmax outputs, the attention score matrix, feed-forward
hidden states, layer-norm statistics — and the backward pass needs every one
of them to compute local gradients via the chain rule. Unless something
special is done (§3), *all of them, for every layer,* have to stay resident
in memory from the forward pass until the backward pass reaches that layer.

This repo's own memory estimator (`orchestrator_framework_layer/akanade/backend/app/llm/memory_estimator.py:32-41`)
implements the standard per-layer activation formula:

```
elements_per_layer = 32·S·B·H + 4·A·S²·B
```

where `S` = sequence length, `B` = batch size, `H` = hidden size, `A` =
number of attention heads. The `32·S·B·H` term covers the self-attention
block's Q/K/V projections and output projection, the two feed-forward linear
layers plus activation function, and layer norms. The `4·A·S²·B` term is the
attention **score matrix** itself (`S×S` per head, per batch element) — the
term that makes activation memory grow *quadratically* with sequence length,
not linearly. `test_memory_estimator.py:44-54` verifies this quadratic
growth is real, not just claimed.

For training, multiply by `num_layers`, because every layer's activations
must be retained simultaneously (inference only needs one layer's worth live
at a time, since it streams layer-by-layer with no backward pass — see
`inference_memory_estimate()` at `memory_estimator.py:48-63`).

### Worked example: 7B model, fp16, Adam — weights + gradients + optimizer states

Take a 7B-parameter model (roughly Llama-2-7B's shape: 32 layers, hidden
size 4096, 32 attention heads) trained naively with everything — weights,
gradients, and both Adam moment buffers — stored in fp16 (2 bytes/element).

```
weights:      7e9 params × 2 bytes = 14,000,000,000 bytes = 14 GB
gradients:    7e9 params × 2 bytes = 14 GB
optimizer m:  7e9 params × 2 bytes = 14 GB
optimizer v:  7e9 params × 2 bytes = 14 GB
------------------------------------------------------------
total:                               56 GB
```

That's exactly the 4× relationship from §1(c): `4 × 14 GB = 56 GB`. Already
larger than a single consumer GPU's VRAM, and this is *before* a single
activation tensor has been allocated.

This naive fp16-everything number understates reality, though — real Adam
implementations don't keep the moment buffers in fp16, for numerical
stability reasons covered next. §2 refines this to the number that actually
shows up in production training runs: **112 GB** for the same 7B model.

## 2. Mixed precision training

Mixed precision training runs the forward and backward pass — the actual
matmuls — in a 16-bit format (fp16 or bf16), because Tensor Cores execute
16-bit matmuls at roughly 2-8× the throughput of fp32 matmuls and every
activation/weight/gradient tensor involved is half the size. But it does
**not** mean everything is fp16. Two additions are needed to make this
numerically stable:

**FP32 master weights.** The actual optimizer *update* — `w = w - lr ×
update` — involves adding a small update (scaled by a learning rate often in
the 1e-4 to 1e-5 range) to a weight value. In fp16, which has only 10
mantissa bits, an update smaller than about `2^-10` relative to the weight's
magnitude silently rounds to zero — the update vanishes and the parameter
never changes. So the canonical trick (Micikevicius et al., 2017) is to keep
one master copy of the weights in fp32, apply optimizer updates to *that*
copy at full precision, and only round down to fp16 for the forward/backward
pass. The fp16 copy is what actually multiplies against activations; the
fp32 copy is what accumulates the true, precise value over thousands of
tiny updates.

**Loss scaling.** fp16 has a narrow exponent range (5 bits, dynamic range
down to about `6e-5` before flushing to zero) compared to fp32's 8-bit
exponent. Gradients late in training, or gradients for parameters far from
the loss (early layers), are often small enough that they underflow this
range and are stored as exact zero — silently killing learning for that
parameter. Loss scaling fixes this cheaply: multiply the loss by a scale
factor `S` (e.g. 1024, or a value chosen adaptively) right before calling
`.backward()`. By linearity of differentiation, every gradient in the
backward pass is now also scaled by `S`, which shifts small gradients up
into fp16's representable range instead of flushing them to zero. After
backward, **before** the optimizer step, all gradients are divided by `S`
again to undo the scaling (typically fused into the optimizer's gradient
unscale step, alongside gradient-norm inspection for skipping the step on
`inf`/`NaN`, which is how dynamic loss scaling auto-tunes `S` upward when
things are stable and downward when it detects overflow).

**Why bf16 has mostly retired loss scaling.** bf16 ("brain float 16") uses
the *same* 8-bit exponent field as fp32 — identical dynamic range, from
about `1e-38` to `3e38` — and only shrinks the mantissa to 7 bits. The
underflow-to-zero failure mode that loss scaling exists to solve is
specifically an *exponent range* problem, and bf16 doesn't have that
problem: a gradient that would underflow fp16's exponent range is still
perfectly representable in bf16 (just with less precision in the mantissa,
which is a *rounding* cost, not a total-loss-of-signal cost). This is why
modern large-model training (most LLM pretraining on A100/H100-class
hardware since roughly 2022) defaults to bf16 mixed precision with **no**
loss scaler at all — one less piece of training-stability machinery, one
less hyperparameter (the initial scale factor and growth/backoff intervals)
to tune, at the cost of bf16's coarser mantissa introducing more rounding
noise per individual op than fp16 would (which turns out to matter much less
in practice than exponent underflow does).

### The refined memory number

Because the master weights and Adam's `m`/`v` buffers need fp32 for
stability, the realistic per-parameter memory cost of mixed-precision Adam
training is:

```
fp16 weights (forward/backward compute copy):  2 bytes
fp16 gradients:                                 2 bytes
fp32 master weights:                            4 bytes
fp32 momentum (m):                              4 bytes
fp32 variance (v):                              4 bytes
------------------------------------------------------
total:                                          16 bytes/param
```

This is the widely-cited "16Ψ bytes" figure from the ZeRO paper
(Rajbhandari et al., 2019), where Ψ is the parameter count. For the same 7B
model:

```
16 bytes × 7,000,000,000 params = 112,000,000,000 bytes = 112 GB
```

That's **double** the naive fp16-only estimate from §1 (56 GB → 112 GB) —
and notably, it comes out to exactly the same number as if you'd trained the
whole thing in *pure fp32* with the naive 4× rule (`4 × 4 bytes/param × 7e9 =
112 GB`). Mixed precision training doesn't shrink the weight+gradient+
optimizer memory footprint at all relative to plain fp32 training — the fp32
master weight and moment buffers are exactly as large as they'd be in a
pure-fp32 run. What mixed precision actually buys you is **activation
memory** (activations are computed and stored in fp16/bf16, half the size of
fp32 activations) and **speed** (Tensor Core throughput on 16-bit matmuls),
not a reduction in the optimizer-state category.

## 3. Gradient checkpointing (activation checkpointing)

Continuing the 7B example: with `S=2048`, `B=1`, `H=4096`, `A=32`, plug into
the activation formula from §1(d):

```
elements/layer = 32·S·B·H + 4·A·S²·B
              = 32×2048×1×4096  +  4×32×2048²×1
              = 268,435,456     +  536,870,912
              = 805,306,368 elements

bytes/layer (fp16, 2 bytes/element) = 1,610,612,736 bytes ≈ 1.61 GB

× 32 layers ≈ 51.5 GB just for activations, at batch size 1
```

Add that to the 112 GB from §2 and a single 7B model, at batch size 1,
needs roughly **164 GB** to train with plain full fine-tuning — more than a
single 80 GB H100/A100, and activation memory only grows *linearly* from
here as batch size increases (batch size 4 would need on the order of 200 GB
just for activations). This is the concrete motivation for gradient
checkpointing.

### The mechanism

Instead of keeping every layer's activations resident from forward pass to
backward pass, gradient checkpointing (Chen et al., 2016, "Training Deep
Nets with Sublinear Memory Cost") keeps only a handful of **checkpoint**
activations — the input to every `c`-th layer — and discards everything in
between during the forward pass. When the backward pass reaches a segment
whose interior activations were discarded, it **re-runs the forward pass for
just that segment** from its nearest checkpoint, recomputing the interior
activations on the spot, uses them immediately for the local backward
computation, and discards them again once done.

### The sqrt(n) result

With `L` layers and a checkpoint placed every `c` layers, peak memory has two
components: the checkpoints themselves (`L/c` of them) and the activations
being recomputed for the segment currently under the backward pass (up to
`c` of them). Peak memory is proportional to `L/c + c`. Minimizing this over
`c` (by calculus, or just AM-GM) gives the optimum at `c = √L`, where the
memory bound becomes proportional to `2√L` — **O(√L) instead of O(L)**.

For the 7B model's 32 layers: `√32 ≈ 5.66`, round to checkpointing every 6
layers. Peak stored activations scale down to roughly `2×√32 ≈ 11.3`
layer-equivalents out of 32, i.e. about **35%** of the uncheckpointed
activation memory:

```
51.5 GB × 0.35 ≈ 18 GB
```

Activation memory drops from ~51.5 GB to ~18 GB — a ~65% reduction — pulling
the model's total memory need from ~164 GB down to roughly `112 + 18 = 130
GB`. Still large (this is why checkpointing is usually combined with ZeRO or
sharding, §5), but a substantial and real reduction from one technique.

### The cost

The trade is compute, not just a free lunch: every checkpointed segment's
forward pass runs **twice** — once during the actual forward pass (to get
to the next checkpoint) and once again during the backward pass (to
regenerate the discarded interior activations). This adds roughly one extra
forward pass's worth of FLOPs on top of the normal forward+backward, which
in practice measures out to **about 20-30% wall-clock slowdown** per
training step (not more, because the recomputed forward passes overlap
reasonably well with backward-pass compute and don't add proportionally to
total FLOPs — a training step is normally ~1 forward + 2 backward-equivalent
FLOPs, and checkpointing adds roughly one more forward's worth on top).

This repo's real nightly LoRA fine-tune job
(`scripts/finetune_local_slm.py`, documented in
`orchestrator_framework_layer/akanade/README_MEMORY_MANAGEMENT.md:119-128`)
explicitly does **not** use gradient checkpointing — and says so for a
concrete, checkable reason rather than by omission: at LoRA's ~0.88%
trainable-parameter scale the job already fits comfortably in CPU RAM (see
§7), so there's no memory pressure to trade compute for. That's the right
way to reason about checkpointing generally: it's a lever to pull when
activation memory is the binding constraint, not a default to flip on
regardless.

## 4. Gradient accumulation

Gradient accumulation simulates a larger batch size than fits in memory at
once by splitting the target batch into several **micro-batches**, running
forward+backward on each micro-batch separately, summing (accumulating)
their gradients in the `.grad` buffers, and only calling `optimizer.step()`
once every `accumulation_steps` micro-batches:

```
effective_batch_size = micro_batch_size × accumulation_steps × data_parallel_world_size
```

Mechanically: gradients are additive across examples in a batch (the loss
for a batch is an average, so gradients accumulate linearly before that
average is taken — implementations typically accumulate raw sums and divide
by `accumulation_steps` at the end, or scale the loss by `1/accumulation_steps`
per micro-batch before calling backward). So running 4 micro-batches of size
8 and summing gradients before stepping the optimizer is mathematically
equivalent (up to how the loss is normalized) to one batch of size 32.

**Why it doesn't cost extra activation memory.** The key detail is *when*
activations get freed. For micro-batch `i`, the forward pass builds a
computation graph and stores its activations; the backward pass for
micro-batch `i` consumes those activations to produce gradients, and once
that backward pass completes, the autograd engine frees the graph and its
activations — because gradients (which are being *accumulated* into the
persistent `.grad` tensors) are all that need to survive past this point,
not the activations that produced them. Micro-batch `i+1`'s forward pass
then allocates its own activations from scratch, using the memory just
freed. So peak activation memory at any instant is determined by **one
micro-batch's activation footprint**, never by the effective (accumulated)
batch size. Only the (much smaller) gradient buffers — one accumulator the
size of the model's trainable parameters — persist across the whole
accumulation window.

This is what makes gradient accumulation the standard first lever for
"model doesn't fit at the batch size I want": it doesn't reduce the model
states categories at all (weights/gradients/optimizer state are unaffected —
same size regardless of accumulation steps), but it decouples the *effective*
batch size a training run achieves from the *peak* activation memory any
single forward/backward pass needs, at the cost of more sequential steps
(and thus wall-clock time) to reach the same number of optimizer updates —
no compute is wasted, but the extra micro-batches can't be pipelined into
one bigger matmul the way a real larger batch would.

## 5. ZeRO (Zero Redundancy Optimizer)

Standard data parallelism (DP) replicates the *entire* model state — weights,
gradients, and optimizer state, the full 16Ψ bytes from §2 — on **every**
GPU in the data-parallel group. Each GPU processes a different data shard
and only the gradients are averaged (all-reduced) across GPUs; but every GPU
redundantly holds an identical, full copy of everything else. For `N` GPUs,
this means `N` copies of the same 16Ψ bytes exist across the cluster, even
though no single GPU needs anything but its own shard of the data.

ZeRO (Rajbhandari, Rasley, Ruwase, He — "ZeRO: Memory Optimizations Toward
Training Trillion Parameter Models", 2019/2020, implemented in Microsoft's
DeepSpeed) eliminates this redundancy by **partitioning** the model state
across the `N` GPUs in the data-parallel group instead of replicating it,
then reconstructing whatever a GPU actually needs, just in time, via
collective communication (all-gather / reduce-scatter). It has three
progressively more aggressive stages, applied to the three model-state
categories from §1(a)-(c) one at a time (activations are handled
separately, by combining ZeRO with checkpointing — it's an orthogonal
technique to §3).

Using Ψ = parameter count, `N_d` = data-parallel degree (number of GPUs
sharing the shard), and the 16-bytes/param breakdown from §2 (2Ψ fp16
weights + 2Ψ fp16 grads + 4Ψ fp32 master + 4Ψ fp32 momentum + 4Ψ fp32
variance = 12Ψ of that is "optimizer state" and 4Ψ is "weights+gradients"):

**ZeRO-1 — partition optimizer states.** Each GPU keeps a full replica of
the weights and gradients, but only owns a `1/N_d` shard of the 12Ψ bytes of
optimizer state. Per-GPU memory: `4Ψ + 12Ψ/N_d`. At the end of the backward
pass, each GPU updates only its own shard of the parameters (using its
shard of the optimizer state and a reduce-scattered slice of the gradient),
then an all-gather redistributes the updated parameter shards so every GPU
again has the full weight tensor for the next forward pass.

**ZeRO-2 — also partition gradients.** Once a GPU only owns a shard of the
optimizer state, it only ever *needs* the matching shard of the gradient —
there's no reason to keep a full gradient replica around. ZeRO-2 uses
reduce-scatter (instead of all-reduce) so each GPU ends up holding only its
own `1/N_d` shard of the gradients, immediately releasing the rest. Per-GPU
memory: `2Ψ + 14Ψ/N_d` (full weights still replicated; grad+optimizer both
sharded).

**ZeRO-3 — also partition the weights themselves.** The most aggressive
stage: no GPU holds a full copy of the weights at rest either. Each GPU
permanently owns only a `1/N_d` shard of the parameters. During the forward
pass, just before each layer executes, an all-gather collects that layer's
full weight tensor from all shards, uses it for the forward computation,
then **immediately discards** the gathered copy (keeping only its own
permanent shard). The same just-in-time all-gather happens again during the
backward pass for that layer. Per-GPU memory: `16Ψ/N_d` — a **fully linear**
reduction with the number of GPUs, no floor.

| Stage | Per-GPU memory | At `N_d`→∞ | Communication vs. baseline DP |
|---|---|---|---|
| Baseline DP (no ZeRO) | `16Ψ` | `16Ψ` (no reduction) | 1× (one all-reduce of gradients) |
| ZeRO-1 | `4Ψ + 12Ψ/N_d` | → `4Ψ` (floor) | ~1× (same as baseline DP) |
| ZeRO-2 | `2Ψ + 14Ψ/N_d` | → `2Ψ` (floor) | ~1× (same as baseline DP) |
| ZeRO-3 | `16Ψ/N_d` | → `0` (no floor) | ~1.5× (extra all-gathers of weights in fwd *and* bwd) |

Concretely, for the running 7B-parameter example (`Ψ = 7×10⁹`, `16Ψ = 112
GB`), per-GPU memory at increasing data-parallel width:

| `N_d` (GPUs) | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|
| 1 | 112 GB | 112 GB | 112 GB |
| 4 | 49 GB | 38.5 GB | 28 GB |
| 8 | 38.5 GB | 26.25 GB | 14 GB |
| 16 | 33.25 GB | 20.1 GB | 7 GB |
| 64 | 29.3 GB | 15.5 GB | 1.75 GB |

Notice ZeRO-1 and ZeRO-2 both hit a floor (28 GB and 14 GB respectively) no
matter how many GPUs are added, because they never shard the weights
themselves — the un-sharded piece dominates once `N_d` is large. ZeRO-3 has
no floor: it keeps dividing by `N_d`, which is exactly the property that
lets it scale to genuinely enormous models (the "trillion parameter" in the
paper's title) given enough GPUs, at the cost of the extra weight-gathering
communication every single layer, every single step.

### PyTorch FSDP

PyTorch's native **Fully Sharded Data Parallel** (`torch.distributed.fsdp`,
`FullyShardedDataParallel`) implements the same core idea as ZeRO-3 without
requiring the DeepSpeed dependency: parameters, gradients, and optimizer
state are all sharded across the data-parallel group, and each layer's full
weight is materialized via all-gather just before it's used and freed right
after. FSDP's `ShardingStrategy` enum makes the ZeRO-stage correspondence
explicit: `FULL_SHARD` ≈ ZeRO-3, `SHARD_GRAD_OP` ≈ ZeRO-2 (shard gradients
and optimizer state, keep full parameters resident), and `NO_SHARD` ≈
plain DP. It composes with mixed precision (`MixedPrecision` config),
activation checkpointing (`torch.utils.checkpoint` wraps individual
transformer blocks, exactly the technique from §3), and CPU offload
(`CPUOffload(offload_params=True)`, §6) as independent, stackable knobs
rather than a single DeepSpeed-specific configuration surface.

## 6. CPU / NVMe offloading

Sharding (§5) helps only if the model state, once divided by `N_d`, fits in
the *combined* GPU memory of the cluster. When it doesn't — a single GPU, or
a model too large even for the full cluster's aggregate HBM — the remaining
lever is **offloading**: physically moving tensors that aren't needed *this
instant* out of GPU HBM and into CPU RAM, or further, onto NVMe SSDs, and
streaming them back over PCIe when needed.

**ZeRO-Offload** (2021) moves the optimizer states and the fp32 master
weights to CPU RAM (which is typically far larger and cheaper per GB than
GPU HBM), and performs the actual Adam update **on the CPU** using a
vectorized (AVX-optimized), multi-threaded implementation. Only the fp16
gradients (computed on GPU) need to cross PCIe to the CPU, and only the
updated fp16 weights need to cross back — the large fp32 optimizer state
(12Ψ bytes, the dominant chunk) never has to move, it just sits in CPU RAM
and gets read/written in place by the CPU-side optimizer step.

**ZeRO-Infinity** (2021) extends this one level further: when even CPU RAM
isn't enough, model states spill to **NVMe** SSDs, using a dedicated I/O
engine to prefetch the tensors about to be needed and overlap that I/O with
GPU compute. This decouples trainable model size from GPU *and* CPU RAM
capacity entirely — the practical ceiling becomes disk capacity, which is
why ZeRO-Infinity's headline claim in the paper is fine-tuning models with
tens of trillions of parameters on a single node.

The catch is bandwidth. Every one of these hops is dramatically slower than
GPU HBM:

| Path | Approximate bandwidth | Slowdown vs. GPU HBM |
|---|---|---|
| GPU HBM (A100, ~2 TB/s / H100, ~3.3 TB/s) | ~2,000-3,300 GB/s | 1× (baseline) |
| PCIe Gen4 x16 (GPU ↔ CPU) | ~32 GB/s | ~60-100× slower |
| CPU DRAM (DDR4/5, per socket) | ~20-50 GB/s | ~50-150× slower |
| Single NVMe SSD (sequential) | ~3.5-7 GB/s | ~300-900× slower |
| RAID of several NVMe drives in parallel | ~20-30 GB/s | ~70-150× slower |

That two-to-three-orders-of-magnitude gap is exactly why offloading is a
capability-of-last-resort lever, not a default: it turns a compute-bound
training step into one gated by data movement, so wall-clock time per step
can go up several-fold. The trade is legitimate when the alternative is "the
job doesn't run at all" — a 65B-parameter model that has no way to fit in
GPU+CPU RAM together can still be *trained*, just slowly, by spilling to
NVMe, whereas without offloading it simply can't start.

## 7. How LoRA/QLoRA sidesteps most of this problem

Everything in §1-§6 is about how to fit the *full* 16Ψ-bytes-per-parameter
model-state cost (§2) into available memory, by sharding it, offloading it,
or shrinking activation memory around it. LoRA takes a different approach
entirely: it makes most of that cost **not exist in the first place**.

The full LoRA/QLoRA math — how the low-rank update `ΔW = BA` is
parameterized, what rank `r` and `alpha` control, and how QLoRA's NF4
quantization plus double quantization and paged optimizers work — is
covered in [`08-lora-qlora/README.md`](../08-lora-qlora/README.md) and
isn't re-derived here. What matters for memory accounting is one structural
fact: **the base model's weights are frozen**. Frozen means `requires_grad =
False` — no gradient is ever computed for those parameters, and since Adam's
`m`/`v` buffers only exist for parameters the optimizer is tracking, there
is no optimizer state for them either. Categories (b) and (c) from §1 — the
2× (or in mixed precision, 14× per §2) memory cost that dominates full
fine-tuning — simply **don't get allocated** for the frozen base weights.
Only the small LoRA adapter matrices (`A` and `B`, together typically well
under 1% of the base model's parameter count) are trainable, so gradients
and optimizer state only ever exist for that tiny slice.

This repo's own nightly fine-tune job (`scripts/finetune_local_slm.py`) is a
real, measured instance of this, and `memory_estimator.lora_finetune_memory_estimate()`
(`orchestrator_framework_layer/akanade/backend/app/llm/memory_estimator.py:66-99`)
computes it directly: at that job's real configuration (`r=8, alpha=16` on
the attention/MLP projections), **~0.88% of parameters are trainable**. Applied
to a 500M-parameter model, full fine-tuning's optimizer state alone would
cost `500,000,000 × 14 bytes = 7 GB`; LoRA's trainable slice is `500,000,000
× 0.0088 ≈ 4.4M` parameters, so its optimizer state costs `4.4M × 14 bytes ≈
61.6 MB` — a **>100× reduction**, asserted directly in
`test_memory_estimator.py:57-69`'s
`test_lora_finetune_estimate_matches_the_real_nightly_job_shape` (which
checks LoRA's optimizer footprint against `full_finetune_optimizer_gb / 50`
and passes with room to spare).

Scaling that same ~0.88% trainable fraction up to the running 7B example
from §1-§3 (the exact fraction in practice depends on rank and which
projections are targeted — see 08-lora-qlora for how `r` determines
parameter count — but this is a realistic ballpark for a small-rank LoRA
config):

```
trainable params:            7e9 × 0.0088 ≈ 61.6M

Adam optimizer + grad state for ONLY the adapters
  (14 bytes/trainable-param: 2 fp16 grad + 4 fp32 master + 4+4 fp32 m,v):
                              61.6M × 14 bytes ≈ 0.86 GB

Base weights, LoRA (bf16, frozen):      7e9 × 2 bytes    = 14 GB
Base weights, QLoRA (int4, frozen):     7e9 × 0.5 bytes  = 3.5 GB
```

Compare full totals for the 7B model, batch size 1, sequence length 2048:

| Config | Weights | Grad+optimizer (trainable only) | Activations | Total |
|---|---|---|---|---|
| Full fine-tune (§1-§3, mixed precision, no checkpointing) | 14 GB | 98 GB | 51.5 GB | **~164 GB** |
| Full fine-tune + gradient checkpointing | 14 GB | 98 GB | ~18 GB | **~130 GB** |
| LoRA (bf16 base, no checkpointing) | 14 GB | 0.86 GB | 51.5 GB | **~66 GB** |
| LoRA (bf16 base) + gradient checkpointing | 14 GB | 0.86 GB | ~18 GB | **~33 GB** |
| QLoRA (int4 base) + gradient checkpointing | 3.5 GB | 0.86 GB | ~18 GB | **~22 GB** |

Two things worth being precise about here, because they're easy to
overstate: LoRA's weight+gradient+optimizer reduction (98 GB → 0.86 GB, on
the *trainable* categories) is dramatic and matches the repo's measured
>100× number for optimizer state specifically — but **activation memory is
untouched by LoRA on its own**. Backpropagation still has to flow through
every frozen layer to reach the adapters (the chain rule doesn't care that a
layer's weights are frozen — it still needs that layer's input activation to
compute the gradient of the *next* layer down), so a naive LoRA run still
pays the full ~51.5 GB activation cost from §3. This is exactly why real
QLoRA implementations enable gradient checkpointing by default rather than
treating it as optional — the two techniques are complementary, and
LoRA/QLoRA's famous "fine-tune a 65B model on one GPU" results depend on
combining low-rank adapters (kills grad+optimizer memory) with checkpointing
(kills most activation memory) and 4-bit base quantization (kills weight
memory) simultaneously, not on any single one of the three.

## 8. Memory breakdown — full fine-tuning vs. LoRA

Visualizing the same 7B-model, mixed-precision, batch-size-1, seq-2048
numbers from §7 (each `█` ≈ 2 GB; both rows use gradient checkpointing, so
the two runs are compared on equal footing for activation memory):

```
                          0        20        40        60        80       100  110  GB
                          |----|----|----|----|----|----|----|----|----|----|----|
FULL FINE-TUNE
  weights      (14 GB)    ███████
  gradients    (14 GB)    ███████
  optimizer    (84 GB)    ██████████████████████████████████████████
  activations  (18 GB)    █████████
                          |----|----|----|----|----|----|----|----|----|----|----|
  TOTAL: ~130 GB          ███████████████████████████████████████████████████████████████

LoRA (bf16 base)
  weights      (14 GB)    ███████         <- frozen base, unavoidable, same as above
  gradients  (0.06 GB)    ▏               <- adapters only: ~61.6M params
  optimizer  (0.80 GB)    ▎               <- adapters only: m + v + fp32 master
  activations  (18 GB)    █████████       <- backward still flows through frozen layers
                          |----|----|----|----|----|----|----|----|----|----|----|
  TOTAL: ~33 GB           ████████████████▍

QLoRA (int4 base)
  weights     (3.5 GB)    █▊              <- frozen base, quantized to int4
  gradients  (0.06 GB)    ▏
  optimizer  (0.80 GB)    ▎
  activations  (18 GB)    █████████
                          |----|----|----|----|----|----|----|----|----|----|----|
  TOTAL: ~22 GB           ███████████▍
```

The gradients and optimizer bars are the ones that nearly vanish — that's
the direct consequence of §7's "no trainable parameters, no gradient, no
optimizer state" argument. The weights bar shrinks only when the base model
is *also* quantized (QLoRA), and the activations bar is unaffected by either
LoRA or QLoRA on their own — it only shrinks because checkpointing (§3) was
applied to both rows equally.

## 9. ZeRO stages — what's replicated vs. sharded, per GPU

Four GPUs, ZeRO's three stages plus the baseline they improve on. `[FULL]`
means every GPU holds an identical, complete copy of that tensor (redundant
across the group); `[1/4: shard k]` means each GPU holds only its own
one-quarter slice, with no overlap between GPUs — the four shards together
add up to exactly one full copy, cluster-wide.

```
BASELINE DATA PARALLELISM — nothing sharded, everything replicated
┌────────┬─────────────────┬─────────────────┬─────────────────────┐
│        │     Weights     │    Gradients     │  Optimizer States    │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ GPU 0  │     [FULL]      │     [FULL]      │       [FULL]         │
│ GPU 1  │     [FULL]      │     [FULL]      │       [FULL]         │
│ GPU 2  │     [FULL]      │     [FULL]      │       [FULL]         │
│ GPU 3  │     [FULL]      │     [FULL]      │       [FULL]         │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ bytes  │       2Ψ        │       2Ψ        │         12Ψ          │
└────────┴─────────────────┴─────────────────┴─────────────────────┘
                                                per-GPU total: 16Ψ bytes

ZeRO-1 — shard the optimizer states only
┌────────┬─────────────────┬─────────────────┬─────────────────────┐
│        │     Weights     │    Gradients     │  Optimizer States    │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ GPU 0  │     [FULL]      │     [FULL]      │  [1/4: shard 0]      │
│ GPU 1  │     [FULL]      │     [FULL]      │  [1/4: shard 1]      │
│ GPU 2  │     [FULL]      │     [FULL]      │  [1/4: shard 2]      │
│ GPU 3  │     [FULL]      │     [FULL]      │  [1/4: shard 3]      │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ bytes  │       2Ψ        │       2Ψ        │       12Ψ/4          │
└────────┴─────────────────┴─────────────────┴─────────────────────┘
                                                per-GPU total: 4Ψ + 12Ψ/4 = 7Ψ bytes

ZeRO-2 — also shard the gradients
┌────────┬─────────────────┬─────────────────┬─────────────────────┐
│        │     Weights     │    Gradients     │  Optimizer States    │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ GPU 0  │     [FULL]      │  [1/4: shard 0]  │  [1/4: shard 0]      │
│ GPU 1  │     [FULL]      │  [1/4: shard 1]  │  [1/4: shard 1]      │
│ GPU 2  │     [FULL]      │  [1/4: shard 2]  │  [1/4: shard 2]      │
│ GPU 3  │     [FULL]      │  [1/4: shard 3]  │  [1/4: shard 3]      │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ bytes  │       2Ψ        │       2Ψ/4       │       12Ψ/4          │
└────────┴─────────────────┴─────────────────┴─────────────────────┘
                                                per-GPU total: 2Ψ + 14Ψ/4 = 5.5Ψ bytes

ZeRO-3 — also shard the weights (≈ PyTorch FSDP's FULL_SHARD)
┌────────┬─────────────────┬─────────────────┬─────────────────────┐
│        │     Weights     │    Gradients     │  Optimizer States    │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ GPU 0  │  [1/4: shard 0] │  [1/4: shard 0]  │  [1/4: shard 0]      │
│ GPU 1  │  [1/4: shard 1] │  [1/4: shard 1]  │  [1/4: shard 1]      │
│ GPU 2  │  [1/4: shard 2] │  [1/4: shard 2]  │  [1/4: shard 2]      │
│ GPU 3  │  [1/4: shard 3] │  [1/4: shard 3]  │  [1/4: shard 3]      │
├────────┼─────────────────┼─────────────────┼─────────────────────┤
│ bytes  │       2Ψ/4       │       2Ψ/4       │       12Ψ/4          │
└────────┴─────────────────┴─────────────────┴─────────────────────┘
                                                per-GPU total: 16Ψ/4 = 4Ψ bytes
        (full per-layer weight shard is all-gathered on demand during
         fwd/bwd, used, then immediately freed — never resident at rest)
```

Reading down the "Weights" column is the whole story: ZeRO-1 and ZeRO-2
leave it at `[FULL]` on every row, which is exactly why both stages hit a
memory floor as `N_d` grows (§5's table) — that unsharded 2Ψ never shrinks.
ZeRO-3 is the only stage where every cell in every column is a shard, which
is why it's the only one whose per-GPU cost keeps falling as `1/N_d` with no
floor.

## Putting it together

None of these techniques are exclusive alternatives — production training
runs stack most of them simultaneously:

| Technique | Reduces | Cost | Typical use |
|---|---|---|---|
| Mixed precision (bf16) | Activation memory, compute time | Slight accuracy/rounding noise; fp16 needs loss scaling | Default for nearly all training today |
| Gradient checkpointing | Activations: O(L) → O(√L) | ~20-30% more compute (extra forward passes) | When activations are the binding constraint |
| Gradient accumulation | Nothing in peak memory; decouples effective batch size from per-step activation memory | More sequential steps, same total FLOPs | Simulating a large batch on limited-memory GPUs |
| ZeRO-1/2 (DeepSpeed) | Optimizer state, gradients | Communication ≈ same as baseline DP | Multi-GPU training where optimizer state dominates |
| ZeRO-3 / FSDP | Weights, gradients, optimizer state — all of it, linearly in `N_d` | ~1.5× communication (extra weight all-gathers) | Models too large for any single GPU |
| CPU/NVMe offload (ZeRO-Offload/-Infinity) | GPU memory pressure, at any scale | Large slowdown (PCIe/NVMe bandwidth, 60-900× slower than HBM) | Model doesn't fit in GPU+CPU RAM combined at all |
| LoRA/QLoRA | Gradient + optimizer memory for ~99%+ of parameters; weight memory too if base is quantized | Slightly lower fine-tuning expressiveness than full FT; activation memory unaffected unless paired with checkpointing | Fine-tuning (not pretraining) a model whose base weights don't need to change |

## References in this repo

- `orchestrator_framework_layer/akanade/backend/app/llm/memory_estimator.py` —
  real implementations of `param_memory_bytes()`, the activation formula
  (`_activation_elements()`, lines 32-41), `inference_memory_estimate()`
  (lines 48-63), `lora_finetune_memory_estimate()` (lines 66-99, the
  frozen-base + trainable-adapter accounting this doc's §7 walks through),
  and `quantization_savings()` (lines 102-111).
- `orchestrator_framework_layer/akanade/backend/tests/test_memory_estimator.py` —
  the assertions that keep those formulas honest, including the
  >100× LoRA-vs-full-fine-tune optimizer-memory test (lines 57-69) and the
  quadratic-in-sequence-length activation growth test (lines 44-54).
- `orchestrator_framework_layer/akanade/README_MEMORY_MANAGEMENT.md` — this
  repo's own memory-sizing writeup for its real nightly LoRA job, including
  the explicit, reasoned decision *not* to use gradient checkpointing at its
  current model/parameter scale.
- [`08-lora-qlora/README.md`](../08-lora-qlora/README.md) — the LoRA/QLoRA
  math this doc deliberately doesn't re-derive: low-rank decomposition,
  `r`/`alpha` hyperparameters, NF4 quantization, and how
  `scripts/finetune_local_slm.py` applies them.
- [`19-distributed-training/README.md`](../19-distributed-training/README.md) —
  data/tensor/pipeline parallelism, the broader context ZeRO/FSDP sharding
  sits inside.
- [`15-quantization/README.md`](../15-quantization/README.md) — INT8/INT4
  quantization techniques underlying QLoRA's frozen-base memory reduction.
