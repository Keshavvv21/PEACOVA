# LoRA & QLoRA — Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) is the family of techniques that let you specialize a
multi-billion-parameter language model on new data without paying the memory and compute cost of
updating every one of its parameters. LoRA (Low-Rank Adaptation) is the dominant technique in that
family, and QLoRA is the trick that combines LoRA with 4-bit quantization of the frozen base model
so the whole thing fits on a single consumer or prosumer GPU. This document builds up the math from
first principles, walks every LoRA hyperparameter, explains merging vs. adapter-swapping, explains
QLoRA's three pieces (NF4, double quantization, paged optimizers), and then ties all of it to the
two real, running implementations in this repo:
[`finetune_local_slm.py`](../../../orchestrator_framework_layer/akanade/backend/scripts/finetune_local_slm.py)
(the nightly production pipeline) and
[`quickstart_lora_example/`](../../../orchestrator_framework_layer/akanade/backend/scripts/quickstart_lora_example/)
(a minimal single-file teaching example).

## Table of contents

1. [The problem: full fine-tuning doesn't scale](#1-the-problem-full-fine-tuning-doesnt-scale)
2. [LoRA's core insight and math](#2-loras-core-insight-and-math)
3. [Every LoRA hyperparameter, in detail](#3-every-lora-hyperparameter-in-detail)
4. [Merging vs. keeping adapters separate](#4-merging-vs-keeping-adapters-separate)
5. [QLoRA: fitting a 7B–70B model on one GPU](#5-qlora-fitting-a-7b70b-model-on-one-gpu)
6. [Standard fine-tuning hyperparameters that still apply](#6-standard-fine-tuning-hyperparameters-that-still-apply)
7. [Block diagram: the parallel low-rank path](#7-block-diagram-the-parallel-low-rank-path)
8. [How this repo uses it](#8-how-this-repo-uses-it)
9. [References](#9-references)

---

## 1. The problem: full fine-tuning doesn't scale

To understand why LoRA exists, first count what full fine-tuning actually requires you to hold in
GPU memory simultaneously for a model with `N` parameters, trained with the Adam/AdamW optimizer in
mixed precision (the standard recipe):

| Quantity | Precision | Bytes per parameter | Total for `N` params |
|---|---|---|---|
| Model weights | fp16/bf16 | 2 | `2N` |
| Gradients | fp16/bf16 | 2 | `2N` |
| Adam optimizer state — first moment (`m`) | fp32 | 4 | `4N` |
| Adam optimizer state — second moment (`v`) | fp32 | 4 | `4N` |
| Master weights (fp32 copy, mixed-precision training) | fp32 | 4 | `4N` |
| **Total** | | | **≈ 16 bytes/param** |

This is the standard mixed-precision-training accounting (Micikevicius et al., 2017; popularized in
this exact form by the QLoRA paper, Dettmers et al. 2023, Table 1). For a 7B-parameter model that is
`7 × 10^9 × 16 bytes ≈ 112 GB` just for weights/gradients/optimizer state — before a single
activation is stored for backprop, and before you've picked a batch size. A single 80 GB A100 cannot
hold that. A 70B model needs roughly 1.1 TB, requiring a multi-node cluster just to fine-tune, let
alone serve.

The two extra copies of the *entire model* that Adam requires (`m` and `v`, one scalar of running
gradient-moment statistics per parameter) are the crux of the problem: optimizer state, not the
weights themselves, is usually the single largest consumer of memory in full fine-tuning. Every
parameter you decide to update costs you not one number but (in the fp32-master-weights recipe) up
to eight numbers of bookkeeping.

**PEFT's answer**: freeze the entire pretrained model (`W_0`, no gradient, no optimizer state at
all — those parameters are just read during the forward pass) and introduce a small number of new,
trainable parameters that are added on top. If the new parameters are, say, 0.1–1% of `N`, then
gradients + optimizer state for them are 0.1–1% of the full-fine-tuning cost, and the frozen base
weights are held once at inference precision (or lower, per §5) with no gradient/optimizer overhead
at all. LoRA is the specific, empirically dominant way of shaping those new parameters.

---

## 2. LoRA's core insight and math

### 2.1 The hypothesis: low intrinsic rank

LoRA (Hu et al., 2021, *"LoRA: Low-Rank Adaptation of Large Language Models"*) starts from an
empirical observation from prior work on intrinsic dimensionality (Aghajanyan et al., 2020): even
though a pretrained model has billions of parameters, the *update* it needs during fine-tuning to
adapt to a new task tends to live in a much lower-dimensional subspace than the full parameter space
would suggest. Concretely: if `W_0 ∈ R^(d×k)` is a frozen pretrained weight matrix (e.g. one of the
attention projection matrices), full fine-tuning would learn a dense update `ΔW ∈ R^(d×k)` with the
same shape and — implicitly — the same rank budget as `W_0` itself (`rank(ΔW)` can be as large as
`min(d,k)`). LoRA's hypothesis is that the *useful* part of `ΔW` — the part that actually captures
the task-specific adaptation — has a much smaller intrinsic rank `r ≪ min(d,k)`.

If that hypothesis holds, you don't need to parameterize `ΔW` as a full `d × k` matrix at all. You
can instead write it as a product of two much smaller matrices:

```
ΔW = B·A,      B ∈ R^(d×r),   A ∈ R^(r×k),   r ≪ min(d, k)
```

`B·A` is rank ≤ `r` by construction (rank of a product is bounded by the rank of its smallest
factor), so this is exactly a low-rank factorization of the update. The number of trainable
parameters drops from `d·k` to `r·(d+k)`. For a typical attention projection with `d = k = 4096`
and `r = 8`: full-rank would be `16,777,216` parameters; the LoRA factorization is `8 × 8192 =
65,536` — a **256× reduction** for that single matrix.

### 2.2 The forward pass

During fine-tuning, the frozen path and the low-rank path run **in parallel** and their outputs are
summed. For an input activation `x` at the layer where `W_0` normally computes `h = W_0 x`, LoRA
instead computes:

```
h = W_0·x + (α / r)·B·A·x
```

where:
- `W_0 ∈ R^(d×k)` is the original pretrained weight, **frozen** — no gradient ever flows into it,
  it is not touched by the optimizer, and its memory footprint at inference precision is the only
  cost it contributes.
- `A ∈ R^(r×k)` and `B ∈ R^(d×r)` are the new trainable matrices — this is where 100% of the
  gradient signal from fine-tuning goes.
- `α` (`lora_alpha`) is a fixed scalar hyperparameter; `α / r` is the **LoRA scaling factor**
  applied to the low-rank path's contribution before it's added to the frozen path's output.

Because `A` and `B` are applied strictly additively next to `W_0` (never multiplied into it during
training), the frozen matrix does not need a gradient buffer, and the optimizer's `m`/`v` state only
ever has to cover `A` and `B`'s `r·(d+k)` parameters, not `W_0`'s `d·k`.

### 2.3 Initialization: training starts as a no-op

LoRA's initialization is chosen deliberately so that the adapted model is numerically *identical* to
the unmodified base model at step 0 of training:

- `A` is initialized with small random values, typically Kaiming/Gaussian initialization (this
  matches how a normal linear layer would be initialized, since `A` is the matrix that first
  "sees" the input).
- `B` is initialized to **all zeros**.

Since `ΔW = B·A` and `B = 0`, we get `ΔW = 0` at initialization regardless of what `A` contains, so
`h = W_0·x + (α/r)·(0)·A·x = W_0·x` exactly. This matters for two reasons: (1) training is stable
from the very first gradient step because the model starts at a known-good point (the pretrained
model's own behavior) rather than at a random, potentially catastrophic perturbation of it; and
(2) it guarantees that if training is aborted after zero useful steps, merging the adapter back
in changes nothing.

### 2.4 Why the `α/r` scaling exists

`α/r` decouples "how large is `r`" from "how strongly does the adapter's contribution get weighted."
Without this scaling, the effective magnitude of `B·A·x` would shrink as `r` grows (more terms
being summed inside the matrix product changes the output's variance in a way that interacts with
initialization), which would silently force you to retune the learning rate every time you change
`r`. Dividing by `r` and multiplying by a separate, freely-chosen `α` means you can change `r`
(a purely capacity-related choice) largely independently of the effective step size the adapter
sees. In practice, a common convention (used in both examples in this repo — see §8) is to set
`α = 2r`, giving a fixed scaling factor of `2` regardless of the specific rank chosen.

### 2.5 Which weight is actually being decomposed

LoRA is applied per weight matrix, not once for the whole model. In a transformer block the natural
candidates are the four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and the three
MLP/feed-forward projections in a gated-MLP architecture like Llama/Qwen (`gate_proj`, `up_proj`,
`down_proj`). Each targeted matrix gets its **own independent** `A`/`B` pair — a 32-layer model with
LoRA on all 7 of the above matrices has `32 × 7 = 224` separate `(A, B)` pairs, all trained jointly
by ordinary backprop through the frozen network.

---

## 3. Every LoRA hyperparameter, in detail

These are the fields of `peft.LoraConfig`, the class both implementations in this repo instantiate
directly (`finetune_local_slm.py:384-388`,
`quickstart_lora_example/train.py:143-164`).

### `r` (rank)

The rank of the low-rank factorization — the shared inner dimension of `A` and `B`. This is the
single most important capacity knob: it directly sets the number of trainable parameters
(`r·(d+k)` per targeted matrix) and, per the intrinsic-rank hypothesis in §2.1, the ceiling on how
expressive the learned update can be.

- **Typical values**: 4, 8, 16, 32, 64. The original LoRA paper found diminishing (and sometimes
  negative, due to overfitting on small datasets) returns well before `r = 64` on the tasks they
  tested.
- **Tradeoff**: larger `r` → more trainable parameters → more capacity to fit a complex or
  large fine-tuning dataset, but more memory/compute for the adapter's own gradients and optimizer
  state, and higher risk of overfitting on a small dataset. Smaller `r` → cheaper, more
  regularized, but may underfit if the target task genuinely needs a higher-rank update (e.g.
  learning a substantially different output format or a new language).
- This repo uses `r = 8` by default in the nightly pipeline (`finetune_local_slm.py:311`,
  `--lora-r` CLI flag, default `8`) and `r = 16` in the quickstart example
  (`quickstart_lora_example/train.py:145`) — consistent with `r=8` being a conservative choice for
  a small, high-signal nightly dataset (reward-filtered production logs, capped at 500 examples per
  `--max-examples`, `finetune_local_slm.py:304`) versus `r=16` being reasonable for a
  from-scratch teaching example with more headroom to overfit a tiny toy dataset intentionally.

### `lora_alpha` (scaling)

The numerator of the `α/r` scaling factor from §2.4. Controls how strongly the adapter's output is
weighted relative to the frozen base path, independent of `r`.

- **Typical convention**: set `lora_alpha = 2 × r`, so the scaling factor `α/r` is always `2`
  regardless of chosen rank. Both examples in this repo follow this convention exactly:
  `finetune_local_slm.py:312` sets `--lora-alpha` default `16` against `--lora-r` default `8`
  (`16/8 = 2`); `quickstart_lora_example/train.py:147` sets `lora_alpha=32` against `r=16`
  (`32/16 = 2`).
- **Tradeoff**: raising `α` relative to `r` (i.e., breaking the `2×` convention) increases the
  effective learning signal reaching the adapter without changing `r` itself — closer in spirit to
  raising the learning rate specifically for the adapted layers. Too high and training destabilizes
  or the adapter overwhelms the base model's pretrained behavior; too low and the adapter
  under-influences the output no matter how long you train.

### `lora_dropout`

Standard dropout applied to the input of the LoRA path (i.e., to `x` before it hits `A`), active
only during training. It regularizes the adapter specifically, which is useful because the adapter
is often trained on a much smaller dataset than the base model ever saw, so it is more prone to
overfitting.

- **Typical values**: 0.0 (no dropout, common) to 0.1. Both implementations in this repo use
  `0.05` (`finetune_local_slm.py:387`, `quickstart_lora_example/train.py:149`) — a mild
  regularizer appropriate for small fine-tuning sets.

### `target_modules`

The list of linear-layer names (by their attribute name in the model's module tree) that get a
LoRA adapter attached. This is architecture-specific — the names must match how the base model's
`transformers` implementation names its projection layers.

- **Attention-only**: `q_proj, k_proj, v_proj, o_proj` — the four linear projections inside
  self-attention (query, key, value, output). This is the minimal, most common starting point from
  the original LoRA paper, which found attention weights sufficient for many tasks while leaving
  the MLP frozen.
- **Attention + MLP**: additionally targeting `gate_proj, up_proj, down_proj` (the three linear
  layers of a SwiGLU-style gated feed-forward block, as used in Llama/Qwen architectures) gives the
  adapter influence over the feed-forward computation too, at the cost of roughly 1.75× more
  trainable parameters than attention-only (7 matrices vs. 4). Later PEFT literature and practice
  (e.g. the QLoRA paper's own ablations) generally found targeting *all* linear layers to be
  advantageous for closing the gap to full fine-tuning quality, at modest extra cost since the
  per-matrix cost is still `r·(d+k)`, tiny relative to `d·k`.
- Both implementations in this repo target the full seven-matrix set:
  `finetune_local_slm.py:313-316` (`--target-modules` default
  `"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"`, documented there as "Qwen2's
  attention+MLP projection names by default") and `quickstart_lora_example/train.py:151-159`
  (same seven names, hardcoded).
- **Not typically targeted**: the embedding matrix and the final LM head — these are enormous
  (`vocab_size × hidden_size`, often hundreds of millions of parameters) and adapting them
  meaningfully would defeat the "small number of new parameters" goal; PEFT supports it (as
  `modules_to_save` or a dedicated embedding-LoRA variant) but it's an advanced, situational choice
  not used in either implementation here.

### `bias`

Controls whether the (much smaller) bias terms of the targeted linear layers are also unfrozen and
trained.

- `"none"` — no bias parameters are trained (the strict LoRA formulation: only `A`/`B` are
  trainable). This is what both implementations in this repo use
  (`finetune_local_slm.py:387`, `quickstart_lora_example/train.py:161`).
- `"all"` — every bias in the model becomes trainable.
- `"lora_only"` — only the biases belonging to the layers that also got an `A`/`B` adapter become
  trainable.
- **Tradeoff**: `"none"` is the cheapest and most common default; many modern base models (Llama,
  Qwen) don't even have bias terms on their linear layers to begin with, making the distinction
  moot for them.

### `task_type`

Tells PEFT which model-wrapping and forward-pass conventions to use (e.g. how to handle labels,
what the model's `forward()` signature looks like) — it's a dispatch string, not a mathematical
hyperparameter. `"CAUSAL_LM"` (used in both files here:
`finetune_local_slm.py:387`, `quickstart_lora_example/train.py:163`) is correct for
decoder-only autoregressive language models like Qwen; other values exist for sequence
classification, seq2seq, token classification, etc.

---

## 4. Merging vs. keeping adapters separate

Once training is done, you have `W_0` (frozen, untouched) and a trained `(A, B)` pair per targeted
matrix. There are two ways to use them at inference time.

### 4.1 Merging

Because the forward pass is a pure sum, `h = W_0 x + (α/r) B A x = (W_0 + (α/r) B A) x`, you can
precompute:

```
W_merged = W_0 + (α/r)·B·A
```

once, offline, and then simply replace `W_0` with `W_matched` in the model. From that point on,
inference is **identical in shape, latency, and memory** to the original unadapted base model — the
`A`/`B` matrices are gone, absorbed into a single dense matrix of the original size. There is zero
extra inference-time compute, zero extra parameters to load, and no risk of adapter-loading bugs
in the serving path. This is the right choice when you have one fine-tuned "personality" you always
want a given deployment to have.

The tradeoff: merging is a one-way operation from a serving-simplicity perspective — you get back a
single dense checkpoint the same size as the base model (LoRA's whole storage-efficiency benefit
is lost once merged; the win was during *training*, not in the final artifact), and if you want to
switch to a different fine-tune, or turn the adaptation off, you need a different full checkpoint
or the original unmerged base weights kept around separately.

### 4.2 Keeping adapters separate (multi-adapter serving)

Alternatively, you can ship the base model once and keep many different `(A, B)` adapter checkpoints
around — each one is tiny (megabytes, not gigabytes) since it's only `r·(d+k)` parameters per
targeted matrix. At request time, the serving stack picks which adapter(s) to apply on top of the
shared frozen base, e.g. a different adapter per tenant, per task, or per user preference, without
ever needing multiple copies of the multi-gigabyte base weights in memory. Frameworks like
`peft.PeftModel` support hot-swapping the active adapter, and serving systems like vLLM/S-LoRA
support batching *different* adapters' requests together against one shared base-model forward
pass. The tradeoff is a small amount of extra inference-time compute (the `B·A·x` term must actually
be computed at request time rather than folded in ahead of time) and added system complexity
(adapter registry, swap logic, cache invalidation) in exchange for far greater deployment
flexibility and far smaller storage/transfer cost per fine-tune.

`finetune_local_slm.py` uses the merge path: after training it calls
`model.merge_and_unload()` (`finetune_local_slm.py:416`) to fold the adapter into the base
weights and then saves the result as a plain HF model directory
(`finetune_local_slm.py:422-425`) — appropriate since the nightly job produces exactly one
updated model to hand off to the next stage of the pipeline (GGUF conversion and Ollama
registration), not a multi-tenant adapter-swapping deployment. `quickstart_lora_example/inference.py`
demonstrates the *other* path deliberately, for teaching purposes: it loads the base model and
wraps it with `PeftModel.from_pretrained(model, ADAPTER)`
(`quickstart_lora_example/inference.py:34-52`) **without merging**, so the adapter stays a
separate, swappable artifact — its own docstring notes this is "no merge — just to see the
quickstart result answer before deciding whether it's worth carrying further into the real
pipeline" (`quickstart_lora_example/inference.py:1-5`).

---

## 5. QLoRA: fitting a 7B–70B model on one GPU

QLoRA (Dettmers, Pagnoni, Holtzman & Zettlemoyer, 2023, *"QLoRA: Efficient Finetuning of Quantized
LLMs"*) is not a different adapter math from §2 — the `h = W_0 x + (α/r) B A x` formula is
unchanged. What QLoRA changes is the **precision and memory representation of the frozen base
model `W_0`**, combined with two supporting memory-management tricks, so that the (dominant, since
`W_0` is frozen and 100–1000× larger than the adapter) memory cost of just *holding* the base model
drops enough to fit a 7B–70B model's base weights on a single 24–48 GB GPU while still training a
full-precision-quality adapter on top of it.

### 5.1 4-bit NormalFloat (NF4)

The base model's weights are quantized from their native fp16/bf16 (16 bits/parameter) down to a
custom 4-bit format, **NF4 (4-bit NormalFloat)**, roughly a 4× reduction in the base model's static
memory footprint (a 7B model drops from ~14 GB in fp16 to ~3.5–4 GB in 4-bit).

Why not plain INT4? A standard integer quantization scheme places quantization levels (the discrete
values a weight can be rounded to) *uniformly* across the range of values being represented — e.g.
16 evenly spaced buckets between the minimum and maximum weight value in a block. But pretrained
neural network weights are empirically well approximated as zero-centered Gaussian/normal
distributions, not uniform ones: most weight values cluster tightly around zero, with a long,
sparsely-populated tail. Uniform INT4 buckets waste most of their 16 representable levels on the
sparsely populated tail region and under-resolve the densely populated region near zero, where most
of the actual weight mass — and therefore most of the quantization error that matters for output
accuracy — lives.

NF4 instead places its 16 quantization levels at the **quantiles of a standard normal distribution**
`N(0, 1)` — i.e., the levels are chosen so that an equal *probability mass* of a normally-distributed
source falls into each bucket, not equal *range width*. Concretely: each block of weights is first
normalized to have (empirically) unit-ish variance, mapped through the standard normal's quantile
function at 16 evenly-spaced probability points to get the fixed NF4 codebook of 16 representable
values, and every weight is rounded to whichever of those 16 fixed values is closest after
normalization by the block's scale. Because the codebook is derived to be information-theoretically
optimal for values that really are normally distributed, and pretrained weights really are close to
normally distributed, NF4 achieves lower quantization error than INT4 at the identical bit width — this
is the core empirical result the QLoRA paper reports as making 4-bit finetuning viable without a
significant quality regression versus a 16-bit base model. This is what `bnb_4bit_quant_type="nf4"`
selects in both `BitsAndBytesConfig` instantiations in this repo
(`finetune_local_slm.py:287`, `quickstart_lora_example/train.py:65`).

Critically, quantization is applied **only to the frozen base weights** `W_0`. The LoRA adapter
matrices `A` and `B` are kept and trained in a normal higher-precision floating-point format (bf16
in `finetune_local_slm.py:288`'s `bnb_4bit_compute_dtype=torch.bfloat16`; fp16 in
`quickstart_lora_example/train.py:66`'s `bnb_4bit_compute_dtype=torch.float16`). At the moment
a quantized weight actually participates in a matmul, it is dequantized on-the-fly to that compute
dtype (a small, transient full-precision buffer, not a persistent one), the matmul runs at that
precision, and the result is used and discarded — so `W_0` stays 4-bit **at rest** while all real
arithmetic (and 100% of the gradient computation) happens in full compute precision. This is why
QLoRA does not simply inherit low-precision-training's usual accuracy problems: gradients never
have to flow through, or be computed in, 4-bit arithmetic at all, because `W_0` never receives a
gradient in the first place (it's frozen) — only `A` and `B`, which are never quantized, do.

### 5.2 Double quantization

Quantization is done per-block (typically blocks of 64 contiguous weight values sharing one scale
constant), because a single scale factor for an entire multi-million-parameter tensor would be far
too coarse to give each region of the tensor an appropriately sized quantization step. But that
means you need to store one extra floating-point scale (quantization constant) per block, and with
small blocks this adds up: at a block size of 64 and one fp32 scale per block, the *overhead* of
storing the scale constants themselves is `32 bits / 64 values ≈ 0.5 bits per parameter`, which is
non-trivial next to a 4-bit-per-parameter base.

**Double quantization** applies the same idea recursively: quantize the (many, small) first-level
scale constants themselves down to a lower precision (8-bit, in a further block-wise quantization
scheme), storing a much smaller number of higher-precision second-level constants to dequantize
them. This shaves roughly another `0.37 bits per parameter` off the effective footprint on top of
the base 4-bit weights (per the QLoRA paper's own accounting) — a genuinely second-order effect but
one that matters at the scale of billions of parameters. It's toggled by
`bnb_4bit_use_double_quant=True`, set in both quantization configs in this repo
(`finetune_local_slm.py:288`, `quickstart_lora_example/train.py:67`).

### 5.3 Paged optimizers

Even with the base model shrunk to 4-bit and only the adapter's small optimizer state to track,
training can still hit sudden memory spikes — most commonly from **gradient checkpointing**
(recomputing, rather than storing, intermediate activations during the backward pass to save
memory overall, at the cost of occasional bursts of extra activation memory during the recompute).
On a GPU that's already sized tightly around the quantized base model plus the adapter, one of
these transient spikes is enough to trigger an out-of-memory crash, terminating an otherwise
successful long training run.

QLoRA addresses this by using NVIDIA's **unified memory** feature to implement a *paged optimizer*:
the optimizer state (which, being fp32 `m`/`v` buffers, is not touched by every single forward pass
the way weights are) is allowed to be transparently paged out from GPU memory to CPU RAM and paged
back in on demand, the same way an operating system pages virtual memory out to disk under pressure
— automatic, transparent to the training loop, and only invoked when an actual spike would otherwise
overflow GPU memory, so it doesn't meaningfully slow down the common case where no spike occurs. In
practice this is exposed as `bitsandbytes`' `paged_adamw_8bit` optimizer (paged optimizer state, and
additionally 8-bit rather than fp32 optimizer state, for a further memory reduction), which
`quickstart_lora_example/train.py:191` selects directly via `optim="paged_adamw_8bit"` in its
`TrainingArguments`.

### 5.4 Putting it together: why this fits on one GPU

Stacking the three pieces: NF4 shrinks the dominant cost (the frozen base weights) roughly 4× versus
fp16; double quantization shaves a further fraction of a bit per parameter off that; and paged
optimizers remove the risk that a transient activation/gradient-checkpointing spike, not the steady-
state footprint, is what actually causes an out-of-memory crash. Combined with LoRA itself (§1–2)
already having eliminated the need for *any* gradient or optimizer-state memory for the base model's
`d·k`-sized weights (only the adapter's much smaller `r·(d+k)` parameters need that), the net result
is that a 7B model — which needs on the order of 112 GB for full fp16 fine-tuning per the §1 table —
can be fine-tuned with well under 10 GB of GPU memory for the base model's static footprint plus a
small, adapter-sized amount of trainable-parameter/gradient/optimizer memory. This is precisely what
makes QLoRA fine-tuning of 7B–13B models practical on a single consumer GPU (e.g. a 24 GB RTX 3090/
4090) and 33B–70B models practical on a single prosumer/datacenter GPU (e.g. a single 48 GB A6000 or
80 GB A100), where full fine-tuning of those same models would require many GPUs or an entire node.

---

## 6. Standard fine-tuning hyperparameters that still apply

LoRA/QLoRA change *what* gets trained and how many parameters that involves, but the training loop
around it is still ordinary supervised fine-tuning with the usual optimization hyperparameters
layered on top.

- **Learning rate**: because only a small number of newly-initialized parameters (`A`/`B`) are
  being trained, and they start at a true no-op (§2.3) rather than at a pretrained optimum, LoRA
  fine-tuning tolerates — and generally needs — a noticeably **higher** learning rate than full
  fine-tuning of the same model would use. Full fine-tuning of an LLM typically uses learning rates
  in the `1e-5`–`5e-5` range; LoRA fine-tuning commonly uses `1e-4`–`2e-4`, often 5–10× higher,
  because the adapter has to learn its entire contribution from scratch and there is no large
  pretrained-weight momentum to disturb if a step is a little too aggressive. Both implementations
  in this repo sit exactly in that band: `finetune_local_slm.py:400` sets
  `learning_rate=1e-4`; `quickstart_lora_example/train.py:181` sets `learning_rate=2e-4`.
- **Batch size / gradient accumulation**: the *effective* batch size (`per_device_train_batch_size
  × gradient_accumulation_steps × num_devices`) still governs gradient-estimate noise and training
  stability the same way it does in any other training run; gradient accumulation is the standard
  way to reach a reasonable effective batch size when per-step GPU/CPU memory only allows a tiny
  physical batch. `finetune_local_slm.py:396-397` uses
  `per_device_train_batch_size=1` with `gradient_accumulation_steps=4` (effective batch 4) — a
  CPU-friendly, memory-conservative choice consistent with that script's default full-precision
  (non-QLoRA) path being designed to run without a GPU at all. `quickstart_lora_example/train.py:177-179`
  uses `per_device_train_batch_size=2` with `gradient_accumulation_steps=8` (effective batch 16).
- **Epochs / max steps**: how many full passes over the training set (or, equivalently, a hard step
  count) to run. On a small, high-signal dataset, very few epochs are needed before the adapter
  starts overfitting given its regardless-small parameter count already limits overfitting somewhat.
  `quickstart_lora_example/train.py:175` sets `num_train_epochs=3` for its tiny hand-written
  dataset. `finetune_local_slm.py:398-399` sets `num_train_epochs=1` but also caps training at
  `max_steps=args.max_steps` (default `200`, `finetune_local_slm.py:308`) and, notably, at a hard
  **wall-clock** deadline via the custom `WallClockStop` callback
  (`finetune_local_slm.py:211-222`, `--max-minutes` default `120.0` at
  `finetune_local_slm.py:306-307`) — the module docstring explains this is because "CPU training
  speed varies enough by machine that a step count doesn't reliably bound wall-clock time on an
  unattended nightly job" (`finetune_local_slm.py:40-43`), so step/epoch counts alone are not
  trusted as the real safety bound in that pipeline.
- **Warmup ratio**: gradually ramping the learning rate up from zero (or a small value) over the
  first fraction of training steps, rather than applying the full learning rate from step one, is a
  standard stabilization technique — it's just as applicable to a LoRA adapter's own optimizer as
  it is to full fine-tuning, since the adapter's weights, however few, are still subject to the
  same "large steps at high LR early in training can be destabilizing" dynamics. Neither script in
  this repo explicitly sets a non-default warmup ratio (both rely on `TrainingArguments`'s default),
  which is worth flagging as a real gap to tune if training instability (loss spikes early in a run)
  is ever observed on the actual accumulated production dataset.

---

## 7. Block diagram: the parallel low-rank path

```
                              input activation  x  (dim k)
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
                 ▼                     ▼                     │
   ┌─────────────────────────┐   ┌───────────┐               │
   │                         │   │           │               │
   │                         │   │     A     │  A ∈ R^(r×k)  │
   │      W_0  (frozen)      │   │  (r × k)  │  random init  │
   │      d × k              │   │           │               │
   │  pretrained weights,    │   └─────┬─────┘               │
   │  NO gradient, NO        │         │  A·x  → dim r        │
   │  optimizer state.       │         ▼                      │
   │  Under QLoRA: stored    │   ┌───────────┐               │
   │  as 4-bit NF4, de-      │   │           │               │
   │  quantized on the fly   │   │     B     │  B ∈ R^(d×r)  │
   │  for each matmul.       │   │  (d × r)  │  zero init    │
   │                         │   │           │               │
   │                         │   └─────┬─────┘               │
   └────────────┬────────────┘         │  B·A·x  → dim d      │
                 │                     │  scaled by (α/r)      │
                 │  W_0·x → dim d      │                      │
                 ▼                     ▼                      │
                 └──────────►  (+)  ◄──────────────────────────
                              │
                              ▼
                 h = W_0·x + (α/r)·B·A·x     (dim d)
```

The wide block (`W_0`) is frozen and, under QLoRA, lives in GPU memory at 4-bit precision. The two
thin blocks (`A`, `B`) are the entire trainable surface — together they hold `r·(d+k)` parameters
versus `W_0`'s `d·k`, and their product `B·A` is what gets added to the frozen path's output before
the result continues into the rest of the transformer block. After training, this same diagram is
either (a) collapsed into a single `d×k` block `W_merged = W_0 + (α/r)BA` for zero-overhead
inference (§4.1), or (b) kept exactly as drawn, with different `(A, B)` pairs swapped in per
request for multi-adapter serving (§4.2).

---

## 8. How this repo uses it

Two real, standalone scripts in `orchestrator_framework_layer/akanade/backend/scripts/` use PEFT.
Both are deliberately isolated from the always-running chatbot backend — neither is imported by the
live serving app, because `torch`/`transformers`/`peft`/`bitsandbytes` are heavy, often GPU-specific
dependencies with no business inside an always-on serving image
(`finetune_local_slm.py:1-7`, `quickstart_lora_example/train.py:1-11`).

### 8.1 `finetune_local_slm.py` — the nightly production pipeline

This is stage 2 of a larger pipeline (stage 1 is `agent/finetune_log.py`'s collection of real
production `(messages, response, reward)` turns, curated by `data_engine.py`), and its own
docstring states it is intended to be invoked by a nightly cron job via
`deploy/nightly_finetune.sh`, run in its own container/venv, decoupled entirely from the live
request-serving app (`finetune_local_slm.py:1-58`).

**LoRA configuration** — constructed at `finetune_local_slm.py:384-388`:

```python
lora_config = LoraConfig(
    r=args.lora_r, lora_alpha=args.lora_alpha,
    target_modules=args.target_modules.split(","),
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

with CLI-configurable defaults `--lora-r 8` (`finetune_local_slm.py:311`), `--lora-alpha 16`
(`finetune_local_slm.py:312`, giving the conventional `α/r = 2` scaling from §2.4), and
`--target-modules` defaulting to all seven Qwen2 attention+MLP projections
(`finetune_local_slm.py:313-316`). These three values (`lora_r`, `lora_alpha`, and separately
`min_reward`, the reward-filtering threshold on training examples) are also the exact three
hyperparameters an optional bandit-based auto-tuner (`peft_hyperparam_optimizer.py`) can override
per night via `--auto-hyperparams`, selecting them based on past regression-gate outcomes rather
than the fixed CLI defaults (`finetune_local_slm.py:334-361`) — off by default until the
feedback loop through `regression_gate.py --update-peft-bandit` is wired up, per the script's own
comment (`finetune_local_slm.py:338-340`).

**QLoRA path** — `load_base_model()` (`finetune_local_slm.py:261-292`) defaults to a
full-precision (`torch.float32`) CPU-friendly load, and only builds the 4-bit QLoRA path when
`--qlora` is passed:

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=quant_config, device_map="auto")
return prepare_model_for_kbit_training(model)
```

(`finetune_local_slm.py:286-292`). This is NF4 quantization (§5.1) with double quantization
enabled (§5.2) and a bf16 compute dtype for dequantized matmuls and for the LoRA adapter itself.
`prepare_model_for_kbit_training` (imported from `peft` at `finetune_local_slm.py:284`) is PEFT's
helper that does the bookkeeping a k-bit-quantized base model needs before LoRA training can proceed
correctly — among other things, casting select components (like layer norms) to higher precision for
numerical stability and enabling gradient computation on the (otherwise-frozen, quantized) model's
input embeddings so gradients can actually flow into the adapter. The script refuses cleanly, rather
than crashing deep inside `bitsandbytes`, if `--qlora` is requested without a CUDA GPU present
(`finetune_local_slm.py:271-275`) — 4-bit kernels are CUDA-only — and the module docstring flags
this QLoRA path as new and specifically not yet validated end-to-end on the real CPU-primary deploy
box, "since it needs a CUDA box this CPU-only pipeline was never validated against"
(`finetune_local_slm.py:68-70`).

**`RewardWeightedTrainer` — a custom loss on top of LoRA, not part of LoRA itself**
(`finetune_local_slm.py:190-208`): this subclasses HuggingFace's `Trainer` and overrides
`compute_loss` to implement reward-weighted supervised fine-tuning — described in the module
docstring as "a well-established lightweight stand-in for full online RL (PPO-style rollouts), which
is not remotely practical on a CPU-only box" (`finetune_local_slm.py:33-35`). Concretely, per
`finetune_local_slm.py:197-208`: it computes an ordinary token-level cross-entropy loss with
`ignore_index=-100` (masking out the prompt/system-prompt tokens so loss is only ever computed on
the assistant's actual answer, per the loss-masking in `build_dataset()` at
`finetune_local_slm.py:137-170`), averages that per-example over the unmasked (assistant-turn)
tokens only, and then computes a **reward-weighted mean across the batch**:

```python
weighted = (per_example_loss * rewards).sum() / rewards.sum().clamp(min=1e-6)
```

(`finetune_local_slm.py:207`), where `rewards` is each example's `reward * training_weight`
product computed back in `build_dataset()` (`finetune_local_slm.py:168`, documented at
`finetune_local_slm.py:141-152` as keeping "was this answer correct" (`reward`) and "how much
should a correct example of this kind count" (`training_weight`, a separate knob set by
`agent/finetune_log.py`'s `log_ground_truth_turn()`) as two distinct, separately multiplied fields
rather than conflating them at the logging layer). The practical effect: a logged production turn
that scored a reward of `1.0` contributes proportionally more to the gradient than one that scored
`0.7`, rather than both contributing identically as in plain SFT — this reward-weighting is
completely orthogonal to LoRA itself (it would apply the same way under full fine-tuning) but is the
one deliberate departure from vanilla supervised fine-tuning in this pipeline.

After training, the script calls `model.merge_and_unload()` (`finetune_local_slm.py:416`) to fold
the trained adapter into the base weights (§4.1's merge path — appropriate here since the goal is
one updated model handed to the next pipeline stage, not multi-tenant adapter serving), optionally
applies real magnitude pruning to the merged linear layers (`finetune_local_slm.py:225-258`,
`417-421`, off by default at `--prune-sparsity 0.0`), and saves the plain merged HF model directory
(`finetune_local_slm.py:422-425`) for `convert_and_register_ollama_model.sh` (GGUF conversion +
registering with the live chatbot) to pick up as its next stage — the module docstring is explicit
that this handoff, not this script itself, is what eventually reaches production, and that the
nightly fine-tune "is deliberately a STANDALONE script... never imported by the live chatbot app"
(`finetune_local_slm.py:3-7`).

### 8.2 `quickstart_lora_example/` — a minimal, from-scratch teaching example

`train.py` and `inference.py` are explicitly *not* the production pipeline — the module docstring
states plainly that this "is NOT the production nightly pipeline... This script trains on the tiny
hand-written `data/train.jsonl` below and never touches Ollama or the live chatbot — it's for
learning the shape of the pipeline and iterating on a data format fast"
(`quickstart_lora_example/train.py:12-23`). It is a useful second, simpler worked example of the
exact same LoRA/QLoRA mechanics:

- Loads `Qwen/Qwen3-0.6B` (`quickstart_lora_example/train.py:39`) in 4-bit via
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`
  (`quickstart_lora_example/train.py:63-68`) — the same NF4 + double-quantization combination as
  `finetune_local_slm.py`'s QLoRA path, but unconditionally (this script assumes a GPU is
  available, unlike the CPU-first default in the nightly pipeline) and with an fp16 rather than
  bf16 compute dtype.
- Builds a `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=[...7 projections...],
  bias="none", task_type="CAUSAL_LM")` (`quickstart_lora_example/train.py:143-164`) — same
  `α/r = 2` convention, same dropout, same bias/task_type choices as the nightly pipeline, just a
  larger rank appropriate for a from-scratch toy dataset.
- Uses TRL's `SFTTrainer` (`quickstart_lora_example/train.py:202-214`) with `peft_config=lora_config`
  passed directly to the trainer — TRL applies `get_peft_model` internally, whereas
  `finetune_local_slm.py` calls `get_peft_model` explicitly itself
  (`finetune_local_slm.py:389`) and uses a plain `transformers.Trainer` subclass (needed precisely
  because it requires the custom reward-weighted `compute_loss` that `SFTTrainer` doesn't provide
  out of the box).
- Uses `optim="paged_adamw_8bit"` (`quickstart_lora_example/train.py:191`) — the paged-optimizer
  mechanism from §5.3, not used in `finetune_local_slm.py`, whose default CPU path has no GPU
  memory-paging concern to address in the first place.
- `inference.py` loads the base model at fp16 (`quickstart_lora_example/inference.py:34-42`) and
  attaches the trained adapter unmerged via `PeftModel.from_pretrained(model, ADAPTER)`
  (`quickstart_lora_example/inference.py:49-52`), the multi-adapter-style separate-adapter path
  from §4.2, explicitly so a user can inspect the fine-tuned answer quality before deciding whether
  to invest in merging and carrying it into the real pipeline
  (`quickstart_lora_example/inference.py:1-5`, `quickstart_lora_example/train.py:19-23`).

### 8.3 The common thread

Both scripts land on essentially the same LoRA hyperparameter conventions (`α/r = 2`, `dropout =
0.05`, `bias = "none"`, the same seven target modules, `task_type = "CAUSAL_LM"`) and the same NF4 +
double-quantization QLoRA recipe when quantization is used at all — strong evidence these are
treated as this codebase's settled defaults for this base-model family (Qwen), with only rank/alpha
magnitude and the choice of trainer/loss varying between "minimal teaching example" and "reward-
aware nightly production pipeline that is validated against real accumulated production data and
gated by a separate regression check before its output is ever registered with the live chatbot."

---

## 9. References

- Hu, E. J., Shen, Y., Wallis, P., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language
  Models.* arXiv:2106.09685.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning
  of Quantized LLMs.* arXiv:2305.14314.
- Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). *Intrinsic Dimensionality Explains the
  Effectiveness of Language Model Fine-Tuning.* arXiv:2012.13255.
- Micikevicius, P., et al. (2017). *Mixed Precision Training.* arXiv:1710.03740.
- In-repo: `orchestrator_framework_layer/akanade/backend/scripts/finetune_local_slm.py`,
  `orchestrator_framework_layer/akanade/backend/scripts/quickstart_lora_example/train.py`,
  `orchestrator_framework_layer/akanade/backend/scripts/quickstart_lora_example/inference.py`.
- See also: [15-quantization](../15-quantization/README.md) for INT8/INT4/GPTQ/AWQ/GGUF beyond
  NF4, and [16-memory-optimization-training](../16-memory-optimization-training/README.md) for
  gradient checkpointing and mixed precision in more depth; for what feeds and tunes this pipeline,
  see [23-data-pipeline-slm-finetuning](../23-data-pipeline-slm-finetuning/README.md) (where the
  training examples come from) and
  [22-peft-hyperparameter-optimization](../22-peft-hyperparameter-optimization/README.md) (how
  `lora_r`/`lora_alpha`/`min_reward` are picked automatically instead of fixed).
