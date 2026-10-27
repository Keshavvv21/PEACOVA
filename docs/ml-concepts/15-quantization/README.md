# LLM Quantization

## 1. What Quantization Is

A model trained in 16-bit or 32-bit floating point stores every weight as a wide, high-precision
number: FP32 uses 32 bits (1 sign, 8 exponent, 23 mantissa), FP16 uses 16 bits (1 sign, 5 exponent,
10 mantissa), and BF16 uses 16 bits with FP32's 8-bit exponent range but only 7 mantissa bits. These
formats give training the dynamic range and precision it needs for gradient updates to accumulate
correctly over millions of steps.

Inference does not need that much precision. A trained weight matrix is just a fixed set of numbers
being multiplied against activations — the question is not "can this format represent a gradient
update precisely" but "can this format represent *this particular value* closely enough that the
matrix multiply still produces approximately the same output." Quantization answers that: it
represents weights (and sometimes activations) using far fewer bits — typically INT8 (8 bits),
INT4 (4 bits), or specialized narrow floating-point formats like NF4 (4-bit NormalFloat) or FP8 —
in place of the FP32/FP16/BF16 used during training.

The payoff is threefold:

- **Memory footprint.** A 7B-parameter model stored in FP16 needs ~14 GB just for weights. The same
  model in INT4 needs ~3.5 GB — a 4x reduction. This is frequently what determines whether a model
  fits on a given GPU or consumer machine at all.
- **Memory bandwidth / throughput.** LLM inference (particularly single-sequence decoding) is
  usually memory-bandwidth-bound, not compute-bound: the GPU spends more time moving weights from
  HBM into registers than doing the actual multiply-accumulate. Halving or quartering the number of
  bytes that must move per token directly increases tokens/second.
- **Compute throughput (when activations are also quantized).** Hardware has dedicated fast paths
  for low-precision integer matrix multiplication (e.g., INT8 tensor cores). Exploiting these
  requires both operands of the matmul — weights *and* activations — to be in the low-precision
  format, not just the weights sitting in memory that way.

The cost is accuracy: mapping a continuous range of float values onto a small number of discrete
levels (256 for INT8, 16 for INT4) throws away information. The engineering problem of
quantization is entirely about *how* to throw that information away — which values to prioritize,
how finely to calibrate the mapping, and whether to let the model adapt to the loss — so that
downstream task performance degrades as little as possible for the bits saved.

## 2. The Core Math: Linear (Affine) Quantization

### 2.1 Quantize / dequantize formulas

The dominant scheme for LLM weight quantization is **affine (linear, uniform) quantization**. Given
a floating-point tensor with observed range `[min, max]`, and a target integer range
`[qmin, qmax]` (e.g., `[-128, 127]` for signed INT8, or `[0, 255]` for unsigned INT8), we compute a
**scale** and a **zero-point**:

```
scale      = (max - min) / (qmax - qmin)
zero_point = round(qmin - min / scale)
```

Quantization (float → int):

```
q = round(x / scale) + zero_point,   clamped to [qmin, qmax]
```

Dequantization (int → float, the approximation actually used at compute time):

```
x_approx = (q - zero_point) * scale
```

`scale` is a positive float that sets how much real-valued range one integer step covers.
`zero_point` is an integer offset that lets the representable range shift so that a real value of
0.0 maps to an *exact* integer (important, because tensors like padded activations or ReLU outputs
have exact zeros that must round-trip exactly — any zero-point error compounds every time that
value is reused downstream).

**Symmetric vs. asymmetric.** When a tensor's distribution is roughly symmetric around zero (which
is the common case for trained weight matrices), it's simpler to fix `zero_point = 0` and use a
symmetric range like `[-127, 127]`. This is "symmetric quantization" — it costs one bit of range
efficiency (127 vs. 128 possible negative codes) but removes an addition/subtraction from every
dequantization and avoids the zero-point's own rounding error. Asymmetric quantization
(`zero_point != 0`) is used when the distribution is skewed — activations after a ReLU/GELU, for
instance, are all non-negative, so an asymmetric `[0, 255]` mapping uses the full integer range
instead of wasting half of it on values that never occur.

### 2.2 Quantization error

The rounding step `round(x/scale)` introduces an error bounded by `±scale/2` in the *integer*
domain, i.e., `±scale²/2`... more precisely, the reconstruction error is:

```
error = x_approx - x,   |error| <= scale / 2
```

This means the **scale is the single number that controls quantization error** — halve the scale
(by using a narrower observed range, or more integer levels) and you halve the worst-case error.
This is exactly why granularity (next section) matters so much: it is a lever for shrinking `scale`
locally instead of globally.

### 2.3 Per-tensor vs. per-channel vs. per-group quantization

The naive approach computes a single `(scale, zero_point)` pair for an entire weight matrix
(**per-tensor quantization**). This is cheap — one scale/zero-point pair per tensor — but a weight
matrix in a real transformer layer has different columns/rows with very different magnitude
distributions. A single global scale is forced to cover the *widest* range in the whole tensor,
which means every other row/column is quantized with a scale far coarser than it actually needs,
wasting precision on the majority of values to accommodate a few outlier rows.

**Per-channel quantization** assigns a separate `(scale, zero_point)` to each output channel
(typically each row of the weight matrix, corresponding to one output neuron). Each channel's scale
is fit to that channel's own min/max, so a channel with small weight magnitudes gets a
correspondingly small scale and thus finer resolution, instead of being dragged by an outlier
channel elsewhere in the tensor.

**Per-group (block-wise) quantization** subdivides even further: instead of one scale per row, the
row is split into contiguous groups of `g` elements (commonly `g = 32, 64, or 128`), each with its
own scale and zero-point. This is the granularity GPTQ, AWQ, and GGUF's K-quants all use in
practice, because weight magnitude can vary meaningfully even *within* a single output channel of a
large LLM layer.

The general accuracy/overhead trade-off:

```
per-tensor  →  1 scale for the whole matrix     → cheapest metadata, worst accuracy
per-channel →  1 scale per row/column           → moderate metadata, better accuracy
per-group   →  1 scale per g-sized block of a row → more metadata, best accuracy at a given bit-width
```

The "overhead" is real: each extra scale factor is typically stored as an FP16 value, so finer
granularity means more bytes spent on metadata rather than the compressed weights themselves. A
group size of 128 with 4-bit weights adds roughly `16 bits / (128 × 4 bits) ≈ 3%` metadata overhead
— a good trade almost everywhere, which is why `g=128` (or 32/64 for extra accuracy) is the default
in nearly every modern 4-bit LLM quantizer.

### 2.4 Why weight distributions being bell-shaped motivates NF4

Trained neural network weights, empirically, are not uniformly distributed across their range —
they cluster densely around zero and thin out toward the extremes, closely resembling a **zero-
centered Gaussian (normal) distribution**. Standard integer quantization (INT4, INT8) uses
*uniformly spaced* levels: the gap between representable value `k` and `k+1` is the same constant
`scale` everywhere in the range. This is a poor match for a bell-shaped input distribution — it
allocates just as many representable levels to the sparsely-populated tails as it does to the
densely-populated region near zero, where nearly all of the actual mass (and therefore nearly all
of the *quantization error mass*) lives.

**NF4 (NormalFloat4)** — introduced with QLoRA — instead defines its 16 representable 4-bit levels
to be the **quantiles of a standard normal distribution** rather than uniformly spaced points. This
places more representable values in the dense center of the distribution and fewer in the tails,
so that a normally-distributed weight tensor incurs less average quantization error per bit than it
would under uniform INT4 for an information-theoretically principled reason: it is
information-theoretically optimal for normally-distributed data, in the sense of minimizing
expected quantization error for a fixed number of levels. In practice, weights are first normalized
into `[-1, 1]` (by dividing by the per-block absolute-max, i.e., a form of per-group scaling as in
§2.3) so the fixed quantile grid can be reused as a lookup table across every block, and each block
still carries its own scale to map back to the tensor's actual magnitude.

This library's detailed treatment of NF4 in the context of QLoRA fine-tuning — including the
double-quantization of the scale factors themselves — lives in the LoRA/QLoRA README at
[`docs/ml-concepts/08-lora-qlora/README.md`](../08-lora-qlora/README.md); this document covers NF4
only as one point on the general quantization-format spectrum.

## 3. Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)

### 3.1 PTQ

**Post-Training Quantization** takes a model that has already finished training in FP16/BF16/FP32
and converts its weights to a lower-precision format directly, with no further gradient-based
training. Most PTQ methods do use a small **calibration set** — a few hundred to a few thousand
unlabeled text samples run through the model in a forward pass — to *observe* activation and/or
weight statistics (ranges, Hessian sensitivity, outlier channels) so the quantization parameters
(`scale`, `zero_point`, which channels to protect) can be chosen well. Crucially, no weights are
updated by gradient descent; the calibration pass only informs how the existing weights get mapped
onto the low-bit grid.

- **Pros:** fast (minutes to a couple of hours for a 7B–70B model, dominated by the calibration
  forward passes and any per-layer reconstruction step), requires no labeled dataset or training
  infrastructure, and can be applied to any already-trained checkpoint.
- **Cons:** because the model was never exposed to quantization noise during training, it has no
  chance to compensate for it. Accuracy loss is generally small at 8-bit but grows sharply at 4-bit
  and below, particularly for smaller models with less redundancy to absorb the error (see §6).

GPTQ, AWQ, and the GGUF k-quant schemes described in §4 are all PTQ methods — this is why they can
be applied to any public pretrained checkpoint in a single offline pass rather than requiring access
to the original training pipeline.

### 3.2 QAT

**Quantization-Aware Training** inserts the quantize/dequantize round-trip (§2.1) directly into the
forward pass *during* training or fine-tuning, using "fake quantization": weights (and optionally
activations) are quantized to the target low-bit grid and immediately dequantized back to float
before the matmul, so the forward pass sees the same rounding error inference will see, while the
backward pass computes gradients as if the operation were identity (the **straight-through
estimator**, since `round()` has zero gradient almost everywhere). Over many training steps, the
optimizer nudges the *full-precision master weights* to values whose rounded/quantized counterparts
minimize the loss — effectively teaching the model to be robust to the exact rounding noise it will
experience at inference time.

- **Pros:** because the model's weights are optimized with quantization noise already "baked in,"
  QAT models generally hold accuracy substantially better than PTQ at the same bit-width, especially
  at aggressive settings (sub-4-bit, or full weight+activation INT8).
- **Cons:** requires a full (or fine-tuning-scale) training loop, a labeled/representative dataset,
  and meaningfully more compute and engineering than PTQ. It is often impractical for very large
  models where even a fine-tuning run is expensive, which is one reason PTQ methods (GPTQ, AWQ,
  GGUF) dominate the open-source LLM quantization ecosystem: they let anyone quantize a 70B model on
  a single GPU in an afternoon, whereas QAT for the same model requires a training cluster.

QLoRA sits in an interesting middle position: the *base* model is PTQ-quantized to NF4 and frozen
(never updated), while small LoRA adapter matrices are trained on top in higher precision — so the
frozen base takes a one-time PTQ hit, but the adapters can partially compensate for it during
fine-tuning. See the LoRA/QLoRA README for the mechanics of that adapter training.

## 4. Modern LLM Quantization Methods

### 4.1 GPTQ (Gradient-free Post-Training Quantization)

GPTQ quantizes a model **layer by layer** (each linear/projection weight matrix in turn), and within
a layer, **column by column** (or, in the common ordering, one row of the weight matrix / one
input-channel group at a time), using second-order curvature information to decide, at each step,
how to compensate not-yet-quantized weights for the error just introduced by rounding a weight
already handled.

Mechanism, at a level of detail useful for reasoning about why it works:

1. For a given layer with weight matrix `W` and calibration input activations `X`, GPTQ works with
   the layer's **Hessian** of the layer-wise squared error objective, `H = 2 X Xᵀ` (a proxy for how
   sensitive the layer's output is to a perturbation in each weight, derived from the second-order
   Taylor expansion of the squared reconstruction error).
2. It quantizes weights **one column at a time**. When column `q` is rounded to its nearest
   grid point, that introduces an error `δ = w_q - quantize(w_q)`.
3. GPTQ then **updates all remaining (not-yet-quantized) columns** to compensate for `δ`, using the
   inverse Hessian to compute the optimal compensating update — closely related to the classical
   **Optimal Brain Surgeon / Optimal Brain Quantization** result that the weight update minimizing
   the increase in loss from removing/quantizing one weight is proportional to `H⁻¹` applied to the
   error, restricted to the remaining weights.
4. This is done efficiently via a Cholesky decomposition of the (dampened) inverse Hessian computed
   once per layer, so that the per-column compensation reduces to a cheap rank-1-style update rather
   than a full re-inversion at each step.

The net effect: rounding error introduced early in the column order does not just sit there — it is
actively pushed into the weights that haven't been quantized yet, in the direction that best
preserves the layer's original output on the calibration data. This is why GPTQ can push weights
down to 3–4 bits with much smaller output-error accumulation than naively rounding every weight
independently (round-to-nearest, or "RTN") would produce. GPTQ typically operates per-group
(§2.3, group sizes of 128 are common) and is entirely calibration-data-driven and gradient-free in
the backprop sense — no loss.backward() over the original training objective is ever run; the
"gradient" here is the closed-form layer-reconstruction Hessian, not a training gradient.

### 4.2 AWQ (Activation-aware Weight Quantization)

AWQ starts from a different observation than GPTQ: not all weight *channels* are equally important
to preserve, and importance is best judged not from the weight values themselves but from the
**magnitude of the activations that multiply them**. A weight channel that consistently gets
multiplied by large activation values has an outsized effect on the layer's output — quantizing it
coarsely produces a large absolute error downstream, even if the weight values themselves look
unremarkable. Empirically, only a small fraction (often around 1%) of channels are "salient" in this
sense, but protecting them is disproportionately important for preserving accuracy.

Mechanism:

1. Run the calibration set through the model and record the **average magnitude of activations**
   per input channel, `s_j = mean(|X_{:,j}|)`, for each channel `j` feeding into a given weight
   matrix `W`.
2. Identify the channels with the largest activation magnitudes as salient.
3. Rather than literally keeping salient channels in higher precision (which would produce a
   mixed-precision matmul, awkward for hardware), AWQ instead **scales** each input channel `j` up
   by a per-channel factor `s_j^α` (for a tuned `α`, found by a small grid search on the calibration
   set) *before* quantizing the weights, and correspondingly divides the matching activations by the
   same factor:

```
Y = (X · diag(1/s)) · (diag(s) · W)
```

   This is a **mathematically exact re-scaling** — the product is unchanged — but it shifts a
   salient weight channel's values into a numeric range where the *fixed-bit-width* quantization
   grid represents it more precisely (larger values relative to the fixed number of quantization
   levels means each level represents comparatively less error as a fraction of that channel's
   scale, while the rescaled activation absorbs the inverse scale in FP16 at negligible cost since
   activations are not being quantized in the weight-only setting AWQ was designed for).
4. All weights are then quantized uniformly (typically per-group, RTN or with light optimization),
   but the important channels have already been pre-conditioned to survive that quantization better.

The key conceptual difference from GPTQ: GPTQ *reactively compensates* for error after it's
introduced, column by column, using Hessian information derived from `XXᵀ`; AWQ *proactively
protects* the weights that matter most before quantizing at all, using activation statistics
directly rather than a Hessian, and does so without touching the model's actual weight *values*
until the final quantization step. AWQ is also simpler and faster to run than GPTQ (no per-layer
Cholesky/Hessian-inversion procedure), which is part of why it is popular for very large models.

### 4.3 GGUF and the llama.cpp Quantization Zoo

**GGUF** (GPT-Generated Unified Format, the successor to the earlier GGML format) is a *file
format*, not a quantization algorithm — it's the container `llama.cpp` and related runtimes (Ollama,
LM Studio, etc.) use to package quantized model weights, tokenizer data, and architecture metadata
into a single portable file, and it supports a whole family of internal quantization schemes at
different bit-widths.

The naming convention on a GGUF quant variant (e.g., `Q4_K_M`, `Q5_K_M`, `Q8_0`) encodes, roughly:

- **The leading number** — `Q4`, `Q5`, `Q8` — is the nominal bits per weight (4, 5, 8). This is an
  average; some tensors inside the same file may be kept at a different precision (see below).
- **The scheme suffix** — `_0`, `_1`, `_K` — identifies the quantization *block scheme*:
  - `_0` is the original, simplest scheme: fixed-size blocks (historically 32 weights) each with a
    single FP16 scale and no zero-point (symmetric quantization per block).
  - `_1` adds a zero-point per block (asymmetric), trading a little extra metadata for better
    handling of skewed blocks.
  - `_K` ("k-quants") is the newer, more elaborate scheme: it groups weights into larger
    superblocks that are themselves subdivided into smaller sub-blocks, each sub-block getting its
    own scale, and the scales themselves are quantized too (a form of nested/double quantization
    conceptually similar to bitsandbytes' double quantization in §5) — this recovers most of the
    accuracy of a smaller group size while keeping metadata overhead low.
- **The trailing size letter** — `_S`, `_M`, `_L` (small/medium/large) — on k-quant variants
  indicates how aggressively *mixed precision* is applied across the model: k-quant schemes
  typically keep more sensitive tensors (e.g., attention output projections, or the final few
  layers) at a slightly higher effective bit-width than the nominal one, and `_S`/`_M`/`_L` dial how
  much of the model gets that higher-precision treatment — `_S` is the most aggressively
  low-bit-width version, `_M` is the common recommended middle ground, `_L` spends more bits (and
  file size) for closer accuracy to fp16.

So `Q4_K_M` reads as: "4-bit nominal, k-quant block scheme, medium mixed-precision profile" — the
de facto default for a strong accuracy/size trade-off in the llama.cpp ecosystem — while `Q8_0`
reads as "8-bit nominal, original simple per-block-scale scheme," commonly used when near-lossless
quality is wanted and the extra size over 4-bit is acceptable. GGUF quantization is applied
post-training (it is a PTQ family), generally via round-to-nearest per block rather than GPTQ/AWQ's
optimization objective, though llama.cpp tooling can also *import* GPTQ/AWQ-style calibration data
("importance matrix" / `imatrix` quantization) to improve accuracy at very low bit-widths.

## 5. The Role of `bitsandbytes`

`bitsandbytes` is the library that makes 8-bit and 4-bit quantization usable *directly inside a
PyTorch/Hugging Face training or inference pipeline*, as drop-in replacements for `nn.Linear`
layers, rather than as an offline export step producing a separate file format (contrast with
GGUF, which is specifically for exporting to a standalone runtime).

Its two headline capabilities:

- **`LLM.int8()`** — 8-bit inference/training that handles the outlier problem in transformer
  activations (see §7) by decomposing each matrix multiply into a high-precision path for the small
  number of outlier feature dimensions and a fast INT8 path for everything else, then recombining
  the results. This is what backs `load_in_8bit=True` in Hugging Face `transformers`.
- **NF4 (§2.4) with double quantization** — 4-bit weight storage for `load_in_4bit=True` /
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. **Double quantization**
  additionally quantizes the per-block scale factors themselves (normally stored as FP32, one per
  64-weight block) down to 8-bit, saving roughly another 0.4 bits per parameter on average with
  negligible extra error — small per-tensor, but meaningful in aggregate across a 7B+ parameter
  model.

The mechanism that matters most for **QLoRA** specifically is **on-the-fly dequantization during
the forward pass**: the frozen base model's weights are stored compressed in 4-bit NF4 in GPU
memory at rest, but `bitsandbytes` transparently dequantizes each weight block back to a compute
dtype (typically BF16, via `bnb_4bit_compute_dtype`) immediately before it is used in a matmul, then
discards the dequantized copy. The model never materializes a full-precision copy of all its
weights simultaneously — only one block/layer's worth at a time — which is precisely what allows a
model far larger than the GPU's memory-at-fp16 budget to still run forward and backward passes for
LoRA training, at the cost of the extra dequantization compute on every forward call. This library's
own use of exactly this path is detailed in §9 and cross-referenced in the LoRA/QLoRA README.

## 6. The Accuracy / Size / Speed Trade-off Surface

Quantization is not a free lunch, and the size of the "lunch" you get depends heavily on bit-width:

- **FP16/BF16 → INT8** is, empirically, **near-lossless** for most LLMs on most benchmarks. The
  extra 8 bits of headroom (256 vs. 16 levels) gives enough resolution that per-channel or
  per-tensor affine quantization introduces error small relative to the model's own noise floor;
  perplexity increases are typically well under 1%, and downstream task accuracy differences are
  often within run-to-run noise.
- **INT8 → INT4** is where degradation becomes measurable and method-dependent. Simple
  round-to-nearest INT4 can produce noticeable perplexity increases and occasional qualitative
  failures (repetition, factual slips) especially in smaller models; GPTQ/AWQ/NF4 close much of this
  gap by being smarter about *where* the remaining error budget goes, but a gap generally remains
  versus INT8 or FP16.
- **Model size interacts with bit-width.** Larger models tend to be more robust to aggressive
  quantization than smaller ones, because they have more redundant capacity — many
  near-duplicate/interchangeable directions in weight space — that can absorb quantization noise
  without changing the function the model computes very much. A 1–3B parameter model pushed to
  INT4 loses proportionally more capability than a 70B model pushed to the same bit-width, because
  the smaller model has less redundancy to spend on absorbing the noise. This is a practical reason
  to be more conservative (INT8, or a higher-quality 4-bit scheme with a larger group size) when
  quantizing small models.

**How the damage is measured.** Two complementary metrics are standard:

- **Perplexity** on a held-out text corpus (WikiText-2 is the most common benchmark in quantization
  papers) — a smooth, cheap-to-compute proxy for "how well does this quantized model still predict
  next tokens the way the original did." Perplexity is exponentiated cross-entropy loss:
  `PPL = exp( (1/N) Σ -log p(x_i | x_<i}) )`. It is sensitive to small distributional shifts and easy
  to compute without any task-specific labels, which is why it is the default first check when
  evaluating a new quantization method or setting.
- **Downstream task benchmarks** (MMLU, HellaSwag, GSM8K, HumanEval, etc.) — because perplexity can
  understate real damage: a model can retain low perplexity (it still "sounds like itself"
  token-by-token) while losing sharper capabilities like multi-step arithmetic or code correctness,
  which only show up when you measure task accuracy directly. A thorough quantization evaluation
  reports both: perplexity as a fast sanity check, and a battery of task benchmarks as the real
  measure of whether the quantized model is still fit for purpose.

The practical rule of thumb this produces: default to INT8 (or skip quantization entirely) when
accuracy is paramount and memory is not the binding constraint; drop to a well-implemented 4-bit
scheme (NF4, GPTQ, AWQ, or GGUF `Q4_K_M`) when memory/throughput is the binding constraint and a
small, measured accuracy cost is acceptable; and always validate the specific model/bit-width/method
combination with perplexity plus a couple of downstream benchmarks rather than assuming published
numbers from a different model transfer directly.

## 7. Weight-Only vs. Weight+Activation Quantization

### 7.1 Weight-only quantization

Weight-only quantization compresses the weight tensors to a low bit-width but keeps activations
(the intermediate values flowing between layers at inference time) in their original floating-point
precision (typically FP16/BF16). The matmul, at execution time, dequantizes the weight block back to
float (§5) and performs a standard float matmul against the float activations.

This is, by a wide margin, the **most common** approach for LLM inference, for a simple reason:
during autoregressive decoding (generating one token at a time), the weight matrices are far larger
than the activation vectors being multiplied against them, so **memory traffic is dominated by
moving weights**, not activations. Shrinking only the weights already captures nearly all of the
available memory-bandwidth win, without having to solve the harder problem of quantizing
activations (below). GPTQ, AWQ, NF4/bitsandbytes 4-bit, and most GGUF quant levels are all
weight-only schemes in their default/common configurations.

### 7.2 Weight+activation quantization

To get an actual speedup from low-precision *matrix multiplication itself* — using a hardware
integer tensor core instead of dequantizing back to float before the matmul — **both operands** of
the matmul must be in the low-precision format. This means activations, not just weights, must be
quantized on the fly during inference (typically dynamically, per forward pass, since the range of
activation values shifts with the input).

This is considerably harder than weight-only quantization because **activation distributions in
transformers contain large outliers** concentrated in a small number of fixed feature dimensions
(this is the same outlier phenomenon `LLM.int8()` was built to handle, §5). A handful of channels
can have magnitudes 10–100x larger than the rest of the activation tensor. Under a single
per-tensor `scale` (§2.1), that scale is forced to accommodate the outlier channels, which crushes
the resolution available to every other (normal-magnitude) channel down near the rounding floor —
producing much larger quantization error than the equivalent weight quantization problem, where no
comparable outlier structure exists.

**SmoothQuant** addresses this directly by observing that the *difficulty* of quantization can be
migrated between activations and weights via a mathematically equivalent rescaling, similar in
spirit to AWQ's channel scaling (§4.2) but applied in the opposite direction of primary benefit.
For a linear layer `Y = X W`, SmoothQuant introduces a diagonal per-channel smoothing factor `Λ`:

```
Y = X W = (X · Λ⁻¹) (Λ · W)
```

`Λ` is chosen (per input channel `j`, based on calibration statistics of both `X` and `W`) to
**divide down** the outlier activation channels by roughly their magnitude while **multiplying up**
the corresponding weight channels by the same factor, controlled by a migration-strength
hyperparameter (commonly denoted `α`) that balances how much of the outlier "difficulty" moves into
the weights versus staying in the activations. Weights are smooth and well-behaved to begin with
(no outlier structure, §2.4), so they can absorb this extra multiplicative range cheaply; activations
become correspondingly flatter and easier to quantize accurately with a simple per-tensor or
per-token scale. The rescaling is exact (the product is unchanged before quantization), so the only
error introduced is the quantization step itself, now performed on two much better-conditioned
tensors — which is what finally makes full INT8 weight+activation quantization practical for
transformer models without a large accuracy penalty, and what unlocks real INT8 tensor-core
throughput rather than just memory savings.

## 8. Diagrams and Comparison

### 8.1 Quantize/dequantize round trip for one weight matrix

```
 FP16 weight matrix W                 Calibration /            Quantized storage
 (rows = output channels)             statistics pass           (INT4 or INT8 codes
┌─────────────────────────┐                                     + per-group metadata)
│  0.031  -0.482   0.117  │           for each group of      ┌───────────────────────┐
│ -0.201   0.056  -0.334  │  ──────▶  g weights, compute:     │  q  q  q │ scale₁     │
│  0.412  -0.019   0.208  │           scale = (max-min)/      │  q  q  q │ zp₁ (opt.) │
│    ...     ...     ...  │                     (qmax-qmin)   │  q  q  q │ scale₂     │
└─────────────────────────┘           zero_point = round(     │  q  q  q │ zp₂ (opt.) │
        │                                qmin - min/scale)    │   ...    │   ...      │
        │  q = round(x/scale)                                 └───────────────────────┘
        │      + zero_point,                                          │
        │      clamped to [qmin,qmax]                                 │
        ▼                                                             │
┌─────────────────────────┐                                           │
│  quantized INT codes    │ ◀─────────────────────────────────────────┘
│  (4 or 8 bits each)     │      stored on disk / in GPU memory (this is the
└─────────────────────────┘      compressed, resident representation)

                    At matmul time (weight-only quant, §7.1):
┌─────────────────────────┐   x_approx = (q - zero_point) * scale   ┌─────────────────────────┐
│  quantized INT codes    │ ───────────────────────────────────────▶│ dequantized FP16 weight │
│  + scale/zero_point     │        (per group, using that           │  block  (transient,     │
└─────────────────────────┘         group's own scale/zp)           │  discarded after use)   │
                                                                     └───────────┬─────────────┘
                                                                                 │
                                                            FP16 activations ────┤
                                                                                 ▼
                                                                    ┌─────────────────────────┐
                                                                    │   standard FP16 matmul   │
                                                                    │  Y = X · W_dequantized   │
                                                                    └─────────────────────────┘
```

### 8.2 Bit-width comparison

| Format | Bits | Approx. relative size (vs. FP32) | Typical use case |
|---|---|---|---|
| FP32 | 32 | 1.00x | Training master weights, optimizer states; rarely used for LLM inference |
| FP16 | 16 | 0.50x | Training / inference on hardware with good FP16 tensor-core support; standard "full precision" inference baseline |
| BF16 | 16 | 0.50x | Training (preferred over FP16 for its wider dynamic range, avoids overflow); also used as a QLoRA compute dtype |
| INT8 | 8 | 0.25x | Near-lossless weight-only PTQ; `LLM.int8()`; also used for activation quantization when paired with SmoothQuant/GPTQ for full INT8 matmul speedups |
| NF4 | 4 (+ small metadata overhead) | ~0.125x (plus ~5–10% metadata) | QLoRA frozen base-model weights; bitsandbytes 4-bit inference |
| INT4 | 4 (+ small metadata overhead) | ~0.125x (plus metadata for scales/zero-points) | GPTQ, AWQ, GGUF `Q4_*` weight-only quantization for memory-constrained inference |

("Relative size" counts weight storage only, per parameter, ignoring the small per-group
scale/zero-point metadata discussed in §2.3 — real file sizes for 4-bit formats end up a few
percent above the raw `bits/8` figure once that metadata is included.)

## 9. Quantization in This Repository

This codebase uses 4-bit quantization concretely in
`orchestrator_framework_layer/akanade/backend/scripts/finetune_local_slm.py`, in its `--qlora` code
path (see `load_base_model`). When `--qlora` is passed (and a CUDA GPU is available — bitsandbytes'
4-bit kernels are GPU-only, so the script refuses cleanly on CPU-only boxes rather than crashing
inside bitsandbytes), the base model is loaded with:

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=quant_config, device_map="auto")
```

This is exactly the mechanism described in §2.4 and §5 above: the base model's weights are stored
in 4-bit **NF4** format (`bnb_4bit_quant_type="nf4"`), with **double quantization** enabled
(`bnb_4bit_use_double_quant=True`) to additionally compress the per-block scale factors, and
dequantized on the fly to **BF16** (`bnb_4bit_compute_dtype=torch.bfloat16`) for each forward-pass
matmul. The base model stays frozen in this compressed form throughout fine-tuning; only small
LoRA adapter matrices (trained in higher precision on top of the frozen quantized base) receive
gradient updates. The adapter mechanics, rank/alpha choices, and the `prepare_model_for_kbit_training`
step that follows are QLoRA-specific concerns and are covered in full in the LoRA/QLoRA README at
[`docs/ml-concepts/08-lora-qlora/README.md`](../08-lora-qlora/README.md) rather than re-derived
here — this document's job is only to explain the quantization half of that pipeline (what NF4 is,
why double quantization saves memory, and how on-the-fly dequantization lets a model larger than the
GPU's fp16-equivalent memory budget still be fine-tuned).
