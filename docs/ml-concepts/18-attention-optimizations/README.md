# Attention-Layer Optimizations in Modern Transformers

Self-attention is the computational core of every transformer, and it is also
the layer that scales worst with sequence length and is hardest to serve
cheaply at inference time. This document works through the baseline
mathematics, the two independent bottlenecks (compute and memory), and the
four families of optimizations that production LLMs actually ship with:
FlashAttention (algorithmic), Multi-Query and Grouped-Query Attention
(architectural), and sliding-window attention (structural). Each section
includes the underlying math and, where useful, an ASCII diagram of the data
flow.

---

## 1. Baseline: Scaled Dot-Product Multi-Head Attention

### 1.1 The formula

For a single attention head, given query, key, and value matrices
`Q, K, V ∈ R^(n × d_k)` (sequence length `n`, per-head dimension `d_k`):

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

Expanded step by step:

1. **Score matrix**: `S = Q K^T`, shape `(n, n)`. Entry `S_ij = q_i · k_j` is
   the raw (unnormalized) similarity between query token `i` and key token
   `j`.
2. **Scale**: divide by `sqrt(d_k)`. Without this, as `d_k` grows the dot
   products `q_i · k_j` grow in variance roughly proportional to `d_k`
   (assuming unit-variance, zero-mean components), which pushes the softmax
   into a saturated regime with vanishing gradients. Dividing by `sqrt(d_k)`
   keeps `Var(q_i · k_j / sqrt(d_k)) ≈ 1` regardless of dimension.
3. **Softmax over rows**: `P = softmax(S / sqrt(d_k))`, applied
   independently to each row `i`, so `P_ij = exp(S_ij) / Σ_j' exp(S_ij')`.
   Each row is a probability distribution over which key positions token `i`
   attends to.
4. **Weighted sum of values**: `O = P V`, shape `(n, d_k)`. Output token `i`
   is a convex combination of value vectors, weighted by attention
   probability.

### 1.2 Multi-head attention

Instead of a single attention computation over the full model dimension
`d_model`, multi-head attention (MHA) runs `H` independent attention
computations ("heads") in parallel, each in a lower-dimensional subspace
`d_k = d_model / H`:

```
head_h = Attention(Q W_h^Q, K W_h^K, V W_h^V),  h = 1 ... H
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W^O
```

Each head has its own learned projection matrices `W_h^Q, W_h^K, W_h^V ∈
R^(d_model × d_k)`. The motivation is that different heads can specialize —
one head might learn positional/local patterns, another might learn
long-range coreference, another syntactic dependency structure — and the
final `W^O ∈ R^(d_model × d_model)` projection mixes the per-head outputs
back into the residual stream.

Critically for everything that follows: **every head in standard MHA has
its own K and V projections**, not just its own Q. This means the KV state
that needs to be cached during autoregressive decoding scales linearly with
`H`.

### 1.3 Why naive attention is compute-expensive: O(n²)

Computing `S = Q K^T` requires, for every one of the `n` queries, a dot
product against every one of the `n` keys, each dot product costing `O(d_k)`
multiply-adds. Total cost for `QK^T`: `O(n² · d_k)`. The subsequent
`P V` matmul is likewise `O(n² · d_k)`. So a single head costs `O(n² d_k)`
FLOPs, and with `H` heads over the full model, attention costs
`O(n² · d_model)` per layer.

The quadratic term `n²` is the problem: doubling the context length
quadruples the attention FLOPs (and, as shown next, quadruples the memory
footprint of the score matrix). This is the reason context-length scaling
has historically been so expensive — an 8K-context model doing attention
costs 4x the attention compute of a 4K-context model at equal batch size,
independent of any change in model width or depth.

### 1.4 Why naive attention is memory-expensive: materializing the n×n matrix

The compute cost above is only half the story, and on modern GPUs it is
usually not even the dominant half. The naive implementation, as expressed
directly by the formula, requires **materializing the full score matrix
`S` (and softmax output `P`) in memory**, for every head, every layer,
every element in the batch:

```
memory(S) = batch_size × H × n × n × bytes_per_element
```

For a single sequence of length `n = 8192` in fp16, one head's score matrix
alone is `8192 × 8192 × 2 bytes ≈ 128 MB`. Multiply by, say, 32 heads and a
batch of even modest size, and this matrix alone can exceed the capacity of
a GPU's on-chip memory by orders of magnitude, forcing it to live in
high-bandwidth memory (HBM) — the GPU's DRAM, which is large (tens of GB)
but comparatively slow to access relative to the on-chip SRAM caches.

This matters because the naive attention implementation, as typically
written in PyTorch (`Q @ K.T`, then `softmax`, then `@ V` as three separate
kernel calls), performs several **full round trips to HBM**:

1. Write `S = QK^T` out to HBM.
2. Read `S` back from HBM, compute `softmax`, write `P` back to HBM.
3. Read `P` and `V` from HBM, compute `P @ V`, write output to HBM.

Each of these reads and writes moves `O(n²)` elements across the
HBM-to-SRAM bus. As Section 2 explains, on modern accelerators this
data-movement cost — not the arithmetic itself — is what actually limits
attention's wall-clock speed.

---

## 2. FlashAttention: Making Attention IO-Aware

### 2.1 The core insight: attention is memory-bandwidth-bound, not compute-bound

Every operation on a GPU has an **arithmetic intensity**: the ratio of
FLOPs performed to bytes moved between HBM and on-chip SRAM. Whether an
operation is compute-bound or memory-bound depends on how its arithmetic
intensity compares to the GPU's own FLOPs-to-bandwidth ratio (its "ridge
point" on a roofline model).

Modern GPUs have enormous compute throughput (hundreds of TFLOPs in fp16
with tensor cores) relative to HBM bandwidth (on the order of 1.5–3 TB/s
for high-end datacenter GPUs, roughly 100-200x less traffic than FLOPs
available per second at peak). Standard attention's constituent operations
— the softmax, masking, and dropout applied elementwise to a giant `n × n`
matrix — do very little arithmetic per byte read/written. They are
therefore **memory-bandwidth-bound**: the GPU's compute units sit idle
waiting for data to arrive from HBM, and runtime is dominated by how many
bytes of the `n × n` matrix get shuttled back and forth, not by the number
of multiply-adds in the `QK^T` and `PV` matmuls.

FlashAttention's central observation is: if you can avoid ever writing
the full `n × n` matrix to HBM in the first place — restructuring the
computation so intermediate results stay resident in the GPU's fast
on-chip SRAM — you eliminate the dominant cost, even though you still
perform (almost) the same number of FLOPs. It is a classic **IO-aware**
algorithm redesign: same asymptotic compute complexity, drastically less
data movement.

### 2.2 The obstacle: softmax needs a global normalizer

The difficulty is that softmax is not a naturally "blockable" (tileable)
operation. To compute `P_ij = exp(S_ij) / Σ_j' exp(S_ij')` for a given
row `i`, you conventionally need to see the *entire* row of scores first
— to find its normalizing sum (and, for numerical stability, its max) —
before you can compute a single normalized probability. If you only have
one block of `K`/`V` in SRAM at a time, you don't yet know the full row
sum, so you can't finish normalizing.

FlashAttention solves this with **online (running) softmax**: process
key/value blocks one at a time, and maintain running statistics that let
you correct previously-computed partial output whenever a new block
reveals a larger max or contributes more to the normalizing sum — without
ever holding the full row of scores in memory at once.

### 2.3 Online softmax: the running max/sum rescaling trick

For numerical stability, softmax is always computed relative to the row's
max value:

```
softmax(x)_j = exp(x_j - max(x)) / Σ_j' exp(x_j' - max(x))
```

Subtracting the max prevents `exp()` from overflowing for large scores;
it does not change the result since the max cancels in the ratio.

The problem: when processing scores incrementally in blocks, you don't
know the true row-max in advance. Online softmax handles this by tracking,
for each query row `i`, a **running max** `m_i` and a **running
(unnormalized) sum** `l_i`, updating them block by block and *rescaling*
previously accumulated output whenever the running max changes.

Concretely, suppose after processing blocks `1..k-1` you have:

- `m_i^(k-1)`: running max of scores seen so far for row `i`
- `l_i^(k-1)`: running sum of `exp(s - m_i^(k-1))` over scores seen so far
- `O_i^(k-1)`: running (unnormalized) weighted sum of `V` rows seen so far,
  computed relative to `m_i^(k-1)`

When a new block `k` arrives with local scores `s_i^(k)` (against that
block's `K`, `V`) and local block max `m̃ = max(s_i^(k))`:

1. Compute the new running max: `m_i^(k) = max(m_i^(k-1), m̃)`
2. Rescale the old accumulator to the new max, since it was normalized
   relative to the stale `m_i^(k-1)`:
   ```
   correction = exp(m_i^(k-1) - m_i^(k))
   l_i^(k) = l_i^(k-1) · correction + Σ_j exp(s_ij^(k) - m_i^(k))
   O_i^(k) = O_i^(k-1) · correction + exp(s_i^(k) - m_i^(k)) · V^(k)
   ```
3. After the last block `T`, the true softmax-weighted output is recovered
   by a single final division: `O_i = O_i^(T) / l_i^(T)`.

Because `exp(a - c) = exp(a - b) · exp(b - c)`, rescaling by
`exp(m^(k-1) - m^(k))` exactly accounts for the shift in reference max, so
the incrementally-computed result is **mathematically identical** to
computing softmax over the full row at once — it is not an approximation.
This is the trick that lets FlashAttention process one `K`/`V` block at a
time, entirely within SRAM, and still emit a numerically correct,
numerically stable softmax at the end — without ever materializing the
full `n × n` row of scores.

### 2.4 Tiling: the full algorithm sketch

FlashAttention picks block sizes `B_r` (query rows per block) and `B_c`
(key/value rows per block) sized so that a `Q` block, `K` block, `V`
block, and the running statistics all fit simultaneously in on-chip SRAM
(on NVIDIA GPUs, this is the shared memory / L1-adjacent scratchpad local
to a streaming multiprocessor, tens to a couple hundred KB — vs. tens of
GB of HBM, several orders of magnitude smaller but roughly two orders of
magnitude faster to access).

Outer loop over query blocks, inner loop over key/value blocks
(conceptually — the real kernel parallelizes the outer loop across GPU
thread blocks):

```
for each query block Q_i  (size B_r × d_k):
    initialize O_i = 0, l_i = 0, m_i = -inf     # per-row running stats, held in SRAM

    for each key/value block K_j, V_j  (size B_c × d_k):
        load K_j, V_j from HBM into SRAM        # only these small blocks touch HBM
        S_ij = Q_i K_j^T / sqrt(d_k)             # compute in SRAM, size B_r × B_c only
        m_new  = max(m_i, rowmax(S_ij))
        P_ij   = exp(S_ij - m_new)
        correction = exp(m_i - m_new)
        l_i    = l_i * correction + rowsum(P_ij)
        O_i    = O_i * correction + P_ij @ V_j
        m_i    = m_new

    O_i = O_i / l_i                              # final normalization
    write O_i to HBM                             # one write per query block, not per n×n entry
```

The full `n × n` score matrix `S` (and the softmax matrix `P`) is never
materialized — at any instant, only a `B_r × B_c` tile of it exists, and
only inside fast SRAM. HBM traffic drops from `O(n²)` (read/write the full
matrix, potentially multiple times) to `O(n² d_k / M)` where `M` is
proportional to SRAM size — in practice, the number of HBM accesses is
reduced by roughly the SRAM-tile-size factor, which is why the technique
gives outsized wall-clock speedups despite doing essentially the same
number of FLOPs. The backward pass uses the same tiling and online-softmax
machinery, recomputing needed values from the saved `(m_i, l_i)` statistics
rather than storing the full `n × n` attention matrix for the backward
pass — trading a bit of recompute for a large reduction in memory traffic
and peak memory usage (attention memory goes from `O(n²)` to `O(n)`).

### 2.5 FlashAttention-2 and FlashAttention-3

FlashAttention-2 and FlashAttention-3 keep the same fundamental
tiling/online-softmax algorithm and push further on **hardware
utilization and parallelism**, rather than changing the underlying math:

- **FlashAttention-2** reduces non-matmul FLOPs (fewer rescaling
  operations, restructured loop order so the outer loop is over query
  blocks and parallelism is exposed across both the sequence dimension and
  attention heads), improving GPU occupancy and reducing warp-level
  synchronization overhead — better utilizing the GPU's streaming
  multiprocessors, especially for long sequences and cases with fewer
  heads/larger batch where the original scheme under-parallelized.
- **FlashAttention-3** targets newer accelerator architectures (e.g.
  Hopper-class GPUs) directly, exploiting asynchronous tensor core
  instructions and overlapping matmul and softmax work between the GPU's
  tensor-core and non-tensor-core execution paths (warp specialization /
  producer-consumer pipelining), and adding support for lower-precision
  (fp8) execution paths — squeezing more of the theoretical hardware peak
  out of the same tiling/online-softmax algorithm.

The conceptual takeaway: FlashAttention v1 established the IO-aware
algorithm; v2 and v3 are engineering refinements that better saturate
increasingly capable hardware, without changing the exact-softmax,
no-materialized-n×n-matrix contract.

---

## 3. Multi-Query Attention (MQA): Shrinking the KV Cache

### 3.1 Motivation: autoregressive decoding is dominated by the KV cache

FlashAttention addresses attention cost during a single forward pass (as
in training or the prefill phase of inference). But autoregressive
**decoding** — generating tokens one at a time — has a different
bottleneck. At each decode step, the model must attend the current token's
query against *all* previously generated keys and values. To avoid
recomputing K and V for the entire prefix at every step, implementations
maintain a **KV cache**: the K and V projections for every past token, for
every layer, for every head, kept resident in GPU memory.

Standard MHA's KV cache size (per sequence) is:

```
KV_cache_size = 2 × n_layers × H × n × d_k × bytes_per_element
```

(the factor of 2 is for both K and V). This grows linearly with sequence
length and, critically, linearly with the number of heads `H`. During
decoding, at every single generated token, the GPU must **read the entire
KV cache from HBM** to compute attention scores against it — and because
generating one token does very little arithmetic (a single query row
against the cache) relative to the bytes read, decoding is emphatically
memory-bandwidth-bound. The KV cache read is the dominant per-token cost.
Larger KV caches (from more heads, longer context, or larger batch) both
consume more GPU memory (limiting how many sequences can be served
concurrently) and take longer to stream from HBM per decode step (limiting
per-token latency and throughput).

### 3.2 The MQA design

Multi-Query Attention's fix: keep **per-head queries**, but use a
**single, shared K projection and single, shared V projection across all
heads**. Formally:

```
head_h = Attention(Q W_h^Q, K W^K, V W^V),   h = 1 ... H
```

Note `W^K` and `W^V` no longer carry a subscript `h` — every head
projects the same underlying `K`/`V` from the sequence and only differs
in its query projection `W_h^Q`. All `H` heads attend using the same set
of keys/values; they simply ask different "questions" of them via
distinct `Q` projections.

This shrinks the KV cache by exactly a factor of `H`:

```
KV_cache_size_MQA = 2 × n_layers × 1 × n × d_k × bytes_per_element
                   = KV_cache_size_MHA / H
```

With, say, `H = 32` heads, MQA cuts KV cache memory (and the HBM traffic
required to read it every decode step) by 32x. This directly translates
to: more sequences fit in GPU memory at once (higher serving batch size /
throughput), and each decode step reads far less data (lower per-token
latency), since the KV-cache read is the dominant cost of decoding.

### 3.3 The cost: expressiveness loss

The tradeoff is architectural expressiveness. In full MHA, each head can
develop its own *view* of the sequence — one head's K/V projection might
emphasize different features (e.g., positional cues) than another head's.
Collapsing all heads onto one shared K/V projection removes this degree of
freedom: heads can still ask different questions (`Q` differs) but they
must all query the *same* representation of keys and values. Empirically
this manifests as a measurable, if often modest, drop in model quality
relative to an equivalently-sized MHA model, particularly noticeable on
tasks that benefit from rich, multi-perspective attention patterns.
This is the reason MQA, while very cheap for serving, has been less
universally adopted for training new large models from scratch than the
compromise described next.

---

## 4. Grouped-Query Attention (GQA): The Practical Middle Ground

### 4.1 Design: interpolating between MHA and MQA

Grouped-Query Attention generalizes both MHA and MQA with a single
parameter, the number of groups `G` (where `1 ≤ G ≤ H` and typically
`G` divides `H`). The `H` query heads are partitioned into `G` groups;
all query heads within a group share one K/V projection:

```
group(h) = ceil(h / (H / G))                     # which of the G groups head h belongs to
head_h   = Attention(Q W_h^Q, K W_{group(h)}^K, V W_{group(h)}^V)
```

Each of the `G` groups has its own `W_g^K, W_g^V`, and all `H / G` query
heads assigned to that group share it. This gives:

```
KV_cache_size_GQA = 2 × n_layers × G × n × d_k × bytes_per_element
                   = KV_cache_size_MHA × (G / H)
```

The two extremes fall out naturally:

- `G = H` (each head is its own group, no sharing at all) → **identical to
  standard MHA**.
- `G = 1` (all heads share one group) → **identical to MQA**.

Values in between — e.g., `G = 8` with `H = 32` (4 query heads per KV
group) — recover most of the KV-cache savings of MQA (a `32/8 = 4x`
reduction here) while giving up much less representational capacity,
since groups of a few heads still retain distinct K/V views from each
other, just not fully independent per-head views.

### 4.2 Why GQA is the industry default

GQA is used by Llama 2 (34B/70B variants) and Llama 3 (all sizes), Mistral
(alongside sliding-window attention, see Section 5), and most other
current-generation open LLMs, because empirically it sits very close to
full MHA quality while capturing most of MQA's inference efficiency gains
— an attractive point on the quality/cost Pareto frontier. It is also
practical from a training standpoint: an existing MHA-pretrained
checkpoint can be *converted* to GQA by mean-pooling each group's original
per-head K/V projections into one shared projection per group, then doing
a short continued-pretraining ("uptraining") phase to recover any quality
lost from the conversion — meaning GQA doesn't have to be designed in from
step zero, it can be retrofitted onto existing MHA architectures at a
fraction of the cost of training from scratch.

### 4.3 Compute vs. memory: what GQA does and doesn't change

It's worth being precise about what GQA optimizes. The `QK^T` and `PV`
matmul FLOP counts are governed by the number of *query* heads `H` (since
every query head still produces its own attention output), so GQA does
**not** reduce attention's raw compute (`O(n² d_model)` FLOPs is
unchanged). What it reduces is the number of distinct K/V projections
that must be **computed, cached, and read from HBM** — which is precisely
the bottleneck that matters for memory-bandwidth-bound autoregressive
decoding, per Section 3.1. This is why GQA/MQA are framed as inference
(specifically decode-time) optimizations rather than training-time
FLOP-reduction techniques.

---

## 5. Sliding Window Attention: Bounding the Receptive Field

### 5.1 Design: attend only to the last W tokens

Sliding window attention (also called local attention) restricts each
query position `i` to attend only to keys within a fixed window of size
`W` immediately preceding it (and typically itself), rather than the full
prefix of length `n`:

```
Attention_window(i) = softmax( q_i K_{[i-W+1 : i]}^T / sqrt(d_k) ) V_{[i-W+1 : i]}
```

This is implemented via a banded attention mask: instead of a full lower
triangular causal mask, positions `j < i - W + 1` are also masked out
(set to `-inf` before softmax) in addition to the standard causal mask
`j > i`.

### 5.2 Complexity: O(n·W) instead of O(n²)

Since each of the `n` query positions now attends to at most `W` keys
(instead of up to `n`), both compute and the KV cache footprint per layer
drop from `O(n²)` / `O(n)`-growing-unbounded to:

```
compute per layer:  O(n · W · d_k)      instead of O(n² · d_k)
KV cache per layer:  O(W)  (bounded!)    instead of O(n) (growing with sequence length)
```

When `W` is a fixed constant (e.g., 4096 tokens) chosen independently of
`n`, the KV cache no longer needs to grow with sequence length at all —
old keys/values that fall outside the trailing window can be **evicted**
from the cache. This is implemented as a **rolling (ring) buffer**: a
fixed-size buffer of capacity `W` where new K/V entries overwrite the
oldest ones once the buffer is full, rather than an ever-growing
allocation. Mistral's architecture uses exactly this: sliding window
attention (window size 4096 in the original Mistral 7B) paired with a
rolling buffer cache, so memory use during decoding is bounded regardless
of how long the generated sequence gets — a qualitatively different
scaling behavior from vanilla attention's unboundedly-growing KV cache.

### 5.3 Receptive field grows with depth, not width

A single sliding-window layer only lets token `i` see back `W` positions.
But **stacking `L` layers of sliding-window attention** allows information
to propagate further than `W`, because each layer's output already mixes
information from its own window, so the next layer's window effectively
reaches further back in the *original* input sequence. This is exactly
analogous to how stacking convolutional layers with a small fixed kernel
size grows the receptive field linearly with depth in a CNN.

Concretely, after `L` stacked layers each with window `W`, a token can be
influenced by input tokens up to roughly:

```
effective_receptive_field ≈ L × W
```

positions back (the exact bound depends on details like whether windows
overlap token-for-token, but the linear-in-depth scaling is the key
property). For example, with `W = 4096` and `L = 32` transformer layers,
the theoretical effective receptive field is on the order of `32 × 4096 =
131072` tokens — far beyond what any single layer can see directly —
even though every individual layer only ever performs the cheap, bounded
`O(n · W)` local computation. Deeper layers implicitly aggregate
increasingly long-range context because the "sight" of tokens they
mix compounds across depth, just as deep CNNs build large image receptive
fields from small local kernels.

### 5.4 What is lost

Sliding window attention imposes a strong inductive bias: any single
layer literally cannot directly attend to a token more than `W` positions
away, so any information that needs to travel farther must be relayed
indirectly through the residual stream across several layers rather than
retrieved in one attention hop. For tasks requiring genuinely
long-range, single-hop dependency resolution (e.g., precisely recalling
a fact stated tens of thousands of tokens earlier, verbatim), this
indirect relay can be lossy compared to full attention, which can look
directly at the relevant token from any layer. In practice this is often
mitigated by combining sliding-window layers with occasional full-attention
layers (a design used in some models to blend the cost savings of local
attention with the reach of global attention at a few strategic layers).

---

## 6. Diagram: MHA vs. MQA vs. GQA

```
STANDARD MULTI-HEAD ATTENTION (MHA)              MULTI-QUERY ATTENTION (MQA)
  H independent K/V projections                   1 shared K/V projection for all H heads

  Q1  K1  V1        Q2  K2  V2                     Q1   Q2   Q3   Q4
   |   |   |          |   |   |                     |    |    |    |
   +---+---+          +---+---+                     +----+----+----+
       |                  |                                |
    head_1             head_2      ...  (H heads)      shared K, V
       |                  |                                |
       +--------+---------+                     +----------+----------+
                |                                |    |    |    |
   Q3  K3  V3        Q4  K4  V4               head_1 head_2 head_3 head_4  (H heads,
    |   |   |          |   |   |                  |    |    |    |          1 KV set)
    +---+---+          +---+---+                  +----+----+----+
        |                  |                                |
     head_3             head_4                          Concat -> W^O -> output
        |                  |
        +--------+---------+
                 |
             Concat -> W^O -> output

  KV cache size:  H x (K,V)                        KV cache size:  1 x (K,V)
                                                     -> H times smaller than MHA


                 GROUPED-QUERY ATTENTION (GQA), example H=4, G=2
                 pairs of heads share one K/V projection

     Q1     Q2              Q3     Q4
      \     /                \     /
       \   /                  \   /
     group 1: K,V           group 2: K,V
        |   |                  |   |
     head_1 head_2          head_3 head_4
        |     |                |     |
        +--+--+                +--+--+
           |                      |
           +----------+-----------+
                      |
              Concat -> W^O -> output

  KV cache size: G x (K,V)   (interpolates between MHA [G=H] and MQA [G=1])
```

---

## 7. Diagram: FlashAttention Tiling vs. Naive Materialization

```
NAIVE ATTENTION (materializes full n x n matrix in slow HBM)

   Q (n x d)   K (n x d)                       HBM (slow, large)
      |           |                            +----------------------------+
      +----matmul-+                            |  S = Q K^T   (n x n)       |  <- full write to HBM
              |                                |  written here, then        |
              v                                |  read back for softmax     |
      S (n x n) ---------- write ------------> +----------------------------+
              |                                              |
        read back from HBM                                   |
              |                                               v
              v                                     P = softmax(S)  (n x n)
      softmax(S) = P  --------- write -----------> +----------------------------+
              |                                    |  P written to HBM,        |
        read P and V back from HBM                 |  then read back again     |
              |                                    +----------------------------+
              v
      O = P @ V  ---> output

   Multiple full O(n^2) round trips to HBM: this IS the bottleneck (memory-bandwidth bound)


FLASHATTENTION (tiled, fused, stays in fast on-chip SRAM)

   HBM: Q, K, V (only these ever fully live in HBM; O(n) each, not O(n^2))
    |
    | stream small blocks in
    v
   +-------------------- on-chip SRAM (fast, small) --------------------+
   |                                                                    |
   |   Q_i block          K_1,V_1   K_2,V_2   K_3,V_3   ...  K_T,V_T    |
   |   (B_r x d)             |         |         |              |      |
   |      |                  v         v         v              v      |
   |      |            +---------+ +---------+ +---------+  +--------+ |
   |      +----------->| S_i1    | | S_i2    | | S_i3    |..| S_iT   | | each tile
   |                   | (B_r x  | | (B_r x  | | (B_r x  |  |        | | B_r x B_c,
   |                   |  B_c)   | |  B_c)   | |  B_c)   |  |        | | never leaves
   |                   +----+----+ +----+----+ +----+----+  +---+----+ | SRAM
   |                        |            |           |           |    |
   |                        v            v           v           v    |
   |                 running softmax stats carried block-to-block:    |
   |                   m_i  (running max)                             |
   |                   l_i  (running sum of exp(s - m_i))             |
   |                   O_i  (running weighted V accumulator)          |
   |                 rescale O_i, l_i by exp(m_old - m_new) each step |
   |                                                                    |
   +----------------------------- final: O_i = O_i / l_i --------------+
                                        |
                                        v
                              write O_i to HBM (O(n) total, not O(n^2))

   The full n x n matrix S (and P) never exists anywhere in memory at once,
   in either HBM or SRAM -- only one B_r x B_c tile at a time.
```

---

## 8. Synthesis: One Bottleneck, Four Angles of Attack

Despite looking like unrelated tricks, FlashAttention, MQA, GQA, and
sliding-window attention are all responses to the **same two underlying
costs** — the `O(n²)` compute/memory footprint of attention, and the
memory-bandwidth cost of reading the KV cache at every autoregressive
decode step — attacked from different layers of the stack:

| Technique | Layer of attack | What it reduces | What's unchanged |
|---|---|---|---|
| FlashAttention | **Algorithmic** (how the same math is computed) | HBM read/write traffic for the `n×n` score matrix; peak activation memory (`O(n²)` -> `O(n)`) | The exact math (exact softmax, same FLOPs); the KV cache size |
| MQA | **Architectural** (what gets computed and cached) | KV cache size and per-decode-step HBM reads, by exactly `H`x | Attention FLOPs (still `O(n² d_model)`); introduces some quality loss |
| GQA | **Architectural** (tunable middle ground) | KV cache size by `H/G`x, with a quality/memory dial | Attention FLOPs (still governed by `H` query heads) |
| Sliding window | **Structural** (what positions are visible at all) | Both compute (`O(nW)` vs `O(n²)`) *and* KV cache size (bounded by `W`, not `n`) | Direct single-hop long-range access (relies on depth to propagate) |

They compose rather than compete: a production model can (and typically
does) use FlashAttention-style fused, tiled kernels *and* a GQA (or
sliding-window + GQA) architecture simultaneously — FlashAttention makes
whatever attention pattern is computed IO-efficient, while GQA/sliding
window determine how much there is to compute and cache in the first
place. Mistral, for instance, combines GQA with sliding-window attention
and a rolling-buffer KV cache, then runs the resulting (small, banded)
attention pattern through a FlashAttention-style kernel.

At the systems level, the choice of KV-cache footprint per token (driven
by GQA's `G` and/or a sliding window's `W`) is the primary input to how
many concurrent sequences a serving system can batch, how much
paging/eviction pressure a KV-cache manager faces, and how attention
compares to the rest of the memory budget (model weights, activation
buffers). See:

- [`16-memory-optimization-training`](../16-memory-optimization-training/README.md)
  for how activation memory, gradient checkpointing, and optimizer state
  interact with the same HBM-bandwidth and capacity constraints during
  *training* (as opposed to the inference-time KV cache concerns here).
- [`17-inference-serving-optimization`](../17-inference-serving-optimization/README.md)
  for how KV cache size (as shaped by the choice of MHA/MQA/GQA and window
  size described above) drives request batching, paged/blocked KV cache
  allocation, and eviction policy at the serving-system level — the
  systems-level consumer of everything discussed in this document.

---

## 9. Summary Reference

| Formula / quantity | Expression |
|---|---|
| Scaled dot-product attention | `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V` |
| Naive attention compute (per head) | `O(n² d_k)` |
| Naive attention memory (score matrix, per head) | `O(n²)` |
| Online softmax rescale factor | `exp(m_old - m_new)` |
| FlashAttention peak memory | `O(n)` (vs. `O(n²)` naive) |
| MHA KV cache size | `2 · n_layers · H · n · d_k · bytes` |
| MQA KV cache size | `2 · n_layers · 1 · n · d_k · bytes` (= MHA / `H`) |
| GQA KV cache size | `2 · n_layers · G · n · d_k · bytes` (= MHA × `G/H`) |
| Sliding window compute (per layer) | `O(n · W · d_k)` |
| Sliding window KV cache (per layer) | `O(W)` (bounded, independent of `n`) |
| Effective receptive field, `L` sliding-window layers | `≈ L · W` |
