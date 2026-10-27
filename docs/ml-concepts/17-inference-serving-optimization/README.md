# Inference-Time / Serving Optimization for LLMs

Training an LLM is a one-time (or occasional) cost. Serving it is not — every user
request re-runs the forward pass, and a production deployment might field thousands
of concurrent requests per second, indefinitely. The economics of an LLM product are
therefore dominated not by training FLOPs but by **inference cost per token**, and
inference cost per token is dominated by how well the serving system exploits GPU
compute and memory bandwidth. This document covers the core techniques that modern
serving stacks (vLLM, TGI, TensorRT-LLM) use to close the gap between naive
autoregressive decoding and hardware-efficient serving.

---

## 1. Why LLM Inference Is Uniquely Awkward to Serve

Most deep learning inference workloads (image classification, a single BERT forward
pass, an embedding lookup) are **one-shot**: you feed an input, you get an output in
a single forward pass, and the whole thing can be batched trivially because every
request does the same fixed amount of work.

LLM text generation is different because it is **autoregressive**: to produce a
sequence of output tokens, the model must generate them one at a time, and each new
token depends on every token that came before it (the prompt plus everything
generated so far).

```
Step 1: [prompt tokens]                          -> predict token 1
Step 2: [prompt tokens, token 1]                  -> predict token 2
Step 3: [prompt tokens, token 1, token 2]         -> predict token 3
Step 4: [prompt tokens, token 1, token 2, token 3] -> predict token 4
...
```

Each of those steps is, mechanically, a full forward pass through every transformer
layer. The self-attention operation inside each layer computes:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

where Q, K, V are projections of the *entire* sequence seen so far. If you implement
this the naive way — recompute Q, K, and V for **every token in the sequence, at
every single decoding step** — then generating the n-th token requires attending
over all n previous tokens, and you redo that projection work for tokens 1..n-1 that
you already projected at the previous step. Total work to generate a sequence of
length N from scratch at each step is:

```
sum_{n=1}^{N} O(n) = O(N^2)
```

per layer, per generated sequence — quadratic in sequence length, and almost all of
it is *redundant* recomputation of K and V vectors for tokens whose values never
change once computed (a token's Key and Value projections depend only on that
token's own hidden state at that layer, which is fixed once the token has been
processed forward through the network). This is the first problem that any serving
system must solve, and it is solved almost universally by the KV cache (Section 2).

Beyond the algorithmic redundancy, LLM inference has three other structural
awkwardnesses that don't show up in typical inference workloads:

- **Unknown, variable-length output.** You don't know how many tokens a request will
  generate until it emits an end-of-sequence token or hits a length cap. This makes
  static resource planning (how much memory to reserve, how long a batch will run)
  fundamentally guesswork.
- **Two very different compute regimes in one request.** Processing the prompt
  (prefill) and generating tokens (decode) have opposite performance
  characteristics — see Section 3.
- **Enormous, awkwardly-shaped state.** The KV cache for a single long-context
  request can be gigabytes, and it has to live in GPU memory for the entire duration
  of that request, competing with the memory needed for model weights and for other
  concurrent requests' caches.

Every technique below exists to attack one of these three problems.

---

## 2. The KV Cache

### The idea

Since a token's Key and Value projections at a given layer never change after being
computed, we can **cache** them instead of recomputing them. On each decoding step,
instead of recomputing Q, K, V for the full sequence:

- Compute Q, K, V for **only the newly generated token**.
- Append the new token's K and V vectors to a per-layer cache.
- Compute attention using the new token's Q against **all cached K/V** (old + new).
- Discard Q after this step (Q for old tokens is never needed again).

```
                     ┌────────────────────────────────────────┐
                     │        KV Cache (per layer)             │
                     │  K: [k_1, k_2, k_3, ..., k_{n-1}]        │
                     │  V: [v_1, v_2, v_3, ..., v_{n-1}]        │
                     └────────────────────────────────────────┘
                                       │  append
                                       ▼
new token n  ──► project Q_n, K_n, V_n ──► K_cache += K_n, V_cache += V_n
                                       │
                                       ▼
                     attention(Q_n, K_cache, V_cache) ──► output for token n
```

This turns the per-step cost from O(n) (recompute K/V for all n tokens) into O(1)
new K/V projections plus an O(n) attention *read* (dot product of Q_n against n
cached K vectors, which is comparatively cheap and, crucially, does not require
redoing the linear projections themselves). Summed over generating N tokens, total
work drops from O(N^2) projection work down to O(N) projection work (O(N) total new
K/V vectors computed, one set per step) plus O(N^2) attention score computations —
but the attention score computation is a much cheaper operation (a dot product) than
the linear projections it replaces, and more importantly it's the part that was
already unavoidable (you always need attention scores between every query and every
past key, that's what attention *is*). The redundant *linear projection* work is
what's eliminated. In practice this is the difference between inference being
usable and unusable — without KV caching, generating a 2,000-token response would
mean the 2000th token's forward pass recomputes projections for all 1,999 prior
tokens, at every layer.

Virtually every production LLM serving system implements a KV cache; it is not
optional, it is the baseline.

### The KV cache's own cost

The KV cache is not free — it trades compute for memory, and at scale that memory
becomes the binding constraint on how many requests you can serve concurrently and
how long a context you can support.

For each token, at each layer, you store one Key vector and one Value vector, each
of dimension `head_dim * num_heads` (i.e., the full hidden dimension used for
attention, since KV cache stores per-head K/V before the output projection). The
total KV cache memory for a batch of sequences is:

```
KV cache memory (bytes) =
    2  ×  num_layers  ×  num_heads  ×  head_dim  ×  seq_len  ×  batch_size  ×  bytes_per_element
    │      │              │             │            │            │              │
    │      │              │             │            │            │              └─ 2 (fp16/bf16) or 4 (fp32) or 1 (int8/fp8)
    │      │              │             │            │            └─ number of concurrent sequences
    │      │              │             │            └─ current sequence length (prompt + generated tokens so far)
    │      │              │             └─ dimension per attention head
    │      │              └─ number of attention heads (K/V heads, if using GQA/MQA — see below)
    │      └─ number of transformer layers
    └─ factor of 2 because we store both K and V
```

**Worked example.** Take a Llama-2-13B-class model: `num_layers = 40`,
`num_heads = 40`, `head_dim = 128` (so hidden size = 5120), running in fp16
(`bytes_per_element = 2`), with `seq_len = 4096` and `batch_size = 1`:

```
KV cache = 2 × 40 × 40 × 128 × 4096 × 1 × 2 bytes
         = 2 × 40 × 40 × 128 × 4096 × 2
         = 3,355,443,200 bytes
         ≈ 3.36 GB   for a single 4K-token sequence
```

That's **3.36 GB of KV cache for one request**, on top of the ~26 GB of fp16 model
weights already resident on the GPU. Now consider serving this model with a batch
of 32 concurrent requests, each with a 4096-token context:

```
KV cache = 3.36 GB × 32 ≈ 107.5 GB
```

That single number already exceeds the memory of a single 80 GB A100/H100 — the KV
cache alone, not the model weights, becomes the memory bottleneck. This is exactly
why long-context serving and high-concurrency serving are hard: **KV cache size
grows linearly in both sequence length and batch size**, and it competes directly
with the fixed cost of the model weights for the same GPU HBM.

This is also why several architectural mitigations exist purely to shrink this
formula:

- **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** reduce the
  effective `num_heads` term for K/V (sharing K/V heads across multiple Q heads),
  cutting KV cache size by up to `num_query_heads / num_kv_heads`×  without changing
  `head_dim` or `num_layers`. This is why most modern serving-optimized models
  (Llama 3, Mistral, Qwen2, etc.) use GQA.
  - **Sliding-window attention** caps the effective `seq_len` term by only keeping
  the most recent W tokens' K/V in the cache.
- **KV cache quantization** (int8, fp8, even int4 for K/V specifically) reduces
  `bytes_per_element`, trading some numerical precision for cache size — often a
  good trade because attention is somewhat tolerant of reduced K/V precision.
- **Multi-head Latent Attention (MLA)**, used in DeepSeek-V2/V3, compresses K/V into
  a much lower-dimensional latent vector before caching, reconstructing full K/V
  on the fly, which can cut KV cache size by an order of magnitude versus GQA at
  comparable model quality.

All of these are attempts to bend the same formula above, either by shrinking one of
its multiplicative terms or by changing which term (`num_heads`, effectively)
dominates.

---

## 3. Prefill vs. Decode: Two Phases, Two Performance Regimes

A single inference request actually runs through two qualitatively different
compute phases.

### Prefill

When a request arrives, the entire prompt (all P input tokens) is known up front.
The model can process **all P tokens in a single parallel forward pass** — compute
Q, K, V for all P positions at once, compute the full P×P (causal-masked) attention
matrix, and populate the KV cache for all P tokens in one shot. This is a large
matrix-multiply-heavy operation:

- **Compute-bound.** The GPU is doing large, dense GEMMs (the batch/sequence
  dimension is P, which is typically hundreds to thousands of tokens), so it can
  achieve high arithmetic intensity and saturate the GPU's FLOPs.
- **High GPU utilization.** Tensor cores stay busy; this phase is close to
  peak-FLOPs throughput for the hardware.
- Cost scales roughly with `O(P^2)` for the attention part (one-time, not repeated)
  and `O(P)` for everything else, but it all happens once, up front, per request.

### Decode

After prefill, the model generates the response one token at a time. On each decode
step:

- Only **one new token's worth of Q/K/V** is computed (a tiny matrix-vector
  operation per layer, not a matrix-matrix operation).
- To compute attention for that one token, the GPU must **read the entire KV cache
  from HBM** (all previously cached K and V, potentially thousands of tokens) — and
  it must also read **all of the model's weight matrices** from HBM to do the
  linear projections and MLP for this one token.
- The FLOPs being performed are tiny relative to the bytes being moved. This makes
  decode **memory-bandwidth-bound**, not compute-bound: the GPU's tensor cores sit
  mostly idle while it waits for weights and KV cache to stream in from memory.
- If decode is run for a **single sequence at a time**, GPU utilization is
  abysmal — often single-digit percent of peak FLOPs — because the actual
  arithmetic work per step is minuscule compared to the time spent waiting on
  memory bandwidth.

This is the core insight that motivates batching for decode: since decode is
memory-bandwidth-bound and the dominant memory traffic (streaming the model's
weights from HBM) is *shared* across all sequences in a batch, increasing the batch
size lets you amortize that weight-loading cost over more tokens generated per pass,
turning idle tensor-core cycles into productive work — up to the point where the KV
cache reads (which do *not* amortize — each sequence has its own K/V) or compute
itself becomes the bottleneck instead.

```
PREFILL                                    DECODE
────────────────────────────               ────────────────────────────
[prompt: P tokens, parallel]                [1 token at a time, sequential]
one big GEMM per layer                      one tiny GEMV per layer, per step
compute-bound                               memory-bandwidth-bound
GPU util: high (~peak FLOPs)                GPU util (batch=1): low (single digits %)
happens once per request                    happens once PER OUTPUT TOKEN
```

This split is also why serving systems report two separate latency metrics: **TTFT
(time to first token)**, dominated by prefill latency, and **TPOT / ITL (time per
output token / inter-token latency)**, dominated by decode throughput. Optimizing
one does not automatically optimize the other, and some techniques trade one for
the other (e.g., chunked prefill, discussed below, sacrifices a bit of prefill
efficiency to reduce its interference with ongoing decode latency).

---

## 4. Static Batching vs. Continuous (In-Flight) Batching

### Static / naive batching

The obvious way to batch LLM requests: collect a batch of B requests, run prefill
for all of them, then run decode steps together, advancing all B sequences by one
token per step — **until every sequence in the batch has finished** (hit EOS or its
max-token limit). Only then is a new batch formed and started.

The problem: different requests in a batch almost never finish at the same length.
One user asks "what's 2+2?" (finishes in 3 tokens); another asks for a 1500-word
essay (finishes in ~2000 tokens). In static batching, the short request's GPU slot
sits **idle** (or is padded and wastefully computed on) for the remaining ~1997
decode steps, waiting for the longest sequence in the batch to finish before the
GPU can be freed for new work.

```
STATIC BATCHING (batch size = 4)

time ──────────────────────────────────────────────────────────────►

req A  [prefill][decode][decode][decode][DONE]···················
req B  [prefill][decode][decode][decode][decode][decode][decode][DONE]
req C  [prefill][decode][DONE]·····································
req D  [prefill][decode][decode][decode][decode][DONE]·············
                                                                 ▲
                                          batch can't start next batch
                                          until the LAST one (B) finishes
                                          '·' = GPU slot allocated but
                                                idle / wasted for that seq

       └── new batch (E, F, G, H) only starts here ──►
```

Slots A, C, D sit wasted for a large fraction of the batch's total decode time.
Effective GPU throughput can be a small fraction of theoretical peak, and the waste
gets worse as request length variance increases — exactly the situation with
real-world chat traffic, where output length is highly unpredictable.

### Continuous batching (in-flight batching)

Continuous batching (the term vLLM and the broader community use; NVIDIA's
TensorRT-LLM/Triton documentation calls the same idea **in-flight batching**)
removes the "wait for the whole batch" constraint entirely. The scheduler operates
at the granularity of a single decode **step**, not a whole batch lifetime:

- After every decode step, check which sequences in the current batch have
  finished.
- Immediately evict finished sequences and free their KV cache slots.
- Immediately admit new, waiting requests into the now-available batch slots — a
  newly admitted request first does its prefill (which can itself be scheduled
  alongside ongoing decode steps, see "chunked prefill" below), then joins the
  decode loop.
- The GPU's batch dimension is kept as full as possible at every single step,
  continuously, rather than only at batch boundaries.

```
CONTINUOUS BATCHING

time ──────────────────────────────────────────────────────────────►

slot 1  [pre A][dec A][dec A][dec A][DONE]  [pre E][dec E][dec E]...
slot 2  [pre B][dec B][dec B][dec B][dec B][dec B][dec B][DONE][pre F]...
slot 3  [pre C][dec C][DONE]  [pre G][dec G][dec G][dec G]...
slot 4  [pre D][dec D][dec D][dec D][dec D][DONE]  [pre H][dec H]...
                     ▲                    ▲                ▲
              new request slotted   new request slotted   new request slotted
              in the instant a      in the instant a      in the instant a
              slot frees up         slot frees up         slot frees up

        No idle slots. Every step, the GPU processes a full batch of
        (mixed prefill + decode) work. Throughput ≈ hardware limit.
```

Because request arrival and completion are now decoupled from any notion of "batch
boundary," this is often called **in-flight batching**: requests are batched
in-flight, dynamically, rather than in fixed cohorts. This is widely cited as **the
single largest throughput improvement** in modern LLM serving — commonly reported
as multi-x (often 10x-plus under realistic, highly variable-length traffic)
improvements in requests-served-per-second over static batching at the same
latency SLO, precisely because it eliminates the structural GPU idling that static
batching cannot avoid.

A related refinement, **chunked prefill** (also called "prefill-decode fusion" or
"piggybacking"), splits a long prompt's prefill into smaller chunks and interleaves
those chunks with ongoing decode steps for other sequences in the same batch step,
so that admitting a new long-prompt request doesn't stall the decode latency of
sequences already in flight. This further smooths the interaction between the
compute-bound prefill phase and the latency-sensitive decode phase when they must
coexist in the same continuously-batched scheduler.

Continuous batching does introduce a new requirement, though: since sequences are
now entering and leaving the batch unpredictably and at different lengths, the
serving system needs a way to manage per-sequence KV cache memory that doesn't
require pre-reserving a worst-case-length contiguous block per slot. That
requirement is exactly what motivated PagedAttention.

---

## 5. PagedAttention

### The problem: fragmentation from contiguous KV cache allocation

Before PagedAttention (introduced by the vLLM project, Kwon et al., 2023), the
standard implementation allocated KV cache memory the "obvious" way: for each
request, reserve one **contiguous** block of GPU memory sized for the *maximum
possible* sequence length that request could reach (e.g., the model's max context
length, or a configured max-tokens setting).

This causes severe memory waste for several distinct reasons, all versions of
memory fragmentation:

- **Internal fragmentation.** Almost no request actually generates all the way out
  to the max length; if you reserve for 4096 tokens but the sequence finishes at
  200, you've wasted ~95% of that reservation for the entire lifetime of the
  request, and none of that reserved-but-unused memory can be lent to other
  requests in the meantime.
- **Reservation for the unknown.** Because you don't know a request's eventual
  length in advance (Section 1), you're forced to over-reserve defensively — you
  can't reserve "as needed" with plain contiguous allocation, because a contiguous
  block can't be grown in place if something else already occupies the adjacent
  memory.
- **External fragmentation.** As requests of different reserved sizes come and go,
  the GPU memory becomes checkerboarded with holes too small individually to fit a
  new request's max-length reservation, even if the *total* free memory would be
  sufficient.

The net effect reported in the vLLM paper: naive contiguous KV cache management
wasted on the order of 60-80% of allocated KV cache memory in realistic serving
scenarios — which directly caps how many concurrent sequences (i.e., how large a
batch) the system can support, which directly caps throughput via the continuous
batching mechanism described above. Fragmented, wasted KV cache memory is wasted
batch capacity.

### The solution: OS-style paging for the KV cache

PagedAttention borrows the classic solution operating systems use for exactly this
class of problem — external/internal fragmentation in memory allocation — namely
**virtual memory paging**:

- The KV cache is divided into fixed-size **blocks** (pages), each holding K/V for
  a small, fixed number of tokens (e.g., 16 tokens per block) across all layers/heads.
- These physical blocks are allocated **non-contiguously** anywhere in GPU memory,
  from a shared pool, on demand — one block at a time, only as a sequence actually
  grows enough to need another block.
- Each sequence has a **block table** (analogous to an OS page table) that maps its
  *logical* token positions (0, 1, 2, 3, ... in generation order) to the *physical*
  block addresses that actually store that range of tokens' K/V, wherever those
  blocks happen to physically live in GPU memory.
- The attention kernel is modified to gather K/V from these scattered physical
  blocks according to the block table when computing attention — instead of
  assuming K/V for a sequence are one contiguous memory span.

```
PAGEDATTENTION — LOGICAL SEQUENCE  →  BLOCK TABLE  →  PHYSICAL KV CACHE

Logical view (sequence, in generation order):
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │   (block size = 4 tokens/block)
  └────┴────┴────┴────┴────┴────┴────┴────┘
    logical block 0        logical block 1

Per-sequence block table:
  logical block 0  ──►  physical block  7
  logical block 1  ──►  physical block  2

Physical GPU memory (shared pool, blocks owned by many sequences, any order):
  ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
  │ blk 0 │ blk 1 │ blk 2 │ blk 3 │ blk 4 │ blk 5 │ blk 6 │ blk 7 │
  │ (seq  │ (free)│ (THIS │ (seq  │ (free)│ (seq  │ (seq  │ (THIS │
  │  X)   │       │ seq,  │  Y)   │       │  Z)   │  X)   │ seq,  │
  │       │       │ lblk1)│       │       │       │       │ lblk0)│
  └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

  New token needs a new block?  → grab ANY free physical block from the
  pool (say, block 4) and append it to this sequence's block table.
  No contiguity required. No pre-reservation of a max-length span.
```

This design gives three concrete wins:

1. **Near-zero internal fragmentation.** Memory is allocated in small block
   increments (only the last, partially-filled block per sequence wastes any
   space — bounded by `block_size - 1` tokens, not by however far from max-length
   the sequence stopped). A sequence that finishes at 200 tokens only ever
   consumed ~200/16 ≈ 13 blocks, not a reservation for 4096.
2. **No external fragmentation.** Since blocks are fixed-size and any free block
   can satisfy any sequence's next allocation, there's no "hole too small to use"
   problem — free blocks are fungible.
3. **Memory sharing via shared block tables.** Multiple logical sequences can point
   to the *same physical block* when they share a common prefix — e.g., **beam
   search** (multiple candidate beams sharing the same prompt and early generated
   tokens before diverging) or **parallel sampling** (multiple samples from the
   same prompt) can have their block tables point at identical physical blocks for
   the shared portion, using copy-on-write semantics only once a sequence starts
   diverging from the others. This turns what used to be N full copies of a shared
   prefix's KV cache into effectively one copy, referenced N times — a direct
   generalization of the idea behind prefix caching (Section 7).

PagedAttention's memory-efficiency win is what lets a serving system actually run
continuous batching (Section 4) at high concurrency: without it, per-sequence KV
cache over-reservation caps concurrent batch size low enough that continuous
batching has little slack to work with; with it, GPU memory utilization for KV
cache approaches its theoretical requirement, so far more concurrent sequences fit,
and the batch stays fuller for longer.

---

## 6. Speculative Decoding

### The problem it targets

Decode is memory-bandwidth-bound (Section 3): generating one token means streaming
the entire model's weights (and the growing KV cache) from HBM, largely regardless
of how much actual arithmetic is performed per step. This has an interesting
consequence: **verifying K tokens in a single forward pass costs barely more than
verifying 1 token**, because the memory-bandwidth cost (reading the weights once)
is shared across all K token positions being evaluated in that one pass — only the
(comparatively tiny) compute cost scales with K. In other words, the large model is
"latency-insensitive" to how many tokens it evaluates per forward pass, up to the
point where compute, not bandwidth, becomes the bottleneck.

### The technique

Speculative decoding exploits this asymmetry with a two-model setup:

1. A small, fast **draft model** (far fewer parameters, or the same model
   quantized/distilled, or even a rule-based n-gram predictor) autoregressively
   generates a short run of, say, K = 4-8 candidate tokens, cheaply — because it's
   small, its own decode steps are fast.
2. The large **target model** (the model whose output distribution you actually
   want) then runs **one single forward pass** over the draft's K proposed tokens
   *plus* the existing context, in parallel — effectively treating the K draft
   tokens as if they were a mini "prefill" chunk. This one pass produces the target
   model's own next-token probability distribution at each of the K positions,
   as if it had generated them step by step.
3. **Verification, left to right**: for each draft token position i, check whether
   the draft token matches the token the target model would have sampled from its
   own distribution at that position (for greedy decoding, this is exact-match; for
   sampled decoding, a rejection-sampling scheme is used that still guarantees the
   final samples are drawn from the target model's exact distribution). Accept all
   draft tokens up to and including the last matching position.
4. At the **first mismatch**, discard the draft's token and all draft tokens after
   it, and instead take the target model's own correct token at that position
   (which you already have for free, since the target model's single forward pass
   computed a distribution at every position, not just the matching ones).
5. Resume drafting from that corrected position for the next round.

```
draft model proposes:     [t1][t2][t3][t4][t5]     (fast, small model, sequential)
target model verifies
in ONE forward pass:      [t1][t2][t3][t4'][t5']    (target's actual distribution
                              ✓   ✓   ✓   ✗    (unused, discarded)
                                          │
                                          └─ mismatch at position 4:
                                             accept t1,t2,t3 from the draft,
                                             take target's OWN token 4 (call it t4*)
                                             for free from the same forward pass,
                                             then resume drafting after t4*

net effect this round: 4 tokens produced (t1,t2,t3,t4*) for the cost of
ONE target-model forward pass (roughly the cost of producing 1 token normally)
plus the cheap draft-model generation of 5 tokens.
```

### Why this speeds up wall-clock time without changing output quality

The critical property: the acceptance criterion is an **exact match against what
the target model itself would have produced** (or, for sampled decoding, a
mathematically exact rejection-sampling correction) — never an approximation, and
never a fallback to the draft model's own (weaker) distribution. Whenever a draft
token is rejected, the *target model's own correct token* is substituted in its
place, straight from the same verification pass. So the sequence of tokens that
actually gets emitted is **provably distributed identically** to what the target
model alone, run token-by-token in the ordinary way, would have produced. Nothing
about the target model's output distribution is changed, approximated, or
degraded — speculative decoding is a pure latency optimization, not a quality
trade-off.

The wall-clock win comes entirely from exploiting the bandwidth-bound nature of
decode: instead of paying one full memory-bandwidth-bound forward pass of the
*target* model per output token, you pay one full forward pass of the target model
per **accepted run of tokens** (which, with a reasonably well-matched draft model,
averages several tokens per round), plus the comparatively cheap cost of the small
draft model generating its candidates. When the draft model's acceptance rate is
high (draft and target models agree often — true when the draft is a smaller
version of the same model family, or trained/distilled specifically to mimic the
target), the number of expensive target-model forward passes needed to produce N
final output tokens can be a fraction of N, directly cutting decode wall-clock
time. Overall speedups of roughly 2-3x on decode-bound workloads are commonly
reported, and vary with how well the draft model predicts the target.

---

## 7. Prompt / Prefix Caching

Many real workloads repeatedly send requests that share a **common prefix**:

- A fixed **system prompt** prepended to every request in an application.
- **Few-shot examples** included in every prompt for a given task.
- A long shared **document or codebase context** that many different user
  questions are asked against.
- Multi-turn **conversation history**, where each new turn re-sends the entire
  prior conversation as part of the prompt.

Without prefix caching, every one of these requests pays the full prefill cost
(Section 3) for the shared portion, every single time, even though the K/V values
for those shared tokens are *identical* across requests (attention's K/V for a
token depend only on that token and everything before it in the sequence, and if
the prefix is byte-identical across requests, its K/V are byte-identical too).

**Prefix caching** stores the KV cache produced for a prefix the first time it's
computed, keyed (typically) by a hash of the token sequence, and on subsequent
requests that share the same prefix, the serving system **reuses the cached K/V
directly** instead of recomputing them — the request's prefill work is reduced to
just the *new*, non-shared suffix (e.g., just the user's actual question, if the
system prompt and few-shot examples are cached).

```
Request 1:  [ system prompt (cached KV) ][ few-shot examples (cached KV) ][ Q1 ]
                    ↑ compute once                ↑ compute once            ↑ compute (new)

Request 2:  [ system prompt (CACHE HIT) ][ few-shot examples (CACHE HIT) ][ Q2 ]
                    ↑ reused, no recompute         ↑ reused, no recompute    ↑ compute (new)

Request 3:  [ system prompt (CACHE HIT) ][ few-shot examples (CACHE HIT) ][ Q3 ]
                    ↑ reused, no recompute         ↑ reused, no recompute    ↑ compute (new)
```

This directly attacks TTFT (time to first token) for prefix-heavy workloads —
sometimes cutting prefill work (and thus TTFT) by an order of magnitude when the
shared prefix dominates the total prompt length, as is common with long system
prompts or large few-shot blocks. It composes naturally with PagedAttention's
block-based KV cache: since KV cache already lives in fixed-size, individually
addressable blocks with per-sequence block tables (Section 5), caching a prefix is
just a matter of letting multiple sequences' block tables point at the **same**
physical blocks for the shared prefix portion, with copy-on-write once/if a
sequence needs to diverge — architecturally, prefix caching and PagedAttention's
beam-search/parallel-sampling sharing are the same mechanism applied at a longer
time horizon (across requests, not just within one request's lifetime).

Prefix caching does require eviction policy design (typically LRU-style, since the
cache pool is finite GPU memory shared with the KV cache of currently-active
requests), and it only helps to the extent that traffic actually repeats prefixes —
for fully unique, non-overlapping prompts it provides no benefit — but for
system-prompt-heavy production traffic (which is most production LLM traffic) it
is a substantial and nearly-free win once the block-based KV cache infrastructure
exists to support it.

---

## 8. Putting It Together: Real Systems

No single technique above is sufficient on its own; production serving throughput
comes from combining them. Three systems dominate the current landscape, each with
a different origin and emphasis, though their feature sets have converged
substantially over time:

- **vLLM** — originated PagedAttention (Section 5) and is generally considered the
  reference implementation for it; also implements continuous batching, prefix
  caching (via its block-based KV cache — the same block-table mechanism that
  backs PagedAttention supports both), chunked prefill, and speculative decoding.
  Open source, widely used as the default high-throughput serving engine for
  Hugging Face-format models.
- **TGI (Text Generation Inference)** — Hugging Face's production serving
  toolkit; implements continuous batching, KV-cache-aware scheduling, and
  quantization support, tightly integrated with the Hugging Face model ecosystem
  and Hub.
- **TensorRT-LLM** — NVIDIA's serving engine, built on TensorRT; implements
  in-flight batching (its name for continuous batching), paged KV cache management,
  speculative decoding, and aggressively exploits NVIDIA-specific kernel fusion,
  custom attention kernels, and quantization formats (fp8, int4/int8 with
  NVIDIA-specific calibration) tuned per GPU architecture (Ampere, Hopper,
  Blackwell) — typically the highest-throughput option on NVIDIA hardware at the
  cost of being NVIDIA-specific and requiring an explicit model-compilation step
  (building a TensorRT engine) rather than TGI/vLLM's more dynamic, load-and-run
  model support.

All three sit on the same conceptual foundation described in this document: use a
KV cache to avoid redundant recomputation (Section 2), understand that decode is
memory-bandwidth-bound so batch aggressively via continuous/in-flight batching
(Sections 3-4), manage the resulting KV cache memory efficiently via block-based
paging to keep batches full (Section 5), and layer additional wins on top —
speculative decoding to shrink the number of expensive target-model passes
(Section 6) and prefix caching to avoid recomputing shared context (Section 7) —
where the workload characteristics make them pay off. Choosing between them in
practice is less about which techniques they support (the feature sets largely
overlap) and more about deployment fit: hardware target, model format ecosystem,
operational maturity, and how much compilation/build complexity is acceptable in
exchange for peak throughput.

---

## Summary Table

| Technique | Problem solved | Primary win |
|---|---|---|
| KV cache | Redundant O(n²) recomputation of K/V across decode steps | O(n) decode cost instead of O(n²) |
| Prefill/decode split awareness | Two very different compute regimes in one request | Correct scheduling, separate TTFT/TPOT optimization |
| Continuous (in-flight) batching | Static batching idles on early-finishing sequences | GPU batch dimension stays full; largest single throughput win |
| PagedAttention | Contiguous KV allocation wastes memory to fragmentation | Near-zero fragmentation; more concurrent sequences fit in memory |
| Speculative decoding | Decode is memory-bandwidth-bound per step regardless of tokens verified | Multiple tokens emitted per expensive target-model pass, same output distribution |
| Prefix/prompt caching | Shared prefixes recomputed on every request | Reused KV cache for shared context; large TTFT reduction |
