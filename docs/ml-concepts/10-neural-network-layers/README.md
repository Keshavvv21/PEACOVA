# The Layers Inside a Modern Transformer

This document walks through what is physically inside a decoder-only
transformer LLM (GPT-style, Llama-style), layer by layer, with the actual
tensor shapes and math at each step. The goal is that after reading this you
could, in principle, implement a forward pass from scratch and know exactly
what every matrix multiply is doing and why it exists.

For the internals of the attention *optimizations* used in production
serving — Multi-Query Attention (MQA), Grouped-Query Attention (GQA),
FlashAttention, sliding-window attention — see
`docs/ml-concepts/18-attention-optimizations/README.md`. This document covers
the baseline architecture; that one covers making it fast and cheap at scale.

---

## 1. Overview: The Shape of the Whole Network

A decoder-only transformer is, structurally, three things stacked together:

```
Token IDs  ->  [ Embedding Layer ]  ->  [ Decoder Block ] x N  ->  [ Output Layer ]  ->  Next-token probabilities
```

1. An **embedding layer** that turns discrete token IDs into dense vectors
   and injects positional information.
2. A stack of `N` **identical decoder blocks** (also called transformer
   layers), each containing a self-attention sublayer and a feed-forward
   sublayer, wrapped in normalization and residual connections. Llama 2 70B
   has `N = 80` of these; GPT-3 175B has `N = 96`; a small model might have
   `N = 12` to `32`.
3. An **output (unembedding) layer** that projects back from the model's
   internal vector space to a probability distribution over the vocabulary.

### 1.1 The one tensor shape that flows through everything

The single most important fact to internalize: from the moment tokens are
embedded until the moment they reach the final output projection, the
activation tensor has a **constant shape**:

```
x.shape == (batch_size, sequence_length, d_model)
```

Common notation: `B` = batch size, `T` or `n` = sequence length (number of
tokens processed so far / context length), `d_model` (also written `d`,
`d_h`, or `h` in some papers) = the model's hidden dimension — e.g. 4096 for
Llama 2 7B, 8192 for Llama 2 70B, 12288 for GPT-3 175B.

Every decoder block takes in a `(B, T, d_model)` tensor and returns a
`(B, T, d_model)` tensor. This is what makes the blocks stackable and
"identical in shape" — you can chain arbitrarily many of them, or even route
different tokens through different numbers of them (as in early-exit /
mixture-of-depths schemes), because the interface never changes. The model's
depth (`N`) and width (`d_model`) are the two primary knobs that determine
capacity, and everything downstream (number of attention heads, FFN hidden
size) is typically defined as a function of `d_model`.

Keep this shape invariant in your head for every section below — every
sublayer's job is to read a `(B, T, d_model)` tensor, do some computation
that lets tokens exchange information (attention) or transform their own
representation (FFN), and write out a `(B, T, d_model)` tensor of the exact
same shape via a residual add.

---

## 2. The Embedding Layer

The embedding layer's job is to turn a sequence of integers (token IDs) into
a sequence of vectors the rest of the network can do linear algebra on, and
to inject information about *where* each token sits in the sequence (since
attention itself is permutation-invariant and has no inherent notion of
order).

### 2.1 Token embeddings: a lookup table

The token embedding is a single learned weight matrix:

```
W_E ∈ R^(vocab_size × d_model)
```

For example, Llama 2's vocabulary is 32,000 tokens with `d_model = 4096`,
giving `W_E` shape `(32000, 4096)` — about 131M parameters, just for this
one table.

Given an input sequence of token IDs `tokens ∈ Z^(B, T)` (integers in
`[0, vocab_size)`), the embedding operation is a **row gather**, not a matrix
multiply in the traditional sense:

```
x_embed[b, t, :] = W_E[tokens[b, t], :]
```

Equivalently, if you one-hot encode each token ID into a `vocab_size`-length
vector `onehot(tokens[b,t]) ∈ {0,1}^vocab_size`, then
`x_embed[b,t,:] = onehot(tokens[b,t]) @ W_E`, which is why it's called an
"embedding matrix multiply" in some texts even though in practice it's
implemented as an indexed lookup for efficiency (no need to materialize a
sparse one-hot vector and do a real matmul).

Output shape: `(B, T, d_model)` — token IDs have become dense vectors.

### 2.2 Positional information

Self-attention (Section 3) computes a weighted sum over all positions with
weights that depend only on content similarity (`q_i · k_j`) — if you
shuffled the input sequence, plain attention would produce the same set of
output vectors, just permuted. That's a problem: "the dog bit the man" and
"the man bit the dog" would look identical to raw attention. Positional
encoding is what breaks that symmetry. There are three approaches worth
understanding, in the order the field adopted and then moved away from them.

#### 2.2.1 Learned positional embeddings

The simplest approach (used in GPT-2, BERT, the original GPT): a second
lookup table, this time indexed by *position* rather than token identity:

```
W_P ∈ R^(max_seq_len × d_model)
```

```
x[b, t, :] = x_embed[b, t, :] + W_P[t, :]
```

Position `t`'s embedding is just added elementwise to the token embedding.
`W_P` is a trained parameter matrix, learned via backprop like everything
else. The major limitation: `max_seq_len` is baked into the model's weights
at training time. Position 4097 simply has no row in `W_P` if the model was
trained with `max_seq_len = 4096`, so the model cannot generalize to longer
sequences without retraining or interpolating the table.

#### 2.2.2 Sinusoidal positional encodings

The original "Attention Is All You Need" Transformer used a fixed
(non-learned) function of position instead of a trained table:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

where `pos` is the token's position (0, 1, 2, ...) and `i` indexes pairs of
dimensions (`i = 0 ... d_model/2 - 1`). Each dimension pair oscillates at a
different frequency: low `i` (early dimensions) oscillate fast (short
wavelength), high `i` (late dimensions) oscillate slowly (long wavelength,
up to `2π · 10000`). This gives every position a unique fixed vector, and
because sine/cosine are defined for any real-valued input, the encoding
extends to positions never seen during training (though the model's ability
to *use* those positions well is a separate matter). A useful property: for
any fixed offset `k`, `PE(pos+k)` is a linear function of `PE(pos)` (a
rotation in each 2D sub-plane), which the paper argued would make it easy for
attention to learn to attend by relative position.

Both 2.2.1 and 2.2.2 share a structural weakness: position information is
**added to the content vector once, at the input**, and then has to survive
being linearly mixed through every subsequent layer's Q/K/V projections. By
layer 40 of an 80-layer network, the positional signal has been through 40
rounds of matrix multiplication and could in principle be attenuated or
distorted. This motivated a different idea: don't add position to the
*content*, inject it directly into the *attention computation itself*.

#### 2.2.3 RoPE — Rotary Position Embeddings

RoPE (used in Llama, Mistral, GPT-NeoX, Qwen, and most modern open LLMs) is
applied at every attention layer, not once at the input — and instead of
being *added* to Q and K vectors, it *rotates* them.

**The core intuition**: take a query vector `q` and a key vector `k`, and
instead of encoding "this token is at position 5" as an additive offset,
encode it as a rotation angle. Pair up each vector's dimensions into 2D
planes `(x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})`, and for a token at
position `m`, rotate each 2D pair by an angle `m · θ_i` (a different base
frequency `θ_i` per pair, same frequency schedule idea as sinusoidal PE):

```
θ_i = 10000^(-2i / d_k),   i = 0 ... d_k/2 - 1

RoPE(x, m)_pair_i = [ x_{2i}   * cos(m θ_i) - x_{2i+1} * sin(m θ_i) ]
                     [ x_{2i}   * sin(m θ_i) + x_{2i+1} * cos(m θ_i) ]
```

This is applied to `q` at its position and to `k` at its position, *before*
the dot product `q · k` is taken (see Section 3). Why does this work? A 2D
rotation matrix `R(θ)` has the property `R(θ_1)^T R(θ_2) = R(θ_2 - θ_1)`.
So when you compute the dot product between a rotated query at position `m`
and a rotated key at position `n`:

```
(R(mθ) q) · (R(nθ) k) = q^T R(mθ)^T R(nθ) k = q^T R((n-m)θ) k
```

The absolute positions `m` and `n` cancel out, and what survives inside the
attention score is a function of the **relative** distance `(n - m)` only.
This is exactly the property you want for language: "the previous token" or
"3 tokens back" should mean the same thing whether you're at position 10 or
position 10,000. RoPE gets relative-position sensitivity "for free" as an
algebraic consequence of rotation, without ever materializing a relative
position matrix, and — because it's applied fresh at every layer to the
current Q/K rather than added once at the input — the positional signal
doesn't have to survive being carried unchanged through the residual stream.
It also composes cleanly with length-extrapolation tricks (NTK-aware
scaling, YaRN, position interpolation) that rescale `θ_i` post-hoc to stretch
a model's effective context window without retraining from scratch.

Note RoPE has no learned parameters at all — `θ_i` is a fixed geometric
schedule, exactly analogous to sinusoidal PE's frequency schedule, just
applied as a rotation to Q/K per-layer instead of an addition to the
embedding once.

---

## 3. Self-Attention in Full Detail

This is the sublayer that lets tokens exchange information with each other —
every other sublayer in the network (FFN, norm) operates on each token's
vector independently. Attention is the *only* place where token `t`'s output
can depend on token `t' ≠ t`'s input.

### 3.1 Linear projections: Q, K, V

Given the input to the attention sublayer `x ∈ R^(B, T, d_model)` (after
normalization — see Section 6), three separate learned weight matrices
project `x` into query, key, and value spaces:

```
W_Q, W_K, W_V ∈ R^(d_model × d_model)

Q = x W_Q     shape (B, T, d_model)
K = x W_K     shape (B, T, d_model)
V = x W_V     shape (B, T, d_model)
```

Intuition for the three roles: for each token, `Q` ("what am I looking
for?") is compared against every token's `K` ("what do I contain, as a
label/key?") to compute a relevance score, and that score is used to weight
a combination of every token's `V` ("what do I actually contribute, as
content?"). The Q/K comparison determines *how much* attention flows from
token to token; V determines *what* gets transferred.

### 3.2 Splitting into heads

Rather than run one attention computation over the full `d_model` dimension,
`Q`, `K`, `V` are each reshaped/split along the feature dimension into `H`
heads of size `d_k = d_model / H`:

```
Q.reshape(B, T, H, d_k).transpose -> (B, H, T, d_k)
K.reshape(B, T, H, d_k).transpose -> (B, H, T, d_k)
V.reshape(B, T, H, d_k).transpose -> (B, H, T, d_k)
```

For Llama 2 7B: `d_model = 4096`, `H = 32` heads, so `d_k = 128` per head.
No new parameters are introduced by splitting — this is purely a reshape of
the already-projected `Q`/`K`/`V` tensors, computed either as one big
`(d_model, d_model)` projection reshaped afterward, or equivalently as `H`
separate `(d_model, d_k)` matrices concatenated — the two are mathematically
identical.

### 3.3 Scaled dot-product attention

For each head independently (`Q_h, K_h, V_h ∈ R^(B, T, d_k)`, dropping the
batch dimension for clarity so `Q_h, K_h, V_h ∈ R^(T, d_k)`):

```
Attention(Q_h, K_h, V_h) = softmax( (Q_h K_h^T) / sqrt(d_k) + M ) V_h
```

Step by step:

1. **Raw scores**: `S = Q_h K_h^T`, shape `(T, T)`. `S[i,j] = q_i · k_j` —
   how much query token `i` "wants" to attend to key token `j`.
2. **Scale by `1/sqrt(d_k)`**: if `q_i` and `k_j` have components that are
   roughly independent, zero-mean, unit-variance random values, their dot
   product `Σ q_i[d] k_j[d]` has variance that grows linearly with `d_k`. As
   `d_k` gets large, unscaled dot products get large in magnitude, and
   softmax of large-magnitude logits saturates — one logit dominates, the
   rest get gradient ≈ 0. Dividing by `sqrt(d_k)` keeps the variance of the
   scores at approximately 1 regardless of head dimension, keeping softmax
   in a well-conditioned regime.
3. **Add the mask `M`** (causal masking — see 3.4 below), then apply
   **softmax row-wise**: `P[i, :] = softmax(S[i, :] / sqrt(d_k) + M[i, :])`.
   Each row `i` becomes a probability distribution over positions `j` that
   sums to 1 — "how much of my output comes from each other token."
4. **Weighted sum over values**: `O_h = P V_h`, shape `(T, d_k)`. Output
   token `i` is `Σ_j P[i,j] * V_h[j, :]` — a convex combination of value
   vectors.

### 3.4 Causal masking

Decoder-only LLMs are trained to predict the next token given only the
tokens that came before it (autoregressive language modeling). If token `i`
were allowed to attend to token `j > i` (a future token), the model could
"cheat" during training by looking at the very answer it's supposed to
predict — the loss would go to zero without the model learning anything
useful, and at inference time future tokens don't even exist yet since
generation is sequential. Causal masking enforces that token `i`'s output
can only depend on tokens `0 ... i`, never `i+1 ... T-1`.

Mechanically, this is done by adding a mask matrix `M ∈ R^(T, T)` to the raw
scores *before* softmax, where `M[i,j] = 0` for allowed positions (`j <= i`)
and `M[i,j] = -∞` (in practice, a very large negative number like `-1e9`,
since real `-∞` causes NaNs when every value in a row is masked, e.g. an
empty-context edge case) for disallowed positions (`j > i`). Since
`exp(-∞) = 0`, those positions get exactly zero attention weight after
softmax, without needing to change the softmax formula itself.

Concrete example for a 5-token sequence (`T = 5`), where rows are query
position `i` and columns are key position `j`:

```
Mask M (0 = allowed, -inf = blocked):

           j=0    j=1    j=2    j=3    j=4
  i=0  [    0,   -inf,  -inf,  -inf,  -inf  ]
  i=1  [    0,     0,   -inf,  -inf,  -inf  ]
  i=2  [    0,     0,     0,   -inf,  -inf  ]
  i=3  [    0,     0,     0,     0,   -inf  ]
  i=4  [    0,     0,     0,     0,     0   ]
```

This is a lower-triangular structure. Row `i=2` (the third token) can attend
to keys at `j=0,1,2` — itself and everything before it — but positions
`j=3,4` are masked to `-inf`, so after softmax those columns get probability
0 in that row. After adding this mask to `S / sqrt(d_k)` and applying
softmax, the resulting `P` matrix is lower-triangular (all mass in the
upper-right triangle is exactly zero), and `O = P V` guarantees output token
`i` is a function only of `V_0 ... V_i`. This is what makes the architecture
"causal" / autoregressive, and it's also what makes KV-caching correct: since
token `i`'s output never depends on tokens after it, previously computed keys
and values never need to be recomputed as generation proceeds — they're
frozen results, not something that would change if the model glimpsed
tokens generated after them.

(Encoder architectures like BERT use no causal mask — every position can
attend bidirectionally to every other position — because they're trained on
tasks like masked-token infilling rather than left-to-right generation.)

### 3.5 Concatenating heads and the output projection

After each of the `H` heads independently produces `O_h ∈ R^(B, T, d_k)`,
the heads are concatenated back along the feature dimension:

```
O_concat = Concat(O_1, O_2, ..., O_H)   shape (B, T, H * d_k) = (B, T, d_model)
```

and passed through one final learned projection:

```
W_O ∈ R^(d_model × d_model)
MultiHeadAttention(x) = O_concat W_O        shape (B, T, d_model)
```

`W_O` is important, not a formality: it's what lets information from
different heads' subspaces mix together into a single unified representation
before being added back to the residual stream. Without it, the `H`
per-head outputs would just sit in disjoint blocks of the output vector,
unable to combine.

Parameter count for one attention sublayer: `W_Q, W_K, W_V, W_O` are each
`(d_model, d_model)`, so `4 * d_model²` parameters total (plus small bias
terms if the implementation uses them, which many modern LLMs — Llama, PaLM
— omit entirely for training stability and to save memory).

---

## 4. Why Multiple Heads?

A natural question: why split into `H` heads of dimension `d_k = d_model/H`
each, rather than run one attention operation over the full `d_model`
dimension? The total FLOPs for the `QK^T` and `PV` matmuls are actually
roughly the same either way (many small attention computations vs. one big
one costs about the same in raw multiply-adds) — so the motivation is
representational, not computational efficiency.

A single attention operation over all `d_model` dimensions produces exactly
**one** attention pattern (`P` matrix) per query token — one weighting over
which other tokens matter, applied uniformly across the entire representation.
But language has many *simultaneous, independent* kinds of relationships
between tokens: which word a pronoun refers to (coreference), which verb a
subject agrees with (syntax/agreement), which token is positionally adjacent
(local n-gram structure), which distant token shares a topic (semantic/
long-range). These relationships don't share the same "shape" of attention
pattern — coreference resolution wants a very different set of `P[i,:]`
weights than adjacent-token tracking does. Splitting into heads gives the
model `H` independent, differently-parameterized attention patterns per
layer (because each head has its own `W_h^Q, W_h^K`), each free to specialize
on a different type of relationship, and each operating on a lower-
dimensional slice of the representation so no one head has to represent
every relationship type at once. Empirically, when researchers visualize
attention patterns in trained models, individual heads often do show
interpretable specializations — e.g., a head that always attends to the
previous token, a head that tracks the subject of the current clause, a head
that attends to the first token of the sequence (an "attention sink"
pattern) — supporting the idea that multiple heads let different relational
patterns be learned in parallel rather than forcing one attention
computation to represent a blend of all of them at once.

The specific efficiency question of *how many* KV projections you actually
need per query head — i.e., whether every head needs its own full K/V
(standard MHA), all heads can share one K/V (Multi-Query Attention), or
groups of heads share (Grouped-Query Attention) — plus hardware-level
optimizations like FlashAttention that make the attention computation itself
faster without changing its output, are covered in
`docs/ml-concepts/18-attention-optimizations/README.md`. That is purely
about *serving cost and memory bandwidth*; the number of independent
attention *patterns* the model can express (`H`, as discussed above) is
unaffected by which MQA/GQA variant is used.

---

## 5. The Feed-Forward Network (FFN / MLP) Layer

If attention is where tokens exchange information *with each other*, the
FFN is where each token's representation is transformed *independently* —
the exact same FFN weights are applied to every position `t` with no mixing
across positions at all (it is sometimes literally called a
"position-wise feed-forward network" for this reason).

### 5.1 Structure: two linear layers with a nonlinearity between them

```
FFN(x) = W_2 * activation( W_1 * x + b_1 ) + b_2

W_1 ∈ R^(d_model × d_ff)     "up-projection"
W_2 ∈ R^(d_ff × d_model)     "down-projection"
```

`x` here has shape `(B, T, d_model)`; the first matmul expands each
token's vector up to a much wider intermediate dimension `d_ff`, applies a
nonlinearity, then the second matmul projects back down to `d_model`. The
expand-then-contract shape is often described as an hourglass or bottleneck
in reverse — the model "thinks" in a higher-dimensional space in the middle
of the block.

`d_ff` is conventionally **4x `d_model`** in the original Transformer and
GPT-family models (e.g. Llama 2's base variant loosely follows this,
adjusted for SwiGLU's parameter accounting — see below). Why 4x specifically?
It's an empirically-settled convention rather than a theoretical necessity:
enough extra capacity in the intermediate layer to let the network compute
useful nonlinear combinations of input features before compressing back down,
without making the FFN so dominant that attention's relative capacity share
becomes negligible. Some models use different ratios (e.g. ~2.7x with
SwiGLU to keep total parameter count comparable to a 4x ReLU FFN, since
SwiGLU needs three weight matrices instead of two — see 5.3).

### 5.2 Why not ReLU anymore: GELU and SwiGLU

**ReLU**: `ReLU(x) = max(0, x)`. Simple, cheap, and the historical default
for deep nets. But it has a hard corner at 0 — zero gradient for any negative
input ("dying ReLU"), and a discontinuous derivative that doesn't reflect
that most quantities in language modeling should have some graceful
transition rather than an on/off switch.

**GELU** (Gaussian Error Linear Unit), used in GPT-2/GPT-3, BERT:

```
GELU(x) = x * Φ(x)
```

where `Φ(x)` is the standard Gaussian cumulative distribution function
(the probability that a standard normal random variable is ≤ `x`). Commonly
approximated as:

```
GELU(x) ≈ 0.5x * (1 + tanh( sqrt(2/π) * (x + 0.044715 x^3) ))
```

Intuition: rather than a hard cutoff at zero, GELU **weights the input by
how likely it is to be "kept"** — it's smooth everywhere (no kink), roughly
linear for large positive `x`, roughly zero for large negative `x`, and,
notably, non-monotonic — it dips slightly negative just below `x=0` before
flattening out, unlike ReLU which is exactly 0 for all `x<0`. This smoothness
gives better-behaved gradients during training compared to ReLU's hard hinge.

**SwiGLU**, used in Llama, PaLM, Mistral, and most current open-weight LLMs,
goes a step further and makes the *activation itself* gated — instead of
one linear projection into the nonlinearity, it computes two independent
linear projections and lets one gate the other:

```
SwiGLU(x) = ( Swish(x W_gate) ⊙ (x W_up) ) W_down

Swish(z) = z * sigmoid(z)     (a.k.a. SiLU)
⊙  =  elementwise multiplication
```

Here `W_gate, W_up ∈ R^(d_model × d_ff)` and `W_down ∈ R^(d_ff × d_model)` —
**three** weight matrices instead of the plain FFN's two. The `Swish(x
W_gate)` term acts as a learned, per-element, continuous gate (values roughly
in `[0, ∞)` scaled by how "activated" that gate unit is) multiplying the
`x W_up` content term elementwise — the network learns *both* what content to
compute (`W_up`) *and* how much of it to let through per dimension
(`W_gate`), rather than applying the same fixed nonlinear curve to every
unit. This extra multiplicative interaction gives the FFN more expressive
power per parameter, and empirically (per the GLU Variants paper and
subsequent LLM ablations) SwiGLU FFNs measurably outperform ReLU/GELU FFNs at
matched parameter/compute budgets — which is why essentially every modern
open-weight LLM (Llama 1/2/3, Mistral, Qwen, Gemma) uses it despite the extra
matrix. To keep total FFN parameter count roughly comparable to a plain 2-
matrix, 4x-`d_ff` FFN despite adding a third matrix, implementations
typically shrink `d_ff` to about `(2/3) * 4 * d_model ≈ 2.67 * d_model`
(rounded to a hardware-friendly multiple, e.g. a multiple of 256), so total
FFN parameters stay close to `2 * (4 * d_model) * d_model = 8 * d_model²`
either way.

### 5.3 Why the FFN holds most of the parameters

Per-sublayer parameter counts, in terms of `d_model` and `d_ff ≈ 4 * d_model`:

```
Attention sublayer:  W_Q, W_K, W_V, W_O, each (d_model × d_model)
                     = 4 * d_model²

FFN sublayer:        W_1 (d_model × d_ff) + W_2 (d_ff × d_model)
                     = 2 * d_model * d_ff
                     ≈ 2 * d_model * (4 * d_model)
                     = 8 * d_model²
```

Per decoder block, the FFN has **roughly 2x as many parameters as the entire
attention sublayer** (`8 d_model²` vs `4 d_model²`), and since this ratio
holds at every one of the `N` layers, across the whole network the FFN
sublayers collectively hold about two-thirds of all non-embedding
parameters. This is a direct, mechanical consequence of `d_ff` being set to
`~4x d_model` — attention's weight matrices are all `d_model × d_model`
(no dimension multiplier), while both FFN matrices touch the wide `d_ff`
dimension. It's part of why the FFN sublayer is often described as where the
model stores most of its "knowledge" (memorized facts, associations) — it
has, quite literally, most of the capacity — while attention is
comparatively lean and specialized for routing/mixing information between
positions.

---

## 6. Layer Normalization vs. RMSNorm

Every sublayer in the network (attention, FFN) is preceded (in the
pre-norm convention — see 6.3) by a normalization step. Normalization
exists to keep activation magnitudes in a well-behaved range as they flow
through dozens of stacked layers, preventing values from exploding or
shrinking to numerical extremes over many layers of matrix multiplication,
and empirically it makes training with larger learning rates stable, which
speeds up convergence substantially.

### 6.1 LayerNorm

For an input vector `x ∈ R^d` (i.e., one token's `d_model`-dimensional
vector), LayerNorm normalizes across the feature dimension:

```
μ = (1/d) * Σ_i x_i                         (mean)
σ² = (1/d) * Σ_i (x_i - μ)²                 (variance)

LayerNorm(x)_i = ( (x_i - μ) / sqrt(σ² + ε) ) * γ_i + β_i
```

`γ, β ∈ R^d` are learned per-dimension scale and shift parameters
(initialized to 1 and 0 respectively), and `ε` is a small constant (e.g.
1e-5) to avoid division by zero. LayerNorm re-centers the vector to zero
mean and rescales it to unit variance, then lets the network learn its own
preferred scale/shift on top of that normalized base.

### 6.2 RMSNorm

RMSNorm (Root Mean Square Normalization), used by Llama, Mistral, Qwen,
Gemma, T5, and most current LLMs, drops the mean-centering step entirely and
normalizes only by the root-mean-square magnitude:

```
RMS(x) = sqrt( (1/d) * Σ_i x_i²  +  ε )

RMSNorm(x)_i = ( x_i / RMS(x) ) * γ_i
```

Only one learned parameter vector `γ ∈ R^d` (no `β` shift term). Two
concrete simplifications relative to LayerNorm:

1. **No mean subtraction**: RMSNorm doesn't compute or subtract `μ`. The
   authors of the RMSNorm paper (Zhang & Sennrich, 2019) argued and showed
   empirically that the *re-scaling* invariance LayerNorm provides is what
   matters for training stability, while the *re-centering* (mean
   subtraction) contributes comparatively little to that benefit — so it can
   be dropped with negligible impact on model quality.
2. **No shift/bias term**: only a multiplicative `γ`, no additive `β`.

This is not merely a stylistic simplification — it directly reduces
compute: RMSNorm requires computing one reduction (`Σ x_i²`) instead of two
(`Σ x_i` for the mean, then `Σ (x_i - μ)²` for the variance), and applies one
elementwise op (`x/RMS * γ`) instead of two (`(x-μ)/σ * γ + β`). At the scale
of billions of forward/backward passes across a full pretraining run, this
per-call saving (papers report roughly 10-64% faster normalization
depending on hardware/implementation, translating to a few percent overall
training speedup) adds up, which — combined with negligible quality loss —
is why essentially every recent open-weight LLM (Llama family, Mistral,
Qwen, Gemma, DeepSeek) has standardized on RMSNorm over LayerNorm.

### 6.3 Pre-norm vs. post-norm

This is a separate axis from *which* normalization formula is used — it's
about *where in the residual computation* the norm is applied.

**Post-norm** (used in the original 2017 Transformer): normalize *after*
adding the residual:

```
x_out = Norm( x_in + Sublayer(x_in) )
```

**Pre-norm** (used by GPT-2 onward, and essentially all modern LLMs):
normalize *before* the sublayer, and add the residual around the whole
thing untouched:

```
x_out = x_in + Sublayer( Norm(x_in) )
```

The critical difference is what the residual stream itself looks like as
you go deeper. In pre-norm, `x_in` flows into the addition **completely
unmodified** — there is an unbroken, un-normalized path of pure addition
from the embedding layer all the way to the final layer (an "identity
highway"), and each sublayer only ever sees a normalized *copy* of the
running state as its input, never disturbing the running total itself. In
post-norm, every single sublayer output gets renormalized before becoming
the next layer's residual base, which repeatedly rescales the magnitude of
the accumulated signal and, in deep stacks (dozens of layers), leads to
unstable gradients during training — post-norm transformers are empirically
much harder to train past a certain depth without careful learning-rate
warmup, and often fail to converge at all without it. Pre-norm trades a
small amount of representational power (the final output distribution is
technically less "normalized-clean" since raw unnormalized activations
accumulate along the residual path, which is why most pre-norm models add
one final extra norm right before the output projection — see Section 8) for
dramatically more stable, warmup-friendlier training at large depth, which
is why virtually every modern LLM (GPT-2/3, Llama, Mistral, PaLM) uses
pre-norm.

---

## 7. Residual (Skip) Connections

Independent of the normalization choice, every sublayer is wrapped in a
residual connection:

```
x_out = x_in + Sublayer( Norm(x_in) )
```

where `Sublayer` is either the multi-head attention block or the FFN block.
The `+ x_in` term is the whole point: rather than forcing the sublayer to
compute the *entire* next representation from scratch, it only has to
compute a **delta** — how much to adjust the existing representation by. The
identity path (`x_in` added straight through) means that even if
`Sublayer(Norm(x_in))` were exactly zero (e.g. at initialization, or if a
particular sublayer turns out not to be useful for a given input), the block
still passes its input through unchanged rather than destroying it.

The concrete motivation is gradient flow in very deep networks. During
backpropagation, the gradient of the loss with respect to `x_in` for a
residual block is:

```
∂L/∂x_in = ∂L/∂x_out * ( I + ∂Sublayer/∂x_in )
```

The `I` (identity) term guarantees that a gradient signal of magnitude 1
always flows straight backward through every block, regardless of what the
Jacobian `∂Sublayer/∂x_in` of the attention or FFN sublayer looks like. Even
if `∂Sublayer/∂x_in` is close to zero or poorly conditioned (which is common
in deep networks — this is exactly the vanishing gradient problem: repeatedly
multiplying many small/poorly-scaled Jacobians together across dozens of
layers without a residual path shrinks the gradient toward zero
exponentially in depth) the `+I` term means the gradient reaching layer `k`
is never smaller than the gradient at layer `k+1`. Without residual
connections, training a plain 80-layer network (Llama 2 70B's depth) end to
end would be extremely difficult — gradients from the loss (computed after
the last layer) would have to survive being multiplied through 80
successive nonlinear transformations to reach the earliest layers, and in
practice would vanish (or explode) long before that. Residual connections
are what made training networks with tens to hundreds of layers practical in
the first place (originally introduced for ResNets in vision, then adopted
by the Transformer, and now universal in every deep LLM architecture).

---

## 8. The Final Layer: Unembedding + Softmax

After the input passes through all `N` decoder blocks, the final
`(B, T, d_model)` tensor needs to be converted back into a probability
distribution over the vocabulary so the model can predict the next token.

### 8.1 Final normalization

Since the architecture is pre-norm (Section 6.3), the residual stream
carries raw, un-normalized accumulated activations all the way through — so
one last normalization is applied before the output projection:

```
x_final = RMSNorm(x_out_of_last_block)     shape (B, T, d_model)
```

### 8.2 Output projection (unembedding)

A linear layer projects from `d_model` back up to `vocab_size`:

```
W_U ∈ R^(d_model × vocab_size)

logits = x_final @ W_U        shape (B, T, vocab_size)
```

Each position `t`'s `vocab_size`-length vector `logits[b, t, :]` holds one
raw (unnormalized) score per vocabulary token, representing the model's
prediction for the token that comes *after* position `t`.

**Weight tying**: `W_U` can either be its own independently-learned
`(d_model, vocab_size)` matrix ("untied"), or set to be exactly the
*transpose* of the input token embedding matrix, `W_U = W_E^T` ("tied").
Weight tying (used in GPT-2, and many other LLMs) roughly halves the
parameter cost of the embedding/unembedding pair — a nontrivial saving,
since `vocab_size * d_model` can be tens to hundreds of millions of
parameters — and rests on the intuition that the mapping "token → vector"
and "vector → likely-token" are related tasks that can plausibly share a
representation: a token's embedding vector and the direction in space that
makes the unembedding layer predict that same token should be similar.
Some larger/newer models (e.g. some Llama variants) untie them, trading
extra parameters for slightly more representational flexibility, since the
best vector for *representing* a token as input isn't necessarily the exact
best vector for *predicting* it as output.

### 8.3 Softmax: logits to probabilities

```
P(next_token = v | context) = softmax(logits)_v
                             = exp(logits_v) / Σ_v' exp(logits_v')
```

applied over the `vocab_size` dimension, independently for each position
`(b, t)`. This turns the raw logit vector into a valid probability
distribution (nonnegative, sums to 1) over the vocabulary. At training time,
cross-entropy loss compares this distribution (at each position) against the
actual next token in the training sequence. At inference/generation time,
only the final position's distribution (`P(next_token | full context so
far)`) is actually needed, and a sampling strategy (greedy argmax, top-k,
top-p/nucleus, temperature scaling) picks the next token to emit; softmax
itself is typically fused into the sampling step rather than computed as a
separate materialized tensor, since normalizing constants aren't needed for
argmax or for one common relative-probability comparisons like top-k/top-p.

---

## 9. Full Decoder Block: ASCII Diagram

Putting Sections 2–8 together, here is one full pre-norm decoder block using
RoPE, causal multi-head self-attention, RMSNorm, and a SwiGLU FFN — the
architecture used by Llama-family models — stacked `N` times between the
embedding and output layers.

```
                              Token IDs  (B, T)
                                   |
                                   v
                  +--------------------------------+
                  |  Token Embedding Lookup  W_E    |   (vocab_size x d_model)
                  |  x = W_E[tokens]                |
                  +--------------------------------+
                                   |
                                   v
                         x : (B, T, d_model)
                                   |
      =====================================================================
      |                    DECODER BLOCK  (repeated N times)              |
      |                                                                   |
      |    x_in ----------------------------------------------------+    |
      |     |                                                        |    |
      |     v                                                        |    |
      |  +------------+                                              |    |
      |  |  RMSNorm   |                                              |    |
      |  +------------+                                              |    |
      |     |                                                        |    |
      |     v                                                        |    |
      |  +-----------------------------------------------------+     |    |
      |  |         Multi-Head Causal Self-Attention             |    |    |
      |  |                                                       |    |    |
      |  |   Q = x W_Q     K = x W_K     V = x W_V               |    |    |
      |  |     |             |             |                     |    |    |
      |  |   split H heads  split H heads split H heads          |    |    |
      |  |     |             |             |                     |    |    |
      |  |     v             v             |                     |    |    |
      |  |  [apply RoPE]  [apply RoPE]     |                     |    |    |
      |  |     |             |             |                     |    |    |
      |  |     +------+------+             |                     |    |    |
      |  |            v                    |                     |    |    |
      |  |     S = Q K^T / sqrt(d_k)       |                     |    |    |
      |  |            |                    |                     |    |    |
      |  |            v                    |                     |    |    |
      |  |     S_masked = S + CausalMask   |                     |    |    |
      |  |     (upper triangle -> -inf)    |                     |    |    |
      |  |            |                    |                     |    |    |
      |  |            v                    |                     |    |    |
      |  |     P = softmax(S_masked)       |                     |    |    |
      |  |            |                    |                     |    |    |
      |  |            +--------> O_h = P V_h  <-------------------+    |    |
      |  |                          |  (per head, h = 1..H)             |    |
      |  |                          v                                   |    |
      |  |          Concat(O_1, ..., O_H)   (B, T, d_model)              |    |
      |  |                          |                                   |    |
      |  |                          v                                   |    |
      |  |                  out = Concat(O) @ W_O                       |    |
      |  +-----------------------------------------------------+        |    |
      |     |                                                            |    |
      |     v                                                            |    |
      |  (+)  <----------------------------------------------------------+    |
      |   |         residual add:  x_mid = x_in + Attention(RMSNorm(x_in))    |
      |   v                                                                   |
      |  x_mid ---------------------------------------------------------+    |
      |   |                                                              |    |
      |   v                                                              |    |
      |  +------------+                                                  |    |
      |  |  RMSNorm   |                                                  |    |
      |  +------------+                                                  |    |
      |   |                                                              |    |
      |   v                                                              |    |
      |  +-----------------------------------------------------+         |    |
      |  |               SwiGLU Feed-Forward Network             |        |    |
      |  |                                                       |        |    |
      |  |   gate = Swish( x_mid_norm @ W_gate )   (B,T,d_ff)     |        |    |
      |  |   up   =        x_mid_norm @ W_up       (B,T,d_ff)     |        |    |
      |  |   h    = gate ⊙ up                       (B,T,d_ff)    |        |    |
      |  |   out  = h @ W_down                      (B,T,d_model) |        |    |
      |  +-----------------------------------------------------+         |    |
      |   |                                                              |    |
      |   v                                                              |    |
      |  (+)  <----------------------------------------------------------+    |
      |   |         residual add:  x_out = x_mid + FFN(RMSNorm(x_mid))        |
      |   v                                                                   |
      |  x_out : (B, T, d_model)   -->  fed as x_in to the next block         |
      |                                                                       |
      =========================================================================
                                   |
                          (after N blocks stacked)
                                   v
                  +--------------------------------+
                  |     Final RMSNorm               |
                  +--------------------------------+
                                   |
                                   v
                  +--------------------------------+
                  |  Output Projection  W_U         |   (d_model x vocab_size)
                  |  logits = x_final @ W_U         |   [may be tied to W_E^T]
                  +--------------------------------+
                                   |
                                   v
                         logits : (B, T, vocab_size)
                                   |
                                   v
                  +--------------------------------+
                  |            Softmax              |
                  +--------------------------------+
                                   |
                                   v
                 P(next_token | context) : (B, T, vocab_size)
```

Two residual adds per block (one around attention, one around the FFN),
each preceded by its own RMSNorm, is the entire repeating unit. Everything
in Sections 2–8 is either inside this diagram or wraps it (embedding before,
final norm + unembed + softmax after).

---

## 10. Parameter Counting

You can estimate a dense (non-MoE) decoder-only transformer's total
parameter count almost entirely from five numbers: `d_model`, `num_layers`
(`N`), `num_heads` (`H`, needed only insofar as it fixes `d_k = d_model/H` —
the *count* of Q/K/V/O parameters is independent of `H` itself, since
splitting into heads is a free reshape), `vocab_size`, and the FFN
hidden dimension `d_ff`.

### 10.1 Per-block parameter count

```
Attention:  W_Q + W_K + W_V + W_O, each (d_model x d_model)
          = 4 * d_model^2

FFN (plain, 2-matrix, ReLU/GELU):  W_1 (d_model x d_ff) + W_2 (d_ff x d_model)
          = 2 * d_model * d_ff

FFN (SwiGLU, 3-matrix):  W_gate + W_up (each d_model x d_ff) + W_down (d_ff x d_model)
          = 3 * d_model * d_ff

RMSNorm (x2 per block, attention + FFN):  2 * d_model
          (negligible relative to the matmul weights)

Params per block  ≈  4*d_model^2  +  {2 or 3}*d_model*d_ff
```

With the common convention `d_ff ≈ 4 * d_model` (plain FFN):

```
Params per block  ≈  4*d_model^2 + 8*d_model^2  =  12 * d_model^2
```

With SwiGLU and the correspondingly-shrunk `d_ff ≈ (8/3)*d_model` (chosen so
`3*d_model*d_ff ≈ 8*d_model^2`, matching the plain-FFN parameter budget):

```
Params per block  ≈  4*d_model^2 + 8*d_model^2  =  12 * d_model^2   (same order)
```

### 10.2 Whole-model parameter count

```
Total params  ≈  (embedding)  +  N * (params per block)  +  (final norm + unembed)

Embedding:            vocab_size * d_model
Unembed (if untied):  d_model * vocab_size     (0 if tied to embedding)
Decoder stack:        N * 12 * d_model^2        (using the ~12*d_model^2/block estimate)
Final norm:           d_model                   (negligible)

Total  ≈  N * 12 * d_model^2  +  (1 or 2) * vocab_size * d_model
```

### 10.3 Worked example: a Llama-2-7B-shaped model

```
d_model    = 4096
N          = 32
H          = 32   (d_k = 128)
vocab_size = 32000
d_ff       ≈ 11008   (Llama 2 7B's actual SwiGLU hidden dim)
```

```
Per-block:      4 * 4096^2  +  3 * 4096 * 11008
              = 67,108,864  +  135,266,304
              ≈ 202,375,168   (~202M params per block)

Decoder stack:  32 * 202,375,168  ≈  6.48 billion

Embedding:      32000 * 4096  =  131,072,000  (~131M)
Unembed:        untied in Llama 2 -> another ~131M

Total: 6.48B (blocks) + 0.131B (embed) + 0.131B (unembed) ≈ 6.74B
```

This lands close to the actual reported 6.7B parameters for Llama 2 7B
(the small residual gap is rounding in `d_ff` and minor architectural
details like norm parameters not modeled here). The general takeaway from
this exercise, consistent with Section 5.3: for realistic `d_ff ≈ 4*d_model`
(or SwiGLU's ~2.67x with a 3-matrix FFN), the `N * 12 * d_model^2` decoder
stack term dominates total parameter count for any model of meaningful
depth, and within that term the FFN matrices outweigh the attention
matrices roughly 2:1 — the embedding and unembedding tables are usually a
small fraction of the total unless `vocab_size` is unusually large relative
to `d_model` (e.g. multilingual models with 250k+ token vocabularies) or the
model itself is small enough that a ~65k-to-256k-row embedding table becomes
comparable in size to a shallow decoder stack.

---

## Summary

| Layer | Input shape | Output shape | Learned params | Role |
|---|---|---|---|---|
| Token embedding | `(B,T)` ints | `(B,T,d_model)` | `vocab_size × d_model` | token ID → vector |
| Positional (learned/sinusoidal) | `(B,T,d_model)` | `(B,T,d_model)` | `max_len × d_model` (learned) or none (sinusoidal) | inject order |
| RoPE | applied to Q,K per layer | same shape | none | inject *relative* order into attention scores |
| RMSNorm | `(B,T,d_model)` | `(B,T,d_model)` | `d_model` (γ only) | stabilize activation scale |
| Multi-head causal self-attention | `(B,T,d_model)` | `(B,T,d_model)` | `4·d_model²` | mix information across positions |
| SwiGLU FFN | `(B,T,d_model)` | `(B,T,d_model)` | `~3·d_model·d_ff ≈ 8·d_model²` | per-token nonlinear transform; bulk of model "knowledge" |
| Residual adds | — | — | none | gradient flow across depth |
| Final RMSNorm | `(B,T,d_model)` | `(B,T,d_model)` | `d_model` | pre-output normalization |
| Unembedding + softmax | `(B,T,d_model)` | `(B,T,vocab_size)` | `d_model × vocab_size` (0 if tied) | vector → next-token distribution |

The whole network is: embed → (norm → causal attention → residual add →
norm → FFN → residual add) × N → norm → unembed → softmax. Every
architectural variation between Llama, GPT, Mistral, Qwen, and other modern
LLMs is a variation on this same skeleton — different choices of positional
scheme, normalization formula, activation function, and attention KV-sharing
strategy (see `docs/ml-concepts/18-attention-optimizations/README.md`), but
the same fundamental stack of embed → repeated (attend, then transform) →
unembed.
