# What Is an LLM? A First-Principles Technical Guide

## Audience and scope

This document explains how large language models actually work, mechanically, from tokenization through to sampled output. It assumes you can read code and math but have no prior ML background. It is not a product overview — there is no discussion of chat UIs, agents, or prompting tricks. The goal is that after reading this, you could sketch the forward pass of a transformer decoder on a whiteboard and explain what every matrix multiplication is for.

---

## 1. What an LLM actually is

Strip away the hype and a large language model is a single mathematical object: a function that takes a sequence of tokens and outputs a probability distribution over "what token comes next."

```
P(token_n | token_1, token_2, ..., token_{n-1})
```

That's it. Everything else — chatbots, code assistants, agents — is built on top of repeatedly calling this one function and feeding its own outputs back in as new inputs.

Concretely, the model is a parameterized function `f_θ` where `θ` is somewhere between hundreds of millions and hundreds of billions of real-valued numbers (weights). Given an input sequence, `f_θ` produces a vector of logits — one real number per entry in the vocabulary — which is then turned into a probability distribution via softmax:

```
logits = f_θ(token_1, ..., token_{n-1})          # shape: [vocab_size]
P(next_token = i) = softmax(logits)_i = exp(logits_i) / Σ_j exp(logits_j)
```

The "large" in LLM refers to `θ` (billions of parameters) and to the training corpus (trillions of tokens of text). The "language model" part is the object described above: a next-token predictor. There is no separate "understanding module," no symbolic reasoning engine, no database of facts consulted at inference time. Every capability you observe — writing code, answering questions, holding a conversation — is an emergent behavior of a system trained to be very good at predicting the next token, when that system is queried in the right way (e.g., by feeding it a prompt and letting it continue).

The specific function family `f_θ` used by virtually every modern LLM is the **transformer**, specifically the decoder-only transformer. The rest of this document builds that function up piece by piece: how raw text becomes numbers (tokenization), how the transformer processes those numbers (architecture), how a full response is produced from repeated single-token predictions (decoding), and how `θ` is chosen in the first place (training).

---

## 2. Tokenization: how text becomes numbers

Neural networks operate on tensors of numbers, not characters. The first step in any LLM pipeline is converting a string into a sequence of integers, each identifying an entry in a fixed vocabulary. This is tokenization, and it happens identically at training time and inference time — it is a fixed, non-learned preprocessing step (the tokenizer is trained once, separately, before model training begins, and then frozen).

### Why not just use characters or words?

Two naive options and why they fail:

- **Character-level**: vocabulary is tiny (~100-200 symbols), but sequences become very long (a 500-word passage might be 2,500+ characters), and the model has to spend enormous learned capacity re-deriving the concept of "words" and "subwords" from scratch. Long sequences are expensive (see Section 6 on the O(n²) cost of attention), so this is wasteful.
- **Word-level**: vocabulary must contain every word the model might ever see. This blows up to hundreds of thousands or millions of entries once you account for morphology, typos, code identifiers, and every language on earth, and it still fails catastrophically on any word not seen during vocabulary construction (out-of-vocabulary tokens), including new words, names, and misspellings.

### The solution: subword tokenization (BPE)

Almost all production LLMs use a subword scheme, most commonly **Byte-Pair Encoding (BPE)** or a close variant (e.g., BPE over UTF-8 bytes, or SentencePiece's unigram model). The idea: common whole words get their own single token, while rare or unseen words get split into smaller, reusable pieces — sometimes down to individual bytes in the worst case, so *anything* is representable.

**How BPE vocabulary construction works** (done once, offline, on a large text sample):

1. Start with a base vocabulary of individual bytes or characters (e.g., 256 byte values for byte-level BPE — this guarantees full coverage of any input, including emoji and unknown scripts).
2. Count all adjacent symbol pairs in the training corpus. E.g., if the corpus is full of `t h e`, the pair `(t, h)` and `(h, e)` will be extremely frequent.
3. Merge the single most frequent pair into a new symbol (e.g., `t`+`h` → `th`). Add this new symbol to the vocabulary.
4. Re-scan the (now partially merged) corpus and repeat: find the next most frequent adjacent pair, merge it, add to vocabulary.
5. Stop after a target vocabulary size is reached (typically 32,000 to 200,000+ merges/tokens for modern LLMs).

The output of this process is a **merge table**: an ordered list of which pairs get merged into which new symbols. At inference/training time, encoding a new string means applying these merges greedily in the learned order until no more merges apply. Decoding is the inverse: look up each token ID's string in the vocabulary table and concatenate.

Example (illustrative, not exact for any specific real tokenizer):

```
Input text:      "tokenization"
Base bytes:       t o k e n i z a t i o n
After merges:     token  ization        (2 tokens)
Token IDs:        [14832, 2298]
```

Common words like "the", "and", or "function" typically end up as single tokens. Rare words, made-up words, or non-English text get split into multiple subword pieces. A useful mental model: **one token is roughly 3-4 characters of English text**, or about 0.75 words.

### Why the tokenizer matters so much

- **It sets the effective granularity the model reasons over.** The model never sees individual letters unless a word is rare enough to be split down that far. This is why LLMs have historically been bad at character-level tasks like counting letters in a word ("how many r's in strawberry") — the model sees the token `strawberry` (or a couple of subword chunks), not the seven individual letters, so counting letters requires it to have memorized spelling patterns rather than actually iterating over characters.
- **It determines sequence length, which determines cost.** Since attention cost scales quadratically with sequence length (Section 6), and generation cost scales linearly with number of tokens generated, a tokenizer that represents the same text in fewer tokens directly reduces compute cost and increases effective context window. This is why tokenizer efficiency varies a lot by language — a tokenizer trained mostly on English text will often need 2-3x more tokens to represent the equivalent content in, say, Thai or Burmese, because those scripts are underrepresented in the merge-table training data.
- **It's frozen after training and can't be casually changed.** The model's embedding matrix (Section 3) has exactly one learned row per vocabulary entry. If you change the tokenizer, every token ID means something different, and the model's learned embeddings for those IDs are meaningless — you'd have to retrain from scratch (or at least redo the embedding layer and fine-tune substantially). This is why vocabulary size is a decision made once, early, with long-lasting consequences.

### Vocabulary size tradeoffs

| Smaller vocabulary (e.g., 32K) | Larger vocabulary (e.g., 128K-256K) |
|---|---|
| Longer token sequences for the same text | Shorter token sequences for the same text (cheaper inference, more effective context) |
| Smaller embedding matrix and output projection (fewer parameters, less memory) | Larger embedding matrix and output softmax layer (more parameters — this can be a meaningful fraction of total model size) |
| More composition needed per word (model must learn to "reassemble" meaning from more pieces) | Better single-token coverage of common words/phrases, and often better multilingual coverage |
| Each token is seen more often during training (denser gradient signal per token) | Each individual token is seen somewhat less often (sparser signal for rare tokens) |

Modern frontier models have trended toward larger vocabularies (100K-250K+) specifically to improve multilingual and code tokenization efficiency, accepting the larger embedding/output-layer parameter cost in exchange.

---

## 3. The transformer architecture, end to end

Once text is a sequence of integer token IDs, here is the full pipeline a decoder-only transformer (the architecture behind GPT-style, Llama-style, and essentially all modern chat-oriented LLMs) applies to turn those IDs into a next-token probability distribution.

### 3.1 High-level flow

```
 token IDs [t1, t2, ..., tn]
        │
        ▼
 ┌─────────────────────┐
 │   Token Embedding    │   lookup: id -> vector (d_model dims)
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │  Positional Encoding │   inject position info (added or via RoPE in attention)
 └─────────────────────┘
        │
        ▼
 ┌───────────────────────────────────────┐
 │  Decoder Block 1                       │
 │  ┌───────────────────────────────┐    │
 │  │ Masked Multi-Head Self-Attn    │    │
 │  └───────────────────────────────┘    │
 │             + residual, LayerNorm      │
 │  ┌───────────────────────────────┐    │
 │  │ Feed-Forward Network (MLP)     │    │
 │  └───────────────────────────────┘    │
 │             + residual, LayerNorm      │
 └───────────────────────────────────────┘
        │
        ▼
       ...  (repeated N times, e.g. N = 32, 80, 96...)
        │
        ▼
 ┌───────────────────────────────────────┐
 │  Decoder Block N                       │
 └───────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │   Final LayerNorm    │
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │  Output Projection   │   linear layer: d_model -> vocab_size
 │  (often tied to      │
 │  embedding weights)  │
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │       Softmax         │   logits -> probability distribution over vocab
 └─────────────────────┘
        │
        ▼
 P(next token | t1..tn)
```

Each of the N decoder blocks is architecturally identical (same shapes, independently learned weights). A model described as "32 layers, d_model=4096" means N=32 copies of the block above are stacked, each operating on 4096-dimensional vectors.

### 3.2 Token embedding

Each of the V vocabulary entries has a learned vector of length `d_model` (e.g., 4096). The embedding layer is literally a lookup table — a matrix `E` of shape `[V, d_model]`. Token ID `t` gets mapped to row `E[t]`. This is a learned parameter, not a hand-designed feature: during training, gradient descent shapes these vectors so that, empirically, tokens used in similar contexts end up with similar vectors (this is where the well-known "king - man + woman ≈ queen" style geometric relationships emerge — as a byproduct of the training objective, not a designed property).

Input to the rest of the network: a matrix of shape `[n, d_model]` where `n` is the sequence length — one row per token position.

### 3.3 Positional encoding

Self-attention (below) is, by construction, permutation-invariant: without extra information, it would treat "dog bites man" identically to "man bites dog," because attention computes a weighted sum over positions with no inherent notion of order. Positional encoding fixes this by injecting position information.

Two dominant approaches:

- **Sinusoidal / learned absolute embeddings** (original transformer, early GPT models): add a position-dependent vector `P[i]` to each token embedding at position `i`, where `P` is either a fixed sinusoidal function of position or a learned embedding table indexed by position, exactly analogous to the token embedding table but indexed by position instead of token ID.
- **Rotary Position Embeddings (RoPE)** — used by most current models (Llama, Mistral, Qwen, etc.): instead of adding a position vector, the query and key vectors (see 3.4) are *rotated* by an angle proportional to their position, inside the attention computation itself. This has the nice property that the attention score between two tokens ends up depending only on their *relative* distance, not their absolute positions, which generalizes better to sequence lengths not seen during training.

Either way, by this point the network has, per position, a vector that encodes both "what token is here" and "where in the sequence this is."

### 3.4 Self-attention — the actual math

This is the mechanism that lets the model mix information across positions — i.e., let a token's representation be updated based on other tokens in the sequence. It is the core innovation of the transformer.

**Step 1 — project into Query, Key, Value.** For each position's input vector `x` (shape `d_model`), compute three separate linear projections using three learned weight matrices `W_Q`, `W_K`, `W_V` (each shape `[d_model, d_k]`):

```
Q = X · W_Q      # shape [n, d_k]   "what am I looking for?"
K = X · W_K      # shape [n, d_k]   "what do I contain?"
V = X · W_V      # shape [n, d_v]   "what do I offer if attended to?"
```

Here `X` is the `[n, d_model]` matrix of all position vectors stacked together, so `Q`, `K`, `V` are computed for every position simultaneously via matrix multiplication.

**Step 2 — compute attention scores.** For every pair of positions `(i, j)`, compute how much position `i` should attend to position `j` by taking the dot product of `i`'s query with `j`'s key:

```
scores = Q · K^T             # shape [n, n]
scores_ij = Q_i · K_j  =  how relevant is position j's content to position i's query
```

**Step 3 — scale.** Divide by `sqrt(d_k)`. Without this, as `d_k` grows, dot products grow large in magnitude, which pushes softmax into regions with extremely small gradients (it saturates). Scaling keeps the values in a range where softmax has a well-behaved gradient.

**Step 4 — mask (causal masking).** For a decoder — the architecture LLMs use for generation — position `i` must not be allowed to see positions `j > i` (future tokens), or the model could "cheat" during training by looking ahead at the answer. This is enforced by setting `scores_ij = -infinity` for all `j > i`, before the softmax, so that after softmax those positions get exactly zero weight.

**Step 5 — softmax, then weighted sum of values.**

```
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + mask ) · V
```

This is the full scaled dot-product attention formula. The softmax is applied row-wise (i.e., independently per query position `i`, normalizing over all valid `j`), turning each row of scores into a probability distribution that sums to 1. The output at position `i` is then a weighted average of all the value vectors `V_j` (for `j ≤ i`), weighted by how much position `i` "attends to" position `j`. This is literally how information moves between positions in a transformer: token `i`'s new representation is built by pulling information out of every earlier token's value vector, in proportion to a learned relevance score.

**Concretely what this buys you**: if the input contains "The cat sat on the mat because it was tired," the word "it" needs to figure out that it refers to "cat," not "mat." Attention lets the representation at the position of "it" compute a high score against the key vector at "cat" and pull "cat"'s value vector into its own representation. Which words attend to which is not hand-coded — it emerges entirely from `W_Q`, `W_K`, `W_V` being trained via gradient descent on the language-modeling objective.

### 3.5 Multi-head attention

A single attention computation forces the model to use one shared notion of "relevance" for everything. In practice you want the model to simultaneously track multiple different kinds of relationships — e.g., one head might specialize in tracking subject-verb agreement, another in coreference ("it" → "cat"), another in local syntax. Multi-head attention does this by running several independent attention computations in parallel, each in a lower-dimensional subspace, and concatenating the results.

```
head_i = Attention(X·W_Q_i, X·W_K_i, X·W_V_i)     for i = 1 .. h

MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

If `d_model = 4096` and there are `h = 32` heads, each head operates with `d_k = d_v = 4096 / 32 = 128`. Each head has its own independently-learned `W_Q_i, W_K_i, W_V_i`. The `h` head outputs (each `[n, 128]`) are concatenated back to `[n, 4096]` and passed through one more learned linear projection `W_O` (`[d_model, d_model]`) to mix information across heads. Total compute is roughly the same as one big attention operation, but the model gets `h` independent "relevance channels" instead of one.

**A note on KV cache**: at inference time, since generation is autoregressive (Section 4) and causal masking means past keys/values never change once computed, production systems cache the `K` and `V` projections for all previous positions rather than recomputing them on every new token. This is the single biggest practical optimization in LLM serving and is why memory (not just compute) becomes a bottleneck for long contexts — the KV cache grows linearly with sequence length and with the number of layers and heads.

### 3.6 Feed-forward network (FFN / MLP)

After attention mixes information *across* positions, each position independently passes through a position-wise feed-forward network — a small 2-layer MLP applied identically (same weights) to every position's vector:

```
FFN(x) = activation(x · W1 + b1) · W2 + b2
```

Typically `W1` projects from `d_model` up to a larger intermediate size (commonly 4x `d_model`, e.g., 4096 → 16384), an activation function is applied (GeLU, SwiGLU, and similar variants are common in modern models), and `W2` projects back down to `d_model`. This block is where a large fraction of the model's total parameters actually live (attention parameters scale with `d_model²` per head config, but the FFN's up/down projections at 4x expansion are often the single largest parameter consumer per layer). Conceptually, if attention is "gather relevant information from other positions," the FFN is "process/transform that information" — this is frequently where factual recall and pattern-completion behavior are attributed in interpretability research, though this is an active research area, not a settled fact.

### 3.7 Residual connections and layer normalization

Every sub-block (attention, and separately the FFN) is wrapped as:

```
x = x + Sublayer(Norm(x))          # "pre-norm" — used in most modern LLMs
```

or, in the original transformer paper's ordering:

```
x = Norm(x + Sublayer(x))          # "post-norm"
```

Two distinct mechanisms are doing work here:

- **Residual (skip) connections**: adding the input `x` back to the sublayer's output. Without this, gradients have to flow backward through every one of the N stacked layers during training, and in deep networks this causes vanishing gradients — the earliest layers get almost no learning signal. The residual path gives gradients a direct, unimpeded route back to every layer, which is what makes it practical to stack 50-100+ layers at all.
- **Layer normalization**: rescales the activations at each position to have consistent mean/variance (independently per token position, unlike batch norm which normalizes across the batch). This stabilizes training by keeping the scale of activations from drifting or exploding as they pass through dozens of layers. Pre-norm (normalizing before the sublayer, rather than after adding the residual) has become standard because it produces more stable gradients in very deep transformer stacks.

### 3.8 Final projection and softmax

After the last decoder block, one final layer norm is applied, and the resulting `[n, d_model]` matrix is projected back to vocabulary size via a learned linear layer (the "unembedding" or "LM head"), often literally using the transpose of the token embedding matrix `E` from Section 3.2 (weight tying) to save parameters:

```
logits = final_norm(x) · E^T        # shape [n, vocab_size]
```

Only the logits at the *last* position are needed to predict the next token during generation (though during training, logits at every position are used simultaneously — see Section 5.1). Softmax converts the last position's logit vector into the next-token probability distribution described in Section 1.

---

## 4. Autoregressive decoding: generating text one token at a time

The transformer, as described above, is a function that maps a sequence to a probability distribution over the *single next* token. To generate a full response, this function is called repeatedly, each time appending the newly produced token to the input and running the whole forward pass again:

```
sequence = prompt_tokens
loop:
    logits = transformer(sequence)              # shape [vocab_size], from last position
    probs  = softmax(logits)
    next_token = sample(probs)                  # see strategies below
    sequence = sequence + [next_token]
    if next_token == <EOS> or len(sequence) hits limit: stop
```

This is why LLM output is generated left-to-right, one token at a time, and why latency scales with output length — each token requires a full forward pass through all N layers (mitigated in practice by KV caching, which avoids recomputing attention over already-processed tokens, but the sequential dependency itself is fundamental: token `k+1` cannot be computed before token `k` is chosen, in standard decoding).

### Sampling strategies

`sample(probs)` is not fixed — how you turn a probability distribution into an actual chosen token is a deliberate design choice with major behavioral consequences:

- **Greedy decoding**: always pick `argmax(probs)`, the single highest-probability token. Deterministic, but tends to produce repetitive, generic text, and is vulnerable to getting stuck in loops, because it never explores lower-probability-but-reasonable continuations.

- **Temperature sampling**: before softmax, divide logits by a temperature `T`:

  ```
  probs = softmax(logits / T)
  ```

  `T = 1` leaves the distribution unchanged. `T < 1` sharpens the distribution (makes it more peaked, closer to greedy, more deterministic/conservative). `T > 1` flattens it (more uniform, more random/creative, higher chance of incoherent output). `T → 0` converges to greedy decoding.

- **Top-k sampling**: restrict sampling to only the `k` highest-probability tokens (e.g., top 40), renormalize their probabilities to sum to 1, then sample from that restricted set. This prevents the model from ever sampling a wildly unlikely token from the long tail, while still allowing some randomness among plausible options.

- **Top-p / nucleus sampling**: instead of a fixed count `k`, keep the smallest set of highest-probability tokens whose cumulative probability exceeds threshold `p` (e.g., `p = 0.9`), then sample from that set. This adapts the size of the candidate pool to the actual shape of the distribution: when the model is very confident (one token has 95% probability), the nucleus might contain just 1-2 tokens; when the model is uncertain (probability spread thinly across many tokens), the nucleus naturally grows larger. This adaptivity is why top-p is generally preferred over a fixed top-k in production systems.

These strategies are commonly combined (e.g., apply temperature, then top-k, then top-p, then sample) and are typically exposed directly as API parameters, because the "right" amount of randomness is task-dependent: near-zero temperature for deterministic tasks like code generation or extraction, higher temperature for open-ended creative writing.

---

## 5. Training stages: pretraining, SFT, and alignment

The parameters `θ` are not designed — they are learned by iteratively adjusting billions of numbers to make the model's predictions match observed data, via gradient descent. Modern LLM training happens in distinct stages, each with a different objective, different data, and a different effect on model behavior.

```
┌─────────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
│   1. PRETRAINING     │ ───► │  2. SUPERVISED       │ ───► │  3. ALIGNMENT         │
│                       │      │     FINE-TUNING (SFT)│      │     (RLHF / DPO)      │
│  Data: trillions of   │      │  Data: thousands-    │      │  Data: comparisons /  │
│  tokens of raw text   │      │  millions of curated  │      │  preference pairs     │
│  (web, books, code)   │      │  (prompt, ideal       │      │  ("response A is      │
│                       │      │   response) pairs     │      │   better than B")     │
│  Objective: predict   │      │  Objective: imitate   │      │  Objective: shift     │
│  next token           │      │  the demonstrated     │      │  output distribution  │
│  (self-supervised,    │      │  response style/format│      │  toward preferred     │
│  no labels needed)    │      │  (supervised)         │      │  behavior             │
│                       │      │                       │      │                       │
│  Result: a raw        │      │  Result: a model that │      │  Result: a model      │
│  "text completion"    │      │  follows instructions │      │  tuned to be more      │
│  engine with broad    │      │  and answers in a     │      │  helpful, honest,     │
│  world/language       │      │  conversational/      │      │  and less harmful,    │
│  knowledge, but no    │      │  assistant format      │      │  per human judgment   │
│  notion of being a    │      │                       │      │                       │
│  helpful assistant    │      │                       │      │                       │
└─────────────────────┘      └─────────────────────┘      └──────────────────────┘
```

### 5.1 Pretraining

This is where the vast majority of compute (often >95%) is spent. The model is trained on a huge, largely unlabeled corpus — web crawls, books, code repositories, papers — with a single, simple, self-supervised objective: given a chunk of real text, predict each next token from the tokens before it.

Concretely, for a training document tokenized as `t_1, ..., t_n`, the model computes, in a single forward pass, a probability distribution for the next token at *every* position simultaneously (this is possible because of causal masking — position `i`'s output only depends on `t_1..t_i`, so all positions can be computed in parallel during training even though generation is sequential at inference time). The loss function is **cross-entropy loss**, averaged over all positions:

```
Loss = -(1/n) * Σ_i  log P(t_{i+1} | t_1, ..., t_i)
```

In words: for every position, look at the probability the model assigned to the *actual* next token that appeared in the real document, take the negative log of that probability, and average this over the whole sequence (and over the whole training batch). If the model assigns high probability to the correct next token, `-log(P)` is small (low loss); if it assigns low probability to the correct token, `-log(P)` is large (high loss). Gradient descent (via backpropagation) then nudges every parameter in `θ` in the direction that would have made the correct token slightly more probable.

This is repeated for trillions of tokens, over enormous compute budgets. No human labels the "correct" answer for anything — the correct next token is just whatever token actually appears next in the real text, which is why this is called self-supervised learning. The astonishing empirical finding of the last decade is that this simple objective, at sufficient scale, produces a model that has implicitly learned grammar, facts, reasoning patterns, and coding ability, purely as a side effect of getting extremely good at predicting plausible continuations of arbitrary text.

The output of this stage is often called a **base model**. Base models complete text but do not reliably behave like an assistant — asked "What's the capital of France?" a base model might continue with more quiz questions ("What's the capital of Germany?") rather than answering, because that's a highly plausible continuation of text that looks like a quiz, given its training data. This is the problem the next two stages fix.

### 5.2 Supervised Fine-Tuning (SFT)

The base model is next fine-tuned on a much smaller (thousands to low millions of examples), high-quality dataset of curated `(prompt, ideal response)` pairs — human-written or human-vetted examples of exactly the assistant-like behavior desired: answering questions directly, following instructions, refusing certain requests, formatting output a particular way. Training mechanics are identical to pretraining (same next-token-prediction, same cross-entropy loss), but now applied only to this curated conversational data, and typically the loss is computed only over the *response* tokens (the prompt is given as context but not "predicted" against).

The effect is not to teach the model new facts or reasoning ability — that headroom was mostly already set during pretraining — but to teach it *the format and behavior of being a helpful assistant*: how to structure an answer, when to ask a clarifying question, how to follow explicit instructions. This is why SFT works with orders of magnitude less data than pretraining: it's steering an already-capable model's output distribution toward a narrower, desired behavioral style, not building knowledge from scratch.

### 5.3 Alignment: RLHF and DPO

Even after SFT, there can be many "reasonable" responses to a given prompt, and picking among them well (more helpful, more honest, safer, better reasoning, less verbose) is hard to specify with more labeled examples alone. Alignment stages use *comparative human (or AI) judgment* rather than exact target outputs.

**RLHF (Reinforcement Learning from Human Feedback)**, the earlier and more complex approach:
1. Generate multiple candidate responses to the same prompt from the SFT model.
2. Have human raters (or a trained proxy) rank these candidates from best to worst.
3. Train a separate **reward model** to predict these human preference rankings — i.e., a model that takes `(prompt, response)` and outputs a scalar "quality" score, trained to match human rankings.
4. Use reinforcement learning (typically PPO, Proximal Policy Optimization) to further update the LLM's weights so that it produces responses the reward model scores highly, while a KL-divergence penalty keeps it from drifting too far from the SFT model (to prevent it from degenerating into meaninglessly gaming the reward model).

**DPO (Direct Preference Optimization)**, a simpler and now widely-used alternative:
- Skips training a separate reward model and skips reinforcement learning entirely.
- Directly optimizes the LLM on preference pairs `(prompt, chosen_response, rejected_response)` using a loss function derived mathematically to have the same optimal solution as the RLHF objective, but computed as a straightforward supervised-style loss: increase the model's relative probability of the chosen response versus the rejected response.
- Simpler to implement and more stable to train than full RLHF (no separate reward model, no RL training instability), which is why it has become popular, though RLHF-style pipelines (or hybrids) are still used, particularly at the largest labs.

Either way, the *purpose* of this stage is distinct from SFT: SFT teaches the model *what a good response looks like* via direct demonstration; alignment teaches the model *which of several plausible responses is preferred* via comparative feedback, which can capture nuanced preferences (tone, safety, helpfulness tradeoffs) that are hard to express as a single "correct" demonstration. It's also the stage most directly responsible for behaviors like refusing harmful requests, hedging appropriately on uncertain claims, and generally "sounding like" the assistant persona users expect.

---

## 6. Context window: why it exists and why it's expensive

The context window is the maximum number of tokens (prompt + generated output combined) a model can process in one forward pass. It exists because of a hard mathematical property of self-attention: **computing attention scores between every pair of positions is O(n²) in sequence length**.

Recall from Section 3.4: `scores = Q · K^T` produces an `[n, n]` matrix — every position attends to every other position. If you double the sequence length, you don't double the compute for attention, you quadruple it: `n=1000` requires computing and storing ~1,000,000 pairwise scores; `n=100,000` requires ~10,000,000,000. This quadratic term is on top of a linear cost elsewhere (embeddings, FFN, projections scale as O(n)), and at large enough `n`, the quadratic attention term dominates total cost.

This has concrete practical consequences:

- **Compute cost of processing a prompt grows superlinearly with length.** Doubling your prompt more than doubles the time and cost to process it (though real systems mitigate the constant factor with optimized kernels like FlashAttention, which reduce memory traffic but don't change the underlying O(n²) compute scaling).
- **Memory cost is dominated by the KV cache during generation**, which grows *linearly* with sequence length but linearly in a way that multiplies by number of layers and number of heads — for a large model with a long context, the KV cache alone can require many gigabytes of accelerator memory, often becoming the binding constraint on how many concurrent requests a server can serve, independent of the O(n²) compute question.
- **Various architectural tricks exist to push past naive quadratic attention** — sliding-window attention (only attend to a fixed nearby window instead of the full sequence), grouped-query attention (share K/V projections across multiple query heads to shrink the KV cache), and other sparse/linear-attention variants. These are engineering responses to a mathematical constraint, not ways of eliminating it — they all trade off some amount of long-range modeling fidelity for reduced compute/memory.
- **Context window is not the same as "effective usable context."** Even models with technically large advertised context windows can show degraded attention/recall for information placed in the middle of a very long context (sometimes called "lost in the middle" behavior) — a documented empirical finding, not a formula-derivable one, but consistent with the intuition that a fixed-capacity attention mechanism spreading itself over more positions has less "budget" to sharply attend to any single one.

---

## 7. Scaling laws: why bigger predictably means better (up to a point)

One of the more surprising empirical findings in this field is that model performance — measured as pretraining cross-entropy loss (Section 5.1) — follows remarkably smooth, predictable power-law curves as a function of three quantities: number of model parameters `N`, size of training dataset `D` (in tokens), and total compute `C` used for training (roughly `C ≈ 6ND` for transformer training, a widely used approximation).

The empirical relationship, found originally by Kaplan et al. (2020) and refined by Hoffmann et al. (2022, the "Chinchilla" paper), takes roughly the form:

```
Loss(N, D) ≈ E + A / N^α + B / D^β
```

where `E` is an irreducible loss floor (entropy inherent to language/text itself, no model can go below this), and `A, B, α, β` are empirically fit constants. The key qualitative takeaways:

- **Loss decreases smoothly and predictably as you scale up N, D, or C** — there is no plateau observed yet at the scales tested by major labs; more parameters and more data reliably produce lower loss, in a way you can extrapolate before ever training the larger model, which is why labs can plan multi-hundred-million-dollar training runs with reasonable confidence about resulting loss.
- **There is an optimal *ratio* between model size and data size for a fixed compute budget.** The Chinchilla finding specifically corrected an earlier industry tendency to over-invest in parameters relative to data: for a fixed compute budget, many earlier large models (e.g., the original 175B-parameter GPT-3) were significantly under-trained on data relative to their size, and a smaller model trained on proportionally more data would have reached *lower* loss for the *same* compute cost. Chinchilla's finding was roughly that model size and training tokens should scale in tandem (very roughly ~20 tokens per parameter, though exact optimal ratios depend on other assumptions and have been debated/refined since).
- **Lower pretraining loss reliably correlates with better downstream task performance** (question answering, reasoning benchmarks, coding), though the relationship between "loss" and "capability at task X" is not perfectly linear or fully understood — some capabilities appear to emerge fairly abruptly at certain scale thresholds rather than improving smoothly (a phenomenon called "emergence" in the literature, itself somewhat contested as partly a measurement artifact of discontinuous evaluation metrics).
- This is *not* a law of physics — it's a robust empirical regularity within the regimes tested. It does not by itself guarantee indefinite improvement, does not account for data quality differences, and doesn't by itself explain post-training (SFT/alignment) behavior, which follows different, less well-quantified dynamics.

Practically, scaling laws are why frontier labs can make informed bets on training runs costing tens to hundreds of millions of dollars: small-scale experiments at lower `N`/`D`/`C` let you fit the power-law curve and extrapolate expected loss (and, with more uncertainty, expected downstream capability) at the target scale before committing the full budget.

---

## 8. Why this matters for building products

Every concept above has a direct, practical consequence for anyone building software on top of LLMs, not just researchers training them.

- **Context window limits are a real architectural constraint, not an arbitrary product limitation.** Because attention cost is O(n²) and KV cache memory scales with sequence length × layers × heads, "just increase the context window" has real cost and latency consequences at the serving layer. If your product needs to reason over more content than fits in context, you need retrieval (fetch only the relevant chunks) or summarization/compression strategies — this is a direct consequence of Section 6, not a workaround for a missing feature.

- **Cost per token is not incidental — it maps directly to the forward-pass compute described in Section 3.** Every token you send in a prompt, and every token generated in a response, requires a full pass through every layer of the model (embedding, N × [attention + FFN], output projection). Larger models (more layers, larger `d_model`) cost more per token because there is literally more matrix multiplication happening per token. This is why prompt engineering that reduces token count (shorter system prompts, trimmed context, concise instructions) has a direct, linear-ish effect on cost, and why output length matters more than input length for latency (input tokens can be processed in one parallel forward pass; output tokens are generated one at a time, sequentially, per Section 4).

- **Tokenization efficiency affects cost and quality unevenly across languages and content types**, as discussed in Section 2 — a prompt in a language poorly represented in the tokenizer's training data, or unusual content like dense code with lots of symbols, may consume noticeably more tokens (and thus more cost and more of the context budget) than equivalent English prose, for identical semantic content.

- **Smaller, fine-tuned models can beat larger general models on narrow tasks, and understanding why requires distinguishing the training stages in Section 5.** A large general-purpose model's pretraining spends its parameter budget across an enormous breadth of world knowledge, languages, and skills. A smaller model that undergoes targeted SFT (and possibly further alignment) on a narrow task distribution — say, extracting structured data from a specific document format, or writing code against a specific internal API — is having its entire capacity specialized toward that task's specific distribution of inputs and outputs, rather than spreading representational capacity across everything else the larger model also had to learn. This is precisely why SFT is effective with comparatively little data (Section 5.2): it's steering, not building knowledge from scratch, and a smaller model has less "irrelevant" capability to overcome for a narrow task. The practical implication: before defaulting to the largest available model, it's worth asking whether the task is narrow and well-specified enough that a smaller fine-tuned model reaches equivalent or better quality at a fraction of the per-token cost and latency.

- **Sampling parameters (Section 4) are not cosmetic — they're a correctness lever.** A task requiring deterministic, reproducible, or precise output (structured data extraction, code generation, anything checked programmatically) should use low or zero temperature; a task requiring variety (brainstorming, creative writing, generating diverse synthetic training examples) benefits from higher temperature and/or top-p sampling. Treating temperature as a fixed default rather than a per-task decision is a common and avoidable source of product quality issues.

- **The gap between "pretrained" and "aligned" behavior (Section 5) explains a lot of observed model behavior that otherwise looks mysterious** — refusals, hedging, particular response formatting conventions, and sensitivity to how a request is phrased are largely products of the SFT and alignment stages, not the underlying pretrained knowledge. If a model "knows" something (it was almost certainly in the pretraining corpus) but won't say it, or says it only in a particular hedged way, that is alignment-stage behavior, and it is often more tractable to address via prompting or system-level instructions than via the (false) assumption that the underlying knowledge is simply missing.

Understanding the mechanism — tokens, attention, the training pipeline, and the compute/memory constraints those mechanisms impose — turns most "why does the model do that" product questions from mysteries into predictable consequences of a system you can now reason about directly.
