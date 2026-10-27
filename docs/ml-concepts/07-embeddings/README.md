# Embeddings

## 1. What an embedding actually is

An **embedding** is a dense, fixed-length vector of real numbers that
represents a discrete or high-dimensional piece of data — a word, a
sentence, a document, an image, a line of code, a user, a product — as a
point in a continuous vector space, chosen so that **geometric distance in
that space encodes semantic similarity** in the original data.

Concretely: an embedding model is a function

```
f: X -> R^d
```

where `X` is the space of raw inputs (strings, pixels, tokens, whatever)
and `d` is a fixed dimensionality (e.g. 384, 768, 1536, 3072). The function
is learned, not hand-designed, and it is trained so that:

- Inputs that are semantically similar map to vectors that are *close
  together* (small distance / high similarity) in `R^d`.
- Inputs that are semantically unrelated map to vectors that are *far
  apart*.

"Semantically similar" is doing a lot of work in that sentence, and what it
means depends entirely on the training objective. A word2vec embedding
considers two words similar if they appear in similar contexts. A
sentence-transformer embedding considers two sentences similar if humans
labeled them as paraphrases, or if they are a genuine (query, passage) pair
in a retrieval dataset. A CLIP embedding considers an image and a caption
similar if they were scraped together from the same web page. The vector
space's geometry is a direct, learned consequence of whatever pairs of
"similar" and "dissimilar" examples the model was shown during training —
it is not an intrinsic property of language or images that a model simply
"discovers."

Two properties fall out of this that matter for everything downstream:

1. **Fixed dimensionality regardless of input size.** A single word and a
   3-page document can both be represented as a single 768-dimensional
   vector (for a document, via pooling — see §3). This is what makes
   embeddings usable as keys in a vector index: every row has the same
   shape, so distance/similarity between any two rows is always
   well-defined.
2. **The vector's individual coordinates are not directly interpretable.**
   You cannot look at dimension 217 of a word2vec vector and say "this
   measures formality." Meaning is encoded in the *relative geometry* —
   directions and distances relative to other vectors — not in any single
   axis. (There are decomposition techniques that recover interpretable
   directions post hoc, e.g. probing classifiers, but the raw dimensions
   are not designed to be human-readable.)

The canonical illustration is the word2vec analogy result:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

This isn't magic — it falls directly out of how the training objective
shapes the space (see §2.2). The vector offset from "man" to "king"
captures something like a "royalty" direction, and because the space is
linear, adding that same offset to "woman" lands near "queen." This
property does not hold reliably for every relation or every embedding
model, but it's the clearest demonstration that these spaces encode
*structured* semantic relationships, not just a similarity score.

## 2. History and evolution

### 2.1 One-hot encoding — the baseline, and why it fails

Before learned embeddings, the standard way to represent a discrete token
(e.g. a word from a vocabulary) numerically was **one-hot encoding**: given
a vocabulary of size `V`, word `i` is represented as a vector of length `V`
that is all zeros except for a `1` at index `i`.

```
vocab = ["cat", "dog", "car", "the"]

one_hot("cat") = [1, 0, 0, 0]
one_hot("dog") = [0, 1, 0, 0]
one_hot("car") = [0, 0, 1, 0]
one_hot("the") = [0, 0, 0, 1]
```

This fails for two structural reasons:

1. **No notion of similarity.** Every pair of distinct one-hot vectors is
   equidistant from every other pair — the dot product between any two
   *different* one-hot vectors is always exactly `0`, and the Euclidean
   distance between any two is always exactly `sqrt(2)`. "cat" and "dog"
   are just as "far apart" as "cat" and "car," even though the first pair
   is semantically related (animals) and the second is not. The
   representation carries zero information about meaning — it is purely an
   arbitrary index.
2. **Dimensionality equals vocabulary size, and the vectors are maximally
   sparse.** A realistic vocabulary is 30,000–300,000+ tokens. Every
   one-hot vector is a `V`-dimensional vector with a single nonzero entry,
   which is enormously wasteful to store and compute with, and gives a
   downstream model no way to generalize: a model that has seen "the cat
   sat" gets zero transferable signal toward "the dog sat," because "cat"
   and "dog" share no structure in their representation at all.

Everything in the rest of this document is, in one way or another, a
response to these two failures: compress the representation into a
low-dimensional *dense* vector, and choose that vector so that distance
reflects meaning.

### 2.2 Static word embeddings

#### word2vec (Mikolov et al., 2013)

word2vec learns a `d`-dimensional vector per vocabulary word (typically
`d = 100`–`300`) by training a shallow neural network on a proxy task —
predicting words from their local context — and then throwing away the
network and keeping only the learned weight matrix as the embedding table.
There are two architectures:

**CBOW (Continuous Bag of Words)** predicts the *center* word from its
surrounding *context* words. Given a window of size `c`, the input is the
average (or sum) of the context words' embeddings, and the model is trained
to predict the center word:

```
context: [the, ___, sat, on, the]   (window = 2 on each side)
target:  cat

CBOW objective: maximize P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})
```

**Skip-gram** does the reverse: given the center word, predict each context
word independently.

```
center: cat
targets: the, sat, on, the   (predict each context word separately)

Skip-gram objective: maximize P(w_{t-c}, ..., w_{t+c} | w_t)
                    = sum over each context position j:
                      maximize P(w_{t+j} | w_t)
```

Skip-gram tends to work better for rare words (it generates more
training pairs per rare word, since each occurrence produces `2c` training
examples instead of one), while CBOW trains faster and works slightly
better for frequent words. In practice skip-gram (often with negative
sampling) became the more widely used variant.

**The actual training objective.** Naively, `P(context word | center word)`
would be a softmax over the entire vocabulary:

```
P(w_o | w_c) = exp(v'_wo · v_wc) / sum_{w=1}^{V} exp(v'_w · v_wc)
```

This is computationally intractable for large `V` because the denominator
requires a sum over the entire vocabulary for every single training step.
word2vec's key practical contribution is **negative sampling**, which
replaces the full softmax with a much cheaper binary classification
objective: for each true (center, context) pair, sample `k` random
"negative" words that did *not* appear in that context, and train the model
to distinguish the true pair from the negative ones using a sigmoid, not a
softmax:

```
L = -log(sigmoid(v'_wo · v_wc)) - sum_{i=1}^{k} log(sigmoid(-v'_wi · v_wc))
```

where `v_wc` is the input-side embedding of the center word, `v'_wo` is the
output-side embedding of the true context word, and `v'_wi` are output-side
embeddings of `k` sampled negative words (typically `k = 5`–`20`, sampled
proportional to `unigram_frequency^0.75`, an empirically-tuned exponent
that down-weights extremely frequent words like "the" relative to their raw
frequency). Minimizing this loss pushes true (center, context) pairs' dot
products up (toward classifying them as "real") and pushes random
non-co-occurring pairs' dot products down. After training, the input-side
embedding matrix is kept as the word vectors.

The critical structural point: **because words that occur in similar
contexts get pushed toward predicting similar context distributions, they
end up with similar vectors.** "cat" and "dog" both frequently appear near
words like "the," "pet," "food," and "walked," so their embeddings are
pulled toward being close together, even though the model never sees a
single labeled example claiming "cat is similar to dog." This is the
**distributional hypothesis** in action: *a word is characterized by the
company it keeps* (Firth, 1957).

#### GloVe (Pennington et al., 2014)

GloVe (Global Vectors) takes a different route to the same kind of vector:
instead of a predictive local-context task, it directly factorizes a
**global word-word co-occurrence matrix**.

1. Build a co-occurrence matrix `X` where `X_ij` counts how often word `j`
   appears in the context window of word `i`, across the *entire* corpus
   (not just word2vec's local sliding-window training signal — this is a
   corpus-wide statistic computed once, up front).
2. Learn word vectors `w_i`, `w_j` and scalar biases `b_i`, `b_j` such that
   their dot product approximates the log of the co-occurrence count:

```
w_i · w_j + b_i + b_j ≈ log(X_ij)

Loss = sum_{i,j} f(X_ij) * (w_i · w_j + b_i + b_j - log(X_ij))^2
```

where `f(X_ij)` is a weighting function that caps the influence of very
frequent co-occurrences (so that pairs like "the"/"of," which co-occur
enormously often, don't dominate the loss) while still giving near-zero
weight to co-occurrence counts of zero (avoiding `log(0)`).

The conceptual difference from word2vec: word2vec is a **predictive,
local-window** model trained with stochastic gradient descent over
individual (center, context) pairs; GloVe is a **count-based, global**
model that explicitly factorizes an aggregate co-occurrence statistic,
closer in spirit to classical matrix-factorization methods like LSA
(Latent Semantic Analysis). In practice the two produce embeddings of
similar quality and similar analogy-completion behavior, and both are now
mostly of historical/pedagogical interest — superseded by contextual
embeddings for almost every production use case.

#### The shared limitation: one vector per word, no context

Both word2vec and GloVe assign **exactly one fixed vector per word type**,
regardless of context. "bank" gets a single vector that has to somehow
average over "river bank" and "bank account" — it cannot represent either
sense precisely, and it certainly cannot distinguish them at inference
time. This is the central limitation that contextual embeddings solve.

### 2.3 Contextual embeddings

#### ELMo (Peters et al., 2018) — the first major step

ELMo (Embeddings from Language Models) produces a different vector for the
same word depending on its sentence context, by running a **deep
bidirectional LSTM** language model over the input sentence and using the
LSTM's internal hidden states (from multiple layers, combined with learned
weights) as the word's embedding. Because the hidden state at position `t`
is a function of everything the LSTM has seen up to (and, in the backward
pass, after) that position, "bank" in "I sat on the river bank" and "bank"
in "I deposited money at the bank" get genuinely different vectors, each
shaped by their surrounding words. This was the first widely-adopted
demonstration that context-dependent word representations meaningfully
outperform static ones on downstream NLP tasks.

#### Transformer-based contextual embeddings: BERT and successors

BERT (Devlin et al., 2018) replaced ELMo's LSTM backbone with a
**transformer encoder**, trained with a masked-language-modeling objective
(predict randomly masked tokens using bidirectional self-attention over
the full sentence) plus (in the original version) a next-sentence
prediction objective. The mechanism that produces context-sensitivity is
**self-attention**: at every layer, each token's representation is updated
by attending to (taking a weighted combination of) every other token's
representation in the same sequence, with the attention weights computed
fresh for that specific input. There is no fixed per-word lookup table
downstream of the input embedding layer — the representation of "bank" at
the output of the model is literally computed from, and mixed with, the
representations of every other word in that specific sentence.

This is why the same word produces different embedding vectors in
different sentences with a transformer encoder:

```
"I sat on the river bank"        -> embedding("bank") = v1
"I deposited cash at the bank"   -> embedding("bank") = v2

v1 != v2, because self-attention mixes in different neighboring
tokens ("river" vs. "deposited", "cash") when computing bank's
final hidden state at each layer.
```

Each transformer layer only has token-level (sub-word/word-piece) outputs;
there is no single vector automatically representing the whole sentence or
document. Producing a single fixed-size vector for a longer span of text
requires an explicit pooling step, covered next.

### 2.4 Sentence and document embeddings

Given a transformer that outputs one vector per token, there are three
common ways to collapse that sequence of token vectors into a single
fixed-size embedding for the whole input:

**Mean pooling** — average the token vectors (usually only over
non-padding tokens, masked appropriately):

```
sentence_embedding = (1/n) * sum_{i=1}^{n} token_embedding_i
```

This is simple, requires no extra trained parameters, and in practice is a
strong default — it tends to outperform naive `[CLS]`-token pooling from a
model that wasn't specifically fine-tuned for sentence-level similarity.

**`[CLS]` token pooling** — BERT-style models prepend a special `[CLS]`
token to every input, and its final-layer hidden state is sometimes used
directly as the "whole sequence" representation, since `[CLS]` attends to
every other token. However, a big caveat: in *vanilla* BERT, the `[CLS]`
vector was trained only for the next-sentence-prediction objective, not for
producing a similarity-preserving sentence vector — raw `[CLS]` embeddings
from off-the-shelf BERT are known to perform *poorly* on semantic
similarity/retrieval tasks (they tend to cluster together based on
superficial features like sentence length rather than meaning). `[CLS]`
pooling only becomes reliable when the model has been explicitly
fine-tuned with a sentence-level objective that uses that pooled vector
directly (e.g. Sentence-BERT below), because then gradient descent
actually shapes `[CLS]` to be a good whole-sentence summary rather than
being a side effect of an unrelated pretraining task.

**Dedicated sentence-embedding models trained with contrastive loss** —
the current best practice. Rather than pooling the output of a model
trained for something else, you fine-tune (or train from scratch) the
model with an objective that directly optimizes the *pooled* vector for
similarity:

- **Sentence-BERT / SBERT (Reimers & Gurevych, 2019)** fine-tunes a
  BERT-style encoder in a **siamese/twin-tower architecture**: two copies
  of the same encoder (shared weights) each process one sentence, produce a
  pooled embedding (mean pooling, in the original SBERT), and are trained
  jointly using a loss computed over pairs or triples of sentences (e.g.
  cosine-similarity regression against a human similarity label, or a
  triplet loss over (anchor, positive, negative) sentences). This is a
  practical breakthrough for speed: to find the most similar pair among
  10,000 sentences with vanilla BERT you'd need to feed every pair through
  the model jointly (`~50 million` forward passes with cross-attention
  between every pair) because BERT was designed to score sentence *pairs*
  jointly rather than to produce comparable independent vectors; with
  SBERT you embed each of the 10,000 sentences once, then compare vectors
  with a cheap cosine similarity, cutting the effort from tens of millions
  of transformer forward passes to 10,000 forward passes plus vector
  arithmetic.
- **SimCSE (Gao et al., 2021)** uses a much simpler and remarkably
  effective trick for the **unsupervised** case: feed the *same* sentence
  through the encoder twice with standard dropout enabled (dropout is
  usually turned off at inference, but SimCSE deliberately leaves it on).
  Because dropout randomly zeroes different neurons on each of the two
  passes, the two resulting embeddings are for the identical sentence but
  are not numerically identical — they become the "positive pair" for a
  contrastive loss (§7), while other sentences in the same training batch
  serve as negatives. This requires no labeled paraphrase data at all, and
  performs competitively with fully supervised approaches. SimCSE also has
  a supervised variant that uses labeled NLI (natural language inference)
  entailment/contradiction pairs as stronger positive/negative supervision.

Document embeddings (representing a multi-paragraph document, not just a
sentence) are typically produced by one of: (a) pooling over chunk-level
sentence embeddings, (b) truncating/chunking the document and embedding
each chunk separately (the standard approach in RAG pipelines — see
`docs/ml-concepts/02-rag`), or (c) using a long-context embedding model
(e.g. models built on top of long-context transformers) that can directly
accept several thousand tokens and pool over the whole thing. Chunking is
usually preferred in retrieval systems regardless of context-length limits,
because a single embedding vector for a 20-page document averages away too
much information to support fine-grained retrieval — a chunk-per-vector
scheme lets a query match the specific paragraph that's actually relevant.

## 3. Types of embeddings used in practice, and when to use each

| Type | Typical model family | What it represents | Typical use |
|---|---|---|---|
| Word embeddings | word2vec, GloVe, fastText | One vector per vocabulary word, static | Legacy NLP pipelines, lightweight feature inputs, explainability work where per-word vectors are inspected directly; mostly superseded |
| Contextual token embeddings | BERT, RoBERTa, and other encoder-only transformers | One vector per token, context-dependent | Token classification (NER, POS tagging), as a feature extractor inside a larger fine-tuned model |
| Sentence embeddings | Sentence-BERT, SimCSE, E5, GTE, BGE, OpenAI/Cohere embedding APIs | One vector per sentence/short passage | Semantic search, RAG retrieval, deduplication, clustering, semantic textual similarity |
| Document embeddings | Chunked sentence embeddings, long-context embedding models | One vector per chunk or per document | RAG corpora, document clustering, large-scale semantic search over long-form content |
| Image embeddings | CLIP image tower, DINOv2, ResNet/ViT feature extractors | One vector per image | Reverse image search, image deduplication, visual similarity, as input features for a classifier |
| Multimodal embeddings | CLIP, ALIGN, and successors | Text and images mapped into the *same* shared vector space | Text-to-image search ("find images matching this caption"), image-to-text search, zero-shot image classification |
| Code embeddings | CodeBERT, StarCoder-embedding-family, OpenAI/Voyage code-embedding models | One vector per function/file/snippet, trained on code + docstring pairs or code-clone pairs | Code search, "find similar function," duplicate-code detection, code-aware RAG for coding agents |

A few of these deserve more detail:

**CLIP (Radford et al., 2021)** trains two separate encoders — an image
encoder (ViT or ResNet) and a text encoder (transformer) — jointly on a
large corpus of (image, caption) pairs scraped from the web, using a
contrastive objective (§7) that pulls a matching (image, caption) pair's
embeddings together and pushes all other in-batch pairings apart. The
result is that images and text share one embedding space: `embed("a photo
of a dog")` lands close to `embed(<actual photo of a dog>)` even though the
two inputs come from completely different modalities and passed through
completely different network architectures. This is what enables
text-to-image search, image-to-image search, and zero-shot classification
(classify an image by computing which of a set of candidate text-label
embeddings is closest to the image's embedding, with no task-specific
training at all).

**Code embeddings** are trained on objectives specific to source code:
matching a function's body to its docstring/comment, matching semantically
equivalent code across different implementations ("clone detection"), or
contrastive pairs mined from commit history and code-review pairs. A
critical practical property: code embedding models generally need to be
trained (or at least fine-tuned) on code-shaped tokenization, because
natural-language embedding models tokenize identifiers, indentation, and
symbols poorly — `getUserById` and `get_user_by_id` should embed close
together for a code-search use case, which requires vocabulary/tokenizer
choices (or explicit normalization) tuned for code rather than prose.

## 4. Similarity metrics

Once two pieces of data are embedded as vectors `A` and `B` in `R^d`, "how
similar are they" reduces to a purely geometric question. Three metrics
dominate in practice.

### 4.1 Cosine similarity

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

where:
  A · B    = sum_{i=1}^{d} A_i * B_i          (dot product)
  ||A||    = sqrt(sum_{i=1}^{d} A_i^2)        (L2 norm / magnitude of A)
```

Cosine similarity measures the **angle** between two vectors, ignoring
their magnitude entirely: it ranges from `-1` (pointing in exactly opposite
directions) through `0` (orthogonal) to `1` (pointing in exactly the same
direction, regardless of length). Cosine *distance*, when needed as an
actual distance metric, is usually defined as `1 - cosine_similarity`.

### 4.2 Dot product

```
dot_product(A, B) = A · B = sum_{i=1}^{d} A_i * B_i
```

The dot product is cosine similarity *without* the normalization — it is
sensitive to both the angle between the vectors and their magnitudes. Two
vectors pointing the same direction but with large magnitude will have a
larger dot product than two identical-direction vectors with small
magnitude, even though their cosine similarity is identical.

### 4.3 Euclidean distance (L2 distance)

```
euclidean_distance(A, B) = sqrt(sum_{i=1}^{d} (A_i - B_i)^2)
```

This measures straight-line distance between the two points in `R^d`,
independent of angle in isolation — but note it is *not* independent of
magnitude either: two vectors with the same direction but very different
lengths will have a large Euclidean distance despite being "similar" in
the cosine sense.

### 4.4 Why cosine similarity is preferred for high-dimensional, normalized embeddings

Three reasons, in order of practical importance:

1. **Magnitude is often an artifact, not a signal.** For many embedding
   models, a vector's magnitude correlates with things like input length,
   token frequency, or numerical scale of the training process, rather
   than with anything semantically meaningful. If magnitude carried no
   real semantic information, using a metric that's sensitive to magnitude
   (dot product or Euclidean distance) would inject noise into the
   similarity signal. Cosine similarity sidesteps this by comparing only
   direction.
2. **If embeddings are pre-normalized to unit length (`||A|| = 1`), cosine
   similarity, dot product, and Euclidean distance all become monotonic
   transformations of each other** — so the choice stops being a modeling
   concern and becomes purely an engineering/indexing convenience. The
   algebraic identity for unit vectors:

```
If ||A|| = ||B|| = 1, then:
  ||A - B||^2 = ||A||^2 + ||B||^2 - 2*(A · B) = 2 - 2*(A · B)

=> euclidean_distance(A, B) = sqrt(2 - 2 * cosine_similarity(A, B))
=> dot_product(A, B) = cosine_similarity(A, B)   (since norms are both 1)
```

   This is why so many production embedding models (Sentence-BERT
   variants, OpenAI's `text-embedding-*` family, most open sentence
   encoders) explicitly L2-normalize their output vectors: it lets a
   vector database use the cheapest metric to compute (raw dot product,
   which is a single fused multiply-add pass with no square-root or
   division) while still getting cosine-similarity *rankings* out of it,
   since ranking by dot product on unit vectors produces the same order
   as ranking by cosine similarity.
3. **High-dimensional geometry makes raw distances less discriminative.**
   As dimensionality `d` grows, random high-dimensional vectors tend to
   become nearly orthogonal to each other and pairwise Euclidean distances
   between random points concentrate toward a similar value (a
   consequence of the well-known "curse of dimensionality" phenomenon) —
   which flattens the contrast between "close" and "far" points under an
   unnormalized metric. Normalizing away magnitude and focusing purely on
   angular relationships tends to preserve a more usable, higher-contrast
   similarity signal in these high-dimensional embedding spaces (commonly
   500–3000+ dimensions in modern models).

In short: use **dot product on L2-normalized vectors** as the fast path in
production (this is what most ANN libraries optimize for), understanding
that it is mathematically equivalent to ranking by cosine similarity; use
**Euclidean distance** when the embedding space is explicitly designed to
be metric (e.g. some clustering-oriented or geometric embeddings where
scale is meaningful, or for certain image-embedding use cases); use raw,
unnormalized dot product only when you have a specific reason to want
magnitude to matter (e.g. some recommendation-system embeddings deliberately
encode "popularity" or "confidence" as vector magnitude).

## 5. Embeddings for retrieval: vector databases and ANN search

### 5.1 The retrieval pipeline at a glance

```
+-----------+     +-----------+     +---------+     +---------+     +------------------+
|  Raw text  | --> | Tokenizer | --> | Encoder | --> | Pooling | --> | Embedding vector |
| "How do I |     | [101,2129,|     |(transf- |     | (mean / |     |  [0.021, -0.114, |
|  reset my |     |  2079,...]|     | ormer   |     |  CLS /  |     |   0.302, ...,    |
|  password?"     |           |     | layers) |     |  last-  |     |   0.008]  (d=768)|
+-----------+     +-----------+     +---------+     |  token) |     +------------------+
                                                     +---------+              |
                                                                              v
                                          +----------------------------------------+
                                          |             Vector Database             |
                                          |  (stores embedding + ANN index + the    |
                                          |   original text/metadata as a payload)  |
                                          +----------------------------------------+
                                                              |
                          query embedding  -----------------> |
                     (same tokenizer + encoder + pooling!)    |
                                                              v
                                          +----------------------------------------+
                                          |     Approximate Nearest Neighbor (ANN)  |
                                          |     similarity search (HNSW / IVF)      |
                                          |     -> top-k most similar vectors       |
                                          +----------------------------------------+
                                                              |
                                                              v
                                          +----------------------------------------+
                                          |  Retrieved passages, ranked by score,   |
                                          |  returned to the RAG pipeline / caller  |
                                          +----------------------------------------+
```

At index time, every document/chunk is pushed through
tokenizer -> encoder -> pooling to produce one vector per chunk, and those
vectors (plus the original text and any metadata) are stored in a **vector
database**. At query time, the *incoming query string* is pushed through
the exact same pipeline to produce a single query vector, and the vector
database returns the `k` stored vectors most similar to it (by whichever
metric from §4 the index was built with), along with their associated
payload (the original text chunk).

### 5.2 Why approximate nearest neighbor (ANN) search, not exact search

Exact nearest-neighbor search (compute the similarity of the query vector
against *every* stored vector, then sort) is `O(N * d)` per query, where
`N` is the number of stored vectors and `d` is the embedding dimension.
This is completely fine for `N` in the thousands, but breaks down at the
scale real retrieval systems operate at (millions to billions of vectors),
where a linear scan per query becomes too slow for interactive use. **ANN
algorithms trade a small amount of recall (they may occasionally miss the
true nearest neighbor) for a large speedup**, typically bringing query time
down to sub-linear, often logarithmic-ish in practice, at the cost of
building an index structure ahead of time.

**HNSW (Hierarchical Navigable Small World graphs)** builds a
multi-layer graph over the stored vectors. Each vector is a node; nodes are
connected to their approximate near neighbors. The top layer is sparse (few
nodes, long-range links), and each layer below is progressively denser,
down to the bottom layer, which contains every vector. A query is answered
by starting at an entry point in the top (sparsest) layer, greedily walking
to whichever neighbor is closest to the query vector, and once no
neighbor at that layer improves the distance, dropping down one layer and
repeating — like a skip-list, but for a similarity graph instead of a
sorted list. This gives you coarse, fast navigation at the top and
fine-grained refinement near the bottom, so you can navigate a huge index
in a small number of hops. HNSW generally gives excellent recall/latency
tradeoffs and doesn't require the index to be "trained" against the data
distribution up front, but it is memory-hungry (the graph structure itself,
with multiple links per node across multiple layers, sits on top of the
raw vectors) and index construction is nontrivially expensive to update
incrementally at very large scale.

**IVF (Inverted File Index)** takes a different approach: it first
**clusters** the vector space (typically with k-means) into `nlist`
partitions ("cells"), each represented by a centroid. Every stored vector
is assigned to its nearest centroid's cell. At query time, instead of
comparing the query against every stored vector, the search first finds
the `nprobe` closest centroids to the query vector, and then does an exact
(or further-approximated) search only within those `nprobe` cells. This
avoids scanning the entire dataset while still checking every vector
within the small number of relevant clusters, at the cost of an
occasional true nearest neighbor sitting just across a cell boundary in a
cluster that wasn't probed (mitigated by increasing `nprobe` — a direct
recall/latency dial). IVF requires a training step (fitting the k-means
centroids) before insertion, and is frequently combined with a compression
technique like Product Quantization (IVF-PQ) to shrink memory footprint
by encoding each vector as a small set of quantized sub-vector codes
instead of storing full-precision floats — a common choice at billion-scale
where HNSW's memory overhead becomes prohibitive.

The high-level tradeoff: HNSW generally wins on recall and query latency
for a given memory budget at small-to-medium scale, and is the default in
many modern vector databases (Qdrant, Weaviate, Milvus's HNSW mode,
pgvector's HNSW index type); IVF (and IVF-PQ) tends to win on raw memory
efficiency and is more common at very large scale where the full dataset
plus a graph's worth of link pointers cannot fit affordably in memory. Many
real deployments combine ideas from both (e.g. IVF for coarse partitioning
plus HNSW or PQ within cells).

### 5.3 Embedding dimensionality tradeoffs

Choosing an embedding dimension `d` is a genuine design tradeoff, not
"bigger is always better":

- **Retrieval quality** generally improves with `d` up to a point — more
  dimensions can encode finer-grained distinctions — but returns diminish,
  and past a model-specific point, added dimensions mostly encode noise or
  redundant signal rather than new discriminative information.
- **Storage cost scales linearly with `d`.** A million 1536-dimensional
  float32 vectors is `1,000,000 * 1536 * 4 bytes ≈ 6.1 GB` just for the raw
  vectors, before any index overhead; the same corpus at 384 dimensions is
  `≈ 1.5 GB`. This matters directly for cost at scale, and for how much of
  the index can be kept in RAM (which matters enormously for query
  latency).
- **ANN index build time and query latency both scale with `d`** — every
  distance computation is `O(d)`, so doubling `d` roughly doubles the cost
  of every single similarity comparison, and HNSW graph construction cost
  scales accordingly too.
- **Practical response: Matryoshka Representation Learning (MRL)-style
  models** (e.g. OpenAI's `text-embedding-3` family, several recent open
  sentence-transformer releases) are explicitly trained so that
  *truncating* the vector to a shorter prefix (e.g. keeping only the first
  256 of 1536 dimensions) still produces a usable, well-ordered embedding,
  letting a system choose a smaller `d` for storage/speed without
  retraining or re-encoding a completely different smaller model. This is
  increasingly the practical way teams navigate the dimensionality
  tradeoff: pick a large `d` at training/encoding time, then truncate
  per-deployment based on the actual latency/storage budget.

## 6. Training an embedding model: contrastive learning

Modern sentence, image, and multimodal embedding models are almost all
trained with some form of **contrastive learning**: the model is not told
"this is the correct absolute vector for this sentence" (there's no such
ground truth) — instead, it's shown pairs or groups of examples and trained
so that vectors of items known to be related end up close together, and
vectors of items known to be unrelated end up far apart.

### 6.1 Positive and negative pairs

Every contrastive training example is built from:

- An **anchor** — the reference item (e.g. a search query, or one view of
  a sentence).
- A **positive** — an item known (by construction, or by human/heuristic
  label) to be semantically related to the anchor (e.g. the passage that
  actually answers the query; a paraphrase of the anchor sentence; the
  caption that was scraped alongside the anchor image).
- One or more **negatives** — items known (or assumed) to be unrelated to
  the anchor.

Where these pairs come from varies by domain: for sentence embeddings, NLI
datasets provide labeled entailment (positive) / contradiction (negative)
pairs; for retrieval-specific embedding models, (query, relevant passage)
pairs come from search logs or QA datasets; for CLIP, (image, caption)
pairs come from co-occurrence on the same web page; for SimCSE's
unsupervised variant, the "positive" is the same sentence run through the
encoder twice with different dropout masks (§2.4).

### 6.2 The InfoNCE loss

The dominant loss function for this setting is **InfoNCE** (Information
Noise-Contrastive Estimation), sometimes seen under the name "multiple
negatives ranking loss" in sentence-embedding training code. For a batch
containing an anchor `a`, its positive `p`, and a set of negatives
`{n_1, ..., n_K}` (often simply every *other* positive example in the same
training batch — "in-batch negatives" — rather than separately curated
ones):

```
InfoNCE loss for anchor a:

L_a = -log( exp(sim(a, p) / τ) / sum_{j in {p, n_1, ..., n_K}} exp(sim(a, j) / τ) )
```

where `sim(x, y)` is a similarity function (almost always cosine
similarity or dot product on normalized vectors, §4) and `τ` (tau) is a
learned or fixed **temperature** hyperparameter that controls how sharply
the softmax distinguishes close scores — a smaller `τ` makes the loss
punish near-miss negatives more aggressively.

Reading the formula operationally: it is a softmax classification loss
where the "correct class" is the positive `p` among the full set of
candidates `{p, n_1, ..., n_K}`. Minimizing it simultaneously does two
things through gradient descent: it pushes `sim(a, p)` up (numerator grows)
and pushes every `sim(a, n_i)` down (competing terms in the denominator
shrink relative to the numerator). Averaged over a whole training set, the
net effect is exactly the geometric structure described in §1: related
things get pulled together, unrelated things get pushed apart.

**Why in-batch negatives are attractive in practice**: if a training batch
contains `B` (anchor, positive) pairs, every anchor's own positive is
paired with it, and *every other pair's positive* in the same batch can be
reused as a free negative for this anchor — with no additional forward
passes needed. This turns a batch of `B` pairs into `B` training instances
each with `B - 1` negatives, essentially for free, which is why larger
batch sizes tend to directly improve contrastive-trained embedding quality
(more negatives per step means a harder, more informative classification
problem for the loss to optimize).

### 6.3 Hard negative mining

Random or in-batch negatives are usually *easy* — a randomly chosen
unrelated sentence is often trivially distinguishable from the true
positive, so the gradient signal it provides is weak once the model has
learned the coarse structure of the space. **Hard negatives** — examples
that are superficially or lexically similar to the anchor but are not
actually the correct match — provide a much stronger training signal
because they sit close to the decision boundary the model still needs to
sharpen.

Example: for the query "how to reset a Python virtual environment," an
easy negative might be a completely unrelated passage about cooking
recipes (already far away in embedding space, contributes almost no
gradient once the model is minimally competent). A **hard negative** would
be a passage about "how to *create* a Python virtual environment" — highly
lexically overlapping, topically adjacent, and genuinely wrong as an
answer to a "reset" question. Training on hard negatives forces the model
to encode a finer-grained distinction than surface word overlap.

Common ways hard negatives are mined in practice:

- **BM25 / lexical-retrieval mining**: run a classical sparse-retrieval
  method (BM25) over the corpus for each anchor, and take its high-ranking
  but *not actually labeled relevant* results as hard negatives — these
  are guaranteed to be lexically close but not the true positive.
- **Mining from a weaker or earlier embedding model**: encode the whole
  corpus with a prior-generation embedding model, retrieve its top-k
  nearest neighbors for each anchor, and use ones that aren't the true
  positive as hard negatives (sometimes called "ANN mining"). This is
  iterative in practice — many production embedding models are trained in
  rounds, where each round's hard negatives are mined using the *previous*
  round's checkpoint.
- **Cross-encoder-scored mining**: use a more expensive but more accurate
  cross-encoder re-ranker model (which jointly encodes the (query,
  passage) pair with full cross-attention, rather than the two-tower
  independent-encoding setup embedding models use) to score candidate
  passages, and select negatives that the cross-encoder confidently
  scores as low-relevance despite superficial similarity.

A caution worth naming explicitly: hard negative mining can introduce
**false negatives** — a mined "hard negative" might actually be a valid
answer that simply wasn't labeled as relevant in the source dataset. This
is a known failure mode in retrieval training pipelines and is typically
mitigated by filtering out mined negatives that score above some
similarity threshold against the anchor, or by periodic human/LLM-based
auditing of a training-negative sample.

## 7. Embedding drift and the same-model requirement

A subtlety that causes real production bugs: **the embedding vector for a
given text is only meaningful relative to other vectors produced by the
exact same model** (same architecture, same trained weights, same
tokenizer, same pooling strategy, and ideally the same normalization
step). There is no universal, model-independent coordinate system for
"meaning" — each model learns its own private geometry during training,
shaped by its own architecture and training data. Two different embedding
models — even two versions of the *same* model family — place semantically
identical sentences at completely different, incomparable coordinates.

This has one direct operational consequence that gets violated
surprisingly often in real systems: **you must use the identical embedding
model to encode both the corpus at index time and every query at search
time.** If a vector database was populated using `model-v1` and a later
code change switches the query-time embedding call to `model-v2` (a
version bump, a provider migration, even a seemingly "compatible"
fine-tune of the same base model) without re-indexing the corpus, cosine
similarity between the new query vectors and the old stored vectors becomes
**meaningless** — the two sets of vectors live in unrelated, un-aligned
coordinate systems, even if both models happen to output the same `d`.
Similarity scores won't error out or obviously fail; they'll just quietly
return poor, semantically nonsensical rankings, which is a much more
dangerous failure mode than a crash because it's easy to miss in testing
and shows up as "retrieval quality silently got worse" rather than as a
loud bug.

This phenomenon and its close cousin are usually both called **embedding
drift**:

- **Model-swap drift**: exactly the scenario above — an index built with
  one model version being queried with a different model version.
- **Distributional drift**: even holding the model fixed, if the
  *distribution* of incoming queries/documents shifts substantially over
  time from what the model was trained/evaluated on (new vocabulary, new
  domain jargon, a product pivoting into a new subject area), retrieval
  quality can degrade gradually even though the model itself never
  changed — because the model's learned geometry may not have accurate,
  well-separated regions for the new kind of content.

Practical mitigations used in real retrieval systems:

- **Pin and version the embedding model explicitly** as part of the vector
  database's schema/metadata — store which model (including exact version
  string / checkpoint hash) produced each stored vector, and refuse to
  query an index with a mismatched model without a full re-index.
- **Full re-index on every embedding model change.** There is no
  incremental migration path between two different models' vector spaces —
  every stored vector must be re-computed from the original source text
  with the new model. This is often the single most expensive step in an
  embedding-model upgrade project, and pipelines should be designed
  (keeping original raw text alongside vectors, not just the vectors) so
  that this re-embedding is actually possible.
- **Blue/green index migration**: build the new index fully under the new
  model in parallel with the old one still serving live queries, validate
  retrieval quality against the new index offline, and cut traffic over
  only once the new index is confirmed healthy — rather than mutating a
  live index in place, which risks a window where old and new vectors
  coexist in the same index and get compared against each other.
- **Track retrieval-quality metrics over time** (e.g. recall@k against a
  held-out labeled query set, or click-through/answer-acceptance rate in
  production) specifically so that distributional drift shows up as a
  measurable trend rather than only being noticed anecdotally after user
  complaints.

## 8. Summary

An embedding is a learned function from discrete, high-dimensional, or
unstructured data into a fixed-size dense vector, engineered so that
distance in that vector space tracks semantic similarity. The field's
history is a series of responses to the previous approach's core
limitation: one-hot encoding has no similarity structure at all and is
absurdly high-dimensional and sparse; static word embeddings (word2vec,
GloVe) fix that but assign one fixed vector per word regardless of
context; contextual embeddings (ELMo, then transformer encoders like
BERT) fix that by making every token's vector a function of its full
surrounding context; and sentence embeddings built with contrastive
objectives (Sentence-BERT, SimCSE, and the current generation of
retrieval-tuned encoder models) fix the remaining gap by training the
*pooled* representation directly for similarity rather than treating
pooling as an afterthought bolted onto a model trained for something else.
In production, the choice of similarity metric, embedding dimensionality,
and ANN index structure are genuine engineering tradeoffs with real cost
and latency consequences, and the single most common way these systems
silently break is a mismatch between the model used to build the index and
the model used to query it — a discipline (version pinning, full re-index
on model change) that matters as much as getting the modeling choices
right in the first place.
