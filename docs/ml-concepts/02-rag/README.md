# Retrieval-Augmented Generation (RAG)

## Table of Contents

1. [What RAG is and why it exists](#1-what-rag-is-and-why-it-exists)
2. [The full pipeline, stage by stage](#2-the-full-pipeline-stage-by-stage)
3. [Full pipeline diagram](#3-full-pipeline-diagram)
4. [Naive RAG vs. intermediate RAG patterns](#4-naive-rag-vs-intermediate-rag-patterns)
5. [Failure modes](#5-failure-modes)
6. [Evaluation](#6-evaluation)
7. [RAG vs. fine-tuning vs. long-context](#7-rag-vs-fine-tuning-vs-long-context)
8. [Where to go deeper](#8-where-to-go-deeper)

---

## 1. What RAG is and why it exists

A large language model's knowledge is **parametric**: everything it "knows" is baked into its weights during pretraining, at a fixed cost, at a fixed point in time. This has three structural consequences that show up immediately in production:

- **Knowledge cutoff.** The model has no idea about anything that happened after its training data was collected. It cannot answer "what changed in our API last week" because "last week" doesn't exist in its weights.
- **Staleness.** Even facts the model once knew can become wrong (a company's pricing, an employee's role, a library's current API) and there is no way to patch a single fact without retraining or fine-tuning the whole model.
- **Hallucination under uncertainty.** When asked about something outside its training distribution — a private codebase, an internal wiki, a niche or recent fact — the model does not reliably say "I don't know." It produces the most statistically plausible continuation of the prompt, which is often a fluent, confident, and wrong answer.

Retrieval-Augmented Generation (RAG) addresses all three by keeping the model's parametric weights untouched and instead giving it **non-parametric memory at inference time**: an external corpus of documents that is searched at query time, with the most relevant chunks inserted directly into the prompt as context. The model is then asked to answer *using that context*, rather than solely from what it memorized during training.

This reframes the model's job from "recall a fact" to "read this passage and answer the question" — a much easier and more reliable task, because reading comprehension over a short, relevant passage is something LLMs are very good at, whereas long-tail factual recall from parametric memory is exactly where they are weakest.

The practical benefits fall out of this reframing directly:

- **No retraining to add or update knowledge.** Add a new document to the index and it's immediately queryable. Delete or edit a document and outdated information stops being retrievable. This is the single biggest operational win over fine-tuning: knowledge updates become a data-engineering problem (keep the index fresh) instead of an ML problem (retrain a model).
- **Grounding and attribution.** Because the model is shown the source text, the system can cite which document(s) an answer came from, and a human can verify the answer against the source — something a purely parametric answer can never offer.
- **Reduced (not eliminated) hallucination.** When the retrieved context actually contains the answer, and the model is instructed to answer only from context, ungrounded fabrication drops substantially. This is not a guarantee — see [Section 5](#5-failure-modes) — but it is a real, measurable improvement.
- **Access to private/proprietary/long-tail data.** A model can answer questions about your internal Confluence pages, ticket history, or codebase without ever having seen them during pretraining, and without exposing that data to a third party by fine-tuning a hosted model on it.

The tradeoff is architectural complexity: RAG turns a single LLM call into a multi-stage system (ingest, chunk, embed, index, retrieve, rerank, assemble, generate), and every stage is a place where errors compound. Most of the engineering effort in a mature RAG system is spent on the retrieval half of this pipeline, not the generation half — a point that will recur throughout this document.

---

## 2. The full pipeline, stage by stage

RAG systems have two distinct temporal phases that most engineers new to RAG conflate: an **indexing (offline) phase**, which runs once per document and again whenever documents change, and a **query (online) phase**, which runs on every user request. Understanding which stage belongs to which phase matters because indexing-time cost can be amortized (you can afford an expensive chunking or embedding step if it happens once), while query-time cost is paid on every request and directly affects latency.

### 2.1 Document ingestion

Raw sources — PDFs, HTML pages, Word docs, Markdown, Slack exports, database rows, code files — are pulled in and normalized to plain text (or text + structural metadata). This stage is unglamorous but is where a huge fraction of real-world RAG quality is won or lost:

- **PDF extraction** must handle multi-column layouts, tables, headers/footers, and OCR for scanned documents. A naive PDF-to-text conversion frequently interleaves columns or drops table structure, silently corrupting the very content you're about to chunk and embed.
- **Metadata extraction** (title, author, section headings, timestamps, source URL, permissions/ACLs) should be preserved and attached to each resulting chunk, because it is often needed later for filtering (e.g., "only search documents the current user has permission to see") and for citation.
- **Deduplication** matters when the same document exists in multiple systems (a wiki page mirrored into Slack, a PDF attached to five different tickets) — otherwise near-duplicate chunks flood the top-k results with the same content.

### 2.2 Chunking strategies

An LLM's context window and a vector index's per-vector granularity both force you to split documents into smaller pieces ("chunks") before embedding. Chunking is arguably the highest-leverage, most underrated design decision in the entire pipeline, because **the chunk is the atomic unit of retrieval** — you can only retrieve whole chunks, never sub-chunk spans, so whatever the chunk fails to contain cannot be retrieved by it, and whatever irrelevant content the chunk *does* contain dilutes its embedding and adds noise to the LLM's context.

**Fixed-size chunking.** Split the raw text every *N* characters or tokens (e.g., every 512 tokens), typically without regard to sentence or paragraph boundaries. This is the simplest strategy and the fastest to implement, but it routinely cuts sentences and even words in half at chunk boundaries, which both degrades the embedding (a truncated sentence embeds to something that doesn't match a full-sentence query) and produces a "chunk boundary problem" for generation (see [Section 5.3](#53-chunk-boundary-problems)).

**Recursive character/text splitting.** Instead of a single fixed split point, recursively try a prioritized list of separators — first paragraph breaks (`\n\n`), then sentence breaks, then word breaks, then characters — until each resulting piece is under the target size. This is the de facto default in most RAG frameworks (e.g., LangChain's `RecursiveCharacterTextSplitter`) because it respects natural document structure whenever possible while still guaranteeing a hard size ceiling. It is a heuristic improvement over fixed-size chunking, not a semantic one: it still has no notion of *topic*, only of *punctuation*.

**Semantic chunking.** Split based on meaning rather than character count: compute sentence-level (or small-window) embeddings across the document, then cut a chunk boundary wherever the embedding similarity between consecutive sentences/windows drops below a threshold (a "semantic discontinuity"). The idea is that a chunk should correspond to one coherent idea, not an arbitrary character count. This produces variable-length chunks that align with topic shifts, at the cost of an extra embedding pass over the whole document at index time and more implementation complexity (threshold tuning, degenerate cases where similarity never dips).

**Sliding window with overlap.** Regardless of which splitting strategy is used, chunks are usually generated with an **overlap** — e.g., 512-token chunks with a 50-token overlap between consecutive chunks — so that content near a boundary appears in full in at least one chunk even if it's split across two. Overlap directly mitigates the chunk-boundary problem at the cost of redundant storage and redundant (near-duplicate) retrieval candidates.

**Chunk size tradeoffs, explicitly:**

| Chunk size | Embedding precision | Recall of relevant facts | Context noise | Cost |
|---|---|---|---|---|
| Small (e.g., 100–200 tokens) | High — embedding represents one narrow idea, so it matches narrow queries precisely | Lower — a fact's supporting context (the sentence before/after that disambiguates it) may live in a different chunk and not get retrieved | Low — little irrelevant text makes it into the LLM's context | More chunks → more vectors to store/search, more re-ranker calls |
| Large (e.g., 1000+ tokens) | Lower — the embedding is an average over many different ideas, so it's a blurrier match to any single query, hurting retrieval precision | Higher per-chunk, since more surrounding context is present | High — retrieved chunks bring in a lot of tangential text, pushing the true answer further from where the LLM attends best (see lost-in-the-middle) | Fewer chunks, but each one consumes more of the LLM's context budget |

There is no universal "correct" chunk size — it is workload-dependent. Dense technical FAQs favor small chunks close to a single Q/A pair; long-form narrative or legal documents where meaning depends on surrounding clauses favor larger chunks or explicit overlap. In practice, teams tune chunk size empirically against a retrieval evaluation set (see [Section 6](#6-evaluation)) rather than picking it a priori.

### 2.3 Embedding

Each chunk is passed through an embedding model — a neural network trained (typically via contrastive learning) to map text into a fixed-dimensionality dense vector such that semantically similar texts land close together in vector space (by cosine similarity or dot product) and dissimilar texts land far apart. Common choices are dedicated embedding models (OpenAI's `text-embedding-3`, Cohere's `embed-v3`, open models like BGE, E5, or GTE) rather than generation models — embedding and generation are different objectives, and a model purpose-trained for retrieval consistently outperforms an ad-hoc embedding taken from a generative model's hidden states.

Key properties that matter operationally:

- **Dimensionality** (e.g., 384, 768, 1536, 3072) trades off representational fidelity against storage and search cost — doubling dimensionality roughly doubles both vector storage size and the cost of the distance computation at query time.
- **Symmetric vs. asymmetric embedding.** Some models embed a search query and a document with the same function ("symmetric," suited to sentence-similarity tasks); most modern retrieval models are trained **asymmetrically** with distinct query/passage encoding (often signaled via an instruction prefix like `"query: ..."` vs. `"passage: ..."`), because a question and its answer are not phrased the same way and a symmetric model would systematically under-match them. Using the wrong prefix, or a symmetric model for an asymmetric task, is a common silent source of poor retrieval.
- **Domain shift.** An embedding model trained mostly on general web/Wikipedia text will underperform on domain-specific corpora (legal, medical, internal codebases) where the surface vocabulary and the semantic relationships between terms differ from its training distribution. This is the root of the *retrieval mismatch* failure mode covered in [Section 5.1](#51-retrieval-mismatch-embedding-space-misalignment).

The output is one dense vector per chunk, which is what gets stored in the index.

### 2.4 Indexing into a vector store

The chunk embeddings (plus the chunk text and metadata) are written into a vector store — a system built to answer nearest-neighbor queries over high-dimensional vectors efficiently. Exact nearest-neighbor search is `O(n)` per query, which is too slow at scale, so production vector stores use **Approximate Nearest Neighbor (ANN)** indexing structures:

- **HNSW (Hierarchical Navigable Small World graphs)** — builds a multi-layer graph where each vector is a node connected to its approximate neighbors; search greedily descends through layers. This is the dominant algorithm in most modern vector databases (Pinecone, Weaviate, Qdrant, Milvus, pgvector) because it gives an excellent recall/latency tradeoff.
- **IVF (Inverted File Index)** — clusters vectors (e.g., via k-means) and restricts search to the nearest clusters (`nprobe` of them) instead of the whole index; used heavily in FAISS.
- **Product Quantization (PQ)** — compresses vectors by splitting each into sub-vectors and quantizing each sub-vector independently against a small codebook, trading some accuracy for large memory savings; often layered on top of IVF (IVF-PQ) for billion-scale indexes.

All of these are **approximate**: they trade a small amount of recall for orders-of-magnitude faster search, and every ANN index exposes tunable parameters (`ef_search`/`M` for HNSW, `nprobe` for IVF) that let you move along that recall/latency curve. This is why "we use a vector database" is not a complete answer to "how does retrieval work" — the index's approximation parameters materially affect whether the truly-best-matching chunk is even a candidate for retrieval.

Alongside the dense vector index, most production systems also build a **sparse/lexical index** (an inverted index over terms, as used by BM25) over the same chunks, because exact term matching solves problems that dense embeddings handle poorly — see hybrid search in [Section 4.1](#41-hybrid-search-sparse--dense-with-reciprocal-rank-fusion).

This entire ingestion → chunk → embed → index sequence happens **offline**, ahead of any user query, and is re-run incrementally whenever documents are added, updated, or deleted.

### 2.5 Query time: embedding, retrieval, re-ranking, and generation

At inference time, a fresh sequence runs for every user request:

**Step 1 — Embed the query.** The user's query is passed through the *same* embedding model used at indexing time (with the query-side prefix/instruction if the model is asymmetric). Using a different embedding model, or a different version of the same model, for queries than was used for documents is a subtle but common bug: the two vector spaces are not guaranteed to be comparable, and similarity scores become meaningless.

**Step 2 — Similarity search (top-k retrieval).** The query vector is compared against the indexed chunk vectors using cosine similarity or dot product, and the *k* nearest chunks are returned (top-k retrieval, e.g., k=20 or k=50 at this stage — deliberately larger than what will ultimately be shown to the LLM, to give the next stage material to work with). This is a **bi-encoder** retrieval step: query and documents were embedded independently and never interact with each other, which is what makes it fast enough to search millions of vectors in milliseconds, but also caps its precision (see the discussion of two-stage retrieval below).

**Step 3 — Optional re-ranking.** The top-k candidates from step 2 are re-scored by a **cross-encoder** re-ranker: a model that takes the *(query, chunk)* pair concatenated together as a single input and outputs a relevance score directly, rather than comparing two independently-computed vectors. Because the cross-encoder attends jointly over the query and the document tokens together, it can capture fine-grained interactions (negation, word order, exact term overlap, argument structure) that two separately-embedded vectors compressed into a fixed-size representation necessarily lose. The top *n* (n << k, e.g., n=5) chunks by cross-encoder score are kept.

**Why a two-stage retrieve-then-rerank pipeline improves precision, concretely:** a bi-encoder must compress an entire chunk's meaning into one fixed-size vector *before ever seeing the query*, so it cannot ask "does this specific chunk answer this specific question" — it can only ask "is this chunk's precomputed vector geometrically close to this query's precomputed vector," which is a lossy proxy. A cross-encoder pays no such compression penalty because it processes the query and document jointly, at the cost of being far more computationally expensive (it requires one full forward pass per candidate pair, so it cannot be run against millions of documents — only against the small shortlist a bi-encoder has already narrowed down). The two-stage design gets the best of both: bi-encoder search is cheap enough to scan a huge corpus and recall a candidate set that's likely to contain the answer, and the cross-encoder is precise enough to correctly rank that small candidate set, fixing exactly the ordering mistakes a bi-encoder alone would make. Empirically, this consistently moves the true best chunk from somewhere in the top-20 up into the top-3–5 that actually get shown to the LLM.

**Step 4 — Context assembly / prompt construction.** The final *n* chunks are formatted into the LLM's prompt, typically with:
- A system instruction that tells the model to answer using only the provided context, and to say when the context doesn't contain the answer.
- Each chunk clearly delimited (e.g., wrapped in `<document>` tags or numbered) and usually accompanied by source metadata (title, URL, section) so the model can cite it.
- The chunk ordering considered deliberately, not left to retrieval-score order, because of the lost-in-the-middle effect (see [Section 5.2](#52-lost-in-the-middle)) — a common mitigation is to put the highest-scoring chunk(s) at both the very start and the very end of the context block.
- The user's original query placed after the context.

**Step 5 — LLM generation.** The LLM produces its answer conditioned on the assembled prompt. Because the answer is now conditioned on retrieved text rather than solely on parametric memory, the generation step becomes closer to a reading-comprehension / summarization task than an open-ended recall task — this is the entire mechanism by which RAG reduces hallucination.

---

## 3. Full pipeline diagram

```
                                   INDEXING PATH (offline, runs when documents change)
 ┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌───────────────┐
 │  Raw docs│──▶│ Ingestion │──▶│   Chunking   │──▶│ Embedding │──▶│  Vector store │
 │ PDFs,HTML│   │ parse,    │   │ fixed-size / │   │  model    │   │  (HNSW/IVF)   │
 │ wiki, DBs│   │ clean,    │   │ recursive /  │   │ chunk ──▶ │   │  + metadata   │
 │          │   │ dedupe,   │   │ semantic,    │   │  vector   │   │  + optional   │
 │          │   │ metadata  │   │ w/ overlap   │   │           │   │  BM25 index   │
 └──────────┘   └───────────┘   └──────────────┘   └───────────┘   └───────┬───────┘
                                                                            │
                                                                            │ indexed once,
                                                                            │ queried many times
                                                                            ▼
                                    QUERY PATH (online, runs on every request)
 ┌──────────┐   ┌───────────┐   ┌────────────────────┐   ┌─────────────────┐
 │   User   │──▶│  Embed    │──▶│  Similarity search  │──▶│   Re-ranking    │
 │  query   │   │  query    │   │  top-k (e.g. k=20-50)│   │  cross-encoder  │
 │          │   │(same model│   │  dense (+ BM25,      │   │  scores each    │
 │          │   │ as index) │   │  fused via RRF)      │   │  (query,chunk)  │
 └──────────┘   └───────────┘   └──────────┬──────────┘   │  pair; keep     │
                                            │              │  top-n (n<<k)   │
                                            │              └────────┬────────┘
                                            │ (skip rerank in                │
                                            │  naive RAG)                    │
                                            └───────────────┬────────────────┘
                                                             ▼
                                            ┌────────────────────────────┐
                                            │   Context assembly /       │
                                            │   prompt construction      │
                                            │   - order chunks (avoid    │
                                            │     lost-in-the-middle)    │
                                            │   - attach citations       │
                                            │   - system instructions    │
                                            └─────────────┬──────────────┘
                                                           ▼
                                            ┌────────────────────────────┐
                                            │       LLM generation       │
                                            │  answer conditioned on     │
                                            │  retrieved context, not    │
                                            │  parametric memory alone   │
                                            └─────────────┬──────────────┘
                                                           ▼
                                                 ┌───────────────────┐
                                                 │  Answer + sources │
                                                 └───────────────────┘
```

The two paths share nothing at request time except the vector store they both touch — indexing is amortized cost paid once per document, while every box in the query path is on the critical path of user-perceived latency.

---

## 4. Naive RAG vs. intermediate RAG patterns

"Naive RAG" refers to the minimal pipeline: chunk → embed → single dense vector search → stuff top-k directly into the prompt → generate, with no re-ranking, no query transformation, and no lexical search. It is a reasonable starting point and works acceptably when queries closely resemble the phrasing of the source documents, but it degrades quickly outside that regime. The following patterns are the standard next steps beyond naive RAG, before reaching for the techniques covered in the companion advanced-techniques doc.

### 4.1 Hybrid search: sparse (BM25) + dense, with Reciprocal Rank Fusion

Dense vector search excels at *semantic* matches (synonyms, paraphrase, conceptual similarity) but is comparatively weak at *exact* matches — product codes, error messages, acronyms, identifiers, rare proper nouns, or numeric values, all of which an embedding model tends to smear into a generic regional area of vector space rather than pinning down precisely. **BM25** (Best Matching 25), a classical term-frequency/inverse-document-frequency scoring function over an inverted index, is the complementary strength: it matches exact terms extremely well and needs no training data or embedding model, but has no notion of synonymy or paraphrase (a BM25 search for "car" will not match a document that only says "automobile").

BM25's scoring function for a query `Q` with terms `q_1...q_n` against document `D`:

```
score(D, Q) = Σ IDF(q_i) · [ f(q_i, D) · (k1 + 1) ] / [ f(q_i, D) + k1 · (1 - b + b · |D| / avgdl) ]
```

where `f(q_i, D)` is the term frequency of `q_i` in `D`, `|D|` is the document length, `avgdl` is the average document length in the corpus, and `k1` (term-frequency saturation, typically ~1.2–2.0) and `b` (length-normalization strength, typically ~0.75) are tunable constants.

**Hybrid search** runs both a dense vector search and a BM25 search over the same chunk corpus in parallel, producing two separately-ranked result lists, then merges them. Because BM25 scores and cosine-similarity scores live on incompatible scales (there is no principled way to say a BM25 score of 8.3 is "worth" a cosine similarity of 0.71), the merge is typically done on **rank**, not raw score, using **Reciprocal Rank Fusion (RRF):**

```
RRF_score(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))
```

where `rank_r(d)` is the position of document `d` in ranker `r`'s result list (1-indexed), the sum runs over each ranking system being fused (dense, BM25, and potentially more), and `k` is a small constant (commonly 60) that dampens the influence of very high individual ranks so that one ranker's #1 pick doesn't automatically dominate. A document ranked highly by *either* system gets a large contribution to its fused score; a document ranked highly by *both* systems gets a large contribution from both terms and typically rises to the top of the fused list — which is exactly the desired outcome, since agreement between a lexical and a semantic signal is strong evidence of relevance. Documents that only one system retrieved at all are naturally still eligible (their absent ranker simply contributes 0), so hybrid search strictly does not lose candidates that either retrieval method alone would have found.

The net effect: hybrid search captures both "this document uses the same words as the query" (BM25) and "this document means the same thing as the query" (dense), and RRF combines them without requiring score calibration between fundamentally different scoring systems.

### 4.2 Query rewriting and expansion

User queries are often short, ambiguous, or phrased very differently from how the answer is written in the source corpus (e.g., a user asks "why is my bill so high," while the billing FAQ is phrased as "understanding overage charges"). Query rewriting/expansion uses an LLM (usually a fast, cheap one) as a preprocessing step to transform the raw query before it's embedded/searched:

- **Rewriting** paraphrases the query into a form closer to expected document phrasing, resolves coreference (e.g., turning a follow-up question like "what about the second one?" into a fully self-contained query using prior conversation turns), and can strip conversational filler.
- **Expansion** generates multiple related query variants (or adds synonyms/related terms) and searches with all of them, then merges the result sets — increasing the odds that at least one phrasing matches the vocabulary actually used in the target document. This overlaps with the multi-query pattern covered in the advanced-techniques doc.

The cost is an extra LLM call (added latency and cost) on every query, so it's typically used selectively — e.g., only when initial retrieval confidence is low, or always in domains where the user query and the document corpus are known to use very different vocabularies (support tickets vs. technical documentation, for instance).

### 4.3 HyDE (Hypothetical Document Embeddings)

HyDE addresses the same underlying problem — query phrasing and document phrasing occupy different regions of embedding space — but from a different angle: instead of rewriting the query into a better query, it asks an LLM to **hallucinate a hypothetical answer** to the query first, and then embeds *that hypothetical answer* to use as the search vector, discarding the LLM-generated text itself (it is never shown to the user or checked for accuracy).

The intuition: a real document that answers the question and a plausible-sounding LLM-generated answer to the question are likely to be phrased much more similarly to each other (same register, same vocabulary, same structural shape — "an answer to this question looks like X") than the original terse user question is to either of them. Because embedding models measure surface-and-semantic similarity, embedding a hypothetical-answer-shaped text tends to land closer in vector space to a real-answer-shaped document than embedding a short, differently-phrased question would.

Concretely: query "What causes lost-in-the-middle behavior in LLMs?" → LLM generates a plausible-sounding (possibly wrong in its specifics) explanatory paragraph → that paragraph is embedded and used for the vector search, instead of embedding the short question directly. The hypothetical document's *factual correctness is irrelevant* — it's used purely as an embedding anchor, not as content. This works well when queries are short/sparse relative to document style (e.g., web search-style queries against long-form articles) but adds an LLM call to every query's latency budget and can misfire when the hallucinated answer heads in a confidently wrong direction that no real document resembles.

---

## 5. Failure modes

RAG's reliability is bounded by whichever stage in the pipeline fails, and failures at different stages look superficially similar (a wrong or missing answer) but require entirely different fixes.

### 5.1 Retrieval mismatch (embedding space misalignment)

The single most common root cause of bad RAG output is not the LLM — it's that the correct chunk was never retrieved in the first place, so the LLM was asked to answer from irrelevant context (and either hallucinates an answer anyway, or the retrieval fails silently and it says "I don't know" even though the answer existed in the corpus). This happens when the query and the relevant document(s), despite being about the same real-world fact, land far apart in the embedding model's vector space — for example, because the embedding model was trained on general-domain text and doesn't capture domain-specific term relationships (a legal or medical corpus, an internal codebase with project-specific jargon), because the query is phrased as a question while the document is phrased as a declarative statement and the model wasn't trained asymmetrically to bridge that gap, or because the query uses an abbreviation/synonym the embedding model doesn't associate strongly with the document's terminology. This is precisely the failure mode hybrid search (4.1), query rewriting (4.2), and HyDE (4.3) exist to mitigate — and why retrieval evaluation (Section 6) has to be measured independently of generation quality: a perfect LLM cannot compensate for a chunk that was never in its context window.

### 5.2 Lost-in-the-middle

Empirically, LLMs do not attend uniformly across a long input context — accuracy on information placed at the very start or very end of the prompt is measurably higher than accuracy on the same information placed in the middle, even when the model's stated context window is large enough to hold everything comfortably. This means that in naive RAG, simply concatenating the top-k chunks in retrieval-score order (best chunk first, worst last) can bury the actual answer-bearing chunk in the middle of the assembled context if it wasn't literally the single top-ranked result, and the model may fail to use it even though it's technically present in the prompt. Mitigations include keeping the number of chunks passed to the LLM small (favoring the two-stage retrieve-then-rerank pipeline over throwing a large raw top-k at the model), and deliberately re-ordering the final chunk sequence so the highest-confidence chunks sit at the boundaries of the context rather than the middle.

### 5.3 Chunk boundary problems

When a fact-bearing sentence, or a sentence and the pronoun/reference it depends on, is split across a chunk boundary by the chunking process, neither resulting chunk fully contains the fact — one chunk may end mid-sentence, and the other begins mid-sentence, so the embedding of neither chunk represents the complete idea well, hurting retrieval, and even if the wrong-but-adjacent chunk *is* retrieved, the LLM sees a fragment that reads as incomplete or ambiguous ("It increased by 15% compared to the prior year." with "It" and the sentence identifying what "it" refers to sitting in the previous, unretrieved chunk). Overlap (Section 2.2) is the standard mitigation, along with structure-aware splitting (recursive splitting on paragraph/sentence boundaries rather than raw character counts) so that at least splits tend to fall between ideas rather than through them; late chunking and contextual retrieval (covered in the advanced-techniques doc) attack this same problem more fundamentally by having each chunk's representation carry information from its surrounding context.

### 5.4 Context window overflow

Retrieving too many chunks, or chunks that are individually too large, for the amount of context budget available (the LLM's max context length minus the space reserved for the system prompt, conversation history, and the model's own output) forces either silent truncation (the least relevant — or worse, an arbitrarily cut — portion of the assembled context is dropped, which can remove the answer entirely without any visible error) or an outright API error. This is compounded in multi-turn conversational RAG, where prior turns and their retrieved contexts accumulate in the prompt unless explicitly pruned. The fix is disciplined context budgeting: cap the number and size of chunks passed to generation (which is exactly what the retrieve-then-rerank pattern enforces by construction — the reranker narrows k candidates down to a small, fixed n before assembly) rather than relying on the LLM's context window being "big enough" as headroom for pipeline sloppiness.

### 5.5 Stale indexes

Because the whole value proposition of RAG is that updating knowledge means updating the index rather than retraining the model, a RAG system is only as current as its last successful ingestion run. If document updates, deletions, or permission changes aren't propagated into the vector store (and any parallel BM25 index) promptly, the system will confidently retrieve and cite outdated or even revoked content — a failure mode that is easy to overlook because it produces no error, just quietly wrong answers sourced from stale data. This requires treating the ingestion pipeline as a first-class, monitored data pipeline (with freshness SLAs, change-data-capture from source systems, and reindexing triggers) rather than a one-time setup step.

---

## 6. Evaluation

RAG evaluation must be split into **retrieval evaluation** (did we find the right chunks?) and **generation evaluation** (given those chunks, did the model produce a good answer?), because the two questions have different failure causes and different fixes, and conflating them (e.g., only measuring end-to-end answer correctness) makes it impossible to tell whether a bad answer came from a retrieval miss or a generation error.

### 6.1 Retrieval metrics

These require a labeled evaluation set of `(query, set of relevant document/chunk IDs)` pairs.

**Recall@k** — of all the chunks that are actually relevant to the query, what fraction appear somewhere in the top-k retrieved results?

```
Recall@k = |relevant chunks ∩ top-k retrieved chunks| / |all relevant chunks|
```

Recall@k measures whether the system is *capable* of answering the query at all — if the relevant chunk isn't in the top-k, no amount of re-ranking or prompt engineering downstream can recover it. This is why recall is typically measured at a larger k (e.g., Recall@20 or Recall@50) at the raw-retrieval stage, before re-ranking narrows the set further.

**MRR (Mean Reciprocal Rank)** — for each query, take the reciprocal of the rank position of the *first* relevant result (1 if it's ranked #1, 1/2 if it's ranked #2, 0 if no relevant result appears at all within the considered range), then average across all queries in the evaluation set:

```
MRR = (1/|Q|) · Σ_{q ∈ Q} 1 / rank_q
```

MRR is most meaningful when there is typically exactly one correct answer chunk per query (e.g., single-fact lookups) and you care about how quickly the user/LLM gets to it, rather than about finding every relevant chunk.

**NDCG (Normalized Discounted Cumulative Gain)** — used when relevance isn't simply binary (relevant/irrelevant) but graded (e.g., a chunk can be highly relevant, somewhat relevant, or irrelevant), and rewards ranking the *most* relevant results earliest, not merely ranking *a* relevant result earliest (which is MRR's limitation). DCG at rank `p` discounts each result's relevance score logarithmically by its position:

```
DCG@p = Σ_{i=1}^{p}  rel_i / log2(i + 1)
```

`rel_i` is the graded relevance of the result at position `i`. This is then normalized by the DCG of the *ideal* ranking (i.e., the same set of results sorted in perfect relevance order), giving a score bounded in `[0, 1]`:

```
NDCG@p = DCG@p / IDCG@p
```

where `IDCG@p` is `DCG@p` computed on the ideal ordering. NDCG is the standard metric of choice for tuning and comparing rerankers, since re-ranking's whole job is producing a better *ordering*, not just a better *set*, of results.

### 6.2 Generation metrics

Once the right chunks are confirmed to be present in context (isolating away retrieval failures), generation quality is evaluated along two axes that are commonly measured with an LLM-as-judge approach (a separate, usually more capable, model scores the answer against the source context and/or a reference answer):

**Faithfulness / groundedness** — does every claim in the generated answer actually follow from the retrieved context, or did the model add unsupported claims (hallucinate) on top of it? This is typically scored by decomposing the answer into individual claims and checking each one against the provided context (can it be inferred from the context: yes/no/partially), then aggregating into a faithfulness score such as `(# claims supported by context) / (# total claims)`. This is the metric that most directly measures whether RAG achieved its core goal of grounding — a system can have perfect retrieval and still score poorly here if the LLM ignores the context and answers from parametric memory instead, or embellishes beyond what the context supports.

**Answer relevancy** — does the generated answer actually address the user's question, independent of whether it's grounded? A perfectly faithful answer that is grounded in the retrieved context but doesn't actually answer what was asked is still a failure. One common approach: have an LLM generate several candidate questions that the produced answer *would* be a good response to, embed those generated questions, embed the original user question, and score relevancy by the average similarity between them — a high similarity implies the answer is on-topic for the question actually asked.

A healthy evaluation practice tracks recall@k and NDCG as retrieval health metrics on every index/chunking/embedding-model change, and faithfulness/relevancy as generation health metrics on every prompt or model change — treating a drop in either category as a signal to look specifically at that half of the pipeline rather than re-tuning the other half by trial and error.

---

## 7. RAG vs. fine-tuning vs. long-context

These three approaches are frequently pitted against each other, but they solve different problems and are frequently used together rather than as mutually exclusive alternatives.

**RAG** is the right tool when the need is **knowledge injection**: the model needs access to facts, documents, or data that either didn't exist at training time, changes frequently, is too voluminous to fit in any context window, or must remain attributable/auditable (you need to know and show *which document* backed an answer). Its cost is architectural complexity and per-query retrieval latency, and its ceiling is bounded by retrieval quality — a great model with bad retrieval performs no better than a mediocre model with good retrieval, and often worse, because it will confidently fill gaps.

**Fine-tuning** is the right tool when the need is **behavior or style injection**, not knowledge injection: teaching a model a specific output format, a domain-specific tone or terminology, a task pattern it doesn't currently follow (e.g., always responding in a certain JSON schema, or adopting a particular reasoning style), or improving performance on a narrow task via many labeled examples. Fine-tuning is a poor tool for keeping a model up to date on rapidly changing facts, because every knowledge update requires re-running a training job, and fine-tuning on new facts has a well-documented tendency to make the model *more* confidently wrong about related but untrained facts, rather than well-calibrated — it doesn't teach "I don't know," it teaches new completions. See `08-lora-qlora` in this repo's ml-concepts docs for the mechanics of parameter-efficient fine-tuning specifically.

**Long-context** (feeding an entire corpus, or a very large fraction of it, directly into a large context window instead of retrieving a subset) is viable when the total relevant corpus is small enough to fit the model's context window with room to spare, and is attractive because it sidesteps retrieval failures entirely — there's no top-k cutoff that can miss the right passage, because everything is present. But it doesn't eliminate lost-in-the-middle (a bigger haystack doesn't fix attention being uneven across it — if anything it usually makes the effect more pronounced across more content), it's expensive and slow to re-process a huge context on every single query (whereas RAG's search step is comparatively cheap, and prompt-caching techniques help long-context reuse but only for a static, unchanging prefix), and it doesn't scale once the true corpus exceeds even a generous context window (a support knowledge base with tens of thousands of articles will never fit, no matter how large context windows grow).

**In practice, these compose:** a common production pattern is RAG for knowledge grounding, combined with a fine-tuned (or well-prompted) model for output style/format, and long-context used opportunistically *within* the RAG pipeline itself — e.g., retrieving fewer, larger chunks (or even whole documents) when the context budget allows, rather than being forced into narrow chunking purely to fit through a small context window. The decision isn't "RAG or fine-tuning or long-context" so much as "which axis — knowledge freshness/scale, behavior/style, or context budget — is the actual constraint for this problem."

---

## 8. Where to go deeper

This document covers the fundamentals and the naive-to-intermediate techniques that form the baseline for any production RAG system. Two companion docs in this repository go further into specialized territory that's deliberately out of scope here:

- **[`../20-advanced-rag-techniques/README.md`](../20-advanced-rag-techniques/README.md)** — covers techniques that address the deeper limitations of the pipeline described above: **contextual retrieval** (prepending LLM-generated context to each chunk before embedding, to fix the chunk-boundary problem at its root rather than just overlapping chunks), **RAPTOR** (building a recursive tree of abstractive summaries over the corpus so broad, thematic queries can retrieve at the right level of abstraction instead of only ever retrieving narrow chunks), **ColBERT / late-interaction retrieval** (token-level multi-vector representations that sit between cheap bi-encoders and expensive cross-encoders), **late chunking** (embedding the full document first and pooling chunk vectors afterward, so each chunk's embedding carries whole-document context), and **agentic RAG** (turning the fixed retrieve-then-generate pipeline into an LLM-driven loop that can decide to re-query, decompose a question, or call tools before answering).
- **[`../13-graph-rag/README.md`](../13-graph-rag/README.md)** — covers **GraphRAG**: building a knowledge graph (entities and relationships) over the corpus instead of, or alongside, a flat vector index, and querying it with local search (for questions about specific named entities and their connections) or global search (for broad, corpus-wide thematic questions) — the right tool when the vector-only pipeline described in this document hits its structural blind spot around multi-hop reasoning and relationship-centric queries.

Readers new to RAG should treat this document as the prerequisite for both: the concepts here — chunking, embeddings, indexing, bi-encoder vs. cross-encoder retrieval, RRF, and the retrieval/generation evaluation split — are assumed background in both companion docs.
