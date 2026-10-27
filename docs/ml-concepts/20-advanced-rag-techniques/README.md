# Advanced & Production RAG Techniques

This document covers techniques that go **beyond the basic retrieve-then-generate pipeline**. If you haven't already, read [`../02-rag/README.md`](../02-rag/README.md) first — it covers chunking strategies, embedding models, vector index structures, basic top-k retrieval, cross-encoder re-ranking, hybrid (dense + BM25) search, HyDE, Self-RAG, and standard evaluation metrics (recall@k, MRR, faithfulness/groundedness scoring, etc.). None of that is repeated here.

This document assumes you've hit the ceiling of naive RAG in production: chunks that lose their surrounding context, single vector-per-document embeddings that blur fine-grained matches, a system that can answer "what did the contract say about termination fees" but falls over on "summarize the overall risk posture across all twelve contracts," and a fixed one-shot pipeline that can't recover when the first retrieval pass returns garbage. The six techniques below — contextual retrieval, RAPTOR, ColBERT-style late interaction, late chunking, agentic RAG, and multi-query ensemble retrieval — address those failure modes, roughly ordered from "cheap fix to an existing pipeline" to "architectural rethink."

We also cross-reference [`../13-graph-rag/README.md`](../13-graph-rag/README.md) throughout, since GraphRAG solves some of the same problems (especially the "broad/thematic query" problem) via a different mechanism — an entity-relationship graph instead of a summary hierarchy or an agent loop. Section 8 gives a decision framework across all of these, including GraphRAG.

---

## 1. Contextual Retrieval

### The problem

Chunking is necessary — you can't embed a 50-page document as one vector without destroying granularity — but it creates a well-known failure mode: **a chunk in isolation loses the context that made it findable and interpretable.**

Take a chunk pulled from the middle of an SEC filing:

> "The company's revenue grew 3% over the previous quarter."

Embedded on its own, this chunk is nearly useless for retrieval. Which company? Which quarter? A query like "What was Acme Corp's Q2 2023 revenue growth?" will not score this chunk highly, because nothing in the chunk's text mentions Acme Corp or Q2 2023 — the embedding model has no signal to work with. The same problem hits a BM25/keyword index even harder: BM25 is pure term-overlap, so if "Acme" and "Q2 2023" never appear in the chunk's tokens, BM25 gives it a zero score regardless of semantic relevance. Larger chunks reduce this problem somewhat but reintroduce the tradeoffs basic RAG already covers (worse embedding precision, more irrelevant tokens fed to the generator, higher cost per retrieved unit).

### The fix

Anthropic's **Contextual Retrieval** technique (introduced in 2024) addresses this at indexing time, before embedding: for every chunk, use an LLM to generate a short (50–100 token) **context statement** that situates the chunk within the document, and prepend it to the chunk before it goes into either the embedding index or the BM25 index.

For the example above, the context-generation prompt is given the *whole document* plus the *specific chunk*, and asked to produce something like:

> "This chunk is from Acme Corp's Q2 2023 SEC 10-Q filing. It appears in the Financial Results section discussing quarter-over-quarter revenue comparisons."

The chunk that actually gets embedded and indexed becomes:

```
This chunk is from Acme Corp's Q2 2023 SEC 10-Q filing. It appears in the
Financial Results section discussing quarter-over-quarter revenue comparisons.

The company's revenue grew 3% over the previous quarter.
```

Now both the dense embedding and the BM25 posting list carry the disambiguating tokens ("Acme Corp," "Q2 2023," "revenue"). A query embedding for "Acme Corp Q2 2023 revenue growth" will have much higher cosine similarity to this augmented chunk than to the bare original, and BM25 will now actually match on "Acme" and "Q2 2023" as terms. Critically, the *generated* text is only used for indexing — you typically still return the original, un-prefixed chunk (or optionally the prefixed version) to the generator LLM at answer time, since the goal is to fix retrievability, not to pollute the context window with meta-commentary (though many implementations keep the prefix in the context passed to the generator too, since it's cheap and can help downstream synthesis).

Anthropic's published implementation combines two augmented indexes:

- **Contextual Embeddings**: embed `context + chunk` with a standard embedding model.
- **Contextual BM25**: build a BM25 index over `context + chunk` (not just the embedding index) — this matters because BM25 catches exact-term matches (part numbers, proper nouns, jargon) that dense embeddings sometimes miss, and without contextualizing the BM25 corpus too, you'd only fix half the retrieval path.

Results reported in Anthropic's write-up: contextual embeddings alone reduced retrieval failure rate (top-20 chunk miss rate) by ~35% versus plain embeddings; combining contextual embeddings + contextual BM25 reduced it by ~49%; adding a re-ranking pass on top of that combination brought the total reduction to ~67%. This stacks cleanly with the hybrid search and re-ranking techniques already covered in `02-rag` — contextual retrieval is not a replacement for hybrid search or re-ranking, it's a preprocessing step that makes both of them work better.

### Why this is cheap at scale: prompt caching

The naive way to generate a context blurb per chunk is to send `(whole document, this chunk)` to an LLM, once per chunk. For a document with 200 chunks, that's 200 LLM calls, each re-sending the *entire document* as input. For a large corpus this is the dominant cost and would normally be prohibitive — you'd be paying full input-token price for the whole document, 200 times over, just to index it once.

**Prompt caching** eliminates almost all of this redundant cost. The document is placed in the prompt as a cached prefix; the model provider (Anthropic's API supports this natively via `cache_control` breakpoints) hashes and stores the KV-cache state for that prefix after the first call. Every subsequent call in the same indexing job that reuses the identical document prefix — one call per chunk, varying only the "here is the specific chunk to contextualize" suffix — hits the cache instead of reprocessing the document from scratch. Cached input tokens are billed at a small fraction of normal input pricing (roughly 10% of the base input rate for a cache read, with a modest write premium on the first call), and latency drops correspondingly because the model isn't re-running attention over the full document repeatedly.

The practical shape of the indexing pipeline:

```
1. Split document into N chunks (using whatever chunking strategy from 02-rag).
2. Build one prompt template:
   [SYSTEM/CACHED PREFIX]: full document text, cache_control: {"type": "ephemeral"}
   [PER-CHUNK SUFFIX]: "Here is the chunk to situate: <chunk_i>.
                        Give a short context (50-100 tokens) situating
                        this chunk within the overall document for the
                        purposes of improving search retrieval."
3. For i in 1..N: call LLM with (cached document prefix, chunk_i suffix).
   -> first call pays full price + a cache-write premium
   -> calls 2..N pay only the cache-read rate for the document tokens,
      plus normal rate for the short suffix + short output
4. Prepend generated context to chunk_i; embed and index (both dense + BM25).
```

This is why contextual retrieval is described as "cheap at scale despite doing an LLM call per chunk" — the marginal cost per chunk after the first call is dominated by a few hundred tokens of suffix + output at cache-read pricing, not by re-billing the whole document N times. For a corpus that's indexed once and queried many times, this one-time indexing cost (even un-cached) is usually justified anyway, since it directly reduces retrieval failures on every query thereafter; caching just makes it close to free.

A secondary, often overlooked cost lever: you can batch contextualization requests using an async/batch API (half the normal price, higher latency tolerance) since indexing is not latency-sensitive the way query-time inference is — batch + cache stacks multiplicatively.

---

## 2. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

### The problem it solves

Standard flat-chunk RAG (even with contextual retrieval applied) has a structural ceiling: retrieval returns individual chunks, so it's well-suited to **factoid, localized queries** ("what was the termination clause in section 4.2") but poorly suited to **broad, thematic, or aggregative queries** ("what are the main risks discussed across this entire document" or "summarize the company's strategic priorities this year"). No single chunk contains the answer to a thematic question — the answer is a synthesis that spans dozens or hundreds of chunks, and top-k retrieval over flat chunks will return k semantically similar but individually incomplete fragments, none of which let the generator produce a coherent, complete synthesis.

This is the same underlying problem that [GraphRAG](../13-graph-rag/README.md) addresses via a knowledge graph of entities and relationships plus community summaries. **RAPTOR addresses it through hierarchical summarization instead of a graph.**

### How it works

RAPTOR builds a **tree** at indexing time, bottom-up:

1. **Leaves**: start with the normal base-level chunks (the same chunking strategy from `02-rag`), each embedded normally.
2. **Clustering**: cluster the chunk embeddings into semantically coherent groups. RAPTOR's original implementation uses a soft-clustering approach (Gaussian Mixture Models over UMAP-reduced embeddings) rather than hard k-means, because a chunk can legitimately belong to more than one thematic cluster (e.g., a chunk discussing "supply chain risk in Q3 due to a regional plant closure" plausibly belongs to both a "supply chain" cluster and a "Q3 operational events" cluster). Soft clustering allows a chunk to be assigned, with some probability, to multiple clusters, and lets the algorithm not fix the number of clusters in advance (a Bayesian Information Criterion selects cluster count).
3. **Summarization**: for each cluster, an LLM generates a summary of the chunks in that cluster. This summary becomes a new "node" one level up the tree.
4. **Recurse**: treat this new layer of summary nodes exactly like a new layer of "chunks" — embed them, cluster them, summarize the clusters — and repeat. Each recursion produces a smaller number of increasingly abstract summary nodes.
5. **Stop** when a level produces a single cluster (or a cluster count small enough to summarize into one root node), giving a root that represents a summary of the entire document (or corpus).

The result is a tree: many leaf chunks at the bottom, progressively fewer and more abstract summary nodes going up, and one (or a small number of) root node(s) capturing the document's overall themes at the top.

```
                              ┌───────────────────────┐
                    LEVEL 3   │   ROOT SUMMARY NODE    │   (whole-document /
                              │  "overall themes and   │    corpus-level
                              │   conclusions"         │    synthesis)
                              └───────────┬────────────┘
                                          │ summarizes
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
              ┌─────────▼───────┐ ┌───────▼────────┐ ┌──────▼─────────┐
    LEVEL 2   │ Summary cluster │ │ Summary cluster│ │ Summary cluster│
              │   "Finance"     │ │  "Operations"  │ │  "Strategy"    │
              └────────┬────────┘ └───────┬────────┘ └───────┬────────┘
                       │  summarizes      │ summarizes       │ summarizes
              ┌────────┼────────┐    ┌────┼────┐      ┌──────┼──────┐
              │        │        │    │    │    │      │      │      │
    LEVEL 1 ┌─▼──┐  ┌──▼─┐  ┌──▼─┐┌──▼─┐┌─▼──┐┌▼───┐┌──▼─┐┌───▼┐┌────▼┐
  (clusters)│sumA│  │sumB│  │sumC││sumD││sumE││sumF││sumG││sumH││sumI │
            └─┬──┘  └─┬──┘  └─┬──┘└─┬──┘└─┬──┘└─┬──┘└─┬──┘└─┬──┘└──┬──┘
              │clusters│chunks│      │      │      │      │      │
    LEVEL 0 ┌─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┬─▼─┐
  (leaves)  │c1 │c2 │c3 │c4 │c5 │c6 │c7 │c8 │c9 │c10│c11│c12│c13│c14│  ...raw
            └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘   chunks
```

### Retrieval over the tree

At query time, RAPTOR supports two retrieval strategies:

- **Collapsed-tree retrieval** (the approach RAPTOR's authors found to perform best in practice): flatten *every node at every level* — leaves and all intermediate/root summaries — into one pool, embed the query once, and run standard top-k cosine similarity search over the entire pool. Because leaf chunks and high-level summaries coexist in the same retrieval pool, a factoid query naturally scores highest against fine-grained leaf chunks (since their embeddings are narrow and specific), while a broad/thematic query naturally scores highest against upper-level summary nodes (since their embeddings capture aggregated themes) — no manual routing logic is needed; relevance ranking does the routing implicitly.
- **Tree-traversal retrieval**: start at the root, compute similarity to the query, descend into the top-scoring child node(s), and repeat down to a target depth — a more structured, greedy top-down walk. This is more constrained and depends on early routing decisions being correct; the RAPTOR paper found collapsed-tree retrieval generally outperformed it, since a wrong turn early in a top-down traversal can prune away the right leaf.

Either way, the retrieved set for a given query can span multiple levels simultaneously — e.g., one root-level summary node plus three leaf chunks — giving the generator both the broad synthesis and the specific supporting detail in the same context window.

### RAPTOR vs GraphRAG for broad queries

Both techniques target the same failure mode (flat-chunk RAG can't answer "what are the overarching themes" questions), but the mechanisms and tradeoffs differ:

| | RAPTOR | GraphRAG |
|---|---|---|
| Structure built | Hierarchy of LLM-generated summaries over semantic clusters | Entity/relationship knowledge graph + community detection + community summaries |
| Best suited to | Thematic/aggregative summarization queries over a document or corpus with soft topical structure | Multi-hop relational queries ("how is entity A connected to entity C") and thematic queries grounded in explicit entities |
| Indexing cost | Multiple rounds of clustering + LLM summarization (moderate) | Entity/relation extraction pass (LLM-heavy, often the most expensive indexing step) + graph community detection + community summarization (heavy) |
| Query-time cost | Single embedding search over the collapsed pool (cheap, same cost profile as flat vector RAG) | Depends on mode: local search (graph-neighborhood lookup) is moderate; global search (map-reduce over community summaries) is expensive |
| Interpretability | Tree of prose summaries, easy to inspect | Explicit graph, supports structured querying and relationship-level explainability |
| Weakness | Clustering can produce awkward groupings on documents without clean topical structure; summaries can drift/hallucinate details from cluster contents | Extraction quality depends heavily on the LLM's ability to correctly identify entities/relations; expensive to build and to keep in sync as documents change |

In short: reach for RAPTOR when the corpus has natural topical/document structure and you want cheap query-time cost with strong summarization behavior; reach for GraphRAG when the value is in explicit entities and their relationships (org charts, supply chains, citation networks, "who reports to whom," "which drugs interact with which proteins") rather than purely topical clustering. See Section 8 for the full decision framework.

---

## 3. ColBERT / Late-Interaction Retrieval

### The problem with single-vector embeddings

Standard dense retrieval (as covered in `02-rag`) collapses an entire query and an entire document/chunk into **one vector each**, then scores relevance as a single cosine similarity or dot product between those two vectors. This is fast and index-friendly, but it forces a huge amount of information through one bottleneck vector — a 768- or 1024-dimensional vector has to simultaneously represent every token, entity, and nuance in a passage. Two passages that share a small but critical overlapping detail (a specific product SKU, a rare technical term, a precise date) can end up with similar overall embeddings to many other passages, because that one matching detail gets diluted in the average-like pooling that produces the single vector. Fine-grained, term-level matching signal is lost.

The other extreme — a **cross-encoder** (already covered in `02-rag`'s re-ranking section) — solves this by jointly encoding the query and document together through a transformer, attending query tokens directly to document tokens. This captures fine-grained interactions extremely well, but it is expensive: you must run a full transformer forward pass per (query, document) pair at query time, so it can only be applied to re-rank a small candidate set (e.g., top 50–100), never to search a full corpus of millions of documents directly.

### ColBERT: late interaction as the middle ground

**ColBERT** ("Contextualized Late Interaction over BERT") sits between these two extremes. Instead of pooling a passage into one vector, it keeps **one contextualized embedding per token** — for both the query and the document. At query time, relevance is computed via a **MaxSim** operation:

For every query token vector, find the *maximum* similarity (dot product or cosine) it achieves against *any* document token vector, then **sum these per-query-token maxima** across all query tokens to get the final relevance score:

```
score(Q, D) = Σ_{i in Q}  max_{j in D}  sim(q_i, d_j)
```

Where `Q = {q_1, ..., q_n}` are the query's token embeddings and `D = {d_1, ..., d_m}` are the document's token embeddings.

```
                    ColBERT MaxSim Late-Interaction Scoring

  Query tokens (each a vector)         Document tokens (each a vector)
  ─────────────────────────────        ────────────────────────────────
                                          d1: "Acme"
   q1: "revenue" ───────┐                d2: "Corp"
                         ├────max────►    d3: "Q2"
                         │                d4: "2023"
                         └──────────────► d5: "revenue"  ◄── highest sim
                                          d6: "grew"
                                          d7: "3%"
                                          d8: "over"
                                          d9: "prior"
                                         d10: "quarter"

   q2: "growth" ─────────────────max───► d6: "grew"      ◄── highest sim

   q3: "Q2" ─────────────────────max───► d3: "Q2"        ◄── highest sim
                                          (exact contextual match)

   q4: "2023" ───────────────────max───► d4: "2023"      ◄── highest sim

  ─────────────────────────────────────────────────────────────────────
  score(Q,D) =  max_sim(q1) + max_sim(q2) + max_sim(q3) + max_sim(q4)
             =     0.91      +    0.83     +    0.97      +   0.95
             =     3.66   (final relevance score for this document)
```

Each query token independently "shops" for its single best match anywhere in the document, and the scores are summed. This means a document that contains a strong match for *every* distinct query token scores highly, even if those matching tokens are scattered across the passage and even if the document as a whole is "about" something slightly different — exactly the fine-grained matching behavior that gets blurred away by single-vector pooling. Rare/specific tokens (an exact model number, a named entity, a specific quarter) get their own dedicated matching opportunity instead of being diluted into a document-level average.

### Why it's cheap despite being token-level

The key architectural trick that makes ColBERT tractable at corpus scale — unlike a cross-encoder — is that **query and document are encoded independently**, never jointly:

- Document token embeddings are computed once, offline, at indexing time, and stored in the index (typically compressed — ColBERTv2 uses residual compression against per-cluster centroids to shrink storage, since storing a full vector per token for every document would otherwise be very storage-heavy).
- At query time, only the query needs to be freshly encoded into its token vectors (one small forward pass, since queries are short).
- MaxSim itself is just a set of vector similarity lookups (approximate nearest-neighbor search can be used per query token against the document token index) followed by summation — no transformer forward pass over the concatenated (query, document) pair is required at query time.

This gives ColBERT a cost profile close to standard dense bi-encoder retrieval (document representations precomputed and indexed, only the query is encoded live) while achieving matching quality much closer to a cross-encoder (fine-grained token-level interaction instead of single-vector similarity). In practice, ColBERT is commonly deployed either as a first-stage retriever in its own right (via specialized ANN indexes over token vectors, e.g. PLAID/ColBERTv2's engine) or as a strong, still-cheap-relative-to-cross-encoders re-ranking stage over a candidate set already narrowed by a bi-encoder or BM25 pass — giving three tiers of increasing cost and precision: bi-encoder (cheapest, coarsest) → ColBERT late-interaction (moderate cost, fine-grained) → cross-encoder (most expensive, most precise, only viable on small candidate sets).

The storage cost is the main tradeoff: a token-level index is much larger than a single-vector-per-document index (roughly proportional to total token count rather than document count), which is why compression schemes and careful engineering (PLAID, ColBERTv2) matter for making this viable at scale.

---

## 4. Late Chunking

### Same problem as contextual retrieval, different layer

Late chunking targets the identical failure mode as contextual retrieval — "a chunk embedded in isolation loses the context from the rest of the document" — but fixes it **inside the embedding model's forward pass** rather than by asking an LLM to write a context blurb.

The standard ("early chunking") pipeline is: **chunk first, then embed each chunk independently.** Each chunk is fed through the embedding model's transformer layers on its own, so the token representations inside the transformer are only ever contextualized by the other tokens *within that same chunk* — the model has literally no access to the rest of the document while computing those representations. A pronoun like "it," or a company referred to only as "the company" in a later paragraph after being named "Acme Corp" in an earlier one, gets a token representation with no way to resolve what "it" or "the company" refers to, because the resolving text isn't in the transformer's input window for that chunk.

### The fix: contextualize first, chunk (pool) after

Late chunking inverts the order of operations:

1. Run the **entire document** (or as much of it as fits in the embedding model's context window — this technique requires a long-context embedding model, e.g. one supporting 8K tokens) through the transformer's layers **as a single pass**, producing one contextualized token embedding for every token in the document. Because self-attention lets every token attend to every other token in the same forward pass, each token's representation is now informed by the whole document — "it" now has access to "Acme Corp" many sentences earlier, and its final hidden state reflects that resolved reference.
2. **Only after** these full-document contextualized token embeddings exist, apply the chunk boundaries (using whatever chunking strategy — fixed-size, sentence-based, semantic) and **mean-pool (or otherwise pool) the token embeddings within each chunk's span** to produce that chunk's final embedding.

The key insight: the transformer layers that produce contextual meaning run over the *whole document*; chunking becomes purely a pooling-boundary decision applied *after* contextualization, not a decision about what the model gets to see. This is why it's called "late" chunking — chunking happens late in the pipeline (at pooling time), not early (at input time).

```
EARLY CHUNKING (standard):                 LATE CHUNKING:

  doc → split into chunks first              doc → full document through
                                                     transformer layers first
  chunk1 → transformer → pool → emb1          (all tokens see all other
  chunk2 → transformer → pool → emb2           tokens in one forward pass)
  chunk3 → transformer → pool → emb3
                                              → THEN split into chunk spans
  each chunk's tokens only see                 and pool each span's already-
  tokens within that same chunk                contextualized token vectors

                                              → emb1, emb2, emb3
                                                (each now informed by the
                                                 rest of the document)
```

### Why this preserves cross-chunk context

Consider a document: *"Acme Corp released its Q2 results yesterday. The company's revenue grew 3% over the previous quarter, driven largely by its cloud division."* If chunked early into two chunks — `["Acme Corp released its Q2 results yesterday."]` and `["The company's revenue grew 3% over the previous quarter, driven largely by its cloud division."]` — the second chunk's embedding is computed by a transformer that never saw the word "Acme." The token embedding for "company" in that chunk has no mechanism to encode "this refers to Acme Corp" because "Acme Corp" wasn't in the input at all.

With late chunking, both sentences are processed together in one transformer pass. The self-attention layers let the token embedding for "company" (and "its," referring to the cloud division being Acme's) attend back to "Acme Corp" in the first sentence. When pooling is then applied to just the span of the second sentence, the resulting chunk embedding is a pooled representation of tokens that *already encode the Acme Corp reference internally*, even though the literal text "Acme Corp" never appears in that chunk's surface text. The embedding for the second chunk ends up meaningfully closer, in vector space, to queries like "What was Acme Corp's revenue growth?" than an early-chunked embedding of the same text would be — without any LLM call, without any prepended text, purely as a side effect of when contextualization happens relative to when pooling happens.

### Late chunking vs contextual retrieval

Both fix the same problem; the practical differences:

- **Cost model**: contextual retrieval costs one LLM call per chunk at indexing time (mitigated by prompt caching, Section 1); late chunking costs nothing extra beyond a normal embedding pass over the full document — no LLM call at all, just a long-context embedding model and a different pooling implementation. Late chunking is typically cheaper.
- **Requires**: late chunking needs a long-context embedding model (the whole document, or a large window of it, must fit in the embedding model's max sequence length) and access to the model's internals to pool at custom token spans rather than using the model's built-in `[CLS]`/mean-pool-over-full-input behavior — this isn't always exposed by hosted embedding APIs, so it's more natural with self-hosted or open-weight embedding models. Contextual retrieval works with any embedding model (even a black-box API) since it just prepends text.
- **Also fixes BM25**: contextual retrieval explicitly improves the BM25/keyword index too, since it literally adds disambiguating tokens to the text. Late chunking only affects the dense embedding, since it operates purely inside the embedding model's vector space — it does nothing for a parallel BM25 index, whose input text is unchanged.
- **Combinable**: these are not mutually exclusive. You can late-chunk to get better base embeddings and still run contextual retrieval's LLM-generated blurbs into the BM25 index, capturing both benefits.

---

## 5. Agentic RAG

### From a fixed pipeline to a decision-making loop

Everything covered so far, and everything in `02-rag`, still assumes the same overall shape: **retrieve once, generate once.** The retrieval step runs a fixed procedure (embed the query, search the index, maybe re-rank) and hands whatever it finds to the generator, regardless of whether that retrieval actually found the right information. If the first retrieval pass misses — because the query was ambiguous, because the question requires combining information found in two unrelated parts of the corpus, or because the query as originally phrased doesn't match how the answer is phrased in the documents — a single-shot pipeline has no mechanism to notice or recover.

**Agentic RAG** replaces the fixed pipeline with an LLM-driven control loop that treats retrieval as one of several tools it can invoke, iteratively, based on its own judgment of whether it currently has enough information to answer. A representative loop:

```
loop:
  1. Given the user's question and everything retrieved/found so far,
     the agent LLM decides ONE of:
       a. "I have enough information — answer now."
       b. "I need to retrieve — here's a search query to run"
            (possibly reformulated/decomposed from the original question)
       c. "I need to call a different tool" (e.g. a SQL query against
            structured data, a calculator, a web search, an API call)
       d. "The question is ambiguous — ask the user a clarifying question"
  2. If (b) or (c): execute the retrieval/tool call, append results to
     the agent's working context, and go back to step 1.
  3. If (a): generate the final answer, optionally with the agent also
     self-critiquing/verifying the answer against the retrieved evidence
     before returning it (similar in spirit to Self-RAG's reflection
     tokens, covered in 02-rag, but driven by explicit agent reasoning
     steps and tool calls rather than trained-in reflection tokens).
```

Concretely, this lets the system handle patterns a fixed pipeline cannot:

- **Multi-hop questions**: "Which of the vendors we terminated contracts with in 2023 also had a security incident reported in their SOC 2?" requires first retrieving which vendors were terminated in 2023, *then* issuing a second, dependent retrieval query per vendor name found in the first step. A fixed single-shot retrieval never sees the vendor names until generation time, so it can't have searched for their SOC 2 incidents. An agent explicitly plans this as two sequential (or looped) retrieval steps.
- **Query reformulation on low-confidence retrieval**: if the agent judges (e.g., by inspecting retrieved chunk relevance scores, or by attempting an answer and finding it's not well supported) that the first retrieval pass returned weak or irrelevant results, it can rewrite the query — generalizing an overly specific query, specializing an overly vague one, swapping in synonyms, or trying a HyDE-style hypothetical-answer reformulation (from `02-rag`) — and retrieve again, rather than silently generating an answer from bad context.
- **Tool combination**: a question like "How does our Q2 revenue growth compare to the industry average?" may need one retrieval call against internal documents *and* a separate call to a different tool entirely (a live web search, a structured database query) — an agent can route to whichever tool fits each sub-need, where a fixed RAG pipeline only ever has one retrieval mechanism wired in.
- **Knowing when *not* to retrieve**: for a purely conversational or already-answered-in-context question, an agent can skip retrieval entirely, avoiding wasted latency/cost — a fixed pipeline typically retrieves unconditionally on every turn.

### Cost/latency vs quality tradeoff

This capability is not free, and the tradeoff is the central design decision when adopting agentic RAG:

- **Multiple LLM calls per user question** instead of one: at minimum one call to decide the next action, one per tool/retrieval invocation's worth of reasoning, and one to generate the final answer — often 3–8+ calls for a moderately complex multi-hop question, each adding its own latency.
- **Higher and less predictable latency**: because the number of loop iterations is data-dependent (the agent might resolve in one hop or need five), p99 latency is much harder to bound than a fixed pipeline's roughly-constant retrieval + generation time. Production systems typically cap the number of iterations (e.g., max 4–5 tool calls) to bound worst-case latency and cost, accepting that a small fraction of very hard queries will be cut off before fully resolving.
- **Higher cost per query**: proportional to the number of LLM calls; for high-QPS, low-value queries (e.g., simple FAQ lookups) this overhead is wasted, since a fixed pipeline would have answered correctly in one pass.
- **New failure modes**: the agent can loop unproductively (re-issuing similar queries without making progress), can be over-confident that it has enough information when it doesn't (terminating too early), or can accumulate irrelevant retrieved context across iterations that dilutes the final generation step (context pollution). These require their own guardrails — iteration caps, deduplication of retrieved context, explicit "is this new information actually useful" checks — that add engineering complexity beyond a fixed pipeline.

In exchange, agentic RAG meaningfully outperforms fixed-pipeline RAG specifically on **multi-hop reasoning and ambiguous/underspecified queries**, where a single retrieval pass structurally cannot gather what's needed. For queries that are well-served by one good retrieval pass — the majority of factoid lookups — agentic RAG adds cost and latency without a corresponding quality gain, which is why most production systems apply it selectively (e.g., route only queries a classifier or heuristic flags as "complex" into the agentic path, and serve the rest via the cheaper fixed pipeline) rather than universally.

---

## 6. Multi-Query / Ensemble Retrieval (Query Decomposition)

A lighter-weight technique that captures some of agentic RAG's benefit without a full agent loop: instead of embedding the user's question once and retrieving once, **generate multiple query variants up front** (in a single LLM call, not an iterative loop) and retrieve for each variant independently, then merge the results.

Two common patterns:

- **Query decomposition into sub-questions**: for a compound question like "What were the main causes of the 2008 financial crisis and how did regulatory responses differ between the US and EU?", an LLM decomposes it into sub-questions — "What were the main causes of the 2008 financial crisis?", "What regulatory responses did the US take after 2008?", "What regulatory responses did the EU take after 2008?" — retrieves separately for each, and passes all retrieved sets to the generator together. This ensures each sub-topic gets its own dedicated retrieval pass rather than one blended query diluting relevance across all sub-topics at once.
- **Query rewriting / multi-perspective variants** (e.g., RAG-Fusion): generate several rephrasings of the same underlying question (different phrasings, different levels of specificity, different likely-vocabulary matches to the source documents) and retrieve for each, then merge/deduplicate the result sets — commonly using **Reciprocal Rank Fusion (RRF)**, which scores each retrieved document by summing `1 / (k + rank)` across all the variant queries' result lists, rewarding documents that rank well consistently across multiple query phrasings over documents that only one variant happened to surface.

This is strictly cheaper than full agentic RAG (one batch of parallel retrievals up front, no iterative "is this enough" loop, no tool-use branching) but still meaningfully improves recall on compound or ambiguously-phrased questions, since it hedges against any single query formulation missing the right vocabulary or missing a sub-topic entirely. It composes with everything else in this document — the underlying per-query retrieval can itself use contextual retrieval, ColBERT, or RAPTOR's collapsed tree; multi-query is a wrapper around the retrieval call, not a replacement for it.

---

## 7. Decision Framework: Which Technique, When

None of these techniques are mutually exclusive — a mature production system often layers several (e.g., contextual retrieval + hybrid search + RAPTOR for summarization queries + a ColBERT re-ranking stage + agentic routing for the subset of queries a classifier flags as multi-hop). But when reaching for the *next* increment of complexity, use query type and budget as the two axes:

| Query type | Best-fit technique(s) | Why |
|---|---|---|
| **Factoid, single-fact lookup** ("What is the termination fee in section 4.2?") | Plain vector RAG + hybrid search + re-ranking (from `02-rag`); add **contextual retrieval** if chunks currently retrieve poorly due to lost context; add **ColBERT** if fine-grained term matches (SKUs, exact figures, named entities) are being missed by single-vector embeddings | The answer lives in one chunk; you mainly need that chunk to be *findable*. Contextual retrieval and late chunking directly fix findability; ColBERT fixes precision on exact-term queries. No need for hierarchy or agentic looping. |
| **Thematic / aggregative / "summarize across"** ("What are the main risks discussed in this filing?") | **RAPTOR** (if the corpus has natural topical/document structure) or **GraphRAG global search** (if the value is in explicit entity relationships, not just topic clusters) | Flat chunk retrieval cannot synthesize across many chunks in one pass; you need a precomputed hierarchy (RAPTOR) or precomputed community structure (GraphRAG) so a single retrieval hop returns something already synthesized at the right altitude. |
| **Multi-hop / relational** ("Which vendors we terminated also had a security incident?", "How is Project X related to the org's Q3 layoffs?") | **Agentic RAG** (iterative retrieval, can chain dependent lookups) or **GraphRAG local search** (if the relationships are already modeled as graph edges) | The answer requires combining information not co-located in any single chunk or cluster; you need either an explicit graph of relationships to traverse, or an agent that can issue a second retrieval informed by the first one's results. |
| **Ambiguous / underspecified** ("Tell me about the Henderson account") | **Agentic RAG** (can ask a clarifying question or try multiple interpretations) or lightweight **multi-query/ensemble retrieval** (hedge across interpretations without a full agent loop) | A fixed single retrieval pass commits to one interpretation of an ambiguous query; agentic RAG can detect low-confidence retrieval and recover, multi-query hedges cheaply without the full loop. |
| **Compound questions with independent sub-parts** ("Causes of X, and how did A vs B respond") | **Multi-query decomposition** (Section 6) | Cheaper than a full agent loop; a single LLM call splits the question and retrieves per sub-question in parallel, no iterative reasoning needed. |

And by budget:

| Constraint | Implication |
|---|---|
| **Indexing budget is tight, query volume is high** | Favor **late chunking** (near-zero marginal indexing cost beyond normal embedding) over contextual retrieval (LLM call per chunk) or RAPTOR/GraphRAG (expensive multi-pass indexing). Query-time cost stays at baseline vector search cost either way. |
| **Indexing budget is generous, corpus is large but mostly static** | **Contextual retrieval** and **RAPTOR** front-load cost into a one-time (or infrequent) indexing pass — good fit when the corpus doesn't change often, since you amortize the indexing cost over many queries. Use prompt caching + batch APIs to keep contextual retrieval's indexing cost low even at scale. |
| **Corpus updates frequently (streaming/incremental documents)** | RAPTOR and GraphRAG both require re-clustering/re-summarizing or re-extracting-entities when new documents arrive, which is expensive to redo from scratch and awkward to do incrementally — plain hybrid search, contextual retrieval, or late chunking (all per-chunk/per-document local operations, no global corpus-level recompute) are far cheaper to keep in sync. |
| **Query-time latency is strictly bounded (e.g., a live chat SLA)** | Avoid full agentic RAG for the default path, or cap its iteration count hard; prefer single-shot retrieval (with contextual retrieval/ColBERT to maximize first-pass quality) and reserve the agentic path for a slower "deep answer" mode or for queries a cheap classifier flags as complex. |
| **Query-time latency is flexible (e.g., an overnight report generation job, an analyst research tool)** | Agentic RAG and RAPTOR's collapsed-tree retrieval (still fast) or GraphRAG global search (slower, map-reduce over communities) all become affordable — spend the latency budget where it improves answer quality most. |
| **Need exact-term precision** (legal citations, part numbers, medical codes) | **ColBERT** late interaction, or at minimum hybrid search with BM25 weighted appropriately — single dense-vector embeddings alone under-perform on exact rare-term matching regardless of which other techniques are layered on. |

A practical rollout order for a team hitting naive-RAG's ceiling: (1) add hybrid search + re-ranking if not already present (from `02-rag`) — usually the highest ROI first step; (2) add **contextual retrieval**, since it's a pure indexing-time win with no query-time cost increase and stacks with everything else; (3) if thematic/summarization queries are a meaningful fraction of traffic, add **RAPTOR** (or evaluate GraphRAG if the domain is genuinely relational); (4) if exact-term precision matters and hybrid search isn't sufficient, add a **ColBERT** re-ranking stage; (5) only after the above, and only for the subset of traffic that actually needs it, introduce **agentic RAG** — it's the most powerful and most expensive lever, so it should be reached for last and applied selectively, not as the default path for every query.

---

## Summary

| Technique | Fixes | Cost added | Where it fits in the pipeline |
|---|---|---|---|
| Contextual Retrieval | Chunks losing document-level context | LLM call per chunk at index time (cheap via prompt caching) | Indexing, before embedding/BM25 |
| RAPTOR | Can't answer broad/thematic queries from flat chunks | Clustering + LLM summarization at index time; near-zero extra query cost | Indexing (builds a tree); retrieval (collapsed-tree search) |
| ColBERT / late interaction | Single-vector embeddings blur fine-grained term matches | Larger index (token-level storage); query-time MaxSim computation | Retrieval or re-ranking stage |
| Late Chunking | Same as contextual retrieval, without an LLM call | Requires long-context embedding model + custom pooling; no extra LLM cost | Embedding step itself |
| Agentic RAG | Fixed pipeline can't handle multi-hop/ambiguous queries | Multiple LLM calls per query, variable latency | Replaces the whole retrieve-then-generate control flow |
| Multi-query/ensemble | Single query formulation misses recall on compound questions | One extra LLM call to generate variants + parallel retrievals | Wraps the retrieval call |

For the underlying fundamentals these techniques build on — chunking strategies, embedding model choice, index structures, basic top-k retrieval, cross-encoder re-ranking, hybrid dense+BM25 search, HyDE, Self-RAG, and evaluation methodology — see [`../02-rag/README.md`](../02-rag/README.md). For the graph-based alternative to RAPTOR's hierarchical-summarization approach to broad queries, see [`../13-graph-rag/README.md`](../13-graph-rag/README.md).
