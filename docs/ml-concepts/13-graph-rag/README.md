# GraphRAG — Graph-Based Retrieval-Augmented Generation

Standard RAG (see [02-rag](../02-rag/README.md)) retrieves the top-k text chunks most similar to a query embedding and stuffs them into a prompt. This works well when the answer to a question lives inside one or two chunks. It structurally fails when the answer requires connecting information that is scattered across dozens or hundreds of chunks, or reasoning over relationships between entities that are never stated together in any single passage. GraphRAG is a family of techniques that fixes this by building a **knowledge graph** over the corpus at indexing time, and using that graph — rather than (or in addition to) raw chunk similarity — to answer queries.

This document uses Microsoft's GraphRAG (Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," 2024) as the canonical reference architecture, described generically so the ideas transfer to any implementation (LlamaIndex's `PropertyGraphIndex`, LangChain's graph retrievers, Neo4j's GraphRAG stack, or a custom pipeline).

> **Status in this repo:** GraphRAG is not currently implemented anywhere in this codebase. This document is a from-first-principles reference, part of the same concept series as [02-rag](../02-rag/README.md) and [20-advanced-rag-techniques](../20-advanced-rag-techniques/README.md). If a graph-based retrieval layer is added to the Algo Akanade stack in the future, this file should be updated with exact file:line pointers, matching the convention used elsewhere in this series.

## Table of Contents

1. [The structural blind spot of vector-only RAG](#1-the-structural-blind-spot-of-vector-only-rag)
2. [What a knowledge graph is, in this context](#2-what-a-knowledge-graph-is-in-this-context)
3. [The GraphRAG indexing pipeline](#3-the-graphrag-indexing-pipeline)
4. [Query time: local search vs. global search](#4-query-time-local-search-vs-global-search)
5. [Tradeoffs vs. vector RAG](#5-tradeoffs-vs-vector-rag)
6. [Hybrid approaches](#6-hybrid-approaches)
7. [Graph representations used in practice](#7-graph-representations-used-in-practice)
8. [Worked example](#8-worked-example)
9. [Summary](#9-summary)

---

## 1. The structural blind spot of vector-only RAG

### 1.1 How vector RAG retrieves

Vector RAG's retrieval step is, mechanically, a nearest-neighbor search in embedding space:

```
query --[embed]--> q_vec
corpus --[chunk, embed]--> {c_1_vec, c_2_vec, ..., c_n_vec}
retrieve = top_k( cosine_sim(q_vec, c_i_vec) for all i )
```

Every chunk is scored **independently** against the query. There is no mechanism for a chunk to "know about" any other chunk. The scoring function only ever asks: *does this one chunk, in isolation, look like it discusses the same thing as the query?*

### 1.2 Where this breaks

This works fine for **factoid lookup** — "What is the maximum context window of model X?", "Who is the CEO of company Y?" — because the answer is a self-contained fact that lives in one place in the source text, and that passage's embedding will legitimately be close to the query's embedding.

It breaks down for two distinct classes of questions:

**Class A — Corpus-level synthesis / thematic questions.** *"What are the major themes across these 50 incident reports?"* *"What risks does this company face, according to its last 10 filings?"* No single chunk contains "the themes." The themes are a property that emerges only when you look at the corpus as a whole. Top-k similarity search will retrieve at most k chunks (k is typically 5–20), which is a tiny, essentially random-looking sample of a 50-document corpus with respect to a query like "what are the themes" — that query's embedding doesn't preferentially match any particular subset of chunks, because "themes" isn't a topic any one passage discusses. You get a biased, incomplete sample and an LLM that confidently synthesizes from whatever fragment it was handed, with no way to know it's missing 90% of the relevant material.

**Class B — Multi-hop relational questions.** *"How is Entity A connected to Entity C?"* where the only path is A → B → C, and no document ever mentions A and C together. Suppose Document 1 says "Alice founded Acme Corp," and Document 2, written years later by a different author, says "Acme Corp was acquired by Globex in 2023." A query "What is Alice's connection to Globex?" embeds close to neither chunk individually in a way that reliably surfaces both — chunk 1 is about Alice/Acme, chunk 2 is about Acme/Globex, and the query mentions Alice and Globex, so lexical/semantic overlap with either chunk alone is partial at best. Even if both chunks are retrieved, the LLM still has to notice, unaided, that "Acme Corp" is the same entity in both chunks and chain the two facts together, with no explicit signal that this chaining is required or that it's now two hops deep. As the hop count grows (A→B→C→D), the odds that *all* the intermediate chunks land in the same top-k retrieval, and that the LLM correctly stitches them in the right order, drop fast.

### 1.3 The root cause

Both failure classes trace back to the same structural fact: **vector similarity is a proxy for "aboutness," not for "connectedness" or "aggregate coverage."** Embeddings compress a chunk of text into a single point in space that captures roughly what topic that passage is about. They do not, and cannot, encode:

- *relationships* between entities that live in different chunks
- *the existence of a global structure* (a graph, a hierarchy of themes) spanning the whole corpus
- *how many other pieces of evidence exist* for a claim, or how they relate to each other

Reranking, larger k, hybrid BM25+dense retrieval, and better chunking (see [20-advanced-rag-techniques](../20-advanced-rag-techniques/README.md)) all improve the *precision* of the same fundamentally local operation. They do not add a global or relational structure. GraphRAG's premise is that some questions require indexing that structure explicitly, once, at build time — rather than trying to reconstruct it implicitly, per-query, from independent chunk lookups.

---

## 2. What a knowledge graph is, in this context

A knowledge graph here is **not** a general-purpose ontology built by hand (like Wikidata or a curated enterprise taxonomy). It's a graph *automatically extracted from unstructured text by an LLM*, specific to the corpus being indexed.

- **Nodes = entities.** People, organizations, concepts, products, locations, events — any noun phrase the extraction prompt judges to be an entity worth tracking. Each node carries: a canonical `name`, a `type` (e.g., `PERSON`, `ORGANIZATION`, `CONCEPT`, `EVENT`), and a **natural-language description** — an LLM-generated summary of what this entity is and does, synthesized from every chunk it was mentioned in.
- **Edges = relationships.** A directed or undirected connection between two entities, extracted because a chunk of text asserted some relationship between them (e.g., "Alice founded Acme Corp" → edge `(Alice) --[founded]--> (Acme Corp)`). Like nodes, edges carry an LLM-generated **description** of the relationship (not just a single verb label — a sentence or two summarizing the nature of the connection, since real relationships extracted from prose are rarely a clean single-word predicate), and often a numeric `weight` or `strength` (e.g., how many chunks/documents assert this relationship, used as a rough confidence/salience signal).

Concretely, a single extracted triple looks like:

```
Node: { name: "Alice Chen", type: "PERSON",
        description: "Alice Chen is the founder and former CEO of Acme Corp,
                      credited with pivoting the company toward cloud
                      infrastructure in 2015." }

Node: { name: "Acme Corp", type: "ORGANIZATION",
        description: "Acme Corp is a cloud infrastructure company founded
                      in 2010, acquired by Globex Inc. in 2023." }

Edge: { source: "Alice Chen", target: "Acme Corp", relationship: "founder",
        description: "Alice Chen founded Acme Corp in 2010 and served as
                      CEO until 2019.",
        weight: 4   // asserted across 4 source-chunk mentions
      }
```

Crucially, both node and edge descriptions are **LLM-synthesized summaries**, not raw extracted spans. This matters for two reasons: (1) the same entity is typically mentioned dozens of times across a corpus in fragmentary ways ("Alice," "the founder," "Chen"), and the description is the mechanism that consolidates all those mentions into one coherent, information-dense blurb; (2) it's this pre-digested description — not the original chunk text — that later gets fed back into the LLM at query time, which is what makes local and global search cheap enough to run at query time (the expensive synthesis already happened once, during indexing).

The graph is a **property graph**: nodes and edges are typed and can carry arbitrary key/value attributes (description, weight, source chunk IDs, community ID — see below), as opposed to a bare set of `(subject, predicate, object)` triples with no room for attached metadata. See [Section 7](#7-graph-representations-used-in-practice) for how this is actually stored.

---

## 3. The GraphRAG indexing pipeline

This is the expensive, offline, one-time (or periodically re-run) phase. It converts unstructured documents into a hierarchical graph structure with pre-computed summaries at multiple levels of granularity.

### 3.1 Pipeline stages

**(a) Chunk source documents.** Same as vector RAG — split raw documents into overlapping windows of text (e.g., 300–600 tokens), sized to fit comfortably inside a single LLM extraction call along with the extraction prompt and few-shot examples. Chunk size here trades off differently than in vector RAG: too large and the extraction LLM misses entities buried in a long chunk (recall drops as chunk size grows, since LLM extraction quality on a fixed task degrades with more competing content); too small and you fragment relationships across chunk boundaries, forcing more cross-chunk merging work in step (c).

**(b) Entity and relationship extraction (LLM call per chunk).** For every chunk, an LLM is prompted (via a structured extraction prompt, usually with few-shot examples and a fixed schema) to return two lists:

- entities mentioned in the chunk (name, type, short description grounded in *this chunk only*)
- relationships between pairs of those entities, as evidenced by *this chunk's text* (source entity, target entity, relationship description, evidence strength)

This is typically done as a single combined prompt asking for both lists in one pass (cheaper than two separate calls), often with a **multi-round "gleaning" step**: after the first extraction pass, the LLM is asked "did you miss anything?" and given one or more follow-up turns to catch entities it skipped, since a single extraction pass on a dense chunk reliably misses secondary entities. This is the single most expensive part of the pipeline — one (or several) LLM calls **per chunk**, and a corpus with tens of thousands of chunks means tens of thousands of LLM calls before any querying happens.

**(c) Graph construction — entity resolution / deduplication.** Every chunk was processed independently, so "Acme Corp," "Acme," and "Acme Corporation" from three different chunks need to be recognized as the *same* node before they're merged. This step:

- merges nodes with matching (or LLM/embedding-judged equivalent) names into a single node
- concatenates and then **re-summarizes** that node's description via another LLM call, folding together everything every chunk said about this entity into one coherent description (this is also where the description gets to synthesize information the LLM didn't have in any single chunk — e.g., chunk 1 says "Alice founded Acme," chunk 40 says "Acme was later acquired," and the merged description for Acme Corp can now state both)
- merges parallel edges between the same entity pair similarly, combining evidence and incrementing the weight/strength counter

This is the step most prone to error: naive string-matching entity resolution conflates distinct entities that share a name (two different "John Smith"s) or fails to merge genuinely identical entities referred to inconsistently ("the plaintiff" vs. "Jane Doe"). Production systems address this with embedding-similarity thresholds, LLM-judged merge decisions, or human-in-the-loop review for high-stakes corpora — none of which is free, and all of which can go wrong silently.

**(d) Community detection.** The resulting graph — potentially tens of thousands of nodes and edges — is partitioned into **communities**: clusters of nodes that are more densely connected to each other than to the rest of the graph. Microsoft's GraphRAG uses the **Leiden algorithm** (an improvement on Louvain that guarantees well-connected communities and avoids Louvain's disconnected-community pathology). Leiden is run **hierarchically**: it's applied once to get a coarse, top-level partition into a handful of large communities, then applied again *within each community* to get a finer sub-partition, recursively, producing a multi-level tree — level 0 (whole-graph, few large communities) down to level N (many small, tightly-related entity clusters). This hierarchy exists because different questions need different granularity: "what's this corpus about overall" wants level-0 summaries; "tell me about this narrow sub-topic" wants a deep-level community.

**(e) Community summarization (LLM call per community, per level).** For every community at every level of the hierarchy, an LLM is given that community's member entities, their descriptions, and the relationships between them, and asked to write a **structured natural-language summary** of the community — typically including a title, an overview of what the community represents, and its most important entities/claims, ranked by the edge-weight/degree salience of its members (so the most well-evidenced facts appear first, which matters because a large community's full entity list may not fit in one prompt and gets truncated by importance rank). This is the second major LLM-cost center of indexing — one call per community per level, though there are far fewer communities than chunks, so it's cheaper than step (b).

The output of indexing is: the entity/relationship graph itself, plus a **hierarchical set of community summaries** — pre-computed, reusable across every future query, however many queries are eventually asked.

### 3.2 Indexing pipeline diagram

```
┌──────────────┐
│   Source     │   raw corpus: PDFs, docs, tickets, transcripts...
│  Documents   │
└──────┬───────┘
       │ (a) chunk
       ▼
┌──────────────┐
│    Chunks    │   c_1, c_2, c_3, ... c_n   (overlapping text windows)
└──────┬───────┘
       │ (b) LLM extraction call PER CHUNK
       │     "extract entities + relationships from this chunk"
       │     (+ optional multi-round gleaning pass)
       ▼
┌──────────────────────────────────────────────────┐
│      Raw entity/relationship mentions             │
│  (Alice, PERSON) (Acme, ORG) (Alice--founded-->Acme)   per-chunk, unmerged
│  (Alice Chen, PERSON) (Acme Corp, ORG) ...             duplicated across chunks
└──────┬─────────────────────────────────────────────┘
       │ (c) entity resolution / dedup + merge
       │     LLM re-summarizes merged node/edge descriptions
       ▼
┌──────────────────────────────────────────────────┐
│              Knowledge Graph                      │
│                                                    │
│      (Alice Chen)──founded──►(Acme Corp)          │
│           │                        │              │
│        advised                 acquired_by        │
│           ▼                        ▼              │
│      (Bob Ito)              (Globex Inc.)         │
│                                    │               │
│                              competitor_of         │
│                                    ▼               │
│                             (Initech LLC)          │
└──────┬─────────────────────────────────────────────┘
       │ (d) Leiden community detection (hierarchical, recursive)
       ▼
┌───────────────────────────────────────────────────────────┐
│                 Community Hierarchy                        │
│                                                             │
│  Level 0 (coarse)      [ Community A ]      [ Community B ]│
│                          /        \                        │
│  Level 1 (finer)  [A.1]      [A.2]        [B.1]      [B.2] │
│                                                             │
│  each node above = {entities, relationships} subgraph      │
└──────┬──────────────────────────────────────────────────────┘
       │ (e) LLM summarization call PER COMMUNITY PER LEVEL
       ▼
┌───────────────────────────────────────────────────────────┐
│           Hierarchical Community Summaries                 │
│                                                             │
│  L0-A: "This cluster covers cloud infra M&A activity..."   │
│  L1-A.1: "Acme Corp's founding and early leadership..."    │
│  L1-A.2: "The Globex acquisition and its aftermath..."     │
│  ... (stored, indexed, reused by every future query)       │
└──────────────────────────────────────────────────────────────┘
```

Everything above the community-hierarchy box happens **once**, offline, ahead of any query. This is the up-front cost that Section 5 quantifies.

---

## 4. Query time: local search vs. global search

GraphRAG defines two distinct retrieval modes, chosen based on the *shape* of the question, not run interchangeably.

### 4.1 Local search — for questions about specific, named entities

Used when the query names or clearly implies specific entities: *"What products does Acme Corp sell?"*, *"Who did Alice Chen work with before founding Acme?"*

Mechanism:
1. Embed the query and match it against entity **description** embeddings (not chunk embeddings) to find one or more seed entities in the graph — this is the one place local search still uses vector similarity, just against a much smaller, denser index (entity descriptions) than raw chunks.
2. From each seed entity, **traverse outward** through its direct relationship edges (1-hop neighbors), and optionally 2-hop neighbors for entities the schema/weights suggest are highly relevant.
3. Collect the descriptions of the seed entities, their neighboring entities, and the connecting relationship descriptions. Optionally pull back in the original source chunks/text units those entities/relationships were extracted from, for extra grounding detail.
4. Assemble all of that (entity descriptions + relationship descriptions + optionally source chunks, ranked by relevance/weight and truncated to fit the context window) into the prompt, and generate the answer.

This is structurally similar to vector RAG's "retrieve then generate," except what's retrieved is a **local neighborhood of pre-summarized graph structure** rather than independently-scored chunks — so a fact spread across two source chunks (Alice→Acme in doc 1, Acme→Globex in doc 2) is retrievable in one hop because the merge step in indexing (§3.1c) already fused them into a single node's neighborhood, something plain top-k chunk retrieval cannot do.

### 4.2 Global search — for broad, thematic, corpus-wide questions

Used when the query cannot be anchored to a small number of entities: *"What are the main themes in this corpus?"*, *"Summarize the key risks discussed across these reports."*

Mechanism (map-reduce over pre-computed community summaries):
1. Pick a level of the community hierarchy to query (coarser levels for very broad questions, finer levels for more detail — often configurable, or all levels are tried and results compared).
2. **Map step:** send the query, in parallel, to an LLM alongside *each* community summary at that level, asking it to produce a partial answer plus a self-rated "how useful/relevant is this community summary for answering the query" helpfulness score, from that one community's perspective alone.
3. Filter out partial answers with a helpfulness score of zero (communities irrelevant to this query).
4. **Reduce step:** sort the remaining partial answers by helpfulness score, concatenate them (subject to the context window budget — a shorter budget means fewer, more helpful partial answers get through), and ask an LLM to synthesize a single final answer from that pool of partial answers.

This is the mechanism that actually answers Class A questions from Section 1: no single chunk, and no single entity's local neighborhood, ever needed to contain "the themes" — the map step distributes the "does this part of the corpus relate to the query" judgment across every community *in parallel*, and the reduce step is where the synthesis genuinely happens, over pre-digested per-cluster summaries rather than raw text, which is exactly what makes it feasible instead of just re-running vector RAG with a much larger k.

### 4.3 Local vs. global search diagram

```
LOCAL SEARCH                              GLOBAL SEARCH
(query names specific entities)           (broad/thematic query, no single entity anchor)

  "What does Acme Corp make?"             "What are the themes across this corpus?"
         │                                          │
         ▼                                          ▼
  ┌─────────────┐                    ┌──────────────────────────────┐
  │ embed query,│                    │  pick a community-hierarchy   │
  │ match entity│                    │  level (e.g. level 1)         │
  │ description │                    └───────────────┬───────────────┘
  │  embeddings │                                    │
  └──────┬──────┘                     ┌──────────────┼──────────────┬─────────────┐
         │ seed entity found:         ▼              ▼              ▼             ▼
         │  (Acme Corp)         ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
         ▼                      │ Summary │   │ Summary │   │ Summary │   │ Summary │
  ┌──────────────┐              │  A.1    │   │  A.2    │   │  B.1    │   │  B.2    │
  │ traverse 1-2 │              └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
  │ hops via     │                   │ MAP: LLM(query, summary) -> partial answer
  │ relationship │                   │      + helpfulness score, IN PARALLEL
  │    edges     │                   ▼             ▼              ▼             ▼
  └──────┬───────┘              partial_A.1   partial_A.2    partial_B.1   partial_B.2
         │                      (score: 8)    (score: 2)     (score: 9)    (score: 0, dropped)
         ▼                             │             │              │
  (Alice Chen)──founded──►(Acme Corp)  └──────┬──────┴──────┬───────┘
       │                        │              ▼             ▼
    advised                 acquired_by    sort by helpfulness score, keep top-scoring,
       ▼                        ▼          fill context budget
  (Bob Ito)              (Globex Inc.)            │
                                                    ▼
  collect: entity descriptions              ┌──────────────┐
  + relationship descriptions               │  REDUCE: LLM │
  + (optional) source chunks                │  synthesizes │
       │                                    │ final answer │
       ▼                                    │  from ranked │
  ┌──────────────┐                          │partial answers│
  │  LLM generates│                         └──────┬───────┘
  │  final answer │                                │
  └──────────────┘                                 ▼
                                            "Final synthesized answer"
```

The key structural difference: local search does a **single-pass, deep, narrow traversal** anchored at one or a few entities; global search does a **shallow, parallel, wide fan-out** across every community, followed by a synthesis pass. Neither mode falls back to plain top-k chunk similarity as its primary mechanism — that's the point of building the graph in the first place.

---

## 5. Tradeoffs vs. vector RAG

### 5.1 Cost

| | Vector RAG | GraphRAG |
|---|---|---|
| Indexing cost | 1 embedding call per chunk | 1+ LLM extraction call per chunk (with gleaning, often 2-3), plus 1 LLM call per merged entity/edge re-summarization, plus 1 LLM call per community per hierarchy level |
| Indexing cost driver | Embedding models are cheap, fast, and run in large batches | Every extraction/summarization step is a full LLM completion call, not a cheap embedding call — the dominant cost is number of LLM calls × tokens per call, and both are large |
| Query-time cost (factoid) | 1 embedding call + top-k retrieval + 1 generation call | Local search: 1 embedding call + graph traversal + 1 generation call (comparable to vector RAG once the graph exists) |
| Query-time cost (thematic) | Not well answerable at all (see §1) | Global search: 1 LLM call per community in the map step (parallelizable, but that's still N LLM calls for N communities) + 1 reduce call |
| Re-indexing on corpus update | Cheap — embed and insert new chunks | Expensive — new entities/relationships must be extracted, merged into the existing graph (re-resolving affected nodes), and affected communities' summaries regenerated; incremental graph updates are an active area of engineering complexity, not a solved, cheap operation |

Microsoft's own GraphRAG documentation and follow-up work is explicit that indexing cost is the primary adoption barrier — for a large corpus, the number of LLM calls at indexing time can be an order of magnitude or more than a comparable vector-RAG ingestion pass, because vector RAG's indexing is "one cheap embedding call per chunk" while GraphRAG's is "one or more expensive LLM completions per chunk, plus more LLM completions for merging and summarization on top."

### 5.2 Error surface

Graph construction introduces failure modes vector RAG simply doesn't have:

- **Extraction errors.** The LLM extracting entities/relationships from a chunk can hallucinate relationships not actually stated in the text, miss entities (especially in dense or jargon-heavy chunks), or mis-type an entity (classifying a product as an organization).
- **Entity resolution errors.** Under-merging (treating "Acme" and "Acme Corp" as different nodes) fragments the graph and weakens the traversal/community structure that's the whole point of building it. Over-merging (conflating two different "John Smith"s into one node) actively corrupts the graph — every fact ever said about either John Smith is now attributed to a single conflated entity, poisoning every future query touching that node, silently and without an easy warning signal.
- **Community-boundary artifacts.** Leiden clustering is a heuristic; near a community boundary, semantically related entities can land in different clusters, and each level's community summary is only as good as the entities the algorithm happened to group together at that resolution.
- **Compounding across the pipeline.** Because each stage (extraction → merge → community detection → summarization) feeds the next, an error introduced early (a missed entity in extraction) can silently propagate and shape every downstream community summary that entity should have been part of, with no single obvious place to catch it.

None of this is unfixable — production systems mitigate it with better prompts, embedding-assisted entity resolution, human review loops, and evaluation harnesses — but it is real engineering surface area that a vector-only pipeline (chunk → embed → store) simply does not have to deal with.

### 5.3 When each wins

- **Vector RAG wins** for narrow factoid lookup, FAQ-style Q&A, and any corpus where the questions are known in advance to be answerable from a single passage. It's cheaper to build, cheaper to update, simpler to operate, and there's a decade of tooling (vector DBs, rerankers, hybrid search) built around it.
- **GraphRAG wins** for questions that are structurally about the corpus as a whole (thematic summarization, "what connects these things," sensemaking over a large document set) or that require chaining facts across documents that were never co-located in any single passage. These are exactly the queries where vector RAG's top-k similarity search has no mechanism to even *attempt* an answer, regardless of how much you tune k, chunking, or reranking.

The honest framing from the GraphRAG literature itself: this is not "GraphRAG is better than RAG," it's "GraphRAG answers a class of queries vector RAG cannot answer at all, at a meaningfully higher indexing cost, and for the queries vector RAG *can* already answer, plain vector RAG remains cheaper and just as accurate."

---

## 6. Hybrid approaches

Because the two approaches have non-overlapping strengths, production systems increasingly run **both indexes side by side** rather than choosing one:

1. **Query routing.** Classify the incoming query (via a lightweight classifier, an LLM call, or heuristics on query phrasing — presence of "compare," "summarize across," "how are X and Y related" vs. "what is," "when did") and route factoid-shaped queries to the vector index and thematic/relational-shaped queries to the graph's global or local search. This is the same routing-by-intent principle used elsewhere in RAG system design (see [06-intent-types](../06-intent-types/README.md) for this repo's general approach to intent classification) applied specifically to picking a retrieval backend.
2. **Vector search as a graph entry point.** Rather than always doing plain entity-name matching to seed local search (§4.1), embed entity descriptions into the same (or a parallel) vector index used for chunk retrieval, and use similarity search to find the best-matching entity node(s) for a query that doesn't name an entity explicitly, then traverse from there. This gets local search's contextual-neighborhood benefit even when the query is phrased loosely rather than as an exact entity lookup.
3. **Combined context assembly.** For a single query, retrieve both top-k chunks (vector) *and* the local-search graph neighborhood for any entities matched in the query, then merge both into the generation prompt — chunks supply verbatim detail and quotes, graph context supplies cross-document relationship structure and pre-synthesized descriptions. LlamaIndex's `PropertyGraphIndex` and several commercial GraphRAG offerings (Neo4j, Microsoft's own reference implementation extensions) support exactly this dual-retrieval pattern natively.
4. **Fallback chains.** Attempt local search first (cheap, fast, works if the query anchors to known entities); if the seed-entity match confidence is low (no entity in the graph is a good semantic match for the query), fall back to plain vector chunk retrieval rather than forcing a graph traversal from a bad seed.

The unifying principle: **the graph index and the vector index are not competing designs, they are two different indexes over the same corpus, each good at answering a different shape of question** — and a system that has invested in building both can route to whichever fits, rather than compromising on a single retrieval strategy for every query.

---

## 7. Graph representations used in practice

Once entities and relationships are extracted, they need to be stored and queried somewhere. Three representations dominate:

### 7.1 Property graphs

The most common representation for GraphRAG specifically. Nodes and edges are typed objects that can carry arbitrary attributes — the `description`, `weight`, `source_chunk_ids`, `community_id` fields described in Section 2 attach naturally as node/edge properties. Queried with a graph query language such as **Cypher** (Neo4j, Memgraph) or **Gremlin** (Apache TinkerPop-compatible stores, Amazon Neptune). A Cypher query for local search's traversal step looks roughly like:

```cypher
MATCH (seed:Entity {name: "Acme Corp"})-[r:RELATED_TO]-(neighbor:Entity)
RETURN seed.description, r.description, neighbor.description, r.weight
ORDER BY r.weight DESC
LIMIT 20
```

Property graphs are the natural fit because GraphRAG's value comes largely from the rich natural-language descriptions attached to nodes/edges, and a triple-store schema (below) makes attaching a paragraph-length description to an edge awkward — you'd need to reify the edge as its own node just to hang a `description` property off it. Microsoft's reference GraphRAG implementation stores its output as a set of parquet files (entities, relationships, communities, community reports tables) that can be loaded into a property-graph database or queried directly with a dataframe library, rather than mandating a specific graph database.

### 7.2 RDF triples

The traditional Semantic Web representation: every fact is a `(subject, predicate, object)` triple, e.g. `(:AcmeCorp :foundedBy :AliceChen)`, queried with **SPARQL**. RDF triple stores (Apache Jena, Blazegraph, GraphDB) are the standard choice when interoperability with existing ontologies (schema.org, Wikidata, domain-specific ontologies like SNOMED CT for medicine) matters more than attaching free-text LLM-generated descriptions — RDF's reification mechanism for attaching metadata to a triple (making a statement *about* a statement) is more cumbersome than a property graph's native edge properties, so RDF is less commonly chosen for LLM-extracted GraphRAG specifically, but remains dominant in domains with pre-existing large-scale curated ontologies that a GraphRAG-style extraction pipeline might want to align against or extend.

### 7.3 Lightweight in-memory representations

For smaller corpora (a few hundred to a few thousand chunks — the range where a dedicated graph database is operational overhead the project doesn't need), the entire graph is just held as **in-memory Python objects** — a `networkx.Graph` or `networkx.DiGraph`, or even bare dictionaries/dataclasses (`{entity_id: EntityNode}`, `{(src, tgt): EdgeRecord}`) serialized to disk as JSON or a pickle between runs. `networkx` specifically is popular for this because it ships Leiden/Louvain-compatible community-detection bindings (via `python-igraph` or `leidenalg` interop) directly usable on an in-memory graph, so the entire pipeline in Section 3 — build graph, run Leiden, walk neighbors for local search — can be implemented without standing up any external database at all. This is the right choice whenever the graph comfortably fits in memory and doesn't need concurrent multi-process access or a query language beyond "get neighbors of this node" and "get nodes in this community," which covers the large majority of GraphRAG use cases below enterprise scale.

**Choosing between them:** in-memory (`networkx`) for prototypes and small/medium corpora with a single-process indexing+query pipeline; a property-graph database (Neo4j et al.) once the graph needs to be queried by multiple services, updated incrementally in production, or grows past comfortable single-machine memory; RDF/SPARQL specifically when the project needs to interoperate with or extend an existing formal ontology rather than working purely from LLM-extracted, corpus-specific entities.

---

## 8. Worked example

Corpus: 3 short internal documents about a fictional company.

- Doc 1: "Alice Chen founded Acme Corp in 2010, building it into a cloud infrastructure provider."
- Doc 2: "In 2023, Globex Inc. acquired Acme Corp for $400M, citing Acme's strong customer base in fintech."
- Doc 3: "Globex Inc.'s main competitor, Initech LLC, responded to the Acme acquisition by accelerating its own cloud roadmap."

**Indexing:**
- Chunking: each document is short enough to be its own chunk (3 chunks total).
- Extraction: chunk 1 → entities `{Alice Chen: PERSON, Acme Corp: ORG}`, relationship `Alice Chen --founded--> Acme Corp`. Chunk 2 → entities `{Globex Inc.: ORG, Acme Corp: ORG}`, relationship `Globex Inc. --acquired--> Acme Corp`. Chunk 3 → entities `{Globex Inc.: ORG, Initech LLC: ORG, Acme Corp: ORG}`, relationship `Initech LLC --competitor_of--> Globex Inc.`.
- Merge: `Acme Corp` appears in all three chunks and is merged into one node with a synthesized description: "Acme Corp is a cloud infrastructure provider founded by Alice Chen in 2010, with a strong fintech customer base, acquired by Globex Inc. in 2023 for $400M." Note this sentence combines facts from all three source chunks — no single chunk said all of that.
- Community detection: with only 4 entities and 3 edges, Leiden would likely put all of them in one community at level 0 (too small a graph to meaningfully sub-partition) — this toy example is below the scale where hierarchical communities add value, but at corpus scale (hundreds of entities) this is where distinct communities ("Acme/Globex M&A cluster" vs. an unrelated "product roadmap cluster" elsewhere in the corpus) would emerge.
- Community summary: "This cluster describes the 2023 acquisition of cloud provider Acme Corp by Globex Inc., and competitive response from Initech LLC."

**Query — local search:** *"What happened to the company Alice Chen founded?"* → embed query, match entity description embeddings → seed = `Acme Corp` node (via its description mentioning Alice Chen) → traverse 1 hop → neighbors `Alice Chen` (founded), `Globex Inc.` (acquired_by) → assemble descriptions → answer: "Alice Chen founded Acme Corp in 2010; it was later acquired by Globex Inc. in 2023 for $400M." This chains a fact from doc 1 and a fact from doc 2 that never co-occur in either source document — exactly the multi-hop case vector RAG structurally struggles with (Section 1.2, Class B), resolved here because the merge step in indexing already fused "Acme Corp" into one node whose neighborhood spans both facts.

**Query — global search:** *"What are the main events discussed in this corpus?"* → map step queries the (single, in this toy example) community summary → partial answer: "The main event is the 2023 acquisition of Acme Corp by Globex Inc., with competitive fallout involving Initech LLC" → reduce step (trivial with one partial answer here) → final answer matches the community summary directly. At real corpus scale, this same mechanism would run across dozens of communities in parallel and reduce-synthesize across all of them — which is the part a top-k vector search over the raw chunks has no equivalent mechanism for producing.

---

## 9. Summary

- Vector-only RAG scores chunks independently against a query; it has no representation of cross-chunk relationships or corpus-wide structure, so it cannot answer questions that require synthesis across many documents or multi-hop reasoning over entity relationships — no single chunk contains the answer to those questions.
- GraphRAG addresses this by extracting a knowledge graph (entities as nodes, relationships as edges, both with LLM-generated natural-language descriptions) from the corpus at indexing time, then clustering it into a hierarchy of communities (via Leiden) and pre-summarizing every community at every level.
- At query time, **local search** traverses the graph outward from matched entities for entity-specific questions; **global search** runs a map-reduce over pre-computed community summaries for broad, thematic questions that no single passage or neighborhood covers.
- This capability comes at a real cost: indexing requires many more LLM calls than vector RAG (extraction, entity resolution/merging, community summarization at every hierarchy level), and the graph-construction steps (extraction, entity resolution) are genuinely error-prone in ways vector RAG's indexing simply isn't.
- Vector RAG remains the better, cheaper choice for narrow factoid lookup; GraphRAG earns its cost specifically on thematic/relational queries vector RAG cannot structurally answer — which is why production systems increasingly run both indexes side by side and route queries to whichever fits, rather than treating this as an either/or choice.
