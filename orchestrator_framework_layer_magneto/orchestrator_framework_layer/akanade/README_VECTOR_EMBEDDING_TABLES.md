# Vector embedding tables — every place this system stores and searches embeddings

Five real, independent embedding stores exist in this codebase, each
solving a different problem. None of them share data with each other —
this document is the one map of what each "table" actually holds, how
it embeds text, and how it's searched.

## 1. The two embedders

Every store below embeds text via ONE of these two interchangeable
implementations (`app/retrieval/vectorstore.py`):

| | `HashEmbedder` | `MiniLMEmbedder` |
|---|---|---|
| Dimensions | 384 | 384 |
| Dependency | None — stdlib `hashlib`/`math` only | `sentence-transformers` (`all-MiniLM-L6-v2`, ~80MB) |
| How it embeds | Deterministic hashed bag-of-words: each non-stopword token's MD5 hash picks a bucket in a 384-slot vector, incremented, then L2-normalized | A real trained sentence embedding model |
| Real semantic understanding (synonyms, paraphrase) | No — "cancel my subscription" and "end my plan" share zero hashed buckets unless they share literal tokens | Yes |
| Where it's used | Every store below EXCEPT `ChromaStore`'s optional MiniLM path | `ChromaStore` when `EMBEDDER=auto` or `minilm` AND the dependency is actually installed |

`HashEmbedder` filters common stopwords (`the`, `is`, `my`, `how`, ...)
before hashing — a real, empirically-confirmed fix for a specific
failure mode: two topically UNRELATED queries that happen to share
function words ("how do I reset my password" vs. "how do I delete my
account") were scoring as similar as genuinely related ones, because
stopword overlap dominated the hashed vector. This is still a
bag-of-words scheme, not a claim of semantic equivalence to MiniLM —
`make_embedder()` picks MiniLM automatically when it's actually
installed (`EMBEDDER=auto`, the default) and falls back to
`HashEmbedder` otherwise, so the pipeline is always real even before
the model file lands on a box.

Both embedders return **unit-normalized** vectors, which is why every
store below computes similarity as a plain dot product (`Σ xᵢyᵢ`) —
mathematically identical to cosine similarity once both vectors already
have norm 1, without paying for the extra division.

## 2. The five tables

### 2.1 The PM domain's task/note index (`app/retrieval/vectorstore.py`)

The original §5 vector store — one Chroma collection named `"algopm"`
(via `chromadb.PersistentClient`, disk-persisted under `CHROMA_DIR`),
falling back to a pure-Python `MemoryStore` (cosine over an in-process
dict) if `chromadb` isn't importable at all. Populated automatically
by `pm_service`'s own mutation hooks — every task create/update
re-embeds `"{title}. {description}"` and upserts it, not through any
generic "index this text" call. RBAC-filtered AFTER retrieval
(`post_filter()`): a vector index has no concept of "who can see this,"
so results are scoped down to the caller's visible projects before
anything reaches the LLM or user — Executive sees everything, every
other role (including the confusingly-named `Admin`, mapped from
Senior Manager) only sees projects they're actually a member of.

**Known, documented CVE exposure**: `chromadb==1.5.9`, as pinned, has 5
real HIGH-severity CVEs (checked against OSV.dev), but all five are in
Chroma's NETWORKED HTTP server mode (multi-tenant auth, `/api/v2/
tenants/...`) — `PersistentClient` here never starts a server or
listens on a network at all, so they're out of scope for how this
code actually runs. Re-check OSV.dev before ever switching this store
to `HttpClient`/`chroma run` server mode; that would put these exact
CVEs back in play.

### 2.2 The RAG registry — named, pluggable document stores (`app/framework/rag.py`)

`RagRegistry` holds any number of named `VectorStoreAdapter`s an
application or agent can address by name (an agent's
`knowledge_sources`, and every `POST /api/v1/rag/*` endpoint's `store`
field) — ships with two by default:

- **`"pm-index"`** (`ChromaVectorStoreAdapter`) — a read-through wrapper
  around §2.1's SAME Chroma table above. Deliberately has no
  `add_documents()` (raises `NotImplementedError`) — that store's RBAC
  filtering assumes every entry is a real task/note with a real owning
  project, so generic documents can't be shoved into it.
- **`"default"`** (`InMemoryVectorStoreAdapter`) — a genuinely separate,
  independent table: pure-Python cosine similarity over `HashEmbedder`
  vectors, runtime-writable via `add_documents()`, zero extra
  dependencies. This is what `document_pipeline.py`'s CSV/PDF/TXT/MD
  ingestion writes into, and what `ground_truth.py` (§ the SLM
  Generation Layer, `README_SLM_GENERATION_LAYER.md`) optionally
  retrieves grounding context from.

Register more at runtime with `rag_registry.register(name, your_adapter)`
— any object satisfying the tiny `VectorStoreAdapter` Protocol
(`add_documents`, `search`) works, no subclassing required.

**Retrieval-speed optimization** — once an `InMemoryVectorStoreAdapter`'s
corpus crosses `IVF_MIN_VECTORS` (50), `search()` automatically builds
and uses an IVF (Inverted File) index instead of a brute-force scan —
see §3 below. Below that threshold, brute force stays both exact and
already fast enough that clustering overhead wouldn't pay for itself.

### 2.3 Agent-routing memory (`app/framework/task_memory.py`)

Not a document store at all — a bounded `deque` (default `maxlen=1000`)
of `(quantized_embedding, agent_id, reward)` triples, one per task the
Adaptive Router's bandit has actually scored an agent against.
`bonus_for(task_text, agent_id)` finds the closest-matching past entry
by cosine similarity (`_MATCH_THRESHOLD = 0.75` — looser than §2.4's
0.93, since this wants "the same KIND of request," not "the identical
string") and returns its reward, scaled into `[-0.5, +0.5]`, as a
WARM-START bonus on top of the hand-designed `utility_prior()` — see
`ARCHITECTURE.md` §3. Every stored vector is scalar-quantized (§3
below) before being kept in the deque.

### 2.4 Query-refinement memory (`app/framework/query_refinement.py`'s `RefinementMemory`)

The same shape as §2.3, applied one layer earlier in the pipeline:
every ACCEPTED past refinement (raw query → cleaned-up query) is
remembered by the raw query's quantized embedding. A new query close
enough to one already seen (cosine ≥ 0.93 — much tighter than §2.3's
0.75, since this wants near-duplicate phrasing, not merely "similar
kind of task") reuses that cached refinement directly, skipping the
LLM call entirely. This is what makes query refinement genuinely
cheaper over time rather than a fixed per-turn tax: the effective LLM
round-trip rate on repeat/near-duplicate phrasing drops the longer the
process runs.

### 2.5 Domain-SLM routing memory (`app/llm/domain_router.py`'s use of `TaskMemory`)

A SEPARATE `TaskMemory` instance (not shared with §2.3's agent-routing
one — each `DomainRouter`/`ContextualThompsonRouter` owns its own),
remembering `(query, domain, reward)` triples so Domain mode
(`README_SLM_ALL.md` §6) can tell "which registered Algo-Akanade-SLM
domain variant has handled a query LIKE this one well before." Unlike
§2.3's usage (a one-time cold-start seed via `LinearBanditCore.ensure()`),
this bonus is added live on EVERY routing decision — see that module's
docstring for why domain relevance needs a per-call content signal a
structured feature vector can't provide, while agent selection already
has one (`TaskContext.task_type`).

## 3. How embeddings are actually stored compactly (`app/framework/vector_storage_optimizer.py`)

Two real, independently-useful techniques, layered on top of §2.2's
`InMemoryVectorStoreAdapter` and reused directly by §2.3/§2.4's memory
tables:

**Scalar quantization** (used by every `TaskMemory`/`RefinementMemory`
entry) — asymmetric per-vector uint8 quantization:

```
v_min = min(v),  v_max = max(v)
scale = (v_max - v_min) / 255            (1.0 if v_max == v_min, avoids ÷0)
q_i   = round((v_i - v_min) / scale)     clamped to [0, 255], stored as 1 byte
dequantize:  v_i' = q_i * scale + v_min
```

A real 4x memory reduction per vector (float32/float64 → uint8), with
a small, BOUNDED approximation error — each dequantized component is
within `scale/2` of the original. The same technique FAISS's
`ScalarQuantizer` and most production vector databases use as their
cheapest compression tier; nothing exotic, just applied consistently
everywhere this codebase keeps embeddings in memory long-term.

**IVF (Inverted File) clustering** (used by `InMemoryVectorStoreAdapter`
once it crosses 50 vectors) — brute-force search is `O(n·d)`: every
query compares against every stored vector. IVF partitions the `n`
vectors into `K = round(√n)` clusters via Lloyd's k-means:

```
assignment:  cluster(xᵢ) = argmin_k ‖xᵢ − μ_k‖²
update:      μ_k = (1/|C_k|) Σ_{xᵢ ∈ C_k} xᵢ
```

repeated until assignments stop changing (or `max_iters`). A query then
only computes exact distance against the `n_probe` nearest clusters'
members instead of all `n` vectors:

```
search cost ≈ O(K·d) [scan every centroid] + O((n/K)·n_probe·d) [scan probed clusters]
```

minimized at `K = √n`, giving `O(√n·d)` total — the standard
first-level technique real ANN systems (FAISS's `IndexIVFFlat`, etc.)
use before adding further compression on top. `n_probe=1` (search only
the single best-matching cluster) trades a little recall for the full
`√n` speedup; `InMemoryVectorStoreAdapter` uses `n_probe = n_clusters // 4`
as a recall/speed middle ground.

Both techniques are opt-in and scale-gated (`IVF_MIN_VECTORS`) — a
handful of vectors gets zero benefit from clustering overhead and stays
on plain brute-force scan, the same "small corpus = simple defaults"
philosophy `document_pipeline.py`'s own retrieval-breadth scaling uses.

## 4. Quick reference — which table am I actually looking at?

| Table | File | Embedder | Persisted? | Searchable via |
|---|---|---|---|---|
| PM task/note index | `retrieval/vectorstore.py` (`vector_index`) | MiniLM or Hash | Yes — Chroma on disk (`CHROMA_DIR`) | `VectorIndex.search()`, RBAC-filtered |
| `"pm-index"` RAG store | `framework/rag.py` (`ChromaVectorStoreAdapter`) | Same as above | Same table as above | `POST /api/v1/rag/search`\|`/query` |
| `"default"` RAG store | `framework/rag.py` (`InMemoryVectorStoreAdapter`) | Hash | No — in-process only | Same endpoints, `store="default"` |
| Agent-routing memory | `framework/task_memory.py` | Hash | No — in-process `deque`, bounded | `bonus_for()`, internal to the Adaptive Router |
| Query-refinement memory | `framework/query_refinement.py` | Hash | No — in-process `deque`, bounded | Internal to `query_refinement.py`'s gate |
| Domain-SLM routing memory | `llm/domain_router.py` (its own `TaskMemory`) | Hash | No — in-process `deque`, bounded | Internal to `DomainRouter.choose()` |

Everything under "No — in-process" resets on a process restart — these
are warm-start/optimization layers on top of the REAL decision-makers
(the bandits, the LLM calls), never the only copy of anything a user
would notice missing. The two Chroma-backed rows are the only
genuinely persistent embedding storage in the system.
