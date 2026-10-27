# Data model — every store, table, and embedding this service persists

One backend, three kinds of persistence, all real and grounded in
`backend/app/db.py`/`backend/app/retrieval/vectorstore.py` — no
speculative future schema, only what's actually written to today.

## The document store (`app/db.py`)

Every table below shares ONE tiny API — `put(pk, sk, item)` /
`get(pk, sk)` / `scan(pk_prefix, sk_prefix)` / `delete(pk, sk)` — backed
by either `LocalStore` (one JSON file per table, `DATA_DIR`, zero
infra — the default, and what every test in this repo runs against) or
`DynamoStore` (real DynamoDB or `dynamodb-local`, `STORAGE_BACKEND=dynamodb`).
Neither the service layer nor any table below knows which backend is
active — see `db.py`'s own docstring.

| Table | PK | SK | What it holds |
|---|---|---|---|
| `Users` | `USER#{username}` | `PROFILE` | Demo-mode user accounts (name, role, password hash) — unused when `REAL_APP_API_BASE_URL` is set, since auth then defers to the real PM app entirely |
| `Spaces` | `SPACE#{id}` | `META` | Demo PM data — top-level spaces |
| `Projects` | `PROJECT#{id}` | `META` | Demo PM data — projects, `member_roles` map |
| `Tasks` | `TASK#{id}` | `META` | Demo PM data — tasks |
| `Sprints` | `SPRINT#{id}` | `META` | Demo PM data — sprints |
| `Subtasks` | `SUBTASK#{id}` | `META` | Demo PM data — subtasks |
| `ChatSessions` | `USER#{user_id}` | `SESSION#{session_id}` | One row per (user, chat session): rolling message window, `pendingConfirmation`/`pendingClarification`, the running `summary` string (see `agent/memory.py`) |
| `Providers` | `PROVIDER#{id}` | `META` | Admin-added LLM providers (`admin_providers.py`) — API keys stored **plaintext** (a known, accepted gap — see `OAuthConnections` below for the higher-stakes case that DOES get encrypted) |
| `OAuthConnections` | `USER#{user_id}` | `PROVIDER#{github\|linear\|google}` | Per-user OAuth tokens (`oauth_store.py`) — `access_token`/`refresh_token` **encrypted at rest** (`app/crypto.py`, Fernet), `expires_at`, `scopes`, `connected_at` |
| `BanditState` | `OPTIMIZER#{name}` | `STATE` | RL optimizer posteriors, durable across restarts (`framework/bandit_persistence.py`) — currently `security_optimizer` is the one wired-up example; see the Security README |

Demo-mode tables (`Users`/`Spaces`/`Projects`/`Tasks`/`Sprints`/`Subtasks`)
exist so the chatbot works fully standalone against a seeded Compass OS
dataset with zero external dependencies — see `seed.py`. When
`REAL_APP_API_BASE_URL` is set, PM reads/writes go to the real app's own
REST API instead (`services/real_pm_service.py`) and these tables sit
unused for that traffic.

## The vector store (`app/retrieval/vectorstore.py`)

Two interchangeable backends, selected automatically (never a hard
dependency on either):

- **`ChromaStore`** — `chromadb.PersistentClient`, disk-persisted at
  `CHROMA_DIR`, one collection (`"algopm"`), cosine similarity
  (`hnsw:space: cosine`). The real backend whenever `chromadb` is
  installed (it's a required dependency — see `requirements.txt`).
- **`MemoryStore`** — plain in-process dict + cosine, the test double
  and zero-infra fallback if Chroma ever fails to initialize.

Neither store cares which embedder produced a vector — `upsert(doc_id,
vector, text, meta)` / `query(vector, k)` is the whole contract.

**Embeddings — two interchangeable embedders** (`make_embedder()`,
`EMBEDDER=auto|minilm|hash`):

- **`MiniLMEmbedder`** — real semantic embeddings, `sentence-transformers`'
  `all-MiniLM-L6-v2` (384-dim, local, CPU, ~80MB) — used automatically
  whenever that optional dependency is installed (`EMBEDDER=auto`, the
  default).
- **`HashEmbedder`** — a deterministic, dependency-free fallback: a
  384-dim hashed bag-of-words vector (MD5 token → bucket index, L2
  normalized), so retrieval is always REAL — never a stub — even before
  the MiniLM model file lands on the box, and what every hermetic test
  in this repo runs against. Stopword-filtered before hashing (a real,
  empirically-confirmed fix — see the module's own docstring for the
  "how do I reset my password" vs "how do I delete my account"
  false-positive it closed).

What gets embedded: `app/framework/document_pipeline.py`'s
`ingest_path()` chunks a document (`rag.py`'s `chunk_text`, word-boundary-
safe overlap) and upserts each chunk's embedding + text + metadata
(source doc id, chunk index) into whichever store/embedder pair is
active. `framework/rag_optimizer.py` is the RL layer deciding WHICH
registered store + WHICH retrieval algorithm (`vector`/`lexical_rerank`/
`hybrid`) answers a given query — see the main README's RL section and
`README_REINFORCEMENT_LEARNING.md`.

## RBAC on retrieval

A vector index has no concept of "who can see this chunk" — `post_filter()`
(`vectorstore.py`) runs on every query result, filtering out anything
the calling user's real project membership wouldn't let them see,
*before* it ever reaches the LLM or the user. Retrieval never bypasses
the same visibility rules a direct PM API call would enforce.
