# SLM Coding Test Suite

`scripts/run_slm_test_suite.py` — 105 real coding prompts run against a
named provider (default `ollama-coder`), each scored for pass/fail,
latency, token usage, and a set of cheap deterministic quality checks.

## Categories

| Category | Count | What it exercises |
|---|---|---|
| `generate` | 30 | New code from scratch — functions, classes, a FastAPI endpoint, a Dockerfile, an async fetcher |
| `debug` | 25 | Real, specific bugs (race conditions, off-by-one, mutable defaults, silent exception swallowing, a multi-symptom incident) |
| `refactor` | 15 | Structural changes required to preserve exact behavior |
| `test` | 15 | Writing tests, including edge cases and mocking |
| `repo_analysis` | 10 | Repository-wide questions (TODOs, env vars, security review across a whole codebase) |
| `general_coding` | 10 | Plain knowledge/reasoning questions, deliberately NOT keyword-matched by the intent reasoner — the fallback bucket needs its own coverage too |

These are the same six categories `llm/intent_reasoner.py` classifies
into (see `README_SLM_REASONING_LAYER.md`) — the suite and the
reasoning layer share one taxonomy on purpose.

## The full prompt list

The source of truth is `PROMPTS` at the top of
`scripts/run_slm_test_suite.py` — this is a generated, exact listing of
it, not a separately hand-maintained copy.

### Generate (30)

- **GEN-01** — Write a Python function that reverses a string.
- **GEN-02** — Write a Python function that checks whether a number is prime.
- **GEN-03** — Write a function that returns the nth Fibonacci number iteratively.
- **GEN-04** — Write a Python function to flatten a nested list of arbitrary depth.
- **GEN-05** — Write a function that merges two sorted lists into one sorted list.
- **GEN-06** — Write a Python decorator that times how long a function takes to run.
- **GEN-07** — Write a context manager that temporarily changes the working directory.
- **GEN-08** — Write a function that validates whether a string is a valid email address.
- **GEN-09** — Write a Python class representing a simple stack with push/pop/peek.
- **GEN-10** — Write a binary search function over a sorted list.
- **GEN-11** — Write a function that counts word frequency in a block of text.
- **GEN-12** — Write a FastAPI endpoint that accepts a POST with a JSON body and echoes it back.
- **GEN-13** — Write a Dockerfile for a Python 3.12 service served by uvicorn on port 8000.
- **GEN-14** — Write a bash script that rotates and gzips log files older than 30 days.
- **GEN-15** — Write a Python script that reads a CSV and prints the row and column counts.
- **GEN-16** — Implement a thread-safe LRU cache in Python supporting concurrent reads and writes.
- **GEN-17** — Write a Python function that computes the SHA-256 hash of a file's contents.
- **GEN-18** — Write a generator function that yields prime numbers indefinitely.
- **GEN-19** — Write a function that deep-merges two Python dictionaries.
- **GEN-20** — Write a simple token-bucket rate limiter class in Python.
- **GEN-21** — Write a Python function that converts a snake_case string to camelCase.
- **GEN-22** — Write a retry decorator with exponential backoff for a flaky function.
- **GEN-23** — Write a Python function that parses a simple INI-style config string into a dict.
- **GEN-24** — Write a function that finds the longest common prefix among a list of strings.
- **GEN-25** — Write a Python class for a min-heap supporting push and pop-min.
- **GEN-26** — Write a function that validates balanced parentheses/brackets/braces in a string.
- **GEN-27** — Write a Python async function that fetches multiple URLs concurrently and returns their status codes.
- **GEN-28** — Write a function that computes the edit distance (Levenshtein) between two strings.
- **GEN-29** — Write a simple event emitter class supporting on/off/emit.
- **GEN-30** — Write a Python function that paginates a list into chunks of a given size.

### Debug (25)

- **DBG-01** — This function should return the max of a list but returns the wrong value for negative numbers (mutated-default-style bug); fix it.
- **DBG-02** — Debug why this Kafka consumer keeps rebalancing every few seconds.
- **DBG-03** — Find and fix a race condition in a Python asyncio producer-consumer system.
- **DBG-04** — This recursive factorial function causes a RecursionError for n=0. Why, and how do you fix it?
- **DBG-05** — Debug "MongoServerError: $changeStream only supported on replica sets."
- **DBG-06** — Why does this loop never terminate (a `continue` before the increment)?
- **DBG-07** — This Python dict comprehension raises "unhashable type: list" — why, and what's the fix?
- **DBG-08** — Debug why two batch pods land on the same node despite anti-affinity.
- **DBG-09** — This function leaks file handles under load — find the bug.
- **DBG-10** — Fix this Kubernetes Deployment YAML — the readiness probe never passes.
- **DBG-11** — This binary search has an off-by-one bug causing an infinite loop. Find and fix it.
- **DBG-12** — Diagnose a multi-symptom incident: Kafka consumer lag climbing, a TRT engine stalled, rising false positives — likely root cause and fix order?
- **DBG-13** — This Python service's memory grows continuously over hours — likely causes, and how to confirm which one?
- **DBG-14** — This mutable-default-argument bug causes state to leak between calls — find it.
- **DBG-15** — Fix a PostgreSQL transaction bug causing duplicate financial transactions under concurrent requests.
- **DBG-16** — This async function never actually awaits its coroutine — find the bug.
- **DBG-17** — Why would AUROC of 0.91 offline and deployed precision of only 0.18 both be true at once?
- **DBG-18** — This code silently swallows every exception with a bare `except: pass` — find and fix it.
- **DBG-19** — Analyze and fix a database deadlock involving multiple concurrent workers.
- **DBG-20** — This regex-based email validator rejects valid addresses with a "+" — why, and fix it.
- **DBG-21** — 35% of served responses still use the old model after a deployment meant to fully switch over — likely causes?
- **DBG-22** — This function compares floats with `==` and fails intermittently — find the bug.
- **DBG-23** — Given failing logs showing a connection-pool-exhausted error under load, identify the root cause and minimal production-safe fix.
- **DBG-24** — This code mutates a list while iterating over it, skipping elements — find the bug.
- **DBG-25** — Profile a slow API and describe how you'd determine whether the bottleneck is CPU, database, network, or serialization.

### Refactor (15)

- **REF-01** — Refactor this 400-line function into smaller modules without changing behavior.
- **REF-02** — Refactor this nested-if chain into a cleaner form, preserving exact behavior.
- **REF-03** — Refactor this repeated try/except block into a reusable decorator without changing what it catches.
- **REF-04** — Extract the duplicated validation logic in three functions into one shared helper.
- **REF-05** — Refactor this class to use dataclasses instead of a hand-written `__init__`/`__repr__`.
- **REF-06** — Rewrite this callback-based function using async/await, preserving external behavior exactly.
- **REF-07** — Refactor this global-mutable-state counter into a proper class, without changing any caller's code.
- **REF-08** — Simplify this deeply nested list comprehension into clearer, equivalent code.
- **REF-09** — Refactor a function that returns different types depending on a flag into two clearly-named functions.
- **REF-10** — Rename and reorganize a poorly-named `utils.py`'s functions into cohesive groups without breaking imports.
- **REF-11** — Remove magic numbers, replacing them with named constants.
- **REF-12** — Refactor a synchronous file-processing loop to process files concurrently, preserving output order.
- **REF-13** — Convert a class-based singleton into a simpler module-level singleton, without changing its public API.
- **REF-14** — Refactor a SQL query built via string concatenation into a parameterized query, without changing results.
- **REF-15** — De-duplicate a pair of near-identical functions into one, parameterized by their one real difference.

### Test (15)

- **TST-01** — Write unit tests for this alert-rule cooldown logic.
- **TST-02** — Write pytest tests for a function that returns precision, recall, and F1 without zero-division errors.
- **TST-03** — Write a test that reproduces a parser bug (crashes on empty input) before fixing it.
- **TST-04** — Write unit tests covering edge cases (empty list, single element, duplicates) for binary search.
- **TST-05** — Write comprehensive pytest tests for a FastAPI service including failures, concurrency, and edge cases.
- **TST-06** — Write a test that mocks an external HTTP call and asserts the function retries on a 500.
- **TST-07** — Write tests for a rate limiter verifying it allows exactly N requests per window and rejects N+1.
- **TST-08** — Write a parametrized pytest test for an `is_prime` function (positive/negative/zero).
- **TST-09** — Write a test asserting a thread-safe counter produces the correct total under concurrent increments.
- **TST-10** — Write tests for a CSV parser handling quoted commas and missing trailing newlines.
- **TST-11** — Write a test verifying a cache eviction policy actually evicts the least-recently-used entry.
- **TST-12** — Write tests asserting a JSON schema validator rejects a payload missing a required field.
- **TST-13** — Write a test verifying a decorator preserves the wrapped function's name and docstring.
- **TST-14** — Write tests for a function that should raise `ValueError` on negative input, not silently clamp it.
- **TST-15** — Write a test suite for a stack class covering push, pop-on-empty, and peek-on-empty.

### Repository Analysis (10)

- **REPO-01** — Find every TODO/FIXME in this codebase with file and line number.
- **REPO-02** — Extract all API endpoints and their HTTP methods from this FastAPI project.
- **REPO-03** — List every environment variable referenced anywhere in this repository.
- **REPO-04** — Find all Kafka topic names used across this repository.
- **REPO-05** — Review this entire codebase for obvious security issues.
- **REPO-06** — Identify every place that reads an environment variable without a default value.
- **REPO-07** — Find all functions in this repository longer than 100 lines, by file.
- **REPO-08** — Reconcile `requirements.txt` against what's actually imported anywhere in the codebase.
- **REPO-09** — Find every place this project catches a bare `except:` with no specific exception type.
- **REPO-10** — Identify which modules in this repository have no corresponding test file.

### General Coding / Reasoning (10)

- **GEN-REASON-01** — What's the difference between a Kubernetes Deployment and a StatefulSet?
- **GEN-REASON-02** — Compare vLLM vs. TGI vs. Ollama for self-hosting a 13B model on one GPU box.
- **GEN-REASON-03** — What is the purpose of a nodeSelector vs. a taint/toleration in Kubernetes?
- **GEN-REASON-04** — Explain the tradeoff between polling a database vs. change streams for a live dashboard.
- **GEN-REASON-05** — When would you choose AWQ over GGUF quantization for a self-hosted model?
- **GEN-REASON-06** — Explain what a Kafka consumer group is and why a service would use one.
- **GEN-REASON-07** — Difference between at-least-once and exactly-once delivery, and why it matters for payments.
- **GEN-REASON-08** — Compare a rules engine vs. a small language model for structured decisions on latency/determinism.
- **GEN-REASON-09** — Explain why adding more retrieved chunks to a RAG pipeline can make answers worse.
- **GEN-REASON-10** — Give a decision framework for one large model vs. a router with several small specialist models.

## Running it

Needs a reachable provider — run inside `ak-chatbot` (has the real
network path to `ak-ollama` and every hosted provider's API keys):

```bash
docker exec ak-chatbot python -m scripts.run_slm_test_suite \
    --provider ollama-coder --domain coding --out /srv/data/slm_test_report.json
```

- `--provider` — any provider key from `llm/router.py` (default
  `ollama-coder`). Point it at a hosted model instead to get a
  comparison baseline.
- `--timeout` — per-prompt timeout in seconds (default 90 — several of
  these prompts genuinely ask for long, detailed answers; see
  `README_COMPLEX_EXAMPLE_TRAINING.md`'s own note on this).
- `--out` — where to write the JSON report.

Calls `llm.router.call_provider_directly()` — the SAME "bypass every
tier/chain/bandit, get the NAMED provider's own real answer or a clear
error" mechanism `scripts/log_complex_ground_truth.py` uses, for the
identical reason: a silent substitute from `TierClient`'s own fallback
chain would make this "the SLM's own test report" in name only.

## How pass/fail is decided

A prompt **passes** when: the call didn't error, the response is
non-empty, and (if it contains a fenced code block) that code parses as
valid Python. This is a real, reproducible, mechanical check — not a
judgment call — and it is **not** a claim that the code is *correct*,
only that it *exists and parses*. Reading `response_preview` (the first
400 characters of each answer, in the JSON report) is still how you'd
judge whether an answer is actually good.

## What the report contains

```json
{
  "provider": "ollama-coder",
  "summary": {
    "total_prompts": 105,
    "passed": 91,
    "pass_rate": 0.867,
    "avg_latency_ms": 4820.3,
    "p95_latency_ms": 21400.0,
    "intent_reasoner_accuracy": 0.94
  },
  "by_category": { "generate": {"total": 30, "passed": 28, "avg_latency_ms": 3900.1}, ... },
  "results": [ {"id": "GEN-01", "category": "generate", "passed": true,
                "latency_ms": 210.4, "prompt_tokens": 42, "completion_tokens": 88,
                "response_preview": "...", "quality_notes": [] }, ... ]
}
```

`intent_reasoner_accuracy` is a live check that
`llm/intent_reasoner.py`'s classifier agrees with this suite's own
human-assigned category for each prompt (excluding `general_coding`,
which is the deliberate fallback bucket, not something the classifier
is expected to "detect").

## Honest limits

- The quality check is deterministic and cheap, not a judge model — it
  catches "didn't answer" and "the code doesn't parse," not "the code
  is wrong in a way that still parses." Real quality assessment still
  needs a human (or a real LLM-judge pass, not built here) reading
  `response_preview`/the full response.
- Results depend on server load, network conditions, and (for a local
  SLM) whether the model was already warm (see `providers.py`'s
  `OLLAMA_KEEP_ALIVE`) — a report run cold will show materially worse
  `avg_latency_ms` than one run warm. Compare reports run under the
  same warmth conditions, not a cold run against a warm one.
- This tests ONE provider's raw answers via `call_provider_directly` —
  it does not exercise CodingAgent's tool-calling loop (`read_file`/
  `write_file`/`run_code`), Auto Hybrid's routing decision, or the
  approval-bar UI. It answers "can this model produce a good ANSWER to
  these prompts," not "does the whole agent pipeline work end to end"
  — that's what a live, manual demo run (or a browser/VS-Code-driven
  integration test) is for.
