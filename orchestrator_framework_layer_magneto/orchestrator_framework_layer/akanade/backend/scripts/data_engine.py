"""Data Engine — the real curation step between the raw fine-tune log
(agent/finetune_log.py's JSONL) and the training set finetune_local_slm.py
actually trains on. Two real, verifiable transforms, applied in order:

  1. scrub_pii() — redacts credential/PII-shaped substrings from each
     example's text BEFORE it can ever be baked into model weights.
     Training on raw production chat logs without this is a real data-
     exfiltration risk a fine-tuned model can't un-learn after the fact
     — unlike a logging/display-layer redaction, this has to happen
     upstream of the actual gradient step.
  2. deduplicate() — drops near-duplicate examples (cosine similarity
     above `threshold` on a cheap bag-of-words hash) so ten copies of
     the same common question don't dominate a 500-example training set
     over genuinely varied ones.

Deliberately dependency-free (re/hashlib/math only, stdlib) — this
module is imported by scripts/finetune_local_slm.py, which is a
STANDALONE script with its own heavy venv (torch/transformers/peft) and
never imports the main app package (see that file's own module
docstring); pulling in app.retrieval.vectorstore.HashEmbedder here would
break that isolation, so the same hashing idea is reimplemented locally
instead, at ~15 lines, rather than shared.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

# ── PII / credential scrubbing ──────────────────────────────────────────

# Each pattern is deliberately conservative (favors false positives over
# false negatives) — over-redacting a training example costs a little
# signal; under-redacting bakes a real secret into model weights, which
# is not the same class of mistake. Order matters: more specific
# patterns (API keys) before more general ones (bare numbers) so a key
# embedded in a longer string is caught by its own rule first.
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE)),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
]


def scrub_pii(text: str) -> str:
    """Replaces every PII/credential-shaped substring with a
    `[REDACTED_<KIND>]` placeholder. Idempotent and order-independent in
    practice since each pattern's own placeholder never re-matches a
    later pattern in this list."""
    if not text:
        return text
    for kind, pattern in _PII_PATTERNS:
        text = pattern.sub(f"[REDACTED_{kind}]", text)
    return text


def scrub_example(example: dict) -> dict:
    """Scrubs every message's content plus the response — the two real
    places PII could appear in a fine-tune record (agent/finetune_log.py's
    schema). Returns a new dict; never mutates the input."""
    scrubbed = dict(example)
    scrubbed["messages"] = [
        {**m, "content": scrub_pii(m.get("content", ""))} for m in example.get("messages", [])
    ]
    scrubbed["response"] = scrub_pii(example.get("response", ""))
    return scrubbed


# ── near-duplicate detection ─────────────────────────────────────────────

_DIM = 256
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "at", "for", "with", "i", "me", "my", "you", "your",
})


def _hash_embed(text: str) -> list[float]:
    """A tiny, self-contained hashed bag-of-words vector — same idea as
    retrieval/vectorstore.py's HashEmbedder, reimplemented locally (see
    module docstring) rather than imported, to keep this script's venv
    independent of the main app."""
    v = [0.0] * _DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % _DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _example_text(example: dict) -> str:
    user_turns = [m.get("content", "") for m in example.get("messages", []) if m.get("role") == "user"]
    return " ".join(user_turns[-1:]) or example.get("response", "")


def deduplicate(examples: list[dict], *, threshold: float = 0.95) -> list[dict]:
    """Keeps the FIRST occurrence of each near-duplicate cluster — since
    callers pass examples most-recent-first (finetune_local_slm.py's
    load_examples()), that means the most recent copy of a repeated
    question wins, not an arbitrary one. O(n^2) cosine comparisons: fine
    at the ~500-example scale this pipeline caps at; not meant for a
    much larger corpus without an index, same tradeoff
    vector_storage_optimizer.py documents for its own brute-force floor."""
    kept: list[dict] = []
    kept_vectors: list[list[float]] = []
    dropped = 0
    for example in examples:
        vector = _hash_embed(_example_text(example))
        if any(_cosine(vector, kv) >= threshold for kv in kept_vectors):
            dropped += 1
            continue
        kept.append(example)
        kept_vectors.append(vector)
    if dropped:
        print(f"data_engine.deduplicate: dropped {dropped} near-duplicate example(s) "
             f"(threshold={threshold}), kept {len(kept)}")
    return kept


def curate(examples: list[dict], *, dedup_threshold: float = 0.95) -> list[dict]:
    """A stateless, one-shot curation pass: scrub every example, THEN
    deduplicate WITHIN this batch only. Useful on its own (and simple to
    test), but see DataEngineStore below for the real nightly path — it
    dedupes against every example ever kept, not just tonight's batch.
    Scrubbing first means two examples that differ only in a redacted
    email/phone number are still recognized as near-duplicates
    afterward, not kept separately because a raw PII substring happened
    to make them look different."""
    scrubbed = [scrub_example(e) for e in examples]
    return deduplicate(scrubbed, threshold=dedup_threshold)


# ── persistent store — "if it's new, keep it; the corpus only grows" ────

class DataEngineStore:
    """The real Data Engine: a durable, ever-growing curated corpus, not
    a value recomputed from scratch every night. Each `ingest()` call
    scrubs incoming examples, checks them against EVERY example this
    store has ever kept (not just the current batch — the whole point of
    persistence), and appends only the genuinely new ones to disk. A
    prompt logged Monday still blocks its near-duplicate from being
    re-added Thursday; `finetune_local_slm.py` trains on `load_all()`'s
    full accumulated set, not a single night's raw log.

    File format: one JSON object per line (same shape as
    agent/finetune_log.py's own records, post-scrub), so a curated
    example is inspectable with the same tools as the raw log.
    """

    def __init__(self, path: str | Path, *, dedup_threshold: float = 0.95):
        self.path = Path(path)
        self.dedup_threshold = dedup_threshold
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._vectors: list[list[float]] = [
            _hash_embed(_example_text(ex)) for ex in self.load_all()
        ]

    def load_all(self) -> list[dict]:
        """Every example this store has ever kept, oldest first."""
        if not self.path.exists():
            return []
        kept: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    kept.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return kept

    def ingest(self, examples: list[dict]) -> list[dict]:
        """Scrubs each example, keeps only the ones NOT a near-duplicate
        of anything already stored (this call's own earlier examples
        included — two brand-new near-identical prompts in the same
        batch still only add one), appends those to disk, and returns
        exactly what was newly added (empty list if nothing was new)."""
        added: list[dict] = []
        with self.path.open("a", encoding="utf-8") as f:
            for example in examples:
                scrubbed = scrub_example(example)
                vector = _hash_embed(_example_text(scrubbed))
                if any(_cosine(vector, kv) >= self.dedup_threshold for kv in self._vectors):
                    continue
                f.write(json.dumps(scrubbed, ensure_ascii=False) + "\n")
                self._vectors.append(vector)
                added.append(scrubbed)
        if added:
            print(f"data_engine.DataEngineStore: added {len(added)} new example(s) to "
                 f"{self.path} ({len(self._vectors)} total curated)")
        return added
