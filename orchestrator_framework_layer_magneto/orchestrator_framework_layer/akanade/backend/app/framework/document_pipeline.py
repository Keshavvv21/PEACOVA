"""Document ingestion + RAG-optimizer-backed summaries/prompts — "given a
folder or doc, run it through the best RAG pipeline, produce a summary
or follow-up prompts." Three real, additive pieces on top of rag.py +
rag_optimizer.py:

1. **ingest_path()** — walks a folder (or a single file), chunks every
   supported text file found, and indexes the chunks into a FRESH,
   dedicated VectorStoreAdapter registered under a name derived from the
   path — not the shared "default"/"pm-index" stores, so summarizing one
   document never gets diluted by whatever else has been indexed, and
   the exact same content immediately becomes queryable by the live PM
   chat too (any perceive_intent-relevant question can retrieve from it,
   same as pm-index/default — see rag_optimizer.py's wiring into
   native_agent.py and its siblings).
2. **summarize_document()** — pulls the most representative chunks for
   the ingested content via rag_optimizer.retrieve_with_optimizer() (RL-
   picked retrieval algorithm, scoped to just this one store so the
   bandit only ever has to learn "which algorithm," not also re-explore
   "which store" the way the general chat flow does), then asks the
   LLM router's "bulk" tier — exactly the batch/background workload that
   tier exists for (see llm/router.py's CHAINS) — for a real summary
   grounded in those chunks.
3. **suggest_prompts()** — same retrieval, a different ask: N concrete
   follow-up questions a user could pose about this content.

Both LLM-calling functions are deliberately NOT on the hot per-message
chat path (unlike token_optimizer/rag_optimizer, which run on every
relevant turn) — this is an explicit, on-demand action (someone points
at a document and asks for a summary), so a real LLM round trip here is
the right tradeoff, not the one those two modules specifically avoid.

Security note: ingest_path() only ever reads files that resolve to
INSIDE base_dir (default settings().RAG_INGEST_DIR) — a REST caller
supplies a relative path, not a trusted server operator, so "any file on
the server" must never be reachable through this.
"""
from __future__ import annotations

import csv
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..agent import finetune_log
from ..config import settings
from ..llm.router import get_llm_client
from .rag import InMemoryVectorStoreAdapter, RagDocument, chunk_text, rag_registry
from .rag_optimizer import RetrievalResult, rag_optimizer, retrieve_with_optimizer

# Enterprise-data formats beyond plain text — each gets a real,
# format-aware extraction below (_extract_text) rather than being read
# as raw bytes/text, so a CSV's rows or a PDF's pages actually produce
# sensible chunk_text() input instead of garbage.
_SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".pdf"}


class UnsupportedFormatError(ValueError):
    """Raised by _extract_text for a real, recognized-but-unavailable
    format (currently just .pdf without pypdf installed) — distinct from
    a plain OSError, so ingest_path() can report a clear, actionable
    skip reason instead of a bare "couldn't read this file"."""


def _extract_csv(path: Path) -> str:
    """One line of readable text per row — 'header: value, header: value,
    ...' — using the first row as field names (csv.DictReader), so a
    spreadsheet-shaped export (a CRM contact list, a task export, ...)
    chunks into something a retrieval query or a fine-tune example can
    actually match against, instead of a wall of raw commas. Falls back
    to plain comma-joined cells if the file has no header row csv.Sniffer
    can find (still real text, just without field names)."""
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False
        if has_header:
            rows = [", ".join(f"{k}: {v}" for k, v in row.items() if v)
                    for row in csv.DictReader(f)]
        else:
            rows = [", ".join(cell for cell in row if cell) for row in csv.reader(f)]
    return "\n".join(rows)


def _extract_pdf(path: Path) -> str:
    """Real per-page text extraction via pypdf — a genuine optional
    dependency (see requirements.txt), not bundled by default since most
    deployments never ingest PDFs. Raises UnsupportedFormatError (never
    a bare ImportError) if it isn't installed, so the caller's skip
    reason is actionable ('pip install pypdf') instead of a stack trace."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise UnsupportedFormatError(
            "pypdf not installed — run `pip install pypdf` to ingest PDF files") from e
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


def _extract_text(path: Path) -> str:
    """The one dispatcher ingest_path() calls — every supported suffix's
    real extraction logic lives here, so adding the NEXT enterprise
    format (e.g. .docx) is one more branch, not a change anywhere else
    in the ingestion/chunking/indexing path below."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _extract_csv(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


class UnsafePathError(ValueError):
    """Raised when a requested path resolves outside the allowed ingest
    base directory — see this module's security note."""


def resolve_within(path: str | Path, base_dir: str | Path) -> Path:
    """Public — reused by the security agents (agents/code_security_agent.py,
    supply_chain_agent.py, cloud_config_agent.py) for the exact same
    "a REST caller supplies a relative path, not a trusted server
    operator" guard this module's own ingest_path() needs."""
    base = Path(base_dir).resolve()
    resolved = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise UnsafePathError(f"{path!r} resolves outside the allowed directory {base}") from None
    return resolved


_resolve_within = resolve_within  # back-compat alias for this module's own call site below


def _store_name_for(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "doc"
    return f"doc-{slug}-{uuid.uuid4().hex[:8]}"


def _recommended_retrieval_params(n_chunks: int) -> tuple[int, int]:
    """Retrieval breadth scaled to corpus size, min to max — a 3-chunk
    note has nothing to gain from k=15 (there's nothing else to
    retrieve), and a 500-chunk corpus is under-served by the same k=6
    that's plenty for a short note. A bigger k is also pointless on its
    own if max_context_chars is too small to let those extra chunks
    actually survive build_context's truncation, so both scale together.
    This is the "optimize once, based on folder size" step — computed
    from the store's actual size, not re-guessed per query."""
    if n_chunks <= 5:
        return min(n_chunks, 3), 2000
    if n_chunks <= 20:
        return 6, 4000
    if n_chunks <= 100:
        return 10, 6000
    return 15, 8000


def _corpus_size(store_name: str) -> int | None:
    """Best-effort chunk count for the named store — only
    InMemoryVectorStoreAdapter (what ingest_path() always creates)
    exposes this directly; returns None for anything else (e.g. the
    Chroma-backed pm-index), so callers fall back to a fixed default
    rather than guessing at an unknown corpus's size."""
    try:
        store = rag_registry.get(store_name)
    except KeyError:
        return None
    docs = getattr(store, "_docs", None)
    return len(docs) if docs is not None else None


@dataclass
class IngestResult:
    store_name: str
    files_indexed: int
    chunks_indexed: int
    skipped_files: list[str] = field(default_factory=list)
    # The k/max_context_chars summarize_document()/suggest_prompts() will
    # use by default for this store, unless a caller overrides them —
    # surfaced here so "what will querying this store actually do" is
    # visible right when you ingest, not a hidden internal choice.
    recommended_k: int = 6
    recommended_max_context_chars: int = 4000


def ingest_path(path: str | Path, *, base_dir: str | Path | None = None,
                chunk_size: int = 800, store_name: str | None = None) -> IngestResult:
    """Chunks + indexes every supported file under `path` (a single file
    or a folder, recursive) into a fresh store, registered into
    rag_registry under `store_name` (auto-derived from the path if not
    given). `path` must resolve inside `base_dir` (default
    settings().RAG_INGEST_DIR) — raises UnsafePathError otherwise."""
    p = _resolve_within(path, base_dir or settings().RAG_INGEST_DIR)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())

    store = InMemoryVectorStoreAdapter(name=store_name or _store_name_for(p))
    documents: list[RagDocument] = []
    skipped: list[str] = []
    for f in files:
        if f.suffix.lower() not in _SUPPORTED_SUFFIXES:
            skipped.append(str(f))
            continue
        try:
            text = _extract_text(f)
        except (OSError, UnsupportedFormatError) as e:
            skipped.append(f"{f}: {e}" if isinstance(e, UnsupportedFormatError) else str(f))
            continue
        if not text.strip():
            skipped.append(f"{f}: extracted no text")
            continue
        for chunk in chunk_text(text, chunk_size=chunk_size):
            documents.append(RagDocument(text=chunk, metadata={
                "source_path": str(f), "filename": f.name,
            }))

    store.add_documents(documents)
    rag_registry.register(store.name, store)
    rec_k, rec_chars = _recommended_retrieval_params(len(documents))
    return IngestResult(store_name=store.name, files_indexed=len(files) - len(skipped),
                        chunks_indexed=len(documents), skipped_files=skipped,
                        recommended_k=rec_k, recommended_max_context_chars=rec_chars)


_SUMMARY_SYSTEM = {"role": "system", "content": (
    "You summarize documents faithfully and concisely, grounded ONLY in the "
    "provided excerpts. Never invent facts not present in the excerpts.")}

_PROMPTS_SYSTEM = {"role": "system", "content": (
    "Given document excerpts, propose concrete, specific follow-up questions "
    "a reader might actually want answered — one per line, no numbering, no "
    "preamble.")}


def _record(result: RetrievalResult, *, success: bool) -> None:
    if result.selection is not None:
        rag_optimizer.record_outcome(result.selection, success=success, n_hits=result.n_hits)


def summarize_document(store_name: str, *, query: str = "summarize the main points",
                       k: int | None = None, max_context_chars: int | None = None) -> dict:
    """Returns {"summary", "sources_used", "provider"} — sources_used is
    the same citation shape rag_optimizer.py returns elsewhere, so a
    caller can render it identically. k/max_context_chars default to
    None — auto-scaled from this store's actual size (see
    _recommended_retrieval_params); pass real values to override."""
    if k is None or max_context_chars is None:
        auto_k, auto_chars = _recommended_retrieval_params(_corpus_size(store_name) or 20)
        k, max_context_chars = k or auto_k, max_context_chars or auto_chars
    result = retrieve_with_optimizer(query, store_names=[store_name], k=k,
                                     max_context_chars=max_context_chars, refine_query=False)
    if not result.context:
        _record(result, success=False)
        return {"summary": "Nothing indexed yet for this document — ingest it first.",
               "sources_used": [], "provider": None}

    client = get_llm_client("bulk")
    messages = [_SUMMARY_SYSTEM,
               {"role": "user", "content": f"Excerpts:\n\n{result.context}\n\n"
                                          f"Write a concise summary of these excerpts."}]
    res = client.chat("summarize_document", messages)
    success = bool(res.text) and not res.error
    _record(result, success=success)
    if success:
        finetune_log.log_document_turn(messages=messages, reply=res.text, provider=res.provider)
    return {"summary": res.text or "Couldn't generate a summary.",
           "sources_used": result.sources_used, "provider": res.provider}


def suggest_prompts(store_name: str, *, query: str = "what is this document about", n: int = 3,
                    k: int | None = None, max_context_chars: int | None = None) -> dict:
    """Returns {"prompts", "sources_used", "provider"} — up to `n` concrete
    follow-up questions grounded in the ingested content. k/max_context_chars
    default to None — auto-scaled from this store's actual size, same as
    summarize_document()."""
    if k is None or max_context_chars is None:
        auto_k, auto_chars = _recommended_retrieval_params(_corpus_size(store_name) or 20)
        k, max_context_chars = k or auto_k, max_context_chars or auto_chars
    result = retrieve_with_optimizer(query, store_names=[store_name], k=k,
                                     max_context_chars=max_context_chars, refine_query=False)
    if not result.context:
        _record(result, success=False)
        return {"prompts": [], "sources_used": [], "provider": None}

    client = get_llm_client("bulk")
    messages = [_PROMPTS_SYSTEM,
               {"role": "user", "content": f"Excerpts:\n\n{result.context}\n\n"
                                          f"Propose {n} follow-up questions."}]
    res = client.chat("suggest_prompts", messages)
    prompts = [line.strip("-•* ").strip() for line in (res.text or "").splitlines() if line.strip()][:n]
    _record(result, success=bool(prompts))
    if prompts:
        finetune_log.log_document_turn(messages=messages, reply=res.text, provider=res.provider)
    return {"prompts": prompts, "sources_used": result.sources_used, "provider": res.provider}
