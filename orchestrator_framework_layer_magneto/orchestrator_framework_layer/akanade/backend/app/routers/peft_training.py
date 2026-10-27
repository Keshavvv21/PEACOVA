"""The "PEFT" button's REST surface — the same request_peft_training
chat tool (agent/tools.py, agent/peft_training_tool.py), reachable
without going through a chat turn at all, for a real upload widget: a
user picks a domain, uploads a document (or pastes a URL/text). Two
real things happen to it, not one:

1. It's chunked and indexed into that domain's own RAG store
   (framework/rag.py's rag_registry, same InMemoryVectorStoreAdapter/
   chunk_text/RagDocument shape document_pipeline.ingest_path() already
   uses) — queryable immediately, this session, via retrieve_with_
   optimizer()/the "domain" chat mode.
2. It's logged as a real ground-truth training example
   (agent/peft_training_tool.py) for that domain's own nightly
   fine-tune log file — picked up on the NEXT scheduled run, never
   trained synchronously.

Same access level as the chat tool itself (any authenticated user, no
Admin gate) — this is additive convenience data entry, not a
privileged action.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..agent import peft_training_tool
from ..auth import current_user
from ..framework import peft_job_status
from ..framework.document_pipeline import _extract_text
from ..framework.rag import InMemoryVectorStoreAdapter, RagDocument, chunk_text, rag_registry

router = APIRouter(prefix="/api/peft-training", tags=["peft-training"])

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB — a training example, not a bulk archive


def _rag_store_name(domain: str) -> str:
    return f"peft-training-{domain or 'general'}"


def _index_into_rag(domain: str, text: str, *, source: str) -> int:
    """Adds `text` to `domain`'s own RAG store, creating it on first use
    (rag_registry.get() raises KeyError for a name never registered —
    same pattern document_pipeline.ingest_path() already follows).
    Returns the number of chunks indexed."""
    store_name = _rag_store_name(domain)
    try:
        store = rag_registry.get(store_name)
    except KeyError:
        store = InMemoryVectorStoreAdapter(name=store_name)
        rag_registry.register(store_name, store)
    chunks = chunk_text(text, chunk_size=800)
    store.add_documents([RagDocument(text=c, metadata={"source": source, "domain": domain})
                        for c in chunks])
    return len(chunks)


def _running_job_note(domain: str) -> str | None:
    """"If others run, wait and tell" — a training job already running
    for THIS domain doesn't block a new submission (appending to the
    log file is always safe), but the submitter should know their
    example won't be part of tonight's already-in-progress run, only
    the next one."""
    if peft_job_status.is_domain_running(domain):
        return (f"A training job is already running for the '{domain}' domain right now — "
                f"this example will be included in the NEXT run, not the one in progress. "
                f"Check back or retry later if you want to confirm status.")
    return None


class TrainingLinkRequest(BaseModel):
    domain: str
    topic: str
    url: str


@router.post("/link")
def submit_training_link(payload: TrainingLinkRequest, _: dict = Depends(current_user)):
    """The "add a link" half of the upload widget — fetches and scrapes
    the URL ONCE (agent/peft_training_tool.py's request_training() also
    knows how to fetch a url= directly, but that would mean fetching
    the same page twice — once for RAG, once for the training log — so
    this fetches here and passes the result to both as content=)."""
    try:
        text = peft_training_tool.fetch_url_text(payload.url)
        result = peft_training_tool.request_training(
            domain=payload.domain, topic=payload.topic, content=text)
    except peft_training_tool.TrainingRequestError as e:
        raise HTTPException(400, str(e)) from e
    result["source"] = payload.url  # request_training(content=...) would otherwise report "pasted text"
    result["rag_chunks_indexed"] = _index_into_rag(payload.domain, text, source=payload.url)
    result["job_status_note"] = _running_job_note(payload.domain)
    return result


@router.post("/upload")
async def upload_training_document(
    domain: str = Form(...),
    topic: str = Form(...),
    file: UploadFile = File(...),
    _: dict = Depends(current_user),
):
    """Extracts text the same way RAG document ingestion already does
    (document_pipeline._extract_text — .pdf/.csv/plain text) from an
    uploaded file, indexes it into RAG, and logs it as a ground-truth
    training example for `domain`. Never writes the upload anywhere
    permanent — a throwaway temp file just long enough to run the same
    extractor ingest_path() itself uses."""
    raw = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large — {_MAX_UPLOAD_BYTES // (1024*1024)}MB max.")
    if not raw:
        raise HTTPException(400, "Empty file.")

    suffix = Path(file.filename or "").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            text = _extract_text(Path(tmp.name))
        except Exception as e:  # noqa: BLE001 — any parser failure becomes a clean 400, never a 500
            raise HTTPException(400, f"Could not extract text from {file.filename!r}: {e}") from e

    if not text.strip():
        raise HTTPException(400, f"No readable text found in {file.filename!r}.")

    try:
        result = peft_training_tool.request_training(domain=domain, topic=topic, content=text)
    except peft_training_tool.TrainingRequestError as e:
        raise HTTPException(400, str(e)) from e
    result["rag_chunks_indexed"] = _index_into_rag(domain, text, source=file.filename or "upload")
    result["job_status_note"] = _running_job_note(domain)
    return result
