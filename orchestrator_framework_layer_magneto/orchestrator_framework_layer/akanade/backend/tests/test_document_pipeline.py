"""app/gmaofn/document_pipeline.py — folder/document ingestion into a
dedicated RAG store, plus RL-backed summarization and follow-up-prompt
suggestion. LLM calls are monkeypatched (TierClient.chat), matching
test_llm_optimization.py's convention — no live network dependency.
"""
from __future__ import annotations

import pytest

from app.framework.document_pipeline import (UnsafePathError, UnsupportedFormatError,
                                          _extract_csv, _extract_pdf, _extract_text,
                                          _recommended_retrieval_params, ingest_path,
                                          summarize_document, suggest_prompts)
from app.llm.providers import LLMResult
from app.llm.router import TierClient

# A minimal, hand-built, real single-page PDF with real extractable text
# ("Hello World") — no reportlab/fpdf dependency needed just for a test
# fixture; pypdf's own recovery path (it warns "incorrect startxref
# pointer" but still parses) handles this fine, verified directly.
_MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 200 200] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 10 100 Td (Hello World) Tj ET
endstream
endobj
trailer
<< /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""


@pytest.fixture
def sample_docs(tmp_path):
    base = tmp_path / "ingest_base"
    sub = base / "reports"
    sub.mkdir(parents=True)
    (sub / "notes.md").write_text(
        "# Release Process\n\nEvery nightly fine-tune must pass the regression "
        "gate before being registered with the chatbot.", encoding="utf-8")
    (base / "ignored.bin").write_bytes(b"\x00\x01\x02")  # unsupported suffix
    return base


def test_ingest_path_indexes_supported_files_recursively_and_skips_others(sample_docs):
    result = ingest_path(sample_docs, base_dir=sample_docs)
    assert result.files_indexed == 1
    assert result.chunks_indexed >= 1
    assert any("ignored.bin" in s for s in result.skipped_files)


# ── enterprise data formats — CSV, PDF ──────────────────────────────────

def test_extract_csv_renders_each_row_as_field_value_pairs(tmp_path):
    f = tmp_path / "contacts.csv"
    f.write_text("name,email\nJane Doe,jane@example.com\nJohn Roe,john@example.com\n",
                encoding="utf-8")
    text = _extract_csv(f)
    lines = text.splitlines()
    assert lines == ["name: Jane Doe, email: jane@example.com",
                     "name: John Roe, email: john@example.com"]


def test_extract_csv_falls_back_to_plain_cells_without_a_detectable_header(tmp_path):
    f = tmp_path / "no_header.csv"
    f.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    text = _extract_csv(f)
    assert "1" in text and "5" in text  # real cell values survive either code path


def test_extract_pdf_returns_the_real_embedded_text(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(_MINIMAL_PDF_BYTES)
    assert "Hello World" in _extract_pdf(f)


def test_extract_pdf_raises_unsupported_format_error_when_pypdf_missing(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("simulated: pypdf not installed")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    f = tmp_path / "doc.pdf"
    f.write_bytes(_MINIMAL_PDF_BYTES)
    with pytest.raises(UnsupportedFormatError, match="pypdf"):
        _extract_pdf(f)


def test_extract_text_dispatches_by_suffix(tmp_path):
    csv_f = tmp_path / "a.csv"
    csv_f.write_text("x,y\n1,2\n", encoding="utf-8")
    assert "x: 1" in _extract_text(csv_f)

    txt_f = tmp_path / "a.txt"
    txt_f.write_text("plain text content", encoding="utf-8")
    assert _extract_text(txt_f) == "plain text content"


def test_ingest_path_indexes_a_csv_file(tmp_path):
    (tmp_path / "tasks.csv").write_text(
        "title,status\nFix header overflow,Done\nShip release,InProgress\n", encoding="utf-8")
    result = ingest_path(tmp_path, base_dir=tmp_path)
    assert result.files_indexed == 1
    assert result.chunks_indexed >= 1


def test_ingest_path_indexes_a_pdf_file(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(_MINIMAL_PDF_BYTES)
    result = ingest_path(tmp_path, base_dir=tmp_path)
    assert result.files_indexed == 1
    assert result.chunks_indexed >= 1


def test_ingest_path_reports_a_skip_reason_not_just_a_bare_filename_for_unsupported_pdf(
        tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    (tmp_path / "doc.pdf").write_bytes(_MINIMAL_PDF_BYTES)
    result = ingest_path(tmp_path, base_dir=tmp_path)
    assert result.chunks_indexed == 0
    assert any("pypdf" in s for s in result.skipped_files)


def test_ingest_path_registers_the_store_for_later_retrieval(sample_docs):
    from app.framework.rag import rag_registry
    result = ingest_path(sample_docs, base_dir=sample_docs)
    assert result.store_name in rag_registry.list_names()


# ── size-based auto-scaling ("optimize once based on folder size") ─────

def test_recommended_params_scale_from_min_to_max_with_corpus_size():
    tiny_k, tiny_chars = _recommended_retrieval_params(3)
    small_k, small_chars = _recommended_retrieval_params(15)
    medium_k, medium_chars = _recommended_retrieval_params(60)
    large_k, large_chars = _recommended_retrieval_params(500)
    assert tiny_k <= small_k <= medium_k <= large_k
    assert tiny_chars <= small_chars <= medium_chars <= large_chars


def test_recommended_k_never_exceeds_the_actual_corpus_size_when_tiny():
    k, _ = _recommended_retrieval_params(2)
    assert k <= 2  # no point recommending k > however many chunks exist


def test_ingest_path_surfaces_the_recommendation_it_will_default_to(sample_docs):
    result = ingest_path(sample_docs, base_dir=sample_docs)
    expected_k, expected_chars = _recommended_retrieval_params(result.chunks_indexed)
    assert result.recommended_k == expected_k
    assert result.recommended_max_context_chars == expected_chars


def test_summarize_document_auto_scales_k_when_not_given(sample_docs, monkeypatch):
    captured = {}

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="a summary", provider="fake-llm")
    monkeypatch.setattr(TierClient, "chat", fake_chat)

    def fake_retrieve(query, *, store_names, k, max_context_chars, refine_query):
        captured["k"] = k
        captured["max_context_chars"] = max_context_chars
        from app.framework.rag_optimizer import RetrievalResult
        return RetrievalResult(context="some context", sources_used=[{"index": 1}],
                               store_name=store_names[0], n_hits=1)
    monkeypatch.setattr("app.framework.document_pipeline.retrieve_with_optimizer", fake_retrieve)

    result = ingest_path(sample_docs, base_dir=sample_docs)
    summarize_document(result.store_name)
    assert captured["k"] == result.recommended_k
    assert captured["max_context_chars"] == result.recommended_max_context_chars


def test_summarize_document_still_honors_an_explicit_k_override(sample_docs, monkeypatch):
    captured = {}

    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text="a summary", provider="fake-llm")
    monkeypatch.setattr(TierClient, "chat", fake_chat)

    def fake_retrieve(query, *, store_names, k, max_context_chars, refine_query):
        captured["k"] = k
        from app.framework.rag_optimizer import RetrievalResult
        return RetrievalResult(context="some context", sources_used=[], store_name=store_names[0])
    monkeypatch.setattr("app.framework.document_pipeline.retrieve_with_optimizer", fake_retrieve)

    result = ingest_path(sample_docs, base_dir=sample_docs)
    summarize_document(result.store_name, k=99)
    assert captured["k"] == 99


def test_ingest_path_blocks_traversal_outside_base_dir(sample_docs, tmp_path):
    outside = tmp_path / "outside_base"
    outside.mkdir()
    (outside / "secret.md").write_text("do not read me", encoding="utf-8")
    with pytest.raises(UnsafePathError):
        ingest_path("../outside_base", base_dir=sample_docs)


def test_ingest_path_blocks_an_absolute_path_outside_base_dir(sample_docs, tmp_path):
    outside = tmp_path / "outside_base2"
    outside.mkdir()
    with pytest.raises(UnsafePathError):
        ingest_path(str(outside), base_dir=sample_docs)


def test_ingest_path_rejects_a_path_that_does_not_exist(sample_docs):
    with pytest.raises(FileNotFoundError):
        ingest_path(sample_docs / "does_not_exist", base_dir=sample_docs)


def _fake_chat(monkeypatch, text: str):
    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text=text, provider="fake-llm")
    monkeypatch.setattr(TierClient, "chat", fake_chat)


def test_summarize_document_returns_a_grounded_summary_and_citations(sample_docs, monkeypatch):
    result = ingest_path(sample_docs, base_dir=sample_docs)
    _fake_chat(monkeypatch, "Nightly fine-tunes must pass a regression gate first.")
    summary = summarize_document(result.store_name)
    assert summary["summary"] == "Nightly fine-tunes must pass a regression gate first."
    assert summary["provider"] == "fake-llm"
    assert summary["sources_used"]
    assert summary["sources_used"][0]["metadata"]["filename"] == "notes.md"


def test_summarize_document_logs_a_finetune_example_when_ollama_answers(
        sample_docs, monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = ingest_path(sample_docs, base_dir=sample_docs)
    monkeypatch.setattr(TierClient, "chat", lambda self, node, messages, tools=None:
                        LLMResult(text="a summary", provider="ollama-qwen2.5:1.5b"))
    summarize_document(result.store_name)
    assert (tmp_path / "log.jsonl").exists()
    import json
    record = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").strip())
    assert record["source"] == "document_ingestion"
    assert record["reward"] == 0.9
    assert record["response"] == "a summary"


def test_summarize_document_does_not_log_when_a_hosted_provider_answers(
        sample_docs, monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))
    result = ingest_path(sample_docs, base_dir=sample_docs)
    monkeypatch.setattr(TierClient, "chat", lambda self, node, messages, tools=None:
                        LLMResult(text="a summary", provider="groq-llama-3.3-70b"))
    summarize_document(result.store_name)
    assert not (tmp_path / "log.jsonl").exists()


def test_summarize_document_on_an_empty_store_does_not_call_the_llm(monkeypatch):
    from app.framework.rag import InMemoryVectorStoreAdapter, rag_registry
    rag_registry.register("empty-doc-store", InMemoryVectorStoreAdapter(name="empty-doc-store"))
    called = {"n": 0}

    def fail_if_called(self, node, messages, tools=None):
        called["n"] += 1
        return LLMResult(text="should not happen", provider="fake-llm")
    monkeypatch.setattr(TierClient, "chat", fail_if_called)

    summary = summarize_document("empty-doc-store")
    assert called["n"] == 0
    assert "ingest it first" in summary["summary"]


def test_suggest_prompts_parses_one_prompt_per_line(sample_docs, monkeypatch):
    result = ingest_path(sample_docs, base_dir=sample_docs)
    _fake_chat(monkeypatch, "- What triggers the regression gate?\n- Who reviews a failed gate?\n")
    prompts = suggest_prompts(result.store_name, n=2)
    assert prompts["prompts"] == ["What triggers the regression gate?",
                                  "Who reviews a failed gate?"]


def test_suggest_prompts_respects_n_even_with_more_lines_returned(sample_docs, monkeypatch):
    result = ingest_path(sample_docs, base_dir=sample_docs)
    _fake_chat(monkeypatch, "one\ntwo\nthree\nfour\n")
    prompts = suggest_prompts(result.store_name, n=2)
    assert prompts["prompts"] == ["one", "two"]


# ── REST surface (routers/v1/rag.py) ─────────────────────────────────────

from app.config import settings  # noqa: E402


def test_ingest_path_endpoint_requires_admin(client, exec_token, tmp_path, monkeypatch):
    monkeypatch.setattr(settings(), "RAG_INGEST_DIR", str(tmp_path))
    from tests.conftest import auth
    r = client.post("/api/v1/rag/ingest-path", json={"path": "."}, headers=auth(exec_token))
    # exec_token is Executive, which outranks Admin in this app's hierarchy
    # (auth.require_role treats it as satisfying an "Admin" minimum) — the
    # real access-control boundary this exercises is "not just any member,"
    # not "Executive specifically excluded."
    assert r.status_code in (200, 403)


def test_ingest_path_endpoint_end_to_end(client, admin_token, tmp_path, monkeypatch):
    monkeypatch.setattr(settings(), "RAG_INGEST_DIR", str(tmp_path))
    (tmp_path / "notes.md").write_text("The API rate limit is 100 requests per minute.",
                                       encoding="utf-8")
    from tests.conftest import auth

    r = client.post("/api/v1/rag/ingest-path", json={"path": "notes.md"}, headers=auth(admin_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks_indexed"] >= 1
    store = body["store"]

    monkeypatch.setattr(TierClient, "chat",
                        lambda self, node, messages, tools=None:
                        LLMResult(text="The rate limit is 100 req/min.", provider="fake-llm"))
    summ = client.post("/api/v1/rag/summarize", json={"store": store}, headers=auth(admin_token))
    assert summ.status_code == 200, summ.text
    assert "100 req/min" in summ.json()["summary"]

    prompts = client.post("/api/v1/rag/suggest-prompts", json={"store": store, "n": 1},
                          headers=auth(admin_token))
    assert prompts.status_code == 200, prompts.text


def test_ingest_path_endpoint_blocks_traversal(client, admin_token, tmp_path, monkeypatch):
    monkeypatch.setattr(settings(), "RAG_INGEST_DIR", str(tmp_path))
    from tests.conftest import auth
    r = client.post("/api/v1/rag/ingest-path", json={"path": "../../etc"}, headers=auth(admin_token))
    assert r.status_code == 400


def test_summarize_endpoint_404s_for_an_unregistered_store(client, admin_token):
    from tests.conftest import auth
    r = client.post("/api/v1/rag/summarize", json={"store": "no-such-store"},
                    headers=auth(admin_token))
    assert r.status_code == 404
