"""POST /api/peft-training/upload and /api/peft-training/link — the
"PEFT button" upload widget's REST surface, same underlying
agent/peft_training_tool.py the chat-triggered path already uses.
"""
from __future__ import annotations

import io

import httpx

from app.agent.finetune_log import _log_path
from app.framework import peft_job_status
from app.framework.rag import rag_registry
from tests.conftest import auth


def test_upload_a_plain_text_file_logs_a_training_example(client, member_token):
    file_content = b"Momentum trading means buying assets with recent upward price trends."
    r = client.post(
        "/api/peft-training/upload",
        data={"domain": "trading-upload-test", "topic": "Momentum trading"},
        files={"file": ("notes.txt", io.BytesIO(file_content), "text/plain")},
        headers=auth(member_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["domain"] == "trading-upload-test"
    assert body["chars_logged"] > 0

    path = _log_path("trading-upload-test")
    assert "Momentum trading means buying" in path.read_text()


def test_upload_also_indexes_into_a_domain_rag_store(client, member_token):
    file_content = b"RSI measures the speed and magnitude of recent price changes."
    r = client.post(
        "/api/peft-training/upload",
        data={"domain": "rag-upload-test", "topic": "RSI"},
        files={"file": ("notes.txt", io.BytesIO(file_content), "text/plain")},
        headers=auth(member_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rag_chunks_indexed"] >= 1
    store = rag_registry.get("peft-training-rag-upload-test")
    hits = store.search("RSI price changes", k=3)
    assert any("RSI measures" in h.text for h in hits)


def test_upload_tells_the_user_when_a_job_is_already_running(client, member_token):
    job_id = peft_job_status.start_job(domain="running-domain-test", mode="cpu", slot=0)
    try:
        r = client.post(
            "/api/peft-training/upload",
            data={"domain": "running-domain-test", "topic": "test"},
            files={"file": ("notes.txt", io.BytesIO(b"some real content here"), "text/plain")},
            headers=auth(member_token),
        )
        assert r.status_code == 200, r.text
        note = r.json()["job_status_note"]
        assert note is not None and "already running" in note
    finally:
        peft_job_status.finish_job(job_id, ok=True, summary="cleanup")


def test_upload_has_no_job_status_note_when_nothing_is_running(client, member_token):
    r = client.post(
        "/api/peft-training/upload",
        data={"domain": "idle-domain-test", "topic": "test"},
        files={"file": ("notes.txt", io.BytesIO(b"some real content here"), "text/plain")},
        headers=auth(member_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["job_status_note"] is None


def test_upload_rejects_an_empty_file(client, member_token):
    r = client.post(
        "/api/peft-training/upload",
        data={"domain": "trading-upload-test-2", "topic": "empty"},
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=auth(member_token),
    )
    assert r.status_code == 400


def test_upload_requires_auth(client):
    r = client.post(
        "/api/peft-training/upload",
        data={"domain": "x", "topic": "y"},
        files={"file": ("notes.txt", io.BytesIO(b"some text"), "text/plain")},
    )
    assert r.status_code in (401, 403)


def test_submit_training_link_fetches_and_logs(client, member_token, monkeypatch):
    class _FakeResponse:
        text = "<html><body><p>Backtesting means testing a strategy on historical data.</p></body></html>"
        status_code = 200

        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResponse())
    r = client.post(
        "/api/peft-training/link",
        json={"domain": "trading-upload-test-3", "topic": "Backtesting",
             "url": "https://example.com/backtesting"},
        headers=auth(member_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "https://example.com/backtesting"
    path = _log_path("trading-upload-test-3")
    assert "Backtesting means testing a strategy" in path.read_text()


def test_submit_training_link_also_indexes_into_rag_and_fetches_only_once(client, member_token, monkeypatch):
    fetch_count = {"n": 0}

    class _FakeResponse:
        text = "<html><body><p>Moving averages smooth out price data over time.</p></body></html>"
        status_code = 200

        def raise_for_status(self):
            pass

    def _fake_get(url, **kw):
        fetch_count["n"] += 1
        return _FakeResponse()

    monkeypatch.setattr(httpx, "get", _fake_get)
    r = client.post(
        "/api/peft-training/link",
        json={"domain": "rag-link-test", "topic": "Moving averages",
             "url": "https://example.com/ma"},
        headers=auth(member_token),
    )
    assert r.status_code == 200, r.text
    assert fetch_count["n"] == 1  # NOT fetched twice for RAG + training log
    body = r.json()
    assert body["rag_chunks_indexed"] >= 1
    assert body["source"] == "https://example.com/ma"
    store = rag_registry.get("peft-training-rag-link-test")
    hits = store.search("moving averages price data", k=3)
    assert any("Moving averages smooth" in h.text for h in hits)


def test_submit_training_link_requires_auth(client):
    r = client.post("/api/peft-training/link",
                    json={"domain": "x", "topic": "y", "url": "https://example.com"})
    assert r.status_code in (401, 403)
