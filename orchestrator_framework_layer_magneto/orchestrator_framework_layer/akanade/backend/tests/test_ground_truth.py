"""Tests for framework/ground_truth.py — the enterprise pipeline's
"Create ground truth" stage: deliberately curated (prompt, verified-
answer) pairs, optionally grounded in a RAG store, logged for tonight's
fine-tune corpus. LLM calls monkeypatched (TierClient.chat), matching
test_document_pipeline.py's convention — no live network dependency.
"""
from __future__ import annotations

import json

import pytest

from app.framework.ground_truth import create_ground_truth
from app.framework.rag import InMemoryVectorStoreAdapter, RagDocument, rag_registry
from app.llm.providers import LLMResult
from app.llm.router import TierClient


def _fake_chat(monkeypatch, text: str, provider: str = "fake-llm"):
    def fake_chat(self, node, messages, tools=None):
        return LLMResult(text=text, provider=provider)
    monkeypatch.setattr(TierClient, "chat", fake_chat)


def test_ungrounded_prompt_answers_from_general_knowledge(monkeypatch):
    _fake_chat(monkeypatch, "A precise, verified answer.")
    [result] = create_ground_truth(["What is the capital of France?"], log_for_training=False)
    assert result.ok
    assert result.answer == "A precise, verified answer."
    assert result.grounded is False
    assert result.sources_used == []


def test_grounded_prompt_retrieves_real_context_first(monkeypatch):
    store = InMemoryVectorStoreAdapter(name="gt-test-store")
    store.add_documents([RagDocument(text="The API rate limit is 100 requests per minute.",
                                     metadata={"filename": "api.md"})])
    rag_registry.register("gt-test-store", store)

    captured = {}

    def fake_chat(self, node, messages, tools=None):
        captured["messages"] = messages
        return LLMResult(text="The rate limit is 100 req/min.", provider="fake-llm")
    monkeypatch.setattr(TierClient, "chat", fake_chat)

    [result] = create_ground_truth(["What's the API rate limit?"], store_name="gt-test-store",
                                   log_for_training=False)
    assert result.grounded is True
    assert result.sources_used
    assert "100 requests per minute" in captured["messages"][1]["content"]


def test_logs_a_finetune_example_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))
    _fake_chat(monkeypatch, "A verified answer.", provider="groq-llama-3.3-70b")
    create_ground_truth(["A real enterprise question"], log_for_training=True)

    record = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").strip())
    assert record["source"] == "ground_truth"
    assert record["reward"] == 1.0
    assert record["response"] == "A verified answer."
    assert record["provider"] == "groq-llama-3.3-70b"


def test_logs_from_a_hosted_provider_too_not_just_ollama(monkeypatch, tmp_path):
    # Unlike log_ollama_turn, ground truth is deliberately NOT gated to
    # "only if ollama answered" — the whole point is distilling from
    # whichever model produced the best verified answer.
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))
    _fake_chat(monkeypatch, "answer", provider="claude-opus")
    create_ground_truth(["question"], log_for_training=True)
    assert (tmp_path / "log.jsonl").exists()


def test_log_for_training_false_never_writes_a_record(monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))
    _fake_chat(monkeypatch, "answer")
    create_ground_truth(["question"], log_for_training=False)
    assert not (tmp_path / "log.jsonl").exists()


def test_a_failed_llm_call_is_not_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))

    def failing_chat(self, node, messages, tools=None):
        return LLMResult(text="", provider="fake-llm", error="every rung failed")
    monkeypatch.setattr(TierClient, "chat", failing_chat)

    [result] = create_ground_truth(["question"], log_for_training=True)
    assert result.ok is False
    assert not (tmp_path / "log.jsonl").exists()


def test_grounded_context_is_included_as_a_system_message_when_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(tmp_path / "log.jsonl"))
    store = InMemoryVectorStoreAdapter(name="gt-test-store-2")
    store.add_documents([RagDocument(text="Deploys happen every Tuesday at 2pm UTC.",
                                     metadata={"filename": "deploys.md"})])
    rag_registry.register("gt-test-store-2", store)
    _fake_chat(monkeypatch, "Deploys are every Tuesday at 2pm UTC.")

    create_ground_truth(["When do deploys happen?"], store_name="gt-test-store-2",
                        log_for_training=True)
    record = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").strip())
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user"]
    assert "Tuesday" in record["messages"][0]["content"]
    # the ground-truth-GENERATION system prompt itself must never leak
    # into the logged training example — only the retrieved context does
    assert "training data" not in record["messages"][0]["content"]


def test_multiple_prompts_each_get_their_own_result(monkeypatch):
    _fake_chat(monkeypatch, "an answer")
    results = create_ground_truth(["q1", "q2", "q3"], log_for_training=False)
    assert [r.prompt for r in results] == ["q1", "q2", "q3"]
    assert all(r.ok for r in results)


def test_ground_truth_endpoint_requires_admin(client, member_token):
    from tests.conftest import auth
    r = client.post("/api/v1/rag/ground-truth", json={"prompts": ["a question"]},
                    headers=auth(member_token))
    assert r.status_code == 403


def test_ground_truth_endpoint_rejects_empty_prompts(client, admin_token):
    from tests.conftest import auth
    r = client.post("/api/v1/rag/ground-truth", json={"prompts": []}, headers=auth(admin_token))
    assert r.status_code == 400


def test_ground_truth_endpoint_404s_for_an_unregistered_store(client, admin_token):
    from tests.conftest import auth
    r = client.post("/api/v1/rag/ground-truth",
                    json={"prompts": ["q"], "store": "no-such-store"}, headers=auth(admin_token))
    assert r.status_code == 404


def test_ground_truth_endpoint_end_to_end(client, admin_token, monkeypatch):
    from tests.conftest import auth
    _fake_chat(monkeypatch, "A verified answer.")
    r = client.post("/api/v1/rag/ground-truth",
                    json={"prompts": ["What is our refund policy?"], "log_for_training": False},
                    headers=auth(admin_token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["answer"] == "A verified answer."
    assert body["results"][0]["ok"] is True
    assert body["logged_for_training"] == 0
