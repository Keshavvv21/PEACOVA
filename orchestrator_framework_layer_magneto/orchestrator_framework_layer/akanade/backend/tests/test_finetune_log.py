"""Tests for the nightly-fine-tuning data collector (app.agent.finetune_log)."""
import json

import pytest

from app.agent import finetune_log


def test_compute_reward_full_marks_for_a_clean_first_try():
    assert finetune_log.compute_reward(retried=False, out_of_scope=False) == 1.0


def test_compute_reward_penalizes_a_retry_and_out_of_scope_independently():
    assert finetune_log.compute_reward(retried=True, out_of_scope=False) == 0.7
    assert finetune_log.compute_reward(retried=False, out_of_scope=True) == 0.5


def test_compute_reward_clamps_to_zero_when_both_penalties_stack():
    # 1.0 - 0.3 - 0.5 = 0.2, not negative, but stacking further than that
    # (e.g. a hypothetical third penalty) must never go below 0.
    assert finetune_log.compute_reward(retried=True, out_of_scope=True) == pytest.approx(0.2)


def test_log_ollama_turn_ignores_every_non_ollama_provider(tmp_path, monkeypatch):
    log_path = tmp_path / "ollama_turns.jsonl"
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(log_path))
    for provider in ("groq-llama-3.3-70b", "rule-based", "gpt-5.6-luna", "", None):
        finetune_log.log_ollama_turn(messages=[{"role": "user", "content": "hi"}],
                                     reply="hello", provider=provider,
                                     retried=False, out_of_scope=False)
    assert not log_path.exists()


def test_log_ollama_turn_appends_a_jsonl_record_for_an_ollama_reply(tmp_path, monkeypatch):
    log_path = tmp_path / "nested" / "ollama_turns.jsonl"
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(log_path))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "list projects"}]

    finetune_log.log_ollama_turn(messages=messages, reply="2 project(s): ...",
                                 provider="ollama-qwen2.5:1.5b", retried=False, out_of_scope=False)
    finetune_log.log_ollama_turn(messages=messages, reply="The capital of France is Paris.",
                                 provider="ollama-qwen2.5:1.5b", retried=True, out_of_scope=True)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["messages"] == messages
    assert first["response"] == "2 project(s): ..."
    assert first["provider"] == "ollama-qwen2.5:1.5b"
    assert first["reward"] == 1.0
    assert second["reward"] == pytest.approx(0.2)  # retried AND out_of_scope
    assert "timestamp" in first and "timestamp" in second


def test_log_hybrid_escalation_turn_writes_a_record_even_for_a_hosted_provider(tmp_path, monkeypatch):
    """Unlike log_ollama_turn, this one is FOR a non-ollama provider — it's
    the training signal for "local couldn't do this, here's what worked."""
    log_path = tmp_path / "ollama_turns.jsonl"
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(log_path))
    messages = [{"role": "user", "content": "a genuinely hard question"}]

    finetune_log.log_hybrid_escalation_turn(
        messages=messages, reply="a real hosted-model answer",
        hosted_provider="groq-llama-3.3-70b", retried=False, out_of_scope=False)

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["provider"] == "groq-llama-3.3-70b"
    assert record["source"] == "hybrid_escalation"
    assert record["response"] == "a real hosted-model answer"
    assert record["reward"] == 1.0


def test_log_hybrid_escalation_turn_ignores_a_blank_provider(tmp_path, monkeypatch):
    log_path = tmp_path / "ollama_turns.jsonl"
    monkeypatch.setenv("FINETUNE_LOG_PATH", str(log_path))
    finetune_log.log_hybrid_escalation_turn(
        messages=[{"role": "user", "content": "hi"}], reply="x",
        hosted_provider="", retried=False, out_of_scope=False)
    assert not log_path.exists()
