"""Tests for app.agent.ground_truth_validator — the daily comparison of
a freshly fine-tuned candidate's answers against Groq-generated
reference answers for that night's own new training prompts."""

import json
from datetime import datetime, timezone

import pytest

from app.agent import ground_truth_validator as gtv
from app.llm.providers import LLMResult


def test_candidate_provider_name_uses_the_domain_naming_convention():
    assert gtv._candidate_provider_name("legal") == "algo-akanade-slm-legal"
    assert gtv._candidate_provider_name(None) == "algo-akanade-slm"


def test_similarity_is_1_for_identical_text_and_0_for_empty():
    assert gtv._similarity("the quick brown fox", "the quick brown fox") == 1.0
    assert gtv._similarity("", "something") == 0.0
    assert gtv._similarity("something", "") == 0.0


def test_similarity_is_lower_for_unrelated_answers():
    close = gtv._similarity("A race condition is unsynchronized concurrent access to shared state.",
                            "A race condition happens from unsynchronized access to shared state.")
    far = gtv._similarity("A race condition is unsynchronized concurrent access to shared state.",
                          "Bananas are a good source of potassium.")
    assert close > far


def _write_log(path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_todays_prompts_reads_only_todays_records_newest_first(tmp_path, monkeypatch):
    log_path = tmp_path / "legal_turns.jsonl"
    monkeypatch.setattr(gtv, "_log_path", lambda domain: log_path)
    today = datetime.now(timezone.utc).date().isoformat()
    _write_log(log_path, [
        {"timestamp": "2020-01-01T00:00:00+00:00",
         "messages": [{"role": "user", "content": "an old prompt from a past day"}]},
        {"timestamp": f"{today}T01:00:00+00:00",
         "messages": [{"role": "user", "content": "first prompt today"}]},
        {"timestamp": f"{today}T02:00:00+00:00",
         "messages": [{"role": "user", "content": "second prompt today"}]},
    ])
    prompts = gtv._todays_prompts("legal", sample_size=8)
    assert prompts == ["second prompt today", "first prompt today"]


def test_todays_prompts_dedupes_and_caps_at_sample_size(tmp_path, monkeypatch):
    log_path = tmp_path / "legal_turns.jsonl"
    monkeypatch.setattr(gtv, "_log_path", lambda domain: log_path)
    today = datetime.now(timezone.utc).date().isoformat()
    _write_log(log_path, [
        {"timestamp": f"{today}T0{i}:00:00+00:00",
         "messages": [{"role": "user", "content": "same prompt" if i < 2 else f"prompt {i}"}]}
        for i in range(5)
    ])
    prompts = gtv._todays_prompts("legal", sample_size=2)
    assert len(prompts) == 2
    assert len(set(prompts)) == 2


def test_todays_prompts_returns_empty_when_no_log_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(gtv, "_log_path", lambda domain: tmp_path / "does_not_exist.jsonl")
    assert gtv._todays_prompts("legal", sample_size=8) == []


def test_run_skips_cleanly_when_no_prompts_logged_today(monkeypatch):
    monkeypatch.setattr(gtv, "_todays_prompts", lambda domain, sample_size: [])
    result = gtv.run(domain="legal")
    assert result["skipped"] == "no new prompts logged today"


def test_run_compares_groq_and_candidate_and_persists(monkeypatch):
    monkeypatch.setattr(gtv, "_todays_prompts",
                        lambda domain, sample_size: ["What is a race condition?"])

    def fake_call(name, messages, tools=None, timeout=None):
        if name == "groq":
            return LLMResult(text="Unsynchronized concurrent access to shared state.", provider="groq")
        return LLMResult(text="Unsynchronized concurrent access to shared state.", provider=name)

    monkeypatch.setattr(gtv, "call_provider_directly", fake_call)
    captured = {}

    def fake_create_run(payload):
        captured["payload"] = payload
        return type("R", (), {"id": "abc123", "avg_match_score": 1.0})()

    monkeypatch.setattr(gtv, "create_run", fake_create_run)

    result = gtv.run(domain="legal")
    assert result["run_id"] == "abc123"
    assert result["n_items"] == 1
    assert captured["payload"].domain == "legal"
    assert captured["payload"].provider == "algo-akanade-slm-legal"
    assert captured["payload"].items[0].match_score == 1.0


def test_run_skips_a_prompt_whose_provider_call_errored(monkeypatch):
    monkeypatch.setattr(gtv, "_todays_prompts", lambda domain, sample_size: ["a prompt"])

    def fake_call(name, messages, tools=None, timeout=None):
        if name == "groq":
            return LLMResult(error="rate limited", provider="groq")
        return LLMResult(text="an answer", provider=name)

    monkeypatch.setattr(gtv, "call_provider_directly", fake_call)
    result = gtv.run(domain="legal")
    assert result["skipped"] == "no successful Groq/candidate comparisons"
