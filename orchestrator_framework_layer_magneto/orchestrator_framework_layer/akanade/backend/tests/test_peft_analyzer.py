"""app/agent/peft_analyzer.py — scrapes admin-configured topics into
real ground-truth training pairs for tonight's fine-tune, with a capped,
audit-only local blob store.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.agent import peft_analyzer
from app.db import store


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=self)

    def json(self):
        return self._payload


@pytest.fixture
def blob_dir(tmp_path):
    return tmp_path / "peft_blobs"


def _fake_get_with_extract(extract: str):
    def _get(url, **kwargs):
        return _FakeResponse({"title": "Whatever", "extract": extract})
    return _get


def test_scrape_topic_writes_a_blob_and_logs_ground_truth(monkeypatch, blob_dir):
    monkeypatch.setattr(httpx, "get", _fake_get_with_extract("ALGO AKANADE is Algoanalytics' agent platform."))
    logged = []
    monkeypatch.setattr(peft_analyzer, "log_ground_truth_turn",
                        lambda **kw: logged.append(kw))

    status = peft_analyzer.scrape_topic("ALGO AKANADE", blob_dir=blob_dir, max_bytes=10_000, day="2026-08-30")
    assert status == "scraped"
    assert len(list(blob_dir.glob("*.json"))) == 1
    assert len(logged) == 1
    assert logged[0]["messages"] == [{"role": "user", "content": "What is ALGO AKANADE?"}]
    assert "Algoanalytics" in logged[0]["reply"]
    assert logged[0]["provider"] == "peft-analyzer"


def test_scrape_topic_is_idempotent_per_day(monkeypatch, blob_dir):
    monkeypatch.setattr(httpx, "get", _fake_get_with_extract("Some summary."))
    logged = []
    monkeypatch.setattr(peft_analyzer, "log_ground_truth_turn",
                        lambda **kw: logged.append(kw))

    first = peft_analyzer.scrape_topic("Topic A", blob_dir=blob_dir, max_bytes=10_000, day="2026-08-30")
    second = peft_analyzer.scrape_topic("Topic A", blob_dir=blob_dir, max_bytes=10_000, day="2026-08-30")
    assert first == "scraped"
    assert second == "skipped_already_done"
    assert len(logged) == 1  # not logged twice


def test_scrape_topic_handles_no_extract_gracefully(monkeypatch, blob_dir):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResponse({"title": "X"}))  # no "extract"
    status = peft_analyzer.scrape_topic("Empty Topic", blob_dir=blob_dir, max_bytes=10_000, day="2026-08-30")
    assert status == "skipped_no_data"
    assert list(blob_dir.glob("*.json")) == []


def test_scrape_topic_handles_network_failure_gracefully(monkeypatch, blob_dir):
    def _raise(*a, **k):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(httpx, "get", _raise)
    status = peft_analyzer.scrape_topic("Unreachable", blob_dir=blob_dir, max_bytes=10_000, day="2026-08-30")
    assert status == "skipped_no_data"


def test_cap_eviction_removes_oldest_blob_first(monkeypatch, blob_dir):
    blob_dir.mkdir(parents=True)
    # Two pre-existing blobs, oldest first by mtime.
    old = blob_dir / "2026-08-28__old.json"
    newer = blob_dir / "2026-08-29__newer.json"
    old.write_text(json.dumps({"extract": "x" * 50}))
    newer.write_text(json.dumps({"extract": "y" * 50}))
    import os
    import time
    os.utime(old, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time() - 50, time.time() - 50))

    monkeypatch.setattr(httpx, "get", _fake_get_with_extract("z" * 50))
    monkeypatch.setattr(peft_analyzer, "log_ground_truth_turn", lambda **kw: None)

    # A tiny cap that only leaves room for ONE existing blob plus the new one.
    small_cap = old.stat().st_size + newer.stat().st_size + 40
    peft_analyzer.scrape_topic("New Topic", blob_dir=blob_dir, max_bytes=small_cap, day="2026-08-30")

    remaining = {f.name for f in blob_dir.glob("*.json")}
    assert "2026-08-28__old.json" not in remaining  # evicted — it was oldest
    assert any("New_Topic" in name for name in remaining)  # the new one made it in


def test_enabled_scrape_topics_reads_only_enabled_configs_deduplicated():
    # Unique domain keys AND unique topic strings — this table is shared
    # across the whole test session (see conftest.py's one DATA_DIR),
    # so asserting exact totals/set-equality here would be flaky against
    # whatever other tests happen to run in the same session. Membership
    # checks on distinctly-named topics are what test_admin_finetune_
    # config.py's own shared-table tests already do for the same reason.
    store("FinetuneConfigs").put("FINETUNECFG#peft-test-a", "META", {
        "id": "peft-test-a", "enabled": True, "scrapeTopics": ["Peft Analyzer Topic Alpha", "Peft Analyzer Topic Beta"],
    })
    store("FinetuneConfigs").put("FINETUNECFG#peft-test-b", "META", {
        "id": "peft-test-b", "enabled": False, "scrapeTopics": ["Peft Analyzer Topic Gamma"],
    })
    store("FinetuneConfigs").put("FINETUNECFG#peft-test-c", "META", {
        "id": "peft-test-c", "enabled": True, "scrapeTopics": ["Peft Analyzer Topic Alpha", "Peft Analyzer Topic Delta"],
    })
    topics = peft_analyzer.enabled_scrape_topics()
    assert "Peft Analyzer Topic Alpha" in topics
    assert "Peft Analyzer Topic Beta" in topics
    assert "Peft Analyzer Topic Delta" in topics
    assert "Peft Analyzer Topic Gamma" not in topics  # excluded — its config is disabled
    assert topics.count("Peft Analyzer Topic Alpha") == 1  # deduplicated, not counted twice


def test_run_returns_a_summary_shape(monkeypatch, blob_dir):
    store("FinetuneConfigs").put("FINETUNECFG#peft-test-run", "META", {
        "id": "peft-test-run", "enabled": True,
        "scrapeTopics": ["Peft Analyzer Run Topic One", "Peft Analyzer Run Topic Two"],
    })
    monkeypatch.setattr(httpx, "get", _fake_get_with_extract("A real summary."))
    monkeypatch.setattr(peft_analyzer, "log_ground_truth_turn", lambda **kw: None)

    summary = peft_analyzer.run(blob_dir=blob_dir, max_bytes=10_000_000)
    # >= , not ==: other configs left in the shared table by other tests
    # (or an earlier run in this same session) may add more topics — the
    # two this test just registered are guaranteed present, not exhaustive.
    assert summary["topics_scraped"] >= 2
    assert summary["new_examples_added"] == summary["topics_scraped"]
    assert summary["blob_bytes_used"] > 0
    assert summary["blob_dir"] == str(blob_dir)
