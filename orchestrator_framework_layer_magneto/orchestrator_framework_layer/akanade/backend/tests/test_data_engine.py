"""Tests for scripts/data_engine.py — the nightly fine-tune pipeline's
PII-scrub + persistent-dedup curation step. Lives in the main (light)
test suite deliberately: unlike finetune_local_slm.py, this module has
zero heavy deps (re/hashlib/math/json/pathlib only), so it doesn't need
the standalone torch/transformers/peft venv to test for real.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import data_engine  # noqa: E402


def _example(user_text: str, response: str = "ok") -> dict:
    return {"messages": [{"role": "user", "content": user_text}], "response": response,
           "reward": 1.0}


class TestScrubPii:
    def test_redacts_an_email_address(self):
        assert data_engine.scrub_pii("reach me at jane.doe@example.com please") \
            == "reach me at [REDACTED_EMAIL] please"

    def test_redacts_an_ssn(self):
        assert "[REDACTED_SSN]" in data_engine.scrub_pii("my ssn is 123-45-6789")

    def test_redacts_an_api_key_shaped_token(self):
        assert "[REDACTED_API_KEY]" in data_engine.scrub_pii("token: sk-abcdEFGH12345678")

    def test_leaves_ordinary_text_untouched(self):
        text = "list my overdue tasks in Compass OS"
        assert data_engine.scrub_pii(text) == text

    def test_handles_empty_string(self):
        assert data_engine.scrub_pii("") == ""

    def test_scrub_example_redacts_both_messages_and_response(self):
        ex = {"messages": [{"role": "user", "content": "email a@b.com"}],
             "response": "sent to a@b.com"}
        scrubbed = data_engine.scrub_example(ex)
        assert "a@b.com" not in scrubbed["messages"][0]["content"]
        assert "a@b.com" not in scrubbed["response"]
        # never mutates the input
        assert "a@b.com" in ex["messages"][0]["content"]


class TestDeduplicate:
    def test_drops_an_exact_repeat(self):
        examples = [_example("list my projects"), _example("list my projects")]
        kept = data_engine.deduplicate(examples)
        assert len(kept) == 1

    def test_keeps_genuinely_different_examples(self):
        examples = [_example("list my projects"), _example("what tasks are overdue")]
        kept = data_engine.deduplicate(examples)
        assert len(kept) == 2

    def test_keeps_the_first_occurrence_of_a_duplicate_cluster(self):
        first = _example("list my projects", response="first answer")
        second = _example("list my projects", response="second answer")
        kept = data_engine.deduplicate([first, second])
        assert kept == [first]

    def test_empty_input_returns_empty(self):
        assert data_engine.deduplicate([]) == []


class TestCurate:
    def test_scrubs_then_dedupes_in_one_pass(self):
        a = _example("email a@b.com about the project")
        b = _example("email a@b.com about the project")  # same after scrubbing too
        kept = data_engine.curate([a, b])
        assert len(kept) == 1
        assert "a@b.com" not in kept[0]["messages"][0]["content"]


class TestDataEngineStore:
    def test_ingest_persists_new_examples_to_disk(self, tmp_path):
        store = data_engine.DataEngineStore(tmp_path / "curated.jsonl")
        added = store.ingest([_example("list my projects")])
        assert len(added) == 1
        assert (tmp_path / "curated.jsonl").exists()

    def test_ingest_skips_a_duplicate_of_something_already_stored(self, tmp_path):
        path = tmp_path / "curated.jsonl"
        store = data_engine.DataEngineStore(path)
        store.ingest([_example("list my projects")])
        added_again = store.ingest([_example("list my projects")])
        assert added_again == []
        assert len(store.load_all()) == 1

    def test_a_duplicate_within_the_same_ingest_call_is_only_added_once(self, tmp_path):
        store = data_engine.DataEngineStore(tmp_path / "curated.jsonl")
        added = store.ingest([_example("list my projects"), _example("list my projects")])
        assert len(added) == 1

    def test_reloading_the_store_sees_everything_persisted_across_instances(self, tmp_path):
        path = tmp_path / "curated.jsonl"
        data_engine.DataEngineStore(path).ingest([_example("list my projects")])
        reloaded = data_engine.DataEngineStore(path)
        assert len(reloaded.load_all()) == 1
        # and dedup against the RELOADED history, not just this instance's own writes
        added = reloaded.ingest([_example("list my projects")])
        assert added == []

    def test_ingest_scrubs_pii_before_persisting(self, tmp_path):
        store = data_engine.DataEngineStore(tmp_path / "curated.jsonl")
        store.ingest([_example("call me at 555-123-4567")])
        [stored] = store.load_all()
        assert "555-123-4567" not in stored["messages"][0]["content"]

    def test_dedup_threshold_is_configurable(self, tmp_path):
        # These two share some but not all words — bag-of-words cosine
        # similarity lands around 0.58 (verified directly), a real
        # middle value to split a low vs. high threshold on, not 0 or 1.
        first, second = "list my projects please", "list my projects for the client today"

        lenient = data_engine.DataEngineStore(tmp_path / "lenient.jsonl", dedup_threshold=0.5)
        lenient.ingest([_example(first)])
        assert lenient.ingest([_example(second)]) == []  # merged away as a near-duplicate

        strict = data_engine.DataEngineStore(tmp_path / "strict.jsonl", dedup_threshold=0.9)
        strict.ingest([_example(first)])
        assert len(strict.ingest([_example(second)])) == 1  # kept as genuinely distinct
