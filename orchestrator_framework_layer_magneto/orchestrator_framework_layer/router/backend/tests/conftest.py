"""Sets DATA_DIR/STORAGE_BACKEND BEFORE any `app.*` module is imported —
`app.config.Settings`'s fields are evaluated once, at class-definition
(import) time, same pattern Akanade's own app/config.py uses, so a
per-test monkeypatch fixture would be too late to matter. One shared
temp dir for the whole test session is fine here: tests either use
distinct candidate/arm ids (deterministic.order is pure and doesn't
care) or explicitly account for the two-arm bandit's persisted state
being shared across tests in this file (see test_thompson.py).
"""
import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="ak-router-test-")
os.environ["DATA_DIR"] = _TMP
os.environ["STORAGE_BACKEND"] = "local"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_data_dir():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)
