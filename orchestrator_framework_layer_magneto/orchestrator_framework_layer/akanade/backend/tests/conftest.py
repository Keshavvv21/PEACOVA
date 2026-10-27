import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# isolated stores per test session, BEFORE app imports
_tmp = tempfile.mkdtemp(prefix="algopm-test-")
os.environ["DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["CHROMA_DIR"] = os.path.join(_tmp, "chroma")
os.environ["CHAT_AGENT_FRAMEWORK"] = "native"

# Force every LLM provider rung unavailable, BEFORE config.py's load_dotenv()
# runs (that happens on `from app.main import app` below) — tests must be
# hermetic and free regardless of what real keys a developer's own
# backend/.env happens to hold. python-dotenv only fills in variables NOT
# already in os.environ, so setting these to "" here makes load_dotenv()
# a no-op for all of them; the deterministic rule-based floor is what
# every test in this suite is written against (see README.md: "that's
# what the tests pin"). Real-key testing is a deliberate, separate,
# by-hand step (curl against a running `uvicorn`), never `pytest`.
for _var in ("LUNA_BASE_URL", "LUNA_API_KEY", "MAX_BASE_URL", "MAX_API_KEY",
            "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "GROQ_API_KEY",
            "ALGO_ENGINE_BASE_URL", "OLLAMA_BASE_URL", "_BASE_URL",
            "EXTRA_LLM_PROVIDERS", "REAL_APP_API_BASE_URL", "CLAUDE_API_KEY",
            "GITHUB_OAUTH_CLIENT_ID", "GITHUB_OAUTH_CLIENT_SECRET",
            "LINEAR_OAUTH_CLIENT_ID", "LINEAR_OAUTH_CLIENT_SECRET",
            "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"):
    os.environ[_var] = ""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import seed


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:      # runs startup → seed
        yield c


@pytest.fixture(scope="session")
def exec_token(client):
    r = client.post("/api/auth/login",
                    json={"username": "aniruddha", "password": "demo123"})
    assert r.status_code == 200
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def member_token(client):
    r = client.post("/api/auth/login",
                    json={"username": "priya", "password": "demo123"})
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def admin_token(client):
    # "meera" is seeded with the demo store's "Admin" role, which maps from
    # the real app's Manager2/Senior Manager (see auth._REAL_ROLE_MAP) — NOT
    # this demo's own top rank (that's "Executive"). She isn't added to the
    # seeded Compass OS project, which is exactly what the confidentiality
    # tests in test_auth_and_pm.py rely on.
    r = client.post("/api/auth/login",
                    json={"username": "meera", "password": "demo123"})
    return r.json()["access_token"]


def auth(tok):
    return {"Authorization": f"Bearer {tok}"}
