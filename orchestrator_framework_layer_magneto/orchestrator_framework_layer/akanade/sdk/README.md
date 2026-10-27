# gmaofn-client

A small, standalone Python SDK for calling a **deployed** ALGO AKANADE
chatbot's REST API from anywhere — a different repo, a notebook, a cron
script, another service. It has no dependency on this repo's code, only
on `httpx` — copy the `gmaofn_client/` directory anywhere, or install it
properly:

```bash
pip install -e sdk/          # editable install from this checkout
# or, published on PyPI:
pip install gmaofn-client
# or straight from GitHub, no PyPI needed:
pip install "git+https://github.com/algoanalytics-pvt/Algo-Kaizen.git#subdirectory=sdk"
```

## Usage

```python
from gmaofn_client import AKANADEClient

with AKANADEClient("https://your-deploy-host") as client:
    client.login("aniruddha", "demo123")

    reply = client.send_message("what is overdue in Compass OS")
    print(reply.reply, "—", reply.provider)

    # Hybrid mode — RL decides local SLM vs hosted model, per message
    reply = client.send_message("summarize this quarter's velocity",
                                provider="hybrid")

    # RAG: ingest a server-side document (Admin role required) and
    # ask for a grounded summary
    ingested = client.rag_ingest_path("reports/q3-notes.md")
    summary = client.rag_summarize(ingested["store"])
    print(summary["summary"])
    print("sources:", summary["sources_used"])
```

Every method maps to one real endpoint in
`akanade/backend/app/routers/` — `client.py`'s docstrings name the exact
route each call hits. `AKANADEError` carries the real HTTP status code
and response body for anything that comes back non-2xx.

`ChatReply` also carries the answering provider's real per-turn token
usage (`reply.usage`), whether an earlier provider in the chain hit its
own context/output limit and got escalated past
(`reply.context_limit_switch`), and the query-refinement layer's result
for the turn (`reply.refinement`, `None` when the bandit gate skipped
refinement) — `reply.raw` always has the complete, unparsed response if
a newer server field isn't modeled here yet.

## Status

Published: **[pypi.org/project/gmaofn-client](https://pypi.org/project/gmaofn-client/)**
(package/import name kept stable across the framework's GMAOFN →
ALGO AKANADE rename; the exported class names now match the current name).
6 smoke tests (`test_gmaofn_client.py`, `httpx.MockTransport` — no live
server needed) cover request shape and response parsing; no live-server
round trip has been run yet, so validate against a real deploy before
depending on it for anything production-facing.

## Publishing a new version (maintainers)

PyPI never lets you re-upload the same version number, even after
deleting it — bump `version` in `pyproject.toml` first (e.g.
`0.1.0` → `0.1.1`).

```bash
cp sdk/.pypirc.example sdk/.pypirc   # gitignored — fill in a real token,
                                     # never the one in this example file
cd sdk
rm -rf dist/ build/ *.egg-info/
python -m build
python -m twine upload --config-file .pypirc dist/*
```

Get a token from **pypi.org → Account settings → API tokens**. The
first upload of a brand-new project needs an account-scoped token
(the project doesn't exist yet to scope one to); swap it for a
project-scoped token immediately after, and delete the broad one.
Optional dry run first: `python -m twine upload --config-file .pypirc
--repository testpypi dist/*`, then `pip install --index-url
https://test.pypi.org/simple/ gmaofn-client`.

**Never commit a real token** — `sdk/.pypirc` is gitignored specifically
for this; only `.pypirc.example` (no real credential in it) is tracked.
A leaked publish token lets anyone push a malicious version under this
package's name — treat it like any other production secret.
