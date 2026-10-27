"""Durable persistence for LinearBanditCore posteriors — without this,
every RL optimizer's learned state (token_optimizer, rag_optimizer,
security_optimizer, ...) lives ONLY in process memory and resets to a
blank prior on every restart/redeploy, silently discarding everything
it's learned so far. This is the generic save/load pair any of them can
call into — same PK/SK document store (app/db.py's store()) every other
durable piece of state in this codebase already goes through
(OAuthConnections, ChatSessions, Providers), not the Chroma vector
store: a bandit posterior is a small numpy array per arm, key-value
state that gains nothing from similarity search.

Currently wired into app/security_optimizer.py (see its __init__/
record_feedback) as the concrete example — the other bandit-backed
optimizers (llm/token_optimizer.py, framework/rag_optimizer.py,
llm/optimization_router.py, framework/router.py's agent-selection core,
framework/query_refinement.py) can adopt the identical two-call pattern
(`load(name, self._core)` in `__init__`, `save(name, self._core)` after
`update()`) without any change to this module.
"""
from __future__ import annotations

from .bandit import LinearBanditCore
from ..db import store

_TABLE = "BanditState"


def save(name: str, core: LinearBanditCore) -> None:
    """Best-effort — a persistence failure (e.g. a misbehaving DynamoDB
    endpoint) degrades to "this update's learning isn't durable" rather
    than breaking the actual decision/response the bandit is backing,
    same fail-soft rule every other optional layer in this codebase
    follows."""
    try:
        store(_TABLE).put(f"OPTIMIZER#{name}", "STATE", {"arms": core.to_state()})
    except Exception:  # noqa: BLE001
        pass


def load(name: str, core: LinearBanditCore) -> None:
    try:
        row = store(_TABLE).get(f"OPTIMIZER#{name}", "STATE")
    except Exception:  # noqa: BLE001
        return
    if row and row.get("arms"):
        core.load_state(row["arms"])
