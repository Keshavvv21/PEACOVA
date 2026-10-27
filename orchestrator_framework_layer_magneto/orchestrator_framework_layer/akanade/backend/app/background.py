"""One tiny, shared "fire-and-forget, never block the caller" helper —
the exact shape app/agent/mcp_client.py and app/agent/adk_agent.py
already hand-roll their own `threading.Thread(daemon=True)` for
(bridging an async call onto a sync one), generalized here so the chat
response path can defer real disk/DynamoDB writes that carry no
correctness dependency for the CURRENT request: nothing later in this
same request reads back the value being written (learning-signal
bandit persistence, fine-tune logging, security-event/metrics history
records — see the request path's own comments at each call site for
why each one is safe to defer).

Deliberately NOT deferring anything whose result the request itself, or
an immediately-following request on the same session, needs to see —
session state (`sessions.save()`) stays synchronous for exactly that
reason. This is a genuine "learning happens in the background, the
reply doesn't wait for it" pattern, not a blanket "make everything
async" — see README_SLM_LATENCY_INVESTIGATION.md's own distinction
between the model call's real latency (never worth faking) and
bookkeeping around it (worth not making the user wait for).

A daemon thread, not a thread pool: these are rare-relative-to-request-
volume, sub-second writes (a JSONL append, one DynamoDB/local-file
put), not a hot inner loop — spinning up one thread per call is
simpler and doesn't need a shutdown/drain story that a pool would.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_in_background(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Runs `fn(*args, **kwargs)` on its own daemon thread and returns
    immediately — never raises into the caller, and never delays it.
    Any exception `fn` raises is logged, not propagated: the same
    fail-soft rule every writer this defers (bandit_persistence.save(),
    finetune_log's appends, security_events.log_event()) already follows
    for itself when called synchronously — deferring the call must not
    make a previously-safe failure suddenly visible to the request."""

    def _target() -> None:
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 — a background write's failure must never surface here
            logger.exception("run_in_background: deferred call %r failed", getattr(fn, "__name__", fn))

    threading.Thread(target=_target, daemon=True).start()
