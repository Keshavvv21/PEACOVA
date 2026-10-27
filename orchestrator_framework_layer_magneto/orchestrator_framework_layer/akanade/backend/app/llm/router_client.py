"""The ONLY thing TierClient.chat() knows about ak-router: two HTTP
calls, both fire-safe — a network problem here must never break a chat
turn. `call_route()` returns None on ANY failure (timeout, connection
refused, disabled, malformed response), and the caller's own fallback is
simply "use the fixed CHAINS order," never a raised exception. See
router_service_config.py for the separate on/off switch checked BEFORE
this module is even asked to make a call.
"""
from __future__ import annotations

import logging
import threading
import time

import httpx

from ..config import settings
from . import router_service_config

logger = logging.getLogger(__name__)

# Circuit breaker for /generic-architecture/route only — a real, observed gap:
# with "routellm" as primary_pipeline.py's DEFAULT, EVERY chat call with no
# explicit preferred provider tries this, and ROUTELLM_TIMEOUT_S (3s,
# sized for RouteLLM's real "mf" -> OpenAI -> fallback round trip) becomes
# a per-call latency tax on the ENTIRE system the moment ak-router is
# unreachable — confirmed live: it multiplied the akanade-backend test
# suite's own runtime past 10 minutes. After a failure, skip trying again
# for this cooldown window and fall straight through to None (Akanade's own
# chain) — same "don't hammer a down dependency" pattern a real load
# balancer's health check follows, not just a nicety.
_CIRCUIT_COOLDOWN_S = 30.0
_circuit_lock = threading.Lock()
_circuit_open_until = 0.0


def _circuit_is_open() -> bool:
    with _circuit_lock:
        return time.monotonic() < _circuit_open_until


def _trip_circuit() -> None:
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_S


def _reset_circuit() -> None:
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = 0.0


def call_route(candidates: list[dict], context: dict) -> dict | None:
    """`candidates`: [{"id", "cost_per_1k_tokens", "p95_latency_ms",
    "capability_tier"}, ...]. `context`: {"tier", "est_tokens",
    "task_min_capability"}. Returns ak-router's raw JSON response
    ({"ordered_ids", "strategy_used", "shadow_pick"}) or None if the
    service is disabled, unreachable, or errored — the caller treats
    None exactly like "no opinion, use the fixed chain order."
    """
    if not router_service_config.is_enabled():
        return None
    try:
        resp = httpx.post(
            f"{settings().ROUTER_SERVICE_URL}/route",
            json={"candidates": candidates, "context": context},
            timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("ak-router /route call failed, falling back to fixed chain order: %s", e)
        return None


def call_generic_architecture_route(prompt: str) -> dict | None:
    """Same fire-safe shape as call_route() above, for the "primary
    pipeline = RouteLLM" setting (primary_pipeline.py) — returns
    ak-router's raw {"model_name", "tier"} decision, or None on ANY
    failure (disabled, unreachable, timeout). Note this does NOT gate on
    router_service_config.is_enabled() — that flag is specifically
    "should TierClient ask ak-router to ORDER the fixed chain," a
    separate concern from "should the primary-pipeline preference call
    RouteLLM," which primary_pipeline.py gates on its own."""
    if _circuit_is_open():
        return None
    try:
        resp = httpx.post(
            f"{settings().ROUTER_SERVICE_URL}/generic-architecture/route",
            json={"prompt": prompt},
            timeout=settings().ROUTELLM_TIMEOUT_S)
        resp.raise_for_status()
        _reset_circuit()
        return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("ak-router /generic-architecture/route call failed — opening circuit for %ss: %s",
                       _CIRCUIT_COOLDOWN_S, e)
        _trip_circuit()
        return None


def report_outcome(*, candidate_id: str, ok: bool, latency_ms: float, cost_usd: float,
                   strategy_used: str, shadow_pick: str | None, tier: str,
                   bert_hint: str | None = None, ucb_pick: str | None = None) -> None:
    """Fire-and-forget — never raises, never blocks the reply on it (the
    plan's own "don't block the reply on it" note). A dropped outcome
    just means one fewer RoutingOutcomes row / one fewer bandit update,
    not a broken turn. `bert_hint` is the admin-opt-in shadow signal's
    own verdict (see ak-router's bert_shadow.py) — None whenever that
    signal is off, exactly like shadow_pick when Thompson Sampling
    hasn't been asked for. `ucb_pick` is the same posterior's UCB-rule
    pick (see ak-router's thompson.py:pick_ucb()), same shadow-only shape."""
    if not router_service_config.is_enabled():
        return
    try:
        httpx.post(
            f"{settings().ROUTER_SERVICE_URL}/outcome",
            json={"candidate_id": candidate_id, "ok": ok, "latency_ms": latency_ms,
                  "cost_usd": cost_usd, "strategy_used": strategy_used,
                  "shadow_pick": shadow_pick, "bert_hint": bert_hint,
                  "ucb_pick": ucb_pick, "tier": tier},
            timeout=settings().ROUTER_SERVICE_TIMEOUT_S)
    except httpx.HTTPError as e:
        logger.warning("ak-router /outcome call failed (non-fatal): %s", e)
