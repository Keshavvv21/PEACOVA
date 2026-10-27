"""Domain mode — an RL-selected choice among MULTIPLE locally-registered
Algo-Akanade-SLM variants (see README_SLM_GENERATION_LAYER.md), each
fine-tuned for a different domain via the SAME nightly pipeline, just
pointed at different ground-truth/training data (a legal-support SLM, a
customer-support SLM, a coding SLM, ...). Sits alongside
llm/hybrid_router.py as a second UI-selectable top-level mode
(ChatMessageIn.provider == "domain") — same "decide a preferred
provider for THIS message, hand off to TierClient.preferred, degrade
through the normal fallback chain on a bad pick" shape hybrid mode
already established, just choosing among N local models by DOMAIN
instead of choosing between local and hosted by COMPLEXITY.

Domains are discovered dynamically, by naming convention, from whatever
is currently registered via POST /api/admin/providers (see
scripts/convert_and_register_ollama_model.sh's DOMAIN env var): any
enabled provider named "algo-akanade-slm-<domain>" is a candidate arm
named "<domain>". No code change is needed to add a new domain — train
it, register it under that name, and it becomes selectable on the very
next chat turn, same "live, no restart" story as every other
admin-managed provider.

Why this ISN'T hybrid_router.py's cheap word/clause-count feature
vector reused as the whole signal: local-vs-hosted genuinely correlates
with message complexity, but "which DOMAIN is this about" is a
question about CONTENT, which a word count can't answer. Domain
relevance instead comes from `framework/task_memory.py`'s embedding
similarity — the same "have I seen a request like THIS one before"
memory router.py's own agent-selection bandit already uses for
warm-starting a brand-new agent. The difference here: TaskMemory's
bonus is folded into the score on EVERY call, not just used once to
seed a brand-new arm's posterior via LinearBanditCore.ensure() (the
role it plays for agent routing, where TaskContext's structured
task_type already gives each agent real per-arm signal in the shared
context vector). A domain SLM has no such structured signal available
— the accumulated embedding memory of which past queries actually
worked well on which domain arm IS the signal, and it needs to inform
every decision, not just the first one.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from ..db import store as db_store
from ..framework import bandit_persistence
from ..framework.bandit import LinearBanditCore
from ..framework.task_memory import TaskMemory
from .router import available_providers

_FEATURE_DIM = 3  # same cheap shape as hybrid_router.py: [word_count/50, clause_count/4, tier_is_delay_ok]

DOMAIN_PROVIDER_PREFIX = "algo-akanade-slm-"
GENERAL_ARM = "general"

_PERSIST_NAME = "domain_router"
_COUNTS_TABLE = "DomainRouterCounts"


def _complexity_features(message: str, tier: str) -> list[float]:
    words = message.split()
    clauses = len(re.split(r"[,;]| and | or ", message))
    return [
        min(len(words) / 50.0, 1.0),
        min(clauses / 4.0, 1.0),
        1.0 if tier == "delay_ok" else 0.0,
    ]


def registered_domains() -> dict[str, str]:
    """{domain: provider_key} for every ENABLED, currently-available
    provider matching the "algo-akanade-slm-<domain>" naming
    convention — read fresh every call, same as available_providers()
    itself, so a newly-registered (or disabled) domain SLM takes effect
    on the very next turn."""
    return {p["id"][len(DOMAIN_PROVIDER_PREFIX):]: p["id"] for p in available_providers()
           if p["id"].startswith(DOMAIN_PROVIDER_PREFIX) and p["available"]}


@dataclass
class DomainSelection:
    arm: str  # a domain name, or GENERAL_ARM
    context: list[float]
    task_text: str


class DomainRouter:
    """One LinearBanditCore + one TaskMemory, arms = GENERAL_ARM plus
    whatever domain SLMs are currently registered — a genuinely DYNAMIC
    arm set (LinearBanditCore.select()/score() take arm_ids fresh every
    call and lazily create a new arm's posterior on first sight, exactly
    like a newly-registered agent gets picked up by router.py's own
    bandit with no code change)."""

    def __init__(self, *, ridge: float = 1.0, exploration_scale: float = 0.4):
        self._core = LinearBanditCore(_FEATURE_DIM, ridge=ridge, exploration_scale=exploration_scale)
        self._memory = TaskMemory()
        self._lock = threading.Lock()
        # Real learning signal has flowed into this bandit since day one
        # (native_agent.py/langgraph_agent.py/adk_agent.py all call
        # record_outcome()) — it just evaporated on every restart until
        # now. Same rag_optimizer.py adoption of bandit_persistence.py.
        bandit_persistence.load(_PERSIST_NAME, self._core)
        self._counts: dict[str, dict[str, int]] = self._load_counts()

    @staticmethod
    def _load_counts() -> dict[str, dict[str, int]]:
        row = db_store(_COUNTS_TABLE).get("DOMAIN_ROUTER", "COUNTS")
        return (row or {}).get("arms", {})

    def _save_counts(self) -> None:
        db_store(_COUNTS_TABLE).put("DOMAIN_ROUTER", "COUNTS", {"arms": self._counts})

    def choose(self, message: str, tier: str) -> DomainSelection:
        domains = registered_domains()
        context = _complexity_features(message, tier)
        if not domains:
            return DomainSelection(arm=GENERAL_ARM, context=context, task_text=message)
        arm_ids = [GENERAL_ARM] + list(domains.keys())
        with self._lock:
            scored = {}
            for arm_id in arm_ids:
                sampled, _ = self._core.score(arm_id, context)
                # Live per-query content signal — see module docstring
                # for why this is added every time, not just used to
                # seed a brand-new arm.
                scored[arm_id] = sampled + self._memory.bonus_for(message, arm_id)
            winner = max(scored, key=scored.get)
        return DomainSelection(arm=winner, context=context, task_text=message)

    def is_confidently_relevant(self, selection: DomainSelection) -> bool:
        """True only when TaskMemory has an actual, content-based (embedding
        cosine similarity, not complexity-bucket) match recording a past WIN
        for THIS domain arm on a message like this one — see module
        docstring's `bonus_for()` gate at `_MATCH_THRESHOLD`. Deliberately
        NOT the same check as `choose()` picking a non-GENERAL_ARM winner:
        `choose()`'s core LinearBanditCore score is fit purely on
        `_complexity_features()` (word/clause count, tier), which has no
        notion of TOPIC at all — a short, simple trading question and a
        short, simple coding question look identical to it, so a domain
        arm can out-score GENERAL_ARM there for a message that domain has
        nothing to do with. Used by the "true auto" callers
        (native_agent.py/langgraph_agent.py/adk_agent.py's `_caller_sent_
        nothing` branch) specifically to require genuine per-message-
        content evidence before silently overriding the normal, safe
        hosted-first chain with an unproven local domain SLM guess — the
        explicit "domain" mode (a caller literally opting into cross-
        domain exploration) is untouched and still uses `choose()`'s raw
        winner as-is."""
        if selection.arm == GENERAL_ARM:
            return False
        return self._memory.bonus_for(selection.task_text, selection.arm) > 0.0

    def preferred_provider_for(self, selection: DomainSelection) -> str | None:
        """None means "don't override" — let the tier's normal chain
        decide (hosted rungs first, then whatever plain "algo-akanade-slm"
        or "ollama" rung is in it), same semantics as
        hybrid_router.preferred_provider_for()'s "hosted" arm."""
        if selection.arm == GENERAL_ARM:
            return None
        return f"{DOMAIN_PROVIDER_PREFIX}{selection.arm}"

    def record_outcome(self, selection: DomainSelection, *, success: bool) -> None:
        reward = 1.0 if success else -0.5
        with self._lock:
            self._core.update(selection.arm, selection.context, reward)
            self._memory.remember(selection.task_text, selection.arm, reward)
            counts = self._counts.setdefault(selection.arm, {"calls": 0, "wins": 0})
            counts["calls"] += 1
            if success:
                counts["wins"] += 1
            self._save_counts()
        bandit_persistence.save(_PERSIST_NAME, self._core)

    def stats(self) -> list[dict]:
        """Current per-domain-arm performance (GENERAL_ARM plus every
        registered domain SLM, live or historical) — the "who's winning
        for what" dashboard's real data source. Same shape discipline as
        rag_optimizer.stats()."""
        with self._lock:
            arm_ids = set(self._counts) | set(self._core._B)
            rows = []
            for arm_id in sorted(arm_ids):
                counts = self._counts.get(arm_id, {"calls": 0, "wins": 0})
                mean = self._core.posterior_mean(arm_id)
                rows.append({
                    "domain": arm_id,
                    "calls": counts["calls"],
                    "wins": counts["wins"],
                    "win_rate": round(counts["wins"] / counts["calls"], 4) if counts["calls"] else None,
                    "posterior_mean_weights": [round(float(x), 4) for x in mean] if mean is not None else None,
                })
            return rows


# Process-wide singleton, same lifecycle as hybrid_router/optimization_router/token_optimizer/rag_optimizer.
domain_router = DomainRouter()
