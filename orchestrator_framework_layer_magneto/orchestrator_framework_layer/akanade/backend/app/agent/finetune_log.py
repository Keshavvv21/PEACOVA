"""Nightly-fine-tuning data collection — logs (prompt, response, reward)
for every turn the local Ollama SLM actually answers, so
scripts/finetune_local_slm.py has real, reward-scored examples to train
on. Deliberately dependency-free (json/pathlib/threading only) so this
lives safely inside the always-running chatbot image; the actual
training script (torch/peft/transformers/datasets) is a separate,
occasionally-run tool with its own requirements file — see
scripts/finetune_requirements.txt — never imported by this module or
anything on the live request path.

The reward heuristic here is intentionally simple and transparent, not
a reuse of gmaofn/reward.py's richer formula (which needs signals — a
Shapley-attributed multi-agent split, real human feedback — that don't
exist on a single PM-chatbot turn today):
  1.0  the turn reached a genuine completed answer (a successfully
       executed tool result, or free text) — the finish()-in-each-track
       call site this is invoked from only ever calls it on that path,
       never for a confirmation prompt, a clarifying question, or a
       guard/execution failure (see native_agent.py's call site)
  -0.3 a retry was needed (perceive_intent's output_guard rejected the
       model's first attempt) — the first answer wasn't usable as-is
  -0.5 out_of_scope (general-knowledge text instead of a grounded PM
       action) — this pipeline exists to make the local model BETTER at
       PM tool use specifically, so these turns are de-weighted, not
       excluded outright (a well-formed refusal is still worth keeping)
clamped to [0, 1]. Day-one heuristic, not a claim of RL-optimality —
swap for a richer signal (e.g. real human ratings once the widget has
one) without changing anything downstream: the training script only
ever reads the "reward" field, never how it was computed.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings

_LOCK = threading.Lock()


def _log_path(domain: str | None = None) -> Path:
    # Under settings().DATA_DIR by default (the same "local persistent
    # storage" root everything else in local mode uses — see config.py),
    # not a separate hardcoded location; FINETUNE_LOG_PATH overrides the
    # GENERAL file explicitly if the training script needs to point
    # somewhere else. `domain`, when given, writes to
    # "<domain>_turns.jsonl" instead — deploy/nightly_finetune.sh's
    # fetch_configs() already prefers exactly this filename over the
    # general one for a domain's own training run (its own comment:
    # "how you seed a domain-specific corpus") — this is what actually
    # populates that file, rather than requiring an operator to place it
    # there by hand.
    if domain:
        default = os.path.join(settings().DATA_DIR, "finetune", f"{domain}_turns.jsonl")
        path = Path(default)
    else:
        default = os.path.join(settings().DATA_DIR, "finetune", "ollama_turns.jsonl")
        path = Path(os.getenv("FINETUNE_LOG_PATH", default))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def compute_reward(*, retried: bool, out_of_scope: bool) -> float:
    reward = 1.0
    if retried:
        reward -= 0.3
    if out_of_scope:
        reward -= 0.5
    return max(0.0, min(1.0, reward))


def _append(record: dict, *, domain: str | None = None) -> None:
    try:
        with _LOCK, _log_path(domain).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_ollama_turn(*, messages: list[dict], reply: str, provider: str,
                    retried: bool, out_of_scope: bool,
                    routing_strategy: str | None = None,
                    primary_pipeline: str | None = None,
                    primary_pipeline_mode: str | None = None) -> None:
    """Fire-and-forget: appends one JSONL record if `provider` is the
    local Ollama rung, silently no-ops for every other provider (Groq,
    Luna, rule-based, ...) since this dataset is specifically for
    fine-tuning THIS model, not distilling every provider's behavior
    into it. Never raises — a logging failure must not break tonight's
    reply over tomorrow's training run.

    `routing_strategy` — which ak-router policy actually picked this
    provider ("deterministic" | "thompson_sampling"), or None on a
    deployment where ROUTER_SERVICE_ENABLED is off and llm/router.py's
    own fixed-CHAINS fallback picked it instead. Purely descriptive:
    never used by scripts/finetune_local_slm.py's own training filter,
    only by the Router Performance dashboard to break results down by
    which policy produced them.

    `primary_pipeline` — "routellm" or "akanade" (llm/primary_pipeline.py's
    blue/green switch) — which pipeline's preference actually produced
    this turn, so nightly PEFT data (and the dashboards reading it) can
    tell a RouteLLM-primary turn's example apart from a Akanade-secondary
    one, per the "secondary pipeline stores responses for PEFT too"
    requirement.

    `primary_pipeline_mode` — the admin's CONFIGURED setting at call time
    ("akanade" | "routellm" | "routellm_only"), distinct from
    `primary_pipeline` above (what actually answered). Lets a dashboard
    compute "routellm_only" mode's real leak rate — how often a turn
    configured for primary-only coverage still needed Akanade's chain."""
    if not provider or not provider.startswith("ollama"):
        return
    _append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "response": reply,
        "provider": provider,
        "reward": compute_reward(retried=retried, out_of_scope=out_of_scope),
        "routingStrategy": routing_strategy,
        "primaryPipeline": primary_pipeline,
        "primaryPipelineMode": primary_pipeline_mode,
    })


def log_ground_truth_turn(*, messages: list[dict], reply: str, provider: str,
                          domain: str | None = None, training_weight: float = 1.0) -> None:
    """The "Create ground truth" stage of the enterprise pipeline
    (framework/ground_truth.py's create_ground_truth()) — a DELIBERATELY
    curated (prompt, verified-answer) pair for a prompt the organization
    actually cares about, not organic usage. Unlike log_ollama_turn(),
    NOT gated on `provider.startswith("ollama")`: the whole point of
    this stage is teaching the local model from a real answer that may
    have come from a STRONGER hosted tier — the same "distill from
    whichever model actually did it well" reasoning
    log_hybrid_escalation_turn() already applies to an escalation.
    Fixed reward of 1.0 — the highest of any logger here — since this
    example was specifically generated and reviewed for training, not
    an incidental organic turn scored by a heuristic.

    `training_weight` is a SEPARATE knob from reward — reward says "was
    this answer correct," training_weight says "how much should this
    ONE example count relative to others of equal reward" (both end up
    multiplied together into the single loss-scale scripts/
    finetune_local_slm.py already applies per example — see its own
    build_dataset() comment). Use > 1.0 to make the model see a genuinely
    hard/complex example's gradient more strongly per epoch than an
    easy one — the concrete lever for "train harder on the complex
    examples specifically" without needing to hand-duplicate rows in the
    JSONL log to get the same effect. Defaults to 1.0 (no emphasis,
    identical to every example logged before this existed)."""
    if not provider:
        return
    _append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "response": reply,
        "provider": provider,
        "source": "ground_truth",
        "reward": 1.0,
        "training_weight": training_weight,
    }, domain=domain)


def log_document_turn(*, messages: list[dict], reply: str, provider: str) -> None:
    """The "Enterprise Data" path into the fine-tune corpus: a real,
    grounded summary/follow-up-prompts answer from
    framework/document_pipeline.py's summarize_document()/
    suggest_prompts() — i.e. a CSV/PDF/txt/md file someone actually
    ingested, not organic chat. Fixed reward of 0.9 (not the full 1.0
    log_ollama_turn() gives a clean chat answer): grounded-in-real-
    excerpts is a strong signal on its own, but there's no retried/
    out_of_scope check here the way a live chat turn gets, so a hair
    below full confidence is the honest number. Same "only if the local
    Ollama rung actually answered" gate as log_ollama_turn() — a
    document summarized by a hosted provider isn't training data for
    THIS model. Never raises, same reason as every other logger here."""
    if not provider or not provider.startswith("ollama"):
        return
    _append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "response": reply,
        "provider": provider,
        "source": "document_ingestion",
        "reward": 0.9,
    })


def log_hybrid_escalation_turn(*, messages: list[dict], reply: str, hosted_provider: str,
                               retried: bool, out_of_scope: bool) -> None:
    """Fire-and-forget: the counterpart to log_ollama_turn() for Hybrid
    mode's OTHER outcome — the local SLM was preferred but didn't answer
    in time (llm/hybrid_router.py's LOCAL_TIMEOUT_SECONDS) or errored, and
    a hosted provider answered instead. That hosted answer is genuinely
    useful training signal: a real, worked example of what the local
    model should have been able to do itself, distilled from a bigger
    model rather than invented. Reuses the same JSONL log/reward shape as
    log_ollama_turn() (scripts/finetune_local_slm.py filters purely on
    the "reward" field, not "provider" — no downstream change needed),
    written even though `hosted_provider` isn't the ollama rung, since
    the whole point of this record is "teach ollama to answer like this."
    Never raises, same reason as log_ollama_turn()."""
    if not hosted_provider:
        return
    _append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "response": reply,
        "provider": hosted_provider,
        "source": "hybrid_escalation",
        "reward": compute_reward(retried=retried, out_of_scope=out_of_scope),
    })
