"""GENERIC ARCHITECTURE — the CEO memo's own reference design (Default
tier: GLM-5.3-Flash/Qwen3.8-Flash; Escalation tier: GLM-5.3 full/Sol
xHigh/Opus 5 — pared down to exactly the candidates with a REAL,
callable provider on this deployment; the earlier 5-candidate list
included Muse Spark 1.3 xHigh, which has no provider configured at
all, and Gemini 3.8 Flash High, which only loosely maps to the
registered "gemini-flash" — both confirmed live, in the first A/B test,
as the cause of 2 of 15 prompts getting zero answer), implemented with
RouteLLM (github.com/lm-sys/routellm) — a calibrated classifier
predicting whether the cheap tier would perform comparably to the
expensive tier FOR THIS PROMPT, genuinely difficulty-aware rather than
a load-balancer's error/latency-based fallback (LiteLLM's Router, used
earlier, doesn't have this concept at all — see the git history on this
file for that attempt). This is the closer semantic match to the CEO
memo's actual ask: "a calibrated acceptance-probability model."

("Generic" because this arm isn't tied to any one vendor's naming —
it's the CEO's OWN reference design, evaluated on equal footing against
Algo aKANADE's own models, not a competitor's product.)

RouteLLM is natively a WEAK/STRONG two-model router, not an N-model
tier router — `weak_model`/`strong_model` below are just LABELS
("default-tier"/"escalation-tier"), never real API model names; RouteLLM
only ever decides WHICH LABEL wins (.route() below), never makes a real
API call itself. Once RouteLLM has picked a tier, which SPECIFIC model
within that tier gets used is a plain, deterministic round-robin (no
second library needed — a tier's own real difficulty-based decision
already happened; picking among near-equivalent tier-mates doesn't
need one).

PRIMARY, not comparison-only: llm/primary_pipeline.py (akanade/backend —
"akanade" names the infrastructure/codebase only, never a user-facing
product name; the product is Algo aKANADE) now calls this service's own
decision as the FIRST preference for real live chat traffic when the
admin blue/green switch is set to "routellm" (the default) — Algo
aKANADE's own deterministic.py/thompson.py stay fully intact as the
automatic-failover path (see that module's own docstring), not
replaced.

LAZY, NOT eager: RouteLLM pulls in torch/transformers/scikit-learn (see
requirements.txt's own comment) — a completely different weight class
than the rest of this lightweight service, same "heavy ML deps live
apart from the always-running API image" principle
orchestrator_framework_layer/akanade/backend/Dockerfile already follows
for algo-kaizen-finetune. The Controller is built on FIRST USE of
pick()/escalate(), not at import/app-startup time, so Algo aKANADE's
own `/route` hot path (every real chat turn) never pays RouteLLM's
import/checkpoint-download cost — only an actual generic-architecture
call does.
"""
from __future__ import annotations

import itertools
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_TIER_MODELS = ["glm-5.3-flash", "qwen3.8-flash"]
# Exactly the escalation candidates with a real, registered provider on
# this deployment (see llm/primary_pipeline.py's REGISTERED_MODEL_MAP,
# akanade/backend side) — "glm-5.3-full-max"/"gpt-5.6-sol-xhigh"/
# "claude-opus-5-medium" map to the real "glm-5.3-full"/"sol-xhigh"/
# "claude" rungs. Muse Spark 1.3 xHigh and Gemini 3.8 Flash High were
# dropped after the first live A/B test: RouteLLM picked them for 2 of
# 15 prompts, and both got zero real answer (no provider at all, or a
# capability mismatch) — a tier should only ever offer candidates this
# deployment can actually serve.
ESCALATION_TIER_MODELS = ["glm-5.3-full-max", "gpt-5.6-sol-xhigh", "claude-opus-5-medium"]

_WEAK_LABEL = "default-tier"
_STRONG_LABEL = "escalation-tier"

# The predicted-strong-win-rate cutoff above which RouteLLM escalates.
# 0.5 (neutral) until this is calibrated against our OWN workload
# (eval/test_assets/*.yaml — the CEO memo's own "test against our real
# workload profile" ask) via RouteLLM's own calibrate_threshold(); see
# this module's _calibrated_threshold() below for where that plugs in.
_THRESHOLD = float(os.getenv("GENERIC_ARCH_ROUTELLM_THRESHOLD", "0.5"))

_default_cycle = itertools.cycle(DEFAULT_TIER_MODELS)
_escalation_cycle = itertools.cycle(ESCALATION_TIER_MODELS)

_controller = None  # built lazily — see module docstring


# Tried in this order — most-informative-but-fragile classifier first,
# "random" (always constructible, zero external deps) last as the
# guaranteed floor:
#   "mf"         — matrix factorization; needs a genuinely funded OpenAI
#                  key to embed each prompt at CALL time (see
#                  _decide_tier's own comment) — this deployment has
#                  none, so every "mf" call fails today, but the rung
#                  stays first so it activates automatically the moment
#                  a real key is configured, with zero code changes.
#   "bert"       — a fine-tuned DeBERTa/XLM-RoBERTa classifier
#                  (checkpoint "routellm/bert_gpt4_augmented", ~1.1GB,
#                  not gated). Self-contained: no external API call at
#                  inference time, runs on CPU with no code changes —
#                  the one candidate here that actually WORKS on this
#                  deployment's hardware with no key/license/GPU
#                  needed, so in practice this is today's real
#                  difficulty-aware classifier.
#   "sw_ranking" — similarity-weighted ranking over real Arena "battle"
#                  data; same real gap as "mf" (needs a funded OpenAI
#                  key to embed each prompt), PLUS its default dataset
#                  id upstream has since moved/redirected — kept as a
#                  rung anyway since a working key would activate it
#                  the same way "mf" would.
#
# "causal_llm" is deliberately excluded, not just deprioritized:
# RouteLLM's own routers/causal_llm/model.py hardcodes `.to("cuda")`
# with no CPU fallback (unlike matrix_factorization's router, which
# checks device), so it throws immediately at construction on a
# CPU-only box — and its default checkpoint's tokenizer/base model
# (meta-llama/Meta-Llama-3-8B) is gated on Hugging Face, needing a
# separate license approval + token this deployment doesn't have.
# Neither gap is fixable without forking RouteLLM's own source.
_CANDIDATE_ROUTERS = ["mf", "bert", "sw_ranking"]


def _get_controller():
    global _controller
    if _controller is not None:
        return _controller
    # RouteLLM's similarity_weighted/utils.py eagerly instantiates a
    # module-level OpenAI() client at IMPORT time, which raises if
    # OPENAI_API_KEY is unset — a real, observed gap: without this
    # guard, importing routellm here would crash the whole ak-router
    # process at startup (this module is imported by
    # routes_generic.py, which main.py mounts unconditionally), not
    # just this one feature.
    # A placeholder key is enough to construct the client object; see
    # _decide_tier()'s own per-call fallback below for the SEPARATE,
    # real gap this doesn't cover (each of "mf"/"sw_ranking" needs a
    # WORKING key for every actual routing decision, not just at
    # construction).
    os.environ.setdefault("OPENAI_API_KEY", "unconfigured-placeholder")
    from routellm.controller import Controller

    # Each candidate is probed as its OWN single-router Controller
    # first — one router's construction failure (e.g. "sw_ranking"'s
    # dataset load, sensitive to network/dataset-id churn upstream)
    # must never take down a DIFFERENT router (e.g. "bert") that would
    # have worked fine on its own. This is a one-time cost paid once
    # per process (Controller is this module's lazy singleton), not
    # per request — a working router's checkpoint loads twice at
    # startup (once to probe, once in the real combined Controller
    # below), never again after that.
    working_routers: list[str] = []
    for name in _CANDIDATE_ROUTERS:
        try:
            Controller(routers=[name], strong_model=_STRONG_LABEL, weak_model=_WEAK_LABEL)
            working_routers.append(name)
        except Exception as e:
            logger.warning("RouteLLM '%s' router unavailable (%s) — skipping", name, e)
    working_routers.append("random")  # guaranteed floor, always last

    _controller = Controller(
        routers=working_routers, strong_model=_STRONG_LABEL, weak_model=_WEAK_LABEL)
    return _controller


def _decide_tier(prompt: str) -> str:
    controller = _get_controller()
    last_error: Exception | None = None
    for name in _CANDIDATE_ROUTERS:
        if name not in controller.routers:
            continue  # failed to construct at all — see _get_controller
        try:
            # Real, observed gap for "mf"/"sw_ranking" specifically:
            # neither is a self-contained local classifier — both call
            # OpenAI's real embeddings API to featurize `prompt` on
            # EVERY call, not just at Controller construction. Without
            # a genuinely funded OPENAI_API_KEY (this deployment has no
            # such key configured anywhere — every hosted provider here
            # is its own custom OpenAI-COMPATIBLE endpoint, not OpenAI
            # itself), every call to either fails at call time, not
            # construction time — caught here, per call, per router,
            # same "present but gracefully degrades" story as every
            # provider in Algo aKANADE's own llm/router.py, rather than
            # a one-time fallback that can't recover once a key IS
            # configured. "bert" needs no external call, so it's the
            # rung that actually decides most calls today.
            return controller.route(prompt=prompt, router=name, threshold=_THRESHOLD)
        except Exception as e:
            last_error = e
            logger.warning("RouteLLM '%s' call failed (%s) — trying the next router", name, e)
    try:
        return controller.route(prompt=prompt, router="random", threshold=_THRESHOLD)
    except Exception as e:
        # "random" has no external deps and always constructs — reaching
        # here would mean something is wrong with the Controller object
        # itself, not any one router; surface it rather than swallow it.
        raise RuntimeError("RouteLLM's 'random' fallback router failed") from (e or last_error)


def pick(prompt: str = "") -> dict:
    """Returns {"model_name", "tier"}. `prompt` is the actual text
    RouteLLM scores for difficulty — an empty string (the default) is a
    valid input and simply reflects "no real prompt available yet" back
    as whatever the router's prior favors, same as any other cold-start
    case in this codebase."""
    tier_label = _decide_tier(prompt)
    if tier_label == _STRONG_LABEL:
        return {"model_name": next(_escalation_cycle), "tier": "escalation"}
    return {"model_name": next(_default_cycle), "tier": "default"}


def escalate(prompt: str = "") -> dict:
    """Force the escalation tier regardless of RouteLLM's own decision
    — the explicit "the default tier's answer wasn't good enough,
    retry with the strong tier" path, same escalation-on-failure shape
    Algo aKANADE's own TierClient.chat() already uses for its real
    chain."""
    del prompt  # not scored — this is an explicit override, not a decision
    return {"model_name": next(_escalation_cycle), "tier": "escalation"}


def bert_tier_hint(prompt: str) -> str | None:
    """The "bert" router's OWN verdict for this prompt ("default" or
    "escalation"), independent of the mf -> bert -> sw_ranking -> random
    cascade _decide_tier() walks above. Used by ak-router's secondary
    pipeline (see bert_shadow.py) as a pure difficulty-awareness input
    for its own admin-opt-in shadow signal — reuses this module's own
    lazily-loaded Controller rather than loading a second copy of the
    checkpoint into that pipeline. Returns None (never raises) if
    "bert" failed to construct (see _get_controller) or this specific
    call fails for any reason — a shadow signal that occasionally goes
    quiet is fine; one that can crash a caller is not."""
    if not prompt:
        return None
    controller = _get_controller()
    if "bert" not in controller.routers:
        return None
    try:
        label = controller.route(prompt=prompt, router="bert", threshold=_THRESHOLD)
        return "escalation" if label == _STRONG_LABEL else "default"
    except Exception as e:
        logger.warning("bert_tier_hint call failed (%s)", e)
        return None
