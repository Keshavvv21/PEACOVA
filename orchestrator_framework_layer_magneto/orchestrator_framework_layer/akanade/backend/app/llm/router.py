"""§3.4 — the model router: get_llm_client(tier) with fallback chains.

The ONLY sanctioned entry point to any LLM call (§16's lint rule). Each
tier resolves to a chain; a call walks the chain until a provider
succeeds, recording {node, tier, provider, fallbackUsed} per call into
the session's modelTierLog (§9) via the returned TierClient.

Chains (per the directive — no 4o-mini, no Gemini, anywhere) — this is
the COLD-START order only, per provider_optimizer.py's seed_utility:
  realtime : Luna    → Groq → Deepseek(shadow) → Ollama                        → rule-based
  delay_ok : Max     → Luna → Groq → Deepseek → GLM → Hy4 → Ollama → Algo Engine → rule-based
  bulk     : Deepseek→ Groq → Ollama            → Algo Engine → rule-based
The rule-based floor means the stack degrades to deterministic parsing
rather than an error when every hosted rung is absent or down — it
always stays LAST no matter what, never reordered.

The order every REAL AVAILABLE provider above it is actually tried in
is learned, not fixed — `provider_optimizer.py`'s Thompson Sampling
bandit reorders the chain before each call, from real success/latency
per tier (§7 of README_RL_ALGORITHMS.md). The lists above are what a
brand-new deployment starts at; a system that's been running for a
while may genuinely try Groq before Luna for `realtime` if that's what
the real data says. An explicit `preferred` (a user's own model pick,
or Hybrid mode's local/hosted choice) is applied AFTER this reorder and
still wins outright, unconditionally.

Two ways to extend this without editing _providers()/CHAINS (see llm/README.md):
  - EXTRA_LLM_PROVIDERS (config-driven registry) — a JSON env var alone
    adds a new rung. Each entry's "type" (default "openai_compatible")
    is looked up in _PROVIDER_FACTORIES below to build the actual
    provider — adding a genuinely new *kind* of provider (not just a new
    OpenAI-compatible host) is one factory-registry entry, not a change
    to the parsing/validation logic in _extra_providers() (Open/Closed:
    open to new provider kinds, closed against editing that function).
  - Ollama, registered below under the "ollama" type as a real LangChain
    provider (providers.LangChainOllama) rather than another
    OpenAICompatible — the "add an SLM" path, distinct from "add another
    hosted API" above, and itself just one more _PROVIDER_FACTORIES entry.

Every provider — the five named tiers, Ollama, and anything
EXTRA_LLM_PROVIDERS constructs — is typed as providers.LLMProvider, a
structural Protocol (see providers.py): TierClient never needs to know
which concrete class it's holding (Liskov substitution), and that
Protocol is deliberately just {name, available(), chat()} — nothing a
caller doesn't use (Interface Segregation).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Callable, Literal

from .. import optimization_events
from ..config import settings
from . import capability_tiers
from . import policy_gate
from . import pricing_table
from . import primary_pipeline
from . import router_client
from .guardrails import REFUSAL_MESSAGE
from .optimization import OptimizationProfile
from .cost_optimizer import cost_optimizer, cost_and_savings_for_call
from .optimization_router import optimization_router
from . import provider_budget
from .provider_optimizer import provider_optimizer
from .providers import (AnthropicNative, CustomProvider, HuggingFaceProvider, LangChainOllama,
                        LLMProvider, LLMResult, OpenAICompatible, RuleBased)
from .token_optimizer import optimize_tools_for_provider, token_optimizer

Tier = Literal["realtime", "delay_ok", "bulk"]

# Maps an EXTRA_LLM_PROVIDERS entry's "type" to the factory that builds it.
# Add a new provider *kind* by adding one line here — nothing else in this
# module changes (Open/Closed). Each factory takes the raw config dict and
# returns something satisfying providers.LLMProvider.
_PROVIDER_FACTORIES: dict[str, Callable[[dict], LLMProvider]] = {
    "openai_compatible": lambda e: OpenAICompatible(
        e["name"], e["base_url"], e.get("api_key", ""), e.get("model", ""),
        requires_key=e.get("requires_key", True), extra_body=e.get("extra_body", {})),
    "ollama": lambda e: LangChainOllama(
        e["name"], e["base_url"], e.get("model", "")),
    # ALGO AKANADE's generic model layer — add ANY Hugging Face model via
    # EXTRA_LLM_PROVIDERS config alone: {"type":"huggingface","name":...,
    # "model":"meta-llama/Llama-3.1-8B-Instruct","api_key":"hf_...",
    # "extra_body":{...}, "optimization":{...}}.
    "huggingface": lambda e: HuggingFaceProvider(
        e["name"], e.get("api_key", ""), e.get("model", ""),
        endpoint_url=e.get("endpoint_url", ""), provider=e.get("provider", "auto"),
        extra_body=e.get("extra_body", {}),
        optimization=OptimizationProfile.from_dict(e.get("optimization", {}))),
    # "any custom/local model" — same OpenAI-compatible contract as
    # "openai_compatible" above, named "custom" per the spec's own
    # ModelProvider.CustomProvider so config/docs can say what it's for.
    # "extra_body"/"optimization" — see optimization.py.
    "custom": lambda e: CustomProvider(
        e["name"], e["base_url"], e.get("api_key", ""), e.get("model", ""),
        requires_key=e.get("requires_key", False), extra_body=e.get("extra_body", {}),
        optimization=OptimizationProfile.from_dict(e.get("optimization", {}))),
}


def _parse_json_dict(raw: str) -> dict:
    try:
        parsed = json.loads(raw) if raw and raw.strip() else {}
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _providers() -> dict[str, LLMProvider]:
    s = settings()
    provs: dict[str, LLMProvider] = {
        "luna": OpenAICompatible("gpt-5.6-luna", s.LUNA_BASE_URL,
                                 s.LUNA_API_KEY, s.LUNA_MODEL),
        "max": OpenAICompatible("gpt-5.6-max", s.MAX_BASE_URL,
                                s.MAX_API_KEY, s.MAX_MODEL),
        # GLM 5.3 Flash / Hy4 (Tencent Hunyuan) — real "delay_ok" escalation
        # rungs (see CHAINS above), not just VS Code model-picker options
        # a user opts into directly anymore: independent benchmarks put
        # GLM-5.3-Flash's capability roughly on par with Groq's model at
        # comparable cost, and the tier needed real escalation depth (a
        # single Groq rate-limit previously took the whole "delay_ok" tier
        # down — see CHAINS's own comment). Still explicitly pinnable
        # (preferred="glm"/"hy4"), unchanged. Present even unconfigured
        # (API key empty) — same graceful "just another unavailable rung
        # until an operator sets the env var" story as every other
        # optional provider here.
        "glm": OpenAICompatible("glm-5.3-flash", s.GLM_BASE_URL,
                                s.GLM_API_KEY, s.GLM_MODEL),
        "hy4": OpenAICompatible("hunyuan-turbos-latest", s.HY4_BASE_URL,
                                s.HY4_API_KEY, s.HY4_MODEL),
        "deepseek": OpenAICompatible("deepseek-v4-flash", s.DEEPSEEK_BASE_URL,
                                     s.DEEPSEEK_API_KEY, s.DEEPSEEK_MODEL),
        "groq": OpenAICompatible("groq-llama-3.3-70b", s.GROQ_BASE_URL,
                                 s.GROQ_API_KEY, s.GROQ_MODEL),
        # Algo Engine (§19) — ALGO AKANADE's own self-hosted model execution
        # layer, OpenAI-compatible via `algo-engine serve`; self-hosted, no
        # auth, so requires_key=False (see providers.py). A real CHAINS rung
        # (delay_ok/bulk), not just a pinnable extra. Model-architecture-
        # agnostic (dense/MoE/hybrid) — its own Go source
        # (performance_optimizer/algo_optimzer_engine) picks its own
        # execution strategy (pure Go/CPU, cgo/CPU, or CUDA/GPU, gated by
        # build tag) per model+hardware, not a single fixed backend. Present
        # even when unconfigured (base_url ""), same as every other optional
        # rung: .available() gates it out, TierClient filters it silently —
        # set ALGO_ENGINE_BASE_URL to a running `algo-engine serve` to
        # actually put it in rotation.
        "algo-engine": OpenAICompatible("algo-engine-local",
                                        s.ALGO_ENGINE_BASE_URL, "", "algo-engine",
                                        requires_key=False),
        # the local SLM tier — real LangChain client, see providers.py.
        # Present even when unconfigured (base_url ""), same as every other
        # rung: .available() gates it out, TierClient filters it silently.
        "ollama": LangChainOllama(f"ollama-{s.OLLAMA_MODEL}",
                                  s.OLLAMA_BASE_URL, s.OLLAMA_MODEL,
                                  keep_alive=s.OLLAMA_KEEP_ALIVE),
        # A second, code-specialized model on the SAME Ollama server (see
        # config.py's OLLAMA_CODER_MODEL) — deliberately absent from every
        # CHAINS entry below; CodingAgent (framework/agents/coding_agent.py)
        # is the only caller that ever names this key via `preferred=`.
        # Present even when OLLAMA_BASE_URL is unset or the model was
        # never pulled — same "just another unavailable rung" story.
        "ollama-coder": LangChainOllama(f"ollama-{s.OLLAMA_CODER_MODEL}",
                                        s.OLLAMA_BASE_URL, s.OLLAMA_CODER_MODEL,
                                        keep_alive=s.OLLAMA_KEEP_ALIVE),
        # The "claude" agent framework's own provider (§ agent/claude_agent.py)
        # — deliberately not in any CHAINS entry below, see config.py's
        # CLAUDE_API_KEY comment. Present here (rather than only inside
        # claude_agent.py) so TierClient(preferred="claude") can find it from
        # ANY track, and so it shows up in available_providers() for the UI.
        "claude": AnthropicNative(s.CLAUDE_MODEL, s.CLAUDE_API_KEY, s.CLAUDE_MODEL),
        # ALGO AKANADE's generic model layer, always present (unconfigured by
        # default) — same "just another unavailable rung until an
        # operator sets the env vars" story as every provider above.
        "huggingface": HuggingFaceProvider(
            s.HUGGINGFACE_MODEL or "huggingface", s.HUGGINGFACE_API_KEY, s.HUGGINGFACE_MODEL,
            endpoint_url=s.HUGGINGFACE_ENDPOINT_URL, provider=s.HUGGINGFACE_INFERENCE_PROVIDER,
            extra_body=_parse_json_dict(s.HUGGINGFACE_EXTRA_BODY),
            optimization=OptimizationProfile.from_dict(_parse_json_dict(s.HUGGINGFACE_OPTIMIZATION))),
        "custom": CustomProvider(
            s.CUSTOM_MODEL_NAME, s.CUSTOM_MODEL_BASE_URL, s.CUSTOM_MODEL_API_KEY,
            s.CUSTOM_MODEL_NAME, requires_key=bool(s.CUSTOM_MODEL_API_KEY),
            extra_body=_parse_json_dict(s.CUSTOM_MODEL_EXTRA_BODY),
            optimization=OptimizationProfile.from_dict(_parse_json_dict(s.CUSTOM_MODEL_OPTIMIZATION))),
        "rule": RuleBased(),
    }
    for _cfg, p in _extra_providers():
        provs[p.name] = p
    return provs


def _db_provider_entries() -> list[dict]:
    """Admin-managed providers (see routers/admin_providers.py) — added,
    edited, and enabled/disabled live via the UI. Read fresh on every call,
    same as EXTRA_LLM_PROVIDERS below, so a toggle in the UI takes effect
    on the very next chat turn — no redeploy, no process restart. A store
    read failure (backend genuinely down) degrades to "no DB providers"
    rather than breaking every chat turn over an admin-only feature."""
    from ..db import store
    try:
        rows = store("Providers").scan("PROVIDER#")
    except Exception:
        return []
    s = settings()
    entries = []
    for r in rows:
        if not r.get("enabled", True):
            continue
        entry = {"name": r["name"], "base_url": r["baseUrl"], "model": r.get("model", ""),
                 "api_key": r.get("apiKey", ""), "tiers": r.get("tiers") or ["realtime", "bulk"],
                 "capability_tier": r.get("capabilityTier", 0.3),
                 "type": "openai_compatible"}
        # Real gap found live: every nightly-registered "algo-akanade-slm[-
        # <domain>]" candidate (scripts/convert_and_register_ollama_model.sh)
        # lands here as a plain OpenAICompatible hitting Ollama's OpenAI-
        # compatible endpoint, which has NO notion of LangChainOllama's
        # `keep_alive` field at all — unlike the hand-written "ollama"/
        # "ollama-coder" rungs below, these candidates were silently relying
        # on Ollama's own un-configured default keep-alive (5 minutes),
        # meaning the very model the nightly fine-tune just spent an hour
        # training could still cold-start-reload on its first real chat
        # call. Ollama's OpenAI-compat shim DOES accept `keep_alive` as a
        # normal top-level request field (verified against its own docs),
        # and OpenAICompatible.chat() already merges `extra_body` verbatim
        # into every request payload — so this is a real fix, not a
        # workaround, and only applied to OUR OWN naming convention
        # (never to an arbitrary admin-added hosted API, which could be a
        # strict server that rejects an unrecognized field).
        if entry["name"] == "algo-akanade-slm" or entry["name"].startswith("algo-akanade-slm-"):
            entry["extra_body"] = {"keep_alive": s.OLLAMA_KEEP_ALIVE}
        entries.append(entry)
    return entries


def _extra_providers() -> list[tuple[dict, LLMProvider]]:
    """Parses EXTRA_LLM_PROVIDERS (a JSON array) plus every enabled
    admin-managed provider (_db_provider_entries above) into (config,
    provider) pairs via _PROVIDER_FACTORIES. Malformed/empty config, or an
    entry naming an unregistered "type", is skipped rather than erroring —
    this is optional, additive configuration, not a required one, so a
    typo shouldn't take the whole bot down."""
    raw = settings().EXTRA_LLM_PROVIDERS
    try:
        entries = json.loads(raw) if raw and raw.strip() else []
    except (json.JSONDecodeError, TypeError):
        entries = []
    entries = list(entries) + _db_provider_entries()
    out = []
    for e in entries:
        if not isinstance(e, dict) or not e.get("name") or not e.get("base_url"):
            continue
        factory = _PROVIDER_FACTORIES.get(e.get("type", "openai_compatible"))
        if factory is None:
            continue
        out.append((e, factory(e)))
    return out


def capability_tiers_for(keys: list[str]) -> dict[str, float]:
    """capability_tier (0..1) for each of `keys` — a DB-backed
    provider's own field (admin-set or nightly-seeded, see
    admin_providers.py/capability_tiers.py's module docstring) when one
    exists, else the hand-designed default for a built-in provider. This
    is the ONE place ak-router's deterministic strategy input gets built
    from — see router_client.py's call site in TierClient.chat()."""
    db_tiers = {cfg["name"]: cfg.get("capability_tier", 0.3) for cfg, _p in _extra_providers()}
    return {k: db_tiers.get(k, capability_tiers.default_tier_for(k)) for k in keys}


CHAINS: dict[Tier, list[str]] = {
    "realtime": ["luna", "groq", "deepseek", "ollama", "rule"],
    # Real, observed gap this closes: on a deployment where only Groq's
    # key is actually set, "delay_ok" had exactly ONE real rung — Max/
    # Luna were configured-but-keyless, Algo Engine has no self-hosted
    # server running, so a single Groq per-minute rate-limit (confirmed
    # live: an 8000-token/minute cap, exhausted by one verbose reply)
    # took the WHOLE tier down with nothing real left to try. GLM/Hy4
    # were previously excluded from every CHAINS entry on purpose (see
    # _providers()'s old comment) — genuine escalation-tier candidates
    # per real independent benchmarks (comparable capability to Groq's
    # model, still cheap), not reordered ahead of Groq. Ollama here is
    # the same generic local SLM "realtime"/"bulk" already fall back to
    # — this tier had none before. Every new entry stays exactly what
    # every other CHAINS rung already is: present-but-filtered-out via
    # .available() until an operator actually sets its API key/base URL
    # — adding the rung costs nothing on a deployment that leaves it
    # unconfigured.
    "delay_ok": ["max", "luna", "groq", "deepseek", "glm", "hy4", "ollama", "algo-engine", "rule"],
    "bulk": ["deepseek", "groq", "ollama", "algo-engine", "rule"],
}

# Real gap found live: a call to a local, CPU-only Ollama-family rung
# reached via the NORMAL chain (i.e. NOT an explicit `preferred=`/
# `preferred_timeout=` pick — see this class's own preferred_timeout
# field, which only ever applied to an EXPLICIT pick) had no hard
# latency ceiling at all. Confirmed live: a single such call took 145
# seconds despite LangChainOllama.timeout defaulting to 30.0 — that
# field is passed straight through to ChatOllama, but doesn't reliably
# bound the actual wall-clock wait under real load. This is a genuine,
# deterministic wall-clock cap (ThreadPoolExecutor.result(timeout=...)),
# not a cooperative one, applied to EVERY call to a provider key
# starting with one of these prefixes — explicit picks included, since
# the underlying enforcement gap is the same either way. A timeout here
# raises the stdlib TimeoutError (a real Exception subclass), so it
# flows through TierClient.chat()'s EXISTING `except Exception` handler
# unchanged — recorded as a real failure, chain moves to the next rung.
_LOCAL_PROVIDER_PREFIXES = ("ollama", "algo-akanade-slm")
_LOCAL_HARD_TIMEOUT_S = 12.0


def _effective_chains() -> dict[Tier, list[str]]:
    """CHAINS (the fixed defaults) plus any EXTRA_LLM_PROVIDERS entries
    spliced into their named tier(s), always just before "rule" so the
    deterministic floor stays the last rung no matter what's configured."""
    chains = {tier: list(rungs) for tier, rungs in CHAINS.items()}
    for cfg, p in _extra_providers():
        for tier in (cfg.get("tiers") or list(chains.keys())):
            if tier in chains and p.name not in chains[tier]:
                chains[tier].insert(chains[tier].index("rule"), p.name)
    return chains


def _last_user_text(messages: list[dict]) -> str:
    """The text primary_pipeline.py's RouteLLM call scores for difficulty
    — the most recent user turn, same "what is this request actually
    asking" scope query_refinement.py's own cache-key logic uses.
    Empty string (a valid, harmless input) if no user message exists."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


@dataclass
class TierClient:
    tier: Tier
    log: list[dict] = field(default_factory=list)
    # Which strategy actually produced THIS call's order — "deterministic"
    # (ak-router's default, or the fixed-CHAINS fallback when ak-router
    # is disabled/unreachable) or "thompson_sampling" (only when an
    # admin has explicitly opted in on ak-router's own side). Read by
    # native_agent.py's finish() to thread into finetune_log's new
    # routing_strategy field, and appended onto every self.log entry
    # below so the Router Performance dashboard can attribute an
    # outcome to the policy that produced it.
    strategy_used: str = "deterministic"
    # What ak-router's shadow Thompson-Sampling policy would have
    # picked this call, even while it isn't live — None when ak-router
    # is disabled/unreachable, or when neither designated arm
    # (groq/algo-akanade-slm) was a candidate this call.
    _shadow_pick: str | None = None
    # ak-router's admin-opt-in BERT difficulty shadow signal's own
    # verdict ("default"/"escalation") for this call's prompt — None
    # whenever that signal is off (the default), ak-router is
    # disabled/unreachable, or bert itself isn't available there. Same
    # "never affects real routing" shape as _shadow_pick above; see
    # ak-router's bert_shadow.py.
    _bert_hint: str | None = None
    # Same shape as _shadow_pick, but from the SAME posterior's UCB rule
    # instead of a Thompson-Sampling draw — see ak-router's
    # thompson.py:pick_ucb(). Comparison-only, never live.
    _ucb_pick: str | None = None
    # "routellm" when primary_pipeline.py's blue/green switch is set to
    # RouteLLM AND its pick actually had a real registered provider to
    # prefer this call; "akanade" otherwise (the switch is set to Akanade,
    # OR RouteLLM's pick failed over — see primary_pipeline.py's own
    # docstring on automatic failover). Threaded into finetune_log so
    # PEFT training data can be attributed to whichever pipeline
    # actually produced it.
    primary_pipeline_used: str = "akanade"
    # The admin's CONFIGURED mode at call time ("akanade" / "routellm" /
    # "routellm_only" — see primary_pipeline.py's module docstring),
    # distinct from primary_pipeline_used (which pipeline actually
    # answered). In "routellm_only" mode, primary_pipeline_used=="akanade"
    # here means an unintended failover "leak," not routine behavior —
    # the whole point of that mode is measuring RouteLLM's own,
    # unassisted coverage. In "routellm" mode the same fallback is
    # expected and not a leak. Threaded into finetune_log alongside
    # primary_pipeline_used so a dashboard can compute a real leak rate.
    primary_pipeline_mode: str = "akanade"
    # In-memory only — deliberately NOT part of `log` (which gets persisted
    # into session state as modelTierLog, JSON-serialized; a Selection
    # dataclass instance isn't). The caller (native_agent.py's finish() and
    # its langgraph/adk mirrors) reads this after a real completed turn is
    # known, to record the token-optimizer's reward with the true
    # retried/out_of_scope signal — see llm/token_optimizer.py.
    tool_selections: list = field(default_factory=list)
    # An operator/user-expressed preference (e.g. the UI's model picker,
    # §"select model in UI") — if named and actually available, tried
    # FIRST, ahead of the tier's normal order; every other rung stays in
    # the chain right behind it, so a bad pick still degrades gracefully
    # instead of hard-failing. None (the default) leaves the chain exactly
    # as before this field existed — no observable change for any caller
    # that doesn't pass it.
    preferred: str | None = None
    # A shorter per-call timeout (seconds) applied ONLY to `preferred`'s
    # own attempt, temporarily overriding that provider's normal default
    # (e.g. LangChainOllama's 30s) — Hybrid mode's escalation policy: a
    # local SLM pick that's genuinely slow this turn should fail fast into
    # a hosted rung rather than make the user wait out the provider's full
    # timeout. None (the default) leaves every provider's own timeout
    # untouched, including for a user's plain "pick Ollama" selection —
    # this only kicks in when a caller explicitly asks for it (Hybrid mode).
    preferred_timeout: float | None = None
    # True once `preferred`'s own attempt has been tried AND failed (any
    # reason — the timeout above or a genuine error) and a later rung in
    # the chain answered instead. Hybrid mode's finish() reads this to
    # decide whether tonight's fine-tune log should get an extra "the
    # local model couldn't do this, here's what worked" example — see
    # finetune_log.log_hybrid_escalation_turn().
    preferred_escalated: bool = False
    # Real bug found live: the deterministic "rule" floor's reply is a
    # generic PM-tool capability menu ("I can create spaces/projects/
    # tasks/sprints...") — a reasonable last resort for the PM/chat
    # pipeline it was designed for, but actively wrong and confusing
    # handed back for a coding or trading request (confirmed live: "code
    # a fibonacci" got this exact PM menu text once Groq's real answer
    # was discarded as truncated and the next rung, algo-engine, was
    # unavailable). CodingAgent (and, once it exists, a Trading agent)
    # pass this True so "rule" is never even a candidate for their
    # calls — every real rung still gets tried; only the non-answer at
    # the very end is removed, replaced by an honest "every model rung
    # failed" error the caller already renders as plain text.
    exclude_rule_based: bool = False
    # CodingAgent's own fixed, always-on model cascade (see
    # llm/coding_pipeline.py) — set once at TierClient construction time,
    # never toggled by any admin/user setting. Independent of both
    # `preferred` and `pipeline_override` above: when set, it always
    # splices in ahead of this tier's own normal chain, the same
    # "walk in order, escalate only on a genuine failure" mechanism the
    # primary_pipeline RouteLLM cascade already uses, just for a caller
    # (CodingAgent) that needs its own fixed order regardless of the
    # deployment's global primary/secondary switch.
    explicit_cascade: list[str] | None = None

    def chat(self, node: str, messages: list[dict],
             tools: list[dict] | None = None) -> LLMResult:
        provs = _providers()
        # Policy gate (llm/policy_gate.py) — runs BEFORE every other
        # routing preference below and can override them outright:
        # "policy always overrides the classifier," not the other way
        # around. A block short-circuits the whole call — no provider is
        # ever contacted, same as guardrails.input_guard()'s own refusal
        # path, just reachable from every TierClient caller, not only
        # native_agent.py's chat turn.
        policy = policy_gate.evaluate(messages)
        if policy.action == "block":
            self.log.append({"node": node, "tier": self.tier, "provider": "policy-gate",
                             "policyAction": policy.action, "reason": policy.reason, "ms": 0})
            return LLMResult(text=REFUSAL_MESSAGE, provider="policy-gate")
        # provider_budget.is_exceeded() is a cheap no-op (no store read)
        # for the common case of no budget configured — see that
        # module's own docstring for why a maxed-out $/token cap
        # excludes a provider here, the SAME filtering stage
        # `.available()` already uses, so a real outage and a blown
        # budget degrade identically: skip straight to the next rung.
        available_keys = [n for n in _effective_chains()[self.tier]
                          if provs[n].available() and not provider_budget.is_exceeded(n)
                          and not (self.exclude_rule_based and n == "rule")]
        if policy.action == "force_local":
            # A secret-shaped string must never reach a hosted API — kept
            # to registry keys this deployment itself runs, plus the
            # deterministic "rule" floor (never a real model call, so
            # nothing to leak). Real outage/budget filtering above still
            # applies on top of this, same graceful-degradation story.
            available_keys = [k for k in available_keys
                              if k in policy_gate.LOCAL_PROVIDER_KEYS or k == "rule"]
        # ak-router (see router_client.py) — the actual v1 default is a
        # deterministic, cost-aware order; Thompson Sampling only ever
        # runs there as an explicit admin opt-in, restricted to two arms
        # (groq/algo-akanade-slm). NOT provider_optimizer.py/
        # cost_optimizer.py's own unconditional full-chain Thompson
        # Sampling reorder below anymore — that was the CEO memo's
        # exact objection ("do not make Thompson Sampling the default
        # router"). Their record_outcome() calls further down are left
        # in place (existing RL Performance/cost-savings dashboards
        # still read their accumulated stats), just no longer used to
        # decide THIS call's order.
        self.strategy_used = "deterministic"  # ak-router's own answer when it ran; see below
        rule_present = "rule" in available_keys
        orderable_keys = [k for k in available_keys if k != "rule"]
        router_result = None
        if orderable_keys:
            tiers = capability_tiers_for(orderable_keys)
            rates = {k: pricing_table.rates_for(k) for k in orderable_keys}
            candidates_payload = [
                {"id": k, "cost_per_1k_tokens": rates[k][0] + rates[k][1],
                 "p95_latency_ms": 2000.0, "capability_tier": tiers[k]}
                for k in orderable_keys]
            # "prompt" is the one field here that's genuinely new (see
            # ak-router's RouteContext docstring) — only ever used by the
            # admin-opt-in BERT difficulty shadow signal there, which is
            # OFF by default and never touches ordered_ids on its own;
            # every other field above is the same lightweight metadata
            # this call has always sent.
            context_payload = {"tier": self.tier,
                               "est_tokens": pricing_table.estimate_prompt_tokens(messages),
                               "task_min_capability": 0.0,
                               "prompt": _last_user_text(messages)}
            router_result = router_client.call_route(candidates_payload, context_payload)
        if router_result:
            self.strategy_used = router_result.get("strategy_used", "deterministic")
            self._shadow_pick = router_result.get("shadow_pick")
            self._bert_hint = router_result.get("bert_hint")
            self._ucb_pick = router_result.get("ucb_pick")
            # Defensive: only ever use ids ak-chatbot itself offered as
            # candidates — a malformed/stale response can't smuggle in a
            # provider key that was never actually eligible this call.
            ordered_from_service = [k for k in router_result.get("ordered_ids", [])
                                    if k in orderable_keys]
            missing = [k for k in orderable_keys if k not in ordered_from_service]
            chain_keys = ordered_from_service + missing
        else:
            # ak-router disabled/unreachable/errored: the fixed CHAINS
            # order, unchanged — never the old unconditional full-chain
            # bandit reorder. See router_service_config.py's own
            # docstring: this is the deliberate default behavior, not
            # just an error path.
            self._shadow_pick = None
            self._bert_hint = None
            self._ucb_pick = None
            chain_keys = orderable_keys
        if rule_present:
            chain_keys = chain_keys + ["rule"]
        if policy.action == "force_hosted":
            # De-prioritize, never remove — a destructive-sounding
            # request still degrades gracefully to a local rung if every
            # hosted one is genuinely down, same as any other outage.
            # "rule" stays pinned last regardless (never a real provider
            # to prefer over), matching _effective_chains()'s own
            # invariant that it's never reordered ahead of a real one.
            rule_last = [k for k in chain_keys if k == "rule"]
            reordered = [k for k in chain_keys if k != "rule"]
            reordered.sort(key=lambda k: k in policy_gate.LOCAL_PROVIDER_KEYS)
            chain_keys = reordered + rule_last
        # Blue/green primary-pipeline switch (primary_pipeline.py) — only
        # ever consulted when the caller didn't already express a GENUINE
        # explicit preference of their own (a VS Code/API caller's own
        # provider param — Hybrid mode's own local/hosted default pick is
        # handled separately, see coding_agent.py's _resolve_hybrid(),
        # which now defers to this switch itself rather than always
        # substituting its own choice): a real caller override still
        # wins outright, unconditionally, same as always.
        #
        # Splices the FULL cascade (primary_pipeline.CASCADE_ORDER: GLM-
        # 5.3-Flash, Qwen3.8-Flash, GLM-5.3-full, Sol xHigh, Opus 5), not
        # just one pick, at the FRONT of chain_keys — Default tier tried
        # first, Escalation tier only reached once every Default-tier
        # rung has genuinely failed (this loop's own per-rung
        # exception/unavailable handling below), same real-failure-driven
        # escalation every other rung in this chain already uses, not a
        # predicted-but-unverified difficulty score. TierClient's normal
        # chain (ending in groq) still follows as the always-on final
        # safety net if every cascade candidate fails too.
        pipeline_mode = primary_pipeline.get_pipeline()
        self.primary_pipeline_mode = pipeline_mode
        routellm_cascade: list[str] = []
        # Distinguishes CodingAgent's own cascade from primary_pipeline's:
        # the latter is explicitly designed to fail over into the REST of
        # this tier's normal chain (primary_pipeline.py's own docstring:
        # "a chat turn must never hard-fail" — that invariant belongs to
        # THAT mechanism, not to explicit_cascade below), so it still
        # gets prefixed onto chain_keys, leftover tail intact. CodingAgent's
        # cascade is meant to be the WHOLE story for that turn (a
        # deliberately curated, sequential Default->Escalation order,
        # per its own admin-configured tiers) — see is_exclusive below.
        is_exclusive_cascade = self.explicit_cascade is not None
        if self.explicit_cascade is not None:
            # CodingAgent's own cascade (llm/coding_pipeline.py) —
            # deliberately independent of the admin's global primary_pipeline
            # switch above: which MODEL answers here is never user-
            # toggleable (see coding_pipeline.py's module docstring), though
            # WHICH TIERS are in play now is, via coding_pipeline's own
            # admin-controlled escalation toggle. Same splice mechanism
            # as the routellm_cascade branch below, just a different,
            # fixed source list — and exclusive, not just a prefix (see
            # is_exclusive_cascade above).
            routellm_cascade = self.explicit_cascade
        elif self.preferred is None and pipeline_mode in primary_pipeline.ROUTELLM_MODES:
            routellm_cascade = primary_pipeline.resolve_routellm_cascade()
        effective_preferred = self.preferred
        if policy.action == "force_local" and effective_preferred \
                and effective_preferred not in policy_gate.LOCAL_PROVIDER_KEYS \
                and effective_preferred != "rule":
            effective_preferred = None  # ignore a hosted preference when a secret is in play
        if policy.action == "force_hosted" and effective_preferred in policy_gate.LOCAL_PROVIDER_KEYS:
            effective_preferred = None  # ignore e.g. Hybrid mode's "local" pick on a destructive turn
        if policy.action == "force_local":
            routellm_cascade = []  # the cascade is all-hosted — never applicable when a secret is in play
        if (effective_preferred and effective_preferred in provs and provs[effective_preferred].available()
                and not provider_budget.is_exceeded(effective_preferred)):
            chain_keys = [effective_preferred] + [k for k in chain_keys if k != effective_preferred]
        elif routellm_cascade:
            available_cascade = [k for k in routellm_cascade if k in provs and provs[k].available()
                                 and not provider_budget.is_exceeded(k)]
            if available_cascade:
                # Real gap found live: this used to ALWAYS prefix
                # available_cascade onto the full leftover chain — fine
                # for primary_pipeline's routellm_cascade (which wants
                # that fallback net), but for CodingAgent's own explicit,
                # curated sequence (e.g. Default -> GLM-full -> Sol-xHigh
                # -> Opus 5 -> Groq -> the local coder SLM, nothing else)
                # it silently let every OTHER configured provider (max,
                # luna, deepseek, hy4, algo-engine, ...) answer once the
                # named sequence was exhausted — exactly the opposite of
                # "every other provider should be off" for this agent.
                # Exclusive here means chain_keys IS available_cascade,
                # full stop; if every named candidate is unavailable,
                # falling through to the wider chain below (the `else`
                # branch, which leaves chain_keys untouched) is still the
                # graceful-degradation floor for that edge case only.
                chain_keys = available_cascade if is_exclusive_cascade \
                    else available_cascade + [k for k in chain_keys if k not in available_cascade]
                self.primary_pipeline_used = "routellm"
            else:
                self.primary_pipeline_used = "akanade"
        # `chain` pairs each provider with its own REGISTRY key ("luna",
        # "ollama", "rule", ...) — the same key CHAINS/`provs`/
        # provider_optimizer.py's arms use — never `.name` (the display
        # string, e.g. "gpt-5.6-luna", tier_log's own "provider" field):
        # two different namespaces in this module, and `.name` isn't
        # guaranteed unique the way a registry key is (an admin-added
        # provider could share a display name with a built-in one).
        # Comparing `.name` against `self.preferred` (a key, per
        # `self.preferred in provs` above) was a real, silent bug for
        # every BUILT-IN provider — `preferred_timeout`/`preferred_escalated`
        # below never actually fired for Hybrid mode's own Ollama pick,
        # since "ollama-llama3.2:1b" != "ollama". Keying the loop by the
        # registry key throughout fixes that as a side effect of adding
        # provider_optimizer's own key-based bookkeeping.
        chain = [(k, provs[k]) for k in chain_keys]
        first = True
        # Real bug found live: a truncated real answer (below) is
        # deliberately discarded to "try for something better" — correct
        # when a genuinely better rung is still ahead, but the "delay_ok"
        # chain's own rung after Groq (algo-engine) is unavailable on this
        # deployment, so that better attempt always instantly fails and
        # the loop falls all the way to the deterministic "rule" floor,
        # replacing a real (if cut-off) code answer with the generic PM
        # capability-menu text — strictly worse than what we already had.
        # Kept here so the "rule" branch below can hand it back instead of
        # ever calling the floor when a real answer already came in.
        best_truncated: LLMResult | None = None
        for key, p in chain:
            if key == "rule" and best_truncated is not None:
                return best_truncated
            # RL-selected optimization arm (llm/optimization_router.py) —
            # EVERY call, not just once per task: which real generation-
            # parameter combination to send, learned from this call's own
            # latency/success. Only applied if this provider actually has
            # an extra_body slot (OpenAICompatible/CustomProvider/
            # HuggingFaceProvider do; RuleBased/LangChainOllama/
            # AnthropicNative don't accept one, so they're simply skipped
            # — same "just doesn't apply here" story as every other
            # optional knob in this module).
            arm = optimization_router.select(p.name, self.tier)
            has_extra_body = hasattr(p, "extra_body")
            original_extra_body = p.extra_body if has_extra_body else None
            if has_extra_body and arm.extra_body:
                p.extra_body = {**original_extra_body, **arm.extra_body}
            has_timeout = hasattr(p, "timeout")
            original_timeout = p.timeout if has_timeout else None
            if has_timeout and self.preferred_timeout is not None and key == self.preferred:
                p.timeout = self.preferred_timeout
            # Token optimization layer (llm/token_optimizer.py) — ollama-*
            # rungs only, every other provider's `tools` is untouched.
            # DESTRUCTIVE/MCP tools are never trimmed (its own guardrail
            # hierarchy); the RL-selected reduction only narrows the rest.
            call_tools, tool_selection = optimize_tools_for_provider(p.name, messages, tools)
            t0 = time.time()
            try:
                # See _LOCAL_PROVIDER_PREFIXES/_LOCAL_HARD_TIMEOUT_S's own
                # module-level comment — a real, deterministic wall-clock
                # cap for local rungs, regardless of whether this call was
                # explicitly preferred or reached via the normal chain.
                if key.startswith(_LOCAL_PROVIDER_PREFIXES):
                    executor = ThreadPoolExecutor(max_workers=1)
                    try:
                        res = executor.submit(p.chat, messages, call_tools).result(
                            timeout=_LOCAL_HARD_TIMEOUT_S)
                    except FutureTimeoutError:
                        raise TimeoutError(
                            f"{key} exceeded the {_LOCAL_HARD_TIMEOUT_S}s hard local timeout")
                    finally:
                        executor.shutdown(wait=False)  # never block on an abandoned slow call
                else:
                    res = p.chat(messages, call_tools)
                ms = (time.time() - t0) * 1000
                optimization_router.update(p.name, arm, self.tier, ok=True, latency_ms=ms)
                provider_optimizer.record_outcome(self.tier, key, ok=True, latency_ms=ms)
                # Cost-aware optimizer (llm/cost_optimizer.py) — real
                # token counts and real (or admin-configured) $/1k rates,
                # right where provider_optimizer records the SAME call's
                # success/latency outcome above. cost_and_savings_for_call
                # is the SAME per-call helper the aggregate savings_summary()
                # baseline is built from (pricing_table.counterfactual_cost_usd)
                # — so this ONE call's costUsd/savingsUsd below, shown after
                # every real API/model call, is on the exact same basis as
                # the dashboard's whole-system total.
                usage = res.usage or {}
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost_optimizer.record_outcome(
                    self.tier, key, ok=True, latency_ms=ms, truncated=res.truncated,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                call_cost = cost_and_savings_for_call(key, prompt_tokens, completion_tokens)
                # ak-router feedback loop — a no-op unless ROUTER_SERVICE_
                # ENABLED is on (see router_client.py). Only ever
                # meaningfully updates the bandit for "groq"/
                # "algo-akanade-slm" (thompson.py's own guard); every
                # outcome still lands in RoutingOutcomes for the Router
                # Performance dashboard, real or shadow.
                router_client.report_outcome(
                    candidate_id=key, ok=True, latency_ms=ms,
                    cost_usd=call_cost["cost_usd"], strategy_used=self.strategy_used,
                    shadow_pick=self._shadow_pick, bert_hint=self._bert_hint,
                    ucb_pick=self._ucb_pick, tier=self.tier)
                self.log.append({"node": node, "tier": self.tier,
                                 "provider": p.name, "fallbackUsed": not first,
                                 "ms": int(ms), "optimizationArm": arm.arm_id,
                                 "usage": res.usage, "truncated": res.truncated,
                                 "costUsd": call_cost["cost_usd"],
                                 "savingsUsd": call_cost["savings_usd"],
                                 "routingStrategy": self.strategy_used,
                                 "primaryPipeline": self.primary_pipeline_used,
                                 "primaryPipelineMode": self.primary_pipeline_mode})
                optimization_events.log_call(node=node, tier=self.tier, provider=p.name,
                                             arm_id=arm.arm_id, ok=True, latency_ms=ms,
                                             fallback_used=not first)
                if res.usage:
                    provider_budget.record_usage(
                        key, prompt_tokens=res.usage.get("prompt_tokens", 0),
                        completion_tokens=res.usage.get("completion_tokens", 0))
                # Not recorded yet: an HTTP-level success doesn't mean the
                # (possibly-trimmed) tool list still led to the RIGHT tool
                # call — that's only known once retried/out_of_scope are
                # computed, further up the call stack. Stashed here for
                # the caller to finalize; see this field's docstring above.
                if tool_selection:
                    self.tool_selections.append(tool_selection)
                # The provider's OWN signal says this reply was cut off by
                # its context/output limit (LLMResult.truncated) — a real
                # answer came back, but an incomplete one. If there's
                # another rung left to try, prefer a genuinely complete
                # answer from it over returning a truncated one — same
                # "keep the same messages, try the next provider" path a
                # hard failure already takes, so conversation context is
                # preserved across the switch with no extra plumbing.
                if res.truncated and key != chain[-1][0]:
                    if best_truncated is None:
                        best_truncated = res
                    if key == self.preferred:
                        self.preferred_escalated = True
                    first = False
                    continue
                return res
            except Exception as e:  # network/5xx/timeout → next rung (§6.1 step 12)
                ms = (time.time() - t0) * 1000
                router_client.report_outcome(
                    candidate_id=key, ok=False, latency_ms=ms, cost_usd=0.0,
                    strategy_used=self.strategy_used, shadow_pick=self._shadow_pick,
                    bert_hint=self._bert_hint, ucb_pick=self._ucb_pick, tier=self.tier)
                self.log.append({"node": node, "tier": self.tier,
                                 "provider": p.name, "fallbackUsed": not first,
                                 "error": str(e)[:120], "ms": int(ms),
                                 "optimizationArm": arm.arm_id,
                                 "costUsd": 0.0, "savingsUsd": 0.0,
                                 "routingStrategy": self.strategy_used,
                                 "primaryPipeline": self.primary_pipeline_used,
                                 "primaryPipelineMode": self.primary_pipeline_mode})
                optimization_router.update(p.name, arm, self.tier, ok=False, latency_ms=ms)
                provider_optimizer.record_outcome(self.tier, key, ok=False, latency_ms=ms)
                cost_optimizer.record_outcome(self.tier, key, ok=False, latency_ms=ms,
                                              truncated=False, prompt_tokens=0, completion_tokens=0)
                optimization_events.log_call(node=node, tier=self.tier, provider=p.name,
                                             arm_id=arm.arm_id, ok=False, latency_ms=ms,
                                             fallback_used=not first)
                # A hard failure here IS a decisive, immediate signal for
                # this specific reduced tool list (unlike the success path
                # above, there's no "did it still pick the right tool"
                # ambiguity to wait for) — recorded right away.
                if tool_selection:
                    token_optimizer.record_outcome(tool_selection, success=False)
                if key == self.preferred:
                    self.preferred_escalated = True
                first = False
            finally:
                if has_extra_body:
                    p.extra_body = original_extra_body
                if has_timeout:
                    p.timeout = original_timeout
        # Same reasoning as the "rule" branch above, for a tier whose
        # chain doesn't end in "rule" at all — a real, truncated answer
        # still beats reporting a hard failure to the caller.
        if best_truncated is not None:
            return best_truncated
        return LLMResult(error="all providers failed", provider="none")


def get_llm_client(tier: Tier, preferred: str | None = None,
                   preferred_timeout: float | None = None,
                   exclude_rule_based: bool = False,
                   explicit_cascade: list[str] | None = None) -> TierClient:
    return TierClient(tier=tier, preferred=preferred, preferred_timeout=preferred_timeout,
                      exclude_rule_based=exclude_rule_based, explicit_cascade=explicit_cascade)


def call_provider_directly(name: str, messages: list[dict],
                           tools: list[dict] | None = None,
                           timeout: float | None = None) -> LLMResult:
    """Real, observed gap this closes: TierClient's `preferred` is a
    HINT, not a guarantee — it still degrades through the tier's own
    (bandit-reordered) fallback chain the moment the preferred rung
    errors, times out, or the bandit's own live-learned order tries a
    DIFFERENT rung first for reasons unrelated to what the caller
    actually asked for (confirmed live: requesting gemini-flash
    specifically still landed on groq, and once even the rule-based
    floor and the local ollama SLM, with zero error surfaced — exactly
    wrong for scripts/log_complex_ground_truth.py, which needs the
    NAMED strong model's own answer to distill from, never a silent
    substitute). This bypasses every chain/tier/bandit entirely: one
    named provider, called directly — a real error (unavailable,
    exception) propagates as LLMResult.error instead of triggering any
    fallback, so the caller always knows exactly which model actually
    answered."""
    provs = _providers()
    if name not in provs:
        return LLMResult(error=f"no such provider {name!r}", provider=name)
    provider = provs[name]
    if not provider.available():
        return LLMResult(error=f"provider {name!r} is not available (unconfigured?)", provider=name)
    # Same temporary mutate-then-restore TierClient.chat() already uses
    # for preferred_timeout (see its own "has_timeout" branch above) —
    # every built-in provider is a plain dataclass with its own mutable
    # `timeout` field, so this is a real override for THIS call, not a
    # standing change to the provider's normal (shorter, interactive-
    # latency-appropriate) default.
    has_timeout = hasattr(provider, "timeout")
    original_timeout = provider.timeout if has_timeout else None
    if has_timeout and timeout is not None:
        provider.timeout = timeout
    try:
        return provider.chat(messages, tools)
    except Exception as e:  # noqa: BLE001 — surfaced as a normal LLMResult.error, not a crash
        return LLMResult(error=str(e), provider=name)
    finally:
        if has_timeout and timeout is not None:
            provider.timeout = original_timeout


def available_provider_keys_for_tier(tier: Tier) -> list[str]:
    """The same availability+budget filter TierClient.chat() applies to
    itself (line ~289 above), exposed as a clean public accessor so a
    caller outside this module (routers/v1/cost.py's cost-predict
    endpoint) can list real candidates for a tier without reaching into
    this module's own private _providers()/_effective_chains()."""
    provs = _providers()
    chains = _effective_chains()
    if tier not in chains:
        return []
    return [n for n in chains[tier] if provs[n].available() and not provider_budget.is_exceeded(n)]


def available_providers() -> list[dict]:
    """Every provider named in any chain, deduped, for the UI's model
    picker — {id, name, available}. "id" is the router key (what a POST
    /api/chat/message.provider value must be — it's what TierClient.preferred
    is matched against); "name" is the same display name that shows up in
    tier_log's "provider" field, so the UI can tell whether the rung it
    asked for is the one that actually answered. "rule" (the deterministic
    floor) is always available so the dropdown never offers a dead end.
    """
    def _describe(key: str, p) -> dict:
        entry = {"id": key, "name": p.name, "available": p.available()}
        optimization = getattr(p, "optimization", None)
        if optimization is not None:
            entry["optimization"] = optimization.to_dict()
        return entry

    provs = _providers()
    seen: dict[str, dict] = {}
    for rungs in _effective_chains().values():
        for key in rungs:
            if key not in seen:
                seen[key] = _describe(key, provs[key])
    # "claude", "huggingface", "custom", and "ollama-coder" are
    # deliberately not in any chain above (config.py's own comments on
    # each) but should still be pickable/visible — added explicitly
    # rather than via a chain. "glm"/"hy4" no longer need to be listed
    # here — they're real "delay_ok" rungs now (see CHAINS), so the loop
    # above already picks them up.
    for extra_id in ("claude", "huggingface", "custom", "ollama-coder"):
        if extra_id not in seen:
            seen[extra_id] = _describe(extra_id, provs[extra_id])
    return list(seen.values())
