"""Pydantic schemas for the mini Aksha PM API + chat wire shapes."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["Executive", "Admin", "Manager", "Member"]
# Project-scoped role — distinct from the global Role above (mirrors the real
# app's Owner/CoOwner/Manager/Member membership model). The project creator
# is always "Owner"; add_member assigns any of the other three to anyone else.
ProjectRole = Literal["Owner", "CoOwner", "Manager", "Member"]
Status = Literal["Backlog", "Todo", "InProgress", "Done"]
Priority = Literal["Low", "Medium", "High", "Urgent"]


class LoginIn(BaseModel):
    # External clients (e.g. the Tally connector, via @algo-akanade/connector)
    # always send {email, password} — this local demo store has no separate
    # email field and keys users by username, so `email` is accepted as an
    # alias: whichever of the two is sent is looked up as the username.
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    user_id: str
    name: str


class SpaceIn(BaseModel):
    name: str


class ProjectIn(BaseModel):
    name: str
    space_id: str
    board: Literal["Kanban", "Scrum"] = "Kanban"


class TaskIn(BaseModel):
    title: str
    description: str = ""
    assignee_id: Optional[str] = None
    start_date: Optional[str] = None  # ISO date
    due_date: Optional[str] = None  # ISO date
    priority: Priority = "Medium"
    labels: list[str] = Field(default_factory=list)


class TaskPatch(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    assignee_id: Optional[str] = None
    start_date: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[Priority] = None
    status: Optional[Status] = None
    labels: Optional[list[str]] = None


class SprintIn(BaseModel):
    name: str


class SubtaskIn(BaseModel):
    title: str


class MemberAddIn(BaseModel):
    user_id: str
    role: ProjectRole = "Member"


class UserCreateIn(BaseModel):
    name: str
    role: Role
    password: str = "demo123"
    email: str = ""


class ChatMessageIn(BaseModel):
    message: str
    session_id: str = "default"
    # The UI's model picker (see llm/router.py's TierClient.preferred) —
    # a provider name from GET /api/chat/providers, or None/omitted to let
    # the tier's normal fallback chain decide, unchanged from before this
    # field existed.
    provider: Optional[str] = None
    # Which tab the web/mobile UI's chat is currently on ("trading" or
    # omitted/anything else = the default PM tab) — the Trading Agent's
    # tools (app/agent/trading_tools.py) already ride the SAME tool-
    # calling pipeline as PM (see README_TRADING_AGENT.md), with no
    # request-shape difference before this field existed, so the model
    # had no signal at all that a plain-text (non-tool-call) answer on
    # the Trading tab is a real miss, not "off-topic PM chit-chat" —
    # confirmed live: a genuine backtest request got the "not grounded
    # in your PM data" out-of-scope treatment. Used to (a) nudge the
    # model toward the trading tools for a finance-shaped question, and
    # (b) let the UI show trading-appropriate copy instead of the PM-
    # specific out-of-scope message for an out-of-scope reply on this
    # tab.
    target: Optional[str] = None


class ChatReply(BaseModel):
    reply: str
    pending_confirmation: bool = False
    confirm_prompt: Optional[str] = None
    session_id: str
    framework: str
    tier_log: list[dict] = Field(default_factory=list)
    # True when this reply is free-text general-knowledge/opinion, not an
    # answer grounded in any PM tool/data — the widget gates reveal of it
    # behind an explicit "show me anyway" instead of presenting it with
    # the same trust as a real PM lookup. See agent/state.py's AgentReply.
    out_of_scope: bool = False
    # Which provider actually answered (e.g. "ollama-qwen2.5:1.5b") — lets
    # the widget show which model handled this turn.
    provider: Optional[str] = None
    # llm/token_optimizer.py's per-call estimate for THIS turn — present
    # only when the local model's tool schema was actually reduced.
    token_stats: Optional[dict] = None
    # gmaofn/rag_optimizer.py's citation list for THIS turn — [] when RAG
    # wasn't attempted or found nothing.
    sources_used: list[dict] = Field(default_factory=list)
    # gmaofn/query_refinement.py's result for THIS turn — None when the
    # bandit gate chose "skip" (message was already clear). Present (even
    # with was_repaired=False) whenever refinement actually ran, so the UI
    # can show what changed, missing clarifications, or suggested rephrasings.
    refinement: Optional[dict] = None
    # The answering provider's own reported token usage for THIS turn —
    # {"prompt_tokens", "completion_tokens", "total_tokens"}, None for a
    # provider that doesn't report it (e.g. the local Ollama rung).
    usage: Optional[dict] = None
    # True when an earlier provider in this turn's chain hit its own
    # context/output limit and TierClient escalated to a later one instead
    # of returning an incomplete answer — see llm/router.py's `truncated`.
    context_limit_switch: bool = False
    # security_guard.py's aggregate flag for this reply — "safe" |
    # "needs_review" | "risk" (agent/state.py's AgentReply.security_flag,
    # passed through verbatim). None only for a reply path that predates
    # this field (shouldn't happen — every finish() computes it).
    security_flag: Optional[str] = None
    security_findings: list[str] = Field(default_factory=list)


# Admin-managed LLM providers (see routers/admin_providers.py, llm/router.py's
# _db_provider_entries()) — the UI equivalent of hand-editing
# EXTRA_LLM_PROVIDERS in .env, except it takes effect immediately (no
# redeploy) and can be toggled on/off per provider. Executive-role only
# (real app's actual Admin — see auth.py's _REAL_ROLE_MAP) since this
# stores real third-party API keys.
class ProviderIn(BaseModel):
    name: str
    base_url: str
    model: str
    api_key: str
    tiers: list[Literal["realtime", "delay_ok", "bulk"]] = Field(
        default_factory=lambda: ["realtime", "bulk"])
    # ak-router's deterministic strategy input — see
    # llm/capability_tiers.py. 0.3 (not 0.5) for a brand-new
    # admin-added provider: a conservative "unproven" starting point per
    # the router-service plan's "Router flag/update" closed loop, raised
    # once a real model-evaluation benchmark run (or nightly PEFT
    # promotion) has real quality numbers for it.
    capability_tier: float = 0.3


class ProviderUpdate(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    tiers: Optional[list[Literal["realtime", "delay_ok", "bulk"]]] = None
    enabled: Optional[bool] = None
    capability_tier: Optional[float] = None


class ProviderOut(BaseModel):
    id: str
    name: str
    base_url: str
    model: str
    # Never the real key — just whether one's set, so the UI can show
    # "configured" without ever re-displaying (or re-transmitting) the
    # actual secret after creation.
    api_key_set: bool
    tiers: list[str]
    enabled: bool
    capability_tier: float
    created_at: str


# One record per nightly SLM-generation-layer run (see
# routers/admin_slm.py, scripts/convert_and_register_ollama_model.sh) —
# the local, inspectable version history the Admin dashboard reads,
# independent of whether that night's candidate actually got registered
# as a provider. Executive-role only, same bar as ProviderIn/Out since
# this is part of the same "which model is actually answering" surface.
class SlmVersionIn(BaseModel):
    tag: str
    base_tag: str
    quant_type: str
    sanity_check: Literal["passed", "failed"]
    regression_gate: Optional[Literal["passed", "failed", "could_not_run"]] = None
    registered: bool = False
    # None = the general-purpose provider ("algo-akanade-slm"); a value
    # (e.g. "legal") = one of possibly several domain-specific variants
    # llm/domain_router.py RL-selects among — see
    # README_SLM_GENERATION_LAYER.md §5.
    domain: Optional[str] = None


# What SHOULD get trained tonight — one row per domain (None = the
# general-purpose model), editable via the Admin UI instead of hand-
# editing shell-script env vars on the deploy box. deploy/nightly_
# finetune.sh fetches every `enabled` row here at 03:30 UTC and runs
# scripts/finetune_local_slm.py + convert_and_register_ollama_model.sh
# once per row — see README_SLM_GENERATION_LAYER.md §5's "choose any
# base model, multiple domains, no shell access needed" story.
class FinetuneConfigIn(BaseModel):
    # None = the general-purpose model; a value (e.g. "legal") produces
    # "algo-akanade-slm-<domain>" — same naming convention
    # llm/domain_router.py already discovers by.
    domain: Optional[str] = None
    # Any HuggingFace model id finetune_local_slm.py's --base-model
    # accepts (Qwen/SmolLM/Llama, ...) — "choose any SLM" from the UI
    # rather than editing deploy/finetune.env by hand.
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # The EXISTING Ollama tag this domain's candidate is gated against
    # (scripts/eval/regression_gate.py) — must already be pulled into
    # ak-ollama; changing base_model without updating this to match
    # compares apples to oranges on gate night. Ignored when
    # serve_backend="llama_server" (there's no Ollama tag in that mode —
    # the gate compares against whatever GGUF is currently live for
    # this domain instead, see convert_and_register_ollama_model.sh).
    ollama_base_tag: str = "qwen2.5:1.5b"
    # "ollama" (default, unchanged behavior) registers the candidate as
    # an Ollama tag via `ollama create`. "llama_server" skips Ollama
    # entirely and serves the exported GGUF directly with llama.cpp's
    # own OpenAI-compatible server (deploy/Dockerfile.llama-server) —
    # genuinely "any SLM" since GGUF conversion is already model-
    # family-agnostic (Qwen, SmolLM, Llama, ...), not limited to
    # whatever Ollama happens to ship a Modelfile template for. See
    # README_SLM_GENERATION_LAYER.md §5 for the full mechanics.
    serve_backend: Literal["ollama", "llama_server"] = "ollama"
    max_minutes: int = 120
    enabled: bool = True
    # The PEFT Analyzer's own input (app/agent/peft_analyzer.py, run by
    # deploy/nightly_finetune.sh before the fine-tune step below) —
    # Wikipedia article titles to scrape into real (prompt, verified-
    # answer) ground-truth pairs, so a domain with little organic chat
    # traffic still has something real to train on tonight. Empty means
    # "scrape nothing for this config" — an existing config's behavior
    # is unchanged until an Admin actually adds topics.
    scrape_topics: list[str] = []


class FinetuneConfigOut(FinetuneConfigIn):
    id: str
    created_at: str
    updated_at: str


class SlmVersionOut(SlmVersionIn):
    id: str
    created_at: str


# One row of scripts/eval/regression_gate.py's own per-(candidate, test-
# asset) scoring — see harness.py's score_item(). Kept only for the
# "new" (candidate) tag, not the base tag it was compared against; the
# base tag's own scores already exist from whichever earlier night
# produced ITS report.
class SlmEvalItemResult(BaseModel):
    item_id: str
    category: str
    scoring_type: str
    score: float
    error: Optional[str] = None
    answer: str = ""


# The daily verification report every regression-gate run produces —
# "did tonight's candidate actually get tested, on what, and how did it
# score item by item" — shown in the Admin dashboard the same way
# admin_security.py's daily digest is: a real report an operator can
# read without SSHing into the deploy box. Executive-role only, same
# bar as SlmVersionIn/Out since this is the detailed evidence behind
# THAT record's regression_gate field.
class SlmEvalReportIn(BaseModel):
    tag: str
    base_tag: str
    domain: Optional[str] = None
    tolerance: float
    old_score: float
    new_score: float
    passed: bool
    old_by_category: dict[str, Optional[float]] = Field(default_factory=dict)
    new_by_category: dict[str, Optional[float]] = Field(default_factory=dict)
    items: list[SlmEvalItemResult] = Field(default_factory=list)


class SlmEvalReportOut(SlmEvalReportIn):
    id: str
    created_at: str


# A daily token/$ cap for one provider (llm/provider_budget.py) — keyed
# by the same registry key llm/router.py's chains use ("claude", or an
# admin-added provider's own name). Executive-role only, same bar as
# ProviderIn/Out since this configures spend against a real API key.
class ProviderBudgetIn(BaseModel):
    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    # At least one of these two is required if max_cost_usd is set —
    # see provider_budget.set_budget()'s own validation for why.
    cost_per_1k_prompt_tokens: Optional[float] = None
    cost_per_1k_completion_tokens: Optional[float] = None


class ProviderBudgetOut(ProviderBudgetIn):
    provider_key: str
    today_tokens: int
    today_cost_usd: float
    exceeded: bool
    updated_at: str


# The list endpoint's shape — everything EXCEPT `items`, so paging
# through report history doesn't ship every prompt/answer over the
# wire; fetch GET .../{id} for the full per-item breakdown.
class SlmEvalReportSummaryOut(BaseModel):
    id: str
    tag: str
    base_tag: str
    domain: Optional[str] = None
    tolerance: float
    old_score: float
    new_score: float
    passed: bool
    n_items: int
    n_item_errors: int
    created_at: str


# Ground Truth Validation (app/agent/ground_truth_validator.py) — a
# SEPARATE signal from the eval reports above. Those score a candidate
# against static, hand-written gold answers (eval/test_assets/*.yaml);
# this instead asks a strong hosted model (Groq) to generate a reference
# answer, on the spot, for prompts drawn from TONIGHT's own new curated
# training data — "does the freshly fine-tuned model still roughly agree
# with what a much bigger model would have said about the exact thing it
# was just trained on." `match_score` is a cheap, real cosine-similarity
# comparison (app.retrieval.vectorstore.HashEmbedder) between the two
# answers' embeddings, NOT an LLM judge call — see that module's
# docstring for why this reuses the same technique data_engine.py's
# dedup and task_memory.py's routing-relevance checks already use,
# rather than adding a second LLM-as-judge dependency.
class GroundTruthRunItem(BaseModel):
    prompt: str
    groq_answer: str
    model_answer: str
    match_score: float


class GroundTruthRunIn(BaseModel):
    domain: Optional[str] = None
    provider: str
    items: list[GroundTruthRunItem] = Field(default_factory=list)


class GroundTruthRunOut(GroundTruthRunIn):
    id: str
    avg_match_score: float
    created_at: str


# The list endpoint's shape — everything except `items` (same reasoning
# as SlmEvalReportSummaryOut above): fetch GET .../{id} for the full
# per-prompt prompt/groq-answer/model-answer/score breakdown.
class GroundTruthRunSummaryOut(BaseModel):
    id: str
    domain: Optional[str] = None
    provider: str
    n_items: int
    avg_match_score: float
    created_at: str
