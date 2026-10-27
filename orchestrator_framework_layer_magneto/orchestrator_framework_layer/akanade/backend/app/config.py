"""Central configuration — every knob the design names, env-driven.

API keys live in backend/.env (never committed — see .gitignore), exactly
as §14 specifies. With no keys set, the model router (§3.4) bottoms out at
the built-in rule-based provider so the whole stack runs fully offline —
the same philosophy as the design's shadow/evaluation modes.

Loading that file is this module's job: `load_dotenv()` below only sets a
variable if it isn't ALREADY in the environment (python-dotenv's default),
so a real `export FOO=bar` or a container's `environment:` block still wins
over `.env` — `.env` is the fallback for "just running uvicorn directly",
not an override mechanism.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Explicit path, not load_dotenv()'s default upward-search — this must find
# backend/.env regardless of the working directory `uvicorn` was launched
# from. Must run before the Settings class body below (its os.getenv calls
# are evaluated once, at import time, not per-instantiation).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _jwt_secret_from_env() -> str:
    """JWT_SECRET, resolved with a FILE taking priority over the plain env
    var. Real incident this exists to stop recurring: ak-backend and
    ak-chatbot each got their secret from a separate `-e JWT_SECRET=...`
    docker-run flag, computed from the same shared file
    (deploy/update_containers.sh's $JWT_SECRET_FILE) but copy-pasted into
    two different command lines — when a stale/regressed copy of that
    script landed back on the real deployment, one container's flag went
    missing and the two silently disagreed again, breaking all auth
    between them, more than once. A file mount is structurally harder to
    lose than one `-e` flag among many: point JWT_SECRET_FILE at the
    SAME host file for every container that needs this secret (see
    update_containers.sh's `-v $JWT_SECRET_FILE:...:ro`) and they can
    never disagree, no matter how stale the rest of a deploy script gets
    — there's only one value to read, not two copies to keep in sync.
    Falls back to the plain env var (unchanged old behavior) when
    JWT_SECRET_FILE isn't set, so this is a no-op everywhere except a
    deployment that opts into the file mount."""
    jwt_secret_file = os.getenv("JWT_SECRET_FILE", "")
    if jwt_secret_file and os.path.isfile(jwt_secret_file):
        secret = Path(jwt_secret_file).read_text(encoding="utf-8").strip()
        if secret:
            return secret
    return os.getenv("JWT_SECRET", "dev-secret-change-me")


class Settings:
    # ── app ──────────────────────────────────────────────────────────
    APP_NAME = "ALGO AKANADE — Multi-Agent GenAI Network with Evolving Training and Orchestration"
    JWT_SECRET: str = _jwt_secret_from_env()
    JWT_ALG = "HS256"
    JWT_TTL_HOURS = 12

    # ── framework adapter (§2.2) ─────────────────────────────────────
    # "native"    — the reference implementation of the §6.1 algorithm
    # "langgraph" — Track A (real StateGraph + interrupt)
    # "adk"       — Track B (optional; requires google-adk installed)
    # "claude"    — Track C (optional; requires the `anthropic` package and
    #               CLAUDE_API_KEY. Real Anthropic Messages API tool-use
    #               loop, with the MCP-sourced tools below merged in
    #               alongside the built-in PM tool set — see
    #               agent/claude_agent.py.)
    CHAT_AGENT_FRAMEWORK: str = os.getenv("CHAT_AGENT_FRAMEWORK", "langgraph")

    # ── model tiers (§3) — OpenAI-compatible endpoints per provider ──
    LUNA_BASE_URL: str = os.getenv("LUNA_BASE_URL", "")
    LUNA_API_KEY: str = os.getenv("LUNA_API_KEY", "")
    LUNA_MODEL: str = os.getenv("LUNA_MODEL", "gpt-5.6-luna")

    MAX_BASE_URL: str = os.getenv("MAX_BASE_URL", "")
    MAX_API_KEY: str = os.getenv("MAX_API_KEY", "")
    MAX_MODEL: str = os.getenv("MAX_MODEL", "gpt-5.6-max")

    # Zhipu AI's GLM family, OpenAI-compatible endpoint — real, public
    # default base URL pre-filled (same precedent as GROQ_BASE_URL below),
    # only the API key needs supplying to put this rung in rotation.
    GLM_BASE_URL: str = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    GLM_API_KEY: str = os.getenv("GLM_API_KEY", "")
    GLM_MODEL: str = os.getenv("GLM_MODEL", "glm-5.3-flash")

    # Tencent Hunyuan, OpenAI-compatible endpoint — same "real public
    # default base URL, key required" story as GLM above.
    HY4_BASE_URL: str = os.getenv("HY4_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1")
    HY4_API_KEY: str = os.getenv("HY4_API_KEY", "")
    HY4_MODEL: str = os.getenv("HY4_MODEL", "hunyuan-turbos-latest")

    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    # deepseek-v4-flash was this design's placeholder name — deepseek-chat
    # is the real, current model id (verify at api-docs.deepseek.com if
    # this drifts). Note DeepSeek is pay-as-you-go, not a free tier — a 402
    # from this rung means the account has no funds, not a code problem.
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_SHADOW: bool = os.getenv("DEEPSEEK_SHADOW", "1") == "1"  # §3.4

    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # llama-3.3-70b-versatile (this design's original default) has since
    # been retired from Groq's live lineup — verified against the real
    # /v1/models endpoint and a real tool-calling request before switching.
    # Groq's free-tier model list changes over time; re-check if this 404s.
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Algo Engine (performance_optimizer/algo_optimzer_engine) — ALGO AKANADE's
    # own self-hosted model execution layer, exposed OpenAI-compatible via
    # `algo-engine serve`. A real rung in llm/router.py's CHAINS (delay_ok/
    # bulk), not just a pinnable extra. Point this at a running instance to
    # put it in rotation; empty (default) means .available() gates it out,
    # same as every other unconfigured provider. NOT MoE-only and NOT
    # GPU-only: Algo Engine is model-architecture-agnostic (dense/MoE/
    # hybrid) and picks its own execution strategy (Go/CPU, cgo/CPU, CUDA/
    # GPU, or hybrid across VRAM/RAM/SSD) per model+hardware — ALGO AKANADE
    # decides WHAT to run, Algo Engine decides HOW. See
    # performance_optimizer/Arhitecture_Opitimization.md.
    ALGO_ENGINE_BASE_URL: str = os.getenv("ALGO_ENGINE_BASE_URL", "")

    # The REAL Algo-Kaizen app's own REST API (NOT the Algo Engine model
    # tier above — that's an LLM inference endpoint, this is real PM data).
    # When set, auth.py's current_user() stops decoding the bot's own JWT
    # and instead validates the incoming token against this API's own
    # GET /api/users/me, and tools.py's read tools (list_projects,
    # query_tasks) call this API for real projects/tasks instead of the
    # demo LocalStore. Empty (the default) keeps the bot fully standalone
    # against its own Compass OS demo dataset — this is opt-in, not a
    # silent behavior change.
    REAL_APP_API_BASE_URL: str = os.getenv("REAL_APP_API_BASE_URL", "")

    # ── ak-router — the standalone provider-ordering service ──────────
    # See llm/router_client.py/router_service_config.py. The DB-backed
    # "enabled" toggle (router_service_config.py) is the real live
    # on/off switch (admin-editable, no redeploy); this URL is
    # deployment topology, not a feature flag — it only ever changes if
    # ak-router's own container/port changes.
    ROUTER_SERVICE_URL: str = os.getenv("ROUTER_SERVICE_URL", "http://ak-router:8002")
    ROUTER_SERVICE_TIMEOUT_S: float = float(os.getenv("ROUTER_SERVICE_TIMEOUT_S", "0.8"))
    # A real, observed gap: RouteLLM's "mf" router attempts a live OpenAI
    # embeddings call before falling back to "random" (see
    # generic_architecture.py's own per-call fallback) — that round trip
    # routinely exceeds the 0.8s budget above, which is sized for the
    # plain deterministic /route call, not this heavier one. Used only
    # by primary_pipeline.py's call to /generic-architecture/route.
    ROUTELLM_TIMEOUT_S: float = float(os.getenv("ROUTELLM_TIMEOUT_S", "3.0"))

    # ── extensible providers (llm/router.py's _extra_providers) ───────
    # Add ANY OpenAI-compatible API (OpenRouter, Together, Anyscale, a
    # second Groq-shaped host, ...) without touching router.py: a JSON
    # array of {name, base_url, api_key, model, tiers}. "tiers" (default
    # every tier) is which chain(s) — realtime/delay_ok/bulk — it joins,
    # inserted just before the rule-based floor. See llm/README.md.
    EXTRA_LLM_PROVIDERS: str = os.getenv("EXTRA_LLM_PROVIDERS", "[]")

    # A local small language model via Ollama (LangChain's ChatOllama) —
    # genuinely offline, no API key, runs on a laptop GPU/CPU. Only
    # activates if both this URL is set AND langchain-ollama is installed
    # (an optional dependency, see requirements.txt); otherwise it's just
    # another unavailable rung, same as any hosted tier with no key set.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    # How long Ollama keeps a model loaded in memory after its last
    # request before unloading it — Ollama's own default is "5m", real
    # enough that a normal gap between turns (reading a reply, switching
    # tasks) routinely triggers a ~10-15s reload on the NEXT message.
    # 30 minutes trades a modest amount of idle RAM/VRAM for that reload
    # essentially never firing during a real working session.
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    # A SECOND, code-specialized local model on the SAME Ollama server
    # (same OLLAMA_BASE_URL — no separate service to run) — registered as
    # its own "ollama-coder" provider key (llm/router.py) that CodingAgent
    # (framework/agents/coding_agent.py) explicitly prefers, so a coding
    # request opportunistically tries a model actually trained on code
    # first, without touching what OLLAMA_MODEL serves for every other
    # domain. Deliberately NOT in any CHAINS entry (realtime/delay_ok/
    # bulk) — the PM chatbot's own traffic never routes here by default,
    # only an explicit `preferred="ollama-coder"` pick does. Requires
    # `ollama pull qwen2.5-coder:1.5b` (or whatever size fits — 7b/14b/32b
    # all exist) on the same box `ollama serve` runs on; unpulled just
    # means this rung's own `.available()` fails through like any other,
    # same graceful-degradation story as every other optional provider.
    OLLAMA_CODER_MODEL: str = os.getenv("OLLAMA_CODER_MODEL", "qwen2.5-coder:1.5b")

    # The real Anthropic Claude API — a genuinely different wire format
    # (native tool_use/tool_result content blocks) from every OpenAI-
    # compatible rung above, so it's its own provider class
    # (llm/providers.py's AnthropicNative), not another OpenAICompatible.
    # Registered in the router but deliberately NOT spliced into the
    # realtime/delay_ok/bulk CHAINS below — it answers when the "claude"
    # framework (agent/claude_agent.py) asks for it by name via
    # TierClient(preferred="claude"), and degrades to that same chain if
    # unset/unreachable, same graceful-fallback story as every other rung.
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

    # ALGO AKANADE's generic model layer — Hugging Face (llm/providers.py's
    # HuggingFaceProvider, real `huggingface_hub` SDK). Any model on HF
    # Inference Providers works via HUGGINGFACE_MODEL alone; set
    # HUGGINGFACE_ENDPOINT_URL instead to target a dedicated HF
    # Inference Endpoint.
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "")
    HUGGINGFACE_ENDPOINT_URL: str = os.getenv("HUGGINGFACE_ENDPOINT_URL", "")
    HUGGINGFACE_INFERENCE_PROVIDER: str = os.getenv("HUGGINGFACE_INFERENCE_PROVIDER", "auto")
    # Same real pass-through / declarative-metadata pair as
    # CUSTOM_MODEL_EXTRA_BODY/CUSTOM_MODEL_OPTIMIZATION below — see
    # llm/optimization.py.
    HUGGINGFACE_EXTRA_BODY: str = os.getenv("HUGGINGFACE_EXTRA_BODY", "{}")
    HUGGINGFACE_OPTIMIZATION: str = os.getenv("HUGGINGFACE_OPTIMIZATION", "{}")

    # ALGO AKANADE's generic model layer — "any custom or locally-hosted model"
    # (llm/providers.py's CustomProvider — an OpenAICompatible alias):
    # vLLM, llama.cpp's server, LM Studio, a hand-rolled inference API,
    # anything that speaks the OpenAI chat-completions wire format.
    # requires_key defaults to false (most local servers have no auth);
    # set CUSTOM_MODEL_API_KEY if yours does.
    CUSTOM_MODEL_BASE_URL: str = os.getenv("CUSTOM_MODEL_BASE_URL", "")
    CUSTOM_MODEL_API_KEY: str = os.getenv("CUSTOM_MODEL_API_KEY", "")
    CUSTOM_MODEL_NAME: str = os.getenv("CUSTOM_MODEL_NAME", "custom-model")
    # A JSON object merged verbatim into every request payload — real
    # pass-through to whatever this endpoint's server actually supports
    # (vLLM/TGI-specific sampling params, a speculative-decoding knob,
    # ...). e.g. '{"top_k": 40, "repetition_penalty": 1.1}'
    CUSTOM_MODEL_EXTRA_BODY: str = os.getenv("CUSTOM_MODEL_EXTRA_BODY", "{}")
    # Declarative only (llm/optimization.py's module docstring) — what
    # THIS endpoint's server is actually deployed with, e.g.
    # '{"kv_cache": true, "continuous_batching": true, "tensor_parallel_size": 4}'
    # Reported via GET /api/v1/models; never changes how a request is made.
    CUSTOM_MODEL_OPTIMIZATION: str = os.getenv("CUSTOM_MODEL_OPTIMIZATION", "{}")

    # ── MCP (Model Context Protocol) client (agent/mcp_client.py) ────
    # A JSON array of {name, command, args, env}, one per external MCP
    # server to launch over stdio and pull tools from — same config-
    # driven-registry shape as EXTRA_LLM_PROVIDERS above, so adding a
    # real external MCP server (filesystem, GitHub, Slack, ...) is an
    # env var, not a code change. Two default entries:
    #   "utility" — the bundled, offline, zero-side-effect example server
    #     (mcp_servers/utility_server.py), so the integration is exercised
    #     with zero operator setup and nothing that reaches a real system.
    #   "fetch" — Anthropic's own reference fetch MCP server
    #     (mcp-server-fetch, see requirements.txt): a genuinely useful
    #     external ability — read a real web page — that's still free,
    #     open source, and needs no API key/account. Every tool call
    #     through it (like every MCP tool) requires an explicit yes/no
    #     before it runs — see native_agent.requires_confirmation.
    MCP_SERVERS: str = os.getenv(
        "MCP_SERVERS",
        '[{"name": "utility", "command": "python", '
        '"args": ["-m", "app.mcp_servers.utility_server"]}, '
        '{"name": "fetch", "command": "python", '
        '"args": ["-m", "mcp_server_fetch"]}]',
    )

    # ── OAuth connections (app/oauth/) — per-user, write-capable MCP ──
    # Real "connect your account" OAuth 2.0 for MCP servers that require
    # it (GitHub, Linear, Google) — a genuinely different, bigger
    # mechanism than MCP_SERVERS' static headers above: each USER
    # authorizes their OWN account (a browser redirect/consent screen),
    # not one shared token for everyone. See app/oauth_store.py and
    # app/agent/user_mcp_client.py. Any provider left without a
    # client id/secret here is just unavailable — same "optional,
    # additive" story as every other unconfigured provider/rung.
    GITHUB_OAUTH_CLIENT_ID: str = os.getenv("GITHUB_OAUTH_CLIENT_ID", "")
    GITHUB_OAUTH_CLIENT_SECRET: str = os.getenv("GITHUB_OAUTH_CLIENT_SECRET", "")
    LINEAR_OAUTH_CLIENT_ID: str = os.getenv("LINEAR_OAUTH_CLIENT_ID", "")
    LINEAR_OAUTH_CLIENT_SECRET: str = os.getenv("LINEAR_OAUTH_CLIENT_SECRET", "")
    GOOGLE_OAUTH_CLIENT_ID: str = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
    GOOGLE_OAUTH_CLIENT_SECRET: str = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
    # This service's own public URL — used to build each provider's
    # redirect_uri (must exactly match what's registered on that
    # provider's OAuth app, protocol+host+port+path all included).
    OAUTH_REDIRECT_BASE_URL: str = os.getenv("OAUTH_REDIRECT_BASE_URL", "http://localhost:8001")
    # The PM frontend's own URL — where /api/oauth/{provider}/callback
    # 302-redirects the browser back to once a connection is saved.
    FRONTEND_BASE_URL: str = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")
    # Fernet key (Fernet.generate_key()) encrypting stored OAuth tokens
    # at rest (app/crypto.py) — unlike admin_providers.py's plaintext
    # LLM API keys, a stolen OAuth token grants live write access to a
    # real person's real GitHub/Linear/Google account, a materially
    # higher-value target. Empty in dev falls back to a fixed
    # (non-secret) key so local/test runs don't need one set — never
    # rely on that default outside a laptop/CI run.
    OAUTH_TOKEN_ENCRYPTION_KEY: str = os.getenv("OAUTH_TOKEN_ENCRYPTION_KEY", "")

    # ── storage ──────────────────────────────────────────────────────
    # "local" (JSON files, zero infra — dev/tests) | "dynamodb"
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    DYNAMO_ENDPOINT: str = os.getenv("DYNAMO_ENDPOINT", "")  # dynamodb-local
    DYNAMO_REGION: str = os.getenv("AWS_REGION", "ap-south-1")

    # ── retrieval ────────────────────────────────────────────────────
    GRAPH_TTL_SECONDS: int = int(os.getenv("GRAPH_TTL_SECONDS", "60"))  # §4.1
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", os.path.join(os.getcwd(), "chroma_db"))
    EMBEDDER: str = os.getenv("EMBEDDER", "auto")  # auto|minilm|hash
    # gmaofn/document_pipeline.py's ingest_path() only ever reads files
    # resolved to be INSIDE this directory (path-traversal guard — the
    # REST endpoint takes a relative path from a caller, not a trusted
    # server operator, so "any file on the server" must never be reachable).
    # Anchored to DATA_DIR, NOT os.getcwd() directly — the deployed
    # chatbot image bakes DATA_DIR=/srv/data (Dockerfile), matching the
    # ak-chatbot-data volume mount; os.getcwd() there is /srv/app
    # (WORKDIR), an ephemeral path nothing persists. Piggybacking on
    # DATA_DIR means an ingested document survives a container restart
    # for free, same as everything else that already lives there
    # (finetune_log.py's JSONL, local-mode JSON stores).
    RAG_INGEST_DIR: str = os.getenv("RAG_INGEST_DIR", os.path.join(DATA_DIR, "rag_ingest"))
    # CodingAgent's read_file/write_file tools (agents/coding_agent.py) —
    # same path-traversal guard as RAG_INGEST_DIR above (document_pipeline.
    # resolve_within(), reused directly), a SEPARATE directory so a coding
    # session can never read/overwrite an ingested RAG document just
    # because both live under DATA_DIR. write_file additionally requires
    # an explicit confirmation round-trip regardless of this scoping —
    # this only bounds WHERE a write can land, not WHETHER one happens.
    CODING_WORKSPACE_DIR: str = os.getenv("CODING_WORKSPACE_DIR",
                                          os.path.join(DATA_DIR, "coding_workspace"))

    # ── sessions (§9) ────────────────────────────────────────────────
    SESSION_WINDOW_TURNS: int = int(os.getenv("SESSION_WINDOW_TURNS", "12"))
    SESSION_TTL_DAYS: int = int(os.getenv("SESSION_TTL_DAYS", "7"))

    # ── intent-answer cache (framework/intent_cache.py) ────────────────
    # A hit skips the SLM/LLM call entirely and returns a previously-
    # generated answer. Gated by CALLERS to only pure-conversational
    # messages (chat.py) and tool-free generate/general_coding turns
    # (coding_agent.py) — the TTLs below are a staleness bound on top of
    # that gating, not the primary safety mechanism.
    INTENT_CACHE_ENABLED: bool = os.getenv("INTENT_CACHE_ENABLED", "true").lower() == "true"
    INTENT_CACHE_MAX_ENTRIES: int = int(os.getenv("INTENT_CACHE_MAX_ENTRIES", "500"))
    INTENT_CACHE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("INTENT_CACHE_SIMILARITY_THRESHOLD", "0.96"))
    CONVERSATIONAL_CACHE_TTL_SECONDS: int = int(
        os.getenv("CONVERSATIONAL_CACHE_TTL_SECONDS", str(24 * 3600)))
    CODING_CACHE_TTL_SECONDS: int = int(os.getenv("CODING_CACHE_TTL_SECONDS", "3600"))


@lru_cache
def settings() -> Settings:
    return Settings()
