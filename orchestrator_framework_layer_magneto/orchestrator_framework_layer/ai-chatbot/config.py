"""Central configuration — every knob the design names, env-driven.

API keys live in .env (never committed), exactly as §14 specifies.
With no keys set, the model router (§3.4) bottoms out at the built-in
rule-based provider so the whole stack runs fully offline — the same
philosophy as the design's shadow/evaluation modes.
"""
from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    # ── app ──────────────────────────────────────────────────────────
    APP_NAME = "Algo PM Chat Bot"
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-me")
    JWT_ALG = "HS256"
    JWT_TTL_HOURS = 12

    # ── framework adapter (§2.2) ─────────────────────────────────────
    # "native"    — the reference implementation of the §6.1 algorithm
    # "langgraph" — Track A (real StateGraph + interrupt)
    # "adk"       — Track B (optional; requires google-adk installed)
    CHAT_AGENT_FRAMEWORK: str = os.getenv("CHAT_AGENT_FRAMEWORK", "langgraph")

    # ── model tiers (§3) — OpenAI-compatible endpoints per provider ──
    LUNA_BASE_URL: str = os.getenv("LUNA_BASE_URL", "")
    LUNA_API_KEY: str = os.getenv("LUNA_API_KEY", "")
    LUNA_MODEL: str = os.getenv("LUNA_MODEL", "gpt-5.6-luna")

    MAX_BASE_URL: str = os.getenv("MAX_BASE_URL", "")
    MAX_API_KEY: str = os.getenv("MAX_API_KEY", "")
    MAX_MODEL: str = os.getenv("MAX_MODEL", "gpt-5.6-max")

    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    DEEPSEEK_SHADOW: bool = os.getenv("DEEPSEEK_SHADOW", "1") == "1"  # §3.4

    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # The future fifth tier (§18/§19 of the design): a local Algo Engine
    # OpenAI-compatible endpoint. A config-table change, as promised.
    ALGO_ENGINE_BASE_URL: str = os.getenv("ALGO_ENGINE_BASE_URL", "")

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

    # ── sessions (§9) ────────────────────────────────────────────────
    SESSION_WINDOW_TURNS: int = int(os.getenv("SESSION_WINDOW_TURNS", "12"))
    SESSION_TTL_DAYS: int = int(os.getenv("SESSION_TTL_DAYS", "7"))


@lru_cache
def settings() -> Settings:
    return Settings()