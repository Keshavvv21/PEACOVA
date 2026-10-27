"""§10 — the Model Abstraction Layer's REST surface: every configured
model provider (OpenAI-compatible hosts, Anthropic, Hugging Face, Ollama,
a custom/local endpoint, ...), independent of which agent uses it."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...auth import current_user
from ...llm.router import available_providers

router = APIRouter(prefix="/api/v1/models", tags=["gmaofn-models"])


@router.get("")
def list_models(_: dict = Depends(current_user)):
    return {"providers": available_providers()}
