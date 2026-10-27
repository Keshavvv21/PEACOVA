"""Per-provider daily token/cost caps (llm/provider_budget.py) —
"if I add a Claude API key, I want a way to cap what it can spend, and
have the system auto-fall-through to another provider once it's hit."
Not Claude-specific: any provider key works, built-in ("claude",
configured via CLAUDE_API_KEY) or admin-added
(routers/admin_providers.py's own `name`). Setting/deleting a budget
(PUT/DELETE) is Executive-role only, same bar as admin_providers.py —
this configures spend against a real API key. Reading is Admin-role
(admits the real app's Manager2/"Senior Manager", per auth.py's
_REAL_ROLE_MAP) — budget numbers and today's usage aren't secrets the
way admin_providers.py's api_key is.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_role
from ..llm import provider_budget
from ..models import ProviderBudgetIn, ProviderBudgetOut

router = APIRouter(prefix="/api/admin/provider-budgets", tags=["admin-provider-budgets"])


def _to_out(row: dict) -> ProviderBudgetOut:
    usage = provider_budget.today_usage(row["providerKey"])
    return ProviderBudgetOut(
        provider_key=row["providerKey"], max_tokens=row.get("maxTokens"),
        max_cost_usd=row.get("maxCostUsd"),
        cost_per_1k_prompt_tokens=row.get("costPer1kPromptTokens"),
        cost_per_1k_completion_tokens=row.get("costPer1kCompletionTokens"),
        today_tokens=usage.get("tokens", 0), today_cost_usd=usage.get("costUsd", 0.0),
        exceeded=provider_budget.is_exceeded(row["providerKey"]), updated_at=row["updatedAt"])


@router.get("", response_model=list[ProviderBudgetOut])
def list_budgets(_: dict = Depends(require_role("Admin"))):
    rows = provider_budget.list_budgets()
    return [_to_out(r) for r in sorted(rows, key=lambda r: r["providerKey"])]


@router.put("/{provider_key}", response_model=ProviderBudgetOut)
def upsert_budget(provider_key: str, payload: ProviderBudgetIn,
                  _: dict = Depends(require_role("Executive"))):
    try:
        row = provider_budget.set_budget(
            provider_key, max_tokens=payload.max_tokens, max_cost_usd=payload.max_cost_usd,
            cost_per_1k_prompt_tokens=payload.cost_per_1k_prompt_tokens,
            cost_per_1k_completion_tokens=payload.cost_per_1k_completion_tokens)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _to_out(row)


@router.delete("/{provider_key}", status_code=204)
def delete_budget(provider_key: str, _: dict = Depends(require_role("Executive"))):
    if not provider_budget.get_budget(provider_key):
        raise HTTPException(404, "No budget configured for that provider")
    provider_budget.delete_budget(provider_key)
