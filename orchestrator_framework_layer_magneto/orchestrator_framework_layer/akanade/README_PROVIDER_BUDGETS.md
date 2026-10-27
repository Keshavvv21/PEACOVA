# Provider budgets — daily token/cost caps, with automatic fallback

The problem this solves: an admin adds a real API key for a paid
provider (Claude is the motivating case, but this applies to ANY
provider with a real per-token cost — Groq, an admin-added OpenRouter
model, anything) and now has no way to cap what it can spend. A single
runaway session, a bug in a client, or just normal usage growth can run
up a real bill with no ceiling. `llm/provider_budget.py` closes that
gap: set a daily token limit and/or a daily dollar limit for any
provider, and once either is hit, that provider is automatically
excluded from the fallback chain for the rest of the day — the exact
same "just try the next rung" mechanism a real provider outage already
takes, so a maxed-out budget degrades gracefully instead of erroring or
silently overspending.

## How it works

```
Real chat turn arrives
        │
        ▼
TierClient.chat() builds the candidate chain for this tier
        │
        ▼
For each candidate provider key:
   provs[key].available()          ── base_url/api_key present?
        AND
   NOT provider_budget.is_exceeded(key)   ── today's usage under both caps?
        │
        ▼
Only providers passing BOTH checks are tried, in order — an explicit
"preferred" pick (Domain mode, Hybrid mode, or a user's own model
choice) gets the SAME check, so an admin capping "claude" can't be
bypassed by picking it directly.
        │
        ▼
A call succeeds → its real usage (prompt_tokens + completion_tokens,
from the provider's own response) is recorded against today's running
total for that provider.
        │
        ▼
Next call: if the running total now meets or exceeds either configured
cap, this same provider is excluded — automatically, no admin action,
no restart.
        │
        ▼
UTC midnight: a new day's usage row starts at zero — the provider is
available again with no manual reset.
```

**Token caps** work with zero extra configuration — every provider that
reports usage (`LLMResult.usage`) already has real numbers to check
against. **Cost caps** need an admin-supplied $/1k-token rate (there's
no live pricing feed built in for every possible provider) — setting a
`max_cost_usd` without also giving at least one of
`cost_per_1k_prompt_tokens`/`cost_per_1k_completion_tokens` is rejected
outright, rather than silently accepting a cap that could never
mathematically trip.

## Using it

**From the Admin UI**: Admin > AI Models > "Provider budgets" — **+ Add
budget**, enter the provider key (`claude` for the built-in Claude
rung, or the exact `name` of an admin-added model above), a max
tokens/day and/or max cost/day, and (for a cost cap) the real $/1k
rates from that provider's own pricing page. The table shows each
provider's today's usage and whether it's currently `OK` or `Exceeded`,
live — no refresh needed beyond the page's own poll. **Export CSV** on
that table downloads the current caps + usage snapshot.

**From the API** (`app/routers/admin_provider_budgets.py`,
Executive-role only, same bar as `admin_providers.py` since this
configures spend against a real key):

```bash
# Cap the built-in Claude rung at 500k tokens/day
curl -X PUT https://.../api/admin/provider-budgets/claude \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"max_tokens": 500000}'

# Cap it at $10/day too, using Claude's real per-token pricing
curl -X PUT https://.../api/admin/provider-budgets/claude \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"max_cost_usd": 10.0, "cost_per_1k_prompt_tokens": 0.003, "cost_per_1k_completion_tokens": 0.015}'

# See every configured budget + today's usage
curl https://.../api/admin/provider-budgets -H "Authorization: Bearer $TOKEN"

# Remove a cap entirely
curl -X DELETE https://.../api/admin/provider-budgets/claude -H "Authorization: Bearer $TOKEN"
```

`provider_key` is whatever registry key `llm/router.py`'s chains use —
the built-in rungs (`claude`, `luna`, `groq`, `deepseek`, `ollama`,
...) or the exact `name` an admin gave a custom provider via
`POST /api/admin/providers`. Setting a budget for a key that doesn't
currently exist is harmless (it just never trips, since that provider
is never in a chain to check) — useful for pre-configuring a cap before
adding the actual key.

## What's verified

- `llm/provider_budget.py`'s own logic (`tests/test_provider_budget.py`,
  6 tests): no-budget-configured never exceeds, a cost cap without a
  rate is rejected, a token cap trips at the right point, a cost cap
  trips at the right point, `record_usage` is a genuine no-op with no
  budget set, deleting a budget lifts the cap.
- The admin API (`tests/test_admin_provider_budgets.py`, 5 tests): role
  gate, set + list, cost-without-rate rejected (400), cost-with-rate
  accepted, delete.
- **The actual claim — real router integration, not mocked**
  (`tests/test_provider_budget_router_integration.py`, 2 tests):
  registers a REAL admin provider, exhausts its budget, and confirms
  `TierClient.chat()` genuinely never calls it — a different provider
  answers instead — for both the normal chain-order path AND an
  explicit `preferred` pick. This is the one that matters: it proves
  the "auto-switch" behavior works through the real chain-building
  logic, not just that the budget module's own functions return the
  right booleans in isolation.
