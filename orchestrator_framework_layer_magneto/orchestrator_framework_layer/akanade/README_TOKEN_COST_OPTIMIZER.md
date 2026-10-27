# Token & Cost Optimizer

ALGO AKANADE's cost layer — every LLM call already goes through a small
number of paid, hosted models (Claude/GPT-class tiers, Groq, DeepSeek)
alongside free local SLMs (Ollama). This layer makes that choice
cost-aware: it predicts what a call would cost before it's made,
learns (via RL) to prefer the cheapest option that's still actually
answering successfully, and tracks real token usage, latency, quality,
and $ savings so the answer to "how much is AI actually costing us,
and is it going down" is a real number, not a guess.

## What it does

1. **Predicts cost before execution** — `POST /api/v1/cost/predict`
   estimates token count and $ cost across every model currently
   available for a tier, before any of them is actually called.
2. **Selects the most cost-effective model while maintaining quality**
   — a Thompson Sampling bandit re-ranks a tier's provider chain,
   seeded toward cheaper candidates but learning from real outcomes:
   success matters most, then latency, then whether the reply was
   truncated (a real quality signal), then real dollar cost.
3. **Uses RL feedback to continuously improve** — every real call
   updates the bandit's posterior for that (tier, provider) pair; the
   ranking keeps adapting as real evidence accumulates, the same
   Thompson Sampling primitive (`framework/bandit.py`) every other RL
   layer in this codebase already uses.
4. **Tracks token usage, latency, quality, and cost savings** — `GET
   /api/v1/metrics/cost` (Admin) returns real per-(tier, provider)
   totals and a whole-system savings estimate.

## Where it lives

- `app/llm/pricing_table.py` — real, best-effort $/1k-token rates per
  provider (public list prices, admin-overridable — see below), plus
  the chars/4 token-count heuristic every estimate is built from.
- `app/llm/cost_optimizer.py` — the RL layer itself: `estimate()`
  (pre-execution prediction), `order()` (cost-aware re-ranking),
  `record_outcome()` (real per-call reward), `stats()` /
  `savings_summary()` (the dashboard's data source).
- `app/routers/v1/cost.py` — `POST /api/v1/cost/predict` (any
  authenticated user).
- `app/routers/v1/metrics.py` — `GET /api/v1/metrics/cost` (Admin).
- `tool_layer_akanade/frontend/src/pages/CostOptimizerDashboard.tsx` —
  the dashboard (Admin + Senior Manager), reachable from the sidebar as
  **Token & Cost Optimizer**.

## How it composes with the existing routing stack

`app/llm/router.py`'s `TierClient.chat()` already had a provider-
ordering bandit (`llm/provider_optimizer.py`) — its own docstring is
explicit that its reward is "pure success+latency... no cost term at
all." This layer doesn't replace that; it adds a second re-rank
immediately after it:

```
policy gate (safety) → availability/budget filter
  → provider_optimizer.order()   (success + latency, existing)
  → cost_optimizer.order()       (+ real $ cost, this layer)
  → force_local/force_hosted policy resort (safety, still wins outright)
  → preferred override (a user's own model pick, still wins outright)
```

Same "augments, never replaces the safety net" rule every optimizer in
this codebase follows: this never removes a candidate, never lets the
deterministic `rule` floor move ahead of a real provider, and never
overrides a policy decision or an explicit user pick.

## Pricing — real rates, admin-overridable

`llm/provider_budget.py` already lets an admin set a real $/1k-token
rate per provider for budget-capping purposes — this layer checks that
rate FIRST, so a real operator-entered number always wins. Where no
admin rate is configured, `pricing_table.py`'s own best-effort public
list-price table is used instead. Ollama (local) and the deterministic
`rule` floor are always exactly $0.00 — there's no metered API call
happening for either.

**Honesty note**: the default table's rates are approximate public
list prices as of 2026-08, for the model class each hosted tier is
wired to in this deployment — not a live pricing feed, and not a claim
of billing-accurate reconciliation. Good enough for relative
cost-aware ranking and directional savings tracking; set a real
admin rate via `provider_budget.set_budget()` for anything that needs
to match an actual invoice.

## Quality, honestly

"Quality" here means two real, synchronously-available signals inside
`TierClient.chat()`: whether the call actually succeeded, and whether
the provider's own reply was truncated (`LLMResult.truncated` — an
HTTP-successful-but-cut-off reply is real evidence of lower quality
even though the call technically "worked"). This deliberately does
NOT require the deferred-finalize wiring `llm/token_optimizer.py`
needs (that module has to wait for `retried`/`out_of_scope` because a
trimmed tool schema can silently "succeed" at the HTTP level while
still picking the wrong tool) — a provider's own success/truncation
signal has no such ambiguity, so this stays fully synchronous, at the
same complexity tier as `provider_optimizer.py`.

## Savings — a real counterfactual, not a fabricated percentage

`savings_summary()` compares real total spend against what the SAME
total token volume would have cost if every call had gone to the
single priciest provider this system's own pricing table knows about,
instead. A deployment that has been routing everything to a free local
model still gets a real, honest "here's what that would have cost on
the priciest real alternative" number — not an artificial zero just
because the pricier option happened to never actually get called.

## Testing

```bash
cd orchestrator_framework_layer/akanade/backend
.venv/bin/python3 -m pytest tests/test_pricing_table.py tests/test_cost_optimizer.py tests/test_cost_api.py -v
```

31 unit tests (`pricing_table`/`cost_optimizer`, including the real
wiring into `TierClient.chat()`) + 6 REST-endpoint tests, all passing.
Full backend suite (747 tests) run clean alongside this layer — see
this session's own testing log for two real, pre-existing bugs this
work surfaced and fixed along the way:
1. `routers/admin_finetune_config.py`'s `_to_out()` did a hard
   `row["baseModel"]` dict lookup — a partially-written config row
   (as some other tests' fixtures intentionally write) 500'd the whole
   `GET .../enabled` listing for every caller. Now falls back to the
   same defaults `FinetuneConfigIn`'s own model declares.
2. `tests/test_retrieval_and_router.py`'s truncation-escalation test
   pinned `provider_optimizer.order()` to a deterministic identity
   function but didn't know about this layer's own new re-rank step
   right after it — updated to pin both.

## Example: predicting cost from the API

```bash
curl -s -X POST https://<host>/chatbot/api/v1/cost/predict \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "Backtest AAPL for the last 10 years", "tier": "realtime"}'
```

Returns one estimate per candidate provider actually available for
that tier right now, cheapest first — real inputs (chars/4 token
estimate × each provider's real/admin-configured rate), one honest
assumption (an estimated completion length, since the real completion
doesn't exist yet to measure).
