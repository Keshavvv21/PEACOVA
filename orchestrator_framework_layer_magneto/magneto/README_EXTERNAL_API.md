# External API — integrating a third-party app with ALGO AKANADE

This is the real, currently-shipping REST surface any external
application (a React app, an internal tool, a script in an ERP's
scripting layer) can call. Everything below is a plain HTTPS JSON API
— no SDK is required to use it; a generated OpenAPI schema is
available at `GET /openapi.json` on the chatbot service
(`ak-chatbot`, e.g. `https://<your-host>/chatbot/openapi.json` behind
the deployed nginx) for any codegen tool (openapi-generator, Postman,
`orval`) to build a typed client from directly.

**Honesty note**: what follows is the same REST surface the web app,
mobile app, and VS Code extension all call — the one real integration
path for a third-party system to talk to ALGO AKANADE. A working Tally
plugin now exists on top of this exact surface —
[`integrations/tally-connector`](../../integrations/tally-connector) —
plus the plain client module it's built on,
[`integrations/algo-akanade-connector`](../../integrations/algo-akanade-connector),
which any React or React Native app (or another ERP's scripting layer)
can import directly. Neither is published to a package registry; both
are used as local/`file:` dependencies today. The Tally plugin's XML
parsing was tested against realistic sample data, not a live Tally
instance — see that package's own README for exactly what is and isn't
verified.

## 1. Authentication

```
POST /api/auth/login
Content-Type: application/json
{ "email": "you@company.com", "password": "..." }

-> { "access_token": "<JWT>", ... }
```

Send the token on every subsequent call:

```
Authorization: Bearer <access_token>
```

The JWT expires; `401` means re-login. There is no refresh-token flow
today — a script should catch `401` and call `/api/auth/login` again.

### Roles (RBAC)

Every protected endpoint is gated by `require_role(minimum)` against
one of four ordered roles: `Member < Manager < Admin < Executive`. A
call with an insufficient role gets a plain `403`. Admin-only
endpoints below are marked explicitly; everything else needs only a
valid token (any role).

## 2. Talk to an agent — the one endpoint every platform actually uses

```
POST /api/chat/message
{ "message": "your prompt", "session_id": "any-string-you-control", "provider": null }

-> { "reply": "...", "pending_confirmation": false, "confirm_prompt": null, "provider": "groq-llama-3.3-70b" }
```

This is the **same endpoint** the web app, the mobile app, and (via
its own tools) VS Code all call. It already reaches:
- **PM Tool** — task/project/sprint lookups, natural language.
- **Trading Agent** — "Backtest AAPL for the last 10 years" returns a
  real reply with CAGR/volatility/Sharpe and an embedded PNG chart
  (base64 `data:image/png;base64,...` inside the `reply` markdown —
  render it as an `<img>` in whatever your caller displays).
- **Train Algo AI** — "Train the trading model on this: <text>" logs a
  real training example; see §4.

`provider`: omit or `null` for auto-routing; `"hybrid"` prefers a local
SLM for easy turns; a specific provider id (from
`GET /api/chat/providers`) pins one model.

If `pending_confirmation` is `true`, send the literal string `"yes"`
or `"no"` as the next message on the same `session_id` — there is no
separate confirm endpoint.

`GET /api/chat/history?session_id=...` / `DELETE /api/chat/history?session_id=...`
manage a session's transcript.

## 3. Coding Agent — sandboxed code generation/execution

```
POST /api/v1/orchestration/execute
{ "task": "write a function that ...", "session_id": "...",
  "context": { "task_type": "code_generation", "required_capabilities": ["code_generation"] },
  "client_tools": false }

-> { "selected_agents": ["coding-agent"], "result": { "text": "...", "ok": true, ... } }
```

Set `"client_tools": true` only if your integration can execute a
returned `pending_tool_call` locally and report the result back via
`POST /api/v1/orchestration/tool_result` — this is the VS Code
extension's own "run the sandbox on the developer's machine" mode; a
server-side integration should leave this `false` and let the backend
run its own sandbox.

## 4. Train Algo AI — feed a domain model real training data

```
POST /api/peft-training/link
{ "domain": "trading", "topic": "RSI", "url": "https://..." }

POST /api/peft-training/upload   (multipart/form-data)
  domain=trading  topic=RSI  file=<.txt/.md/.csv/.pdf, 5MB max>
```

Both: any authenticated user, no Admin gate. The content is indexed
into that domain's RAG store immediately (searchable right away) and
logged as a real ground-truth example for the **next** scheduled
nightly fine-tune — never trained synchronously inside the request.

`GET /api/admin/peft-jobs` (**Admin**) — `{jobs: [...], running: [...]}`,
whether a domain is training right now.

## 5. Token & Cost Optimizer — predict cost before you spend it

```
POST /api/v1/cost/predict
{ "message": "your prompt", "tier": "realtime" }

-> { "tier": "realtime", "estimates": [
     { "provider": "ollama", "predicted_prompt_tokens": 12,
       "predicted_completion_tokens": 6, "predicted_cost_usd": 0.0 },
     { "provider": "luna", "predicted_prompt_tokens": 12,
       "predicted_completion_tokens": 6, "predicted_cost_usd": 0.00009 }
   ] }
```

Any authenticated user. One estimate per model actually available for
`tier` right now, cheapest first — real token estimate × each
provider's real (or admin-configured) rate, before any real call is
made. `tier` is one of `realtime` / `delay_ok` / `bulk`; omit for
`realtime`. See `README_TOKEN_COST_OPTIMIZER.md` for the full layer —
model selection is cost-aware automatically on every real
`/api/chat/message` call too, no extra parameter needed.

`GET /api/v1/metrics/cost` (**Admin**) — real tracked cost/token/
latency/quality per model, plus a whole-system savings estimate.

## 6. Everything else, by area

| Area | Base path | Notes |
|---|---|---|
| PM data (projects/tasks/sprints) | `/api/*` | The mini-PM REST API `pm.py` exposes directly, if you need structured data rather than chat. |
| Provider config | `/api/admin/providers` | **Admin**. Add/manage LLM provider API keys. |
| Fine-tune config | `/api/admin/finetune-config` | **Admin**. Per-domain nightly training settings, including `scrape_topics`. |
| SLM versions / eval reports | `/api/admin/slm-versions`, `/api/admin/slm-eval-reports` | **Admin**. Read-only history. |
| Security findings | `/api/admin/security/findings`, `/api/admin/security/events` | **Admin**. Real scan results + flagged prompts. |
| RL/routing metrics | `/api/v1/metrics`, `/api/v1/metrics/routing` | **Admin**. Bandit weights, per-domain/provider win rates. |
| RAG | `/api/v1/rag/metrics`, `/api/v1/rag/*` | Retrieval stats and document ingestion. |
| Agent registry | `/api/v1/agents` | List every registered agent and its declared tools/capabilities. |

Full request/response schemas for all of the above: `GET /openapi.json`.

## 7. Rate limits, timeouts, errors

- No explicit rate limiting is enforced today — a high-volume
  integration should still self-throttle; nothing here is designed for
  bulk/batch traffic.
- A chat turn that triggers a slow local model can take minutes; set a
  generous client timeout (the deployed nginx proxy is configured for
  200s) rather than retrying aggressively.
- Errors are plain HTTP status + `{"detail": "..."}` (FastAPI's
  default `HTTPException` shape).

## 8. A minimal integration, in curl

```bash
TOKEN=$(curl -s -X POST https://<host>/chatbot/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@company.com","password":"..."}' | python3 -c "import json,sys;print(json.load(sys.stdin)['access_token'])")

curl -s -X POST https://<host>/chatbot/api/chat/message \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message":"Backtest AAPL for the last 10 years","session_id":"integration-test-1"}'
```

That one call is the entire integration surface a React app, a Tally
add-on script, or any other ERP scripting layer needs to reach every
agent ALGO AKANADE ships — the same call the official web/mobile/VS Code
clients make.
