# One implementation, three platforms — how to test it

**The architecture point this doc exists to prove**: PM Tool, Trading
Agent, and Train Algo AI are implemented exactly **once**, in the
backend's shared tool-calling pipeline behind `POST /api/chat/message`
(`app/agent/tools.py`, `app/agent/trading_tools.py`,
`app/agent/peft_training_tool.py`). The web app, the mobile app, and
the VS Code extension's PM/chat surface all call that **same**
endpoint — there is no separate per-platform reimplementation of any
agent's logic. Coding Agent is the one exception: it also has a
sandboxed/local-execution mode behind `POST /api/v1/orchestration/execute`,
used only by the VS Code extension when it needs to run generated code
on the developer's own machine.

Because the implementation is shared, **the same prompt list below
is valid on all three platforms** — you're testing one backend
behavior each time, just through three different front doors.

## Prompt list — run each of these on Web, VS Code, and Mobile

### PM Tool
1. `Who knows Python?`
2. `Tell me about <a real user's name>`
3. `Create a task called "Test unified agents" in <a real project>`

### Trading Agent
4. `Backtest AAPL for the last 10 years`
   — expect: a reply with CAGR, buy-and-hold CAGR, annualized
   volatility, Sharpe, max drawdown, and an inline chart image.
5. `Fetch the price history for BTC-USD from 2024-01-01 to 2024-06-01`
6. `Run a 10/30 day moving average crossover on MSFT`

### Train Algo AI (chat-triggered)
7. `Train the coding model on this: always write docstrings for public functions`
   — expect: a reply confirming characters logged and which domain.
8. `Train Algo AI on this link for the trading domain: <a real URL>`

### Coding Agent
9. `Write a Python function that returns the nth Fibonacci number`
10. (VS Code only, via the "+" → *Attach file…* / *Ask about selection*
    flow) — ask the agent to review or modify a real open file.

### Train Algo AI button (Web + VS Code + Mobile)
11. Open the **Train Algo AI** button/menu item on each platform,
    submit a short pasted text under a test domain (e.g. `qa-test`),
    and confirm the same success message shape on all three
    (`chars logged`, `rag_chunks_indexed`, domain name).

### Token & Cost Optimizer
12. `POST /api/v1/cost/predict` with `{"message": "Backtest AAPL for
    the last 10 years", "tier": "realtime"}` — expect one estimate per
    available model, cheapest first, with a real (nonzero) predicted
    cost for any hosted provider and exactly `0.0` for Ollama.
13. Open the **Token & Cost Optimizer** dashboard (Admin/Senior
    Manager) — after some real chat traffic, confirm per-model
    calls/cost/latency rows are populated and the savings summary
    shows a nonzero `estimated_savings_usd` once at least one call has
    gone to a provider cheaper than this system's priciest known rate.
    See `README_TOKEN_COST_OPTIMIZER.md` for the full layer.

## What "pass" looks like

- The **reply text itself** differs turn to turn (different LLM
  providers may answer), but the **tool that got called** and the
  **shape of the result** (chart present for Trading Agent prompts,
  domain/chars-logged for Train Algo AI prompts) should be consistent
  across all three platforms for the same prompt.
- If a small local model answers a Trading Agent prompt with a plain
  code suggestion instead of calling the real tool, that's a known,
  live-confirmed provider-quality gap (not a bug in this pipeline) —
  retry once; a stronger provider (Groq) reliably calls the tool.
- `GET /api/admin/peft-jobs` (Admin token) should show `running` grow
  to 1 during a real nightly training run and drop back to 0 after.

## Where to actually run this

- **Web**: the chat page, any account.
- **VS Code**: the PM Tool tab in the extension's chat panel (Trading
  Agent and Train Algo AI both reach through this exact tab, since
  it's the same `/api/chat/message` call).
- **Mobile**: the AI Assistant screen.

Running this same list on all three and comparing outcomes is the
actual regression test for "implemented once, used everywhere" — a
divergence between platforms on the same prompt means one client is
calling something other than the shared endpoint, which should never
happen by design.
