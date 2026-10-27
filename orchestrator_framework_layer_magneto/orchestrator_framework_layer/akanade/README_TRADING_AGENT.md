# Trading Agent — real backtests, in the same chat pipeline as everything else

A third capability alongside the PM Tool and Coding Agent: real
historical price data and a backtest, reachable from the exact same
chat UI (web, mobile, VS Code) that's already deployed everywhere — no
separate agent-picker, no new endpoint to wire up client-side.

## Why this shape, not a separate registered "agent"

ALGO AKANADE has a second, separate multi-agent framework
(`app/framework/agents/coding_agent.py` and friends, reached via
`POST /api/v1/orchestration/execute`) — but investigation found neither
the web frontend nor the mobile app call that endpoint at all; only the
VS Code extension does. Wiring a whole new UI surface to reach it (or
migrating the existing, working PM chat pipeline onto it) is a much
bigger, riskier change than this actually needs.

Instead, the Trading Agent's tools are added to the **same tool-calling
pipeline** the PM Tool already uses (`app/agent/tools.py`, shared by the
native/LangGraph/ADK tracks) — the identical mechanism an MCP tool like
"fetch" already uses. That means it's automatically available on web,
mobile, and anywhere else `/api/chat/message` is already reachable,
today, with zero new frontend/mobile code.

## What it does (`app/agent/trading_tools.py`)

- **`fetch_price_history(symbol, start, end)`** — real daily
  OHLCV data from Yahoo Finance's free, keyless chart JSON endpoint
  (`https://query1.finance.yahoo.com/v8/finance/chart/<symbol>`) — the
  *one* hardcoded external domain this tool ever reaches (no arbitrary
  URL parameter, so this introduces no SSRF-shaped hole). `symbol` is a
  real Yahoo Finance ticker, e.g. `"AAPL"`, `"BTC-USD"`. Stooq's own CSV
  endpoint was the original choice — genuinely keyless and scriptable
  historically — but live testing found it now sits behind a
  JavaScript proof-of-work bot challenge a plain HTTP client can't
  satisfy; rather than automate around that, this switched to Yahoo's
  endpoint, confirmed working live.
- **`run_backtest(symbol, start, end, fast_window=20, slow_window=50)`**
  — a moving-average-crossover backtest: long whenever the fast SMA is
  above the slow SMA, flat otherwise. Fully vectorized (`numpy`), no
  lookahead bias (a day's position is decided by the *previous* day's
  signal, never its own close) and a small relative-epsilon guard on
  the crossover comparison (two SMAs over equal-but-differently-sized
  windows can differ by ~1e-13 from floating-point summation order
  alone on a flat/near-flat market — without the guard that noise
  manufactures phantom trades). Returns `total_return_pct`,
  `buy_and_hold_return_pct` (the honest baseline to compare against),
  `sharpe_ratio` (`None` on zero-variance returns, never a fake zero),
  `max_drawdown_pct`, and `n_trades`.

Both are **read-only** — no confirmation gate (`native_agent.
requires_confirmation` returns `False` for both), same treatment any
other read gets. Neither takes a PM entity argument, so
`native_agent.resolve_arguments` passes their arguments through
unresolved, exactly like it already does for MCP tools.

## "Combining with the Coding Agent"

There's no special two-agent hand-off endpoint for this pipeline (that
would need the orchestration framework mentioned above). In practice,
the model itself already writes the backtest's own analysis and any
requested Python alongside the tool's real numeric result in one reply
— ask for both in one message (*"backtest a 20/50 crossover on AAPL for
2024 and show me the Python that would reproduce it"*) and the model's
free-text portion covers the "Coding Agent" half, grounded by the tool
call's real numbers rather than invented ones. For a strategy that
needs genuine local file/workspace access to iterate on, the VS Code
extension's real Coding Agent (`app/framework/agents/coding_agent.py`,
`README_POLICY_GATE_AND_CODING_AGENT.md`) is still the right tool.

## Safety notes

- Backtests are read-only analysis — no real order is ever placed, no
  brokerage/exchange account is ever touched. This is explicitly a
  research/analysis tool, not a live-trading system.
- The built-in strategy is intentionally narrow (one crossover rule) —
  anything more custom is what the Coding Agent's own sandboxed
  `run_code` tool is for; this module never executes model-generated
  code itself.

## Verification

`.venv/bin/python3 -m pytest tests/test_trading_tools.py -q` — price
parsing, backtest math (uptrend/flat/no-lookahead-bias cases), and the
tool-calling wiring (schema registration, argument passthrough,
confirmation gating, `tools.execute()` dispatch) all covered.
