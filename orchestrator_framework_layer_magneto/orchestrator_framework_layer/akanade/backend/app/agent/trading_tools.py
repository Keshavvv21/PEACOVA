"""Trading Agent tools — real historical price data (Yahoo Finance's
free, keyless chart JSON endpoint — the one hardcoded external domain
these tools ever reach, no arbitrary URL parameter, so this introduces
no SSRF-shaped hole) and a simple, safe vectorized backtest on top of
it.

Stooq's own CSV endpoint (`stooq.com/q/d/l/`) was the original choice
here — genuinely keyless and scriptable historically, but live testing
found it now sits behind a JavaScript proof-of-work bot challenge
(`/__verify`, a SHA-256 grind + fetch from the browser) that a plain
HTTP client can't satisfy. Rather than automate around that challenge,
this uses Yahoo Finance's undocumented-but-widely-used chart API
instead (`query1.finance.yahoo.com/v8/finance/chart/<symbol>`) —
confirmed live, keyless, and a plain `User-Agent` header is all it
needs.

Same "translation layer only" shape as agent/tools.py's PM tools: one
function per tool, no PM-entity resolution needed (symbol/dates are
literal values — native_agent.py's resolve_arguments bypasses entity
resolution for these two tool names the same way it already does for
MCP tools).

Deliberately narrow scope for the backtest itself — a single built-in
strategy (moving-average crossover), not an arbitrary-strategy engine.
Anything more custom than that is exactly what the Coding Agent's own
sandboxed run_code tool is for (see coding_agent.py) — this module
never executes model-generated code itself.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
from dataclasses import dataclass

import httpx
import numpy as np

# Agg BEFORE any pyplot import — the one true way to guarantee no
# display/GUI backend is ever probed for on a headless server; importing
# pyplot first and switching backends after is a well-known source of
# "no display name and no $DISPLAY" crashes in exactly this deployment
# shape (a container with no X server).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use())

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_REQUEST_TIMEOUT_S = 10.0
_HEADERS = {"User-Agent": "Mozilla/5.0 (Algo-Akanade-Trading-Agent/1.0)"}
_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_LOOKBACK_YEARS = 10


def _to_unix(date_str: str) -> int:
    return int(dt.datetime.strptime(date_str, "%Y-%m-%d")
              .replace(tzinfo=dt.timezone.utc).timestamp())


def default_date_range() -> tuple[str, str]:
    """"The last N years" — the default a trader actually means by
    "backtest this strategy" with no range named, per the DEFAULT_
    LOOKBACK_YEARS constant above. A real calendar date, not an
    approximation — end is today (UTC), start is exactly that many
    years earlier."""
    end = dt.datetime.now(dt.timezone.utc).date()
    try:
        start = end.replace(year=end.year - _DEFAULT_LOOKBACK_YEARS)
    except ValueError:  # Feb 29 with no leap year N years back
        start = end.replace(month=2, day=28, year=end.year - _DEFAULT_LOOKBACK_YEARS)
    return start.isoformat(), end.isoformat()


@dataclass
class PricePoint:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceDataError(Exception):
    """Raised for anything that stops real backtest math from running —
    a bad/unknown symbol, an empty range, or a network failure. Always
    carries a human-readable message a chat reply can show as-is."""


def fetch_price_history(symbol: str, start: str, end: str) -> list[PricePoint]:
    """`symbol` is a real ticker as Yahoo Finance itself recognizes it
    (e.g. "AAPL", "BTC-USD") — no validation beyond what the API itself
    accepts; an unknown symbol comes back as a clean `chart.error`,
    turned into a PriceDataError below rather than a confusing empty
    list."""
    try:
        r = httpx.get(_CHART_URL.format(symbol=symbol.upper()),
                      params={"period1": _to_unix(start), "period2": _to_unix(end), "interval": "1d"},
                      headers=_HEADERS, timeout=_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise PriceDataError(f"Could not reach the price data source: {e}") from e

    chart = data.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        reason = (chart.get("error") or {}).get("description", "no data returned")
        raise PriceDataError(f"No price data found for {symbol!r} between {start} and {end}: {reason}")

    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]

    points: list[PricePoint] = []
    for i, ts in enumerate(timestamps):
        o, h, l, c, v = (quote["open"][i], quote["high"][i], quote["low"][i],
                        quote["close"][i], quote["volume"][i])
        if None in (o, h, l, c):  # a non-trading day inside the range — Yahoo pads with nulls
            continue
        points.append(PricePoint(
            date=dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d"),
            open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v or 0),
        ))
    if not points:
        raise PriceDataError(f"No usable price rows for {symbol!r} between {start} and {end}.")
    return points


def _max_drawdown_pct(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(drawdowns.min() * 100)


def _sharpe_ratio(daily_returns: np.ndarray) -> float | None:
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return None
    # Annualized, 252 trading days — the standard convention, not a claim
    # this backtest accounts for risk-free rate or any real trading cost.
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


def _annualized_volatility_pct(daily_returns: np.ndarray) -> float:
    return float(daily_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR) * 100)


def _cagr_pct(total_return_multiple: float, n_trading_days: int) -> float | None:
    """Compound Annual Growth Rate — the standard "what steady annual
    rate would have produced this same total return" figure every
    trader expects alongside a raw total-return percentage, especially
    over a multi-year backtest where total return alone is hard to
    compare across different date ranges. None for a total wipeout
    (can't take a root of a non-positive base) or under a year of data
    (not enough history for "annual" to mean anything)."""
    years = n_trading_days / _TRADING_DAYS_PER_YEAR
    if total_return_multiple <= 0 or years < 1:
        return None
    return float((total_return_multiple ** (1 / years) - 1) * 100)


def _yearly_returns_pct(dates: list[str], strategy_equity: np.ndarray) -> list[tuple[str, float]]:
    """The strategy's own equity curve, sliced into real calendar years
    and turned into one "how did this specific year go" percentage per
    year — the bar-chart panel below plots exactly this. `dates[i]` and
    `strategy_equity[i]` are the same length and index-aligned (see
    run_backtest's own aligned_dates/equity_curve construction) — a
    year's return is its last day's cumulative equity divided by the
    equity right before its first day (the previous trading day's
    close), or, for a year with no "day before" in range (the backtest's
    very first calendar year), the equity curve's own starting 1.0
    baseline. A year only appears once at least one trading day of it
    exists in the range; a partial first/last year (the range doesn't
    start/end on Jan 1/Dec 31) still shows, correctly reflecting only
    the days actually in range, never padded to a fake full year."""
    years = [d[:4] for d in dates]
    out: list[tuple[str, float]] = []
    for year in dict.fromkeys(years):  # de-duplicated, first-seen order == chronological here
        idx = [i for i, y in enumerate(years) if y == year]
        first_i, last_i = idx[0], idx[-1]
        baseline_i = first_i - 1 if first_i > 0 else first_i
        start_equity = strategy_equity[baseline_i]
        end_equity = strategy_equity[last_i]
        out.append((year, float((end_equity / start_equity - 1) * 100)))
    return out


# A small, deliberately calm palette — real financial-app conventions
# (green = up, red = down) rather than matplotlib's default loud blue/
# orange, so the bar chart reads at a glance the way a trader already
# expects, on both a light and dark chat surface (the PNG itself always
# renders on a white canvas — no theme-awareness needed there, only in
# the CHOICE of colors already being ones that hold up on either).
_COLOR_STRATEGY = "#1a56db"
_COLOR_BUY_HOLD = "#94a3b8"
_COLOR_UP = "#16a34a"
_COLOR_DOWN = "#dc2626"
_COLOR_GRID = "#e5e7eb"


def _render_backtest_chart(dates: list[str], strategy_equity: np.ndarray,
                           buy_hold_equity: np.ndarray, *, symbol: str,
                           fast_window: int, slow_window: int) -> str:
    """Returns a base64-encoded PNG — embedded directly in the chat
    reply as a markdown image (see state.py's format_result), so the
    graph shows up inline in whichever chat surface is already
    rendering markdown (web/mobile/VS Code), no new UI or endpoint
    needed. Two panels, one figure: the full equity curve (strategy vs.
    buy-and-hold, the shape of the whole backtest) on top, and a
    year-by-year return bar chart (the concrete "how did each year go"
    trades actually get asked for) below — same real numbers the reply
    text already reports, just visualized instead of only prose."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    yearly = _yearly_returns_pct(dates, strategy_equity)
    fig, (ax_curve, ax_bars) = plt.subplots(
        2, 1, figsize=(8.5, 7.5), dpi=130, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor("#ffffff")

    # --- Top: cumulative equity curve ---
    x = range(len(strategy_equity))
    ax_curve.plot(x, (strategy_equity - 1) * 100, label=f"{fast_window}/{slow_window}-day crossover",
                 color=_COLOR_STRATEGY, linewidth=2.0)
    ax_curve.plot(x, (buy_hold_equity - 1) * 100, label="Buy & hold",
                 color=_COLOR_BUY_HOLD, linewidth=1.4, linestyle="--")
    ax_curve.axhline(0, color=_COLOR_GRID, linewidth=0.9, zorder=0)
    ax_curve.set_title(f"{symbol} — {dates[0]} to {dates[-1]}", fontsize=13, fontweight="bold", pad=10)
    ax_curve.set_ylabel("Cumulative return (%)", fontsize=10)
    ax_curve.set_xticks([0, len(x) - 1])
    ax_curve.set_xticklabels([dates[0], dates[-1]], fontsize=9)
    ax_curve.legend(loc="upper left", frameon=False, fontsize=9)
    ax_curve.grid(axis="y", color=_COLOR_GRID, linewidth=0.7, zorder=0)
    ax_curve.set_axisbelow(True)

    # --- Bottom: year-by-year strategy return ---
    years = [y for y, _ in yearly]
    values = [v for _, v in yearly]
    bar_colors = [_COLOR_UP if v >= 0 else _COLOR_DOWN for v in values]
    bars = ax_bars.bar(years, values, color=bar_colors, width=0.6, zorder=3)
    ax_bars.axhline(0, color="#9ca3af", linewidth=1.0, zorder=2)
    ax_bars.set_title(f"{symbol} strategy return by year", fontsize=11, fontweight="bold", pad=8)
    ax_bars.set_ylabel("Return (%)", fontsize=10)
    ax_bars.grid(axis="y", color=_COLOR_GRID, linewidth=0.7, zorder=0)
    ax_bars.set_axisbelow(True)
    ax_bars.tick_params(axis="x", labelsize=9)
    # A value label on every bar — "beautiful" here means readable at a
    # glance without hovering/zooming, not just colorful; a label makes
    # the exact number available even when the bar itself is too short
    # to eyeball precisely.
    span = max(values) - min(values) if len(values) > 1 else abs(values[0]) or 1.0
    pad = span * 0.03
    for bar, v in zip(bars, values):
        offset = pad if v >= 0 else -pad
        va = "bottom" if v >= 0 else "top"
        ax_bars.text(bar.get_x() + bar.get_width() / 2, v + offset, f"{v:+.1f}%",
                    ha="center", va=va, fontsize=8.5, color="#1f2937", fontweight="bold")

    for ax in (ax_curve, ax_bars):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5db")
        ax.spines["bottom"].set_color("#d1d5db")

    fig.tight_layout(pad=1.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_backtest(symbol: str, start: str | None = None, end: str | None = None, *,
                 fast_window: int = 20, slow_window: int = 50) -> dict:
    """A moving-average-crossover backtest: long whenever the fast SMA is
    above the slow SMA, flat otherwise — the one built-in strategy this
    tool runs (see module docstring on why this isn't an arbitrary-
    strategy engine). Fully vectorized with numpy, no model-generated
    code ever executed here. `start`/`end` default to the last
    DEFAULT_LOOKBACK_YEARS (10) — "backtest a strategy for the last 10
    years" needs no explicit dates named."""
    if not start or not end:
        default_start, default_end = default_date_range()
        start, end = start or default_start, end or default_end
    if fast_window <= 0 or slow_window <= 0 or fast_window >= slow_window:
        raise PriceDataError("fast_window must be a smaller positive number than slow_window.")

    points = fetch_price_history(symbol, start, end)
    closes = np.array([p.close for p in points], dtype=float)
    if len(closes) <= slow_window:
        raise PriceDataError(
            f"Only {len(closes)} trading days of data for {symbol!r} in that range — "
            f"need more than slow_window={slow_window} to compute a crossover.")

    fast_sma = np.convolve(closes, np.ones(fast_window) / fast_window, mode="valid")
    slow_sma = np.convolve(closes, np.ones(slow_window) / slow_window, mode="valid")
    # Align both SMAs to the SAME trailing window (slow_window is always
    # the longer one, so it determines how many leading closes are unusable).
    offset = slow_window - fast_window
    fast_sma = fast_sma[offset:]
    aligned_closes = closes[slow_window - 1:]
    aligned_dates = [p.date for p in points[slow_window - 1:]]

    # A plain `>` here is genuinely wrong on a flat/near-flat market: two
    # SMAs computed from equal-but-differently-sized windows can differ
    # by ~1e-13 purely from np.convolve's floating-point summation order,
    # not any real signal — spuriously flipping "long" on and off on
    # data that never actually crossed. A small relative tolerance avoids
    # manufacturing trades out of floating-point noise.
    position = ((fast_sma - slow_sma) > 1e-9 * np.abs(slow_sma)).astype(float)  # 1.0 = long, 0.0 = flat
    daily_returns = np.diff(aligned_closes) / aligned_closes[:-1]
    # A day's return only counts if we were ALREADY positioned going into
    # it (position at the START of the day, i.e. yesterday's signal) —
    # no lookahead bias.
    strategy_returns = daily_returns * position[:-1]

    equity_curve = np.concatenate([[1.0], np.cumprod(1 + strategy_returns)])
    buy_hold_curve = aligned_closes / aligned_closes[0]
    total_return_pct = float((equity_curve[-1] - 1.0) * 100)
    buy_and_hold_pct = float((buy_hold_curve[-1] - 1.0) * 100)
    trades = int(np.sum(np.abs(np.diff(position)) > 0))
    n_days = len(aligned_closes)

    chart_base64 = _render_backtest_chart(
        aligned_dates, equity_curve, buy_hold_curve, symbol=symbol,
        fast_window=fast_window, slow_window=slow_window)

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "n_trading_days": n_days,
        "n_trades": trades,
        "total_return_pct": round(total_return_pct, 2),
        "buy_and_hold_return_pct": round(buy_and_hold_pct, 2),
        "cagr_pct": round(c, 2) if (c := _cagr_pct(equity_curve[-1], n_days)) is not None else None,
        "buy_and_hold_cagr_pct": round(c, 2) if (c := _cagr_pct(buy_hold_curve[-1], n_days)) is not None else None,
        "annualized_volatility_pct": round(_annualized_volatility_pct(strategy_returns), 2),
        "sharpe_ratio": round(s, 3) if (s := _sharpe_ratio(strategy_returns)) is not None else None,
        "max_drawdown_pct": round(_max_drawdown_pct(equity_curve), 2),
        "chart_base64": chart_base64,
    }
