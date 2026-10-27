"""app/agent/trading_tools.py — real historical price data (mocked
Yahoo Finance chart JSON, real parsing) and the moving-average-crossover
backtest math.
"""
from __future__ import annotations

import base64
import datetime as dt

import httpx
import numpy as np
import pytest

from app.agent import native_agent, tools, trading_tools
from app.agent.state import format_result


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=self)

    def json(self):
        return self._payload


def _chart_payload(closes: list[float], start_date: str = "2024-01-01") -> dict:
    date = dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    timestamps = []
    for _ in closes:
        timestamps.append(int(date.timestamp()))
        date += dt.timedelta(days=1)
    return {"chart": {"result": [{
        "timestamp": timestamps,
        "indicators": {"quote": [{
            "open": list(closes), "high": list(closes), "low": list(closes),
            "close": list(closes), "volume": [1000] * len(closes),
        }]},
    }], "error": None}}


def _no_data_payload() -> dict:
    return {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}


def _fake_get_with(payload: dict):
    return lambda url, **kw: _FakeResponse(payload)


def test_fetch_price_history_parses_real_chart_json_shape(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload([100, 101, 99])))
    points = trading_tools.fetch_price_history("AAPL", "2024-01-01", "2024-01-03")
    assert len(points) == 3
    assert points[0].close == 100
    assert points[0].date == "2024-01-01"


def test_fetch_price_history_skips_null_padded_non_trading_days(monkeypatch):
    payload = _chart_payload([100, 101, 102])
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = None
    payload["chart"]["result"][0]["indicators"]["quote"][0]["open"][1] = None
    monkeypatch.setattr(httpx, "get", _fake_get_with(payload))
    points = trading_tools.fetch_price_history("AAPL", "2024-01-01", "2024-01-03")
    assert len(points) == 2  # the null-padded middle day is dropped, not crashed on


def test_fetch_price_history_raises_on_no_data(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get_with(_no_data_payload()))
    with pytest.raises(trading_tools.PriceDataError):
        trading_tools.fetch_price_history("NONEXISTENTXYZ", "2024-01-01", "2024-01-03")


def test_fetch_price_history_raises_on_network_failure(monkeypatch):
    def _raise(*a, **k):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(httpx, "get", _raise)
    with pytest.raises(trading_tools.PriceDataError):
        trading_tools.fetch_price_history("AAPL", "2024-01-01", "2024-01-03")


def test_run_backtest_rejects_a_non_positive_or_inverted_window(monkeypatch):
    with pytest.raises(trading_tools.PriceDataError):
        trading_tools.run_backtest("AAPL", "2024-01-01", "2024-06-01",
                                   fast_window=50, slow_window=20)


def test_run_backtest_rejects_too_little_data(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload([100] * 10)))
    with pytest.raises(trading_tools.PriceDataError):
        trading_tools.run_backtest("AAPL", "2024-01-01", "2024-01-10",
                                   fast_window=5, slow_window=20)


def test_run_backtest_on_a_steady_uptrend_beats_flat(monkeypatch):
    # A clean, steadily rising series — the fast SMA should stay above
    # the slow SMA almost the whole time once past the warm-up window,
    # so the strategy should capture most of the real move.
    closes = list(np.linspace(100, 200, 120))
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload(closes)))
    result = trading_tools.run_backtest("UPTREND", "2024-01-01", "2024-06-01",
                                        fast_window=10, slow_window=30)
    assert result["symbol"] == "UPTREND"
    assert result["total_return_pct"] > 0
    assert result["buy_and_hold_return_pct"] > 0
    assert result["n_trading_days"] > 0
    assert result["max_drawdown_pct"] <= 0  # a drawdown is always <= 0 by definition


def test_run_backtest_on_a_flat_series_has_no_trades_and_no_return(monkeypatch):
    closes = [100.0] * 120
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload(closes)))
    result = trading_tools.run_backtest("FLAT", "2024-01-01", "2024-06-01",
                                        fast_window=10, slow_window=30)
    assert result["n_trades"] == 0
    assert result["total_return_pct"] == 0.0
    assert result["sharpe_ratio"] is None  # zero-variance returns — undefined, not zero


def test_default_date_range_is_exactly_ten_years():
    start, end = trading_tools.default_date_range()
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    assert end_d - dt.timedelta(days=1) <= dt.date.today() <= end_d + dt.timedelta(days=1)
    assert start_d.year == end_d.year - 10


def test_run_backtest_defaults_to_a_ten_year_range_when_omitted(monkeypatch):
    closes = list(np.linspace(100, 200, 2600))  # ~10 years of trading days
    captured_range = {}

    def _fake_fetch(symbol, start, end):
        captured_range["start"] = start
        captured_range["end"] = end
        return [trading_tools.PricePoint(date=f"2015-01-{(i % 28) + 1:02d}", open=c, high=c,
                                         low=c, close=c, volume=1000) for i, c in enumerate(closes)]

    monkeypatch.setattr(trading_tools, "fetch_price_history", _fake_fetch)
    trading_tools.run_backtest("AAPL")
    expected_start, expected_end = trading_tools.default_date_range()
    assert captured_range == {"start": expected_start, "end": expected_end}


def test_run_backtest_reports_cagr_close_to_a_known_doubling():
    # A price series that exactly doubles over 5 years (1260 trading
    # days) has a real, hand-computable CAGR: 2**(1/5) - 1 ≈ 14.87%.
    closes = list(np.geomspace(100, 200, 1260))
    dates = [f"20{15 + i // 252:02d}-{(i % 12) + 1:02d}-01" for i in range(len(closes))]
    points = [trading_tools.PricePoint(date=d, open=c, high=c, low=c, close=c, volume=1000)
             for d, c in zip(dates, closes)]

    import httpx as httpx_module
    import matplotlib.pyplot as _plt  # noqa: F401  (already imported by trading_tools)

    class _Result:
        pass

    # Bypass the network entirely — feed fetch_price_history's OWN return
    # shape via a direct monkeypatch of the module-level function it
    # calls internally, same as the other tests in this file.
    import app.agent.trading_tools as tt
    orig = tt.fetch_price_history
    tt.fetch_price_history = lambda *a, **k: points
    try:
        result = tt.run_backtest("DOUBLER", "2015-01-01", "2020-01-01",
                                 fast_window=5, slow_window=20)
    finally:
        tt.fetch_price_history = orig
    # Buy-and-hold literally doubled — its CAGR must be very close to
    # the real, hand-computed 14.87%, not a rough guess.
    assert result["buy_and_hold_cagr_pct"] == pytest.approx(14.87, abs=0.5)


def test_run_backtest_includes_a_real_nonempty_chart(monkeypatch):
    closes = list(np.linspace(100, 200, 120))
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload(closes)))
    result = trading_tools.run_backtest("AAPL", "2024-01-01", "2024-06-01",
                                        fast_window=10, slow_window=30)
    assert result["chart_base64"]
    raw = base64.b64decode(result["chart_base64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG file signature, not a placeholder string


def test_run_backtest_never_has_lookahead_bias(monkeypatch):
    # A single, sharp one-day spike on the LAST day only. If the
    # backtest had lookahead bias, it could "see" this coming and
    # capture the move; a same-day-signal-only backtest cannot, since
    # position at day N is decided by day N-1's SMA relationship.
    closes = [100.0] * 119 + [500.0]
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload(closes)))
    result = trading_tools.run_backtest("SPIKE", "2024-01-01", "2024-06-01",
                                        fast_window=10, slow_window=30)
    # The strategy could not have been positioned for a spike that only
    # reveals itself on the final closing price — the huge buy-and-hold
    # number must NOT be matched by the strategy's own return.
    assert result["buy_and_hold_return_pct"] > 100
    assert result["total_return_pct"] < result["buy_and_hold_return_pct"]


# ── wiring into the shared tool-calling pipeline (agent/tools.py) ───────

def test_trading_tools_are_registered_in_tool_schemas():
    assert "fetch_price_history" in tools.TOOL_SCHEMAS
    assert "run_backtest" in tools.TOOL_SCHEMAS
    assert set(tools.TOOL_SCHEMAS["run_backtest"]["required"]) == {"symbol"}  # start/end default to 10y


def test_resolve_arguments_passes_trading_args_through_unresolved():
    user = {"sub": "test-user"}
    resolved, clarify = native_agent.resolve_arguments(
        user, "run_backtest", {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-06-01"})
    assert clarify is None
    assert resolved == {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-06-01"}


def test_run_backtest_never_requires_confirmation():
    user = {"sub": "test-user"}
    assert native_agent.requires_confirmation(user, "run_backtest") is False
    assert native_agent.requires_confirmation(user, "fetch_price_history") is False


def test_execute_run_backtest_returns_ok_and_metrics(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload([100] * 60)))
    user = {"sub": "test-user", "role": "Member"}
    result = tools.execute(user, "run_backtest",
                           {"symbol": "FLAT", "start": "2024-01-01", "end": "2024-03-01"})
    assert result["ok"] is True
    assert result["kind"] == "backtest"
    assert "total_return_pct" in result


def test_execute_fetch_price_history_returns_ok_and_points(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get_with(_chart_payload([100, 101, 102])))
    user = {"sub": "test-user", "role": "Member"}
    result = tools.execute(user, "fetch_price_history",
                           {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-01-03"})
    assert result["ok"] is True
    assert result["n_points"] == 3


def test_format_result_phrases_a_backtest_not_a_generic_done():
    # Regression test for a real bug this feature surfaced live: with no
    # "backtest"/"price_history" branch in format_result, a real,
    # successful tool call phrased as a bare "Done." — technically
    # correct, useless to read. Both LangGraph and native tracks share
    # this same function (app/agent/state.py), so this covers both.
    result = {"ok": True, "kind": "backtest", "symbol": "AAPL", "start": "2023-01-01",
             "end": "2024-01-01", "fast_window": 20, "slow_window": 50,
             "n_trading_days": 201, "n_trades": 2, "total_return_pct": 18.55,
             "buy_and_hold_return_pct": 25.84, "cagr_pct": 18.5, "buy_and_hold_cagr_pct": 25.8,
             "annualized_volatility_pct": 12.3, "sharpe_ratio": 1.554, "max_drawdown_pct": -10.12,
             "chart_base64": "Zm9v"}
    text = format_result(result)
    assert text != "Done."
    assert "AAPL" in text and "18.55" in text and "25.84" in text
    assert "CAGR" in text
    assert "data:image/png;base64,Zm9v" in text  # the chart is embedded, not just described


def test_format_result_phrases_a_backtest_with_no_chart_gracefully():
    # A caller that didn't ask for (or couldn't generate) a chart still
    # gets a real phrased result, not a crash on a missing key.
    result = {"ok": True, "kind": "backtest", "symbol": "AAPL", "start": "2023-01-01",
             "end": "2024-01-01", "fast_window": 20, "slow_window": 50,
             "n_trading_days": 201, "n_trades": 2, "total_return_pct": 18.55,
             "buy_and_hold_return_pct": 25.84, "cagr_pct": None, "buy_and_hold_cagr_pct": None,
             "annualized_volatility_pct": 12.3, "sharpe_ratio": None, "max_drawdown_pct": -10.12,
             "chart_base64": None}
    text = format_result(result)
    assert text != "Done."
    assert "data:image" not in text


def test_format_result_phrases_price_history_not_a_generic_done():
    result = {"ok": True, "kind": "price_history", "symbol": "AAPL", "n_points": 3,
             "points": [{"date": "2024-01-03", "open": 1, "high": 1, "low": 1,
                        "close": 125.0, "volume": 1000}]}
    text = format_result(result)
    assert text != "Done."
    assert "AAPL" in text and "125.0" in text


def test_execute_reports_a_clean_error_not_a_crash_on_bad_symbol(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get_with(_no_data_payload()))
    user = {"sub": "test-user", "role": "Member"}
    result = tools.execute(user, "run_backtest",
                           {"symbol": "NONEXISTENTXYZ", "start": "2024-01-01", "end": "2024-06-01"})
    assert result["ok"] is False
    assert "error" in result
