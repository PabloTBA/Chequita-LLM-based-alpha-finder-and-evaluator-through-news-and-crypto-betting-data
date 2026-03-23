"""
Tests for ExecutionAdvisor
==========================
Cycle 1  — output contract
Cycle 2  — only active signals included
Cycle 3  — spread & slippage calculation
Cycle 4  — market impact classification
Cycle 5  — portfolio risk aggregation
Cycle 6  — empty active signals safe fallback
Cycle 7  — yfinance fetch failure graceful degradation
"""

import pytest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution_advisor import ExecutionAdvisor


# ── fixtures ────────────────────────────────────────────────────────────────

def _strategy(ticker="AAPL", signal_active=True, dollar_risk=500.0,
              position_size=100, entry_price=150.0, stop_price=145.0,
              current_atr=3.0, adv=1_000_000):
    setup = {
        "entry_price":   entry_price,
        "stop_price":    stop_price,
        "stop_dist":     entry_price - stop_price,
        "position_size": position_size,
        "dollar_risk":   dollar_risk,
        "current_atr":   current_atr,
        "current_ma":    148.0,
        "target":        None,
    }
    return {
        "ticker":          ticker,
        "strategy":        "Momentum",
        "adjusted_params": {"lookback": 20, "atr_mult": 2.0},
        "current_signal": {
            "signal_active": signal_active,
            "details":       "Breakout confirmed" if signal_active else "No signal",
            "setup":         setup if signal_active else None,
        },
        "_adv":            adv,   # injected by tests for market-impact calculations
    }


def _mock_ticker(bid=149.90, ask=150.10):
    """Returns a mock yfinance Ticker whose .info has bid/ask."""
    class _Info(dict):
        pass

    class _MockTicker:
        info = {"bid": bid, "ask": ask, "averageVolume": 1_000_000}

    return _MockTicker()


# ── Cycle 1: output contract ─────────────────────────────────────────────────

class TestOutputContract:
    def test_returns_dict_with_required_keys(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([_strategy()])

        assert isinstance(result, dict)
        assert "active_signals" in result
        assert "inactive_count" in result
        assert "portfolio_risk" in result
        assert "warnings"       in result

    def test_active_signals_is_list(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([_strategy()])

        assert isinstance(result["active_signals"], list)

    def test_portfolio_risk_is_dict(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([_strategy()])

        assert isinstance(result["portfolio_risk"], dict)

    def test_warnings_is_list(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([_strategy()])

        assert isinstance(result["warnings"], list)


# ── Cycle 2: only active signals included ────────────────────────────────────

class TestActiveSignalsFilter:
    def test_inactive_strategy_not_in_active_signals(self):
        strategies = [
            _strategy("AAPL", signal_active=True),
            _strategy("MSFT", signal_active=False),
        ]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise(strategies)

        tickers = [s["ticker"] for s in result["active_signals"]]
        assert "AAPL" in tickers
        assert "MSFT" not in tickers

    def test_inactive_count_reflects_inactive_strategies(self):
        strategies = [
            _strategy("AAPL", signal_active=True),
            _strategy("MSFT", signal_active=False),
            _strategy("GOOG", signal_active=False),
        ]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise(strategies)

        assert result["inactive_count"] == 2

    def test_all_active_strategies_appear(self):
        strategies = [
            _strategy("AAPL", signal_active=True),
            _strategy("MSFT", signal_active=True),
        ]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise(strategies)

        tickers = [s["ticker"] for s in result["active_signals"]]
        assert set(tickers) == {"AAPL", "MSFT"}


# ── Cycle 3: spread & slippage calculation ───────────────────────────────────

class TestSpreadAndSlippage:
    def test_active_signal_has_spread_key(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker(bid=149.90, ask=150.10)):
            result = advisor.advise([_strategy(position_size=100)])

        sig = result["active_signals"][0]
        assert "spread" in sig

    def test_spread_equals_ask_minus_bid(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker(bid=149.90, ask=150.10)):
            result = advisor.advise([_strategy(position_size=100)])

        sig = result["active_signals"][0]
        assert abs(sig["spread"] - 0.20) < 0.001

    def test_slippage_equals_half_spread_times_position(self):
        """Total slippage = (ask - bid) / 2 × position_size."""
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker(bid=149.80, ask=150.20)):
            result = advisor.advise([_strategy(position_size=200)])

        sig = result["active_signals"][0]
        # spread = 0.40, slippage per share = 0.20, total = 0.20 * 200 = 40.0
        assert abs(sig["slippage_total"] - 40.0) < 0.01

    def test_adjusted_risk_equals_dollar_risk_plus_slippage(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker(bid=149.80, ask=150.20)):
            result = advisor.advise([_strategy(dollar_risk=500.0, position_size=200)])

        sig = result["active_signals"][0]
        # slippage = 40.0, adjusted_risk = 500 + 40 = 540.0
        assert abs(sig["adjusted_risk"] - 540.0) < 0.01


# ── Cycle 4: market impact classification ────────────────────────────────────

class TestMarketImpact:
    def test_small_order_is_negligible(self):
        """position_size = 100 shares, adv = 1_000_000 → 0.01% → negligible."""
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        strat = _strategy(position_size=100, adv=1_000_000)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([strat])

        assert result["active_signals"][0]["market_impact"] == "negligible"

    def test_moderate_order_classified_correctly(self):
        """position_size = 20_000 shares, adv = 1_000_000 → 2% → moderate."""
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        strat = _strategy(position_size=20_000, adv=1_000_000)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([strat])

        assert result["active_signals"][0]["market_impact"] == "moderate"

    def test_significant_order_classified_correctly(self):
        """position_size = 60_000 shares, adv = 1_000_000 → 6% → significant."""
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        strat = _strategy(position_size=60_000, adv=1_000_000)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([strat])

        assert result["active_signals"][0]["market_impact"] == "significant"

    def test_significant_impact_adds_warning(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        strat = _strategy("TINY", position_size=60_000, adv=1_000_000)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise([strat])

        assert any("TINY" in w for w in result["warnings"])


# ── Cycle 5: portfolio risk aggregation ──────────────────────────────────────

class TestPortfolioRisk:
    def test_total_dollar_risk_sums_active_adjusted_risks(self):
        strategies = [
            _strategy("AAPL", dollar_risk=500.0,  position_size=100),
            _strategy("MSFT", dollar_risk=300.0,  position_size=100),
        ]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker(bid=149.90, ask=150.10)):
            result = advisor.advise(strategies)

        # slippage per signal = 0.10 * 100 = 10, adjusted_risk AAPL=510, MSFT=310, total=820
        assert abs(result["portfolio_risk"]["total_dollar_risk"] - 820.0) < 1.0

    def test_portfolio_risk_pct_of_portfolio(self):
        strategies = [_strategy("AAPL", dollar_risk=1_000.0, position_size=100)]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker(bid=149.90, ask=150.10)):
            result = advisor.advise(strategies)

        pct = result["portfolio_risk"]["pct_of_portfolio"]
        assert 0 < pct < 5.0   # sanity: <5% for a single 1k risk on 100k portfolio

    def test_portfolio_risk_has_active_count(self):
        strategies = [
            _strategy("AAPL", signal_active=True),
            _strategy("MSFT", signal_active=False),
        ]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", return_value=_mock_ticker()):
            result = advisor.advise(strategies)

        assert result["portfolio_risk"]["active_count"] == 1


# ── Cycle 6: empty active signals safe fallback ───────────────────────────────

class TestEmptyFallback:
    def test_all_inactive_returns_empty_active_signals(self):
        strategies = [_strategy(signal_active=False), _strategy("MSFT", signal_active=False)]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        result = advisor.advise(strategies)

        assert result["active_signals"] == []

    def test_all_inactive_total_risk_is_zero(self):
        strategies = [_strategy(signal_active=False)]
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        result = advisor.advise(strategies)

        assert result["portfolio_risk"]["total_dollar_risk"] == 0.0

    def test_empty_strategies_list(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        result = advisor.advise([])

        assert result["active_signals"] == []
        assert result["inactive_count"] == 0
        assert result["portfolio_risk"]["total_dollar_risk"] == 0.0


# ── Cycle 7: yfinance fetch failure — ATR-based fallback ─────────────────────

class TestYFinanceFetchFailure:
    def test_fetch_failure_does_not_raise(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = advisor.advise([_strategy(current_atr=3.0, position_size=100)])

        assert isinstance(result, dict)
        assert len(result["active_signals"]) == 1

    def test_fallback_slippage_is_atr_based(self):
        """Fallback: slippage_per_share = ATR * 0.1 (Corwin-Schultz proxy)."""
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = advisor.advise([_strategy(current_atr=3.0, position_size=100)])

        sig = result["active_signals"][0]
        # ATR=3.0, proxy = 3.0 * 0.1 = 0.30 per share, total = 30.0
        assert abs(sig["slippage_total"] - 30.0) < 0.01

    def test_fallback_sets_spread_to_none(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = advisor.advise([_strategy()])

        sig = result["active_signals"][0]
        assert sig["spread"] is None

    def test_fallback_adds_warning(self):
        advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = advisor.advise([_strategy("AAPL")])

        assert any("AAPL" in w and "live" in w.lower() for w in result["warnings"])
