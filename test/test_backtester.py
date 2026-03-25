"""
Tests for Backtester.
All tests use synthetic OHLCV fixtures with known entry/exit bars — no real data.
"""
import numpy as np
import pandas as pd
import pytest

from backtester import Backtester

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ohlcv(closes, volumes=None, hl_spread=2.0, start="2022-01-03"):
    closes  = np.array(closes, dtype=float)
    n       = len(closes)
    highs   = closes + hl_spread / 2
    lows    = closes - hl_spread / 2
    opens   = closes - 0.1
    vols    = np.full(n, 1_000_000.0) if volumes is None else np.array(volumes, dtype=float)
    idx     = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

# FLAT: 60 days at close=100, vol=1M → no momentum signals
FLAT_OHLCV = make_ohlcv(np.full(60, 100.0))

# STOP_LOSS: entry at bar 30 (close=101, vol=2M), stop fires at bar 31 (close=97)
# ATR at entry = 2.0  →  stop = 101 - 1.5×2 = 98  →  97 < 98 ✓
_sl_closes  = np.concatenate([np.full(30, 100.0), [101.0, 97.0], np.full(28, 97.0)])
_sl_volumes = np.concatenate([np.full(30, 1_000_000.0), [2_000_000.0], np.full(29, 1_000_000.0)])
STOP_LOSS_OHLCV = make_ohlcv(_sl_closes, volumes=_sl_volumes)

# TRAILING_STOP: entry at bar 30, price rises to 110 (bars 31-34), drops to 104 at bar 35
# peak=110, ATR≈2.78 at bar 35  →  trailing=110-2×2.78≈104.44  →  104 < 104.44 ✓
_ts_closes  = np.concatenate([np.full(30, 100.0), [101.0], np.full(4, 110.0), [104.0], np.full(24, 104.0)])
_ts_volumes = np.concatenate([np.full(30, 1_000_000.0), [2_000_000.0], np.full(29, 1_000_000.0)])
TRAILING_STOP_OHLCV = make_ohlcv(_ts_closes, volumes=_ts_volumes)

# MAX_HOLDING: entry at bar 30, price stays at 103 — above all stops — for 20 days
# max_holding_days=20 → exit at bar 50
_mh_closes  = np.concatenate([np.full(30, 100.0), [101.0], np.full(29, 103.0)])
_mh_volumes = np.concatenate([np.full(30, 1_000_000.0), [2_000_000.0], np.full(29, 1_000_000.0)])
MAX_HOLDING_OHLCV = make_ohlcv(_mh_closes, volumes=_mh_volumes)

# MEAN_REV: flat at 100 for 40 days, then sharp drops: 85,70,55,40,25
# At bar 40: RSI=0 < 30 AND close=85 below lower BB  →  entry fires
# At bar 41: close=70 < stop(80.5=85-1.5×3)  →  stop loss fires
_mr_closes = np.concatenate([
    np.full(40, 100.0),
    [85.0, 70.0, 55.0, 40.0, 25.0],
    np.full(15, 25.0),
])
MEAN_REV_OHLCV = make_ohlcv(_mr_closes)

# ── Strategy dicts ────────────────────────────────────────────────────────────

MOMENTUM_STRATEGY = {
    "strategy": "Momentum",
    "adjusted_params": {
        "entry_lookback":    20,
        "volume_multiplier": 1.5,
        "trailing_stop_atr": 2.0,
        "ma_exit_period":    10,
        "stop_loss_atr":     1.5,
        "max_holding_days":  20,
    },
}

MEAN_REV_STRATEGY = {
    "strategy": "Mean-Reversion",
    "adjusted_params": {
        "rsi_entry_threshold": 30,
        "rsi_exit_threshold":  55,
        "bb_period":           20,
        "bb_std":              2.0,
        "stop_loss_atr":       1.5,
        "max_holding_days":    10,
    },
}

RESULT_REQUIRED_KEYS = {"ticker", "strategy", "trade_log", "equity_curve", "returns", "summary"}
TRADE_REQUIRED_KEYS  = {
    "entry_date", "entry_price", "exit_date", "exit_price",
    "holding_days", "position_size", "pnl", "exit_reason",
}
VALID_EXIT_REASONS = {"stop_loss", "trailing_stop", "ma_exit", "max_holding", "rsi_exit"}


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_result_has_required_keys(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert RESULT_REQUIRED_KEYS == set(r.keys())

    def test_ticker_matches_input(self):
        bt = Backtester()
        r  = bt.run("NVDA", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["ticker"] == "NVDA"

    def test_strategy_matches_input(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["strategy"] == "Momentum"

    def test_trade_log_entries_have_required_keys(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert len(r["trade_log"]) >= 1
        for trade in r["trade_log"]:
            assert TRADE_REQUIRED_KEYS == set(trade.keys())

    def test_exit_reason_is_valid(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        for trade in r["trade_log"]:
            assert trade["exit_reason"] in VALID_EXIT_REASONS


# ── Cycle 2: no signals → no trades ──────────────────────────────────────────

class TestNoSignals:
    def test_flat_price_produces_no_trades(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, FLAT_OHLCV)
        assert r["trade_log"] == []

    def test_empty_trade_log_trade_count_zero(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, FLAT_OHLCV)
        assert r["summary"]["trade_count"] == 0


# ── Cycle 3: momentum entry ───────────────────────────────────────────────────

class TestMomentumEntry:
    def test_breakout_with_volume_generates_trade(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert len(r["trade_log"]) == 1

    def test_entry_price_matches_breakout_close(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["trade_log"][0]["entry_price"] == pytest.approx(101.0, abs=0.01)


# ── Cycle 4: stop loss exit ───────────────────────────────────────────────────

class TestStopLoss:
    def test_stop_loss_exit_reason(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["trade_log"][0]["exit_reason"] == "stop_loss"

    def test_stop_loss_exit_price(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["trade_log"][0]["exit_price"] == pytest.approx(97.0, abs=0.01)

    def test_stop_loss_holding_days_is_one(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["trade_log"][0]["holding_days"] == 1


# ── Cycle 5: trailing stop exit ───────────────────────────────────────────────

class TestTrailingStop:
    def test_trailing_stop_exit_reason(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, TRAILING_STOP_OHLCV)
        assert r["trade_log"][0]["exit_reason"] == "trailing_stop"

    def test_trailing_stop_exit_above_hard_stop(self):
        """Exit price should be above the hard stop floor (101 - 1.5×2 = 98)."""
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, TRAILING_STOP_OHLCV)
        assert r["trade_log"][0]["exit_price"] > 98.0


# ── Cycle 6: max holding exit ─────────────────────────────────────────────────

class TestMaxHolding:
    def test_max_holding_exit_reason(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, MAX_HOLDING_OHLCV)
        assert r["trade_log"][0]["exit_reason"] == "max_holding"

    def test_max_holding_days_equals_param(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, MAX_HOLDING_OHLCV)
        assert r["trade_log"][0]["holding_days"] == MOMENTUM_STRATEGY["adjusted_params"]["max_holding_days"]


# ── Cycle 7: position sizing ──────────────────────────────────────────────────

class TestPositionSizing:
    def test_position_size_formula(self):
        """position_size = (portfolio × 0.01) / (stop_loss_atr × ATR_at_entry)
        ATR at entry bar 30 = 2.0  →  size = (100_000 × 0.01)/(1.5×2) = 333.33"""
        bt   = Backtester(initial_portfolio=100_000.0)
        r    = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        expected = (100_000.0 * 0.01) / (1.5 * 2.0)
        assert r["trade_log"][0]["position_size"] == pytest.approx(expected, rel=0.01)

    def test_position_size_scales_with_portfolio(self):
        """Doubling portfolio doubles position size."""
        bt1 = Backtester(initial_portfolio=100_000.0)
        bt2 = Backtester(initial_portfolio=200_000.0)
        r1  = bt1.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        r2  = bt2.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r2["trade_log"][0]["position_size"] == pytest.approx(
            2 * r1["trade_log"][0]["position_size"], rel=0.01
        )


# ── Cycle 8: P&L accuracy ────────────────────────────────────────────────────

class TestPnL:
    def test_pnl_formula(self):
        """pnl = (exit_price - entry_price) × position_size"""
        bt    = Backtester(initial_portfolio=100_000.0)
        r     = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        trade = r["trade_log"][0]
        expected_pnl = (trade["exit_price"] - trade["entry_price"]) * trade["position_size"]
        assert trade["pnl"] == pytest.approx(expected_pnl, rel=0.001)

    def test_losing_trade_has_negative_pnl(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["trade_log"][0]["pnl"] < 0


# ── Cycle 9: equity curve ────────────────────────────────────────────────────

class TestEquityCurve:
    def test_equity_curve_starts_at_initial_portfolio(self):
        bt = Backtester(initial_portfolio=100_000.0)
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["equity_curve"].iloc[0] == pytest.approx(100_000.0)

    def test_equity_curve_length_matches_ohlcv(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert len(r["equity_curve"]) == len(STOP_LOSS_OHLCV)

    def test_equity_curve_decreases_after_losing_trade(self):
        bt = Backtester(initial_portfolio=100_000.0)
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert r["equity_curve"].iloc[-1] < 100_000.0

    def test_returns_series_length_matches_ohlcv(self):
        bt = Backtester()
        r  = bt.run("AAPL", MOMENTUM_STRATEGY, STOP_LOSS_OHLCV)
        assert len(r["returns"]) == len(STOP_LOSS_OHLCV)


# ── Cycle 10: mean-reversion entry ───────────────────────────────────────────

class TestMeanReversionEntry:
    def test_oversold_series_generates_trade(self):
        bt = Backtester()
        r  = bt.run("AAPL", MEAN_REV_STRATEGY, MEAN_REV_OHLCV)
        assert len(r["trade_log"]) >= 1

    def test_mean_reversion_exit_reason_is_valid(self):
        bt = Backtester()
        r  = bt.run("AAPL", MEAN_REV_STRATEGY, MEAN_REV_OHLCV)
        for trade in r["trade_log"]:
            assert trade["exit_reason"] in VALID_EXIT_REASONS
