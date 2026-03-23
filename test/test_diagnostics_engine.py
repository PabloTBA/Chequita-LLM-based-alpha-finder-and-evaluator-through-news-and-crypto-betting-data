"""
Tests for DiagnosticsEngine.
All tests use synthetic returns/trade-log fixtures — no real market data.
LLM behavior tested with mocked llm_client.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from diagnostics_engine import DiagnosticsEngine

# ── Fixtures ──────────────────────────────────────────────────────────────────

_idx = pd.date_range("2024-01-02", periods=252, freq="B")

# BAD_SHARPE: alternating +0.02 / -0.019 → mean≈0.0005, std≈0.0195, Sharpe≈0.41
_bad_sharpe_arr = np.where(np.arange(252) % 2 == 0, 0.02, -0.019)
BAD_SHARPE_RETURNS = pd.Series(_bad_sharpe_arr, index=_idx)

# DEEP_DRAWDOWN: 84 days +0.005 → 84 days -0.007 → 84 days +0.005
# Sharpe ≈ 2.81 (passes), peak-to-trough drawdown ≈ 44.6% (fails)
DEEP_DRAWDOWN_RETURNS = pd.Series(
    np.concatenate([np.full(84, 0.005), np.full(84, -0.007), np.full(84, 0.005)]),
    index=_idx,
)

# BAD_WALKFORWARD: strong in-sample, weak OOS
# first 126: alternating 0.007/0.003 → mean=0.005, std≈0.002, Sharpe≈39.7
# last  126: alternating 0.0015/-0.0005 → mean=0.0005, std≈0.001, Sharpe≈7.9
# degradation ≈ 80% > 50%; overall Sharpe ≈ 15.8 (passes floor); max DD ≈ 0
BAD_WALKFORWARD_RETURNS = pd.Series(
    np.concatenate([
        np.where(np.arange(126) % 2 == 0, 0.007,   0.003),
        np.where(np.arange(126) % 2 == 0, 0.0015, -0.0005),
    ]),
    index=_idx,
)

# HEALTHY: alternating 0.004/0.002 — mean=0.003, std≈0.001, Sharpe≈47.6, max DD≈0
# identical halves → walk-forward degradation ≈ 0
HEALTHY_RETURNS = pd.Series(
    np.where(np.arange(252) % 2 == 0, 0.004, 0.002),
    index=_idx,
)

# Trade logs
HEALTHY_TRADE_LOG  = [{"pnl": (2.0 if i % 2 == 0 else -1.0)} for i in range(40)]  # 50%, 40 trades
LOW_WIN_RATE_LOG   = [{"pnl": (10.0 if i % 5 == 0 else -1.0)} for i in range(40)]  # 20%, 40 trades
FEW_TRADES_LOG     = [{"pnl": (1.0 if i % 2 == 0 else -0.5)} for i in range(5)]   # 60% WR, only 5 trades

REQUIRED_KEYS    = {"ticker", "strategy", "passed", "reject_reason", "metrics", "llm_commentary"}
REQUIRED_METRICS = {"sharpe", "max_drawdown", "win_rate", "walk_forward_degradation", "trade_count"}

LLM_COMMENTARY = "Strong momentum with solid risk-adjusted returns. Watch for drawdown risk."


def make_llm(response: str = LLM_COMMENTARY) -> MagicMock:
    return MagicMock(return_value=response)


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_run_returns_required_keys(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert REQUIRED_KEYS == set(r.keys())

    def test_ticker_matches_input(self):
        eng = DiagnosticsEngine()
        r = eng.run("NVDA", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert r["ticker"] == "NVDA"

    def test_strategy_matches_input(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Mean-Reversion", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert r["strategy"] == "Mean-Reversion"

    def test_metrics_contains_required_keys(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert REQUIRED_METRICS.issubset(set(r["metrics"].keys()))


# ── Cycle 2: Sharpe floor ─────────────────────────────────────────────────────

class TestSharpeFloor:
    def test_low_sharpe_auto_rejects(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_SHARPE_RETURNS)
        assert r["passed"] is False

    def test_low_sharpe_reject_reason_mentions_sharpe(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_SHARPE_RETURNS)
        assert "sharpe" in r["reject_reason"].lower()

    def test_sharpe_metric_reflects_bad_value(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_SHARPE_RETURNS)
        assert r["metrics"]["sharpe"] < 0.5


# ── Cycle 3: max drawdown floor ───────────────────────────────────────────────

class TestMaxDrawdownFloor:
    def test_deep_drawdown_auto_rejects(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, DEEP_DRAWDOWN_RETURNS)
        assert r["passed"] is False

    def test_deep_drawdown_reject_reason_mentions_drawdown(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, DEEP_DRAWDOWN_RETURNS)
        assert "drawdown" in r["reject_reason"].lower()

    def test_drawdown_metric_reflects_bad_value(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, DEEP_DRAWDOWN_RETURNS)
        assert r["metrics"]["max_drawdown"] > 0.30


# ── Cycle 4: win rate floor ───────────────────────────────────────────────────

class TestWinRateFloor:
    def test_low_win_rate_auto_rejects(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", LOW_WIN_RATE_LOG, HEALTHY_RETURNS)
        assert r["passed"] is False

    def test_low_win_rate_reject_reason_mentions_win_rate(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", LOW_WIN_RATE_LOG, HEALTHY_RETURNS)
        assert "win" in r["reject_reason"].lower()

    def test_win_rate_metric_reflects_bad_value(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", LOW_WIN_RATE_LOG, HEALTHY_RETURNS)
        assert r["metrics"]["win_rate"] < 0.35


# ── Cycle 5: walk-forward floor ───────────────────────────────────────────────

class TestWalkForwardFloor:
    def test_bad_walkforward_auto_rejects(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_WALKFORWARD_RETURNS)
        assert r["passed"] is False

    def test_bad_walkforward_reject_reason_mentions_walk_forward(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_WALKFORWARD_RETURNS)
        assert "walk" in r["reject_reason"].lower()

    def test_walk_forward_degradation_metric_reflects_bad_value(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_WALKFORWARD_RETURNS)
        assert r["metrics"]["walk_forward_degradation"] > 0.50


# ── Cycle 6: trade count floor ────────────────────────────────────────────────

class TestTradeCountFloor:
    def test_few_trades_auto_rejects(self):
        """5 trades passes all other floors but should fail trade count floor."""
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", FEW_TRADES_LOG, HEALTHY_RETURNS)
        assert r["passed"] is False

    def test_few_trades_reject_reason_mentions_trade(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", FEW_TRADES_LOG, HEALTHY_RETURNS)
        assert "trade" in r["reject_reason"].lower()

    def test_trade_count_metric_reflects_low_value(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", FEW_TRADES_LOG, HEALTHY_RETURNS)
        assert r["metrics"]["trade_count"] < 10


# ── Cycle 7: pass case ─────────────────────────────────────────────────────────

class TestPassCase:
    def test_healthy_strategy_passes(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert r["passed"] is True

    def test_passing_strategy_has_no_reject_reason(self):
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert r["reject_reason"] is None


# ── Cycle 8: LLM behavior ─────────────────────────────────────────────────────

class TestLLMBehavior:
    def test_llm_not_called_on_auto_reject(self):
        llm = make_llm()
        eng = DiagnosticsEngine(llm_client=llm)
        eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_SHARPE_RETURNS)
        assert llm.call_count == 0

    def test_llm_called_once_on_passing_strategy(self):
        llm = make_llm()
        eng = DiagnosticsEngine(llm_client=llm)
        eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert llm.call_count == 1

    def test_llm_commentary_none_on_reject(self):
        eng = DiagnosticsEngine(llm_client=make_llm())
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, BAD_SHARPE_RETURNS)
        assert r["llm_commentary"] is None

    def test_llm_commentary_present_on_pass(self):
        eng = DiagnosticsEngine(llm_client=make_llm())
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert r["llm_commentary"] == LLM_COMMENTARY

    def test_no_llm_client_passes_without_commentary(self):
        eng = DiagnosticsEngine()   # no llm_client
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, HEALTHY_RETURNS)
        assert r["passed"] is True
        assert r["llm_commentary"] is None


# ── Cycle 9: metrics accuracy ─────────────────────────────────────────────────

class TestMetricsAccuracy:
    def test_sharpe_near_zero_for_zero_mean_returns(self):
        """Alternating +0.01/-0.01 → mean=0, Sharpe≈0."""
        returns = pd.Series(
            np.where(np.arange(252) % 2 == 0, 0.01, -0.01),
            index=_idx,
        )
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, returns)
        assert abs(r["metrics"]["sharpe"]) < 0.1

    def test_max_drawdown_50pct_for_single_halving(self):
        """All zero returns then one -50% return → equity halves → max DD = 50%."""
        arr = np.zeros(252)
        arr[125] = -0.5
        returns = pd.Series(arr, index=_idx)
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", HEALTHY_TRADE_LOG, returns)
        assert abs(r["metrics"]["max_drawdown"] - 0.50) < 0.02

    def test_win_rate_30pct_for_three_winners_in_ten(self):
        """3 winners / 10 trades → win rate = 30%."""
        log = [{"pnl": 1.0} if i < 3 else {"pnl": -1.0} for i in range(10)]
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", log, HEALTHY_RETURNS)
        assert abs(r["metrics"]["win_rate"] - 0.30) < 0.01

    def test_trade_count_matches_log_length(self):
        log = [{"pnl": 1.0} for _ in range(17)]
        eng = DiagnosticsEngine()
        r = eng.run("AAPL", "Momentum", log, HEALTHY_RETURNS)
        assert r["metrics"]["trade_count"] == 17
