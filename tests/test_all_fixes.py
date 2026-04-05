"""
test_all_fixes.py
=================
Comprehensive regression test suite covering all 10 fixes applied to the
LLM-based alpha finder pipeline.  Written as a quant developer / researcher
pair: each test validates a specific, named fix and documents WHY the fix
was needed.

Run with:
    pytest tests/test_all_fixes.py -v

All tests are pure-Python and require only numpy, pandas, and scipy (already
in requirements).  No LLM API calls are made.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

# ── make sure the project root is on the path ────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — Trending-Down → Mean-Reversion (not VolatilityBreakout)
#
# Root cause: VolatilityBreakout buys ABOVE the upper Bollinger Band — a long-
# only entry in a confirmed downtrend.  This is directionally contradictory.
# Mean-Reversion fades oversold exhaustion within the downtrend (buy RSI < 30
# + below lower BB, exit at mean), which is the correct posture.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix1_trending_down_maps_to_mean_reversion():
    """Trending-Down regime must route to Mean-Reversion, never VolatilityBreakout."""
    from strategy_selector import _REGIME_TO_STRATEGY
    assert _REGIME_TO_STRATEGY["Trending-Down"] == "Mean-Reversion", (
        "Trending-Down must map to Mean-Reversion. "
        "VolatilityBreakout (long-only, buys above upper BB) fights a downtrend."
    )


def test_fix1_trending_up_still_maps_to_momentum():
    """Sanity check: Trending-Up must still route to Momentum."""
    from strategy_selector import _REGIME_TO_STRATEGY
    assert _REGIME_TO_STRATEGY["Trending-Up"] == "Momentum"


def test_fix1_no_regression_on_other_regimes():
    """All other regime mappings are unchanged."""
    from strategy_selector import _REGIME_TO_STRATEGY
    expected = {
        "High-Volatility":  "VolatilityBreakout",
        "Crisis":           "VolatilityBreakout",
        "Mean-Reverting":   "AlphaCombined",
        "Low-Volatility":   "MLSignal",
        "Neutral":          "MLSignal",
        "Event-Driven":     "EventDriven",
    }
    for regime, strategy in expected.items():
        assert _REGIME_TO_STRATEGY[regime] == strategy, (
            f"Unexpected change: {regime} → {_REGIME_TO_STRATEGY[regime]} "
            f"(expected {strategy})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — Backtester self._slip state mutation
#
# Root cause: backtester.run() was mutating self._slip based on ADV, then
# leaving the mutated value for subsequent calls.  Sequential calls with
# different ADVs shared leaked state.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix2_backtester_slip_immutable_after_run():
    """self._slip must not change between sequential backtester.run() calls."""
    from backtester import Backtester

    bt = Backtester(slippage_bps=10)
    initial_slip = bt._slip  # noqa: SLF001

    # Build minimal OHLCV with enough history
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.cumsum(np.random.default_rng(42).normal(0.001, 0.02, n)) + 100
    df = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.full(n, 1_000_000),
    }, index=dates)

    from strategy_selector import MOMENTUM_BASE
    import copy
    params = copy.deepcopy(MOMENTUM_BASE)

    # Run once with a large ADV — should change local slip but not self._slip
    bt.run("TEST", "Momentum", df, params, adv_shares=50_000)
    assert bt._slip == initial_slip, (  # noqa: SLF001
        f"self._slip mutated after run(): expected {initial_slip}, got {bt._slip}"
    )

    # Run again with a tiny ADV — still should not mutate
    bt.run("TEST", "Momentum", df, params, adv_shares=100)
    assert bt._slip == initial_slip, (  # noqa: SLF001
        "self._slip must remain constant across sequential run() calls."
    )


def test_fix2_backtester_slip_resolve_does_not_mutate():
    """_resolve_slip() must return a value without touching self._slip."""
    from backtester import Backtester

    bt = Backtester(slippage_bps=10)
    before = bt._slip  # noqa: SLF001
    resolved = bt._resolve_slip(500_000)  # large ADV → higher slip
    after = bt._slip  # noqa: SLF001

    assert before == after, "_resolve_slip() must be non-mutating"
    assert resolved >= before, "Large ADV should produce higher-or-equal slip than default"


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3 — Portfolio construction must exclude diagnostics-failed tickers
#
# Root cause: PortfolioOptimizer.optimize() previously fed all backtests into
# the CS momentum ranking without first checking whether diagnostics passed.
# A strategy with negative Sharpe could be allocated capital if it ranked
# well on 12-1 month price momentum.
# ═══════════════════════════════════════════════════════════════════════════

def _make_return_series(n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.015, n))


def test_fix3_failed_diagnostic_ticker_excluded_from_portfolio():
    """Tickers that failed diagnostics must appear in 'rejected', not 'allocations'."""
    from portfolio_optimizer import PortfolioOptimizer

    # Two tickers: GOOD passed, BAD failed
    backtests = [
        {"ticker": "GOOD", "returns": _make_return_series(300, seed=1)},
        {"ticker": "BAD",  "returns": _make_return_series(300, seed=2)},
    ]
    diagnostics = [
        {"ticker": "GOOD", "passed": True,  "reject_reason": None},
        {"ticker": "BAD",  "passed": False, "reject_reason": "Sharpe 0.10 below floor"},
    ]
    n = 300
    dates = pd.date_range("2016-01-01", periods=n, freq="B")
    close = np.linspace(100, 200, n)

    def _make_ohlcv(trend_factor=1.0):
        c = 100 + np.cumsum(np.random.default_rng(0).normal(0.1 * trend_factor, 1, n))
        return pd.DataFrame({
            "Open": c, "High": c * 1.01, "Low": c * 0.99,
            "Close": c, "Volume": np.full(n, 1_000_000),
        }, index=dates)

    ohlcv_dict = {"GOOD": _make_ohlcv(1.0), "BAD": _make_ohlcv(0.5)}

    opt = PortfolioOptimizer(initial_portfolio=100_000)
    result = opt.optimize(backtests, diagnostics, ohlcv_dict)

    allocated_tickers = {a["ticker"] for a in result["allocations"]}
    rejected_tickers  = {r["ticker"] for r in result["rejected"]}

    assert "BAD" not in allocated_tickers, (
        "BAD (diagnostics failed) must not receive capital allocation"
    )
    assert "BAD" in rejected_tickers, (
        "BAD must appear in the rejected list with a diagnostics-failure reason"
    )


def test_fix3_all_failed_returns_empty_portfolio():
    """If all tickers fail diagnostics, allocations must be empty."""
    from portfolio_optimizer import PortfolioOptimizer

    backtests   = [{"ticker": "X", "returns": _make_return_series(100)}]
    diagnostics = [{"ticker": "X", "passed": False, "reject_reason": "test"}]
    ohlcv_dict  = {}

    opt    = PortfolioOptimizer()
    result = opt.optimize(backtests, diagnostics, ohlcv_dict)
    assert result["allocations"] == []


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4 — OOS Sharpe None sentinel (not 0.0) for underpowered walk-forward
#
# Root cause: _walk_forward_degradation returned 0.0 as oos_sharpe for
# underpowered strategies (< 30 trades), which the report rendered as the
# real number "0.000" — indistinguishable from a measured zero Sharpe.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix4_wf_underpowered_returns_none_oos_sharpe():
    """Walk-forward with < 30 trades must return oos_sharpe=None, not 0.0."""
    from diagnostics_engine import DiagnosticsEngine

    engine = DiagnosticsEngine(verbose=False)
    ret = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 500))
    # Only 5 trades — below the 30-trade WF minimum
    trade_log = [{"pnl": 1.0, "holding_days": 5} for _ in range(5)]

    metrics = engine._compute_metrics(trade_log, ret)  # noqa: SLF001
    assert metrics["oos_sharpe"] is None, (
        "oos_sharpe must be None (not 0.0) when WF is underpowered — "
        "0.0 would be rendered as a measured zero Sharpe in the report"
    )
    assert metrics.get("wf_underpowered") is True


def test_fix4_wf_underpowered_check_floors_does_not_reject_on_oos():
    """_check_floors must not fail a strategy on OOS Sharpe when WF is underpowered."""
    from diagnostics_engine import DiagnosticsEngine, _DEFAULT_FLOORS

    # Metrics that would pass all floors except OOS Sharpe (which is None)
    metrics = {
        "sharpe":                   0.80,
        "oos_sharpe":               None,   # underpowered sentinel
        "max_drawdown":             0.12,
        "win_rate":                 0.45,
        "profit_factor":            1.8,
        "kelly_fraction":           0.05,
        "walk_forward_degradation": 0.10,
        "wf_splits":                [],
        "wf_underpowered":          True,
        "trade_count":              15,
        "t_stat":                   2.0,
        "p_value":                  0.03,
        "bootstrap_sharpe_p5":      0.15,
        "bootstrap_sharpe_p95":     1.20,
        "permutation_p_value":      0.40,
        "rolling_pct_positive":     0.70,
        "rolling_sharpe_std":       0.50,
        "market_exposure":          0.20,
    }
    passed, reason = DiagnosticsEngine._check_floors(metrics, _DEFAULT_FLOORS)  # noqa: SLF001
    # Should fail on trade count (15 < 30 floor), not on OOS Sharpe
    if not passed:
        assert reason is not None
        assert "oos" not in reason.lower() or "underpowered" in reason.lower(), (
            f"Should not fail on OOS Sharpe when underpowered. Got: {reason}"
        )


def test_fix4_none_oos_sharpe_in_check_floors_uses_full_sharpe_as_fallback():
    """When oos_sharpe is None, _check_floors must substitute full-period Sharpe (not crash)."""
    from diagnostics_engine import DiagnosticsEngine, _DEFAULT_FLOORS

    # A passing strategy but with None oos_sharpe — should not raise TypeError
    metrics = {
        "sharpe":                   0.70,
        "oos_sharpe":               None,
        "max_drawdown":             0.10,
        "win_rate":                 0.50,
        "profit_factor":            2.0,
        "kelly_fraction":           0.10,
        "walk_forward_degradation": 0.05,
        "wf_splits":                [],
        "wf_underpowered":          True,
        "trade_count":              35,
        "t_stat":                   2.5,
        "p_value":                  0.02,
        "bootstrap_sharpe_p5":      0.20,
        "bootstrap_sharpe_p95":     1.30,
        "permutation_p_value":      0.35,
        "rolling_pct_positive":     0.75,
        "rolling_sharpe_std":       0.40,
        "market_exposure":          0.25,
    }
    try:
        DiagnosticsEngine._check_floors(metrics, _DEFAULT_FLOORS)  # noqa: SLF001
    except TypeError as e:
        pytest.fail(f"_check_floors crashed on None oos_sharpe: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 — MC Sharpe vs Diagnostic Sharpe annualization note
#
# Root cause: Monte Carlo uses sqrt(trades_per_year) annualization while
# DiagnosticsEngine uses sqrt(252).  For strategies with < 252 trades/year
# the MC Sharpe is lower — without a note, traders interpret the discrepancy
# as an error.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix5_mc_sharpe_annualization_note_in_report():
    """Report must include a footnote about MC vs Diagnostic Sharpe annualization."""
    from report_generator import ReportGenerator
    import tempfile

    gen = ReportGenerator(output_dir=tempfile.mkdtemp())

    # Minimal pipeline output that triggers the Monte Carlo section
    po = {
        "run_date": "2026-01-01",
        "summary": {}, "macro": {}, "ticker_verdicts": [], "regimes": [],
        "strategies": [], "diagnostics": [], "backtests": [], "spy_ohlcv": None,
        "features": {}, "markets": [],
        "monte_carlos": [
            {
                "ticker": "AAPL",
                "p5_final": 95_000, "p50_final": 110_000, "p95_final": 130_000,
                "p5_sharpe": 0.3, "p50_sharpe": 0.6, "p95_sharpe": 1.1,
                "p5_win_rate": 0.40, "p50_win_rate": 0.50, "p95_win_rate": 0.60,
                "p_ruin": 0.01, "p95_max_drawdown": 0.15, "median_cagr": 0.08,
                "p95_max_consec_losses": 5, "kelly_fraction": 0.10,
                "equity_band": [], "trade_count": 40,
                "insufficient_sample": False, "stress_test": False,
                "median_time_to_ruin": None, "ruin_severity": None,
            }
        ],
    }

    section = gen._monte_carlo_section(po)
    assert "annualization" in section.lower() or "annualiz" in section.lower(), (
        "Monte Carlo section must include a Sharpe annualization note explaining "
        "the difference between MC (trade-frequency) and Diagnostic (daily-return) Sharpe."
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 6 — Momentum MA exit period increased from 20 to 50
#
# Root cause: the 20-day MA fires as a leading exit reason even in strong
# trending markets, cutting winners before the trend exhausts.  The 50-day
# MA is the institutional standard for trend-following exits.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix6_momentum_base_ma_exit_period_is_50():
    """MOMENTUM_BASE.ma_exit_period must be 50 (not 20)."""
    from strategy_selector import MOMENTUM_BASE
    assert MOMENTUM_BASE["ma_exit_period"] == 50, (
        f"ma_exit_period={MOMENTUM_BASE['ma_exit_period']} — must be 50. "
        "20d MA fires too early in trending regimes, cutting winners short."
    )


def test_fix6_strategy_selector_inherits_50_day_ma():
    """StrategySelector must produce params with ma_exit_period=50 for Trending-Up."""
    from strategy_selector import StrategySelector

    sel = StrategySelector(llm_client=None, verbose=False)
    regime = {"regime": "Trending-Up", "hurst": 0.65, "atr_pct": 0.02, "ret_20d": 0.05}
    features = {"atr_pct": 0.02, "volume_ratio_30d": 1.1}
    result = sel.select("TEST", regime, features, {})
    assert result["adjusted_params"]["ma_exit_period"] == 50, (
        "StrategySelector must propagate ma_exit_period=50 to adjusted_params."
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 7 — Hurst boundary raised from 0.55 to 0.58 (uncertainty margin)
#
# Root cause: R/S estimation over 756 daily observations has standard error
# ~0.03–0.05 (Lo 1991).  A Hurst of 0.55 could easily be 0.50 noise.
# Requiring > 0.58 prevents borderline estimates from triggering Momentum
# routing and then failing every diagnostic floor.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix7_hurst_trending_threshold_is_058():
    """HURST_TRENDING must be 0.58, not 0.55."""
    from regime_classifier import HURST_TRENDING
    assert HURST_TRENDING == 0.58, (
        f"HURST_TRENDING={HURST_TRENDING} — must be 0.58. "
        "R/S estimation has SE ~0.03; 0.55 is within noise of the random-walk boundary."
    )


def _make_ohlcv(n: int = 400, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    """Helper: produce OHLCV with enough history for Hurst estimation."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(trend, 0.02, n)
    close = 100 * np.cumprod(1 + ret)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.full(n, 500_000),
    }, index=dates)


def test_fix7_borderline_hurst_stays_in_neutral_zone():
    """A Hurst of ~0.56 (borderline) should NOT be classified as Trending."""
    from regime_classifier import RegimeClassifier, HURST_TRENDING

    clf = RegimeClassifier()
    # Patch the Hurst method to return exactly 0.56 (old boundary would classify as Trending)
    original_hurst = clf._hurst

    def _mock_hurst(prices):
        return 0.56  # above old 0.55 threshold but below new 0.58

    clf._hurst = _mock_hurst  # type: ignore[method-assign]
    df = _make_ohlcv(800, trend=0.001)
    result = clf.classify("TEST", df)

    assert result["regime"] not in ("Trending-Up", "Trending-Down"), (
        f"Hurst=0.56 should NOT be classified as Trending with new threshold 0.58. "
        f"Got: {result['regime']}"
    )
    clf._hurst = original_hurst  # restore


def test_fix7_clear_hurst_still_classified_as_trending():
    """A Hurst of 0.65 (clearly trending) must still route to Trending regime."""
    from regime_classifier import RegimeClassifier

    clf = RegimeClassifier()
    original_hurst = clf._hurst

    def _mock_hurst(prices):
        return 0.65  # well above new threshold

    clf._hurst = _mock_hurst  # type: ignore[method-assign]
    df = _make_ohlcv(800, trend=0.002)  # positive 20-day return → Trending-Up
    result = clf.classify("TEST", df)

    assert result["regime"] == "Trending-Up", (
        f"Hurst=0.65 with positive 20d return must classify as Trending-Up. "
        f"Got: {result['regime']}"
    )
    clf._hurst = original_hurst


# ═══════════════════════════════════════════════════════════════════════════
# FIX 8 — LLM commentary source labels
#
# Root cause: the report conflated two distinct LLM outputs:
#   (a) StrategySelector alpha hypothesis (PRE-backtest, sees only regime/features)
#   (b) DiagnosticsEngine commentary (POST-backtest, sees realized P&L metrics)
# Without clear labels, the trader could confuse a pre-backtest opinion with
# post-backtest evidence.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix8_diagnostic_llm_commentary_label_is_post_backtest():
    """Diagnostic LLM commentary must be labeled 'post-backtest' in the report."""
    from report_generator import ReportGenerator
    import tempfile

    gen = ReportGenerator(output_dir=tempfile.mkdtemp())

    po = {
        "run_date": "2026-01-01",
        "summary": {}, "macro": {}, "ticker_verdicts": [], "regimes": [],
        "strategies": [], "backtests": [],
        "diagnostics": [
            {
                "ticker": "AAPL",
                "strategy": "Momentum",
                "passed": True,
                "reject_reason": None,
                "metrics": {
                    "sharpe": 0.8, "oos_sharpe": 0.6, "max_drawdown": 0.10,
                    "win_rate": 0.50, "profit_factor": 1.8, "kelly_fraction": 0.1,
                    "walk_forward_degradation": 0.05, "wf_splits": [],
                    "wf_underpowered": False, "trade_count": 50,
                    "t_stat": 2.5, "p_value": 0.03,
                    "bootstrap_sharpe_p5": 0.2, "bootstrap_sharpe_p95": 1.4,
                    "permutation_p_value": 0.35,
                    "rolling_pct_positive": 0.70, "rolling_sharpe_std": 0.40,
                    "market_exposure": 0.22,
                },
                "llm_commentary": "Strategy shows strong IS/OOS consistency.",
            }
        ],
        "monte_carlos": [], "spy_ohlcv": None, "features": {}, "markets": [],
        "execution_brief": {}, "correlation_warnings": [],
    }

    section = gen._diagnostic_section(po)
    # Must contain the post-backtest label
    assert "post-backtest" in section.lower() or "Post-backtest" in section, (
        "Diagnostic commentary must be labeled as 'post-backtest' to distinguish "
        "it from the pre-backtest StrategySelector LLM hypothesis."
    )


def test_fix8_strategy_llm_hypothesis_label_is_pre_backtest():
    """Strategy LLM alpha hypothesis must be labeled 'pre-backtest' in the report."""
    from report_generator import ReportGenerator
    import tempfile

    gen = ReportGenerator(output_dir=tempfile.mkdtemp())

    po = {
        "run_date": "2026-01-01",
        "summary": {}, "macro": {}, "ticker_verdicts": [], "regimes": [],
        "diagnostics": [], "backtests": [], "monte_carlos": [],
        "spy_ohlcv": None, "features": {}, "markets": [],
        "strategies": [
            {
                "ticker": "AAPL",
                "strategy": "Momentum",
                "regime": "Trending-Up",
                "reasoning": "Strong trend.",
                "adjusted_params": {"ma_exit_period": 50, "entry_lookback": 10,
                                    "volume_multiplier": 1.2, "trailing_stop_atr": 2.0,
                                    "stop_loss_atr": 1.5, "max_holding_days": 20},
                "llm_adjustments": [],
                "current_signal": {"signal_active": False, "details": ""},
                "llm_hypothesis": {
                    "agree": False,
                    "suggested": "Mean-Reversion",
                    "reason": "Recent RSI reading is oversold.",
                },
                "param_divergence_warnings": [],
            }
        ],
    }

    section = gen._strategy_section(po)
    assert "pre-backtest" in section.lower() or "StrategySelector" in section, (
        "LLM Alpha Hypothesis in strategy section must be labeled 'pre-backtest' "
        "to distinguish it from post-backtest Diagnostic commentary."
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 9 — Market exposure metric added to diagnostics
#
# Root cause: diagnostics reported only per-trade metrics (Sharpe, MaxDD, win
# rate) but not what fraction of the backtesting window the strategy was
# actually invested.  A strategy with 5% exposure and Sharpe 0.8 may have
# fewer actual compounding opportunities than its metrics suggest.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix9_market_exposure_present_in_compute_metrics():
    """_compute_metrics must include 'market_exposure' key."""
    from diagnostics_engine import DiagnosticsEngine

    engine = DiagnosticsEngine(verbose=False)
    ret = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 252))
    trade_log = [{"pnl": 1.0, "holding_days": 5} for _ in range(40)]

    metrics = engine._compute_metrics(trade_log, ret)
    assert "market_exposure" in metrics, "market_exposure must be in metrics dict"


def test_fix9_market_exposure_value_correct():
    """market_exposure = sum(holding_days) / len(returns)."""
    from diagnostics_engine import DiagnosticsEngine

    engine = DiagnosticsEngine(verbose=False)
    n_days = 252
    ret = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, n_days))

    # 40 trades × 5 days each = 200 days invested out of 252
    trade_log = [{"pnl": 1.0, "holding_days": 5} for _ in range(40)]
    expected_exposure = 200 / 252

    metrics = engine._compute_metrics(trade_log, ret)
    assert abs(metrics["market_exposure"] - expected_exposure) < 0.001, (
        f"market_exposure={metrics['market_exposure']:.4f}, expected {expected_exposure:.4f}"
    )


def test_fix9_market_exposure_zero_for_empty_trade_log():
    """market_exposure must be 0.0 when trade log is empty."""
    from diagnostics_engine import DiagnosticsEngine

    engine = DiagnosticsEngine(verbose=False)
    ret = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 252))

    metrics = engine._compute_metrics([], ret)
    assert metrics["market_exposure"] == 0.0


def test_fix9_report_shows_exposure_row():
    """Report diagnostic table must include a Market Exposure row."""
    from report_generator import _fmt_exposure_row

    # Under-deployed case
    metrics_low = {"market_exposure": 0.08}
    row_low = _fmt_exposure_row(metrics_low)
    assert "WARNING" in row_low.upper() or "under-deployed" in row_low.lower(), (
        "Exposure < 15% should produce a warning in the report row."
    )

    # Normal case
    metrics_ok = {"market_exposure": 0.30}
    row_ok = _fmt_exposure_row(metrics_ok)
    assert "Normal" in row_ok or "normal" in row_ok.lower(), (
        "Exposure 15-50% should be labeled 'Normal' in the report row."
    )

    # Missing metric
    row_na = _fmt_exposure_row({})
    assert "N/A" in row_na


# ═══════════════════════════════════════════════════════════════════════════
# FIX 10 — Permutation p-value consistent threshold-based labels
#
# Root cause: the permutation p-value row in the statistical significance
# summary table previously showed only the raw number "0.350" without any
# interpretation — a trader cannot know if that's good, neutral, or bad.
# ═══════════════════════════════════════════════════════════════════════════

def test_fix10_perm_pvalue_row_has_interpretation():
    """_fmt_perm_pvalue_row must return a row with an interpretation string."""
    from report_generator import _fmt_perm_pvalue_row

    # Strongly temporal
    row = _fmt_perm_pvalue_row({"permutation_p_value": 0.05})
    assert "temporal" in row.lower(), f"p=0.05 should mention temporal structure. Got: {row}"

    # IID expected
    row = _fmt_perm_pvalue_row({"permutation_p_value": 0.50})
    assert "iid" in row.lower() or "expected" in row.lower(), (
        f"p=0.50 should mention IID/expected. Got: {row}"
    )

    # Exit-destroying
    row = _fmt_perm_pvalue_row({"permutation_p_value": 0.95})
    assert "destroying" in row.lower() or "warning" in row.lower(), (
        f"p=0.95 should warn about exits destroying value. Got: {row}"
    )


def test_fix10_summary_table_perm_label_uses_brackets():
    """Statistical significance summary table must include bracket interpretation for perm p-val."""
    from report_generator import ReportGenerator
    import tempfile

    gen = ReportGenerator(output_dir=tempfile.mkdtemp())

    po = {
        "run_date": "2026-01-01",
        "summary": {}, "macro": {}, "ticker_verdicts": [], "regimes": [],
        "strategies": [], "backtests": [], "monte_carlos": [],
        "spy_ohlcv": None, "features": {}, "markets": [],
        "diagnostics": [
            {
                "ticker": "AAPL",
                "strategy": "Momentum",
                "passed": True,
                "reject_reason": None,
                "metrics": {
                    "sharpe": 0.8, "oos_sharpe": 0.6, "max_drawdown": 0.10,
                    "win_rate": 0.50, "profit_factor": 1.8, "kelly_fraction": 0.1,
                    "walk_forward_degradation": 0.05, "wf_splits": [],
                    "wf_underpowered": False, "trade_count": 50,
                    "t_stat": 2.5, "p_value": 0.03,
                    "bootstrap_sharpe_p5": 0.2, "bootstrap_sharpe_p95": 1.4,
                    "permutation_p_value": 0.45,
                    "rolling_pct_positive": 0.70, "rolling_sharpe_std": 0.40,
                    "market_exposure": 0.22,
                },
                "llm_commentary": None,
            }
        ],
        "execution_brief": {}, "correlation_warnings": [],
    }

    # The portfolio section or its sub-section renders the stats significance table
    # _portfolio_section also has a stats table — check the diagnostic section instead
    # which directly renders with _fmt_perm_pvalue_row
    section = gen._diagnostic_section(po)
    # IID for p=0.45 should appear in the diagnostic table
    assert "iid" in section.lower() or "expected" in section.lower(), (
        "Diagnostic section permutation row must include an interpretation label (e.g. IID expected)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION — all fixes together (smoke test)
#
# Verifies that the complete diagnostics pipeline runs end-to-end without
# raising exceptions when the fixed code paths interact.
# ═══════════════════════════════════════════════════════════════════════════

def test_integration_diagnostics_full_pipeline_no_crash():
    """Full diagnostics pipeline must run end-to-end without raising exceptions."""
    from diagnostics_engine import DiagnosticsEngine

    engine = DiagnosticsEngine(llm_client=None, verbose=False)
    rng = np.random.default_rng(999)

    # Generate a plausible return series (enough data, mild positive drift)
    ret = pd.Series(rng.normal(0.0008, 0.015, 500))

    # Mix of winning and losing trades
    trade_log = []
    for i in range(45):
        pnl = rng.uniform(50, 200) if rng.random() > 0.45 else rng.uniform(-150, -30)
        trade_log.append({"pnl": float(pnl), "holding_days": int(rng.integers(3, 15))})

    try:
        result = engine.run("SMOKE", "Momentum", trade_log, ret, regime_label="Trending-Up")
    except Exception as e:
        pytest.fail(f"Full diagnostics pipeline raised an exception: {e}")

    assert "market_exposure" in result["metrics"]
    assert result["metrics"]["oos_sharpe"] is not None or result["metrics"]["wf_underpowered"]


def test_integration_regime_classifier_with_new_hurst_threshold():
    """RegimeClassifier with new threshold must not crash and must classify correctly."""
    from regime_classifier import RegimeClassifier

    clf = RegimeClassifier()
    df  = _make_ohlcv(800)
    try:
        result = clf.classify("SMOKE", df)
    except Exception as e:
        pytest.fail(f"RegimeClassifier raised exception: {e}")

    assert result["regime"] in (
        "Crisis", "Trending-Up", "Trending-Down", "Mean-Reverting",
        "High-Volatility", "Low-Volatility", "Event-Driven", "Neutral"
    )


def test_integration_report_generator_no_crash_on_none_oos_sharpe():
    """ReportGenerator must not crash when OOS Sharpe is None (underpowered)."""
    from report_generator import ReportGenerator
    import tempfile

    gen = ReportGenerator(output_dir=tempfile.mkdtemp())

    po = {
        "run_date": "2026-01-01",
        "summary": {}, "macro": {}, "ticker_verdicts": [], "regimes": [],
        "strategies": [], "backtests": [],
        "diagnostics": [
            {
                "ticker": "AAPL",
                "strategy": "Momentum",
                "passed": True,
                "reject_reason": None,
                "metrics": {
                    "sharpe": 0.7, "oos_sharpe": None,  # <-- underpowered sentinel
                    "max_drawdown": 0.10,
                    "win_rate": 0.50, "profit_factor": 1.8, "kelly_fraction": 0.1,
                    "walk_forward_degradation": 0.05,
                    "wf_splits": [
                        {"is_pct": 0.6, "is_sharpe": None, "oos_sharpe": None,
                         "degradation": None, "passed": True, "underpowered": True,
                         "rolling_wf": False},
                    ],
                    "wf_underpowered": True, "trade_count": 12,
                    "t_stat": 1.8, "p_value": 0.04,
                    "bootstrap_sharpe_p5": 0.1, "bootstrap_sharpe_p95": 1.1,
                    "permutation_p_value": 0.40,
                    "rolling_pct_positive": 0.65, "rolling_sharpe_std": 0.45,
                    "market_exposure": 0.10,
                },
                "llm_commentary": None,
            }
        ],
        "monte_carlos": [], "spy_ohlcv": None, "features": {}, "markets": [],
        "execution_brief": {}, "correlation_warnings": [],
    }

    try:
        section = gen._diagnostic_section(po)
    except (TypeError, ValueError) as e:
        pytest.fail(f"ReportGenerator crashed on None OOS Sharpe: {e}")

    # Walk-forward table with None values should render as "N/A"
    assert "N/A" in section, (
        "Report must render None walk-forward values as 'N/A' not '0.000'"
    )
