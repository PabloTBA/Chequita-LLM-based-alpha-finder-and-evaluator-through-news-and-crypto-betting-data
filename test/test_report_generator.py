"""
Tests for ReportGenerator.
All tests use a synthetic pipeline_output fixture — no real data, no LLM.
Tests verify file creation, section headings, and key data presence.
"""
import os
import math
import tempfile
import numpy as np
import pandas as pd
import pytest

from report_generator import ReportGenerator

# ── Synthetic pipeline output ─────────────────────────────────────────────────

_idx = pd.date_range("2024-01-02", periods=252, freq="B")
_equity = pd.Series(
    100_000 * (1 + np.linspace(0, 0.25, 252)),
    index=_idx,
)
_returns = _equity.pct_change().fillna(0.0)

PIPELINE_OUTPUT = {
    "run_date": "2026-03-19",
    "summary": {
        "market_bias":   "bullish",
        "top_themes":    ["AI infrastructure", "semiconductor demand"],
        "key_risks":     ["tariff escalation", "Fed policy"],
        "article_count": 87,
        "window_start":  "2026-03-12",
        "window_end":    "2026-03-19",
    },
    "macro": {
        "market_bias":        "bullish",
        "favored_sectors":    ["Technology", "Industrials"],
        "avoid_sectors":      ["Energy", "Real Estate"],
        "active_macro_risks": ["tariff", "rates"],
        "reasoning":          "Tech sector remains resilient amid macro uncertainty.",
    },
    "ticker_verdicts": [
        {"ticker": "AAPL", "verdict": "buy",   "reasoning": "Strong breakout setup."},
        {"ticker": "MSFT", "verdict": "watch", "reasoning": "Waiting for confirmation."},
        {"ticker": "NVDA", "verdict": "avoid", "reasoning": "Overextended, high ATR."},
    ],
    "regimes": [
        {"ticker": "AAPL", "regime": "Trending",      "hurst": 0.68, "atr_pct": 0.022},
        {"ticker": "MSFT", "regime": "Neutral",       "hurst": 0.51, "atr_pct": 0.018},
        {"ticker": "NVDA", "regime": "High-Volatility","hurst": 0.52, "atr_pct": 0.045},
    ],
    "strategies": [
        {
            "ticker":          "AAPL",
            "strategy":        "Momentum",
            "regime":          "Trending",
            "base_params":     {"entry_lookback": 20, "stop_loss_atr": 1.5},
            "adjusted_params": {"entry_lookback": 25, "stop_loss_atr": 1.5},
            "llm_adjustments": ["Extended entry_lookback from 20 to 25 — strong trend context"],
            "reasoning":       "Wider lookback suits sustained trend.",
        },
    ],
    "diagnostics": [
        {
            "ticker":    "AAPL",
            "strategy":  "Momentum",
            "passed":    True,
            "reject_reason": None,
            "metrics": {
                "sharpe":                   1.42,
                "max_drawdown":             0.14,
                "win_rate":                 0.58,
                "walk_forward_degradation": 0.21,
                "trade_count":              34,
            },
            "llm_commentary": "Solid risk-adjusted returns with manageable drawdown.",
        },
    ],
    "backtests": [
        {
            "ticker":   "AAPL",
            "strategy": "Momentum",
            "trade_log": [
                {
                    "entry_date":    _idx[20],
                    "entry_price":   150.0,
                    "exit_date":     _idx[35],
                    "exit_price":    162.0,
                    "holding_days":  15,
                    "position_size": 66.67,
                    "pnl":           800.0,
                    "exit_reason":   "trailing_stop",
                },
                {
                    "entry_date":    _idx[50],
                    "entry_price":   165.0,
                    "exit_date":     _idx[55],
                    "exit_price":    160.0,
                    "holding_days":  5,
                    "position_size": 66.67,
                    "pnl":          -333.0,
                    "exit_reason":   "stop_loss",
                },
            ],
            "equity_curve": _equity,
            "returns":      _returns,
            "summary": {
                "total_return": 0.25,
                "trade_count":  34,
                "win_rate":     0.58,
            },
        },
    ],
    "monte_carlos": [
        {
            "ticker":               "AAPL",
            "p5_final":             88_000.0,
            "p50_final":            107_000.0,
            "p95_final":            131_000.0,
            "p_ruin":               0.03,
            "p95_max_drawdown":     0.22,
            "median_cagr":          0.07,
            "p5_sharpe":            0.38,
            "p50_sharpe":           1.15,
            "p95_sharpe":           2.41,
            "p5_win_rate":          0.44,
            "p50_win_rate":         0.58,
            "p95_win_rate":         0.73,
            "p95_max_consec_losses": 5,
            "kelly_fraction":       0.14,
            "median_time_to_ruin":  None,
            "ruin_severity":        None,
            "equity_band": [
                {"step": i * 2, "p5": 88_000.0 + i * 500,
                 "p50": 100_000.0 + i * 700, "p95": 115_000.0 + i * 900}
                for i in range(20)
            ],
        }
    ],
}

REQUIRED_SECTIONS = [
    "## Executive Summary",
    "## Macro Environment",
    "## Shortlisted Tickers",
    "## Regime Classification",
    "## Strategy Parameters",
    "## Diagnostic Results",
    "## Backtest Results",
    "## Monte Carlo Stress Test",
]


@pytest.fixture
def tmp_gen(tmp_path):
    return ReportGenerator(output_dir=str(tmp_path))


@pytest.fixture
def report_path(tmp_gen):
    return tmp_gen.generate(PIPELINE_OUTPUT)


@pytest.fixture
def report_text(report_path):
    with open(report_path, encoding="utf-8") as f:
        return f.read()


# ── Cycle 1: file creation ────────────────────────────────────────────────────

class TestFileCreation:
    def test_generate_returns_string(self, tmp_gen):
        path = tmp_gen.generate(PIPELINE_OUTPUT)
        assert isinstance(path, str)

    def test_file_exists_on_disk(self, report_path):
        assert os.path.isfile(report_path)

    def test_filename_contains_run_date(self, report_path):
        assert "2026-03-19" in os.path.basename(report_path)

    def test_file_has_md_extension(self, report_path):
        assert report_path.endswith(".md")

    def test_file_is_non_empty(self, report_path):
        assert os.path.getsize(report_path) > 0


# ── Cycle 2: section headings ────────────────────────────────────────────────

class TestSections:
    def test_report_starts_with_heading(self, report_text):
        assert report_text.lstrip().startswith("#")

    def test_all_required_sections_present(self, report_text):
        for section in REQUIRED_SECTIONS:
            assert section in report_text, f"Missing section: {section}"


# ── Cycle 3: executive summary ───────────────────────────────────────────────

class TestExecutiveSummary:
    def test_run_date_in_summary(self, report_text):
        assert "2026-03-19" in report_text

    def test_article_count_in_summary(self, report_text):
        assert "87" in report_text

    def test_market_bias_in_summary(self, report_text):
        assert "bullish" in report_text.lower()


# ── Cycle 4: macro section ────────────────────────────────────────────────────

class TestMacroSection:
    def test_favored_sectors_listed(self, report_text):
        assert "Technology" in report_text

    def test_avoid_sectors_listed(self, report_text):
        assert "Energy" in report_text

    def test_macro_risks_listed(self, report_text):
        assert "tariff" in report_text.lower()


# ── Cycle 5: ticker verdicts ──────────────────────────────────────────────────

class TestTickerVerdicts:
    def test_all_tickers_present(self, report_text):
        for tv in PIPELINE_OUTPUT["ticker_verdicts"]:
            assert tv["ticker"] in report_text

    def test_verdicts_present(self, report_text):
        for verdict in ("buy", "watch", "avoid"):
            assert verdict in report_text.lower()


# ── Cycle 6: regime section ───────────────────────────────────────────────────

class TestRegimeSection:
    def test_regime_labels_present(self, report_text):
        assert "Trending" in report_text

    def test_hurst_values_present(self, report_text):
        assert "0.68" in report_text

    def test_high_volatility_label_present(self, report_text):
        assert "High-Volatility" in report_text


# ── Cycle 7: strategy section ────────────────────────────────────────────────

class TestStrategySection:
    def test_strategy_name_present(self, report_text):
        assert "Momentum" in report_text

    def test_adjusted_params_present(self, report_text):
        # entry_lookback adjusted to 25
        assert "25" in report_text

    def test_llm_adjustment_note_present(self, report_text):
        assert "entry_lookback" in report_text


# ── Cycle 8: diagnostic section ──────────────────────────────────────────────

class TestDiagnosticSection:
    def test_sharpe_ratio_present(self, report_text):
        assert "1.42" in report_text

    def test_max_drawdown_present(self, report_text):
        assert "14" in report_text   # 14% drawdown

    def test_win_rate_present(self, report_text):
        assert "58" in report_text   # 58% win rate

    def test_pass_status_present(self, report_text):
        assert "PASS" in report_text or "pass" in report_text.lower()

    def test_llm_commentary_present(self, report_text):
        assert "Solid risk-adjusted returns" in report_text

    def test_walk_forward_degradation_present(self, report_text):
        assert "21" in report_text   # 21% degradation


# ── Cycle 9: advanced metrics ────────────────────────────────────────────────

class TestAdvancedMetrics:
    def test_profit_factor_present(self, report_text):
        assert "Profit Factor" in report_text or "profit_factor" in report_text.lower()

    def test_sortino_present(self, report_text):
        assert "Sortino" in report_text

    def test_calmar_present(self, report_text):
        assert "Calmar" in report_text

    def test_cagr_present(self, report_text):
        assert "CAGR" in report_text

    def test_var_present(self, report_text):
        assert "VaR" in report_text

    def test_cvar_present(self, report_text):
        assert "CVaR" in report_text

    def test_max_consecutive_losses_present(self, report_text):
        assert "Consecutive" in report_text or "consecutive" in report_text

    def test_exit_reason_breakdown_present(self, report_text):
        assert "trailing_stop" in report_text or "stop_loss" in report_text


# ── Cycle 10: backtest section ────────────────────────────────────────────────

class TestBacktestSection:
    def test_total_return_present(self, report_text):
        assert "25" in report_text   # 25% total return

    def test_trade_log_entry_dates_present(self, report_text):
        assert "2024" in report_text  # entry dates are in 2024

    def test_pnl_values_present(self, report_text):
        assert "800" in report_text   # winning trade pnl

    def test_exit_reasons_in_trade_log(self, report_text):
        assert "trailing_stop" in report_text


# ── Cycle 11: graph-ready tables ─────────────────────────────────────────────

class TestGraphReadyTables:
    def test_equity_curve_table_present(self, report_text):
        assert "Equity Curve" in report_text or "equity_curve" in report_text.lower()

    def test_drawdown_table_present(self, report_text):
        assert "Drawdown" in report_text

    def test_walk_forward_table_present(self, report_text):
        assert "Walk-Forward" in report_text or "Walk Forward" in report_text

    def test_return_distribution_present(self, report_text):
        assert "Distribution" in report_text or "distribution" in report_text


# ── Cycle 12: Monte Carlo section ─────────────────────────────────────────────

class TestMonteCarloSection:
    def test_section_heading_present(self, report_text):
        assert "## Monte Carlo Stress Test" in report_text

    def test_ticker_heading_present(self, report_text):
        assert "### AAPL" in report_text

    def test_p_ruin_present(self, report_text):
        assert "P(Ruin)" in report_text or "Ruin" in report_text

    def test_p50_final_equity_present(self, report_text):
        assert "107,000" in report_text  # p50_final = 107_000

    def test_kelly_fraction_present(self, report_text):
        assert "Kelly" in report_text

    def test_p50_sharpe_present(self, report_text):
        assert "1.15" in report_text  # p50_sharpe

    def test_equity_band_table_present(self, report_text):
        assert "Equity Confidence Band" in report_text

    def test_median_cagr_present(self, report_text):
        assert "7.00%" in report_text or "7%" in report_text or "CAGR" in report_text

    def test_p95_max_consec_losses_present(self, report_text):
        assert "5" in report_text  # p95_max_consec_losses

    def test_time_to_ruin_na_when_none(self, report_text):
        assert "N/A" in report_text  # median_time_to_ruin is None
