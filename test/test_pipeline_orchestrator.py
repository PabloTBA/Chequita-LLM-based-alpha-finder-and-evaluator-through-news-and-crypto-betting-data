"""
Tests for PipelineOrchestrator.
All module dependencies are injected via _modules dict — no real APIs, no Ollama.
"""
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from pipeline_orchestrator import PipelineOrchestrator

# ── Synthetic stage outputs ───────────────────────────────────────────────────

MOCK_ARTICLES = {
    "stock_news":    pd.DataFrame({"ticker": ["AAPL"], "title": ["Apple"], "composite_score": [10.0]}),
    "global_news":   pd.DataFrame(),
    "industry_news": pd.DataFrame(),
    "ticker_news":   pd.DataFrame(),
}

MOCK_SUMMARY = {
    "market_bias": "bullish", "top_themes": ["AI"], "key_risks": ["tariff"],
    "article_count": 45, "window_start": "2026-03-12", "window_end": "2026-03-19",
}

MOCK_MACRO = {
    "market_bias": "bullish", "favored_sectors": ["Technology"],
    "avoid_sectors": [], "active_macro_risks": ["tariff"], "reasoning": "Tech strong.",
}

MOCK_TOP50 = pd.DataFrame({"ticker": ["AAPL"], "composite_score": [10.0]})

MOCK_FEATURES = {
    "return_20d": 0.05, "rsi_14": 55.0, "atr_14": 3.2,
    "atr_pct": 0.014, "52w_high_prox": 0.95, "52w_low_prox": 1.4,
    "volume_ratio_30d": 1.2,
}

MOCK_SHORTLISTED = ["AAPL"]

MOCK_VERDICTS = [{"ticker": "AAPL", "verdict": "buy", "reasoning": "Strong setup."}]

MOCK_REGIMES = [{"ticker": "AAPL", "regime": "Trending", "hurst": 0.68, "atr_pct": 0.022}]

MOCK_STRATEGY = {
    "ticker": "AAPL", "strategy": "Momentum", "regime": "Trending",
    "base_params": {"entry_lookback": 20},
    "adjusted_params": {
        "entry_lookback": 20, "volume_multiplier": 1.5,
        "trailing_stop_atr": 2.0, "ma_exit_period": 10,
        "stop_loss_atr": 1.5, "max_holding_days": 20,
    },
    "llm_adjustments": [], "reasoning": "Momentum aligned.",
}

MOCK_BACKTEST = {
    "ticker": "AAPL", "strategy": "Momentum",
    "trade_log":    [],
    "equity_curve": pd.Series([100_000.0] * 252),
    "returns":      pd.Series([0.0] * 252),
    "summary":      {"total_return": 0.05, "trade_count": 5, "win_rate": 0.6},
}

MOCK_DIAGNOSTIC = {
    "ticker": "AAPL", "strategy": "Momentum",
    "passed": True, "reject_reason": None,
    "metrics": {
        "sharpe": 1.2, "max_drawdown": 0.12,
        "win_rate": 0.6, "walk_forward_degradation": 0.2, "trade_count": 5,
    },
    "llm_commentary": None,
}

MOCK_REPORT_PATH = "/tmp/report_2026-03-19_083000.md"

CONFIG = {
    "benzinga_api_key": "test_key",
    "llm_client":       MagicMock(return_value="{}"),
    "output_dir":       "/tmp/reports",
    "cache_dir":        "/tmp/cache",
    "initial_portfolio": 100_000.0,
    "window_days":      7,
}

RESULT_REQUIRED_KEYS = {
    "report_path", "run_date", "summary", "macro",
    "ticker_verdicts", "regimes", "strategies", "diagnostics", "backtests",
}


def make_mock_modules(report_path: str = MOCK_REPORT_PATH) -> dict:
    mods = {k: MagicMock() for k in (
        "collector", "summarizer", "macro_screener", "screener",
        "fetcher", "classifier", "selector", "diagnostics", "backtester", "reporter",
    )}
    mods["collector"].collect_range.return_value      = MOCK_ARTICLES
    mods["summarizer"].summarize.return_value         = MOCK_SUMMARY
    mods["macro_screener"].screen.return_value        = MOCK_MACRO
    mods["screener"].prefilter.return_value           = MOCK_TOP50
    mods["screener"].shortlist.return_value           = MOCK_SHORTLISTED
    mods["screener"].screen_tickers.return_value      = MOCK_VERDICTS
    mods["fetcher"].fetch.return_value                = {"AAPL": MagicMock(name="ohlcv_aapl")}
    mods["fetcher"].compute_features.return_value     = MOCK_FEATURES
    mods["classifier"].classify_all.return_value      = MOCK_REGIMES
    mods["selector"].select.return_value              = MOCK_STRATEGY
    mods["backtester"].run.return_value               = MOCK_BACKTEST
    mods["diagnostics"].run.return_value              = MOCK_DIAGNOSTIC
    mods["reporter"].generate.return_value            = report_path
    return mods


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_result_has_all_required_keys(self):
        orch = PipelineOrchestrator(CONFIG, _modules=make_mock_modules())
        r = orch.run("2026-03-19")
        assert RESULT_REQUIRED_KEYS == set(r.keys())

    def test_report_path_is_string(self):
        orch = PipelineOrchestrator(CONFIG, _modules=make_mock_modules())
        r = orch.run("2026-03-19")
        assert isinstance(r["report_path"], str)

    def test_run_date_matches_input(self):
        orch = PipelineOrchestrator(CONFIG, _modules=make_mock_modules())
        r = orch.run("2026-03-19")
        assert r["run_date"] == "2026-03-19"

    def test_backtests_is_list(self):
        orch = PipelineOrchestrator(CONFIG, _modules=make_mock_modules())
        r = orch.run("2026-03-19")
        assert isinstance(r["backtests"], list)

    def test_diagnostics_is_list(self):
        orch = PipelineOrchestrator(CONFIG, _modules=make_mock_modules())
        r = orch.run("2026-03-19")
        assert isinstance(r["diagnostics"], list)


# ── Cycle 2: date handling ────────────────────────────────────────────────────

class TestDateHandling:
    def test_default_date_is_yesterday_ph(self):
        tz_ph     = timezone(timedelta(hours=8))
        yesterday = (datetime.now(tz_ph) - timedelta(days=1)).strftime("%Y-%m-%d")
        orch = PipelineOrchestrator(CONFIG, _modules=make_mock_modules())
        r = orch.run()   # no date arg
        assert r["run_date"] == yesterday

    def test_collect_range_end_date_matches_run_date(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        _, end = mods["collector"].collect_range.call_args[0]
        assert end == "2026-03-19"

    def test_collect_range_window_is_90_days(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        start, end = mods["collector"].collect_range.call_args[0]
        dt_start = datetime.strptime(start, "%Y-%m-%d")
        dt_end   = datetime.strptime(end,   "%Y-%m-%d")
        assert (dt_end - dt_start).days == 90


# ── Cycle 3: stage invocation ─────────────────────────────────────────────────

class TestStageInvocation:
    def test_collector_collect_range_called_once(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        mods["collector"].collect_range.assert_called_once()

    def test_summarizer_called_once(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        mods["summarizer"].summarize.assert_called_once()

    def test_macro_screener_called_once(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        mods["macro_screener"].screen.assert_called_once()

    def test_reporter_called_once(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        mods["reporter"].generate.assert_called_once()

    def test_backtester_called_once_per_regime(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert mods["backtester"].run.call_count == len(MOCK_REGIMES)

    def test_diagnostics_called_once_per_backtest(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert mods["diagnostics"].run.call_count == len(MOCK_REGIMES)


# ── Cycle 4: data flow ────────────────────────────────────────────────────────

class TestDataFlow:
    def test_summarizer_receives_articles_from_collector(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        articles_passed = mods["summarizer"].summarize.call_args[0][0]
        assert articles_passed == MOCK_ARTICLES

    def test_summarizer_receives_run_date_as_of_date(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        kwargs = mods["summarizer"].summarize.call_args[1]
        assert kwargs.get("as_of_date") == "2026-03-19"

    def test_macro_screener_receives_summary(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        summary_passed = mods["macro_screener"].screen.call_args[0][0]
        assert summary_passed == MOCK_SUMMARY

    def test_reporter_receives_pipeline_output_with_required_keys(self):
        mods = make_mock_modules()
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        po = mods["reporter"].generate.call_args[0][0]
        for key in ("run_date", "macro", "ticker_verdicts", "regimes",
                    "strategies", "diagnostics", "backtests"):
            assert key in po

    def test_result_report_path_matches_reporter_output(self):
        mods = make_mock_modules(report_path="/tmp/my_report.md")
        r = PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert r["report_path"] == "/tmp/my_report.md"


# ── Cycle 5: stage failure handling ──────────────────────────────────────────

class TestStageFailure:
    def test_collector_failure_does_not_crash(self):
        mods = make_mock_modules()
        mods["collector"].collect_range.side_effect = Exception("API unavailable")
        r = PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert "report_path" in r

    def test_summarizer_failure_does_not_crash(self):
        mods = make_mock_modules()
        mods["summarizer"].summarize.side_effect = Exception("LLM timeout")
        r = PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert "report_path" in r

    def test_backtester_failure_does_not_crash(self):
        mods = make_mock_modules()
        mods["backtester"].run.side_effect = Exception("Insufficient data")
        r = PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert r["run_date"] == "2026-03-19"

    def test_no_tickers_found_returns_result_with_empty_lists(self):
        mods = make_mock_modules()
        mods["screener"].prefilter.return_value = pd.DataFrame()   # empty
        r = PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        assert "report_path" in r
        assert r["ticker_verdicts"] == []

    def test_reporter_always_called_even_after_upstream_failures(self):
        mods = make_mock_modules()
        mods["classifier"].classify_all.side_effect = Exception("Classify failed")
        PipelineOrchestrator(CONFIG, _modules=mods).run("2026-03-19")
        mods["reporter"].generate.assert_called_once()
