"""
PipelineOrchestrator
====================
Runs the full alpha-finder pipeline end-to-end.

Stage order
-----------
1.  collect_range(start, end)       → articles dict
2.  summarize(articles, as_of_date) → market summary
3.  screen(summary)                 → macro analysis
4.  prefilter(articles)             → top-50 DataFrame
5.  fetch(tickers)                  → OHLCV dict
6.  compute_features(df)            → feature dict (per ticker)
7.  shortlist(tickers, features, macro) → shortlisted tickers
8.  screen_tickers(tickers, …)      → verdicts list
9.  classify_all(tickers, ohlcv)    → regime list
10. (per regime) select + backtest + diagnostics
11. generate(pipeline_output)       → report path

Dependency injection
--------------------
Pass ``_modules`` (a dict of MagicMocks or real instances) to override
the default module construction. This is the sole hook used by tests.

Public interface
----------------
  orch   = PipelineOrchestrator(config)
  result = orch.run("2026-03-19")   # or orch.run() for yesterday UTC+8
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _yesterday() -> str:
    tz_ph = timezone(timedelta(hours=8))
    return (datetime.now(tz_ph) - timedelta(days=1)).strftime("%Y-%m-%d")


def _subtract_days(date_str: str, days: int) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (dt - timedelta(days=days)).strftime("%Y-%m-%d")


# ── orchestrator ──────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    def __init__(self, config: dict, _modules: dict | None = None):
        self._cfg = config
        self._modules = _modules if _modules is not None else self._build_modules()

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, date: str | None = None) -> dict:
        run_date = date if date is not None else _yesterday()
        start    = _subtract_days(run_date, 90)

        m = self._modules

        # ── Stage 1: collect ──────────────────────────────────────────────────
        articles = self._safe(
            "collector",
            lambda: m["collector"].collect_range(start, run_date),
            {},
        )

        # ── Stage 2: summarize ────────────────────────────────────────────────
        summary = self._safe(
            "summarizer",
            lambda: m["summarizer"].summarize(articles, as_of_date=run_date),
            {},
        )

        # ── Stage 3: macro screen ─────────────────────────────────────────────
        macro = self._safe(
            "macro_screener",
            lambda: m["macro_screener"].screen(summary),
            {},
        )

        # ── Stage 4: prefilter ────────────────────────────────────────────────
        top50 = self._safe(
            "screener.prefilter",
            lambda: m["screener"].prefilter(articles),
            pd.DataFrame(),
        )

        # Early exit: no tickers found
        if top50 is None or (isinstance(top50, pd.DataFrame) and top50.empty):
            return self._finish(
                m, run_date, summary, macro,
                ticker_verdicts=[], regimes=[], strategies=[], diagnostics=[], backtests=[],
            )

        tickers = top50["ticker"].tolist()

        # ── Stage 5: fetch OHLCV ──────────────────────────────────────────────
        ohlcv_raw = self._safe(
            "fetcher.fetch",
            lambda: m["fetcher"].fetch(tickers),
            {},
        )

        # ── Stage 6: compute features ─────────────────────────────────────────
        features: dict[str, Any] = {}
        for ticker in tickers:
            df = (ohlcv_raw or {}).get(ticker)
            if df is not None:
                features[ticker] = self._safe(
                    f"fetcher.compute_features({ticker})",
                    lambda _df=df: m["fetcher"].compute_features(_df),
                    {},
                )

        # ── Stage 7: shortlist ────────────────────────────────────────────────
        shortlisted = self._safe(
            "screener.shortlist",
            lambda: m["screener"].shortlist(top50, macro, features),
            tickers,
        )
        shortlisted = shortlisted or tickers
        max_tickers = self._cfg.get("max_tickers", 15)
        shortlisted = shortlisted[:max_tickers]

        # ── Stage 8: verdicts ─────────────────────────────────────────────────
        ticker_verdicts = self._safe(
            "screener.screen_tickers",
            lambda: m["screener"].screen_tickers(shortlisted, macro, features),
            [],
        )

        # ── Stage 9: regime classification ───────────────────────────────────
        ohlcv_shortlisted = {t: (ohlcv_raw or {}).get(t) for t in shortlisted}
        regimes = self._safe(
            "classifier.classify_all",
            lambda: m["classifier"].classify_all(ohlcv_shortlisted),
            [],
        )

        # ── Stage 10: per-ticker strategy / backtest / diagnostics / MC ─────
        strategies:    list[dict] = []
        backtests:     list[dict] = []
        diagnostics:   list[dict] = []
        monte_carlos:  list[dict] = []

        for regime in (regimes or []):
            ticker  = regime["ticker"]
            ohlcv   = (ohlcv_raw or {}).get(ticker)
            feats   = features.get(ticker, {})

            strategy = self._safe(
                f"selector.select({ticker})",
                lambda _t=ticker, _r=regime, _f=feats: m["selector"].select(_t, _r, _f, macro),
                None,
            )

            # Attach current signal status so the report can show live entry conditions
            if strategy and ohlcv is not None:
                _portfolio = self._cfg.get("initial_portfolio", 100_000.0)
                sig = self._safe(
                    f"backtester.signal_status({ticker})",
                    lambda _s=strategy, _o=ohlcv, _p=_portfolio: m["backtester"].signal_status(
                        _s["strategy"], _o, _s["adjusted_params"], _p
                    ),
                    {"signal_active": None, "details": "N/A", "setup": None},
                )
                strategy = {**strategy, "current_signal": sig}

            if strategy:
                strategies.append(strategy)

            backtest = self._safe(
                f"backtester.run({ticker})",
                lambda _t=ticker, _s=strategy, _o=ohlcv: m["backtester"].run(_t, _s, _o),
                None,
            )
            if backtest:
                backtests.append(backtest)

            diagnostic = self._safe(
                f"diagnostics.run({ticker})",
                lambda _t=ticker, _s=strategy, _bt=backtest: m["diagnostics"].run(
                    _t,
                    _s["strategy"] if isinstance(_s, dict) else _s,
                    _bt["trade_log"],
                    _bt["returns"],
                ) if _bt else None,
                None,
            )
            if diagnostic:
                diagnostics.append(diagnostic)

            # Monte Carlo — only for strategies that passed diagnostics
            if diagnostic and diagnostic.get("passed") and backtest:
                portfolio = self._cfg.get("initial_portfolio", 100_000.0)
                mc_result = self._safe(
                    f"monte_carlo.run({ticker})",
                    lambda _bt=backtest, _p=portfolio: m["monte_carlo"].run(
                        _bt["trade_log"], _p
                    ),
                    None,
                )
                if mc_result:
                    monte_carlos.append({"ticker": ticker, **mc_result})

        return self._finish(
            m, run_date, summary, macro,
            ticker_verdicts=ticker_verdicts or [],
            regimes=regimes or [],
            strategies=strategies,
            diagnostics=diagnostics,
            backtests=backtests,
            monte_carlos=monte_carlos,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe(stage: str, fn, fallback):
        try:
            return fn()
        except Exception as exc:
            return fallback

    @staticmethod
    def _finish(m, run_date, summary, macro, **kwargs) -> dict:
        pipeline_output = {
            "run_date":        run_date,
            "summary":         summary,
            "macro":           macro,
            **kwargs,
        }
        report_path = m["reporter"].generate(pipeline_output)
        return {
            "report_path": report_path,
            "run_date":    run_date,
            "summary":     summary,
            "macro":       macro,
            **kwargs,
        }

    def _build_modules(self) -> dict:
        """Construct real module instances from config (used in production)."""
        cfg = self._cfg
        llm = cfg["llm_client"]

        from Stage1DataCollector import Stage1DataCollector
        from news_summarizer      import NewsSummarizer
        from macro_screener       import MacroScreener
        from ticker_screener      import TickerScreener
        from ohlcv_fetcher        import OHLCVFetcher
        from regime_classifier    import RegimeClassifier
        from strategy_selector    import StrategySelector
        from backtester           import Backtester
        from diagnostics_engine   import DiagnosticsEngine
        from monte_carlo_engine   import MonteCarloEngine
        from report_generator     import ReportGenerator

        return {
            "collector":     Stage1DataCollector(
                                api_key=cfg["benzinga_api_key"],
                                cache_dir=cfg.get("cache_dir", "data/cache"),
                             ),
            "summarizer":    NewsSummarizer(
                                llm_client=llm,
                                window_days=cfg.get("window_days", 7),
                             ),
            "macro_screener": MacroScreener(llm_client=llm),
            "screener":      TickerScreener(llm_client=llm),
            "fetcher":       OHLCVFetcher(),
            "classifier":    RegimeClassifier(),
            "selector":      StrategySelector(llm_client=llm),
            "backtester":    Backtester(
                                initial_portfolio=cfg.get("initial_portfolio", 100_000.0)
                             ),
            "diagnostics":   DiagnosticsEngine(llm_client=llm),
            "monte_carlo":   MonteCarloEngine(
                                n_simulations=cfg.get("mc_simulations", 10_000),
                                ruin_threshold=cfg.get("mc_ruin_threshold", 0.40),
                             ),
            "reporter":      ReportGenerator(
                                output_dir=cfg.get("output_dir", "reports")
                             ),
        }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    import ollama
    import os
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="MFT Alpha Finder pipeline")
    parser.add_argument("date",          nargs="?",  default=None,  help="Run date YYYY-MM-DD (default: yesterday UTC+8)")
    parser.add_argument("--days",        type=int,   default=7,     help="News summary window in days (default: 7, max: 14)")
    parser.add_argument("--max-tickers", type=int,   default=15,    help="Max tickers to fully analyse (default: 15)")
    args = parser.parse_args()

    def llm(prompt: str) -> str:
        resp = ollama.chat(model="qwen3:8b", messages=[{"role": "user", "content": prompt}])
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    config = {
        "benzinga_api_key":  os.getenv("BENZINGA_API"),
        "llm_client":        llm,
        "output_dir":        "reports",
        "cache_dir":         "data/cache",
        "initial_portfolio": 100_000.0,
        "window_days":       min(args.days, 14),
        "max_tickers":       args.max_tickers,
    }

    date = args.date

    result = PipelineOrchestrator(config).run(date)

    print(f"\nReport : {result['report_path']}")
    print(f"Date   : {result['run_date']}")
    print(f"Bias   : {result['summary'].get('market_bias', 'n/a')}")
    print(f"Tickers: {len(result['ticker_verdicts'])} verdicts | "
          f"{len(result['backtests'])} backtests | "
          f"{len(result['diagnostics'])} diagnostics | "
          f"{len(result['monte_carlos'])} MC sims")
