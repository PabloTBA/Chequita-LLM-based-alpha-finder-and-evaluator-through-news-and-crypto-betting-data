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

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ── sector map for diversity enforcement ──────────────────────────────────────

_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "AMD": "Technology",
    "QCOM": "Technology", "AMAT": "Technology", "LRCX": "Technology",
    "KLAC": "Technology", "ADI": "Technology", "MU": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "TXN": "Technology",
    "INTU": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "NOW": "Technology", "PANW": "Technology", "CRWD": "Technology",
    "SNOW": "Technology", "PLTR": "Technology", "ACN": "Technology",
    # Communication
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication",
    "NFLX": "Communication", "T": "Communication", "DIS": "Communication",
    "BIDU": "Communication",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
    "UBER": "Consumer Discretionary", "RBLX": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary", "LCID": "Consumer Discretionary",
    "SHOP": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "MDLZ": "Consumer Staples",
    "MO": "Consumer Staples", "PM": "Consumer Staples", "CL": "Consumer Staples",
    # Healthcare
    "LLY": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "AMGN": "Healthcare", "ISRG": "Healthcare",
    "SYK": "Healthcare", "GILD": "Healthcare", "VRTX": "Healthcare",
    "BSX": "Healthcare", "REGN": "Healthcare", "MDT": "Healthcare",
    "EW": "Healthcare", "ZTS": "Healthcare", "BDX": "Healthcare",
    "HUM": "Healthcare", "HCA": "Healthcare", "JNJ": "Healthcare",
    # Financials
    "JPM": "Financials", "BRK.B": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "SPGI": "Financials",
    "AXP": "Financials", "C": "Financials", "SCHW": "Financials",
    "CB": "Financials", "MMC": "Financials", "AON": "Financials",
    "MCO": "Financials", "ICE": "Financials", "CME": "Financials",
    "USB": "Financials", "AIG": "Financials", "FI": "Financials",
    "PYPL": "Financials", "SQ": "Financials", "SOFI": "Financials",
    "COIN": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "FCX": "Energy",
    # Industrials
    "GE": "Industrials", "HON": "Industrials", "CAT": "Industrials",
    "ETN": "Industrials", "RTX": "Industrials", "NOC": "Industrials",
    "DE": "Industrials", "ITW": "Industrials", "EMR": "Industrials",
    "NSC": "Industrials", "GD": "Industrials", "BA": "Industrials",
    "MMM": "Industrials", "PH": "Industrials", "ROP": "Industrials",
    # Materials
    "LIN": "Materials", "SHW": "Materials",
    # Real Estate
    "PLD": "Real Estate", "PSA": "Real Estate",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities",
    # Speculative / Other
    "NIO": "Speculative",
}

_MAX_PER_SECTOR = 2  # hard cap on tickers per sector in the shortlist


def _enforce_sector_diversity(tickers: list[str], max_per_sector: int = _MAX_PER_SECTOR) -> list[str]:
    """
    Cap tickers per GICS sector to avoid concentrated bets.
    Unknown tickers (not in _SECTOR_MAP) are treated as their own sector
    so they always pass through.
    """
    sector_count: dict[str, int] = {}
    result: list[str] = []
    for ticker in tickers:
        sector = _SECTOR_MAP.get(ticker)
        if sector is None:
            result.append(ticker)   # unknown — let it through
            continue
        if sector_count.get(sector, 0) < max_per_sector:
            result.append(ticker)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        else:
            print(f"  [Diversity] {ticker} dropped — {sector} sector already has {max_per_sector} tickers")
    return result


# ── parameter validation alternatives (tried if LLM params yield Sharpe < 0) ──

_PARAM_ALTERNATIVES: dict[str, list[dict]] = {
    "Momentum": [
        # Conservative: wider stops, longer lookback, lower volume bar
        {"entry_lookback": 30, "volume_multiplier": 1.3, "trailing_stop_atr": 2.5,
         "ma_exit_period": 20, "stop_loss_atr": 2.0, "max_holding_days": 30},
        # Aggressive entry filter: higher volume confirmation, tighter MA exit
        {"entry_lookback": 15, "volume_multiplier": 2.0, "trailing_stop_atr": 1.5,
         "ma_exit_period": 15, "stop_loss_atr": 1.0, "max_holding_days": 15},
    ],
    "Mean-Reversion": [
        # Deeper oversold, wider BB — enters on more extreme dislocations only
        {"rsi_entry_threshold": 25, "rsi_exit_threshold": 60, "bb_period": 20,
         "bb_std": 2.5, "stop_loss_atr": 2.0, "max_holding_days": 15},
        # Quicker cycle: tighter BB, earlier RSI exit
        {"rsi_entry_threshold": 30, "rsi_exit_threshold": 50, "bb_period": 15,
         "bb_std": 1.8, "stop_loss_atr": 1.5, "max_holding_days": 8},
    ],
}


def _quick_sharpe(returns: pd.Series) -> float:
    """Fast annualised Sharpe estimate (no risk-free rate) for param comparison."""
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * (252 ** 0.5))


# ── helpers ───────────────────────────────────────────────────────────────────

def _yesterday() -> str:
    tz_ph = timezone(timedelta(hours=8))
    return (datetime.now(tz_ph) - timedelta(days=1)).strftime("%Y-%m-%d")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
        run_date     = date if date is not None else _yesterday()
        start        = _subtract_days(run_date, 90)
        collect_end  = _today_utc()   # include today's UTC articles even if run_date is yesterday

        m = self._modules

        # ── Stage 1: collect ──────────────────────────────────────────────────
        print(f"\n[Stage 1] Collecting news {start} → {collect_end} ...")
        articles = self._safe(
            "collector",
            lambda: m["collector"].collect_range(start, collect_end),
            {},
        )

        # ── Stage 1b: RAG insert news ─────────────────────────────────────────
        print("[Stage 1b] Inserting news into RAG store ...")
        self._safe(
            "rag_store.insert_news",
            lambda: m["rag_store"].insert_news(articles),
            None,
        )

        # ── Stage 1c: fetch + embed prediction markets ────────────────────────
        print("[Stage 1c] Fetching prediction markets ...")
        markets = self._safe(
            "market_client.fetch",
            lambda: m["market_client"].fetch(collect_end),
            [],
        ) or []

        # ── Stage 2: summarize ────────────────────────────────────────────────
        print("[Stage 2] Summarizing news ...")
        summary = self._safe(
            "summarizer",
            lambda: m["summarizer"].summarize(articles, as_of_date=run_date, markets=markets),
            {},
        )

        # ── Stage 3: macro screen ─────────────────────────────────────────────
        print("[Stage 3] Running macro screen ...")
        macro = self._safe(
            "macro_screener",
            lambda: m["macro_screener"].screen(summary),
            {},
        )

        # ── Stage 4: prefilter ────────────────────────────────────────────────
        print("[Stage 4] Prefiltering tickers ...")
        top50 = self._safe(
            "screener.prefilter",
            lambda: m["screener"].prefilter(articles),
            pd.DataFrame(),
        )

        # Early exit: no tickers found
        if top50 is None or (isinstance(top50, pd.DataFrame) and top50.empty):
            return self._finish(
                m, run_date, summary, macro,
                markets=markets,
                ticker_verdicts=[], regimes=[], strategies=[], diagnostics=[], backtests=[],
                monte_carlos=[],
                execution_brief={"active_signals": [], "inactive_count": 0,
                                 "portfolio_risk": {"active_count": 0, "total_dollar_risk": 0.0, "pct_of_portfolio": 0.0},
                                 "warnings": []},
                spy_ohlcv=None,
                correlation_warnings=[],
            )

        tickers = top50["ticker"].tolist()

        # ── Stage 5: fetch OHLCV (tickers + SPY benchmark) ───────────────────
        print(f"[Stage 5] Fetching OHLCV for {len(tickers)} tickers + SPY benchmark ...")
        all_tickers = list(dict.fromkeys(tickers + ["SPY"]))  # deduplicate, preserve order
        ohlcv_raw = self._safe(
            "fetcher.fetch",
            lambda: m["fetcher"].fetch(all_tickers),
            {},
        )

        # ── Stage 5b: data integrity check ───────────────────────────────────
        if ohlcv_raw:
            close_map = {
                t: df["Close"].astype(float).round(2)
                for t, df in ohlcv_raw.items()
                if df is not None and "Close" in df.columns
            }
            ticker_list_check = list(close_map.keys())
            for i in range(len(ticker_list_check)):
                for j in range(i + 1, len(ticker_list_check)):
                    ta, tb = ticker_list_check[i], ticker_list_check[j]
                    sa, sb = close_map[ta].align(close_map[tb], join="inner")
                    if len(sa) > 10 and sa.equals(sb):
                        print(f"  [WARN] Data integrity: {ta} and {tb} have identical Close series — possible yfinance cache collision. {tb} will be dropped.")
                        ohlcv_raw[tb] = None

        # ── Stage 5c: earnings blackout dates ────────────────────────────────
        print(f"[Stage 5c] Fetching earnings blackout dates ...")
        for _t in tickers:
            if (ohlcv_raw or {}).get(_t) is not None:
                ohlcv_raw[_t] = self._safe(
                    f"fetcher.add_earnings_blackout({_t})",
                    lambda _ticker=_t, _df=ohlcv_raw[_t]: m["fetcher"].add_earnings_blackout(_ticker, _df),
                    ohlcv_raw[_t],
                )

        # ── Stage 6: compute features ─────────────────────────────────────────
        print(f"[Stage 6] Computing features ...")
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
        print("[Stage 7] Shortlisting tickers ...")
        shortlisted = self._safe(
            "screener.shortlist",
            lambda: m["screener"].shortlist(top50, macro, features),
            tickers,
        )
        shortlisted = shortlisted or tickers
        shortlisted = _enforce_sector_diversity(shortlisted)
        max_tickers = self._cfg.get("max_tickers", 15)
        shortlisted = shortlisted[:max_tickers]

        # ── Stage 8: verdicts ─────────────────────────────────────────────────
        print(f"[Stage 8] Screening {len(shortlisted)} tickers ...")
        ticker_verdicts = self._safe(
            "screener.screen_tickers",
            lambda: m["screener"].screen_tickers(
                shortlisted, macro, features, rag_store=m["rag_store"]
            ),
            [],
        )

        # ── Stage 9: regime classification (skip AVOID tickers) ──────────────
        avoid_set = {v["ticker"] for v in (ticker_verdicts or []) if v.get("verdict", "").lower() == "avoid"}
        actionable = [t for t in shortlisted if t not in avoid_set]
        if avoid_set:
            print(f"[Stage 9] Skipping {len(avoid_set)} AVOID ticker(s): {', '.join(sorted(avoid_set))}")
        print(f"[Stage 9] Classifying market regimes for {len(actionable)} ticker(s) ...")
        ohlcv_shortlisted = {t: (ohlcv_raw or {}).get(t) for t in actionable}
        regimes = self._safe(
            "classifier.classify_all",
            lambda: m["classifier"].classify_all(ohlcv_shortlisted),
            [],
        )

        # ── Stage 10: per-ticker strategy / backtest / diagnostics / MC ─────
        print(f"[Stage 10] Running per-ticker analysis for {len(regimes or [])} tickers in parallel ...")
        strategies:    list[dict] = []
        backtests:     list[dict] = []
        diagnostics:   list[dict] = []
        monte_carlos:  list[dict] = []

        portfolio = self._cfg.get("initial_portfolio", 100_000.0)

        def _analyse_ticker(regime: dict) -> dict:
            ticker = regime["ticker"]
            ohlcv  = (ohlcv_raw or {}).get(ticker)
            feats  = features.get(ticker, {})

            # AVOID gate: skip full analysis if LLM rated this ticker AVOID
            tv = next((v for v in (ticker_verdicts or []) if v["ticker"] == ticker), None)
            if tv and tv.get("verdict", "").lower() == "avoid":
                print(f"  [Stage 10] {ticker} — skipping strategy/backtest (AVOID verdict)")
                return {"ticker": ticker, "strategy": None, "backtest": None,
                        "diagnostic": None, "mc_result": None}

            print(f"  [Stage 10] {ticker} — strategy select / backtest / diagnostics ...")

            strategy = self._safe(
                f"selector.select({ticker})",
                lambda _t=ticker, _r=regime, _f=feats: m["selector"].select(_t, _r, _f, macro),
                None,
            )

            if strategy and ohlcv is not None:
                sig = self._safe(
                    f"backtester.signal_status({ticker})",
                    lambda _s=strategy, _o=ohlcv, _p=portfolio: m["backtester"].signal_status(
                        _s["strategy"], _o, _s["adjusted_params"], _p
                    ),
                    {"signal_active": None, "details": "N/A", "setup": None},
                )
                strategy = {**strategy, "current_signal": sig}

            # Use base_params for backtest to avoid circular LLM tuning —
            # LLM-adjusted params are only used for the live signal check above.
            backtest = self._safe(
                f"backtester.run({ticker})",
                lambda _t=ticker, _s=strategy, _o=ohlcv: m["backtester"].run(
                    _t, {**_s, "adjusted_params": _s.get("base_params", _s["adjusted_params"])}, _o
                ),
                None,
            )

            # ── Parameter validation loop ──────────────────────────────────────
            # If LLM params yield negative Sharpe, try predefined alternatives
            # and keep whichever produces the best Sharpe.
            if backtest and strategy:
                current_sharpe = _quick_sharpe(backtest["returns"])
                if current_sharpe < 0.0:
                    strat_type = strategy["strategy"]
                    for alt_params in _PARAM_ALTERNATIVES.get(strat_type, []):
                        alt_strategy = {
                            **strategy,
                            "adjusted_params": alt_params,
                            "llm_adjustments": strategy.get("llm_adjustments", []) + [
                                f"[Auto] alternative params tried — LLM params yielded Sharpe {current_sharpe:.3f}"
                            ],
                        }
                        alt_backtest = self._safe(
                            f"backtester.run({ticker})[alt]",
                            lambda _t=ticker, _s=alt_strategy, _o=ohlcv: m["backtester"].run(_t, _s, _o),
                            None,
                        )
                        if alt_backtest:
                            alt_sharpe = _quick_sharpe(alt_backtest["returns"])
                            if alt_sharpe > current_sharpe:
                                print(f"  [Stage 10] {ticker} — alt params improved Sharpe {current_sharpe:.3f} → {alt_sharpe:.3f}")
                                strategy       = alt_strategy
                                backtest       = alt_backtest
                                current_sharpe = alt_sharpe

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

            mc_result = None
            if backtest and diagnostic:
                diag_passed    = diagnostic.get("passed", False)
                diag_sharpe    = diagnostic.get("metrics", {}).get("sharpe", -999)
                near_threshold = (not diag_passed) and (0.0 < diag_sharpe < 0.5)
                should_run_mc  = diag_passed or near_threshold
                trade_count    = len(backtest.get("trade_log", []))

                if should_run_mc:
                    if trade_count < 30:
                        label = "PASS" if diag_passed else "STRESS"
                        print(f"  [Stage 10] {ticker} — MC skipped ({label}, only {trade_count} trades, need 30+)")
                        mc_result = {"insufficient_sample": True, "trade_count": trade_count,
                                     "stress_test": not diag_passed}
                    else:
                        label = "passed" if diag_passed else f"near-threshold Sharpe={diag_sharpe:.3f}"
                        print(f"  [Stage 10] {ticker} — running Monte Carlo ({label}, {trade_count} trades) ...")
                        _mc = self._safe(
                            f"monte_carlo.run({ticker})",
                            lambda _bt=backtest, _p=portfolio: m["monte_carlo"].run(_bt["trade_log"], _p),
                            None,
                        )
                        if _mc:
                            mc_result = {**_mc, "trade_count": trade_count,
                                         "stress_test": not diag_passed}

            return {
                "ticker":     ticker,
                "strategy":   strategy,
                "backtest":   backtest,
                "diagnostic": diagnostic,
                "mc_result":  mc_result,
            }

        ticker_order = [r["ticker"] for r in (regimes or [])]
        results_map: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_analyse_ticker, r): r["ticker"] for r in (regimes or [])}
            for future in as_completed(futures):
                res = future.result()
                results_map[res["ticker"]] = res

        for ticker in ticker_order:
            res = results_map.get(ticker, {})
            if res.get("strategy"):
                strategies.append(res["strategy"])
            if res.get("backtest"):
                backtests.append(res["backtest"])
            if res.get("diagnostic"):
                diagnostics.append(res["diagnostic"])
            if res.get("mc_result"):
                monte_carlos.append({"ticker": ticker, **res["mc_result"]})

        # ── Correlation warnings ──────────────────────────────────────────────
        # Flag pairs of shortlisted tickers with >0.95 return correlation (e.g. GOOG/GOOGL)
        correlation_warnings: list[str] = []
        try:
            close_data = {
                t: (ohlcv_raw or {}).get(t)["Close"].astype(float)
                for t in shortlisted
                if (ohlcv_raw or {}).get(t) is not None
            }
            if len(close_data) >= 2:
                ret_df = pd.DataFrame(close_data).pct_change().dropna()
                corr   = ret_df.corr()
                tlist  = list(corr.columns)
                for i in range(len(tlist)):
                    for j in range(i + 1, len(tlist)):
                        c = corr.iloc[i, j]
                        if c > 0.95:
                            correlation_warnings.append(
                                f"{tlist[i]} / {tlist[j]} — correlation {c:.2f} "
                                f"(near-identical exposure, running both doubles concentration risk)"
                            )
        except Exception:
            pass

        # ── Reconcile verdicts against diagnostics ────────────────────────────
        # A BUY verdict that has a failed (or missing) diagnostic is contradictory.
        # Downgrade such tickers to WATCH so the report is self-consistent.
        passed_tickers = {d["ticker"] for d in diagnostics if d.get("passed")}
        reconciled_verdicts = []
        for v in (ticker_verdicts or []):
            if v.get("verdict", "").lower() == "buy" and v["ticker"] not in passed_tickers:
                reconciled_verdicts.append({**v, "verdict": "watch",
                    "reasoning": f"[Downgraded from BUY — backtest diagnostic did not pass] {v.get('reasoning', '')}"})
            else:
                reconciled_verdicts.append(v)
        ticker_verdicts = reconciled_verdicts

        # ── Stage 11: execution brief ─────────────────────────────────────────
        print("[Stage 11] Building execution brief ...")
        execution_brief = self._safe(
            "execution_advisor.advise",
            lambda: m["execution_advisor"].advise(strategies),
            {"active_signals": [], "inactive_count": 0,
             "portfolio_risk": {"active_count": 0, "total_dollar_risk": 0.0, "pct_of_portfolio": 0.0},
             "warnings": []},
        )

        spy_ohlcv = (ohlcv_raw or {}).get("SPY")

        return self._finish(
            m, run_date, summary, macro,
            markets=markets,
            ticker_verdicts=ticker_verdicts or [],
            regimes=regimes or [],
            strategies=strategies,
            diagnostics=diagnostics,
            backtests=backtests,
            monte_carlos=monte_carlos,
            execution_brief=execution_brief,
            spy_ohlcv=spy_ohlcv,
            correlation_warnings=correlation_warnings,
            features=features,
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
        from datetime import datetime as _dt
        timestamp = _dt.now().strftime("%H%M%S")
        print("[Stage 12] Generating full report ...")
        report_path   = m["reporter"].generate(pipeline_output, timestamp=timestamp)
        print("[Stage 12] Generating trader summary report ...")
        summary_path  = m["reporter"].generate_summary(pipeline_output, timestamp=timestamp)
        return {
            "report_path":  report_path,
            "summary_path": summary_path,
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
        from monte_carlo_engine          import MonteCarloEngine
        from report_generator            import ReportGenerator
        from execution_advisor           import ExecutionAdvisor
        from rag_store                   import RAGStore
        from prediction_market_client    import PredictionMarketClient

        rag = RAGStore(persist_dir=cfg.get("chroma_dir", "data/chroma"))
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
            "execution_advisor": ExecutionAdvisor(
                                initial_portfolio=cfg.get("initial_portfolio", 100_000.0)
                             ),
            "rag_store":     rag,
            "market_client": PredictionMarketClient(
                                cache_dir=cfg.get("cache_dir", "data/cache"),
                                min_volume=cfg.get("market_min_volume", 100_000.0),
                                max_markets=cfg.get("market_max_markets", 50),
                                categories=cfg.get("market_categories",
                                                   ["Economics", "Politics", "Crypto"]),
                                rag_store=rag,
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
    parser.add_argument("--max-tickers", type=int,   default=15,       help="Max tickers to fully analyse (default: 15)")
    parser.add_argument("--min-volume",  type=float, default=10_000.0, help="Min prediction market volume in USD (default: 10000)")
    parser.add_argument("--max-markets", type=int,   default=50,       help="Max prediction markets to store and embed (default: 50)")
    args = parser.parse_args()

    def llm(prompt: str) -> str:
        resp = ollama.chat(model="qwen3:8b", messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.0})
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    config = {
        "benzinga_api_key":  os.getenv("BENZINGA_API"),
        "llm_client":        llm,
        "output_dir":        "reports",
        "cache_dir":         "data/cache",
        "initial_portfolio": 100_000.0,
        "window_days":       min(args.days, 14),
        "max_tickers":       args.max_tickers,
        "market_min_volume":  args.min_volume,
        "market_max_markets": args.max_markets,
    }

    date = args.date

    result = PipelineOrchestrator(config).run(date)

    print(f"\nReport  : {result['report_path']}")
    print(f"Summary : {result.get('summary_path', 'n/a')}")
    print(f"Date    : {result['run_date']}")
    print(f"Bias   : {result['summary'].get('market_bias', 'n/a')}")
    print(f"Tickers: {len(result['ticker_verdicts'])} verdicts | "
          f"{len(result['backtests'])} backtests | "
          f"{len(result['diagnostics'])} diagnostics | "
          f"{len(result['monte_carlos'])} MC sims")
