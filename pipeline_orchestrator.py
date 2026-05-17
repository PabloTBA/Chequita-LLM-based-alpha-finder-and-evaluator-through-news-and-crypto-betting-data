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
# Base map covers S&P 500 constituents.  _extend_sector_map() dynamically
# fills in missing tickers via yfinance so the diversity cap works across
# the full expanded universe, not just the hardcoded S&P 500 names.

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

# Tickers that are economic duplicates of a preferred ticker.
# If the preferred ticker is present, the alias is always dropped.
_TICKER_ALIASES: dict[str, str] = {
    "GOOG": "GOOGL",   # GOOG class-C and GOOGL class-A are >0.99 correlated — keep GOOGL
}


def _deduplicate_aliases(tickers: list[str]) -> list[str]:
    """Drop any ticker whose preferred alias is already in the list."""
    ticker_set = set(tickers)
    return [t for t in tickers
            if not (_TICKER_ALIASES.get(t) in ticker_set)]


def _extend_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    Build a merged sector map: base _SECTOR_MAP + yfinance lookups for unknown tickers.
    Returns a NEW dict — does NOT mutate _SECTOR_MAP.
    Unknown tickers that fail lookup are absent from the returned dict.
    """
    result = dict(_SECTOR_MAP)  # shallow copy
    missing = [t for t in tickers if t not in result]
    if not missing:
        return result

    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        try:
            ticker_objects = __import__("yfinance").Tickers(" ".join(batch))
            for t in batch:
                try:
                    info = ticker_objects.tickers.get(t)
                    if info and hasattr(info, "info"):
                        d = info.info or {}
                        sector = d.get("sector") or d.get("industry") or ""
                        if sector:
                            result[t] = str(sector)
                except Exception:
                    pass
            __import__("time").sleep(0.3)
        except Exception as e:
            print(f"  [SectorMap] Batch lookup failed: {e}")

    return result


def _enforce_sector_diversity(tickers: list[str], max_per_sector: int = _MAX_PER_SECTOR) -> list[str]:
    """
    Cap tickers per GICS sector to avoid concentrated bets.
    Dynamically extends the sector map for unknown tickers via yfinance.
    Tickers that still can't be resolved are treated as their own sector
    so they always pass through.
    """
    sector_map = _extend_sector_map(tickers)

    sector_count: dict[str, int] = {}
    result: list[str] = []
    for ticker in tickers:
        sector = sector_map.get(ticker)
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
    "VolatilityBreakout": [
        # Tighter squeeze — only enter the most compressed setups
        {"bb_period": 20, "squeeze_pct": 0.10, "squeeze_lookback": 5,
         "volume_mult": 1.5, "stop_loss_atr": 2.0, "trailing_stop_atr": 2.5, "max_holding_days": 15},
        # Looser squeeze, longer hold, higher volume bar
        {"bb_period": 20, "squeeze_pct": 0.30, "squeeze_lookback": 7,
         "volume_mult": 2.0, "stop_loss_atr": 2.5, "trailing_stop_atr": 3.0, "max_holding_days": 20},
    ],
    "MLSignal": [
        # Higher conviction required, wider stops, shorter hold
        {"ml_threshold": 0.65, "reversal_threshold": 0.45,
         "stop_loss_atr": 2.0, "trailing_stop_atr": 2.5, "max_holding_days": 7},
        # Looser entry, tighter stops — higher frequency
        {"ml_threshold": 0.55, "reversal_threshold": 0.38,
         "stop_loss_atr": 1.0, "trailing_stop_atr": 1.5, "max_holding_days": 15},
    ],
}


def _ts() -> str:
    """Return current time as HH:MM:SS for progress output."""
    return datetime.now().strftime("%H:%M:%S")

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

        # ── Meta-learning: read historical verdict outcomes ────────────────────
        # Computes per-regime/strategy pass rates from prior runs so the report
        # can flag combinations with historically poor OOS performance.
        meta_insights = PipelineOrchestrator._load_meta_insights()
        if meta_insights.get("insights"):
            print(f"  [MetaLearning] Loaded {meta_insights['total_runs']} historical runs — "
                  f"{len(meta_insights['insights'])} regime/strategy combinations tracked.")
        if meta_insights.get("warnings"):
            for w in meta_insights["warnings"]:
                print(f"  [MetaLearning] WARNING: {w}")

        m = self._modules

        # ── Stage 1: collect ──────────────────────────────────────────────────
        benzinga_key = self._cfg.get("benzinga_api_key")
        if not benzinga_key:
            print("[ERROR] Stage 1: BENZINGA_API is not set — news collection will be empty. "
                  "Set BENZINGA_API in your .env file.")

        print(f"\n[{_ts()}] [Stage 1] Collecting news {start} → {collect_end} ...")
        articles = self._safe(
            "collector",
            lambda: m["collector"].collect_range(start, collect_end),
            {},
        )

        # Count total articles across all sources/dates for early-warning
        total_articles = sum(
            len(df) for df in articles.values() if hasattr(df, "__len__")
        ) if articles else 0
        if total_articles == 0:
            print("[WARN] Stage 1: 0 articles collected. Check BENZINGA_API key, "
                  "network access, and cache directory. All downstream stages will "
                  "have no news context.")
        else:
            print(f"[{_ts()}] [Stage 1] Collected {total_articles} articles total.")

        # ── Stage 1c: Crucix OSINT enrichment ────────────────────────────────
        # Pulls real-time macro/geopolitical intelligence from the Crucix sidecar
        # (node server.mjs in Crucix-master/, port 3117).  Gracefully skipped when
        # Crucix is not running — the pipeline continues with Benzinga data only.
        crucix_snapshot  = None
        crucix_macro_ctx = {}
        _crucix          = None   # kept in scope so Stage 3 can call to_summary_text
        print(f"[{_ts()}] [Stage 1c] Checking Crucix OSINT sidecar ...")
        try:
            from crucix_adapter import CrucixAdapter
            _crucix = CrucixAdapter(
                base_url=self._cfg.get("crucix_url", "http://localhost:3117")
            )
            crucix_snapshot = _crucix.fetch()
            if crucix_snapshot:
                crucix_macro_ctx = _crucix.to_macro_context(crucix_snapshot)
                crucix_df        = _crucix.to_dataframe(crucix_snapshot)
                # articles is {source_type: DataFrame} — add Crucix as its own source
                # so RAG store and NewsSummarizer can iterate it like any other source.
                if not crucix_df.empty:
                    articles["crucix_osint"] = crucix_df
                print(f"[{_ts()}] [Stage 1c] Crucix: {len(crucix_df)} OSINT articles merged. "
                      f"Macro signals: {len(crucix_macro_ctx.get('cross_domain_risk_signals', []))}")
            else:
                print(f"[{_ts()}] [Stage 1c] Crucix not available — continuing without OSINT enrichment.")
        except Exception as _e:
            print(f"[{_ts()}] [Stage 1c] Crucix enrichment error ({_e}) — continuing.")

        # ── Stage 1b: RAG insert news ─────────────────────────────────────────
        print(f"[{_ts()}] [Stage 1b] Inserting news into RAG store ...")
        self._safe(
            "rag_store.insert_news",
            lambda: m["rag_store"].insert_news(articles),
            None,
        )

        # ── Stage 2: summarize ────────────────────────────────────────────────
        print(f"[{_ts()}] [Stage 2] Summarizing news ...")
        summary = self._safe(
            "summarizer",
            lambda: m["summarizer"].summarize(articles, as_of_date=run_date),
            {},
        )

        # ── Stage 3: macro screen ─────────────────────────────────────────────
        print(f"[{_ts()}] [Stage 3] Running macro screen ...")
        # Merge Crucix live macro data into the summary so MacroScreener has
        # real-time FRED, EIA, VIX, yield curve, and conflict signals.
        if crucix_macro_ctx:
            if isinstance(summary, dict):
                summary.setdefault("crucix_macro", crucix_macro_ctx)
            elif isinstance(summary, str) and crucix_snapshot and _crucix is not None:
                summary = summary + "\n\n" + _crucix.to_summary_text(crucix_snapshot)
        macro = self._safe(
            "macro_screener",
            lambda: m["macro_screener"].screen(summary),
            {},
        )

        # ── Stage 4: prefilter ────────────────────────────────────────────────
        print(f"[{_ts()}] [Stage 4] Prefiltering tickers ...")
        top50 = self._safe(
            "screener.prefilter",
            lambda: m["screener"].prefilter(articles),
            pd.DataFrame(),
        )

        # Early exit: no tickers found
        _empty_market_state = {"market_state": "Normal", "spy_atr_pct": 0.0,
                               "crisis_fraction": 0.0, "stress_fraction": 0.0,
                               "stable_days": 0, "recovery_score": 1.0}
        if top50 is None or (isinstance(top50, pd.DataFrame) and top50.empty):
            return self._finish(
                m, run_date, summary, macro,
                ticker_verdicts=[], regimes=[], strategies=[], diagnostics=[], backtests=[],
                monte_carlos=[],
                execution_brief={"active_signals": [], "inactive_count": 0,
                                 "portfolio_risk": {"active_count": 0, "total_dollar_risk": 0.0, "pct_of_portfolio": 0.0},
                                 "warnings": []},
                spy_ohlcv=None,
                correlation_warnings=[],
                meta_insights=meta_insights,
                market_state=_empty_market_state,
            )

        tickers = top50["ticker"].tolist()

        # max_markets cap: limit how many tickers proceed to OHLCV fetch
        max_markets = self._cfg.get("max_markets", 100)
        if len(tickers) > max_markets:
            tickers = tickers[:max_markets]
            top50 = top50.head(max_markets)
            print(f"  [MaxMarkets] Capped prefilter to top {max_markets} tickers")

        # ── Stage 5: fetch OHLCV (tickers + SPY benchmark) ───────────────────
        print(f"[{_ts()}] [Stage 5] Fetching OHLCV for {len(tickers)} tickers + SPY benchmark ...")
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
                    sa, sb = close_map[ta].align(close_map[tb], join="inner", axis=0)
                    if len(sa) > 10 and sa.equals(sb):
                        print(f"  [WARN] Data integrity: {ta} and {tb} have identical Close series — possible yfinance cache collision. {tb} will be dropped.")
                        ohlcv_raw[tb] = None

        # ── Stage 5c: earnings blackout dates ────────────────────────────────
        print(f"[{_ts()}] [Stage 5c] Fetching earnings blackout dates ...")
        for _t in tickers:
            if (ohlcv_raw or {}).get(_t) is not None:
                ohlcv_raw[_t] = self._safe(
                    f"fetcher.add_earnings_blackout({_t})",
                    lambda _ticker=_t, _df=ohlcv_raw[_t]: m["fetcher"].add_earnings_blackout(_ticker, _df),
                    ohlcv_raw[_t],
                )

        # ── Stage 5d: cross-sectional alpha signal enrichment ─────────────────
        # AlphaEngine computes CS-MR, residual reversion, volume-spike, and
        # short-term momentum signals across the full universe, then injects
        # alpha_signal into each OHLCV DataFrame.  The backtester reads this
        # column when running AlphaCombined strategy — no look-ahead because
        # AlphaEngine uses shift(1) throughout.
        print(f"[{_ts()}] [Stage 5d] Computing cross-sectional alpha signals ...")
        if ohlcv_raw:
            _bias = (macro or {}).get("market_bias", "neutral")
            ohlcv_raw = self._safe(
                "alpha_engine.compute",
                lambda: m["alpha_engine"].compute(dict(ohlcv_raw), market_bias=_bias),
                ohlcv_raw,
            )

        # ── Stage 5e: joint market+news crisis detector ──────────────────────
        # Systemic risk-off circuit breaker. Fires when BOTH conditions are
        # true simultaneously, producing a `joint_crisis` flag on the macro
        # dict that StrategySelector reads to force AlphaCombined with Crisis
        # parameters for every ticker (tighter stops, higher alpha threshold,
        # shorter holds) regardless of its individual Hurst/ATR regime.
        #
        #   1. SPY 14d ATR / SPY close > 6% (ATR_CRISIS threshold — a market
        #      event, not a single-stock event). Uses the same RegimeClassifier
        #      ATR math so the threshold stays in lockstep with per-ticker logic.
        #
        #   2. Macro screener bias == "bearish" AND ≥2 active_macro_risks. One
        #      bearish tweet doesn't trigger it — the news classifier must have
        #      flagged multiple independent systemic risks (e.g. Fed hike +
        #      credit-spread blowout + liquidity squeeze).
        #
        # Without this, the per-ticker Hurst regime would keep routing names
        # like AAPL into Momentum during a Lehman-style panic.
        try:
            from regime_classifier import RegimeClassifier, ATR_CRISIS
            _spy_df = (ohlcv_raw or {}).get("SPY")
            _market_atr_pct: float | None = None
            if _spy_df is not None and len(_spy_df) >= 30:
                _clf = RegimeClassifier()
                _high  = _spy_df["High"].astype(float).values
                _low   = _spy_df["Low"].astype(float).values
                _close = _spy_df["Close"].astype(float).values
                _atr   = _clf._atr(_high, _low, _close, period=14)
                if _close[-1] > 0:
                    _market_atr_pct = float(_atr / _close[-1])

            _bias           = str((macro or {}).get("market_bias", "neutral")).lower()
            _active_risks   = list((macro or {}).get("active_macro_risks", []) or [])
            _market_crisis  = (_market_atr_pct is not None) and (_market_atr_pct > ATR_CRISIS)
            _news_crisis    = (_bias == "bearish") and (len(_active_risks) >= 2)
            _joint_crisis   = bool(_market_crisis and _news_crisis)

            if isinstance(macro, dict):
                macro["joint_crisis"]    = _joint_crisis
                macro["market_atr_pct"]  = _market_atr_pct
                macro["market_crisis"]   = _market_crisis
                macro["news_crisis"]     = _news_crisis

            _mkt_str = f"{_market_atr_pct:.2%}" if _market_atr_pct is not None else "n/a"
            if _joint_crisis:
                print(
                    f"[{_ts()}] [Stage 5e] JOINT CRISIS DETECTED — "
                    f"SPY ATR%={_mkt_str} (>6%) AND bearish macro with "
                    f"{len(_active_risks)} active risks. Forcing AlphaCombined "
                    f"+ Crisis params for all tickers."
                )
            else:
                print(
                    f"[{_ts()}] [Stage 5e] Joint crisis check: "
                    f"market_crisis={_market_crisis} (SPY ATR%={_mkt_str}), "
                    f"news_crisis={_news_crisis} (bias={_bias}, "
                    f"risks={len(_active_risks)}) — normal routing."
                )
        except Exception as _e:
            print(f"[{_ts()}] [Stage 5e] Joint crisis detector failed: {_e} — defaulting to no override")
            if isinstance(macro, dict):
                macro.setdefault("joint_crisis", False)

        # ── Stage 6: compute features ─────────────────────────────────────────
        print(f"[{_ts()}] [Stage 6] Computing features ...")
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
        print(f"[{_ts()}] [Stage 7] Shortlisting tickers ...")
        shortlisted = self._safe(
            "screener.shortlist",
            lambda: m["screener"].shortlist(top50, macro, features),
            tickers,
        )
        shortlisted = shortlisted or tickers
        shortlisted = _deduplicate_aliases(shortlisted)
        shortlisted = _enforce_sector_diversity(shortlisted)
        max_tickers = self._cfg.get("max_tickers", 15)

        # ── Stage 6b: volume filter ──────────────────────────────────────────────
        min_volume = self._cfg.get("min_volume", 0)
        if min_volume > 0 and features:
            dropped = []
            for t in list(features):
                adv = features[t].get("adv_20d", 0)
                if adv < min_volume:
                    dropped.append(t)
                    del features[t]
                    if t in tickers:
                        tickers.remove(t)
            if dropped:
                print(f"  [Volume] Dropped {len(dropped)} ticker(s) below {min_volume:,} ADV: {dropped}")
            # Also trim top50 so backfill candidates respect volume filter
            if top50 is not None and not top50.empty and dropped:
                top50 = top50[~top50["ticker"].isin(dropped)]

        # Backfill: if shortlist is below max_tickers, pull the next-best
        # candidates from top50 that are not already shortlisted.
        if len(shortlisted) < max_tickers and top50 is not None and not top50.empty:
            shortlisted_set = set(shortlisted)
            candidates = [
                t for t in top50["ticker"].tolist()
                if t not in shortlisted_set
            ]
            combined = _deduplicate_aliases(shortlisted + candidates)
            shortlisted = combined[:max_tickers]
            added = [t for t in shortlisted if t not in shortlisted_set]
            if added:
                print(f"  [Backfill] Added {len(added)} ticker(s) from top-50 reserve: {added}")

        shortlisted = shortlisted[:max_tickers]

        # ── Stage 7b: ML signal enrichment (shortlisted tickers only) ────────
        # Run after shortlisting so the cross-sectional GBM (Model 1) gets
        # meaningful z-scores across the ~15 shortlisted tickers rather than
        # all 50 raw candidates.  Only Low-Volatility and Neutral regime tickers
        # will actually consume ml_signal, but we need all shortlisted tickers
        # in the panel now so cross-sectional features are well-calibrated.
        if ohlcv_raw and shortlisted:
            ml_subset = {t: ohlcv_raw[t] for t in shortlisted if t in ohlcv_raw}
            print(f"[{_ts()}] [Stage 7b] Computing ML signals for {len(ml_subset)} shortlisted ticker(s) ...")
            enriched = self._safe(
                "ml_signal_engine.compute",
                lambda: m["ml_signal_engine"].compute(ml_subset),
                ml_subset,
            )
            ohlcv_raw.update(enriched)

        # ── Stage 8: verdicts ─────────────────────────────────────────────────
        print(f"[{_ts()}] [Stage 8] Screening {len(shortlisted)} tickers ...")
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
            print(f"[{_ts()}] [Stage 9] Skipping {len(avoid_set)} AVOID ticker(s): {', '.join(sorted(avoid_set))}")
        print(f"[{_ts()}] [Stage 9] Classifying market regimes for {len(actionable)} ticker(s) ...")
        ohlcv_shortlisted = {t: (ohlcv_raw or {}).get(t) for t in actionable}
        regimes = self._safe(
            "classifier.classify_all",
            lambda: m["classifier"].classify_all(ohlcv_shortlisted),
            [],
        )

        # ── Stage 9c: portfolio-level market state detection ─────────────────
        # Determines whether the whole market is in Crisis, High-Volatility, or
        # Normal mode.  This single value drives four downstream adaptations:
        #   1. No-trade gate  : Crisis tickers are skipped entirely (no recommendations).
        #   2. Holding cap    : max_holding_days capped at 5 (Crisis) or 10 (High-Vol).
        #   3. Momentum gate  : 12-1m gate disabled in Crisis (calibrated for trending mkts).
        #   4. Trade-count floor: Crisis floor = 10 trades instead of 30.
        print(f"[{_ts()}] [Stage 9c] Detecting portfolio-level market state ...")
        try:
            from regime_classifier import MarketStateDetector as _MSD
            _spy_df    = (ohlcv_raw or {}).get("SPY")
            _msd       = _MSD()
            market_state_info = self._safe(
                "market_state_detector.detect",
                lambda: _msd.detect(_spy_df, regimes or []),
                {"market_state": "Normal", "spy_atr_pct": 0.0,
                 "crisis_fraction": 0.0, "stress_fraction": 0.0,
                 "stable_days": 0, "recovery_score": 1.0},
            )
        except Exception as _e:
            print(f"[{_ts()}] [Stage 9c] MarketStateDetector unavailable ({_e}), defaulting to Normal")
            market_state_info = {"market_state": "Normal", "spy_atr_pct": 0.0,
                                 "crisis_fraction": 0.0, "stress_fraction": 0.0,
                                 "stable_days": 0, "recovery_score": 1.0}

        _ms = market_state_info.get("market_state", "Normal")
        if _ms == "Crisis":
            print(
                f"[{_ts()}] [Stage 9c] CRISIS MODE ACTIVE — "
                f"Crisis tickers will be skipped (no recommendations). "
                f"SPY_ATR%={market_state_info['spy_atr_pct']:.2%}  "
                f"crisis_frac={market_state_info['crisis_fraction']:.0%}  "
                f"recovery_score={market_state_info['recovery_score']:.2f}"
            )
        elif _ms == "High-Volatility":
            print(
                f"[{_ts()}] [Stage 9c] HIGH-VOLATILITY MODE — "
                f"max_holding_days capped at 10; entry thresholds tightened. "
                f"recovery_score={market_state_info['recovery_score']:.2f}  "
                f"stable_days={market_state_info['stable_days']}"
            )
        else:
            print(
                f"[{_ts()}] [Stage 9c] NORMAL MODE — standard pipeline parameters. "
                f"recovery_score={market_state_info['recovery_score']:.2f}  "
                f"stable_days={market_state_info['stable_days']}"
            )

        # ── Stage 9b: pair trading analysis ──────────────────────────────────
        # Runs after regime classification so we have the full shortlisted
        # universe available.  Pair analysis is cross-sectional — it needs
        # all tickers' OHLCV simultaneously to find cointegrated pairs.
        pair_analyses: list[dict] = []
        # Use the full shortlisted universe for pair discovery — AVOID tickers are
        # excluded from directional single-leg strategies but can still form valid
        # market-neutral spreads.  A cointegrated pair is evaluated on the spread's
        # own backtest and diagnostic, not on either leg's individual verdict.
        _pair_universe = [t for t in shortlisted if (ohlcv_raw or {}).get(t) is not None]
        if ohlcv_raw and len(_pair_universe) >= 2:
            print(f"[{_ts()}] [Stage 9b] Scanning for cointegrated pairs in {len(_pair_universe)} tickers "
                  f"(includes {len(_pair_universe) - len(actionable)} AVOID ticker(s)) ...")
            try:
                from pair_selector import PairSelector
                from strategy_selector import PAIR_TRADING_BASE, _compute_pair_trading_params
                import copy as _copy

                _pair_ohlcv = {t: ohlcv_raw[t] for t in _pair_universe if ohlcv_raw.get(t) is not None}
                _selector   = PairSelector(min_corr=0.70, max_coint_pv=0.10, max_pairs=5)
                _pairs      = self._safe("pair_selector.find_pairs",
                                         lambda: _selector.find_pairs(_pair_ohlcv), [])

                for _pair in (_pairs or []):
                    _ta, _tb     = _pair["ticker_a"], _pair["ticker_b"]
                    _oa          = ohlcv_raw.get(_ta)
                    _ob          = ohlcv_raw.get(_tb)
                    if _oa is None or _ob is None:
                        continue

                    # Use average Hurst and ATR of the two tickers for param tuning
                    _ra = next((r for r in (regimes or []) if r["ticker"] == _ta), {})
                    _rb = next((r for r in (regimes or []) if r["ticker"] == _tb), {})
                    _avg_hurst   = (_ra.get("hurst", 0.5) + _rb.get("hurst", 0.5)) / 2.0
                    _avg_atr_pct = (_ra.get("atr_pct", 0.02) + _rb.get("atr_pct", 0.02)) / 2.0
                    _params, _rule_log = _compute_pair_trading_params(_avg_hurst, _avg_atr_pct)

                    print(f"[{_ts()}]   [Stage 9b] Backtesting pair {_ta}/{_tb} "
                          f"(corr={_pair['correlation']:.3f}  coint_p={_pair['coint_pvalue']:.4f}  "
                          f"half_life={_pair['halflife_days']:.1f}d) ...")

                    _adv_a = float(features.get(_ta, {}).get("adv_20d", 0))
                    _adv_b = float(features.get(_tb, {}).get("adv_20d", 0))
                    _bt_pair = self._safe(
                        f"backtester.run_pair({_ta}/{_tb})",
                        lambda _a=_ta, _b=_tb, _oa2=_oa, _ob2=_ob, _p=_params, _hr=_pair["hedge_ratio"]: m["backtester"].run_pair(
                            _a, _b, _oa2, _ob2, _p,
                            hedge_ratio=_hr, adv_a=_adv_a, adv_b=_adv_b,
                        ),
                        None,
                    )

                    if _bt_pair and len(_bt_pair.get("trade_log", [])) >= 5:
                        _diag_pair = self._safe(
                            f"diagnostics.run({_ta}/{_tb})",
                            lambda _ta2=_ta, _bt2=_bt_pair: m["diagnostics"].run(
                                f"{_ta2}/{_tb}",
                                "PairTrading",
                                _bt2["trade_log"],
                                _bt2["returns"],
                                regime_label="Mean-Reverting",
                            ),
                            None,
                        )
                        # Live pair signal — computed on latest bar so the
                        # report can render a full execution brief (direction,
                        # leg sizing, z-score, exit/stop bands) without
                        # requiring additional research by the trader.
                        _pair_signal = self._safe(
                            f"backtester.pair_signal_status({_ta}/{_tb})",
                            lambda _oa2=_oa, _ob2=_ob, _p=_params,
                                   _hr=_pair["hedge_ratio"]: m["backtester"].pair_signal_status(
                                _oa2, _ob2, _p,
                                hedge_ratio=_hr,
                                initial_portfolio=self._cfg.get("initial_portfolio", 100_000.0),
                            ),
                            {"signal_active": None, "details": "unavailable",
                             "setup": None, "projected_setup": None},
                        )
                        pair_analyses.append({
                            "pair":           f"{_ta}/{_tb}",
                            "ticker_a":       _ta,
                            "ticker_b":       _tb,
                            "pair_stats":     _pair,
                            "params":         _params,
                            "rule_log":       _rule_log,
                            "backtest":       _bt_pair,
                            "diagnostic":     _diag_pair,
                            "current_signal": _pair_signal,
                        })
                        status = "PASS" if (_diag_pair or {}).get("passed") else "FAIL"
                        sharpe = (_diag_pair or {}).get("metrics", {}).get("sharpe", float("nan"))
                        print(f"[{_ts()}]   [Stage 9b] {_ta}/{_tb}: {status}  "
                              f"Sharpe={sharpe:.3f}  trades={len(_bt_pair['trade_log'])}")
                    else:
                        print(f"[{_ts()}]   [Stage 9b] {_ta}/{_tb}: skipped — "
                              f"insufficient trades ({len((_bt_pair or {}).get('trade_log', []))})")

            except Exception as _e:
                print(f"[{_ts()}] [Stage 9b] Pair analysis error: {_e}")

        if pair_analyses:
            print(f"[{_ts()}] [Stage 9b] Found {len(pair_analyses)} pair(s) with sufficient trade history.")

        # ── Stage 10: per-ticker strategy / backtest / diagnostics / MC ─────
        print(f"[{_ts()}] [Stage 10] Running per-ticker analysis for {len(regimes or [])} tickers in parallel ...")
        strategies:    list[dict] = []
        backtests:     list[dict] = []
        diagnostics:   list[dict] = []
        monte_carlos:  list[dict] = []

        portfolio = self._cfg.get("initial_portfolio", 100_000.0)

        _market_state = market_state_info.get("market_state", "Normal")

        def _analyse_ticker(regime: dict) -> dict:
            ticker       = regime["ticker"]
            ticker_regime = regime.get("regime", "Neutral")
            ohlcv        = (ohlcv_raw or {}).get(ticker)
            feats        = features.get(ticker, {})

            # AVOID gate: skip full analysis if LLM rated this ticker AVOID
            tv = next((v for v in (ticker_verdicts or []) if v["ticker"] == ticker), None)
            if tv and tv.get("verdict", "").lower() == "avoid":
                print(f"[{_ts()}]   [Stage 10] {ticker} — skipping strategy/backtest (AVOID verdict)")
                return {"ticker": ticker, "strategy": None, "backtest": None,
                        "diagnostic": None, "mc_result": None}

            # NO-TRADE gate: Crisis market state + Crisis ticker regime → skip entry.
            # Rationale: forcing Mean-Reversion entries into a sustained sell-off
            # (buying falling knives) has negative expected value.  The system outputs
            # no recommendation rather than a low-confidence trade.
            # High-Volatility market state alone does NOT trigger a skip — the
            # strategy parameters are adapted instead (shorter holds, tighter gates).
            if _market_state == "Crisis" and ticker_regime == "Crisis":
                print(
                    f"[{_ts()}]   [Stage 10] {ticker} — NO TRADE (Crisis market + Crisis ticker). "
                    f"Skipping to avoid buying falling knives."
                )
                return {
                    "ticker":     ticker,
                    "strategy":   None,
                    "backtest":   None,
                    "diagnostic": None,
                    "mc_result":  None,
                    "no_trade_reason": "Crisis market state — no new entries during sustained sell-off",
                }

            print(f"[{_ts()}]   [Stage 10] {ticker} — strategy select / backtest / diagnostics ...")

            strategy = self._safe(
                f"selector.select({ticker})",
                lambda _t=ticker, _r=regime, _f=feats, _v=tv, _ms=_market_state: m["selector"].select(
                    _t, _r, _f, macro, ticker_verdict=_v, market_state=_ms
                ),
                None,
            )

            # MLSignal fallback: if ml_signal column is absent or all-NaN
            # (thin history < 252 bars), downgrade to AlphaCombined so the
            # backtest and live signal check still produce a valid result.
            if strategy and strategy.get("strategy") == "MLSignal" and ohlcv is not None:
                ml_col = ohlcv.get("ml_signal") if hasattr(ohlcv, "get") else (
                    ohlcv["ml_signal"] if "ml_signal" in ohlcv.columns else None
                )
                no_signal = (
                    ml_col is None
                    or (hasattr(ml_col, "isna") and ml_col.isna().all())
                )
                if no_signal:
                    from strategy_selector import ALPHA_COMBINED_BASE as _AC_BASE
                    import copy as _copy
                    print(f"[{_ts()}]   [Fallback] {ticker}: MLSignal → AlphaCombined "
                          f"(insufficient ML training data)")
                    _ac_params = _copy.deepcopy(_AC_BASE)
                    strategy = {
                        **strategy,
                        "strategy":        "AlphaCombined",
                        "base_params":     _ac_params,
                        "adjusted_params": _ac_params,
                        "llm_adjustments": strategy.get("llm_adjustments", []) + [
                            "[Auto] MLSignal → AlphaCombined fallback (< 252 bars of history)"
                        ],
                    }

            if strategy and ohlcv is not None:
                sig = self._safe(
                    f"backtester.signal_status({ticker})",
                    lambda _s=strategy, _o=ohlcv, _p=portfolio: m["backtester"].signal_status(
                        _s["strategy"], _o, _s["adjusted_params"], _p
                    ),
                    {"signal_active": None, "details": "N/A", "setup": None},
                )
                adv = int(feats.get("adv_20d", 0))
                strategy = {**strategy, "current_signal": sig, "_adv": adv}

            # ── Param divergence flag ─────────────────────────────────────────
            # Warn when adjusted params differ materially from base params so the
            # trader knows the live signal uses a different stop/band than the
            # validated backtest.
            param_warnings: list[str] = []
            if strategy:
                base_p = strategy.get("base_params", {})
                adj_p  = strategy.get("adjusted_params", {})
                for key in ("stop_loss_atr", "trailing_stop_atr", "bb_std", "volume_mult"):
                    b_val = base_p.get(key)
                    a_val = adj_p.get(key)
                    if b_val is not None and a_val is not None:
                        diff = abs(a_val - b_val)
                        if diff > 0.5:
                            param_warnings.append(
                                f"{key}: base={b_val} vs adjusted={a_val} (Δ={diff:.1f} > 0.5) — "
                                f"live signal uses wider params than validated backtest"
                            )
                if param_warnings:
                    strategy = {**strategy, "param_divergence_warnings": param_warnings}
                    for w in param_warnings:
                        print(f"  [WARN] {ticker} param divergence: {w}")

            # Use base_params for backtest to avoid circular LLM tuning —
            # LLM-adjusted params are only used for the live signal check above.
            backtest = self._safe(
                f"backtester.run({ticker})",
                lambda _t=ticker, _s=strategy, _o=ohlcv, _adv=adv: m["backtester"].run(
                    _t,
                    {**_s, "adjusted_params": _s.get("base_params", _s["adjusted_params"])},
                    _o,
                    adv_shares=float(_adv),
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
                                print(f"[{_ts()}]   [Stage 10] {ticker} — alt params improved Sharpe {current_sharpe:.3f} → {alt_sharpe:.3f}")
                                strategy       = alt_strategy
                                backtest       = alt_backtest
                                current_sharpe = alt_sharpe

            # ── LLM hypothesis annotation (narrative only — no strategy switch) ──
            # The LLM may disagree with the regime-rule strategy selection.
            # Its suggestion is surfaced in the report for the trader's awareness,
            # but does NOT trigger a re-backtest or strategy class change.
            #
            # Rationale: selecting a strategy by comparing in-sample Sharpe ratios
            # (algo vs challenger) is data mining — the challenger that fits the
            # historical data better wins regardless of whether the fit is genuine
            # or overfit.  The regime-rule mapping is the primary selection;
            # the LLM provides qualitative narrative context only.
            hypothesis = (strategy or {}).get("llm_hypothesis", {})
            if (
                strategy
                and not hypothesis.get("agree", True)
                and hypothesis.get("suggested")
                and hypothesis["suggested"] != strategy["strategy"]
            ):
                ch_name   = hypothesis["suggested"]
                ch_reason = hypothesis.get("reason", "")
                print(
                    f"[{_ts()}]   [LLM Hypothesis] {ticker}: LLM suggests {ch_name} "
                    f"(narrative only — no backtest, no strategy switch) | {ch_reason}"
                )
                strategy = {
                    **strategy,
                    "llm_challenger": {
                        "strategy": ch_name,
                        "reason":   ch_reason,
                        "verdict":  "narrative_only",
                        "note":     "LLM disagreement surfaced for trader review; "
                                    "regime-rule selection retained to avoid in-sample data mining.",
                    },
                }

            diagnostic = self._safe(
                f"diagnostics.run({ticker})",
                lambda _t=ticker, _s=strategy, _bt=backtest, _rl=regime.get("regime", "Neutral"): m["diagnostics"].run(
                    _t,
                    _s["strategy"] if isinstance(_s, dict) else _s,
                    _bt["trade_log"],
                    _bt["returns"],
                    regime_label=_rl,
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
                        print(f"[{_ts()}]   [Stage 10] {ticker} — MC skipped ({label}, only {trade_count} trades, need 30+)")
                        mc_result = {"insufficient_sample": True, "trade_count": trade_count,
                                     "stress_test": not diag_passed}
                    else:
                        label = "passed" if diag_passed else f"near-threshold Sharpe={diag_sharpe:.3f}"
                        print(f"[{_ts()}]   [Stage 10] {ticker} — running Monte Carlo ({label}, {trade_count} trades) ...")
                        _ohlcv_yrs = None
                        _odf = (ohlcv_raw or {}).get(ticker)
                        if _odf is not None and not _odf.empty:
                            try:
                                _ohlcv_yrs = (_odf.index[-1] - _odf.index[0]).days / 365.25
                            except Exception:
                                pass
                        _mc = self._safe(
                            f"monte_carlo.run({ticker})",
                            lambda _bt=backtest, _p=portfolio, _oy=_ohlcv_yrs: m["monte_carlo"].run(
                                _bt["trade_log"], _p, ohlcv_years=_oy
                            ),
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

        # ── AlphaCombined portfolio backtest ──────────────────────────────────
        # AlphaCombined is a cross-sectional strategy: its signals are z-scored
        # across the universe, so individual per-ticker backtests destroy the
        # market-neutral property.  After the per-ticker loop, re-backtest all
        # AlphaCombined-selected tickers together as a portfolio and run
        # diagnostics on the aggregate result.  The portfolio diagnostic is
        # ADDED to (not replacing) the per-ticker results so individual
        # signal-status entries are preserved for the execution brief.
        ac_tickers = [
            s["ticker"]
            for s in strategies
            if isinstance(s, dict) and s.get("strategy") == "AlphaCombined"
        ]
        if len(ac_tickers) >= 2 and ohlcv_raw:
            print(f"[{_ts()}] [Stage 10b] Running AlphaCombined cross-sectional portfolio backtest "
                  f"on {len(ac_tickers)} tickers: {ac_tickers} ...")
            ac_ohlcv = {t: ohlcv_raw[t] for t in ac_tickers if ohlcv_raw.get(t) is not None}
            # Use params from the first AC strategy (all share same algo params for this regime)
            ac_params = next(
                (s["adjusted_params"] for s in strategies if s.get("strategy") == "AlphaCombined"),
                {},
            )
            ac_bt = self._safe(
                "backtester.run_alpha_combined_portfolio",
                lambda: m["backtester"].run_alpha_combined_portfolio(ac_ohlcv, ac_params),
                None,
            )
            if ac_bt and ac_bt.get("trade_log"):
                # Use the regime of the first AC ticker for regime-conditional floors
                ac_regime_label = next(
                    (r.get("regime", "Neutral") for r in (regimes or []) if r["ticker"] in ac_tickers),
                    "Neutral",
                )
                ac_diag = self._safe(
                    "diagnostics.run(AlphaCombined_Portfolio)",
                    lambda: m["diagnostics"].run(
                        "AlphaCombined_Portfolio",
                        "AlphaCombined",
                        ac_bt["trade_log"],
                        ac_bt["returns"],
                        regime_label=ac_regime_label,
                    ),
                    None,
                )
                if ac_diag:
                    diagnostics.append(ac_diag)
                backtests.append(ac_bt)
                # Monte Carlo on portfolio if diagnostic passed or near-threshold
                ac_diag_passed = (ac_diag or {}).get("passed", False)
                ac_diag_sharpe = (ac_diag or {}).get("metrics", {}).get("sharpe", -999)
                ac_near_thresh = (not ac_diag_passed) and (0.0 < ac_diag_sharpe < 0.5)
                ac_trade_count = len(ac_bt["trade_log"])
                if (ac_diag_passed or ac_near_thresh) and ac_trade_count >= 10:
                    _ohlcv_yrs = max(
                        (len(v) for v in ac_ohlcv.values() if v is not None), default=504
                    ) / 252
                    ac_mc = self._safe(
                        "monte_carlo.run(AlphaCombined_Portfolio)",
                        lambda: m["monte_carlo"].run(
                            ac_bt["trade_log"], portfolio, ohlcv_years=_ohlcv_yrs
                        ),
                        None,
                    )
                    if ac_mc:
                        monte_carlos.append({
                            "ticker":       "AlphaCombined_Portfolio",
                            "trade_count":  ac_trade_count,
                            "stress_test":  not ac_diag_passed,
                            **ac_mc,
                        })
        elif len(ac_tickers) == 1 and ohlcv_raw:
            print(f"[{_ts()}] [Stage 10b] Only 1 AlphaCombined ticker ({ac_tickers[0]}) — "
                  f"cross-sectional portfolio requires >= 2 tickers; per-ticker result retained.")

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
                ret_df = pd.DataFrame(close_data).pct_change(fill_method=None).dropna()
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
        print(f"[{_ts()}] [Stage 11] Building execution brief ...")
        execution_brief = self._safe(
            "execution_advisor.advise",
            lambda: m["execution_advisor"].advise(strategies),
            {"active_signals": [], "inactive_count": 0,
             "portfolio_risk": {"active_count": 0, "total_dollar_risk": 0.0, "pct_of_portfolio": 0.0},
             "warnings": []},
        )

        spy_ohlcv = (ohlcv_raw or {}).get("SPY")

        # ── Stage 11b: portfolio optimisation ────────────────────────────────
        print(f"[{_ts()}] [Stage 11b] Running portfolio optimisation ...")
        portfolio_result = self._safe(
            "portfolio_optimizer.optimize",
            lambda: m["portfolio_optimizer"].optimize(
                backtests, diagnostics, ohlcv_shortlisted or {}
            ),
            {"cs_momentum_ranks": [], "allocations": [], "rejected": [],
             "portfolio_metrics": {}, "portfolio_returns": None},
        )

        return self._finish(
            m, run_date, summary, macro,
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
            meta_insights=meta_insights,
            portfolio_result=portfolio_result,
            pair_analyses=pair_analyses,
            market_state=market_state_info,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe(stage: str, fn, fallback):
        try:
            return fn()
        except Exception as exc:
            import traceback
            print(f"  [ERROR] Stage '{stage}' failed: {exc}")
            print(traceback.format_exc())
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

        # ── Verdict accountability log ────────────────────────────────────────
        # Append {run_date, ticker, verdict, sharpe, passed, strategy, regime} to
        # a flat CSV so that after 30–50 runs a Kruskal-Wallis test can determine
        # whether LLM buy/watch/avoid verdicts actually correlate with backtest outcome.
        PipelineOrchestrator._log_verdict_outcomes(run_date, pipeline_output)

        print(f"[{_ts()}] [Stage 12] Generating full report ...")
        SummaryReport = m["reporter"].generate(pipeline_output, timestamp=timestamp)
        print(f"[{_ts()}] [Stage 12] Generating trader summary report ...")
        TraderReport  = m["reporter"].generate_summary(pipeline_output, timestamp=timestamp)
        return {
            "SummaryReport": SummaryReport,
            "TraderReport":  TraderReport,
            "run_date":    run_date,
            "summary":     summary,
            "macro":       macro,
            **kwargs,
        }

    @staticmethod
    def _load_meta_insights() -> dict:
        """
        Read data/verdict_log.csv and compute per-(regime+strategy) performance stats.

        Returns a dict with keys:
          insights       : dict[str, dict] — keyed by "Regime+Strategy"
                           values: {n, avg_sharpe, pass_rate}
          warnings       : list[str] — combinations with historically poor pass rate
          total_runs     : int
          sample_too_small: bool — set when fewer than 10 rows

        After ~30 runs, this gives the system memory of which strategy/regime
        combinations have historically produced passing OOS Sharpes, allowing
        the report to flag when the current selection is historically weak.
        """
        import csv as _csv
        import os as _os
        from collections import defaultdict as _ddict

        log_path = _os.path.join("data", "verdict_log.csv")
        if not _os.path.exists(log_path):
            return {"sample_too_small": True, "total_runs": 0, "insights": {}, "warnings": []}

        rows = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                rows   = list(reader)
        except Exception:
            return {"sample_too_small": True, "total_runs": 0, "insights": {}, "warnings": []}

        if len(rows) < 10:
            return {"sample_too_small": True, "total_runs": len(rows), "insights": {}, "warnings": []}

        groups: dict = _ddict(list)
        for row in rows:
            regime   = row.get("regime", "Unknown")
            strategy = row.get("strategy", "Unknown")
            key      = f"{regime}+{strategy}"
            try:
                groups[key].append({
                    "sharpe": float(row.get("sharpe", 0)),
                    "passed": int(row.get("passed", 0)),
                })
            except (ValueError, TypeError):
                pass

        insights: dict = {}
        for key, entries in groups.items():
            if len(entries) >= 5:
                avg_sharpe = sum(e["sharpe"] for e in entries) / len(entries)
                pass_rate  = sum(e["passed"] for e in entries) / len(entries)
                insights[key] = {
                    "n":          len(entries),
                    "avg_sharpe": round(avg_sharpe, 3),
                    "pass_rate":  round(pass_rate, 3),
                }

        warnings: list[str] = []
        for key, stats in insights.items():
            if stats["pass_rate"] < 0.10 and stats["n"] >= 10:
                warnings.append(
                    f"{key}: only {stats['pass_rate']:.0%} historical pass rate "
                    f"over {stats['n']} runs (avg Sharpe {stats['avg_sharpe']:.3f}) — "
                    f"this regime/strategy combination has shown weak edge"
                )

        return {
            "sample_too_small": False,
            "total_runs":       len(rows),
            "insights":         insights,
            "warnings":         warnings,
        }

    @staticmethod
    def _log_verdict_outcomes(run_date: str, pipeline_output: dict) -> None:
        """
        Append one row per ticker to data/verdict_log.csv.

        Columns: run_date, ticker, verdict, strategy, regime, sharpe,
                 max_drawdown, trade_count, passed, wf_underpowered

        After ~50 runs, run a Kruskal-Wallis test on sharpe grouped by verdict
        to verify the LLM screening stage is contributing alpha rather than noise.
        """
        import csv, os as _os

        verdicts   = {v["ticker"]: v for v in pipeline_output.get("ticker_verdicts", [])}
        diags      = {d["ticker"]: d for d in pipeline_output.get("diagnostics", [])}
        strategies = {s["ticker"]: s for s in pipeline_output.get("strategies", [])}
        regimes    = {r["ticker"]: r for r in pipeline_output.get("regimes", [])}

        if not verdicts:
            return

        log_path = _os.path.join("data", "verdict_log.csv")
        _os.makedirs("data", exist_ok=True)
        write_header = not _os.path.exists(log_path)

        rows = []
        for ticker, v in verdicts.items():
            d   = diags.get(ticker, {})
            s   = strategies.get(ticker, {})
            reg = regimes.get(ticker, {})
            m   = d.get("metrics", {})
            rows.append({
                "run_date":       run_date,
                "ticker":         ticker,
                "verdict":        v.get("verdict", "watch"),
                "strategy":       s.get("strategy", ""),
                "regime":         reg.get("regime", ""),
                "sharpe":         round(m.get("sharpe", 0.0), 4),
                "max_drawdown":   round(m.get("max_drawdown", 0.0), 4),
                "trade_count":    m.get("trade_count", 0),
                "passed":         int(d.get("passed", False)),
                "wf_underpowered": int(m.get("wf_underpowered", False)),
            })

        try:
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
            print(f"  [VerdictLog] Appended {len(rows)} rows → {log_path}")
        except Exception as e:
            print(f"  [VerdictLog] Write failed (non-fatal): {e}")

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
        from alpha_engine                import AlphaEngine
        from portfolio_optimizer         import PortfolioOptimizer
        from ml_signal_engine            import MLSignalEngine
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
            "rag_store":           rag,
            "portfolio_optimizer": PortfolioOptimizer(
                                initial_portfolio=cfg.get("initial_portfolio", 100_000.0)
                             ),
            "alpha_engine":        AlphaEngine(),
            "ml_signal_engine":    MLSignalEngine(),
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
    parser.add_argument("--min-volume",  type=int,   default=0,     help="Minimum avg daily volume to include (default: 0 = no filter)")
    parser.add_argument("--max-markets", type=int,   default=100,   help="Max tickers to prefilter from news (default: 100)")
    args = parser.parse_args()

    def llm(prompt: str) -> str:
        resp = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.0}, think=True)
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    config = {
        "benzinga_api_key":  os.getenv("BENZINGA_API"),
        "llm_client":        llm,
        "output_dir":        "reports",
        "cache_dir":         "data/cache",
        "initial_portfolio": 100_000.0,
        "window_days":       min(args.days, 14),
        "max_tickers":       args.max_tickers,
        "min_volume":        args.min_volume,
        "max_markets":       min(args.max_markets, 100),
    }

    date = args.date

    result = PipelineOrchestrator(config).run(date)

    print(f"\nSummaryReport : {result['SummaryReport']}")
    print(f"TraderReport  : {result.get('TraderReport', 'n/a')}")
    print(f"Date    : {result['run_date']}")
    print(f"Bias   : {result['summary'].get('market_bias', 'n/a')}")
    diag_passed = {d["ticker"] for d in result.get("diagnostics", []) if d.get("passed")}
    mc_passed   = sum(
        1 for mc in result.get("monte_carlos", [])
        if not mc.get("insufficient_sample")
        and not mc.get("stress_test")
        and mc["ticker"] in diag_passed
    )
    print(f"Tickers: {len(result['ticker_verdicts'])} verdicts | "
          f"{len(result['backtests'])} backtests | "
          f"{len(result['diagnostics'])} diagnostics | "
          f"{len(result['monte_carlos'])} MC sims ({mc_passed} passed)")
