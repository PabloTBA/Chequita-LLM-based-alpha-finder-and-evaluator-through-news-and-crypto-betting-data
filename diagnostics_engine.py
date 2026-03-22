"""
DiagnosticsEngine
=================
Runs diagnostic checks on a strategy's returns and trade log.
Applies 4 hard reject floors; calls LLM for qualitative commentary only when
all floors pass.

Hard floors (checked in order)
-------------------------------
  Sharpe ratio              < 0.5   → auto-reject
  Max drawdown              > 30%   → auto-reject
  Win rate                  < 35%   → auto-reject
  Walk-forward degradation  > 50%   → auto-reject

Public interface
----------------
  engine = DiagnosticsEngine(llm_client=fn, verbose=False)
  result = engine.run(ticker, strategy_name, trade_log, returns)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Hard floor constants (PRD defaults) ───────────────────────────────────────

SHARPE_FLOOR          = 0.50
MAX_DD_FLOOR          = 0.20   # tightened from 30% → 20% (institutional standard)
WIN_RATE_FLOOR        = 0.35
WALKFWD_DEGRAD_FLOOR  = 0.50
MIN_TRADE_COUNT       = 30     # raised from 10 → 30 (minimum for statistical significance)
TRADING_DAYS          = 252
RISK_FREE_RATE        = 0.045  # annualised risk-free rate (~current Fed funds); subtract from Sharpe


class DiagnosticsEngine:
    def __init__(self, llm_client: Optional[Callable] = None, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(
        self,
        ticker:     str,
        strategy:   str,
        trade_log:  list[dict],
        returns:    pd.Series,
    ) -> dict:
        """
        Run all diagnostic checks on a strategy.

        Parameters
        ----------
        ticker    : str
        strategy  : str  (e.g. "Momentum", "Mean-Reversion")
        trade_log : list of dicts, each with at least {"pnl": float}
        returns   : pd.Series of daily returns (decimal), DatetimeIndex preferred

        Returns
        -------
        dict with keys: ticker, strategy, passed, reject_reason, metrics, llm_commentary
        """
        metrics = self._compute_metrics(trade_log, returns)
        self._log(f"[DiagnosticsEngine] {ticker}: {metrics}")

        passed, reject_reason = self._check_floors(metrics)
        status = "PASS ✓" if passed else f"FAIL — {reject_reason}"
        print(f"  [Diag] {ticker}: Sharpe={metrics['sharpe']:.3f}  MaxDD={metrics['max_drawdown']:.1%}  "
              f"WinRate={metrics['win_rate']:.1%}  WFDegrad={metrics['walk_forward_degradation']:.1%}  "
              f"Trades={metrics['trade_count']}  → {status}")

        llm_commentary: Optional[str] = None
        if passed and self.llm_client is not None:
            llm_commentary = self._get_llm_commentary(ticker, strategy, metrics)

        return {
            "ticker":         ticker,
            "strategy":       strategy,
            "passed":         passed,
            "reject_reason":  reject_reason,
            "metrics":        metrics,
            "llm_commentary": llm_commentary,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, trade_log: list[dict], returns: pd.Series) -> dict:
        return {
            "sharpe":                   self._sharpe(returns),
            "max_drawdown":             self._max_drawdown(returns),
            "win_rate":                 self._win_rate(trade_log),
            "walk_forward_degradation": self._walk_forward_degradation(returns),
            "trade_count":              len(trade_log),
        }

    @staticmethod
    def _check_floors(metrics: dict) -> tuple[bool, Optional[str]]:
        sharpe = metrics["sharpe"]
        if sharpe < SHARPE_FLOOR:
            return False, f"Sharpe ratio {sharpe:.3f} below floor {SHARPE_FLOOR}"

        max_dd = metrics["max_drawdown"]
        if max_dd > MAX_DD_FLOOR:
            return False, f"Max drawdown {max_dd:.1%} exceeds floor {MAX_DD_FLOOR:.0%}"

        win_rate = metrics["win_rate"]
        if win_rate < WIN_RATE_FLOOR:
            return False, f"Win rate {win_rate:.1%} below floor {WIN_RATE_FLOOR:.0%}"

        wf = metrics["walk_forward_degradation"]
        if wf > WALKFWD_DEGRAD_FLOOR:
            return False, f"Walk-forward degradation {wf:.1%} exceeds floor {WALKFWD_DEGRAD_FLOOR:.0%}"

        tc = metrics["trade_count"]
        if tc < MIN_TRADE_COUNT:
            return False, f"Trade count {tc} below minimum {MIN_TRADE_COUNT} for statistical significance"

        return True, None

    def _get_llm_commentary(self, ticker: str, strategy: str, metrics: dict) -> str:
        print(f"  [LLM] DiagnosticsEngine: commentary for {ticker} ({strategy})...")
        prompt = (
            f"You are a quantitative strategist reviewing a strategy that passed all diagnostic floors.\n"
            f"Ticker: {ticker}\n"
            f"Strategy: {strategy}\n"
            f"Sharpe={metrics['sharpe']:.2f}, MaxDD={metrics['max_drawdown']:.1%}, "
            f"WinRate={metrics['win_rate']:.1%}, "
            f"WalkFwdDegradation={metrics['walk_forward_degradation']:.1%}\n\n"
            f"Provide 2-3 sentences of qualitative commentary on strengths and weaknesses."
        )
        result = self.llm_client(prompt)
        print(f"  [LLM] DiagnosticsEngine: {ticker} done")
        return result

    # ── math ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        """Annualized Sharpe ratio net of risk-free rate."""
        std = returns.std(ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        daily_rf = RISK_FREE_RATE / TRADING_DAYS
        return float((returns.mean() - daily_rf) / std * np.sqrt(TRADING_DAYS))

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Maximum drawdown of the equity curve built from daily returns."""
        equity      = (1.0 + returns).cumprod()
        rolling_max = equity.cummax()
        dd          = (equity - rolling_max) / rolling_max
        return float(-dd.min())

    @staticmethod
    def _win_rate(trade_log: list[dict]) -> float:
        """Fraction of trades with pnl > 0."""
        if not trade_log:
            return 0.0
        wins = sum(1 for t in trade_log if t.get("pnl", 0) > 0)
        return wins / len(trade_log)

    @staticmethod
    def _walk_forward_degradation(returns: pd.Series) -> float:
        """
        Split returns 70/30 in-sample / out-of-sample (industry standard).
        Degradation = (IS_Sharpe - OOS_Sharpe) / IS_Sharpe, clamped to [0, 1].
        Returns 1.0 when IS_Sharpe ≤ 0 (degenerate / already bad in-sample).
        """
        split     = int(len(returns) * 0.70)
        in_sample = returns.iloc[:split]
        oos       = returns.iloc[split:]

        def sharpe(r: pd.Series) -> float:
            std = r.std(ddof=1)
            if std == 0 or np.isnan(std):
                return 0.0
            daily_rf = RISK_FREE_RATE / TRADING_DAYS
            return float((r.mean() - daily_rf) / std * np.sqrt(TRADING_DAYS))

        is_sharpe  = sharpe(in_sample)
        oos_sharpe = sharpe(oos)

        # OOS better than or equal to IS → no degradation (improvement)
        if oos_sharpe >= is_sharpe:
            return 0.0

        # IS bad but OOS also bad → fully degraded
        if is_sharpe <= 0:
            return 1.0

        # IS positive, OOS worse → graded degradation
        degrad = (is_sharpe - oos_sharpe) / is_sharpe
        return float(np.clip(degrad, 0.0, 1.0))


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ollama, os, sys
    from dotenv import load_dotenv; load_dotenv()
    from Stage1DataCollector import Stage1DataCollector
    from news_summarizer import NewsSummarizer
    from macro_screener import MacroScreener
    from ticker_screener import TickerScreener
    from ohlcv_fetcher import OHLCVFetcher
    from regime_classifier import RegimeClassifier
    from strategy_selector import StrategySelector
    from datetime import datetime, timedelta

    date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    def llm(prompt):
        resp = ollama.chat(model="qwen3:8b",
                           messages=[{"role": "user", "content": prompt}])
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    collector = Stage1DataCollector(api_key=os.getenv("BENZINGA_API"), cache_dir="data/cache")
    articles  = collector.collect(date)
    summary   = NewsSummarizer(llm_client=llm, window_days=7).summarize(articles, as_of_date=date)
    macro     = MacroScreener(llm_client=llm).screen(summary)

    screener  = TickerScreener(llm_client=llm)
    top50     = screener.prefilter(articles)
    fetcher   = OHLCVFetcher()
    ohlcv_raw = fetcher.fetch(top50["ticker"].head(5).tolist())
    ohlcv     = {t: fetcher.compute_features(df) for t, df in ohlcv_raw.items() if df is not None}

    clf = RegimeClassifier()
    sel = StrategySelector(llm_client=llm, verbose=True)
    eng = DiagnosticsEngine(llm_client=llm, verbose=True)

    for ticker, feats in ohlcv.items():
        regime   = clf.classify(ticker, ohlcv_raw[ticker])
        strategy = sel.select(ticker, regime, feats, macro)

        # Build a synthetic trade log from OHLCV for the smoke test
        # (real trade log comes from backtester — not yet built)
        df   = ohlcv_raw[ticker].dropna()
        rets = df["Close"].pct_change().dropna()
        fake_log = [{"pnl": float(r)} for r in rets.tail(30)]

        result = eng.run(ticker, strategy["strategy"], fake_log, rets)
        print(f"\n{'='*60}")
        print(f"  {ticker}  passed={result['passed']}  reason={result['reject_reason']}")
        print(f"  metrics={result['metrics']}")
        if result["llm_commentary"]:
            print(f"  commentary: {result['llm_commentary'][:200]}")
