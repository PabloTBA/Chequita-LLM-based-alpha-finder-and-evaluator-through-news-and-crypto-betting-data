"""
DiagnosticsEngine
=================
Runs diagnostic checks on a strategy's returns and trade log.
Applies hard reject floors; calls LLM for qualitative commentary only when
all floors pass.

Hard floors (checked in order)
-------------------------------
  Sharpe ratio              < 0.5   → auto-reject  (OOS rescue: pass if OOS ≥ floor)
  Max drawdown              > 20%   → auto-reject
  Win rate                  < 35%   → auto-reject  (bypass if profit_factor ≥ 1.5)
  Kelly fraction            < 0.0   → auto-reject  (negative expectancy = provably losing)
  Walk-forward degradation  > 50%   → auto-reject
  Trade count               < 30    → auto-reject

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
PROFIT_FACTOR_FLOOR   = 1.5    # bypass win-rate floor when profit_factor >= this (high-payoff strategies)
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
        status = "PASS" if passed else f"FAIL -- {reject_reason}"
        print(f"  [Diag] {ticker}: Sharpe={metrics['sharpe']:.3f}  OOS_Sharpe={metrics['oos_sharpe']:.3f}  "
              f"MaxDD={metrics['max_drawdown']:.1%}  WinRate={metrics['win_rate']:.1%}  "
              f"WFDegrad={metrics['walk_forward_degradation']:.1%}  "
              f"Trades={metrics['trade_count']}  -> {status}")

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
        sharpe = self._sharpe(returns)
        wf_degrad, oos_sharpe = self._walk_forward_degradation(returns)
        return {
            "sharpe":                   sharpe,
            "oos_sharpe":               oos_sharpe,
            "max_drawdown":             self._max_drawdown(returns),
            "win_rate":                 self._win_rate(trade_log),
            "profit_factor":            self._profit_factor(trade_log),
            "kelly_fraction":           self._kelly_fraction(trade_log),
            "walk_forward_degradation": wf_degrad,
            "trade_count":              len(trade_log),
        }

    @staticmethod
    def _check_floors(metrics: dict) -> tuple[bool, Optional[str]]:
        # Use the better of full-period and OOS Sharpe so that a strategy
        # which degrades in IS but is strong OOS is not wrongly rejected.
        sharpe     = metrics["sharpe"]
        oos_sharpe = metrics.get("oos_sharpe", sharpe)
        best_sharpe = max(sharpe, oos_sharpe)
        if best_sharpe < SHARPE_FLOOR:
            return False, (f"Sharpe ratio {sharpe:.3f} (OOS {oos_sharpe:.3f}) "
                           f"both below floor {SHARPE_FLOOR}")

        max_dd = metrics["max_drawdown"]
        if max_dd > MAX_DD_FLOOR:
            return False, f"Max drawdown {max_dd:.1%} exceeds floor {MAX_DD_FLOOR:.0%}"

        win_rate      = metrics["win_rate"]
        profit_factor = metrics.get("profit_factor", 0.0)
        # Low win rate bypassed when profit factor is strong (high-payoff strategies such as
        # trend-following with 30% wins but 3:1 payoff ratio are valid and should not be rejected)
        if win_rate < WIN_RATE_FLOOR and profit_factor < PROFIT_FACTOR_FLOOR:
            return False, (f"Win rate {win_rate:.1%} below floor {WIN_RATE_FLOOR:.0%} "
                           f"and profit factor {profit_factor:.2f} below {PROFIT_FACTOR_FLOOR:.1f}")

        # Kelly fraction: negative Kelly means the strategy has provably negative expected value
        # regardless of win rate or profit factor (e.g. 50% win rate but losses >> wins).
        # Default 0.0 when key absent (callers that pass metric dicts directly without kelly).
        kelly = metrics.get("kelly_fraction", 0.0)
        if kelly < 0.0:
            return False, (f"Negative Kelly fraction ({kelly:.4f}) — strategy has provably "
                           f"negative expected value; do not size any position")

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
        """Annualized Sharpe ratio net of risk-free rate.
        Guard uses 1e-10 (not == 0) because 0.003 is not exactly representable
        in float64 — std of a nominally-constant series can be ~1e-19, which
        would produce an astronomically large Sharpe if uncapped."""
        std = returns.std(ddof=1)
        if std < 1e-10 or np.isnan(std):
            return 0.0
        daily_rf = RISK_FREE_RATE / TRADING_DAYS
        raw = float((returns.mean() - daily_rf) / std * np.sqrt(TRADING_DAYS))
        return float(np.clip(raw, -20.0, 20.0))   # cap at ±20 — physically impossible otherwise

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
    def _profit_factor(trade_log: list[dict]) -> float:
        """Gross profit / gross loss.  Returns 0.0 when there are no losing trades."""
        if not trade_log:
            return 0.0
        gross_profit = sum(t.get("pnl", 0) for t in trade_log if t.get("pnl", 0) > 0)
        gross_loss   = sum(-t.get("pnl", 0) for t in trade_log if t.get("pnl", 0) < 0)
        if gross_loss < 1e-10:
            return 0.0 if gross_profit <= 0 else 999.0   # no losses → best possible
        return gross_profit / gross_loss

    @staticmethod
    def _kelly_fraction(trade_log: list[dict]) -> float:
        """
        Optimal Kelly fraction: f* = W/L − (1−W)/G

        Where W = win rate, G = mean win size, L = mean absolute loss.

        This is the exact Kelly formula derived by maximising E[ln(1 + f·R)].
        Returns:
          0.0  when trade log is empty or when there are no wins *or* no losses
               (degenerate — treat as zero edge rather than ±∞).
         -1.0  clipped minimum  (worst possible; signals strongly negative edge)
          1.0  clipped maximum  (theoretical; do not use full Kelly in practice)
        """
        if not trade_log:
            return 0.0
        pnls   = [t.get("pnl", 0.0) for t in trade_log]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        if not wins or not losses:
            # All wins: infinite Kelly → cap at 1.0; all losses: cap at -1.0; all flat: 0.0
            if not losses and wins:
                return 1.0
            if not wins and losses:
                return -1.0
            return 0.0
        W = len(wins) / len(pnls)           # win rate
        G = sum(wins) / len(wins)           # mean win (positive)
        L = sum(-p for p in losses) / len(losses)  # mean loss (positive)
        kelly = W / L - (1.0 - W) / G
        return float(np.clip(kelly, -1.0, 1.0))

    @staticmethod
    def _walk_forward_degradation(returns: pd.Series) -> tuple[float, float]:
        """
        Split returns 70/30 in-sample / out-of-sample (industry standard).
        Degradation = (IS_Sharpe - OOS_Sharpe) / IS_Sharpe, clamped to [0, 1].
        Returns 1.0 degradation when IS_Sharpe <= 0 (degenerate / already bad in-sample).

        Returns
        -------
        (degradation, oos_sharpe) — both needed by the caller.
        """
        split     = int(len(returns) * 0.70)
        in_sample = returns.iloc[:split]
        oos       = returns.iloc[split:]

        def sharpe(r: pd.Series) -> float:
            std = r.std(ddof=1)
            if std < 1e-10 or np.isnan(std):
                return 0.0
            daily_rf = RISK_FREE_RATE / TRADING_DAYS
            raw = float((r.mean() - daily_rf) / std * np.sqrt(TRADING_DAYS))
            return float(np.clip(raw, -20.0, 20.0))

        is_sharpe  = sharpe(in_sample)
        oos_sharpe = sharpe(oos)

        # OOS better than or equal to IS → no degradation (improvement)
        if oos_sharpe >= is_sharpe:
            return 0.0, oos_sharpe

        # IS bad but OOS also bad → fully degraded
        if is_sharpe <= 0:
            return 1.0, oos_sharpe

        # IS positive, OOS worse → graded degradation
        degrad = (is_sharpe - oos_sharpe) / is_sharpe
        return float(np.clip(degrad, 0.0, 1.0)), oos_sharpe


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
        resp = ollama.chat(model="qwen3:14b",
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.0})
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
