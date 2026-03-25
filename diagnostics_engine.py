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

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd
import scipy.stats as _stats

# ── Hard floor constants (PRD defaults) ───────────────────────────────────────

WF_MIN_TRADE_COUNT    = 100    # minimum trades for walk-forward to have statistical power
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
        tc = len(trade_log)
        wf_degrad, oos_sharpe, wf_splits = self._walk_forward_degradation(returns, trade_count=tc)
        t_stat, p_value                  = self._tstat(returns)
        bs_p5, bs_p95                    = self._bootstrap_sharpe_ci(returns)
        return {
            "sharpe":                   sharpe,
            "oos_sharpe":               oos_sharpe,
            "max_drawdown":             self._max_drawdown(returns),
            "win_rate":                 self._win_rate(trade_log),
            "profit_factor":            self._profit_factor(trade_log),
            "kelly_fraction":           self._kelly_fraction(trade_log),
            "walk_forward_degradation": wf_degrad,
            "wf_splits":                wf_splits,
            "trade_count":              tc,
            "wf_underpowered":          tc < WF_MIN_TRADE_COUNT,
            # Robustness / statistical significance
            "t_stat":                   t_stat,     # Lo (2002) autocorr-corrected t-stat
            "p_value":                  p_value,    # one-tailed H1: Sharpe > 0
            "bootstrap_sharpe_p5":      bs_p5,      # 5th pct of bootstrap Sharpe dist
            "bootstrap_sharpe_p95":     bs_p95,     # 95th pct
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

        # Walk-forward: require passing at least 2 of 3 splits (60/40, 70/30, 80/20).
        # Underpowered splits count as passes so they don't wrongly reject.
        wf_splits = metrics.get("wf_splits", [])
        if wf_splits and not metrics.get("wf_underpowered", False):
            n_pass = sum(1 for s in wf_splits if s.get("passed", True))
            if n_pass < 2:
                degrad_str = " | ".join(
                    f"{int(s['is_pct']*100)}/{int((1-s['is_pct'])*100)}: {s['degradation']:.1%}"
                    for s in wf_splits
                )
                return False, (
                    f"Walk-forward failed ≥2 of 3 splits ({degrad_str}) — "
                    f"median degradation {metrics['walk_forward_degradation']:.1%} "
                    f"suggests IS overfit"
                )
        elif not wf_splits:
            # Fallback: single-split check for callers that don't provide split detail
            wf = metrics["walk_forward_degradation"]
            if wf > WALKFWD_DEGRAD_FLOOR:
                return False, f"Walk-forward degradation {wf:.1%} exceeds floor {WALKFWD_DEGRAD_FLOOR:.0%}"

        tc = metrics["trade_count"]
        if tc < MIN_TRADE_COUNT:
            return False, f"Trade count {tc} below minimum {MIN_TRADE_COUNT} for statistical significance"

        return True, None

    def _get_llm_commentary(self, ticker: str, strategy: str, metrics: dict) -> str:
        print(f"  [LLM] DiagnosticsEngine: commentary for {ticker} ({strategy})...")
        wf_splits   = metrics.get("wf_splits", [])
        # Pick the 70/30 split for the report (most commonly cited)
        wf_70 = next((s for s in wf_splits if abs(s.get("is_pct", 0) - 0.70) < 0.01), {})
        is_sharpe  = wf_70.get("is_sharpe",  metrics.get("sharpe", 0.0))
        oos_sharpe = wf_70.get("oos_sharpe", metrics.get("oos_sharpe", 0.0))
        wf_note = (
            f"IS Sharpe={is_sharpe:.3f}, OOS Sharpe={oos_sharpe:.3f} "
            f"({'OOS better than IS — strategy improved out-of-sample' if oos_sharpe > is_sharpe else 'OOS worse than IS — some degradation' if oos_sharpe < is_sharpe * 0.5 else 'IS and OOS broadly consistent'})"
        )
        prompt = (
            f"You are a quantitative strategist reviewing a strategy that passed all diagnostic floors.\n"
            f"Ticker: {ticker}\n"
            f"Strategy: {strategy}\n"
            f"Full-period Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_drawdown']:.1%}, "
            f"WinRate={metrics['win_rate']:.1%}, ProfitFactor={metrics.get('profit_factor', 0):.2f}\n"
            f"Walk-forward (70/30 split): {wf_note}\n"
            f"WalkFwdDegradation={metrics['walk_forward_degradation']:.1%}\n\n"
            f"Rules:\n"
            f"- If OOS Sharpe > IS Sharpe, state this explicitly — it means the strategy is NOT overfitted.\n"
            f"- If IS Sharpe is negative but OOS Sharpe is positive, say 'in-sample underperformed but out-of-sample recovered'.\n"
            f"- Do NOT call performance 'consistent' unless IS and OOS Sharpe are within 0.2 of each other.\n"
            f"- Be specific about numbers. Do not give generic praise.\n\n"
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
    def _tstat(returns: pd.Series) -> tuple[float, float]:
        """
        t-statistic for H0: mean excess return = 0, with Lo (2002) correction
        for serial autocorrelation in the return series.

        Returns (t_stat, p_value) where p_value is one-tailed (H1: Sharpe > 0).
        A p_value < 0.05 means the Sharpe is statistically distinguishable from
        noise at 95% confidence.
        """
        n = len(returns)
        if n < 10:
            return 0.0, 1.0
        daily_rf = RISK_FREE_RATE / TRADING_DAYS
        excess   = np.array(returns, dtype=float) - daily_rf
        mean_e   = float(excess.mean())
        std_e    = float(excess.std(ddof=1))
        if std_e < 1e-10 or math.isnan(std_e):
            return 0.0, 1.0

        # Lo (2002) autocorrelation correction — Bartlett kernel
        q    = max(1, int(n ** 0.25))
        acf  = 0.0
        for k in range(1, q + 1):
            if n - k > 0:
                rho = float(np.corrcoef(excess[:-k], excess[k:])[0, 1])
                if not math.isnan(rho):
                    acf += rho * (1.0 - k / (q + 1.0))
        acf_factor = max(1.0 + 2.0 * acf, 0.1)   # floor at 0.1 to avoid div/0

        sr_daily = mean_e / std_e
        t_stat   = sr_daily * math.sqrt(n / acf_factor)
        p_value  = float(1.0 - _stats.t.cdf(t_stat, df=n - 1))

        return round(float(t_stat), 3), round(p_value, 4)

    @staticmethod
    def _bootstrap_sharpe_ci(
        returns: pd.Series,
        n_bootstrap: int = 1000,
        block_size:  int = 20,
    ) -> tuple[float, float]:
        """
        Block-bootstrap 90% confidence interval for annualised Sharpe.

        Block size = 20 trading days (≈ 1 month) preserves the serial
        dependence structure of returns (volatility clustering, autocorrelation).

        Returns (p5, p95): the 5th and 95th percentile of the bootstrap
        Sharpe distribution.  A bootstrap p5 > 0 provides strong evidence
        that the Sharpe is genuinely positive, not a sampling artefact.
        """
        r = np.array(returns, dtype=float)
        n = len(r)
        if n < 40:
            return 0.0, 0.0
        daily_rf = RISK_FREE_RATE / TRADING_DAYS
        sharpes: list[float] = []
        rng = np.random.default_rng(seed=42)   # reproducible
        for _ in range(n_bootstrap):
            # Circular block bootstrap
            n_blocks = math.ceil(n / block_size)
            starts   = rng.integers(0, n, size=n_blocks)
            sample   = np.concatenate([
                np.roll(r, -int(s))[:block_size] for s in starts
            ])[:n]
            std = float(sample.std(ddof=1))
            if std > 1e-10:
                sr = (float(sample.mean()) - daily_rf) / std * math.sqrt(TRADING_DAYS)
                sharpes.append(sr)
        if not sharpes:
            return 0.0, 0.0
        return round(float(np.percentile(sharpes, 5)),  3), \
               round(float(np.percentile(sharpes, 95)), 3)

    @staticmethod
    def _walk_forward_degradation(
        returns: pd.Series, trade_count: int = 0
    ) -> tuple[float, float, list[dict]]:
        """
        Multi-split walk-forward: run at IS/OOS ratios of 60/40, 70/30, 80/20.
        Strategy passes walk-forward only if degradation ≤ WALKFWD_DEGRAD_FLOOR
        in at least 2 of the 3 splits.  Reported degradation is the median
        across passing splits (or worst split if all fail).

        Reduces sensitivity to the specific IS/OOS cut-point: a strategy that
        passes only because the 30% OOS window happened to be a favourable
        regime will typically fail at least one of the other two cuts.

        When trade_count < WF_MIN_TRADE_COUNT: returns neutral scores tagged as
        underpowered — the gate is not applied on fewer than 100 trades because
        a 30% OOS of 30 trades is 9 trades, which cannot distinguish real Sharpe
        from sampling noise.

        Returns
        -------
        (median_degradation, median_oos_sharpe, split_detail_list)
        split_detail_list contains one dict per split with keys:
            is_pct, is_sharpe, oos_sharpe, degradation, passed
        """
        if trade_count > 0 and trade_count < WF_MIN_TRADE_COUNT:
            stub = [
                {"is_pct": p, "is_sharpe": 0.0, "oos_sharpe": 0.0,
                 "degradation": 0.0, "passed": True, "underpowered": True}
                for p in (0.60, 0.70, 0.80)
            ]
            return 0.0, 0.0, stub

        def _sharpe(r: pd.Series) -> float:
            std = r.std(ddof=1)
            if std < 1e-10 or np.isnan(std):
                return 0.0
            daily_rf = RISK_FREE_RATE / TRADING_DAYS
            return float(np.clip(
                (r.mean() - daily_rf) / std * np.sqrt(TRADING_DAYS), -20.0, 20.0
            ))

        splits = []
        for is_pct in (0.60, 0.70, 0.80):
            cut       = int(len(returns) * is_pct)
            is_ret    = returns.iloc[:cut]
            oos_ret   = returns.iloc[cut:]
            is_s      = _sharpe(is_ret)
            oos_s     = _sharpe(oos_ret)

            if oos_s >= is_s or is_s <= 0:
                degrad = 0.0
            else:
                degrad = float(np.clip((is_s - oos_s) / is_s, 0.0, 1.0))

            splits.append({
                "is_pct":      is_pct,
                "is_sharpe":   is_s,
                "oos_sharpe":  oos_s,
                "degradation": degrad,
                "passed":      degrad <= WALKFWD_DEGRAD_FLOOR,
                "underpowered": False,
            })

        degradations = [s["degradation"] for s in splits]
        oos_sharpes  = [s["oos_sharpe"]  for s in splits]

        # Reported values: median across splits for robustness
        median_degrad    = float(np.median(degradations))
        median_oos_sharpe = float(np.median(oos_sharpes))

        return median_degrad, median_oos_sharpe, splits


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
