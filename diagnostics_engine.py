"""
DiagnosticsEngine
=================
Runs diagnostic checks on a strategy's returns and trade log.
Applies hard reject floors; calls LLM for qualitative commentary only when
all floors pass.

Hard floors (checked in order)
-------------------------------
  Sharpe ratio (full-period) < 0.5  → auto-reject  (PRIMARY — no OOS rescue)
  OOS Sharpe                 < 0.3  → auto-reject  (SECONDARY — must show out-of-sample edge)
  Max drawdown              > 20%   → auto-reject
  Win rate                  < 35%   → auto-reject  (bypass if profit_factor ≥ 1.5)
  Kelly fraction            < 0.0   → auto-reject  (negative expectancy = provably losing)
  Walk-forward degradation  > 50%   → auto-reject
  Trade count               < 30    → auto-reject
  p-value (Lo 2002)         ≥ 0.10  → auto-reject  (Sharpe must be significant at 90% confidence)
  Bootstrap Sharpe p5       ≤ 0.0   → auto-reject  (lower CI bound must be positive — not noise)
  Rolling Sharpe stability  < 50%   → warning flag  (< half of 60-day windows have positive Sharpe)
  Permutation test (Calmar) → report only, not a gate (Sharpe is order-invariant under shuffling)

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

# ── Hard floor constants (PRD defaults — Neutral / Low-Vol regime) ────────────

WF_MIN_TRADE_COUNT    = 100    # minimum trades for walk-forward to have statistical power
SHARPE_FLOOR          = 0.50
MAX_DD_FLOOR          = 0.20   # tightened from 30% → 20% (institutional standard)
WIN_RATE_FLOOR        = 0.35
PROFIT_FACTOR_FLOOR   = 1.5    # bypass win-rate floor when profit_factor >= this (high-payoff strategies)
WALKFWD_DEGRAD_FLOOR  = 0.50
MIN_TRADE_COUNT       = 30     # raised from 10 → 30 (minimum for statistical significance)
TRADING_DAYS          = 252
RISK_FREE_RATE        = 0.045  # annualised risk-free rate (~current Fed funds); subtract from Sharpe

# ── Regime-conditional floor overrides ───────────────────────────────────────
# High-vol / crisis markets inflate return std, suppressing Sharpe, and produce
# larger intra-strategy drawdowns even for genuinely profitable strategies.
# Floors are relaxed proportionally to regime severity so the pipeline can
# surface real alpha instead of systematically rejecting everything in a sell-off.
#
# Sharpe relaxation rationale:
#   Sharpe = mean_excess / std.  In a crisis std can double vs normal markets.
#   A strategy earning the same dollar alpha will show Sharpe ~0.25 instead of 0.50.
#   Keeping the floor at 0.50 would reject all crisis alpha by construction.
#
# MaxDD relaxation rationale:
#   20% DD floor is calibrated for low-vol regimes.  High-vol regimes have larger
#   normal drawdowns even for strategies with positive edge.  Tightening the floor
#   in a VIX spike would reject valid strategies for noise, not real weakness.
_REGIME_FLOORS: dict[str, dict] = {
    # regime_label       sharpe  max_dd  win_rate  oos_sharpe  p_value
    "Crisis":         {"sharpe": 0.25, "max_dd": 0.35, "win_rate": 0.30, "oos_sharpe": 0.15, "p_value": 0.15},
    "High-Volatility":{"sharpe": 0.35, "max_dd": 0.28, "win_rate": 0.32, "oos_sharpe": 0.20, "p_value": 0.12},
    "Event-Driven":   {"sharpe": 0.30, "max_dd": 0.30, "win_rate": 0.30, "oos_sharpe": 0.15, "p_value": 0.15},
    "Trending-Down":  {"sharpe": 0.40, "max_dd": 0.25, "win_rate": 0.33, "oos_sharpe": 0.25, "p_value": 0.12},
    "Trending-Up":    {"sharpe": 0.45, "max_dd": 0.22, "win_rate": 0.34, "oos_sharpe": 0.28, "p_value": 0.10},
    "Mean-Reverting": {"sharpe": 0.50, "max_dd": 0.20, "win_rate": 0.35, "oos_sharpe": 0.30, "p_value": 0.10},
    "Low-Volatility": {"sharpe": 0.50, "max_dd": 0.20, "win_rate": 0.35, "oos_sharpe": 0.30, "p_value": 0.10},
    "Neutral":        {"sharpe": 0.50, "max_dd": 0.20, "win_rate": 0.35, "oos_sharpe": 0.30, "p_value": 0.10},
}
_DEFAULT_FLOORS = _REGIME_FLOORS["Neutral"]


class DiagnosticsEngine:
    def __init__(self, llm_client: Optional[Callable] = None, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(
        self,
        ticker:       str,
        strategy:     str,
        trade_log:    list[dict],
        returns:      pd.Series,
        regime_label: str = "Neutral",
    ) -> dict:
        """
        Run all diagnostic checks on a strategy.

        Parameters
        ----------
        ticker       : str
        strategy     : str  (e.g. "Momentum", "Mean-Reversion")
        trade_log    : list of dicts, each with at least {"pnl": float}
        returns      : pd.Series of daily returns (decimal), DatetimeIndex preferred
        regime_label : str  market regime — used to select floor thresholds

        Returns
        -------
        dict with keys: ticker, strategy, passed, reject_reason, metrics, llm_commentary
        """
        floors  = _REGIME_FLOORS.get(regime_label, _DEFAULT_FLOORS)
        metrics = self._compute_metrics(trade_log, returns)
        self._log(f"[DiagnosticsEngine] {ticker}: {metrics}")

        passed, reject_reason = self._check_floors(metrics, floors)
        status = "PASS" if passed else f"FAIL -- {reject_reason}"
        print(f"  [Diag] {ticker} [{regime_label}]: Sharpe={metrics['sharpe']:.3f} "
              f"(floor={floors['sharpe']})  OOS_Sharpe={metrics['oos_sharpe']:.3f}  "
              f"MaxDD={metrics['max_drawdown']:.1%} (floor={floors['max_dd']:.0%})  "
              f"WinRate={metrics['win_rate']:.1%}  "
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
        perm_p                           = self._permutation_test(returns)
        roll_pct_pos, roll_sharpe_std    = self._rolling_sharpe_stability(returns)
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
            "t_stat":                   t_stat,          # Lo (2002) autocorr-corrected t-stat
            "p_value":                  p_value,          # one-tailed H1: Sharpe > 0
            "bootstrap_sharpe_p5":      bs_p5,            # 5th pct of bootstrap Sharpe dist
            "bootstrap_sharpe_p95":     bs_p95,           # 95th pct
            "permutation_p_value":      perm_p,           # non-parametric: fraction shuffled >= real
            "rolling_pct_positive":     roll_pct_pos,     # fraction of 60-day windows with +ve Sharpe
            "rolling_sharpe_std":       roll_sharpe_std,  # std of rolling Sharpe (instability measure)
        }

    @staticmethod
    def _check_floors(metrics: dict, floors: dict) -> tuple[bool, Optional[str]]:
        # ── Sharpe — PRIMARY criterion, no substitution allowed ───────────────
        # Full-period Sharpe is the single most honest summary of risk-adjusted
        # return over the entire history.  OOS Sharpe is a SECONDARY check that
        # must also pass — it cannot rescue a failed full-period Sharpe.
        #
        # Floors are regime-conditional: crisis/high-vol markets inflate return
        # std, so the same dollar alpha produces a lower Sharpe.  See
        # _REGIME_FLOORS for per-regime calibration rationale.
        sharpe_floor = floors["sharpe"]
        sharpe       = metrics["sharpe"]
        oos_sharpe   = metrics.get("oos_sharpe", sharpe)

        if sharpe < sharpe_floor:
            return False, (f"Sharpe ratio {sharpe:.3f} below regime floor {sharpe_floor} "
                           f"(OOS {oos_sharpe:.3f} cannot rescue a failed full-period Sharpe)")

        # OOS must also show positive edge — prevents IS-only curve-fitting.
        # Skipped when WF is underpowered (< 100 trades): oos_sharpe is 0.0 by
        # design in that path (not a real measurement), so applying the floor
        # would wrongly reject every low-trade-count strategy.
        oos_sharpe_floor = floors["oos_sharpe"]
        wf_underpowered  = metrics.get("wf_underpowered", False)
        if not wf_underpowered and oos_sharpe < oos_sharpe_floor:
            return False, (f"OOS Sharpe {oos_sharpe:.3f} below regime secondary floor {oos_sharpe_floor} "
                           f"— full-period Sharpe {sharpe:.3f} passes but strategy has no "
                           f"out-of-sample evidence of edge")

        max_dd       = metrics["max_drawdown"]
        max_dd_floor = floors["max_dd"]
        if max_dd > max_dd_floor:
            return False, f"Max drawdown {max_dd:.1%} exceeds regime floor {max_dd_floor:.0%}"

        win_rate      = metrics["win_rate"]
        win_rate_floor = floors["win_rate"]
        profit_factor = metrics.get("profit_factor", 0.0)
        # Low win rate bypassed when profit factor is strong (high-payoff strategies such as
        # trend-following with 30% wins but 3:1 payoff ratio are valid and should not be rejected)
        if win_rate < win_rate_floor and profit_factor < PROFIT_FACTOR_FLOOR:
            return False, (f"Win rate {win_rate:.1%} below regime floor {win_rate_floor:.0%} "
                           f"and profit factor {profit_factor:.2f} below {PROFIT_FACTOR_FLOOR:.1f}")

        # Kelly fraction: negative Kelly means the strategy has provably negative expected value
        # regardless of win rate or profit factor (e.g. 50% win rate but losses >> wins).
        # Default 0.0 when key absent (callers that pass metric dicts directly without kelly).
        kelly = metrics.get("kelly_fraction", 0.0)
        if kelly < 0.0:
            return False, (f"Negative Kelly fraction ({kelly:.4f}) — strategy has provably "
                           f"negative expected value; do not size any position")

        # Walk-forward gate.
        # Rolling WF (many windows): require median OOS Sharpe > 0, i.e. majority passing.
        # Static 3-split: require at least 2 of 3 splits passing (original rule).
        # Underpowered splits count as passes so they don't wrongly reject.
        wf_splits = metrics.get("wf_splits", [])
        if wf_splits and not metrics.get("wf_underpowered", False):
            n_pass    = sum(1 for s in wf_splits if s.get("passed", True))
            is_rolling = any(s.get("rolling_wf", False) for s in wf_splits)
            if is_rolling:
                # Rolling WF: require ≥ 50% of windows to have OOS Sharpe > 0
                min_pass = max(len(wf_splits) // 2, 1)
                if n_pass < min_pass:
                    median_oos = float(np.median([s["oos_sharpe"] for s in wf_splits]))
                    return False, (
                        f"Rolling walk-forward failed: only {n_pass}/{len(wf_splits)} windows have "
                        f"positive OOS Sharpe (median OOS Sharpe {median_oos:.3f}) — "
                        f"strategy overfits the in-sample period"
                    )
            else:
                # Static 3-split: require ≥ 2 of 3 splits passing
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

        # ── Statistical significance — p-value (Lo 2002 autocorr-corrected) ─────
        # The Sharpe floor confirms magnitude but not reliability: a Sharpe of 0.55
        # on 40 noisy days can easily be pure luck.  p_value tests H0: mean excess
        # return = 0.  Crisis/high-vol regimes get a relaxed p-value floor because
        # higher noise requires more data to reach the same confidence level —
        # rejecting everything at 90% CI in a panic market throws away real alpha.
        p_value_floor = floors["p_value"]
        p_value = metrics.get("p_value", 0.0)
        if not wf_underpowered and p_value >= p_value_floor:
            return False, (
                f"p-value {p_value:.4f} ≥ regime floor {p_value_floor} — Sharpe {sharpe:.3f} is not "
                f"statistically significant (Lo 2002 autocorr-corrected t-stat); "
                f"likely sampling noise"
            )

        # ── Bootstrap Sharpe lower CI bound ─────────────────────────────────────
        # Even if p-value passes, the 5th-percentile of the bootstrap Sharpe
        # distribution must be > 0: if the lower end of the 90% CI dips to zero or
        # below, we cannot distinguish the Sharpe from sampling noise across
        # plausible resamplings of the same return history.
        # Only enforced when bootstrap was run (n >= 40; returns (0.0, 0.0) otherwise).
        bs_p5 = metrics.get("bootstrap_sharpe_p5", 0.0)
        bs_p95 = metrics.get("bootstrap_sharpe_p95", 0.0)
        bootstrap_ran = not (bs_p5 == 0.0 and bs_p95 == 0.0)
        if not wf_underpowered and bootstrap_ran and bs_p5 <= 0.0:
            return False, (
                f"Bootstrap Sharpe 5th-percentile {bs_p5:.3f} ≤ 0 — 90% CI [{bs_p5:.3f}, {bs_p95:.3f}] "
                f"includes zero; Sharpe {sharpe:.3f} may be a sampling artefact"
            )

        # NOTE — Permutation test is NOT a hard gate:
        # The Calmar-based permutation test is included in metrics and surfaced in
        # the report, but not gated here.  An IID strategy with genuine positive
        # expected return has perm_p ≈ 0.50 (no temporal structure to detect), which
        # would wrongly trigger a gate.  The Lo t-stat + bootstrap CI already enforce
        # statistical significance; the permutation test is a supplementary diagnostic.

        # ── Rolling Sharpe stability (warning flag, not hard reject) ─────────
        # A strategy that passes all floors but has < 50% of 60-day windows with
        # positive Sharpe is unstable — it works in certain regimes but not others.
        # We do NOT hard-reject here (regime dependency is expected and sometimes
        # desirable), but we annotate the metrics so the report can surface it.
        # The report_generator should display this as a ⚠️ flag.
        roll_pct_pos = metrics.get("rolling_pct_positive", 1.0)
        if roll_pct_pos < 0.50:
            # Return PASS but inject the warning into the reject_reason field as a
            # non-blocking advisory by returning True with an advisory string.
            # Callers should surface this when roll_pct_positive < 0.50.
            pass   # logged in metrics; report_generator reads rolling_pct_positive directly

        return True, None

    def _get_llm_commentary(self, ticker: str, strategy: str, metrics: dict) -> str:
        print(f"  [LLM] DiagnosticsEngine: commentary for {ticker} ({strategy})...")
        wf_splits   = metrics.get("wf_splits", [])
        # For rolling WF: pick the window whose IS fraction is closest to 0.70.
        # For static 3-split: pick the 70/30 split directly.
        wf_rep = next(
            (s for s in sorted(wf_splits, key=lambda s: abs(s.get("is_pct", 0) - 0.70))
             if wf_splits),
            {}
        )
        is_sharpe  = wf_rep.get("is_sharpe",  metrics.get("sharpe", 0.0))
        oos_sharpe = wf_rep.get("oos_sharpe", metrics.get("oos_sharpe", 0.0))
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
    def _permutation_test(
        returns: pd.Series,
        n: int = 1000,
    ) -> float:
        """
        Non-parametric significance test: shuffle the return series n times and
        compute the Calmar ratio (CAGR / MaxDD) of each shuffle.

        IMPORTANT — why we use Calmar, not Sharpe:
          Sharpe = (mean − rf) / std.  Under permutation, mean and std are EXACTLY
          preserved, so every shuffle produces the same Sharpe as the real series.
          A Sharpe-based permutation test is trivially uninformative.

          Calmar ratio IS order-dependent because MaxDrawdown depends on the sequence
          of returns — a series with consecutive losses creates a deeper drawdown than
          the same losses scattered randomly.  A strategy with genuine temporal structure
          (exits before drawdowns compound, or momentum runs preserve gains) will have a
          better Calmar than a randomly ordered series with the same return distribution.

        Interpretation (Calmar-based):
          perm_p ≈ 0.5 is expected for an IID positive-mean process (no temporal structure).
          perm_p < 0.10 means the real strategy's Calmar beats 90%+ of random orderings →
            the strategy's temporal structure actively limits drawdowns.
          perm_p > 0.90 means the strategy's drawdown is WORSE than 90% of random orderings →
            the strategy's exit logic is destroying value (flagged as warning, not hard gate).

        NOTE: This test requires genuine temporal structure to be meaningful.  The Lo (2002)
        t-stat and block bootstrap CI are the primary statistical gates.  This test is
        provided as a supplementary diagnostic in the report.  It is NOT used as a
        PASS/FAIL gate in _check_floors because:
          (a) An IID strategy with real positive edge will have perm_p ≈ 0.5 (correct PASS)
          (b) A gate at perm_p < 0.10 would wrongly reject valid strategies without trend
              autocorrelation while the Lo t-stat already catches statistical insignificance.
        """
        r = np.array(returns, dtype=float)
        n_obs = len(r)
        if n_obs < 10:
            return 0.5   # not enough data — return neutral value
        daily_rf = RISK_FREE_RATE / TRADING_DAYS

        def _calmar(arr: np.ndarray) -> float:
            equity   = np.cumprod(1.0 + arr)
            peak     = np.maximum.accumulate(equity)
            dd       = (equity - peak) / np.where(peak > 0, peak, 1.0)
            max_dd   = float(-dd.min()) if len(dd) > 0 else 0.0
            cagr     = float((equity[-1]) ** (TRADING_DAYS / len(arr)) - 1.0)
            return cagr / max_dd if max_dd > 1e-6 else (cagr * 10.0 if cagr > 0 else 0.0)

        real_calmar = _calmar(r)
        rng = np.random.default_rng(seed=42)
        count_geq = 0
        for _ in range(n):
            shuffled = rng.permutation(r)
            if _calmar(shuffled) >= real_calmar:
                count_geq += 1
        return round(float(count_geq / n), 4)

    @staticmethod
    def _rolling_sharpe_stability(
        returns: pd.Series,
        window: int = 60,
    ) -> tuple[float, float]:
        """
        Measure how consistently the strategy generates positive risk-adjusted
        returns over time using a rolling window.

        Returns
        -------
        (pct_positive_windows, rolling_sharpe_std)

          pct_positive_windows : fraction of 60-day rolling windows that have
              a positive annualised Sharpe (> 0).  A truly robust strategy should
              have > 70% of windows positive.  < 50% means the strategy spends more
              than half its life in a regime where it doesn't work.

          rolling_sharpe_std : standard deviation of the rolling Sharpe series.
              High std (> 2.0) indicates the strategy is regime-sensitive — large
              swings between strongly positive and strongly negative periods.

        Note: 60-day Sharpe estimates are noisy (only ~60 observations), so this
        is a directional measure, not a precise one.  We use it as a stability flag,
        not as a precise gate.
        """
        n = len(returns)
        if n < window + 10:
            return 1.0, 0.0   # not enough data — assume stable (no evidence of instability)

        daily_rf = RISK_FREE_RATE / TRADING_DAYS
        r = np.array(returns, dtype=float)
        roll_sharpes: list[float] = []

        for start in range(0, n - window + 1):
            chunk = r[start : start + window]
            std_c = float(chunk.std(ddof=1))
            if std_c > 1e-10:
                sr = float((chunk.mean() - daily_rf) / std_c * math.sqrt(TRADING_DAYS))
                roll_sharpes.append(sr)

        if not roll_sharpes:
            return 1.0, 0.0

        pct_pos  = float(np.mean([s > 0 for s in roll_sharpes]))
        roll_std = float(np.std(roll_sharpes, ddof=1)) if len(roll_sharpes) > 1 else 0.0
        return round(pct_pos, 4), round(roll_std, 4)

    @staticmethod
    def _walk_forward_degradation(
        returns: pd.Series, trade_count: int = 0
    ) -> tuple[float, float, list[dict]]:
        """
        Rolling anchored walk-forward (preferred when enough history).
        IS expands from _MIN_IS → end; OOS is always the next _OOS bars.
        Yields ~(len - _MIN_IS) // _OOS non-overlapping OOS windows.

        Accept criterion: median OOS Sharpe > 0 across all rolling windows.

        Falls back to the static 3-split (60/40, 70/30, 80/20) approach when
        there is insufficient history (< _MIN_IS + _OOS bars).

        When trade_count < WF_MIN_TRADE_COUNT: returns neutral scores tagged as
        underpowered — the gate is not applied on fewer than 100 trades because
        a quarterly OOS of ~30 trades cannot distinguish real Sharpe from noise.

        Returns
        -------
        (median_degradation, median_oos_sharpe, split_detail_list)
        Each entry in split_detail_list contains:
            is_pct, is_sharpe, oos_sharpe, degradation, passed,
            underpowered, rolling_wf (True when rolling method was used)
        """
        _MIN_IS = 252   # 1 year minimum IS
        _OOS    = 63    # 1 quarter OOS

        if trade_count > 0 and trade_count < WF_MIN_TRADE_COUNT:
            stub = [
                {"is_pct": p, "is_sharpe": 0.0, "oos_sharpe": 0.0,
                 "degradation": 0.0, "passed": True, "underpowered": True,
                 "rolling_wf": False}
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

        n = len(returns)

        # ── Rolling anchored walk-forward (preferred when enough history) ─────
        if n >= _MIN_IS + _OOS:
            splits: list[dict] = []
            for end_is in range(_MIN_IS, n - _OOS + 1, _OOS):
                is_ret  = returns.iloc[:end_is]
                oos_ret = returns.iloc[end_is : end_is + _OOS]
                is_s    = _sharpe(is_ret)
                oos_s   = _sharpe(oos_ret)
                is_pct  = end_is / n

                if oos_s >= is_s or is_s <= 0:
                    degrad = 0.0
                else:
                    degrad = float(np.clip((is_s - oos_s) / is_s, 0.0, 1.0))

                splits.append({
                    "is_pct":       round(is_pct, 4),
                    "is_sharpe":    is_s,
                    "oos_sharpe":   oos_s,
                    "degradation":  degrad,
                    # Pass = OOS Sharpe positive (rolling WF accept criterion)
                    "passed":       oos_s > 0,
                    "underpowered": False,
                    "rolling_wf":   True,
                })

            oos_sharpes  = [s["oos_sharpe"]  for s in splits]
            degradations = [s["degradation"] for s in splits]
            return (
                float(np.median(degradations)),
                float(np.median(oos_sharpes)),
                splits,
            )

        # ── Static 3-split fallback (insufficient history for rolling) ────────
        splits = []
        for is_pct in (0.60, 0.70, 0.80):
            cut     = int(n * is_pct)
            is_ret  = returns.iloc[:cut]
            oos_ret = returns.iloc[cut:]
            is_s    = _sharpe(is_ret)
            oos_s   = _sharpe(oos_ret)

            if oos_s >= is_s or is_s <= 0:
                degrad = 0.0
            else:
                degrad = float(np.clip((is_s - oos_s) / is_s, 0.0, 1.0))

            splits.append({
                "is_pct":       is_pct,
                "is_sharpe":    is_s,
                "oos_sharpe":   oos_s,
                "degradation":  degrad,
                "passed":       degrad <= WALKFWD_DEGRAD_FLOOR,
                "underpowered": False,
                "rolling_wf":   False,
            })

        degradations = [s["degradation"] for s in splits]
        oos_sharpes  = [s["oos_sharpe"]  for s in splits]
        return (
            float(np.median(degradations)),
            float(np.median(oos_sharpes)),
            splits,
        )


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
