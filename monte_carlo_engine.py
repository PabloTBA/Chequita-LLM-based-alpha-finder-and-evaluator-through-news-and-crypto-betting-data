"""
MonteCarloEngine
================
Bootstrap-resamples a strategy's trade log to stress-test whether the
historical backtest results are robust or a product of lucky sequencing.

Method
------
Each simulation draws ``len(trade_log)`` trades with replacement from the
actual trade log, builds an equity curve from ``initial_portfolio``, and
records per-simulation metrics. 10,000 simulations are aggregated into
percentile distributions.

Output (all keys always present)
---------------------------------
  Equity distribution
    p5_final, p50_final, p95_final   — 5th/50th/95th percentile final equity
    p_ruin                           — fraction of sims that hit the ruin floor
    p95_max_drawdown                 — 95th-pct worst drawdown across sims
    median_cagr                      — median CAGR across sims
    equity_band                      — list of 20 {step,p5,p50,p95} dicts

  Sharpe distribution
    p5_sharpe, p50_sharpe, p95_sharpe

  Win-rate distribution
    p5_win_rate, p50_win_rate, p95_win_rate

  Tail risk
    p95_max_consec_losses            — int

  Position sizing
    kelly_fraction                   — median optimal Kelly across sims

  Ruin detail (None when p_ruin == 0)
    median_time_to_ruin              — median trade index at first ruin
    ruin_severity                    — mean final equity across ruined sims

Public interface
----------------
  mc     = MonteCarloEngine(n_simulations=10_000, ruin_threshold=0.40, seed=None)
  result = mc.run(trade_log, initial_portfolio=100_000.0)
"""

from __future__ import annotations

import math
import numpy as np

TRADING_DAYS = 252
EQUITY_BAND_STEPS = 20


class MonteCarloEngine:
    def __init__(
        self,
        n_simulations: int = 10_000,
        ruin_threshold: float = 0.40,
        seed: int | None = None,
    ):
        self.n_simulations  = n_simulations
        self.ruin_threshold = ruin_threshold
        self._rng           = np.random.default_rng(seed)

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, trade_log: list[dict], initial_portfolio: float = 100_000.0) -> dict:
        if not trade_log:
            return self._empty_result(initial_portfolio)

        pnls      = np.array([t["pnl"] for t in trade_log], dtype=float)
        n_trades  = len(pnls)
        ruin_floor = initial_portfolio * (1.0 - self.ruin_threshold)

        # Draw all simulations at once: shape (n_simulations, n_trades)
        indices = self._rng.integers(0, n_trades, size=(self.n_simulations, n_trades))
        sampled = pnls[indices]                          # (n_sims, n_trades)

        # Equity curves: shape (n_sims, n_trades+1)
        equity = np.hstack([
            np.full((self.n_simulations, 1), initial_portfolio),
            initial_portfolio + np.cumsum(sampled, axis=1),
        ])

        # ── per-simulation metrics ────────────────────────────────────────────
        final_equity = equity[:, -1]

        # Max drawdown per sim
        rolling_max  = np.maximum.accumulate(equity, axis=1)
        drawdowns    = (rolling_max - equity) / np.where(rolling_max > 0, rolling_max, 1)
        max_dd       = drawdowns.max(axis=1)

        # CAGR per sim (treat n_trades as n_trades trading days for scaling)
        years        = n_trades / TRADING_DAYS
        with np.errstate(invalid="ignore", divide="ignore"):
            cagr = np.where(
                (initial_portfolio > 0) & (final_equity > 0),
                (final_equity / initial_portfolio) ** (1.0 / years) - 1.0,
                -1.0,
            )

        # Sharpe per sim: daily P&L returns
        daily_returns  = sampled / initial_portfolio          # (n_sims, n_trades)
        ret_mean       = daily_returns.mean(axis=1)
        ret_std        = daily_returns.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe = np.where(
                ret_std > 0,
                ret_mean / ret_std * math.sqrt(TRADING_DAYS),
                np.where(ret_mean > 0, 1e6, np.where(ret_mean < 0, -1e6, 0.0)),
            )

        # Win rate per sim
        wins     = (sampled > 0).sum(axis=1)
        win_rate = wins / n_trades

        # Max consecutive losses per sim
        max_consec = self._max_consec_losses_batch(sampled)

        # Kelly fraction per sim: f* = W/L - (1-W)/G  where W=win_rate, G=avg_win, L=avg_loss
        kelly = self._kelly_batch(sampled, win_rate)

        # Ruin detection: did equity ever fall below ruin_floor?
        ruined      = (equity < ruin_floor).any(axis=1)      # bool (n_sims,)
        p_ruin      = float(ruined.mean())

        # Time-to-ruin: first trade index where equity < ruin_floor (per ruined sim)
        time_to_ruin: int | None = None
        ruin_severity: float | None = None
        if p_ruin > 0.0:
            ruin_mask       = equity < ruin_floor              # (n_sims, n_trades+1)
            # argmax finds first True (step 0 = before any trades = initial portfolio)
            first_ruin_step = ruin_mask.argmax(axis=1)         # (n_sims,) — 0 if no ruin
            ruined_steps    = first_ruin_step[ruined]          # only ruined sims
            # step index is into equity (0=start), so trade number = step - 1 (min 1)
            ruin_trade_nums = np.maximum(ruined_steps - 1, 1)
            time_to_ruin    = int(np.median(ruin_trade_nums))
            ruin_severity   = float(final_equity[ruined].mean())

        # ── equity confidence band (20 evenly-spaced steps) ──────────────────
        band_indices = np.linspace(0, n_trades, EQUITY_BAND_STEPS, dtype=int)
        equity_band  = [
            {
                "step": int(s),
                "p5":   float(np.percentile(equity[:, s], 5)),
                "p50":  float(np.percentile(equity[:, s], 50)),
                "p95":  float(np.percentile(equity[:, s], 95)),
            }
            for s in band_indices
        ]

        return {
            # equity distribution
            "p5_final":            float(np.percentile(final_equity, 5)),
            "p50_final":           float(np.percentile(final_equity, 50)),
            "p95_final":           float(np.percentile(final_equity, 95)),
            "p_ruin":              p_ruin,
            "p95_max_drawdown":    float(np.percentile(max_dd, 95)),
            "median_cagr":         float(np.median(cagr)),
            "equity_band":         equity_band,
            # Sharpe distribution
            "p5_sharpe":           float(np.percentile(sharpe, 5)),
            "p50_sharpe":          float(np.percentile(sharpe, 50)),
            "p95_sharpe":          float(np.percentile(sharpe, 95)),
            # win-rate distribution
            "p5_win_rate":         float(np.percentile(win_rate, 5)),
            "p50_win_rate":        float(np.percentile(win_rate, 50)),
            "p95_win_rate":        float(np.percentile(win_rate, 95)),
            # tail risk
            "p95_max_consec_losses": int(np.percentile(max_consec, 95)),
            # position sizing
            "kelly_fraction":      float(np.median(kelly)),
            # ruin detail
            "median_time_to_ruin": time_to_ruin,
            "ruin_severity":       ruin_severity,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _max_consec_losses_batch(sampled: np.ndarray) -> np.ndarray:
        """
        Max consecutive losses per simulation row.
        sampled: (n_sims, n_trades) — P&L values
        Returns: (n_sims,) int array
        """
        losses   = (sampled <= 0).astype(np.int32)   # 1 where loss
        n_sims, n_trades = losses.shape
        max_cl   = np.zeros(n_sims, dtype=np.int32)
        streak   = np.zeros(n_sims, dtype=np.int32)
        for j in range(n_trades):
            streak   = (streak + 1) * losses[:, j]
            max_cl   = np.maximum(max_cl, streak)
        return max_cl

    @staticmethod
    def _kelly_batch(sampled: np.ndarray, win_rate: np.ndarray) -> np.ndarray:
        """
        Kelly fraction per simulation: f* = W/L - (1-W)/G
        W = win_rate, G = mean win size, L = mean loss size (absolute).
        Returns 0.0 when no wins or no losses.
        """
        wins   = np.where(sampled > 0, sampled, 0.0)
        losses = np.where(sampled < 0, -sampled, 0.0)

        win_count  = (sampled > 0).sum(axis=1)
        loss_count = (sampled < 0).sum(axis=1)

        avg_win  = np.where(win_count  > 0, wins.sum(axis=1)   / np.maximum(win_count,  1), 1e-9)
        avg_loss = np.where(loss_count > 0, losses.sum(axis=1) / np.maximum(loss_count, 1), 1e-9)

        loss_rate = 1.0 - win_rate
        kelly = win_rate / avg_loss - loss_rate / avg_win
        return kelly

    @staticmethod
    def _empty_result(initial_portfolio: float) -> dict:
        band = [
            {"step": 0, "p5": initial_portfolio, "p50": initial_portfolio, "p95": initial_portfolio}
        ] * EQUITY_BAND_STEPS
        return {
            "p5_final":              initial_portfolio,
            "p50_final":             initial_portfolio,
            "p95_final":             initial_portfolio,
            "p_ruin":                0.0,
            "p95_max_drawdown":      0.0,
            "median_cagr":           0.0,
            "equity_band":           band,
            "p5_sharpe":             0.0,
            "p50_sharpe":            0.0,
            "p95_sharpe":            0.0,
            "p5_win_rate":           0.0,
            "p50_win_rate":          0.0,
            "p95_win_rate":          0.0,
            "p95_max_consec_losses": 0,
            "kelly_fraction":        0.0,
            "median_time_to_ruin":   None,
            "ruin_severity":         None,
        }
