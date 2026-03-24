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

TRADING_DAYS    = 252
EQUITY_BAND_STEPS = 20
RISK_FREE_RATE  = 0.045          # must match diagnostics_engine.py
DAILY_RF        = RISK_FREE_RATE / TRADING_DAYS


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

    def run(self, trade_log: list[dict], initial_portfolio: float = 100_000.0,
            ohlcv_years: float | None = None) -> dict:
        if not trade_log:
            print("  [MC]  No trades — returning empty result")
            return self._empty_result(initial_portfolio)

        pnls      = np.array([t["pnl"] for t in trade_log], dtype=float)
        n_trades  = len(pnls)
        ruin_floor = initial_portfolio * (1.0 - self.ruin_threshold)
        print(f"  [MC]  Running {self.n_simulations:,} simulations ({n_trades} trades, portfolio ${initial_portfolio:,.0f}) ...")

        # Use the full OHLCV window duration (passed from orchestrator) as the
        # annualisation denominator — NOT first_entry→last_exit, which understates
        # the window when trades cluster in one sub-period and wildly overstates CAGR.
        if ohlcv_years is not None and ohlcv_years > 0:
            backtest_years = ohlcv_years
        else:
            try:
                first_entry = min(t["entry_date"] for t in trade_log)
                last_exit   = max(t["exit_date"]  for t in trade_log)
                backtest_years = max((last_exit - first_entry).days / 365.25, 1 / 12)
            except Exception:
                backtest_years = n_trades / TRADING_DAYS

        # Block bootstrap: sample overlapping blocks of trades to preserve
        # the serial autocorrelation in momentum strategies (losing streaks
        # cluster during regime changes — IID bootstrap understates tail risk).
        # Block size = average holding period (in trades), minimum 2, maximum 10.
        avg_hold   = float(np.mean([max(t.get("holding_days", 1), 1) for t in trade_log]))
        # Upper bound: n_trades // 4 ensures at least 4 independent blocks per sim.
        # No hard cap at 10 — strategies with longer holds (e.g. 15–20 day swings)
        # need larger blocks to preserve the autocorrelation of multi-day losing streaks.
        block_size = int(np.clip(round(avg_hold), 2, max(2, n_trades // 4)))
        indices    = self._block_bootstrap(n_trades, self.n_simulations, block_size)
        sampled    = pnls[indices]                          # (n_sims, n_trades)

        # Equity curves: shape (n_sims, n_trades+1)
        equity = np.hstack([
            np.full((self.n_simulations, 1), initial_portfolio),
            initial_portfolio + np.cumsum(sampled, axis=1),
        ])

        # ── per-simulation metrics ────────────────────────────────────────────
        final_equity = equity[:, -1]

        # Max drawdown per sim — use initial_portfolio as the safe divisor floor so
        # that drawdown stays in [0, 1] even when equity momentarily goes negative
        # (possible with unleveraged PnL if a single trade loss exceeds the portfolio).
        rolling_max  = np.maximum.accumulate(equity, axis=1)
        safe_denom   = np.maximum(rolling_max, initial_portfolio * 1e-6)  # never divide by ≤ 0
        drawdowns    = (rolling_max - equity) / safe_denom
        max_dd       = drawdowns.max(axis=1)

        # CAGR per sim — annualised over actual backtest calendar duration.
        # Clamp to [-1, +∞): you cannot lose more than 100% in a non-leveraged account.
        with np.errstate(invalid="ignore", divide="ignore"):
            cagr = np.where(
                (initial_portfolio > 0) & (final_equity > 0),
                (final_equity / initial_portfolio) ** (1.0 / backtest_years) - 1.0,
                -1.0,   # total ruin or worse → -100% CAGR
            )

        # Sharpe per sim — convert each trade's dollar P&L to a per-day return so
        # that multi-day holds are correctly annualised.  Without this, a 20-day trade
        # returning $500 on $100k looks like a daily return of 0.5%, which when
        # annualised by √252 produces a Sharpe ≈ 3-4× higher than the diagnostics
        # engine (which operates on the actual daily returns series).
        holding_days = np.array(
            [max(t.get("holding_days", 1), 1) for t in trade_log], dtype=float
        )  # (n_trades,)
        # Broadcast: each bootstrapped trade gets its per-day P&L
        per_day_pnl   = pnls / holding_days                   # (n_trades,)
        sampled_daily = per_day_pnl[indices] / initial_portfolio  # (n_sims, n_trades)
        excess_mean   = sampled_daily.mean(axis=1) - DAILY_RF
        ret_std       = sampled_daily.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe = np.where(
                ret_std > 1e-10,
                excess_mean / ret_std * math.sqrt(TRADING_DAYS),
                np.where(excess_mean > 0, 20.0, np.where(excess_mean < 0, -20.0, 0.0)),
            )
        sharpe = np.clip(sharpe, -20.0, 20.0)

        # Win rate per sim
        wins     = (sampled > 0).sum(axis=1)
        win_rate = wins / n_trades

        # Max consecutive losses per sim
        max_consec = self._max_consec_losses_batch(sampled)

        # Kelly fraction per sim: f* = W/L - (1-W)/G  where W=win_rate, G=avg_win, L=avg_loss
        # Raw Kelly is clipped to [0, 1] — values > 1 imply leverage which is out of scope.
        kelly = np.clip(self._kelly_batch(sampled, win_rate), 0.0, 1.0)

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

        median_kelly      = float(np.median(kelly))
        half_kelly        = median_kelly / 2.0
        if median_kelly > 0.25:
            print(f"  [MC]  WARNING: full Kelly ({median_kelly:.1%}) > 25% — recommend half-Kelly "
                  f"({half_kelly:.1%}) to reduce variance.  Never bet full Kelly in live trading.")

        print(f"  [MC]  Done — p_ruin={p_ruin:.1%}  median_CAGR={float(np.median(cagr)):.1%}  "
              f"p50_equity=${float(np.percentile(final_equity, 50)):,.0f}  "
              f"p95_maxDD={float(np.percentile(max_dd, 95)):.1%}  "
              f"Kelly={median_kelly:.1%}  half-Kelly={half_kelly:.1%}")
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
            # position sizing — always surface both full and half-Kelly
            "kelly_fraction":      median_kelly,
            "half_kelly_fraction": half_kelly,
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
        Kelly fraction per simulation: f* = W/L − (1−W)/G

        W = win_rate, G = mean win size (absolute), L = mean loss size (absolute).

        Edge cases handled explicitly — using sentinel floats avoids NaN propagation
        that would occur if 1e-9 denominators were passed through division chains:
          • No wins  in a sim → all-losing sequence → Kelly = -1.0 (worst)
          • No losses in a sim → all-winning sequence → Kelly = +1.0 (full)
          • All flat trades   → zero variance → Kelly = 0.0

        The outer np.clip(kelly, 0, 1) in the caller then floors negative values at 0.
        """
        wins   = np.where(sampled > 0, sampled, 0.0)
        losses = np.where(sampled < 0, -sampled, 0.0)   # positive values

        win_count  = (sampled > 0).sum(axis=1)
        loss_count = (sampled < 0).sum(axis=1)

        has_wins   = win_count  > 0
        has_losses = loss_count > 0

        # Safe averages: only divide when we have the relevant trades
        avg_win  = np.where(has_wins,   wins.sum(axis=1)   / np.maximum(win_count,   1), 0.0)
        avg_loss = np.where(has_losses, losses.sum(axis=1) / np.maximum(loss_count, 1), 0.0)

        loss_rate = 1.0 - win_rate

        # Compute Kelly only for sims that have both wins and losses; use sentinels otherwise.
        # np.where evaluates both branches so guard against division by zero explicitly.
        both = has_wins & has_losses
        kelly = np.where(
            both,
            win_rate / np.where(has_losses, avg_loss, 1.0)
            - loss_rate / np.where(has_wins, avg_win, 1.0),
            np.where(has_wins, 1.0, -1.0),   # all-win → 1.0; all-loss (or flat) → -1.0
        )
        return kelly

    def _block_bootstrap(self, n_trades: int, n_sims: int, block_size: int) -> np.ndarray:
        """
        Circular block bootstrap: sample overlapping blocks of trades.

        Each simulation samples ceil(n_trades / block_size) random starting
        positions and takes blocks of length block_size, wrapping circularly.
        Preserves local autocorrelation between consecutive trades.

        Returns indices array of shape (n_sims, n_trades).
        """
        n_blocks = math.ceil(n_trades / block_size)
        # Random starting positions: (n_sims, n_blocks)
        starts  = self._rng.integers(0, n_trades, size=(n_sims, n_blocks))
        offsets = np.arange(block_size, dtype=np.int64)  # (block_size,)
        # Build all block indices: (n_sims, n_blocks, block_size)
        block_indices = (starts[:, :, None] + offsets[None, None, :]) % n_trades
        # Flatten to (n_sims, n_blocks * block_size) and trim to n_trades
        return block_indices.reshape(n_sims, -1)[:, :n_trades]

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
            "half_kelly_fraction":   0.0,
            "median_time_to_ruin":   None,
            "ruin_severity":         None,
        }
