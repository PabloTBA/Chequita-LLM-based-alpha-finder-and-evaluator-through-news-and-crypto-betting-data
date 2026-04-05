"""
Backtester
==========
Executes one of six strategy rules on the full OHLCV history supplied by the
caller (typically 5–10 years of daily data) and returns a trade log, equity
curve, daily returns series, and summary stats.

Strategies
----------
  Momentum          — N-day high breakout + volume confirmation + 12-1m gate
  Mean-Reversion    — RSI oversold + below lower Bollinger Band
  VolatilityBreakout— BB squeeze → expansion breakout + volume confirmation
  AlphaCombined     — pre-computed cross-sectional multi-factor alpha signal
  MLSignal          — ensemble ML probability signal (4-model average)
  EventDriven       — Post-Earnings Announcement Drift (PEAD): enter after blackout,
                      ride earnings gap drift; exit on PEAD fade or max hold

Position sizing (volatility-adjusted)
--------------------------------------
  risk per trade  = 1% of current portfolio
  stop distance   = stop_loss_atr × ATR_at_entry
  position_size   = (portfolio × 0.01) / stop_distance

Exit priority (Momentum)
-------------------------
  1. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  2. Trailing stop    : close < peak  − trailing_stop_atr × current_ATR
  3. MA cross         : close < ma_exit_period-day MA
  4. Max holding      : holding_days ≥ max_holding_days

Exit priority (Mean-Reversion)
-------------------------------
  1. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  2. RSI exit         : RSI(14) > rsi_exit_threshold
  3. Middle BB        : close ≥ bb_period-day MA
  4. Max holding      : holding_days ≥ max_holding_days

Exit priority (VolatilityBreakout)
------------------------------------
  1. Trailing stop    : close < peak  − trailing_stop_atr × ATR
  2. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  3. Max holding      : holding_days ≥ max_holding_days

Exit priority (AlphaCombined)
------------------------------
  1. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  2. Trailing stop    : close < peak  − trailing_stop_atr × ATR
  3. Alpha reversal   : alpha_signal < reversal_threshold  (signal flips negative)
  4. Max holding      : holding_days ≥ max_holding_days

Exit priority (MLSignal)
--------------------------
  1. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  2. Trailing stop    : close < peak  − trailing_stop_atr × ATR
  3. ML reversal      : ml_signal < reversal_threshold  (model loses conviction)
  4. Max holding      : holding_days ≥ max_holding_days

Entry conditions (EventDriven)
--------------------------------
  1. Positive earnings gap > gap_threshold occurred within last entry_window_bars
  2. pead_signal > pead_min_signal  (PEAD drift still active; filled forward 60d)
  3. Close > ma_filter_period-day MA  (confirms upward drift continuation)
  4. Volume > volume_mult × 20-bar average  (elevated post-announcement flow)
  5. NOT inside earnings blackout window  (trade the drift, not the announcement)

Exit priority (EventDriven)
-----------------------------
  1. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  2. Trailing stop    : close < peak  − trailing_stop_atr × ATR
  3. PEAD fade        : pead_signal < pead_exit_threshold  (drift reversed)
  4. Max holding      : holding_days ≥ max_holding_days

Public interface
----------------
  bt     = Backtester(initial_portfolio=100_000.0)
  result = bt.run(ticker, strategy, ohlcv_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_PER_TRADE    = 0.01   # 1% of portfolio per trade
ATR_PERIOD        = 14
RSI_PERIOD        = 14
DEFAULT_SLIP_BPS  = 10     # 10 basis points (0.10%) per side — fallback only
ANNUAL_RF         = 0.045  # risk-free rate — must match diagnostics_engine.py
DAILY_RF          = ANNUAL_RF / 252  # T-bill daily return earned on idle (flat) days
BORROW_RATE_DAILY = 0.005 / 252      # 50 bps/yr short-leg stock borrow cost (liquid large caps)

# ADV-tiered slippage (basis points per side).
# Mega-caps trade inside the spread; small-caps incur significant market impact.
# Tiers calibrated to typical NYSE/NASDAQ retail execution:
#   > 5M shares/day  → liquid, spread < 1¢ on most names → 5bps
#   1M–5M            → normal mid-cap execution           → 10bps
#   100K–1M          → thin mid-cap, wider spreads        → 25bps
#   < 100K           → illiquid — expect 50–150bps realized → 75bps
_SLIP_TIERS: list[tuple[float, float]] = [
    (5_000_000, 5.0),    # ADV ≥ 5M  → 5bps
    (1_000_000, 10.0),   # ADV ≥ 1M  → 10bps
    (  100_000, 25.0),   # ADV ≥ 100K → 25bps
    (       0,  75.0),   # ADV < 100K → 75bps
]


def _slip_bps_for_adv(adv_shares: float) -> float:
    """Return the appropriate one-side slippage in basis points given ADV."""
    for threshold, bps in _SLIP_TIERS:
        if adv_shares >= threshold:
            return bps
    return 75.0   # safety fallback


class Backtester:
    def __init__(self, initial_portfolio: float = 100_000.0, slippage_bps: float = DEFAULT_SLIP_BPS):
        self.initial_portfolio = initial_portfolio
        self._default_slip_bps = slippage_bps
        # NOTE: _slip is the DEFAULT rate only; per-call slippage is computed
        # locally inside run() / run_pair() and never mutates this instance attribute.
        self._slip = slippage_bps / 10_000  # convert to fraction

    # ── public ────────────────────────────────────────────────────────────────

    def _resolve_slip(self, adv_shares: float) -> float:
        """Return the slippage fraction for a single call without mutating state."""
        if adv_shares > 0:
            return _slip_bps_for_adv(adv_shares) / 10_000
        return self._default_slip_bps / 10_000

    def run(self, ticker: str, strategy: dict, ohlcv: pd.DataFrame,
            adv_shares: float = 0.0) -> dict:
        """
        Back-test a strategy against OHLCV data.

        Parameters
        ----------
        ticker     : str
        strategy   : dict — output of StrategySelector.select(); must contain
                     "strategy" (str) and "adjusted_params" (dict)
        ohlcv      : pd.DataFrame with Open/High/Low/Close/Volume columns
        adv_shares : float — 20-day average daily volume in shares.  When > 0,
                     overrides the instance default with the ADV-tiered slippage
                     so backtest costs reflect actual liquidity.

        Returns
        -------
        dict with keys: ticker, strategy, trade_log, equity_curve, returns, summary,
                        slippage_bps (the rate actually used)
        """
        # Compute slippage locally — never mutate self._slip so parallel / sequential
        # calls do not bleed state into each other.
        slip = self._resolve_slip(adv_shares)

        strategy_type = strategy["strategy"]
        params        = strategy["adjusted_params"]

        if strategy_type == "Momentum":
            trade_log = self._run_momentum(ohlcv, params, slip)
        elif strategy_type == "VolatilityBreakout":
            trade_log = self._run_volatility_breakout(ohlcv, params, slip)
        elif strategy_type == "AlphaCombined":
            trade_log = self._run_alpha_combined(ohlcv, params, slip)
        elif strategy_type == "MLSignal":
            trade_log = self._run_ml_signal(ohlcv, params, slip)
        elif strategy_type == "EventDriven":
            trade_log = self._run_event_driven(ohlcv, params, slip)
        else:
            trade_log = self._run_mean_reversion(ohlcv, params, slip)

        equity_curve = self._build_equity_curve(ohlcv, trade_log)
        returns      = self._build_returns(ohlcv, trade_log, equity_curve)
        if strategy_type == "EventDriven":
            summary = self._summarize_event_driven(trade_log, equity_curve)
        else:
            summary = self._summarize(trade_log, equity_curve)

        # in_position series — used for exposure-adjusted benchmark
        in_pos = pd.Series(False, index=ohlcv.index)
        for trade in trade_log:
            try:
                in_pos.loc[trade["entry_date"]:trade["exit_date"]] = True
            except Exception:
                pass

        used_slip_bps = round(slip * 10_000, 1)
        return {
            "ticker":       ticker,
            "strategy":     strategy_type,
            "trade_log":    trade_log,
            "equity_curve": equity_curve,
            "returns":      returns,
            "in_position":  in_pos,
            "summary":      summary,
            "slippage_bps": used_slip_bps,   # surfaced in report for transparency
        }

    # ── portfolio backtest (cross-sectional) ──────────────────────────────────

    def run_alpha_combined_portfolio(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        params: dict,
    ) -> dict:
        """
        Cross-sectional portfolio backtest for AlphaCombined.

        AlphaCombined signals are z-scored across the universe, so a single
        ticker's signal has no meaning in isolation — the market-neutral property
        requires simultaneous long positions across multiple tickers ranked by
        their cross-sectional alpha_signal.

        Simulation rules
        ----------------
        - Each day: enter equally-weighted long positions in every ticker where
          alpha_signal > alpha_threshold (and not in earnings blackout).
        - Each position sizes by 1% portfolio risk / (stop_loss_atr × ATR).
        - Portfolio equity curve = running sum of all active position PnL.
        - Exit per-position: same rules as per-ticker AlphaCombined
          (stop, trailing stop, alpha reversal, max hold).

        Returns
        -------
        dict with keys: ticker ("AlphaCombined_Portfolio"), tickers (list),
                        strategy, trade_log, equity_curve, returns, summary
        """
        alpha_th  = float(params.get("alpha_threshold",   0.40))
        rev_th    = float(params.get("reversal_threshold", -0.50))
        stop_atr  = float(params.get("stop_loss_atr",     1.5))
        trail_atr = float(params.get("trailing_stop_atr", 2.0))
        max_hold  = int(params.get("max_holding_days",    10))

        # Precompute per-ticker series on their native indices
        close_map:    dict[str, pd.Series] = {}
        alpha_map:    dict[str, pd.Series] = {}
        atr_map:      dict[str, pd.Series] = {}
        blackout_map: dict[str, pd.Series] = {}

        for ticker, ohlcv in ohlcv_dict.items():
            if ohlcv is None or ohlcv.empty:
                continue
            close_s = ohlcv["Close"].astype(float)
            high_s  = ohlcv["High"].astype(float)
            low_s   = ohlcv["Low"].astype(float)
            atr_s   = self._atr(high_s, low_s, close_s)
            alpha_s = (
                ohlcv["alpha_signal"].astype(float)
                if "alpha_signal" in ohlcv.columns
                else pd.Series(0.0, index=ohlcv.index)
            )
            bl_s = (
                ohlcv["earnings_blackout"].astype(bool)
                if "earnings_blackout" in ohlcv.columns
                else pd.Series(False, index=ohlcv.index)
            )
            close_map[ticker]    = close_s
            alpha_map[ticker]    = alpha_s
            atr_map[ticker]      = atr_s
            blackout_map[ticker] = bl_s

        # Portfolio-level backtest uses the default slippage (no ADV override available
        # at the portfolio level — individual tickers have mixed liquidity).
        slip = self._default_slip_bps / 10_000

        if not close_map:
            empty = pd.Series(dtype=float)
            return {
                "ticker": "AlphaCombined_Portfolio", "tickers": [],
                "strategy": "AlphaCombined", "trade_log": [],
                "equity_curve": empty, "returns": empty,
                "summary": {}, "slippage_bps": round(slip * 10_000, 1),
            }

        # Build a common date index (union so no ticker data is silently dropped)
        all_indices = [s.index for s in close_map.values()]
        common_idx  = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.union(idx)
        common_idx = common_idx.sort_values()

        tickers = list(close_map.keys())
        warmup  = 30   # ATR + alpha warmup bars

        # Per-ticker open-position state
        positions: dict[str, dict] = {}

        equity      = self.initial_portfolio
        all_trades: list[dict]          = []
        equity_pts: list[tuple]         = []   # (date, equity)

        for i, date in enumerate(common_idx):
            if i < warmup:
                equity_pts.append((date, equity))
                continue

            for ticker in tickers:
                c_s  = close_map[ticker]
                a_s  = atr_map[ticker]
                al_s = alpha_map[ticker]
                bl_s = blackout_map[ticker]

                if date not in c_s.index:
                    continue

                c  = float(c_s.loc[date])
                a  = float(a_s.loc[date]) if date in a_s.index else float("nan")
                s  = float(al_s.loc[date]) if date in al_s.index else 0.0
                bl = bool(bl_s.loc[date])  if date in bl_s.index else False

                if np.isnan(c) or np.isnan(a) or a <= 0:
                    continue
                if np.isnan(s):
                    s = 0.0

                if ticker not in positions:
                    # Entry: cross-sectional signal exceeds threshold
                    if s > alpha_th and not bl:
                        entry_px = c * (1 + slip)
                        stop_px  = entry_px - stop_atr * a
                        pos_size = (equity * RISK_PER_TRADE) / (stop_atr * a)
                        positions[ticker] = {
                            "entry_price":  entry_px,
                            "stop_price":   stop_px,
                            "trail_stop":   stop_px,
                            "pos_size":     pos_size,
                            "peak":         c,
                            "entry_date":   date,
                            "holding_days": 0,
                            "target_1r":    entry_px + stop_atr * a,
                            "reached_1r":   False,
                        }
                else:
                    pos = positions[ticker]
                    pos["holding_days"] += 1
                    pos["peak"]          = max(pos["peak"], c)
                    pos["trail_stop"]    = max(
                        pos["peak"] - trail_atr * a, pos["stop_price"]
                    )
                    if c >= pos["target_1r"]:
                        pos["reached_1r"] = True

                    exit_reason: str | None = None
                    if c < pos["trail_stop"]:
                        exit_reason = (
                            "stop_loss"
                            if pos["trail_stop"] <= pos["stop_price"] + 1e-6
                            else "trailing_stop"
                        )
                    elif s < rev_th:
                        exit_reason = "alpha_reversal"
                    elif pos["holding_days"] >= max_hold:
                        exit_reason = "max_holding"

                    if exit_reason:
                        exit_px   = c * (1 - slip)
                        gross_pnl = (c - (pos["entry_price"] / (1 + slip))) * pos["pos_size"]
                        pnl       = (exit_px - pos["entry_price"]) * pos["pos_size"]
                        equity   += pnl
                        all_trades.append(_make_trade(
                            pos["entry_date"], pos["entry_price"],
                            date, exit_px,
                            pos["holding_days"], pos["pos_size"], pnl, exit_reason,
                            gross_pnl=gross_pnl,
                            slippage_cost=abs(gross_pnl - pnl),
                            reached_1r=pos["reached_1r"],
                        ))
                        del positions[ticker]

            equity_pts.append((date, equity))

        # Force-close any remaining open positions at last known price
        for ticker, pos in list(positions.items()):
            c_s = close_map[ticker]
            if len(c_s) == 0:
                continue
            c        = float(c_s.iloc[-1])
            exit_px  = c * (1 - slip)
            gross_pnl = (c - (pos["entry_price"] / (1 + slip))) * pos["pos_size"]
            pnl      = (exit_px - pos["entry_price"]) * pos["pos_size"]
            equity  += pnl
            all_trades.append(_make_trade(
                pos["entry_date"], pos["entry_price"],
                c_s.index[-1], exit_px,
                pos["holding_days"], pos["pos_size"], pnl, "end_of_backtest",
                gross_pnl=gross_pnl,
                slippage_cost=abs(gross_pnl - pnl),
                reached_1r=pos["reached_1r"],
            ))

        equity_curve = pd.Series(
            [v for _, v in equity_pts],
            index=pd.DatetimeIndex([d for d, _ in equity_pts]),
            name="equity",
        )

        # Daily returns: flat days earn DAILY_RF (idle cash)
        raw_returns  = equity_curve.pct_change().fillna(0.0)
        in_position  = pd.Series(False, index=equity_curve.index)
        for trade in all_trades:
            try:
                in_position.loc[trade["entry_date"]:trade["exit_date"]] = True
            except Exception:
                pass
        raw_returns[~in_position] = DAILY_RF

        summary = self._summarize(all_trades, equity_curve)

        return {
            "ticker":       "AlphaCombined_Portfolio",
            "tickers":      tickers,
            "strategy":     "AlphaCombined",
            "trade_log":    all_trades,
            "equity_curve": equity_curve,
            "returns":      raw_returns,
            "summary":      summary,
            "slippage_bps": round(slip * 10_000, 1),
        }

    # ── pair trading ─────────────────────────────────────────────────────────

    def run_pair(
        self,
        ticker_a:    str,
        ticker_b:    str,
        ohlcv_a:     pd.DataFrame,
        ohlcv_b:     pd.DataFrame,
        params:      dict,
        hedge_ratio: float = 1.0,
        adv_a:       float = 0.0,
        adv_b:       float = 0.0,
    ) -> dict:
        """
        Back-test Relative Value / Pair Trading on two cointegrated tickers.

        Strategy mechanics (simulated long/short, no real broker required)
        ------------------------------------------------------------------
        The spread is defined as:
            spread[t] = log(P_A[t]) - beta * log(P_B[t])

        where beta (hedge_ratio) is estimated via rolling OLS over beta_window days
        so it adapts to slow structural shifts without overfitting.

        The spread is then z-scored over the same z_window:
            z[t] = (spread[t] - mean(spread, z_window)) / std(spread, z_window)

        Entry signals:
            z > entry_z  → short spread: SHORT A (overvalued), LONG B (undervalued)
            z < -entry_z → long spread:  LONG A (undervalued), SHORT B (overvalued)

        Position sizing (dollar-neutral per leg):
            risk_per_trade = 1% of portfolio on a (stop_z - entry_z) spread move
            notional_a     = portfolio × 0.01 / (stop_z - entry_z)
            size_a (shares)= notional_a / close_a
            size_b (shares)= size_a × beta  (dollar neutrality via hedge ratio)

        Short-leg simulation:
            LONG  leg PnL  = (exit_px - entry_px) × size     (buy low, sell high)
            SHORT leg PnL  = (entry_px - exit_px) × size - borrow_cost
            borrow_cost    = entry_px × size × BORROW_RATE_DAILY × holding_days

        Slippage applied to all four executions (entry A, entry B, exit A, exit B).

        Exit conditions (priority order):
            1. Stop loss     : |z| > stop_z   (spread diverging — thesis failed)
            2. Target        : |z| < exit_z   (spread converged to mean)
            3. Max hold      : holding_days ≥ max_holding_days

        Returns
        -------
        dict with keys:
            ticker_a, ticker_b, strategy ("PairTrading"),
            trade_log, equity_curve, returns, summary,
            hedge_ratio, slippage_bps
        """
        # Compute local slip — never mutate self._slip
        if adv_a > 0 or adv_b > 0:
            avg_adv = (adv_a + adv_b) / max(int(adv_a > 0) + int(adv_b > 0), 1)
            slip    = _slip_bps_for_adv(avg_adv) / 10_000
        else:
            slip    = self._default_slip_bps / 10_000

        trade_log    = self._run_pair_trading(ohlcv_a, ohlcv_b, params, hedge_ratio, slip)
        equity_curve = self._build_pair_equity_curve(ohlcv_a, ohlcv_b, trade_log)
        returns      = self._build_pair_returns(equity_curve, trade_log)
        summary      = self._summarize_pair(trade_log, equity_curve)

        return {
            "ticker_a":     ticker_a,
            "ticker_b":     ticker_b,
            "strategy":     "PairTrading",
            "trade_log":    trade_log,
            "equity_curve": equity_curve,
            "returns":      returns,
            "summary":      summary,
            "hedge_ratio":  round(hedge_ratio, 4),
            "slippage_bps": round(slip * 10_000, 1),
        }

    def _run_pair_trading(
        self, ohlcv_a: pd.DataFrame, ohlcv_b: pd.DataFrame,
        params: dict, initial_hedge: float, slip: float = 0.0,
    ) -> list[dict]:
        """
        Core pair trading simulation engine.

        Uses a rolling hedge ratio (60-day OLS) so the spread remains
        stationary even as the fundamental relationship slowly drifts.
        The z-score is computed over the same window with shift(1) throughout
        to prevent look-ahead bias.
        """
        entry_z       = float(params.get("entry_z",           2.0))
        exit_z        = float(params.get("exit_z",            0.25))
        stop_z        = float(params.get("stop_z",            3.5))
        beta_window   = int(params.get("beta_window",         60))
        z_window      = int(params.get("z_window",            60))
        max_hold      = int(params.get("max_holding_days",    15))
        borrow_rate   = float(params.get("borrow_rate_annual", 0.005)) / 252

        # Align close series on common trading dates
        close_a = ohlcv_a["Close"].astype(float)
        close_b = ohlcv_b["Close"].astype(float)
        close_a, close_b = close_a.align(close_b, join="inner")

        if len(close_a) < beta_window + z_window + 5:
            return []

        log_a = np.log(close_a.values.astype(float))
        log_b = np.log(close_b.values.astype(float))

        # ── Rolling OLS hedge ratio ───────────────────────────────────────────
        # beta[t] estimated from (t - beta_window) to (t-1) — shift(1) via
        # indexing [i-beta_window : i] where i is the current bar.
        beta_vals = np.full(len(log_a), np.nan)
        for i in range(beta_window, len(log_a)):
            y_w = log_a[i - beta_window: i]
            x_w = log_b[i - beta_window: i]
            x_dm = x_w - x_w.mean()
            y_dm = y_w - y_w.mean()
            var_x = float((x_dm ** 2).sum())
            if var_x > 1e-12:
                beta_vals[i] = float((x_dm * y_dm).sum() / var_x)
            else:
                beta_vals[i] = initial_hedge

        beta_series = pd.Series(beta_vals, index=close_a.index)

        # ── Spread = log(P_A) - beta * log(P_B) ──────────────────────────────
        spread_vals = log_a - beta_vals * log_b
        spread      = pd.Series(spread_vals, index=close_a.index)

        # ── Rolling z-score of spread (shift(1) to prevent look-ahead) ───────
        # The rolling mean/std use windows ending at t-1, so z-score on bar t
        # is based purely on history before bar t.
        spread_mean = spread.rolling(z_window).mean().shift(1)
        spread_std  = spread.rolling(z_window).std(ddof=1).shift(1)
        z_raw       = (spread - spread_mean) / spread_std.replace(0, np.nan)

        start = beta_window + z_window + 1

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        direction    = ""    # "long_spread" or "short_spread"
        entry_date   = None
        holding_days = 0
        entry_ca = entry_cb = size_a = size_b = 0.0
        entry_z_val = 0.0

        for i in range(start, len(close_a)):
            date = close_a.index[i]
            ca   = float(close_a.iloc[i])
            cb   = float(close_b.iloc[i])
            z    = float(z_raw.iloc[i]) if not np.isnan(z_raw.iloc[i]) else np.nan
            beta = float(beta_series.iloc[i]) if not np.isnan(beta_series.iloc[i]) else initial_hedge

            if np.isnan(z):
                continue

            if not in_position:
                if abs(z) < entry_z:
                    continue

                # Dollar-neutral sizing: risk 1% portfolio on (stop_z - entry_z) spread move
                # notional_a = portfolio × 0.01 / (stop_z - entry_z)
                # This ensures a stop-out at stop_z costs at most 1% per trade.
                denom = max(stop_z - abs(z), 0.5)   # floor at 0.5z to avoid gigantic sizing
                notional_a = (equity * RISK_PER_TRADE) / denom
                size_a     = notional_a / (ca * (1 + slip))
                size_b     = size_a * abs(beta)

                in_position  = True
                holding_days = 0
                entry_date   = date
                entry_z_val  = z
                beta_at_entry = beta   # rolling hedge ratio used for this trade
                direction    = "short_spread" if z > 0 else "long_spread"

                if direction == "short_spread":
                    # Short A (entry: sell at ca with slip credit lost)
                    # Long  B (entry: buy  at cb with slip paid)
                    entry_ca = ca * (1 - slip)   # received for short
                    entry_cb = cb * (1 + slip)   # paid for long
                else:
                    # Long  A (entry: buy  at ca)
                    # Short B (entry: sell at cb)
                    entry_ca = ca * (1 + slip)
                    entry_cb = cb * (1 - slip)
            else:
                holding_days += 1

                exit_reason: str | None = None

                if direction == "short_spread":
                    if z < -stop_z:            # spread blew out in wrong direction
                        exit_reason = "stop_loss"
                    elif z > stop_z:           # also stop-out: spread diverged further
                        exit_reason = "stop_loss"
                    elif z < exit_z:           # converged toward mean — target reached
                        exit_reason = "z_cross"
                else:  # long_spread
                    if z > stop_z:
                        exit_reason = "stop_loss"
                    elif z < -stop_z:
                        exit_reason = "stop_loss"
                    elif z > -exit_z:
                        exit_reason = "z_cross"

                if exit_reason is None and holding_days >= max_hold:
                    exit_reason = "max_holding"

                if exit_reason:
                    if direction == "short_spread":
                        # Cover short A (buy back at ca), sell long B (at cb)
                        exit_ca = ca * (1 + slip)   # pay slip to cover
                        exit_cb = cb * (1 - slip)   # receive minus slip
                        pnl_a   = (entry_ca - exit_ca) * size_a   # short A: sold high, buy back
                        pnl_b   = (exit_cb  - entry_cb) * size_b  # long  B: buy low, sold high
                    else:
                        # Sell long A, cover short B
                        exit_ca = ca * (1 - slip)
                        exit_cb = cb * (1 + slip)
                        pnl_a   = (exit_ca  - entry_ca) * size_a  # long  A
                        pnl_b   = (entry_cb - exit_cb) * size_b   # short B: sold high, buy back

                    # Borrow cost: charged on the notional of the short leg
                    short_notional = (
                        entry_ca * size_a if direction == "short_spread"
                        else entry_cb * size_b
                    )
                    borrow_cost = short_notional * borrow_rate * holding_days

                    # Gross PnL: computed without slippage (using raw entry/exit prices
                    # before the slip adjustment was applied at execution time)
                    raw_entry_ca = ca if direction == "short_spread" else ca   # pre-slip entry
                    raw_entry_cb = cb if direction == "short_spread" else cb
                    if direction == "short_spread":
                        gross_pnl = (entry_ca / (1 - slip) - exit_ca / (1 + slip)) * size_a \
                                  + (exit_cb  / (1 - slip) - entry_cb / (1 + slip)) * size_b
                    else:
                        gross_pnl = (exit_ca  / (1 - slip) - entry_ca / (1 + slip)) * size_a \
                                  + (entry_cb / (1 - slip) - exit_cb  / (1 + slip)) * size_b
                    net_pnl   = pnl_a + pnl_b - borrow_cost
                    slip_cost = abs(gross_pnl - (pnl_a + pnl_b))
                    equity   += net_pnl

                    trades.append({
                        "entry_date":   entry_date,
                        "exit_date":    date,
                        "holding_days": int(holding_days),
                        "direction":    direction,
                        "entry_z":      round(float(entry_z_val), 4),
                        "exit_z":       round(float(z), 4),
                        "beta_at_entry": round(float(beta_at_entry), 4),
                        "entry_price_a": round(float(entry_ca), 4),
                        "entry_price_b": round(float(entry_cb), 4),
                        "exit_price_a":  round(float(exit_ca), 4),
                        "exit_price_b":  round(float(exit_cb), 4),
                        "size_a":        round(float(size_a), 4),
                        "size_b":        round(float(size_b), 4),
                        "pnl_a":         round(float(pnl_a), 2),
                        "pnl_b":         round(float(pnl_b), 2),
                        "borrow_cost":   round(float(borrow_cost), 2),
                        "pnl":           round(float(net_pnl), 2),
                        "gross_pnl":     round(float(gross_pnl), 2),
                        "slippage_cost": round(float(slip_cost), 2),
                        "exit_reason":   exit_reason,
                        "reached_1r":    abs(z) > stop_z - 0.5 if exit_reason == "z_cross" else False,
                    })
                    in_position = False

        return trades

    def _build_pair_equity_curve(
        self,
        ohlcv_a: pd.DataFrame,
        ohlcv_b: pd.DataFrame,
        trade_log: list[dict],
    ) -> pd.Series:
        """
        Daily mark-to-market equity for a pair trade.

        For an open trade, unrealised P&L on each day is:
            long leg:  (close_today - entry_price) × size
            short leg: (entry_price - close_today) × size  (inverted for short)
        Combined unrealised = pnl_long + pnl_short at today's close.
        """
        close_a = ohlcv_a["Close"].astype(float)
        close_b = ohlcv_b["Close"].astype(float)
        close_a, close_b = close_a.align(close_b, join="inner")

        equity = pd.Series(self.initial_portfolio, index=close_a.index, dtype=float)
        if not trade_log:
            return equity

        date_to_idx = {d: i for i, d in enumerate(close_a.index)}
        cash        = self.initial_portfolio

        for trade in sorted(trade_log, key=lambda t: t["entry_date"]):
            ei = date_to_idx.get(trade["entry_date"])
            xi = date_to_idx.get(trade["exit_date"])
            if ei is None or xi is None:
                cash += trade["pnl"]
                continue

            direction  = trade["direction"]
            ea, eb     = trade["entry_price_a"], trade["entry_price_b"]
            sa, sb     = trade["size_a"],         trade["size_b"]

            hold_ca = close_a.iloc[ei: xi + 1].values
            hold_cb = close_b.iloc[ei: xi + 1].values

            if direction == "short_spread":
                # Short A: profit when price drops; Long B: profit when price rises
                unrealised = (ea - hold_ca) * sa + (hold_cb - eb) * sb
            else:
                # Long A: profit when price rises; Short B: profit when price drops
                unrealised = (hold_ca - ea) * sa + (eb - hold_cb) * sb

            equity.iloc[ei: xi + 1] = cash + unrealised

            cash += trade["pnl"]
            equity.iloc[xi + 1:] = cash

        return equity

    def _build_pair_returns(
        self, equity_curve: pd.Series, trade_log: list[dict]
    ) -> pd.Series:
        """Daily returns; flat days earn DAILY_RF (idle cash earns T-bill rate)."""
        returns     = equity_curve.pct_change().fillna(0.0)
        in_position = pd.Series(False, index=equity_curve.index)
        for trade in trade_log:
            try:
                in_position.loc[trade["entry_date"]: trade["exit_date"]] = True
            except Exception:
                pass
        returns[~in_position] = DAILY_RF
        return returns

    @staticmethod
    def _summarize_pair(trade_log: list[dict], equity_curve: pd.Series) -> dict:
        """
        Extended summary for pair trades — includes leg-level breakdown and
        direction analysis on top of the standard per-strategy metrics.
        """
        if not trade_log:
            return {
                "total_return": 0.0, "trade_count": 0, "win_rate": 0.0,
                "total_slippage_cost": 0.0, "gross_return": 0.0,
                "entry_efficiency": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "payoff_ratio": 0.0, "exit_reason_breakdown": {},
                "avg_holding_days": 0.0,
                # Pair-specific
                "total_borrow_cost": 0.0,
                "direction_breakdown": {},
                "avg_entry_z": 0.0,
                "avg_exit_z":  0.0,
            }

        initial      = equity_curve.iloc[0]
        final        = equity_curve.iloc[-1]
        total_return = (final - initial) / initial if initial != 0 else 0.0

        wins   = [t for t in trade_log if t["pnl"] > 0]
        losses = [t for t in trade_log if t["pnl"] < 0]

        total_slip   = sum(t.get("slippage_cost", 0.0) for t in trade_log)
        total_borrow = sum(t.get("borrow_cost",   0.0) for t in trade_log)
        gross_total  = sum(t.get("gross_pnl",     t["pnl"]) for t in trade_log)
        gross_return = gross_total / initial if initial != 0 else 0.0

        avg_win  = float(np.mean([t["pnl"] for t in wins]))      if wins   else 0.0
        avg_loss = float(np.mean([abs(t["pnl"]) for t in losses])) if losses else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 1e-6 else 0.0

        exit_reasons: dict[str, int] = {}
        for t in trade_log:
            r = t.get("exit_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        direction_breakdown: dict[str, int] = {}
        for t in trade_log:
            d = t.get("direction", "unknown")
            direction_breakdown[d] = direction_breakdown.get(d, 0) + 1

        avg_hold    = float(np.mean([t.get("holding_days", 0) for t in trade_log]))
        avg_entry_z = float(np.mean([abs(t.get("entry_z", 0)) for t in trade_log]))
        avg_exit_z  = float(np.mean([abs(t.get("exit_z",  0)) for t in trade_log]))

        beta_vals = [t["beta_at_entry"] for t in trade_log if t.get("beta_at_entry") is not None]
        avg_beta  = float(np.mean(beta_vals)) if beta_vals else 0.0

        n = len(trade_log)
        return {
            "total_return":              round(float(total_return), 6),
            "gross_return":              round(float(gross_return), 6),
            "total_slippage_cost":       round(float(total_slip),   2),
            "total_borrow_cost":         round(float(total_borrow), 2),
            "avg_borrow_cost_per_trade": round(float(total_borrow) / n, 2),
            "trade_count":               n,
            "win_rate":                  round(len(wins) / n, 4),
            "entry_efficiency":          round(sum(1 for t in trade_log if t.get("reached_1r")) / n, 4),
            "avg_win":                   round(avg_win,       2),
            "avg_loss":                  round(avg_loss,       2),
            "payoff_ratio":              round(payoff_ratio,   3),
            "exit_reason_breakdown":     exit_reasons,
            "direction_breakdown":       direction_breakdown,
            "avg_holding_days":          round(avg_hold,       1),
            "avg_entry_z":               round(avg_entry_z,    3),
            "avg_exit_z":                round(avg_exit_z,     3),
            # Hedge ratio audit: avg_beta_at_entry should match the static hedge_ratio
            # from PairSelector.  Large deviations mean the relationship has drifted
            # and dollar-neutrality may have been temporarily compromised.
            "avg_beta_at_entry":         round(avg_beta,       4),
        }

    # ── pair signal status ────────────────────────────────────────────────────

    def _pair_trading_signal(
        self, ohlcv_a: pd.DataFrame, ohlcv_b: pd.DataFrame,
        params: dict, portfolio: float, hedge_ratio: float,
    ) -> dict:
        """
        Current-bar signal check for a pair.

        Computes the live z-score on the most recent bar and returns whether
        entry conditions are met, plus the full projected trade setup.
        """
        close_a = ohlcv_a["Close"].astype(float)
        close_b = ohlcv_b["Close"].astype(float)
        close_a, close_b = close_a.align(close_b, join="inner")

        if len(close_a) < 62:
            return {
                "signal_active": False,
                "details": "Insufficient history for pair z-score (<62 bars after alignment).",
                "setup": None, "projected_setup": None,
            }

        entry_z   = float(params.get("entry_z",        2.0))
        stop_z    = float(params.get("stop_z",         3.5))
        beta_w    = int(params.get("beta_window",      60))
        z_w       = int(params.get("z_window",         60))

        log_a     = np.log(close_a.values.astype(float))
        log_b     = np.log(close_b.values.astype(float))

        # Compute hedge ratio from last beta_window bars
        y_w, x_w  = log_a[-beta_w:], log_b[-beta_w:]
        x_dm      = x_w - x_w.mean()
        y_dm      = y_w - y_w.mean()
        var_x     = float((x_dm ** 2).sum())
        beta_live = float((x_dm * y_dm).sum() / var_x) if var_x > 1e-12 else hedge_ratio

        spread    = log_a - beta_live * log_b
        sp_series = pd.Series(spread, index=close_a.index)
        sp_mean   = float(sp_series.iloc[-z_w:-1].mean())
        sp_std    = float(sp_series.iloc[-z_w:-1].std(ddof=1))

        ca_last   = float(close_a.iloc[-1])
        cb_last   = float(close_b.iloc[-1])
        sp_last   = spread[-1]

        if sp_std < 1e-10:
            return {"signal_active": False, "details": "Spread std ≈ 0 — no signal.", "setup": None, "projected_setup": None}

        z_live    = (sp_last - sp_mean) / sp_std
        active    = abs(z_live) > entry_z
        direction = "short_spread" if z_live > 0 else "long_spread"

        setup = None
        if active:
            denom       = max(stop_z - abs(z_live), 0.5)
            notional_a  = (portfolio * RISK_PER_TRADE) / denom
            s_a         = notional_a / ca_last
            s_b         = s_a * abs(beta_live)
            setup = {
                "direction":     direction,
                "z_score":       round(z_live, 3),
                "entry_z":       entry_z,
                "stop_z":        stop_z,
                "size_a":        round(s_a, 2),
                "size_b":        round(s_b, 2),
                "notional_a":    round(s_a * ca_last, 2),
                "notional_b":    round(s_b * cb_last, 2),
                "dollar_risk":   round(portfolio * RISK_PER_TRADE, 2),
                "beta_live":     round(beta_live, 4),
                "spread_std":    round(sp_std, 6),
            }

        return {
            "signal_active":  active,
            "z_score":        round(float(z_live), 3),
            "entry_z":        entry_z,
            "direction":      direction if active else None,
            "close_a":        ca_last,
            "close_b":        cb_last,
            "beta_live":      round(beta_live, 4),
            "spread_std":     round(sp_std, 6),
            "setup":          setup,
            "projected_setup": setup,
            "details": (
                f"Pair z={z_live:.3f} {'>' if z_live > 0 else '<'} "
                f"±{entry_z:.1f} → {direction.replace('_', ' ').upper() if active else 'NO SIGNAL'}"
                f" | beta={beta_live:.3f}  spread_std={sp_std:.5f}"
            ),
        }

    # ── strategy engines ──────────────────────────────────────────────────────

    def _run_momentum(self, ohlcv: pd.DataFrame, params: dict, slip: float = 0.0) -> list[dict]:
        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)
        volume = ohlcv["Volume"].astype(float)

        entry_lookback    = params["entry_lookback"]
        vol_multiplier    = params["volume_multiplier"]
        trailing_stop_atr = params["trailing_stop_atr"]
        ma_period         = params["ma_exit_period"]
        stop_loss_atr     = params["stop_loss_atr"]
        max_holding       = params["max_holding_days"]
        # momentum_gate_active=False disables the 12-1 month filter (Crisis/High-Vol
        # markets: the gate is calibrated for trending markets and blocks all entries
        # after a crash, where the exact tickers that need evaluation have recently fallen).
        gate_active   = bool(params.get("momentum_gate_active", True))
        mom_lookback  = int(params.get("momentum_lookback", 0)) if gate_active else 0

        atr           = self._atr(high, low, close)
        ma            = close.rolling(ma_period).mean().shift(1)   # shift(1): no look-ahead
        vol_ma        = volume.rolling(20).mean().shift(1)
        rolling_high  = close.rolling(entry_lookback).max().shift(1)
        blackout      = ohlcv["earnings_blackout"] if "earnings_blackout" in ohlcv.columns else pd.Series(False, index=ohlcv.index)

        # 12-1 month momentum gate: return from [t-252] to [t-21], fully non-look-ahead.
        # Skipping the most recent month avoids short-term reversal contamination.
        # Only computed when momentum_lookback > 0 AND gate_active (Momentum strategy only).
        if mom_lookback > 0:
            mom_filter = (close.shift(22) / close.shift(mom_lookback + 1) - 1)
        else:
            mom_filter = None   # gate disabled — all tickers pass the momentum filter

        start = max(entry_lookback, 20, ma_period, ATR_PERIOD,
                    mom_lookback + 2 if mom_lookback > 0 else 0)

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        entry_price  = stop_price = pos_size = peak = 0.0
        entry_date   = None
        holding_days = 0

        for i in range(start, len(ohlcv)):
            c  = float(close.iloc[i])
            v  = float(volume.iloc[i])
            a  = float(atr.iloc[i])
            rh = float(rolling_high.iloc[i]) if not np.isnan(rolling_high.iloc[i]) else np.inf
            vm = float(vol_ma.iloc[i])        if not np.isnan(vol_ma.iloc[i])        else 0.0
            m  = float(ma.iloc[i])            if not np.isnan(ma.iloc[i])            else 0.0

            if np.isnan(a) or a <= 0:
                continue

            if not in_position:
                mom_ok = True
                if mom_filter is not None:
                    mv = float(mom_filter.iloc[i])
                    mom_ok = not np.isnan(mv) and mv > 0.0
                if c > rh and v > vol_multiplier * vm and mom_ok and not bool(blackout.iloc[i]):
                    in_position  = True
                    entry_price  = c * (1 + slip)   # pay spread on entry
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_loss_atr * a
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_loss_atr * a)
                    peak         = c
                    holding_days = 0
                    target_1r    = entry_price + stop_loss_atr * a  # 1R above entry
                    reached_1r   = False
            else:
                holding_days += 1
                peak          = max(peak, c)
                # Trailing stop ratchets up with peak but never below the hard stop floor.
                # At entry peak==entry so trailing_stop starts at stop_price level.
                # Once price rises, trailing_stop lifts above stop_price and locks in profit.
                trailing_stop = max(peak - trailing_stop_atr * a, stop_price)
                h_bar         = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < trailing_stop:
                    # If trailing_stop == stop_price → hard stop fired; otherwise profit trailing stop
                    exit_reason = "stop_loss" if trailing_stop <= stop_price + 1e-6 else "trailing_stop"
                elif c < m:
                    exit_reason = "ma_exit"
                elif holding_days >= max_holding:
                    exit_reason = "max_holding"

                if exit_reason:
                    exit_price      = c * (1 - slip)   # lose spread on exit
                    gross_pnl       = (c - (entry_price / (1 + slip))) * pos_size
                    pnl             = (exit_price - entry_price) * pos_size
                    equity         += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], exit_price,
                        holding_days, pos_size, pnl, exit_reason,
                        gross_pnl=gross_pnl,
                        slippage_cost=abs(gross_pnl - pnl),
                        reached_1r=reached_1r,
                    ))
                    in_position = False

        return trades

    def _run_mean_reversion(self, ohlcv: pd.DataFrame, params: dict, slip: float = 0.0) -> list[dict]:
        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)

        rsi_entry   = params["rsi_entry_threshold"]
        rsi_exit_th = params["rsi_exit_threshold"]
        bb_period   = params["bb_period"]
        bb_std_mult = params["bb_std"]
        stop_atr    = params["stop_loss_atr"]
        max_holding = params["max_holding_days"]

        atr      = self._atr(high, low, close)
        rsi_ser  = self._rsi(close)
        bb_ma    = close.rolling(bb_period).mean().shift(1)        # shift(1): no look-ahead
        bb_std   = close.rolling(bb_period).std(ddof=1).shift(1)   # shift(1): no look-ahead
        lower_bb = bb_ma - bb_std_mult * bb_std
        middle_bb = bb_ma
        blackout  = ohlcv["earnings_blackout"] if "earnings_blackout" in ohlcv.columns else pd.Series(False, index=ohlcv.index)

        start = max(bb_period, RSI_PERIOD, ATR_PERIOD)

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        entry_price  = stop_price = pos_size = 0.0
        entry_date   = None
        holding_days = 0

        for i in range(start, len(ohlcv)):
            c  = float(close.iloc[i])
            a  = float(atr.iloc[i])
            r  = float(rsi_ser.iloc[i]) if not np.isnan(rsi_ser.iloc[i]) else 50.0
            lb = float(lower_bb.iloc[i]) if not np.isnan(lower_bb.iloc[i]) else -np.inf
            mb = float(middle_bb.iloc[i]) if not np.isnan(middle_bb.iloc[i]) else np.inf

            if np.isnan(a) or a <= 0:
                continue

            if not in_position:
                if r < rsi_entry and c <= lb and not bool(blackout.iloc[i]):
                    in_position  = True
                    entry_price  = c * (1 + slip)   # pay spread on entry
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_atr * a
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_atr * a)
                    holding_days = 0
                    target_1r    = entry_price + stop_atr * a  # 1R above entry
                    reached_1r   = False
            else:
                holding_days += 1
                h_bar = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < stop_price:
                    exit_reason = "stop_loss"
                elif r > rsi_exit_th:
                    exit_reason = "rsi_exit"
                elif c >= mb:
                    exit_reason = "ma_exit"
                elif holding_days >= max_holding:
                    exit_reason = "max_holding"

                if exit_reason:
                    exit_price    = c * (1 - slip)   # lose spread on exit
                    gross_pnl     = (c - (entry_price / (1 + slip))) * pos_size
                    pnl           = (exit_price - entry_price) * pos_size
                    equity       += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], exit_price,
                        holding_days, pos_size, pnl, exit_reason,
                        gross_pnl=gross_pnl,
                        slippage_cost=abs(gross_pnl - pnl),
                        reached_1r=reached_1r,
                    ))
                    in_position = False

        return trades

    def _run_alpha_combined(self, ohlcv: pd.DataFrame, params: dict, slip: float = 0.0) -> list[dict]:
        """
        AlphaCombined strategy engine.

        Alpha source: pre-computed cross-sectional multi-factor signal injected
        by AlphaEngine as the ``alpha_signal`` column.  The signal combines:
          - Cross-sectional 5-day mean reversion (40%)
          - Market-neutral idiosyncratic residual reversion (30%)
          - Volume-spike exhaustion fade (20%)
          - 2-day short-term momentum (10%)

        Entry:  alpha_signal > alpha_threshold  AND  not in earnings blackout
        Exit (priority order):
          1. Hard stop  : close < entry - stop_loss_atr × ATR
          2. Trailing   : close < peak  - trailing_stop_atr × ATR
          3. Reversal   : alpha_signal < reversal_threshold (signal flips negative)
          4. Max hold   : holding_days ≥ max_holding_days

        Trade frequency is much higher than RSI+BB because the cross-sectional
        z-score threshold fires on the bottom ~30-40% of the universe each day.
        """
        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)

        alpha_th  = float(params.get("alpha_threshold",   0.40))
        rev_th    = float(params.get("reversal_threshold", -0.50))
        stop_atr  = float(params.get("stop_loss_atr",     1.5))
        trail_atr = float(params.get("trailing_stop_atr", 2.0))
        max_hold  = int(params.get("max_holding_days",    10))

        atr = self._atr(high, low, close)

        # alpha_signal is pre-computed and already shift(1)-lagged by AlphaEngine
        if "alpha_signal" in ohlcv.columns:
            alpha_sig = ohlcv["alpha_signal"].astype(float)
        else:
            alpha_sig = pd.Series(0.0, index=ohlcv.index)

        blackout = (
            ohlcv["earnings_blackout"]
            if "earnings_blackout" in ohlcv.columns
            else pd.Series(False, index=ohlcv.index)
        )

        start = max(ATR_PERIOD + 5, 25)

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        entry_price  = stop_price = trail_stop = pos_size = peak = 0.0
        entry_date   = None
        holding_days = 0
        target_1r    = 0.0
        reached_1r   = False

        for i in range(start, len(ohlcv)):
            c = float(close.iloc[i])
            a = float(atr.iloc[i])
            s = float(alpha_sig.iloc[i]) if not np.isnan(alpha_sig.iloc[i]) else 0.0

            if np.isnan(a) or a <= 0:
                continue

            if not in_position:
                if s > alpha_th and not bool(blackout.iloc[i]):
                    in_position  = True
                    entry_price  = c * (1 + slip)
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_atr * a
                    trail_stop   = stop_price
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_atr * a)
                    peak         = c
                    holding_days = 0
                    target_1r    = entry_price + stop_atr * a
                    reached_1r   = False
            else:
                holding_days += 1
                peak          = max(peak, c)
                trail_stop    = max(peak - trail_atr * a, stop_price)
                h_bar         = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < trail_stop:
                    exit_reason = (
                        "stop_loss" if trail_stop <= stop_price + 1e-6
                        else "trailing_stop"
                    )
                elif s < rev_th:
                    exit_reason = "alpha_reversal"
                elif holding_days >= max_hold:
                    exit_reason = "max_holding"

                if exit_reason:
                    exit_price  = c * (1 - slip)
                    gross_pnl   = (c - (entry_price / (1 + slip))) * pos_size
                    pnl         = (exit_price - entry_price) * pos_size
                    equity     += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], exit_price,
                        holding_days, pos_size, pnl, exit_reason,
                        gross_pnl=gross_pnl,
                        slippage_cost=abs(gross_pnl - pnl),
                        reached_1r=reached_1r,
                    ))
                    in_position = False

        return trades

    def _run_ml_signal(self, ohlcv: pd.DataFrame, params: dict, slip: float = 0.0) -> list[dict]:
        """
        MLSignal strategy engine.

        Alpha source: per-ticker gradient-boosting classifier probability
        injected by MLSignalEngine as the ``ml_signal`` column.  The signal
        represents P(5-day forward return > 0) and is fully out-of-sample
        (walk-forward trained) with all features shift(1)-lagged.

        Entry:  ml_signal > ml_threshold  AND  not in earnings blackout
        Exit (priority order):
          1. Hard stop    : close < entry - stop_loss_atr × ATR
          2. Trailing     : close < peak  - trailing_stop_atr × ATR
          3. ML reversal  : ml_signal < reversal_threshold (model loses conviction)
          4. Max hold     : holding_days ≥ max_holding_days

        Falls back gracefully when ``ml_signal`` column is absent (returns []).
        """
        close = ohlcv["Close"].astype(float)
        high  = ohlcv["High"].astype(float)
        low   = ohlcv["Low"].astype(float)

        ml_th     = float(params.get("ml_threshold",      0.60))
        rev_th    = float(params.get("reversal_threshold", 0.40))
        stop_atr  = float(params.get("stop_loss_atr",      1.5))
        trail_atr = float(params.get("trailing_stop_atr",  2.0))
        max_hold  = int(params.get("max_holding_days",     10))

        atr = self._atr(high, low, close)

        # ml_signal is pre-computed and shift(1)-lagged by MLSignalEngine
        if "ml_signal" not in ohlcv.columns:
            return []  # no signal available — pipeline fallback should have caught this
        ml_sig = ohlcv["ml_signal"].astype(float)

        blackout = (
            ohlcv["earnings_blackout"]
            if "earnings_blackout" in ohlcv.columns
            else pd.Series(False, index=ohlcv.index)
        )

        start = max(ATR_PERIOD + 5, 25)

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        entry_price  = stop_price = trail_stop = pos_size = peak = 0.0
        entry_date   = None
        holding_days = 0
        target_1r    = 0.0
        reached_1r   = False

        for i in range(start, len(ohlcv)):
            c = float(close.iloc[i])
            a = float(atr.iloc[i])
            s = float(ml_sig.iloc[i]) if not np.isnan(ml_sig.iloc[i]) else np.nan

            if np.isnan(a) or a <= 0:
                continue

            if not in_position:
                # Skip bars where ml_signal is not yet available (early history)
                if np.isnan(s):
                    continue
                if s > ml_th and not bool(blackout.iloc[i]):
                    in_position  = True
                    entry_price  = c * (1 + slip)
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_atr * a
                    trail_stop   = stop_price
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_atr * a)
                    peak         = c
                    holding_days = 0
                    target_1r    = entry_price + stop_atr * a
                    reached_1r   = False
            else:
                holding_days += 1
                peak          = max(peak, c)
                trail_stop    = max(peak - trail_atr * a, stop_price)
                h_bar         = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < trail_stop:
                    exit_reason = (
                        "stop_loss" if trail_stop <= stop_price + 1e-6
                        else "trailing_stop"
                    )
                elif not np.isnan(s) and s < rev_th:
                    exit_reason = "ml_reversal"
                elif holding_days >= max_hold:
                    exit_reason = "max_holding"

                if exit_reason:
                    exit_price = c * (1 - slip)
                    gross_pnl  = (c - (entry_price / (1 + slip))) * pos_size
                    pnl        = (exit_price - entry_price) * pos_size
                    equity    += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], exit_price,
                        holding_days, pos_size, pnl, exit_reason,
                        gross_pnl=gross_pnl,
                        slippage_cost=abs(gross_pnl - pnl),
                        reached_1r=reached_1r,
                    ))
                    in_position = False

        return trades

    def _run_volatility_breakout(self, ohlcv: pd.DataFrame, params: dict, slip: float = 0.0) -> list[dict]:
        """
        VolatilityBreakout strategy engine.

        Alpha source: Bollinger Band compression (squeeze) followed by
        directional breakout is a well-documented precursor to large moves.
        Volatility compresses → bands narrow → price breaks above the upper band
        on elevated volume → the move is underway.

        Entry conditions (ALL required):
          1. BB width was in the bottom squeeze_pct percentile within the last
             squeeze_lookback bars — confirms prior compression phase.
          2. Close > upper Bollinger Band — breakout direction = long.
          3. Volume > volume_mult × 20-bar average — confirms institutional participation.
          4. Not inside earnings blackout window.

        Exit (priority order):
          1. Trailing stop: close < peak − trailing_stop_atr × ATR
          2. Hard stop:     close < entry − stop_loss_atr × ATR  (floor for trailing stop)
          3. Max holding days
        """
        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)
        volume = ohlcv["Volume"].astype(float)

        bb_period         = params["bb_period"]
        squeeze_pct       = params["squeeze_pct"]
        squeeze_lookback  = params.get("squeeze_lookback", 5)
        volume_mult       = params.get("volume_mult", 1.5)
        stop_loss_atr     = params["stop_loss_atr"]
        trailing_stop_atr = params["trailing_stop_atr"]
        max_holding       = params["max_holding_days"]

        atr      = self._atr(high, low, close)
        bb_ma    = close.rolling(bb_period).mean().shift(1)
        bb_std_s = close.rolling(bb_period).std(ddof=1).shift(1)
        upper_bb = bb_ma + 2.0 * bb_std_s

        # Normalised BB width: 4σ/mid (proportional to %BB width) — shift(1) no look-ahead
        bb_width = (4.0 * bb_std_s) / bb_ma.replace(0, np.nan)
        # squeeze_threshold: rolling bb_period-bar quantile of BB width — shift(1) no look-ahead
        squeeze_threshold = bb_width.rolling(bb_period).quantile(squeeze_pct).shift(1)
        # Was there a squeeze in the last squeeze_lookback bars?
        # shift(1) on the squeeze flag prevents same-bar look-ahead
        squeezed_flag = (bb_width.shift(1) <= squeeze_threshold)
        was_squeezed_recently = squeezed_flag.rolling(squeeze_lookback).max().fillna(0).astype(bool)

        vol_ma   = volume.rolling(20).mean().shift(1)
        blackout = ohlcv["earnings_blackout"] if "earnings_blackout" in ohlcv.columns else pd.Series(False, index=ohlcv.index)

        # Need bb_period + 20 bars (for percentile) + ATR warmup + lookback
        start = max(bb_period + squeeze_lookback + 5, ATR_PERIOD + 5, 30)

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        entry_price  = stop_price = pos_size = peak = 0.0
        entry_date   = None
        holding_days = 0

        for i in range(start, len(ohlcv)):
            c   = float(close.iloc[i])
            v   = float(volume.iloc[i])
            a   = float(atr.iloc[i])
            ub  = float(upper_bb.iloc[i])  if not np.isnan(upper_bb.iloc[i])  else np.inf
            vm  = float(vol_ma.iloc[i])    if not np.isnan(vol_ma.iloc[i])    else 0.0
            was_sq = bool(was_squeezed_recently.iloc[i])

            if np.isnan(a) or a <= 0 or ub == np.inf:
                continue

            if not in_position:
                vol_confirmed = v > volume_mult * vm and vm > 0
                bb_breakout   = c > ub

                if was_sq and bb_breakout and vol_confirmed and not bool(blackout.iloc[i]):
                    in_position   = True
                    entry_price   = c * (1 + slip)
                    entry_date    = close.index[i]
                    stop_price    = entry_price - stop_loss_atr * a
                    pos_size      = (equity * RISK_PER_TRADE) / (stop_loss_atr * a)
                    peak          = c
                    holding_days  = 0
                    target_1r     = entry_price + stop_loss_atr * a
                    reached_1r    = False
            else:
                holding_days += 1
                peak          = max(peak, c)
                trailing_stop = max(peak - trailing_stop_atr * a, stop_price)
                h_bar         = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < trailing_stop:
                    exit_reason = "stop_loss" if trailing_stop <= stop_price + 1e-6 else "trailing_stop"
                elif holding_days >= max_holding:
                    exit_reason = "max_holding"

                if exit_reason:
                    exit_price  = c * (1 - slip)
                    gross_pnl   = (c - (entry_price / (1 + slip))) * pos_size
                    pnl         = (exit_price - entry_price) * pos_size
                    equity     += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], exit_price,
                        holding_days, pos_size, pnl, exit_reason,
                        gross_pnl=gross_pnl,
                        slippage_cost=abs(gross_pnl - pnl),
                        reached_1r=reached_1r,
                    ))
                    in_position = False

        return trades

    def _run_event_driven(self, ohlcv: pd.DataFrame, params: dict, slip: float = 0.0) -> list[dict]:
        """
        EventDriven (PEAD) strategy engine.

        Alpha source: Post-Earnings Announcement Drift (PEAD).
        After a positive earnings surprise the market systematically under-reacts
        (Bernard & Thomas 1989, 1990): prices drift higher for 5–60 days in
        proportion to the gap magnitude.  This strategy enters AFTER the earnings
        blackout window lifts and rides the drift, exiting when the signal fades.

        Entry conditions (ALL required):
          1. A positive earnings gap > gap_threshold occurred within the last
             entry_window_bars bars — detected via rolling max of earnings_gap.
             Default 10 bars ensures entry is possible after the ±3-bar blackout.
          2. pead_signal > pead_min_signal — PEAD drift signal is still active.
             Filled forward for up to 60 days by OHLCVFetcher; scale is −1 to +1
             where +1 = maximum drift (gap ≥ 10%).
          3. Close > ma_filter_period-day MA — confirms upward drift continuation
             and filters out gap-and-trap patterns that quickly reverse.
          4. Volume > volume_mult × 20-bar average — elevated post-announcement
             institutional participation confirms continued market attention.
          5. NOT inside earnings blackout window — we trade the drift, not the
             announcement itself (avoids straddling the uncertainty window).

        Exit (priority order):
          1. Hard stop  : close < entry − stop_loss_atr × ATR_at_entry
          2. Trailing   : close < peak  − trailing_stop_atr × ATR
          3. PEAD fade  : pead_signal < pead_exit_threshold (drift has reversed;
                          default −0.10 gives a small buffer before exiting)
          4. Max hold   : holding_days ≥ max_holding_days (7 default; 10 for large gaps)

        Fall-back: returns [] if pead_signal or earnings_gap columns are absent
        (OHLCVFetcher.add_earnings_drift_features() was not called).
        """
        if "pead_signal" not in ohlcv.columns or "earnings_gap" not in ohlcv.columns:
            return []   # PEAD features absent — pipeline should have added them

        # If no earnings events exist at all for this ticker, PEAD has no signal to trade.
        # fillna(0.0) would make pead_signal always 0, which is below pead_min_signal (0.20),
        # so no entries would ever fire — but we return [] explicitly for clarity.
        if ohlcv["earnings_gap"].notna().sum() == 0:
            return []

        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)
        volume = ohlcv["Volume"].astype(float)

        gap_threshold  = float(params.get("gap_threshold",        0.02))
        pead_min_sig   = float(params.get("pead_min_signal",       0.20))
        pead_exit_th   = float(params.get("pead_exit_threshold",  -0.10))
        entry_window   = int(params.get("entry_window_bars",       10))
        vol_mult       = float(params.get("volume_mult",           1.3))
        ma_period      = int(params.get("ma_filter_period",        5))
        stop_atr       = float(params.get("stop_loss_atr",         1.5))
        trail_atr      = float(params.get("trailing_stop_atr",     2.0))
        max_hold       = int(params.get("max_holding_days",        7))

        atr      = self._atr(high, low, close)
        # NaN pead_signal means no active PEAD drift — keep as NaN so the pead_ok
        # check (ps > pead_min_sig) correctly rejects bars with no drift signal.
        pead_sig = ohlcv["pead_signal"].astype(float)
        earn_gap = ohlcv["earnings_gap"].fillna(0.0).astype(float)

        # Short MA to confirm upward drift continuation after the gap.
        # shift(1) prevents same-bar look-ahead.
        ma_short = close.rolling(ma_period).mean().shift(1)
        vol_ma   = volume.rolling(20).mean().shift(1)

        blackout = (
            ohlcv["earnings_blackout"]
            if "earnings_blackout" in ohlcv.columns
            else pd.Series(False, index=ohlcv.index)
        )

        # Recent positive gap flag: True if a gap > gap_threshold occurred within
        # the last entry_window bars.  shift(1) prevents same-bar look-ahead.
        # With a ±3-bar blackout and entry_window=10, the strategy can enter on
        # bars t+4 through t+13 after an earnings event at bar t.
        recent_gap_flag = (
            earn_gap.gt(gap_threshold)
            .rolling(entry_window)
            .max()
            .shift(1)
            .fillna(0)
            .astype(bool)
        )

        start = max(ATR_PERIOD + 5, ma_period + 5, 25)

        trades       = []
        in_position  = False
        equity       = self.initial_portfolio
        entry_price  = stop_price = trail_stop = pos_size = peak = 0.0
        entry_date   = None
        holding_days = 0
        target_1r    = 0.0
        reached_1r   = False

        for i in range(start, len(ohlcv)):
            c  = float(close.iloc[i])
            a  = float(atr.iloc[i])
            ps_raw = pead_sig.iloc[i]
            ps = float(ps_raw) if not np.isnan(ps_raw) else float("nan")
            v  = float(volume.iloc[i])
            vm = float(vol_ma.iloc[i])   if not np.isnan(vol_ma.iloc[i])   else 0.0
            m5 = float(ma_short.iloc[i]) if not np.isnan(ma_short.iloc[i]) else 0.0

            if np.isnan(a) or a <= 0:
                continue

            if not in_position:
                gap_ok     = bool(recent_gap_flag.iloc[i])
                pead_ok    = (not np.isnan(ps)) and ps > pead_min_sig
                ma_ok      = c > m5 and m5 > 0
                vol_ok     = v > vol_mult * vm and vm > 0
                no_blackout = not bool(blackout.iloc[i])

                if gap_ok and pead_ok and ma_ok and vol_ok and no_blackout:
                    in_position  = True
                    entry_price  = c * (1 + slip)
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_atr * a
                    trail_stop   = stop_price
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_atr * a)
                    peak         = c
                    holding_days = 0
                    target_1r    = entry_price + stop_atr * a
                    reached_1r   = False
                    # Capture PEAD context at entry for per-trade diagnostics
                    _pead_at_entry = ps
                    _gap_at_entry  = float(earn_gap.iloc[max(0, i - entry_window): i].max())
            else:
                holding_days += 1
                peak          = max(peak, c)
                trail_stop    = max(peak - trail_atr * a, stop_price)
                h_bar         = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < trail_stop:
                    exit_reason = (
                        "stop_loss" if trail_stop <= stop_price + 1e-6
                        else "trailing_stop"
                    )
                elif (not np.isnan(ps)) and ps < pead_exit_th:
                    exit_reason = "pead_fade"
                elif holding_days >= max_hold:
                    exit_reason = "max_holding"

                if exit_reason:
                    exit_price  = c * (1 - slip)
                    gross_pnl   = (c - (entry_price / (1 + slip))) * pos_size
                    pnl         = (exit_price - entry_price) * pos_size
                    equity     += pnl
                    trade = _make_trade(
                        entry_date, entry_price, close.index[i], exit_price,
                        holding_days, pos_size, pnl, exit_reason,
                        gross_pnl=gross_pnl,
                        slippage_cost=abs(gross_pnl - pnl),
                        reached_1r=reached_1r,
                    )
                    # PEAD-specific context: required to audit whether signal strength
                    # at entry predicts outcome (core thesis of the strategy).
                    trade["pead_signal_at_entry"] = round(float(_pead_at_entry), 4)
                    trade["earnings_gap_size"]    = round(float(_gap_at_entry), 4)
                    trade["pead_signal_at_exit"]  = round(float(ps), 4) if not np.isnan(ps) else None
                    trades.append(trade)
                    in_position = False

        return trades

    # ── current signal ────────────────────────────────────────────────────────

    def signal_status(
        self,
        strategy_type:     str,
        ohlcv:             pd.DataFrame,
        params:            dict,
        initial_portfolio: float = 100_000.0,
    ) -> dict:
        """
        Check whether the entry signal is active on the most recent bar and,
        if so, compute the full trade setup (entry, stop, size, risk).

        Returns
        -------
        dict with keys: signal_active (bool), details (str), setup (dict or None)
        """
        try:
            if strategy_type == "Momentum":
                return self._momentum_signal(ohlcv, params, initial_portfolio)
            if strategy_type == "VolatilityBreakout":
                return self._volatility_breakout_signal(ohlcv, params, initial_portfolio)
            if strategy_type == "AlphaCombined":
                return self._alpha_combined_signal(ohlcv, params, initial_portfolio)
            if strategy_type == "MLSignal":
                return self._ml_signal_signal(ohlcv, params, initial_portfolio)
            if strategy_type == "EventDriven":
                return self._event_driven_signal(ohlcv, params, initial_portfolio)
            return self._mean_rev_signal(ohlcv, params, initial_portfolio)
        except Exception as e:
            return {"signal_active": None, "details": f"Signal check failed: {e}", "setup": None}

    def _momentum_signal(
        self, ohlcv: pd.DataFrame, params: dict, portfolio: float
    ) -> dict:
        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)
        volume = ohlcv["Volume"].astype(float)

        entry_lookback = params["entry_lookback"]
        vol_multiplier = params["volume_multiplier"]
        stop_loss_atr  = params["stop_loss_atr"]
        ma_period      = params["ma_exit_period"]

        atr          = self._atr(high, low, close)
        rolling_high = close.rolling(entry_lookback).max().shift(1)
        vol_ma       = volume.rolling(20).mean().shift(1)
        ma           = close.rolling(ma_period).mean()

        c   = float(close.iloc[-1])
        a   = float(atr.iloc[-1])   if not pd.isna(atr.iloc[-1])          else 0.0
        rh  = float(rolling_high.iloc[-1]) if not pd.isna(rolling_high.iloc[-1]) else float("inf")
        v   = float(volume.iloc[-1])
        vm  = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1])       else 0.0
        m   = float(ma.iloc[-1])     if not pd.isna(ma.iloc[-1])           else 0.0

        breakout  = c > rh
        vol_ok    = v > vol_multiplier * vm
        active    = breakout and vol_ok

        setup = None
        if active and a > 0:
            stop_dist   = stop_loss_atr * a
            pos_size    = int((portfolio * RISK_PER_TRADE) / stop_dist)
            stop_price  = c - stop_dist
            dollar_risk = portfolio * RISK_PER_TRADE
            setup = {
                "entry_price":  c,
                "stop_price":   stop_price,
                "stop_dist":    stop_dist,
                "position_size": pos_size,
                "dollar_risk":  dollar_risk,
                "current_atr":  a,
                "current_ma":   m,
                "target":       None,   # momentum has no fixed target
            }

        # Projected setup: always compute trade parameters based on current bar
        # so the execution brief can show the trader what the trade will look like
        # when (not just if) the signal fires.
        projected_setup = None
        if a > 0:
            stop_loss_atr = params["stop_loss_atr"]
            proj_entry    = rh * (1 + self._slip) if rh < np.inf else c * (1 + self._slip)
            stop_dist     = stop_loss_atr * a
            proj_stop     = proj_entry - stop_dist
            proj_size     = int((portfolio * RISK_PER_TRADE) / stop_dist)
            projected_setup = {
                "entry_price":   proj_entry,
                "stop_price":    proj_stop,
                "stop_dist":     stop_dist,
                "position_size": proj_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "current_ma":    m,
                "target":        None,
                "entry_trigger": rh,        # price that must be broken
                "volume_needed": vol_multiplier * vm,  # volume threshold to confirm
            }

        return {
            "signal_active":    active,
            "close":            c,
            "rolling_high":     rh,
            "volume":           v,
            "vol_threshold":    vol_multiplier * vm,
            "breakout":         breakout,
            "volume_confirmed": vol_ok,
            "setup":            setup,
            "projected_setup":  projected_setup,
            "details": (
                f"Close {c:.2f} {'>' if breakout else '<='} {entry_lookback}d high {rh:.2f}"
                f" | Volume {v:,.0f} {'>' if vol_ok else '<='} "
                f"{vol_multiplier}× avg {vol_multiplier * vm:,.0f}"
            ),
        }

    def _mean_rev_signal(
        self, ohlcv: pd.DataFrame, params: dict, portfolio: float
    ) -> dict:
        close = ohlcv["Close"].astype(float)
        high  = ohlcv["High"].astype(float)
        low   = ohlcv["Low"].astype(float)

        rsi_entry   = params["rsi_entry_threshold"]
        bb_period   = params["bb_period"]
        bb_std_mult = params["bb_std"]
        stop_atr    = params["stop_loss_atr"]

        atr      = self._atr(high, low, close)
        rsi_ser  = self._rsi(close)
        bb_ma    = close.rolling(bb_period).mean()
        bb_std   = close.rolling(bb_period).std(ddof=1)
        lower_bb = bb_ma - bb_std_mult * bb_std

        c   = float(close.iloc[-1])
        a   = float(atr.iloc[-1])     if not pd.isna(atr.iloc[-1])     else 0.0
        r   = float(rsi_ser.iloc[-1]) if not pd.isna(rsi_ser.iloc[-1]) else 50.0
        lb  = float(lower_bb.iloc[-1]) if not pd.isna(lower_bb.iloc[-1]) else -float("inf")
        mid = float(bb_ma.iloc[-1])    if not pd.isna(bb_ma.iloc[-1])    else float("inf")

        oversold = r < rsi_entry
        below_bb = c <= lb
        active   = oversold and below_bb

        setup = None
        if active and a > 0:
            stop_dist   = stop_atr * a
            pos_size    = int((portfolio * RISK_PER_TRADE) / stop_dist)
            stop_price  = c - stop_dist
            dollar_risk = portfolio * RISK_PER_TRADE
            pot_gain    = (mid - c) * pos_size if mid > c else 0.0
            setup = {
                "entry_price":   c,
                "stop_price":    stop_price,
                "stop_dist":     stop_dist,
                "position_size": pos_size,
                "dollar_risk":   dollar_risk,
                "current_atr":   a,
                "current_ma":    mid,
                "target":        mid,        # middle BB = mean-reversion target
                "potential_gain": pot_gain,
            }

        # Projected setup: always compute so execution brief is always populated
        projected_setup = None
        if a > 0 and lb > -np.inf:
            stop_atr_param = params["stop_loss_atr"]
            proj_entry     = lb * (1 + self._slip)
            stop_dist      = stop_atr_param * a
            proj_stop      = proj_entry - stop_dist
            proj_size      = int((portfolio * RISK_PER_TRADE) / stop_dist)
            pot_gain       = (mid - lb) * proj_size if mid > lb else 0.0
            projected_setup = {
                "entry_price":    proj_entry,
                "stop_price":     proj_stop,
                "stop_dist":      stop_dist,
                "position_size":  proj_size,
                "dollar_risk":    portfolio * RISK_PER_TRADE,
                "current_atr":    a,
                "current_ma":     mid,
                "target":         mid,
                "potential_gain": pot_gain,
                "entry_trigger":  lb,      # RSI < rsi_entry AND close ≤ lower_BB
                "rsi_needed":     rsi_entry,
            }

        return {
            "signal_active": active,
            "close":         c,
            "rsi":           r,
            "rsi_threshold": rsi_entry,
            "lower_bb":      lb,
            "middle_bb":     mid,
            "oversold":      oversold,
            "below_bb":      below_bb,
            "setup":         setup,
            "projected_setup": projected_setup,
            "details": (
                f"RSI {r:.1f} {'<' if oversold else '>='} {rsi_entry}"
                f" | Close {c:.2f} {'<=' if below_bb else '>'} Lower BB {lb:.2f}"
            ),
        }

    def _alpha_combined_signal(
        self, ohlcv: pd.DataFrame, params: dict, portfolio: float
    ) -> dict:
        """Current-bar signal check for AlphaCombined strategy."""
        close = ohlcv["Close"].astype(float)
        high  = ohlcv["High"].astype(float)
        low   = ohlcv["Low"].astype(float)

        alpha_th = float(params.get("alpha_threshold",   0.40))
        stop_atr = float(params.get("stop_loss_atr",     1.5))

        atr = self._atr(high, low, close)

        if "alpha_signal" in ohlcv.columns:
            alpha_sig = ohlcv["alpha_signal"].astype(float)
        else:
            alpha_sig = pd.Series(0.0, index=ohlcv.index)

        c = float(close.iloc[-1])
        a = float(atr.iloc[-1])   if not pd.isna(atr.iloc[-1])        else 0.0
        s = float(alpha_sig.iloc[-1]) if not pd.isna(alpha_sig.iloc[-1]) else 0.0

        active = s > alpha_th

        setup = None
        if active and a > 0:
            stop_dist   = stop_atr * a
            pos_size    = int((portfolio * RISK_PER_TRADE) / stop_dist)
            stop_price  = c - stop_dist
            setup = {
                "entry_price":   c,
                "stop_price":    stop_price,
                "stop_dist":     stop_dist,
                "position_size": pos_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "target":        None,
            }

        projected_setup = None
        if a > 0:
            stop_dist  = stop_atr * a
            proj_entry = c * (1 + self._slip)
            proj_stop  = proj_entry - stop_dist
            proj_size  = int((portfolio * RISK_PER_TRADE) / stop_dist)
            projected_setup = {
                "entry_price":   proj_entry,
                "stop_price":    proj_stop,
                "stop_dist":     stop_dist,
                "position_size": proj_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "target":        None,
                "entry_trigger": f"alpha_signal > {alpha_th:.2f}",
            }

        return {
            "signal_active":    active,
            "close":            c,
            "alpha_signal":     s,
            "alpha_threshold":  alpha_th,
            "setup":            setup,
            "projected_setup":  projected_setup,
            "details": (
                f"alpha_signal {s:.3f} {'>' if active else '<='} threshold {alpha_th:.2f}"
            ),
        }

    def _ml_signal_signal(
        self, ohlcv: pd.DataFrame, params: dict, portfolio: float
    ) -> dict:
        """Current-bar signal check for MLSignal strategy."""
        close = ohlcv["Close"].astype(float)
        high  = ohlcv["High"].astype(float)
        low   = ohlcv["Low"].astype(float)

        ml_th    = float(params.get("ml_threshold",  0.60))
        stop_atr = float(params.get("stop_loss_atr", 1.5))

        atr = self._atr(high, low, close)

        if "ml_signal" in ohlcv.columns:
            ml_sig = ohlcv["ml_signal"].astype(float)
        else:
            ml_sig = pd.Series(np.nan, index=ohlcv.index)

        c = float(close.iloc[-1])
        a = float(atr.iloc[-1])    if not pd.isna(atr.iloc[-1])    else 0.0
        s = float(ml_sig.iloc[-1]) if not pd.isna(ml_sig.iloc[-1]) else np.nan

        active = (not np.isnan(s)) and s > ml_th

        setup = None
        if active and a > 0:
            stop_dist   = stop_atr * a
            pos_size    = int((portfolio * RISK_PER_TRADE) / stop_dist)
            stop_price  = c - stop_dist
            setup = {
                "entry_price":   c,
                "stop_price":    stop_price,
                "stop_dist":     stop_dist,
                "position_size": pos_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "target":        None,
            }

        projected_setup = None
        if a > 0:
            stop_dist  = stop_atr * a
            proj_entry = c * (1 + self._slip)
            proj_stop  = proj_entry - stop_dist
            proj_size  = int((portfolio * RISK_PER_TRADE) / stop_dist)
            projected_setup = {
                "entry_price":   proj_entry,
                "stop_price":    proj_stop,
                "stop_dist":     stop_dist,
                "position_size": proj_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "target":        None,
                "entry_trigger": f"ml_signal > {ml_th:.2f}",
            }

        s_display = f"{s:.3f}" if not np.isnan(s) else "N/A"
        return {
            "signal_active":   active,
            "close":           c,
            "ml_signal":       s if not np.isnan(s) else None,
            "ml_threshold":    ml_th,
            "setup":           setup,
            "projected_setup": projected_setup,
            "details": (
                f"ml_signal {s_display} {'>' if active else '<='} threshold {ml_th:.2f}"
            ),
        }

    def _volatility_breakout_signal(
        self, ohlcv: pd.DataFrame, params: dict, portfolio: float
    ) -> dict:
        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)
        volume = ohlcv["Volume"].astype(float)

        bb_period        = params["bb_period"]
        squeeze_pct      = params["squeeze_pct"]
        squeeze_lookback = params.get("squeeze_lookback", 5)
        volume_mult      = params.get("volume_mult", 1.5)
        stop_loss_atr    = params["stop_loss_atr"]

        atr      = self._atr(high, low, close)
        bb_ma    = close.rolling(bb_period).mean()
        bb_std_s = close.rolling(bb_period).std(ddof=1)
        upper_bb = bb_ma + 2.0 * bb_std_s
        bb_width = (4.0 * bb_std_s) / bb_ma.replace(0, np.nan)
        sq_thresh        = bb_width.rolling(bb_period).quantile(squeeze_pct).shift(1)
        squeezed_flag    = (bb_width.shift(1) <= sq_thresh)
        was_sq_recently  = bool(squeezed_flag.rolling(squeeze_lookback).max().iloc[-1] or False)

        vol_ma = volume.rolling(20).mean()

        c   = float(close.iloc[-1])
        a   = float(atr.iloc[-1])     if not pd.isna(atr.iloc[-1])     else 0.0
        ub  = float(upper_bb.iloc[-1]) if not pd.isna(upper_bb.iloc[-1]) else float("inf")
        v   = float(volume.iloc[-1])
        vm  = float(vol_ma.iloc[-1])   if not pd.isna(vol_ma.iloc[-1])  else 0.0
        mid = float(bb_ma.iloc[-1])    if not pd.isna(bb_ma.iloc[-1])   else 0.0

        vol_confirmed = v > volume_mult * vm and vm > 0
        bb_breakout   = c > ub and ub < float("inf")
        active        = was_sq_recently and bb_breakout and vol_confirmed

        setup = None
        if active and a > 0:
            stop_dist  = stop_loss_atr * a
            pos_size   = int((portfolio * RISK_PER_TRADE) / stop_dist)
            stop_price = c - stop_dist
            setup = {
                "entry_price":   c,
                "stop_price":    stop_price,
                "stop_dist":     stop_dist,
                "position_size": pos_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "current_ma":    mid,
                "target":        None,
            }

        projected_setup = None
        if a > 0 and ub < float("inf"):
            stop_dist  = stop_loss_atr * a
            proj_entry = ub * (1 + self._slip)
            proj_stop  = proj_entry - stop_dist
            proj_size  = int((portfolio * RISK_PER_TRADE) / stop_dist)
            projected_setup = {
                "entry_price":     proj_entry,
                "stop_price":      proj_stop,
                "stop_dist":       stop_dist,
                "position_size":   proj_size,
                "dollar_risk":     portfolio * RISK_PER_TRADE,
                "current_atr":     a,
                "current_ma":      mid,
                "target":          None,
                "entry_trigger":   ub,
                "volume_needed":   volume_mult * vm,
                "squeeze_lookback": squeeze_lookback,
                "squeeze_detected": was_sq_recently,
            }

        return {
            "signal_active":      active,
            "close":              c,
            "upper_bb":           ub,
            "squeeze_detected":   was_sq_recently,
            "bb_breakout":        bb_breakout,
            "volume_confirmed":   vol_confirmed,
            "setup":              setup,
            "projected_setup":    projected_setup,
            "details": (
                f"Squeeze (last {squeeze_lookback}d): {'YES' if was_sq_recently else 'NO'}"
                f" | Close {c:.2f} vs Upper BB {ub:.2f}"
                f" | Volume {v:,.0f} vs {volume_mult}x avg {vm:,.0f}"
            ),
        }

    def _event_driven_signal(
        self, ohlcv: pd.DataFrame, params: dict, portfolio: float
    ) -> dict:
        """Current-bar signal check for EventDriven (PEAD) strategy."""
        if "pead_signal" not in ohlcv.columns or "earnings_gap" not in ohlcv.columns:
            return {
                "signal_active": False,
                "details": (
                    "EventDriven: PEAD features absent — call "
                    "OHLCVFetcher.add_earnings_drift_features() first."
                ),
                "setup": None,
                "projected_setup": None,
            }

        close  = ohlcv["Close"].astype(float)
        high   = ohlcv["High"].astype(float)
        low    = ohlcv["Low"].astype(float)
        volume = ohlcv["Volume"].astype(float)

        gap_threshold  = float(params.get("gap_threshold",        0.02))
        pead_min_sig   = float(params.get("pead_min_signal",       0.20))
        entry_window   = int(params.get("entry_window_bars",       10))
        vol_mult       = float(params.get("volume_mult",           1.3))
        ma_period      = int(params.get("ma_filter_period",        5))
        stop_atr       = float(params.get("stop_loss_atr",         1.5))

        atr      = self._atr(high, low, close)
        pead_sig = ohlcv["pead_signal"].fillna(0.0).astype(float)
        earn_gap = ohlcv["earnings_gap"].fillna(0.0).astype(float)
        ma_short = close.rolling(ma_period).mean()
        vol_ma   = volume.rolling(20).mean().shift(1)
        blackout = (
            ohlcv["earnings_blackout"]
            if "earnings_blackout" in ohlcv.columns
            else pd.Series(False, index=ohlcv.index)
        )

        recent_gap_flag = (
            earn_gap.gt(gap_threshold)
            .rolling(entry_window)
            .max()
            .shift(1)
            .fillna(0)
            .astype(bool)
        )

        c  = float(close.iloc[-1])
        a  = float(atr.iloc[-1])   if not pd.isna(atr.iloc[-1])   else 0.0
        ps = float(pead_sig.iloc[-1])
        v  = float(volume.iloc[-1])
        vm = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else 0.0
        m5 = float(ma_short.iloc[-1]) if not pd.isna(ma_short.iloc[-1]) else 0.0

        gap_ok     = bool(recent_gap_flag.iloc[-1])
        pead_ok    = ps > pead_min_sig
        ma_ok      = c > m5 and m5 > 0
        vol_ok     = v > vol_mult * vm and vm > 0
        no_blackout = not bool(blackout.iloc[-1])
        active     = gap_ok and pead_ok and ma_ok and vol_ok and no_blackout

        setup = None
        if active and a > 0:
            stop_dist   = stop_atr * a
            pos_size    = int((portfolio * RISK_PER_TRADE) / stop_dist)
            stop_price  = c - stop_dist
            setup = {
                "entry_price":   c,
                "stop_price":    stop_price,
                "stop_dist":     stop_dist,
                "position_size": pos_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "pead_signal":   ps,
                "target":        c + stop_dist,   # 1R minimum target
            }

        projected_setup = None
        if a > 0:
            stop_dist  = stop_atr * a
            proj_entry = c * (1 + self._slip)
            proj_stop  = proj_entry - stop_dist
            proj_size  = int((portfolio * RISK_PER_TRADE) / stop_dist)
            projected_setup = {
                "entry_price":   proj_entry,
                "stop_price":    proj_stop,
                "stop_dist":     stop_dist,
                "position_size": proj_size,
                "dollar_risk":   portfolio * RISK_PER_TRADE,
                "current_atr":   a,
                "pead_signal":   ps,
                "target":        proj_entry + stop_dist,
                "entry_trigger": (
                    f"pead_signal > {pead_min_sig:.2f} AND "
                    f"positive gap > {gap_threshold:.0%} within last {entry_window} bars"
                ),
                "volume_needed": vol_mult * vm,
            }

        # Detail string breaks down each condition so the report shows exactly
        # which gate is blocking or passing — mirrors the structure of other strategies.
        condition_parts = [
            f"gap>{gap_threshold:.0%} in {entry_window}d: {'YES' if gap_ok else 'NO'}",
            f"PEAD {ps:.2f}>={pead_min_sig:.2f}: {'YES' if pead_ok else 'NO'}",
            f"above MA({ma_period}) {m5:.2f}: {'YES' if ma_ok else 'NO'}",
            f"vol {v:,.0f} vs {vol_mult}x avg {vm:,.0f}: {'YES' if vol_ok else 'NO'}",
            f"outside blackout: {'YES' if no_blackout else 'NO'}",
        ]
        details = (
            f"EventDriven ACTIVE — PEAD drift firing | "
            + " | ".join(condition_parts)
            if active
            else "EventDriven: waiting — " + " | ".join(condition_parts)
        )

        return {
            "signal_active":    active,
            "close":            c,
            "pead_signal":      ps,
            "pead_threshold":   pead_min_sig,
            "recent_gap":       gap_ok,
            "above_ma":         ma_ok,
            "volume_confirmed": vol_ok,
            "outside_blackout": no_blackout,
            "setup":            setup,
            "projected_setup":  projected_setup,
            "details":          details,
        }

    # ── equity curve & summary ────────────────────────────────────────────────

    def _build_equity_curve(self, ohlcv: pd.DataFrame, trade_log: list[dict]) -> pd.Series:
        """
        Daily mark-to-market equity curve.

        During an open trade, unrealized P&L = (close − entry_price) × position_size
        is recognised each bar.  This gives a smooth, realistic equity curve whose
        pct_change() produces a daily-returns series with meaningful Sharpe statistics.
        Without MTM, the returns series has many zero-return flat days interrupted by
        large single-bar jumps at exits, which artificially depresses volatility and
        inflates Sharpe.
        """
        close  = ohlcv["Close"].astype(float)
        equity = pd.Series(self.initial_portfolio, index=ohlcv.index, dtype=float)

        if not trade_log:
            return equity

        date_to_idx: dict = {d: i for i, d in enumerate(ohlcv.index)}
        cash = self.initial_portfolio

        for trade in sorted(trade_log, key=lambda t: t["entry_date"]):
            ei = date_to_idx.get(trade["entry_date"])
            xi = date_to_idx.get(trade["exit_date"])
            if ei is None or xi is None:
                cash += trade["pnl"]
                continue

            ep  = trade["entry_price"]    # includes entry slippage
            sz  = trade["position_size"]

            # Vectorised MTM: equity = cash_before_entry + (close − entry_price) × size
            hold_close = close.iloc[ei : xi + 1].values
            equity.iloc[ei : xi + 1] = cash + (hold_close - ep) * sz

            # Realise PnL at exit (exit_price already has exit slippage baked in)
            cash += trade["pnl"]

            # Flat period after exit will be overwritten when the next trade starts
            equity.iloc[xi + 1 :] = cash

        return equity

    def _build_returns(
        self, ohlcv: pd.DataFrame, trade_log: list[dict], equity_curve: pd.Series
    ) -> pd.Series:
        """
        Daily returns where flat (no-position) days earn DAILY_RF instead of 0.

        Without this, the Sharpe calculation subtracts daily RF from every flat
        day (return 0 - RF = negative), making a modestly positive strategy look
        terrible.  When not in position, idle capital earns T-bill rate — so the
        excess return on those days is exactly zero, not negative.
        """
        returns = equity_curve.pct_change().fillna(0.0)

        # Mark every day that falls within an open trade window as "in position"
        in_position = pd.Series(False, index=ohlcv.index)
        for trade in trade_log:
            entry = trade.get("entry_date")
            exit_ = trade.get("exit_date")
            if entry is not None and exit_ is not None:
                try:
                    in_position.loc[entry:exit_] = True
                except Exception:
                    pass

        # Flat days earn daily_rf so Sharpe numerator uses excess over RF only
        # on invested days — idle cash days contribute zero excess, not negative
        returns[~in_position] = DAILY_RF
        return returns

    @staticmethod
    def _summarize(trade_log: list[dict], equity_curve: pd.Series) -> dict:
        if not trade_log:
            return {
                "total_return": 0.0, "trade_count": 0, "win_rate": 0.0,
                "total_slippage_cost": 0.0, "gross_return": 0.0,
                "entry_efficiency": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "payoff_ratio": 0.0, "exit_reason_breakdown": {},
                "avg_holding_days": 0.0,
            }
        initial   = equity_curve.iloc[0]
        final     = equity_curve.iloc[-1]
        total_return    = (final - initial) / initial if initial != 0 else 0.0
        wins            = [t for t in trade_log if t["pnl"] > 0]
        losses          = [t for t in trade_log if t["pnl"] < 0]
        total_slip      = sum(t.get("slippage_cost", 0.0) for t in trade_log)
        gross_pnl_total = sum(t.get("gross_pnl", t["pnl"]) for t in trade_log)
        gross_return    = gross_pnl_total / initial if initial != 0 else 0.0
        reached_count   = sum(1 for t in trade_log if t.get("reached_1r", False))

        # Payoff asymmetry: the core question of whether the strategy has real edge.
        # avg_win / avg_loss > 1.0 means winners are larger than losers on average.
        # avg_win ≈ avg_loss with win_rate ~0.55 gives profit_factor ≈ 1.2 —
        # the "noise trading" signature the quant critique flagged.
        avg_win  = float(np.mean([t["pnl"] for t in wins]))  if wins   else 0.0
        avg_loss = float(np.mean([abs(t["pnl"]) for t in losses])) if losses else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 1e-6 else 0.0

        # Exit reason breakdown: shows *why* the strategy exits.
        # A healthy strategy has diverse exits; > 60% alpha_reversal means
        # the signal flips before capturing meaningful profit — enter late, exit early.
        exit_reasons: dict[str, int] = {}
        for t in trade_log:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        avg_hold = float(np.mean([t.get("holding_days", 0) for t in trade_log]))

        return {
            "total_return":          float(total_return),
            "gross_return":          float(gross_return),
            "total_slippage_cost":   float(total_slip),
            "trade_count":           len(trade_log),
            "win_rate":              len(wins) / len(trade_log),
            "entry_efficiency":      float(reached_count / len(trade_log)),
            # Payoff asymmetry — the signal quality indicator
            "avg_win":               round(avg_win,  2),
            "avg_loss":              round(avg_loss, 2),
            "payoff_ratio":          round(payoff_ratio, 3),
            # Exit diagnosis — shows if signal is too short-lived
            "exit_reason_breakdown": exit_reasons,
            "avg_holding_days":      round(avg_hold, 1),
        }

    @staticmethod
    def _summarize_event_driven(trade_log: list[dict], equity_curve: pd.Series) -> dict:
        """
        Extended summary for EventDriven (PEAD) trades — adds PEAD signal
        diagnostics on top of the standard per-strategy metrics.

        Extra fields over _summarize():
          avg_pead_signal_at_entry — mean PEAD drift signal at entry.
            A healthy strategy should show avg > pead_min_signal (default 0.20).
            If avg is only slightly above the threshold, the edge is thin.
          avg_earnings_gap_size — mean earnings gap that triggered each trade.
            Larger gaps (e.g., 0.05+) are associated with stronger PEAD drift
            (Bernard & Thomas 1989).
          pead_fade_rate — fraction of trades that exited because the PEAD
            signal faded (pead_fade exit), not because of stop or max hold.
            A high rate (> 30%) means the drift is short-lived.
        """
        base = Backtester._summarize(trade_log, equity_curve)
        if not trade_log:
            base.update({
                "avg_pead_signal_at_entry": 0.0,
                "avg_earnings_gap_size":    0.0,
                "pead_fade_rate":           0.0,
            })
            return base

        pead_sigs = [t["pead_signal_at_entry"] for t in trade_log
                     if t.get("pead_signal_at_entry") is not None]
        gap_sizes = [t["earnings_gap_size"] for t in trade_log
                     if t.get("earnings_gap_size") is not None]
        pead_fade_count = sum(1 for t in trade_log if t.get("exit_reason") == "pead_fade")

        base["avg_pead_signal_at_entry"] = round(float(np.mean(pead_sigs)), 4) if pead_sigs else 0.0
        base["avg_earnings_gap_size"]    = round(float(np.mean(gap_sizes)), 4) if gap_sizes else 0.0
        base["pead_fade_rate"]           = round(pead_fade_count / len(trade_log), 4)
        return base

    # ── indicators ────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = ATR_PERIOD) -> pd.Series:
        """
        Wilder-smoothed ATR — seeds with SMA of first `period` TRs, then RMA.
        Identical seeding to ohlcv_fetcher and regime_classifier.
        """
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        tr_vals  = tr.values.astype(float)
        atr_vals = np.full(len(tr_vals), np.nan)

        # Find first run of `period` consecutive non-NaN TRs (skip the NaN at index 0 from shift)
        first = 1  # index 0 is NaN from shift(1)
        seed_end = first + period  # exclusive
        if seed_end <= len(tr_vals) and not np.any(np.isnan(tr_vals[first:seed_end])):
            atr_vals[seed_end - 1] = tr_vals[first:seed_end].mean()
            for i in range(seed_end, len(tr_vals)):
                if not np.isnan(tr_vals[i]):
                    atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr_vals[i]) / period

        return pd.Series(atr_vals, index=tr.index)

    @staticmethod
    def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100 - 100 / (1 + rs)
        return rsi.fillna(100.0)   # all-gain bars → RSI = 100


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_trade(
    entry_date, entry_price, exit_date, exit_price,
    holding_days, position_size, pnl, exit_reason,
    gross_pnl: float = 0.0, slippage_cost: float = 0.0,
    reached_1r: bool = False,
) -> dict:
    return {
        "entry_date":    entry_date,
        "entry_price":   float(entry_price),
        "exit_date":     exit_date,
        "exit_price":    float(exit_price),
        "holding_days":  int(holding_days),
        "position_size": float(position_size),
        "pnl":           float(pnl),
        "gross_pnl":     float(gross_pnl),
        "slippage_cost": float(slippage_cost),
        "exit_reason":   exit_reason,
        "reached_1r":    bool(reached_1r),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ollama, os, json as _json
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
    ohlcv_ft  = {t: fetcher.compute_features(df) for t, df in ohlcv_raw.items() if df is not None}
    clf       = RegimeClassifier()
    sel       = StrategySelector(llm_client=llm)
    bt        = Backtester()

    for ticker, feats in ohlcv_ft.items():
        regime   = clf.classify(ticker, ohlcv_raw[ticker])
        strategy = sel.select(ticker, regime, feats, macro)
        result   = bt.run(ticker, strategy, ohlcv_raw[ticker])
        print(f"\n{'='*60}")
        print(f"  {ticker}  trades={result['summary']['trade_count']}"
              f"  return={result['summary']['total_return']:.1%}"
              f"  win_rate={result['summary']['win_rate']:.1%}")
        for t in result["trade_log"]:
            print(f"    {t['entry_date'].date()} → {t['exit_date'].date()}"
                  f"  pnl={t['pnl']:+.0f}  reason={t['exit_reason']}")
