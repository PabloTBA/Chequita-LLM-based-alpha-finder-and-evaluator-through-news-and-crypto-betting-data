"""
Backtester
==========
Executes Momentum or Mean-Reversion strategy rules on 2-year OHLCV history.
Returns a trade log, equity curve, daily returns series, and summary stats.

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

Exit priority (AlphaCombined)
------------------------------
  1. Hard stop loss   : close < entry − stop_loss_atr × ATR_at_entry
  2. Trailing stop    : close < peak  − trailing_stop_atr × ATR
  3. Alpha reversal   : alpha_signal < reversal_threshold  (signal flips negative)
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
        self._slip = slippage_bps / 10_000  # convert to fraction

    # ── public ────────────────────────────────────────────────────────────────

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
        # Apply per-ticker ADV-tiered slippage when ADV is known
        if adv_shares > 0:
            self._slip = _slip_bps_for_adv(adv_shares) / 10_000
        else:
            self._slip = self._default_slip_bps / 10_000

        strategy_type = strategy["strategy"]
        params        = strategy["adjusted_params"]

        if strategy_type == "Momentum":
            trade_log = self._run_momentum(ohlcv, params)
        elif strategy_type == "VolatilityBreakout":
            trade_log = self._run_volatility_breakout(ohlcv, params)
        elif strategy_type == "AlphaCombined":
            trade_log = self._run_alpha_combined(ohlcv, params)
        else:
            trade_log = self._run_mean_reversion(ohlcv, params)

        equity_curve = self._build_equity_curve(ohlcv, trade_log)
        returns      = self._build_returns(ohlcv, trade_log, equity_curve)
        summary      = self._summarize(trade_log, equity_curve)

        # in_position series — used for exposure-adjusted benchmark
        in_pos = pd.Series(False, index=ohlcv.index)
        for trade in trade_log:
            try:
                in_pos.loc[trade["entry_date"]:trade["exit_date"]] = True
            except Exception:
                pass

        used_slip_bps = round(self._slip * 10_000, 1)
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

    # ── strategy engines ──────────────────────────────────────────────────────

    def _run_momentum(self, ohlcv: pd.DataFrame, params: dict) -> list[dict]:
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
        mom_lookback      = int(params.get("momentum_lookback", 0))

        atr           = self._atr(high, low, close)
        ma            = close.rolling(ma_period).mean().shift(1)   # shift(1): no look-ahead
        vol_ma        = volume.rolling(20).mean().shift(1)
        rolling_high  = close.rolling(entry_lookback).max().shift(1)
        blackout      = ohlcv["earnings_blackout"] if "earnings_blackout" in ohlcv.columns else pd.Series(False, index=ohlcv.index)

        # 12-1 month momentum gate: return from [t-252] to [t-21], fully non-look-ahead.
        # Skipping the most recent month avoids short-term reversal contamination.
        # Only computed when momentum_lookback > 0 (Momentum strategy only).
        if mom_lookback > 0:
            mom_filter = (close.shift(22) / close.shift(mom_lookback + 1) - 1)
        else:
            mom_filter = None

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
                    entry_price  = c * (1 + self._slip)   # pay spread on entry
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
                    exit_price      = c * (1 - self._slip)   # lose spread on exit
                    slip_cost       = (entry_price - c * (1 - self._slip + self._slip)) * pos_size
                    gross_pnl       = (c - (entry_price / (1 + self._slip))) * pos_size
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

    def _run_mean_reversion(self, ohlcv: pd.DataFrame, params: dict) -> list[dict]:
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
                    entry_price  = c * (1 + self._slip)   # pay spread on entry
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
                    exit_price    = c * (1 - self._slip)   # lose spread on exit
                    gross_pnl     = (c - (entry_price / (1 + self._slip))) * pos_size
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

    def _run_alpha_combined(self, ohlcv: pd.DataFrame, params: dict) -> list[dict]:
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
                    entry_price  = c * (1 + self._slip)
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
                    exit_price  = c * (1 - self._slip)
                    gross_pnl   = (c - (entry_price / (1 + self._slip))) * pos_size
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

    def _run_volatility_breakout(self, ohlcv: pd.DataFrame, params: dict) -> list[dict]:
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
                    entry_price   = c * (1 + self._slip)
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
                    exit_price  = c * (1 - self._slip)
                    gross_pnl   = (c - (entry_price / (1 + self._slip))) * pos_size
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
