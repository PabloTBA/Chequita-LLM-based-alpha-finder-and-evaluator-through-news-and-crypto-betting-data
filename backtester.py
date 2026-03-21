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
DEFAULT_SLIP_BPS  = 10     # 10 basis points (0.10%) per side — conservative retail estimate
ANNUAL_RF         = 0.045  # risk-free rate — must match diagnostics_engine.py
DAILY_RF          = ANNUAL_RF / 252  # T-bill daily return earned on idle (flat) days


class Backtester:
    def __init__(self, initial_portfolio: float = 100_000.0, slippage_bps: float = DEFAULT_SLIP_BPS):
        self.initial_portfolio = initial_portfolio
        self._slip = slippage_bps / 10_000  # convert to fraction

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, ticker: str, strategy: dict, ohlcv: pd.DataFrame) -> dict:
        """
        Back-test a strategy against OHLCV data.

        Parameters
        ----------
        ticker   : str
        strategy : dict — output of StrategySelector.select(); must contain
                   "strategy" (str) and "adjusted_params" (dict)
        ohlcv    : pd.DataFrame with Open/High/Low/Close/Volume columns

        Returns
        -------
        dict with keys: ticker, strategy, trade_log, equity_curve, returns, summary
        """
        strategy_type = strategy["strategy"]
        params        = strategy["adjusted_params"]

        if strategy_type == "Momentum":
            trade_log = self._run_momentum(ohlcv, params)
        else:
            trade_log = self._run_mean_reversion(ohlcv, params)

        equity_curve = self._build_equity_curve(ohlcv, trade_log)
        returns      = self._build_returns(ohlcv, trade_log, equity_curve)
        summary      = self._summarize(trade_log, equity_curve)

        return {
            "ticker":       ticker,
            "strategy":     strategy_type,
            "trade_log":    trade_log,
            "equity_curve": equity_curve,
            "returns":      returns,
            "summary":      summary,
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

        atr           = self._atr(high, low, close)
        ma            = close.rolling(ma_period).mean()
        vol_ma        = volume.rolling(20).mean().shift(1)
        rolling_high  = close.rolling(entry_lookback).max().shift(1)

        start = max(entry_lookback, 20, ma_period, ATR_PERIOD)

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
                if c > rh and v > vol_multiplier * vm:
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
                peak             = max(peak, c)
                trailing_stop    = peak - trailing_stop_atr * a
                h_bar            = float(high.iloc[i])
                if h_bar >= target_1r:
                    reached_1r = True

                exit_reason: str | None = None
                if c < stop_price:
                    exit_reason = "stop_loss"
                elif c < trailing_stop:
                    exit_reason = "trailing_stop"
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
        bb_ma    = close.rolling(bb_period).mean()
        bb_std   = close.rolling(bb_period).std(ddof=1)
        lower_bb = bb_ma - bb_std_mult * bb_std
        middle_bb = bb_ma

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
                if r < rsi_entry and c <= lb:
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

        return {
            "signal_active":    active,
            "close":            c,
            "rolling_high":     rh,
            "volume":           v,
            "vol_threshold":    vol_multiplier * vm,
            "breakout":         breakout,
            "volume_confirmed": vol_ok,
            "setup":            setup,
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
            "details": (
                f"RSI {r:.1f} {'<' if oversold else '>='} {rsi_entry}"
                f" | Close {c:.2f} {'<=' if below_bb else '>'} Lower BB {lb:.2f}"
            ),
        }

    # ── equity curve & summary ────────────────────────────────────────────────

    def _build_equity_curve(self, ohlcv: pd.DataFrame, trade_log: list[dict]) -> pd.Series:
        curve   = pd.Series(self.initial_portfolio, index=ohlcv.index, dtype=float)
        running = self.initial_portfolio
        for trade in sorted(trade_log, key=lambda t: t["exit_date"]):
            running += trade["pnl"]
            curve.loc[trade["exit_date"]:] = running
        return curve

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
            return {"total_return": 0.0, "trade_count": 0, "win_rate": 0.0,
                    "total_slippage_cost": 0.0, "gross_return": 0.0, "entry_efficiency": 0.0}
        initial           = equity_curve.iloc[0]
        final             = equity_curve.iloc[-1]
        total_return      = (final - initial) / initial if initial != 0 else 0.0
        wins              = sum(1 for t in trade_log if t["pnl"] > 0)
        total_slip        = sum(t.get("slippage_cost", 0.0) for t in trade_log)
        gross_pnl_total   = sum(t.get("gross_pnl", t["pnl"]) for t in trade_log)
        gross_return      = gross_pnl_total / initial if initial != 0 else 0.0
        reached_count     = sum(1 for t in trade_log if t.get("reached_1r", False))
        entry_efficiency  = reached_count / len(trade_log)
        return {
            "total_return":        float(total_return),
            "gross_return":        float(gross_return),
            "total_slippage_cost": float(total_slip),
            "trade_count":         len(trade_log),
            "win_rate":            wins / len(trade_log),
            "entry_efficiency":    float(entry_efficiency),
        }

    # ── indicators ────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = ATR_PERIOD) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

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
