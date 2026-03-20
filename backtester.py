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

RISK_PER_TRADE = 0.01   # 1% of portfolio per trade
ATR_PERIOD     = 14
RSI_PERIOD     = 14


class Backtester:
    def __init__(self, initial_portfolio: float = 100_000.0):
        self.initial_portfolio = initial_portfolio

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
        returns      = equity_curve.pct_change().fillna(0.0)
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
                    entry_price  = c
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_loss_atr * a
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_loss_atr * a)
                    peak         = c
                    holding_days = 0
            else:
                holding_days += 1
                peak             = max(peak, c)
                trailing_stop    = peak - trailing_stop_atr * a

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
                    pnl    = (c - entry_price) * pos_size
                    equity += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], c,
                        holding_days, pos_size, pnl, exit_reason,
                    ))
                    in_position = False

        return trades

    def _run_mean_reversion(self, ohlcv: pd.DataFrame, params: dict) -> list[dict]:
        close = ohlcv["Close"].astype(float)
        high  = ohlcv["High"].astype(float)
        low   = ohlcv["Low"].astype(float)

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
                    entry_price  = c
                    entry_date   = close.index[i]
                    stop_price   = entry_price - stop_atr * a
                    pos_size     = (equity * RISK_PER_TRADE) / (stop_atr * a)
                    holding_days = 0
            else:
                holding_days += 1

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
                    pnl    = (c - entry_price) * pos_size
                    equity += pnl
                    trades.append(_make_trade(
                        entry_date, entry_price, close.index[i], c,
                        holding_days, pos_size, pnl, exit_reason,
                    ))
                    in_position = False

        return trades

    # ── equity curve & summary ────────────────────────────────────────────────

    def _build_equity_curve(self, ohlcv: pd.DataFrame, trade_log: list[dict]) -> pd.Series:
        curve   = pd.Series(self.initial_portfolio, index=ohlcv.index, dtype=float)
        running = self.initial_portfolio
        for trade in sorted(trade_log, key=lambda t: t["exit_date"]):
            running += trade["pnl"]
            curve.loc[trade["exit_date"]:] = running
        return curve

    @staticmethod
    def _summarize(trade_log: list[dict], equity_curve: pd.Series) -> dict:
        if not trade_log:
            return {"total_return": 0.0, "trade_count": 0, "win_rate": 0.0}
        initial      = equity_curve.iloc[0]
        final        = equity_curve.iloc[-1]
        total_return = (final - initial) / initial if initial != 0 else 0.0
        wins         = sum(1 for t in trade_log if t["pnl"] > 0)
        return {
            "total_return": float(total_return),
            "trade_count":  len(trade_log),
            "win_rate":     wins / len(trade_log),
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
) -> dict:
    return {
        "entry_date":    entry_date,
        "entry_price":   float(entry_price),
        "exit_date":     exit_date,
        "exit_price":    float(exit_price),
        "holding_days":  int(holding_days),
        "position_size": float(position_size),
        "pnl":           float(pnl),
        "exit_reason":   exit_reason,
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
