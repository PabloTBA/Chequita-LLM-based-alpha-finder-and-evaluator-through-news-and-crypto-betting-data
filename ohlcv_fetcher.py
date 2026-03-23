"""
OHLCVFetcher
============
Fetches 2 years of daily OHLCV data from yfinance for a list of tickers and
computes the summary features required by the screener and regime classifier.

Public interface
----------------
    fetcher = OHLCVFetcher()
    raw     = fetcher.fetch(["AAPL", "MSFT"])          # dict[str, DataFrame | None]
    feats   = fetcher.compute_features(raw["AAPL"])    # dict[str, float]

Features returned by compute_features
--------------------------------------
    return_20d      — 20-day price return
    rsi_14          — RSI(14) using Wilder smoothing
    atr_14          — ATR(14) using Wilder smoothing
    atr_pct         — ATR(14) / last close price
    52w_high_prox   — last close / 52-week high  (1.0 = at the high)
    52w_low_prox    — last close / 52-week low   (>1.0 = above the low)
    volume_ratio_30d— last volume / 30-day average volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Minimum rows needed to compute all features (252 trading days ≈ 1 year)
_MIN_ROWS = 30


class OHLCVFetcher:
    """Fetches and featurises OHLCV data from yfinance."""

    # ── public ────────────────────────────────────────────────────────────────

    def fetch(self, tickers: list[str], period: str = "10y") -> dict[str, pd.DataFrame | None]:
        """
        Download daily OHLCV for each ticker via yfinance.

        Returns a dict keyed by ticker symbol.  If yfinance returns an empty
        DataFrame for a ticker (unknown symbol, delisted, etc.) the value is
        None.  Other tickers are unaffected.
        """
        def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame | None]:
            try:
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if df.empty:
                    return ticker, None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return ticker, df
            except Exception:
                return ticker, None

        # max_workers=1 avoids yfinance internal cache collisions under threading
        data: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, df = future.result()
                data[ticker] = df
        return data

    def add_earnings_blackout(self, ticker: str, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """
        Add an 'earnings_blackout' boolean column to the OHLCV dataframe.
        True on days within ±window trading days of any historical earnings release.
        """
        df = df.copy()
        df["earnings_blackout"] = False
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is None or ed.empty:
                return df
            for edate in ed.index:
                edate_ts = pd.Timestamp(edate).normalize().tz_localize(None)
                # find nearest index position
                pos = df.index.searchsorted(edate_ts)
                lo  = max(0, pos - window)
                hi  = min(len(df), pos + window + 1)
                df.iloc[lo:hi, df.columns.get_loc("earnings_blackout")] = True
            count = int(df["earnings_blackout"].sum())
            if count > 0:
                print(f"  [Earnings] {ticker}: {count} blackout days (±{window} trading days around earnings)")
        except Exception:
            pass
        return df

    def compute_features(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute summary features from a raw OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: Open, High, Low, Close, Volume.
            Must have at least 30 rows.

        Returns
        -------
        dict with keys: return_20d, rsi_14, atr_14, atr_pct,
                        52w_high_prox, 52w_low_prox, volume_ratio_30d
        """
        if len(df) < _MIN_ROWS:
            raise ValueError(
                f"insufficient data: need at least {_MIN_ROWS} rows, got {len(df)}"
            )

        close  = df["Close"].astype(float).values
        high   = df["High"].astype(float).values
        low    = df["Low"].astype(float).values
        volume = df["Volume"].astype(float).values

        return {
            "return_20d":       float(self._return_20d(close)),
            "rsi_14":           float(self._rsi(close, period=14)),
            "atr_14":           float(self._atr(high, low, close, period=14)),
            "atr_pct":          float(self._atr(high, low, close, period=14) / close[-1]),
            "52w_high_prox":    float(self._52w_high_prox(close)),
            "52w_low_prox":     float(self._52w_low_prox(close)),
            "volume_ratio_30d": float(self._volume_ratio(volume, window=30)),
        }

    # ── private feature calculators ───────────────────────────────────────────

    @staticmethod
    def _return_20d(close: np.ndarray) -> float:
        return (close[-1] - close[-21]) / close[-21]

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
        """Wilder-smoothed RSI."""
        deltas = np.diff(close)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Seed with simple average of first `period` values
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        # Wilder smoothing for remaining values
        for g, l in zip(gains[period:], losses[period:]):
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + l) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Wilder-smoothed ATR."""
        prev_close = close[:-1]
        h = high[1:]
        l = low[1:]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))

        # Seed
        atr = tr[:period].mean()
        # Wilder smoothing
        for t in tr[period:]:
            atr = (atr * (period - 1) + t) / period
        return atr

    @staticmethod
    def _52w_high_prox(close: np.ndarray) -> float:
        window = close[-252:] if len(close) >= 252 else close
        return close[-1] / window.max()

    @staticmethod
    def _52w_low_prox(close: np.ndarray) -> float:
        window = close[-252:] if len(close) >= 252 else close
        return close[-1] / window.min()

    @staticmethod
    def _volume_ratio(volume: np.ndarray, window: int = 30) -> float:
        avg = volume[-window:].mean()
        if avg == 0:
            return 1.0
        return volume[-1] / avg


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    tickers = sys.argv[1:] or ["AAPL", "MSFT", "NVDA"]
    print(f"Fetching OHLCV for: {tickers}")
    fetcher = OHLCVFetcher()
    raw     = fetcher.fetch(tickers)

    for ticker, df in raw.items():
        if df is None:
            print(f"  {ticker}: no data (unknown/delisted)")
            continue
        try:
            feats = fetcher.compute_features(df)
            print(f"\n  {ticker} ({len(df)} rows)")
            for k, v in feats.items():
                print(f"    {k:20s}: {v:.4f}")
        except ValueError as e:
            print(f"  {ticker}: {e}")
