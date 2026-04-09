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
    return_20d       — 20-day price return
    rsi_14           — RSI(14) using Wilder smoothing
    atr_14           — ATR(14) using Wilder smoothing
    atr_pct          — ATR(14) / last close price
    52w_high_prox    — last close / 52-week high  (1.0 = at the high)
    52w_low_prox     — last close / 52-week low   (>1.0 = above the low)
    volume_ratio_30d — last volume / 30-day average volume
    pead_signal_recent — most recent post-earnings drift signal (see add_earnings_drift_features)

Columns added by add_earnings_drift_features
---------------------------------------------
    earnings_gap  — at each earnings date: close[t+1]/close[t-1]-1; NaN otherwise.
                    Positive = price gapped up (beat); negative = miss.
    pead_signal   — sign(gap) × |gap|/0.10, clipped to [-1, +1]; filled forward
                    for up to 60 trading days after each earnings date so the
                    alpha engine can combine it as a persistent drift signal.
    pead_drift_5d — realized 5-day return starting the day after earnings
                    (research label; not available forward-looking in live mode).
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

    def add_earnings_drift_features(
        self, ticker: str, df: pd.DataFrame, pead_fill_days: int = 60
    ) -> pd.DataFrame:
        """
        Add post-earnings drift columns to the OHLCV dataframe.

        Columns added
        -------------
        earnings_gap  : float (NaN off earnings dates)
            close[t+1] / close[t-1] - 1 at each earnings announcement date.
            Positive = price gapped up after earnings (beat expectations).
            Negative = price gapped down (miss).  Uses the actual next-day open
            proxy (close[t+1]) so the signal is only available after the event.

        pead_signal   : float in [-1, +1], filled forward up to pead_fill_days
            sign(gap) × |gap|/0.10 clipped to [-1, +1].
            Filled forward for up to `pead_fill_days` trading days after each
            earnings date so alpha_engine can use it as a persistent drift signal.
            Based on Rendleman et al. (1982): post-earnings drift continues ~60 days.

        pead_drift_5d : float (NaN off earnings dates)
            Realized 5-day cumulative return starting the day after earnings.
            Research label — useful for measuring PEAD magnitude historically
            but NOT available forward-looking (only use for diagnostics).
        """
        df = df.copy()
        close = df["Close"].astype(float)

        df["earnings_gap"]  = float("nan")
        df["pead_signal"]   = float("nan")
        df["pead_drift_5d"] = float("nan")

        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is None or ed.empty:
                return df

            for edate in ed.index:
                edate_ts = pd.Timestamp(edate).normalize().tz_localize(None)
                pos = df.index.searchsorted(edate_ts)
                if pos <= 0 or pos >= len(df) - 1:
                    continue

                # earnings_gap: next close / prev close - 1
                gap = float(close.iloc[pos + 1] / close.iloc[pos - 1] - 1)
                df.at[df.index[pos], "earnings_gap"] = gap

                # pead_signal at announcement date
                raw_signal = float(np.sign(gap) * min(abs(gap), 0.10) / 0.10)
                df.at[df.index[pos], "pead_signal"] = raw_signal

                # pead_drift_5d: realized 5-day return starting day after earnings
                end_pos = min(pos + 6, len(df) - 1)
                if end_pos > pos + 1:
                    drift_5d = float(
                        close.iloc[end_pos] / close.iloc[pos + 1] - 1
                    )
                    df.at[df.index[pos], "pead_drift_5d"] = drift_5d

        except Exception:
            pass

        # Fill pead_signal forward for up to pead_fill_days after each earnings date
        df["pead_signal"] = df["pead_signal"].ffill(limit=pead_fill_days)

        count = int(df["earnings_gap"].notna().sum())
        if count > 0:
            print(f"  [PEAD] {ticker}: {count} earnings events; pead_signal filled forward {pead_fill_days}d")

        return df

    def compute_features(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute summary features from a raw OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: Open, High, Low, Close, Volume.
            Must have at least 30 rows.
            If 'pead_signal' column is present (from add_earnings_drift_features),
            pead_signal_recent is included in the output.

        Returns
        -------
        dict with keys: return_20d, rsi_14, atr_14, atr_pct,
                        52w_high_prox, 52w_low_prox, volume_ratio_30d,
                        adv_20d, pead_signal_recent
        """
        if len(df) < _MIN_ROWS:
            raise ValueError(
                f"insufficient data: need at least {_MIN_ROWS} rows, got {len(df)}"
            )

        close  = df["Close"].astype(float).values
        high   = df["High"].astype(float).values
        low    = df["Low"].astype(float).values
        volume = df["Volume"].astype(float).values

        feats = {
            "return_20d":       float(self._return_20d(close)),
            "rsi_14":           float(self._rsi(close, period=14)),
            "atr_14":           float(self._atr(high, low, close, period=14)),
            "atr_pct":          float(self._atr(high, low, close, period=14) / close[-1]),
            "52w_high_prox":    float(self._52w_high_prox(close)),
            "52w_low_prox":     float(self._52w_low_prox(close)),
            "volume_ratio_30d": float(self._volume_ratio(volume, window=30)),
            "adv_20d":          float(self._adv_shares(volume, window=20)),
        }

        # Include most-recent PEAD signal if the column is present
        if "pead_signal" in df.columns:
            pead_series = df["pead_signal"].dropna()
            feats["pead_signal_recent"] = float(pead_series.iloc[-1]) if len(pead_series) > 0 else 0.0
        else:
            feats["pead_signal_recent"] = 0.0

        return feats

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
        # Use [-window-1:-1] to exclude today — avoids look-ahead bias in the denominator
        hist = volume[-window - 1 : -1]
        avg  = hist.mean() if len(hist) > 0 else 0.0
        if avg == 0:
            return 1.0
        return volume[-1] / avg

    @staticmethod
    def _adv_shares(volume: np.ndarray, window: int = 20) -> float:
        """20-day average daily volume in shares (used for market-impact sizing)."""
        hist = volume[-window - 1 : -1]   # exclude today — same non-look-ahead logic
        return float(hist.mean()) if len(hist) > 0 else float(volume.mean())


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
