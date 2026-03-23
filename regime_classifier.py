"""
RegimeClassifier
================
Pure-math module. Classifies each ticker into a market regime using:
  1. Hurst exponent (R/S analysis on last 30 trading days of log returns)
  2. ATR(14) / last close price (fallback when Hurst is in the neutral zone)

Regime labels
-------------
    Hurst > 0.55               → "Trending"
    Hurst < 0.45               → "Mean-Reverting"
    0.45 ≤ Hurst ≤ 0.55:
        ATR/price > 3%         → "High-Volatility"
        ATR/price < 1.5%       → "Low-Volatility"
        else                   → "Neutral"

Public interface
----------------
    clf  = RegimeClassifier()
    r    = clf.classify("AAPL", ohlcv_df)      # single ticker
    rs   = clf.classify_all(ohlcv_dict)         # dict[str, DataFrame | None]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Regime thresholds (PRD defaults) ─────────────────────────────────────────

HURST_TRENDING       = 0.55
HURST_MEAN_REVERTING = 0.45
ATR_HIGH_VOL         = 0.03    # 3%
ATR_LOW_VOL          = 0.015   # 1.5%
MIN_ROWS             = 30
HURST_WINDOW         = 100     # trading days used for Hurst (30 is too noisy; 100 is industry minimum)
ATR_PERIOD           = 14


class RegimeClassifier:
    """Classifies price series into market regimes using Hurst + ATR/price."""

    # ── public ────────────────────────────────────────────────────────────────

    def classify(self, ticker: str, ohlcv: pd.DataFrame) -> dict:
        """
        Classify a single ticker.

        Parameters
        ----------
        ticker : str
        ohlcv  : pd.DataFrame — must have Close, High, Low columns; ≥ 30 rows.

        Returns
        -------
        dict with keys: ticker, hurst, atr_pct, regime
        """
        if len(ohlcv) < MIN_ROWS:
            raise ValueError(
                f"insufficient data for {ticker}: need ≥ {MIN_ROWS} rows, got {len(ohlcv)}"
            )

        close  = ohlcv["Close"].astype(float).values
        high   = ohlcv["High"].astype(float).values
        low    = ohlcv["Low"].astype(float).values

        hurst   = self._hurst(close[-HURST_WINDOW:])
        atr     = self._atr(high, low, close, period=ATR_PERIOD)
        atr_pct = float(atr / close[-1])
        regime  = self._label(hurst, atr_pct)

        return {
            "ticker":  ticker,
            "hurst":   float(hurst),
            "atr_pct": atr_pct,
            "regime":  regime,
        }

    def classify_all(self, ohlcv_dict: dict[str, pd.DataFrame | None]) -> list[dict]:
        """
        Classify all tickers in the dict.  None entries are silently skipped.
        """
        results = []
        for ticker, df in ohlcv_dict.items():
            if df is None or df.empty:
                continue
            try:
                r = self.classify(ticker, df)
                print(f"  [Regime] {ticker}: {r['regime']}  Hurst={r['hurst']:.3f}  ATR%={r['atr_pct']:.2%}")
                results.append(r)
            except ValueError:
                continue
        return results

    # ── private: Hurst exponent (R/S analysis) ────────────────────────────────

    @staticmethod
    def _hurst(prices: np.ndarray) -> float:
        """
        Estimate Hurst exponent via R/S analysis on log returns.

        Uses multiple lag scales from 2 to n//2, fits log(R/S) ~ H*log(lag).
        Returns 0.5 if estimation is not possible (too few unique R/S values).
        """
        if len(prices) < 4:
            return 0.5

        log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
        n       = len(log_ret)

        lags    = range(2, max(3, n // 2))
        log_rs  = []
        log_lag = []

        for lag in lags:
            rs_vals = []
            for start in range(0, n - lag + 1, lag):
                chunk = log_ret[start : start + lag]
                if len(chunk) < 2:
                    continue
                mean = chunk.mean()
                dev  = np.cumsum(chunk - mean)
                R    = dev.max() - dev.min()
                S    = chunk.std(ddof=1)
                if S > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                log_rs.append(np.log(np.mean(rs_vals)))
                log_lag.append(np.log(lag))

        if len(log_lag) < 2:
            return 0.5

        H = float(np.polyfit(log_lag, log_rs, 1)[0])
        # Clamp to [0, 1] — numerical edge cases can push slightly outside
        return float(np.clip(H, 0.0, 1.0))

    # ── private: ATR (Wilder smoothing) ──────────────────────────────────────

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> float:
        prev_close = close[:-1]
        h, l       = high[1:], low[1:]
        tr         = np.maximum(h - l, np.maximum(np.abs(h - prev_close),
                                                   np.abs(l - prev_close)))
        atr = tr[:period].mean()
        for t in tr[period:]:
            atr = (atr * (period - 1) + t) / period
        return float(atr)

    # ── private: regime label ─────────────────────────────────────────────────

    @staticmethod
    def _label(hurst: float, atr_pct: float) -> str:
        if hurst > HURST_TRENDING:
            return "Trending"
        if hurst < HURST_MEAN_REVERTING:
            return "Mean-Reverting"
        # Neutral Hurst zone — fall back to ATR/price
        if atr_pct > ATR_HIGH_VOL:
            return "High-Volatility"
        if atr_pct < ATR_LOW_VOL:
            return "Low-Volatility"
        return "Neutral"


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ohlcv_fetcher import OHLCVFetcher

    tickers = sys.argv[1:] or ["AAPL", "MSFT", "NVDA", "SPY"]
    print(f"Classifying regimes for: {tickers}\n")

    fetcher = OHLCVFetcher()
    raw     = fetcher.fetch(tickers)
    clf     = RegimeClassifier()
    results = clf.classify_all(raw)

    for r in results:
        print(f"  {r['ticker']:6s}  regime={r['regime']:16s}  hurst={r['hurst']:.3f}  atr_pct={r['atr_pct']:.3%}")
