"""
RegimeClassifier
================
Pure-math module. Classifies each ticker into one of eight industry-standard
market regimes using Hurst exponent, ATR/price, 20-day return direction, and
earnings proximity.

Regime labels (priority order)
-------------------------------
    ATR/price > 6%                             → "Crisis"
                                                 (extreme vol — all strategies
                                                  use tight params / skip entry)
    earnings_blackout within last 5 bars       → "Event-Driven"
                                                 (imminent catalyst — fade gaps)
    Hurst > 0.55  AND  20d return > 0          → "Trending-Up"
    Hurst > 0.55  AND  20d return ≤ 0          → "Trending-Down"
    Hurst < 0.45                               → "Mean-Reverting"
    0.45 ≤ Hurst ≤ 0.55:
        ATR/price > 3%                         → "High-Volatility"
        ATR/price < 1.5%                       → "Low-Volatility"
        else                                   → "Neutral"

Strategy mapping
----------------
    Trending-Up    → Momentum          (follow direction)
    Trending-Down  → Mean-Reversion    (fade downtrend / buy dips)
    High-Volatility→ VolatilityBreakout(squeeze → expansion)
    Mean-Reverting → Mean-Reversion
    Low-Volatility → Mean-Reversion
    Crisis         → Mean-Reversion    (extreme moves revert; tight params)
    Event-Driven   → Mean-Reversion    (post-earnings gap fill)
    Neutral        → Momentum          (default)

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
ATR_CRISIS           = 0.06    # 6% — extreme panic/distress; takes priority
ATR_HIGH_VOL         = 0.03    # 3%
ATR_LOW_VOL          = 0.015   # 1.5%
RET_LOOKBACK         = 20      # trading days for directional trend split
EARNINGS_LOOKBACK    = 5       # bars back to flag imminent earnings
MIN_ROWS             = 30
HURST_WINDOW         = 756     # 3 years — minimum for reliable R/S estimation on equity daily data
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

        close_s = ohlcv["Close"].astype(float)
        close   = close_s.values
        high    = ohlcv["High"].astype(float).values
        low     = ohlcv["Low"].astype(float).values

        hurst   = self._hurst(close[-HURST_WINDOW:])
        atr     = self._atr(high, low, close, period=ATR_PERIOD)
        atr_pct = float(atr / close[-1])

        # 20-day return for directional regime split
        ret_20d = float(close_s.iloc[-1] / close_s.iloc[-RET_LOOKBACK - 1] - 1) \
                  if len(close_s) > RET_LOOKBACK else 0.0

        # Earnings proximity — True if any of the last 5 bars were in blackout
        near_earnings = False
        if "earnings_blackout" in ohlcv.columns:
            near_earnings = bool(ohlcv["earnings_blackout"].iloc[-EARNINGS_LOOKBACK:].any())

        regime  = self._label(hurst, atr_pct, ret_20d, near_earnings)

        return {
            "ticker":        ticker,
            "hurst":         float(hurst),
            "atr_pct":       atr_pct,
            "ret_20d":       ret_20d,
            "near_earnings": near_earnings,
            "regime":        regime,
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
                direction = f"  ret20d={r['ret_20d']:+.1%}" if r.get('ret_20d') is not None else ""
                event_tag = "  [EARNINGS]" if r.get("near_earnings") else ""
                print(f"  [Regime] {ticker}: {r['regime']}  Hurst={r['hurst']:.3f}  ATR%={r['atr_pct']:.2%}{direction}{event_tag}")
                results.append(r)
            except ValueError:
                continue
        return results

    # ── private: Hurst exponent (R/S analysis) ────────────────────────────────

    @staticmethod
    def _hurst(prices: np.ndarray) -> float:
        """
        Estimate Hurst exponent via R/S analysis on raw log returns.

        Key design choices that produce realistic equity values (0.45–0.65):
          1. Operate on raw log returns (already stationary) — NOT on detrended
             price residuals.  Detrending prices and re-differencing re-introduces
             structural-break artefacts that inflate H toward 0.8–0.9.
          2. Logarithmically-spaced lags starting at 10 — skipping lags 2–9
             avoids the well-known short-lag upward bias caused by volatility
             clustering (GARCH effects inflate R/S at short horizons).
          3. Require ≥ 8 non-overlapping windows per lag (relaxed from 10 to
             preserve enough lags over a 3-year window).
          4. Weight OLS by number of sub-windows — stable short-lag estimates
             dominate the slope fit.

        With HURST_WINDOW = 756 (3 years) this typically yields 0.48–0.62 for
        liquid large-cap equities, matching the academic literature (Lo 1991;
        Di Matteo et al. 2005).
        """
        if len(prices) < 100:
            return 0.5

        log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
        n = len(log_ret)

        if n < 50:
            return 0.5

        # Logarithmically spaced lags: 10 → n//8 (≈ 10 lags over a 3-year window)
        max_lag = max(n // 8, 10)
        lags = np.unique(
            np.logspace(1.0, np.log10(max_lag), 15).astype(int)
        )
        lags = lags[(lags >= 10) & (lags <= max_lag)]

        log_rs_vals: list[float] = []
        log_lag_vals: list[float] = []
        weights:      list[float] = []

        for lag in lags:
            n_windows = n // lag
            if n_windows < 8:
                continue

            rs_vals: list[float] = []
            for start in range(0, n_windows * lag, lag):
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
                log_rs_vals.append(np.log(np.mean(rs_vals)))
                log_lag_vals.append(np.log(lag))
                weights.append(float(n_windows))

        if len(log_lag_vals) < 2:
            return 0.5

        w = np.array(weights)
        w = w / w.sum()
        H = float(np.polyfit(log_lag_vals, log_rs_vals, 1, w=w)[0])
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
    def _label(hurst: float, atr_pct: float,
               ret_20d: float = 0.0, near_earnings: bool = False) -> str:
        # 1. Crisis — extreme volatility takes priority over all other signals
        if atr_pct > ATR_CRISIS:
            return "Crisis"
        # 2. Event-Driven — imminent earnings catalyst overrides statistical regime
        if near_earnings:
            return "Event-Driven"
        # 3. Trending — split by 20-day price direction
        if hurst > HURST_TRENDING:
            return "Trending-Up" if ret_20d >= 0 else "Trending-Down"
        # 4. Mean-Reverting
        if hurst < HURST_MEAN_REVERTING:
            return "Mean-Reverting"
        # 5. Neutral Hurst zone — fall back to ATR/price
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
