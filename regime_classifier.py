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
    Hurst > 0.55  AND  20d return > 0          → "Trending-Up"
    Hurst > 0.55  AND  20d return ≤ 0          → "Trending-Down"
    Hurst < 0.45                               → "Mean-Reverting"
    0.45 ≤ Hurst ≤ 0.55:
        ATR/price > 3%                         → "High-Volatility"
        ATR/price < 1.5%                       → "Low-Volatility"
        earnings_blackout within last 5 bars   → "Event-Driven"
                                                 (only in neutral Hurst zone —
                                                  catalyst overlaid on ambiguous
                                                  structural regime)
        else                                   → "Neutral"

Design rationale
----------------
    Event-Driven is intentionally the LOWEST-priority regime. A trending stock
    near earnings is still trending — its Hurst signal is the primary information.
    Overriding with Event-Driven before Hurst analysis was checked discarded all
    structural regime information for any ticker with an upcoming earnings date,
    routing everything into AlphaCombined regardless of actual market structure.

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

MarketStateDetector
===================
Portfolio-level market state: "Crisis", "High-Volatility", or "Normal".
Uses SPY ATR/price and the fraction of the universe in stress regimes.

Adaptive thresholds (entry is immediate; exit requires sustained stability)
---------------------------------------------------------------------------
    Crisis    : spy_atr_pct ≥ 4%  OR  crisis_fraction ≥ 40%
    High-Vol  : spy_atr_pct ≥ 2.5% OR  stress_fraction ≥ 25%
    Normal    : spy_atr_pct < 3%   AND stress_fraction < 20%  for ≥ 5 days
                (hysteresis — requires sustained calm before returning to Normal)

Outputs
-------
    market_state     : "Crisis" | "High-Volatility" | "Normal"
    spy_atr_pct      : SPY's current ATR/price
    crisis_fraction  : fraction of universe in Crisis regime
    stress_fraction  : fraction in Crisis + High-Volatility
    stable_days      : consecutive days SPY ATR has been below recovery threshold
    recovery_score   : 0.0 (deep crisis) → 1.0 (fully normal); linear interpolation

Public interface
----------------
    det    = MarketStateDetector()
    state  = det.detect(spy_ohlcv, regime_results)
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
        # 2. Trending — split by 20-day price direction
        #    Hurst > 0.55 means persistent trend; checked BEFORE earnings so a
        #    trending stock near earnings remains "Trending-Up/Down", not
        #    force-routed to AlphaCombined.
        if hurst > HURST_TRENDING:
            return "Trending-Up" if ret_20d >= 0 else "Trending-Down"
        # 3. Mean-Reverting
        if hurst < HURST_MEAN_REVERTING:
            return "Mean-Reverting"
        # 4. Neutral Hurst zone (0.45–0.55) — fall back to ATR/price
        if atr_pct > ATR_HIGH_VOL:
            return "High-Volatility"
        if atr_pct < ATR_LOW_VOL:
            return "Low-Volatility"
        # 5. Event-Driven — earnings overlay, only when no structural regime found.
        #    Stock is in the ambiguous Hurst zone AND normal ATR range; an
        #    upcoming catalyst is then the dominant information.
        if near_earnings:
            return "Event-Driven"
        return "Neutral"


# ── Market-state thresholds ───────────────────────────────────────────────────

# Entry thresholds (immediate — one bad day is enough to trigger)
MS_CRISIS_SPY_ATR    = 0.04    # SPY ATR/price ≥ 4%  → Crisis
MS_CRISIS_FRAC       = 0.40    # ≥ 40% of universe in Crisis regime → Crisis
MS_HIGHVOL_SPY_ATR   = 0.025   # SPY ATR/price ≥ 2.5% → High-Volatility
MS_HIGHVOL_FRAC      = 0.25    # ≥ 25% of universe in Crisis+High-Vol → High-Volatility

# Exit thresholds (hysteresis — must stay below these for STABLE_DAYS_REQUIRED)
MS_RECOVERY_SPY_ATR  = 0.03    # SPY ATR/price must drop below 3% to start recovery
MS_RECOVERY_FRAC     = 0.20    # stress fraction must drop below 20% to start recovery
STABLE_DAYS_REQUIRED = 5       # consecutive stable days required before returning to Normal


class MarketStateDetector:
    """
    Detects the portfolio-level market state from SPY OHLCV and per-ticker
    regime results.  Uses hysteresis: easy to enter Crisis/High-Vol, requires
    sustained stability to exit back to Normal.
    """

    def detect(
        self,
        spy_ohlcv: "pd.DataFrame | None",
        regime_results: "list[dict]",
    ) -> dict:
        """
        Parameters
        ----------
        spy_ohlcv      : SPY OHLCV DataFrame (Close/High/Low columns).  May be None.
        regime_results : output of RegimeClassifier.classify_all() — list of regime dicts.

        Returns
        -------
        dict with keys:
            market_state     : "Crisis" | "High-Volatility" | "Normal"
            spy_atr_pct      : float — SPY's current ATR/price (0 if SPY unavailable)
            crisis_fraction  : float — fraction of universe in Crisis
            stress_fraction  : float — fraction in Crisis + High-Volatility
            stable_days      : int   — consecutive days SPY ATR < recovery threshold
            recovery_score   : float — 0.0 (deep crisis) → 1.0 (fully normal)
        """
        spy_atr_pct = self._spy_atr_pct(spy_ohlcv)
        stable_days = self._stable_days(spy_ohlcv)

        total = len(regime_results)
        if total == 0:
            crisis_count  = 0
            highvol_count = 0
        else:
            crisis_count  = sum(1 for r in regime_results if r.get("regime") == "Crisis")
            highvol_count = sum(1 for r in regime_results if r.get("regime") == "High-Volatility")

        crisis_fraction = crisis_count / total if total else 0.0
        stress_fraction = (crisis_count + highvol_count) / total if total else 0.0

        # ── State determination (entry is immediate) ──────────────────────────
        if spy_atr_pct >= MS_CRISIS_SPY_ATR or crisis_fraction >= MS_CRISIS_FRAC:
            market_state = "Crisis"
        elif spy_atr_pct >= MS_HIGHVOL_SPY_ATR or stress_fraction >= MS_HIGHVOL_FRAC:
            market_state = "High-Volatility"
        else:
            # Below both High-Vol thresholds, but only return "Normal" when
            # the market has been calm for at least STABLE_DAYS_REQUIRED days.
            # Before that, stay in "High-Volatility" as a buffer.
            if stable_days >= STABLE_DAYS_REQUIRED:
                market_state = "Normal"
            else:
                market_state = "High-Volatility"

        recovery_score = self._recovery_score(spy_atr_pct, stress_fraction, stable_days)

        print(
            f"  [MarketState] {market_state}  SPY_ATR%={spy_atr_pct:.2%}  "
            f"crisis_frac={crisis_fraction:.0%}  stress_frac={stress_fraction:.0%}  "
            f"stable_days={stable_days}  recovery={recovery_score:.2f}"
        )
        return {
            "market_state":    market_state,
            "spy_atr_pct":     spy_atr_pct,
            "crisis_fraction": crisis_fraction,
            "stress_fraction": stress_fraction,
            "stable_days":     stable_days,
            "recovery_score":  recovery_score,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _spy_atr_pct(spy_ohlcv: "pd.DataFrame | None") -> float:
        """Return SPY's current ATR/price.  Returns 0.0 if data unavailable."""
        if spy_ohlcv is None or spy_ohlcv.empty:
            return 0.0
        try:
            close = spy_ohlcv["Close"].astype(float).values
            high  = spy_ohlcv["High"].astype(float).values
            low   = spy_ohlcv["Low"].astype(float).values
            atr   = RegimeClassifier._atr(high, low, close, period=ATR_PERIOD)
            return float(atr / close[-1])
        except Exception:
            return 0.0

    @staticmethod
    def _stable_days(spy_ohlcv: "pd.DataFrame | None") -> int:
        """
        Count consecutive trailing days where SPY ATR/price was below the
        recovery threshold.  Uses a rolling 14-day ATR on each day's window.
        Returns 0 if data is insufficient.
        """
        if spy_ohlcv is None or len(spy_ohlcv) < ATR_PERIOD + 2:
            return 0
        try:
            close  = spy_ohlcv["Close"].astype(float).values
            high   = spy_ohlcv["High"].astype(float).values
            low    = spy_ohlcv["Low"].astype(float).values
            # Compute rolling ATR% for each of the last 30 bars
            lookback = min(30, len(close) - ATR_PERIOD - 1)
            stable   = 0
            for offset in range(lookback):
                end   = len(close) - offset
                if end < ATR_PERIOD + 1:
                    break
                c = close[:end]
                h = high[:end]
                l = low[:end]
                atr     = RegimeClassifier._atr(h, l, c, period=ATR_PERIOD)
                atr_pct = atr / c[-1]
                if atr_pct < MS_RECOVERY_SPY_ATR:
                    stable += 1
                else:
                    break   # streak broken — stop counting
            return stable
        except Exception:
            return 0

    @staticmethod
    def _recovery_score(spy_atr_pct: float, stress_fraction: float, stable_days: int) -> float:
        """
        Linear recovery score 0.0 → 1.0.
        0.0 = deepest crisis (SPY ATR ≥ 6%, 100% stress).
        1.0 = fully normal (SPY ATR < 1.5%, 0% stress, many stable days).
        """
        # ATR component: 0 at crisis threshold (4%), 1 at fully calm (1.5%)
        atr_score    = float(np.clip((MS_CRISIS_SPY_ATR - spy_atr_pct) /
                                     (MS_CRISIS_SPY_ATR - 0.015), 0.0, 1.0))
        # Stress-fraction component: 0 at 40%+, 1 at 0%
        frac_score   = float(np.clip(1.0 - stress_fraction / MS_CRISIS_FRAC, 0.0, 1.0))
        # Stability component: 0 at 0 days, 1 at STABLE_DAYS_REQUIRED+
        stab_score   = float(np.clip(stable_days / STABLE_DAYS_REQUIRED, 0.0, 1.0))
        return round(0.40 * atr_score + 0.40 * frac_score + 0.20 * stab_score, 3)


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
