"""
PairSelector
============
Identifies statistically valid pairs for Relative Value / Pair Trading from a
universe of OHLCV DataFrames.

A valid pair must pass three sequential filters:
  1. Pearson correlation ≥ min_corr  (returns move together — necessary but not sufficient)
  2. Engle-Granger cointegration test  (price levels share a long-run equilibrium)
  3. Mean-reversion half-life in [min_halflife, max_halflife] days
     (spread reverts fast enough to trade within our holding window)

Academic foundations
---------------------
  - Engle & Granger (1987): cointegrated pairs share an error-correction mechanism.
  - Vidyamurthy (2004): hedge ratio estimation via OLS of log-prices.
  - Avellaneda & Lee (2010): OU mean-reversion speed → optimal entry/exit thresholds.
  - Half-life = -ln(2) / ln(1 + rho)  where rho is the AR(1) coefficient on spread changes.

Cointegration test (Engle-Granger two-step)
-------------------------------------------
  Step 1: OLS regression  log(P_A) = alpha + beta * log(P_B) + epsilon
  Step 2: ADF test on epsilon (residuals)
          H0: unit root (NOT cointegrated)
          Reject H0 when t-stat < critical value → cointegrated

  Uses statsmodels when available; falls back to a pure-numpy ADF approximation
  calibrated against MacKinnon (1994) critical values.

Public interface
----------------
    selector = PairSelector()
    pairs    = selector.find_pairs(ohlcv_dict)
    # pairs is a list of dicts sorted by cointegration p-value (best first):
    # {
    #   "ticker_a": str, "ticker_b": str,
    #   "correlation": float,        # Pearson on log-returns
    #   "hedge_ratio": float,        # beta (OLS slope of log-prices)
    #   "coint_pvalue": float,       # Engle-Granger p-value
    #   "halflife_days": float,      # OU mean reversion half-life
    #   "spread_std": float,         # rolling spread std (last 60-bar window)
    #   "n_observations": int,       # aligned price observations used
    # }
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller as _sm_adfuller
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────────────────

_CORR_WINDOW        = 252   # days for correlation and cointegration estimation
_HALFLIFE_MIN       = 2     # minimum mean-reversion speed (days) — faster = noise
_HALFLIFE_MAX       = 30    # maximum mean-reversion speed — slower = won't resolve
_OLS_MIN_OBS        = 120   # minimum aligned observations for reliable OLS

# MacKinnon (1994) ADF critical values for T ≈ 250 (no constant, no trend case,
# adjusted for regression-on-residuals with one regressor):
_ADF_CRIT_1PCT  = -3.45
_ADF_CRIT_5PCT  = -2.87
_ADF_CRIT_10PCT = -2.57


class PairSelector:
    """
    Identifies cointegrated pairs from a universe of OHLCV DataFrames.

    Parameters
    ----------
    min_corr       : float — minimum Pearson correlation on log-returns (default 0.70)
    max_coint_pv   : float — maximum Engle-Granger p-value (default 0.10)
    max_pairs      : int   — maximum pairs to return (default 5, sorted by p-value)
    min_halflife   : float — minimum OU half-life in days (default 2)
    max_halflife   : float — maximum OU half-life in days (default 30)
    lookback_days  : int   — number of calendar days of history to use (default 756 ≈ 3yr)
    """

    def __init__(
        self,
        min_corr:     float = 0.70,
        max_coint_pv: float = 0.10,
        max_pairs:    int   = 5,
        min_halflife: float = float(_HALFLIFE_MIN),
        max_halflife: float = float(_HALFLIFE_MAX),
        lookback_days: int  = 756,
    ) -> None:
        self.min_corr      = min_corr
        self.max_coint_pv  = max_coint_pv
        self.max_pairs     = max_pairs
        self.min_halflife  = min_halflife
        self.max_halflife  = max_halflife
        self.lookback_days = lookback_days

    # ── public ────────────────────────────────────────────────────────────────

    def find_pairs(
        self, ohlcv_dict: dict[str, pd.DataFrame | None]
    ) -> list[dict]:
        """
        Scan all ticker pairs in ohlcv_dict and return those passing all filters,
        sorted by cointegration p-value (most significant first).

        Parameters
        ----------
        ohlcv_dict : dict[ticker → DataFrame | None]
            Must contain Close column.  None entries are silently skipped.

        Returns
        -------
        list[dict] — up to max_pairs entries, each with keys:
            ticker_a, ticker_b, correlation, hedge_ratio, coint_pvalue,
            halflife_days, spread_std, n_observations
        """
        valid: dict[str, pd.Series] = {}
        for ticker, df in ohlcv_dict.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].astype(float).dropna()
            if len(close) < _OLS_MIN_OBS:
                continue
            # Trim to lookback window
            cutoff = close.index[-1] - pd.Timedelta(days=self.lookback_days)
            close  = close[close.index >= cutoff]
            if len(close) >= _OLS_MIN_OBS:
                valid[ticker] = close

        tickers = sorted(valid.keys())
        n       = len(tickers)
        if n < 2:
            return []

        print(f"[PairSelector] Scanning {n*(n-1)//2} pairs from {n} tickers ...")

        candidates: list[dict] = []

        for i in range(n):
            for j in range(i + 1, n):
                ta, tb = tickers[i], tickers[j]
                result = self._evaluate_pair(ta, valid[ta], tb, valid[tb])
                if result is not None:
                    candidates.append(result)

        # Sort by cointegration p-value ascending (most significant = lowest p-value first)
        candidates.sort(key=lambda x: x["coint_pvalue"])
        selected = candidates[: self.max_pairs]

        for p in selected:
            print(
                f"  [PairSelector] {p['ticker_a']}/{p['ticker_b']}  "
                f"corr={p['correlation']:.3f}  coint_p={p['coint_pvalue']:.4f}  "
                f"half_life={p['halflife_days']:.1f}d  beta={p['hedge_ratio']:.3f}"
            )
        if not selected:
            print("  [PairSelector] No valid cointegrated pairs found.")

        return selected

    # ── private ───────────────────────────────────────────────────────────────

    def _evaluate_pair(
        self, ta: str, close_a: pd.Series, tb: str, close_b: pd.Series
    ) -> dict | None:
        """
        Run the three-filter evaluation for a single pair.
        Returns None if any filter fails; dict of stats if all pass.
        """
        # Align on common dates
        ca, cb = close_a.align(close_b, join="inner")
        if len(ca) < _OLS_MIN_OBS:
            return None

        log_a = np.log(ca.values.astype(float))
        log_b = np.log(cb.values.astype(float))

        # ── Filter 1: Pearson correlation on log-returns ──────────────────────
        ret_a = np.diff(log_a)
        ret_b = np.diff(log_b)
        if len(ret_a) < 30:
            return None
        corr = float(np.corrcoef(ret_a, ret_b)[0, 1])
        if corr < self.min_corr:
            return None

        # ── Filter 2: Engle-Granger cointegration ─────────────────────────────
        coint_result = self._cointegration(log_a, log_b)
        if coint_result["pvalue"] > self.max_coint_pv:
            return None

        hedge_ratio = coint_result["hedge_ratio"]
        residuals   = coint_result["residuals"]

        # ── Filter 3: OU mean-reversion half-life ────────────────────────────
        halflife = self._halflife(residuals)
        if halflife < self.min_halflife or halflife > self.max_halflife:
            return None

        # Rolling spread std on last 60 bars (matches backtester z-score window)
        spread = log_a - hedge_ratio * log_b
        spread_std = float(pd.Series(spread).rolling(60).std(ddof=1).iloc[-1])
        if np.isnan(spread_std) or spread_std <= 0:
            spread_std = float(np.std(spread[-60:], ddof=1))

        return {
            "ticker_a":       ta,
            "ticker_b":       tb,
            "correlation":    round(corr, 4),
            "hedge_ratio":    round(hedge_ratio, 4),
            "coint_pvalue":   round(coint_result["pvalue"], 6),
            "halflife_days":  round(halflife, 2),
            "spread_std":     round(spread_std, 6),
            "n_observations": len(ca),
        }

    def _cointegration(self, log_a: np.ndarray, log_b: np.ndarray) -> dict:
        """
        Engle-Granger two-step cointegration test.

        Step 1: OLS  log_a = alpha + beta * log_b + epsilon
        Step 2: ADF test on epsilon (H0: unit root = NOT cointegrated)

        Returns dict with keys: pvalue, hedge_ratio (beta), residuals
        """
        # ── OLS hedge ratio ───────────────────────────────────────────────────
        X = np.column_stack([np.ones(len(log_b)), log_b])
        try:
            coeffs = np.linalg.lstsq(X, log_a, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"pvalue": 1.0, "hedge_ratio": 1.0, "residuals": log_a - log_b}
        alpha_ols, beta = float(coeffs[0]), float(coeffs[1])
        residuals = log_a - (alpha_ols + beta * log_b)

        # ── ADF test on residuals ─────────────────────────────────────────────
        if _STATSMODELS_AVAILABLE:
            try:
                adf_result = _sm_adfuller(residuals, maxlag=1, regression="c", autolag=None)
                tstat  = float(adf_result[0])
                pvalue = float(adf_result[1])
                return {"pvalue": pvalue, "hedge_ratio": beta, "residuals": residuals}
            except Exception:
                pass

        # Fallback: manual ADF (AR(1) on differenced residuals)
        pvalue = self._adf_pvalue_approx(residuals)
        return {"pvalue": pvalue, "hedge_ratio": beta, "residuals": residuals}

    @staticmethod
    def _adf_pvalue_approx(residuals: np.ndarray) -> float:
        """
        Approximate ADF p-value via AR(1) regression on spread changes.

        Regresses: delta_e[t] = phi * e[t-1] + noise
        t-stat = phi / std_err(phi)

        MacKinnon (1994) critical values for cointegrating residuals (1 regressor):
          1%: -3.45   5%: -2.87   10%: -2.57
        Linear interpolation gives approximate p-value.
        """
        e  = residuals[:-1]
        de = np.diff(residuals)
        if len(e) < 10 or np.var(e) < 1e-12:
            return 1.0

        # OLS: de = phi * e + constant
        X      = np.column_stack([np.ones(len(e)), e])
        try:
            coeffs, residuals_ols, _, _ = np.linalg.lstsq(X, de, rcond=None)
        except np.linalg.LinAlgError:
            return 1.0
        phi    = float(coeffs[1])

        # Standard error of phi
        de_hat  = X @ coeffs
        sigma2  = np.sum((de - de_hat) ** 2) / max(len(de) - 2, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_phi  = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
        if se_phi < 1e-10:
            return 1.0
        tstat = phi / se_phi

        # Approximate p-value via piecewise linear interpolation through critical values
        # (one-tailed: H0 is unit root so we want t-stat to be very negative)
        if tstat < _ADF_CRIT_1PCT:
            return 0.005
        elif tstat < _ADF_CRIT_5PCT:
            # Interpolate between 1% and 5%
            frac = (tstat - _ADF_CRIT_1PCT) / (_ADF_CRIT_5PCT - _ADF_CRIT_1PCT)
            return round(0.01 + frac * 0.04, 4)
        elif tstat < _ADF_CRIT_10PCT:
            frac = (tstat - _ADF_CRIT_5PCT) / (_ADF_CRIT_10PCT - _ADF_CRIT_5PCT)
            return round(0.05 + frac * 0.05, 4)
        elif tstat < 0:
            # Between 10% and 50% — linear extrapolation
            frac = min((tstat - _ADF_CRIT_10PCT) / (0 - _ADF_CRIT_10PCT), 1.0)
            return round(0.10 + frac * 0.40, 4)
        return 1.0

    @staticmethod
    def _halflife(residuals: np.ndarray) -> float:
        """
        Estimate Ornstein-Uhlenbeck mean-reversion half-life from spread residuals.

        Fits AR(1):  e[t] = rho * e[t-1] + noise
        Half-life = -ln(2) / ln(rho)

        A half-life of 5 days means the spread reverts halfway to its mean in 5 days.
        Returns infinity if rho >= 1 (non-stationary) or negative (explosive).
        """
        e   = residuals[:-1]
        e1  = residuals[1:]
        if len(e) < 10 or np.var(e) < 1e-12:
            return float("inf")

        # OLS: e1 = rho * e + constant
        X = np.column_stack([np.ones(len(e)), e])
        try:
            coeffs = np.linalg.lstsq(X, e1, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float("inf")
        rho = float(coeffs[1])

        if rho >= 1.0 or rho <= 0.0:
            return float("inf")

        halflife = -np.log(2.0) / np.log(rho)
        return max(0.0, float(halflife))


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ohlcv_fetcher import OHLCVFetcher

    tickers = sys.argv[1:] or ["JPM", "BAC", "GS", "MS", "WFC", "C", "GOOGL", "META", "NVDA", "AMD"]
    print(f"Finding pairs in: {tickers}\n")

    fetcher  = OHLCVFetcher()
    raw      = fetcher.fetch(tickers)
    selector = PairSelector(min_corr=0.65, max_coint_pv=0.10, max_pairs=5)
    pairs    = selector.find_pairs(raw)

    print(f"\nFound {len(pairs)} valid pair(s):")
    for p in pairs:
        print(
            f"  {p['ticker_a']}/{p['ticker_b']}: "
            f"corr={p['correlation']:.3f}  beta={p['hedge_ratio']:.3f}  "
            f"coint_p={p['coint_pvalue']:.4f}  half_life={p['halflife_days']:.1f}d"
        )
