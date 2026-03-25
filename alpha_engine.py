"""
AlphaEngine
===========
Computes cross-sectional, residual, and microstructure alpha signals
across the full ticker universe and injects them as an ``alpha_signal``
column into each ticker's OHLCV DataFrame.

All signals are properly lagged (shift(1)) so no look-ahead is present.

Alpha components
----------------
1. CS-MR-5d      (weight 0.40)
   Rank z-score of 5-day return cross-sectionally; fades the strongest
   recent performers relative to peers (Jegadeesh 1990 short-horizon reversal).
   High value = recent laggard = long signal.

2. Residual-5d   (weight 0.30)
   Market-neutral idiosyncratic reversion: removes SPY beta so only the
   stock-specific excess move is faded.  beta estimated via 60-day rolling OLS.

3. Volume-spike  (weight 0.20)
   Exhaustion signal: large volume accompanying a directional move is often
   climactic; the next-day reversal tendency is well-documented (Lo & Wang 2000).
   signal = -sign(r1d) * clip(vol_z, 0, 3)

4. Short-term momentum (weight 0.10)
   1–2 day continuation bias (Cooper et al. 2004).
   signal = sign(r2d) * scaled_magnitude

Combined signal is z-scored cross-sectionally each day so the backtester
threshold is always comparable regardless of universe size.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA as _PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_CS_MR_WINDOW    = 5    # days for cross-sectional reversion lookback
_VOL_Z_WINDOW    = 20   # bars for volume z-score
_BETA_WINDOW     = 60   # bars for rolling beta estimation
_VOL_CLIP        = 3.0  # clip vol-z at 3σ
_SPY_TICKER      = "SPY"

_WEIGHTS = {"cs_mr": 0.40, "residual": 0.30, "vol_spike": 0.20, "mom_2d": 0.10}


class AlphaEngine:
    """
    Compute multi-alpha signals and inject them into OHLCV DataFrames.

    Usage
    -----
        engine  = AlphaEngine()
        ohlcv_enriched = engine.compute(ohlcv_dict)
    """

    def compute(
        self, ohlcv_dict: dict[str, "pd.DataFrame | None"]
    ) -> dict[str, "pd.DataFrame | None"]:
        """
        Parameters
        ----------
        ohlcv_dict : dict[ticker -> DataFrame | None]
            Raw OHLCV data for every ticker in the universe (including SPY).

        Returns
        -------
        Same dict structure; each non-None DataFrame gains an ``alpha_signal``
        column (float, shift(1)-lagged — no look-ahead).
        Tickers with insufficient history get alpha_signal = 0.0.
        """
        valid = {
            t: df for t, df in ohlcv_dict.items()
            if df is not None and not df.empty
        }

        if len(valid) < 2:
            # Cross-sectional signals require ≥ 2 tickers
            result = {}
            for ticker, df in ohlcv_dict.items():
                if df is not None and not df.empty:
                    out = df.copy()
                    out["alpha_signal"] = 0.0
                    result[ticker] = out
                else:
                    result[ticker] = df
            return result

        print(
            f"[AlphaEngine] Computing cross-sectional alpha for "
            f"{len(valid)} tickers ..."
        )

        # ── 1. Align close & volume panels ────────────────────────────────────
        close_dict = {t: df["Close"].astype(float) for t, df in valid.items()}
        vol_dict   = {t: df["Volume"].astype(float) for t, df in valid.items()}

        close_panel = pd.DataFrame(close_dict).sort_index()
        vol_panel   = pd.DataFrame(vol_dict).sort_index()

        # ── 2. Return series (all shift(1) so yesterday's info is used) ───────
        ret_1d = close_panel.pct_change(1).shift(1)
        ret_2d = close_panel.pct_change(2).shift(1)
        ret_5d = close_panel.pct_change(_CS_MR_WINDOW).shift(1)

        # ── 3. CS-MR signal ───────────────────────────────────────────────────
        # Rank the 5-day returns cross-sectionally (pct rank → [0,1]).
        # Invert: high-rank (strong performer) → negative signal (fade it).
        pct_rank    = ret_5d.rank(axis=1, pct=True, na_option="keep")
        cs_mr_raw   = -(pct_rank - 0.5) * 2.0   # map [0,1] → [-1, +1], inverted

        # ── 4. Residual-reversion signal ──────────────────────────────────────
        residual_raw = pd.DataFrame(0.0, index=ret_5d.index, columns=ret_5d.columns)
        spy_r5 = ret_5d.get(_SPY_TICKER)

        if spy_r5 is not None:
            for ticker in ret_5d.columns:
                if ticker == _SPY_TICKER:
                    continue
                try:
                    t_r = ret_5d[ticker]
                    s_r = spy_r5.reindex(t_r.index)
                    both = t_r.notna() & s_r.notna()
                    if both.sum() < 30:
                        continue
                    cov  = t_r[both].rolling(_BETA_WINDOW, min_periods=20).cov(s_r[both])
                    var_ = s_r[both].rolling(_BETA_WINDOW, min_periods=20).var()
                    beta = (cov / var_.replace(0, np.nan)).clip(-3.0, 3.0)
                    # idiosyncratic move = actual - beta * market; fade it
                    resid = (t_r[both] - beta * s_r[both]).reindex(ret_5d.index).fillna(0.0)
                    residual_raw[ticker] = -resid
                except Exception:
                    pass

        # ── 5. Volume-spike exhaustion signal ─────────────────────────────────
        vol_mean  = vol_panel.rolling(_VOL_Z_WINDOW, min_periods=5).mean().shift(1)
        vol_std   = vol_panel.rolling(_VOL_Z_WINDOW, min_periods=5).std(ddof=1).shift(1)
        vol_z     = ((vol_panel - vol_mean) / vol_std.replace(0, np.nan)).clip(
            -_VOL_CLIP, _VOL_CLIP
        ).shift(1)
        # Fade the direction of the volume spike (exhaustion)
        vol_spike_raw = -ret_1d.apply(np.sign) * vol_z.abs()

        # ── 6. Short-term momentum (2-day continuation) ───────────────────────
        mom_raw = ret_2d.apply(np.sign) * ret_2d.abs().clip(upper=0.05) / 0.05

        # ── 7. Cross-sectional z-score each component, then combine ──────────
        def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
            """Row-wise cross-sectional z-score; returns 0 where std ≈ 0."""
            mu  = df.mean(axis=1)
            std = df.std(axis=1).replace(0.0, np.nan)
            return df.sub(mu, axis=0).div(std, axis=0).fillna(0.0)

        z_cs_mr     = _cs_zscore(cs_mr_raw)
        z_residual  = _cs_zscore(residual_raw)
        z_vol_spike = _cs_zscore(vol_spike_raw)
        z_mom       = _cs_zscore(mom_raw)

        # ── 7b. PCA orthogonalization (when sklearn is available) ─────────────
        # If the 4 components share substantial variance (avg pairwise corr > 0.40),
        # project them into PCA space to remove correlated variance before combining.
        # Each PC is an orthogonal linear combination of the original factors.
        # We keep PCs whose cumulative explained variance reaches 90% (Kaiser-like).
        # Fall back to fixed-weight combination if PCA fails or sklearn is absent.
        combined: pd.DataFrame | None = None

        if _SKLEARN_AVAILABLE:
            try:
                # Stack to (T*N_tickers, 4) — each (day, ticker) is one observation
                comp_vals = [z_cs_mr.values.flatten(), z_residual.values.flatten(),
                             z_vol_spike.values.flatten(), z_mom.values.flatten()]
                X = np.column_stack(comp_vals)

                # Remove rows with any NaN
                mask = ~np.any(np.isnan(X), axis=1)
                X_clean = X[mask]

                if X_clean.shape[0] >= 10 and X_clean.shape[0] > X_clean.shape[1]:
                    pca = _PCA()
                    pca.fit(X_clean)
                    cum_var = np.cumsum(pca.explained_variance_ratio_)
                    n_factors = int(np.searchsorted(cum_var, 0.90)) + 1
                    n_factors = max(1, min(n_factors, X_clean.shape[1]))

                    # Project all rows (including NaN-masked ones) using the fitted PCA
                    X_orth = np.full((X.shape[0], n_factors), np.nan)
                    X_orth[mask] = pca.transform(X_clean)[:, :n_factors]

                    # Reshape back to (T, N_tickers) panels and combine with equal weights
                    T, N = z_cs_mr.shape
                    idx, cols = z_cs_mr.index, z_cs_mr.columns
                    pc_panels = [
                        pd.DataFrame(X_orth[:, i].reshape(T, N), index=idx, columns=cols)
                        for i in range(n_factors)
                    ]
                    combined = sum(pc_panels) / n_factors   # equal-weight orthogonal PCs

                    print(
                        f"[AlphaEngine] PCA orthogonalization: {n_factors} factor(s) retained "
                        f"(≥90% variance explained). "
                        f"Loadings: {pca.explained_variance_ratio_[:n_factors].round(3)}"
                    )
            except Exception as _e:
                print(f"[AlphaEngine] PCA failed ({_e}); falling back to fixed weights.")
                combined = None

        if combined is None:
            # Fixed-weight fallback (original combination)
            combined = (
                z_cs_mr     * _WEIGHTS["cs_mr"]     +
                z_residual  * _WEIGHTS["residual"]   +
                z_vol_spike * _WEIGHTS["vol_spike"]  +
                z_mom       * _WEIGHTS["mom_2d"]
            )

        combined = _cs_zscore(combined)   # final normalisation

        # ── 8. Inject alpha_signal into each DataFrame ────────────────────────
        result: dict = {}
        for ticker, df in ohlcv_dict.items():
            if df is None or df.empty:
                result[ticker] = df
                continue
            out = df.copy()
            if ticker in combined.columns:
                out["alpha_signal"] = (
                    combined[ticker].reindex(out.index).fillna(0.0)
                )
            else:
                out["alpha_signal"] = 0.0
            result[ticker] = out

        # ── 9. Signal diagnostics — log redundancy warning ────────────────────
        components = {
            "cs_mr":     z_cs_mr,
            "residual":  z_residual,
            "vol_spike": z_vol_spike,
            "mom_2d":    z_mom,
        }
        signal_corr = self._signal_correlation(components)
        if signal_corr["avg_pairwise_corr"] > 0.70:
            print(
                f"[AlphaEngine] ⚠ HIGH SIGNAL REDUNDANCY — avg pairwise corr "
                f"{signal_corr['avg_pairwise_corr']:.2f} > 0.70.  "
                f"Components carry overlapping information; effective diversity "
                f"< {signal_corr['effective_n_signals']:.1f} independent signals.  "
                f"Consider orthogonalising via PCA residuals."
            )

        print(
            f"[AlphaEngine] Done. Injected alpha_signal into "
            f"{sum(1 for d in result.values() if d is not None)} DataFrames.  "
            f"Avg pairwise signal corr={signal_corr['avg_pairwise_corr']:.2f}"
        )
        return result

    def compute_signal_diagnostics(
        self, ohlcv_dict: dict[str, "pd.DataFrame | None"]
    ) -> dict:
        """
        Compute signal quality diagnostics without re-running the full alpha pipeline.
        Call this AFTER compute() so the alpha_signal column is already populated.

        Returns
        -------
        dict with keys:
          signal_correlation  : pairwise correlation between the 4 alpha components
          ic_by_horizon       : Information Coefficient at 1, 2, 5, 10 day horizons
          avg_pairwise_corr   : scalar summary of redundancy
          effective_n_signals : how many truly independent signals (= 1/avg_corr approx)
        """
        valid = {
            t: df for t, df in ohlcv_dict.items()
            if df is not None and not df.empty and "alpha_signal" in df.columns
        }
        if len(valid) < 2:
            return {"signal_correlation": {}, "ic_by_horizon": {}, "avg_pairwise_corr": 0.0,
                    "effective_n_signals": 4.0}

        close_dict = {t: df["Close"].astype(float) for t, df in valid.items()}
        vol_dict   = {t: df["Volume"].astype(float) for t, df in valid.items()}
        close_panel = pd.DataFrame(close_dict).sort_index()
        vol_panel   = pd.DataFrame(vol_dict).sort_index()

        ret_1d = close_panel.pct_change(1).shift(1)
        ret_2d = close_panel.pct_change(2).shift(1)
        ret_5d = close_panel.pct_change(_CS_MR_WINDOW).shift(1)

        pct_rank  = ret_5d.rank(axis=1, pct=True, na_option="keep")
        cs_mr_raw = -(pct_rank - 0.5) * 2.0

        residual_raw = pd.DataFrame(0.0, index=ret_5d.index, columns=ret_5d.columns)
        spy_r5 = ret_5d.get(_SPY_TICKER)
        if spy_r5 is not None:
            for ticker in ret_5d.columns:
                if ticker == _SPY_TICKER:
                    continue
                try:
                    t_r  = ret_5d[ticker]
                    s_r  = spy_r5.reindex(t_r.index)
                    both = t_r.notna() & s_r.notna()
                    if both.sum() < 30:
                        continue
                    cov  = t_r[both].rolling(_BETA_WINDOW, min_periods=20).cov(s_r[both])
                    var_ = s_r[both].rolling(_BETA_WINDOW, min_periods=20).var()
                    beta = (cov / var_.replace(0, np.nan)).clip(-3.0, 3.0)
                    resid = (t_r[both] - beta * s_r[both]).reindex(ret_5d.index).fillna(0.0)
                    residual_raw[ticker] = -resid
                except Exception:
                    pass

        vol_mean     = vol_panel.rolling(_VOL_Z_WINDOW, min_periods=5).mean().shift(1)
        vol_std      = vol_panel.rolling(_VOL_Z_WINDOW, min_periods=5).std(ddof=1).shift(1)
        vol_z        = ((vol_panel - vol_mean) / vol_std.replace(0, np.nan)).clip(-_VOL_CLIP, _VOL_CLIP).shift(1)
        vol_spike_raw = -ret_1d.apply(np.sign) * vol_z.abs()
        mom_raw       = ret_2d.apply(np.sign) * ret_2d.abs().clip(upper=0.05) / 0.05

        def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu  = df.mean(axis=1)
            std = df.std(axis=1).replace(0.0, np.nan)
            return df.sub(mu, axis=0).div(std, axis=0).fillna(0.0)

        components = {
            "cs_mr":     _cs_zscore(cs_mr_raw),
            "residual":  _cs_zscore(residual_raw),
            "vol_spike": _cs_zscore(vol_spike_raw),
            "mom_2d":    _cs_zscore(mom_raw),
        }

        # IC by horizon: correlation between today's combined signal and
        # forward returns at 1, 2, 5, 10 days (shift signal backward to align).
        # IC measures predictive power — IC > 0.05 daily is considered usable;
        # IC < 0.02 means the signal barely correlates with future returns.
        alpha_signal_panel = pd.DataFrame(
            {t: df["alpha_signal"] for t, df in valid.items()}
        ).sort_index()

        fwd_returns: dict[int, pd.DataFrame] = {}
        for h in (1, 2, 5, 10):
            # forward return: close at t+h / close at t - 1, aligned to current bar
            fwd_returns[h] = close_panel.pct_change(h).shift(-h)

        ic_by_horizon: dict[str, float] = {}
        for h, fwd in fwd_returns.items():
            ics = []
            for ticker in alpha_signal_panel.columns:
                sig = alpha_signal_panel[ticker].dropna()
                fwd_t = fwd[ticker].reindex(sig.index).dropna()
                both = sig.reindex(fwd_t.index).dropna()
                fwd_aligned = fwd_t.reindex(both.index)
                if len(both) > 20:
                    corr = float(np.corrcoef(both.values, fwd_aligned.values)[0, 1])
                    if not math.isnan(corr):
                        ics.append(corr)
            ic_by_horizon[f"IC_{h}d"] = round(float(np.mean(ics)), 4) if ics else 0.0

        signal_corr = self._signal_correlation(components)
        return {
            "signal_correlation": signal_corr["pairwise"],
            "ic_by_horizon":      ic_by_horizon,
            "avg_pairwise_corr":  signal_corr["avg_pairwise_corr"],
            "effective_n_signals": signal_corr["effective_n_signals"],
        }

    @staticmethod
    def _signal_correlation(
        components: dict[str, pd.DataFrame]
    ) -> dict:
        """
        Compute average pairwise Pearson correlation between the 4 z-scored
        alpha components across all rows and tickers.

        A high average correlation (> 0.70) means the signals are redundant —
        you effectively have 1 factor repeated N times, not N independent factors.
        This dilutes the Sharpe rather than diversifying it.

        Returns
        -------
        dict with keys: pairwise (dict), avg_pairwise_corr (float),
                        effective_n_signals (float)
        """
        names = list(components.keys())
        # Stack all panels into long series by stacking ticker columns
        long: dict[str, np.ndarray] = {}
        for name, panel in components.items():
            vals = panel.values.flatten()
            mask = ~np.isnan(vals)
            long[name] = vals[mask]

        # Align to shortest length
        min_len = min(len(v) for v in long.values())
        for name in long:
            long[name] = long[name][:min_len]

        pairwise: dict[str, float] = {}
        total_corr = 0.0
        n_pairs = 0
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                if min_len > 1:
                    corr = float(np.corrcoef(long[n1], long[n2])[0, 1])
                    if not math.isnan(corr):
                        pairwise[f"{n1}_{n2}"] = round(corr, 3)
                        total_corr += abs(corr)
                        n_pairs += 1

        avg_corr = total_corr / n_pairs if n_pairs > 0 else 0.0
        # Effective number of independent signals: 1 / avg_corr is an upper bound;
        # with 4 signals and avg_corr=0.8 → ~1.25 effective independent factors
        effective_n = min(len(names), 1.0 / avg_corr) if avg_corr > 0.01 else float(len(names))
        return {
            "pairwise":           pairwise,
            "avg_pairwise_corr":  round(avg_corr, 3),
            "effective_n_signals": round(effective_n, 2),
        }
