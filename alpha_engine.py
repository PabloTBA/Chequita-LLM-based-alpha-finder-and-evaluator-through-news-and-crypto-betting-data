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

import numpy as np
import pandas as pd

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

        combined = (
            _cs_zscore(cs_mr_raw)      * _WEIGHTS["cs_mr"]     +
            _cs_zscore(residual_raw)   * _WEIGHTS["residual"]   +
            _cs_zscore(vol_spike_raw)  * _WEIGHTS["vol_spike"]  +
            _cs_zscore(mom_raw)        * _WEIGHTS["mom_2d"]
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

        print(
            f"[AlphaEngine] Done. Injected alpha_signal into "
            f"{sum(1 for d in result.values() if d is not None)} DataFrames."
        )
        return result
