"""
PortfolioOptimizer
==================
Addresses the portfolio construction gap: tickers were evaluated independently
with no ranking, sizing, or correlation logic.

Three layers of construction:

1. Cross-sectional momentum ranking (Jegadeesh-Titman 1993)
   Ranks the universe by 12-1 month return (skip most recent month to avoid
   short-term reversal).  Only the top _CS_MOM_TOP_FRACTION proceed; the rest
   are rejected with an explanation.  This replaces the implicit assumption
   that every shortlisted ticker is equally worth trading.

2. Volatility-parity sizing
   Each position is sized so it contributes equal annualised vol risk.
   weight_i = target_vol / realised_vol_i, then normalised to sum to 1.
   Prevents high-ATR names from dominating portfolio risk even if they have
   good Sharpe ratios.

3. Correlation shrinkage
   When two positions have realized correlation > threshold, both weights are
   scaled down by sqrt(threshold / corr).  Simple, interpretable, and avoids
   the instability of full mean-variance optimisation on small samples.

Public interface
----------------
    opt    = PortfolioOptimizer(initial_portfolio=100_000.0)
    result = opt.optimize(backtests, diagnostics, ohlcv_dict)

Output schema
-------------
{
  "cs_momentum_ranks": [{"ticker", "mom_12_1", "rank"}, ...],
  "allocations":       [{"ticker", "weight", "dollar_allocation",
                          "sharpe", "cs_rank", "cs_momentum_12_1",
                          "rationale"}, ...],
  "rejected":          [{"ticker", "reason"}, ...],
  "portfolio_metrics": {"sharpe", "annual_vol", "var_95", "cvar_95",
                         "max_drawdown"},
  "portfolio_returns": pd.Series,   # daily weighted returns
}
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

# ── tuneable constants ────────────────────────────────────────────────────────

_TARGET_PORTFOLIO_VOL = 0.15   # 15% annualised target vol (institutional standard)
_MAX_POSITION_PCT     = 0.30   # hard cap: no single position > 30% of portfolio
_MIN_POSITION_PCT     = 0.02   # ignore round-off weights below 2%
_CORR_THRESHOLD       = 0.70   # scale down both legs when realised corr > this
_CS_MOM_TOP_FRACTION  = 0.60   # keep top 60% by 12-1 month momentum rank
_TRADING_DAYS         = 252
_RF_ANNUAL            = 0.045


class PortfolioOptimizer:
    def __init__(self, initial_portfolio: float = 100_000.0):
        self._portfolio = initial_portfolio

    # ── public ────────────────────────────────────────────────────────────────

    def optimize(
        self,
        backtests:   list[dict],
        diagnostics: list[dict],
        ohlcv_dict:  dict[str, pd.DataFrame | None],
    ) -> dict:
        """
        Run full portfolio construction pipeline.

        Parameters
        ----------
        backtests   : list of Backtester.run() outputs
        diagnostics : list of DiagnosticsEngine.run() outputs
        ohlcv_dict  : raw OHLCV DataFrames keyed by ticker
        """
        if not backtests:
            return self._empty()

        print("[PortfolioOptimizer] Building cross-sectional momentum ranks ...")
        cs_ranks = self._cs_momentum_ranks(ohlcv_dict)

        # ── build candidate set ───────────────────────────────────────────────
        candidates: list[dict] = []
        rejected:   list[dict] = []

        for bt in backtests:
            ticker    = bt["ticker"]
            daily_ret = bt.get("returns")
            if daily_ret is None or len(daily_ret) < 20:
                rejected.append({"ticker": ticker,
                                  "reason": "insufficient return series (<20 days)"})
                continue
            sharpe     = _sharpe(daily_ret)
            rank_info  = next((r for r in cs_ranks if r["ticker"] == ticker), None)
            candidates.append({
                "ticker":  ticker,
                "returns": daily_ret,
                "sharpe":  sharpe,
                "cs_rank": rank_info["rank"]   if rank_info else len(backtests) + 1,
                "cs_mom":  rank_info["mom_12_1"] if rank_info else 0.0,
            })

        # ── CS momentum filter ────────────────────────────────────────────────
        if len(candidates) > 2:
            n_keep   = max(2, int(len(candidates) * _CS_MOM_TOP_FRACTION))
            by_rank  = sorted(candidates, key=lambda x: x["cs_rank"])
            kept     = by_rank[:n_keep]
            filtered = by_rank[n_keep:]
            for f in filtered:
                rejected.append({
                    "ticker": f["ticker"],
                    "reason": (f"CS momentum filter: rank {f['cs_rank']} / {len(candidates)}, "
                               f"12-1m return {f['cs_mom']:+.1%} — below top-{_CS_MOM_TOP_FRACTION:.0%} cutoff"),
                })
        else:
            kept = candidates

        print(f"[PortfolioOptimizer] {len(kept)} tickers pass CS filter "
              f"({len(rejected)} rejected)")

        if not kept:
            return self._empty(cs_ranks=cs_ranks, rejected=rejected)

        # ── volatility-parity weights ─────────────────────────────────────────
        weights = self._vol_parity_weights([c["returns"] for c in kept])

        # ── correlation shrinkage ─────────────────────────────────────────────
        weights = self._correlation_shrink(weights, [c["returns"] for c in kept])

        # ── cap and normalise ─────────────────────────────────────────────────
        weights = np.clip(weights, 0.0, _MAX_POSITION_PCT)
        total_w = weights.sum()
        if total_w > 1e-6:
            weights = weights / total_w

        # ── build allocations list ────────────────────────────────────────────
        allocations: list[dict] = []
        active_returns: list[pd.Series] = []
        active_weights: list[float]     = []

        for i, c in enumerate(kept):
            w = float(weights[i])
            if w < _MIN_POSITION_PCT:
                rejected.append({
                    "ticker": c["ticker"],
                    "reason": f"post-normalisation weight {w:.2%} below minimum {_MIN_POSITION_PCT:.0%}",
                })
                continue
            allocations.append({
                "ticker":            c["ticker"],
                "weight":            round(w, 4),
                "dollar_allocation": round(w * self._portfolio, 2),
                "sharpe":            round(c["sharpe"], 3),
                "cs_rank":           c["cs_rank"],
                "cs_momentum_12_1":  round(c["cs_mom"], 4),
                "rationale":         (f"vol-parity {w:.1%}; "
                                      f"CS rank {c['cs_rank']}/{len(candidates)}; "
                                      f"12-1m mom {c['cs_mom']:+.1%}"),
            })
            active_returns.append(c["returns"])
            active_weights.append(w)

        allocations.sort(key=lambda x: x["weight"], reverse=True)

        # ── portfolio-level metrics ───────────────────────────────────────────
        port_rets, port_metrics = self._portfolio_metrics(
            active_returns, np.array(active_weights)
        )

        print(f"[PortfolioOptimizer] Portfolio Sharpe={port_metrics['sharpe']:.3f}  "
              f"Vol={port_metrics['annual_vol']:.1%}  "
              f"MaxDD={port_metrics['max_drawdown']:.1%}")

        return {
            "cs_momentum_ranks": cs_ranks,
            "allocations":       allocations,
            "rejected":          rejected,
            "portfolio_metrics": port_metrics,
            "portfolio_returns": port_rets,
        }

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cs_momentum_ranks(ohlcv_dict: dict) -> list[dict]:
        """
        Rank every ticker in ohlcv_dict by 12-1 month momentum.
        12-1 month = return from t-252 to t-21 (skip most recent month).
        """
        scores: list[dict] = []
        for ticker, df in ohlcv_dict.items():
            if df is None or df.empty:
                scores.append({"ticker": ticker, "mom_12_1": 0.0, "rank": 999})
                continue
            close = df["Close"].squeeze().astype(float)
            if len(close) < 253:
                scores.append({"ticker": ticker, "mom_12_1": 0.0, "rank": 999})
                continue
            mom = float(close.iloc[-22] / close.iloc[-253] - 1)
            scores.append({"ticker": ticker, "mom_12_1": round(mom, 4)})

        scores.sort(key=lambda x: x["mom_12_1"], reverse=True)
        for i, s in enumerate(scores):
            s["rank"] = i + 1
        return scores

    @staticmethod
    def _vol_parity_weights(returns_list: list[pd.Series]) -> np.ndarray:
        """
        Inverse-volatility weighting so each position contributes equal risk.
        weight_i ∝ target_vol / annualised_vol_i
        """
        vols = []
        for r in returns_list:
            vol = float(np.array(r).std(ddof=1)) * math.sqrt(_TRADING_DAYS)
            vols.append(max(vol, 0.001))   # floor to avoid div/0 on flat series
        vols    = np.array(vols, dtype=float)
        raw     = _TARGET_PORTFOLIO_VOL / vols
        return raw / raw.sum()

    @staticmethod
    def _correlation_shrink(
        weights: np.ndarray, returns_list: list[pd.Series]
    ) -> np.ndarray:
        """
        Scale down both positions in any pair with realised corr > threshold.
        Scaling factor = sqrt(threshold / corr) applied to both legs.
        """
        n = len(weights)
        if n < 2:
            return weights

        min_len = min(len(r) for r in returns_list)
        ret_mat = np.column_stack([np.array(r.values[-min_len:]) for r in returns_list])
        corr    = np.corrcoef(ret_mat.T)

        adjusted = weights.copy()
        for i in range(n):
            for j in range(i + 1, n):
                c = corr[i, j]
                if not np.isnan(c) and c > _CORR_THRESHOLD:
                    scale = math.sqrt(_CORR_THRESHOLD / c)
                    adjusted[i] *= scale
                    adjusted[j] *= scale

        return adjusted

    @staticmethod
    def _portfolio_metrics(
        returns_list: list[pd.Series],
        weights:      np.ndarray,
    ) -> tuple[pd.Series, dict]:
        """
        Combine individual return series into a weighted portfolio return,
        then compute aggregate risk metrics.
        """
        if not returns_list:
            return pd.Series(dtype=float), _zero_metrics()

        min_len = min(len(r) for r in returns_list)
        ret_mat = np.column_stack([
            np.array(r.values[-min_len:], dtype=float) for r in returns_list
        ])
        w = np.array(weights[:len(returns_list)], dtype=float)
        if w.sum() > 1e-6:
            w = w / w.sum()

        port_arr = ret_mat @ w
        port_ret = pd.Series(port_arr)

        sharpe     = _sharpe(port_ret)
        annual_vol = float(port_arr.std(ddof=1)) * math.sqrt(_TRADING_DAYS)

        sorted_r = np.sort(port_arr)
        var_idx  = max(1, int(len(sorted_r) * 0.05))
        var_95   = float(-sorted_r[var_idx])
        cvar_95  = float(-sorted_r[:var_idx].mean()) if var_idx > 0 else var_95

        equity   = np.cumprod(1 + port_arr)
        roll_max = np.maximum.accumulate(equity)
        max_dd   = float(np.max((roll_max - equity) / np.where(roll_max > 0, roll_max, 1)))

        return port_ret, {
            "sharpe":       round(sharpe,     3),
            "annual_vol":   round(annual_vol, 4),
            "var_95":       round(var_95,     4),
            "cvar_95":      round(cvar_95,    4),
            "max_drawdown": round(max_dd,     4),
        }

    @staticmethod
    def _empty(cs_ranks=None, rejected=None) -> dict:
        return {
            "cs_momentum_ranks": cs_ranks or [],
            "allocations":       [],
            "rejected":          rejected or [],
            "portfolio_metrics": _zero_metrics(),
            "portfolio_returns": pd.Series(dtype=float),
        }


# ── module-level helpers ──────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, rf: float = _RF_ANNUAL) -> float:
    arr  = np.array(returns, dtype=float)
    std  = float(arr.std(ddof=1))
    if std < 1e-10 or math.isnan(std):
        return 0.0
    daily_rf = rf / _TRADING_DAYS
    raw = (float(arr.mean()) - daily_rf) / std * math.sqrt(_TRADING_DAYS)
    return float(np.clip(raw, -20.0, 20.0))


def _zero_metrics() -> dict:
    return {"sharpe": 0.0, "annual_vol": 0.0, "var_95": 0.0,
            "cvar_95": 0.0, "max_drawdown": 0.0}
