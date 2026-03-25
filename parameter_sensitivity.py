"""
ParameterSensitivity
====================
Sweeps strategy parameters to detect overfitting.

For each parameter in param_grid, each value is substituted independently
(one-at-a-time), the backtest is re-run, and the Sharpe change is measured.

A strategy is considered overfit to a parameter when:
  - Sharpe degrades > SHARPE_SENSITIVITY_FLOOR (0.30 units) from a ±30% parameter change

Usage
-----
    from parameter_sensitivity import ParameterSensitivity

    sweep = ParameterSensitivity(backtester, diagnostics_engine)
    results = sweep.run(ticker, strategy, ohlcv_df)

    # results["stable"] = True  → no parameter is overfitted
    # results["unstable_params"] → list of params with Sharpe sensitivity > floor
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd

SHARPE_SENSITIVITY_FLOOR = 0.30   # Sharpe units — reject if any param exceeds this
TRADING_DAYS = 252
RISK_FREE_RATE = 0.045


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio, capped ±20, matching DiagnosticsEngine."""
    arr = np.array(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return 0.0
    std = float(arr.std(ddof=1))
    if std < 1e-10:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    raw = float((arr.mean() - daily_rf) / std * math.sqrt(TRADING_DAYS))
    return float(np.clip(raw, -20.0, 20.0))


class ParameterSensitivity:
    """
    One-at-a-time parameter sensitivity analysis for strategy parameters.

    Parameters
    ----------
    backtester : Backtester instance
    verbose : bool
        Print progress when True.
    """

    def __init__(self, backtester: Any, verbose: bool = False):
        self.bt      = backtester
        self.verbose = verbose

    def run(
        self,
        ticker:     str,
        strategy:   dict,
        ohlcv:      pd.DataFrame,
        param_grid: dict[str, list] | None = None,
    ) -> dict:
        """
        Vary each parameter independently, re-run backtest, measure Sharpe change.

        Parameters
        ----------
        ticker : str
        strategy : dict — strategy dict as produced by StrategySelector.
            Must contain strategy["adjusted_params"] with numeric param values.
        ohlcv : pd.DataFrame — OHLCV data for the ticker.
        param_grid : dict[param_name -> list_of_values] | None
            If None, a default ±30% grid is auto-generated for all numeric params
            in strategy["adjusted_params"].

        Returns
        -------
        dict with keys:
            base_sharpe     : float — Sharpe of the original strategy
            params          : dict[param_name -> {values, sharpes, range, stable}]
            stable          : bool — True if ALL params have Sharpe sensitivity < floor
            unstable_params : list[str] — params whose sensitivity exceeds the floor
        """
        adj_params = strategy.get("adjusted_params", {})

        # Auto-generate ±30% grid when none provided
        if param_grid is None:
            param_grid = self._auto_grid(adj_params)

        if not param_grid:
            return {
                "base_sharpe": 0.0,
                "params": {},
                "stable": True,
                "unstable_params": [],
                "note": "No numeric parameters found in strategy.adjusted_params",
            }

        # Base backtest
        try:
            base_result = self.bt.run(ticker, strategy, ohlcv)
            base_sharpe = _sharpe(base_result.get("returns", pd.Series(dtype=float)))
        except Exception as e:
            return {
                "base_sharpe": 0.0,
                "params": {},
                "stable": True,
                "unstable_params": [],
                "note": f"Base backtest failed: {e}",
            }

        if self.verbose:
            print(
                f"[ParamSensitivity] {ticker} base Sharpe={base_sharpe:.3f}, "
                f"sweeping {len(param_grid)} param(s) ..."
            )

        results: dict[str, dict] = {}

        for param, values in param_grid.items():
            sharpes: list[float] = []
            for val in values:
                modified = copy.deepcopy(strategy)
                if "adjusted_params" not in modified:
                    modified["adjusted_params"] = {}
                modified["adjusted_params"][param] = val
                try:
                    bt_result  = self.bt.run(ticker, modified, ohlcv)
                    s          = _sharpe(bt_result.get("returns", pd.Series(dtype=float)))
                except Exception:
                    s = 0.0
                sharpes.append(s)

            sharpe_range = max(sharpes) - min(sharpes) if sharpes else 0.0
            stable       = sharpe_range < SHARPE_SENSITIVITY_FLOOR

            results[param] = {
                "values":       values,
                "sharpes":      [round(s, 4) for s in sharpes],
                "range":        round(sharpe_range, 4),
                "stable":       stable,
                "base_sharpe":  round(base_sharpe, 4),
                "sensitivity":  round(sharpe_range / max(abs(base_sharpe), 0.01), 4),
            }

            if self.verbose:
                icon = "✅" if stable else "⚠️"
                print(
                    f"  {icon} {param}: range={sharpe_range:.3f} "
                    f"({'stable' if stable else 'UNSTABLE'})"
                )

        unstable = [p for p, r in results.items() if not r["stable"]]
        overall_stable = len(unstable) == 0

        return {
            "base_sharpe":     round(base_sharpe, 4),
            "params":          results,
            "stable":          overall_stable,
            "unstable_params": unstable,
        }

    @staticmethod
    def _auto_grid(adj_params: dict, steps: int = 5, pct: float = 0.30) -> dict[str, list]:
        """
        Generate a ±30% variation grid for each numeric parameter.
        Produces `steps` evenly-spaced values from base*(1-pct) to base*(1+pct).
        Non-numeric or zero-value params are skipped.
        """
        grid: dict[str, list] = {}
        for name, base_val in adj_params.items():
            if not isinstance(base_val, (int, float)):
                continue
            if base_val == 0:
                continue
            lo  = base_val * (1.0 - pct)
            hi  = base_val * (1.0 + pct)
            vals = [round(float(lo + i * (hi - lo) / (steps - 1)), 6) for i in range(steps)]
            # Preserve int type when the original value was an int
            if isinstance(base_val, int):
                vals = [max(1, int(round(v))) for v in vals]
                vals = sorted(set(vals))   # deduplicate after rounding to int
            grid[name] = vals
        return grid
