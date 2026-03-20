"""
ExecutionAdvisor
================
Fetches live bid/ask data from yfinance for active signals and produces
execution-ready trade briefs with slippage estimates and market impact.

Public interface
----------------
  advisor = ExecutionAdvisor(initial_portfolio=100_000.0)
  brief   = advisor.advise(strategies)   # strategies list from pipeline

Output schema
-------------
{
  "active_signals": [
    {
      "ticker":         str,
      "bid":            float | None,
      "ask":            float | None,
      "spread":         float | None,          # ask - bid
      "slippage_per_share": float,
      "slippage_total": float,                 # per_share × position_size
      "adjusted_risk":  float,                 # dollar_risk + slippage_total
      "market_impact":  "negligible"|"moderate"|"significant",
      "execution_note": str,
      "entry_price":    float,
      "stop_price":     float,
      "position_size":  int,
      "dollar_risk":    float,
    }, ...
  ],
  "inactive_count":  int,
  "portfolio_risk": {
    "active_count":       int,
    "total_dollar_risk":  float,
    "pct_of_portfolio":   float,
  },
  "warnings": [str, ...]
}
"""

from __future__ import annotations

from typing import Any

import yfinance as yf


# ── market impact thresholds ──────────────────────────────────────────────────

_MODERATE_ADV_PCT   = 0.01   # 1 % of ADV
_SIGNIFICANT_ADV_PCT = 0.05  # 5 % of ADV

# Corwin-Schultz ATR proxy when live data unavailable
_ATR_SLIPPAGE_FACTOR = 0.10


class ExecutionAdvisor:
    def __init__(self, initial_portfolio: float = 100_000.0):
        self._portfolio = initial_portfolio

    # ── public ────────────────────────────────────────────────────────────────

    def advise(self, strategies: list[dict]) -> dict:
        active_signals: list[dict] = []
        inactive_count = 0
        warnings:       list[str]  = []

        for strat in strategies:
            sig = strat.get("current_signal", {})
            if not sig.get("signal_active"):
                inactive_count += 1
                continue

            brief, strat_warnings = self._build_brief(strat)
            active_signals.append(brief)
            warnings.extend(strat_warnings)

        total_risk = sum(s["adjusted_risk"] for s in active_signals)
        pct        = (total_risk / self._portfolio * 100) if self._portfolio else 0.0

        return {
            "active_signals": active_signals,
            "inactive_count": inactive_count,
            "portfolio_risk": {
                "active_count":      len(active_signals),
                "total_dollar_risk": total_risk,
                "pct_of_portfolio":  pct,
            },
            "warnings": warnings,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_brief(self, strat: dict) -> tuple[dict, list[str]]:
        ticker  = strat["ticker"]
        sig     = strat["current_signal"]
        setup   = sig.get("setup") or {}

        position_size = int(setup.get("position_size", 0))
        dollar_risk   = float(setup.get("dollar_risk", 0.0))
        current_atr   = float(setup.get("current_atr", 0.0))
        adv           = strat.get("_adv", 0)   # injected by tests; 0 means unknown

        warnings: list[str] = []

        # ── live bid/ask ──────────────────────────────────────────────────────
        bid, ask, spread, slippage_per_share, live_ok = self._fetch_spread(ticker, current_atr)

        if not live_ok:
            warnings.append(
                f"{ticker}: live bid/ask unavailable — using ATR-based slippage estimate"
            )

        slippage_total = slippage_per_share * position_size
        adjusted_risk  = dollar_risk + slippage_total

        # ── market impact ─────────────────────────────────────────────────────
        market_impact = self._classify_impact(position_size, adv)
        if market_impact == "significant":
            warnings.append(
                f"{ticker}: order size ({position_size:,} shares) is >{_SIGNIFICANT_ADV_PCT:.0%} of ADV — "
                "consider splitting or using VWAP"
            )

        # ── execution note ────────────────────────────────────────────────────
        note = self._execution_note(market_impact, live_ok, spread)

        return {
            "ticker":              ticker,
            "bid":                 bid,
            "ask":                 ask,
            "spread":              spread,
            "slippage_per_share":  slippage_per_share,
            "slippage_total":      slippage_total,
            "adjusted_risk":       adjusted_risk,
            "market_impact":       market_impact,
            "execution_note":      note,
            "entry_price":         setup.get("entry_price"),
            "stop_price":          setup.get("stop_price"),
            "position_size":       position_size,
            "dollar_risk":         dollar_risk,
        }, warnings

    def _fetch_spread(
        self, ticker: str, current_atr: float
    ) -> tuple[float | None, float | None, float | None, float, bool]:
        """
        Returns (bid, ask, spread, slippage_per_share, live_ok).
        Falls back to ATR proxy when yfinance fails.
        """
        try:
            info = yf.Ticker(ticker).info
            bid  = float(info.get("bid") or 0)
            ask  = float(info.get("ask") or 0)
            if bid > 0 and ask > 0:
                spread              = ask - bid
                slippage_per_share  = spread / 2
                return bid, ask, spread, slippage_per_share, True
        except Exception:
            pass

        # fallback
        slippage_per_share = current_atr * _ATR_SLIPPAGE_FACTOR
        return None, None, None, slippage_per_share, False

    @staticmethod
    def _classify_impact(position_size: int, adv: int) -> str:
        if adv <= 0:
            return "negligible"
        pct = position_size / adv
        if pct < _MODERATE_ADV_PCT:
            return "negligible"
        if pct < _SIGNIFICANT_ADV_PCT:
            return "moderate"
        return "significant"

    @staticmethod
    def _execution_note(market_impact: str, live_ok: bool, spread: float | None) -> str:
        parts = []
        if not live_ok:
            parts.append("ATR-based slippage estimate (no live quote)")
        elif spread is not None:
            parts.append(f"Live spread ${spread:.4f}")

        if market_impact == "moderate":
            parts.append("consider limit orders to reduce market impact")
        elif market_impact == "significant":
            parts.append("split order / use VWAP; significant market impact expected")

        return "; ".join(parts) if parts else "Use market order at open"
