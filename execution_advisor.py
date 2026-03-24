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

import datetime
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf

# ── market hours ──────────────────────────────────────────────────────────────

_ET            = ZoneInfo("America/New_York")
_MARKET_OPEN   = datetime.time(9, 30)
_MARKET_CLOSE  = datetime.time(16, 0)


def _format_td(td: datetime.timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m = rem // 60
    return f"{h}h {m}m" if h else f"{m}m"


def get_market_status() -> dict:
    """Return NYSE open/closed status with human-readable time context."""
    now = datetime.datetime.now(_ET)
    wd  = now.weekday()   # 0=Mon … 6=Sun
    t   = now.time()

    if wd >= 5:
        days = 7 - wd          # days until next Monday
        nxt  = (now + datetime.timedelta(days=days)).replace(hour=9, minute=30, second=0, microsecond=0)
        return {"open": False, "label": "CLOSED — weekend",
                "detail": f"opens Monday 09:30 ET (in {_format_td(nxt - now)})"}

    if t < _MARKET_OPEN:
        nxt = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return {"open": False, "label": "CLOSED — pre-market",
                "detail": f"opens today 09:30 ET (in {_format_td(nxt - now)})"}

    if t >= _MARKET_CLOSE:
        days = 1 if wd < 4 else (7 - wd)
        nxt  = (now + datetime.timedelta(days=days)).replace(hour=9, minute=30, second=0, microsecond=0)
        label = "tomorrow" if wd < 4 else "Monday"
        return {"open": False, "label": "CLOSED — after-hours",
                "detail": f"opens {label} 09:30 ET (in {_format_td(nxt - now)})"}

    closes = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return {"open": True, "label": "OPEN",
            "detail": f"closes 16:00 ET (in {_format_td(closes - now)})"}


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
        """
        Build execution briefs for ALL strategies passed in.

        Active signals  → full live/ATR slippage estimate, market impact check.
        Inactive signals → projected setup (conditions not yet met) + same
                           slippage/impact estimates so the trader knows exactly
                           what to expect and what to watch for before entry.
        """
        active_briefs:   list[dict] = []
        inactive_briefs: list[dict] = []
        warnings:        list[str]  = []

        mkt = get_market_status()
        print(f"  [Execution] NYSE {mkt['label']} — {mkt['detail']}")
        if not mkt["open"]:
            warnings.append(
                f"NYSE {mkt['label']} ({mkt['detail']}) — "
                "live quotes unavailable; all slippage estimates are ATR-based"
            )

        for strat in strategies:
            sig = strat.get("current_signal", {})
            is_active = bool(sig.get("signal_active"))
            brief, strat_warnings = self._build_brief(
                strat, market_open=mkt["open"], signal_active=is_active
            )
            if is_active:
                active_briefs.append(brief)
            else:
                inactive_briefs.append(brief)
            warnings.extend(strat_warnings)

        total_risk = sum(s["adjusted_risk"] for s in active_briefs)
        pct        = (total_risk / self._portfolio * 100) if self._portfolio else 0.0

        return {
            "market_status":   mkt,
            "active_signals":  active_briefs,
            "pending_signals": inactive_briefs,
            "inactive_count":  len(inactive_briefs),
            "portfolio_risk": {
                "active_count":      len(active_briefs),
                "total_dollar_risk": total_risk,
                "pct_of_portfolio":  pct,
            },
            "warnings": warnings,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_brief(self, strat: dict, market_open: bool = True,
                     signal_active: bool = False) -> tuple[dict, list[str]]:
        """
        Build an execution brief regardless of signal state.

        When signal_active is True:  use 'setup' (exact current-bar entry).
        When signal_active is False: use 'projected_setup' (entry at trigger price)
                                     so the trader knows what the trade looks like
                                     when (not if) the signal fires.
        """
        ticker  = strat["ticker"]
        sig     = strat.get("current_signal", {})

        # Active signal: use the exact setup; inactive: use projected setup
        if signal_active:
            setup = sig.get("setup") or {}
        else:
            setup = sig.get("projected_setup") or sig.get("setup") or {}

        position_size = int(setup.get("position_size", 0))
        dollar_risk   = float(setup.get("dollar_risk", 0.0))
        current_atr   = float(setup.get("current_atr", 0.0))
        adv           = strat.get("_adv", 0)

        warnings: list[str] = []

        # ── live bid/ask (skip when market is closed) ─────────────────────────
        if not market_open:
            bid, ask, spread = None, None, None
            slippage_per_share = current_atr * _ATR_SLIPPAGE_FACTOR
            live_ok = False
        else:
            bid, ask, spread, slippage_per_share, live_ok = self._fetch_spread(ticker, current_atr)

        if not live_ok and market_open:
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
        note = self._execution_note(
            market_impact, live_ok, spread, market_open=market_open,
            signal_active=signal_active
        )

        return {
            "ticker":              ticker,
            "signal_active":       signal_active,
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
            "entry_trigger":       setup.get("entry_trigger"),   # price to watch
            "volume_needed":       setup.get("volume_needed"),   # volume threshold
            "rsi_needed":          setup.get("rsi_needed"),      # RSI threshold (MR)
            "target":              setup.get("target"),          # MR target
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
    def _execution_note(market_impact: str, live_ok: bool, spread: float | None,
                        market_open: bool = True, signal_active: bool = True) -> str:
        parts = []
        if not signal_active:
            parts.append("PENDING — conditions not yet met; use this setup when signal fires")
        if not market_open:
            parts.append("NYSE closed — queue order for next open; slippage is ATR-estimated")
        elif not live_ok:
            parts.append("ATR-based slippage estimate (no live quote)")
        elif spread is not None:
            parts.append(f"Live spread ${spread:.4f}")

        if market_impact == "moderate":
            parts.append("consider limit orders to reduce market impact")
        elif market_impact == "significant":
            parts.append("split order / use VWAP; significant market impact expected")

        return "; ".join(parts) if parts else "Use market order at open"
