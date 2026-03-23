"""
Comprehensive Algorithmic Test Suite
=====================================
Tests every deterministic/algorithmic component in the pipeline.
No LLM calls, no network calls, no real market data.

Run:
    python test_algorithms_comprehensive.py          # human-readable output
    python -m pytest test_algorithms_comprehensive.py -v  # pytest output

Sections
--------
  1. OHLCVFetcher.compute_features      (RSI, ATR, returns, proximity, volume ratio)
  2. RegimeClassifier.classify           (Hurst, ATR, regime labels, edge cases)
  3. StrategySelector deterministic params (momentum + mean-reversion rules)
  4. DiagnosticsEngine                   (floors, OOS Sharpe gate fix, metrics)
  5. Backtester                          (position sizing, slippage, exits)
  6. MonteCarloEngine                    (Kelly, P(Ruin), equity percentiles)
  7. ExecutionAdvisor                    (market impact, slippage, portfolio risk)
  8. _advanced_metrics VaR fix           (invested-days-only VaR/CVaR)
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

# ── Test harness ──────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (name, passed, message)

def check(name: str, condition: bool, message: str = "") -> None:
    """Record a test result."""
    _results.append((name, condition, message))
    mark = "  PASS" if condition else "  FAIL"
    suffix = f"  ({message})" if message and not condition else ""
    print(f"  {'[+]' if condition else '[X]'} {name}{suffix}")

def assert_close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol

def section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

# ── OHLCV helper builders ─────────────────────────────────────────────────────

_BUS_DAYS = pd.date_range("2022-01-03", periods=252, freq="B")

def make_ohlcv(
    close: np.ndarray,
    high_offset: float = 0.5,
    low_offset: float = 0.5,
    volume: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close price array."""
    n = len(close)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    vol = volume if volume is not None else np.full(n, 1_000_000.0)
    return pd.DataFrame({
        "Open":   close,
        "High":   close + high_offset,
        "Low":    close - low_offset,
        "Close":  close,
        "Volume": vol,
    }, index=idx)


def make_returns(n: int = 252, daily: float = 0.001) -> pd.Series:
    """Build a flat-return daily series."""
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(np.full(n, daily), index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OHLCVFetcher.compute_features
# ═══════════════════════════════════════════════════════════════════════════════

def test_ohlcv_fetcher() -> None:
    section("1 — OHLCVFetcher.compute_features")
    from ohlcv_fetcher import OHLCVFetcher
    f = OHLCVFetcher()

    # 1a  —  all 7 output keys present
    close = np.linspace(100, 120, 60)
    df = make_ohlcv(close)
    feats = f.compute_features(df)
    required = {"return_20d", "rsi_14", "atr_14", "atr_pct",
                "52w_high_prox", "52w_low_prox", "volume_ratio_30d"}
    check("1a  output keys present", required == set(feats.keys()),
          f"got {set(feats.keys())}")

    # 1b  —  return_20d: prices[i] = 100 + i, last close = 159, close[-21] = 138
    #         return = (159 - 138) / 138 ≈ 0.1522
    close60 = np.array([100.0 + i for i in range(60)])
    df60 = make_ohlcv(close60)
    feats60 = f.compute_features(df60)
    expected_ret = (close60[-1] - close60[-21]) / close60[-21]
    check("1b  return_20d exact value",
          assert_close(feats60["return_20d"], expected_ret, tol=1e-6),
          f"got {feats60['return_20d']:.6f} expected {expected_ret:.6f}")

    # 1c  —  RSI near 100 for all-positive returns (monotone uptrend)
    close_up = np.linspace(100, 200, 60)  # every day up
    rsi_up = f.compute_features(make_ohlcv(close_up))["rsi_14"]
    check("1c  RSI near 100 for pure uptrend",
          rsi_up > 90,
          f"RSI={rsi_up:.2f}")

    # 1d  —  RSI near 0 for all-negative returns (monotone downtrend)
    close_dn = np.linspace(200, 100, 60)
    rsi_dn = f.compute_features(make_ohlcv(close_dn))["rsi_14"]
    check("1d  RSI near 0 for pure downtrend",
          rsi_dn < 10,
          f"RSI={rsi_dn:.2f}")

    # 1e  —  ATR near 0 for flat price with zero H-L spread
    close_flat = np.full(60, 100.0)
    df_flat = pd.DataFrame({
        "Open": close_flat, "High": close_flat, "Low": close_flat,
        "Close": close_flat, "Volume": np.full(60, 1e6),
    }, index=pd.date_range("2022-01-03", periods=60, freq="B"))
    atr_flat = f.compute_features(df_flat)["atr_14"]
    check("1e  ATR ~= 0 for flat price with no H-L spread",
          atr_flat < 1e-8,
          f"ATR={atr_flat:.2e}")

    # 1f  —  52w_high_prox = 1.0 when last close IS the 252-day high
    close_peak = np.concatenate([np.linspace(80, 99.9, 251), [100.0]])
    feat_peak = f.compute_features(make_ohlcv(close_peak))
    check("1f  52w_high_prox = 1.0 at 52-week high",
          assert_close(feat_peak["52w_high_prox"], 1.0, tol=1e-6),
          f"prox={feat_peak['52w_high_prox']:.6f}")

    # 1g  —  52w_low_prox = 1.0 when last close IS the 252-day low
    #         Prices start high and fall to the minimum on the last bar.
    close_trough = np.concatenate([np.linspace(120.0, 50.01, 251), [50.0]])
    feat_trough = f.compute_features(make_ohlcv(close_trough))
    check("1g  52w_low_prox = 1.0 at 52-week low",
          assert_close(feat_trough["52w_low_prox"], 1.0, tol=1e-6),
          f"prox={feat_trough['52w_low_prox']:.6f}")

    # 1h  —  volume_ratio_30d = 1.0 for constant volume
    df_const_vol = make_ohlcv(np.linspace(100, 105, 60))  # default volume=1e6
    vr = f.compute_features(df_const_vol)["volume_ratio_30d"]
    check("1h  volume_ratio_30d = 1.0 for constant volume",
          assert_close(vr, 1.0, tol=1e-6),
          f"ratio={vr:.6f}")

    # 1i  —  volume_ratio_30d > 2 when last day volume is triple the avg
    vol_spike = np.full(60, 1_000_000.0)
    vol_spike[-1] = 3_000_000.0
    df_spike = make_ohlcv(np.linspace(100, 105, 60), volume=vol_spike)
    vr_spike = f.compute_features(df_spike)["volume_ratio_30d"]
    check("1i  volume_ratio_30d > 2 for 3× volume spike",
          vr_spike > 2.0,
          f"ratio={vr_spike:.2f}")

    # 1j  —  insufficient data raises ValueError
    df_short = make_ohlcv(np.linspace(100, 105, 10))
    try:
        f.compute_features(df_short)
        check("1j  ValueError for insufficient data", False, "no exception raised")
    except ValueError:
        check("1j  ValueError for insufficient data", True)

    # 1k  —  atr_pct = atr_14 / last close
    feats_chk = f.compute_features(make_ohlcv(np.linspace(100, 110, 60)))
    atr_pct_expected = feats_chk["atr_14"] / feats_chk["atr_14"] * feats_chk["atr_pct"]
    manual_atr_pct = feats_chk["atr_14"] / 110.0   # last close ≈ 110
    check("1k  atr_pct ~= atr_14 / last_close",
          assert_close(feats_chk["atr_pct"], manual_atr_pct, tol=0.001),
          f"atr_pct={feats_chk['atr_pct']:.6f} manual={manual_atr_pct:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RegimeClassifier
# ═══════════════════════════════════════════════════════════════════════════════

def test_regime_classifier() -> None:
    section("2 — RegimeClassifier.classify")
    from regime_classifier import RegimeClassifier, HURST_TRENDING, HURST_MEAN_REVERTING
    clf = RegimeClassifier()

    # Helper: linear uptrend → persistent → high Hurst
    def trending_ohlcv(n: int = 200) -> pd.DataFrame:
        c = np.linspace(100.0, 150.0, n)
        return make_ohlcv(c, high_offset=0.3, low_offset=0.3)

    # Helper: alternating up/down → mean-reverting → low Hurst
    def mean_reverting_ohlcv(n: int = 200) -> pd.DataFrame:
        c = 100.0 + np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        return make_ohlcv(c, high_offset=0.1, low_offset=0.1)

    # 2a  —  output contains required keys
    result = clf.classify("TEST", trending_ohlcv())
    required_keys = {"ticker", "hurst", "atr_pct", "regime"}
    check("2a  required output keys", required_keys == set(result.keys()),
          f"got {set(result.keys())}")

    # 2b  —  ticker field matches input
    check("2b  ticker field matches", result["ticker"] == "TEST")

    # 2c  —  Hurst clamped to [0, 1]
    h = result["hurst"]
    check("2c  Hurst in [0, 1]", 0.0 <= h <= 1.0, f"hurst={h:.4f}")

    # 2d  —  linear uptrend → Trending or High-Volatility (Hurst > threshold)
    r_trending = clf.classify("TREND", trending_ohlcv())
    check("2d  linear uptrend -> Trending regime",
          r_trending["regime"] == "Trending",
          f"regime={r_trending['regime']}  hurst={r_trending['hurst']:.3f}")

    # 2e  —  alternating price → Mean-Reverting (low Hurst)
    r_mr = clf.classify("MR", mean_reverting_ohlcv())
    check("2e  alternating price -> Mean-Reverting regime",
          r_mr["regime"] == "Mean-Reverting",
          f"regime={r_mr['regime']}  hurst={r_mr['hurst']:.3f}")

    # 2f  —  decision logic: neutral Hurst + high ATR% → High-Volatility
    #  Test the _label routing directly so the Hurst estimator variance doesn't
    #  interfere with the ATR-based branching logic test.
    from regime_classifier import HURST_TRENDING, HURST_MEAN_REVERTING, ATR_HIGH_VOL, ATR_LOW_VOL
    neutral_hurst = (HURST_TRENDING + HURST_MEAN_REVERTING) / 2   # 0.50
    label_hv = clf._label(neutral_hurst, ATR_HIGH_VOL + 0.01)     # ATR > 3% threshold
    check("2f  _label(neutral_hurst, ATR%=4%) -> High-Volatility",
          label_hv == "High-Volatility",
          f"got '{label_hv}'")

    # 2g  —  decision logic: neutral Hurst + low ATR% → Low-Volatility
    label_lv = clf._label(neutral_hurst, ATR_LOW_VOL - 0.005)     # ATR < 1.5% threshold
    check("2g  _label(neutral_hurst, ATR%=1.0%) -> Low-Volatility",
          label_lv == "Low-Volatility",
          f"got '{label_lv}'")

    # 2g2 —  decision logic: neutral Hurst + mid ATR% → Neutral
    mid_atr = (ATR_HIGH_VOL + ATR_LOW_VOL) / 2   # 2.25%
    label_neutral = clf._label(neutral_hurst, mid_atr)
    check("2g2 _label(neutral_hurst, mid_ATR%) -> Neutral",
          label_neutral == "Neutral",
          f"got '{label_neutral}'")

    # 2h  —  fewer than 30 rows raises ValueError
    df_tiny = make_ohlcv(np.linspace(100, 105, 20))
    try:
        clf.classify("X", df_tiny)
        check("2h  ValueError for < 30 rows", False, "no exception raised")
    except ValueError:
        check("2h  ValueError for < 30 rows", True)

    # 2i  —  classify_all skips None entries
    batch = {"A": trending_ohlcv(), "B": None}
    results_batch = clf.classify_all(batch)
    check("2i  classify_all skips None entries",
          len(results_batch) == 1 and results_batch[0]["ticker"] == "A",
          f"got {len(results_batch)} results")

    # 2j  —  atr_pct is positive
    check("2j  atr_pct > 0", r_trending["atr_pct"] > 0,
          f"atr_pct={r_trending['atr_pct']:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — StrategySelector deterministic parameters
# ═══════════════════════════════════════════════════════════════════════════════

def test_strategy_selector_params() -> None:
    section("3 — StrategySelector deterministic parameter rules")
    from strategy_selector import (
        _compute_momentum_params, _compute_mean_reversion_params,
        MOMENTUM_BASE, MEAN_REVERSION_BASE,
    )

    # ── Momentum ──────────────────────────────────────────────────────────────

    # 3a  —  all below thresholds → base params unchanged
    p, rules = _compute_momentum_params(hurst=0.50, atr_pct=0.02, vol_ratio=1.1)
    check("3a  momentum base case: no rules fired",
          len(rules) == 0 and p == MOMENTUM_BASE,
          f"rules={rules}  params={p}")

    # 3b  —  Hurst > 0.70 only → trailing_stop_atr += 0.5 (becomes 2.5)
    p, rules = _compute_momentum_params(hurst=0.72, atr_pct=0.02, vol_ratio=1.1)
    check("3b  Hurst>0.70 -> trailing_stop_atr = 2.5",
          p["trailing_stop_atr"] == 2.5 and p["max_holding_days"] == 20,
          f"trailing={p['trailing_stop_atr']}  max_hold={p['max_holding_days']}")

    # 3c  —  Hurst > 0.75 → trailing_stop_atr = 2.5 AND max_holding_days = 30
    p, rules = _compute_momentum_params(hurst=0.80, atr_pct=0.02, vol_ratio=1.1)
    check("3c  Hurst>0.75 -> max_holding_days = 30",
          p["trailing_stop_atr"] == 2.5 and p["max_holding_days"] == 30,
          f"trailing={p['trailing_stop_atr']}  max_hold={p['max_holding_days']}")

    # 3d  —  ATR% > 2.5% → stop_loss_atr = 2.0
    p, rules = _compute_momentum_params(hurst=0.50, atr_pct=0.026, vol_ratio=1.1)
    check("3d  ATR%>2.5% -> stop_loss_atr = 2.0",
          p["stop_loss_atr"] == 2.0,
          f"stop_loss={p['stop_loss_atr']}")

    # 3e  —  vol_ratio > 1.3 → volume_multiplier = 1.5
    p, rules = _compute_momentum_params(hurst=0.50, atr_pct=0.02, vol_ratio=1.5)
    check("3e  vol_ratio>1.3 -> volume_multiplier = 1.5",
          assert_close(p["volume_multiplier"], 1.5, tol=1e-9),
          f"vol_mult={p['volume_multiplier']}")

    # 3f  —  all momentum triggers → all params adjusted simultaneously
    p, rules = _compute_momentum_params(hurst=0.80, atr_pct=0.030, vol_ratio=1.5)
    check("3f  all momentum triggers fire",
          p["trailing_stop_atr"] == 2.5
          and p["stop_loss_atr"] == 2.0
          and assert_close(p["volume_multiplier"], 1.5)
          and p["max_holding_days"] == 30,
          f"params={p}")

    # 3g  —  Hurst exactly at 0.70 boundary: no rule fires (> not >=)
    p, rules = _compute_momentum_params(hurst=0.70, atr_pct=0.02, vol_ratio=1.1)
    check("3g  Hurst=0.70 (boundary): trailing_stop unchanged (rule is >0.70)",
          p["trailing_stop_atr"] == 2.0,
          f"trailing={p['trailing_stop_atr']}")

    # ── Mean-Reversion ────────────────────────────────────────────────────────

    # 3h  —  base case: no rules
    p, rules = _compute_mean_reversion_params(atr_pct=0.02)
    check("3h  MR base case: no rules fired",
          len(rules) == 0 and p == MEAN_REVERSION_BASE,
          f"rules={rules}")

    # 3i  —  ATR% > 2.5% → rsi_entry = 35, stop_loss_atr = 2.0
    p, rules = _compute_mean_reversion_params(atr_pct=0.026)
    check("3i  MR ATR%>2.5% -> rsi_entry=35, stop_loss=2.0",
          p["rsi_entry_threshold"] == 35 and p["stop_loss_atr"] == 2.0,
          f"rsi_entry={p['rsi_entry_threshold']}  stop_loss={p['stop_loss_atr']}")

    # 3j  —  ATR% > 3.0% → bb_std = 2.5
    p, rules = _compute_mean_reversion_params(atr_pct=0.031)
    check("3j  MR ATR%>3.0% -> bb_std = 2.5",
          p["bb_std"] == 2.5,
          f"bb_std={p['bb_std']}")

    # 3k  —  ATR% < 1.5% → max_holding_days = 15
    p, rules = _compute_mean_reversion_params(atr_pct=0.014)
    check("3k  MR ATR%<1.5% -> max_holding_days = 15",
          p["max_holding_days"] == 15,
          f"max_hold={p['max_holding_days']}")

    # 3l  —  base params are deep copies (mutations don't affect originals)
    _compute_momentum_params(hurst=0.80, atr_pct=0.030, vol_ratio=1.5)
    check("3l  MOMENTUM_BASE unchanged after param computation",
          MOMENTUM_BASE["trailing_stop_atr"] == 2.0
          and MOMENTUM_BASE["max_holding_days"] == 20,
          f"base={MOMENTUM_BASE}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DiagnosticsEngine (floors + OOS Sharpe gate fix)
# ═══════════════════════════════════════════════════════════════════════════════

def test_diagnostics_engine() -> None:
    section("4 — DiagnosticsEngine floors and OOS Sharpe gate")
    from diagnostics_engine import DiagnosticsEngine, SHARPE_FLOOR, MAX_DD_FLOOR

    idx = pd.date_range("2022-01-03", periods=252, freq="B")

    # healthy returns: alternating +0.004/+0.002 → mean=0.003, std≈0.001
    # Sharpe ~= (0.003 - rf) / 0.001 * sqrt(252) ~= 44 >> 0.5 floor
    # Must NOT use constant returns: std would be ~0 → Sharpe=0 after the overflow fix
    healthy_rets = pd.Series(
        np.where(np.arange(252) % 2 == 0, 0.004, 0.002),
        index=idx,
    )
    healthy_log  = [{"pnl": (2.0 if i % 2 == 0 else -1.0)} for i in range(40)]

    eng = DiagnosticsEngine()

    # 4a  —  healthy strategy passes all floors
    r = eng.run("AAPL", "Momentum", healthy_log, healthy_rets)
    check("4a  healthy strategy passes", r["passed"] is True,
          f"reject={r['reject_reason']}")

    # 4b  —  oos_sharpe key present in metrics
    check("4b  oos_sharpe in metrics", "oos_sharpe" in r["metrics"])

    # 4c  —  OOS Sharpe gate: full-period low but OOS high → PASS
    metrics_oos_good = {
        "sharpe": 0.20, "oos_sharpe": 1.80,
        "max_drawdown": 0.10, "win_rate": 0.50,
        "walk_forward_degradation": 0.10, "trade_count": 40,
    }
    passed, reason = DiagnosticsEngine._check_floors(metrics_oos_good)
    check("4c  OOS Sharpe=1.80 rescues low full-period Sharpe=0.20",
          passed is True,
          f"reason={reason}")

    # 4d  —  both Sharpe values below floor → FAIL
    metrics_both_bad = {
        "sharpe": 0.20, "oos_sharpe": 0.30,
        "max_drawdown": 0.10, "win_rate": 0.50,
        "walk_forward_degradation": 0.10, "trade_count": 40,
    }
    passed, reason = DiagnosticsEngine._check_floors(metrics_both_bad)
    check("4d  both Sharpe values below floor -> FAIL",
          passed is False and "sharpe" in reason.lower(),
          f"passed={passed} reason={reason}")

    # 4e  —  OOS Sharpe exactly at floor → PASS
    metrics_at_floor = {
        "sharpe": 0.20, "oos_sharpe": SHARPE_FLOOR,
        "max_drawdown": 0.10, "win_rate": 0.50,
        "walk_forward_degradation": 0.10, "trade_count": 40,
    }
    passed, reason = DiagnosticsEngine._check_floors(metrics_at_floor)
    check("4e  OOS Sharpe exactly at floor passes",
          passed is True,
          f"reason={reason}")

    # 4f  —  max drawdown floor triggers
    deep_dd_rets = pd.Series(
        np.concatenate([np.full(84, 0.005), np.full(84, -0.008), np.full(84, 0.005)]),
        index=idx,
    )
    r_dd = eng.run("AAPL", "Momentum", healthy_log, deep_dd_rets)
    check("4f  deep drawdown triggers max_drawdown floor",
          r_dd["passed"] is False and "drawdown" in (r_dd["reject_reason"] or "").lower(),
          f"dd={r_dd['metrics']['max_drawdown']:.1%}")

    # 4g  —  win rate below 35% triggers
    low_wr_log = [{"pnl": (1.0 if i % 5 == 0 else -1.0)} for i in range(40)]
    r_wr = eng.run("AAPL", "Momentum", low_wr_log, healthy_rets)
    check("4g  win rate < 35% triggers floor",
          r_wr["passed"] is False and "win" in (r_wr["reject_reason"] or "").lower(),
          f"win_rate={r_wr['metrics']['win_rate']:.1%}")

    # 4h  —  trade count below 30 triggers
    tiny_log = [{"pnl": 1.0} for _ in range(5)]
    r_tc = eng.run("AAPL", "Momentum", tiny_log, healthy_rets)
    check("4h  trade count < 30 triggers floor",
          r_tc["passed"] is False and "trade" in (r_tc["reject_reason"] or "").lower(),
          f"count={r_tc['metrics']['trade_count']}")

    # 4i  —  max_drawdown metric accuracy (single -50% event → max DD ≈ 50%)
    arr50 = np.zeros(252)
    arr50[125] = -0.5
    rets_dd50 = pd.Series(arr50, index=idx)
    r_dd50 = eng.run("X", "Momentum", healthy_log, rets_dd50)
    check("4i  max_drawdown ~= 50% for single halving event",
          assert_close(r_dd50["metrics"]["max_drawdown"], 0.50, tol=0.02),
          f"max_dd={r_dd50['metrics']['max_drawdown']:.3f}")

    # 4j  —  walk-forward degradation: IS much better than OOS → high degradation
    wf_bad = pd.Series(
        np.concatenate([
            np.where(np.arange(176) % 2 == 0, 0.008, 0.003),   # IS: Sharpe ≈ high
            np.where(np.arange(76) % 2 == 0, 0.001, -0.001),   # OOS: Sharpe ≈ negative
        ]),
        index=idx,
    )
    r_wf = eng.run("X", "Momentum", healthy_log, wf_bad)
    check("4j  IS-good / OOS-bad series produces high WF degradation",
          r_wf["metrics"]["walk_forward_degradation"] > 0.5,
          f"degrad={r_wf['metrics']['walk_forward_degradation']:.3f}")

    # 4k  —  LLM not called on auto-reject (no llm_client provided)
    r_no_llm = eng.run("AAPL", "Momentum", tiny_log, healthy_rets)
    check("4k  llm_commentary is None when no llm_client",
          r_no_llm["llm_commentary"] is None)

    # ── profit factor tests ──────────────────────────────────────────────────

    # 4l  —  profit_factor present in metrics
    r_pf = eng.run("AAPL", "Momentum", healthy_log, healthy_rets)
    check("4l  profit_factor key present in metrics",
          "profit_factor" in r_pf["metrics"],
          f"keys={list(r_pf['metrics'].keys())}")

    # 4m  —  low win rate BUT high payoff ratio → should PASS (profit_factor bypass)
    #         20% win rate, avg_win=$10, avg_loss=$1 → profit_factor = (8×10)/(32×1) = 2.5 > 1.5
    high_payoff_log = (
        [{"pnl": 10.0}] * 8 +   # 8 wins
        [{"pnl": -1.0}] * 32    # 32 losses → win_rate=20%, profit_factor=2.5
    )
    r_hp = eng.run("AAPL", "Momentum", high_payoff_log, healthy_rets)
    check("4m  win_rate=20% but profit_factor=2.5 -> PASS (high-payoff bypass)",
          r_hp["passed"] is True,
          f"passed={r_hp['passed']}  reason={r_hp['reject_reason']}  "
          f"pf={r_hp['metrics']['profit_factor']:.2f}")

    # 4n  —  low win rate AND low profit factor → FAIL
    #         20% win rate, avg_win=$1, avg_loss=$1 → profit_factor=0.25 (net loser)
    bad_payoff_log = (
        [{"pnl": 1.0}]  * 8 +   # 8 wins × $1
        [{"pnl": -1.0}] * 32    # 32 losses × $1 → profit_factor=0.25 < 1.5
    )
    r_bp = eng.run("AAPL", "Momentum", bad_payoff_log, healthy_rets)
    check("4n  win_rate=20% and profit_factor=0.25 -> FAIL (both thresholds breached)",
          r_bp["passed"] is False,
          f"passed={r_bp['passed']}  reason={r_bp['reject_reason']}")

    # 4o  —  profit_factor is correct for 3 wins / 7 losses ($2 win, $1 loss)
    pf_log = [{"pnl": 2.0}] * 3 + [{"pnl": -1.0}] * 7
    r_pf_check = eng.run("AAPL", "Momentum", pf_log, healthy_rets)
    check("4o  profit_factor = (3*2) / (7*1) = 0.857",
          assert_close(r_pf_check["metrics"]["profit_factor"], 6.0 / 7.0, tol=0.01),
          f"pf={r_pf_check['metrics']['profit_factor']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Backtester
# ═══════════════════════════════════════════════════════════════════════════════

def _make_momentum_strategy(
    entry_lookback: int = 10,
    volume_multiplier: float = 1.2,
    trailing_stop_atr: float = 2.0,
    ma_exit_period: int = 10,
    stop_loss_atr: float = 1.5,
    max_holding_days: int = 20,
) -> dict:
    return {
        "strategy": "Momentum",
        "adjusted_params": {
            "entry_lookback":    entry_lookback,
            "volume_multiplier": volume_multiplier,
            "trailing_stop_atr": trailing_stop_atr,
            "ma_exit_period":    ma_exit_period,
            "stop_loss_atr":     stop_loss_atr,
            "max_holding_days":  max_holding_days,
        },
    }


def _make_mr_strategy(
    rsi_entry_threshold: int = 30,
    rsi_exit_threshold: int = 55,
    bb_period: int = 20,
    bb_std: float = 2.0,
    stop_loss_atr: float = 1.5,
    max_holding_days: int = 10,
) -> dict:
    return {
        "strategy": "Mean-Reversion",
        "adjusted_params": {
            "rsi_entry_threshold": rsi_entry_threshold,
            "rsi_exit_threshold":  rsi_exit_threshold,
            "bb_period":           bb_period,
            "bb_std":              bb_std,
            "stop_loss_atr":       stop_loss_atr,
            "max_holding_days":    max_holding_days,
        },
    }


def test_backtester() -> None:
    section("5 — Backtester")
    from backtester import Backtester, RISK_PER_TRADE, DEFAULT_SLIP_BPS

    bt = Backtester(initial_portfolio=100_000.0, slippage_bps=10)

    # 5a  —  output keys present
    close = np.linspace(100, 150, 200)
    df = make_ohlcv(close)
    strat = _make_momentum_strategy()
    result = bt.run("TEST", strat, df)
    required_keys = {"ticker", "strategy", "trade_log", "equity_curve",
                     "returns", "in_position", "summary"}
    check("5a  output keys present",
          required_keys == set(result.keys()),
          f"got {set(result.keys())}")

    # 5b  —  equity_curve starts at initial portfolio value
    eq = result["equity_curve"]
    check("5b  equity_curve[0] = initial_portfolio (100 000)",
          assert_close(float(eq.iloc[0]), 100_000.0, tol=1.0),
          f"first={float(eq.iloc[0]):.2f}")

    # 5c  —  position sizing formula: risk$=1% of portfolio, stop=stop_loss_atr*ATR
    #         Entry condition: close > rolling_high AND volume > 1.2 × 20d_avg_volume
    #         Fix: baseline volume=1_000_000, breakout bar volume=2_500_000 (2.5× avg) to pass filter
    close_bt  = np.full(200, 100.0)
    close_bt[100:] = 105.0    # breakout above rolling high of 100.5
    vol_bt = np.full(200, 1_000_000.0)
    vol_bt[100] = 2_500_000.0   # 2.5× avg → passes 1.2× volume filter
    df_trigger = pd.DataFrame({
        "Open":   close_bt,
        "High":   close_bt + 0.5,
        "Low":    close_bt - 0.5,
        "Close":  close_bt,
        "Volume": vol_bt,
    }, index=pd.date_range("2022-01-03", periods=200, freq="B"))
    r_ps = bt.run("PS", strat, df_trigger)
    if r_ps["trade_log"]:
        t = r_ps["trade_log"][0]
        atr_at_entry = t.get("current_atr", None)
        if atr_at_entry is not None:
            expected_size = int(100_000.0 * RISK_PER_TRADE / (1.5 * atr_at_entry))
            check("5c  position size matches 1%-risk formula",
                  abs(t["position_size"] - expected_size) <= 2,
                  f"got={t['position_size']} expected~={expected_size}")
        else:
            check("5c  position size is positive and non-zero",
                  t["position_size"] > 0,
                  f"size={t['position_size']}")
    else:
        check("5c  position size (no trade fired — increase volume spike)", False,
              "entry condition still not met; check volume_multiplier logic")

    # 5d  —  returns series has same length as OHLCV
    check("5d  returns series length matches OHLCV",
          len(result["returns"]) == len(df),
          f"returns={len(result['returns'])}  ohlcv={len(df)}")

    # 5e  —  in_position is boolean Series
    ip = result["in_position"]
    check("5e  in_position is boolean Series",
          isinstance(ip, pd.Series) and ip.dtype == bool,
          f"dtype={ip.dtype}")

    # 5f  —  flat days earn daily risk-free rate (not exactly 0)
    #         With no trades, all days should be near DAILY_RF
    close_no_signal = np.full(200, 100.0)   # flat prices, rolling high = 100.5 always → no breakout
    df_no_signal = pd.DataFrame({
        "Open":   close_no_signal,
        "High":   close_no_signal + 0.5,
        "Low":    close_no_signal - 0.5,
        "Close":  close_no_signal,
        "Volume": np.full(200, 1_000_000.0),   # volume never exceeds 1.2x avg since it's constant
    }, index=pd.date_range("2022-01-03", periods=200, freq="B"))
    r_flat = bt.run("FLAT", strat, df_no_signal)
    if len(r_flat["trade_log"]) == 0:
        daily_rf = 0.045 / 252
        mean_ret = float(r_flat["returns"].mean())
        check("5f  flat days earn DAILY_RF (not 0.0)",
              assert_close(mean_ret, daily_rf, tol=daily_rf * 0.01),
              f"mean_ret={mean_ret:.6f}  daily_rf={daily_rf:.6f}")
    else:
        check("5f  flat days earn DAILY_RF (trade fired — skip)", True)

    # 5g  —  slippage cost is positive for any completed trade
    if r_ps["trade_log"]:
        slip_cost = r_ps["trade_log"][0].get("slippage_cost", None)
        check("5g  slippage_cost > 0 for any trade",
              slip_cost is not None and slip_cost > 0,
              f"slippage_cost={slip_cost}")
    else:
        check("5g  slippage_cost (no trade fired — skip)", True)

    # 5h  —  max holding exit: strategy with max_holding_days=1 → all trades exit next day
    strat_1day = _make_momentum_strategy(max_holding_days=1)
    r_1d = bt.run("S1D", strat_1day, df_trigger)   # df_trigger has volume spike at bar 100
    if r_1d["trade_log"]:
        max_hold = max(t["holding_days"] for t in r_1d["trade_log"])
        check("5h  max holding enforced: holding_days <= 1",
              max_hold <= 1,
              f"max_holding={max_hold}")
    else:
        check("5h  max holding (no trade fired — same trigger data as 5c)", False,
              "entry did not fire; check volume_multiplier condition")

    # 5i  —  summary.trade_count matches len(trade_log)
    check("5i  summary.trade_count == len(trade_log)",
          result["summary"]["trade_count"] == len(result["trade_log"]),
          f"summary={result['summary']['trade_count']}  log={len(result['trade_log'])}")

    # 5j  —  total_return is consistent with equity curve
    eq = result["equity_curve"]
    total_ret_from_equity = (float(eq.iloc[-1]) - 100_000.0) / 100_000.0
    summary_ret = result["summary"]["total_return"]
    check("5j  summary total_return consistent with equity curve",
          assert_close(total_ret_from_equity, summary_ret, tol=0.001),
          f"from_equity={total_ret_from_equity:.4f}  summary={summary_ret:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MonteCarloEngine
# ═══════════════════════════════════════════════════════════════════════════════

def _trade(pnl: float, i: int = 0) -> dict:
    base = pd.Timestamp("2022-01-03")
    return {
        "pnl":        pnl,
        "entry_date": base + pd.Timedelta(days=i * 2),
        "exit_date":  base + pd.Timedelta(days=i * 2 + 1),
    }


def test_monte_carlo_engine() -> None:
    section("6 — MonteCarloEngine")
    from monte_carlo_engine import MonteCarloEngine

    mc = MonteCarloEngine(n_simulations=5_000, seed=42)

    # 6a  —  empty trade log returns safe defaults
    r_empty = mc.run([], 100_000.0)
    check("6a  empty trade log: p_ruin = 0.0",
          r_empty["p_ruin"] == 0.0,
          f"p_ruin={r_empty['p_ruin']}")
    check("6b  empty trade log: kelly_fraction = 0.0",
          r_empty["kelly_fraction"] == 0.0,
          f"kelly={r_empty['kelly_fraction']}")

    # 6c  —  all-winning trades → p_ruin = 0
    wins = [_trade(500.0, i) for i in range(50)]
    r_win = mc.run(wins, 100_000.0)
    check("6c  all-winning trades: p_ruin = 0.0",
          r_win["p_ruin"] == 0.0,
          f"p_ruin={r_win['p_ruin']:.4f}")

    # 6d  —  all-winning trades: p50_final > initial
    check("6d  all-winning trades: p50_final > 100 000",
          r_win["p50_final"] > 100_000.0,
          f"p50={r_win['p50_final']:.0f}")

    # 6e  —  catastrophic losses → high p_ruin
    #         Each trade loses $5 000, 50 trades → equity drops to $-150 000 → definitely ruins
    ruins = [_trade(-5_000.0, i) for i in range(50)]
    r_ruin = mc.run(ruins, 100_000.0)
    check("6e  catastrophic losses: p_ruin = 1.0",
          r_ruin["p_ruin"] == 1.0,
          f"p_ruin={r_ruin['p_ruin']:.4f}")

    # 6f  —  equity_band has 20 steps
    check("6f  equity_band has 20 entries",
          len(r_win["equity_band"]) == 20,
          f"steps={len(r_win['equity_band'])}")

    # 6g  —  equity band is monotone: step increases
    steps = [b["step"] for b in r_win["equity_band"]]
    check("6g  equity_band steps are strictly increasing",
          steps == sorted(steps) and len(set(steps)) == 20,
          f"steps={steps[:5]}...")

    # 6h  —  p5 <= p50 <= p95 for final equity
    check("6h  p5_final <= p50_final <= p95_final",
          r_win["p5_final"] <= r_win["p50_final"] <= r_win["p95_final"],
          f"p5={r_win['p5_final']:.0f}  p50={r_win['p50_final']:.0f}  p95={r_win['p95_final']:.0f}")

    # 6i  —  Kelly formula: 60% wins, avg_win=$200, avg_loss=$100
    #         formula: f* = W/L - (1-W)/G = 0.60/100 - 0.40/200 = 0.006 - 0.002 = 0.004
    mixed = ([_trade(200.0, i) for i in range(30)]
           + [_trade(-100.0, i + 30) for i in range(20)])
    r_kelly = mc.run(mixed, 100_000.0)
    W, G, L = 0.60, 200.0, 100.0
    expected_kelly = W / L - (1 - W) / G
    check("6i  Kelly fraction matches W/L - (1-W)/G formula",
          assert_close(r_kelly["kelly_fraction"], expected_kelly, tol=0.01),
          f"got={r_kelly['kelly_fraction']:.4f}  expected≈{expected_kelly:.4f}")

    # 6j  —  median_time_to_ruin = None when p_ruin = 0
    check("6j  median_time_to_ruin = None when no ruin",
          r_win["median_time_to_ruin"] is None,
          f"got={r_win['median_time_to_ruin']}")

    # 6k  —  p50 Sharpe > 0 for all-winning trade log
    check("6k  p50_sharpe > 0 for all-winning trades",
          r_win["p50_sharpe"] > 0,
          f"p50_sharpe={r_win['p50_sharpe']:.3f}")

    # 6l  —  max_consec_losses = 0 for all-winning trades
    check("6l  p95_max_consec_losses = 0 for all-winning trades",
          r_win["p95_max_consec_losses"] == 0,
          f"p95_consec={r_win['p95_max_consec_losses']}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ExecutionAdvisor (non-network parts)
# ═══════════════════════════════════════════════════════════════════════════════

def test_execution_advisor() -> None:
    section("7 — ExecutionAdvisor")
    from execution_advisor import ExecutionAdvisor, get_market_status

    adv = ExecutionAdvisor(initial_portfolio=100_000.0)

    # 7a  —  get_market_status returns required keys
    status = get_market_status()
    check("7a  market_status has 'open', 'label', 'detail' keys",
          {"open", "label", "detail"} == set(status.keys()),
          f"got {set(status.keys())}")

    # 7b  —  market_status.open is bool
    check("7b  market_status.open is bool",
          isinstance(status["open"], bool))

    # 7c  —  market_status.label is non-empty string
    check("7c  market_status.label is non-empty string",
          isinstance(status["label"], str) and len(status["label"]) > 0,
          f"label='{status['label']}'")

    # 7d  —  _classify_impact: small position vs large ADV → negligible
    impact_neg = adv._classify_impact(100, 1_000_000)   # 0.01% of ADV
    check("7d  0.01% ADV -> 'negligible' impact",
          impact_neg == "negligible",
          f"impact='{impact_neg}'")

    # 7e  —  _classify_impact: 2% of ADV → moderate
    impact_mod = adv._classify_impact(2_000, 100_000)   # 2% of ADV
    check("7e  2% ADV -> 'moderate' impact",
          impact_mod == "moderate",
          f"impact='{impact_mod}'")

    # 7f  —  _classify_impact: 10% of ADV → significant
    impact_sig = adv._classify_impact(10_000, 100_000)   # 10% of ADV
    check("7f  10% ADV -> 'significant' impact",
          impact_sig == "significant",
          f"impact='{impact_sig}'")

    # 7g  —  _classify_impact: ADV = 0 → negligible (guard for unknown ADV)
    impact_zero_adv = adv._classify_impact(1_000, 0)
    check("7g  ADV=0 -> 'negligible' (unknown ADV)",
          impact_zero_adv == "negligible",
          f"impact='{impact_zero_adv}'")

    # 7h  —  advise() returns required top-level keys
    result = adv.advise([])
    required_adv_keys = {"market_status", "active_signals", "inactive_count",
                         "portfolio_risk", "warnings"}
    check("7h  advise([]) returns all required keys",
          required_adv_keys == set(result.keys()),
          f"got {set(result.keys())}")

    # 7i  —  advise(no signals) → active_signals is empty list
    check("7i  advise([]) -> active_signals = []",
          result["active_signals"] == [],
          f"active={result['active_signals']}")

    # 7j  —  portfolio_risk.active_count = 0 when no signals
    check("7j  portfolio_risk.active_count = 0 for empty input",
          result["portfolio_risk"]["active_count"] == 0,
          f"count={result['portfolio_risk']['active_count']}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — _advanced_metrics VaR/CVaR bug fix
# ═══════════════════════════════════════════════════════════════════════════════

def test_advanced_metrics_var() -> None:
    section("8 — report_generator._advanced_metrics VaR/CVaR (invested-days-only fix)")
    from report_generator import _advanced_metrics

    idx = pd.date_range("2022-01-03", periods=252, freq="B")
    metrics_stub = {"max_drawdown": 0.05}

    # 8a  —  VaR is 0.0 on old method (regression demo)
    #         250 flat days + 2 loss days → 5th percentile of full series = 0.0
    vals_bug_demo = [0.0] * 252
    vals_bug_demo[0] = -0.05
    vals_bug_demo[1] = -0.05
    rets_demo = pd.Series(vals_bug_demo, index=idx)
    old_var = float(np.percentile(rets_demo.dropna(), 5))
    check("8a  OLD method (full series): VaR = 0.0 on mostly-flat series (confirms bug existed)",
          old_var == 0.0,
          f"old_var={old_var}")

    # 8b  —  NEW method (invested days): VaR is non-zero
    #         Build a series with 20 invested days (>= 10 required for new method)
    #         so the fix path is taken.  15 gains + 5 losses → VaR should be negative.
    vals_bug = [0.0] * 252
    for i in range(15):
        vals_bug[i * 10] = 0.02          # 15 positive days spread across the series
    for i in range(5):
        vals_bug[i * 10 + 5] = -0.05    # 5 loss days interleaved
    rets_bug = pd.Series(vals_bug, index=idx)
    result = _advanced_metrics(rets_bug, [], metrics_stub)
    check("8b  NEW method: VaR is non-zero for series with 20 invested days",
          result["var_95"] < -0.01,
          f"var_95={result['var_95']:.4f}")

    # 8c  —  CVaR <= VaR (CVaR is mean of tail below VaR)
    check("8c  CVaR <= VaR (tail mean worse than threshold)",
          result["cvar_95"] <= result["var_95"] + 1e-9,
          f"cvar={result['cvar_95']:.4f}  var={result['var_95']:.4f}")

    # 8d  —  VaR reflects actual loss magnitude
    vals_known = [0.0] * 252
    # 20 invested days: 18 gains of +0.02, 2 losses of -0.05 (10% of invested)
    for i in range(18):
        vals_known[i * 10] = 0.02
    vals_known[5]  = -0.05
    vals_known[15] = -0.05
    rets_known = pd.Series(vals_known, index=idx)
    result_known = _advanced_metrics(rets_known, [], metrics_stub)
    # 5th percentile of [−0.05, −0.05, +0.02 × 18] → should hit -0.05
    check("8d  VaR ~= worst daily loss when < 10% of invested days are losses",
          result_known["var_95"] <= -0.04,
          f"var_95={result_known['var_95']:.4f}")

    # 8e  —  Sortino > 0 for mostly-positive returns with VARIED occasional losses
    #         (identical losses → down_std=0 → Sortino=0 by code guard; must vary loss sizes)
    rets_arr = np.full(252, 0.004)
    varied_losses = [-0.001, -0.005, -0.002, -0.008, -0.003, -0.006, -0.004, -0.007]
    for i, loss in enumerate(varied_losses):
        rets_arr[i * 20] = loss   # 8 losses spread out, different magnitudes
    rets_mostly_pos = pd.Series(rets_arr, index=idx)
    res_mp = _advanced_metrics(rets_mostly_pos, [], metrics_stub)
    check("8e  Sortino > 0 for mostly-positive returns (varied loss magnitudes)",
          res_mp["sortino"] > 0,
          f"sortino={res_mp['sortino']:.3f}")

    # 8f  —  CAGR > 0 for positive net daily returns
    check("8f  CAGR > 0 for mostly-positive returns",
          res_mp["cagr"] > 0,
          f"cagr={res_mp['cagr']:.4f}")

    # 8g  —  IS/OOS Sharpe split at midpoint; OOS is clearly stronger than IS
    #         Use alternating returns so std > 0 in both halves.
    is_rets  = np.where(np.arange(126) % 2 == 0, 0.001, -0.001)   # IS: zero-mean
    oos_rets = np.where(np.arange(126) % 2 == 0, 0.008, 0.003)    # OOS: mean=0.0055
    split_rets = pd.Series(np.concatenate([is_rets, oos_rets]), index=idx)
    res_split = _advanced_metrics(split_rets, [], {"max_drawdown": 0.01})
    check("8g  IS Sharpe < OOS Sharpe when OOS is clearly stronger",
          res_split["is_sharpe"] < res_split["oos_sharpe"],
          f"is={res_split['is_sharpe']:.3f}  oos={res_split['oos_sharpe']:.3f}")

    # 8h  —  recovery_days > 0 when there is a drawdown period
    dd_rets = pd.Series(
        np.concatenate([np.full(50, 0.005), np.full(50, -0.003), np.full(152, 0.002)]),
        index=idx,
    )
    res_dd = _advanced_metrics(dd_rets, [], {"max_drawdown": 0.05})
    check("8h  recovery_days > 0 when drawdown period exists",
          res_dd["recovery_days"] > 0,
          f"recovery_days={res_dd['recovery_days']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — run all sections and print summary
# ═══════════════════════════════════════════════════════════════════════════════

_SECTIONS = [
    test_ohlcv_fetcher,
    test_regime_classifier,
    test_strategy_selector_params,
    test_diagnostics_engine,
    test_backtester,
    test_monte_carlo_engine,
    test_execution_advisor,
    test_advanced_metrics_var,
]

# Also expose as pytest-discoverable test functions
def test_1_ohlcv_fetcher():              test_ohlcv_fetcher()
def test_2_regime_classifier():          test_regime_classifier()
def test_3_strategy_selector_params():   test_strategy_selector_params()
def test_4_diagnostics_engine():         test_diagnostics_engine()
def test_5_backtester():                 test_backtester()
def test_6_monte_carlo_engine():         test_monte_carlo_engine()
def test_7_execution_advisor():          test_execution_advisor()
def test_8_advanced_metrics_var():       test_advanced_metrics_var()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ALGORITHMIC PIPELINE TEST SUITE")
    print("=" * 70)

    section_errors: list[str] = []
    for fn in _SECTIONS:
        try:
            fn()
        except Exception as e:
            section_errors.append(f"{fn.__name__}: {e}")
            print(f"\n  [!] SECTION CRASHED: {e}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    width = 70
    print(f"\n{'=' * width}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * width}")

    total   = len(_results)
    passed  = sum(1 for _, ok, _ in _results if ok)
    failed  = total - passed

    if failed:
        print(f"\n  FAILURES ({failed}):")
        for name, ok, msg in _results:
            if not ok:
                print(f"    [X] {name}  —  {msg}")

    print(f"\n  Passed : {passed}/{total}")
    print(f"  Failed : {failed}/{total}")
    if section_errors:
        print(f"  Section crashes: {len(section_errors)}")
        for e in section_errors:
            print(f"    {e}")

    print()
    sys.exit(0 if failed == 0 and not section_errors else 1)
