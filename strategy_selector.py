"""
StrategySelector
================
Maps a regime label to a strategy template, then adjusts parameters via a
deterministic rule-based algorithm (no LLM numeric decisions).  The LLM is
called once only to produce a plain-English explanation of the final params
for the report — it has no influence on the numbers.

<<<<<<< HEAD
Regime -> Strategy mapping  (see _REGIME_TO_STRATEGY for the authoritative dict)
--------------------------
    Trending-Up      -> Momentum           (follow the confirmed uptrend)
    Trending-Down    -> Mean-Reversion     (fade oversold exhaustion within the
                                            downtrend; long-only VolatilityBreakout
                                            buying above the upper-BB fights the trend)
    High-Volatility  -> VolatilityBreakout  (BB squeeze → expansion alpha)
    Crisis           -> VolatilityBreakout  (trade the next breakout, not the dip)
    Mean-Reverting   -> AlphaCombined       (multi-factor MR signal)
    Low-Volatility   -> MLSignal            (quiet markets: ML detects subtle patterns)
    Neutral          -> MLSignal            (no strong structural signal: let ML decide)
    Event-Driven     -> EventDriven         (post-earnings PEAD drift)
=======
Regime -> Strategy mapping (current)
------------------------------------
    Trending-Up      -> Momentum                (follow trend long)
    Trending-Down    -> AlphaCombined           (multi-factor reversion in downtrend)
    High-Volatility  -> VolatilityBreakout      (BB squeeze → expansion alpha)
    Low-Volatility   -> MLSignal                (ML finds subtle nonlinear patterns)
    Mean-Reverting   -> Mean-Reversion (classical RSI/BB) when atr_pct < 2%
                     -> AlphaCombined           otherwise (multi-factor MR)
    Crisis           -> AlphaCombined           (tight params; alpha with defensive stops)
    Event-Driven     -> EventDriven             (PEAD — post-earnings drift)
    Neutral          -> MLSignal                (no structural bias — learn from data)
    Joint Crisis     -> All strategies tightened via `joint_crisis` flag
                        (market crisis ATR > 6% AND news bearish w/ ≥2 active risks)
>>>>>>> main

Parameter adjustment rules (Momentum)
--------------------------------------
    trailing_stop_atr : 2.0 base
        +0.5  if Hurst > 0.70   (strong trend persistence — give more room)
    stop_loss_atr     : 1.5 base
        +0.5  if ATR% > 2.5%    (high volatility — widen hard stop)
    volume_multiplier : 1.2 base
        +0.3  if volume_ratio_30d > 1.3  (elevated volume — tighten confirmation)
    max_holding_days  : 20 base
        -> 30 if Hurst > 0.75   (very strong trend — allow longer ride)
    entry_lookback    : 10 (fixed)
    ma_exit_period    : 50 (fixed — institutional 50d MA; 20d fired prematurely)

Parameter adjustment rules (Mean-Reversion)
--------------------------------------------
    rsi_entry_threshold : 30 base
        -> 35  if ATR% > 2.5%   (wider oversold band in volatile market)
    bb_std              : 2.0 base
        -> 2.5 if ATR% > 3.0%   (wider Bollinger in very volatile market)
    stop_loss_atr       : 1.5 base
        +0.5  if ATR% > 2.5%
    max_holding_days    : 10 base
        -> 15  if ATR% < 1.5%   (slow mean reversion in low-vol market)
    rsi_exit_threshold  : 55 (fixed)
    bb_period           : 20 (fixed)

Public interface
----------------
    sel    = StrategySelector(llm_client=llm_fn, verbose=True)
    result = sel.select(ticker, regime_dict, ohlcv_features, macro_dict)
"""

from __future__ import annotations

import copy
import json

# ── PRD base templates ────────────────────────────────────────────────────────

MOMENTUM_BASE: dict = {
    "entry_lookback":       10,
    "volume_multiplier":    1.2,
    "trailing_stop_atr":    2.0,
    "ma_exit_period":       50,    # widened from 20 → 50 — 20d MA fires too early in trending regimes,
                                   # cutting winners short before the trend exhausts.
                                   # 50d MA is the institutional standard for trend-following exits
                                   # and aligns with the ~3-month holding horizon of Jegadeesh-Titman
                                   # cross-sectional momentum (the regime this strategy exploits).
    "stop_loss_atr":        1.5,
    "max_holding_days":     20,
    "momentum_lookback":    252,   # 12-1 month momentum gate: only enter if 11m return (skip last month) > 0
    "momentum_gate_active": True,  # set False in Crisis/High-Vol to allow entries after crashes
}

MEAN_REVERSION_BASE: dict = {
    "rsi_entry_threshold": 30,
    "rsi_exit_threshold":  55,
    "bb_period":           20,
    "bb_std":              2.0,
    "stop_loss_atr":       1.5,
    "max_holding_days":    10,
}

VOLATILITY_BREAKOUT_BASE: dict = {
    # Bollinger Band squeeze → expansion breakout.
    # Alpha source: volatility compression precedes directional moves.
    # The squeeze compresses THEN the price breaks out — entry conditions:
    #   1. BB width was in the bottom squeeze_pct percentile within squeeze_lookback bars
    #      (confirms prior compression)
    #   2. Close breaks above the upper Bollinger Band (breakout direction = long)
    #   3. Volume > volume_mult × 20-bar average (confirms institutional participation)
    "bb_period":         20,
    "squeeze_pct":       0.20,   # BB width in bottom N-th percentile = squeeze
    "squeeze_lookback":  5,      # bars back to look for prior squeeze
    "volume_mult":       1.5,    # volume must exceed N× 20-bar avg at breakout
    "stop_loss_atr":     2.0,
    "trailing_stop_atr": 2.5,
    "max_holding_days":  15,
}

ALPHA_COMBINED_BASE: dict = {
    # Cross-sectional multi-factor signal strategy.
    # alpha_threshold: minimum combined z-score to enter.
    # reversal_threshold: exit when signal drops below this (signal flipped).
    # Designed for mean-reversion regimes; much higher trade frequency than RSI+BB.
    "alpha_threshold":    0.40,
    "reversal_threshold": -0.50,
    "stop_loss_atr":      1.5,
    "trailing_stop_atr":  2.0,
    "max_holding_days":   10,
}

ML_SIGNAL_BASE: dict = {
    # Gradient-boosting ML probability signal.
    # ml_threshold: P(5d return > 0) must exceed this to enter (0–1 range).
    # reversal_threshold: exit when probability drops below this (model lost conviction).
    "ml_threshold":       0.60,
    "reversal_threshold": 0.40,
    "stop_loss_atr":      1.5,
    "trailing_stop_atr":  2.0,
    "max_holding_days":   10,
}

PAIR_TRADING_BASE: dict = {
    # Relative Value / Pair Trading (Engle-Granger cointegration).
    # Alpha source: cointegrated pairs share a long-run equilibrium; when the
    # spread (log P_A - beta * log P_B) deviates beyond entry_z standard
    # deviations, it reverts toward zero — the trade captures that reversion.
    #
    # entry_z           : z-score threshold to open a position (spread ≥ 2σ from mean).
    #                     Standard academic value (Gatev et al. 2006): 2.0.
    # exit_z            : z-score at which to close (spread converged to mean ≈ 0).
    #                     Slightly above 0 to avoid noise at the mean line.
    # stop_z            : z-score beyond which the spread is diverging, not converging.
    #                     Stop losses are rare in mean-reverting strategies but
    #                     necessary to prevent "it will revert eventually" blow-ups.
    # beta_window        : rolling OLS window for hedge ratio estimation (days).
    #                     60 days: adapts to slow regime shifts without over-fitting.
    # z_window           : rolling mean/std window for z-score normalisation.
    #                     Must match beta_window so the spread distribution is stable.
    # max_holding_days   : hard time stop — if spread hasn't converged in 15 days
    #                      something structural has changed; cut the trade.
    # borrow_rate_annual : annual stock borrow cost on the short leg.
    #                      50 bps is typical for large-cap S&P 500 names.
    "entry_z":           2.0,
    "exit_z":            0.25,
    "stop_z":            3.5,
    "beta_window":       60,
    "z_window":          60,
    "max_holding_days":  15,
    "borrow_rate_annual": 0.005,
}

EVENT_DRIVEN_BASE: dict = {
    # Post-Earnings Announcement Drift (PEAD) strategy.
    # Alpha source: after a positive earnings surprise, prices drift higher for
    # 5–60 days (Rendleman et al. 1982, Bernard & Thomas 1989).  The strategy
    # enters AFTER the earnings blackout window lifts and rides the drift.
    #
    # gap_threshold     : minimum earnings gap to qualify (2% = modest beat).
    # pead_min_signal   : minimum PEAD z-score to enter (scale −1 to +1;
    #                     0.20 corresponds to a 2% gap — same as gap_threshold).
    # pead_exit_threshold: exit when pead_signal drops below this; negative
    #                     means the drift has reversed (earnings re-rated lower).
    # entry_window_bars : look for an earnings gap within this many recent bars.
    #                     Must be > blackout window (±3) so entry is possible
    #                     after the blackout lifts; default 10 = bars t+4…t+13.
    # volume_mult       : volume confirmation — post-earnings institutional flow
    #                     should keep volume elevated above the 20-bar average.
    # ma_filter_period  : close must be above this short MA to confirm upward drift.
    # stop_loss_atr     : hard stop in ATR multiples (wider than MR — earnings
    #                     reactions are noisy; 1.5 gives room for initial volatility).
    # trailing_stop_atr : trailing stop to lock in drift profits as price moves up.
    # max_holding_days  : PEAD drift typically resolves in 5–15 days; cap at 7.
    "gap_threshold":        0.02,
    "pead_min_signal":      0.20,
    "pead_exit_threshold": -0.10,
    "entry_window_bars":   10,
    "volume_mult":          1.3,
    "ma_filter_period":     5,
    "stop_loss_atr":        1.5,
    "trailing_stop_atr":    2.0,
    "max_holding_days":     7,
}

_REGIME_TO_STRATEGY: dict[str, str] = {
    # Directional trend regimes
    "Trending-Up":      "Momentum",          # follow the confirmed uptrend long
    # Trending-Down: in a long-only system, VolatilityBreakout (which buys above the
    # upper BB) fights the downtrend — the breakout signal fires against momentum, not
    # with it.  Mean-Reversion fades oversold exhaustion within a downtrend, which is
    # the correct long-only posture: wait for RSI compression and a lower-BB touch,
    # capture the bounce, exit quickly.  If no oversold setup appears, no trade is taken.
    "Trending-Down":    "Mean-Reversion",    # fade oversold exhaustion in the downtrend
    # Volatility regimes
    "High-Volatility":  "VolatilityBreakout",  # squeeze → expansion alpha
    "Low-Volatility":   "MLSignal",             # quiet markets: ML detects subtle nonlinear patterns
<<<<<<< HEAD
    "Crisis":           "VolatilityBreakout",  # extreme vol: trade the next directional breakout, not the dip
    # Statistical regimes
    "Mean-Reverting":   "AlphaCombined",        # primary regime for multi-factor MR
=======
    "Crisis":           "AlphaCombined",        # extreme moves: use alpha signal with tight stops
    # Statistical regimes — Mean-Reverting is special-cased in _route_strategy()
    # and may map to either "Mean-Reversion" (classical RSI/BB, low-vol) or
    # "AlphaCombined" (multi-factor, normal/high-vol). The entry here is the
    # default fallback if atr_pct cannot be evaluated.
    "Mean-Reverting":   "AlphaCombined",
>>>>>>> main
    "Neutral":          "MLSignal",             # no strong structural bias: ML learns from data
    # Exogenous event regime — dedicated PEAD strategy
    "Event-Driven":     "EventDriven",          # post-earnings drift: enter after blackout, ride PEAD
    # Legacy label — kept for backward compatibility with any cached data
    "Trending":         "Momentum",
}

# ATR threshold below which Mean-Reverting regime routes to the classical
# RSI+Bollinger Mean-Reversion engine instead of the multi-factor AlphaCombined
# engine. Classical MR thrives in quiet tape (<2% ATR/price); AlphaCombined
# handles the noisier end of mean-reversion regimes.
_MR_LOW_VOL_CUTOFF = 0.020


def _route_strategy(regime_label: str, atr_pct: float) -> str:
    """
    Deterministic regime→strategy router.

    Encapsulates the one conditional override on top of ``_REGIME_TO_STRATEGY``:
    Mean-Reverting regimes with low realised volatility (< 2% ATR/price) route
    to the classical Mean-Reversion engine; everything else uses the static
    mapping. This keeps all routing logic in one place and ensures the
    Mean-Reversion branch is actually reachable.
    """
    if regime_label == "Mean-Reverting" and atr_pct < _MR_LOW_VOL_CUTOFF:
        return "Mean-Reversion"
    return _REGIME_TO_STRATEGY.get(regime_label, "Momentum")

_STRATEGY_TO_BASE: dict[str, dict] = {
    "Momentum":           MOMENTUM_BASE,
    "Mean-Reversion":     MEAN_REVERSION_BASE,
    "VolatilityBreakout": VOLATILITY_BREAKOUT_BASE,
    "AlphaCombined":      ALPHA_COMBINED_BASE,
    "MLSignal":           ML_SIGNAL_BASE,
    "EventDriven":        EVENT_DRIVEN_BASE,
    "PairTrading":        PAIR_TRADING_BASE,
}

_HYPOTHESIS_PROMPT = """\
You are a quantitative researcher selecting the best strategy for a ticker.

TICKER: {ticker}
REGIME (algorithmic): {regime}  |  Hurst: {hurst:.3f}
VOLATILITY: ATR/price = {atr_pct:.2%}
MOMENTUM: 20d return = {ret_20d:.2%}  |  RSI(14) = {rsi:.1f}  |  Volume ratio = {vol_ratio:.2f}x
MARKET CONTEXT: {market_bias}
NEWS VERDICT: {news_verdict}
NEWS REASONING: {news_reasoning}

AVAILABLE STRATEGY CLASSES:
1. Momentum            — N-day high breakout + volume confirmation. Edge: trend persistence (Hurst > 0.55).
2. Mean-Reversion      — RSI oversold + below lower Bollinger Band. Edge: oscillation in low-Hurst assets.
3. VolatilityBreakout  — BB squeeze → expansion + ATR surge. Edge: compressed volatility preceding directional move. Works in both up and down breakouts — direction-agnostic.
4. AlphaCombined       — Cross-sectional multi-factor signal (CS-MR + residual + vol-spike + momentum). Edge: diversified alpha, higher trade frequency, market-neutral component.
5. MLSignal            — Gradient-boosting ML probability signal. Edge: learns nonlinear patterns from lagged features in low-structural-bias regimes.
6. EventDriven         — Post-Earnings Announcement Drift (PEAD): enter after blackout, ride earnings gap drift. Edge: systematic under-reaction to earnings surprises (Bernard & Thomas 1989).

REGIME RULE SELECTED: {regime_rule_strategy}

Do you AGREE or DISAGREE with this selection given the news context and current conditions?

Respond in EXACTLY this format (one line only):
VERDICT: AGREE
or
VERDICT: DISAGREE | SUGGESTED: [Momentum|Mean-Reversion|VolatilityBreakout|AlphaCombined|MLSignal] | REASON: [one sentence]
"""

_REASONING_PROMPT = """\
You are writing a one-sentence explanation for a trading report.

The following strategy parameters were set algorithmically for {ticker} \
({strategy} strategy, {regime} regime):

{params_block}

Key inputs used:
  Hurst exponent : {hurst:.3f}
  ATR/price      : {atr_pct:.3%}
  Volume ratio   : {vol_ratio:.2f}
  Market bias    : {market_bias}

Write ONE sentence explaining why these parameters suit this ticker's \
current regime and volatility profile. Be specific. Do not suggest changes.
"""


def _compute_momentum_params(hurst: float, atr_pct: float, vol_ratio: float) -> tuple[dict, list[str]]:
    """Deterministic momentum parameter rules. Returns (params, rule_log)."""
    p = copy.deepcopy(MOMENTUM_BASE)
    rules: list[str] = []

    if hurst > 0.70:
        p["trailing_stop_atr"] += 0.5
        rules.append(f"trailing_stop_atr -> {p['trailing_stop_atr']} (Hurst {hurst:.3f} > 0.70 — strong trend persistence)")

    if atr_pct > 0.025:
        p["stop_loss_atr"] += 0.5
        rules.append(f"stop_loss_atr -> {p['stop_loss_atr']} (ATR% {atr_pct:.2%} > 2.5% — high volatility)")

    if vol_ratio > 1.3:
        p["volume_multiplier"] += 0.3
        rules.append(f"volume_multiplier -> {p['volume_multiplier']:.1f} (volume_ratio {vol_ratio:.2f} > 1.3 — elevated volume)")

    if hurst > 0.75:
        p["max_holding_days"] = 30
        rules.append(f"max_holding_days -> 30 (Hurst {hurst:.3f} > 0.75 — very strong trend)")

    return p, rules


def _compute_mean_reversion_params(atr_pct: float) -> tuple[dict, list[str]]:
    """Deterministic mean-reversion parameter rules. Returns (params, rule_log)."""
    p = copy.deepcopy(MEAN_REVERSION_BASE)
    rules: list[str] = []

    if atr_pct > 0.025:
        p["rsi_entry_threshold"] = 35
        p["stop_loss_atr"]      += 0.5
        rules.append(f"rsi_entry_threshold -> 35, stop_loss_atr -> {p['stop_loss_atr']} (ATR% {atr_pct:.2%} > 2.5% — high volatility)")

    if atr_pct > 0.030:
        p["bb_std"] = 2.5
        rules.append(f"bb_std -> 2.5 (ATR% {atr_pct:.2%} > 3.0% — very high volatility)")

    if atr_pct < 0.015:
        p["max_holding_days"] = 15
        rules.append(f"max_holding_days -> 15 (ATR% {atr_pct:.2%} < 1.5% — slow mean reversion in low-vol market)")

    return p, rules


def _compute_volatility_breakout_params(atr_pct: float, hurst: float) -> tuple[dict, list[str]]:
    """Deterministic VolatilityBreakout parameter rules. Returns (params, rule_log)."""
    p = copy.deepcopy(VOLATILITY_BREAKOUT_BASE)
    rules: list[str] = []

    if atr_pct > 0.04:
        # Extreme volatility: widen stops so we're not stopped out by normal noise
        p["stop_loss_atr"]     = 2.5
        p["trailing_stop_atr"] = 3.0
        rules.append(
            f"stop_loss_atr -> 2.5, trailing_stop_atr -> 3.0 "
            f"(ATR% {atr_pct:.2%} > 4% — extreme volatility, widen stops)"
        )

    if hurst > 0.55:
        # Trending bias in high-vol — allow longer hold to capture continuation
        p["max_holding_days"] = 20
        rules.append(
            f"max_holding_days -> 20 (Hurst {hurst:.3f} > 0.55 — trending bias, extend hold)"
        )

    if atr_pct < 0.015:
        # Very low vol — require more volume confirmation to avoid noise breakouts
        p["volume_mult"] = 2.0
        rules.append(
            f"volume_mult -> 2.0 (ATR% {atr_pct:.2%} < 1.5% — tighten volume confirmation)"
        )

    return p, rules


def _compute_alpha_combined_params(
    atr_pct: float, hurst: float, regime_label: str
) -> tuple[dict, list[str]]:
    """Deterministic AlphaCombined parameter rules. Returns (params, rule_log)."""
    import copy as _copy
    p = _copy.deepcopy(ALPHA_COMBINED_BASE)
    rules: list[str] = []

    # Crisis: tighten stops, lower threshold to enter more defensively
    if regime_label == "Crisis":
        p["stop_loss_atr"]     = 1.0
        p["trailing_stop_atr"] = 1.5
        p["max_holding_days"]  = 5
        p["alpha_threshold"]   = 0.50   # require stronger signal in panic conditions
        rules.append(
            "Crisis: stop_loss_atr=1.0, trailing=1.5, max_hold=5, "
            "alpha_threshold=0.50 (tighter params in extreme vol)"
        )

    # Event-Driven: short gap-fill window
    if regime_label == "Event-Driven":
        p["max_holding_days"]  = 7
        p["alpha_threshold"]   = 0.45
        rules.append(
            "Event-Driven: max_holding_days=7, alpha_threshold=0.45 "
            "(target post-event gap fill within 7 bars)"
        )

    # High ATR: widen stops slightly
    if atr_pct > 0.03:
        p["stop_loss_atr"]    += 0.5
        p["trailing_stop_atr"] += 0.5
        rules.append(
            f"High ATR {atr_pct:.2%}: stop_loss_atr={p['stop_loss_atr']}, "
            f"trailing={p['trailing_stop_atr']}"
        )

    # Low vol: extend max hold (reversion takes longer in quiet markets)
    if atr_pct < 0.015:
        p["max_holding_days"] = max(p["max_holding_days"], 15)
        rules.append(
            f"Low ATR {atr_pct:.2%}: max_holding_days={p['max_holding_days']} "
            "(slow reversion in quiet market)"
        )

    return p, rules


def _compute_ml_signal_params(
    atr_pct: float, regime_label: str
) -> tuple[dict, list[str]]:
    """Deterministic MLSignal parameter rules. Returns (params, rule_log)."""
    p = copy.deepcopy(ML_SIGNAL_BASE)
    rules: list[str] = []

    # Low-Volatility: relax entry threshold — ML signal is more reliable in
    # quiet, low-noise markets; allow longer hold for slow reversion
    if regime_label == "Low-Volatility":
        p["ml_threshold"]     = 0.55
        p["max_holding_days"] = 15
        rules.append(
            "Low-Volatility: ml_threshold=0.55, max_holding_days=15 "
            "(ML signal more reliable in low-noise environment; slower reversion)"
        )

    # High ATR: widen stops — larger price swings need more room
    if atr_pct > 0.03:
        p["stop_loss_atr"]    += 0.5
        p["trailing_stop_atr"] += 0.5
        rules.append(
            f"High ATR {atr_pct:.2%}: stop_loss_atr={p['stop_loss_atr']}, "
            f"trailing={p['trailing_stop_atr']} (widen stops in volatile market)"
        )

    # Very low ATR: extend max hold — slow markets take longer to resolve
    if atr_pct < 0.015:
        p["max_holding_days"] = max(p["max_holding_days"], 15)
        rules.append(
            f"Low ATR {atr_pct:.2%}: max_holding_days={p['max_holding_days']} "
            "(slow market — extend hold)"
        )

    return p, rules


def _compute_event_driven_params(
    atr_pct: float, pead_signal_recent: float
) -> tuple[dict, list[str]]:
    """
    Deterministic EventDriven (PEAD) parameter rules.  Returns (params, rule_log).

    Parameter adjustment logic
    --------------------------
    pead_signal_recent > 0.50  (gap ≥ 5%):
        Large earnings surprise → stronger, longer drift expected.
        Loosen pead_min_signal to 0.10 (don't require signal to stay high),
        extend max_holding_days to 10 (drift lasts longer on big beats).

    atr_pct > 0.025:
        High post-earnings volatility — widen both stops so we are not
        stopped out by the normal noise following an announcement.

    atr_pct < 0.015:
        Very low volatility — drift resolves slowly; extend max hold to 10.
        Require tighter volume confirmation (1.5×) since low-vol names have
        less liquidity expansion post-earnings.
    """
    p = copy.deepcopy(EVENT_DRIVEN_BASE)
    rules: list[str] = []

    if pead_signal_recent > 0.50:
        p["pead_min_signal"]  = 0.10
        p["max_holding_days"] = 10
        rules.append(
            f"Large PEAD signal {pead_signal_recent:.2f} > 0.50 (gap ≥ 5%): "
            "pead_min_signal=0.10, max_holding_days=10 "
            "(strong beat — drift expected to persist longer)"
        )

    if atr_pct > 0.025:
        p["stop_loss_atr"]    += 0.5
        p["trailing_stop_atr"] += 0.5
        rules.append(
            f"High ATR {atr_pct:.2%} > 2.5%: "
            f"stop_loss_atr={p['stop_loss_atr']}, trailing={p['trailing_stop_atr']} "
            "(high post-earnings noise — widen stops)"
        )

    if atr_pct < 0.015:
        p["max_holding_days"] = max(p["max_holding_days"], 10)
        p["volume_mult"]      = 1.5
        rules.append(
            f"Low ATR {atr_pct:.2%} < 1.5%: "
            f"max_holding_days={p['max_holding_days']}, volume_mult=1.5 "
            "(slow drift in low-vol — extend hold, tighten volume confirmation)"
        )

    return p, rules


def _compute_pair_trading_params(
    hurst: float, atr_pct: float
) -> tuple[dict, list[str]]:
    """
    Deterministic PairTrading parameter rules.  Returns (params, rule_log).

    Parameter adjustment logic
    --------------------------
    hurst < 0.45  (strong mean-reversion regime):
        High-conviction convergence environment — tighten entry_z slightly (1.8)
        to enter more frequently when regime structure favors reversion.

    hurst > 0.55  (trending regime):
        Pairs spreads are more likely to trend/diverge; widen entry threshold
        to 2.3 and tighten stop to 3.0 — only trade the most extreme dislocations
        and cut quickly if they continue diverging.

    atr_pct > 0.025  (high volatility):
        Larger daily price swings inflate z-scores spuriously.  Widen stop_z
        to 4.0 so we don't stop out on noise, and extend max holding to 20 days
        as convergence takes longer in volatile markets.

    atr_pct < 0.015  (low volatility):
        Quiet markets have very stable spreads; tighten entry_z to 1.8 to capture
        the smaller but more reliable dislocations.
    """
    p     = copy.deepcopy(PAIR_TRADING_BASE)
    rules: list[str] = []

    if hurst < 0.45:
        p["entry_z"] = 1.8
        rules.append(
            f"Hurst {hurst:.3f} < 0.45 (strong mean-reversion): "
            "entry_z=1.8 — tighter threshold in high-conviction reversion regime"
        )

    if hurst > 0.55:
        p["entry_z"] = 2.3
        p["stop_z"]  = 3.0
        rules.append(
            f"Hurst {hurst:.3f} > 0.55 (trending market): "
            "entry_z=2.3, stop_z=3.0 — only extreme dislocations; cut divergers quickly"
        )

    if atr_pct > 0.025:
        p["stop_z"]           = 4.0
        p["max_holding_days"] = 20
        rules.append(
            f"High ATR {atr_pct:.2%} > 2.5%: "
            "stop_z=4.0, max_holding_days=20 — wider stop in volatile market; "
            "convergence takes longer"
        )

    if atr_pct < 0.015:
        p["entry_z"] = min(p["entry_z"], 1.8)
        rules.append(
            f"Low ATR {atr_pct:.2%} < 1.5%: "
            f"entry_z={p['entry_z']:.1f} — smaller dislocations are reliable in quiet market"
        )

    return p, rules


def _apply_market_state_overrides(
    params: dict,
    rule_log: list[str],
    strategy: str,
    market_state: str,
) -> tuple[dict, list[str]]:
    """
    Apply portfolio-level market state overrides on top of per-ticker regime params.

    Crisis
    ------
    - Disable 12-1 month momentum gate (momentum_gate_active = False).
      Rationale: 12-1m gate filters out tickers that crashed recently — which is
      exactly the universe in a crisis. The gate is designed for normal trending
      markets; keeping it active in a crash guarantees zero entries.
    - Cap max_holding_days at 5 days across ALL strategies.
      Rationale: holding 10-30 days through a crash gets whipsawed by 5-10%
      intraday reversals. Short holds cut risk; the market can reverse completely
      in days during panic conditions.

    High-Volatility
    ---------------
    - Cap max_holding_days at 10 days across ALL strategies.
      Rationale: elevated vol means larger intraday noise relative to signal;
      shorter holds reduce the probability of being stopped out by noise rather
      than signal.
    - Tighten volume_multiplier for entry confirmation where applicable.
    """
    import copy as _copy
    p = _copy.deepcopy(params)

    if market_state == "Crisis":
        # Disable 12-1 month momentum gate
        if "momentum_gate_active" in p:
            p["momentum_gate_active"] = False
            rule_log.append(
                "[MarketState=Crisis] momentum_gate_active=False — "
                "12-1m gate disabled; gate is calibrated for trending markets, not crashes"
            )
        # Cap holding days to 5 across all strategies
        if p.get("max_holding_days", 999) > 5:
            p["max_holding_days"] = 5
            rule_log.append(
                f"[MarketState=Crisis] max_holding_days capped at 5 — "
                f"holding > 5 days through a crash risks full reversal whipsaw"
            )

    elif market_state == "High-Volatility":
        # Cap holding days to 10 across all strategies
        if p.get("max_holding_days", 999) > 10:
            p["max_holding_days"] = 10
            rule_log.append(
                f"[MarketState=High-Volatility] max_holding_days capped at 10 — "
                f"elevated vol shortens the signal-to-noise window"
            )

    return p, rule_log


class StrategySelector:
    def __init__(self, llm_client: callable, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def select(self, ticker: str, regime: dict,
               ohlcv_features: dict, macro: dict,
               ticker_verdict: dict | None = None,
               market_state: str = "Normal") -> dict:
        """
        Deterministically compute strategy parameters, then call LLM twice:
          1. Alpha hypothesis: does the LLM agree with the regime-rule strategy?
             If it disagrees, the disagreement is logged as a signal for the trader.
          2. Reasoning: plain-English explanation of the final params.

        market_state : "Normal" | "High-Volatility" | "Crisis"
            Portfolio-level market environment from MarketStateDetector.
            Crisis  → disable 12-1m momentum gate; cap max_holding_days at 5.
            High-Vol → cap max_holding_days at 10; tighten entry thresholds.
        """
        regime_label = regime.get("regime", "Neutral")
        atr_pct      = float(regime.get("atr_pct", 0.02))
        strategy     = _route_strategy(regime_label, atr_pct)

        # Joint crisis override: when macro signals a systemic panic
        # (market ATR extreme AND multiple active news risks with bearish bias),
        # force AlphaCombined with Crisis-tight params regardless of per-ticker
        # regime. This is the circuit-breaker for system-wide risk-off events.
        joint_crisis = bool(macro.get("joint_crisis", False))
        if joint_crisis:
            strategy     = "AlphaCombined"
            regime_label = "Crisis"  # treat ticker as crisis for downstream param tuning

        base_params  = copy.deepcopy(_STRATEGY_TO_BASE[strategy])

        hurst              = float(regime.get("hurst", 0.5))
        vol_ratio          = float((ohlcv_features or {}).get("volume_ratio_30d", 1.0))
        rsi                = float((ohlcv_features or {}).get("rsi_14", 50.0))
        ret_20d            = float((ohlcv_features or {}).get("return_20d", 0.0))
        pead_signal_recent = float((ohlcv_features or {}).get("pead_signal_recent", 0.0))

        # ── Deterministic parameter computation ───────────────────────────────
        if strategy == "Momentum":
            adjusted_params, rule_log = _compute_momentum_params(hurst, atr_pct, vol_ratio)
        elif strategy == "VolatilityBreakout":
            adjusted_params, rule_log = _compute_volatility_breakout_params(atr_pct, hurst)
        elif strategy == "AlphaCombined":
            adjusted_params, rule_log = _compute_alpha_combined_params(atr_pct, hurst, regime_label)
        elif strategy == "MLSignal":
            adjusted_params, rule_log = _compute_ml_signal_params(atr_pct, regime_label)
        elif strategy == "EventDriven":
            adjusted_params, rule_log = _compute_event_driven_params(atr_pct, pead_signal_recent)
        elif strategy == "Mean-Reversion":
            adjusted_params, rule_log = _compute_mean_reversion_params(atr_pct)
        elif strategy == "PairTrading":
            adjusted_params, rule_log = _compute_pair_trading_params(hurst, atr_pct)
        else:
            # Unknown strategy name — should not happen under current routing,
            # but fall back to Momentum rather than silently mis-dispatching.
            adjusted_params, rule_log = _compute_momentum_params(hurst, atr_pct, vol_ratio)
            rule_log.append(f"unknown strategy '{strategy}' — fell back to Momentum")

        if joint_crisis:
            rule_log.append(
                "JOINT CRISIS override: market ATR > 6% AND bearish macro with "
                "multiple active risks — forcing AlphaCombined/Crisis params"
            )

        # ── Regime-specific overrides ─────────────────────────────────────────
        # Crisis downgrade guard: Momentum is structurally wrong in panic regimes
        # (trend-following buys into collapsing tapes). If anything hand-routes
        # Momentum into a Crisis regime, force a downgrade to VolatilityBreakout
        # — the correct crisis posture (wait for directional expansion, not dip-buy).
        if regime_label == "Crisis" and strategy == "Momentum":
            strategy        = "VolatilityBreakout"
            base_params     = copy.deepcopy(_STRATEGY_TO_BASE[strategy])
            adjusted_params, rule_log_vb = _compute_volatility_breakout_params(atr_pct, hurst)
            rule_log.append("Crisis downgrade: Momentum -> VolatilityBreakout")
            rule_log.extend(rule_log_vb)

        # Trending-Down: tighten entry to only buy deep oversold, reduce hold
        if regime_label == "Trending-Down" and strategy == "Mean-Reversion":
            adjusted_params["rsi_entry_threshold"] = min(adjusted_params.get("rsi_entry_threshold", 30), 25)
            adjusted_params["max_holding_days"]    = min(adjusted_params.get("max_holding_days", 10), 8)
            rule_log.append("Trending-Down override: rsi_entry_threshold <= 25, max_holding_days <= 8")

        # ── Market-state adaptive overrides ──────────────────────────────────
        # Applied on top of regime overrides — portfolio-level environment takes
        # precedence over per-ticker regime for holding-period risk management.
        adjusted_params, rule_log = _apply_market_state_overrides(
            adjusted_params, rule_log, strategy, market_state
        )

        print(f"  [Strategy] {ticker}: {regime_label} -> {strategy} | "
              f"Hurst={hurst:.3f} ATR%={atr_pct:.2%} VolRatio={vol_ratio:.2f}")
        for r in rule_log:
            print(f"    rule: {r}")

        # ── LLM alpha hypothesis ───────────────────────────────────────────────
        # The LLM reviews the regime-rule selection against news/macro context and
        # either confirms or suggests a different strategy class. This is the only
        # place where the LLM contributes alpha signal (not just cosmetic explanation).
        llm_hypothesis = self._get_hypothesis(
            ticker, strategy, regime_label, hurst, atr_pct, vol_ratio,
            rsi, ret_20d, macro, ticker_verdict,
        )

        # ── LLM for explanation only ──────────────────────────────────────────
        reasoning = self._get_reasoning(
            ticker, strategy, regime_label, hurst, atr_pct, vol_ratio,
            adjusted_params, macro,
        )

        return {
            "ticker":          ticker,
            "strategy":        strategy,
            "regime":          regime_label,
            "base_params":     base_params,
            "adjusted_params": adjusted_params,
            "llm_adjustments": rule_log,   # algo rules, not LLM decisions
            "llm_hypothesis":  llm_hypothesis,  # LLM strategy class vote + reasoning
            "reasoning":       reasoning,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _get_hypothesis(
        self, ticker, regime_rule_strategy, regime_label, hurst, atr_pct,
        vol_ratio, rsi, ret_20d, macro, ticker_verdict,
    ) -> dict:
        """
        Ask the LLM whether it agrees with the regime-rule strategy selection.
        Returns a dict with keys: agree (bool), suggested (str|None), reason (str|None).
        When the LLM disagrees, the suggested strategy class and reason are surfaced
        in the report as an alpha hypothesis signal for the trader to consider.
        """
        verdict  = (ticker_verdict or {}).get("verdict", "watch")
        reas_str = (ticker_verdict or {}).get("reasoning", "No news context available.")
        prompt   = _HYPOTHESIS_PROMPT.format(
            ticker               = ticker,
            regime               = regime_label,
            hurst                = hurst,
            atr_pct              = atr_pct,
            ret_20d              = ret_20d,
            rsi                  = rsi,
            vol_ratio            = vol_ratio,
            market_bias          = macro.get("market_bias", "neutral"),
            news_verdict         = verdict.upper(),
            news_reasoning       = reas_str[:300],
            regime_rule_strategy = regime_rule_strategy,
        )
        try:
            raw = (self.llm_client(prompt) or "").strip()
            # Parse: "VERDICT: AGREE" or "VERDICT: DISAGREE | SUGGESTED: X | REASON: Y"
            if raw.upper().startswith("VERDICT: AGREE"):
                result = {"agree": True, "suggested": None, "reason": None}
            elif "DISAGREE" in raw.upper():
                suggested = None
                reason    = None
                for part in raw.split("|"):
                    part = part.strip()
                    if part.upper().startswith("SUGGESTED:"):
                        candidate = part.split(":", 1)[1].strip()
                        # Validate it's a known strategy class
                        if candidate in _STRATEGY_TO_BASE:
                            suggested = candidate
                    elif part.upper().startswith("REASON:"):
                        reason = part.split(":", 1)[1].strip()
                result = {"agree": False, "suggested": suggested, "reason": reason}
                if suggested and suggested != regime_rule_strategy:
                    print(f"  [LLM Hypothesis] {ticker}: disagrees → suggests {suggested} | {reason}")
            else:
                # Unparseable — treat as agree to avoid false signals
                result = {"agree": True, "suggested": None, "reason": raw[:200]}
        except Exception:
            result = {"agree": True, "suggested": None, "reason": None}
        return result

    def _get_reasoning(self, ticker, strategy, regime_label, hurst, atr_pct,
                       vol_ratio, params, macro) -> str:
        params_block = "\n".join(f"  {k}: {v}" for k, v in params.items())
        prompt = _REASONING_PROMPT.format(
            ticker       = ticker,
            strategy     = strategy,
            regime       = regime_label,
            params_block = params_block,
            hurst        = hurst,
            atr_pct      = atr_pct,
            vol_ratio    = vol_ratio,
            market_bias  = macro.get("market_bias", "neutral"),
        )
        try:
            raw = self.llm_client(prompt)
            return raw.strip() if raw.strip() else "Parameters set by regime-based algorithm."
        except Exception:
            return "Parameters set by regime-based algorithm."

    @staticmethod
    def _format_ohlcv(feats: dict) -> str:
        if not feats:
            return "  No OHLCV data."
        return "\n".join(
            f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in feats.items()
        )


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ollama, os, json as _json, sys
    from dotenv import load_dotenv; load_dotenv()
    from Stage1DataCollector import Stage1DataCollector
    from news_summarizer import NewsSummarizer
    from macro_screener import MacroScreener
    from ticker_screener import TickerScreener
    from ohlcv_fetcher import OHLCVFetcher
    from regime_classifier import RegimeClassifier
    from datetime import datetime, timedelta

    date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    def llm(prompt):
        resp = ollama.chat(model="qwen3:14b",
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.0})
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    collector = Stage1DataCollector(api_key=os.getenv("BENZINGA_API"), cache_dir="data/cache")
    articles  = collector.collect(date)
    summary   = NewsSummarizer(llm_client=llm, window_days=7).summarize(articles, as_of_date=date)
    macro     = MacroScreener(llm_client=llm).screen(summary)

    screener  = TickerScreener(llm_client=llm)
    top50     = screener.prefilter(articles)
    fetcher   = OHLCVFetcher()
    ohlcv_raw = fetcher.fetch(top50["ticker"].head(5).tolist())
    ohlcv     = {t: fetcher.compute_features(df) for t, df in ohlcv_raw.items() if df is not None}

    clf       = RegimeClassifier()
    sel       = StrategySelector(llm_client=llm, verbose=True)

    for ticker, feats in ohlcv.items():
        regime = clf.classify(ticker, ohlcv_raw[ticker])
        result = sel.select(ticker, regime, feats, macro)
        print(f"\n{'='*60}")
        print(_json.dumps(result, indent=2))
