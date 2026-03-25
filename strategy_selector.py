"""
StrategySelector
================
Maps a regime label to a strategy template, then adjusts parameters via a
deterministic rule-based algorithm (no LLM numeric decisions).  The LLM is
called once only to produce a plain-English explanation of the final params
for the report — it has no influence on the numbers.

Regime -> Strategy mapping
--------------------------
    Trending         -> Momentum
    High-Volatility  -> VolatilityBreakout  (BB squeeze → expansion alpha)
    Mean-Reverting   -> Mean-Reversion
    Low-Volatility   -> Mean-Reversion
    Neutral          -> Momentum  (default)

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
    ma_exit_period    : 10 (fixed)

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
    "entry_lookback":    10,
    "volume_multiplier": 1.2,
    "trailing_stop_atr": 2.0,
    "ma_exit_period":    20,    # widened from 10 — 10d MA fires too early, cutting winners short
    "stop_loss_atr":     1.5,
    "max_holding_days":  20,
    "momentum_lookback": 252,   # 12-1 month momentum gate: only enter if 11m return (skip last month) > 0
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

_REGIME_TO_STRATEGY: dict[str, str] = {
    # Directional trend regimes
    "Trending-Up":      "Momentum",            # follow the trend long
    "Trending-Down":    "AlphaCombined",        # multi-signal: idiosyncratic reversion + volume exhaustion
    # Volatility regimes
    "High-Volatility":  "VolatilityBreakout",  # squeeze → expansion alpha
    "Low-Volatility":   "AlphaCombined",        # cross-sectional MR fires well in quiet markets
    "Crisis":           "AlphaCombined",        # extreme moves: use alpha signal with tight stops
    # Statistical regimes
    "Mean-Reverting":   "AlphaCombined",        # primary regime for multi-factor MR
    "Neutral":          "AlphaCombined",        # default: cross-sectional alpha more robust than RSI+BB
    # Exogenous event regime
    "Event-Driven":     "AlphaCombined",        # post-event idiosyncratic drift + volume exhaustion
    # Legacy label — kept for backward compatibility with any cached data
    "Trending":         "Momentum",
}

_STRATEGY_TO_BASE: dict[str, dict] = {
    "Momentum":           MOMENTUM_BASE,
    "Mean-Reversion":     MEAN_REVERSION_BASE,
    "VolatilityBreakout": VOLATILITY_BREAKOUT_BASE,
    "AlphaCombined":      ALPHA_COMBINED_BASE,
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
3. VolatilityBreakout  — BB squeeze → expansion + ATR surge. Edge: compressed volatility preceding directional move.
4. AlphaCombined       — Cross-sectional multi-factor signal (CS-MR + residual + vol-spike + momentum). Edge: diversified alpha, higher trade frequency, market-neutral component.

REGIME RULE SELECTED: {regime_rule_strategy}

Do you AGREE or DISAGREE with this selection given the news context and current conditions?

Respond in EXACTLY this format (one line only):
VERDICT: AGREE
or
VERDICT: DISAGREE | SUGGESTED: [Momentum|Mean-Reversion|VolatilityBreakout|AlphaCombined] | REASON: [one sentence]
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

    # Trending-Down: want stronger reversion signal before buying the dip
    if regime_label == "Trending-Down":
        p["alpha_threshold"]   = 0.55   # require stronger signal in downtrend
        p["max_holding_days"]  = 7
        rules.append(
            "Trending-Down: alpha_threshold=0.55, max_holding_days=7 "
            "(only trade strongest idiosyncratic bounce signals)"
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


class StrategySelector:
    def __init__(self, llm_client: callable, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def select(self, ticker: str, regime: dict,
               ohlcv_features: dict, macro: dict,
               ticker_verdict: dict | None = None) -> dict:
        """
        Deterministically compute strategy parameters, then call LLM twice:
          1. Alpha hypothesis: does the LLM agree with the regime-rule strategy?
             If it disagrees, the disagreement is logged as a signal for the trader.
          2. Reasoning: plain-English explanation of the final params.
        """
        regime_label = regime.get("regime", "Neutral")
        strategy     = _REGIME_TO_STRATEGY.get(regime_label, "Momentum")
        base_params  = copy.deepcopy(_STRATEGY_TO_BASE[strategy])

        hurst     = float(regime.get("hurst", 0.5))
        atr_pct   = float(regime.get("atr_pct", 0.02))
        vol_ratio = float((ohlcv_features or {}).get("volume_ratio_30d", 1.0))
        rsi       = float((ohlcv_features or {}).get("rsi_14", 50.0))
        ret_20d   = float((ohlcv_features or {}).get("return_20d", 0.0))

        # ── Deterministic parameter computation ───────────────────────────────
        if strategy == "Momentum":
            adjusted_params, rule_log = _compute_momentum_params(hurst, atr_pct, vol_ratio)
        elif strategy == "VolatilityBreakout":
            adjusted_params, rule_log = _compute_volatility_breakout_params(atr_pct, hurst)
        elif strategy == "AlphaCombined":
            adjusted_params, rule_log = _compute_alpha_combined_params(atr_pct, hurst, regime_label)
        else:
            adjusted_params, rule_log = _compute_mean_reversion_params(atr_pct)

        # ── Regime-specific overrides (Momentum only — AlphaCombined handles its own) ──
        # Crisis: tighten stops and shorten max hold to reduce exposure in panic conditions
        if regime_label == "Crisis" and strategy == "Momentum":
            adjusted_params["stop_loss_atr"]    = min(adjusted_params.get("stop_loss_atr", 1.5), 1.0)
            adjusted_params["max_holding_days"] = min(adjusted_params.get("max_holding_days", 10), 5)
            rule_log.append("Crisis override: stop_loss_atr <= 1.0, max_holding_days <= 5")

        # Event-Driven: target short gap-fill window (5–7 days post-earnings)
        if regime_label == "Event-Driven" and strategy == "Mean-Reversion":
            adjusted_params["max_holding_days"]      = min(adjusted_params.get("max_holding_days", 10), 7)
            adjusted_params["rsi_entry_threshold"]   = max(adjusted_params.get("rsi_entry_threshold", 30), 35)
            rule_log.append("Event-Driven override: max_holding_days <= 7 (gap-fill window)")

        # Trending-Down: tighten entry to only buy deep oversold, reduce hold
        if regime_label == "Trending-Down" and strategy == "Mean-Reversion":
            adjusted_params["rsi_entry_threshold"] = min(adjusted_params.get("rsi_entry_threshold", 30), 25)
            adjusted_params["max_holding_days"]    = min(adjusted_params.get("max_holding_days", 10), 8)
            rule_log.append("Trending-Down override: rsi_entry_threshold <= 25, max_holding_days <= 8")

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
