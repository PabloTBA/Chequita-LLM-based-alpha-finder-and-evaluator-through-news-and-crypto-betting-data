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
    High-Volatility  -> Momentum
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
    "ma_exit_period":    10,
    "stop_loss_atr":     1.5,
    "max_holding_days":  20,
}

MEAN_REVERSION_BASE: dict = {
    "rsi_entry_threshold": 30,
    "rsi_exit_threshold":  55,
    "bb_period":           20,
    "bb_std":              2.0,
    "stop_loss_atr":       1.5,
    "max_holding_days":    10,
}

_REGIME_TO_STRATEGY: dict[str, str] = {
    "Trending":        "Momentum",
    "High-Volatility": "Momentum",
    "Neutral":         "Momentum",
    "Mean-Reverting":  "Mean-Reversion",
    "Low-Volatility":  "Mean-Reversion",
}

_STRATEGY_TO_BASE: dict[str, dict] = {
    "Momentum":       MOMENTUM_BASE,
    "Mean-Reversion": MEAN_REVERSION_BASE,
}

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


class StrategySelector:
    def __init__(self, llm_client: callable, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def select(self, ticker: str, regime: dict,
               ohlcv_features: dict, macro: dict) -> dict:
        """
        Deterministically compute strategy parameters, then call LLM once
        for a plain-English explanation only.
        """
        regime_label = regime.get("regime", "Neutral")
        strategy     = _REGIME_TO_STRATEGY.get(regime_label, "Momentum")
        base_params  = copy.deepcopy(_STRATEGY_TO_BASE[strategy])

        hurst     = float(regime.get("hurst", 0.5))
        atr_pct   = float(regime.get("atr_pct", 0.02))
        vol_ratio = float((ohlcv_features or {}).get("volume_ratio_30d", 1.0))

        # ── Deterministic parameter computation ───────────────────────────────
        if strategy == "Momentum":
            adjusted_params, rule_log = _compute_momentum_params(hurst, atr_pct, vol_ratio)
        else:
            adjusted_params, rule_log = _compute_mean_reversion_params(atr_pct)

        print(f"  [Strategy] {ticker}: {regime_label} -> {strategy} | "
              f"Hurst={hurst:.3f} ATR%={atr_pct:.2%} VolRatio={vol_ratio:.2f}")
        for r in rule_log:
            print(f"    rule: {r}")

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
            "llm_adjustments": rule_log,   # now shows algo rules, not LLM decisions
            "reasoning":       reasoning,
        }

    # ── private ───────────────────────────────────────────────────────────────

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
