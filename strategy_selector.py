"""
StrategySelector
================
Maps a regime label to a strategy template (pure mapping), then calls the LLM
once to adjust parameters within that template.  LLM adjustments are logged
alongside the hard regime signal so every change is auditable.

Regime → Strategy mapping
--------------------------
    Trending         → Momentum
    High-Volatility  → Momentum
    Mean-Reverting   → Mean-Reversion
    Low-Volatility   → Mean-Reversion
    Neutral          → Momentum  (default)

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
    "entry_lookback":    20,   # days: entry on close > N-day high
    "volume_multiplier": 1.5,  # volume must be > N × 20-day avg
    "trailing_stop_atr": 2.0,  # trailing stop at N × ATR below peak
    "ma_exit_period":    10,   # exit if close < N-day MA
    "stop_loss_atr":     1.5,  # hard stop at N × ATR below entry
    "max_holding_days":  20,   # forced exit after N trading days
}

MEAN_REVERSION_BASE: dict = {
    "rsi_entry_threshold": 30,   # enter when RSI(14) < N
    "rsi_exit_threshold":  55,   # exit when RSI(14) > N
    "bb_period":           20,   # Bollinger Band period
    "bb_std":              2.0,  # Bollinger Band sigma
    "stop_loss_atr":       1.5,  # hard stop at N × ATR below entry
    "max_holding_days":    10,   # forced exit after N trading days
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

_PROMPT_TEMPLATE = """\
You are a quantitative strategist. Given the regime classification and market \
context for {ticker}, review the base strategy parameters and suggest any \
adjustments. Only change parameters that are clearly warranted by the evidence.

Ticker:   {ticker}
Regime:   {regime}
Strategy: {strategy}
Hurst:    {hurst:.3f}
ATR/price:{atr_pct:.3%}

OHLCV features:
{ohlcv_block}

Macro context:
  Bias:    {market_bias}
  Favored: {favored}
  Risks:   {risks}

Base parameters (PRD defaults):
{base_params}

Respond ONLY with valid JSON:
{{
  "adjusted_params": {{"param_name": new_value, ...}},
  "llm_adjustments": ["Changed X from A to B because ...", ...],
  "reasoning": "one or two sentence summary"
}}
Only include params in adjusted_params that you are actually changing.
"""


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
        Map regime to strategy template, call LLM once for parameter adjustment.

        Parameters
        ----------
        ticker        : str
        regime        : RegimeClassifier.classify() output
        ohlcv_features: OHLCVFetcher.compute_features() output
        macro         : MacroScreener.screen() output

        Returns
        -------
        dict with keys: ticker, strategy, regime, base_params,
                        adjusted_params, llm_adjustments, reasoning
        """
        regime_label = regime.get("regime", "Neutral")
        strategy     = _REGIME_TO_STRATEGY.get(regime_label, "Momentum")
        base_params  = copy.deepcopy(_STRATEGY_TO_BASE[strategy])

        print(f"  [LLM] StrategySelector: {ticker} ({regime_label} → {strategy})...")
        self._log(f"[StrategySelector] {ticker}: {regime_label} → {strategy}")

        prompt = _PROMPT_TEMPLATE.format(
            ticker      = ticker,
            regime      = regime_label,
            strategy    = strategy,
            hurst       = regime.get("hurst", 0.5),
            atr_pct     = regime.get("atr_pct", 0.02),
            ohlcv_block = self._format_ohlcv(ohlcv_features),
            market_bias = macro.get("market_bias", "neutral"),
            favored     = ", ".join(macro.get("favored_sectors", [])),
            risks       = ", ".join(macro.get("active_macro_risks", [])),
            base_params = json.dumps(base_params, indent=2),
        )

        raw = self.llm_client(prompt)
        print(f"  [LLM] StrategySelector: {ticker} done")
        self._log(f"[StrategySelector] LLM response: {raw[:300]}")

        adjusted_params, llm_adjustments, reasoning = self._parse(raw, base_params)

        return {
            "ticker":          ticker,
            "strategy":        strategy,
            "regime":          regime_label,
            "base_params":     base_params,
            "adjusted_params": adjusted_params,
            "llm_adjustments": llm_adjustments,
            "reasoning":       reasoning,
        }

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_ohlcv(feats: dict) -> str:
        if not feats:
            return "  No OHLCV data."
        return "\n".join(
            f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in feats.items()
        )

    @staticmethod
    def _parse(raw: str, base_params: dict) -> tuple[dict, list[str], str]:
        """Parse LLM response; fall back to base_params on any parse failure."""
        try:
            data            = json.loads(raw)
            overrides       = data.get("adjusted_params", {})
            llm_adjustments = data.get("llm_adjustments", [])
            reasoning       = data.get("reasoning", "")

            # Merge: start from base, apply only valid overrides
            adjusted = copy.deepcopy(base_params)
            for k, v in overrides.items():
                if k in adjusted:          # only touch known params
                    adjusted[k] = v

            if not isinstance(llm_adjustments, list):
                llm_adjustments = []
            llm_adjustments = [s for s in llm_adjustments if isinstance(s, str)]

            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = "No reasoning provided."

            return adjusted, llm_adjustments, reasoning

        except (json.JSONDecodeError, ValueError):
            return copy.deepcopy(base_params), [], "LLM parse failed — using base parameters."


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
        resp = ollama.chat(model="qwen3:8b",
                           messages=[{"role": "user", "content": prompt}],
                           format="json")
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
