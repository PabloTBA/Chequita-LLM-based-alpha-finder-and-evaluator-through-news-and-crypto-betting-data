"""
Tests for StrategySelector.
Template mapping tests need no LLM.
LLM behavior tests use mocked llm_client.
"""
import copy
import json
import pytest
from unittest.mock import MagicMock

from strategy_selector import StrategySelector

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOMENTUM_REGIME = {
    "ticker": "AAPL", "hurst": 0.68, "atr_pct": 0.022, "regime": "Trending"
}
MEAN_REV_REGIME = {
    "ticker": "AAPL", "hurst": 0.32, "atr_pct": 0.018, "regime": "Mean-Reverting"
}
HIGH_VOL_REGIME = {
    "ticker": "AAPL", "hurst": 0.50, "atr_pct": 0.045, "regime": "High-Volatility"
}
LOW_VOL_REGIME = {
    "ticker": "AAPL", "hurst": 0.50, "atr_pct": 0.008, "regime": "Low-Volatility"
}
NEUTRAL_REGIME = {
    "ticker": "AAPL", "hurst": 0.50, "atr_pct": 0.020, "regime": "Neutral"
}

OHLCV = {
    "return_20d": 0.05, "rsi_14": 55.0, "atr_14": 3.2,
    "atr_pct": 0.014, "52w_high_prox": 0.95,
    "52w_low_prox": 1.4, "volume_ratio_30d": 1.2,
}

MACRO = {
    "favored_sectors": ["Technology"], "avoid_sectors": [],
    "active_macro_risks": ["tariff"], "market_bias": "bullish",
    "reasoning": "Tech strong.",
}

REQUIRED_KEYS = {
    "ticker", "strategy", "regime", "base_params",
    "adjusted_params", "llm_adjustments", "reasoning",
}

MOMENTUM_BASE_KEYS    = {"entry_lookback", "volume_multiplier", "trailing_stop_atr",
                          "ma_exit_period", "stop_loss_atr", "max_holding_days"}
MEAN_REV_BASE_KEYS    = {"rsi_entry_threshold", "rsi_exit_threshold", "bb_period",
                          "bb_std", "stop_loss_atr", "max_holding_days"}

VALID_LLM_RESPONSE = json.dumps({
    "adjusted_params": {"entry_lookback": 30},
    "llm_adjustments": ["Changed entry_lookback from 20 to 30 — strong trend context"],
    "reasoning": "Extended lookback suits current macro momentum.",
})


def make_llm(response: str = VALID_LLM_RESPONSE) -> MagicMock:
    return MagicMock(return_value=response)


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_select_returns_all_required_keys(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert REQUIRED_KEYS == set(result.keys())

    def test_ticker_matches_input(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("NVDA", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["ticker"] == "NVDA"

    def test_regime_matches_input(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["regime"] == "Trending"


# ── Cycle 2: template mapping — Momentum ─────────────────────────────────────

class TestMomentumMapping:
    def test_trending_maps_to_momentum(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["strategy"] == "Momentum"

    def test_momentum_base_params_has_correct_keys(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert MOMENTUM_BASE_KEYS == set(result["base_params"].keys())

    def test_momentum_base_params_prd_defaults(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        bp     = result["base_params"]
        assert bp["entry_lookback"]    == 20
        assert bp["volume_multiplier"] == 1.5
        assert bp["trailing_stop_atr"] == 2.0
        assert bp["ma_exit_period"]    == 10
        assert bp["stop_loss_atr"]     == 1.5
        assert bp["max_holding_days"]  == 20


# ── Cycle 3: template mapping — Mean-Reversion ───────────────────────────────

class TestMeanReversionMapping:
    def test_mean_reverting_maps_to_mean_reversion(self):
        llm    = make_llm(json.dumps({
            "adjusted_params": {}, "llm_adjustments": [], "reasoning": "ok",
        }))
        sel    = StrategySelector(llm_client=llm)
        result = sel.select("AAPL", MEAN_REV_REGIME, OHLCV, MACRO)
        assert result["strategy"] == "Mean-Reversion"

    def test_mean_reversion_base_params_has_correct_keys(self):
        llm    = make_llm(json.dumps({
            "adjusted_params": {}, "llm_adjustments": [], "reasoning": "ok",
        }))
        sel    = StrategySelector(llm_client=llm)
        result = sel.select("AAPL", MEAN_REV_REGIME, OHLCV, MACRO)
        assert MEAN_REV_BASE_KEYS == set(result["base_params"].keys())

    def test_mean_reversion_base_params_prd_defaults(self):
        llm    = make_llm(json.dumps({
            "adjusted_params": {}, "llm_adjustments": [], "reasoning": "ok",
        }))
        sel    = StrategySelector(llm_client=llm)
        result = sel.select("AAPL", MEAN_REV_REGIME, OHLCV, MACRO)
        bp     = result["base_params"]
        assert bp["rsi_entry_threshold"] == 30
        assert bp["rsi_exit_threshold"]  == 55
        assert bp["bb_period"]           == 20
        assert bp["bb_std"]              == 2.0
        assert bp["stop_loss_atr"]       == 1.5
        assert bp["max_holding_days"]    == 10


# ── Cycle 4: base_params immutability ────────────────────────────────────────

class TestBaseParamsImmutability:
    def test_base_params_not_mutated_by_llm(self):
        """LLM changes entry_lookback — base_params must still show 20."""
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["base_params"]["entry_lookback"] == 20
        assert result["adjusted_params"]["entry_lookback"] == 30


# ── Cycle 5: LLM call budget ─────────────────────────────────────────────────

class TestLLMCallBudget:
    def test_llm_called_exactly_once(self):
        llm    = make_llm()
        sel    = StrategySelector(llm_client=llm)
        sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert llm.call_count == 1

    def test_prompt_contains_regime_label(self):
        captured = {}
        def llm(prompt):
            captured["prompt"] = prompt
            return VALID_LLM_RESPONSE
        sel = StrategySelector(llm_client=llm)
        sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert "Trending" in captured["prompt"]

    def test_prompt_contains_ticker(self):
        captured = {}
        def llm(prompt):
            captured["prompt"] = prompt
            return VALID_LLM_RESPONSE
        sel = StrategySelector(llm_client=llm)
        sel.select("NVDA", MOMENTUM_REGIME, OHLCV, MACRO)
        assert "NVDA" in captured["prompt"]


# ── Cycle 6: LLM adjustments merged ──────────────────────────────────────────

class TestLLMAdjustments:
    def test_adjusted_params_contains_llm_change(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["adjusted_params"]["entry_lookback"] == 30

    def test_adjusted_params_inherits_unchanged_base_keys(self):
        """Keys not changed by LLM still present in adjusted_params."""
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert "stop_loss_atr" in result["adjusted_params"]
        assert result["adjusted_params"]["stop_loss_atr"] == 1.5

    def test_llm_adjustments_is_list_of_strings(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert isinstance(result["llm_adjustments"], list)
        assert all(isinstance(s, str) for s in result["llm_adjustments"])

    def test_reasoning_is_non_empty_string(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


# ── Cycle 7: malformed JSON fallback ─────────────────────────────────────────

class TestMalformedJSONFallback:
    def test_bad_json_adjusted_params_equals_base(self):
        sel    = StrategySelector(llm_client=make_llm("not json"))
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["adjusted_params"] == result["base_params"]

    def test_bad_json_llm_adjustments_empty(self):
        sel    = StrategySelector(llm_client=make_llm("not json"))
        result = sel.select("AAPL", MOMENTUM_REGIME, OHLCV, MACRO)
        assert result["llm_adjustments"] == []


# ── Cycle 8: fallback regime mappings ────────────────────────────────────────

class TestFallbackRegimeMappings:
    def test_high_volatility_maps_to_momentum(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", HIGH_VOL_REGIME, OHLCV, MACRO)
        assert result["strategy"] == "Momentum"

    def test_low_volatility_maps_to_mean_reversion(self):
        llm    = make_llm(json.dumps({
            "adjusted_params": {}, "llm_adjustments": [], "reasoning": "ok",
        }))
        sel    = StrategySelector(llm_client=llm)
        result = sel.select("AAPL", LOW_VOL_REGIME, OHLCV, MACRO)
        assert result["strategy"] == "Mean-Reversion"

    def test_neutral_maps_to_momentum(self):
        sel    = StrategySelector(llm_client=make_llm())
        result = sel.select("AAPL", NEUTRAL_REGIME, OHLCV, MACRO)
        assert result["strategy"] == "Momentum"
