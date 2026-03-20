"""
Tests for MacroScreener.
LLM is always mocked — no real Ollama calls.
"""
import json
import pytest
from unittest.mock import MagicMock

from macro_screener import MacroScreener

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SUMMARY = {
    "window_start":  "2026-03-11",
    "window_end":    "2026-03-18",
    "article_count": 80,
    "summary":       "Tech earnings strong, Fed held rates, oil supply tightening.",
    "top_themes":    ["AI spending", "rate pause", "energy supply"],
    "market_bias":   "bullish",
    "key_risks":     ["tariff escalation", "CPI surprise"],
}

NO_DATA_SUMMARY = {
    "window_start":  "2026-03-11",
    "window_end":    "2026-03-18",
    "article_count": 0,
    "summary":       "No data available.",
    "top_themes":    [],
    "market_bias":   "neutral",
    "key_risks":     [],
}

REQUIRED_KEYS = {"favored_sectors", "avoid_sectors", "active_macro_risks", "market_bias", "reasoning"}

VALID_LLM_RESPONSE = json.dumps({
    "favored_sectors":    ["Technology", "Energy"],
    "avoid_sectors":      ["Real Estate", "Utilities"],
    "active_macro_risks": ["Fed rate decision", "tariff escalation"],
    "market_bias":        "bullish",
    "reasoning":          "Tech earnings strong and energy supply constraints favour these sectors.",
})

NEUTRAL_DEFAULT = {
    "favored_sectors":    [],
    "avoid_sectors":      [],
    "active_macro_risks": [],
    "market_bias":        "neutral",
    "reasoning":          "Insufficient data to form a macro view.",
}


def make_llm(response: str = VALID_LLM_RESPONSE) -> MagicMock:
    m = MagicMock(return_value=response)
    return m


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_screen_returns_all_required_keys(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(VALID_SUMMARY)
        assert REQUIRED_KEYS == set(result.keys())

    def test_favored_sectors_is_list_of_strings(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(VALID_SUMMARY)
        assert isinstance(result["favored_sectors"], list)
        assert all(isinstance(s, str) for s in result["favored_sectors"])

    def test_avoid_sectors_is_list_of_strings(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(VALID_SUMMARY)
        assert isinstance(result["avoid_sectors"], list)
        assert all(isinstance(s, str) for s in result["avoid_sectors"])

    def test_active_macro_risks_is_list_of_strings(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(VALID_SUMMARY)
        assert isinstance(result["active_macro_risks"], list)
        assert all(isinstance(s, str) for s in result["active_macro_risks"])

    def test_market_bias_is_valid_value(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(VALID_SUMMARY)
        assert result["market_bias"] in {"bullish", "bearish", "neutral"}

    def test_reasoning_is_string(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(VALID_SUMMARY)
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


# ── Cycle 2: LLM call budget ──────────────────────────────────────────────────

class TestLLMCallBudget:
    def test_llm_called_exactly_once(self):
        llm      = make_llm()
        screener = MacroScreener(llm_client=llm)
        screener.screen(VALID_SUMMARY)
        assert llm.call_count == 1

    def test_prompt_contains_summary_text(self):
        captured = {}
        def llm(prompt):
            captured["prompt"] = prompt
            return VALID_LLM_RESPONSE
        screener = MacroScreener(llm_client=llm)
        screener.screen(VALID_SUMMARY)
        assert VALID_SUMMARY["summary"] in captured["prompt"]


# ── Cycle 3: no-data path ─────────────────────────────────────────────────────

class TestNoDataPath:
    def test_no_data_summary_skips_llm(self):
        llm      = make_llm()
        screener = MacroScreener(llm_client=llm)
        screener.screen(NO_DATA_SUMMARY)
        assert llm.call_count == 0

    def test_no_data_summary_returns_neutral_default(self):
        screener = MacroScreener(llm_client=make_llm())
        result   = screener.screen(NO_DATA_SUMMARY)
        assert result["market_bias"]   == "neutral"
        assert result["favored_sectors"]    == []
        assert result["avoid_sectors"]      == []
        assert result["active_macro_risks"] == []


# ── Cycle 4: malformed JSON fallback ─────────────────────────────────────────

class TestMalformedJSON:
    def test_bad_json_returns_neutral_default_no_crash(self):
        screener = MacroScreener(llm_client=make_llm("this is not json"))
        result   = screener.screen(VALID_SUMMARY)
        assert result["market_bias"]        == "neutral"
        assert result["favored_sectors"]    == []
        assert result["avoid_sectors"]      == []
        assert result["active_macro_risks"] == []

    def test_missing_keys_in_llm_response_uses_defaults(self):
        partial = json.dumps({"favored_sectors": ["Tech"]})  # missing other keys
        screener = MacroScreener(llm_client=make_llm(partial))
        result   = screener.screen(VALID_SUMMARY)
        assert "avoid_sectors"      in result
        assert "active_macro_risks" in result
        assert "market_bias"        in result
        assert "reasoning"          in result

    def test_invalid_market_bias_value_corrected_to_neutral(self):
        bad = json.dumps({
            "favored_sectors":    ["Tech"],
            "avoid_sectors":      [],
            "active_macro_risks": [],
            "market_bias":        "very_bullish",   # invalid value
            "reasoning":          "...",
        })
        screener = MacroScreener(llm_client=make_llm(bad))
        result   = screener.screen(VALID_SUMMARY)
        assert result["market_bias"] in {"bullish", "bearish", "neutral"}
