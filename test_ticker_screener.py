"""
Tests for TickerScreener.
Pre-filter tests use synthetic DataFrames — no LLM.
Shortlist + screen_tickers tests use mocked LLM.
"""
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from ticker_screener import TickerScreener

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_articles(tickers: list[str], scores: list[float], source: str = "stock_news") -> pd.DataFrame:
    """Build a minimal articles DataFrame with ticker + composite_score columns."""
    return pd.DataFrame({
        "ticker":          tickers,
        "composite_score": scores,
        "headline":        [f"News about {t}" for t in tickers],
        "source":          source,
    })


def make_top50_df(n: int = 50) -> pd.DataFrame:
    tickers = [f"T{i:03d}" for i in range(n)]
    scores  = list(range(n, 0, -1))   # T000 has highest score
    return pd.DataFrame({"ticker": tickers, "composite_score": scores})


MACRO = {
    "favored_sectors":    ["Technology", "Energy"],
    "avoid_sectors":      ["Real Estate"],
    "active_macro_risks": ["tariff escalation"],
    "market_bias":        "bullish",
    "reasoning":          "Tech earnings strong.",
}

OHLCV = {
    "AAPL": {"return_20d": 0.05, "rsi_14": 55.0, "atr_14": 3.2,
             "atr_pct": 0.014, "52w_high_prox": 0.95, "52w_low_prox": 1.4,
             "volume_ratio_30d": 1.2},
}

VALID_SHORTLIST_RESPONSE = json.dumps({
    "tickers": [f"T{i:03d}" for i in range(20)]
})

VALID_VERDICT_RESPONSE = json.dumps({
    "verdict":   "buy",
    "reasoning": "Strong momentum and favored sector.",
})


def make_llm(response: str = VALID_VERDICT_RESPONSE) -> MagicMock:
    return MagicMock(return_value=response)


# ── Cycle 1: prefilter — top 50 cap ──────────────────────────────────────────

class TestPrefilter:
    def test_returns_exactly_50_when_more_available(self):
        screener = TickerScreener(llm_client=make_llm())
        tickers  = [f"T{i:03d}" for i in range(80)]
        scores   = list(range(80, 0, -1))
        articles = {"stock_news": make_articles(tickers, scores), "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result   = screener.prefilter(articles)
        assert len(result) == 50

    def test_returns_all_when_fewer_than_50(self):
        screener = TickerScreener(llm_client=make_llm())
        articles = {"stock_news": make_articles(["AAPL", "MSFT"], [10.0, 5.0]), "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result   = screener.prefilter(articles)
        assert len(result) == 2

    def test_ranked_by_composite_score_descending(self):
        screener = TickerScreener(llm_client=make_llm())
        tickers  = ["LOW", "MID", "HIGH"]
        scores   = [1.0, 5.0, 10.0]
        articles = {"stock_news": make_articles(tickers, scores), "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result   = screener.prefilter(articles)
        assert result.iloc[0]["ticker"] == "HIGH"
        assert result.iloc[-1]["ticker"] == "LOW"

    def test_combines_all_three_sources(self):
        screener   = TickerScreener(llm_client=make_llm())
        stock_df   = make_articles(["AAPL"], [10.0], "stock_news")
        global_df  = make_articles(["MSFT"], [8.0],  "global_news")
        industry_df= make_articles(["NVDA"], [6.0],  "industry_news")
        articles   = {"stock_news": stock_df, "global_news": global_df, "industry_news": industry_df}
        result     = screener.prefilter(articles)
        assert set(result["ticker"]) == {"AAPL", "MSFT", "NVDA"}

    def test_sums_scores_for_same_ticker_across_articles(self):
        """Same ticker appearing in multiple articles → scores summed."""
        screener = TickerScreener(llm_client=make_llm())
        df = make_articles(["AAPL", "AAPL", "MSFT"], [3.0, 4.0, 10.0])
        articles = {"stock_news": df, "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result   = screener.prefilter(articles)
        aapl_score = result.loc[result["ticker"] == "AAPL", "composite_score"].iloc[0]
        assert abs(aapl_score - 7.0) < 1e-9

    def test_excludes_rows_with_empty_ticker(self):
        screener = TickerScreener(llm_client=make_llm())
        df = make_articles(["AAPL", "", None, "MSFT"], [5.0, 3.0, 2.0, 4.0])
        articles = {"stock_news": df, "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result   = screener.prefilter(articles)
        assert "" not in result["ticker"].values
        assert result["ticker"].isna().sum() == 0


# ── Cycle 2: shortlist — LLM picks 20 ────────────────────────────────────────

class TestShortlist:
    def test_returns_exactly_20_tickers(self):
        llm      = make_llm(VALID_SHORTLIST_RESPONSE)
        screener = TickerScreener(llm_client=llm)
        result   = screener.shortlist(make_top50_df(), MACRO, OHLCV)
        assert len(result) == 20

    def test_tickers_are_strings(self):
        llm      = make_llm(VALID_SHORTLIST_RESPONSE)
        screener = TickerScreener(llm_client=llm)
        result   = screener.shortlist(make_top50_df(), MACRO, OHLCV)
        assert all(isinstance(t, str) for t in result)

    def test_llm_called_exactly_once(self):
        llm      = make_llm(VALID_SHORTLIST_RESPONSE)
        screener = TickerScreener(llm_client=llm)
        screener.shortlist(make_top50_df(), MACRO, OHLCV)
        assert llm.call_count == 1

    def test_malformed_json_falls_back_to_top20_from_top50(self):
        screener = TickerScreener(llm_client=make_llm("not json"))
        top50    = make_top50_df(50)
        result   = screener.shortlist(top50, MACRO, OHLCV)
        assert len(result) == 20
        assert set(result) == set(top50.iloc[:20]["ticker"].tolist())

    def test_shortlisted_tickers_subset_of_top50(self):
        llm      = make_llm(VALID_SHORTLIST_RESPONSE)
        screener = TickerScreener(llm_client=llm)
        top50    = make_top50_df(50)
        result   = screener.shortlist(top50, MACRO, OHLCV)
        top50_set = set(top50["ticker"].tolist())
        assert all(t in top50_set for t in result)


# ── Cycle 3: screen_tickers — per-ticker verdicts ────────────────────────────

class TestScreenTickers:
    TICKERS = ["AAPL", "MSFT", "NVDA"]

    def test_returns_dict_for_every_ticker(self):
        screener = TickerScreener(llm_client=make_llm())
        result   = screener.screen_tickers(self.TICKERS, MACRO, OHLCV)
        returned = {r["ticker"] for r in result}
        assert returned == set(self.TICKERS)

    def test_verdict_is_valid_value(self):
        screener = TickerScreener(llm_client=make_llm())
        result   = screener.screen_tickers(self.TICKERS, MACRO, OHLCV)
        for r in result:
            assert r["verdict"] in {"buy", "watch", "avoid"}

    def test_reasoning_is_non_empty_string(self):
        screener = TickerScreener(llm_client=make_llm())
        result   = screener.screen_tickers(self.TICKERS, MACRO, OHLCV)
        for r in result:
            assert isinstance(r["reasoning"], str)
            assert len(r["reasoning"]) > 0

    def test_malformed_json_defaults_to_watch(self):
        screener = TickerScreener(llm_client=make_llm("bad json"))
        result   = screener.screen_tickers(["AAPL"], MACRO, OHLCV)
        assert result[0]["verdict"] == "watch"

    def test_invalid_verdict_value_defaults_to_watch(self):
        bad = json.dumps({"verdict": "strong_buy", "reasoning": "great stock"})
        screener = TickerScreener(llm_client=make_llm(bad))
        result   = screener.screen_tickers(["AAPL"], MACRO, OHLCV)
        assert result[0]["verdict"] == "watch"

    def test_llm_called_once_per_ticker(self):
        llm      = make_llm()
        screener = TickerScreener(llm_client=llm)
        screener.screen_tickers(self.TICKERS, MACRO, OHLCV)
        assert llm.call_count == len(self.TICKERS)
