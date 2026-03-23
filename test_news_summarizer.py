"""
Tests for NewsSummarizer — TDD, red-green-refactor.

Run unit tests (no Ollama needed):
    python -m pytest test_news_summarizer.py -v

Run integration test (requires local Ollama + Qwen2.5-7B):
    python -m pytest test_news_summarizer.py -v --integration
"""

import json
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call

from news_summarizer import NewsSummarizer


# ── Helpers ───────────────────────────────────────────────────

def _make_articles(dates: list[str], source: str = "Reuters") -> pd.DataFrame:
    """Build a minimal scored DataFrame for testing."""
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date":            d,
            "title":           f"Article {i} on {d}",
            "composite_score": 10 - i,
            "catalyst_type":   "Earnings",
            "publisher":       source,
            "url":             f"http://example.com/{d}/{i}",
            "keywords_matched": "earnings",
        })
    return pd.DataFrame(rows)


def _make_llm_response(**kwargs) -> str:
    """Return a valid JSON LLM response string."""
    base = {
        "summary":     "Markets showed mixed signals with earnings driving individual moves.",
        "top_themes":  ["Fed policy", "Tech earnings", "Oil supply"],
        "market_bias": "neutral",
        "key_risks":   ["tariff escalation", "inflation persistence"],
    }
    base.update(kwargs)
    return json.dumps(base)


# ── Cycle 1: Window guard ─────────────────────────────────────

class TestWindowGuard:
    """NewsSummarizer raises ValueError for out-of-range window_days."""

    def test_window_over_14_raises(self):
        llm = MagicMock()
        with pytest.raises(ValueError, match="14"):
            NewsSummarizer(llm_client=llm, window_days=15)

    def test_window_under_1_raises(self):
        llm = MagicMock()
        with pytest.raises(ValueError, match="1"):
            NewsSummarizer(llm_client=llm, window_days=0)

    def test_window_exactly_7_is_valid(self):
        llm = MagicMock()
        s = NewsSummarizer(llm_client=llm, window_days=7)
        assert s.window_days == 7

    def test_window_exactly_14_is_valid(self):
        llm = MagicMock()
        s = NewsSummarizer(llm_client=llm, window_days=14)
        assert s.window_days == 14

    def test_default_window_is_7(self):
        llm = MagicMock()
        s = NewsSummarizer(llm_client=llm)
        assert s.window_days == 7


# ── Cycle 2: Date filtering ───────────────────────────────────

class TestDateFiltering:
    """Articles outside the window are not sent to the LLM."""

    def test_articles_outside_window_excluded(self):
        """Only articles within window_days reach the LLM prompt."""
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        as_of = "2026-03-18"
        # One article inside (2 days ago), one outside (30 days ago)
        inside_date  = "2026-03-16"
        outside_date = "2026-02-16"

        articles = {
            "stock_news":    _make_articles([inside_date, outside_date]),
            "global_news":   pd.DataFrame(),
            "industry_news": pd.DataFrame(),
        }
        s.summarize(articles, as_of_date=as_of)

        prompt_sent = llm.call_args[0][0]
        assert inside_date  in prompt_sent
        assert outside_date not in prompt_sent

    def test_all_articles_outside_window_returns_no_data(self):
        """If all articles are outside the window, LLM is not called."""
        llm = MagicMock()
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        old_date = "2025-01-01"
        articles = {
            "stock_news":    _make_articles([old_date]),
            "global_news":   pd.DataFrame(),
            "industry_news": pd.DataFrame(),
        }
        result = s.summarize(articles, as_of_date="2026-03-18")

        llm.assert_not_called()
        assert result["article_count"] == 0


# ── Cycle 3: Output contract ──────────────────────────────────

class TestOutputContract:
    """summarize() always returns a dict with the required keys."""

    REQUIRED_KEYS = {"window_start", "window_end", "article_count",
                     "summary", "top_themes", "market_bias", "key_risks"}

    def test_required_keys_present_with_articles(self):
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {
            "stock_news":    _make_articles(["2026-03-17", "2026-03-16"]),
            "global_news":   _make_articles(["2026-03-15"]),
            "industry_news": pd.DataFrame(),
        }
        result = s.summarize(articles, as_of_date="2026-03-18")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_required_keys_present_with_empty_articles(self):
        llm = MagicMock()
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {k: pd.DataFrame() for k in ("stock_news", "global_news", "industry_news")}
        result = s.summarize(articles, as_of_date="2026-03-18")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_top_themes_is_list(self):
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {"stock_news": _make_articles(["2026-03-17"]),
                    "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result = s.summarize(articles, as_of_date="2026-03-18")
        assert isinstance(result["top_themes"], list)

    def test_market_bias_is_valid_value(self):
        llm = MagicMock(return_value=_make_llm_response(market_bias="bullish"))
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {"stock_news": _make_articles(["2026-03-17"]),
                    "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()}
        result = s.summarize(articles, as_of_date="2026-03-18")
        assert result["market_bias"] in ("bullish", "bearish", "neutral")


# ── Cycle 4: No-data path ─────────────────────────────────────

class TestNoDataPath:
    """Empty input produces a sensible default without touching the LLM."""

    def test_empty_all_sources_no_llm_call(self):
        llm = MagicMock()
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {k: pd.DataFrame() for k in ("stock_news", "global_news", "industry_news")}
        result   = s.summarize(articles, as_of_date="2026-03-18")

        llm.assert_not_called()
        assert result["article_count"] == 0
        assert result["summary"] != ""       # always returns something readable
        assert result["top_themes"] == []
        assert result["key_risks"]  == []


# ── Cycle 5: LLM call budget ──────────────────────────────────

class TestLLMCallBudget:
    """LLM is called exactly once per summarize(), combining all sources."""

    def test_llm_called_exactly_once(self):
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {
            "stock_news":    _make_articles(["2026-03-17", "2026-03-16"]),
            "global_news":   _make_articles(["2026-03-15"]),
            "industry_news": _make_articles(["2026-03-14"]),
        }
        s.summarize(articles, as_of_date="2026-03-18")
        assert llm.call_count == 1

    def test_prompt_contains_all_three_sources(self):
        """Combined prompt references all three source types."""
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        articles = {
            "stock_news":    _make_articles(["2026-03-17"]),
            "global_news":   _make_articles(["2026-03-16"]),
            "industry_news": _make_articles(["2026-03-15"]),
        }
        s.summarize(articles, as_of_date="2026-03-18")
        prompt = llm.call_args[0][0]

        assert "stock"    in prompt.lower()
        assert "global"   in prompt.lower() or "macro" in prompt.lower()
        assert "industry" in prompt.lower()

    def test_only_top_5_per_source_in_prompt(self):
        """Prompt is capped to top 5 articles per source to control token count."""
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=14)

        # 10 articles for stock_news, all within window
        dates = [(datetime(2026, 3, 18) - timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range(1, 11)]
        articles = {
            "stock_news":    _make_articles(dates),
            "global_news":   pd.DataFrame(),
            "industry_news": pd.DataFrame(),
        }
        s.summarize(articles, as_of_date="2026-03-18")
        prompt = llm.call_args[0][0]

        # Only top 5 titles should appear (composite_score descending → first 5 dates)
        included = sum(1 for d in dates[:5]  if d in prompt)
        excluded = sum(1 for d in dates[5:]  if d in prompt)
        assert included == 5
        assert excluded == 0


# ── Cycle 6: Date math ────────────────────────────────────────

class TestDateMath:
    """window_start and window_end in result match expected dates."""

    def test_window_start_equals_as_of_minus_window(self):
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        as_of  = "2026-03-18"
        result = s.summarize(
            {"stock_news": _make_articles(["2026-03-17"]),
             "global_news": pd.DataFrame(), "industry_news": pd.DataFrame()},
            as_of_date=as_of,
        )
        expected_start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        assert result["window_start"] == expected_start
        assert result["window_end"]   == as_of

    def test_article_count_reflects_window_filtered_count(self):
        llm = MagicMock(return_value=_make_llm_response())
        s   = NewsSummarizer(llm_client=llm, window_days=7)

        as_of         = "2026-03-18"
        inside_dates  = ["2026-03-17", "2026-03-15"]
        outside_dates = ["2026-02-01"]

        articles = {
            "stock_news":    _make_articles(inside_dates + outside_dates),
            "global_news":   pd.DataFrame(),
            "industry_news": pd.DataFrame(),
        }
        result = s.summarize(articles, as_of_date=as_of)
        assert result["article_count"] == 2
