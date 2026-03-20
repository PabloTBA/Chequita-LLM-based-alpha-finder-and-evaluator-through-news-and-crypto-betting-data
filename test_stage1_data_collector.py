"""
Tests for Stage1DataCollector — TDD, red-green-refactor.

Run unit tests (no API key needed):
    python -m pytest test_stage1_data_collector.py -v

Run integration tests (requires BENZINGA_API in .env):
    python -m pytest test_stage1_data_collector.py -v --integration
"""

import json
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

from Stage1DataCollector import Stage1DataCollector, score_article

load_dotenv()


# ──────────────────────────────────────────────────────────────
# CYCLE 1 — score_article: pure composite scoring
# ──────────────────────────────────────────────────────────────

class TestScoreArticle:
    """score_article(article, date_str) → scored dict with composite_score."""

    def test_composite_score_known_input(self):
        """
        Given an article with:
          - title containing 3 keyword hits ("earnings", "beat", "guidance")
          - 3 tickers
          - trusted publisher ("Reuters")
          - published exactly at market open (14:30 UTC) → recency_score = 3

        Expected:
          keyword_score  = 3  → ×3 = 9
          ticker_score   = 3  → ×2 = 6
          publisher_score= 2  (Reuters is trusted)
          recency_score  = 3  (published at open)
          composite      = 20
        """
        article = {
            "title":     "Company earnings beat guidance expectations",
            "tickers":   ["AAPL", "MSFT", "GOOG"],
            "source":    "Reuters",
            "published": "2026-03-19T14:30:00Z",
        }
        result = score_article(article, "2026-03-19")

        assert result["keyword_score"]   == 3
        assert result["ticker_score"]    == 3
        assert result["publisher_score"] == 2
        assert result["recency_score"]   == 3
        assert result["composite_score"] == 20

    def test_ticker_score_capped_at_10(self):
        """ticker_score is capped at 10 even if >10 tickers are present."""
        # Benzinga returns tickers in "stocks": [{"name": "AAPL", "exchange": "NYSE"}, ...]
        article = {
            "title":     "Market update",
            "stocks":    [{"name": f"T{chr(65+i)}{chr(65+i)}", "exchange": "NYSE"} for i in range(15)],
            "source":    "Unknown Blog",
            "published": "2026-03-19T00:00:00Z",
        }
        result = score_article(article, "2026-03-19")
        assert result["ticker_score"] == 10

    def test_missing_fields_default_to_zero(self):
        """Article with no tickers/publisher/timestamp still scores without error."""
        article = {"title": "Some neutral article"}
        result = score_article(article, "2026-03-19")

        assert result["composite_score"] >= 0
        assert result["ticker_score"]    == 0
        assert result["publisher_score"] == 0

    def test_required_output_keys_present(self):
        """scored dict always contains all required keys."""
        article = {"title": "Fed raises rates", "tickers": ["SPY"], "source": "Bloomberg"}
        result = score_article(article, "2026-03-19")

        required = {
            "date", "title", "url", "tickers", "publisher",
            "keyword_score", "ticker_score", "publisher_score",
            "recency_score", "composite_score", "catalyst_type",
            "keywords_matched",
        }
        assert required.issubset(result.keys())

    def test_recency_within_30_min_scores_3(self):
        """Article published 15 minutes after market open gets recency_score 3."""
        article = {
            "title":     "Fed raises rates",
            "tickers":   [],
            "source":    "",
            "published": "2026-03-19T14:45:00Z",  # 15 min after open
        }
        result = score_article(article, "2026-03-19")
        assert result["recency_score"] == 3

    def test_recency_more_than_6h_scores_0(self):
        """Article published >6 hours from market open gets recency_score 0."""
        article = {
            "title":     "Fed raises rates",
            "tickers":   [],
            "source":    "",
            "published": "2026-03-19T22:00:00Z",  # 7.5h after open
        }
        result = score_article(article, "2026-03-19")
        assert result["recency_score"] == 0


# ──────────────────────────────────────────────────────────────
# CYCLE 2 — Cache: hit vs miss
# ──────────────────────────────────────────────────────────────

class TestCaching:
    """collect(date) serves from cache when available; calls API when not."""

    def test_cache_hit_skips_http(self, tmp_path):
        """If cache file exists for all sources, no HTTP call is made."""
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))
        date = "2026-03-19"

        # Pre-populate cache for all 4 sources (including ticker_news)
        for source in ("stock_news", "global_news", "industry_news", "ticker_news"):
            cache_file = tmp_path / f"{date}_{source}.json"
            cache_file.write_text(json.dumps([{
                "date": date, "title": f"Cached {source}", "url": f"http://example.com/{source}",
                "tickers": "AAPL", "publisher": "Reuters",
                "keyword_score": 1, "ticker_score": 1, "publisher_score": 2, "recency_score": 3,
                "composite_score": 10, "catalyst_type": "Earnings", "keywords_matched": "earnings",
            }]))

        with patch("Stage1DataCollector.requests.get") as mock_get:
            collector.collect(date)
            mock_get.assert_not_called()

    def test_cache_miss_calls_benzinga(self, tmp_path):
        """If no cache file, requests.get is called for each source."""
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))
        date = "2026-03-19"

        benzinga_response = MagicMock()
        benzinga_response.status_code = 200
        benzinga_response.json.return_value = []  # empty — no articles

        with patch("Stage1DataCollector.requests.get", return_value=benzinga_response) as mock_get:
            collector.collect(date)
            assert mock_get.call_count >= 3  # one call per source type

    def test_cache_file_created_after_fetch(self, tmp_path):
        """After a successful fetch, cache files are written for all 3 sources."""
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))
        date = "2026-03-19"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            collector.collect(date)

        for source in ("stock_news", "global_news", "industry_news"):
            assert (tmp_path / f"{date}_{source}.json").exists()


# ──────────────────────────────────────────────────────────────
# CYCLE 3 — collect() output shape
# ──────────────────────────────────────────────────────────────

class TestCollectOutputShape:
    """collect() always returns a dict with exactly 4 DataFrames."""

    def test_returns_four_dataframes(self, tmp_path):
        """collect() result has keys: stock_news, global_news, industry_news, ticker_news."""
        import pandas as pd
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            result = collector.collect("2026-03-19")

        assert set(result.keys()) == {"stock_news", "global_news", "industry_news", "ticker_news"}
        for df in result.values():
            assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_required_columns(self, tmp_path):
        """Each DataFrame contains the required columns for downstream use."""
        import pandas as pd
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))

        # One real article in stock_news cache
        date = "2026-03-19"
        article = {
            "date": date, "title": "Earnings beat", "url": "http://example.com/1",
            "tickers": "AAPL", "publisher": "Reuters",
            "keyword_score": 2, "ticker_score": 1, "publisher_score": 2, "recency_score": 3,
            "composite_score": 15, "catalyst_type": "Earnings", "keywords_matched": "earnings",
        }
        (tmp_path / f"{date}_stock_news.json").write_text(json.dumps([article]))
        (tmp_path / f"{date}_global_news.json").write_text(json.dumps([]))
        (tmp_path / f"{date}_industry_news.json").write_text(json.dumps([]))

        with patch("Stage1DataCollector.requests.get"):
            result = collector.collect(date)

        required_cols = {"date", "title", "url", "composite_score", "catalyst_type"}
        for df in result.values():
            if not df.empty:
                assert required_cols.issubset(df.columns)


# ──────────────────────────────────────────────────────────────
# CYCLE 4 — collect_range() window guard
# ──────────────────────────────────────────────────────────────

class TestCollectRange:
    """collect_range() enforces the 3-month rolling window."""

    def test_window_over_3_months_raises(self, tmp_path):
        """collect_range() raises ValueError if date range exceeds 3 months."""
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))
        start = "2025-01-01"
        end   = "2026-03-19"   # > 3 months apart

        with pytest.raises(ValueError, match="3.month"):
            collector.collect_range(start, end)

    def test_exactly_3_month_window_is_allowed(self, tmp_path):
        """Exactly 3-month window does not raise."""
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))

        # 2026-03-19 minus 90 days = 2025-12-19
        start = "2025-12-19"
        end   = "2026-03-19"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            result = collector.collect_range(start, end)

        assert isinstance(result, dict)


# ──────────────────────────────────────────────────────────────
# CYCLE 5 — CSV export
# ──────────────────────────────────────────────────────────────

class TestCsvExport:
    """save_csv=True writes one CSV per source to csv_dir."""

    def test_csv_files_created_when_save_csv_true(self, tmp_path):
        """collect(date, save_csv=True) creates 3 CSV files in csv_dir."""
        cache_dir = tmp_path / "cache"
        csv_dir   = tmp_path / "csv"
        collector = Stage1DataCollector(
            api_key="test_key",
            cache_dir=str(cache_dir),
            csv_dir=str(csv_dir),
        )
        date = "2026-03-19"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "title":     "Earnings beat",
            "url":       "http://example.com/1",
            "published": "2026-03-19T14:30:00Z",
            "tickers":   [{"name": "AAPL"}],
            "source":    "Reuters",
        }]

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            collector.collect(date, save_csv=True)

        for source in ("stock_news", "global_news", "industry_news"):
            assert (csv_dir / f"{date}_{source}.csv").exists()

    def test_csv_not_created_when_save_csv_false(self, tmp_path):
        """collect() without save_csv=True writes no CSV files."""
        csv_dir   = tmp_path / "csv"
        collector = Stage1DataCollector(
            api_key="test_key",
            cache_dir=str(tmp_path / "cache"),
            csv_dir=str(csv_dir),
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            collector.collect("2026-03-19")

        assert not csv_dir.exists()

    def test_csv_contains_correct_rows(self, tmp_path):
        """CSV file row count matches the DataFrame returned by collect()."""
        import pandas as pd
        cache_dir = tmp_path / "cache"
        csv_dir   = tmp_path / "csv"
        collector = Stage1DataCollector(
            api_key="test_key",
            cache_dir=str(cache_dir),
            csv_dir=str(csv_dir),
        )
        date = "2026-03-19"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"title": f"Article {i}", "url": f"http://example.com/{i}",
             "published": "2026-03-19T14:30:00Z", "tickers": [], "source": "Reuters"}
            for i in range(5)
        ]

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            result = collector.collect(date, save_csv=True)

        csv_path = csv_dir / f"{date}_stock_news.csv"
        df_csv   = pd.read_csv(csv_path)
        assert len(df_csv) == len(result["stock_news"])


# ──────────────────────────────────────────────────────────────
# CYCLE 6 — Deduplication
# ──────────────────────────────────────────────────────────────

class TestDeduplication:
    """Duplicate articles (same URL) appear only once in output."""

    def test_duplicate_urls_deduplicated_within_source(self, tmp_path):
        """If Benzinga returns same URL twice in one fetch, output keeps only one per source."""
        date = "2026-03-19"
        duplicate = {
            "id":        "abc123",
            "title":     "Earnings beat",
            "url":       "http://example.com/same",
            "published": "2026-03-19T14:30:00Z",
            "tickers":   [{"name": "AAPL"}],
            "channels":  [{"name": "News"}],
            "source":    "Reuters",
        }
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [duplicate, duplicate]  # same article twice

        with patch("Stage1DataCollector.requests.get", return_value=mock_resp):
            result = collector.collect(date)

        # Each source deduplicates internally — stock_news should have exactly 1 row
        stock_df = result["stock_news"]
        assert not stock_df.empty
        assert (stock_df["url"] == "http://example.com/same").sum() == 1


# ──────────────────────────────────────────────────────────────
# CYCLE 7 — Error resilience: failed fetch must not be cached
# ──────────────────────────────────────────────────────────────

class TestErrorResilience:
    """A failed (non-200) fetch must not write a cache file."""

    def test_failed_fetch_not_cached(self, tmp_path):
        """HTTP 400 response → no cache file written → next call retries API."""
        collector = Stage1DataCollector(api_key="bad_key", cache_dir=str(tmp_path))
        date = "2026-03-18"

        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.text = '{"error": "invalid token"}'

        with patch("Stage1DataCollector.requests.get", return_value=error_resp):
            collector.collect(date)

        # No cache files should exist after a failed fetch
        for source in ("stock_news", "global_news", "industry_news"):
            assert not (tmp_path / f"{date}_{source}.json").exists()

    def test_failed_fetch_retried_on_next_call(self, tmp_path):
        """After a failed fetch, the next collect() call hits the API again."""
        collector = Stage1DataCollector(api_key="test_key", cache_dir=str(tmp_path))
        date = "2026-03-18"

        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.text = '{"error": "bad request"}'

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = []

        with patch("Stage1DataCollector.requests.get", return_value=error_resp):
            collector.collect(date)

        # Second call should hit API again (not serve empty cache)
        with patch("Stage1DataCollector.requests.get", return_value=ok_resp) as mock_get:
            collector.collect(date)
            assert mock_get.call_count >= 3  # one per source


# ──────────────────────────────────────────────────────────────
# INTEGRATION — hits the real Benzinga API (needs BENZINGA_API)
# Run with: pytest --integration
# ──────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestBenzingaIntegration:
    """
    Live tests against the real Benzinga API.
    Skipped by default — run with: pytest --integration
    """

    @pytest.fixture(autouse=True)
    def skip_without_flag(self, request):
        if not request.config.getoption("--integration"):
            pytest.skip("Pass --integration to run live API tests")

    @pytest.fixture(autouse=True)
    def require_api_key(self, skip_without_flag):
        if not os.getenv("BENZINGA_API"):
            pytest.skip("BENZINGA_API not set in environment")

    def test_stock_news_returns_articles(self, tmp_path):
        """
        Real API call: stock_news for a recent weekday returns at least 1 article
        with the expected columns populated.
        """
        api_key   = os.getenv("BENZINGA_API")
        collector = Stage1DataCollector(api_key=api_key, cache_dir=str(tmp_path))

        # Use yesterday (US date) — most recent trading day with guaranteed data
        date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = collector.collect(date)

        stock_df = result["stock_news"]
        print(f"\n[INTEGRATION] stock_news for {date}: {len(stock_df)} articles")

        assert not stock_df.empty, (
            f"Got 0 articles for {date}. "
            f"Check BENZINGA_API key and channel names. "
            f"Run collector.debug_raw('{date}') to inspect the raw response."
        )

        # Verify required columns are present and populated
        assert stock_df["title"].notna().any(),         "title column is all null"
        assert stock_df["composite_score"].notna().any(), "composite_score column is all null"

        # Print first row for inspection
        print(stock_df[["date", "composite_score", "catalyst_type", "title", "publisher"]].head(3).to_string())

    def test_debug_raw_does_not_raise(self, tmp_path):
        """debug_raw() prints the raw response without raising."""
        api_key   = os.getenv("BENZINGA_API")
        collector = Stage1DataCollector(api_key=api_key, cache_dir=str(tmp_path))
        date      = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        # Should not raise regardless of API response
        collector.debug_raw(date)
