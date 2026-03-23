"""
Tests for RAGStore
==================
Cycle 1  — tracer: insert article, retrieve returns non-empty list
Cycle 2  — news deduplication by URL
Cycle 3  — k parameter respected
Cycle 4  — empty collection returns [] without raising
Cycle 5  — article with no URL skipped silently
Cycle 6  — insert_markets + retrieve from prediction_markets collection
Cycle 7  — markets deduplication by market_id
Cycle 8  — collection isolation (news vs prediction_markets)
Cycle 9  — persistence across two RAGStore instances on same persist_dir
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_store import RAGStore


# ── fixtures ─────────────────────────────────────────────────────────────────

def _articles(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """Wrap a list of article dicts into the dict[str, DataFrame] format Stage1 returns."""
    return {"stock_news": pd.DataFrame(rows)}


def _article(title="Apple beats earnings", url="https://news.example.com/aapl-1", ticker="AAPL"):
    return {"title": title, "url": url, "tickers": ticker,
            "date": "2026-03-19", "composite_score": 8.0, "source": "stock_news"}


def _market(market_id="mkt-001", event="Fed cuts rates by 25bps",
            probability=0.72, volume=5_000_000.0):
    return {
        "market_id":      market_id,
        "event":          event,
        "probability":    probability,
        "volume":         volume,
        "status":         "active",
        "resolved_yes":   None,
        "end_date":       "2026-06-30",
        "category":       "Economics",
        "formatted_text": f"Event: {event} | Probability: {int(probability*100)}% | Volume: $5.0M",
    }


# ── Cycle 1: tracer — insert article, retrieve non-empty ─────────────────────

class TestTracerBullet:
    def test_insert_news_then_retrieve_returns_results(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_news(_articles([_article()]))
        results = store.retrieve("AAPL earnings", collection="news", k=3)
        assert len(results) > 0

    def test_retrieve_returns_list_of_strings(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_news(_articles([_article()]))
        results = store.retrieve("AAPL", collection="news", k=3)
        assert all(isinstance(r, str) for r in results)


# ── Cycle 2: news deduplication by URL ───────────────────────────────────────

class TestNewsDeduplication:
    def test_same_url_twice_stored_once(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        art = _article(url="https://news.example.com/aapl-dup")
        store.insert_news(_articles([art]))
        store.insert_news(_articles([art]))   # second insert same URL
        results = store.retrieve("AAPL", collection="news", k=10)
        # titles in results should not be duplicated
        assert results.count(results[0]) == 1

    def test_different_urls_both_stored(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_news(_articles([
            _article(title="Apple beats earnings", url="https://news.example.com/a1"),
            _article(title="Microsoft cloud growth", url="https://news.example.com/a2", ticker="MSFT"),
        ]))
        results = store.retrieve("earnings growth", collection="news", k=10)
        assert len(results) >= 2


# ── Cycle 3: k parameter respected ───────────────────────────────────────────

class TestKParameter:
    def test_retrieve_k1_returns_at_most_one(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        arts = [_article(title=f"Article {i}", url=f"https://example.com/{i}") for i in range(5)]
        store.insert_news(_articles(arts))
        results = store.retrieve("article", collection="news", k=1)
        assert len(results) <= 1

    def test_retrieve_k3_returns_at_most_three(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        arts = [_article(title=f"Headline {i}", url=f"https://example.com/h{i}") for i in range(10)]
        store.insert_news(_articles(arts))
        results = store.retrieve("headline", collection="news", k=3)
        assert len(results) <= 3


# ── Cycle 4: empty collection returns [] without raising ─────────────────────

class TestEmptyCollection:
    def test_retrieve_empty_news_returns_empty_list(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        results = store.retrieve("AAPL", collection="news", k=3)
        assert results == []

    def test_retrieve_empty_markets_returns_empty_list(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        results = store.retrieve("Fed rates", collection="prediction_markets", k=3)
        assert results == []


# ── Cycle 5: article with no URL skipped silently ────────────────────────────

class TestNoUrlSkipped:
    def test_article_without_url_does_not_raise(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        no_url = {"title": "No URL article", "url": None,
                  "tickers": "AAPL", "date": "2026-03-19",
                  "composite_score": 5.0, "source": "stock_news"}
        store.insert_news(_articles([no_url]))   # must not raise

    def test_article_with_empty_string_url_skipped(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        empty_url = {"title": "Empty URL article", "url": "",
                     "tickers": "AAPL", "date": "2026-03-19",
                     "composite_score": 5.0, "source": "stock_news"}
        store.insert_news(_articles([empty_url]))   # must not raise

    def test_valid_article_still_inserted_alongside_no_url(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        bad  = {"title": "No URL", "url": None, "tickers": "AAPL",
                "date": "2026-03-19", "composite_score": 3.0, "source": "stock_news"}
        good = _article(title="Valid article about AAPL earnings beat")
        store.insert_news(_articles([bad, good]))
        results = store.retrieve("AAPL earnings", collection="news", k=3)
        assert len(results) > 0


# ── Cycle 6: insert_markets + retrieve from prediction_markets ────────────────

class TestMarketsInsertion:
    def test_insert_markets_then_retrieve_returns_results(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_markets([_market()])
        results = store.retrieve("Fed rate cut", collection="prediction_markets", k=3)
        assert len(results) > 0

    def test_retrieved_market_text_is_string(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_markets([_market(event="US recession probability")])
        results = store.retrieve("recession", collection="prediction_markets", k=3)
        assert all(isinstance(r, str) for r in results)


# ── Cycle 7: markets deduplication by market_id ──────────────────────────────

class TestMarketsDeduplication:
    def test_same_market_id_twice_stored_once(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        mkt = _market(market_id="mkt-dup")
        store.insert_markets([mkt])
        store.insert_markets([mkt])
        results = store.retrieve("Fed", collection="prediction_markets", k=10)
        assert results.count(results[0]) == 1

    def test_updated_probability_reflected_after_upsert(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        original = _market(market_id="mkt-upd", probability=0.40)
        store.insert_markets([original])
        updated  = {**original, "probability": 0.80,
                    "formatted_text": "Event: Fed cuts rates by 25bps | Probability: 80% | Volume: $5.0M"}
        store.insert_markets([updated])
        results = store.retrieve("Fed", collection="prediction_markets", k=3)
        assert any("80%" in r for r in results)


# ── Cycle 8: collection isolation ────────────────────────────────────────────

class TestCollectionIsolation:
    def test_news_query_does_not_return_market_docs(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_markets([_market(event="Fed rate cut unique phrase xyzzy")])
        results = store.retrieve("xyzzy", collection="news", k=3)
        assert results == []

    def test_markets_query_does_not_return_news_docs(self, tmp_path):
        store = RAGStore(persist_dir=str(tmp_path))
        store.insert_news(_articles([_article(title="AAPL unique phrase foobar earnings")]))
        results = store.retrieve("foobar", collection="prediction_markets", k=3)
        assert results == []


# ── Cycle 9: persistence across two RAGStore instances ───────────────────────

class TestPersistence:
    def test_data_survives_new_instance_on_same_persist_dir(self, tmp_path):
        # First instance inserts
        store1 = RAGStore(persist_dir=str(tmp_path))
        store1.insert_news(_articles([_article(title="Persistent AAPL article")]))

        # Second instance on same directory retrieves
        store2 = RAGStore(persist_dir=str(tmp_path))
        results = store2.retrieve("AAPL", collection="news", k=3)
        assert len(results) > 0

    def test_markets_persist_across_instances(self, tmp_path):
        store1 = RAGStore(persist_dir=str(tmp_path))
        store1.insert_markets([_market(event="Persistent Fed market")])

        store2 = RAGStore(persist_dir=str(tmp_path))
        results = store2.retrieve("Fed", collection="prediction_markets", k=3)
        assert len(results) > 0
