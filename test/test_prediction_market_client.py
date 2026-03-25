"""
Tests for PredictionMarketClient
=================================
Cycle 1  — tracer: mock API response returns list of dicts with required keys
Cycle 2  — category filtering
Cycle 3  — volume filtering
Cycle 4  — volume formatting ($M / $K / raw)
Cycle 5  — probability in formatted_text
Cycle 6  — resolved market outcome in formatted_text
Cycle 7  — cache hit skips API call
Cycle 8  — API failure returns [], no cache written
Cycle 9  — RAG insertion called with results
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction_market_client import PredictionMarketClient


# ── helpers ──────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "market_id", "event", "probability", "volume",
    "status", "resolved_yes", "end_date", "category", "formatted_text",
}

def _market_api(
    id="mkt-001",
    question="Will the Fed cut rates by 25bps?",
    yes_price="0.73",
    volume="5000000.00",
    active=True,
    closed=False,
    category="Economics",
    end_date="2026-06-30",
    winner=None,          # "YES" | "NO" | None for active
):
    return {
        "id":           id,
        "question":     question,
        "outcomePrices": [yes_price, str(1 - float(yes_price))],
        "volume":       volume,
        "active":       active,
        "closed":       closed,
        "category":     category,
        "endDateIso":   end_date,
        "winner":       winner,
    }


def _mock_response(markets: list[dict], status_code: int = 200):
    """Returns a mock requests.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = markets
    resp.raise_for_status = MagicMock()
    return resp


def _client(tmp_path, min_volume=100_000,
            categories=("Economics", "Politics", "Crypto"),
            rag_store=None):
    return PredictionMarketClient(
        cache_dir=str(tmp_path),
        min_volume=min_volume,
        categories=list(categories),
        rag_store=rag_store,
    )


# ── Cycle 1: tracer — required keys present ───────────────────────────────────

class TestOutputContract:
    def test_fetch_returns_list(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            result = _client(tmp_path).fetch("2026-03-20")
        assert isinstance(result, list)

    def test_each_market_has_required_keys(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            result = _client(tmp_path).fetch("2026-03-20")
        assert len(result) > 0
        for m in result:
            assert REQUIRED_KEYS.issubset(m.keys()), f"Missing keys: {REQUIRED_KEYS - m.keys()}"

    def test_market_id_matches_api_id(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api(id="mkt-abc")])):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["market_id"] == "mkt-abc"

    def test_event_matches_question(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(question="Will Bitcoin hit $100k?")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["event"] == "Will Bitcoin hit $100k?"

    def test_status_active_for_open_market(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(active=True, closed=False)]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["status"] == "active"

    def test_resolved_yes_is_none_for_active_market(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["resolved_yes"] is None


# ── Cycle 2: category filtering ───────────────────────────────────────────────

class TestCategoryFiltering:
    def test_market_matching_category_included(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(id="eco-1", category="Economics")]
        )):
            result = _client(tmp_path, categories=["Economics"]).fetch("2026-03-20")
        assert any(m["market_id"] == "eco-1" for m in result)

    def test_market_not_matching_category_excluded(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(id="sports-1", category="Sports")]
        )):
            result = _client(tmp_path, categories=["Economics"]).fetch("2026-03-20")
        assert not any(m["market_id"] == "sports-1" for m in result)

    def test_multiple_allowed_categories_all_included(self, tmp_path):
        markets = [
            _market_api(id="eco-1",    category="Economics"),
            _market_api(id="pol-1",    category="Politics"),
            _market_api(id="crypto-1", category="Crypto"),
            _market_api(id="sport-1",  category="Sports"),
        ]
        with patch("requests.get", return_value=_mock_response(markets)):
            result = _client(tmp_path, categories=["Economics", "Politics", "Crypto"]).fetch("2026-03-20")
        ids = {m["market_id"] for m in result}
        assert {"eco-1", "pol-1", "crypto-1"}.issubset(ids)
        assert "sport-1" not in ids


# ── Cycle 3: volume filtering ─────────────────────────────────────────────────

class TestVolumeFiltering:
    def test_market_above_threshold_included(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(id="big", volume="500000.00")]
        )):
            result = _client(tmp_path, min_volume=100_000).fetch("2026-03-20")
        assert any(m["market_id"] == "big" for m in result)

    def test_market_below_threshold_excluded(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(id="tiny", volume="50000.00")]
        )):
            result = _client(tmp_path, min_volume=100_000).fetch("2026-03-20")
        assert not any(m["market_id"] == "tiny" for m in result)

    def test_market_exactly_at_threshold_included(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(id="exact", volume="100000.00")]
        )):
            result = _client(tmp_path, min_volume=100_000).fetch("2026-03-20")
        assert any(m["market_id"] == "exact" for m in result)

    def test_min_volume_is_configurable(self, tmp_path):
        markets = [
            _market_api(id="mid",  volume="300000.00"),
            _market_api(id="low",  volume="50000.00"),
        ]
        with patch("requests.get", return_value=_mock_response(markets)):
            result = _client(tmp_path, min_volume=200_000).fetch("2026-03-20")
        ids = {m["market_id"] for m in result}
        assert "mid" in ids
        assert "low" not in ids


# ── Cycle 4: volume formatting ────────────────────────────────────────────────

class TestVolumeFormatting:
    def test_million_volume_formatted_with_M(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(volume="2300000.00")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert "$2.3M" in result[0]["formatted_text"]

    def test_thousand_volume_formatted_with_K(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(volume="450000.00")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert "$450K" in result[0]["formatted_text"]

    def test_sub_thousand_volume_formatted_as_raw(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(volume="950.00")]
        )):
            result = _client(tmp_path, min_volume=0).fetch("2026-03-20")
        assert "$950" in result[0]["formatted_text"]


# ── Cycle 5: probability in formatted_text ────────────────────────────────────

class TestProbabilityFormatting:
    def test_probability_shown_as_percentage(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(yes_price="0.73")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert "73%" in result[0]["formatted_text"]

    def test_probability_field_is_float(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(yes_price="0.73")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert abs(result[0]["probability"] - 0.73) < 0.001

    def test_formatted_text_contains_event_and_probability_and_volume(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(question="Fed cuts rates", yes_price="0.60", volume="1000000.00")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        ft = result[0]["formatted_text"]
        assert "Fed cuts rates" in ft
        assert "60%" in ft
        assert "$1.0M" in ft


# ── Cycle 6: resolved market outcome ─────────────────────────────────────────

class TestResolvedMarket:
    def test_resolved_yes_market_has_resolved_yes_true(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(active=False, closed=True, winner="YES")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["resolved_yes"] is True

    def test_resolved_no_market_has_resolved_yes_false(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(active=False, closed=True, winner="NO")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["resolved_yes"] is False

    def test_resolved_yes_market_status_is_resolved(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(active=False, closed=True, winner="YES")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result[0]["status"] == "resolved"

    def test_resolved_yes_appears_in_formatted_text(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(active=False, closed=True, winner="YES")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert "Resolved: YES" in result[0]["formatted_text"]

    def test_resolved_no_appears_in_formatted_text(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(active=False, closed=True, winner="NO")]
        )):
            result = _client(tmp_path).fetch("2026-03-20")
        assert "Resolved: NO" in result[0]["formatted_text"]


# ── Cycle 7: cache hit skips API ─────────────────────────────────────────────

class TestCaching:
    def test_second_fetch_uses_cache(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api()])) as mock_get:
            client = _client(tmp_path)
            client.fetch("2026-03-20")            # first call — hits API (active + resolved)
            calls_after_first = mock_get.call_count
            client.fetch("2026-03-20")            # second call — should serve from cache
        assert mock_get.call_count == calls_after_first  # no new API calls on cache hit

    def test_cache_file_written_after_successful_fetch(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            _client(tmp_path).fetch("2026-03-20")
        cache_file = tmp_path / "2026-03-20_prediction_markets.json"
        assert cache_file.exists()

    def test_cache_returns_same_data(self, tmp_path):
        with patch("requests.get", return_value=_mock_response(
            [_market_api(id="mkt-cache-test")]
        )):
            client = _client(tmp_path)
            first  = client.fetch("2026-03-20")
            second = client.fetch("2026-03-20")
        assert first[0]["market_id"] == second[0]["market_id"]


# ── Cycle 8: API failure ──────────────────────────────────────────────────────

class TestApiFailure:
    def test_exception_during_fetch_returns_empty_list(self, tmp_path):
        with patch("requests.get", side_effect=Exception("network error")):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result == []

    def test_no_cache_written_on_failure(self, tmp_path):
        with patch("requests.get", side_effect=Exception("network error")):
            _client(tmp_path).fetch("2026-03-20")
        cache_file = tmp_path / "2026-03-20_prediction_markets.json"
        assert not cache_file.exists()

    def test_non_200_status_returns_empty_list(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([], status_code=503)):
            result = _client(tmp_path).fetch("2026-03-20")
        assert result == []


# ── Cycle 9: RAG insertion ────────────────────────────────────────────────────

class TestRagInsertion:
    def test_insert_markets_called_on_successful_fetch(self, tmp_path):
        mock_rag = MagicMock()
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            _client(tmp_path, rag_store=mock_rag).fetch("2026-03-20")
        mock_rag.insert_markets.assert_called_once()

    def test_insert_markets_receives_list_of_dicts(self, tmp_path):
        mock_rag = MagicMock()
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            _client(tmp_path, rag_store=mock_rag).fetch("2026-03-20")
        args = mock_rag.insert_markets.call_args[0][0]
        assert isinstance(args, list)
        assert len(args) > 0

    def test_rag_not_called_on_api_failure(self, tmp_path):
        mock_rag = MagicMock()
        with patch("requests.get", side_effect=Exception("fail")):
            _client(tmp_path, rag_store=mock_rag).fetch("2026-03-20")
        mock_rag.insert_markets.assert_not_called()

    def test_rag_none_does_not_raise(self, tmp_path):
        with patch("requests.get", return_value=_mock_response([_market_api()])):
            result = _client(tmp_path, rag_store=None).fetch("2026-03-20")
        assert isinstance(result, list)
