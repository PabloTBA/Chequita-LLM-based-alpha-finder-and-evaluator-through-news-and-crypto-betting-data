"""
PredictionMarketClient
======================
Fetches active and recently-resolved prediction markets from the Polymarket
CLOB API, filters by category and minimum volume, caches per-date to disk,
and optionally inserts results into a RAGStore.

Public interface
----------------
  client  = PredictionMarketClient(
                cache_dir, min_volume=100_000,
                categories=["Economics", "Politics", "Crypto"],
                rag_store=None,
                top_n_for_summarizer=20,
            )
  markets = client.fetch(as_of_date)   # list[dict]

Each dict:
  market_id, event, probability, volume, status,
  resolved_yes, end_date, category, formatted_text
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import requests

_BASE_URL        = "https://gamma-api.polymarket.com/markets"
_REQUEST_DELAY   = 0.2   # seconds between paginated requests
_PAGE_SIZE       = 100
_RESOLVE_WINDOW  = 90    # days — match the news corpus rolling window


class PredictionMarketClient:
    def __init__(
        self,
        cache_dir:             str,
        min_volume:            float       = 100_000.0,
        categories:            list[str]   = None,
        rag_store:             Any         = None,
        top_n_for_summarizer:  int         = 20,
    ):
        self._cache_dir   = cache_dir
        self._min_volume  = min_volume
        self._categories  = set(categories or ["Economics", "Politics", "Crypto"])
        self._rag_store   = rag_store
        self._top_n       = top_n_for_summarizer
        os.makedirs(cache_dir, exist_ok=True)

    # ── public ────────────────────────────────────────────────────────────────

    def fetch(self, as_of_date: str) -> list[dict]:
        """
        Return filtered, formatted prediction markets for as_of_date.
        Serves from cache if available; fetches and caches otherwise.
        On any API error returns [] without writing cache.
        """
        cache_path = self._cache_path(as_of_date)

        # ── cache hit ─────────────────────────────────────────────────────────
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  [CACHE] {as_of_date} prediction_markets: {len(data)} markets")
            return data

        # ── live fetch ────────────────────────────────────────────────────────
        try:
            raw = self._fetch_all(as_of_date)
        except Exception as exc:
            print(f"  [ERROR] prediction_markets fetch failed: {exc}")
            return []

        markets = [self._format(m) for m in raw if self._keep(m)]

        if not markets:
            print(f"  [API]   {as_of_date} prediction_markets: 0 markets after filtering")
            return []

        # ── write cache ───────────────────────────────────────────────────────
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(markets, f, ensure_ascii=False, indent=2)
        print(f"  [API]   {as_of_date} prediction_markets: {len(markets)} markets cached")

        # ── RAG insertion ─────────────────────────────────────────────────────
        if self._rag_store is not None:
            self._rag_store.insert_markets(markets)

        return markets

    # ── internal ──────────────────────────────────────────────────────────────

    def _cache_path(self, date_str: str) -> str:
        return os.path.join(self._cache_dir, f"{date_str}_prediction_markets.json")

    def _fetch_all(self, as_of_date: str) -> list[dict]:
        """Fetch active + recently resolved markets from Polymarket."""
        resolve_cutoff = (
            datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=_RESOLVE_WINDOW)
        ).strftime("%Y-%m-%d")

        active   = self._paginate({"active": "true",  "closed": "false"})
        resolved = self._paginate({"active": "false", "closed": "true",
                                   "end_date_min": resolve_cutoff})
        return active + resolved

    def _paginate(self, extra_params: dict) -> list[dict]:
        results = []
        offset  = 0
        while True:
            params = {"limit": _PAGE_SIZE, "offset": offset, **extra_params}
            resp   = requests.get(_BASE_URL, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"  [WARN]  Polymarket HTTP {resp.status_code}")
                break
            page = resp.json() or []
            if not page:
                break
            results.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return results

    def _keep(self, m: dict) -> bool:
        category = m.get("category", "")
        volume   = float(m.get("volume", 0) or 0)
        return category in self._categories and volume >= self._min_volume

    def _format(self, m: dict) -> dict:
        market_id = str(m.get("id", ""))
        event     = str(m.get("question", ""))
        category  = str(m.get("category", ""))
        end_date  = m.get("endDateIso") or m.get("end_date")
        volume    = float(m.get("volume", 0) or 0)
        winner    = m.get("winner")

        # probability from outcomePrices[0] (YES)
        prices = m.get("outcomePrices", [])
        try:
            probability = float(prices[0]) if prices else 0.0
        except (ValueError, TypeError):
            probability = 0.0

        # status
        closed    = m.get("closed", False)
        active    = m.get("active", True)
        status    = "resolved" if closed and not active else "active"

        # resolved outcome
        resolved_yes: bool | None = None
        if status == "resolved" and winner:
            resolved_yes = winner.upper() == "YES"

        # formatted text
        vol_str  = _fmt_volume(volume)
        prob_pct = int(round(probability * 100))
        parts    = [f"Event: {event} | Probability: {prob_pct}% | Volume: {vol_str}"]
        if resolved_yes is True:
            parts.append("Resolved: YES")
        elif resolved_yes is False:
            parts.append("Resolved: NO")
        formatted_text = " | ".join(parts)

        return {
            "market_id":      market_id,
            "event":          event,
            "probability":    probability,
            "volume":         volume,
            "status":         status,
            "resolved_yes":   resolved_yes,
            "end_date":       end_date,
            "category":       category,
            "formatted_text": formatted_text,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_volume(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${int(round(v / 1_000))}K"
    return f"${int(v)}"
