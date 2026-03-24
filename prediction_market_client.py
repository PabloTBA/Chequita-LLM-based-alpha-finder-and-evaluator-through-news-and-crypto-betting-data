"""
PredictionMarketClient
======================
Loads prediction markets from disk cache, filters by category and minimum
volume, and optionally inserts results into a RAGStore.

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
from typing import Any

_DEFAULT_MAX_MARKETS = 50  # cap total markets stored to RAG to avoid embedding lag


class PredictionMarketClient:
    def __init__(
        self,
        cache_dir:             str,
        min_volume:            float       = 100_000.0,
        categories:            list[str]   = None,
        rag_store:             Any         = None,
        top_n_for_summarizer:  int         = 20,
        max_markets:           int         = _DEFAULT_MAX_MARKETS,
    ):
        self._cache_dir   = cache_dir
        self._min_volume  = min_volume
        self._categories  = set(categories or ["Economics", "Politics", "Crypto"])
        self._rag_store   = rag_store
        self._top_n       = top_n_for_summarizer
        self._max_markets = max_markets
        os.makedirs(cache_dir, exist_ok=True)

    # ── public ────────────────────────────────────────────────────────────────

    def fetch(self, as_of_date: str) -> list[dict]:
        """
        Return filtered prediction markets for as_of_date from cache.
        Returns [] if no cache file exists for the date.
        """
        cache_path = self._cache_path(as_of_date)

        print(f"  [PredictionMarkets] Getting prediction market data for {as_of_date} ...")

        if not os.path.exists(cache_path):
            print(f"  [PredictionMarkets] No cached data found for {as_of_date}, skipping.")
            return []

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        markets = [m for m in data if self._keep(m)]
        markets = sorted(markets, key=lambda m: m["volume"], reverse=True)[:self._max_markets]

        print(f"  [PredictionMarkets] Loaded {len(markets)} markets from cache.")

        # ── RAG embedding ─────────────────────────────────────────────────────
        if self._rag_store is not None and markets:
            print(f"  [PredictionMarkets] Embedding {len(markets)} markets into RAG ...")
            self._rag_store.insert_markets(markets)
            print(f"  [PredictionMarkets] RAG embedding complete.")

        return markets

    # ── internal ──────────────────────────────────────────────────────────────

    def _cache_path(self, date_str: str) -> str:
        return os.path.join(self._cache_dir, f"{date_str}_prediction_markets.json")

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
