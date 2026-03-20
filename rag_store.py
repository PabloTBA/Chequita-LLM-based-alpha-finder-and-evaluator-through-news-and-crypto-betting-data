"""
RAGStore
========
ChromaDB-backed vector store with two persistent collections:
  - news               : articles from Stage1DataCollector
  - prediction_markets : events from PredictionMarketClient

Public interface
----------------
  store = RAGStore(persist_dir="data/chroma")
  store.insert_news(articles)          # dict[str, pd.DataFrame]
  store.insert_markets(markets)        # list[dict]
  chunks = store.retrieve(query, collection="news", k=3)  # list[str]
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_NEWS_COLLECTION = "news"
_MARKETS_COLLECTION = "prediction_markets"


class RAGStore:
    def __init__(self, persist_dir: str = "data/chroma"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL
        )
        self._news = self._client.get_or_create_collection(
            name=_NEWS_COLLECTION,
            embedding_function=ef,
        )
        self._markets = self._client.get_or_create_collection(
            name=_MARKETS_COLLECTION,
            embedding_function=ef,
        )

    # ── public ────────────────────────────────────────────────────────────────

    def insert_news(self, articles: dict[str, pd.DataFrame]) -> None:
        """Insert articles from all sources into the news collection."""
        ids, docs, metas = [], [], []

        for source, df in articles.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                url = row.get("url")
                if not url:
                    continue
                ids.append(str(url))
                docs.append(str(row.get("title", "")))
                metas.append({
                    "date":            str(row.get("date", "")),
                    "source":          str(source),
                    "tickers":         str(row.get("tickers", "")),
                    "composite_score": float(row.get("composite_score", 0.0)),
                })

        if ids:
            self._news.upsert(ids=ids, documents=docs, metadatas=metas)
            print(f"  [RAG] news: upserted {len(ids)} articles")

    def insert_markets(self, markets: list[dict]) -> None:
        """Insert prediction market events into the prediction_markets collection."""
        ids, docs, metas = [], [], []

        for m in markets:
            mid = m.get("market_id")
            if not mid:
                continue
            ids.append(str(mid))
            docs.append(str(m.get("formatted_text", m.get("event", ""))))
            metas.append({
                "category":    str(m.get("category", "")),
                "status":      str(m.get("status", "")),
                "volume":      float(m.get("volume", 0.0)),
                "probability": float(m.get("probability", 0.0)),
            })

        if ids:
            self._markets.upsert(ids=ids, documents=docs, metadatas=metas)
            print(f"  [RAG] prediction_markets: upserted {len(ids)} markets")

    def retrieve(
        self,
        query:      str,
        collection: str = _NEWS_COLLECTION,
        k:          int = 3,
    ) -> list[str]:
        """
        Return up to k document texts most semantically similar to query.
        Returns [] if the collection is empty or no results found.
        """
        col = self._news if collection == _NEWS_COLLECTION else self._markets

        try:
            count = col.count()
        except Exception:
            return []

        if count == 0:
            return []

        n = min(k, count)
        results = col.query(query_texts=[query], n_results=n)
        docs = results.get("documents", [[]])[0]
        return [str(d) for d in docs]
