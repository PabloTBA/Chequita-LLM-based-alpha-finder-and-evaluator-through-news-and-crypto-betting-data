## Problem Statement

As a trader using the MFT Alpha Finder pipeline, the per-ticker LLM calls in the screener receive only OHLCV features and macro context — they have no access to the actual news articles that made a ticker worth screening in the first place. A ticker flagged as high-conviction by the composite news score might have scored highly due to a single transformative article (an FDA approval, an earnings beat, a regulatory investigation), but that article's content is never seen by the LLM when it forms its buy/watch/avoid verdict. Similarly, prediction market data fetched by the pipeline has nowhere persistent to live between runs. Without a vector store, the pipeline is making LLM decisions on numeric summaries while ignoring the richest signal it already collects: the actual text.

---

## Solution

A `RAGStore` module that wraps ChromaDB and provides a simple, deep interface for inserting and retrieving documents across two persistent collections: `news` (articles from Stage1DataCollector) and `prediction_markets` (events from PredictionMarketClient). News articles are inserted immediately after collection in Stage 1. For each shortlisted ticker, the screener retrieves the top 3 most semantically relevant news chunks and injects them into the per-ticker LLM prompt. Prediction market data is inserted for future retrieval but consumed directly (not via RAG) by the summarizer in v1. The store persists across runs at `data/chroma` so the knowledge base grows incrementally and never re-embeds documents it has already seen.

---

## User Stories

### Storage and Persistence

1. As a trader, I want news articles and prediction market events stored in a persistent vector database so that the knowledge base accumulates across daily runs rather than being discarded after each pipeline execution.
2. As a trader, I want the vector store to persist on disk at a configurable path so that I can control where the database lives and back it up if needed.
3. As a trader, I want news articles and prediction market events stored in separate ChromaDB collections so that I can retrieve news and macro events independently with distinct relevance criteria.
4. As a trader, I want the store to initialise both collections automatically on first use so that I never have to manually create collections before running the pipeline.
5. As a trader, I want the store to load existing collections from disk on subsequent runs so that previously embedded documents are immediately available without re-embedding.

### Insertion — News

6. As a trader, I want news articles inserted into the RAG store immediately after Stage 1 collection so that the store is fully populated before any retrieval happens later in the pipeline.
7. As a trader, I want articles from all four news sources (stock_news, global_news, industry_news, ticker_news) inserted into the same `news` collection so that per-ticker retrieval searches across all source types simultaneously.
8. As a trader, I want each article stored with its title and any available body text as the embedded document so that semantic search captures topical meaning, not just keyword presence.
9. As a trader, I want each article stored with metadata (date, source, ticker list, composite_score) so that retrieved chunks can be filtered or ranked by recency or relevance score if needed.
10. As a trader, I want articles deduplicated by URL so that re-running the pipeline on the same date does not re-embed articles already in the store, keeping the collection lean and avoiding retrieval duplicates.
11. As a trader, I want the insertion to skip articles with no URL rather than failing so that malformed records from the API do not break the pipeline.
12. As a trader, I want a count of newly inserted vs. skipped (duplicate) articles logged after each insertion batch so that I know how much new content was added to the store.

### Insertion — Prediction Markets

13. As a trader, I want prediction market events inserted into a dedicated `prediction_markets` collection after Stage 1b so that macro event data is stored alongside but separately from news.
14. As a trader, I want each market stored with its formatted text as the embedded document so that semantic search on the prediction markets collection retrieves contextually relevant events.
15. As a trader, I want each market stored with metadata (market_id, category, status, volume, probability) so that downstream consumers can filter by category or volume if needed.
16. As a trader, I want markets deduplicated by Polymarket market ID so that the same market fetched on multiple days is not re-embedded each time.

### Retrieval

17. As a trader, I want to retrieve the top k most semantically relevant news chunks for a given query string so that the ticker screener can inject the most relevant article excerpts into each per-ticker LLM prompt.
18. As a trader, I want k to be configurable at retrieval time (default k=3) so that different callers can request more or fewer chunks based on their context window budget.
19. As a trader, I want retrieval to return plain text strings (not raw ChromaDB result objects) so that callers can inject retrieved chunks directly into LLM prompts without parsing.
20. As a trader, I want retrieval to return an empty list (not raise an exception) when the collection is empty or no results match so that the pipeline degrades gracefully on a fresh install with no data yet.
21. As a trader, I want retrieval to accept the collection name as a parameter so that callers can query either `news` or `prediction_markets` through the same interface.

### Ticker Screener Integration

22. As a trader, I want the per-ticker LLM call in the screener to include the top 3 retrieved news chunks for each ticker so that the verdict is grounded in actual article content, not just numeric OHLCV features.
23. As a trader, I want the retrieval query for each ticker to be the ticker symbol so that the most directly relevant articles are retrieved without injecting strategy assumptions into the query.
24. As a trader, I want the screener to work correctly when no RAG store is provided (rag_store=None) so that the module remains independently testable and usable without ChromaDB installed.
25. As a trader, I want retrieved chunks clearly labelled in the LLM prompt (e.g. "Relevant news:") so that the model understands the provenance of the injected text and does not confuse it with OHLCV data.

### Pipeline Wiring

26. As a trader, I want the RAGStore instantiated once in the pipeline and shared across all stages that need it so that the same in-memory ChromaDB connection is reused rather than re-opened per call.
27. As a trader, I want the RAGStore passed to the ticker screener via the existing dependency injection pattern (_modules dict) so that tests can substitute a mock store without modifying pipeline code.
28. As a trader, I want the pipeline to continue normally if RAGStore instantiation fails (e.g. ChromaDB not installed) so that the pipeline is not hard-blocked on the vector store being present.

---

## Implementation Decisions

### Modules

| Module | Action | Responsibility |
|---|---|---|
| `rag_store` | New | ChromaDB wrapper; two persistent collections; insert and retrieve interface |
| `pipeline_orchestrator` | Modified | Instantiate RAGStore; call insert_news after Stage 1; call insert_markets after Stage 1b; pass rag_store to ticker_screener |
| `ticker_screener` | Modified | Accept optional rag_store parameter in screen_tickers; retrieve 3 chunks per ticker; inject into per-ticker LLM prompt |

### RAGStore Interface

Constructor:
  - persist_dir: str = "data/chroma"
  - embedding_model: str = "all-MiniLM-L6-v2" (ChromaDB default, runs locally, no API key)

Public methods:
  - insert_news(articles: dict mapping source name to DataFrame) -> None
  - insert_markets(markets: list of dicts) -> None
  - retrieve(query: str, collection: str = "news", k: int = 3) -> list[str]

### Collections

**news collection:**
  - Document ID: article URL (used for deduplication)
  - Embedded text: article title (plus body/summary text if available)
  - Metadata: date (str), source (str), tickers (str, comma-separated), composite_score (float)

**prediction_markets collection:**
  - Document ID: Polymarket market_id (used for deduplication)
  - Embedded text: formatted_text field from PredictionMarketClient
  - Metadata: category (str), status (str), volume (float), probability (float)

### Embedding Model

- ChromaDB default embedding function: all-MiniLM-L6-v2
- Runs entirely locally with no API key or external service required
- Applied automatically by ChromaDB on insert and query; RAGStore does not manage embeddings directly

### Deduplication

- On insert, ChromaDB upserts by document ID (URL for news, market_id for markets)
- Existing documents with the same ID are overwritten with the latest version
- This means probability updates on active markets are naturally refreshed each run

### Persistence

- ChromaDB persists to disk at the configured persist_dir
- The store survives process restarts; collections are reloaded from disk on instantiation
- No manual migration or schema management required

### Ticker Screener Changes

- screen_tickers(shortlisted, macro, features, rag_store=None) — rag_store is optional
- For each ticker: if rag_store is not None, call retrieve(ticker, collection="news", k=3)
- Retrieved chunks injected into the per-ticker LLM prompt under a "Relevant news:" label
- If rag_store is None or retrieve returns empty list, prompt is unchanged from current behaviour

### Pipeline Orchestrator Changes

- RAGStore added to _build_modules() with persist_dir from config
- Stage 1: after collect_range(), call rag_store.insert_news(articles)
- Stage 1b: after prediction_market_client.fetch(), call rag_store.insert_markets(markets)
- rag_store passed to ticker_screener via _modules dict

### Graceful Degradation

- If ChromaDB is not installed or RAGStore fails to instantiate, pipeline logs a warning and continues with rag_store=None
- If retrieve() is called on an empty collection, returns [] without raising
- Ticker screener handles rag_store=None by skipping retrieval silently

---

## Testing Decisions

### What Makes a Good Test

Tests verify external behaviour through the public interface — given a known set of inserted documents, assert that retrieval returns the expected results. Tests must not assert on ChromaDB internals, embedding values, or private methods. Use an in-memory or temporary ChromaDB instance (temp directory) for all tests so no disk state leaks between test runs.

### Modules to Test

**rag_store** — highest priority, pure deterministic behaviour:
- Given insert_news called with articles containing 3 unique URLs, assert retrieve returns results (collection is non-empty).
- Given insert_news called twice with the same URL, assert the collection contains only one document for that URL (deduplication).
- Given an article with no URL, assert insert_news does not raise and skips the record.
- Given retrieve called on an empty collection, assert it returns an empty list without raising.
- Given retrieve called with k=2, assert at most 2 results are returned.
- Given insert_markets called with markets containing duplicate market_ids, assert only one document per market_id is stored.
- Given a query that closely matches an inserted document's text, assert that document appears in the retrieve results.
- Given retrieve called with collection="prediction_markets", assert results come from that collection and not from news.

**ticker_screener (modified behaviour)**:
- Given rag_store=None, assert screen_tickers produces the same output as the current implementation (no regression).
- Given a mock rag_store that returns ["Relevant chunk 1", "Relevant chunk 2", "Relevant chunk 3"], assert the LLM prompt contains those strings.
- Given a mock rag_store that returns [], assert the LLM prompt is unchanged.

**pipeline_orchestrator (modified behaviour)**:
- Assert rag_store.insert_news is called with the articles dict returned by Stage 1.
- Assert rag_store.insert_markets is called with the markets list returned by Stage 1b.
- Assert rag_store is passed to the ticker_screener module.
- Assert that if RAGStore raises on instantiation, the pipeline continues with rag_store=None.

### Prior Art

Use the _modules injection pattern from test/test_pipeline_orchestrator.py — inject a MagicMock rag_store to test pipeline wiring. For RAGStore unit tests, use a pytest tmp_path fixture to create a throwaway ChromaDB persist directory that is automatically cleaned up after each test.

---

## Out of Scope

- Per-ticker prediction market retrieval via RAG: prediction market data is inserted into ChromaDB for future use but consumed directly (not via RAG) by the summarizer in v1. RAG retrieval of prediction markets is a future extension.
- Embedding model fine-tuning or custom embeddings: all-MiniLM-L6-v2 is used as-is via the ChromaDB default embedding function.
- Cross-collection retrieval (searching news and prediction_markets in a single query): each retrieve call targets one collection.
- Scheduled or background re-indexing: insertion happens synchronously at Stage 1 during the pipeline run.
- Authentication or multi-user access control: ChromaDB runs as a local embedded database, not a server.
- Article body/full-text scraping: only the title and any text already present in the Benzinga response is embedded. No additional scraping of linked article URLs.
- Semantic deduplication (near-duplicate detection): deduplication is by exact URL or market_id match only.

---

## Further Notes

- ChromaDB embedded mode (no server): the store runs in-process, reading and writing directly to disk. No separate ChromaDB server process is required.
- Cold start behaviour: on first run with an empty store, retrieve returns [] for all queries. The ticker screener handles this gracefully by proceeding without RAG context. After the first run inserts articles, subsequent runs benefit from the populated store immediately.
- Store growth: with ~200 articles per day across 4 sources, and a 90-day rolling window, the store will hold approximately 18,000 documents at steady state. all-MiniLM-L6-v2 produces 384-dimensional embeddings; this is well within ChromaDB's local performance envelope.
- No rolling window eviction in v1: old articles are never deleted. The store grows unboundedly. Eviction by date is a future extension if store size becomes a concern.
- Upsert semantics for market data: because the same active Polymarket market is fetched on multiple days (with updated probabilities), using upsert-by-market_id means the store always holds the latest probability for each market without accumulating stale copies.
