## Problem Statement

As a trader using the MFT Alpha Finder pipeline, I have no visibility into what the prediction market consensus says about near-term macro events. Fed rate decisions, recession probabilities, election outcomes, and sector-specific regulatory events can shift the entire market regime — but none of this signal currently reaches the pipeline. The news summarizer and macro screener only reason over news text; they are blind to the aggregated probabilistic bets of thousands of market participants on Polymarket. This creates a gap where a 73% probability of a rate cut — backed by $5M in volume — goes unseen while the LLM forms its macro thesis purely from news headlines.

---

## Solution

A `PredictionMarketClient` module that fetches active and recently resolved prediction markets from the Polymarket CLOB API, filters by relevant categories and a minimum volume threshold, caches results per-date, and returns structured formatted market data. The formatted text is (1) inserted into the RAG store's dedicated `prediction_markets` ChromaDB collection for retrieval, and (2) passed directly to the NewsSummarizer as an additional context block so that event probabilities and volumes inform the LLM's macro summary alongside news.

The module requires `rag_store.py` to be built first (separate PRD). Once the RAG store exists, `PredictionMarketClient` slots into Stage 1 of the pipeline alongside `Stage1DataCollector`.

---

## User Stories

### Fetching

1. As a trader, I want the system to automatically fetch active Polymarket markets each run so that the current prediction market consensus is included in my daily macro picture.
2. As a trader, I want the system to also fetch markets resolved in the last 3 months so that the macro summarizer can see how recent macro bets actually played out.
3. As a trader, I want fetched markets filtered to the categories Economics, Politics, and Crypto so that noise from Sports, Entertainment, and other irrelevant categories is excluded.
4. As a trader, I want fetched markets filtered to a minimum total volume threshold (default $100,000, configurable) so that only high-conviction, liquid markets are included — thin markets with $500 volume carry no signal.
5. As a trader, I want the volume threshold to be a constructor parameter (not a hardcoded constant) so that I can raise or lower the bar without changing code.
6. As a trader, I want the categories list to be a constructor parameter so that I can add or remove categories without changing code.

### Caching

7. As a trader, I want prediction market data cached to disk per run-date so that re-running the pipeline mid-day does not re-hit the Polymarket API.
8. As a trader, I want the cache to follow the same pattern as `Stage1DataCollector` — JSON files keyed by date — so that the caching behaviour is consistent and predictable across all data sources.
9. As a trader, I want the cache to be skipped on a failed fetch (API error, network timeout) so that a partial or empty response is never written to disk and the next run retries cleanly.
10. As a trader, I want a log message printed for each fetch indicating whether data came from cache or the live API, consistent with the `[CACHE]` / `[API]` / `[ERROR]` log format used by `Stage1DataCollector`.

### Data Format

11. As a trader, I want each market formatted as `"Event: [title] | Probability: [X]% | Volume: $[Y]"` so that the LLM receives clear, numeric-precision text rather than raw JSON fields.
12. As a trader, I want probability to represent the YES outcome only (since NO = 100% minus YES) so that the format is concise without losing information.
13. As a trader, I want the volume formatted as a human-readable dollar amount (e.g. `$2.3M`, `$450K`) so that the LLM can quickly assess market conviction without parsing raw numbers.
14. As a trader, I want each market record to carry metadata fields (event title, raw probability float, raw volume float, status, category, end date) alongside the formatted text so that downstream consumers can filter or sort programmatically if needed.
15. As a trader, I want resolved markets to include their resolution outcome in the formatted text (e.g. `| Resolved: YES`) so that the summarizer can tell the LLM what actually happened.

### RAG Integration

16. As a trader, I want fetched market records inserted into the RAG store's `prediction_markets` collection so that they are retrievable by semantic similarity for downstream LLM calls.
17. As a trader, I want the `prediction_markets` collection kept separate from the `news` collection in ChromaDB so that macro event retrieval and news article retrieval can be queried independently with distinct relevance criteria.
18. As a trader, I want duplicate markets (same Polymarket market ID already inserted) to not be re-inserted into the RAG store so that repeated runs do not bloat the vector database.

### Pipeline Integration

19. As a trader, I want the prediction market fetch to run as Stage 1b in the pipeline, immediately after news collection and before summarisation, so that the markets are available when the NewsSummarizer builds its prompt.
20. As a trader, I want the formatted market texts passed to `NewsSummarizer.summarize()` as an additional parameter so that the LLM summary incorporates event probabilities alongside news themes.
21. As a trader, I want the NewsSummarizer prompt to include a dedicated `Active Prediction Markets` section listing the top markets by volume so that the LLM explicitly reasons over market-implied probabilities when forming its macro thesis.
22. As a trader, I want the pipeline to continue normally if the prediction market fetch fails or returns zero results (graceful degradation) so that a Polymarket outage does not break my daily run.
23. As a trader, I want the number of prediction market records passed to the summarizer capped (default: top 20 by volume) so that the LLM context window is not overwhelmed by hundreds of low-relevance markets.

### Report

24. As a trader, I want the Executive Summary section of the report to list the top 5 prediction market events (by volume) and their probabilities so that I can see the market consensus at a glance without reading the full macro section.
25. As a trader, I want the Macro Environment section of the report to include the full list of fetched prediction markets (above the volume threshold) as a table so that the complete picture is documented.

---

## Implementation Decisions

### Modules

| Module | Action | Responsibility |
|---|---|---|
| `prediction_market_client` | New | Fetch, filter, cache, format, and insert Polymarket markets into RAG |
| `rag_store` | New (separate PRD) | ChromaDB wrapper; must be built first; exposes insert and retrieve |
| `news_summarizer` | Modified | Accept optional markets parameter; add prediction market section to LLM prompt |
| `pipeline_orchestrator` | Modified | Add Stage 1b: fetch markets; pass to summarizer; wire market_client into _build_modules |
| `report_generator` | Modified | Add top-5 prediction markets to Executive Summary; add full markets table to Macro section |

### PredictionMarketClient Interface

Constructor:
  - cache_dir: str
  - min_volume: float = 100_000
  - categories: list[str] = ["Economics", "Politics", "Crypto"]
  - top_n_for_summarizer: int = 20

Public method:
  - fetch(as_of_date: str) -> list[dict]

Each returned dict contains:
  - market_id: str — Polymarket unique market ID (used for RAG deduplication)
  - event: str — market question title
  - probability: float — YES probability, 0.0 to 1.0
  - volume: float — raw total volume in USD
  - status: str — "active" or "resolved"
  - resolved_yes: bool or None — True/False if resolved, None if active
  - end_date: str or None — YYYY-MM-DD
  - category: str
  - formatted_text: str — ready for RAG insert and LLM prompt injection

### Polymarket API

- Base: Polymarket CLOB API, public endpoints only
- No authentication required — read-only market data is publicly accessible
- Fetches active markets and resolved markets (last 3 months) in separate requests
- Paginates until all pages exhausted; small inter-request delay to respect rate limits

### Caching

- Cache file: {cache_dir}/{as_of_date}_prediction_markets.json
- On fetch: check for cache file first; if present, return without hitting API
- Write cache only after a successful, non-empty fetch
- Never cache on API error, timeout, or empty response

### Date Scope

- Active markets: all currently open markets (no date filter)
- Resolved markets: resolved within the last 3 months (aligns with news corpus window)
- Both filtered by category and volume before return

### NewsSummarizer Changes

- summarize(articles, as_of_date, markets=None) — markets is optional; if None or empty, prompt unchanged
- New prompt section injected before the summary request, listing top markets by volume
- Markets sorted by volume descending; top top_n_for_summarizer records included

### Volume Formatting

- >= 1,000,000: format as $XM (e.g. $2.3M)
- >= 1,000: format as $XK (e.g. $450K)
- < 1,000: format as $X (e.g. $950)

### Build Order

1. rag_store.py — built and tested first (separate PRD)
2. prediction_market_client.py — built against the rag_store interface
3. news_summarizer, pipeline_orchestrator, report_generator wiring

---

## Testing Decisions

### What Makes a Good Test

Tests verify external behaviour through the public interface — given a known input (mock API response, fixture), assert the correct output. Tests must not assert on private methods, internal state, or log messages.

### Modules to Test

**prediction_market_client** — highest priority, pure deterministic logic:
- Given a mock API response with markets of mixed categories and volumes, assert only markets matching configured categories and above min_volume are returned.
- Given a cached file on disk, assert fetch() returns cached data without making any HTTP requests.
- Given a successful API response, assert the cache file is written.
- Given an API failure (exception or non-200), assert fetch() returns an empty list and does not write a cache file.
- Given a resolved market with resolved_yes=True, assert formatted_text contains "Resolved: YES".
- Given a market with volume 2,300,000, assert formatted_text contains "$2.3M".
- Given a market with volume 450,000, assert formatted_text contains "$450K".
- Given a market with probability 0.73, assert formatted_text contains "73%".

**news_summarizer (modified behaviour)**:
- Given markets=None, assert the LLM prompt is unchanged from current behaviour.
- Given markets=[], assert the LLM prompt is unchanged.
- Given a non-empty markets list, assert the prompt contains a prediction market context section.

**pipeline_orchestrator (modified behaviour)**:
- Assert market_client.fetch() is called with the correct run date.
- Assert the result is passed to summarizer.summarize() as the markets keyword argument.
- Assert that if market_client.fetch() raises an exception, the pipeline continues and calls summarizer.summarize() with markets=[].

### Prior Art

Follow the _modules injection pattern in test/test_pipeline_orchestrator.py — inject market_client via the _modules dict with a MagicMock. HTTP calls in PredictionMarketClient should be patched with unittest.mock.patch on requests.get, consistent with the approach used in Stage1DataCollector tests.

---

## Out of Scope

- Trading on Polymarket: read-only, no authentication, no private endpoints.
- Per-ticker prediction market retrieval via RAG: part of the RAG store PRD.
- Historical data beyond 3 months: Polymarket public API does not expose deep archives.
- Kalshi or other prediction market APIs: only Polymarket CLOB API in scope.
- Crypto as a tradeable instrument: crypto markets used as macro signal only.
- Real-time / streaming updates: data fetched once per run and cached.

---

## Further Notes

- No API key required: Polymarket exposes market data publicly, so no secrets management is needed for this module.
- Volume as conviction proxy: a 90% probability on a $1,000 market is noise; a 55% probability on a $10M market is a genuine macro signal. The volume threshold enforces this distinction.
- Build order dependency: rag_store.py must exist before prediction_market_client.py is built. The client calls rag_store.insert() after each successful fetch. If the RAG store is unavailable, the client should still return the formatted markets list so Stage 2 (summarizer) is not blocked.
- Context window discipline: the summarizer already receives stock_news, global_news, and industry_news. Capping at top 20 markets by volume keeps the prediction market addition to approximately 600 tokens, well within Qwen3-8B limits.
