# PRD: MFT Alpha Finder & Evaluator — Full System

## Problem Statement

As a retail trader or independent analyst, I need to systematically identify high-quality trading opportunities across US equities every day — before the market opens. Currently, the process of collecting news, assessing macro conditions, screening individual stocks, classifying market regimes, and backtesting candidate strategies is either done manually (slow, inconsistent) or fragmented across disconnected tools (no unified signal). There is no single system that takes raw data all the way through to a fully evaluated, backtested trade recommendation in a single automated pipeline.

---

## Solution

A multi-stage automated pipeline — MFT Alpha Finder & Evaluator — that:

1. **Collects and curates** news, macro signals, and prediction market data into a persistent, queryable knowledge base
2. **Screens** the market using an LLM to identify a shortlist of high-conviction ticker candidates
3. **Classifies the market regime** for each candidate using quantitative signals and selects an appropriate strategy
4. **Validates** each strategy against hard diagnostic thresholds before proceeding
5. **Backtests** surviving strategies and computes trade parameters
6. **Outputs a single Markdown report** with all findings, reasoning, and graph-ready data

The system runs on-demand (recommended pre-market) and produces a structured report the user can act on independently. No automated execution — all trade decisions remain with the human.

---

## User Stories

### Stage 1 — Data Collection & Intelligence

1. As a trader, I want the system to automatically fetch the latest stock news, global macro news, industry news, and prediction market data so that I don't have to manually aggregate sources each day.
2. As a trader, I want news data cached by date so that re-running the pipeline mid-day does not re-scrape articles already collected today.
3. As a trader, I want prediction market data (event probabilities and bet volumes) ingested alongside news so that macro event probabilities inform my thesis.
4. As a trader, I want all collected news embedded into a persistent vector database so that relevant articles can be retrieved per-ticker later in the pipeline.
5. As a trader, I want an LLM-generated daily summary of the last 7–14 days of news so that the screener has a concise, high-signal context to reason over.
6. As a trader, I want the pipeline to be user-triggered (not scheduled) so that I can run it at any time — pre-market, mid-day, or end-of-day.
7. As a trader, I want the news corpus to cover a rolling 3-month window so that medium-term trends and themes are captured.
8. As a trader, I want the system to clearly log which data sources were fetched vs. served from cache on each run so that I know how fresh my data is.

### Stage 2 — Screening & Shortlisting

9. As a trader, I want the system to pre-filter the full list of news-mentioned tickers to the top 50 by composite news score so that the LLM screener operates on a manageable, high-signal candidate set.
10. As a trader, I want the LLM to produce a macro filter call identifying favored sectors/themes, sectors/themes to avoid, and active macro risks so that all subsequent stock analysis is grounded in the current macro environment.
11. As a trader, I want the LLM to shortlist exactly 20 tickers from the top-50 pre-filtered candidates using the macro filter and computed OHLCV summaries as inputs so that I receive a focused, actionable candidate set.
12. As a trader, I want each shortlisted ticker to receive a dedicated focused LLM call using its macro thesis, retrieved RAG news chunks (max 3 per ticker), and computed OHLCV features so that the per-ticker verdict is grounded in both narrative and price evidence.
13. As a trader, I want each per-ticker LLM call to return a structured verdict (buy / watch / avoid) with reasoning so that the output is machine-parseable and consistently formatted.
14. As a trader, I want the system to retry LLM calls that return malformed or unparseable JSON output (up to 3 retries) so that transient model failures don't silently corrupt the pipeline.
15. As a trader, I want OHLCV summaries to include computed features (20-day return, RSI, ATR, 52-week high/low proximity, volume ratio vs. 30-day avg) rather than raw OHLCV rows so that the LLM context window is used efficiently.
16. As a trader, I want tickers that produce unparseable output after all retries to be logged and marked "watch" by default so that the pipeline never silently drops a candidate.

### Stage 3 — Regime Classification & Strategy Selection

17. As a trader, I want the system to compute a quantitative regime label for each shortlisted ticker using the Hurst exponent (30-day rolling) and ATR/price ratio so that strategy selection is driven by objective price behavior, not LLM opinion.
18. As a trader, I want tickers with Hurst > 0.55 classified as "Trending" and mapped to the Momentum strategy family so that trend-following logic is only applied when trend persistence is statistically supported.
19. As a trader, I want tickers with Hurst < 0.45 classified as "Mean-Reverting" and mapped to the Mean Reversion strategy family so that oscillating price behavior is matched to a compatible strategy.
20. As a trader, I want tickers with ATR/price > 3% flagged as "High-Volatility" so that position sizing and risk management can be adjusted accordingly.
21. As a trader, I want tickers with ATR/price < 1.5% flagged as "Low-Volatility" so that breakout and tight-stop strategies can be considered.
22. As a trader, I want the LLM to receive the regime label, news sentiment summary, and price chart summary for each ticker and return adjusted strategy parameters within the pre-defined template so that qualitative context refines but never overrides the quantitative regime signal.
23. As a trader, I want the Momentum strategy template to use the following base rules: entry on close > 20-day high with volume > 1.5× 20-day avg; exit on trailing stop at 2×ATR or close < 10-day MA; stop loss at 1.5×ATR below entry.
24. As a trader, I want the Mean Reversion strategy template to use the following base rules: entry when RSI(14) < 30 AND price touches the lower Bollinger Band (20-day, 2σ); exit when RSI > 55 OR price reaches the middle Bollinger Band; stop loss at 1.5×ATR below entry.
25. As a trader, I want the LLM's parameter adjustments (e.g., "use 30-day high instead of 20-day") to be logged alongside the hard regime signal so that I can audit what the model changed and why.

### Stage 4 — Diagnostics & Validation

26. As a trader, I want each candidate strategy to be evaluated against 7 sequential diagnostic tests — Return & Growth Decomposition, Volatility Profiling, Drawdown Analysis, Risk-Adjusted Ratio Table, Tail Risk & Distribution, Stress & Robustness Testing, Walk-Forward Validation — so that only strategies with real edge proceed to backtesting.
27. As a trader, I want hard reject thresholds applied automatically before any LLM review: Sharpe ratio < 0.5, max drawdown > 30%, walk-forward degradation > 50% vs. in-sample, win rate < 35% → auto-reject so that the LLM is never asked to justify a clearly bad strategy.
28. As a trader, I want strategies that pass all hard floors to be reviewed by the LLM, which adds qualitative commentary on strengths and weaknesses, so that I receive both quantitative validation and narrative context.
29. As a trader, I want auto-rejected strategies to appear in the report with the specific threshold that was breached so that I understand why a candidate was eliminated.
30. As a trader, I want the diagnostic results for each strategy stored in a structured format so that they can be rendered as tables and charts in the final report.

### Stage 5 — Backtesting & Report Generation

31. As a trader, I want backtests to run on 2 years of OHLCV history (fetched via yfinance) so that the strategy evaluation covers sufficient market cycles, even though the news corpus only spans 3 months.
32. As a trader, I want the backtest to use volatility-adjusted position sizing: risk 1% of portfolio per trade, position size = (portfolio × 0.01) / (2 × ATR) so that position sizes are consistent with the regime classifier's ATR outputs.
33. As a trader, I want the backtest to return, for each trade: entry date, entry price, exit date, exit price, holding period, position size, P&L, and the exit reason (trailing stop / MA cross / RSI exit / stop loss) so that I can review the full trade lifecycle.
34. As a trader, I want the final report to be written as a Markdown file containing all pipeline outputs so that I can read it in any Markdown viewer.
35. As a trader, I want the report to contain the following sections: Executive Summary, Macro Environment & Prediction Market Signals, Shortlisted Tickers with Buy/Watch/Avoid Verdicts, Regime Classification per Ticker, Strategy Selected & Parameters, Diagnostic Results, Backtest Results so that the full pipeline reasoning is documented in one place.
36. As a trader, I want the report to include raw data tables and data series for each chart (equity curve, drawdown curve, return distribution, walk-forward results) so that I can render visualizations from the Markdown data without needing the pipeline to generate image files.
37. As a trader, I want the report filename to include the run date and time so that reports from different runs are distinguishable.

---

## Implementation Decisions

### Architecture: Sequential Multi-Stage Pipeline

The system is implemented as a linear pipeline of stages. Each stage consumes the output of the previous stage. The pipeline is triggered by a single user-facing entry point and runs all stages to completion before writing the report.

### Modules

| Module | Responsibility |
|---|---|
| `pipeline_orchestrator` | Top-level entry point; runs stages 1–5 in sequence; handles stage failures |
| `data_collector` | Orchestrates all API scraping with per-date caching; delegates to source-specific clients |
| `prediction_market_client` | Fetches prediction market event probabilities and bet volumes from chosen API (Kalshi or Polymarket — **TBD**) |
| `rag_store` | ChromaDB wrapper; persistent across runs; exposes insert(articles) and retrieve(ticker, k=3) |
| `llm_client` | Local Qwen2.5-7B wrapper; enforces structured JSON output; retries up to 3× on parse failure; marks ticker "watch" after exhausted retries |
| `news_summarizer` | Generates daily LLM summary of last 7–14 days of news for use as screener context |
| `ohlcv_fetcher` | yfinance integration; fetches 2 years of daily OHLCV; computes summary features (20-day return, RSI, ATR, 52-week proximity, volume ratio) |
| `macro_screener` | Runs macro filter LLM call; returns favored sectors, avoid sectors, active macro risks |
| `ticker_screener` | Pre-filters to top 50 by composite news score; runs LLM shortlisting to 20; runs per-ticker focused LLM calls |
| `regime_classifier` | Computes Hurst exponent (30-day rolling) and ATR/price ratio; returns regime label per ticker |
| `strategy_selector` | Maps regime label to strategy template; runs LLM parameter adjustment call; returns final strategy config per ticker |
| `diagnostics_engine` | Runs 7 sequential diagnostic tests; applies hard reject floors; returns pass/fail + metrics per strategy |
| `backtester` | Executes strategy rules on 2-year OHLCV history; returns trade log + equity curve + summary stats |
| `report_generator` | Assembles all stage outputs into structured Markdown report with data tables |

### Key Technical Decisions

- **LLM**: Qwen2.5-7B running locally (via Ollama or llama.cpp). All LLM calls use structured JSON output mode. Parse failures trigger retry up to 3×.
- **Vector Store**: ChromaDB, persistent on disk. Built incrementally — articles are inserted on scrape and never re-embedded if already present (deduplicated by article URL or ID).
- **RAG Retrieval**: Max 3 chunks per ticker per call to control context window size.
- **Pre-filter before LLM shortlisting**: Top 50 tickers by composite news score (keyword score × 3 + ticker mentions × 2 + publisher trust + recency) are passed to the LLM. The LLM picks 20 from this set.
- **Prediction market data**: Treated as text and embedded into ChromaDB alongside news. Probability values should be formatted explicitly in the text (e.g., "Probability: 73%, Volume: $2.3M") to preserve numeric precision in embeddings.
- **Caching**: News scraping checks for existing data by source + date before making API calls. OHLCV data fetched fresh on each run (fast, no rate limit concerns with yfinance).
- **Backtest data**: 2 years of OHLCV from yfinance, independent of the 3-month news corpus window.
- **Position sizing**: Volatility-adjusted. Risk per trade = 1% of portfolio. Stop = 2×ATR. Size = (portfolio × 0.01) / (2 × ATR).
- **Regime thresholds (defaults, tunable)**:
  - Hurst > 0.55 → Trending
  - Hurst < 0.45 → Mean-Reverting
  - 0.45 ≤ Hurst ≤ 0.55 → falls back to ATR classification only
  - ATR/Price > 3% → High-Volatility
  - ATR/Price < 1.5% → Low-Volatility
- **Diagnostic hard floors**: Sharpe < 0.5, Max Drawdown > 30%, Walk-Forward degradation > 50%, Win Rate < 35% → auto-reject without LLM review.
- **Report format**: Single `.md` file per run, filename includes ISO date and run time. No image generation — chart data is written as Markdown tables/code blocks for the user to render.

### Resolved Decisions

1. **Prediction market API**: **Polymarket** — use the Polymarket REST API (CLOB API). Fetch active markets, current probabilities, and total volume. Embed as formatted text: `"Event: [title] | Probability: [X]% | Volume: $[Y]"`.
2. **Max holding period**:
   - **Momentum strategies**: 20 trading days (4 calendar weeks). Momentum trades need time to develop; forced exit at 20 days if neither trailing stop nor MA cross triggers.
   - **Mean reversion strategies**: 10 trading days (2 calendar weeks). Mean reversion is a faster trade; if price hasn't reverted in 2 weeks, the thesis is broken.
3. **Run time SLA**: 90 minutes maximum per full pipeline run. No per-stage time limits — the pipeline runs to completion.
4. **Summary window**: User-configurable at runtime, between 7 and 14 days. Default = 7 days. If run during the current month, window counts back from the current date. If run against a past date, window counts back from that date.

---

## Testing Decisions

### What Makes a Good Test

Tests should verify the external behavior of a module given defined inputs — not its internal implementation. A good test asks: "given this input, does this module produce the correct output?" Tests should not mock internal logic or assert on private state.

### Modules to Test

| Module | What to Test |
|---|---|
| `regime_classifier` | Given known price series with confirmed Hurst values, assert correct regime label is returned. Pure math — fully deterministic, highest priority. |
| `diagnostics_engine` | Given synthetic trade logs, assert that hard floor thresholds correctly trigger auto-reject. Assert diagnostic metrics are computed correctly (Sharpe, max DD, walk-forward degradation). |
| `backtester` | Given synthetic OHLCV data with known patterns, assert that momentum entry/exit rules fire at correct bars, position sizes are computed correctly, and P&L is accurate. |
| `rag_store` | Assert that inserting N articles and retrieving for a known ticker returns ≤ 3 most relevant chunks. Assert that re-inserting a duplicate article does not create duplicates in the store. |
| `llm_client` | Assert that malformed JSON responses trigger retry logic up to the configured limit. Assert that a ticker is marked "watch" after exhausting retries. Use a mock LLM backend for deterministic testing. |
| `ticker_screener` (pre-filter only) | Assert that given a list of scored articles, the pre-filter returns exactly the top 50 by composite score. The LLM shortlisting step is not unit tested (integration concern). |
| `ohlcv_fetcher` | Assert that computed summary features (RSI, ATR, volume ratio) are correctly derived from a known OHLCV fixture. |

### Modules Not Tested

`news_summarizer`, `macro_screener`, `strategy_selector`, `report_generator` — these modules are primarily LLM-call orchestrators or text formatters. Their correctness depends on model output quality, which cannot be meaningfully unit tested. Integration-level review of their output format is sufficient.

---

## Out of Scope

- **Automated trade execution**: The system produces reports only. No brokerage API integration, no order placement.
- **Real-time / streaming data**: The pipeline is batch-run on-demand. No tick data, no intraday streaming.
- **Options, futures, or crypto assets as primary instruments**: The screener targets US equities. Prediction market data is used as a macro signal only, not as a tradeable instrument.
- **Portfolio-level optimization**: Position sizing is per-trade. No cross-ticker correlation, portfolio variance optimization, or multi-leg strategies.
- **Live paper trading simulation**: No connection to a simulated brokerage environment.
- **Web UI or dashboard**: Output is a Markdown file. No frontend.
- **Multi-user support**: Single-user local installation only.
- **Model fine-tuning**: Qwen2.5-7B is used as-is. No fine-tuning on financial data.
- **Women's or other market prediction models**: March Madness predictor already in the repo is a separate, independent module and is not integrated into this pipeline.

---

## Further Notes

- **Cost profile**: All LLM inference is local (Qwen2.5-7B via Ollama or llama.cpp). Only costs are API calls to news sources and yfinance (free tier). The system is designed to run on a consumer GPU or CPU with sufficient RAM.
- **7B model quality caveat**: Qwen2.5-7B is capable but will produce lower-quality reasoning on complex financial prompts compared to larger models. Mitigation: structured output enforcement, retry logic, hard quantitative floors before LLM review, and small context windows per call (≤ 4K tokens per ticker call).
- **3-month news vs. 2-year backtest asymmetry**: The news corpus drives screener and regime inputs. The backtest uses 2 years of price history. These are intentionally decoupled — the screener finds *current* opportunity, the backtest validates *historical* strategy viability.
- **Existing codebase reuse**: `stock_news.py`, `global_news.py`, `industry_news.py`, and `dataset_curator.py` already implement the news fetch + composite scoring pipeline. These should be refactored into the `data_collector` module rather than rebuilt. The composite score they produce feeds directly into the pre-filter step.
- **Recommended run time**: Pre-market (7:00–9:00 AM ET) for maximum news freshness and time to review the report before the open.
