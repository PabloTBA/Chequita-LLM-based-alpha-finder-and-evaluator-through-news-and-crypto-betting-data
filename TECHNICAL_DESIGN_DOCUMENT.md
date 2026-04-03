# Chequita — Technical Design Document

**Version:** 1.0  
**Date:** 2026-04-03  
**Authors:** Chequita Engineering Team  
**Audience:** Engineers & Competition Judges

---

## Table of Contents

1. [System Purpose & Philosophy](#1-system-purpose--philosophy)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Pipeline Flow](#3-pipeline-flow)
4. [Module Reference](#4-module-reference)
   - [4.1 Data Collection Layer](#41-data-collection-layer)
   - [4.2 Signal Generation Layer](#42-signal-generation-layer)
   - [4.3 Strategy & Execution Layer](#43-strategy--execution-layer)
   - [4.4 Backtesting & Evaluation Layer](#44-backtesting--evaluation-layer)
   - [4.5 Orchestration & Reporting](#45-orchestration--reporting)
5. [Design Decisions & Rationale](#5-design-decisions--rationale)
6. [Parameters, Thresholds & Metrics](#6-parameters-thresholds--metrics)
7. [External Dependencies](#7-external-dependencies)
8. [Testing Architecture](#8-testing-architecture)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)

---

## 1. System Purpose & Philosophy

### What Chequita Does

Chequita is a **hybrid LLM-RAG framework for medium-frequency trading alpha discovery and evaluation**. It ingests unstructured news data and prediction market signals, distills macroeconomic and sector context using a locally-hosted LLM, generates quantitative signals over historical price data, backtests those signals under realistic market conditions, and outputs a ranked, risk-managed set of trade recommendations.

The system targets a **holding period of 1–20 trading days** — long enough that execution friction is manageable, short enough that signals remain timely.

### Core Philosophy

**1. Multi-signal ensemble, not a single oracle.** No single indicator reliably predicts markets. Chequita runs four independent alpha components, four independent ML models, and two LLM-backed qualitative filters in parallel — then aggregates. This reduces the chance that any one model's biases dominate the output.

**2. News is macro context, not a direct trading signal.** Raw article sentiment is noisy and subject to media framing. Instead of mapping article sentiment directly to buy/sell decisions, we use an LLM to extract structured qualitative intelligence (favored sectors, active risks, market bias direction) and use that to *modulate weights* in the quantitative pipeline.

**3. Out-of-sample rigor is non-negotiable.** Every signal is applied with a one-bar lag (`shift(1)`) to avoid look-ahead bias. ML models use strict walk-forward validation with a one-year minimum training window before firing their first prediction. Monte Carlo simulation uses block bootstrap (not IID) to preserve the serial autocorrelation of losing streaks.

**4. Regime-awareness as a first-class citizen.** Markets behave fundamentally differently during crises, trending markets, and low-volatility mean-reverting periods. The system classifies market regime before selecting any strategy, ensuring strategies are contextually appropriate rather than applied blindly.

**5. Risk is quantified, not ignored.** Every backtest produces a full diagnostics report: Sharpe, Sortino, maximum drawdown, CVaR-95, ruin probability, and Kelly fraction. Portfolio construction applies volatility parity and correlation shrinkage — institutional-grade risk budgeting with transparent, interpretable math.

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Chequita System                             │
│                                                                    │
│  ┌────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   Data Layer   │    │  Signal Layer    │    │  Eval Layer   │  │
│  │                │    │                  │    │               │  │
│  │ Benzinga API   │───▶│ AlphaEngine      │───▶│ Backtester    │  │
│  │ yfinance       │───▶│ MLSignalEngine   │───▶│ Monte Carlo   │  │
│  │ PredMkts Cache │───▶│ RegimeClassifier │    │ Diagnostics   │  │
│  │ ChromaDB (RAG) │    │ PairSelector     │    │ Portfolio Opt │  │
│  └────────────────┘    └──────────────────┘    └───────────────┘  │
│           │                     │                      │           │
│           ▼                     ▼                      ▼           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               PipelineOrchestrator                          │  │
│  │   (coordinates all modules, manages state, writes reports)   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                        │
│    ┌──────┴───────┐                                               │
│    │   LLM Layer  │                                               │
│    │              │                                               │
│    │ Ollama/Qwen3 │ ◀── NewsSummarizer                           │
│    │   (local)    │ ◀── MacroScreener                            │
│    └──────────────┘                                               │
│           │                                                        │
│    ┌──────┴───────┐                                               │
│    │  Output Layer│                                               │
│    │              │                                               │
│    │ FastAPI /SSE │                                               │
│    │ Markdown Rpts│                                               │
│    │ TradeZero API│                                               │
│    └──────────────┘                                               │
└────────────────────────────────────────────────────────────────────┘
```

The architecture is deliberately **sequential within a ticker, parallel across tickers**. Each stock is processed through its own regime classification → signal generation → strategy selection → backtesting pipeline. Modules communicate via plain Python dicts and Pandas DataFrames — no shared mutable state, no message queues, no databases mid-pipeline.

---

## 3. Pipeline Flow

```
INPUT
├── Date range & ticker universe
├── Benzinga API key
├── Local Ollama endpoint (Qwen3:14b)
└── Configuration (window_days, max_tickers, portfolio size)

STAGE 1 — DATA COLLECTION
├── Stage1DataCollector.collect_range()
│   ├── Fetches stock news, global macro news, industry news from Benzinga
│   ├── Applies composite impact scoring (keyword × 3, ticker × 2, publisher × 1, recency × 1)
│   └── Caches per-date JSON (prevents redundant API calls on re-runs)
├── OHLCVFetcher.fetch()
│   ├── Downloads 2–10 years of daily OHLCV from yfinance
│   ├── Computes 12 technical features (RSI, ATR, Bollinger position, momentum, etc.)
│   └── Appends earnings drift flags
└── PredictionMarketClient.fetch()
    ├── Loads prediction markets from local cache
    ├── Filters by category relevance and minimum volume ($10K)
    └── Optionally embeds markets into ChromaDB RAG store

STAGE 2 — LLM-POWERED NEWS ANALYSIS
├── NewsSummarizer.summarize()
│   ├── Samples top-5 highest-scoring articles from each of 3 news sources
│   ├── Calls Ollama (Qwen3:14b) with structured prompt
│   └── Returns: summary, top_themes, market_bias ("bullish"/"bearish"/"neutral"), key_risks
└── MacroScreener.screen()
    ├── Sends news summary + prediction market context to Ollama
    └── Returns: favored_sectors[], avoid_sectors[], active_macro_risks[], market_bias

STAGE 3 — PER-TICKER SIGNAL GENERATION
For each ticker in universe:
├── RegimeClassifier.classify()
│   ├── R/S Hurst exponent analysis (756-bar window, log-spaced lags)
│   ├── ATR/price ratio (crisis/high/low volatility thresholds)
│   ├── 20-day return direction
│   ├── Earnings proximity check (± 5 days via yfinance calendar)
│   └── Returns: {regime, hurst, atr_pct, ret_20d, near_earnings}
├── AlphaEngine.compute()
│   ├── CS-MR: 5-day cross-sectional mean reversion (weight 0.40 normal / 0.20 bearish)
│   ├── Residual: 60-day rolling beta residual vs SPY (weight 0.30 / 0.10)
│   ├── Vol-spike: 20-day volume z-score (weight 0.20 / 0.45)
│   ├── Mom-2d: 2-day price momentum (weight 0.10 / 0.25)
│   ├── PCA orthogonalization if signal pairwise corr > 0.70
│   └── shift(1) lag → no look-ahead bias
└── MLSignalEngine.compute()
    ├── CS-GBM: Universe-panel GBM on full cross-section
    ├── Regime-GBM: Per-ticker GBM with regime flags as additional features
    ├── Online-SGD: Partial-fit SGD updated every bar (never stale)
    ├── Calibrated-Ensemble: GBM + LR + RF averaged
    └── Walk-forward: min 252-bar train window, 63-bar refit, 5-day forward target

STAGE 4 — STRATEGY SELECTION & EXECUTION ADVISORY
├── StrategySelector.select(ticker, regime, market_bias)
│   ├── Routes to one of: Momentum, MeanReversion, VolatilityBreakout, AlphaCombined
│   ├── Each strategy defines: entry trigger, exit rule, max holding, risk per trade
│   └── Position size = 1% portfolio risk / ATR
└── ExecutionAdvisor.advise()
    ├── Fetches live bid/ask spread from yfinance
    ├── Estimates slippage (live spread or ATR proxy if spread unavailable)
    └── Classifies market impact: negligible / moderate / significant

STAGE 5 — BACKTESTING
├── Backtester.run()
│   ├── Walk-forward entry/exit simulation
│   ├── Slippage: 0.05% per side; Commission: 0.10% per side
│   ├── Multi-day holdings (1–20 days per strategy limits)
│   └── Returns: trade_log[], equity_curve, daily_returns, max_drawdown, sharpe
└── MonteCarloEngine.run()
    ├── 10,000 block-bootstrap simulations of trade_log
    ├── Block size = average holding period (preserves serial autocorrelation)
    └── Returns: equity distribution (p5/p50/p95), Sharpe dist, Kelly, ruin probability

STAGE 6 — DIAGNOSTICS & PORTFOLIO CONSTRUCTION
├── DiagnosticsEngine.run()
│   ├── Sharpe ratio, Sortino ratio, max drawdown, CVaR-95
│   ├── Win rate distribution, trade expectancy, capacity estimate
│   └── Regime stress testing (performance segmented by regime)
└── PortfolioOptimizer.optimize()
    ├── Layer 1: CS-momentum rank filter (keep top 60% by 12-1m momentum)
    ├── Layer 2: Volatility-parity sizing (weight ∝ 1/realized_vol)
    ├── Layer 3: Correlation shrinkage (scale both weights when corr(i,j) > 0.70)
    └── Returns: allocations[], portfolio_sharpe, portfolio_returns

OUTPUT
├── SummaryReport.md — full technical breakdown per ticker/strategy
├── TraderSummary.md — execution brief (entry/exit levels, position sizes, rationale)
├── FastAPI /api/report endpoint (browser-accessible)
└── Optional: TradeZero API order submission
```

---

## 4. Module Reference

### 4.1 Data Collection Layer

#### `Stage1DataCollector` — News Ingestion

**Purpose:** Fetch and score news articles from Benzinga across three dimensions (stock-specific, global macro, industry-level) for a given date range.

**Why Benzinga:** Benzinga provides machine-readable, timestamped financial news with ticker metadata already extracted. This eliminates a full NER pass and gives reliable article-to-ticker mappings. The API delivers structured JSON making batch ingestion trivial.

**Why composite scoring vs. raw sentiment:**

Raw sentiment (positive/negative) from pretrained models is often too noisy for trading decisions. A CFO quote about "cautious optimism" can read as positive while signaling guidance cuts. Instead, the composite score ranks *relevance and urgency* — which directly determines which articles the LLM summarizes and which get embedded in the RAG store:

```
composite_score = (keyword_score × 3) + (ticker_score × 2) + publisher_score + recency_score

keyword_score:    Presence of high-signal terms (earnings, upgrade, catalyst, layoff, acquisition...)
ticker_score:     Number of distinct tickers mentioned (more specific = more actionable)
publisher_score:  Trust weight by source (Reuters/Bloomberg/WSJ > generic aggregators)
recency_score:    Proximity to NYSE open (0–3; articles right before market open score highest)
```

**Why caching per-date JSON:** Running LLM inference and API calls is expensive in both time and cost. If a run fails mid-pipeline or the user re-runs for the same date window, we skip re-fetching. Cached data also enables reproducible backtests — the same news snapshot is used every time.

**Key parameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| `REQUEST_DELAY` | 1.0s | Respect Benzinga API rate limit |
| `MAX_PAGE_SIZE` | 100 articles/request | Benzinga API maximum |
| `TICKER_BATCH_SIZE` | 30 | Stay below URL length limits |

---

#### `OHLCVFetcher` — Price Data & Feature Engineering

**Purpose:** Download historical OHLCV data and compute a standardized 12-feature engineering set used by both the alpha engine and ML signal engine.

**Why yfinance:** Free, reliable daily data going back 10+ years with built-in adjusted-close handling and earnings calendar access. For a research system, this is the appropriate choice. Production would swap to a paid data vendor.

**Why 12 features specifically:**

The feature set is designed to cover four independent dimensions of price behavior without redundancy:

| Feature | Dimension | Formula |
|---------|-----------|---------|
| `ret_5d`, `ret_10d`, `ret_20d` | Short/medium/long momentum | `close / close.shift(n) - 1` |
| `rsi_14` | Overbought/oversold | Standard Wilder RSI |
| `atr_pct` | Realized volatility | `ATR(14) / close` |
| `vol_ratio_20` | Volume activity | `volume / volume.rolling(20).mean()` |
| `bb_position` | Bollinger relative position | `(close - lower) / (upper - lower)` |
| `mom_12_1` | 12-minus-1 month momentum | `ret_252 - ret_21` (skip last month) |
| `vol_z_20` | Volume z-score | `(volume - vol_ma) / vol_std` |
| `close_to_ma50`, `close_to_ma200` | Trend alignment | `close / MA - 1` |
| `realized_vol_20` | Volatility level | `std(daily_ret, 20) × √252` |

The 12-minus-1 month momentum (`mom_12_1`) skips the most recent month deliberately — this is standard academic practice because 1-month returns exhibit short-term reversal, not continuation.

**Why `MIN_ROWS = 30`:** Below 30 bars, rolling statistics (RSI, ATR, Bollinger) produce unreliable estimates that will contaminate signals. Tickers with insufficient history are skipped cleanly.

---

#### `RAGStore` — Vector Store

**Purpose:** Persist news articles and prediction market descriptions as dense vector embeddings using ChromaDB, enabling semantic similarity retrieval at query time.

**Why ChromaDB:** Lightweight, embedded vector database — no external server required. Runs fully in-process, persists to local disk. Appropriate for the data scales involved (thousands of articles, not millions).

**Why two separate collections:** News articles and prediction market entries have very different linguistic patterns. A prediction market headline ("Will the Fed cut rates in June 2025?") is structured as a question with market metadata. Mixing these with news prose would reduce retrieval precision. Separate collections allow precise, targeted retrieval.

**Why SentenceTransformer embeddings:** Local inference, no API call, no latency. The `all-MiniLM-L6-v2` model is well-calibrated for short financial text and runs efficiently on CPU.

---

#### `PredictionMarketClient` — Prediction Market Context

**Purpose:** Incorporate crowd-sourced probability estimates (from prediction markets) as a qualitative signal about macro and sector risk.

**Why prediction markets at all:** Prediction markets aggregate distributed information from participants with financial stakes in accuracy. Unlike analyst consensus (which can be politically constrained or anchored to prior estimates), prediction market prices reflect the current best estimate of event probabilities. They provide a complementary signal to news — what the market currently *believes* will happen, not just what the media is writing about.

**Why filter by volume (`market_min_volume = $10,000`):** Low-volume markets have wide bid-ask spreads and are susceptible to manipulation or anchoring. We require at least $10K in traded volume to consider a market sufficiently liquid for its probability to be reliable.

**Why embed in RAG instead of parsing programmatically:** Prediction market questions are natural language. Embedding them allows the LLM to retrieve the most semantically relevant markets for a given news context — e.g., when news mentions the Fed, markets about "rate cut by June" surface naturally without keyword matching.

---

### 4.2 Signal Generation Layer

#### `RegimeClassifier` — Market Regime Detection

**Purpose:** Classify the current market environment for a given ticker into one of 8 regimes, enabling downstream strategy selection to be contextually appropriate.

**The 8 regimes and their detection logic:**

| Regime | Detection Criteria | Trading Implication |
|--------|-------------------|---------------------|
| Crisis | ATR/price ≥ 6% | Suspend momentum; consider vol-breakout only |
| Trending-Up | Hurst ≥ 0.55 + ret_20d > 0 | Momentum strategies have an edge |
| Trending-Down | Hurst ≥ 0.55 + ret_20d < 0 | Downside momentum; tighten risk |
| Mean-Reverting | Hurst ≤ 0.45 | Mean-reversion strategies work; avoid momentum |
| High-Volatility | ATR/price ≥ 3% | Wider stops; smaller size |
| Low-Volatility | ATR/price ≤ 1.5% | Volatility breakout setups ripen |
| Event-Driven | near_earnings = True + no structural regime | Tactical overlay; use AlphaCombined |
| Neutral | Default fallback | No strong structural signal |

**Why the Hurst exponent (not ADX or a moving average crossover):**

The Hurst exponent H measures the *degree of long-range dependence* in a time series. Unlike ADX (which measures the strength of a trend that already exists) or moving average crossovers (which lag by construction), H captures whether the *underlying price process itself* exhibits persistence (H > 0.5) or anti-persistence (H < 0.5).

A market with H = 0.60 is statistically more likely to continue in its current direction than a market with H = 0.50 (random walk). This is a structural property of the process, not a reactive indicator.

**Why R/S (rescaled range) analysis specifically:**

We use the classic Hurst-Mandelbrot R/S method rather than detrended fluctuation analysis (DFA) or wavelet decomposition. DFA is more robust to polynomial trends but adds complexity that isn't warranted for equity price series (which don't exhibit polynomial non-stationarity at daily resolution). R/S produces realistic H estimates in the 0.45–0.65 range for equity prices, matching the academic literature (Lo 1991).

**Why log-spaced lags in the R/S calculation:**

Using evenly-spaced lags would over-weight long-lag statistics (many data points at large τ). Log-spacing (10 to n//8) gives equal representation across timescales, and skipping very short lags (< 10) avoids contamination from GARCH-type conditional heteroscedasticity, which creates spurious short-lag persistence.

**Why a 756-bar (3-year) Hurst window:**

The Hurst exponent is a *population statistic* about the data-generating process. It requires enough data to reliably estimate. At 252 bars (1 year), sampling noise dominates. At 756 bars (3 years), we have sufficient observations for stable estimates while still being responsive to structural regime shifts (e.g., post-crisis regime changes).

**Why Event-Driven is the lowest-priority regime:**

Near-earnings is a *tactical overlay*, not a structural market property. If a stock is trending strongly (high Hurst), the structural regime (Trending-Up) dominates the classification. Event-Driven fires only when no structural signal is present. This prevents over-classifying every stock approaching earnings as Event-Driven and defaulting everything to AlphaCombined strategy — which would suppress the regime-specific edge captured by Momentum and MeanReversion strategies.

---

#### `AlphaEngine` — Multi-Component Alpha Signal

**Purpose:** Generate a single, composite alpha signal per ticker per day by combining four independent signal components with regime-adaptive weighting.

**The four components and their economic rationale:**

| Component | Lookback | Weight (Normal/Bearish) | Economic Rationale |
|-----------|----------|------------------------|-------------------|
| CS-MR (Cross-sectional Mean Reversion) | 5-day | 0.40 / 0.20 | Stocks that underperform peers over 5 days tend to revert; captures short-term institutional rebalancing |
| Residual (Beta-adjusted) | 60-day beta | 0.30 / 0.10 | Beta-neutralized return captures idiosyncratic alpha rather than market factor exposure |
| Vol-spike | 20-day volume z | 0.20 / 0.45 | Unusual volume precedes price moves; validated extensively in academic literature (Blume et al. 1994) |
| Mom-2d (2-day price momentum) | 2-day | 0.10 / 0.25 | Short-term continuation effect (Jegadeesh & Titman 1993 extended to ultra-short horizon) |

**Why the weights shift in bearish regimes:**

In a bull market, cross-sectional mean reversion (CS-MR) dominates because institutional rebalancing flows are predictable and regular. In a bearish/crisis market, rebalancing breaks down — institutional selling is directional, not mean-reverting. Volatility and momentum signals become more informative because fear-driven selling produces momentum, and volume spikes signal capitulation events. The bearish weight configuration (vol_spike 0.45, mom_2d 0.25) reflects this shift.

**Why cross-sectional z-scoring:**

Each component is z-scored across the universe at each timestep rather than z-scored through time per ticker. This removes market-wide level shifts (if all stocks have high volume today, that's not a signal) and makes the output comparable across tickers — a score of +1.5 means the ticker is 1.5 standard deviations more attractive than the universe median today.

**Why PCA orthogonalization (and when it fires):**

If alpha components are correlated with each other (average pairwise correlation > 0.70), the composite signal is effectively receiving redundant information — the same economic phenomenon measured multiple ways. PCA rotates the components into orthogonal dimensions (principal components), retaining 90% of cumulative variance. The first PC captures the dominant shared signal; later PCs capture residual independent variation. This prevents a single economic mechanism (e.g., a market-wide vol event) from being counted multiple times with inflated effective weight.

The 0.70 threshold is chosen because below that level, the linear combination already behaves near-independently and PCA would only add computational overhead without meaningful signal improvement.

**Why `shift(1)` lag:**

This is the single most important implementation detail for preventing backtest overfitting. All signals are computed from data available at close of day T, then shifted forward by 1 bar so they are only "used" at close of day T+1. This ensures the signal correctly represents what a trader would know the morning after the signal is generated — not information from the same bar being traded.

---

#### `MLSignalEngine` — Walk-Forward Ensemble ML

**Purpose:** Generate a machine learning-based probability signal using an ensemble of four models trained and validated in a strict walk-forward regime, with no future data leakage.

**Why four models instead of one:**

Model diversity is the most reliable hedge against overfitting. Each model captures a different aspect of the signal:

| Model | Type | What It Captures |
|-------|------|-----------------|
| CS-GBM | Gradient Boosted Machine, cross-sectional | Relative value across the universe at each date; which stocks are attractive vs. peers |
| Regime-GBM | GBM with one-hot regime flags | Interaction effects between price features and market regime; learns regime-specific entry patterns |
| Online-SGD | Stochastic Gradient Descent (partial_fit) | Continuously adapts to the most recent data; never stale between quarterly refits |
| Calibrated-Ensemble | GBM + LR + RF averaged | Classic bias-variance tradeoff diversification; LR adds interpretability, RF adds bagging robustness |

**Final ml_signal = mean probability across all four models.** Simple averaging works better than learned stacking in the low-data regime of per-ticker time series — stacking adds parameters that overfit on the small validation sets available.

**Why walk-forward with `_MIN_TRAIN = 252` bars:**

The first 252 bars (approximately 1 trading year) are used as pure training data with no signal output. This ensures the model has seen at least one full market cycle before making predictions. Firing signals after only 3 months of training risks capturing seasonal or regime-specific patterns that don't generalize.

**Why `_REFIT = 63` bars (quarterly):**

Markets are non-stationary. A model trained two years ago has "forgotten" recent regime changes. Quarterly refit (every 63 trading days) balances two competing concerns:
- **Recency:** The model incorporates data from the last quarter, which reflects the current regime.
- **Sample size:** With 63-bar refits, training windows grow steadily, so later models have more data, not less.

**Why `_FORWARD = 5` (5-day forward return as target):**

The system targets 1–20 day holding periods. A 5-day forward return is the natural "sweet spot" target — long enough to filter out 1-day noise, short enough to remain in the medium-frequency regime. Models predicting 1-day returns are dominated by microstructure noise; models predicting 30-day returns leak into fundamental analysis territory.

**Why shallow trees (`_DEPTH = 4`, `_MIN_LEAF = 20`, `_SUBSAMPLE = 0.8`):**

Financial time series has an extremely low signal-to-noise ratio. Deep trees (depth > 6) reliably memorize noise in training data and produce near-zero out-of-sample IC. Shallow trees (depth 3–5) act as weak learners that, when boosted (200 estimators), form a robust ensemble. `min_samples_leaf = 20` ensures no leaf is fit on fewer than 20 samples — preventing the model from specializing on individual historical events. Row subsampling (0.8) adds stochastic regularization (as in Random Forest) to reduce variance.

---

#### `PairSelector` — Statistical Arbitrage Pair Detection

**Purpose:** Identify pairs of tickers with statistically stable cointegrating relationships suitable for pairs trading (statistical arbitrage).

**Why cointegration, not just correlation:**

High correlation between two stocks does not mean their spread will revert to zero. Two stocks that are both trending up will be highly correlated but their spread may be expanding. Cointegration (Engle-Granger test) specifically tests whether a *linear combination* of two price series is stationary — meaning the spread does revert, and we can model it as an Ornstein-Uhlenbeck process with a finite half-life.

**The three-stage filter:**

1. **Pearson correlation ≥ 0.70 on log-returns (252-bar window):** Coarse filter. Pairs that aren't correlated at all have no economic relationship and likely no cointegrating vector. This eliminates 95%+ of pairs cheaply before the expensive statistical test.

2. **Engle-Granger ADF test (p-value ≤ 0.10):** The core cointegration test. We use p ≤ 0.10 rather than the conventional 0.05 because financial data is inherently noisy and a strict 5% threshold would eliminate genuinely tradeable pairs. The downstream OU half-life filter compensates — even if we accept some false positives here, pairs with non-convergent spreads are removed in step 3.

3. **Ornstein-Uhlenbeck half-life ∈ [2, 30] days:** The OU process describes spread mean-reversion speed. Half-life below 2 days means the spread reverts too fast to execute profitably (transaction costs eat the signal). Half-life above 30 days means reversion is too slow — capital is locked up for a month with no guarantee of convergence within a typical holding period. The [2, 30] window targets the medium-frequency sweet spot.

---

### 4.3 Strategy & Execution Layer

#### `StrategySelector` — Regime-to-Strategy Routing

**Purpose:** Map each (ticker, regime) combination to the strategy most likely to have positive expected value given the current market environment.

**The four strategies:**

**1. Momentum**
- *When:* Trending-Up regime + positive 2-day momentum + volume spike
- *Entry:* Close breaks 20-day high
- *Exit:* Close breaks 20-day low OR max 15 days
- *Rationale:* In trending markets (Hurst > 0.55), price persistence is statistically documented. The 20-day channel breakout is the simplest implementation of breakout momentum with century-long empirical support (Donchian 1970, Jegadeesh/Titman 1993).

**2. MeanReversion**
- *When:* Mean-Reverting regime (Hurst < 0.45) OR Neutral
- *Entry:* RSI(14) < 30 (oversold) + bounce from 5-day low
- *Exit:* RSI > 70 OR price approaches 52-week high
- *Rationale:* When Hurst < 0.45, the price process is anti-persistent — overshoots are followed by reversions. RSI < 30 identifies short-term overextension. The 5-day low confirmation prevents catching a falling knife on the first day of a decline.

**3. VolatilityBreakout**
- *When:* Low-Volatility regime (pre-breakout squeeze) OR High-Volatility regime (directional breakout)
- *Entry:* Close exits Bollinger Band squeeze
- *Exit:* Reversion back to band midpoint OR 1× ATR stop
- *Rationale:* Volatility follows a mean-reverting cycle (GARCH documented). Periods of compressed volatility (narrow Bollinger Bands) precede expansion. Trading the expansion captures directional moves without predicting direction in advance.

**4. AlphaCombined**
- *When:* Event-Driven regime OR when the LLM macro filter's confidence warrants multi-signal fusion
- *Entry:* Weighted combination of alpha_signal and ml_signal (IC-weighted)
- *Exit:* Regime-adaptive holding periods
- *Rationale:* When no dominant structural regime exists, multiple independent signals have roughly equal information content. Fusing them reduces variance without sacrificing much bias.

**Why position size = 1% portfolio risk / ATR:**

This is the classic fixed-fractional risk sizing rule (Van Tharp). Sizing by ATR ensures that higher-volatility positions receive smaller allocations in dollar terms, so each position contributes approximately equal dollar volatility to the portfolio — regardless of the stock's price or average daily range. The 1% risk budget per trade is a conservative institutional standard that limits ruin probability while providing meaningful participation.

---

#### `ExecutionAdvisor` — Pre-Trade Market Impact Assessment

**Purpose:** Provide an execution brief estimating real-world costs before trades are submitted.

**Why this exists at all:** Backtests with flat slippage assumptions are optimistic. Execution quality varies significantly by ticker liquidity, time of day, and trade size. The ExecutionAdvisor fetches live bid-ask spreads (not last price) and estimates actual execution cost, flagging positions where slippage could materially impact P&L.

**Market impact classification:**
- **Negligible:** Spread < 0.1% of price. Execute at market.
- **Moderate:** Spread 0.1–0.5%. Use limit orders; be patient.
- **Significant:** Spread > 0.5% or large position relative to ADV. Algo execution or split orders required.

---

### 4.4 Backtesting & Evaluation Layer

#### `Backtester` — Walk-Forward Simulation

**Purpose:** Simulate strategy execution on historical data under realistic market conditions to estimate out-of-sample performance.

**Why walk-forward instead of simple train/test split:**

A single train/test split produces one estimate of out-of-sample performance. Walk-forward testing produces *many* overlapping estimates — each using all available prior data for training, then testing on the next unseen window. This is the most realistic simulation of how a live strategy would perform month by month as it accumulates history.

**Slippage model:**

```
slippage = 0.05% per side (0.10% round-trip)
commission = 0.10% per side (0.20% round-trip)
total_friction = 0.30% round-trip
```

These figures are intentionally conservative. Modern equity execution (IBKR, direct market access) achieves 0.02–0.05% slippage for liquid names. Using 0.05% slippage and 0.10% commission gives realistic pessimistic estimates — strategies that survive these costs are robust.

**Why 1–20 day holding limits:**

Below 1 day: This is not a medium-frequency strategy; intraday dynamics require tick data.
Above 20 days: Signals decay. The ML model's 5-day forward target and alpha component lookbacks (2–20 days) lose information content at longer horizons. Hard-coding a 20-day maximum prevents over-holding losing positions that have clearly stopped working.

---

#### `MonteCarloEngine` — Distributional Risk Quantification

**Purpose:** Estimate the *distribution* of outcomes the strategy could plausibly achieve, not just its point estimate from one historical realization.

**Why block bootstrap (not IID bootstrap):**

IID (independent and identically distributed) bootstrap randomly resamples individual trades with replacement. This assumes trades are independent — but they're not. Losing streaks have positive serial autocorrelation: a strategy that's working poorly today tends to work poorly tomorrow (regime persistence). IID bootstrap underestimates the probability and duration of drawdown events.

Block bootstrap resamples *consecutive blocks* of trades with replacement, where block size equals the average holding period. This preserves the autocorrelation structure of trade outcomes — adjacent blocks represent the same market regime, so winning and losing streaks appear with their realistic frequency and duration.

**Why 10,000 simulations:**

The tail of the distribution (ruin probability, 95th-percentile max drawdown) requires a large number of simulations to estimate stably. Below 1,000 simulations, ruin probability estimates have unacceptably wide confidence intervals. 10,000 simulations brings standard error below 0.5% for ruin probability at the 5% level.

**Why `ruin_threshold = 40%`:**

A 40% portfolio drawdown represents the threshold at which most trading operations would face forced liquidation (either by risk management, investor redemptions, or margin calls). It's not an arbitrary number — it's where the practical consequences of continued trading become severe.

**Kelly fraction:**

The Kelly criterion gives the theoretically optimal fraction of capital to risk per bet for maximum long-run growth. However, full Kelly is aggressive and sensitive to estimation error. We report both full Kelly and half-Kelly. The half-Kelly is the practical recommendation — it sacrifices ~25% of expected log-growth rate in exchange for ~50% reduction in drawdown risk (MacLean, Thorp 2011). If Kelly comes out > 0.30 in simulation, it's capped — very high Kelly fractions typically indicate overfitting on the historical trade log.

**Key output metrics explained:**

| Metric | What It Means | Why It Matters |
|--------|--------------|----------------|
| `p5_final / p50_final / p95_final` | 5th/50th/95th percentile final equity | Shows range of plausible outcomes, not just median |
| `p_ruin` | Probability of 40%+ drawdown before end | Direct risk metric for risk management decisions |
| `p95_max_drawdown` | Worst-case expected drawdown in 95% of simulations | Position sizing calibration |
| `kelly_fraction` | Optimal trade size as fraction of capital | Sizing guidance with theoretical basis |
| `p95_max_consec_losses` | Max losing streak in 95th percentile sim | Emotionally-relevant; drives investor communication |
| `median_time_to_ruin` | Expected months before ruin if it occurs | Useful for capital reserve planning |

---

#### `DiagnosticsEngine` — Per-Strategy Risk Analytics

**Purpose:** Decompose strategy performance into its risk/return components and stress-test against market regimes.

**Why Sortino instead of just Sharpe:**

Sharpe ratio penalizes upside volatility equally with downside volatility. For trading strategies where upside volatility is desirable (momentum strategies with right-skewed returns), this is misleading. Sortino penalizes only downside deviation (returns below a minimum acceptable return, typically 0%). A strategy with Sharpe = 1.2 and Sortino = 2.0 has mostly upside volatility — it's actually better than its Sharpe implies. A strategy with Sharpe = 1.2 and Sortino = 0.9 has heavy downside volatility — it's worse than its Sharpe implies.

**Why CVaR-95 (not just max drawdown):**

Max drawdown is a single historical realization — the worst thing that actually happened. CVaR-95 (Conditional Value at Risk, 95th percentile) is the *expected loss* in the worst 5% of days. CVaR is:
- **Sub-additive:** Portfolio CVaR ≤ sum of position CVaRs (supports diversification math).
- **Forward-looking:** Estimated from the return distribution, not just one historical event.
- **Coherent:** Satisfies all four axioms of coherent risk measures (Artzner et al. 1999).

**Why capacity estimate:**

A strategy that works for $100K might not work for $10M. If average daily volume of tickers in the strategy is $5M and we need to execute $3M, our orders move the market. The capacity estimate flags strategies that can't scale without significant market impact, which is critical information for allocators.

---

#### `PortfolioOptimizer` — Three-Layer Portfolio Construction

**Purpose:** Combine individual strategy signals into a coherent, risk-managed portfolio allocation.

**Why three layers instead of mean-variance optimization (MVO):**

Classic Markowitz MVO is theoretically optimal but practically unreliable at the portfolio sizes and data lengths involved here. MVO is exquisitely sensitive to input estimation error — small errors in expected return forecasts produce wildly different weight allocations. DeMiguel et al. (2007) showed that naive equal-weight portfolios outperform MVO out-of-sample in most empirical settings due to estimation error.

Our three-layer approach is explicitly designed to extract the robust signal from MVO while discarding the estimation-error-sensitive parts:

**Layer 1 — CS-Momentum Rank Filter (keep top 60%):**
12-minus-1 month momentum is the most robust cross-sectional predictor in the academic literature (Fama/French 1996, Asness 1994). Rather than trying to estimate precise expected returns, we simply rank by momentum and discard the bottom 40%. This is a non-parametric filter that doesn't require estimating a covariance matrix.

**Layer 2 — Volatility Parity Sizing:**
```
raw_weight_i = target_vol / realized_vol_i
normalized_weight_i = raw_weight_i / sum(raw_weight_j)
```
Each position is sized inversely proportional to its realized volatility, so every position contributes approximately equal annualized volatility to the portfolio. This is equivalent to the "risk parity" approach used by Bridgewater, AQR, and other institutional managers — it's robust to the key failure mode of MVO (high-vol names dominating portfolio risk).

**Why `_TARGET_PORTFOLIO_VOL = 0.15` (15% annualized):**
15% annualized portfolio volatility is roughly consistent with the long-run volatility of a diversified equity portfolio (S&P 500 historical vol is ~15%). This is a reasonable target for an actively managed book that takes on concentrated positions — it allows meaningful returns without excessive drawdown risk.

**Layer 3 — Correlation Shrinkage:**
```python
if corr(i, j) > 0.70:
    scale_factor = sqrt(0.70 / corr(i, j))
    weight_i *= scale_factor
    weight_j *= scale_factor
```
When two positions are highly correlated, they provide less diversification benefit than their individual weights suggest. The shrinkage formula reduces both weights multiplicatively — the higher the correlation, the more aggressive the reduction. We use `sqrt(threshold/corr)` rather than a hard cutoff because: (1) it's continuous (no cliff-edge at 0.70), (2) it partially accounts for diversification benefit even above the threshold, (3) it's interpretable.

The 0.70 threshold is chosen because below this level, two stocks typically occupy different sectors or have meaningfully different factor exposures. Above 0.70, they often represent the same sector bet placed twice.

**Why `_MAX_POSITION_PCT = 0.30` (hard cap):**
Even if a single ticker ranks #1 by all metrics, concentration above 30% creates unacceptable idiosyncratic risk. A single earnings miss, regulatory action, or sector event can cause a 30–50% single-stock move. Capping at 30% limits the worst-case impact of any one position to ~15% of portfolio value.

---

### 4.5 Orchestration & Reporting

#### `PipelineOrchestrator` — Coordinator

**Purpose:** Coordinates all modules in the correct order, manages data flow between stages, handles errors gracefully, and writes outputs.

**Why a single orchestrator class (not microservices or DAGs):**

The pipeline is sequential within a run, deterministic, and runs in under 30 minutes on a single machine. The overhead of a DAG scheduler (Airflow, Prefect) or microservice architecture would dwarf the value — we'd be managing infrastructure, not building alpha. A single `PipelineOrchestrator.run()` method with clear stage logging is sufficient and dramatically easier to debug.

**Why stdout capture via `_QueueWriter`:**

The FastAPI server needs to stream pipeline logs to the browser in real-time via Server-Sent Events (SSE). Rather than instrumenting every `print()` call, we replace `sys.stdout` with a `_QueueWriter` that writes each line to a thread-safe queue. The SSE endpoint reads from this queue and streams to the client. This is a minimal, non-invasive approach that doesn't require a logging framework change.

---

#### `api_server.py` — FastAPI Web Interface

**Purpose:** Expose the pipeline as an HTTP API with real-time log streaming, enabling browser-based interaction.

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/run` | POST | Start pipeline run (async, returns immediately) |
| `/api/logs` | GET | SSE stream of live pipeline output |
| `/api/status` | GET | Current pipeline status (idle/running/done/error) |
| `/api/report` | GET | Fetch generated SummaryReport.md |
| `/api/summary` | GET | Fetch TraderSummary.md |

**Why SSE instead of WebSockets:**

SSE is unidirectional (server → client) and sufficient for log streaming. WebSockets add bidirectional complexity that isn't needed. SSE also reconnects automatically on dropped connections, which is useful for long pipeline runs.

---

## 5. Design Decisions & Rationale

This section consolidates the key architectural choices and their justifications in one place for quick reference.

| Decision | Alternative Considered | Why We Chose This |
|----------|----------------------|-------------------|
| Local LLM (Ollama/Qwen3) | OpenAI GPT-4 API | No API cost per run; no data leaves the machine; deterministic outputs for reproducible research; Qwen3:14b performs at GPT-4 level for structured financial reasoning |
| ChromaDB embedded | Pinecone / Weaviate | Zero infrastructure overhead; no API calls; local persistence; sufficient for article-scale data |
| Block bootstrap MC | IID bootstrap | Preserves serial autocorrelation of trade outcomes; avoids underestimating drawdown risk |
| Walk-forward ML | k-fold CV | k-fold mixes future data into training; walk-forward is the only valid evaluation methodology for time series |
| Hurst exponent for regime | ADX, RSI-based regime, HMM | Hurst is a structural process property, not a reactive lagging indicator; captures persistence before a trend is obvious |
| 3-layer portfolio construction | Markowitz MVO | MVO is notoriously unstable out-of-sample; our approach extracts the robust components (momentum ranking, vol parity) without estimation-error-sensitive covariance inversion |
| CS z-scoring of alpha | Time-series z-score | Removes market-wide level effects; makes signals comparable across tickers on the same date |
| Shift(1) universal lag | No lag | Absolute requirement for honest backtesting; prevents look-ahead bias that inflates simulated returns |
| 4 independent ML models | Single best model | Diversity reduces overfitting; ensemble average has lower variance than any single model |
| Per-date JSON news cache | Re-fetch on each run | Reproducibility; cost control; enables offline development |
| Composite news scoring | Sentiment analysis | Relevance/urgency scoring is more actionable than valence; avoids media framing artifacts |

---

## 6. Parameters, Thresholds & Metrics

### AlphaEngine

| Parameter | Value | Justification |
|-----------|-------|--------------|
| `_CS_MR_WINDOW` | 5 days | Short-term reversion operates at weekly frequency; longer windows capture too much momentum |
| `_BETA_WINDOW` | 60 days | ~3-month rolling beta is standard in factor model estimation |
| `_VOL_Z_WINDOW` | 20 days | 1-month volume baseline captures stable average trading activity |
| `_VOL_CLIP` | 3.0σ | Beyond 3σ, volume spikes are data artifacts or circuit-breaker events; clip to avoid signal distortion |
| CS-MR weight (normal) | 0.40 | Dominant signal in normal markets; highest IC in academic literature for short-horizon cross-sectional strategies |
| Vol-spike weight (bearish) | 0.45 | Volume precedes price in risk-off environments; dominates in crisis/bearish regimes |
| PCA trigger (corr > 0.70) | 0.70 | Below this, components are sufficiently independent; above, orthogonalization provides meaningful benefit |

### MLSignalEngine

| Parameter | Value | Justification |
|-----------|-------|--------------|
| `_MIN_TRAIN` | 252 bars | ~1 year to see full market cycle |
| `_REFIT` | 63 bars | Quarterly to balance recency vs. sample size |
| `_FORWARD` | 5 days | Medium-frequency sweet spot (avoids micro-noise < 1d, avoids fundamental drift > 30d) |
| `_N_EST` | 200 trees | Sufficient for variance reduction; more trees add diminishing returns |
| `_DEPTH` | 4 | Shallow enough to avoid noise memorization; deep enough to capture interactions |
| `_MIN_LEAF` | 20 | No leaf fit on fewer than 20 samples; prevents event-specific overfit |
| `_LR` | 0.05 | Slow learning rate requires more trees but produces more regularized ensemble |
| `_SUBSAMPLE` | 0.8 | Row subsampling adds stochastic regularization (Friedman 2002) |

### RegimeClassifier

| Threshold | Value | Justification |
|-----------|-------|--------------|
| `HURST_TRENDING` | 0.55 | Above 0.5 = persistence; 0.55 ensures statistically significant departure from random walk |
| `HURST_MEAN_REVERTING` | 0.45 | Symmetric: 0.45 = statistically significant anti-persistence |
| `ATR_CRISIS` | 6% | 6% ATR/price corresponds to ~2σ move per day; rare enough to be genuinely exceptional |
| `ATR_HIGH_VOL` | 3% | ~1σ per day; elevated but not crisis |
| `ATR_LOW_VOL` | 1.5% | Sub-1σ; volatility compression that precedes breakouts |
| `HURST_WINDOW` | 756 bars | 3 years for stable R/S estimation |

### Backtester

| Parameter | Value | Justification |
|-----------|-------|--------------|
| `_SLIPPAGE_PCT` | 0.05% per side | Conservative estimate for liquid US equities |
| `_COMMISSION_PCT` | 0.10% per side | Retail broker estimate; DMA would be lower |
| `_MAX_HOLDING_DAYS` | 20 days | Signal decay horizon for all included strategies |

### MonteCarloEngine

| Parameter | Value | Justification |
|-----------|-------|--------------|
| `n_simulations` | 10,000 | Stable tail estimates; ruin probability SE < 0.5% |
| `ruin_threshold` | 40% | Practical forced-liquidation threshold |
| `RISK_FREE_RATE` | 4.5% | Current approximate US T-bill rate |
| `block_size` | avg holding days | Matches simulation autocorrelation to empirical trade autocorrelation |

### PortfolioOptimizer

| Parameter | Value | Justification |
|-----------|-------|--------------|
| `_CS_MOM_TOP_FRACTION` | 0.60 | Keep top 60%; excludes the most negative-momentum names without over-concentrating |
| `_TARGET_PORTFOLIO_VOL` | 15% annual | Benchmark-consistent target; limits drawdown risk |
| `_MAX_POSITION_PCT` | 30% | Hard idiosyncratic risk cap |
| `_MIN_POSITION_PCT` | 2% | Ignore de-minimis positions that don't move the needle |
| `_CORR_THRESHOLD` | 0.70 | Industry standard for "highly correlated" in portfolio construction |
| `_RF_ANNUAL` | 4.5% | Consistent with Monte Carlo and DiagnosticsEngine |

---

## 7. External Dependencies

| Service / Library | Version | Purpose | Substitution |
|-------------------|---------|---------|--------------|
| **Benzinga API** | REST v2 | News ingestion (stock, global, industry) | Yahoo Finance news, Alpha Vantage News |
| **yfinance** | ≥ 0.2.36 | OHLCV data, earnings calendar, live bid/ask | Polygon.io, IEX Cloud |
| **Ollama + Qwen3:14b** | ≥ 0.2.0 | Local LLM inference for summarization & macro screening | OpenAI API, Anthropic API |
| **ChromaDB** | ≥ 0.5.0 | Persistent vector store (SentenceTransformer embeddings) | FAISS, Pinecone |
| **scikit-learn** | ≥ 1.4 | GBM, LR, RF, SGD, calibration | XGBoost, LightGBM |
| **statsmodels** | ≥ 0.14 | ADF cointegration test | Custom numpy approximation (fallback included) |
| **FastAPI + Uvicorn** | ≥ 0.111 | REST API + SSE | Flask, Starlette |
| **TradeZero API** | Internal | Live order submission (optional) | IBKR API, Alpaca |
| **numpy / pandas / scipy** | Latest stable | Core numerics and time-series | — |

---

## 8. Testing Architecture

Tests are organized in the `/test` directory with one test file per module. The testing philosophy follows the principle that financial code should be tested against known mathematical outcomes, not just "does it run without error."

| Test File | What It Validates |
|-----------|------------------|
| `test_backtester.py` | Trade log structure, P&L arithmetic, slippage application, no look-ahead |
| `test_monte_carlo_engine.py` | Equity distribution percentile ordering (p5 ≤ p50 ≤ p95), Kelly bounds, ruin classification |
| `test_regime_classifier.py` | Hurst calculation correctness on synthetic AR(1) series; regime label consistency |
| `test_strategy_selector.py` | All (regime, market_bias) combinations route to valid strategy; position size math |
| `test_pipeline_orchestrator.py` | End-to-end integration with mock data and LLM stub |
| `test_rag_store.py` | Insert/retrieve semantic similarity; collection isolation |
| `test_diagnostics_engine.py` | Sharpe, Sortino, CVaR-95 arithmetic on known return series |
| `test_execution_advisor.py` | Slippage classification boundaries; ATR proxy fallback |
| `test_prediction_market_client.py` | Volume filter correctness; category filter logic |

**Key testing principles:**
- Regime classifier tests use synthetic AR(1) series with known theoretical Hurst exponents to verify implementation correctness.
- Monte Carlo tests verify distributional properties (p5 ≤ p50 ≤ p95, Kelly ∈ [0, 1]) rather than specific values.
- No test uses live API calls — all external data is mocked via `conftest.py` fixtures.

---

## 9. Known Limitations & Future Work

### Current Limitations

**1. Prediction market data is cached, not live.**
The prediction market client reads from a local cache rather than fetching live prices. This means prediction market signals can be stale. A live integration with a prediction market API (Manifold Markets, Kalshi, Polymarket) would make this a real-time signal.

**2. The LLM macro screening is qualitative.**
The MacroScreener returns text-based sector classifications that are pattern-matched to the ticker universe. There's no formal semantic matching — a macro concern about "semiconductors" must be manually associated with the relevant tickers. A more rigorous implementation would use named entity recognition and sector taxonomy mapping.

**3. Single-machine, single-process architecture.**
The pipeline processes tickers sequentially. For universes > 100 tickers, this is slow. Parallel processing (multiprocessing, async IO) would reduce runtime substantially.

**4. No live order execution validation.**
The TradeZero integration exists but orders are not confirmed round-trip in testing. Production deployment requires a paper trading environment for end-to-end validation.

**5. Earnings data quality depends on yfinance.**
yfinance earnings calendars are scraped from Yahoo Finance and can be inaccurate or delayed. Professional-grade earnings data (Bloomberg, FactSet) would improve the Event-Driven regime classification reliability.

### Future Work

- **Intraday signal extension:** Extend to 15-minute bars using tick data for higher-frequency strategies.
- **Options flow integration:** Unusual options activity is a leading indicator for directional moves; integrating open interest data would improve the vol-spike component.
- **Live prediction market API:** Connect to Kalshi or Polymarket for real-time probability signals.
- **Transformer-based alpha:** Replace the GBM ML signal with a temporal attention model (e.g., Temporal Fusion Transformer) that can capture long-range dependencies in price series.
- **Formal factor attribution:** Decompose strategy returns into Fama-French 5-factor exposures to distinguish genuine alpha from beta mislabeling.
- **Multi-asset extension:** Apply the regime classifier and alpha engine to other asset classes (crypto, commodities, fixed income) using the same architectural framework.

---

*This document reflects the system as of version 1.0 (2026-04-03). For questions, contact the Chequita Engineering Team.*
