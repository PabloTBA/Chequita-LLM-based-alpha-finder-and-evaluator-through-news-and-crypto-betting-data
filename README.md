# Chequita-LLM-based-alpha-finder-and-evaluator-through-news-and-crypto-betting-data

**Chequita** (based on Chiquita from the anime *Jormungand* and the word "cheque")  
is a **Hybrid LLM-RAG Framework** for **News and Crypto Betting Data-Driven Alpha Mining** and **Rigorous Strategy Evaluation** in **Medium-Frequency Trading (MFT)**.

The system ingests **3 months of stock + world news** as well as **3 months of crypto betting data** to discover high-conviction stocks for tactical investment.  
After identifying candidates, it assembles deep context, generates realistic strategies, runs constrained backtests, and applies strict robustness gates before approving any live/paper-trading idea.

![Chiquita](https://static.wikia.nocookie.net/jormungand/images/b/b2/S2_16_Chiquita.jpg/revision/latest/scale-to-width-down/250?cb=20121101181804)

---

## Fully Adapted Flow for 3-Month Data Constraint

**Data Horizon Rule** (fixed across flow)  
- News / narrative context: **full 3 months** (ideal for recency in MFT)  
- Price / volume / OHLCV: **full 3 months** (5-min or 15-min bars recommended)  
- No access to longer historical regimes → all robustness adapted to intra-window splits and synthetic tests

### Step-by-Step Adapted Flow

**Step 1: Data Ingestion & Preprocessing**  
- Ingest 3 months global + sector + stock-specific news (articles, headlines, transcripts if available)  
- Clean 3-month intraday OHLCV (adjusted for splits/dividends), volume, sector ETF comparison  
- Build vector store for RAG (embeddings + metadata: date, ticker, sentiment)  
- Pre-compute cheap features (daily/rolling vol, momentum, sector beta)

**Step 2: Hybrid Alpha Discovery**  
- Lightweight quant pre-filter on 3-month data: volume spikes, momentum breakouts, volatility regime change, sentiment delta  
- Rank → top 50–100 tickers  
- RAG + LLM ranks candidates using grounded news analysis (global macro + industry + ticker events)  
- Output: **5–20 high-conviction tactical candidates** with narrative thesis

**Step 3: Per-Stock Deep Context Assembly**  
- For each candidate: full 3-month price series + relevant news subset + peer/sector context  
- Generate chart images (candlestick + indicators) + feature summary text

**Step 4: Hybrid Strategy Generation**  
- Extract 25–40 technical/quant features (RSI, MACD, Bollinger, vol regime, volume profile, sector correlation, etc.)  
- LLM receives: features text + chart image + narrative context  
- Proposes **3–5 strategy templates** (breakout, mean-reversion post-news, momentum continuation, event fade) with narrow, realistic parameter ranges

**Step 5: Realistic Backtesting Engine**  
- Vectorized backtester with mandatory realism:  
  - Volume-based slippage (1–5 bp or % of ADV)  
  - Commissions  
  - Minimum trade size filter  
  - Partial fills (if modeled)  
- Run primary backtest on full 3 months (in-sample heavy by necessity)

**Step 6: Comprehensive Diagnostics & Risk Suite**  
- Core metrics (net-of-costs): cumulative equity curve, rolling vol, drawdown curve + max/avg DD  
- Full table: Sharpe, Sortino, Calmar, skew, kurtosis, VaR 95/99, CVaR, Omega  
- Monthly heatmap, worst periods, tail visuals (histogram + QQ plot)  
- Adapted additions:  
  - Monte Carlo (2,000–5,000 paths, block bootstrap)  
  - Bootstrapping for confidence intervals  
  - Synthetic stress: vol ×1.5–2×, remove best days, gap risk

**Step 7: Adapted Robustness & Validation Layer**  
- **Micro Walk-Forward Analysis** (4–8 rolling windows) → target Walk-Forward Efficiency (WFE) > 0.4–0.5  
- **Noise Injection** (0.1–0.5% microstructure noise) → performance drop < 40–50%  
- **Intra-Window Distributional Check**: Wasserstein distance between first 1.5 mo vs last 1.5 mo returns (low distance = stable)  
- Pass only if micro-WFE, noise survival, and Wasserstein are all acceptable

**Step 8: Agentic Optimization Loop**  
- LLM reviews full report  
- Suggests tight refinements  
- Bayesian optimizer (Optuna) runs 300–800 trials max  
- Re-run micro walk-forward + diagnostics → decide iterate or kill

**Step 9: Strict Approval Gates & Tactical Reporting**  
Hard filters (tuned for short-data realism):  
- ≥120–200 trades executed  
- In-sample Sharpe > 2.0–2.5  
- Micro walk-forward Sharpe > 1.0–1.5  
- Max DD < 8–12%  
- Positive/neutral skew preferred  
- Noise drop < 40–50%  
- Wasserstein low  
- Monte-Carlo 95%ile DD < 15–18%

**If passes** → generate tactical report:  
- Narrative thesis + strategy template  
- Parameter values + holding period projection  
- Entry/exit logic + position sizing  
- Regime warning (“single-regime only — monitor decay”)  
- Recommended: immediate paper trading + weekly re-run

**Step 10: Monitoring & Decay Trigger**  
- Re-run full pipeline weekly  
- Auto-kill if recent Sharpe < 50% of backtest or drawdown exceeds projection

---

## Final Briefing Notes / Caveats

### Strengths of 3-Month Version
- Extremely timely: captures current regime + fresh news alpha  
- Fast iteration cycle (days–weeks per idea)  
- Strong narrative grounding via RAG + LLM  
- Excellent tactical screening for event/sector plays

### Critical Limitations & Risks
- Single regime → high chance of alpha decay when regime shifts  
- Limited statistical power (few trades)  
- Elevated overfitting risk even with micro walk-forward  
- Robustness tests are proxies only — **no true multi-year stress coverage**

**Treat all outputs as hypotheses for paper/live testing, not proven edges.**

---



 
