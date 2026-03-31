# Chequita-LLM-based-alpha-finder-and-evaluator-through-news-and-crypto-betting-data

**Chequita** (based on Chiquita from the anime *Jormungand* and the word "cheque")  
is a **Hybrid LLM-RAG Framework** for **News and Crypto Betting Data-Driven Alpha Mining** and **Rigorous Strategy Evaluation** in **Medium-Frequency Trading (MFT)**.

The system ingests **3 months of stock + world news** as well as **3 months of crypto betting data** to discover high-conviction stocks for tactical investment.  
After identifying candidates, it assembles deep context, generates realistic strategies, runs constrained backtests, and applies strict robustness gates before approving any live/paper-trading idea.

![Chiquita](https://static.wikia.nocookie.net/jormungand/images/b/b2/S2_16_Chiquita.jpg/revision/latest/scale-to-width-down/250?cb=20121101181804)

Frontend Demo

A timelapse showcasing the user interface, interaction flow, and how insights are presented to the user:

<video src="frontend_timelapse.mp4" controls width="700"></video>

Backend Demo

A timelapse demonstrating the pipeline execution, including data ingestion, RAG processing, alpha generation, and evaluation workflow:

<video src="backend_timelapse.mp4" controls width="700"></video>

# System Overview
Data Sources
Global financial and macro news (3 months)
Crypto betting / prediction market data (3 months)
Core Components
LLM + RAG retrieval system
Signal extraction engine
Strategy generator
Constrained backtesting module
Robustness & validation filters
Output
High-conviction stock ideas
Interpretable strategies
Evaluation metrics (Sharpe, drawdown, stability, etc.)
