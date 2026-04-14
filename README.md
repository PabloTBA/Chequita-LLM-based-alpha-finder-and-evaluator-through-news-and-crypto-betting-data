# Chequita-LLM-based-alpha-finder-and-evaluator-through-news-and-crypto-betting-data

**Chequita** (based on Chiquita from the anime *Jormungand* and the word "cheque")  
is a **Hybrid LLM-RAG Framework** for **News and Crypto Betting Data-Driven Alpha Mining** and **Rigorous Strategy Evaluation** in **Medium-Frequency Trading (MFT)**.

The system ingests **3 months of stock + world news** as well as **3 months of crypto betting data** to discover high-conviction stocks for tactical investment.  
After identifying candidates, it assembles deep context, generates realistic strategies, runs constrained backtests, and applies strict robustness gates before approving any live/paper-trading idea.

![Chiquita](https://static.wikia.nocookie.net/jormungand/images/b/b2/S2_16_Chiquita.jpg/revision/latest/scale-to-width-down/250?cb=20121101181804)

## Frontend Demo

A timelapse showcasing the user interface, interaction flow, and how insights are presented to the user:


https://github.com/user-attachments/assets/bd9e4f9e-10e3-4e50-92dd-334e5d431eef


## Backend Demo

A timelapse demonstrating the pipeline execution, including data ingestion, RAG processing, alpha generation, and evaluation workflow:


https://github.com/user-attachments/assets/9a26a048-4fde-417b-93d8-31a7476b2d22


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


# Prerequisites

Install the following before running the project:

1. Python 3.10+

Download from https://www.python.org/

Verify installation:

python --version
2. Node.js 18+

Download from https://nodejs.org/

Verify installation:

node --version
3. Ollama + Qwen3:14b Model

Install Ollama:

https://ollama.com

Pull the required model (~8GB):

ollama pull qwen3:14b
# Installation
Step 1 — Clone / Navigate to Project
cd "Chequita-LLM-based-alpha-finder-and-evaluator-through-news-and-crypto-betting-data"
Step 2 — Install Python Dependencies
pip install -r requirements.txt

Install PyTorch based on your hardware:

## CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

## NVIDIA GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
Step 3 — Install Frontend Dependencies
cd front-end
npm install
cd ..
Step 4 — Set API Key

Create a .env file in the project root:

BENZINGA_API=your_benzinga_api_key_here

The Benzinga API key is required for news ingestion (Stages 1–4 of the pipeline)

# Running the App

You need two terminals running simultaneously.

Terminal 1 — Backend (FastAPI)
cd "Chequita-LLM-based-alpha-finder-and-evaluator-through-news-and-crypto-betting-data"
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Expected output:

Uvicorn running on http://0.0.0.0:8000
Terminal 2 — Frontend (Vite/React)
cd "Chequita-LLM-based-alpha-finder-and-evaluator-through-news-and-crypto-betting-data\front-end"
npm run dev

Open the local URL shown (typically):

http://localhost:5173
## Using the Web App
Sidebar Sections
Section	Description
Generate	Run the full 13-stage pipeline
Analysis	View analysis outputs
Report Summary	Full pipeline report (Markdown)
Trader Summary	Plain-language trade recommendations
About / Creators	Information pages
## Running the Pipeline
Navigate to Generate
Click Execute_Generation
Sends: POST /api/run
Params: days=14, max_tickers=15
Watch live logs (all 13 stages stream in real time)
When Generation_Complete appears:
Go to Report Summary or Trader Summary
Click Run_Again to restart
## Runtime Expectations

## The pipeline is intentionally compute-heavy

Uses Qwen3:14b locally
Processes up to 15 tickers
Includes full backtesting

Estimated runtime:

10–30 minutes per run (hardware dependent)
## Troubleshooting
Error	Fix
Pipeline_Error: Setup failed -> Ollama isn't running — start it first with ollama serve
No report available yet	-> Pipeline hasn't completed yet, or it errored mid-run
409 A pipeline run is already in progress	->   Wait for the current run to finish
Frontend can't reach backend	-> Make sure the backend is on port 8000; Vite proxies /api to it
Benzinga errors ->	Check your .env has BENZINGA_API set correctly
If there is error when you run api_server.py just paste code below on the cmd:
rmdir /s /q node_modules
del package-lock.json
npm cache clean --force
npm install
