"""
MFT Alpha Finder — News Relevance Pipeline
============================================
Fetches daily stock news from Massive.com, scores each article by
price-impact potential, and returns a ranked Pandas DataFrame of
the top N articles per day across a date range.

Usage:
    pip install requests pandas python-dotenv

    Place your .env file in the same directory as this script:
      MASSIVE_API_KEY=your_key_here

    Or set it directly in the CONFIG block below.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env file from the same directory as this script
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG — edit these values
# ─────────────────────────────────────────────
API_KEY        = os.getenv("MASSIVE_API_KEY", "YOUR_API_KEY_HERE")
START_DATE     = "2025-01-06"   # YYYY-MM-DD
END_DATE       = "2025-01-10"   # YYYY-MM-DD (inclusive)
TOP_N_PER_DAY  = 15             # articles to keep per day (10–20 recommended)
REQUEST_DELAY  = 13             # free tier = 5 calls/min → 1 call per 13s to stay safe

BASE_URL = "https://api.massive.com/v2/reference/news"

# ─────────────────────────────────────────────
# HIGH-IMPACT KEYWORD LISTS
# Each word scores +1 to the keyword_score.
# Adjust/extend to fit your trading thesis.
# ─────────────────────────────────────────────
EARNINGS_KEYWORDS = [
    "earnings", "eps", "revenue", "guidance", "beat", "miss",
    "profit", "loss", "quarter", "results", "outlook", "forecast",
]

CATALYST_KEYWORDS = [
    "fda", "approval", "merger", "acquisition", "takeover", "deal",
    "buyout", "spinoff", "ipo", "offering", "recall", "lawsuit",
    "investigation", "sec", "doj", "settlement", "bankruptcy",
    "restructuring", "layoffs", "ceo", "resignation",
]

MACRO_KEYWORDS = [
    "fed", "fomc", "rate", "inflation", "cpi", "ppi", "gdp",
    "jobs", "payroll", "unemployment", "recession", "tariff",
    "sanction", "war", "geopolitical",
]

ALL_IMPACT_KEYWORDS = set(EARNINGS_KEYWORDS + CATALYST_KEYWORDS + MACRO_KEYWORDS)

# ─────────────────────────────────────────────
# CATALYST TYPE CLASSIFICATION
# Priority order matters — if an article hits
# multiple buckets, the highest-priority type wins.
# Mixed = meaningful hits in 2+ buckets.
# ─────────────────────────────────────────────
CATALYST_PRIORITY = [
    ("Corporate Event",  set(CATALYST_KEYWORDS)),   # M&A, FDA, IPO, CEO change etc.
    ("Earnings",         set(EARNINGS_KEYWORDS)),   # beat/miss, guidance, EPS etc.
    ("Macro",            set(MACRO_KEYWORDS)),      # Fed, CPI, tariff, jobs etc.
]

def classify_catalyst(keyword_hits: list[str]) -> str:
    """
    Returns a catalyst type label based on which keyword buckets were hit.

    Logic:
      - Count hits per bucket
      - If 2+ buckets have hits → "Mixed"
      - Otherwise → whichever single bucket wins (by priority order)
      - No hits → "General"
    """
    if not keyword_hits:
        return "General"

    hit_set     = set(keyword_hits)
    bucket_hits = {}

    for label, keywords in CATALYST_PRIORITY:
        count = len(hit_set & keywords)
        if count > 0:
            bucket_hits[label] = count

    if len(bucket_hits) == 0:
        return "General"
    elif len(bucket_hits) >= 2:
        return "Mixed"
    else:
        return list(bucket_hits.keys())[0]

# ─────────────────────────────────────────────
# TRUSTED PUBLISHER WHITELIST
# Publishers here get a +2 publisher_score bonus.
# ─────────────────────────────────────────────
TRUSTED_PUBLISHERS = {
    "Reuters", "Bloomberg", "The Wall Street Journal", "CNBC",
    "Financial Times", "MarketWatch", "Barron's", "Seeking Alpha",
    "Investor's Business Daily", "The Associated Press",
}

# ─────────────────────────────────────────────
# STEP 1 — Fetch news for a single day
# ─────────────────────────────────────────────
def fetch_news_for_date(date_str: str) -> list[dict]:
    """
    Pulls all available news articles for a given date (YYYY-MM-DD).
    Handles pagination via next_url cursor.
    """
    # Define the day window
    date_start = f"{date_str}T00:00:00Z"
    date_end   = f"{date_str}T23:59:59Z"

    params = {
        "published_utc.gte": date_start,
        "published_utc.lte": date_end,
        "sort":              "published_utc",
        "order":             "desc",
        "limit":             1000,          # max per page
        "apiKey":            API_KEY,
    }

    articles = []
    url = BASE_URL

    while url:
        try:
            resp = resp = requests.get(url, params=params if url == BASE_URL else None, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [ERROR] {date_str}: {e}")
            break

        results = data.get("results", [])
        articles.extend(results)

        # Follow pagination cursor if present
        next_url = data.get("next_url")
        if next_url:
            # Append API key to cursor URL (required by Massive)
            url = next_url + f"&apiKey={API_KEY}"
            params = None          # params are now embedded in the cursor URL
            time.sleep(REQUEST_DELAY)
        else:
            break

    return articles


# ─────────────────────────────────────────────
# STEP 2 — Score a single article
# ─────────────────────────────────────────────
def score_article(article: dict, date_str: str) -> dict:
    """
    Computes a composite price-impact proxy score for one article.

    Score components (each 0–N, then weighted sum):
      keyword_score   — count of high-impact words in title + description
      ticker_score    — number of tickers mentioned (multi-ticker = wider impact)
      publisher_score — +2 if from a trusted publisher, else 0
      recency_score   — articles closer to market open (09:30 ET) score higher

    Final composite_score = keyword_score * 3
                          + ticker_score  * 2
                          + publisher_score
                          + recency_score
    """
    title       = (article.get("title")       or "").lower()
    description = (article.get("description") or "").lower()
    text        = f"{title} {description}"

    tickers     = article.get("tickers", []) or []
    publisher   = (article.get("publisher", {}) or {}).get("name", "")
    pub_utc     = article.get("published_utc", "")

    # --- keyword_score & catalyst_type ---
    keyword_hits   = [kw for kw in ALL_IMPACT_KEYWORDS if kw in text]
    keyword_score  = len(keyword_hits)
    catalyst_type  = classify_catalyst(keyword_hits)

    # --- ticker_score ---
    ticker_score = min(len(tickers), 10)   # cap at 10 to avoid runaway outliers

    # --- publisher_score ---
    publisher_score = 2 if publisher in TRUSTED_PUBLISHERS else 0

    # --- recency_score (0–3) ---
    # Market open is 09:30 ET = 14:30 UTC.
    # Articles published within 2 hours of open get max score.
    recency_score = 0
    if pub_utc:
        try:
            pub_dt      = datetime.fromisoformat(pub_utc.replace("Z", "+00:00"))
            market_open = datetime.fromisoformat(f"{date_str}T14:30:00+00:00")
            delta_mins  = abs((pub_dt - market_open).total_seconds() / 60)
            if delta_mins <= 30:
                recency_score = 3
            elif delta_mins <= 120:
                recency_score = 2
            elif delta_mins <= 360:
                recency_score = 1
        except Exception:
            pass

    # --- composite ---
    composite_score = (
        keyword_score  * 3 +
        ticker_score   * 2 +
        publisher_score    +
        recency_score
    )

    return {
        "date":             date_str,
        "published_utc":    pub_utc,
        "title":            article.get("title", ""),
        "publisher":        publisher,
        "tickers":          ", ".join(tickers) if tickers else "",
        "ticker_count":     len(tickers),
        "catalyst_type":    catalyst_type,
        "keywords_matched": ", ".join(keyword_hits),
        "keyword_score":    keyword_score,
        "ticker_score":     ticker_score,
        "publisher_score":  publisher_score,
        "recency_score":    recency_score,
        "composite_score":  composite_score,
        "url":              article.get("article_url", ""),
        "description":      article.get("description", ""),
    }


# ─────────────────────────────────────────────
# STEP 3 — Run across a date range
# ─────────────────────────────────────────────
def run_pipeline(
    start_date: str,
    end_date:   str,
    top_n:      int = TOP_N_PER_DAY,
) -> pd.DataFrame:
    """
    Main entry point. Iterates day by day between start_date and end_date,
    fetches + scores news, keeps top_n per day, and returns a single DataFrame.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    all_rows = []
    current  = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Processing {date_str} ...", end=" ")

        articles = fetch_news_for_date(date_str)
        print(f"{len(articles)} articles fetched", end=" → ")

        if articles:
            scored = [score_article(a, date_str) for a in articles]
            # Sort descending by composite score, keep top N
            scored.sort(key=lambda x: x["composite_score"], reverse=True)
            top    = scored[:top_n]
            all_rows.extend(top)
            print(f"top {len(top)} kept")
        else:
            print("no articles (weekend/holiday?)")

        current += timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        print("\n[WARNING] No data returned. Check your API key and date range.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Clean up column order for readability
    col_order = [
        "date", "published_utc", "composite_score",
        "catalyst_type",
        "keyword_score", "ticker_score", "publisher_score", "recency_score",
        "title", "tickers", "ticker_count", "keywords_matched",
        "publisher", "url", "description",
    ]
    df = df[col_order]
    df = df.sort_values(["date", "composite_score"], ascending=[True, False])
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  MFT Alpha Finder — News Relevance Pipeline")
    print(f"  Date range : {START_DATE}  →  {END_DATE}")
    print(f"  Top N/day  : {TOP_N_PER_DAY}")
    print(f"{'='*55}\n")

    df = run_pipeline(START_DATE, END_DATE, top_n=TOP_N_PER_DAY)

    if not df.empty:
        print(f"\n{'='*55}")
        print(f"  Done. {len(df)} total rows across {df['date'].nunique()} days.")
        print(f"{'='*55}\n")

        # ── Preview ──────────────────────────────────
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",       180)
        pd.set_option("display.max_colwidth", 60)

        print(df[[
            "date", "composite_score", "catalyst_type",
            "title", "tickers", "keywords_matched", "publisher"
        ]].to_string(index=True))

        # ── Save to CSV ──────────────────────────────
        out_path = f"mft_alpha_news_{START_DATE}_to_{END_DATE}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[Saved] {out_path}")

        # ── The DataFrame is ready for downstream use ─
        # e.g.:
        #   from mft_alpha_news import run_pipeline
        #   df = run_pipeline("2025-01-06", "2025-01-10")
        #   high_impact = df[df["composite_score"] >= 10]