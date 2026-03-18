"""
MFT Alpha Finder — GDELT World & Industry News Pipeline
=========================================================
Fetches global news directly from the GDELT DOC 2.0 API, scores
each article by macro/geopolitical price-impact potential, and
returns a ranked Pandas DataFrame of the top N articles per day.

No API key required — GDELT is completely free and open.

Usage:
    pip install requests pandas python-dotenv

CONFIRMED GDELT API FACTS:
  - Endpoint  : https://api.gdeltproject.org/api/v2/doc/doc
  - Auth      : None required
  - Max/query : 250 articles (hard limit, no pagination)
  - History   : Last 3 months only (rolling window)
  - Date param: STARTDATETIME / ENDDATETIME → format YYYYMMDDHHMMSS
  - Response  : JSON → { "articles": [ {url, title, seendate,
                          domain, language, sourcecountry, ...} ] }
  - Default sort: relevance descending (best for MFT — do NOT override)
  - sourcelang:English goes INSIDE the query string, not as its own param
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIG — edit these values
# ─────────────────────────────────────────────
# 2-month window ending today (Mar 15 2026)
# GDELT supports last 3 months so this is within range
END_DATE       = "2026-03-15"   # YYYY-MM-DD  (today)
START_DATE     = "2026-01-15"   # YYYY-MM-DD  (2 months back)
TOP_N_PER_DAY  = 15             # articles to keep per day after scoring
REQUEST_DELAY  = 8              # 8s between calls — reduces 429 frequency (GDELT limit is 5s)

# ─────────────────────────────────────────────
# GDELT DOC 2.0 — confirmed endpoint
# ─────────────────────────────────────────────
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# ─────────────────────────────────────────────
# SEARCH QUERY
# Rules confirmed from GDELT docs:
#   - Phrases must be in "double quotes"
#   - OR must be uppercase and explicit
#   - sourcelang:English goes INSIDE the query string (not a URL param)
#   - Outer parentheses group the OR block correctly
# ─────────────────────────────────────────────
SEARCH_QUERY = (
    '("federal reserve" OR "interest rate" OR tariff OR sanction OR '
    'recession OR inflation OR commodity OR war OR election OR conflict) '
    'sourcelang:English'
)

# ─────────────────────────────────────────────
# IMPACT KEYWORD BUCKETS
# Checked against article titles (lowercased) for scoring.
# ─────────────────────────────────────────────
MACRO_KEYWORDS = [
    "federal reserve", "fed", "fomc", "interest rate", "rate hike",
    "rate cut", "inflation", "cpi", "ppi", "gdp", "recession",
    "central bank", "monetary policy", "quantitative easing",
]

GEO_KEYWORDS = [
    "war", "conflict", "sanction", "tariff", "trade war", "embargo",
    "geopolitical", "invasion", "coup", "protest", "election",
    "nato", "g7", "g20", "imf", "world bank",
]

COMMODITY_KEYWORDS = [
    "oil", "crude", "brent", "wti", "natural gas", "gold", "silver",
    "copper", "wheat", "corn", "supply chain", "energy", "commodity",
    "shortage", "opec",
]

MARKET_KEYWORDS = [
    "stock market", "stock exchange", "wall street", "s&p", "nasdaq",
    "dow jones", "bear market", "bull market", "market crash",
    "market rally", "volatility", "vix", "yield curve", "bond",
    "treasury", "dollar", "currency", "forex",
]

ALL_KEYWORDS = set(
    MACRO_KEYWORDS + GEO_KEYWORDS + COMMODITY_KEYWORDS + MARKET_KEYWORDS
)

CATALYST_PRIORITY = [
    ("Geopolitical",       set(GEO_KEYWORDS)),
    ("Macro/Central Bank", set(MACRO_KEYWORDS)),
    ("Commodity",          set(COMMODITY_KEYWORDS)),
    ("Market",             set(MARKET_KEYWORDS)),
]

# ─────────────────────────────────────────────
# TRUSTED SOURCE DOMAINS  (+2 domain_score bonus)
# ─────────────────────────────────────────────
TRUSTED_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "cnbc.com", "bbc.com", "bbc.co.uk", "apnews.com",
    "economist.com", "nytimes.com", "theguardian.com",
    "marketwatch.com", "financialtimes.com",
}


# ─────────────────────────────────────────────
# CATALYST TYPE CLASSIFIER
# ─────────────────────────────────────────────
def classify_catalyst(keyword_hits: list[str]) -> str:
    """
    Assigns a catalyst label based on which keyword buckets were hit.
      Mixed   = 2+ buckets hit
      General = no keyword matches
    """
    if not keyword_hits:
        return "General"

    hit_set     = set(keyword_hits)
    bucket_hits = {}

    for label, keywords in CATALYST_PRIORITY:
        count = sum(1 for kw in keywords if kw in hit_set)
        if count > 0:
            bucket_hits[label] = count

    if not bucket_hits:
        return "General"
    if len(bucket_hits) >= 2:
        return "Mixed"
    return list(bucket_hits.keys())[0]


# ─────────────────────────────────────────────
# STEP 1 — Fetch raw articles for a single day
# ─────────────────────────────────────────────
def fetch_gdelt_for_date(date_str: str, debug: bool = False) -> list[dict]:
    """
    Queries GDELT DOC 2.0 directly via HTTP for one calendar day.
    Returns list of article dicts. Max 250 (hard API limit, no pagination).

    Rate limit: 1 request per 5 seconds (enforced by GDELT — HTTP 429 if exceeded).
    This function includes automatic retry on 429 with backoff.

    Confirmed response fields per article:
        url, url_mobile, title, seendate, socialimage,
        domain, language, sourcecountry
    """
    start_dt = f"{date_str.replace('-', '')}000000"
    end_dt   = f"{date_str.replace('-', '')}235959"

    params = {
        "query":         SEARCH_QUERY,
        "mode":          "artlist",
        "maxrecords":    250,
        "startdatetime": start_dt,
        "enddatetime":   end_dt,
        "format":        "json",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_URL, params=params, timeout=30)

            if debug and attempt == 0:
                pass   # URL already printed above
                print(f"\n  [DEBUG] URL: {resp.url}")

            # 429 = rate limited — wait and retry
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)   # 10s, 20s, 30s backoff
                print(f"  [RATE LIMIT] 429 on {date_str} — waiting {wait}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"  [ERROR] HTTP {resp.status_code} for {date_str}: {resp.text[:200]}")
                return []

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                print(f"  [ERROR] Got HTML not JSON for {date_str} — query may be malformed")
                if debug:
                    print(f"          Response: {resp.text[:200]}")
                return []

            data = resp.json()

        except requests.exceptions.Timeout:
            print(f"  [ERROR] Timeout for {date_str}")
            return []
        except requests.exceptions.JSONDecodeError:
            print(f"  [ERROR] JSON parse failed for {date_str}: {resp.text[:200]}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] {date_str}: {e}")
            return []

        articles = data.get("articles") or []

        if not articles and debug:
            print(f"  [DEBUG] No articles. Response keys: {list(data.keys())}")
            print(f"  [DEBUG] Response preview: {str(data)[:400]}")

        return articles

    # All retries exhausted
    print(f"  [ERROR] All {max_retries} retries failed for {date_str} (persistent 429)")
    return []


# ─────────────────────────────────────────────
# STEP 2 — Score a single article
# ─────────────────────────────────────────────
def score_article(article: dict, date_str: str) -> dict:
    """
    Composite price-impact proxy score.

    Components:
      keyword_score  (×3)  — high-impact macro/geo phrases in title
      domain_score   (+2)  — trusted publisher bonus
      language_score (+1)  — English = most actionable for US markets
      recency_score  (0–3) — proximity to NYSE open (09:30 ET = 14:30 UTC)

    composite = keyword_score*3 + domain_score + language_score + recency_score
    """
    title    = (article.get("title")         or "").lower()
    url      = (article.get("url")           or "")
    domain   = (article.get("domain")        or "").lower().replace("www.", "")
    language = (article.get("language")      or "").lower()
    seendate = (article.get("seendate")      or "")
    country  = (article.get("sourcecountry") or "")

    # --- keyword_score + catalyst_type ---
    keyword_hits  = [kw for kw in ALL_KEYWORDS if kw in title]
    keyword_score = len(keyword_hits)
    catalyst_type = classify_catalyst(keyword_hits)

    # --- domain_score ---
    domain_score = 2 if domain in TRUSTED_DOMAINS else 0

    # --- language_score ---
    language_score = 1 if language == "english" else 0

    # --- recency_score (0–3) ---
    # Confirmed GDELT seendate format: "20260220T120000Z"
    recency_score = 0
    if seendate:
        try:
            clean       = seendate.replace("-", "").replace(":", "")
            pub_dt      = datetime.strptime(clean, "%Y%m%dT%H%M%SZ")
            market_open = datetime.strptime(f"{date_str} 14:30:00", "%Y-%m-%d %H:%M:%S")
            delta_mins  = abs((pub_dt - market_open).total_seconds() / 60)
            if delta_mins <= 30:
                recency_score = 3
            elif delta_mins <= 120:
                recency_score = 2
            elif delta_mins <= 360:
                recency_score = 1
        except Exception:
            pass

    composite_score = keyword_score * 3 + domain_score + language_score + recency_score

    return {
        "date":             date_str,
        "seendate":         seendate,
        "composite_score":  composite_score,
        "catalyst_type":    catalyst_type,
        "keyword_score":    keyword_score,
        "domain_score":     domain_score,
        "language_score":   language_score,
        "recency_score":    recency_score,
        "title":            article.get("title", ""),
        "keywords_matched": ", ".join(keyword_hits),
        "domain":           domain,
        "language":         language,
        "source_country":   country,
        "url":              url,
    }


# ─────────────────────────────────────────────
# STEP 3 — Run across a date range
# ─────────────────────────────────────────────
def run_pipeline(
    start_date: str,
    end_date:   str,
    top_n:      int  = TOP_N_PER_DAY,
    debug:      bool = False,
) -> pd.DataFrame:
    """
    Loops day by day, fetches + scores GDELT articles, keeps top_n
    per day, returns a unified ranked DataFrame.

    Args:
        start_date : 'YYYY-MM-DD' — must be within last 3 months of today
        end_date   : 'YYYY-MM-DD' (inclusive)
        top_n      : articles to keep per day (10–20 recommended)
        debug      : prints raw API URL + response details when True
    """
    start    = datetime.strptime(start_date, "%Y-%m-%d")
    end      = datetime.strptime(end_date,   "%Y-%m-%d")
    current  = start
    all_rows = []
    first_call = True      # always show DEBUG URL on the first request

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        # Sleep BEFORE the request — GDELT enforces 1 req/5s server-side
        if not first_call:
            time.sleep(REQUEST_DELAY)

        print(f"Processing {date_str} ...", end=" ", flush=True)

        articles   = fetch_gdelt_for_date(date_str, debug=first_call)
        first_call = False

        print(f"{len(articles)} articles fetched", end=" → ", flush=True)

        if articles:
            scored = [score_article(a, date_str) for a in articles]
            scored.sort(key=lambda x: x["composite_score"], reverse=True)
            top = scored[:top_n]
            all_rows.extend(top)
            print(f"top {len(top)} kept ✓")
        else:
            print("0 kept")

        current += timedelta(days=1)

    if not all_rows:
        print("\n[WARNING] No data collected. Possible causes:")
        print("  1. Date is outside the last 3 months")
        print("  2. Query syntax error — paste the [DEBUG] URL into your browser")
        print("  3. GDELT server temporarily unavailable")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    col_order = [
        "date", "seendate", "composite_score",
        "catalyst_type",
        "keyword_score", "domain_score", "language_score", "recency_score",
        "title", "keywords_matched",
        "domain", "language", "source_country", "url",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["date", "composite_score"], ascending=[True, False])
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  MFT Alpha Finder — GDELT World & Industry News Pipeline")
    print(f"  Date range  : {START_DATE}  →  {END_DATE}  (2 months)")
    print(f"  Top N/day   : {TOP_N_PER_DAY}")
    print(f"  API limit   : 250 articles/day max (GDELT hard limit)")
    print(f"  Sort        : GDELT native relevance (no sort override)")
    print(f"{'='*60}\n")

    df = run_pipeline(START_DATE, END_DATE, top_n=TOP_N_PER_DAY, debug=True)

    if not df.empty:
        print(f"\n{'='*60}")
        print(f"  Done. {len(df)} rows across {df['date'].nunique()} days.")
        print(f"{'='*60}\n")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",       200)
        pd.set_option("display.max_colwidth", 70)

        print(df[[
            "date", "composite_score", "catalyst_type",
            "title", "keywords_matched", "domain", "source_country"
        ]].to_string(index=True))

        out_path = f"mft_gdelt_news_{START_DATE}_to_{END_DATE}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[Saved] → {out_path}")

    # ── Use as a module ───────────────────────────────────────
    # from mft_gdelt_news import run_pipeline
    # df = run_pipeline("2026-01-15", "2026-03-15", top_n=15)
    #
    # ── Filter by catalyst type ───────────────────────────────
    # geo    = df[df["catalyst_type"] == "Geopolitical"]
    # macro  = df[df["catalyst_type"] == "Macro/Central Bank"]
    # commod = df[df["catalyst_type"] == "Commodity"]
    # mixed  = df[df["catalyst_type"] == "Mixed"]