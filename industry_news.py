"""
MFT Alpha Finder — TheNewsAPI Industry News Pipeline (Free Tier)
================================================================
Uses /v1/news/all with search + date range — confirmed available
on ALL plans including free tier.

FREE TIER CONFIRMED LIMITS:
  - 100 requests/day
  - 3 articles per request (hard cap)
  - /v1/news/all available on all plans (confirmed from docs)
  - search parameter available on all plans
  - published_after / published_before available on all plans

STRATEGY:
  Since limit=3 on free, we run multiple targeted search queries
  per day covering different catalyst types to maximise coverage:
    - earnings query    → 3 articles
    - corporate query   → 3 articles
    - macro query       → 3 articles
    - energy query      → 3 articles
    - tech query        → 3 articles
    - general biz query → 3 articles
  Total: up to 18 unique articles/day using 6 requests/day.
  6 req/day × 61 days = 366 total — within 100 req/day free limit.

Setup:
    pip install requests pandas python-dotenv
    .env: THENEWS_API_KEY=your_key_here
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_KEY        = os.getenv("THENEWS_API_KEY", "YOUR_API_KEY_HERE")
START_DATE     = "2026-01-15"
END_DATE       = "2026-03-15"
TOP_N_PER_DAY  = 15
REQUEST_DELAY  = 2.0    # generous delay to avoid timeouts + rate limits

BASE_URL = "https://api.thenewsapi.com/v1/news/all"

# ─────────────────────────────────────────────
# TARGETED SEARCH QUERIES PER DAY
# Each query uses /v1/news/all with search param.
# Docs confirm: | = OR, + = AND, - = NOT, () = grouping
# limit=3 (free tier max), sort=relevance_score when search used
# ─────────────────────────────────────────────
DAILY_QUERIES = [
    {
        "label":      "earnings",
        "search":     "earnings | revenue | guidance | \"eps\" | profit",
        "categories": "business",
    },
    {
        "label":      "corporate",
        "search":     "merger | acquisition | \"ipo\" | bankruptcy | layoffs | \"price target\"",
        "categories": "business",
    },
    {
        "label":      "macro",
        "search":     "\"federal reserve\" | \"interest rate\" | inflation | tariff | recession | \"central bank\"",
        "categories": "business,politics",
    },
    {
        "label":      "energy_commodity",
        "search":     "\"oil price\" | \"crude oil\" | opec | \"natural gas\" | commodity | gold",
        "categories": "business",
    },
    {
        "label":      "tech",
        "search":     "semiconductor | \"artificial intelligence\" | cybersecurity | earnings | acquisition",
        "categories": "tech",
    },
    {
        "label":      "health_pharma",
        "search":     "\"fda approval\" | \"clinical trial\" | biotech | pharmaceutical | drug",
        "categories": "health,science",
    },
]

# ─────────────────────────────────────────────
# IMPACT KEYWORD BUCKETS (for scoring)
# ─────────────────────────────────────────────
EARNINGS_KEYWORDS = [
    "earnings", "eps", "revenue", "guidance", "beat", "miss",
    "profit", "loss", "quarterly", "results", "outlook", "forecast",
    "raised", "lowered", "reaffirm", "surprise",
]
CATALYST_KEYWORDS = [
    "fda", "approval", "merger", "acquisition", "takeover", "deal",
    "buyout", "spinoff", "ipo", "offering", "recall", "lawsuit",
    "investigation", "sec", "doj", "settlement", "bankruptcy",
    "restructuring", "layoffs", "ceo", "resignation",
    "downgrade", "upgrade", "price target", "analyst",
]
MACRO_KEYWORDS = [
    "fed", "fomc", "interest rate", "inflation", "cpi", "gdp",
    "recession", "tariff", "sanction", "jobs", "payroll",
    "unemployment", "rate hike", "rate cut", "central bank",
]
ALL_KEYWORDS = set(EARNINGS_KEYWORDS + CATALYST_KEYWORDS + MACRO_KEYWORDS)

CATALYST_PRIORITY = [
    ("Corporate Event", set(CATALYST_KEYWORDS)),
    ("Earnings",        set(EARNINGS_KEYWORDS)),
    ("Macro",           set(MACRO_KEYWORDS)),
]

TRUSTED_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "cnbc.com", "bbc.com", "bbc.co.uk", "apnews.com",
    "economist.com", "nytimes.com", "theguardian.com",
    "marketwatch.com", "barrons.com", "seekingalpha.com",
    "benzinga.com", "thestreet.com", "businessinsider.com",
    "forbes.com", "fortune.com",
}


# ─────────────────────────────────────────────
# CATALYST TYPE CLASSIFIER
# ─────────────────────────────────────────────
def classify_catalyst(keyword_hits, categories, title, query_label):
    t = title.lower()

    # Title-based detection (most precise)
    if any(w in t for w in ["fda", "approval", "clinical", "trial", "biotech", "pharma"]):
        return "Biotech/Pharma"
    if any(w in t for w in ["merger", "acquisition", "takeover", "buyout"]):
        return "Corporate Event"
    if any(w in t for w in ["earnings", "eps", "revenue", "quarterly", "guidance"]):
        return "Earnings"
    if any(w in t for w in ["ipo", "offering", "spac"]):
        return "Corporate Event"
    if any(w in t for w in ["fed", "fomc", "inflation", "cpi", "tariff", "rate"]):
        return "Macro"
    if any(w in t for w in ["oil", "crude", "opec", "gas", "commodity", "gold"]):
        return "Energy/Commodity"

    # Query label fallback (which search bucket fetched this)
    label_map = {
        "earnings":         "Earnings",
        "corporate":        "Corporate Event",
        "macro":            "Macro",
        "energy_commodity": "Energy/Commodity",
        "tech":             "Technology",
        "health_pharma":    "Biotech/Pharma",
    }
    if query_label in label_map:
        return label_map[query_label]

    # Category fallback
    cat_str = " ".join(categories).lower()
    if "tech" in cat_str:      return "Technology"
    if "health" in cat_str:    return "Healthcare/Science"
    if "science" in cat_str:   return "Healthcare/Science"
    if "politics" in cat_str:  return "Geopolitical"
    if "business" in cat_str:  return "Business"

    # Keyword fallback
    if not keyword_hits:
        return "General"
    hit_set = set(keyword_hits)
    bucket_hits = {}
    for label, keywords in CATALYST_PRIORITY:
        count = sum(1 for kw in keywords if kw in hit_set)
        if count > 0:
            bucket_hits[label] = count
    if not bucket_hits:       return "General"
    if len(bucket_hits) >= 2: return "Mixed"
    return list(bucket_hits.keys())[0]


# ─────────────────────────────────────────────
# STEP 1 — Fetch articles for a single day
# ─────────────────────────────────────────────
def fetch_thenews_for_date(date_str):
    """
    Runs each targeted query for the given date.
    Uses published_after/published_before for precise day scoping.
    sort=relevance_score used since search param is active.
    Deduplicates by uuid across all queries.
    """
    all_articles = []
    seen_uuids   = set()

    after  = f"{date_str}T00:00:00"
    before = f"{date_str}T23:59:59"

    for q in DAILY_QUERIES:
        params = {
            "api_token":        API_KEY,
            "search":           q["search"],
            "search_fields":    "title,description,keywords",
            "categories":       q["categories"],
            "language":         "en",
            "published_after":  after,
            "published_before": before,
            "sort":             "relevance_score",
            "limit":            3,               # free tier max
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)

            if resp.status_code == 401:
                print(f"\n  [ERROR] 401 — check THENEWS_API_KEY in .env")
                return all_articles
            if resp.status_code == 402:
                print(f"\n  [LIMIT] 402 — daily usage limit reached")
                return all_articles
            if resp.status_code == 403:
                print(f"\n  [ERROR] 403 — endpoint not available on your plan")
                return all_articles
            if resp.status_code == 429:
                print(f"\n  [RATE LIMIT] 429 — slow down, waiting 10s")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                time.sleep(REQUEST_DELAY)
                continue

            data     = resp.json()
            articles = data.get("data", [])

            for article in articles:
                uid = article.get("uuid")
                if uid and uid not in seen_uuids:
                    seen_uuids.add(uid)
                    article["_query_label"] = q["label"]  # tag for catalyst_type
                    all_articles.append(article)

        except requests.exceptions.Timeout:
            # Skip this query/day combination on timeout, don't abort whole day
            pass
        except Exception as e:
            print(f"\n  [ERROR] {q['label']}/{date_str}: {e}")

        time.sleep(REQUEST_DELAY)

    return all_articles


# ─────────────────────────────────────────────
# STEP 2 — Score a single article
# ─────────────────────────────────────────────
def score_article(article, date_str):
    title        = (article.get("title")       or "").lower()
    description  = (article.get("description") or "").lower()
    keywords_raw = (article.get("keywords")    or "").lower()
    text         = f"{title} {description} {keywords_raw}"

    source       = (article.get("source")       or "").lower()
    pub_at       = (article.get("published_at") or "")
    url          = (article.get("url")          or "")
    categories   = article.get("categories",     []) or []
    raw_rel      = article.get("relevance_score")
    query_label  = article.get("_query_label",   "")

    # Normalise API relevance score to 0–10
    relevance_norm = round(min(float(raw_rel), 200) / 20, 2) if raw_rel else 0.0

    keyword_hits  = [kw for kw in ALL_KEYWORDS if kw in text]
    keyword_score = len(keyword_hits)
    source_score  = 2 if source in TRUSTED_DOMAINS else 0

    recency_score = 0
    if pub_at:
        try:
            clean    = pub_at.replace("Z", "+00:00")
            pub_dt   = datetime.fromisoformat(clean).replace(tzinfo=None)
            mkt_open = datetime.strptime(f"{date_str} 14:30:00", "%Y-%m-%d %H:%M:%S")
            delta    = abs((pub_dt - mkt_open).total_seconds() / 60)
            if delta <= 30:    recency_score = 3
            elif delta <= 120: recency_score = 2
            elif delta <= 360: recency_score = 1
        except Exception:
            pass

    catalyst_type   = classify_catalyst(keyword_hits, categories, title, query_label)
    composite_score = round(
        relevance_norm * 2 +
        keyword_score  * 3 +
        source_score       +
        recency_score,
        2
    )

    return {
        "date":             date_str,
        "published_at":     pub_at,
        "composite_score":  composite_score,
        "catalyst_type":    catalyst_type,
        "api_relevance":    raw_rel,
        "relevance_norm":   relevance_norm,
        "keyword_score":    keyword_score,
        "source_score":     source_score,
        "recency_score":    recency_score,
        "title":            article.get("title", ""),
        "description":      article.get("description", ""),
        "snippet":          article.get("snippet", ""),
        "categories":       ", ".join(categories),
        "keywords_matched": ", ".join(keyword_hits),
        "query_bucket":     query_label,
        "source":           source,
        "url":              url,
    }


# ─────────────────────────────────────────────
# STEP 3 — Run across a date range
# ─────────────────────────────────────────────
def run_pipeline(start_date, end_date, top_n=TOP_N_PER_DAY):
    start    = datetime.strptime(start_date, "%Y-%m-%d")
    end      = datetime.strptime(end_date,   "%Y-%m-%d")
    current  = start
    all_rows = []

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Processing {date_str} ...", end=" ", flush=True)

        articles = fetch_thenews_for_date(date_str)
        print(f"{len(articles)} articles fetched", end=" -> ", flush=True)

        if articles:
            scored = [score_article(a, date_str) for a in articles]
            scored.sort(key=lambda x: x["composite_score"], reverse=True)
            top = scored[:top_n]
            all_rows.extend(top)
            print(f"top {len(top)} kept v")
        else:
            print("0 kept")

        current += timedelta(days=1)

    if not all_rows:
        print("\n[WARNING] No data returned. Check your API key.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    col_order = [
        "date", "published_at", "composite_score", "catalyst_type",
        "api_relevance", "relevance_norm",
        "keyword_score", "source_score", "recency_score",
        "title", "description", "snippet",
        "categories", "query_bucket", "keywords_matched",
        "source", "url",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["date", "composite_score"], ascending=[True, False])
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*64}")
    print(f"  MFT Alpha Finder — TheNewsAPI Industry News Pipeline")
    print(f"  Date range  : {START_DATE}  ->  {END_DATE}  (~2 months)")
    print(f"  Top N/day   : {TOP_N_PER_DAY}")
    print(f"  Endpoint    : /v1/news/all  (all plans confirmed)")
    print(f"  Strategy    : 6 targeted queries x 3 articles = 18/day max")
    print(f"  Requests    : ~6/day  (within 100/day free limit)")
    print(f"  Timeout     : 30s per request with graceful skip on failure")
    print(f"{'='*64}\n")

    df = run_pipeline(START_DATE, END_DATE, top_n=TOP_N_PER_DAY)

    if not df.empty:
        print(f"\n{'='*64}")
        print(f"  Done. {len(df)} rows across {df['date'].nunique()} days.")
        print(f"{'='*64}\n")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",       220)
        pd.set_option("display.max_colwidth", 65)

        print(df[[
            "date", "composite_score", "catalyst_type",
            "api_relevance", "title", "query_bucket", "source"
        ]].to_string(index=True))

        out_path = f"mft_thenews_{START_DATE}_to_{END_DATE}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[Saved] -> {out_path}")

    # ── Use as module ─────────────────────────────────────────
    # from mft_thenews import run_pipeline
    # df = run_pipeline("2026-01-15", "2026-03-15", top_n=15)
    #
    # ── Filter by catalyst type ───────────────────────────────
    # earnings = df[df["catalyst_type"] == "Earnings"]
    # macro    = df[df["catalyst_type"] == "Macro"]
    # corp     = df[df["catalyst_type"] == "Corporate Event"]
    # energy   = df[df["catalyst_type"] == "Energy/Commodity"]
    # biotech  = df[df["catalyst_type"] == "Biotech/Pharma"]
    # tech     = df[df["catalyst_type"] == "Technology"]
    #
    # ── Filter by query bucket (which search found it) ────────
    # macro_bucket  = df[df["query_bucket"] == "macro"]
    # energy_bucket = df[df["query_bucket"] == "energy_commodity"]