"""
Stage1DataCollector
====================
Fetches stock news, global/macro news, industry news, and ticker-specific news
from the Benzinga API.  Supports per-date JSON caching and composite impact scoring.

Environment:
    BENZINGA_API — Benzinga API token (set in .env)

Cache layout:
    {cache_dir}/YYYY-MM-DD_stock_news.json
    {cache_dir}/YYYY-MM-DD_global_news.json
    {cache_dir}/YYYY-MM-DD_industry_news.json
    {cache_dir}/YYYY-MM-DD_ticker_news.json
"""

import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Benzinga API
# ──────────────────────────────────────────────────────────────
BENZINGA_BASE_URL = "https://api.benzinga.com/api/v2/news"
REQUEST_DELAY     = 1.0   # seconds between API calls
MAX_PAGE_SIZE     = 100   # articles per request
TICKER_BATCH_SIZE = 30    # tickers per Benzinga ticker-news request

# Channels used per source type
STOCK_CHANNELS    = "News"
GLOBAL_CHANNELS   = "Global,Economics,Markets"
INDUSTRY_CHANNELS = "Healthcare,Technology,Energy,Finance,Industrials,ConsumerGoods"

# S&P 500 watchlist — top ~150 by market cap
# Benzinga ticker-filtered news will only tag articles that mention these symbols
SP500_WATCHLIST = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK.B","AVGO","JPM",
    "LLY","UNH","V","XOM","MA","COST","HD","PG","ORCL","WMT","BAC","ABBV",
    "NFLX","AMD","KO","CRM","CVX","MRK","ADBE","PEP","TMO","MCD","CSCO",
    "ABT","ACN","LIN","DHR","TXN","WFC","PM","INTU","NEE","AMGN","IBM","RTX",
    "QCOM","SPGI","LOW","ISRG","GS","CAT","MS","BKNG","NOW","ELV","PFE",
    "AMAT","MDT","T","DE","GILD","AXP","SYK","VRTX","MDLZ","C","ADI","MU",
    "ETN","SCHW","BMY","CB","PANW","ZTS","BSX","REGN","MMC","LRCX","AON",
    "PLD","COP","SO","DUK","ICE","APH","MCO","SHW","ITW","NOC","HUM","TGT",
    "USB","PH","KLAC","CME","EMR","NSC","FCX","BDX","FI","ROP","MAR","HCA",
    "MO","TJX","EOG","AIG","PSA","GD","CL","EW","F","GM","UBER","ABNB",
    "CRWD","SNOW","COIN","PLTR","SOFI","RBLX","RIVN","LCID","NIO","BIDU",
    "JNJ","GE","HON","MMM","BA","DIS","SBUX","NKE","PYPL","SQ","SHOP",
]

# ──────────────────────────────────────────────────────────────
# Scoring: keyword lists
# ──────────────────────────────────────────────────────────────
EARNINGS_KEYWORDS = {
    "earnings", "eps", "revenue", "guidance", "beat", "miss",
    "profit", "loss", "quarter", "results", "outlook", "forecast",
}

CATALYST_KEYWORDS = {
    "fda", "approval", "merger", "acquisition", "takeover", "deal",
    "buyout", "spinoff", "ipo", "offering", "recall", "lawsuit",
    "investigation", "sec", "doj", "settlement", "bankruptcy",
    "restructuring", "layoffs", "ceo", "resignation",
}

MACRO_KEYWORDS = {
    "fed", "fomc", "rate", "inflation", "cpi", "ppi", "gdp",
    "jobs", "payroll", "unemployment", "recession", "tariff",
    "sanction", "war", "geopolitical",
}

ALL_IMPACT_KEYWORDS = EARNINGS_KEYWORDS | CATALYST_KEYWORDS | MACRO_KEYWORDS

TRUSTED_PUBLISHERS = {
    "Reuters", "Bloomberg", "The Wall Street Journal", "CNBC",
    "Financial Times", "MarketWatch", "Barron's", "Seeking Alpha",
    "Investor's Business Daily", "The Associated Press", "Benzinga",
}

CATALYST_PRIORITY = [
    ("Corporate Event", CATALYST_KEYWORDS),
    ("Earnings",        EARNINGS_KEYWORDS),
    ("Macro",           MACRO_KEYWORDS),
]


# ──────────────────────────────────────────────────────────────
# Pure scoring functions (no I/O)
# ──────────────────────────────────────────────────────────────

def _classify_catalyst(keyword_hits: list[str]) -> str:
    if not keyword_hits:
        return "General"
    hit_set     = set(keyword_hits)
    bucket_hits = {label: len(hit_set & kws) for label, kws in CATALYST_PRIORITY if hit_set & kws}
    if not bucket_hits:
        return "General"
    if len(bucket_hits) >= 2:
        return "Mixed"
    return list(bucket_hits.keys())[0]


def score_article(article: dict, date_str: str) -> dict:
    """
    Compute composite impact score for a Benzinga article dict.

    Composite = keyword_score × 3
              + ticker_score  × 2
              + publisher_score
              + recency_score

    Args:
        article:  Raw Benzinga article dict.
        date_str: 'YYYY-MM-DD' of the collection date.

    Returns:
        Flat scored dict ready for DataFrame construction.
    """
    title     = (article.get("title") or "").lower()
    pub_raw   = article.get("published") or article.get("created") or ""
    publisher = article.get("source") or article.get("author") or ""
    url       = article.get("url") or article.get("link") or ""

    # Benzinga returns tickers in "stocks": [{"name": "AAPL", "exchange": "NASDAQ"}, ...]
    # Fallback to legacy "tickers" field for backwards compatibility.
    raw_stocks = article.get("stocks") or article.get("tickers") or []
    if raw_stocks and isinstance(raw_stocks[0], dict):
        tickers = [t.get("name", "").lstrip("$") for t in raw_stocks]
    else:
        tickers = [str(t).lstrip("$") for t in raw_stocks]
    # Keep only clean equity-like symbols (1–5 uppercase letters, no crypto junk)
    tickers = [t for t in tickers if t and t.isalpha() and t.isupper() and 1 <= len(t) <= 5]

    # keyword_score
    keyword_hits  = [kw for kw in ALL_IMPACT_KEYWORDS if kw in title]
    keyword_score = len(keyword_hits)
    catalyst_type = _classify_catalyst(keyword_hits)

    # ticker_score (capped at 10)
    ticker_score = min(len(tickers), 10)

    # publisher_score
    publisher_score = 2 if publisher in TRUSTED_PUBLISHERS else 0

    # recency_score (0–3) — proximity to NYSE open 09:30 ET = 14:30 UTC
    recency_score = 0
    if pub_raw:
        try:
            clean  = pub_raw.replace("Z", "+00:00")
            pub_dt = datetime.fromisoformat(clean).replace(tzinfo=None)
            mkt_open = datetime.strptime(f"{date_str} 14:30:00", "%Y-%m-%d %H:%M:%S")
            delta_mins = abs((pub_dt - mkt_open).total_seconds() / 60)
            if delta_mins <= 30:
                recency_score = 3
            elif delta_mins <= 120:
                recency_score = 2
            elif delta_mins <= 360:
                recency_score = 1
        except Exception:
            pass

    composite_score = keyword_score * 3 + ticker_score * 2 + publisher_score + recency_score

    return {
        "date":             date_str,
        "title":            article.get("title", ""),
        "url":              url,
        "publisher":        publisher,
        "tickers":          ", ".join(tickers),
        "keywords_matched": ", ".join(keyword_hits),
        "catalyst_type":    catalyst_type,
        "keyword_score":    keyword_score,
        "ticker_score":     ticker_score,
        "publisher_score":  publisher_score,
        "recency_score":    recency_score,
        "composite_score":  composite_score,
    }


# ──────────────────────────────────────────────────────────────
# Stage1DataCollector
# ──────────────────────────────────────────────────────────────

class Stage1DataCollector:
    """
    Collects stock, global, and industry news from Benzinga.
    Caches each source per date as a JSON file to avoid redundant API calls.

    Args:
        api_key:   Benzinga API token.
        cache_dir: Directory for per-date JSON cache files.
        csv_dir:   Directory for CSV exports (created on demand).
    """

    SOURCE_CHANNELS = {
        "stock_news":    STOCK_CHANNELS,
        "global_news":   GLOBAL_CHANNELS,
        "industry_news": INDUSTRY_CHANNELS,
    }

    def __init__(self, api_key: str, cache_dir: str = "data/cache", csv_dir: str = "data"):
        self.api_key   = api_key
        self.cache_dir = cache_dir
        self.csv_dir   = csv_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ── Cache helpers ──────────────────────────────────────────

    def _cache_path(self, date_str: str, source: str) -> str:
        return os.path.join(self.cache_dir, f"{date_str}_{source}.json")

    def _load_cache(self, date_str: str, source: str) -> list[dict] | None:
        path = self._cache_path(date_str, source)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, date_str: str, source: str, rows: list[dict]) -> None:
        path = self._cache_path(date_str, source)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    # ── Benzinga fetch ────────────────────────────────────────

    def _fetch_benzinga(self, date_str: str, channels: str) -> list[dict] | None:
        """
        Fetch articles from Benzinga for a single date and channel set.

        Returns:
            list[dict] on success (may be empty if no articles that day).
            None on HTTP error — callers must not cache a None result.
        """
        params = {
            "token":         self.api_key,
            "displayOutput": "full",
            "dateFrom":      date_str,   # Benzinga expects YYYY-MM-DD only
            "dateTo":        date_str,
            "channels":      channels,
            "pageSize":      MAX_PAGE_SIZE,
            "page":          0,
        }

        articles = []
        seen_urls: set[str] = set()
        # Benzinga defaults to XML — must request JSON explicitly (per docs)
        headers  = {"accept": "application/json"}

        while True:
            try:
                resp = requests.get(BENZINGA_BASE_URL, params=params, headers=headers, timeout=30)
                if resp.status_code != 200:
                    print(f"  [WARN] Benzinga HTTP {resp.status_code} for {date_str} channels={channels}")
                    print(f"         Response: {resp.text[:400]}")
                    return None   # error — do NOT cache
                if not resp.text.strip():
                    print(f"  [WARN] Benzinga returned empty body for {date_str} channels={channels}")
                    print(f"         HTTP {resp.status_code} — check API key and plan tier")
                    return None
                page_articles = resp.json() or []
            except Exception as e:
                print(f"  [ERROR] Benzinga fetch failed ({date_str}, {channels}): {e}")
                print(f"         Raw body preview: {getattr(resp, 'text', 'N/A')[:400]}")
                return None       # error — do NOT cache

            if not page_articles:
                break

            for art in page_articles:
                url = art.get("url") or art.get("link") or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append(art)
                elif not url:
                    articles.append(art)

            if len(page_articles) < MAX_PAGE_SIZE:
                break  # last page

            params["page"] += 1
            time.sleep(REQUEST_DELAY)

        return articles

    def _fetch_benzinga_by_tickers(self, date_str: str, tickers: list[str]) -> list[dict] | None:
        """
        Fetch articles from Benzinga filtered by a list of ticker symbols.
        Uses the `tickers` query param instead of `channels`.
        Returns list[dict] on success, None on HTTP error.
        """
        params = {
            "token":         self.api_key,
            "displayOutput": "full",
            "dateFrom":      date_str,
            "dateTo":        date_str,
            "tickers":       ",".join(tickers),
            "pageSize":      MAX_PAGE_SIZE,
            "page":          0,
        }
        headers  = {"accept": "application/json"}
        articles = []
        seen_urls: set[str] = set()

        while True:
            try:
                resp = requests.get(BENZINGA_BASE_URL, params=params, headers=headers, timeout=30)
                if resp.status_code != 200:
                    print(f"  [WARN] Benzinga HTTP {resp.status_code} for {date_str} tickers-batch")
                    return None
                if not resp.text.strip():
                    return None
                page_articles = resp.json() or []
            except Exception as e:
                print(f"  [ERROR] Benzinga ticker-news fetch failed ({date_str}): {e}")
                return None

            if not page_articles:
                break

            for art in page_articles:
                url = art.get("url") or art.get("link") or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append(art)
                elif not url:
                    articles.append(art)

            if len(page_articles) < MAX_PAGE_SIZE:
                break

            params["page"] += 1
            time.sleep(REQUEST_DELAY)

        return articles

    def _collect_ticker_news(self, date_str: str, watchlist: list[str]) -> list[dict]:
        """
        Fetch and score ticker-tagged news for the given watchlist.
        Batches watchlist into groups of TICKER_BATCH_SIZE.
        Deduplicates by URL across batches.
        Caches result as ticker_news.
        """
        cached = self._load_cache(date_str, "ticker_news")
        if cached is not None:
            print(f"  [CACHE] {date_str} ticker_news: {len(cached)} rows")
            return cached

        all_articles: list[dict] = []
        seen_urls: set[str] = set()

        batches = [watchlist[i:i + TICKER_BATCH_SIZE] for i in range(0, len(watchlist), TICKER_BATCH_SIZE)]
        print(f"  [API]   {date_str} ticker_news: fetching {len(batches)} batches × {TICKER_BATCH_SIZE} tickers...")

        for batch in batches:
            raw = self._fetch_benzinga_by_tickers(date_str, batch)
            if raw is None:
                continue
            for art in raw:
                url = art.get("url") or art.get("link") or ""
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                all_articles.append(art)
            time.sleep(REQUEST_DELAY)

        rows = [score_article(art, date_str) for art in all_articles]
        print(f"  [API]   {date_str} ticker_news: {len(rows)} unique articles")

        if rows:
            self._save_cache(date_str, "ticker_news", rows)
        return rows

    def debug_raw(self, date: str) -> None:
        """
        Print the raw Benzinga API response for one date (stock_news only).
        Use this to inspect field names and verify your API key / channel names.

        Example:
            collector.debug_raw("2026-03-18")
        """
        params = {
            "token":         self.api_key,
            "displayOutput": "full",
            "dateFrom":      date,
            "dateTo":        date,
            "pageSize":      2,
            "page":          0,
        }
        headers = {"accept": "application/json"}
        print(f"\n[DEBUG] GET {BENZINGA_BASE_URL}")
        print(f"[DEBUG] params (token redacted): { {k: ('***' if k == 'token' else v) for k, v in params.items()} }")
        print(f"[DEBUG] accept: application/json")
        try:
            resp = requests.get(BENZINGA_BASE_URL, params=params, headers=headers, timeout=30)
            print(f"[DEBUG] HTTP {resp.status_code}")
            print(f"[DEBUG] Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
            print(f"[DEBUG] Body length: {len(resp.text)} chars")
            print(f"[DEBUG] Raw response:\n{resp.text[:2000]}")
        except Exception as e:
            print(f"[DEBUG] Request failed: {e}")

    # ── Per-source collect ─────────────────────────────────────

    def _collect_source(self, date_str: str, source: str) -> list[dict]:
        """
        Return scored rows for one source + date.
        Serves from cache if available; fetches and caches otherwise.
        Never caches a failed (None) fetch — will retry on next run.
        Logs whether data came from cache or API.
        """
        cached = self._load_cache(date_str, source)
        if cached is not None:
            print(f"  [CACHE] {date_str} {source}: {len(cached)} rows")
            return cached

        channels = self.SOURCE_CHANNELS[source]
        raw      = self._fetch_benzinga(date_str, channels)

        if raw is None:
            # API error — return empty but do NOT write cache so next run retries
            print(f"  [ERROR] {date_str} {source}: fetch failed, skipping cache")
            return []

        print(f"  [API]   {date_str} {source}: {len(raw)} articles fetched")

        seen_urls: set[str] = set()
        rows: list[dict] = []
        for art in raw:
            scored = score_article(art, date_str)
            url    = scored["url"]
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            rows.append(scored)

        self._save_cache(date_str, source, rows)
        return rows

    # ── CSV export ─────────────────────────────────────────────

    def _export_csv(self, results: dict[str, pd.DataFrame], label: str) -> None:
        """Write each non-empty DataFrame to {csv_dir}/{label}_{source}.csv."""
        os.makedirs(self.csv_dir, exist_ok=True)
        for source, df in results.items():
            if not df.empty:
                path = os.path.join(self.csv_dir, f"{label}_{source}.csv")
                df.to_csv(path, index=False)
                print(f"  [CSV]   Saved {len(df)} rows → {path}")

    # ── Public API ────────────────────────────────────────────

    def collect(self, date: str, save_csv: bool = False,
                watchlist: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """
        Collect all four news sources for a single date.

        Args:
            date:      'YYYY-MM-DD'
            save_csv:  If True, write each source to {csv_dir}/{date}_{source}.csv
            watchlist: Ticker list for ticker_news fetch (default: SP500_WATCHLIST)

        Returns:
            {'stock_news': df, 'global_news': df, 'industry_news': df, 'ticker_news': df}
            Each DataFrame is sorted by composite_score descending.
        """
        result = {}
        for source in self.SOURCE_CHANNELS:
            rows = self._collect_source(date, source)
            df   = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            result[source] = df

        # Ticker-specific news (Option B: Benzinga tickers= param)
        wl   = watchlist if watchlist is not None else SP500_WATCHLIST
        rows = self._collect_ticker_news(date, wl)
        df   = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        result["ticker_news"] = df

        if save_csv:
            self._export_csv(result, date)

        return result

    def collect_range(self, start_date: str, end_date: str, save_csv: bool = False,
                      watchlist: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """
        Collect all three news sources across a date range (max 3 months).

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD' (inclusive)
            save_csv:   If True, write combined results to {csv_dir}/{start}_{end}_{source}.csv

        Returns:
            {'stock_news': DataFrame, 'global_news': DataFrame, 'industry_news': DataFrame}
            Each DataFrame covers all dates in the range, sorted by date + composite_score.

        Raises:
            ValueError: If the date range exceeds 3 months (92 days).
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")

        if (end - start).days > 92:
            raise ValueError(
                f"Date range {start_date} → {end_date} exceeds the 3-month rolling window. "
                f"Benzinga history is limited to ~3 months."
            )

        all_sources = list(self.SOURCE_CHANNELS) + ["ticker_news"]
        all_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in all_sources}

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily    = self.collect(date_str, watchlist=watchlist)
            for source, df in daily.items():
                if not df.empty:
                    all_frames[source].append(df)
            current += timedelta(days=1)

        result = {}
        for source, frames in all_frames.items():
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined = combined.sort_values(
                    ["date", "composite_score"], ascending=[True, False]
                ).reset_index(drop=True)
                result[source] = combined
            else:
                result[source] = pd.DataFrame()

        if save_csv:
            self._export_csv(result, f"{start_date}_to_{end_date}")

        return result


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MFT Alpha Finder — Stage 1 Data Collector")
    # Default: 3-month rolling window ending yesterday (Philippines = UTC+8, 1 day ahead of US)
    yesterday   = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    three_months = (datetime.today() - timedelta(days=92)).strftime("%Y-%m-%d")
    parser.add_argument("--start-date", default=three_months, help="Range start YYYY-MM-DD (default: 3 months ago)")
    parser.add_argument("--end-date",   default=yesterday,    help="Range end   YYYY-MM-DD (default: yesterday US date)")
    parser.add_argument("--cache-dir",  default="data/cache", help="Cache directory")
    parser.add_argument("--no-csv",     action="store_true",  help="Disable CSV export (CSV is saved by default)")
    parser.add_argument("--csv-dir",    default="data",       help="Directory for CSV output (default: data/)")
    args = parser.parse_args()

    api_key = os.getenv("BENZINGA_API", "")
    if not api_key:
        print("[ERROR] BENZINGA_API not set in environment / .env")
        raise SystemExit(1)

    save_csv  = not args.no_csv   # CSV is ON by default
    collector = Stage1DataCollector(api_key=api_key, cache_dir=args.cache_dir, csv_dir=args.csv_dir)

    print(f"\n  Date range : {args.start_date}  →  {args.end_date}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(f"  CSV dir    : {args.csv_dir}\n")

    results = collector.collect_range(args.start_date, args.end_date, save_csv=save_csv)

    for source, df in results.items():
        print(f"\n── {source}: {len(df)} articles ──")
        if not df.empty:
            pd.set_option("display.max_colwidth", 70)
            print(df[["date", "composite_score", "catalyst_type", "title", "publisher"]].to_string(index=False))
