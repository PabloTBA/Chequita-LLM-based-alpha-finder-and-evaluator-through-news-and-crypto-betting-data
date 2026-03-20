"""
NewsSummarizer
==============
Generates a structured JSON summary of the last N days of news
(from Stage1DataCollector output) using a local LLM via Ollama.

Design:
  - One combined prompt across all 3 sources (stock / global / industry)
  - Top 5 articles per source by composite_score → max 15 articles total
  - Single LLM call per summarize() to minimise token overhead
  - JSON output: summary, top_themes, market_bias, key_risks

Usage:
    import ollama
    from news_summarizer import NewsSummarizer

    def llm_client(prompt: str) -> str:
        resp = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        return resp["message"]["content"]

    summarizer = NewsSummarizer(llm_client=llm_client, window_days=7)
    result = summarizer.summarize(articles, as_of_date="2026-03-18")
"""

import json
from datetime import datetime, timedelta

import pandas as pd

# Articles per source included in the prompt (token budget control)
TOP_N_PER_SOURCE = 5

NO_DATA_RESULT = {
    "summary":     "No news articles available for the specified window.",
    "top_themes":  [],
    "market_bias": "neutral",
    "key_risks":   [],
}

PROMPT_TEMPLATE = """\
You are a senior financial analyst. Summarise the following news articles from the past {window_days} days \
(up to {as_of_date}) for a US equities trader making pre-market decisions.

Return ONLY valid JSON with exactly these keys:
{{
  "summary":     "<2-4 sentence narrative of dominant market themes>",
  "top_themes":  ["<theme 1>", "<theme 2>", "<theme 3>"],
  "market_bias": "<bullish | bearish | neutral>",
  "key_risks":   ["<risk 1>", "<risk 2>"]
}}

--- STOCK NEWS (top {n} by impact score) ---
{stock_news}

--- GLOBAL / MACRO NEWS (top {n} by impact score) ---
{global_news}

--- INDUSTRY NEWS (top {n} by impact score) ---
{industry_news}

Respond with JSON only. No explanation outside the JSON block."""


class NewsSummarizer:
    """
    Generates a daily structured summary of recent news for screener context.

    Args:
        llm_client:  callable(prompt: str) -> str
                     Any function that sends a prompt to the LLM and returns
                     the raw string response (e.g. Ollama wrapper).
        window_days: Lookback window in days. Must be between 1 and 14.
                     Default: 7.
    """

    def __init__(self, llm_client: callable, window_days: int = 7, verbose: bool = False):
        if window_days < 1:
            raise ValueError(f"window_days must be at least 1, got {window_days}")
        if window_days > 14:
            raise ValueError(f"window_days must be at most 14, got {window_days}")
        self.llm_client  = llm_client
        self.window_days = window_days
        self.verbose     = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Internal helpers ──────────────────────────────────────

    def _filter_window(self, df: pd.DataFrame, window_start: str, as_of_date: str) -> pd.DataFrame:
        """Return rows within [window_start, as_of_date] inclusive."""
        if df.empty or "date" not in df.columns:
            return pd.DataFrame()
        mask = (df["date"] >= window_start) & (df["date"] <= as_of_date)
        return df[mask].copy()

    def _top_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Top N by composite_score, sorted descending."""
        if df.empty:
            return df
        col = "composite_score" if "composite_score" in df.columns else df.columns[0]
        return df.sort_values(col, ascending=False).head(TOP_N_PER_SOURCE)

    def _format_articles(self, df: pd.DataFrame) -> str:
        """Format articles as a compact numbered list for the prompt."""
        if df.empty:
            return "(none)"
        lines = []
        for i, row in enumerate(df.itertuples(), 1):
            title = getattr(row, "title", "")
            date  = getattr(row, "date",  "")
            score = getattr(row, "composite_score", "")
            lines.append(f"{i}. [{date}] (score:{score}) {title}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        stock_df: pd.DataFrame,
        global_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        as_of_date: str,
    ) -> str:
        return PROMPT_TEMPLATE.format(
            window_days   = self.window_days,
            as_of_date    = as_of_date,
            n             = TOP_N_PER_SOURCE,
            stock_news    = self._format_articles(self._top_articles(stock_df)),
            global_news   = self._format_articles(self._top_articles(global_df)),
            industry_news = self._format_articles(self._top_articles(industry_df)),
        )

    def _parse_llm_response(self, raw: str) -> dict:
        """
        Parse LLM JSON response. Returns a safe fallback on parse failure
        rather than raising — upstream pipeline should never crash here.
        """
        try:
            # Strip markdown code fences if the model added them
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            return {
                "summary":     str(data.get("summary", "")),
                "top_themes":  list(data.get("top_themes", [])),
                "market_bias": str(data.get("market_bias", "neutral")),
                "key_risks":   list(data.get("key_risks", [])),
            }
        except (json.JSONDecodeError, Exception):
            return {
                "summary":     raw.strip()[:500],   # return raw text as fallback
                "top_themes":  [],
                "market_bias": "neutral",
                "key_risks":   [],
            }

    # ── Public API ────────────────────────────────────────────

    def summarize(
        self,
        articles: dict[str, pd.DataFrame],
        as_of_date: str,
    ) -> dict:
        """
        Generate a structured news summary for the lookback window.

        Args:
            articles:   Output of Stage1DataCollector.collect_range() —
                        {'stock_news': df, 'global_news': df, 'industry_news': df}
            as_of_date: Reference date 'YYYY-MM-DD' (typically yesterday US date).

        Returns:
            {
                'window_start':  'YYYY-MM-DD',
                'window_end':    'YYYY-MM-DD',
                'article_count': int,
                'summary':       str,
                'top_themes':    list[str],
                'market_bias':   str,   # 'bullish' | 'bearish' | 'neutral'
                'key_risks':     list[str],
            }
        """
        as_of_dt     = datetime.strptime(as_of_date, "%Y-%m-%d")
        window_start = (as_of_dt - timedelta(days=self.window_days)).strftime("%Y-%m-%d")

        self._log(f"\n[NewsSummarizer] window: {window_start} → {as_of_date} ({self.window_days} days)")

        stock_df    = self._filter_window(articles.get("stock_news",    pd.DataFrame()), window_start, as_of_date)
        global_df   = self._filter_window(articles.get("global_news",   pd.DataFrame()), window_start, as_of_date)
        industry_df = self._filter_window(articles.get("industry_news", pd.DataFrame()), window_start, as_of_date)

        total_count = len(stock_df) + len(global_df) + len(industry_df)

        self._log(f"[NewsSummarizer] articles in window — stock:{len(stock_df)}  global:{len(global_df)}  industry:{len(industry_df)}  total:{total_count}")

        base = {
            "window_start":  window_start,
            "window_end":    as_of_date,
            "article_count": total_count,
        }

        if total_count == 0:
            self._log("[NewsSummarizer] no articles in window — returning no-data result (LLM skipped)")
            return {**base, **NO_DATA_RESULT}

        prompt = self._build_prompt(stock_df, global_df, industry_df, as_of_date)
        self._log(f"[NewsSummarizer] prompt built — {len(prompt)} chars, sending to LLM...")

        raw    = self.llm_client(prompt)
        self._log(f"[NewsSummarizer] LLM response received — {len(raw)} chars")
        self._log(f"[NewsSummarizer] raw response:\n{raw[:500]}")

        parsed = self._parse_llm_response(raw)
        self._log(f"[NewsSummarizer] parsed result: {parsed}")

        return {**base, **parsed}
