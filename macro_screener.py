"""
MacroScreener
=============
Takes the news_summarizer output and calls the LLM once to produce a macro
filter used by ticker_screener downstream.

Public interface
----------------
    screener = MacroScreener(llm_client=llm_fn, verbose=True)
    result   = screener.screen(summary_dict)

Output keys
-----------
    favored_sectors    — list[str]  sectors to overweight
    avoid_sectors      — list[str]  sectors to avoid
    active_macro_risks — list[str]  macro risks to flag per ticker
    market_bias        — "bullish" | "bearish" | "neutral"
    reasoning          — str        LLM explanation
"""

from __future__ import annotations

import json

VALID_BIASES = {"bullish", "bearish", "neutral"}

_NEUTRAL_DEFAULT = {
    "favored_sectors":    [],
    "avoid_sectors":      [],
    "active_macro_risks": [],
    "market_bias":        "neutral",
    "reasoning":          "Insufficient data to form a macro view.",
}

_SYSTEM_PROMPT = """\
You are a macro analyst. Given a daily news summary, identify:
1. Sectors that look favored by current macro conditions
2. Sectors to avoid
3. Active macro risks that could impact any trade

Respond ONLY with valid JSON matching this schema:
{
  "favored_sectors":    ["string", ...],
  "avoid_sectors":      ["string", ...],
  "active_macro_risks": ["string", ...],
  "market_bias":        "bullish" | "bearish" | "neutral",
  "reasoning":          "string"
}
"""


class MacroScreener:
    def __init__(self, llm_client: callable, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def screen(self, summary: dict) -> dict:
        """
        Produce a macro filter from a news_summarizer output dict.

        Returns neutral defaults immediately if article_count == 0.
        Falls back to neutral defaults on LLM parse failure.
        """
        if summary.get("article_count", 0) == 0:
            self._log("[MacroScreener] no articles — returning neutral default (LLM skipped)")
            return dict(_NEUTRAL_DEFAULT)

        prompt = self._build_prompt(summary)
        self._log(f"[MacroScreener] sending prompt ({len(prompt)} chars) to LLM...")

        raw    = self.llm_client(prompt)
        self._log(f"[MacroScreener] raw response ({len(raw)} chars):\n{raw[:400]}")

        return self._parse(raw)

    # ── private ───────────────────────────────────────────────────────────────

    def _build_prompt(self, summary: dict) -> str:
        themes = ", ".join(summary.get("top_themes", []))
        risks  = ", ".join(summary.get("key_risks",  []))
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"=== News Summary ({summary.get('window_start')} to {summary.get('window_end')}) ===\n"
            f"{summary.get('summary', '')}\n\n"
            f"Top themes: {themes}\n"
            f"Key risks:  {risks}\n"
            f"Overall market bias from news: {summary.get('market_bias', 'neutral')}\n"
        )

    def _parse(self, raw: str) -> dict:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            self._log("[MacroScreener] JSON parse failed — returning neutral default")
            return dict(_NEUTRAL_DEFAULT)

        result = dict(_NEUTRAL_DEFAULT)

        for key in ("favored_sectors", "avoid_sectors", "active_macro_risks"):
            val = data.get(key)
            if isinstance(val, list) and all(isinstance(s, str) for s in val):
                result[key] = val

        bias = data.get("market_bias", "neutral")
        result["market_bias"] = bias if bias in VALID_BIASES else "neutral"

        reasoning = data.get("reasoning", "")
        result["reasoning"] = reasoning if isinstance(reasoning, str) and reasoning else _NEUTRAL_DEFAULT["reasoning"]

        return result


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ollama, os, json as _json, sys
    from dotenv import load_dotenv; load_dotenv()
    from Stage1DataCollector import Stage1DataCollector
    from news_summarizer import NewsSummarizer
    from datetime import datetime, timedelta

    date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    collector  = Stage1DataCollector(api_key=os.getenv("BENZINGA_API"), cache_dir="data/cache")
    articles   = collector.collect(date)

    def llm(prompt):
        resp = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    summary  = NewsSummarizer(llm_client=llm, window_days=7, verbose=True).summarize(articles, as_of_date=date)
    result   = MacroScreener(llm_client=llm, verbose=True).screen(summary)

    print("\n" + "=" * 60)
    print(_json.dumps(result, indent=2))
