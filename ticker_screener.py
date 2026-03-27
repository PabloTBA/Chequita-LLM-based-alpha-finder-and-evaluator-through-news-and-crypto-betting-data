"""
TickerScreener
==============
Three-step screener that goes from raw news articles to 20 per-ticker verdicts.

Step 1 — prefilter  (no LLM)
    Combine all article sources, group by ticker, sum composite_scores,
    return top-50 as a DataFrame.

Step 2 — shortlist  (1 LLM call)
    Feed top-50 tickers + macro filter + OHLCV features to LLM.
    LLM returns exactly 20 tickers.  Falls back to top-20 by score on parse failure.

Step 3 — screen_tickers  (1 LLM call per ticker)
    Per-ticker focused LLM call using macro thesis + OHLCV features.
    Returns buy / watch / avoid verdict + reasoning.
    Malformed JSON defaults to "watch".

Public interface
----------------
    screener = TickerScreener(llm_client=llm_fn, verbose=True)
    top50    = screener.prefilter(articles)          # dict[str, DataFrame]
    short20  = screener.shortlist(top50, macro, ohlcv)
    verdicts = screener.screen_tickers(short20, macro, ohlcv)
"""

from __future__ import annotations

import json
import pandas as pd

VALID_VERDICTS  = {"buy", "watch", "avoid"}
PREFILTER_LIMIT = 50
SHORTLIST_LIMIT = 20

_SHORTLIST_PROMPT = """\
You are a stock screener. Given the top-50 tickers ranked by news score, \
the current macro environment, and OHLCV summaries, select exactly {n} tickers \
that represent the highest-conviction opportunities.

Macro context:
  Favored sectors:    {favored}
  Avoid sectors:      {avoid}
  Active macro risks: {risks}
  Market bias:        {bias}

Sector diversity rules (MANDATORY):
  - Include NO MORE THAN 2 tickers from the same GICS sector.
  - Prefer tickers from favored sectors when scores are similar.
  - Actively spread selections across different sectors to maximise breadth.

Top-50 tickers by news score:
{ticker_list}

OHLCV features (tickers with data):
{ohlcv_block}

Respond ONLY with valid JSON:
{{"tickers": ["TICK1", "TICK2", ...]}}
(exactly {n} ticker symbols)
"""

_VERDICT_PROMPT = """\
You are a stock analyst. Evaluate {ticker} given the macro environment and its \
price action, then return a single structured verdict.

Macro context:
  Favored sectors:    {favored}
  Avoid sectors:      {avoid}
  Active macro risks: {risks}
  Market bias:        {bias}

OHLCV features for {ticker}:
{ohlcv_block}

Respond ONLY with valid JSON:
{{"verdict": "buy" | "watch" | "avoid", "reasoning": "one or two sentence explanation"}}
"""

_DEFAULT_VERDICT = {"verdict": "watch", "reasoning": "Unable to parse LLM response — defaulting to watch."}


class TickerScreener:
    def __init__(self, llm_client: callable, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose    = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Step 1: prefilter ─────────────────────────────────────────────────────

    def prefilter(self, articles: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all article sources, group by ticker, aggregate scores,
        return top-50 as DataFrame with columns [ticker, composite_score].

        Scoring aggregation — hybrid of max and normalised sum:
          per_ticker_score = 0.7 × max_article_score
                           + 0.3 × sum(article_score / n_tickers_in_article)

        The normalisation by n_tickers prevents a single broadside article
        ("AAPL MSFT GOOGL AMZN... lead rally") from giving every listed ticker
        the full composite score.  The max component preserves the signal from
        one genuinely high-quality, ticker-specific article.
        """
        frames = [df for df in articles.values() if isinstance(df, pd.DataFrame) and not df.empty]

        if not frames:
            return pd.DataFrame(columns=["ticker", "composite_score"])

        combined = pd.concat(frames, ignore_index=True)

        # Real articles use 'tickers' (comma-separated string); tests use 'ticker'
        if "tickers" in combined.columns and "ticker" not in combined.columns:
            combined = combined.rename(columns={"tickers": "ticker"})

        # Count tickers per article BEFORE exploding — needed for normalisation
        combined["_n_tickers"] = (
            combined["ticker"].astype(str)
            .str.split(",")
            .apply(lambda ts: max(1, len([t for t in ts if t.strip() and t.strip() != "nan"])))
        )

        # Normalised per-ticker contribution: article score ÷ number of tickers tagged
        combined["_norm_score"] = combined["composite_score"] / combined["_n_tickers"]

        # Explode to one row per ticker
        combined["ticker"] = combined["ticker"].astype(str).str.split(",")
        combined = combined.explode("ticker")
        combined["ticker"] = combined["ticker"].str.strip()

        # Drop rows with missing or empty ticker
        combined = combined[combined["ticker"].notna()]
        combined = combined[combined["ticker"].str.strip() != ""]
        combined = combined[combined["ticker"] != "nan"]

        # Hybrid aggregation per ticker
        agg = combined.groupby("ticker", as_index=False).agg(
            _max_score=("composite_score", "max"),
            _sum_norm =("_norm_score", "sum"),
        )
        agg["composite_score"] = 0.7 * agg["_max_score"] + 0.3 * agg["_sum_norm"]

        scored = (
            agg[["ticker", "composite_score"]]
            .sort_values("composite_score", ascending=False)
            .head(PREFILTER_LIMIT)
            .reset_index(drop=True)
        )

        self._log(f"[TickerScreener] prefilter: {len(scored)} tickers after top-{PREFILTER_LIMIT} cut")
        return scored

    # ── Step 2: shortlist ─────────────────────────────────────────────────────

    def shortlist(self, top50: pd.DataFrame, macro: dict, ohlcv: dict) -> list[str]:
        """LLM picks exactly 20 tickers from top50. Falls back to top-20 by score."""
        fallback   = top50.head(SHORTLIST_LIMIT)["ticker"].tolist()
        ticker_list = "\n".join(
            f"  {i+1:2d}. {row['ticker']} (score={row['composite_score']:.1f})"
            for i, row in top50.iterrows()
        )
        ohlcv_block = self._format_ohlcv(ohlcv)
        prompt = _SHORTLIST_PROMPT.format(
            n           = SHORTLIST_LIMIT,
            favored     = ", ".join(macro.get("favored_sectors", [])),
            avoid       = ", ".join(macro.get("avoid_sectors", [])),
            risks       = ", ".join(macro.get("active_macro_risks", [])),
            bias        = macro.get("market_bias", "neutral"),
            ticker_list = ticker_list,
            ohlcv_block = ohlcv_block,
        )

        print(f"  [LLM] TickerScreener: shortlisting top-{SHORTLIST_LIMIT} from {len(top50)} candidates...")
        self._log(f"[TickerScreener] shortlist: sending prompt ({len(prompt)} chars) to LLM...")
        raw = self.llm_client(prompt)
        print(f"  [LLM] TickerScreener: shortlist done")

        try:
            data    = json.loads(raw)
            tickers = data.get("tickers", [])
            top50_set = set(top50["ticker"].tolist())
            tickers = [t for t in tickers if isinstance(t, str) and t in top50_set]
            if len(tickers) == SHORTLIST_LIMIT:
                self._log(f"[TickerScreener] shortlist: {tickers}")
                return tickers
        except (json.JSONDecodeError, ValueError):
            pass

        self._log(f"[TickerScreener] shortlist: parse failed — falling back to top-{SHORTLIST_LIMIT}")
        return fallback

    # ── Step 3: screen_tickers ────────────────────────────────────────────────

    def screen_tickers(
        self,
        tickers:   list[str],
        macro:     dict,
        ohlcv:     dict,
        rag_store: object = None,
    ) -> list[dict]:
        """One LLM call per ticker, executed in parallel. Returns list of verdict dicts."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(tickers)

        def _screen_one(ticker: str) -> dict:
            # Data quality gate: no OHLCV → skip LLM, return avoid immediately
            if ohlcv.get(ticker) is None:
                print(f"  [Skip] TickerScreener: {ticker} → avoid (no OHLCV data)")
                return {"ticker": ticker, "verdict": "avoid",
                        "reasoning": "DATA_UNAVAILABLE — no OHLCV price data. Cannot evaluate price action."}

            prompt = _VERDICT_PROMPT.format(
                ticker      = ticker,
                favored     = ", ".join(macro.get("favored_sectors", [])),
                avoid       = ", ".join(macro.get("avoid_sectors", [])),
                risks       = ", ".join(macro.get("active_macro_risks", [])),
                bias        = macro.get("market_bias", "neutral"),
                ohlcv_block = self._format_ohlcv({ticker: ohlcv.get(ticker)}),
            )
            if rag_store is not None:
                chunks = rag_store.retrieve(ticker, collection="news", k=3)
                if chunks:
                    news_block = "\n".join(f"- {c}" for c in chunks)
                    prompt += f"\n\nRelevant news:\n{news_block}"
            raw = self.llm_client(prompt)
            verdict = self._parse_verdict(raw)
            print(f"  [LLM] TickerScreener: {ticker} → {verdict.get('verdict', '?')}")
            return {"ticker": ticker, **verdict}

        print(f"  [LLM] TickerScreener: screening {total} tickers in parallel...")
        results_map: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_screen_one, t): t for t in tickers}
            for future in as_completed(futures):
                result = future.result()
                results_map[result["ticker"]] = result

        # preserve original order
        return [results_map[t] for t in tickers if t in results_map]

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_ohlcv(ohlcv: dict) -> str:
        lines = []
        for ticker, feats in ohlcv.items():
            if feats is None:
                lines.append(f"  {ticker}: no data")
                continue
            lines.append(
                f"  {ticker}: return_20d={feats.get('return_20d', 'N/A'):.3f}  "
                f"rsi={feats.get('rsi_14', 'N/A'):.1f}  "
                f"atr_pct={feats.get('atr_pct', 'N/A'):.3f}  "
                f"vol_ratio={feats.get('volume_ratio_30d', 'N/A'):.2f}  "
                f"52w_hi={feats.get('52w_high_prox', 'N/A'):.2f}"
            )
        return "\n".join(lines) if lines else "  No OHLCV data available."

    @staticmethod
    def _parse_verdict(raw: str) -> dict:
        try:
            data    = json.loads(raw)
            verdict = data.get("verdict", "watch")
            if verdict not in VALID_VERDICTS:
                verdict = "watch"
            reasoning = data.get("reasoning", _DEFAULT_VERDICT["reasoning"])
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = _DEFAULT_VERDICT["reasoning"]
            return {"verdict": verdict, "reasoning": reasoning}
        except (json.JSONDecodeError, ValueError):
            return dict(_DEFAULT_VERDICT)


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ollama, os, json as _json
    from dotenv import load_dotenv; load_dotenv()
    from Stage1DataCollector import Stage1DataCollector
    from news_summarizer import NewsSummarizer
    from macro_screener import MacroScreener
    from ohlcv_fetcher import OHLCVFetcher
    from datetime import datetime, timedelta

    date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    def llm(prompt):
        resp = ollama.chat(model="qwen3:14b", messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0.0})
        return resp.message.content if hasattr(resp, "message") else resp["message"]["content"]

    print("Step 1: Collecting news...")
    collector = Stage1DataCollector(api_key=os.getenv("BENZINGA_API"), cache_dir="data/cache")
    articles  = collector.collect(date)

    print("Step 2: Summarising news...")
    summary = NewsSummarizer(llm_client=llm, window_days=7).summarize(articles, as_of_date=date)

    print("Step 3: Macro screen...")
    macro = MacroScreener(llm_client=llm).screen(summary)
    print(f"  bias={macro['market_bias']}  favored={macro['favored_sectors']}")

    print("Step 4: Pre-filtering tickers...")
    screener = TickerScreener(llm_client=llm, verbose=True)
    top50    = screener.prefilter(articles)
    print(f"  top50 head: {top50.head()['ticker'].tolist()}")

    print("Step 5: Shortlisting 20...")
    fetcher = OHLCVFetcher()
    ohlcv_raw = fetcher.fetch(top50["ticker"].tolist())
    ohlcv = {t: fetcher.compute_features(df) for t, df in ohlcv_raw.items() if df is not None}
    short20 = screener.shortlist(top50, macro, ohlcv)
    print(f"  shortlisted: {short20}")

    print("Step 6: Per-ticker verdicts...")
    verdicts = screener.screen_tickers(short20, macro, ohlcv)
    print("\n" + "=" * 60)
    print(_json.dumps(verdicts, indent=2))
