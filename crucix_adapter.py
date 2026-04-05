"""
crucix_adapter.py
=================
Pulls real-time macro/geopolitical intelligence from the Crucix sidecar
(Node.js, port 3117) and converts it into Chequita-compatible data structures.

Crucix runs 27 OSINT sources in parallel every 15 minutes:
  Tier 1  — GDELT, OpenSky, FIRMS (NASA fires), Maritime, Safecast, ACLED,
             ReliefWeb, WHO, OFAC, OpenSanctions, ADS-B
  Tier 2  — FRED (22 macro indicators), Treasury, BLS, EIA, GSCPI,
             USAspending, UN Comtrade
  Tier 3  — NOAA, EPA RadNet, USPTO Patents, Bluesky, Reddit, Telegram, KiwiSDR
  Tier 4  — CelesTrak (satellite tracking)
  Tier 5  — Yahoo Finance (live prices)

Usage
-----
    from crucix_adapter import CrucixAdapter

    adapter = CrucixAdapter()               # optional: base_url="http://localhost:3117"
    snapshot = adapter.fetch()              # None if Crucix is not running
    if snapshot:
        macro_context = adapter.to_macro_context(snapshot)
        news_articles = adapter.to_articles(snapshot)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:3117"
FETCH_TIMEOUT_S  = 10      # seconds before giving up on Crucix
MAX_ARTICLES     = 200     # cap on articles returned to avoid flooding the pipeline

# Mapping from FRED series ID to a human-readable macro field name used by
# MacroScreener / NewsSummarizer.
_FRED_FIELD_MAP: dict[str, str] = {
    "DFF":               "fed_funds_rate",
    "DGS2":              "yield_2y",
    "DGS10":             "yield_10y",
    "DGS30":             "yield_30y",
    "T10Y2Y":            "yield_curve_10y2y",
    "T10Y3M":            "yield_curve_10y3m",
    "CPIAUCSL":          "cpi_all_items",
    "CPILFESL":          "core_cpi",
    "PCEPI":             "pce_price_index",
    "MICH":              "michigan_inflation_exp",
    "UNRATE":            "unemployment_rate",
    "PAYEMS":            "nonfarm_payrolls",
    "ICSA":              "initial_jobless_claims",
    "M2SL":              "m2_money_supply",
    "WALCL":             "fed_balance_sheet",
    "VIXCLS":            "vix",
    "BAMLH0A0HYM2":      "high_yield_spread",
    "DCOILWTICO":        "wti_crude_fred",
    "GOLDAMGBD228NLBM":  "gold_price",
    "MORTGAGE30US":      "mortgage_30y",
    "DTWEXBGS":          "usd_trade_weighted",
}

# Yahoo Finance tickers that Crucix tracks
_YFINANCE_FIELD_MAP: dict[str, str] = {
    "SPY":  "spy_price",
    "QQQ":  "qqq_price",
    "DIA":  "dia_price",
    "VIX":  "vix_live",
    "GLD":  "gold_etf",
    "USO":  "oil_etf",
    "BTC":  "btc_price",
    "ETH":  "eth_price",
    "GC=F": "gold_futures",
    "CL=F": "wti_futures",
    "NG=F": "natgas_futures",
}


class CrucixAdapter:
    """Fetches the latest Crucix intelligence snapshot and converts it for Chequita."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    # ── public API ─────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Return True if Crucix server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def fetch(self) -> dict | None:
        """
        Fetch the latest synthesized data from Crucix's /api/data endpoint.
        Returns None (with a warning) if Crucix is unavailable or hasn't
        completed its first sweep yet.
        """
        try:
            r = requests.get(f"{self.base_url}/api/data", timeout=FETCH_TIMEOUT_S)
            if r.status_code == 503:
                logger.warning("[CrucixAdapter] Crucix is up but first sweep not done yet.")
                return None
            r.raise_for_status()
            data = r.json()
            logger.info("[CrucixAdapter] Fetched Crucix snapshot (sources: %s ok / %s total)",
                        data.get("meta", {}).get("sourcesOk", "?"),
                        data.get("meta", {}).get("sourcesQueried", "?"))
            return data
        except requests.exceptions.ConnectionError:
            logger.warning("[CrucixAdapter] Crucix not running at %s — skipping OSINT enrichment.", self.base_url)
            return None
        except Exception as exc:
            logger.warning("[CrucixAdapter] Failed to fetch from Crucix: %s", exc)
            return None

    def to_macro_context(self, snapshot: dict) -> dict:
        """
        Convert a Crucix snapshot into a flat macro-context dict that
        MacroScreener and NewsSummarizer can consume.

        Keys follow the naming convention used in Chequita's LLM prompts:
          fed_funds_rate, yield_curve_10y2y, vix, wti_crude, ...
        """
        ctx: dict[str, Any] = {"source": "crucix", "timestamp": snapshot.get("crucix", {}).get("timestamp")}

        sources = snapshot.get("sources", {})

        # ── FRED macro indicators ──────────────────────────────────────────────
        fred_indicators = sources.get("FRED", {}).get("indicators", [])
        for ind in fred_indicators:
            field = _FRED_FIELD_MAP.get(ind.get("id", ""))
            if field and ind.get("value") is not None:
                ctx[field] = {
                    "value": ind["value"],
                    "date":  ind.get("date"),
                    "mom_change_pct": ind.get("momChangePct"),
                }

        # ── EIA energy prices ──────────────────────────────────────────────────
        eia = sources.get("EIA", {})
        oil = eia.get("oilPrices", {})
        if oil.get("wti", {}).get("value") is not None:
            ctx["wti_crude"] = {"value": oil["wti"]["value"], "date": oil["wti"].get("date")}
        if oil.get("brent", {}).get("value") is not None:
            ctx["brent_crude"] = {"value": oil["brent"]["value"], "date": oil["brent"].get("date")}
        if eia.get("gasPrice", {}).get("value") is not None:
            ctx["natural_gas"] = {"value": eia["gasPrice"]["value"]}
        ctx["energy_signals"] = eia.get("signals", [])

        # ── Yahoo Finance live prices ──────────────────────────────────────────
        yf = sources.get("YahooFinance", {})
        prices = yf.get("prices", {}) or {}
        for ticker, field in _YFINANCE_FIELD_MAP.items():
            entry = prices.get(ticker, {})
            if entry.get("price") is not None:
                ctx[field] = {
                    "price":       entry["price"],
                    "change_pct":  entry.get("changePct"),
                    "timestamp":   entry.get("timestamp"),
                }

        # ── BLS employment indicators ──────────────────────────────────────────
        bls_indicators = sources.get("BLS", {}).get("indicators", [])
        for ind in bls_indicators:
            series = ind.get("series_id", ind.get("seriesId", "")).upper()
            if series == "LNS14000000":   # unemployment rate
                ctx["unemployment_bls"] = {"value": ind.get("value"), "date": ind.get("date")}
            elif series == "CES0000000001":  # total nonfarm payrolls
                ctx["nonfarm_payrolls_bls"] = {"value": ind.get("value"), "date": ind.get("date")}

        # ── GSCPI (supply chain pressure) ─────────────────────────────────────
        gscpi = sources.get("GSCPI", {}).get("latest")
        if gscpi is not None:
            ctx["supply_chain_pressure"] = gscpi

        # ── ACLED conflict summary ─────────────────────────────────────────────
        acled = sources.get("ACLED", {})
        if not acled.get("error"):
            ctx["conflict_events_total"]     = acled.get("totalEvents", 0)
            ctx["conflict_fatalities_total"] = acled.get("totalFatalities", 0)
            ctx["conflict_by_type"]          = acled.get("byType", {})
            ctx["conflict_signals"]          = acled.get("signals", [])

        # ── Sanctions (OFAC + OpenSanctions) ──────────────────────────────────
        ofac = sources.get("OFAC", {})
        ctx["ofac_total_entries"]     = ofac.get("totalEntries", 0)
        ctx["ofac_recent_additions"]  = ofac.get("recentAdditions", [])
        ctx["ofac_signals"]           = ofac.get("signals", [])

        opensanctions = sources.get("OpenSanctions", {})
        ctx["opensanctions_signals"]  = opensanctions.get("signals", [])

        # ── GDELT geopolitical event count ────────────────────────────────────
        gdelt = sources.get("GDELT", {})
        ctx["gdelt_article_count"]    = gdelt.get("articleCount", 0)
        ctx["gdelt_avg_tone"]         = gdelt.get("avgTone")
        ctx["gdelt_signals"]          = gdelt.get("signals", [])

        # ── Treasury ──────────────────────────────────────────────────────────
        treasury = sources.get("Treasury", {})
        ctx["treasury_signals"]       = treasury.get("signals", [])

        # ── WHO health alerts ──────────────────────────────────────────────────
        who = sources.get("WHO", {})
        ctx["who_disease_alerts"]     = (who.get("diseaseOutbreakNews") or [])[:5]

        # ── NOAA weather alerts ────────────────────────────────────────────────
        noaa = sources.get("NOAA", {})
        ctx["noaa_severe_alerts"]     = noaa.get("totalSevereAlerts", 0)

        # ── Aggregate risk signals (cross-domain) ──────────────────────────────
        all_signals: list[str] = []
        for src_key in ("FRED", "EIA", "ACLED", "OFAC", "GDELT", "Treasury", "Space"):
            for sig in sources.get(src_key, {}).get("signals", []):
                if isinstance(sig, str) and sig:
                    all_signals.append(sig)
        ctx["cross_domain_risk_signals"] = all_signals

        return ctx

    def to_articles(self, snapshot: dict) -> list[dict]:
        """
        Convert Crucix intelligence data into a list of article-like dicts
        compatible with Chequita's article schema:
            { date, headline, summary, source, tickers, impact_score }

        Sources converted:
          - GDELT global news articles
          - Telegram OSINT urgent posts
          - WHO disease outbreak alerts
          - ACLED deadliest conflict events
          - OFAC recent sanctions additions
          - Reddit & Bluesky social sentiment posts
        """
        articles: list[dict] = []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sources = snapshot.get("sources", {})

        # ── GDELT articles ─────────────────────────────────────────────────────
        gdelt_articles = sources.get("GDELT", {}).get("articles", [])
        for art in gdelt_articles[:60]:
            headline = art.get("title") or art.get("url", "")
            if not headline:
                continue
            articles.append({
                "date":         art.get("seendate", today)[:10],
                "headline":     headline[:200],
                "summary":      art.get("domain", ""),
                "source":       "GDELT",
                "tickers":      [],
                "impact_score": _tone_to_impact(art.get("tone", 0)),
            })

        # ── Telegram OSINT posts ───────────────────────────────────────────────
        tg = sources.get("Telegram", {})
        for post in (tg.get("urgentPosts") or [])[:30]:
            text = post.get("text", "")
            if not text or _is_cyrillic(text):
                continue
            articles.append({
                "date":         _parse_date(post.get("date"), today),
                "headline":     text[:200],
                "summary":      f"Telegram/{post.get('channel', 'unknown')} — {post.get('views', 0)} views",
                "source":       "Telegram_OSINT",
                "tickers":      [],
                "impact_score": _urgency_to_impact(post.get("urgentFlags", [])),
            })

        # ── WHO disease alerts ─────────────────────────────────────────────────
        for alert in (sources.get("WHO", {}).get("diseaseOutbreakNews") or [])[:10]:
            title = alert.get("title", "")
            if not title:
                continue
            articles.append({
                "date":         _parse_date(alert.get("date"), today),
                "headline":     title[:200],
                "summary":      alert.get("summary", "")[:300],
                "source":       "WHO",
                "tickers":      [],
                "impact_score": 0.6,   # health alerts are always moderately impactful
            })

        # ── ACLED conflict events ──────────────────────────────────────────────
        acled = sources.get("ACLED", {})
        for evt in (acled.get("deadliestEvents") or [])[:20]:
            notes = evt.get("notes", evt.get("event_type", "Conflict event"))
            articles.append({
                "date":         _parse_date(evt.get("event_date"), today),
                "headline":     f"ACLED: {evt.get('event_type', 'conflict')} in {evt.get('country', '?')} "
                                f"({evt.get('fatalities', 0)} fatalities)",
                "summary":      notes[:300],
                "source":       "ACLED",
                "tickers":      [],
                "impact_score": min(1.0, 0.5 + (evt.get("fatalities", 0) or 0) / 200),
            })

        # ── OFAC sanctions additions ───────────────────────────────────────────
        for entry in (acled.get("recentAdditions") or sources.get("OFAC", {}).get("recentAdditions") or [])[:10]:
            name = entry.get("name", entry.get("uid", "unknown"))
            articles.append({
                "date":         today,
                "headline":     f"OFAC sanctions addition: {name}",
                "summary":      f"Program: {entry.get('programs', ['?'])[0] if entry.get('programs') else '?'}",
                "source":       "OFAC",
                "tickers":      [],
                "impact_score": 0.7,
            })

        # ── Reddit sentiment ───────────────────────────────────────────────────
        reddit = sources.get("Reddit", {})
        for post in (reddit.get("topPosts") or [])[:20]:
            title = post.get("title", "")
            if not title:
                continue
            articles.append({
                "date":         today,
                "headline":     title[:200],
                "summary":      f"r/{post.get('subreddit', '?')} — score {post.get('score', 0)}",
                "source":       "Reddit",
                "tickers":      _extract_tickers(title),
                "impact_score": min(1.0, 0.3 + (post.get("score", 0) or 0) / 10000),
            })

        # ── Bluesky sentiment ──────────────────────────────────────────────────
        bluesky = sources.get("Bluesky", {})
        for post in (bluesky.get("posts") or [])[:20]:
            text = post.get("text", "")
            if not text:
                continue
            articles.append({
                "date":         today,
                "headline":     text[:200],
                "summary":      f"Bluesky — {post.get('likeCount', 0)} likes",
                "source":       "Bluesky",
                "tickers":      _extract_tickers(text),
                "impact_score": 0.3,
            })

        return articles[:MAX_ARTICLES]

    def to_summary_text(self, snapshot: dict) -> str:
        """
        Return a short plaintext summary of the Crucix snapshot suitable for
        injecting into the NewsSummarizer or MacroScreener prompt as extra context.
        """
        ctx = self.to_macro_context(snapshot)
        lines: list[str] = ["=== Crucix OSINT Intelligence Snapshot ==="]

        # Macro indicators
        if "vix" in ctx:
            lines.append(f"VIX: {ctx['vix']['value']}")
        if "yield_curve_10y2y" in ctx:
            lines.append(f"Yield Curve (10Y-2Y): {ctx['yield_curve_10y2y']['value']:.2f}%")
        if "wti_crude" in ctx:
            lines.append(f"WTI Crude: ${ctx['wti_crude']['value']:.2f}")
        if "fed_funds_rate" in ctx:
            lines.append(f"Fed Funds Rate: {ctx['fed_funds_rate']['value']:.2f}%")
        if "high_yield_spread" in ctx:
            lines.append(f"High-Yield Credit Spread: {ctx['high_yield_spread']['value']:.2f}%")

        # Live market
        if "spy_price" in ctx:
            chg = ctx["spy_price"].get("change_pct")
            chg_str = f" ({chg:+.2f}%)" if chg is not None else ""
            lines.append(f"SPY: ${ctx['spy_price']['price']:.2f}{chg_str}")

        # Supply chain
        if "supply_chain_pressure" in ctx:
            gscpi = ctx["supply_chain_pressure"]
            lines.append(f"Global Supply Chain Pressure Index: {gscpi}")

        # Conflict
        total_evts = ctx.get("conflict_events_total", 0)
        if total_evts:
            lines.append(f"Armed Conflict Events (ACLED): {total_evts} events, "
                         f"{ctx.get('conflict_fatalities_total', 0)} fatalities")

        # Cross-domain signals
        signals = ctx.get("cross_domain_risk_signals", [])
        if signals:
            lines.append("\nKey risk signals:")
            for sig in signals[:10]:
                lines.append(f"  • {sig}")

        # WHO
        who_alerts = ctx.get("who_disease_alerts", [])
        if who_alerts:
            lines.append("\nWHO Health Alerts:")
            for a in who_alerts[:3]:
                lines.append(f"  • {a.get('title', '')[:120]}")

        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _tone_to_impact(tone: float | None) -> float:
    """Convert GDELT article tone (negative = bad news) to 0-1 impact score."""
    if tone is None:
        return 0.4
    return min(1.0, max(0.1, 0.5 + abs(float(tone)) / 20))


def _urgency_to_impact(flags: list) -> float:
    return min(1.0, 0.5 + len(flags) * 0.15)


def _parse_date(raw: Any, fallback: str) -> str:
    if not raw:
        return fallback
    try:
        s = str(raw)
        return s[:10]
    except Exception:
        return fallback


_CYRILLIC = set(range(0x0400, 0x0500))


def _is_cyrillic(text: str) -> bool:
    cyrillic_chars = sum(1 for c in text[:80] if ord(c) in _CYRILLIC)
    return cyrillic_chars > 5


# Common ticker patterns in social text
import re as _re
_TICKER_RE = _re.compile(r'\$([A-Z]{1,5})\b')


def _extract_tickers(text: str) -> list[str]:
    return list(dict.fromkeys(_TICKER_RE.findall(text)))[:5]
