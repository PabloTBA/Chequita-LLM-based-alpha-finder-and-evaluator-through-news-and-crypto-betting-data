"""
crucix_adapter.py
=================
Pulls real-time macro/geopolitical intelligence from the Crucix sidecar
(Node.js, port 3117) and converts it into Chequita-compatible data structures.

The Crucix /api/data endpoint returns a synthesized V2 dashboard object:
  {
    meta:     { timestamp, ... }          # sweep metadata
    fred:     [{id, label, value, date, momChange, momChangePct}, ...]
    energy:   {wti, brent, natgas, wtiRecent, signals}
    metals:   {gold, silver, ...}
    bls:      [{...}, ...]
    gscpi:    <number>
    treasury: {totalDebt, signals}
    acled:    {totalEvents, totalFatalities, byType, deadliestEvents}
    gdelt:    {allArticles, avgTone, signals}
    tg:       {posts, urgent, topPosts}
    who:      [{title, date, summary}, ...]
    noaa:     {totalAlerts, alerts}
    markets:  {indexes, commodities, crypto, vix, ...}   # Yahoo Finance
    newsFeed: [{headline, source, type, timestamp}, ...]
  }

Usage
-----
    from crucix_adapter import CrucixAdapter

    adapter  = CrucixAdapter()
    snapshot = adapter.fetch()          # None if Crucix is not running
    if snapshot:
        macro_ctx = adapter.to_macro_context(snapshot)
        osint_df  = adapter.to_dataframe(snapshot)   # pd.DataFrame for RAG/pipeline
"""

from __future__ import annotations

import hashlib
import logging
import re as _re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:3117"
FETCH_TIMEOUT_S  = 10
MAX_ARTICLES     = 200

# Benzinga uses ", " as separator; we match it so prefilter split is consistent
_TICKER_SEP = ", "

# ── None-safe helpers ─────────────────────────────────────────────────────────

def _sd(val: Any) -> dict:
    """Return val if it's a non-None dict, else {}."""
    return val if isinstance(val, dict) else {}


def _sl(val: Any) -> list:
    """Return val if it's a non-None list, else []."""
    return val if isinstance(val, list) else []


# ── Public class ──────────────────────────────────────────────────────────────

class CrucixAdapter:
    """Fetches the latest Crucix intelligence snapshot and converts it for Chequita."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    # ── public ────────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def fetch(self) -> dict | None:
        """
        Fetch the synthesized intelligence snapshot from Crucix /api/data.
        Returns None if Crucix is offline or first sweep not done yet.
        """
        try:
            r = requests.get(f"{self.base_url}/api/data", timeout=FETCH_TIMEOUT_S)
            if r.status_code == 503:
                logger.warning("[CrucixAdapter] Crucix up but first sweep not complete.")
                return None
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                logger.warning("[CrucixAdapter] Unexpected response type: %s", type(data))
                return None
            ts = _sd(data.get("meta")).get("timestamp", "?")
            logger.info("[CrucixAdapter] Fetched Crucix snapshot (ts=%s)", ts)
            return data
        except requests.exceptions.ConnectionError:
            logger.warning("[CrucixAdapter] Crucix not running at %s — skipping.", self.base_url)
            return None
        except Exception as exc:
            logger.warning("[CrucixAdapter] Fetch error: %s", exc)
            return None

    def to_macro_context(self, snapshot: dict) -> dict:
        """
        Convert a Crucix V2 snapshot into a flat macro-context dict.
        All values from Crucix are treated as untrusted — None-safe throughout.
        """
        ctx: dict[str, Any] = {
            "source":    "crucix",
            "timestamp": _sd(snapshot.get("meta")).get("timestamp"),
        }

        # ── FRED macro indicators ──────────────────────────────────────────────
        _FRED_FIELD = {
            "DFF":               "fed_funds_rate",
            "DGS2":              "yield_2y",
            "DGS10":             "yield_10y",
            "DGS30":             "yield_30y",
            "T10Y2Y":            "yield_curve_10y2y",
            "T10Y3M":            "yield_curve_10y3m",
            "CPIAUCSL":          "cpi_all_items",
            "CPILFESL":          "core_cpi",
            "UNRATE":            "unemployment_rate",
            "PAYEMS":            "nonfarm_payrolls",
            "VIXCLS":            "vix_fred",
            "BAMLH0A0HYM2":      "high_yield_spread",
            "DCOILWTICO":        "wti_crude_fred",
            "GOLDAMGBD228NLBM":  "gold_price",
            "DTWEXBGS":          "usd_trade_weighted",
            "M2SL":              "m2_money_supply",
            "WALCL":             "fed_balance_sheet",
        }
        for ind in _sl(snapshot.get("fred")):
            if not isinstance(ind, dict):
                continue
            field = _FRED_FIELD.get(ind.get("id", ""))
            if field and ind.get("value") is not None:
                ctx[field] = {
                    "value":          ind["value"],
                    "date":           ind.get("date"),
                    "mom_change_pct": ind.get("momChangePct"),
                }

        # ── EIA energy prices ──────────────────────────────────────────────────
        energy = _sd(snapshot.get("energy"))
        if energy.get("wti") is not None:
            ctx["wti_crude"]   = {"value": energy["wti"]}
        if energy.get("brent") is not None:
            ctx["brent_crude"] = {"value": energy["brent"]}
        if energy.get("natgas") is not None:
            ctx["natural_gas"] = {"value": energy["natgas"]}
        ctx["energy_signals"] = _sl(energy.get("signals"))

        # ── Metals ────────────────────────────────────────────────────────────
        metals = _sd(snapshot.get("metals"))
        if metals.get("gold") is not None:
            ctx["gold_price_live"] = {
                "value":      metals["gold"],
                "change_pct": metals.get("goldChangePct"),
            }

        # ── Yahoo Finance live market data ─────────────────────────────────────
        markets = _sd(snapshot.get("markets"))
        vix = _sd(markets.get("vix"))
        if vix.get("value") is not None:
            ctx["vix_live"] = {"value": vix["value"], "change_pct": vix.get("changePct")}
        for idx in _sl(markets.get("indexes")):
            if not isinstance(idx, dict):
                continue
            sym = idx.get("symbol", "").replace("^", "").replace("=F", "")
            if sym and idx.get("price") is not None:
                ctx[f"market_{sym.lower()}"] = {
                    "price": idx["price"], "change_pct": idx.get("changePct")
                }
        for com in _sl(markets.get("commodities")):
            if not isinstance(com, dict):
                continue
            sym = com.get("symbol", "").replace("=F", "").replace("^", "")
            if sym and com.get("price") is not None:
                ctx[f"commodity_{sym.lower()}"] = {
                    "price": com["price"], "change_pct": com.get("changePct")
                }
        for cr in _sl(markets.get("crypto")):
            if not isinstance(cr, dict):
                continue
            sym = cr.get("symbol", "").replace("-USD", "")
            if sym and cr.get("price") is not None:
                ctx[f"crypto_{sym.lower()}"] = {
                    "price": cr["price"], "change_pct": cr.get("changePct")
                }

        # ── GSCPI (supply chain pressure) ─────────────────────────────────────
        gscpi = snapshot.get("gscpi")
        if gscpi is not None:
            ctx["supply_chain_pressure"] = gscpi

        # ── ACLED conflict ─────────────────────────────────────────────────────
        acled = _sd(snapshot.get("acled"))
        if acled:
            ctx["conflict_events_total"]     = acled.get("totalEvents", 0) or 0
            ctx["conflict_fatalities_total"] = acled.get("totalFatalities", 0) or 0
            ctx["conflict_by_type"]          = _sd(acled.get("byType"))

        # ── Treasury ──────────────────────────────────────────────────────────
        treasury = _sd(snapshot.get("treasury"))
        ctx["treasury_signals"] = _sl(treasury.get("signals"))

        # ── GDELT ─────────────────────────────────────────────────────────────
        gdelt = _sd(snapshot.get("gdelt"))
        ctx["gdelt_avg_tone"]  = gdelt.get("avgTone")
        ctx["gdelt_signals"]   = _sl(gdelt.get("signals"))

        # ── WHO ───────────────────────────────────────────────────────────────
        ctx["who_disease_alerts"] = _sl(snapshot.get("who"))[:5]

        # ── NOAA ──────────────────────────────────────────────────────────────
        noaa = _sd(snapshot.get("noaa"))
        ctx["noaa_severe_alerts"] = noaa.get("totalAlerts", 0) or 0

        # ── Cross-domain risk signals ──────────────────────────────────────────
        all_signals: list[str] = []
        for sig_list in [
            _sl(energy.get("signals")),
            _sl(treasury.get("signals")),
            _sl(gdelt.get("signals")),
            _sl(acled.get("signals")) if isinstance(acled, dict) else [],
        ]:
            for s in sig_list:
                if isinstance(s, str) and s.strip():
                    all_signals.append(s)
        ctx["cross_domain_risk_signals"] = all_signals

        return ctx

    def to_dataframe(self, snapshot: dict) -> pd.DataFrame:
        """
        Convert Crucix intelligence into a pd.DataFrame matching Chequita's
        article schema (Benzinga-compatible columns):
          url, title, date, tickers, composite_score, source

        The `tickers` column uses ", " as separator (same as Benzinga) so
        TickerScreener.prefilter() splits correctly.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows: list[dict] = []

        # ── newsFeed (RSS + GDELT + Telegram — pre-merged by Crucix) ──────────
        for item in _sl(snapshot.get("newsFeed"))[:80]:
            if not isinstance(item, dict):
                continue
            headline = item.get("headline") or ""
            if not headline:
                continue
            rows.append(_row(
                title=headline,
                date=_parse_date(item.get("timestamp"), today),
                source=f"Crucix/{item.get('source') or 'OSINT'}",
                tickers=_extract_tickers(headline),
                score=0.5,
            ))

        # ── GDELT articles ─────────────────────────────────────────────────────
        gdelt = _sd(snapshot.get("gdelt"))
        for art in _sl(gdelt.get("allArticles"))[:40]:
            if not isinstance(art, dict):
                continue
            title = art.get("title") or ""
            if not title:
                continue
            rows.append(_row(
                title=title,
                date=_parse_date(art.get("seendate"), today),
                source="Crucix/GDELT",
                tickers=_extract_tickers(title),
                score=_tone_to_impact(art.get("tone")),
            ))

        # ── Telegram OSINT urgent posts ────────────────────────────────────────
        tg = _sd(snapshot.get("tg"))
        for post in _sl(tg.get("urgent"))[:30]:
            if not isinstance(post, dict):
                continue
            text = post.get("text") or ""
            if not text or _is_cyrillic(text):
                continue
            rows.append(_row(
                title=text[:200],
                date=_parse_date(post.get("date"), today),
                source=f"Crucix/Telegram/{post.get('channel') or 'OSINT'}",
                tickers=_extract_tickers(text),
                score=_urgency_to_impact(_sl(post.get("urgentFlags"))),
            ))

        # ── WHO disease alerts ─────────────────────────────────────────────────
        for alert in _sl(snapshot.get("who"))[:10]:
            if not isinstance(alert, dict):
                continue
            title = alert.get("title") or ""
            if not title:
                continue
            rows.append(_row(
                title=title,
                date=_parse_date(alert.get("date"), today),
                source="Crucix/WHO",
                tickers=[],
                score=0.6,
            ))

        # ── ACLED deadliest events ─────────────────────────────────────────────
        acled = _sd(snapshot.get("acled"))
        for evt in _sl(acled.get("deadliestEvents"))[:15]:
            if not isinstance(evt, dict):
                continue
            fatalities = evt.get("fatalities") or 0
            title = (
                f"ACLED: {evt.get('event_type') or 'conflict'} in "
                f"{evt.get('country') or '?'} ({fatalities} fatalities)"
            )
            rows.append(_row(
                title=title,
                date=_parse_date(evt.get("event_date"), today),
                source="Crucix/ACLED",
                tickers=[],
                score=min(1.0, 0.5 + fatalities / 200),
            ))

        if not rows:
            return pd.DataFrame(columns=["url", "title", "date", "tickers",
                                         "composite_score", "source"])

        df = pd.DataFrame(rows[:MAX_ARTICLES])
        df = df.drop_duplicates(subset=["url"])
        return df.reset_index(drop=True)

    def to_summary_text(self, snapshot: dict) -> str:
        """Short plaintext summary for injecting into LLM prompts."""
        ctx = self.to_macro_context(snapshot)
        lines: list[str] = ["=== Crucix OSINT Intelligence Snapshot ==="]

        def _fmt(key: str, label: str) -> None:
            v = ctx.get(key)
            val = v.get("value") if isinstance(v, dict) else v
            if val is None:
                return
            try:
                lines.append(f"{label}: {float(val):.2f}")
            except (TypeError, ValueError):
                lines.append(f"{label}: {val}")

        _fmt("vix_fred",          "VIX (FRED)")
        _fmt("vix_live",          "VIX (live)")
        _fmt("yield_curve_10y2y", "Yield Curve 10Y-2Y")
        _fmt("wti_crude",         "WTI Crude $")
        _fmt("fed_funds_rate",    "Fed Funds Rate")
        _fmt("high_yield_spread", "HY Credit Spread")

        gscpi = ctx.get("supply_chain_pressure")
        if gscpi is not None:
            try:
                lines.append(f"GSCPI (supply chain): {float(gscpi):.2f}")
            except (TypeError, ValueError):
                lines.append(f"GSCPI (supply chain): {gscpi}")

        total_evts = ctx.get("conflict_events_total", 0) or 0
        if total_evts:
            lines.append(
                f"Armed Conflict (ACLED): {total_evts} events, "
                f"{ctx.get('conflict_fatalities_total', 0) or 0} fatalities"
            )

        signals = ctx.get("cross_domain_risk_signals") or []
        if signals:
            lines.append("\nKey risk signals:")
            for s in signals[:8]:
                lines.append(f"  \u2022 {s}")

        for alert in (ctx.get("who_disease_alerts") or [])[:3]:
            if isinstance(alert, dict):
                lines.append(f"  [WHO] {(alert.get('title') or '')[:120]}")

        return "\n".join(lines)


# ── module-level helpers ──────────────────────────────────────────────────────

def _row(title: str, date: str, source: str, tickers: list[str], score: float) -> dict:
    """Build a Benzinga-compatible article row with a stable dedup URL."""
    uid = hashlib.md5(f"{source}|{title}".encode()).hexdigest()
    return {
        "url":             f"crucix://{uid}",
        "title":           title,
        "date":            date,
        # Match Benzinga's ", " separator so prefilter's str.split(",") works
        "tickers":         _TICKER_SEP.join(tickers),
        "composite_score": round(float(score), 4),
        "source":          source,
    }


def _tone_to_impact(tone: Any) -> float:
    """GDELT article tone (negative = bad news) → 0–1 impact score."""
    try:
        return min(1.0, max(0.1, 0.5 + abs(float(tone)) / 20))
    except (TypeError, ValueError):
        return 0.4


def _urgency_to_impact(flags: list) -> float:
    """Telegram urgentFlags list length → 0–1 impact score."""
    return min(1.0, 0.5 + len(flags) * 0.15)


_DATE_RE = _re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_date(raw: Any, fallback: str) -> str:
    """
    Extract a YYYY-MM-DD date from raw.  Handles:
      - ISO strings: "2026-04-05T12:00:00Z"
      - Date-only strings: "2026-04-05"
      - Unix timestamps (int/float seconds): 1743916800
      - Any string containing a date pattern
    Falls back to `fallback` on failure.
    """
    if raw is None:
        return fallback
    # Unix timestamp (numeric)
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return fallback
    # String — extract first YYYY-MM-DD pattern
    s = str(raw)
    m = _DATE_RE.search(s)
    if m:
        return m.group(1)
    return fallback


_CYRILLIC = set(range(0x0400, 0x0500))


def _is_cyrillic(text: str) -> bool:
    return sum(1 for c in text[:80] if ord(c) in _CYRILLIC) > 5


_TICKER_RE = _re.compile(r"\$([A-Z]{1,5})\b")


def _extract_tickers(text: str) -> list[str]:
    return list(dict.fromkeys(_TICKER_RE.findall(text or "")))[:5]
