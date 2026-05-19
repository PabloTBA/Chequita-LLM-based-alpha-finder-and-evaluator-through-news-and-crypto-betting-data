"""
Sector Diversity Enforcement
=============================
Tiered per-sector quotas that prevent large-caps from crowding out small-caps.

Features
--------
- Static GICS sector map for S&P 500 constituents
- Dynamic yfinance resolution for unknown tickers (sector + market cap in one call)
- Three-tier quotas: mega (1), large (2), small_mid (3) per sector
- Alias deduplication (e.g. GOOG → GOOGL)
"""

from __future__ import annotations


# ── static sector map (S&P 500 constituents) ─────────────────────────────────

_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "AMD": "Technology",
    "QCOM": "Technology", "AMAT": "Technology", "LRCX": "Technology",
    "KLAC": "Technology", "ADI": "Technology", "MU": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "TXN": "Technology",
    "INTU": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "NOW": "Technology", "PANW": "Technology", "CRWD": "Technology",
    "SNOW": "Technology", "PLTR": "Technology", "ACN": "Technology",
    # Communication
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication",
    "NFLX": "Communication", "T": "Communication", "DIS": "Communication",
    "BIDU": "Communication",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
    "UBER": "Consumer Discretionary", "RBLX": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary", "LCID": "Consumer Discretionary",
    "SHOP": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "MDLZ": "Consumer Staples",
    "MO": "Consumer Staples", "PM": "Consumer Staples", "CL": "Consumer Staples",
    # Healthcare
    "LLY": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "AMGN": "Healthcare", "ISRG": "Healthcare",
    "SYK": "Healthcare", "GILD": "Healthcare", "VRTX": "Healthcare",
    "BSX": "Healthcare", "REGN": "Healthcare", "MDT": "Healthcare",
    "EW": "Healthcare", "ZTS": "Healthcare", "BDX": "Healthcare",
    "HUM": "Healthcare", "HCA": "Healthcare", "JNJ": "Healthcare",
    # Financials
    "JPM": "Financials", "BRK.B": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "SPGI": "Financials",
    "AXP": "Financials", "C": "Financials", "SCHW": "Financials",
    "CB": "Financials", "MMC": "Financials", "AON": "Financials",
    "MCO": "Financials", "ICE": "Financials", "CME": "Financials",
    "USB": "Financials", "AIG": "Financials", "FI": "Financials",
    "PYPL": "Financials", "SQ": "Financials", "SOFI": "Financials",
    "COIN": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "FCX": "Energy",
    # Industrials
    "GE": "Industrials", "HON": "Industrials", "CAT": "Industrials",
    "ETN": "Industrials", "RTX": "Industrials", "NOC": "Industrials",
    "DE": "Industrials", "ITW": "Industrials", "EMR": "Industrials",
    "NSC": "Industrials", "GD": "Industrials", "BA": "Industrials",
    "MMM": "Industrials", "PH": "Industrials", "ROP": "Industrials",
    # Materials
    "LIN": "Materials", "SHW": "Materials",
    # Real Estate
    "PLD": "Real Estate", "PSA": "Real Estate",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities",
    # Speculative / Other
    "NIO": "Speculative",
}

# ── tiered diversity quotas ──────────────────────────────────────────────────

_MEGA_CAP_THRESHOLD       = 200_000_000_000   # $200B+
_LARGE_CAP_THRESHOLD      =  10_000_000_000   # $10B+
_MEGA_CAP_PER_SECTOR      = 1
_LARGE_CAP_PER_SECTOR     = 2
_SMALL_MID_CAP_PER_SECTOR = 3

# ── ticker alias deduplication ───────────────────────────────────────────────

_TICKER_ALIASES: dict[str, str] = {
    "GOOG": "GOOGL",   # class-C and class-A are >0.99 correlated — keep GOOGL
}


def deduplicate_aliases(tickers: list[str]) -> list[str]:
    """Drop any ticker whose preferred alias is already in the list."""
    ticker_set = set(tickers)
    return [t for t in tickers
            if not (_TICKER_ALIASES.get(t) in ticker_set)]


# ── public helpers ───────────────────────────────────────────────────────────

def resolve_ticker_info(
    tickers: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    """
    Single yfinance batch call that extracts both sector and market cap
    for tickers not already in the static sector map.

    Returns (extended_sector_map, market_caps).
    """
    sector_map = dict(_SECTOR_MAP)
    market_caps: dict[str, float] = {}
    unknown = [t for t in tickers if t not in sector_map]

    if not unknown:
        return sector_map, market_caps

    for i in range(0, len(unknown), 50):
        batch = unknown[i : i + 50]
        try:
            ticker_objects = __import__("yfinance").Tickers(" ".join(batch))
            for t in batch:
                try:
                    obj = ticker_objects.tickers.get(t)
                    if obj and hasattr(obj, "info"):
                        d = obj.info or {}
                        sector = d.get("sector") or d.get("industry") or ""
                        mkt_cap = d.get("marketCap") or 0
                        if sector:
                            sector_map[t] = str(sector)
                        market_caps[t] = float(mkt_cap)
                except Exception:
                    pass
            __import__("time").sleep(0.3)
        except Exception as e:
            print(f"  [TickerInfo] Batch lookup failed: {e}")

    return sector_map, market_caps


def classify_market_cap(market_cap: float) -> str:
    """Bucket a market-cap figure into a tier label."""
    if market_cap >= _MEGA_CAP_THRESHOLD:
        return "mega"
    if market_cap >= _LARGE_CAP_THRESHOLD:
        return "large"
    return "small_mid"


def enforce_sector_diversity(
    tickers: list[str],
    *,
    market_caps: dict[str, float] | None = None,
    sector_map: dict[str, str] | None = None,
) -> list[str]:
    """
    Cap tickers per GICS sector using tiered quotas.

    Tiers (per sector):
        mega      (>$200B):  1 slot
        large     ($10B–$200B): 2 slots
        small_mid (<$10B):  3 slots

    Unknown market cap → classified as small_mid (most permissive).
    Unknown sector → passed through unhindered.
    """
    if sector_map is None or market_caps is None:
        sector_map, live_caps = resolve_ticker_info(tickers)
        if market_caps is None:
            market_caps = live_caps

    caps = market_caps or {}

    mega_counts:      dict[str, int] = {}
    large_counts:     dict[str, int] = {}
    small_mid_counts: dict[str, int] = {}

    result: list[str] = []
    for ticker in tickers:
        sector = sector_map.get(ticker)
        if sector is None:
            result.append(ticker)   # unknown — let it through
            continue

        tier = classify_market_cap(caps.get(ticker, 0))

        if tier == "mega":
            quota = _MEGA_CAP_PER_SECTOR
            counter = mega_counts
        elif tier == "large":
            quota = _LARGE_CAP_PER_SECTOR
            counter = large_counts
        else:
            quota = _SMALL_MID_CAP_PER_SECTOR
            counter = small_mid_counts

        current = counter.get(sector, 0)
        if current < quota:
            result.append(ticker)
            counter[sector] = current + 1
        else:
            print(
                f"  [Diversity] {ticker} ({tier}) dropped — "
                f"{sector} sector already has {quota} {tier} ticker(s)"
            )

    return result
