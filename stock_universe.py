"""
StockUniverse — Dynamic Stock Discovery Engine
===============================================
Builds a broad, dynamically-discovered stock universe that goes far beyond
the S&P 500.  Discovers niche, obscure, and newly-booming stocks through
multiple free data sources: index constituents, thematic ETFs, Finviz
screeners, corporate venture investments, and recent IPOs.

Discovery Channels
------------------
1. Core indices      — S&P 500, Nasdaq 100, Russell 2000, S&P 400
2. Sector ETFs       — All holdings from 11 GICS sector SPDR ETFs
3. Thematic ETFs     — AI, EV/Lithium, Clean Energy, Crypto/Blockchain,
                       Biotech, Cybersecurity, Cloud, Semiconductors
4. Finviz screeners  — Top gainers, unusual volume, new highs, oversold
5. Corporate ventures — S&P 500 strategic investments in smaller firms
6. Recent IPOs       — Newly listed stocks (past 12–24 months)

Public Interface
----------------
    builder = StockUniverseBuilder(cache_dir="data/cache")
    universe = builder.build(mode="broad")         # ~2000+ tickers
    universe = builder.build(mode="aggressive")    # ~4000+ tickers (includes micro-caps)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Discovery weights — higher weight = higher priority for news coverage
# ---------------------------------------------------------------------------
WEIGHT_SP500       = 1.0
WEIGHT_NASDAQ100   = 0.9
WEIGHT_RUSSELL2000 = 0.6
WEIGHT_SP400       = 0.7
WEIGHT_SECTOR_ETF  = 0.8
WEIGHT_THEME_ETF   = 0.7
WEIGHT_SCREENER    = 0.5
WEIGHT_CORP_VENTURE = 0.6
WEIGHT_IPO          = 0.4

# Theme ETFs used to discover stocks in hot sectors
THEME_ETFS: dict[str, list[str]] = {
    "AI & Robotics":          ["BOTZ", "AIQ", "ROBT", "ARKQ"],
    "EV & Battery Materials": ["LIT", "DRIV", "KARS", "BATT"],
    "Clean Energy":           ["ICLN", "TAN", "FAN", "QCLN", "PBW"],
    "Crypto & Blockchain":    ["BLOK", "BITQ", "LEGR", "BKCH"],
    "Biotech & Genomics":     ["XBI", "IBB", "ARKG", "FBT", "BBH"],
    "Cybersecurity":          ["HACK", "CIBR", "BUG", "IHAK"],
    "Cloud Computing":        ["CLOU", "SKYY", "WCLD", "FCLD"],
    "Semiconductors":         ["SMH", "SOXX", "PSI", "XSD"],
    "Space Economy":          ["ARKX", "UFO", "ROKT"],
    "Fintech":                ["FINX", "ARKF", "IPAY", "TPAY"],
    "Infrastructure":         ["PAVE", "IFRA", "NFRA"],
    "Water & Resources":      ["PHO", "PIO", "FIW", "CGW"],
    "Agriculture":            ["MOO", "VEGI", "CROP"],
    "Gaming & Metaverse":     ["ESPO", "HERO", "METV", "GAMR"],
    "Defense":                ["ITA", "PPA", "XAR", "DFEN"],
}

# Sector SPDR ETFs — one per GICS sector
SECTOR_ETFS: dict[str, str] = {
    "Technology":                 "XLK",
    "Financials":                 "XLF",
    "Healthcare":                 "XLV",
    "Consumer Discretionary":     "XLY",
    "Communication":              "XLC",
    "Industrials":                "XLI",
    "Energy":                     "XLE",
    "Consumer Staples":           "XLP",
    "Utilities":                  "XLU",
    "Real Estate":                "XLRE",
    "Materials":                  "XLB",
    "Aerospace & Defense":        "XAR",
    "Biotech":                    "XBI",
    "Oil & Gas Exploration":      "XOP",
    "Regional Banks":             "KRE",
    "Homebuilders":               "XHB",
    "Semiconductors":             "SMH",
    "Software":                   "IGV",
    "Retail":                     "XRT",
}

# Corporate Venture Capital arms of major S&P 500 companies and their
# known strategic investments in smaller/public companies.  Updated
# periodically from public filings and news.
CORPORATE_VENTURES: dict[str, list[str]] = {
    "GOOGL": ["RXRX", "DMTK", "IONQ", "API", "ASAN", "GTLB", "CFLT", "S", "PLTR", "SOFI"],
    "MSFT":  ["OPEN", "RXRX", "IONQ", "API", "CFLT", "GTLB", "S", "PATH"],
    "AMZN":  ["IONQ", "RIVN", "ASTR", "PL", "S", "CFLT"],
    "NVDA":  ["RXRX", "IONQ", "SOUN", "AI", "APLD", "NBIS", "SERV"],
    "META":  ["IONQ", "RXRX", "PATH"],
    "AAPL":  ["IONQ", "PATH"],
    "INTC":  ["IONQ", "RXRX", "AI", "PATH", "AUR", "LAZR", "INDI"],
    "TSLA":  [],
    "QCOM":  ["IONQ", "RXRX", "PATH", "CRNC", "INDI"],
    "CRM":   ["IONQ", "SNOW", "PATH", "ASAN", "GTLB"],
    "ORCL":  ["IONQ", "SNOW", "GTLB"],
    "JPM":   ["IONQ", "RXRX", "PATH", "SOFI", "BILL"],
    "GS":    ["IONQ", "SOFI", "RXRX"],
    "BAC":   ["IONQ", "SOFI"],
    "V":     ["IONQ", "SOFI", "PATH", "BILL"],
    "MA":    ["IONQ", "SOFI", "PATH", "BILL"],
    "IBM":   ["IONQ", "RXRX", "PATH"],
    "CVX":   ["PLUG", "FCEL", "BE", "STEM"],
    "XOM":   ["PLUG", "STEM", "BE"],
    "GM":    ["RIVN", "LCID", "QS", "SFTBY", "INDI", "LAZR", "AUR", "BLBD"],
    "F":     ["RIVN", "LCID", "QS", "INDI"],
    "AMGN":  ["RXRX", "BEAM", "NTLA", "CRSP", "DNA", "ABCL", "PRME"],
    "LLY":   ["RXRX", "BEAM", "NTLA", "CRSP", "DNA"],
    "PFE":   ["RXRX", "BEAM", "NTLA", "CRSP", "DNA", "BNTX", "MRNA"],
    "MRK":   ["RXRX", "BEAM", "NTLA", "CRSP"],
    "ABBV":  ["RXRX", "BEAM", "NTLA", "CRSP"],
    "JNJ":   ["RXRX", "BEAM", "NTLA", "CRSP", "DNA"],
    "HD":    ["IONQ"],
    "WMT":   ["IONQ", "SOFI", "PATH"],
    "BA":    ["IONQ", "SPIR", "ASTR", "PL"],
    "RTX":   ["IONQ", "SPIR"],
    "NOC":   ["IONQ", "SPIR"],
    "LMT":   ["IONQ", "SPIR", "PL"],
}

# Proxy ticker lists for sector and thematic ETFs (when yfinance holdings aren't available).
# These are major holdings of each ETF, refreshed periodically.
SECTOR_ETF_PROXIES: dict[str, list[str]] = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN",
            "IBM", "QCOM", "TXN", "INTU", "NOW", "AMAT", "LRCX", "KLAC", "ADI", "MU",
            "PANW", "CRWD", "FTNT", "SNOW", "PLTR", "APH", "MSI", "ROP", "FICO", "IT"],
    "XLF": ["BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP",
            "C", "SCHW", "BLK", "FI", "MMC", "CB", "AON", "MCO", "ICE", "CME",
            "USB", "PNC", "AIG", "MET", "PRU", "TRV", "BK", "COF", "DFS", "MTB"],
    "XLV": ["LLY", "UNH", "ABBV", "JNJ", "MRK", "TMO", "ABT", "DHR", "AMGN", "PFE",
            "ISRG", "SYK", "BSX", "GILD", "VRTX", "REGN", "ZTS", "EW", "BDX", "MDT",
            "HUM", "HCA", "CI", "CNC", "CVS", "BIIB", "IDXX", "DXCM", "IQV", "A"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "LOW", "TJX", "SBUX", "NKE", "CMG",
            "ORLY", "AZO", "MAR", "ABNB", "GM", "F", "LULU", "ROST", "DHI", "LEN",
            "ULTA", "DRI", "YUM", "DPZ", "WYNN", "LVS", "MGM", "BBY", "EBAY", "RCL"],
    "XLC": ["META", "GOOGL", "NFLX", "DIS", "T", "TMUS", "CHTR", "CMCSA", "EA", "TTWO",
            "OMC", "IPG", "LYV", "NWSA", "FOXA", "PARA", "WBD", "MTCH", "RBLX", "SNAP"],
    "XLI": ["GE", "CAT", "RTX", "UNP", "UBER", "HON", "BA", "ETN", "DE", "ADP",
            "LMT", "ITW", "GD", "NOC", "EMR", "CSX", "NSC", "FDX", "UPS", "WM",
            "PH", "CPRT", "GWW", "PCAR", "URI", "FAST", "AME", "ODFL", "DAL", "JCI"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "WMB", "OKE", "KMI",
            "HES", "BKR", "FANG", "DVN", "HAL", "OXY", "CTRA", "EQT", "TPL", "APA"],
    "XLP": ["PG", "COST", "WMT", "KO", "PEP", "PM", "MDLZ", "CL", "MO", "TGT",
            "KMB", "STZ", "KHC", "GIS", "KVUE", "SYY", "MNST", "KR", "HSY", "ADM"],
    "XLU": ["NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "PEG", "ED", "XEL",
            "EIX", "WEC", "AWK", "ES", "PPL", "FE", "DTE", "AEE", "CNP", "CMS"],
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG", "O", "CCI", "PSA", "DLR", "CSGP",
             "EXR", "VICI", "AVB", "EQR", "IRM", "SBAC", "WY", "INVH", "ARE", "UDR"],
    "XLB": ["LIN", "SHW", "FCX", "ECL", "APD", "NEM", "CTVA", "NUE", "DOW", "DD",
            "PPG", "VMC", "MLM", "FMC", "EMN", "ALB", "IFF", "CE", "AVY", "IP"],
}
THEME_PROXIES: dict[str, list[str]] = {
    "BOTZ": ["NVDA", "ISRG", "DT", "PATH", "AI", "SOUN", "BBAI", "OMCL", "ANSS", "CGNX",
             "DDD", "ROK", "AME", "EMR", "ZBRA", "TER", "AMBA", "PRLB", "SSYS", "FARO"],
    "LIT":  ["ALB", "SQM", "LAC", "ALTM", "SGML", "PLL", "SLI", "IONR", "LTHM",
             "ENS", "QS", "MVST", "SLDP", "AMLI", "LITM", "ABAT", "NAK", "LGO",
             "TSLA", "BYD", "NIO", "XPEV", "LI", "RIVN", "LCID", "GM", "F", "TM"],
    "XBI":  ["AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "BNTX", "NVAX", "RXRX",
             "BEAM", "NTLA", "CRSP", "DNA", "ABCL", "PRME", "ALLO", "EDIT", "FATE",
             "ARWR", "ALNY", "IONS", "SRPT", "BBIO", "FOLD", "SAGE", "ACAD", "KRTX",
             "ITCI", "AXSM", "PCRX", "HRTX", "GERN", "AGIO", "PTCT", "MIRM", "RCKT"],
    "HACK": ["PANW", "CRWD", "ZS", "FTNT", "S", "CHKP", "CYBR", "TENB", "VRNS", "QLYS",
             "RPD", "NET", "AKAM", "BAH", "LDOS", "SAIC", "CACI", "PSN"],
    "SMH":  ["NVDA", "AVGO", "AMD", "QCOM", "INTC", "MU", "AMAT", "LRCX", "KLAC", "ADI",
             "TXN", "NXPI", "MRVL", "ON", "MPWR", "STM", "UMC", "TSM", "ASML", "ARM",
             "ALAB", "SITM", "CRDO", "PI", "SYNA", "SLAB", "AMBA", "SIMO", "NVTS"],
    "FINX": ["PYPL", "SQ", "SOFI", "BILL", "AFRM", "FOUR", "MQ", "PAYO", "STNE", "PAGS",
             "DLO", "EVTC", "FLYW", "GPN", "FIS", "FISV", "ADYEN", "WISE", "NU", "HOOD"],
    "TAN":  ["ENPH", "SEDG", "FSLR", "RUN", "SPWR", "NOVA", "NXT", "MAXN", "CSIQ", "JKS",
             "DQ", "ARRY", "SHLS", "FCEL", "PLUG", "BE", "STEM", "OPAL", "CWEN", "AY"],
    "BLOK": ["COIN", "MARA", "RIOT", "CLSK", "HUT", "BTBT", "IREN", "CIFR", "WULF",
             "MSTR", "SQ", "PYPL", "HOOD", "CME", "ICE", "NDAQ", "SI", "CUBI"],
}

# Small-cap stocks in hot sectors — manually curated list of emerging names
# that are often missed by index-based discovery.  Updated periodically.
EMERGING_STOCKS: list[str] = [
    # Semiconductors & Sensors
    "ALMU", "INDI", "LAZR", "OUST", "MVIS", "AEVA", "CPTN", "QUIK", "NVTS",
    "AOSL", "DIOD", "POWI", "SIMO", "CRDO", "SITM", "PI", "SYNA", "SLAB",
    "AMBA", "MTSI", "FORM", "ACLS", "UCTT", "COHU", "ICHR", "VECO", "ACMR",
    # Lithium & Battery Materials
    "LAC", "ALTM", "SGML", "PLL", "SLI", "IONR", "LTHM", "MVST", "SLDP",
    "AMLI", "LITM", "ABAT", "LGO", "NAK", "TMC", "MP", "UEC", "UUUU", "LEU",
    # AI & Quantum Computing
    "IONQ", "RGTI", "QBTS", "QUBT", "QMCO", "BBAI", "AI", "SOUN", "APLD",
    "NBIS", "BMR", "VRT", "SMCI", "DELL", "HPE", "NTAP",
    # EV & Autonomous
    "RIVN", "LCID", "QS", "INDI", "LAZR", "AUR", "OUST", "AEVA", "BLBD",
    "BLNK", "CHPT", "EVGO", "WBX", "FSR", "RIDE", "NKC", "WKHS", "GOEV",
    # Biotech & Genomics (small-cap)
    "RXRX", "BEAM", "NTLA", "CRSP", "DNA", "ABCL", "PRME", "ALLO", "EDIT",
    "FATE", "BBIO", "FOLD", "SAGE", "ACAD", "KRTX", "AXSM", "GERN", "AGIO",
    "PTCT", "MIRM", "RCKT", "ALEC", "GRTS", "ADPT", "NRIX", "KURA", "IDYA",
    # Clean Energy & Hydrogen
    "PLUG", "FCEL", "BE", "STEM", "OPAL", "CWEN", "AY", "NEP", "BEPC",
    "ENPH", "SEDG", "FSLR", "RUN", "NOVA", "NXT", "ARRY", "SHLS", "MAXN",
    # Space Economy
    "ASTR", "SPIR", "PL", "RDW", "BKSY", "MNTS", "RKLB", "LLAP", "SIDU",
    # Crypto & Blockchain
    "MARA", "RIOT", "CLSK", "HUT", "BTBT", "IREN", "CIFR", "WULF",
    "COIN", "MSTR", "HOOD", "CUBI", "SI", "DBS", "NRXP",
    # Fintech (small-cap)
    "SOFI", "BILL", "AFRM", "FOUR", "MQ", "PAYO", "STNE", "PAGS", "DLO",
    "EVTC", "FLYW", "NU", "UPST", "LC", "OPFI", "PRTH", "GDOT",
    # Cybersecurity (small-cap)
    "TENB", "VRNS", "QLYS", "RPD", "S", "NET", "CYBR", "CHKP",
    "ATEN", "CYTK", "ZD", "OSPN", "CRWD", "ZS", "FTNT",
]


class StockUniverseBuilder:
    """Dynamically discovers stocks from multiple free data sources."""

    def __init__(self, cache_dir: str = "data/cache", cache_ttl_hours: int = 24):
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl_hours
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, mode: str = "broad") -> list[str]:
        """
        Build a dynamic stock universe.

        Args:
            mode: "broad" (default) — ~2000+ tickers across all channels
                  "aggressive" — ~4000+ including micro-caps and recent IPOs

        Returns:
            Deduplicated list of ticker symbols, sorted by discovery weight.
        """
        cache_key = f"universe_{mode}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        tickers: dict[str, float] = {}

        # Channel 1: Core indices (always included)
        self._add_weighted(tickers, self._get_sp500(), WEIGHT_SP500)
        self._add_weighted(tickers, self._get_nasdaq100(), WEIGHT_NASDAQ100)
        self._add_weighted(tickers, self._get_russell2000(), WEIGHT_RUSSELL2000)
        self._add_weighted(tickers, self._get_sp400(), WEIGHT_SP400)
        print(f"  [Universe] After core indices: {len(tickers)} tickers")

        # Channel 2: All sector ETF holdings
        sector_tickers = self._get_sector_etf_holdings()
        self._add_weighted(tickers, sector_tickers, WEIGHT_SECTOR_ETF)
        print(f"  [Universe] After sector ETFs: {len(tickers)} tickers")

        # Channel 3: Thematic ETF holdings (hot sectors)
        theme_tickers = self._get_theme_holdings()
        self._add_weighted(tickers, theme_tickers, WEIGHT_THEME_ETF)
        print(f"  [Universe] After theme ETFs: {len(tickers)} tickers")

        # Channel 4: Corporate venture investments
        corp_ventures = self._get_corporate_ventures_list()
        self._add_weighted(tickers, corp_ventures, WEIGHT_CORP_VENTURE)
        print(f"  [Universe] After corp ventures: {len(tickers)} tickers")

        # Channel 5: Emerging / niche stocks (manually curated)
        self._add_weighted(tickers, EMERGING_STOCKS, 0.4)
        print(f"  [Universe] After emerging stocks: {len(tickers)} tickers")

        # Channel 6: Finviz screeners (market movers)
        try:
            screener_tickers = self._get_finviz_screeners()
            self._add_weighted(tickers, screener_tickers, WEIGHT_SCREENER)
            print(f"  [Universe] After Finviz screeners: {len(tickers)} tickers")
        except Exception as e:
            print(f"  [Universe] Finviz screeners skipped ({e})")

        # Channel 7: Recent IPOs (aggressive mode only)
        if mode == "aggressive":
            try:
                ipo_tickers = self._get_recent_ipos()
                self._add_weighted(tickers, ipo_tickers, WEIGHT_IPO)
                print(f"  [Universe] After recent IPOs: {len(tickers)} tickers")
            except Exception as e:
                print(f"  [Universe] Recent IPOs skipped ({e})")

        # Sort by weight (descending), then alphabetically
        sorted_tickers = sorted(tickers, key=lambda t: (-tickers[t], t))
        self._save_cache(cache_key, sorted_tickers)
        return sorted_tickers

    def get_sector_map(self, tickers: list[str]) -> dict[str, str]:
        """
        Build a ticker → GICS sector map for a given universe.
        Uses yfinance to fetch sector/industry for each ticker.
        Falls back to keyword-based sector inference.
        """
        cache_key = "sector_map"
        cached = self._load_cache(cache_key)
        if cached is not None:
            result = {t: cached[t] for t in tickers if t in cached}
            missing = [t for t in tickers if t not in cached]
            if not missing:
                return result
            tickers = missing

        result = {}
        for i in range(0, len(tickers), 50):
            batch = tickers[i:i + 50]
            try:
                ticker_objects = yf.Tickers(" ".join(batch))
                for t in batch:
                    try:
                        info = ticker_objects.tickers.get(t)
                        if info and hasattr(info, "info"):
                            d = info.info or {}
                            sector = d.get("sector") or d.get("industry") or ""
                            if sector:
                                result[t] = str(sector)
                    except Exception:
                        pass
                time.sleep(0.3)
            except Exception as e:
                print(f"  [SectorMap] Batch error: {e}")

        self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Index scraping (Wikipedia)
    # ------------------------------------------------------------------

    def _get_sp500(self) -> list[str]:
        """Scrape S&P 500 constituents from Wikipedia."""
        return self._scrape_wikipedia_table(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            col=0,
        )

    def _get_nasdaq100(self) -> list[str]:
        """Scrape Nasdaq 100 constituents from Wikipedia."""
        return self._scrape_wikipedia_table(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            col=0,
            table_idx=5,  # Table 5 is the components list with 'Ticker' column
        )

    def _get_russell2000(self) -> list[str]:
        """Approximate Russell 2000 via IWM ETF + curated small-cap list."""
        tickers = set()
        tickers.update(self._get_etf_top_holdings("IWM", 500))
        tickers.update(self._curated_russell_sample())
        return list(tickers)

    def _get_sp400(self) -> list[str]:
        """Scrape S&P 400 (MidCap) from Wikipedia."""
        return self._scrape_wikipedia_table(
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            col=0,
        )

    def _scrape_wikipedia_table(self, url: str, col: int = 0, table_idx: int = 0) -> list[str]:
        """Scrape ticker symbols from a Wikipedia table."""
        try:
            from io import StringIO
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            if table_idx < len(tables):
                df = tables[table_idx]
                if col < len(df.columns):
                    symbols = df.iloc[:, col].dropna().astype(str).tolist()
                    # Clean tickers: remove exchange suffixes, keep valid symbols
                    cleaned = []
                    for s in symbols:
                        s = s.strip().replace("\xa0", " ").replace("\n", " ")
                        s = s.split(" ")[0].split("\xa0")[0]
                        # Preserve dots for tickers like BRK.B
                        if s and len(s) <= 6 and s.replace("-", "").replace(".", "").isalpha():
                            cleaned.append(s.upper())
                    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order
        except Exception as e:
            print(f"  [Wikipedia] Failed to scrape {url}: {e}")
        return []

    # ------------------------------------------------------------------
    # ETF holdings
    # ------------------------------------------------------------------

    def _get_sector_etf_holdings(self) -> list[str]:
        """Get all holdings from GICS sector SPDR ETFs."""
        tickers: set[str] = set()
        for sector, etf in SECTOR_ETFS.items():
            holdings = self._get_etf_top_holdings(etf, 100)
            tickers.update(holdings)
            print(f"  [ETFs] {sector} ({etf}): {len(holdings)} holdings")
        return list(tickers)

    def _get_theme_holdings(self) -> list[str]:
        """Get holdings from thematic ETFs (hot sectors)."""
        tickers: set[str] = set()
        for theme, etfs in THEME_ETFS.items():
            theme_set: set[str] = set()
            for etf in etfs:
                # Try yfinance first, fall back to proxy list
                holdings = self._get_etf_top_holdings(etf, 80)
                if len(holdings) < 10:
                    holdings = THEME_PROXIES.get(etf, [])
                theme_set.update(holdings)
            tickers.update(theme_set)
            print(f"  [Themes] {theme}: {len(theme_set)} holdings across {len(etfs)} ETFs")
        return list(tickers)

    def _get_etf_top_holdings(self, etf_symbol: str, limit: int = 200) -> list[str]:
        """
        Fetch top holdings for an ETF via yfinance.
        Falls back to cached proxy lists on failure.
        """
        cache_key = f"etf_{etf_symbol}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached[:limit]

        for attempt in range(3):
            try:
                etf = yf.Ticker(etf_symbol)
                holdings = getattr(etf, "holdings", None)
                if holdings is not None:
                    df = holdings
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        tickers = df.index.tolist()
                        self._save_cache(cache_key, tickers)
                        return tickers[:limit]

                # Fallback via fund info
                info = etf.info or {}
                top_holdings = info.get("holdings", []) or info.get("topHoldings", []) or []
                if top_holdings:
                    tickers = [h.get("symbol", "") for h in top_holdings if h.get("symbol")]
                    tickers = [t for t in tickers if t]
                    if tickers:
                        self._save_cache(cache_key, tickers)
                        return tickers[:limit]
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                continue
            break

        # Ultimate fallback: proxy list
        fallback = SECTOR_ETF_PROXIES.get(etf_symbol, []) or THEME_PROXIES.get(etf_symbol, [])
        return fallback[:limit]

    # ------------------------------------------------------------------
    # Corporate ventures
    # ------------------------------------------------------------------

    def _get_corporate_ventures_list(self) -> list[str]:
        """Return all known corporate venture investments by S&P 500 firms."""
        tickers: set[str] = set()
        for parent, ventures in CORPORATE_VENTURES.items():
            tickers.update(ventures)
        return list(tickers)

    # ------------------------------------------------------------------
    # Finviz screeners (market movers)
    # ------------------------------------------------------------------

    def _get_finviz_screeners(self) -> list[str]:
        """
        Fetch tickers from multiple Finviz screener views.
        Filters: top gainers, unusual volume, new highs, oversold bounces.
        """
        tickers: set[str] = set()

        screeners = [
            # Top gainers today (>5% up, price > $2, volume > 200K)
            ("ta_perf_dup", "sh_avgvol_o200,sh_price_o2,sh_relvol_o1.5"),
            # Unusual volume (relative volume > 3, price > $5)
            ("sh_relvol_o3", "sh_avgvol_o200,sh_price_o5"),
            # New 52-week highs
            ("ta_highlow52w_high", "sh_avgvol_o500,sh_price_o5"),
            # Oversold (RSI < 30) — potential reversals
            ("ta_rsi_os30", "sh_avgvol_o200,sh_price_o5"),
            # High volatility movers (>3% change, volume > 500K)
            ("ta_change_u3", "sh_avgvol_o500,sh_price_o5"),
            # Gap up today
            ("ta_gap_u3", "sh_avgvol_o200,sh_price_o5"),
        ]

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        for primary_filter, extra_filters in screeners:
            try:
                export_url = (
                    f"https://finviz.com/screener.ashx?v=152"
                    f"&f={primary_filter},{extra_filters}"
                    f"&ft=4&o=-change&export=csv"
                )
                csv_resp = requests.get(export_url, headers=headers, timeout=15)
                if csv_resp.status_code == 200 and csv_resp.text.strip():
                    # Parse CSV: first column after header is "Ticker"
                    lines = csv_resp.text.strip().split("\n")
                    if len(lines) > 1:
                        for line in lines[1:]:
                            cols = line.split(",")
                            if cols and cols[0].strip():
                                tickers.add(cols[0].strip().upper())
                time.sleep(0.5)  # Rate limit
            except Exception:
                continue

        return list(tickers)

    # ------------------------------------------------------------------
    # Recent IPOs
    # ------------------------------------------------------------------

    def _get_recent_ipos(self) -> list[str]:
        """
        Get recently listed stocks (past 12-24 months).
        Uses Finviz screener for IPO date filter.
        """
        tickers: set[str] = set()
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AlphaFinder/1.0)"}

        # Finviz: IPO date in last 3 years, price > $2
        url = (
            "https://finviz.com/screener.ashx?v=152"
            "&f=ipo_more3,sh_price_o2,sh_avgvol_o100"
            "&ft=4&o=-change&export=csv"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if len(lines) > 1:
                    for line in lines[1:]:
                        cols = line.split(",")
                        if cols and cols[0].strip():
                            tickers.add(cols[0].strip().upper())
        except Exception:
            pass

        return list(tickers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_weighted(self, target: dict[str, float], tickers: list[str], weight: float) -> None:
        """Add tickers to target dict, capping at the higher weight if already present."""
        for t in tickers:
            if t and len(t) <= 5 and t.replace("-", "").replace(".", "").isalpha():
                t = t.upper()
                target[t] = max(target.get(t, 0.0), weight)

    def _curated_russell_sample(self) -> list[str]:
        """Additional small-cap tickers not covered by IWM ETF top holdings."""
        return [
            "ALMU", "LAC", "ALTM", "SGML", "PLL", "SLI", "IONR", "IONQ",
            "RGTI", "QBTS", "QUBT", "QMCO", "BBAI", "AI", "SOUN", "APLD",
            "NBIS", "RXRX", "BEAM", "NTLA", "CRSP", "DNA", "ABCL", "PRME",
            "RIVN", "LCID", "QS", "INDI", "LAZR", "AUR", "OUST", "AEVA",
            "PLUG", "FCEL", "BE", "STEM", "OPAL", "ASTR", "SPIR", "PL",
            "RDW", "BKSY", "MNTS", "RKLB", "MARA", "RIOT", "CLSK", "HUT",
            "SOFI", "AFRM", "FOUR", "MQ", "PAYO", "NU", "HOOD", "UPST",
            "TENB", "VRNS", "QLYS", "RPD", "S", "NET", "CYBR",
            "MVST", "SLDP", "AMLI", "LITM", "ABAT", "ENPH", "SEDG",
            "FSLR", "RUN", "NOVA", "NXT", "ARRY", "SHLS", "MAXN",
            "BLNK", "CHPT", "EVGO", "WBX", "MSTR", "COIN", "HOOD",
            "ALLO", "EDIT", "FATE", "BBIO", "FOLD", "SAGE", "ACAD",
        ]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> str:
        safe = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return os.path.join(self._cache_dir, f"stock_universe_{safe}.json")

    def _load_cache(self, key: str) -> list | dict | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if (datetime.now() - mtime).total_seconds() > self._cache_ttl * 3600:
                return None
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, data: list | dict) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience function for direct use in Stage1DataCollector
# ---------------------------------------------------------------------------

def build_watchlist(mode: str = "broad", cache_dir: str = "data/cache") -> list[str]:
    """
    Build an expanded watchlist for the pipeline.
    Returns a list of ticker symbols sorted by priority.
    """
    builder = StockUniverseBuilder(cache_dir=cache_dir)
    return builder.build(mode=mode)
