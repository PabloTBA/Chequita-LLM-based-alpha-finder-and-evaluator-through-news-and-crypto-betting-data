"""
PortfolioManager
================
Combines all TraderSummary.md files from the output directory into a single
consolidated trading plan, adjusts every position to the user's actual live
portfolio value, and hands off the merged brief to TradeZeroExecutor.

Design
------
1. Load all TraderSummary*.md files from `summary_dir` (newest first).
2. Parse "ENTER NOW" signals from each file using the existing
   `parse_trader_summary` helper.
3. Merge signals across files:
   - Recency weight: files are scored by age (most recent = highest weight).
   - Consensus boost: if the same ticker appears in multiple summaries,
     its weight increases proportionally.
   - Only the most recent entry price / stop / ATR values are used for sizing.
4. Fetch the live portfolio value from the TradeZero API balance endpoint.
   Falls back to the PORTFOLIO_VALUE env var, then to $100,000.
5. Re-size every position using the real portfolio value (1% risk per trade).
6. Skip any ticker that already has an open position in tradezero_positions.json.
7. Print a consolidated action plan and optionally execute via TradeZeroExecutor.

Usage
-----
    # Dry run — prints plan, no orders sent
    python portfolio_manager.py --dry-run

    # Live run — executes orders at next session open
    python portfolio_manager.py

    # Override summary directory
    python portfolio_manager.py --summary-dir path/to/md --dry-run
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from tradezero_executor import (
    TradeZeroExecutor,
    _TradeZeroClient,
    _load_state,
    parse_trader_summary,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PORTFOLIO  = 100_000.0
_RISK_PCT           = 0.01        # 1% of portfolio per trade
_STOP_ATR_MULT      = 2.0         # stop = entry - 2.0 * ATR
_RECENCY_HALF_LIFE  = 7.0         # days — weight halves every 7 days
_CONSENSUS_FACTOR   = 0.25        # each additional file adds 25% weight bonus
_MAX_SIGNALS        = 10          # hard cap on simultaneous new entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_summary_date(path: Path) -> date:
    """
    Extract the run date from the first heading of a TraderSummary.md file.
    Falls back to the file modification time if the heading cannot be parsed.
    """
    try:
        text = path.read_text(encoding="utf-8")
        import re
        m = re.search(r"#.*?(\d{4}-\d{2}-\d{2})", text)
        if m:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def _recency_weight(file_date: date, today: date) -> float:
    """Exponential decay weight based on how many days ago the file was written."""
    age_days = max(0, (today - file_date).days)
    return math.exp(-math.log(2) * age_days / _RECENCY_HALF_LIFE)


def _resize_signal(signal: dict, portfolio_value: float) -> dict:
    """
    Return a new signal dict with position_size recalculated for portfolio_value.
    Uses 1% portfolio risk / (STOP_ATR_MULT * ATR).
    """
    entry = signal["entry_price"]
    stop  = signal["stop_price"]
    atr   = signal.get("atr14") or (abs(entry - stop) / _STOP_ATR_MULT) or 1.0

    dollar_risk   = portfolio_value * _RISK_PCT
    stop_distance = _STOP_ATR_MULT * atr
    position_size = max(1, int(dollar_risk / stop_distance)) if stop_distance > 0 else 1

    return {
        **signal,
        "position_size": position_size,
        "dollar_risk":   dollar_risk,
        "atr14":         atr,
    }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Reads every TraderSummary.md in `summary_dir`, merges signals with
    recency + consensus weighting, fetches live portfolio value, and
    produces a consolidated execution brief.
    """

    def __init__(
        self,
        summary_dir: str = "front-end/md",
        dry_run: bool = False,
    ) -> None:
        self._summary_dir = Path(summary_dir)
        self._dry_run     = dry_run

        api_key    = os.environ.get("TRADEZERO_API_KEY", "")
        api_secret = os.environ.get("TRADEZERO_API_SECRET", "")
        base_url   = os.environ.get("TRADEZERO_BASE_URL", "https://webapi.tradezero.com")
        account_id = os.environ.get("TRADEZERO_ACCOUNT_ID", "")

        self._has_api = bool(api_key and api_secret)
        self._client  = (
            _TradeZeroClient(api_key, api_secret, base_url, account_id)
            if self._has_api
            else None
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Full pipeline: load → merge → value → resize → print → execute."""
        today = date.today()

        # 1. Load all summary files
        summaries = self._load_summaries()
        if not summaries:
            print("[PortfolioManager] No TraderSummary.md files found in "
                  f"'{self._summary_dir}'. Nothing to do.")
            return

        print(f"[PortfolioManager] Loaded {len(summaries)} summary file(s).")

        # 2. Merge signals
        merged = self._merge_signals(summaries, today)
        if not merged:
            print("[PortfolioManager] No active ENTER NOW signals across all summaries.")
            return

        # 3. Get live portfolio value
        portfolio_value = self._get_portfolio_value()
        print(f"[PortfolioManager] Portfolio value: ${portfolio_value:,.2f}")

        # 4. Re-size positions to real capital
        sized = [_resize_signal(s, portfolio_value) for s in merged]

        # 5. Filter already-open positions
        open_positions = _load_state()
        fresh = [s for s in sized if s["ticker"] not in open_positions]
        skipped = [s["ticker"] for s in sized if s["ticker"] in open_positions]
        if skipped:
            print(f"[PortfolioManager] Skipping already-open: {', '.join(skipped)}")

        # 6. Cap at max simultaneous signals
        if len(fresh) > _MAX_SIGNALS:
            fresh = fresh[:_MAX_SIGNALS]
            print(f"[PortfolioManager] Capped to top {_MAX_SIGNALS} signals by weight.")

        # 7. Print consolidated plan
        self._print_plan(fresh, portfolio_value, open_positions)

        if not fresh:
            print("[PortfolioManager] No new signals to act on.")
            return

        # 8. Execute via TradeZeroExecutor
        if not self._has_api:
            print("[PortfolioManager] No TradeZero credentials — skipping execution.")
            return

        try:
            executor = TradeZeroExecutor(
                portfolio_value=portfolio_value,
                dry_run=self._dry_run,
            )
            executor.enter_signals({"active_signals": fresh})
            executor.summary()
        except EnvironmentError as exc:
            logger.error("TradeZeroExecutor init failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 1: Load summaries
    # ------------------------------------------------------------------

    def _load_summaries(self) -> list[dict]:
        """
        Find all TraderSummary*.md files, parse each, and return a list of
        {date, signals, path} dicts sorted newest-first.
        """
        if not self._summary_dir.exists():
            return []

        files = sorted(
            self._summary_dir.glob("TraderSummary*.md"),
            key=lambda p: _parse_summary_date(p),
            reverse=True,
        )

        results = []
        for f in files:
            parsed = parse_trader_summary(f)
            signals = parsed.get("active_signals", [])
            file_date = _parse_summary_date(f)
            results.append({
                "path":    f,
                "date":    file_date,
                "signals": signals,
            })
            status = f"{len(signals)} signal(s)" if signals else "no signals"
            print(f"  {f.name} ({file_date}) — {status}")

        return results

    # ------------------------------------------------------------------
    # Step 2: Merge signals
    # ------------------------------------------------------------------

    def _merge_signals(
        self,
        summaries: list[dict],
        today: date,
    ) -> list[dict]:
        """
        Combine signals across all summaries with recency + consensus weighting.

        For each ticker that appears as ENTER NOW in at least one summary:
        - weight = recency_weight * (1 + CONSENSUS_FACTOR * (appearances - 1))
        - Use entry/stop/ATR from the most recent file that contained the ticker.

        Returns signals sorted by weight descending.
        """
        # ticker → {weight, signal (from most recent file)}
        ticker_map: dict[str, dict] = {}

        for summary in summaries:
            w = _recency_weight(summary["date"], today)
            for signal in summary["signals"]:
                t = signal["ticker"]
                if t not in ticker_map:
                    ticker_map[t] = {"weight": w, "signal": signal, "count": 1}
                else:
                    # Already seen — boost weight by consensus factor, keep newest signal
                    ticker_map[t]["weight"] += w * _CONSENSUS_FACTOR
                    ticker_map[t]["count"]  += 1
                    # Keep signal from most recent file (already first due to sort)

        if not ticker_map:
            return []

        # Sort by combined weight descending
        ranked = sorted(ticker_map.values(), key=lambda x: x["weight"], reverse=True)

        print(f"\n[PortfolioManager] Merged {len(ranked)} unique signal(s):")
        for item in ranked:
            t = item["signal"]["ticker"]
            print(
                f"  {t:<6} weight={item['weight']:.3f}  "
                f"(appeared in {item['count']} file(s))"
            )
        print()

        return [item["signal"] for item in ranked]

    # ------------------------------------------------------------------
    # Step 3: Fetch live portfolio value
    # ------------------------------------------------------------------

    def _get_portfolio_value(self) -> float:
        """
        Try TradeZero balance endpoint first.
        Falls back to PORTFOLIO_VALUE env var, then to $100,000.
        """
        if self._client:
            try:
                account_id = os.environ.get("TRADEZERO_ACCOUNT_ID", "")
                resp = self._client._get(f"/accounts/{account_id}/balance")
                # TradeZero returns totalEquity or netLiquidation — try both
                value = (
                    resp.get("totalEquity")
                    or resp.get("netLiquidation")
                    or resp.get("cashBalance")
                )
                if value:
                    val = float(value)
                    print(f"[PortfolioManager] Live balance from TradeZero: ${val:,.2f}")
                    return val
            except Exception as exc:
                logger.warning("Could not fetch live balance: %s — using fallback.", exc)

        env_val = os.environ.get("PORTFOLIO_VALUE")
        if env_val:
            val = float(env_val)
            print(f"[PortfolioManager] Portfolio value from env: ${val:,.2f}")
            return val

        print(f"[PortfolioManager] No balance source — using default ${_DEFAULT_PORTFOLIO:,.2f}")
        return _DEFAULT_PORTFOLIO

    # ------------------------------------------------------------------
    # Step 4: Print consolidated plan
    # ------------------------------------------------------------------

    def _print_plan(
        self,
        signals: list[dict],
        portfolio_value: float,
        open_positions: dict,
    ) -> None:
        total_risk = sum(s["dollar_risk"] for s in signals)
        pct_at_risk = total_risk / portfolio_value * 100 if portfolio_value else 0

        print("=" * 60)
        print("  CONSOLIDATED PORTFOLIO PLAN")
        print("=" * 60)
        print(f"  Portfolio value : ${portfolio_value:,.2f}")
        print(f"  Open positions  : {len(open_positions)}")
        print(f"  New signals     : {len(signals)}")
        print(f"  Total $ at risk : ${total_risk:,.2f}  ({pct_at_risk:.1f}% of portfolio)")
        print("-" * 60)

        if open_positions:
            print("  CURRENTLY OPEN:")
            for ticker, pos in open_positions.items():
                entry    = pos.get("entry_price", 0)
                size     = pos.get("shares", 0)
                entered  = pos.get("entry_date", "?")
                print(f"    {ticker:<6} {size} shares @ ${entry:.2f}  (entered {entered})")
            print()

        if signals:
            print("  NEW ENTRIES (at next session open):")
            for s in signals:
                stop_dist = s["entry_price"] - s["stop_price"]
                print(
                    f"    {s['ticker']:<6} "
                    f"entry=${s['entry_price']:.2f}  "
                    f"stop=${s['stop_price']:.2f}  "
                    f"size={s['position_size']} shares  "
                    f"risk=${s['dollar_risk']:,.0f}  "
                    f"stop_dist=${stop_dist:.2f}"
                )
        else:
            print("  No new entries.")

        print("=" * 60)
        mode = "[DRY RUN]" if self._dry_run else "[LIVE]"
        print(f"  Mode: {mode}")
        print("=" * 60)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Consolidated portfolio manager — merges all TraderSummary.md files "
                    "and executes a unified trading plan."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log all actions but do NOT send real orders.",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="front-end/md",
        help="Directory containing TraderSummary*.md files (default: front-end/md).",
    )
    args = parser.parse_args()

    manager = PortfolioManager(
        summary_dir=args.summary_dir,
        dry_run=args.dry_run,
    )
    manager.run()
