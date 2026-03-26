"""
TradeZeroExecutor
=================
Paper-trades signals from the AlphaCombined pipeline via the TradeZero REST API.

Strategy rules (from TraderSummary.md / AlphaCombined):
  Entry  : Market order at next session open when alpha signal > 0.45
  Size   : 1% portfolio risk ÷ (2.0 × ATR₁₄)
  Exit 1 : Trailing stop  — close < highest_close_since_entry − 2.5 × ATR₁₄
  Exit 2 : Alpha reversal — alpha signal drops below −0.5
  Exit 3 : Max holding    — force exit after 7 trading days

Setup
-----
Set these environment variables (or a .env file) before running:

  TRADEZERO_API_KEY      your TradeZero API key
  TRADEZERO_API_SECRET   your TradeZero API secret
  TRADEZERO_BASE_URL     e.g. https://api.tradezero.com/v1  (no trailing slash)
  PORTFOLIO_VALUE        initial portfolio in USD (default 100000)

Usage
-----
  from tradezero_executor import TradeZeroExecutor
  from execution_advisor  import ExecutionAdvisor

  executor = TradeZeroExecutor()

  # --- entry (call once at session open) ---
  executor.enter_signals(advisor_brief)

  # --- daily exit check (call once per day with fresh data) ---
  executor.check_exits(
      current_prices={"NVDA": 181.50},
      current_alpha={"NVDA": 0.30},   # latest alpha z-score from alpha_engine
  )

Position state is persisted to `tradezero_positions.json` so the process can
be restarted without losing track of open trades.

NOTE: Verify all endpoint paths against the official TradeZero developer docs
at https://developer.tradezero.com/ before going live.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_FILE = Path(__file__).parent / "tradezero_positions.json"

_ALPHA_ENTRY_THRESHOLD = 0.45
_ALPHA_EXIT_THRESHOLD = -0.5
_STOP_ATR_MULT = 2.0          # initial stop distance multiplier
_TRAIL_ATR_MULT = 2.5         # trailing stop multiplier
_MAX_HOLDING_DAYS = 7         # trading days
_PORTFOLIO_RISK_PCT = 0.01    # 1 % of portfolio per trade


# ---------------------------------------------------------------------------
# TradeZero API client
# ---------------------------------------------------------------------------

class _TradeZeroClient:
    """Thin wrapper around the TradeZero REST API.

    Auth is header-based — TZ-API-KEY-ID and TZ-API-SECRET-KEY are sent
    on every request. No token exchange required.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str, account_id: str) -> None:
        self._base = base_url.rstrip("/")
        self._account_id = account_id
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type":      "application/json",
            "Accept":            "application/json",
            "TZ-API-KEY-ID":     api_key,
            "TZ-API-SECRET-KEY": api_secret,
        })

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        ticker: str,
        side: str,      # "Buy" or "Sell"
        quantity: int,
    ) -> dict:
        payload = {
            "symbol":        ticker,
            "side":          side,
            "orderQuantity": quantity,
            "orderType":     "Market",
            "securityType":  "Stock",
            "timeInForce":   "Day",
        }
        logger.info("TradeZero: placing %s %s x %d (Market)", side, ticker, quantity)
        return self._post(f"/v1/api/accounts/{self._account_id}/order", payload)

    def get_order(self, order_id: str) -> dict:
        return self._get(f"/v1/api/accounts/{self._account_id}/order/{order_id}")

    def get_orders(self) -> list[dict]:
        return self._get(f"/v1/api/accounts/{self._account_id}/orders")

    def get_orders_from(self, start_date: str) -> list[dict]:
        """start_date: YYYY-MM-DD"""
        return self._get(
            f"/v1/api/accounts/{self._account_id}/orders/start-date/{start_date}"
        )

    def get_routes(self) -> list[dict]:
        return self._get(f"/v1/api/accounts/{self._account_id}/routes")

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/v1/api/accounts/{self._account_id}/order/{order_id}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        resp = self._session.get(f"{self._base}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> Any:
        resp = self._session.post(f"{self._base}{path}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Any:
        resp = self._session.delete(f"{self._base}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Position state helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _trading_days_since(entry_date_str: str) -> int:
    """Count trading days (Mon–Fri) between entry date and today."""
    entry = date.fromisoformat(entry_date_str)
    today = date.today()
    if today <= entry:
        return 0
    count = 0
    current = entry + timedelta(days=1)
    while current <= today:
        if current.weekday() < 5:   # Mon=0 … Fri=4
            count += 1
        current += timedelta(days=1)
    return count


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

class TradeZeroExecutor:
    """
    Executes and manages paper trades on TradeZero using AlphaCombined rules.

    Parameters
    ----------
    portfolio_value : float
        Current portfolio value used for position sizing (default: env var
        PORTFOLIO_VALUE or 100 000).
    dry_run : bool
        If True, log all intended actions but do NOT send orders to TradeZero.
        Useful for testing the logic without touching the account.
    """

    def __init__(
        self,
        portfolio_value: float | None = None,
        dry_run: bool = False,
    ) -> None:
        api_key    = os.environ.get("TRADEZERO_API_KEY", "")
        api_secret = os.environ.get("TRADEZERO_API_SECRET", "")
        base_url   = os.environ.get("TRADEZERO_BASE_URL", "https://webapi.tradezero.com")
        account_id = os.environ.get("TRADEZERO_ACCOUNT_ID", "")

        if not api_key or not api_secret:
            raise EnvironmentError(
                "TRADEZERO_API_KEY and TRADEZERO_API_SECRET must be set."
            )
        if not account_id and not dry_run:
            raise EnvironmentError("TRADEZERO_ACCOUNT_ID must be set.")

        self._portfolio = portfolio_value or float(
            os.environ.get("PORTFOLIO_VALUE", 100_000)
        )
        self._dry_run = dry_run
        self._client = _TradeZeroClient(api_key, api_secret, base_url, account_id)

        self._state: dict = _load_state()   # ticker → position dict
        self._closed_trades: list[dict] = []  # accumulated this session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enter_signals(self, advisor_brief: dict) -> None:
        """
        Place market orders for all active signals in an ExecutionAdvisor brief
        that are not already open.

        Parameters
        ----------
        advisor_brief : dict
            Output of ExecutionAdvisor.advise() — must contain "active_signals".
        """
        for signal in advisor_brief.get("active_signals", []):
            ticker = signal["ticker"]
            if ticker in self._state:
                logger.info("%s: already have an open position, skipping entry.", ticker)
                continue

            shares = signal.get("position_size", 0)
            atr = signal.get("atr14") or self._derive_atr(signal)
            entry_price = signal.get("entry_price", 0.0)
            stop_price = signal.get("stop_price", 0.0)

            if shares <= 0:
                logger.warning("%s: position_size is 0, skipping.", ticker)
                continue

            order_result = self._place_order(ticker, "Buy", shares)

            self._state[ticker] = {
                "ticker": ticker,
                "entry_date": date.today().isoformat(),
                "entry_price": entry_price,
                "stop_price": stop_price,
                "atr14": atr,
                "shares": shares,
                "highest_close": entry_price,   # updated each day
                "order_id": order_result.get("orderId") if order_result else None,
            }
            _save_state(self._state)
            logger.info(
                "ENTERED %s | %d shares @ ~$%.2f | stop $%.2f",
                ticker, shares, entry_price, stop_price,
            )

    def check_exits(
        self,
        current_prices: dict[str, float],
        current_alpha: dict[str, float] | None = None,
    ) -> list[str]:
        """
        Evaluate all three exit rules for every open position.

        Parameters
        ----------
        current_prices : dict
            {ticker: latest_close_price}
        current_alpha : dict, optional
            {ticker: current_alpha_z_score} — needed to evaluate alpha reversal exit.

        Returns
        -------
        list of tickers that were exited this call.
        """
        if current_alpha is None:
            current_alpha = {}

        exited: list[str] = []

        for ticker, pos in list(self._state.items()):
            close = current_prices.get(ticker)
            if close is None:
                logger.warning("%s: no price available, skipping exit check.", ticker)
                continue

            # Update trailing high
            if close > pos["highest_close"]:
                pos["highest_close"] = close
                _save_state(self._state)

            atr = pos["atr14"]
            reason: str | None = None

            # Exit 1 — trailing stop
            trail_stop = pos["highest_close"] - _TRAIL_ATR_MULT * atr
            if close < trail_stop:
                reason = "trailing_stop"

            # Exit 2 — alpha reversal
            elif ticker in current_alpha and current_alpha[ticker] < _ALPHA_EXIT_THRESHOLD:
                reason = "alpha_reversal"

            # Exit 3 — max holding days
            elif _trading_days_since(pos["entry_date"]) >= _MAX_HOLDING_DAYS:
                reason = "max_holding"

            if reason:
                self._place_order(ticker, "Sell", pos["shares"])
                pnl = (close - pos["entry_price"]) * pos["shares"]
                logger.info(
                    "EXITED %s | reason=%s | close=%.2f | entry=%.2f | PnL=$%.2f",
                    ticker, reason, close, pos["entry_price"], pnl,
                )
                self._closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date.today().isoformat(),
                    "entry_price": pos["entry_price"],
                    "exit_price": close,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "reason": reason,
                })
                del self._state[ticker]
                _save_state(self._state)
                exited.append(ticker)

        return exited

    def open_positions(self) -> dict:
        """Return a copy of the current tracked positions."""
        return dict(self._state)

    def summary(self) -> None:
        """Print a session P&L summary of all trades closed this run."""
        trades = self._closed_trades
        if not trades:
            print("\n=== TradeZero Session Summary ===")
            print("No trades closed this session.")
            return

        total_pnl = sum(t["pnl"] for t in trades)
        winners   = [t for t in trades if t["pnl"] > 0]
        losers    = [t for t in trades if t["pnl"] <= 0]
        win_rate  = len(winners) / len(trades) * 100

        print("\n=== TradeZero Session Summary ===")
        print(f"Closed trades : {len(trades)}")
        print(f"Winners       : {len(winners)}  ({win_rate:.1f}%)")
        print(f"Losers        : {len(losers)}  ({100 - win_rate:.1f}%)")
        print(f"Total PnL     : {'+'if total_pnl >= 0 else ''}${total_pnl:,.2f}")
        if winners:
            best = max(winners, key=lambda t: t["pnl"])
            print(f"Best trade    : {best['ticker']} +${best['pnl']:,.2f}")
        if losers:
            worst = min(losers, key=lambda t: t["pnl"])
            print(f"Worst trade   : {worst['ticker']} ${worst['pnl']:,.2f}")
        print("-" * 34)
        for t in trades:
            sign = "+" if t["pnl"] >= 0 else ""
            print(
                f"  {t['ticker']:<6} {t['entry_date']} → {t['exit_date']}"
                f"  {t['shares']}sh  {sign}${t['pnl']:,.2f}  [{t['reason']}]"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _place_order(self, ticker: str, side: str, shares: int) -> dict | None:
        if self._dry_run:
            logger.info("[DRY RUN] Would send: %s %s x %d", side, ticker, shares)
            return {"orderId": "DRY_RUN"}
        try:
            return self._client.place_market_order(ticker, side, shares)
        except requests.HTTPError as exc:
            logger.error("Order failed for %s: %s", ticker, exc)
            return None

    @staticmethod
    def _derive_atr(signal: dict) -> float:
        """Fall back: derive ATR from stop distance (stop_distance = 2.0 × ATR)."""
        entry = signal.get("entry_price", 0.0)
        stop = signal.get("stop_price", 0.0)
        if entry and stop:
            return abs(entry - stop) / _STOP_ATR_MULT
        return 1.0   # safe non-zero default


# ---------------------------------------------------------------------------
# TraderSummary.md parser
# ---------------------------------------------------------------------------

def parse_trader_summary(path: str | Path) -> dict:
    """
    Parse a TraderSummary.md file and return an ExecutionAdvisor-compatible
    brief so the executor can be tested without running the full pipeline.

    Extracts every ticker marked '✅ ... ENTER NOW' from the Action section
    and reads its entry price, stop loss, position size, and ATR₁₄.

    Returns
    -------
    dict with key "active_signals", each entry matching the ExecutionAdvisor
    output schema expected by enter_signals().
    """
    import re

    text = Path(path).read_text(encoding="utf-8")
    signals: list[dict] = []

    # Find each "ENTER NOW" block in the ## Today's Action section
    # e.g.  ### ✅ NVDA — ENTER NOW (AlphaCombined)
    action_tickers = re.findall(
        r"###\s*✅\s+([A-Z]+)\s*[—–-]+\s*ENTER NOW",
        text,
    )

    for ticker in action_tickers:
        # Scope the search to the detailed section for this ticker
        # which starts at "## TICKER —" and runs until the next "## "
        section_match = re.search(
            rf"##\s+{re.escape(ticker)}\b.*?\n(.*?)(?=\n## |\Z)",
            text,
            re.DOTALL,
        )
        section = section_match.group(1) if section_match else text

        def _find(pattern: str, src: str = section) -> float | None:
            m = re.search(pattern, src)
            return float(m.group(1).replace(",", "")) if m else None

        entry_price  = _find(r"\|\s*Entry price\s*\|\s*\$?([\d,.]+)")
        stop_price   = _find(r"\|\s*Stop loss\s*\|\s*\$?([\d,.]+)")
        position_size = _find(r"\|\s*Position size\s*\|\s*([\d,]+)\s*shares")
        atr14        = _find(r"\|\s*Current ATR.*?\|\s*\$?([\d,.]+)")

        # Fallback to the bullet-point summary if table not found
        if entry_price is None:
            entry_price = _find(r"next session open.*?\$?([\d,.]+)\)", text)
        if stop_price is None:
            stop_price = _find(r"Stop loss.*?\$?([\d,.]+)", text)
        if position_size is None:
            pos_m = re.search(rf"{ticker}.*?Position size.*?([\d,]+)\s*shares", text, re.DOTALL)
            position_size = float(pos_m.group(1).replace(",", "")) if pos_m else None
        if atr14 is None:
            atr14 = _find(r"ATR.*?\$?([\d,.]+)", text)

        if entry_price is None:
            logger.warning("Could not parse entry price for %s — skipping.", ticker)
            continue

        dollar_risk = 1000.0   # default; 1% of $100k
        signals.append({
            "ticker":        ticker,
            "entry_price":   entry_price,
            "stop_price":    stop_price or (entry_price * 0.94),
            "position_size": int(position_size or 0),
            "atr14":         atr14 or abs(entry_price - (stop_price or entry_price * 0.94)) / 2.0,
            "dollar_risk":   dollar_risk,
        })
        logger.info(
            "Parsed signal: %s | entry=%.2f | stop=%.2f | size=%d | ATR=%.2f",
            ticker,
            entry_price,
            stop_price or 0,
            int(position_size or 0),
            atr14 or 0,
        )

    if not signals:
        logger.warning("No ENTER NOW signals found in %s", path)

    return {"active_signals": signals}


# ---------------------------------------------------------------------------
# CLI convenience runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TradeZero paper executor.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without sending real orders.",
    )
    parser.add_argument(
        "--check-exits-only",
        action="store_true",
        help="Only run exit checks (no new entries).",
    )
    parser.add_argument(
        "--portfolio",
        type=float,
        default=None,
        help="Override portfolio value for sizing.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a TraderSummary.md file. When provided, skips the live "
             "pipeline and parses signals directly from the file (for testing).",
    )
    args = parser.parse_args()

    import time
    import yfinance as yf
    from zoneinfo import ZoneInfo

    _ET          = ZoneInfo("America/New_York")
    _OPEN        = dtime(9, 30)
    _CLOSE       = dtime(16, 0)
    CHECK_EVERY  = 60   # seconds between exit checks during market hours

    executor = TradeZeroExecutor(
        portfolio_value=args.portfolio,
        dry_run=args.dry_run,
    )

    def _get_brief() -> dict:
        if args.summary_file:
            return parse_trader_summary(args.summary_file)
        import ollama as _ollama
        from execution_advisor import ExecutionAdvisor
        from pipeline_orchestrator import PipelineOrchestrator
        _llm = _ollama.Client()
        _cfg = {
            "benzinga_api_key":   os.environ.get("BENZINGA_API"),
            "llm_client":         _llm,
            "output_dir":         "reports",
            "cache_dir":          "data/cache",
            "chroma_dir":         "data/chroma",
            "initial_portfolio":  args.portfolio or 100_000.0,
            "window_days":        7,
            "max_tickers":        20,
            "market_min_volume":  500_000,
            "market_max_markets": 5,
        }
        result = PipelineOrchestrator(_cfg).run()
        from execution_advisor import ExecutionAdvisor
        return ExecutionAdvisor(initial_portfolio=args.portfolio or 100_000).advise(
            result.get("strategies", [])
        )

    def _market_open() -> bool:
        now = datetime.now(tz=_ET)
        return now.weekday() < 5 and _OPEN <= now.time() <= _CLOSE

    entered_today: str | None = None   # track which date we already entered

    print("TradeZero executor running. Press Ctrl+C to stop.\n")

    try:
        while True:
            now_et = datetime.now(tz=_ET)
            today  = now_et.date().isoformat()

            if _market_open():
                # Enter signals once per trading day
                if entered_today != today and not args.check_exits_only:
                    brief = _get_brief()
                    executor.enter_signals(brief)
                    entered_today = today

                # Check exits on every open position
                tickers = list(executor.open_positions().keys())
                if tickers:
                    prices = {t: yf.Ticker(t).fast_info["lastPrice"] for t in tickers}
                    executor.check_exits(prices)
                else:
                    logger.info("No open positions to monitor.")

                time.sleep(CHECK_EVERY)

            else:
                # Outside market hours — sleep longer
                logger.info("Market closed. Next check in 5 minutes.")
                time.sleep(300)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        executor.summary()
