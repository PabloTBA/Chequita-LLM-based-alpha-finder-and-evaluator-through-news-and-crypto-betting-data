"""
ReportGenerator
===============
Assembles all pipeline stage outputs into a single structured Markdown report.
Computes graph-ready data tables (equity curve, drawdown, walk-forward,
return distribution) and advanced trader metrics from the backtest results.

Public interface
----------------
    gen      = ReportGenerator(output_dir="reports")
    filepath = gen.generate(pipeline_output)
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, pipeline_output: dict) -> str:
        """
        Build and write the Markdown report.

        Parameters
        ----------
        pipeline_output : dict with keys:
            run_date, summary, macro, ticker_verdicts, regimes,
            strategies, diagnostics, backtests

        Returns
        -------
        str — absolute path of the written .md file
        """
        run_date  = pipeline_output.get("run_date", datetime.today().strftime("%Y-%m-%d"))
        timestamp = datetime.now().strftime("%H%M%S")
        filename  = f"report_{run_date}_{timestamp}.md"
        filepath  = os.path.join(self.output_dir, filename)

        sections = [
            self._title(run_date),
            self._executive_summary(pipeline_output),
            self._macro_section(pipeline_output),
            self._tickers_section(pipeline_output),
            self._regime_section(pipeline_output),
            self._strategy_section(pipeline_output),
            self._diagnostic_section(pipeline_output),
            self._backtest_section(pipeline_output),
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(sections))

        return filepath

    # ── sections ──────────────────────────────────────────────────────────────

    @staticmethod
    def _title(run_date: str) -> str:
        return f"# MFT Alpha Finder & Evaluator — Report {run_date}"

    @staticmethod
    def _executive_summary(po: dict) -> str:
        s   = po.get("summary", {})
        mac = po.get("macro", {})
        vds = po.get("ticker_verdicts", [])

        buys    = [v["ticker"] for v in vds if v.get("verdict") == "buy"]
        watches = [v["ticker"] for v in vds if v.get("verdict") == "watch"]
        avoids  = [v["ticker"] for v in vds if v.get("verdict") == "avoid"]

        lines = [
            "## Executive Summary",
            "",
            f"**Run date:** {po.get('run_date', 'N/A')}  ",
            f"**News window:** {s.get('window_start', 'N/A')} → {s.get('window_end', 'N/A')}  ",
            f"**Articles analysed:** {s.get('article_count', 0)}  ",
            f"**Overall market bias:** {mac.get('market_bias', s.get('market_bias', 'neutral')).upper()}  ",
            "",
            f"**Buy candidates ({len(buys)}):** {', '.join(buys) or 'None'}  ",
            f"**Watch ({len(watches)}):** {', '.join(watches) or 'None'}  ",
            f"**Avoid ({len(avoids)}):** {', '.join(avoids) or 'None'}  ",
            "",
            "**Top themes:** " + ", ".join(s.get("top_themes", [])),
            "",
            "**Key risks:** " + ", ".join(s.get("key_risks", [])),
        ]
        return "\n".join(lines)

    @staticmethod
    def _macro_section(po: dict) -> str:
        m = po.get("macro", {})
        lines = [
            "## Macro Environment",
            "",
            f"**Market bias:** {m.get('market_bias', 'neutral').upper()}",
            "",
            f"**Favoured sectors:** {', '.join(m.get('favored_sectors', []))}  ",
            f"**Avoid sectors:** {', '.join(m.get('avoid_sectors', []))}  ",
            f"**Active macro risks:** {', '.join(m.get('active_macro_risks', []))}  ",
            "",
            f"> {m.get('reasoning', '')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _tickers_section(po: dict) -> str:
        verdicts = po.get("ticker_verdicts", [])
        lines = [
            "## Shortlisted Tickers",
            "",
            "| Ticker | Verdict | Reasoning |",
            "|--------|---------|-----------|",
        ]
        for v in verdicts:
            verdict = v.get("verdict", "watch").upper()
            lines.append(f"| {v['ticker']} | **{verdict}** | {v.get('reasoning', '')} |")
        return "\n".join(lines)

    @staticmethod
    def _regime_section(po: dict) -> str:
        regimes = po.get("regimes", [])
        lines = [
            "## Regime Classification",
            "",
            "| Ticker | Regime | Hurst | ATR/Price |",
            "|--------|--------|-------|-----------|",
        ]
        for r in regimes:
            lines.append(
                f"| {r['ticker']} | {r['regime']} "
                f"| {r['hurst']:.3f} | {r['atr_pct']:.2%} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _strategy_section(po: dict) -> str:
        strategies = po.get("strategies", [])
        blocks = ["## Strategy Parameters", ""]
        for s in strategies:
            params = s.get("adjusted_params", {})
            adj    = s.get("llm_adjustments", [])
            blocks += [
                f"### {s['ticker']} — {s['strategy']}",
                "",
                f"**Regime:** {s.get('regime', 'N/A')}  ",
                f"**Reasoning:** {s.get('reasoning', '')}",
                "",
                "**Adjusted parameters:**",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
            for k, v in params.items():
                blocks.append(f"| {k} | {v} |")
            if adj:
                blocks += ["", "**LLM adjustments:**", ""]
                for note in adj:
                    blocks.append(f"- {note}")
            blocks.append("")
        return "\n".join(blocks)

    def _diagnostic_section(self, po: dict) -> str:
        diagnostics = po.get("diagnostics", [])
        backtests   = {b["ticker"]: b for b in po.get("backtests", [])}
        blocks = ["## Diagnostic Results", ""]

        for d in diagnostics:
            ticker  = d["ticker"]
            passed  = d.get("passed", False)
            status  = "✅ PASS" if passed else "❌ FAIL"
            reject  = d.get("reject_reason") or "—"
            metrics = d.get("metrics", {})
            bt      = backtests.get(ticker, {})
            returns = bt.get("returns", pd.Series(dtype=float))
            trade_log = bt.get("trade_log", [])

            adv = _advanced_metrics(returns, trade_log, metrics)

            blocks += [
                f"### {ticker} — {d['strategy']} [{status}]",
                "",
                f"**Reject reason:** {reject}  ",
                "",
                "#### Core Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Sharpe Ratio | {metrics.get('sharpe', 0):.3f} |",
                f"| Sortino Ratio | {adv['sortino']:.3f} |",
                f"| Calmar Ratio | {adv['calmar']:.3f} |",
                f"| CAGR | {adv['cagr']:.2%} |",
                f"| Annualised Volatility | {adv['ann_vol']:.2%} |",
                f"| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |",
                f"| Max DD Recovery (days) | {adv['recovery_days']} |",
                f"| VaR 95% (daily) | {adv['var_95']:.3%} |",
                f"| CVaR 95% (daily) | {adv['cvar_95']:.3%} |",
                "",
                "#### Trade Statistics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Trade Count | {metrics.get('trade_count', 0)} |",
                f"| Win Rate | {metrics.get('win_rate', 0):.1%} |",
                f"| Avg Win | {adv['avg_win']:.2f} |",
                f"| Avg Loss | {adv['avg_loss']:.2f} |",
                f"| Profit Factor | {adv['profit_factor']:.3f} |",
                f"| Max Consecutive Losses | {adv['max_consec_losses']} |",
                "",
                "#### Walk-Forward Validation",
                "",
                "| Period | Sharpe | Total Return |",
                "|--------|--------|--------------|",
                f"| In-Sample | {adv['is_sharpe']:.3f} | {adv['is_return']:.2%} |",
                f"| Out-of-Sample | {adv['oos_sharpe']:.3f} | {adv['oos_return']:.2%} |",
                f"| Degradation | — | {metrics.get('walk_forward_degradation', 0):.1%} |",
                "",
                "#### Exit Reason Breakdown",
                "",
                "| Exit Reason | Count |",
                "|-------------|-------|",
            ]
            for reason, count in adv["exit_breakdown"].items():
                blocks.append(f"| {reason} | {count} |")

            if d.get("llm_commentary"):
                blocks += ["", f"> **LLM commentary:** {d['llm_commentary']}"]

            blocks.append("")

        return "\n".join(blocks)

    def _backtest_section(self, po: dict) -> str:
        backtests = po.get("backtests", [])
        blocks = ["## Backtest Results", ""]

        for bt in backtests:
            ticker    = bt["ticker"]
            summary   = bt.get("summary", {})
            trade_log = bt.get("trade_log", [])
            equity    = bt.get("equity_curve", pd.Series(dtype=float))
            returns   = bt.get("returns", pd.Series(dtype=float))

            blocks += [
                f"### {ticker} — {bt['strategy']}",
                "",
                f"**Total Return:** {summary.get('total_return', 0):.2%}  ",
                f"**Trade Count:** {summary.get('trade_count', 0)}  ",
                f"**Win Rate:** {summary.get('win_rate', 0):.1%}  ",
                "",
            ]

            # Trade log table
            blocks += [
                "#### Trade Log",
                "",
                "| Entry Date | Entry $ | Exit Date | Exit $ | Days | Size | P&L | Exit Reason |",
                "|------------|---------|-----------|--------|------|------|-----|-------------|",
            ]
            for t in trade_log:
                edate = t["entry_date"].strftime("%Y-%m-%d") if hasattr(t["entry_date"], "strftime") else str(t["entry_date"])
                xdate = t["exit_date"].strftime("%Y-%m-%d")  if hasattr(t["exit_date"],  "strftime") else str(t["exit_date"])
                blocks.append(
                    f"| {edate} | {t['entry_price']:.2f} | {xdate} | {t['exit_price']:.2f}"
                    f" | {t['holding_days']} | {t['position_size']:.1f}"
                    f" | {t['pnl']:+.2f} | {t['exit_reason']} |"
                )

            # Best / worst trades
            if trade_log:
                sorted_trades = sorted(trade_log, key=lambda x: x["pnl"])
                worst = sorted_trades[:3]
                best  = sorted_trades[-3:][::-1]
                blocks += ["", "**Best 3 trades:**", ""]
                for t in best:
                    xdate = t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"])
                    blocks.append(f"- {xdate}: P&L = **{t['pnl']:+.2f}** ({t['exit_reason']})")
                blocks += ["", "**Worst 3 trades:**", ""]
                for t in worst:
                    xdate = t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"])
                    blocks.append(f"- {xdate}: P&L = **{t['pnl']:+.2f}** ({t['exit_reason']})")

            # Equity curve table (sampled — every 20 bars to keep report readable)
            if not equity.empty:
                blocks += ["", "#### Equity Curve", "", "| Date | Portfolio Value |", "|------|----------------|"]
                step = max(1, len(equity) // 20)
                for date, val in equity.iloc[::step].items():
                    dstr = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                    blocks.append(f"| {dstr} | {val:,.2f} |")

            # Drawdown table (sampled)
            if not equity.empty:
                dd_series = _drawdown_series(equity)
                blocks += ["", "#### Drawdown Curve", "", "| Date | Drawdown |", "|------|----------|"]
                step = max(1, len(dd_series) // 20)
                for date, val in dd_series.iloc[::step].items():
                    dstr = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                    blocks.append(f"| {dstr} | {val:.2%} |")

            # Walk-Forward table
            if not returns.empty:
                mid     = len(returns) // 2
                is_ret  = returns.iloc[:mid]
                oos_ret = returns.iloc[mid:]
                blocks += ["", "#### Walk-Forward Returns", "",
                           "| Period | Start | End | Cumulative Return |",
                           "|--------|-------|-----|-------------------|"]
                is_cum  = float((1 + is_ret).prod() - 1)
                oos_cum = float((1 + oos_ret).prod() - 1)
                is_start  = returns.index[0].strftime("%Y-%m-%d")  if hasattr(returns.index[0],  "strftime") else str(returns.index[0])
                is_end    = returns.index[mid-1].strftime("%Y-%m-%d") if hasattr(returns.index[mid-1], "strftime") else str(returns.index[mid-1])
                oos_start = returns.index[mid].strftime("%Y-%m-%d")   if hasattr(returns.index[mid],   "strftime") else str(returns.index[mid])
                oos_end   = returns.index[-1].strftime("%Y-%m-%d")    if hasattr(returns.index[-1],    "strftime") else str(returns.index[-1])
                blocks.append(f"| In-Sample    | {is_start}  | {is_end}  | {is_cum:.2%} |")
                blocks.append(f"| Out-of-Sample| {oos_start} | {oos_end} | {oos_cum:.2%} |")

            # Return distribution (histogram bins)
            if not returns.empty:
                blocks += ["", "#### Return Distribution", "",
                           "| Return Bin | Count |",
                           "|------------|-------|"]
                hist, edges = np.histogram(returns.dropna(), bins=10)
                for i, count in enumerate(hist):
                    lo = edges[i]
                    hi = edges[i + 1]
                    blocks.append(f"| {lo:.3%} to {hi:.3%} | {count} |")

            blocks.append("")

        return "\n".join(blocks)


# ── math helpers ──────────────────────────────────────────────────────────────

def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def _advanced_metrics(returns: pd.Series, trade_log: list[dict], metrics: dict) -> dict:
    """Compute advanced metrics from returns series and trade log."""
    result: dict[str, Any] = {}

    # ── returns-based ─────────────────────────────────────────────────────────
    if returns.empty or returns.std() == 0:
        result.update({
            "sortino": 0.0, "calmar": 0.0, "cagr": 0.0, "ann_vol": 0.0,
            "var_95": 0.0, "cvar_95": 0.0, "recovery_days": 0,
            "is_sharpe": 0.0, "oos_sharpe": 0.0,
            "is_return": 0.0, "oos_return": 0.0,
        })
    else:
        ann_vol  = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        mean_ret = float(returns.mean())
        cagr     = float((1 + returns).prod() ** (TRADING_DAYS / max(len(returns), 1)) - 1)
        max_dd   = metrics.get("max_drawdown", 0.0)

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 1e-9
        sortino  = float(mean_ret / down_std * math.sqrt(TRADING_DAYS)) if down_std > 0 else 0.0

        # Calmar
        calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

        # VaR / CVaR 95%
        var_95  = float(np.percentile(returns.dropna(), 5))
        tail    = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

        # Max drawdown recovery
        equity_fake = (1 + returns).cumprod()
        dd          = _drawdown_series(equity_fake)
        in_dd       = dd < 0
        recovery_days = 0
        if in_dd.any():
            # Length of the longest drawdown streak
            max_streak = 0
            streak = 0
            for v in in_dd:
                streak = streak + 1 if v else 0
                max_streak = max(max_streak, streak)
            recovery_days = int(max_streak)

        # Walk-forward split
        mid      = len(returns) // 2
        is_ret   = returns.iloc[:mid]
        oos_ret  = returns.iloc[mid:]

        def _sharpe(r: pd.Series) -> float:
            s = r.std(ddof=1)
            return float(r.mean() / s * math.sqrt(TRADING_DAYS)) if s > 0 else 0.0

        result.update({
            "sortino":       sortino,
            "calmar":        calmar,
            "cagr":          cagr,
            "ann_vol":       ann_vol,
            "var_95":        var_95,
            "cvar_95":       cvar_95,
            "recovery_days": recovery_days,
            "is_sharpe":     _sharpe(is_ret),
            "oos_sharpe":    _sharpe(oos_ret),
            "is_return":     float((1 + is_ret).prod() - 1),
            "oos_return":    float((1 + oos_ret).prod() - 1),
        })

    # ── trade-log-based ────────────────────────────────────────────────────────
    if not trade_log:
        result.update({
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "max_consec_losses": 0, "exit_breakdown": {},
        })
    else:
        pnls    = [t["pnl"] for t in trade_log]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]
        avg_win  = float(np.mean(wins))   if wins   else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max consecutive losses
        max_cl = streak = 0
        for p in pnls:
            streak = streak + 1 if p <= 0 else 0
            max_cl = max(max_cl, streak)

        # Exit reason breakdown
        breakdown: dict[str, int] = {}
        for t in trade_log:
            r = t.get("exit_reason", "unknown")
            breakdown[r] = breakdown.get(r, 0) + 1

        result.update({
            "avg_win":          avg_win,
            "avg_loss":         avg_loss,
            "profit_factor":    profit_factor,
            "max_consec_losses": max_cl,
            "exit_breakdown":   breakdown,
        })

    return result
