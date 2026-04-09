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

    @staticmethod
    def _strip_table_pipes(md: str) -> str:
        """Convert markdown pipe tables to pipe-free aligned plain-text rows."""
        lines = md.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Detect a markdown table block: line starts with '|'
            if line.lstrip().startswith("|"):
                # Collect all consecutive table lines
                block = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    block.append(lines[i])
                    i += 1
                # Parse each row into cells, skip separator rows (---|--- pattern)
                rows = []
                for row in block:
                    cells = [c.strip() for c in row.strip().strip("|").split("|")]
                    if all(set(c.replace("-", "").replace(":", "").replace(" ", "")) == set() for c in cells):
                        continue  # separator row
                    rows.append(cells)
                if not rows:
                    continue
                # Normalise column count
                ncols = max(len(r) for r in rows)
                rows = [r + [""] * (ncols - len(r)) for r in rows]
                # Compute column widths
                widths = [max(len(r[c]) for r in rows) for c in range(ncols)]
                for row in rows:
                    out.append("  ".join(cell.ljust(widths[ci]) for ci, cell in enumerate(row)).rstrip())
            else:
                out.append(line)
                i += 1
        return "\n".join(out)

    def generate(self, pipeline_output: dict, timestamp: str | None = None) -> str:  # noqa: ARG002
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
        filename  = "ReportSummary.md"
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
            self._pairs_section(pipeline_output),
            self._baseline_section(pipeline_output),
            self._portfolio_section(pipeline_output),
            self._monte_carlo_section(pipeline_output),
            self._meta_learning_section(pipeline_output),
            self._execution_brief_section(pipeline_output),
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._strip_table_pipes("\n\n".join(sections)))

        print(f"  [Report] Full report    → {filepath}")
        return filepath

    def generate_summary(self, pipeline_output: dict, timestamp: str | None = None) -> str:  # noqa: ARG002
        """
        Generate a trader-focused summary report containing only tickers
        that passed all 3 stages (backtest → diagnostics → Monte Carlo).
        Ordered by importance to the trader. Numbers preserved for graphing.
        """
        run_date  = pipeline_output.get("run_date", datetime.today().strftime("%Y-%m-%d"))
        filename  = "TraderSummary.md"
        filepath  = os.path.join(self.output_dir, filename)

        # Build lookup maps
        # Only tickers where diagnostic truly passed (stress_test=True means diag failed
        # but Sharpe was 0–0.5; those must NOT appear in the trader summary).
        diag_passed_set = {d["ticker"] for d in pipeline_output.get("diagnostics", [])
                           if d.get("passed")}
        mc_map   = {mc["ticker"]: mc for mc in pipeline_output.get("monte_carlos", [])
                    if not mc.get("insufficient_sample")
                    and not mc.get("stress_test")
                    and mc["ticker"] in diag_passed_set}
        diag_map = {d["ticker"]: d  for d in pipeline_output.get("diagnostics", [])}
        bt_map   = {b["ticker"]: b  for b in pipeline_output.get("backtests", [])}
        strat_map= {s["ticker"]: s  for s in pipeline_output.get("strategies", [])}
        verdict_map = {v["ticker"]: v for v in pipeline_output.get("ticker_verdicts", [])}
        regime_map  = {r["ticker"]: r for r in pipeline_output.get("regimes", [])}
        features    = pipeline_output.get("features", {})
        spy_ohlcv   = pipeline_output.get("spy_ohlcv")
        macro       = pipeline_output.get("macro", {})
        summary     = pipeline_output.get("summary", {})

        # Only tickers that passed all 3 stages
        qualified = [t for t in mc_map]

        # SPY close series — kept as a Series so per-ticker sections can align
        # their comparison to the exact dates the strategy was backtested on.
        spy_close_full: "pd.Series | None" = None
        spy_return_global: float | None = None
        if spy_ohlcv is not None and not spy_ohlcv.empty:
            try:
                spy_close_full     = spy_ohlcv["Close"].astype(float)
                spy_return_global  = float(
                    (spy_close_full.iloc[-1] - spy_close_full.iloc[0])
                    / spy_close_full.iloc[0]
                )
            except Exception:
                pass

        sections = [self._summary_header(run_date, qualified, summary, macro, spy_return_global)]

        # ── Today's action ────────────────────────────────────────────────────
        sections.append(self._summary_action(qualified, strat_map, run_date))

        # ── Per-ticker deep dives ─────────────────────────────────────────────
        for ticker in qualified:
            sections.append(self._summary_ticker(
                ticker, strat_map, bt_map, diag_map, mc_map,
                verdict_map, regime_map, features, spy_close_full,
            ))

        # ── Macro context (brief, at the end) ────────────────────────────────
        sections.append(self._summary_macro(macro, summary))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._strip_table_pipes("\n\n".join(sections)))

        print(f"  [Report] Trader summary → {filepath}")
        return filepath

    # ── summary sub-sections ──────────────────────────────────────────────────

    @staticmethod
    def _summary_header(run_date, qualified, summary, macro, spy_return) -> str:
        spy_str = f" | SPY benchmark: {spy_return:.2%}" if spy_return is not None else ""
        lines = [
            f"# Trader Summary — {run_date}",
            "",
            f"**{len(qualified)} ticker(s) passed all 3 stages** (backtest → diagnostics → Monte Carlo){spy_str}  ",
            f"**Market bias:** {macro.get('market_bias', 'neutral').upper()}  ",
            f"**Favoured sectors:** {', '.join(macro.get('favored_sectors', []))}  ",
            f"**Avoid sectors:** {', '.join(macro.get('avoid_sectors', []))}  ",
            f"**Key risks:** {', '.join(summary.get('key_risks', []))}  ",
            "",
            "_This report shows only tickers that cleared backtest, diagnostic floors, "
            "and Monte Carlo stress testing. All return figures are net of 10bps slippage per side._",
        ]
        if not qualified:
            lines += ["", "**No tickers passed all 3 stages today. No action required.**"]
        return "\n".join(lines)

    @staticmethod
    def _summary_action(qualified, strat_map, run_date) -> str:
        lines = ["## Today's Action", ""]
        active_any = False
        for ticker in qualified:
            s   = strat_map.get(ticker, {})
            sig = s.get("current_signal", {})
            if sig.get("signal_active"):
                active_any = True
                setup = sig.get("setup", {})
                lines += [
                    f"### {ticker} — ENTER NOW ({s.get('strategy', '')})",
                    "",
                    f"- **Order:** Market order at next session open (~${setup.get('entry_price', 0):,.2f})",
                    f"- **Stop loss:** ${setup.get('stop_price', 0):,.2f} "
                    f"(risk ${setup.get('dollar_risk', 0):,.0f} = 1% of portfolio)",
                    f"- **Position size:** {setup.get('position_size', 0):,} shares",
                    f"- **Current ATR₁₄:** ${setup.get('current_atr', 0):,.2f}",
                    "",
                ]
        if not active_any:
            lines += [
                "_No entry signals are active today across all qualified tickers._  ",
                "_Monitor the conditions below — enter on the next session where ALL conditions are met._",
            ]
        return "\n".join(lines)

    @staticmethod
    def _summary_ticker(ticker, strat_map, bt_map, diag_map, mc_map,
                        verdict_map, regime_map, features, spy_close_full) -> str:
        s       = strat_map.get(ticker, {})
        bt      = bt_map.get(ticker, {})
        diag    = diag_map.get(ticker, {})
        mc      = mc_map.get(ticker, {})
        verdict = verdict_map.get(ticker, {})
        regime  = regime_map.get(ticker, {})
        feats   = features.get(ticker, {})
        params  = s.get("adjusted_params", {})
        sig     = s.get("current_signal", {})
        summary = bt.get("summary", {})
        metrics = diag.get("metrics", {})
        trade_log = bt.get("trade_log", [])
        equity    = bt.get("equity_curve", pd.Series(dtype=float))
        returns   = bt.get("returns", pd.Series(dtype=float))

        net_ret = summary.get("total_return", 0)

        # ── Period-aligned SPY comparison ──────────────────────────────────────
        # Align SPY to the EXACT date range of this ticker's backtest so the
        # comparison is apples-to-apples and can never show -65% for a period
        # where SPY was actually positive.
        spy_str = ""
        if spy_close_full is not None and not returns.empty:
            try:
                lo = returns.index[0]
                hi = returns.index[-1]
                spy_window = spy_close_full[
                    (spy_close_full.index >= lo) & (spy_close_full.index <= hi)
                ]
                if len(spy_window) >= 2:
                    spy_period_ret = float(
                        (spy_window.iloc[-1] - spy_window.iloc[0]) / spy_window.iloc[0]
                    )
                    alpha = net_ret - spy_period_ret
                    flag  = "outperform" if alpha >= 0 else "underperform"
                    spy_str = (
                        f" | SPY (same period): {spy_period_ret:+.2%}"
                        f" | Alpha: {alpha:+.2%} ({flag})"
                    )
            except Exception:
                pass

        lines = [
            f"---",
            f"## {ticker} — {s.get('strategy', 'N/A')} | {regime.get('regime', 'N/A')}",
            "",
        ]

        # ── 1. Signal status ──────────────────────────────────────────────────
        lines += ["### 1. Entry Signal (as of run date)", ""]
        if sig.get("signal_active") is True:
            setup = sig.get("setup", {})
            lines += [
                "**Status: ACTIVE — enter at next session open**",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Entry price | ${setup.get('entry_price', 0):,.2f} |",
                f"| Stop loss | ${setup.get('stop_price', 0):,.2f} |",
                f"| Stop distance | ${setup.get('stop_dist', 0):,.2f} |",
                f"| Position size | {setup.get('position_size', 0):,} shares |",
                f"| Dollar risk | ${setup.get('dollar_risk', 0):,.0f} (1% of portfolio) |",
                f"| Current ATR₁₄ | ${setup.get('current_atr', 0):,.2f} |",
            ]
            if setup.get("target"):
                lines.append(f"| Target (mean-reversion) | ${setup['target']:,.2f} |")
        else:
            failed = []
            if sig.get("breakout") is False:          failed.append("price breakout")
            if sig.get("volume_confirmed") is False:  failed.append("volume confirmation")
            if sig.get("oversold") is False:          failed.append("RSI oversold")
            if sig.get("below_bb") is False:          failed.append("below lower BB")
            if sig.get("squeeze_detected") is False:  failed.append("BB squeeze not detected")
            if sig.get("atr_expanding") is False:     failed.append("ATR not yet expanding")
            if sig.get("bb_breakout") is False:       failed.append("close ≤ upper Bollinger Band")
            reason = " + ".join(failed) if failed else "conditions"
            lines += [
                f"**Status: INACTIVE — {reason} not met**",
                "",
                f"```",
                sig.get("details", "N/A"),
                "```",
                "",
                "_Monitor daily. Enter at next session open when ALL conditions are met._",
            ]

        # ── 2. Why this ticker was selected ───────────────────────────────────
        lines += ["", "### 2. Why This Ticker Was Selected", ""]
        if feats:
            lines += [
                "**Screening data (OHLCV features):**",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| 20-day return | {feats.get('return_20d', 0):.2%} |",
                f"| RSI(14) | {feats.get('rsi_14', 0):.1f} |",
                f"| ATR(14) | {feats.get('atr_14', 0):.2f} |",
                f"| ATR % of price | {feats.get('atr_pct', 0):.2%} |",
                f"| 52-week high proximity | {feats.get('52w_high_prox', 0):.3f} |",
                f"| 52-week low proximity | {feats.get('52w_low_prox', 0):.3f} |",
                f"| Volume ratio (30d) | {feats.get('volume_ratio_30d', 0):.2f}× |",
                f"| Hurst exponent | {regime.get('hurst', 0):.3f} |",
                f"| Regime | {regime.get('regime', 'N/A')} |",
            ]
        lines += [
            "",
            f"**Screener verdict:** {verdict.get('verdict', 'N/A').upper()}",
            "",
            f"> {verdict.get('reasoning', 'N/A')}",
        ]

        # ── 3. Strategy rules ─────────────────────────────────────────────────
        lines += ["", "### 3. Strategy Rules", ""]
        lines += _render_mechanics(s.get("strategy", ""), params)
        if s.get("llm_adjustments"):
            lines += ["", "**Parameter adjustments made by LLM:**", ""]
            for adj in s["llm_adjustments"]:
                lines.append(f"- {adj}")

        # ── 4. Monte Carlo risk profile ───────────────────────────────────────
        lines += ["", "### 4. Monte Carlo Risk Profile (10,000 simulations)", ""]
        lines += [
            "| Metric | P5 (worst 5%) | Median | P95 (best 5%) |",
            "|--------|--------------|--------|---------------|",
            f"| Final portfolio ($) | {mc.get('p5_final', 0):,.0f} | {mc.get('p50_final', 0):,.0f} | {mc.get('p95_final', 0):,.0f} |",
            f"| CAGR | — | {mc.get('median_cagr', 0):.2%} | — |",
            f"| Sharpe ratio | {mc.get('p5_sharpe', 0):.2f} | {mc.get('p50_sharpe', 0):.2f} | {mc.get('p95_sharpe', 0):.2f} |",
            f"| Win rate | {mc.get('p5_win_rate', 0):.1%} | {mc.get('p50_win_rate', 0):.1%} | {mc.get('p95_win_rate', 0):.1%} |",
            "",
            "| Risk metric | Value |",
            "|-------------|-------|",
            f"| P(Ruin) >40% drawdown | {mc.get('p_ruin', 0):.2%} |",
            f"| P95 max drawdown | {mc.get('p95_max_drawdown', 0):.2%} |",
            f"| P95 max consecutive losses | {mc.get('p95_max_consec_losses', 0)} |",
            f"| Optimal Kelly fraction | {mc.get('kelly_fraction', 0):.3f} |",
            f"| Suggested position size (½ Kelly) | {mc.get('kelly_fraction', 0) / 2:.3f} of capital |",
            *(
                [
                    "",
                    "> **Kelly = 0 note:** Negative expectancy at the trade-sequence level — "
                    "the formula signals no provable edge. P(Ruin) can still be 0% because "
                    "the fixed 1% position sizing caps total drawdown far below the 40% ruin "
                    "floor even across many consecutive losses. Kelly = 0 is the stronger "
                    "signal: do not trade this setup until edge is demonstrated.",
                ]
                if mc.get("kelly_fraction", 0) <= 0
                else []
            ),
            "",
            "**Equity confidence band** _(for graphing: trade# vs portfolio value)_",
            "",
            "| Trade # | P5 ($) | Median ($) | P95 ($) |",
            "|---------|--------|------------|---------|",
        ]
        for entry in mc.get("equity_band", []):
            lines.append(
                f"| {entry['step']} | {entry['p5']:,.0f} | {entry['p50']:,.0f} | {entry['p95']:,.0f} |"
            )

        # ── 5. Diagnostic scorecard ────────────────────────────────────────────
        lines += ["", "### 5. Diagnostic Scorecard", ""]

        _sharpe    = metrics.get("sharpe", 0.0)
        _raw_oos   = metrics.get("oos_sharpe")
        _oos       = 0.0 if _raw_oos is None else _raw_oos
        _dd        = metrics.get("max_drawdown", 0.0)
        _wr        = metrics.get("win_rate", 0.0)
        _pf        = metrics.get("profit_factor", 0.0)
        _kelly     = metrics.get("kelly_fraction", 0.0)
        _wf        = metrics.get("walk_forward_degradation", 0.0)
        _tc        = metrics.get("trade_count", 0)

        sharpe_ok  = max(_sharpe, _oos) >= 0.50
        dd_ok      = _dd <= 0.20
        win_ok     = _wr >= 0.35 or _pf >= 1.50
        kelly_ok   = _kelly >= 0.0
        wf_ok      = _wf <= 0.50
        tc_ok      = _tc >= 30

        # Annotate when the OOS rescue mechanism saved a low full-period Sharpe
        sharpe_note = ""
        if _sharpe < 0.50 and _oos >= 0.50:
            sharpe_note = f" _(OOS rescue: {_oos:.3f})_"

        floors = [
            ("Sharpe (RF-adjusted)", f"{_sharpe:.3f}{sharpe_note}", "≥ 0.50", sharpe_ok),
            ("Max drawdown",         f"{_dd:.2%}",   "≤ 20%",          dd_ok),
            ("Win rate",             f"{_wr:.1%}",   "≥ 35% (or PF ≥ 1.5)", win_ok),
            ("Kelly fraction",       f"{_kelly:.4f}", "≥ 0.0",          kelly_ok),
            ("Walk-fwd degradation", f"{_wf:.1%}",   "≤ 50%",          wf_ok),
            ("Trade count",          str(_tc),        "≥ 30",           tc_ok),
        ]
        lines += ["| Metric | Value | Floor | Pass |", "|--------|-------|-------|------|"]
        for name, val, floor, ok in floors:
            lines.append(f"| {name} | {val} | {floor} | {'PASS' if ok else 'FAIL'} |")

        # Multi-split walk-forward detail (60/40, 70/30, 80/20)
        wf_splits = metrics.get("wf_splits", [])
        wf_underpowered = metrics.get("wf_underpowered", False)
        if wf_underpowered:
            lines += [
                "",
                "> WARNING: **Walk-forward underpowered** — fewer than 100 trades; "
                "split results are not statistically meaningful and the WF gate was not applied.",
            ]
        elif wf_splits and not returns.empty:
            is_rolling = any(sp.get("rolling_wf") for sp in wf_splits)
            if is_rolling:
                lines += ["", "**Walk-forward (rolling anchored — requires ≥50% windows with OOS Sharpe > 0):**", ""]
            else:
                lines += ["", "**Walk-forward (3-split robustness: 60/40, 70/30, 80/20 — requires 2/3 passes):**", ""]
            lines += [
                "| IS/OOS | IS Sharpe | OOS Sharpe | Degradation | Pass |",
                "|--------|-----------|------------|-------------|------|",
            ]
            for sp in wf_splits:
                is_p  = int(sp.get("is_pct", 0) * 100)
                oos_p = 100 - is_p
                tick  = "PASS" if sp.get("passed") else "FAIL"
                # None sentinel = underpowered — render as N/A
                _is_s  = sp.get("is_sharpe")
                _oos_s = sp.get("oos_sharpe")
                _degrad = sp.get("degradation")
                is_str   = "N/A" if _is_s  is None else f"{_is_s:.3f}"
                oos_str  = "N/A" if _oos_s is None else f"{_oos_s:.3f}"
                deg_str  = "N/A" if _degrad is None else f"{_degrad:.1%}"
                lines.append(f"| {is_p}/{oos_p} | {is_str} | {oos_str} | {deg_str} | {tick} |")
            n_pass = sum(1 for sp in wf_splits if sp.get("passed"))
            n_total = len(wf_splits)
            lines.append(f"")
            if is_rolling:
                lines.append(f"**{n_pass}/{n_total} windows passed** ({'robust' if n_pass >= n_total // 2 else 'fragile — likely IS overfit'})")
            else:
                lines.append(f"**{n_pass}/3 splits passed** ({'robust' if n_pass >= 2 else 'fragile — likely IS overfit'})")

        # ── 6. Backtest performance ────────────────────────────────────────────
        slip_bps = bt.get("slippage_bps", 10.0)
        lines += ["", "### 6. Backtest Performance (10 years)", ""]
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Net return (after slippage) | {net_ret:.2%}{spy_str} |",
            f"| Gross return (pre-cost) | {summary.get('gross_return', 0):.2%} |",
            f"| Total slippage cost | ${summary.get('total_slippage_cost', 0):,.2f} |",
            f"| Slippage rate used | {slip_bps:.0f}bps/side (ADV-tiered) |",
            f"| Trade count | {summary.get('trade_count', 0)} |",
            f"| Win rate | {summary.get('win_rate', 0):.1%} |",
            "",
        ]

        # Param divergence warnings — alert when live params deviate from backtest params
        param_div = s.get("param_divergence_warnings", [])
        if param_div:
            lines += [
                "> WARNING: **Parameter divergence detected** — live signal uses different "
                "parameters than the validated backtest:",
                "",
            ]
            for w in param_div:
                lines.append(f"> - {w}")
            lines.append("")

        # Trade log
        if trade_log:
            lines += [
                "**Trade log** _(entry date, entry $, exit date, exit $, days held, P&L net)_",
                "",
                "| Entry | Entry $ | Exit | Exit $ | Days | P&L ($) | Reason |",
                "|-------|---------|------|--------|------|---------|--------|",
            ]
            for t in trade_log:
                ed = str(t["entry_date"])[:10]
                xd = str(t["exit_date"])[:10]
                lines.append(
                    f"| {ed} | {t['entry_price']:.2f} | {xd} | {t['exit_price']:.2f}"
                    f" | {t['holding_days']} | {t['pnl']:+.2f} | {t['exit_reason']} |"
                )

        # Equity curve (sampled, for graphing)
        if not equity.empty:
            lines += [
                "",
                "**Equity curve** _(for graphing: date vs portfolio value)_",
                "",
                "| Date | Portfolio ($) |",
                "|------|--------------|",
            ]
            step = max(1, len(equity) // 30)
            for date, val in equity.iloc[::step].items():
                lines.append(f"| {str(date)[:10]} | {val:,.2f} |")

        # Drawdown curve (sampled, for graphing)
        if not equity.empty:
            dd_series = _drawdown_series(equity)
            lines += [
                "",
                "**Drawdown curve** _(for graphing: date vs drawdown %)_",
                "",
                "| Date | Drawdown |",
                "|------|----------|",
            ]
            step = max(1, len(dd_series) // 30)
            for date, val in dd_series.iloc[::step].items():
                lines.append(f"| {str(date)[:10]} | {val:.4f} |")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _summary_macro(macro, summary) -> str:
        lines = [
            "## Macro Context",
            "",
            f"**Bias:** {macro.get('market_bias', 'neutral').upper()}  ",
            f"**Favoured:** {', '.join(macro.get('favored_sectors', []))}  ",
            f"**Avoid:** {', '.join(macro.get('avoid_sectors', []))}  ",
            f"**Risks:** {', '.join(macro.get('active_macro_risks', []))}  ",
            "",
            f"> {macro.get('reasoning', '')}",
            "",
            f"**Top news themes:** {', '.join(summary.get('top_themes', []))}  ",
            f"**Articles analysed:** {summary.get('article_count', 0)}  ",
            f"**Window:** {summary.get('window_start', 'N/A')} → {summary.get('window_end', 'N/A')}  ",
        ]
        return "\n".join(lines)

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

        # Portfolio-level risk from active signals — only tickers that PASSED diagnostics
        strategies   = po.get("strategies", [])
        diag_passed  = {d["ticker"] for d in po.get("diagnostics", []) if d.get("passed", False)}
        active_sigs  = [s for s in strategies
                        if s.get("current_signal", {}).get("signal_active")
                        and s["ticker"] in diag_passed]
        n_active     = len(active_sigs)
        total_risk   = sum(
            s["current_signal"]["setup"].get("dollar_risk", 0.0)
            for s in active_sigs
            if s["current_signal"].get("setup")
        )
        portfolio    = 100_000.0   # default; real value embedded in setup sizing
        risk_pct     = (total_risk / portfolio * 100) if portfolio else 0.0

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

        # Prediction markets block — top 5 by volume
        markets = po.get("markets", [])
        if markets:
            top5 = sorted(markets, key=lambda m: m.get("volume", 0), reverse=True)[:5]
            lines += ["", "### Key Prediction Markets", ""]
            lines += [
                "| Event | Probability | Volume |",
                "|-------|-------------|--------|",
            ]
            for m in top5:
                prob = f"{int(round(m.get('probability', 0) * 100))}%"
                vol  = m.get("formatted_text", "").split("Volume: ")[-1].split(" |")[0] if "Volume:" in m.get("formatted_text", "") else "N/A"
                lines.append(f"| {m['event']} | {prob} | {vol} |")

        # Portfolio risk block — only shown when there are active signals
        if n_active > 0:
            lines += [
                "",
                "### Portfolio Risk (Active Signals Today)",
                "",
                f"**{n_active} signal{'s' if n_active != 1 else ''} active** — "
                f"if all are entered simultaneously:",
                "",
                f"- Total capital at risk: **${total_risk:,.0f}** "
                f"({risk_pct:.1f}% of a $100k portfolio)",
                f"- Each position risks ~$1,000 (1% rule); "
                f"{'within normal diversification limits.' if risk_pct <= 5 else 'consider staggering entries to avoid overexposure.'}",
            ]

        return "\n".join(lines)

    @staticmethod
    def _macro_section(po: dict) -> str:
        m       = po.get("macro", {})
        markets = po.get("markets", [])
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
        if markets:
            sorted_mkts = sorted(markets, key=lambda x: x.get("volume", 0), reverse=True)
            lines += [
                "",
                "### Prediction Markets",
                "",
                "| Status | Category | Event | Probability | Volume |",
                "|--------|----------|-------|-------------|--------|",
            ]
            for mk in sorted_mkts:
                prob   = f"{int(round(mk.get('probability', 0) * 100))}%"
                status = mk.get("status", "active").upper()
                vol    = mk.get("formatted_text", "").split("Volume: ")[-1].split(" |")[0] if "Volume:" in mk.get("formatted_text", "") else "N/A"
                lines.append(
                    f"| {status} | {mk.get('category', '')} "
                    f"| {mk['event']} | {prob} | {vol} |"
                )
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
            "| Ticker | Regime | Hurst | ATR/Price | Near Earnings |",
            "|--------|--------|-------|-----------|---------------|",
        ]
        for r in regimes:
            earnings_tag = "Yes" if r.get("near_earnings") else "No"
            lines.append(
                f"| {r['ticker']} | {r['regime']} "
                f"| {r['hurst']:.3f} | {r['atr_pct']:.2%} | {earnings_tag} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _strategy_section(po: dict) -> str:
        strategies  = po.get("strategies", [])
        diag_passed = {d["ticker"] for d in po.get("diagnostics", []) if d.get("passed", False)}
        blocks = ["## Strategy Parameters", ""]
        for s in strategies:
            params = s.get("adjusted_params", {})
            adj    = s.get("llm_adjustments", [])
            sig    = s.get("current_signal", {})
            blocks += [
                f"### {s['ticker']} — {s['strategy']}",
                "",
                f"**Regime:** {s.get('regime', 'N/A')}  ",
                f"**Reasoning:** {s.get('reasoning', '')}",
                "",
                "#### Strategy Mechanics",
                "",
            ]
            blocks += _render_mechanics(s["strategy"], params)
            blocks += [
                "",
                "#### Adjusted Parameters",
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
            # LLM alpha hypothesis — show when LLM disagrees with regime rule.
            # This is PRE-BACKTEST opinion from StrategySelector — the LLM sees
            # only the regime label and OHLCV features, not the actual backtest returns.
            # Contrast with "LLM Diagnostic Commentary" in the Diagnostic Results section,
            # which is POST-BACKTEST and has access to the full realized P&L metrics.
            hyp = s.get("llm_hypothesis", {})
            if hyp and not hyp.get("agree", True) and hyp.get("suggested"):
                blocks += [
                    "",
                    f"> **LLM Alpha Hypothesis (pre-backtest, StrategySelector):** "
                    f"Disagrees with regime-rule selection. "
                    f"Suggests **{hyp['suggested']}** instead."
                    + (f" Reason: _{hyp['reason']}_" if hyp.get("reason") else ""),
                ]

            # Current signal status
            blocks += ["", "#### Current Entry Signal (as of run date)", ""]
            if sig.get("signal_active") is True:
                blocks.append("**Status: ACTIVE — entry condition met on latest bar**")
            elif sig.get("signal_active") is False:
                # Show exactly which condition(s) failed — handles all three strategy types
                failed = []
                if sig.get("breakout") is False:
                    failed.append("price breakout")
                if sig.get("volume_confirmed") is False:
                    failed.append("volume confirmation")
                if sig.get("oversold") is False:
                    failed.append("RSI oversold")
                if sig.get("below_bb") is False:
                    failed.append("below lower BB")
                if sig.get("squeeze_detected") is False:
                    failed.append("BB squeeze not detected")
                if sig.get("atr_expanding") is False:
                    failed.append("ATR not yet expanding")
                if sig.get("bb_breakout") is False:
                    failed.append("close ≤ upper Bollinger Band")
                reason = " + ".join(failed) if failed else "entry condition"
                blocks.append(f"**Status: INACTIVE — {reason} not met**")
            else:
                blocks.append("**Status: N/A**")
            if sig.get("details"):
                blocks.append(f"```\n{sig['details']}\n```")
            # Trade setup — only shown when signal is active AND diagnostics passed.
            # A signal may fire on a failed strategy (backtest has no edge); showing
            # the setup in that case would invite a trader to enter a losing strategy.
            setup = sig.get("setup")
            ticker_passed = s["ticker"] in diag_passed
            if setup and not ticker_passed:
                blocks += [
                    "",
                    "> NOTE: **Trade setup suppressed — this ticker FAILED diagnostic floors.**  ",
                    "> The entry signal fired but the backtest has no demonstrated edge. Do not trade.",
                ]
            elif setup and ticker_passed:
                blocks += [
                    "",
                    "#### Trade Setup",
                    "",
                    "| | Value |",
                    "|---|-------|",
                    f"| Suggested entry | Market order at next open (~${setup['entry_price']:,.2f}) |",
                    f"| Stop loss | ${setup['stop_price']:,.2f}  "
                    f"(entry − {params.get('stop_loss_atr', '?')} × ATR₁₄ ${setup['current_atr']:,.2f}) |",
                    f"| Stop distance | ${setup['stop_dist']:,.2f} |",
                    f"| Position size | {setup['position_size']:,} shares |",
                    f"| Dollar risk | ${setup['dollar_risk']:,.0f}  (1% of portfolio) |",
                    f"| Current ATR₁₄ | ${setup['current_atr']:,.2f} |",
                ]
                _slip_per_share = setup['entry_price'] * 0.0015
                _slip_total     = _slip_per_share * setup['position_size']
                _adj_risk       = setup['dollar_risk'] + _slip_total
                blocks += [
                    f"| Est. slippage (~0.15% of price) | ${_slip_per_share:.4f}/share → ${_slip_total:,.2f} total |",
                    f"| Adjusted net risk (incl. slippage) | ${_adj_risk:,.0f} |",
                ]
                if setup.get("current_ma"):
                    ma_label = f"{params.get('ma_exit_period', '?')}-day MA" if s["strategy"] == "Momentum" else f"{params.get('bb_period', '?')}-day SMA (middle BB)"
                    blocks.append(f"| {ma_label} | ${setup['current_ma']:,.2f} |")
                if setup.get("target"):
                    pot = setup.get("potential_gain", 0)
                    blocks.append(
                        f"| Target (mean-reversion) | ${setup['target']:,.2f}  "
                        f"(potential gain ~${pot:,.0f}) |"
                    )
            blocks.append("")
        return "\n".join(blocks)

    def _diagnostic_section(self, po: dict) -> str:
        diagnostics = po.get("diagnostics", [])
        backtests   = {b["ticker"]: b for b in po.get("backtests", [])}
        blocks = ["## Diagnostic Results", ""]

        for d in diagnostics:
            ticker  = d["ticker"]
            passed  = d.get("passed", False)
            status  = "PASS" if passed else "FAIL"
            reject  = d.get("reject_reason") or "—"
            metrics = d.get("metrics", {})
            bt        = backtests.get(ticker, {})
            returns   = bt.get("returns",     pd.Series(dtype=float))
            equity    = bt.get("equity_curve", pd.Series(dtype=float))
            trade_log = bt.get("trade_log", [])

            adv = _advanced_metrics(returns, trade_log, metrics, equity_curve=equity)

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
                f"| Entry Efficiency (% reaching +1R) | {bt.get('summary', {}).get('entry_efficiency', 0):.1%} |",
                f"| Avg Win | {adv['avg_win']:.2f} |",
                f"| Avg Loss | {adv['avg_loss']:.2f} |",
                f"| Payoff Ratio (avg_win / avg_loss) | {bt.get('summary', {}).get('payoff_ratio', adv['avg_win'] / max(abs(adv['avg_loss']), 1e-6)):.3f} |",
                f"| Avg Holding Days | {bt.get('summary', {}).get('avg_holding_days', 0):.1f} |",
                f"| Profit Factor | {adv['profit_factor']:.3f} |",
                f"| Max Consecutive Losses | {adv['max_consec_losses']} |",
                _fmt_exposure_row(metrics),
                "",
                "#### Alpha Quality Diagnostics",
                "",
                "| Metric | Value | Interpretation |",
                "|--------|-------|----------------|",
                _fmt_perm_pvalue_row(metrics),
                f"| Rolling Sharpe (% positive windows) | {metrics.get('rolling_pct_positive', float('nan')):.1%} | {'Regime-dependent' if metrics.get('rolling_pct_positive', 1.0) < 0.50 else 'Consistent'} |" if not math.isnan(metrics.get('rolling_pct_positive', float('nan'))) else "| Rolling Sharpe (% positive windows) | N/A | Insufficient data |",
                f"| Rolling Sharpe Std Dev | {metrics.get('rolling_sharpe_std', float('nan')):.3f} | Lower = more stable |" if not math.isnan(metrics.get('rolling_sharpe_std', float('nan'))) else "| Rolling Sharpe Std Dev | N/A | Insufficient data |",
                "",
                "#### Walk-Forward Validation",
                "",
                "| Period | Sharpe | Total Return |",
                "|--------|--------|--------------|",
                f"| In-Sample | {adv['is_sharpe']:.3f} | {adv['is_return']:.2%} |",
                f"| Out-of-Sample | {adv['oos_sharpe']:.3f} | {adv['oos_return']:.2%} |",
                _fmt_degradation_row(metrics),
                "",
                "#### Exit Reason Breakdown",
                "",
                "| Exit Reason | Count |",
                "|-------------|-------|",
            ]
            for reason, count in adv["exit_breakdown"].items():
                blocks.append(f"| {reason} | {count} |")

            if d.get("llm_commentary"):
                blocks += [
                    "",
                    f"> **LLM Diagnostic Commentary (post-backtest, from DiagnosticsEngine):** "
                    f"{d['llm_commentary']}",
                ]

            blocks.append("")

        return "\n".join(blocks)

    def _backtest_section(self, po: dict) -> str:
        backtests  = po.get("backtests", [])
        spy_ohlcv  = po.get("spy_ohlcv")
        corr_warns = po.get("correlation_warnings", [])
        blocks     = ["## Backtest Results", ""]

        # Correlation warnings
        if corr_warns:
            blocks += ["### Concentration Risk Warnings", ""]
            for w in corr_warns:
                blocks.append(f"- {w}")
            blocks.append("")

        # SPY full buy-and-hold return (context only)
        spy_bnh: float | None = None
        spy_close_full: "pd.Series | None" = None
        if spy_ohlcv is not None and not spy_ohlcv.empty:
            try:
                spy_close_full = spy_ohlcv["Close"].astype(float)
                spy_bnh = float((spy_close_full.iloc[-1] - spy_close_full.iloc[0]) / spy_close_full.iloc[0])
            except Exception:
                spy_bnh = None

        if spy_bnh is not None:
            blocks += [
                f"**SPY Buy-and-Hold (full window):** {spy_bnh:.2%}  ",
                "_Note: each ticker also shows an exposure-adjusted SPY return — SPY compounded only on days the strategy was invested. This is the fair apples-to-apples comparison._",
                "",
            ]

        for bt in backtests:
            ticker    = bt["ticker"]
            summary   = bt.get("summary", {})
            trade_log = bt.get("trade_log", [])
            equity    = bt.get("equity_curve", pd.Series(dtype=float))
            returns   = bt.get("returns", pd.Series(dtype=float))
            in_pos    = bt.get("in_position", pd.Series(dtype=bool))

            net_ret   = summary.get("total_return", 0)
            gross_ret = summary.get("gross_return", 0)
            slip_cost = summary.get("total_slippage_cost", 0)

            # Exposure-adjusted SPY: compound SPY only on days strategy was invested
            spy_exp_adj: float | None = None
            if spy_close_full is not None and len(in_pos) > 0:
                try:
                    spy_dr      = spy_close_full.pct_change(fill_method=None).fillna(0.0)
                    common      = in_pos.index.intersection(spy_dr.index)
                    in_pos_c    = in_pos.reindex(common, fill_value=False)
                    spy_dr_c    = spy_dr.reindex(common, fill_value=0.0)
                    invested_r  = spy_dr_c[in_pos_c]
                    pct_invested = float(in_pos_c.sum()) / max(len(in_pos_c), 1)
                    if len(invested_r) > 0:
                        spy_exp_adj = float((1 + invested_r).prod() - 1)
                except Exception:
                    spy_exp_adj = None

            vs_spy = ""
            if spy_exp_adj is not None:
                diff   = net_ret - spy_exp_adj
                vs_spy = f"  **vs SPY (exposure-adj): {diff:+.2%}** ({'outperform' if diff >= 0 else 'underperform'})"
            elif spy_bnh is not None:
                diff   = net_ret - spy_bnh
                vs_spy = f"  **vs SPY B&H: {diff:+.2%}**"

            blocks += [
                f"### {ticker} — {bt['strategy']}",
                "",
                f"**Net Return (after slippage):** {net_ret:.2%}{vs_spy}  ",
                f"**Gross Return (pre-cost):** {gross_ret:.2%}  ",
                f"**Total Slippage Cost:** ${slip_cost:,.2f}  ",
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

            # Walk-Forward table (70/30 IS/OOS split — industry standard)
            # Use the equity curve (not returns) to avoid RF-on-flat-days inflation:
            # _build_returns() sets flat days to DAILY_RF so (1+r).prod()-1 compounds
            # the risk-free rate on every uninvested day, overstating cumulative return
            # vs what the equity curve actually shows.
            if not equity.empty:
                split     = int(len(equity) * 0.70)
                split     = max(split, 1)
                is_cum    = float(equity.iloc[split] / equity.iloc[0] - 1)
                oos_cum   = float(equity.iloc[-1]   / equity.iloc[split] - 1)
                is_start  = equity.index[0].strftime("%Y-%m-%d")       if hasattr(equity.index[0],       "strftime") else str(equity.index[0])
                is_end    = equity.index[split - 1].strftime("%Y-%m-%d") if hasattr(equity.index[split-1], "strftime") else str(equity.index[split-1])
                oos_start = equity.index[split].strftime("%Y-%m-%d")    if hasattr(equity.index[split],   "strftime") else str(equity.index[split])
                oos_end   = equity.index[-1].strftime("%Y-%m-%d")       if hasattr(equity.index[-1],      "strftime") else str(equity.index[-1])
                blocks += ["", "#### Walk-Forward Returns (70% IS / 30% OOS)", "",
                           "| Period | Start | End | Cumulative Return |",
                           "|--------|-------|-----|-------------------|"]
                blocks.append(f"| In-Sample (70%)     | {is_start}  | {is_end}  | {is_cum:.2%} |")
                blocks.append(f"| Out-of-Sample (30%) | {oos_start} | {oos_end} | {oos_cum:.2%} |")

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

    @staticmethod
    def _baseline_section(po: dict) -> str:
        """
        Compare every backtested strategy against two simple baselines:
          1. SPY buy-and-hold (full window, fully invested)
          2. SPY 50-day MA cross (long when SPY > 50d MA, else flat — earns RF)

        Showing baselines forces honest evaluation: a strategy with Sharpe 0.6
        and 8% net return looks very different if the baseline is SPY at 24%.
        A strategy that can't beat a simple MA cross has no demonstrated alpha.
        """
        spy_ohlcv  = po.get("spy_ohlcv")
        backtests  = po.get("backtests", [])
        diag_map   = {d["ticker"]: d for d in po.get("diagnostics", [])}

        if spy_ohlcv is None or spy_ohlcv.empty:
            return "## Baseline Comparison\n\n_SPY data unavailable — baselines cannot be computed._"

        try:
            spy_close  = spy_ohlcv["Close"].squeeze().astype(float)
            spy_daily  = spy_close.pct_change(fill_method=None).fillna(0.0)
        except Exception:
            return "## Baseline Comparison\n\n_SPY data malformed._"

        # ── Baseline 1: SPY buy-and-hold ──────────────────────────────────────
        spy_bnh_ret = float(np.array(spy_close.iloc[-1]).flat[0] / np.array(spy_close.iloc[0]).flat[0] - 1)
        spy_bnh_sharpe = _sharpe_from_returns(spy_daily)

        # ── Baseline 2: SPY 50-day MA cross ───────────────────────────────────
        spy_ma50    = spy_close.rolling(50).mean()
        in_pos_mask = (spy_close > spy_ma50).shift(1).fillna(False)
        # Flat days earn RF (same treatment as backtester for apples-to-apples)
        daily_rf    = 0.045 / 252
        ma_rets     = spy_daily.where(in_pos_mask, daily_rf)
        spy_ma_ret    = float((1 + ma_rets).prod() - 1)
        spy_ma_sharpe = _sharpe_from_returns(ma_rets)

        lines = [
            "## Baseline Comparison",
            "",
            "> Every strategy must be judged against what a passive, zero-effort investor would earn.",
            "> A strategy that underperforms SPY buy-and-hold provides negative alpha even with a",
            "> positive return. A strategy that underperforms a simple MA cross has no edge over",
            "> basic trend-following.",
            "",
            "### Reference Baselines",
            "",
            "| Baseline | Total Return | Sharpe | Notes |",
            "|----------|-------------|--------|-------|",
            f"| SPY buy-and-hold | {spy_bnh_ret:.2%} | {spy_bnh_sharpe:.3f} | Fully invested, full window |",
            f"| SPY 50d MA cross | {spy_ma_ret:.2%} | {spy_ma_sharpe:.3f} | Long when SPY > 50d MA, flat (earns RF) otherwise |",
            "",
            "### Strategy vs Baselines",
            "",
            "_Both 'vs' columns show net-return difference (strategy − baseline)._",
            "_A positive value means the strategy returned more; Sharpe is shown separately for quality context._",
            "",
            "| Ticker | Strategy | Net Return | Sharpe | vs SPY B&H (return) | vs 50d MA cross (return) |",
            "|--------|----------|------------|--------|---------------------|--------------------------|",
        ]

        for bt in backtests:
            t       = bt["ticker"]
            summary = bt.get("summary", {})
            net_ret = summary.get("total_return", 0.0)
            diag    = diag_map.get(t, {})
            sharpe  = diag.get("metrics", {}).get("sharpe", 0.0)

            bnh_diff = net_ret - spy_bnh_ret
            ma_diff  = net_ret - spy_ma_ret   # both columns now use return diff

            bnh_icon = "PASS" if bnh_diff >= 0 else "FAIL"
            ma_icon  = "PASS" if ma_diff  >= 0 else "FAIL"

            lines.append(
                f"| {t} | {bt['strategy']} | {net_ret:.2%} | {sharpe:.3f}"
                f" | {bnh_icon} {bnh_diff:+.2%}"
                f" | {ma_icon} {ma_diff:+.2%} |"
            )

        return "\n".join(lines)

    @staticmethod
    def _portfolio_section(po: dict) -> str:
        """
        Portfolio construction results:
         - Cross-sectional momentum ranking (12-1 month)
         - Volatility-parity allocation weights
         - Correlation-adjusted final weights
         - Portfolio-level Sharpe, Vol, VaR, CVaR, MaxDD
         - Per-ticker t-stat and bootstrap Sharpe CI from diagnostics
        """
        pr = po.get("portfolio_result") or {}
        if not pr:
            return "## Portfolio Construction\n\n_Portfolio optimiser did not run._"

        lines = ["## Portfolio Construction", ""]

        # ── CS Momentum Ranking ──────────────────────────────────────────────
        cs_ranks = pr.get("cs_momentum_ranks", [])
        if cs_ranks:
            lines += ["### Cross-Sectional Momentum Ranking (12-1 Month)", ""]
            lines.append("| Rank | Ticker | 12-1m Return | Status |")
            lines.append("|------|--------|-------------|--------|")
            alloc_tickers = {a["ticker"] for a in pr.get("allocations", [])}
            rej_tickers   = {r["ticker"] for r in pr.get("rejected", [])}
            for r in cs_ranks:
                ticker = r["ticker"]
                mom    = r.get("mom_12_1", 0.0)
                if ticker in alloc_tickers:
                    status = "Allocated"
                elif ticker in rej_tickers:
                    status = "Filtered"
                else:
                    status = "—"
                lines.append(f"| {r['rank']} | {ticker} | {mom:+.1%} | {status} |")
            lines.append("")

        # ── Allocations ──────────────────────────────────────────────────────
        allocations = pr.get("allocations", [])
        if allocations:
            lines += ["### Volatility-Parity Allocations", ""]
            lines.append("| Ticker | Weight | $ Allocated | Sharpe | CS Rank | Rationale |")
            lines.append("|--------|--------|-------------|--------|---------|-----------|")
            for a in allocations:
                lines.append(
                    f"| {a['ticker']} | {a['weight']:.1%} | "
                    f"${a['dollar_allocation']:,.0f} | {a['sharpe']:.3f} | "
                    f"{a['cs_rank']} | {a['rationale']} |"
                )
            lines.append("")
        else:
            lines.append("_No tickers passed portfolio construction filters._\n")

        # ── Rejected ─────────────────────────────────────────────────────────
        rejected = pr.get("rejected", [])
        if rejected:
            lines += ["### Rejected by Portfolio Filter", ""]
            for r in rejected:
                lines.append(f"- **{r['ticker']}**: {r['reason']}")
            lines.append("")

        # ── Portfolio Metrics ─────────────────────────────────────────────────
        pm = pr.get("portfolio_metrics") or {}
        if pm and pm.get("sharpe", 0) != 0:
            lines += ["### Portfolio-Level Risk Metrics", ""]
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Portfolio Sharpe | {pm.get('sharpe', 0):.3f} |")
            lines.append(f"| Annualised Vol   | {pm.get('annual_vol', 0):.1%} |")
            lines.append(f"| VaR (95%)        | {pm.get('var_95', 0):.2%} |")
            lines.append(f"| CVaR (95%)       | {pm.get('cvar_95', 0):.2%} |")
            lines.append(f"| Max Drawdown     | {pm.get('max_drawdown', 0):.1%} |")
            lines.append("")

        # ── Per-Ticker Statistical Significance ──────────────────────────────
        diagnostics = po.get("diagnostics", [])
        has_stats = any(
            d.get("metrics", {}).get("t_stat") is not None for d in diagnostics
        )
        if has_stats:
            lines += ["### Statistical Significance (per Ticker)", ""]
            lines.append("_t-stat uses Lo (2002) autocorrelation correction. "
                          "Bootstrap CI is 90% block-bootstrap (block=20 days). "
                          "p-val < 0.05 = statistically significant at 95% confidence. "
                          "Rolling stable = % of 60-day windows with positive Sharpe ≥ 50%. "
                          "Perm p-val = Calmar-based permutation test (lower = more order-dependent return path)._")
            lines.append("")
            lines.append("| Ticker | Sharpe | t-stat | p-value | Bootstrap 90% CI | Rolling Stable? | Perm p-val | Significant? |")
            lines.append("|--------|--------|--------|---------|------------------|-----------------|------------|--------------|")
            for d in diagnostics:
                m      = d.get("metrics", {})
                ticker = d.get("ticker", "?")
                sharpe = m.get("sharpe", 0)
                t_stat = m.get("t_stat", 0)
                p_val  = m.get("p_value", 1)
                bs_p5  = m.get("bootstrap_sharpe_p5", 0)
                bs_p95 = m.get("bootstrap_sharpe_p95", 0)
                sig    = "PASS" if p_val < 0.05 else ("WARNING" if p_val < 0.10 else "FAIL")
                roll_pct = m.get("rolling_pct_positive")
                perm_p   = m.get("permutation_p_value")
                roll_str = f"{'PASS' if roll_pct >= 0.50 else 'WARNING'} {roll_pct:.0%}" if roll_pct is not None and not math.isnan(roll_pct) else "N/A"
                if perm_p is None or (isinstance(perm_p, float) and math.isnan(perm_p)):
                    perm_str = "N/A"
                elif perm_p < 0.10:
                    perm_str = f"{perm_p:.3f} [TEMPORAL EDGE]"
                elif perm_p < 0.30:
                    perm_str = f"{perm_p:.3f} [weak temporal]"
                elif perm_p <= 0.70:
                    perm_str = f"{perm_p:.3f} [IID — expected]"
                else:
                    perm_str = f"{perm_p:.3f} [WARNING: exits destroying value]"
                lines.append(
                    f"| {ticker} | {sharpe:.3f} | {t_stat:.2f} | {p_val:.3f} | "
                    f"[{bs_p5:.2f}, {bs_p95:.2f}] | {roll_str} | {perm_str} | {sig} |"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _meta_learning_section(po: dict) -> str:
        """
        Surface historical regime+strategy performance from verdict_log.csv.
        Helps the trader understand which combinations have shown real edge
        over prior runs versus which are historically weak.
        After 30+ runs this becomes a reliable regime/strategy performance heatmap.
        """
        meta = po.get("meta_insights", {})

        if meta.get("sample_too_small"):
            n = meta.get("total_runs", 0)
            return (
                "## Historical Alpha Learning\n\n"
                f"_Only {n} historical runs recorded. A minimum of 10 is needed to compute "
                "per-regime/strategy statistics. Run the pipeline on more dates to accumulate "
                "the performance database._"
            )

        insights  = meta.get("insights", {})
        warnings  = meta.get("warnings", [])
        total_runs = meta.get("total_runs", 0)

        if not insights:
            return (
                "## Historical Alpha Learning\n\n"
                f"_{total_runs} total runs — no single regime/strategy combination has ≥5 observations yet._"
            )

        lines = [
            "## Historical Alpha Learning",
            "",
            f"_Based on {total_runs} pipeline runs stored in `data/verdict_log.csv`. "
            "Combinations with <5 observations are excluded._",
            "",
        ]

        if warnings:
            lines += ["### Weak Combinations (historically < 10% pass rate)", ""]
            for w in warnings:
                lines.append(f"> FAIL {w}")
            lines.append("")

        lines += [
            "### Per-Regime/Strategy Performance History",
            "",
            "| Regime + Strategy | Observations | Avg Sharpe | Pass Rate |",
            "|-------------------|-------------|------------|-----------|",
        ]

        for key, stats in sorted(insights.items(), key=lambda x: -x[1]["avg_sharpe"]):
            pr_icon = "PASS" if stats["pass_rate"] >= 0.3 else ("WARNING" if stats["pass_rate"] >= 0.1 else "FAIL")
            lines.append(
                f"| {key} | {stats['n']} | {stats['avg_sharpe']:.3f} | "
                f"{pr_icon} {stats['pass_rate']:.0%} |"
            )

        return "\n".join(lines)

    @staticmethod
    def _execution_brief_section(po: dict) -> str:
        """
        Execution brief for ALL qualified tickers — both active (enter now) and
        pending (enter when conditions met).  The trader needs this information
        regardless of whether the signal is firing today, because the brief tells
        them exactly how to size and execute the trade when it does fire.
        """
        brief    = po.get("execution_brief", {})
        mkt      = brief.get("market_status", {})
        active   = brief.get("active_signals", [])
        pending  = brief.get("pending_signals", [])
        p_risk   = brief.get("portfolio_risk", {})
        warnings = brief.get("warnings", [])

        lines = [
            "## Execution Brief",
            "",
            f"**NYSE:** {mkt.get('label', 'N/A')} — {mkt.get('detail', '')}  ",
        ]

        if warnings:
            lines += ["", "**Warnings:**"]
            for w in warnings:
                lines.append(f"- WARNING: {w}")

        def _render_brief(b: dict, active_signal: bool) -> list[str]:
            status = "ENTER NOW" if active_signal else "PENDING — conditions not yet met"
            out = [
                "",
                f"### {b['ticker']} — {status}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Entry price | ${b['entry_price']:,.2f} |" if b.get("entry_price") else "| Entry price | — |",
                f"| Stop loss | ${b['stop_price']:,.2f} |" if b.get("stop_price") else "| Stop loss | — |",
                f"| Position size | {b['position_size']:,} shares |",
                f"| Dollar risk (1% rule) | ${b['dollar_risk']:,.0f} |",
                f"| Slippage est. | ${b['slippage_per_share']:.4f}/share → ${b['slippage_total']:,.2f} total |",
                f"| Adjusted net risk | ${b['adjusted_risk']:,.0f} |",
                f"| Market impact | {b['market_impact']} |",
                f"| ADV (20d) | {int(b.get('_adv', 0)):,} shares |" if b.get("_adv") else "",
            ]
            if b.get("target"):
                out.append(f"| Mean-reversion target | ${b['target']:,.2f} |")
            if not active_signal:
                out += ["", "**Conditions to watch (enter when ALL are met):**", ""]
                trig = b.get("entry_trigger")
                is_str_trig = isinstance(trig, str)

                # String triggers (AlphaCombined / MLSignal / EventDriven)
                if is_str_trig:
                    # MLSignal: "ml_signal > 0.60"
                    if "ml_signal" in trig:
                        out.append(f"- Entry condition: **{trig}** (ensemble ML probability threshold)")
                        if b.get("ml_signal") is not None:
                            out.append(f"- Current ml_signal: **{b['ml_signal']:.3f}** (needs to exceed threshold)")
                    # EventDriven: contains "pead_signal"
                    elif "pead_signal" in trig:
                        out.append(f"- Entry condition: **{trig}**")
                        out.append(f"- Close must be **above the {b.get('ma_filter_period', 5)}-day MA** (drift intact)")
                        if b.get("volume_needed") is not None:
                            out.append(f"- Volume must exceed **{b['volume_needed']:,.0f} shares** (participation confirmed)")
                        out.append(f"- Must be **outside earnings blackout window**")
                        if b.get("pead_signal") is not None:
                            out.append(f"- Current pead_signal: **{b['pead_signal']:.3f}**")
                    # AlphaCombined
                    else:
                        out.append(f"- Entry condition: **{trig}** (cross-sectional alpha signal threshold)")
                # Momentum conditions
                elif trig is not None and b.get("volume_needed") is not None and b.get("squeeze_pct_threshold") is None:
                    out.append(f"- Price must close **above ${trig:,.2f}** (N-day high breakout)")
                    out.append(f"- Volume must exceed **{b['volume_needed']:,.0f} shares** (volume confirmation)")
                # Mean-Reversion conditions
                elif b.get("rsi_needed") is not None:
                    out.append(f"- RSI(14) must drop **below {b['rsi_needed']}** (oversold) AND price ≤ lower Bollinger Band")
                # VolatilityBreakout conditions (numeric trigger + squeeze fields)
                elif trig is not None:
                    out.append(f"- Close must **break above upper Bollinger Band (${trig:,.2f})**")
                    if b.get("squeeze_pct_threshold") is not None:
                        out.append(f"- BB width must be in the bottom {int(b.get('squeeze_pct_threshold', 0)*100 if isinstance(b.get('squeeze_pct_threshold'), float) else 20)}% of its rolling history (squeeze)")
                    if b.get("min_atr_expansion") is not None:
                        out.append(f"- ATR must expand to ≥{b['min_atr_expansion']}× its 20-day average (expansion confirmed)")
                out.append("")
                out.append("_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._")
            out.append(f"")
            out.append(f"**Note:** {b['execution_note']}")
            return out

        if active:
            lines += ["", "### Active Signals — Enter at Next Open", ""]
            for b in active:
                lines += _render_brief(b, active_signal=True)
        else:
            lines += ["", "_No active entry signals today._", ""]

        # Filter pending signals to only those that passed diagnostics
        # (execution_advisor generates briefs for ALL inactive strategies, not just qualified ones)
        diag_passed_set = {d["ticker"] for d in po.get("diagnostics", []) if d.get("passed")}
        mc_set = {mc["ticker"] for mc in po.get("monte_carlos", [])
                  if not mc.get("insufficient_sample") and not mc.get("stress_test")}
        qualified_pending = [b for b in pending if b["ticker"] in diag_passed_set]

        if qualified_pending:
            lines += ["", "---", "", "### Pending Signals — Monitor Daily", ""]
            lines += [
                "_These tickers passed all 3 validation stages (backtest → diagnostics → Monte Carlo)_",
                "_but have not yet triggered their entry signal. The setup below shows what the_",
                "_trade will look like when conditions are met._",
                "",
            ]
            for b in qualified_pending:
                lines += _render_brief(b, active_signal=False)

        if active or qualified_pending:
            lines += [
                "",
                "---",
                "",
                "### Portfolio Risk Summary",
                "",
                f"| | Count | Dollar Risk | % of Portfolio |",
                "|---|---|---|---|",
                f"| Active signals | {p_risk.get('active_count', 0)} | "
                f"${p_risk.get('total_dollar_risk', 0):,.0f} | "
                f"{p_risk.get('pct_of_portfolio', 0):.1f}% |",
            ]

        return "\n".join(lines)

    @staticmethod
    def _pairs_section(po: dict) -> str:
        """
        Full trader-facing rendering of statistical pair-trading analyses.

        For each pair the pipeline discovered (stage 9b), render: pair stats,
        strategy mechanics, backtest summary, diagnostic verdict, and a
        complete execution brief covering both legs — so the trader knows
        exactly which to sell, which to buy, at what size, when to exit,
        and what costs to expect. No additional research required.
        """
        pair_analyses = po.get("pair_analyses", [])
        blocks = ["## Pair Trading (Statistical Arbitrage)", ""]

        if not pair_analyses:
            blocks.append("_No cointegrated pairs were found in this run's universe._")
            return "\n".join(blocks)

        blocks.append(
            f"**{len(pair_analyses)} pair(s) analysed.** "
            "Pairs are market-neutral: returns come from spread convergence, "
            "not market direction. Execution requires simultaneous fills on BOTH legs."
        )
        blocks.append("")

        for pa in pair_analyses:
            pair_name = pa.get("pair", "?/?")
            ta        = pa.get("ticker_a", "A")
            tb        = pa.get("ticker_b", "B")
            stats     = pa.get("pair_stats", {}) or {}
            params    = pa.get("params", {}) or {}
            bt        = pa.get("backtest", {}) or {}
            diag      = pa.get("diagnostic", {}) or {}
            summary   = bt.get("summary", {}) or {}
            metrics   = diag.get("metrics", {}) or {}
            sig       = pa.get("current_signal") or {}
            passed    = bool(diag.get("passed"))

            status_badge = "PASS (tradable)" if passed else "FAIL (do not trade)"
            blocks += [
                "---",
                "",
                f"### {pair_name} — {status_badge}",
                "",
                "#### Pair Statistics",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Correlation (returns) | {stats.get('correlation', float('nan')):.3f} |",
                f"| Cointegration p-value | {stats.get('coint_pvalue', float('nan')):.4f} |",
                f"| Spread half-life | {stats.get('halflife_days', float('nan')):.1f} days |",
                f"| Hedge ratio β (historical) | {stats.get('hedge_ratio', float('nan')):.4f} |",
                "",
                "#### Strategy Mechanics",
                "",
            ]
            blocks += _render_mechanics("PairTrading", params)
            blocks += [
                "",
                "#### Adjusted Parameters",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
            for k, v in params.items():
                blocks.append(f"| {k} | {v} |")

            # ── Backtest summary ─────────────────────────────────────────
            blocks += [
                "",
                "#### Backtest Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Trades | {summary.get('trade_count', len(bt.get('trade_log', [])))} |",
                f"| Win rate | {summary.get('win_rate', 0):.1%} |",
                f"| Total return | {summary.get('total_return', 0):+.2%} |",
                f"| Sharpe | {metrics.get('sharpe', float('nan')):.2f} |",
                f"| Max drawdown | {metrics.get('max_drawdown', 0):.2%} |",
                f"| Avg |entry_z| | {summary.get('avg_entry_z', 0):.2f} |",
                f"| Avg |exit_z| | {summary.get('avg_exit_z', 0):.2f} |",
                f"| Diagnostic verdict | {'PASS' if passed else 'FAIL'} |",
                f"| Reject reason | {diag.get('reject_reason') or '—'} |",
            ]

            # ── Execution brief (live) ───────────────────────────────────
            if not passed:
                blocks += [
                    "",
                    "> NOTE: **Execution brief suppressed — pair FAILED diagnostic floors.**  ",
                    "> Do not trade this pair; the backtest has no demonstrated edge.",
                    "",
                ]
                continue

            blocks += [
                "",
                "#### Execution Brief (Live Signal)",
                "",
            ]
            z_live     = sig.get("z_score")
            active     = bool(sig.get("signal_active"))
            direction  = sig.get("direction")
            setup      = sig.get("setup") or {}

            if z_live is None:
                blocks.append(
                    "_Live pair signal unavailable for this run "
                    "(insufficient recent history or pipeline skipped computation)._"
                )
                blocks.append("")
                continue

            status = "ACTIVE — ENTER NOW" if active else "PENDING — wait for z-score to breach entry band"
            blocks.append(f"**Status:** {status}")
            blocks.append("")

            # Direction-aware leg mapping
            if direction == "short_spread":
                short_leg, long_leg = ta, tb
                action_line = f"**SHORT {ta}** + **LONG {tb}** (spread z > 0; {ta} is relatively overvalued)"
            elif direction == "long_spread":
                short_leg, long_leg = tb, ta
                action_line = f"**LONG {ta}** + **SHORT {tb}** (spread z < 0; {ta} is relatively undervalued)"
            else:
                short_leg, long_leg = None, None
                action_line = "_No trade signal on latest bar — monitor z-score daily._"

            entry_z = params.get("entry_z", 2.0)
            exit_z  = params.get("exit_z",  0.25)
            stop_z  = params.get("stop_z",  3.5)
            mh      = params.get("max_holding_days", "—")

            blocks += [
                "| Field | Value |",
                "|-------|-------|",
                f"| Live spread z-score | {z_live:+.3f} |",
                f"| Entry band | ±{entry_z:.2f} |",
                f"| Profit-take band | ±{exit_z:.2f} (spread mean-reversion target) |",
                f"| Stop-out band | ±{stop_z:.2f} (divergence stop) |",
                f"| Live hedge ratio β | {sig.get('beta_live', float('nan')):.4f} |",
                f"| Spread std | {sig.get('spread_std', float('nan')):.6f} |",
                f"| {ta} last close | ${sig.get('close_a', 0):,.2f} |",
                f"| {tb} last close | ${sig.get('close_b', 0):,.2f} |",
                "",
                f"**Action:** {action_line}",
                "",
            ]

            if active and setup:
                s_a    = float(setup.get("size_a", 0) or 0)
                s_b    = float(setup.get("size_b", 0) or 0)
                na     = float(setup.get("notional_a", 0) or 0)
                nb     = float(setup.get("notional_b", 0) or 0)
                drisk  = float(setup.get("dollar_risk", 0) or 0)
                # Slippage: 10bps per leg, paid twice (entry + exit)
                slip_a = na * 0.0010 * 2
                slip_b = nb * 0.0010 * 2
                # Borrow cost on short leg (50 bps/yr, charged per day of hold)
                short_notional = na if direction == "short_spread" else nb
                borrow_daily   = short_notional * (0.005 / 252)

                blocks += [
                    "##### Order Tickets (submit BOTH simultaneously at next open)",
                    "",
                    "| Leg | Side | Ticker | Shares | ~Notional |",
                    "|-----|------|--------|--------|-----------|",
                    f"| A | {'SELL SHORT' if direction == 'short_spread' else 'BUY'} | "
                    f"{ta} | {s_a:,.0f} | ${na:,.0f} |",
                    f"| B | {'BUY' if direction == 'short_spread' else 'SELL SHORT'} | "
                    f"{tb} | {s_b:,.0f} | ${nb:,.0f} |",
                    "",
                    "##### Risk & Cost Accounting",
                    "",
                    "| Field | Value |",
                    "|-------|-------|",
                    f"| Dollar risk at entry | ${drisk:,.0f} (1% of portfolio) |",
                    f"| Est. round-trip slippage (both legs × entry + exit) | ${slip_a + slip_b:,.0f} |",
                    f"| Short-leg borrow cost (~50 bps/yr) | ${borrow_daily:,.2f} / day held |",
                    f"| Max holding | {mh} trading days |",
                    "",
                    "##### When to Close the Position",
                    "",
                    f"1. **Take profit:** spread z-score converges inside ±{exit_z:.2f}"
                    f" → BUY BACK **{short_leg}**, SELL **{long_leg}**.",
                    f"2. **Stop out:** spread z-score breaches ±{stop_z:.2f}"
                    f" (divergence worsening) → close BOTH legs immediately.",
                    f"3. **Time stop:** still open after {mh} trading days → close BOTH legs at next open.",
                    "",
                    "> **Important:** if only ONE leg fills at entry, cancel the other"
                    " and skip the trade — a one-sided pair trade is a directional bet,"
                    " not an arbitrage.",
                    "",
                ]
            elif active:
                blocks.append("_Signal active but live setup unavailable — recompute before trading._")
                blocks.append("")
            else:
                blocks += [
                    "##### Projected Trade (when signal fires)",
                    "",
                    f"- Will enter when |z| > {entry_z:.2f}. Current z = {z_live:+.3f}.",
                    f"- Direction will be determined by the sign of the breach"
                    f" (positive z → SHORT {ta} / LONG {tb}; negative z → LONG {ta} / SHORT {tb}).",
                    f"- Sizing will be recomputed at entry using live prices and β.",
                    f"- Monitor daily; entry band is ±{entry_z:.2f}.",
                    "",
                ]

        return "\n".join(blocks)

    @staticmethod
    def _monte_carlo_section(po: dict) -> str:
        monte_carlos = po.get("monte_carlos", [])
        blocks = ["## Monte Carlo Stress Test", ""]

        if not monte_carlos:
            blocks.append("_No Monte Carlo results — no strategies passed diagnostics._")
            return "\n".join(blocks)

        for mc in monte_carlos:
            ticker = mc.get("ticker", "Unknown")

            # Insufficient sample gate
            if mc.get("insufficient_sample"):
                blocks += [
                    f"### {ticker}",
                    "",
                    f"WARNING: **Monte Carlo skipped** — only {mc.get('trade_count', '?')} trades in backtest "
                    f"(minimum 30 required). Results would be statistically meaningless with this few observations.",
                    "",
                ]
                continue

            trade_count = mc.get("trade_count", 0)
            disclaimer  = (
                f"\n> WARNING: **Statistical disclaimer:** This simulation is based on only "
                f"**{trade_count} historical trades**. Bootstrap resampling with fewer than 60 trades "
                f"produces wide, unreliable confidence bands. Treat these figures as directional only."
            ) if trade_count and trade_count < 60 else ""

            blocks += [
                f"### {ticker}",
                "",
                "#### Outcome Distribution (10,000 simulations)",
                "",
                "| Metric | P5 | Median | P95 |",
                "|--------|----|--------|-----|",
                f"| Final Portfolio ($) "
                f"| {mc.get('p5_final', 0):,.0f} "
                f"| {mc.get('p50_final', 0):,.0f} "
                f"| {mc.get('p95_final', 0):,.0f} |",
                f"| Sharpe Ratio † "
                f"| {mc.get('p5_sharpe', 0):.3f} "
                f"| {mc.get('p50_sharpe', 0):.3f} "
                f"| {mc.get('p95_sharpe', 0):.3f} |",
                f"| Win Rate "
                f"| {mc.get('p5_win_rate', 0):.1%} "
                f"| {mc.get('p50_win_rate', 0):.1%} "
                f"| {mc.get('p95_win_rate', 0):.1%} |",
                "",
                "#### Risk Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| P(Ruin) — equity falls >40% | {mc.get('p_ruin', 0):.2%} |",
                f"| P95 Max Drawdown | {mc.get('p95_max_drawdown', 0):.2%} |",
                f"| Median CAGR | {mc.get('median_cagr', 0):.2%} |",
                f"| P95 Max Consecutive Losses | {mc.get('p95_max_consec_losses', 0)} |",
                f"| Optimal Kelly Fraction | {mc.get('kelly_fraction', 0):.3f} |",
                *(
                    [
                        "",
                        "> **Kelly = 0 note:** Negative expectancy at the trade-sequence level — "
                        "the formula signals no provable edge. P(Ruin) can still be 0% because "
                        "the fixed 1% position sizing caps total drawdown far below the 40% ruin "
                        "floor even across many consecutive losses. Kelly = 0 is the stronger "
                        "signal: do not trade this setup until edge is demonstrated.",
                    ]
                    if mc.get("kelly_fraction", 0) <= 0
                    else []
                ),
                "",
                "#### Ruin Analysis",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Median Trade at First Ruin | {mc.get('median_time_to_ruin') or 'N/A'} |",
                f"| Mean Portfolio at Ruin | "
                f"{'${:,.0f}'.format(mc['ruin_severity']) if mc.get('ruin_severity') is not None else 'N/A'} |",
                "",
                "#### Equity Confidence Band",
                "",
                "| Trade # | P5 ($) | Median ($) | P95 ($) |",
                "|---------|--------|------------|---------|",
            ]
            for entry in mc.get("equity_band", []):
                blocks.append(
                    f"| {entry['step']} "
                    f"| {entry['p5']:,.0f} "
                    f"| {entry['p50']:,.0f} "
                    f"| {entry['p95']:,.0f} |"
                )
            if disclaimer:
                blocks.append(disclaimer)
            blocks.append(
                "\n† **Sharpe annualization note:** Monte Carlo Sharpe uses trade-frequency "
                "annualization (√(trades/year)), while Diagnostic Sharpe uses daily-return "
                "annualization (√252). For strategies with <252 trades/year the MC Sharpe will "
                "be lower — this is not a discrepancy; it reflects a stricter per-trade view. "
                "Use the Diagnostic Sharpe for regime comparisons; use MC Sharpe for realistic "
                "out-of-sample expectation."
            )
            blocks.append("")

        return "\n".join(blocks)


# ── strategy mechanics renderer ───────────────────────────────────────────────

def _render_mechanics(strategy: str, params: dict) -> list[str]:
    """Return plain-English lines describing entry/exit/sizing rules with params filled in."""
    lines = []
    if strategy == "Momentum":
        el  = params.get("entry_lookback", "N")
        vm  = params.get("volume_multiplier", "N")
        sl  = params.get("stop_loss_atr", "N")
        ts  = params.get("trailing_stop_atr", "N")
        ma  = params.get("ma_exit_period", "N")
        mh  = params.get("max_holding_days", "N")
        lines += [
            "**Why it works:** Momentum strategies exploit the empirical tendency of assets"
            " with high Hurst exponents (H > 0.55) to persist in their current direction."
            " Requiring a volume surge at breakout filters false breakouts driven by thin"
            " liquidity, keeping the signal anchored to genuine institutional participation."
            " ATR-based stops let volatility scale the exit distance, avoiding premature"
            " stops in volatile regimes while still capping loss per trade at ~1% of capital.",
            "",
            "**Order type:** Market order at next session open.",
            "",
            "**Entry (both conditions required):**",
            f"- Price breakout: Close > {el}-day rolling high (prior session close)",
            f"- Volume confirmation: Volume > {vm}× 20-day average volume",
            "",
            f"**Position sizing:** 1% portfolio risk ÷ ({sl} × ATR₁₄) = shares to buy",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Hard stop loss** — Close < entry price − {sl} × ATR₁₄",
            f"2. **Trailing stop** — Close < highest close since entry − {ts} × ATR₁₄",
            f"3. **MA exit** — Close < {ma}-day simple moving average",
            f"4. **Max holding** — Force exit after {mh} trading days",
        ]
    elif strategy == "Mean-Reversion":
        re_ = params.get("rsi_entry_threshold", "N")
        rx  = params.get("rsi_exit_threshold", "N")
        bp  = params.get("bb_period", "N")
        bs  = params.get("bb_std", "N")
        sl  = params.get("stop_loss_atr", "N")
        mh  = params.get("max_holding_days", "N")
        lines += [
            "**Why it works:** Mean-reversion strategies exploit the empirical tendency of"
            " low-Hurst assets (H < 0.45) to oscillate around a statistical mean."
            " Requiring both RSI oversold and a close below the lower Bollinger Band"
            " creates a dual-confirmation filter — RSI measures rate-of-change exhaustion"
            " while Bollinger Bands measure statistical deviation from the rolling mean."
            " The position is sized so that even a full ATR move against the trade risks"
            " only 1% of capital, giving the reversion room to play out over several days.",
            "",
            "**Order type:** Market order at next session open.",
            "",
            "**Entry (both conditions required):**",
            f"- RSI(14) < {re_} (oversold)",
            f"- Close ≤ lower Bollinger Band ({bp}-day MA − {bs}σ)",
            "",
            f"**Position sizing:** 1% portfolio risk ÷ ({sl} × ATR₁₄) = shares to buy",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Hard stop loss** — Close < entry price − {sl} × ATR₁₄",
            f"2. **RSI exit** — RSI(14) > {rx} (overbought)",
            f"3. **MA exit** — Close ≥ {bp}-day SMA (middle Bollinger Band = mean-reversion target)",
            f"4. **Max holding** — Force exit after {mh} trading days",
        ]
    elif strategy == "VolatilityBreakout":
        bp  = params.get("bb_period", "N")
        sq  = params.get("squeeze_pct", "N")
        sl_days = params.get("squeeze_lookback", 5)
        vm  = params.get("volume_mult", 1.5)
        sl  = params.get("stop_loss_atr", "N")
        ts  = params.get("trailing_stop_atr", "N")
        mh  = params.get("max_holding_days", "N")
        sq_pct_str = f"{int(sq*100)}th" if isinstance(sq, float) else str(sq)
        lines += [
            "**Why it works:** Bollinger Band squeeze → breakout is a well-documented alpha source."
            " When a stock's daily range compresses below its historical norm (the 'squeeze'), market"
            " participants are accumulating positions before a catalyst. When price finally breaks"
            " above the upper Bollinger Band on elevated volume, the accumulation phase ends and the"
            " directional move begins. Unlike a simple momentum breakout, the squeeze filter means"
            " we only enter when volatility was genuinely compressed beforehand — avoiding chasing"
            " moves that started without compression.",
            "",
            "**Order type:** Market order at next session open.",
            "",
            "**Entry (ALL conditions required):**",
            f"- BB width was in the bottom {sq_pct_str} percentile of its rolling {bp}-bar history within the last {sl_days} bars (prior compression confirmed)",
            f"- Close > upper Bollinger Band ({bp}-day MA + 2σ) (breakout direction = long)",
            f"- Volume > {vm}× 20-bar average (confirms institutional participation — not a low-volume fake break)",
            "",
            f"**Position sizing:** 1% portfolio risk ÷ ({sl} × ATR₁₄) = shares to buy",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Trailing stop** — Close < highest close since entry − {ts} × ATR₁₄",
            f"2. **Hard stop loss** — Close < entry price − {sl} × ATR₁₄ (floor for trailing stop)",
            f"3. **Max holding** — Force exit after {mh} trading days",
        ]
    elif strategy == "AlphaCombined":
        ath = params.get("alpha_threshold", "N")
        rth = params.get("reversal_threshold", "N")
        sl  = params.get("stop_loss_atr", "N")
        ts  = params.get("trailing_stop_atr", "N")
        mh  = params.get("max_holding_days", "N")
        lines += [
            "**Why it works:** AlphaCombined blends four cross-sectional signals — "
            "cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), "
            "volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar "
            "to prevent look-ahead bias. Because each signal is z-scored cross-sectionally "
            "(ranked across all tickers in the universe each day), the combined alpha is "
            "market-neutral by construction and adapts to whichever tickers are relatively "
            "mis-priced on any given day. This multi-factor approach produces 200+ trades "
            "over a 2-year backtest window, providing the statistical power required for "
            "walk-forward validation.",
            "",
            "**Order type:** Market order at next session open.",
            "",
            "**Entry condition:**",
            f"- Cross-sectional alpha signal > {ath} (normalised z-score threshold)",
            "",
            f"**Position sizing:** 1% portfolio risk ÷ ({sl} × ATR₁₄) = shares to buy",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Trailing stop** — Close < highest close since entry − {ts} × ATR₁₄",
            f"2. **Alpha reversal** — alpha signal drops below {rth} (signal exhaustion)",
            f"3. **Max holding** — Force exit after {mh} trading days",
        ]
    elif strategy == "MLSignal":
        ml_th   = params.get("ml_threshold",    "N")
        sl      = params.get("stop_loss_atr",   "N")
        ts      = params.get("trailing_stop_atr", "N")
        mh      = params.get("max_holding_days", "N")
        lines += [
            "**Why it works:** MLSignal is an ensemble probability signal produced by"
            " averaging four independent models (logistic regression, random forest,"
            " gradient boosting, and a shallow MLP) trained on engineered features"
            " (momentum, volatility, volume, cross-sectional rank). The ensemble is"
            " shift(1)-lagged to prevent look-ahead bias. It excels in Neutral and"
            " Low-Volatility regimes where no single linear factor dominates but"
            " subtle nonlinear interactions (e.g. high short-term momentum combined"
            " with low realised vol) carry predictive content. Threshold-gated entries"
            " only fire when the ensemble probability exceeds a regime-tuned cutoff,"
            " keeping false-positive rate down in quiet markets.",
            "",
            "**Order type:** Market order at next session open.",
            "",
            "**Entry condition:**",
            f"- Ensemble ML probability `ml_signal` > {ml_th} (prior-bar value, no look-ahead)",
            "",
            f"**Position sizing:** 1% portfolio risk ÷ ({sl} × ATR₁₄) = shares to buy",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Hard stop loss** — Close < entry price − {sl} × ATR₁₄",
            f"2. **Trailing stop** — Close < highest close since entry − {ts} × ATR₁₄",
            f"3. **Signal decay** — ml_signal drops back below {ml_th}",
            f"4. **Max holding** — Force exit after {mh} trading days",
        ]
    elif strategy == "EventDriven":
        gap_t   = params.get("gap_threshold",        "N")
        pead_t  = params.get("pead_min_signal",      "N")
        ewin    = params.get("entry_window_bars",    "N")
        vm      = params.get("volume_mult",          "N")
        map_    = params.get("ma_filter_period",     "N")
        sl      = params.get("stop_loss_atr",        "N")
        pex     = params.get("pead_exit_threshold",  "N")
        mh      = params.get("max_holding_days",     "N")
        gap_pct_str = f"{gap_t*100:.1f}%" if isinstance(gap_t, float) else str(gap_t)
        lines += [
            "**Why it works:** Post-Earnings Announcement Drift (PEAD) is one of the"
            " most persistent anomalies in equity markets: stocks that gap up on"
            " earnings continue to drift upward for days to weeks as institutions"
            " re-rate slowly and analyst estimates get revised. The strategy waits"
            " out the immediate earnings blackout window (to avoid adverse selection"
            " by informed traders), then enters when (a) a recent positive gap is"
            " still in force, (b) the PEAD composite signal is above threshold,"
            " (c) price remains above its short MA (drift still intact), and"
            " (d) volume confirms continued participation. Exits cut early when the"
            " PEAD signal fades, capturing the drift and not the mean-reversion that"
            " follows exhaustion.",
            "",
            "**Order type:** Market order at next session open (after blackout lifts).",
            "",
            "**Entry (ALL conditions required):**",
            f"- Positive earnings gap > {gap_pct_str} within the last {ewin} bars",
            f"- PEAD composite signal > {pead_t} (drift still active)",
            f"- Close > {map_}-day simple moving average (drift not broken)",
            f"- Volume > {vm}× 20-bar average (participation confirmed)",
            f"- Outside earnings blackout window",
            "",
            f"**Position sizing:** 1% portfolio risk ÷ ({sl} × ATR₁₄) = shares to buy",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Hard stop loss** — Close < entry price − {sl} × ATR₁₄",
            f"2. **PEAD fade** — pead_signal drops below {pex} (drift reversed)",
            f"3. **MA break** — Close < {map_}-day MA (drift broken)",
            f"4. **Max holding** — Force exit after {mh} trading days",
        ]
    elif strategy in ("PairTrading", "PairsTrading"):
        ez  = params.get("entry_z",       2.0)
        xz  = params.get("exit_z",        0.25)
        sz  = params.get("stop_z",        3.5)
        bw  = params.get("beta_window",   60)
        zw  = params.get("z_window",      60)
        mh  = params.get("max_holding_days", "N")
        lines += [
            "**Why it works:** Statistical pair (spread) trading exploits short-run"
            " dislocations between two historically cointegrated assets. When the"
            " log-spread `log(A) − β·log(B)` deviates from its rolling mean by more"
            " than `entry_z` standard deviations, the market-neutral position is"
            " opened — shorting the relatively expensive leg and going long the"
            " relatively cheap leg. Mean-reversion of the spread (cointegration)"
            " provides the edge; dollar-neutral sizing removes broad market beta"
            " exposure so returns come from convergence rather than market direction.",
            "",
            "**Order type:** Two simultaneous market orders at next session open"
            " (one long leg, one short leg). Both must fill — if only one leg fills,"
            " cancel the other and skip the trade.",
            "",
            "**Entry conditions:**",
            f"- |spread z-score| > {ez} (computed on rolling {zw}-bar window)",
            f"- Hedge ratio β estimated from rolling {bw}-bar OLS of log(A) on log(B)",
            "- Direction:",
            f"  - **z > +{ez}** → **SHORT A**, **LONG B** (A is overvalued vs B)",
            f"  - **z < −{ez}** → **LONG A**,  **SHORT B** (A is undervalued vs B)",
            "",
            "**Position sizing (dollar-neutral):**",
            f"- Risk budget: 1% of portfolio allocated to a ({sz} − |entry_z|) spread move",
            f"- notional_A = portfolio × 0.01 / ({sz} − |z|)",
            f"- size_A     = notional_A / price_A",
            f"- size_B     = size_A × |β|          (β from rolling regression)",
            "- Verify: notional_A ≈ notional_B at entry (leg balance check)",
            "",
            "**Exit rules (checked in priority order each day):**",
            f"1. **Hard stop** — |spread z-score| > {sz} (divergence getting worse; close both legs)",
            f"2. **Target** — |spread z-score| < {xz} (spread has converged to mean; take profit)",
            f"3. **Max holding** — Force close both legs after {mh} trading days",
            "",
            "**Costs to account for:**",
            "- Slippage on BOTH legs (paid twice on entry, twice on exit)",
            "- Borrow fee on the short leg (~50 bps/yr on liquid large caps,"
            " charged daily on short-leg notional)",
            "- Dividend pass-through on the short leg if a dividend falls during the hold",
        ]
    else:
        lines.append(f"_Mechanics not defined for strategy type: {strategy}_")
    return lines


# ── math helpers ──────────────────────────────────────────────────────────────

def _sharpe_from_returns(returns: pd.Series, rf: float = 0.045) -> float:
    """Annualised Sharpe, capped ±20, matching DiagnosticsEngine."""
    returns = pd.Series(np.array(returns).flatten())
    std = float(returns.std(ddof=1))
    if std < 1e-10 or np.isnan(std):
        return 0.0
    daily_rf = rf / 252
    raw = float((float(returns.mean()) - daily_rf) / std * math.sqrt(252))
    return float(np.clip(raw, -20.0, 20.0))


def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def _fmt_perm_pvalue_row(metrics: dict) -> str:
    """Format the permutation p-value diagnostic table row with a dynamic interpretation.

    The permutation test shuffles the return series and measures what fraction of
    random orderings produce a Calmar ratio >= the real strategy's.
    - p < 0.10 : real strategy's temporal structure (entry/exit timing) beats 90%+
                 of random orderings → strong evidence of order-dependent edge
    - 0.10–0.30: weak evidence of temporal structure
    - 0.30–0.70: no detectable temporal structure (expected for IID-return strategies)
    - p > 0.90 : exits are actively destroying value (most shuffles outperform)
    """
    pv = metrics.get("permutation_p_value", float("nan"))
    if math.isnan(pv):
        return "| Permutation p-value (Calmar) | N/A (underpowered) | < 10 trades — test skipped |"
    if pv < 0.10:
        interp = "temporal structure present (exits beat 90%+ of shuffles)"
    elif pv < 0.30:
        interp = "weak temporal structure"
    elif pv <= 0.70:
        interp = "no temporal structure (expected for IID-return strategies)"
    else:
        interp = "WARNING: exits destroying value (90%+ of shuffles outperform)"
    return f"| Permutation p-value (Calmar) | {pv:.3f} | {interp} |"


def _fmt_exposure_row(metrics: dict) -> str:
    """Format market exposure as a trade-statistics table row.

    Market exposure = fraction of backtested days the strategy was invested.
    < 15%  : flag as potentially under-deployed (limited compounding opportunity)
    15–50% : normal for swing / momentum strategies
    > 50%  : high exposure — confirm VaR/CVaR are acceptable before sizing up
    """
    exposure = metrics.get("market_exposure")
    if exposure is None:
        return "| Market Exposure (% days invested) | N/A | Insufficient trade log |"
    if exposure < 0.15:
        interp = f"WARNING: under-deployed — only {exposure:.1%} of days invested; limited compounding"
    elif exposure > 0.50:
        interp = f"High exposure ({exposure:.1%}) — verify VaR/CVaR are within limits"
    else:
        interp = f"Normal ({exposure:.1%})"
    return f"| Market Exposure (% days invested) | {exposure:.1%} | {interp} |"


def _fmt_degradation_row(metrics: dict) -> str:
    """Format the walk-forward degradation table row.

    Degradation is only meaningful when the strategy had positive in-sample
    Sharpe.  When WF is underpowered (< 30 trades) or the IS Sharpe was
    negative, the formula returns 0.0 by convention — display 'N/A' with a
    reason instead of the misleading '0.0%'.
    """
    wf_degrad      = metrics.get("walk_forward_degradation", 0.0)
    wf_underpowered = metrics.get("wf_underpowered", False)
    is_sharpe      = metrics.get("sharpe", 0.0)   # full-period Sharpe; IS ≈ 0 signals no edge

    if wf_underpowered:
        note = "N/A (< 30 trades)"
    elif is_sharpe <= 0:
        note = "N/A (no IS edge)"
    else:
        note = f"{wf_degrad:.1%}"

    return f"| Degradation | — | {note} |"


def _advanced_metrics(
    returns: pd.Series,
    trade_log: list[dict],
    metrics: dict,
    equity_curve: "pd.Series | None" = None,
) -> dict:
    """Compute advanced metrics from returns series and trade log.

    equity_curve is used for IS/OOS return calculation to avoid the RF-inflation
    artefact: _build_returns() earns DAILY_RF on flat/cash days so
    (1+returns).prod()-1 compounds that rate on every uninvested day, reporting
    a higher cumulative return than the equity curve actually shows.
    """
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

        # Sortino (downside deviation) — uses same RF hurdle as Sharpe (4.5% annual)
        # so both ratios are directly comparable.  Capped at 10.0 to prevent artifacts.
        _daily_rf = 0.045 / TRADING_DAYS
        downside = returns[returns < 0]
        down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino  = float((mean_ret - _daily_rf) / down_std * math.sqrt(TRADING_DAYS)) if down_std > 1e-4 else 0.0
        sortino  = min(max(sortino, -10.0), 10.0)

        # Calmar
        calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

        # VaR / CVaR 95% — computed only on invested days (non-zero returns).
        # Flat cash days (return == 0.0) are excluded because including them
        # dilutes the tail and collapses VaR to 0%.
        invested = returns[returns != 0.0].dropna()
        if len(invested) >= 10:
            var_95  = float(np.percentile(invested, 5))
            tail    = invested[invested <= var_95]
            cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
        else:
            # Fallback: too few invested days — use full series
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

        # Walk-forward split — 70/30 IS/OOS matches DiagnosticsEngine so the Sharpe
        # values shown here are consistent with the walk_forward_degradation metric.
        split    = int(len(returns) * 0.70)
        split    = max(split, 1)   # guard against degenerate very-short series
        is_ret   = returns.iloc[:split]
        oos_ret  = returns.iloc[split:]

        _DAILY_RF = 0.045 / TRADING_DAYS   # risk-free rate per day (matches DiagnosticsEngine)

        def _sharpe(r: pd.Series) -> float:
            s = r.std(ddof=1)
            if s < 1e-10:
                return 0.0
            raw = float((r.mean() - _DAILY_RF) / s * math.sqrt(TRADING_DAYS))
            return float(np.clip(raw, -20.0, 20.0))

        # IS/OOS returns: use equity curve to avoid RF-inflation artefact.
        # Falls back to compounding returns only if equity is unavailable.
        if equity_curve is not None and not equity_curve.empty:
            eq_split = max(int(len(equity_curve) * 0.70), 1)
            is_ret_pct  = float(equity_curve.iloc[eq_split] / equity_curve.iloc[0] - 1)
            oos_ret_pct = float(equity_curve.iloc[-1] / equity_curve.iloc[eq_split] - 1)
        else:
            is_ret_pct  = float((1 + is_ret).prod() - 1)
            oos_ret_pct = float((1 + oos_ret).prod() - 1)

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
            "is_return":     is_ret_pct,
            "oos_return":    oos_ret_pct,
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
