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

    def generate(self, pipeline_output: dict, timestamp: str | None = None) -> str:
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
        timestamp = timestamp or datetime.now().strftime("%H%M%S")
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
            self._monte_carlo_section(pipeline_output),
            self._execution_brief_section(pipeline_output),
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(sections))

        print(f"  [Report] Full report    → {filepath}")
        return filepath

    def generate_summary(self, pipeline_output: dict, timestamp: str | None = None) -> str:
        """
        Generate a trader-focused summary report containing only tickers
        that passed all 3 stages (backtest → diagnostics → Monte Carlo).
        Ordered by importance to the trader. Numbers preserved for graphing.
        """
        run_date  = pipeline_output.get("run_date", datetime.today().strftime("%Y-%m-%d"))
        timestamp = timestamp or datetime.now().strftime("%H%M%S")
        filename  = f"summary_{run_date}_{timestamp}.md"
        filepath  = os.path.join(self.output_dir, filename)

        # Build lookup maps
        mc_map   = {mc["ticker"]: mc for mc in pipeline_output.get("monte_carlos", [])
                    if not mc.get("insufficient_sample")}
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
            f.write("\n\n".join(sections))

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
        lines = ["## ⚡ Today's Action", ""]
        active_any = False
        for ticker in qualified:
            s   = strat_map.get(ticker, {})
            sig = s.get("current_signal", {})
            if sig.get("signal_active"):
                active_any = True
                setup = sig.get("setup", {})
                lines += [
                    f"### ✅ {ticker} — ENTER NOW ({s.get('strategy', '')})",
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
                    flag  = "alpha ✅" if alpha >= 0 else "underperform ❌"
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
                "**Status: ✅ ACTIVE — enter at next session open**",
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
            if sig.get("breakout") is False:     failed.append("price breakout")
            if sig.get("volume_confirmed") is False: failed.append("volume confirmation")
            if sig.get("oversold") is False:     failed.append("RSI oversold")
            if sig.get("below_bb") is False:     failed.append("below lower BB")
            reason = " + ".join(failed) if failed else "conditions"
            lines += [
                f"**Status: ⏸ INACTIVE — {reason} not met**",
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
        _oos       = metrics.get("oos_sharpe", 0.0)
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
            lines.append(f"| {name} | {val} | {floor} | {'✅' if ok else '❌'} |")

        wf_split = int(len(returns) * 0.70)
        if not returns.empty and wf_split > 0:
            is_ret  = returns.iloc[:wf_split]
            oos_ret = returns.iloc[wf_split:]
            is_cum  = float((1 + is_ret).prod() - 1)
            oos_cum = float((1 + oos_ret).prod() - 1)
            is_start  = str(returns.index[0])[:10]
            is_end    = str(returns.index[wf_split - 1])[:10]
            oos_start = str(returns.index[wf_split])[:10]
            oos_end   = str(returns.index[-1])[:10]
            lines += [
                "",
                "**Walk-forward (70% IS / 30% OOS):**",
                "",
                "| Period | Start | End | Cumulative Return |",
                "|--------|-------|-----|-------------------|",
                f"| In-sample (70%) | {is_start} | {is_end} | {is_cum:.2%} |",
                f"| Out-of-sample (30%) | {oos_start} | {oos_end} | {oos_cum:.2%} |",
            ]

        # ── 6. Backtest performance ────────────────────────────────────────────
        lines += ["", "### 6. Backtest Performance (10 years)", ""]
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Net return (after slippage) | {net_ret:.2%}{spy_str} |",
            f"| Gross return (pre-cost) | {summary.get('gross_return', 0):.2%} |",
            f"| Total slippage cost | ${summary.get('total_slippage_cost', 0):,.2f} |",
            f"| Trade count | {summary.get('trade_count', 0)} |",
            f"| Win rate | {summary.get('win_rate', 0):.1%} |",
            "",
        ]

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

        # Portfolio-level risk from active signals
        strategies   = po.get("strategies", [])
        active_sigs  = [s for s in strategies if s.get("current_signal", {}).get("signal_active")]
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
            # Current signal status
            blocks += ["", "#### Current Entry Signal (as of run date)", ""]
            if sig.get("signal_active") is True:
                blocks.append("**Status: ✅ ACTIVE — entry condition met on latest bar**")
            elif sig.get("signal_active") is False:
                # Show exactly which condition(s) failed
                failed = []
                if sig.get("breakout") is False:
                    failed.append("price breakout")
                if sig.get("volume_confirmed") is False:
                    failed.append("volume confirmation")
                if sig.get("oversold") is False:
                    failed.append("RSI oversold")
                if sig.get("below_bb") is False:
                    failed.append("below lower BB")
                reason = " + ".join(failed) if failed else "entry condition"
                blocks.append(f"**Status: ⏸ INACTIVE — {reason} not met**")
            else:
                blocks.append("**Status: N/A**")
            if sig.get("details"):
                blocks.append(f"```\n{sig['details']}\n```")
            # Trade setup — only shown when signal is active
            setup = sig.get("setup")
            if setup:
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
                f"| Entry Efficiency (% reaching +1R) | {bt.get('summary', {}).get('entry_efficiency', 0):.1%} |",
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
        backtests  = po.get("backtests", [])
        spy_ohlcv  = po.get("spy_ohlcv")
        corr_warns = po.get("correlation_warnings", [])
        blocks     = ["## Backtest Results", ""]

        # Correlation warnings
        if corr_warns:
            blocks += ["### ⚠️ Concentration Risk Warnings", ""]
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
                    spy_dr      = spy_close_full.pct_change().fillna(0.0)
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
                vs_spy = f"  **vs SPY (exposure-adj): {diff:+.2%}** ({'alpha ✅' if diff >= 0 else 'underperform ❌'})"
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
            if not returns.empty:
                split   = int(len(returns) * 0.70)
                is_ret  = returns.iloc[:split]
                oos_ret = returns.iloc[split:]
                blocks += ["", "#### Walk-Forward Returns (70% IS / 30% OOS)", "",
                           "| Period | Start | End | Cumulative Return |",
                           "|--------|-------|-----|-------------------|"]
                is_cum  = float((1 + is_ret).prod() - 1)
                oos_cum = float((1 + oos_ret).prod() - 1)
                is_start  = returns.index[0].strftime("%Y-%m-%d")      if hasattr(returns.index[0],      "strftime") else str(returns.index[0])
                is_end    = returns.index[split-1].strftime("%Y-%m-%d") if hasattr(returns.index[split-1],"strftime") else str(returns.index[split-1])
                oos_start = returns.index[split].strftime("%Y-%m-%d")   if hasattr(returns.index[split],  "strftime") else str(returns.index[split])
                oos_end   = returns.index[-1].strftime("%Y-%m-%d")      if hasattr(returns.index[-1],     "strftime") else str(returns.index[-1])
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
    def _execution_brief_section(po: dict) -> str:
        eb = po.get("execution_brief", {})
        active   = eb.get("active_signals", [])
        p_risk   = eb.get("portfolio_risk", {})
        warnings = eb.get("warnings", [])

        mkt = eb.get("market_status", {})
        mkt_label  = mkt.get("label", "UNKNOWN")
        mkt_detail = mkt.get("detail", "")
        mkt_line   = f"**NYSE Market Status:** {mkt_label} — {mkt_detail}"

        blocks = ["## Execution Brief", "", mkt_line, ""]

        if not active:
            blocks.append(
                "_No active entry signals today — no execution required._  \n"
                f"**Inactive signals:** {eb.get('inactive_count', 0)}"
            )
            return "\n".join(blocks)

        blocks += [
            "### Portfolio-Level Risk",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Active signals | {p_risk.get('active_count', 0)} |",
            f"| Total adjusted dollar risk | ${p_risk.get('total_dollar_risk', 0):,.0f} |",
            f"| % of portfolio | {p_risk.get('pct_of_portfolio', 0):.2f}% |",
            "",
        ]

        if warnings:
            blocks.append("### Warnings")
            blocks.append("")
            for w in warnings:
                blocks.append(f"- ⚠️  {w}")
            blocks.append("")

        blocks += [
            "### Active Signal Execution Details",
            "",
            "| Ticker | Entry | Stop | Size | Dollar Risk | Adj. Risk | Spread | Slippage | Impact | Note |",
            "|--------|-------|------|------|-------------|-----------|--------|----------|--------|------|",
        ]
        for s in active:
            spread_str   = f"${s['spread']:.4f}"  if s.get("spread")  is not None else "N/A (ATR est.)"
            entry_str    = f"${s['entry_price']:,.2f}" if s.get("entry_price") is not None else "N/A"
            stop_str     = f"${s['stop_price']:,.2f}"  if s.get("stop_price")  is not None else "N/A"
            blocks.append(
                f"| {s['ticker']} "
                f"| {entry_str} "
                f"| {stop_str} "
                f"| {s['position_size']:,} "
                f"| ${s['dollar_risk']:,.0f} "
                f"| ${s['adjusted_risk']:,.0f} "
                f"| {spread_str} "
                f"| ${s['slippage_total']:,.2f} "
                f"| {s['market_impact'].upper()} "
                f"| {s['execution_note']} |"
            )

        blocks.append("")
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
                    f"⚠️ **Monte Carlo skipped** — only {mc.get('trade_count', '?')} trades in backtest "
                    f"(minimum 30 required). Results would be statistically meaningless with this few observations.",
                    "",
                ]
                continue

            trade_count = mc.get("trade_count", 0)
            disclaimer  = (
                f"\n> ⚠️ **Statistical disclaimer:** This simulation is based on only "
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
                f"| Sharpe Ratio "
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
    else:
        lines.append(f"_Mechanics not defined for strategy type: {strategy}_")
    return lines


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

        # Sortino (downside deviation) — capped at 10.0 to prevent artifacts from near-zero downside std
        downside = returns[returns < 0]
        down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino  = float(mean_ret / down_std * math.sqrt(TRADING_DAYS)) if down_std > 1e-4 else 0.0
        sortino  = min(sortino, 10.0)

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
