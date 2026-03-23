"""
visualize_report.py
===================
Parses a generated Markdown report and produces a matplotlib dashboard.

Usage
-----
    python visualize_report.py                        # latest report in reports/
    python visualize_report.py reports/report_X.md   # specific file

Output
------
    One PNG per ticker backtest  +  one overview PNG saved beside the report.
    All figures are also shown interactively (close each window to continue).
"""

from __future__ import annotations

import os
import re
import sys
import glob
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")          # change to "Agg" if no display / running headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_report(directory: str = "reports") -> str:
    files = sorted(glob.glob(os.path.join(directory, "report_*.md")))
    if not files:
        raise FileNotFoundError(f"No report_*.md files found in '{directory}/'")
    return files[-1]


def _parse_md_table(block: str) -> list[dict]:
    """Parse a markdown pipe table into a list of dicts."""
    lines = [l.strip() for l in block.strip().splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return []
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:          # skip header + separator
        vals = [v.strip() for v in line.strip("|").split("|")]
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals)))
    return rows


def _clean(text: str) -> str:
    """Strip markdown bold/italic markers."""
    return re.sub(r"\*+", "", text).strip()


def _pct(s: str) -> float:
    """'12.34%' → 0.1234"""
    try:
        return float(s.replace("%", "").replace(",", "").strip()) / 100
    except ValueError:
        return 0.0


def _num(s: str) -> float:
    try:
        return float(re.sub(r"[^\d.\-+eE]", "", s))
    except ValueError:
        return 0.0


# ── parser ────────────────────────────────────────────────────────────────────

class ReportParser:
    def __init__(self, path: str):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.text = f.read()

    # ── executive summary ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        t = self.text
        data = {}
        m = re.search(r"\*\*Run date:\*\*\s+(.+)", t)
        if m: data["run_date"] = m.group(1).strip()
        m = re.search(r"\*\*Articles analysed:\*\*\s+(\d+)", t)
        if m: data["article_count"] = int(m.group(1))
        m = re.search(r"\*\*Overall market bias:\*\*\s+(\w+)", t)
        if m: data["market_bias"] = m.group(1)

        m = re.search(r"\*\*Buy candidates \((\d+)\):\*\*\s*(.*)", t)
        if m:
            data["buy_count"] = int(m.group(1))
            raw = m.group(2).strip()
            data["buys"] = [] if raw == "None" else [x.strip() for x in raw.split(",")]
        m = re.search(r"\*\*Watch \((\d+)\):\*\*\s*(.*)", t)
        if m:
            data["watch_count"] = int(m.group(1))
        m = re.search(r"\*\*Avoid \((\d+)\):\*\*\s*(.*)", t)
        if m:
            data["avoid_count"] = int(m.group(1))
        return data

    # ── macro ─────────────────────────────────────────────────────────────────

    def macro(self) -> dict:
        t = self.text
        data = {}
        m = re.search(r"\*\*Favoured sectors:\*\*\s*(.+)", t)
        if m: data["favored"] = [x.strip() for x in m.group(1).split(",")]
        m = re.search(r"\*\*Avoid sectors:\*\*\s*(.+)", t)
        if m: data["avoid"] = [x.strip() for x in m.group(1).split(",")]
        m = re.search(r"\*\*Active macro risks:\*\*\s*(.+)", t)
        if m: data["risks"] = [x.strip() for x in m.group(1).split(",")]
        return data

    # ── verdicts table ────────────────────────────────────────────────────────

    def verdicts(self) -> list[dict]:
        section = re.search(r"## Shortlisted Tickers(.+?)(?=\n## )", self.text, re.DOTALL)
        if not section:
            return []
        rows = _parse_md_table(section.group(1))
        out = []
        for r in rows:
            out.append({
                "ticker":  _clean(r.get("Ticker", "")),
                "verdict": _clean(r.get("Verdict", "")).lower(),
            })
        return out

    # ── regimes table ─────────────────────────────────────────────────────────

    def regimes(self) -> list[dict]:
        section = re.search(r"## Regime Classification(.+?)(?=\n## )", self.text, re.DOTALL)
        if not section:
            return []
        rows = _parse_md_table(section.group(1))
        out = []
        for r in rows:
            out.append({
                "ticker":  _clean(r.get("Ticker", "")),
                "regime":  _clean(r.get("Regime", "")),
                "hurst":   _num(r.get("Hurst", "0")),
                "atr_pct": _pct(r.get("ATR/Price", "0")),
            })
        return [x for x in out if x["ticker"]]

    # ── diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> list[dict]:
        section = re.search(r"## Diagnostic Results(.+?)(?=\n## )", self.text, re.DOTALL)
        if not section:
            return []
        results = []
        for block in re.split(r"(?m)^### ", section.group(1)):
            if not block.strip():
                continue
            header_m = re.match(r"(.+?)\s+\[(.+?)\]", block.splitlines()[0])
            if not header_m:
                continue
            label  = header_m.group(1).strip()   # "AAPL — Momentum"
            status = header_m.group(2).strip()    # "✅ PASS" or "❌ FAIL"

            def get_metric(name):
                m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([^\|]+)\|", block)
                return m.group(1).strip() if m else "0"

            results.append({
                "label":        label,
                "passed":       "PASS" in status,
                "sharpe":       _num(get_metric("Sharpe Ratio")),
                "sortino":      _num(get_metric("Sortino Ratio")),
                "calmar":       _num(get_metric("Calmar Ratio")),
                "cagr":         _pct(get_metric("CAGR")),
                "max_drawdown": _pct(get_metric("Max Drawdown")),
                "win_rate":     _pct(get_metric("Win Rate")),
                "profit_factor":_num(get_metric("Profit Factor")),
                "trade_count":  int(_num(get_metric("Trade Count"))),
                "is_sharpe":    _num(get_metric("In-Sample")),
                "oos_sharpe":   _num(get_metric("Out-of-Sample")),
            })
        return results

    # ── backtest equity / drawdown / distribution ─────────────────────────────

    def backtests(self) -> list[dict]:
        section = re.search(r"## Backtest Results(.+?)$", self.text, re.DOTALL)
        if not section:
            return []
        results = []
        for block in re.split(r"(?m)^### ", section.group(1)):
            if not block.strip():
                continue
            label = block.splitlines()[0].strip()

            # Summary line metrics
            m = re.search(r"\*\*Total Return:\*\*\s*([\d.\-+%]+)", block)
            total_return = _pct(m.group(1)) if m else 0.0
            m = re.search(r"\*\*Win Rate:\*\*\s*([\d.%]+)", block)
            win_rate = _pct(m.group(1)) if m else 0.0
            m = re.search(r"\*\*Trade Count:\*\*\s*(\d+)", block)
            trade_count = int(m.group(1)) if m else 0

            # Equity curve
            eq_section = re.search(r"#### Equity Curve(.+?)(?=####|\Z)", block, re.DOTALL)
            equity = {}
            if eq_section:
                for row in _parse_md_table(eq_section.group(1)):
                    d = row.get("Date", "").strip()
                    v = _num(row.get("Portfolio Value", "0"))
                    if d:
                        equity[d] = v

            # Drawdown curve
            dd_section = re.search(r"#### Drawdown Curve(.+?)(?=####|\Z)", block, re.DOTALL)
            drawdown = {}
            if dd_section:
                for row in _parse_md_table(dd_section.group(1)):
                    d = row.get("Date", "").strip()
                    v = _pct(row.get("Drawdown", "0"))
                    if d:
                        drawdown[d] = v

            # Return distribution
            dist_section = re.search(r"#### Return Distribution(.+?)(?=####|\Z)", block, re.DOTALL)
            dist_bins, dist_counts = [], []
            if dist_section:
                for row in _parse_md_table(dist_section.group(1)):
                    bin_str = row.get("Return Bin", "")
                    cnt     = _num(row.get("Count", "0"))
                    m2 = re.search(r"([\-\d.]+)%\s*to\s*([\-\d.]+)%", bin_str)
                    if m2:
                        lo = float(m2.group(1)) / 100
                        hi = float(m2.group(2)) / 100
                        dist_bins.append((lo + hi) / 2)
                        dist_counts.append(int(cnt))

            # Walk-forward
            wf_section = re.search(r"#### Walk-Forward Returns(.+?)(?=####|\Z)", block, re.DOTALL)
            wf = {}
            if wf_section:
                for row in _parse_md_table(wf_section.group(1)):
                    period = _clean(row.get("Period", "")).strip()
                    ret    = _pct(row.get("Cumulative Return", "0"))
                    if period:
                        wf[period] = ret

            # Exit reason breakdown
            exit_section = re.search(r"#### Exit Reason Breakdown(.+?)(?=####|\n\n|\Z)", block, re.DOTALL)
            exits = {}
            if exit_section:
                for row in _parse_md_table(exit_section.group(1)):
                    reason = _clean(row.get("Exit Reason", "")).strip()
                    count  = int(_num(row.get("Count", "0")))
                    if reason:
                        exits[reason] = count

            results.append({
                "label":        label,
                "total_return": total_return,
                "win_rate":     win_rate,
                "trade_count":  trade_count,
                "equity":       equity,
                "drawdown":     drawdown,
                "dist_bins":    dist_bins,
                "dist_counts":  dist_counts,
                "walk_forward": wf,
                "exits":        exits,
            })
        return results

    # ── monte carlo ───────────────────────────────────────────────────────────

    def monte_carlos(self) -> list[dict]:
        section = re.search(r"## Monte Carlo Stress Test(.+?)$", self.text, re.DOTALL)
        if not section:
            return []
        results = []
        for block in re.split(r"(?m)^### ", section.group(1)):
            if not block.strip():
                continue
            ticker = block.splitlines()[0].strip()
            if not ticker or ticker.startswith("_"):
                continue

            def gm(name):
                m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([^\|]+)\|", block)
                return m.group(1).strip() if m else "0"

            def gm3(name):
                m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([^\|]+)\|\s*([^\|]+)\|\s*([^\|]+)\|", block)
                return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip()) if m else ("0","0","0")

            p5f, p50f, p95f = gm3("Final Portfolio ($)")
            p5s, p50s, p95s = gm3("Sharpe Ratio")
            p5w, p50w, p95w = gm3("Win Rate")

            # Equity band
            band_section = re.search(r"#### Equity Confidence Band(.+?)(?=####|\Z)", block, re.DOTALL)
            equity_band = []
            if band_section:
                for row in _parse_md_table(band_section.group(1)):
                    step = _num(row.get("Trade #", "0"))
                    p5   = _num(row.get("P5 ($)", "0"))
                    p50  = _num(row.get("Median ($)", "0"))
                    p95  = _num(row.get("P95 ($)", "0"))
                    if step or p50:
                        equity_band.append({"step": int(step), "p5": p5, "p50": p50, "p95": p95})

            ttr_raw = gm("Median Trade at First Ruin")
            sev_raw = gm("Mean Portfolio at Ruin")

            results.append({
                "ticker":              ticker,
                "p5_final":            _num(p5f),
                "p50_final":           _num(p50f),
                "p95_final":           _num(p95f),
                "p5_sharpe":           _num(p5s),
                "p50_sharpe":          _num(p50s),
                "p95_sharpe":          _num(p95s),
                "p5_win_rate":         _pct(p5w),
                "p50_win_rate":        _pct(p50w),
                "p95_win_rate":        _pct(p95w),
                "p_ruin":              _pct(gm("P(Ruin) — equity falls >40%")),
                "p95_max_drawdown":    _pct(gm("P95 Max Drawdown")),
                "median_cagr":         _pct(gm("Median CAGR")),
                "p95_max_consec":      int(_num(gm("P95 Max Consecutive Losses"))),
                "kelly_fraction":      _num(gm("Optimal Kelly Fraction")),
                "median_time_to_ruin": None if "N/A" in ttr_raw else int(_num(ttr_raw)) if ttr_raw else None,
                "ruin_severity":       None if "N/A" in sev_raw else _num(re.sub(r"[,$]", "", sev_raw)),
                "equity_band":         equity_band,
            })
        return results


# ── chart builders ────────────────────────────────────────────────────────────

COLORS = {
    "buy":    "#2ecc71",
    "watch":  "#f39c12",
    "avoid":  "#e74c3c",
    "pass":   "#2ecc71",
    "fail":   "#e74c3c",
    "equity": "#3498db",
    "dd":     "#e74c3c",
    "is":     "#2ecc71",
    "oos":    "#e67e22",
}


def plot_overview(parser: ReportParser, save_dir: str) -> None:
    """One overview figure: verdict split + regime bar + macro sectors + diagnostics radar."""
    summ     = parser.summary()
    mac      = parser.macro()
    verdicts = parser.verdicts()
    regimes  = parser.regimes()
    diags    = parser.diagnostics()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"MFT Alpha Finder — {summ.get('run_date', '')}   "
        f"Bias: {summ.get('market_bias', '')}   "
        f"Articles: {summ.get('article_count', '')}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Verdict pie ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = {"buy": 0, "watch": 0, "avoid": 0}
    for v in verdicts:
        vc = v["verdict"].replace("*", "").lower()
        if vc in counts:
            counts[vc] += 1
    labels = [k for k, v in counts.items() if v > 0]
    sizes  = [counts[k] for k in labels]
    clrs   = [COLORS[k] for k in labels]
    if sizes:
        ax1.pie(sizes, labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
                colors=clrs, autopct="%1.0f%%", startangle=140,
                textprops={"fontsize": 9})
    ax1.set_title("Ticker Verdicts", fontweight="bold")

    # ── 2. Regime breakdown ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    regime_counts: dict[str, int] = {}
    for r in regimes:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
    if regime_counts:
        labels_r = list(regime_counts.keys())
        vals_r   = list(regime_counts.values())
        bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
        ax2.barh(labels_r, vals_r, color=bar_colors[:len(labels_r)])
        ax2.set_xlabel("# Tickers")
        for i, v in enumerate(vals_r):
            ax2.text(v + 0.05, i, str(v), va="center", fontsize=9)
    ax2.set_title("Regime Classification", fontweight="bold")

    # ── 3. Hurst scatter ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if regimes:
        tickers_r = [r["ticker"] for r in regimes]
        hursts    = [r["hurst"] for r in regimes]
        atrs      = [r["atr_pct"] * 100 for r in regimes]
        regime_color_map = {
            "Trending": "#3498db", "Mean-Reverting": "#2ecc71",
            "High-Volatility": "#e74c3c", "Low-Volatility": "#9b59b6",
            "Neutral": "#95a5a6",
        }
        pt_colors = [regime_color_map.get(r["regime"], "#95a5a6") for r in regimes]
        ax3.scatter(hursts, atrs, c=pt_colors, s=60, zorder=3)
        for t, h, a in zip(tickers_r, hursts, atrs):
            ax3.annotate(t, (h, a), fontsize=7, textcoords="offset points", xytext=(4, 2))
        ax3.axvline(0.55, color="#3498db", linestyle="--", linewidth=0.8, label="Trending (0.55)")
        ax3.axvline(0.45, color="#2ecc71", linestyle="--", linewidth=0.8, label="MR (0.45)")
        ax3.set_xlabel("Hurst Exponent")
        ax3.set_ylabel("ATR/Price (%)")
        ax3.legend(fontsize=7)
    ax3.set_title("Hurst vs Volatility", fontweight="bold")

    # ── 4. Macro sectors ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    favored = mac.get("favored", [])
    avoid   = mac.get("avoid", [])
    all_sectors = favored + avoid
    if all_sectors:
        sector_colors = [COLORS["buy"] if s in favored else COLORS["avoid"] for s in all_sectors]
        y_pos = range(len(all_sectors))
        vals  = [1] * len(all_sectors)
        ax4.barh(list(y_pos), vals, color=sector_colors)
        ax4.set_yticks(list(y_pos))
        ax4.set_yticklabels(all_sectors, fontsize=9)
        ax4.set_xlim(0, 1.5)
        ax4.set_xticks([])
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=COLORS["buy"], label="Favoured"),
                           Patch(facecolor=COLORS["avoid"], label="Avoid")]
        ax4.legend(handles=legend_elements, fontsize=8, loc="lower right")
    ax4.set_title("Macro Sector View", fontweight="bold")

    # ── 5. Diagnostics bar ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if diags:
        labels_d  = [d["label"].split("—")[0].strip() for d in diags]
        sharpes   = [d["sharpe"] for d in diags]
        bar_c     = [COLORS["pass"] if d["passed"] else COLORS["fail"] for d in diags]
        x = np.arange(len(labels_d))
        bars = ax5.bar(x, sharpes, color=bar_c, width=0.5)
        ax5.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Floor (0.5)")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels_d, rotation=30, ha="right", fontsize=8)
        ax5.set_ylabel("Sharpe Ratio")
        ax5.legend(fontsize=8)
        for bar in bars:
            h = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=8)
    ax5.set_title("Sharpe Ratio (green=PASS)", fontweight="bold")

    # ── 6. Walk-forward IS vs OOS ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if diags:
        labels_d = [d["label"].split("—")[0].strip() for d in diags]
        is_s  = [d["is_sharpe"]  for d in diags]
        oos_s = [d["oos_sharpe"] for d in diags]
        x = np.arange(len(labels_d))
        w = 0.35
        ax6.bar(x - w/2, is_s,  w, label="In-Sample",     color=COLORS["is"])
        ax6.bar(x + w/2, oos_s, w, label="Out-of-Sample",  color=COLORS["oos"])
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels_d, rotation=30, ha="right", fontsize=8)
        ax6.set_ylabel("Sharpe")
        ax6.legend(fontsize=8)
    ax6.set_title("Walk-Forward: IS vs OOS Sharpe", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(save_dir, "overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


def plot_ticker(bt: dict, diag: dict | None, save_dir: str) -> None:
    """Four-panel per-ticker figure: equity, drawdown, return dist, exit breakdown."""
    label = bt["label"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(label, fontsize=13, fontweight="bold")

    # ── equity curve ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    if bt["equity"]:
        dates = list(bt["equity"].keys())
        vals  = list(bt["equity"].values())
        ax.plot(range(len(dates)), vals, color=COLORS["equity"], linewidth=1.5)
        step = max(1, len(dates) // 6)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels(dates[::step], rotation=30, ha="right", fontsize=7)
        ax.fill_between(range(len(dates)), vals, alpha=0.15, color=COLORS["equity"])
        ax.set_ylabel("Portfolio Value ($)")
        ret_str = f"{bt['total_return']:+.1%}"
        ax.set_title(f"Equity Curve  [{ret_str}  {bt['trade_count']} trades  WR {bt['win_rate']:.0%}]")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center")
        ax.set_title("Equity Curve")

    # ── drawdown curve ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if bt["drawdown"]:
        dates = list(bt["drawdown"].keys())
        vals  = [v * 100 for v in bt["drawdown"].values()]
        ax.fill_between(range(len(dates)), vals, 0, alpha=0.4, color=COLORS["dd"])
        ax.plot(range(len(dates)), vals, color=COLORS["dd"], linewidth=1.0)
        step = max(1, len(dates) // 6)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels(dates[::step], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Drawdown (%)")
        if vals:
            ax.set_title(f"Drawdown  [max {min(vals):.1f}%]")
        else:
            ax.set_title("Drawdown")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No drawdown data", ha="center", va="center")
        ax.set_title("Drawdown")

    # ── return distribution ───────────────────────────────────────────────────
    ax = axes[1, 0]
    if bt["dist_bins"] and bt["dist_counts"]:
        bins_pct = [b * 100 for b in bt["dist_bins"]]
        bar_colors = [COLORS["avoid"] if b < 0 else COLORS["buy"] for b in bt["dist_bins"]]
        ax.bar(bins_pct, bt["dist_counts"], color=bar_colors,
               width=(bins_pct[1] - bins_pct[0]) * 0.85 if len(bins_pct) > 1 else 0.5,
               edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Return Distribution")
    else:
        ax.text(0.5, 0.5, "No distribution data", ha="center", va="center")
        ax.set_title("Return Distribution")

    # ── exit breakdown + diagnostic metrics ──────────────────────────────────
    ax = axes[1, 1]
    if bt["exits"]:
        reasons = list(bt["exits"].keys())
        counts  = list(bt["exits"].values())
        exit_colors = {"stop_loss": COLORS["avoid"], "trailing_stop": COLORS["watch"],
                       "ma_exit": "#3498db", "max_holding": "#9b59b6", "rsi_exit": "#1abc9c"}
        clrs = [exit_colors.get(r, "#95a5a6") for r in reasons]
        ax.pie(counts, labels=[f"{r}\n({c})" for r, c in zip(reasons, counts)],
               colors=clrs, autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 8})
        ax.set_title("Exit Reason Breakdown")
    elif diag:
        # show diagnostic table as text
        metrics = [
            ("Sharpe",        f"{diag.get('sharpe', 0):.3f}"),
            ("Sortino",       f"{diag.get('sortino', 0):.3f}"),
            ("Calmar",        f"{diag.get('calmar', 0):.3f}"),
            ("CAGR",          f"{diag.get('cagr', 0):.2%}"),
            ("Max DD",        f"{diag.get('max_drawdown', 0):.2%}"),
            ("Win Rate",      f"{diag.get('win_rate', 0):.1%}"),
            ("Profit Factor", f"{diag.get('profit_factor', 0):.3f}"),
        ]
        ax.axis("off")
        table = ax.table(cellText=metrics, colLabels=["Metric", "Value"],
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax.set_title("Diagnostic Metrics")
    else:
        ax.text(0.5, 0.5, "No exit data", ha="center", va="center")
        ax.set_title("Exits")

    plt.tight_layout()
    safe_name = re.sub(r"[^\w\-]", "_", label)
    out_path  = os.path.join(save_dir, f"ticker_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


def plot_monte_carlo(mc: dict, save_dir: str) -> None:
    """3-panel Monte Carlo figure per ticker: equity band, outcome bars, risk table."""
    ticker = mc["ticker"]
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Monte Carlo Stress Test — {ticker}  (10,000 simulations)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Equity confidence band ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])   # span 2 cols
    band = mc["equity_band"]
    if band:
        steps = [e["step"] for e in band]
        p5s   = [e["p5"]   for e in band]
        p50s  = [e["p50"]  for e in band]
        p95s  = [e["p95"]  for e in band]
        ax1.fill_between(steps, p5s, p95s, alpha=0.20, color=COLORS["equity"], label="P5–P95 band")
        ax1.plot(steps, p50s, color=COLORS["equity"], linewidth=2, label="Median")
        ax1.plot(steps, p5s,  color=COLORS["avoid"],  linewidth=1, linestyle="--", label="P5")
        ax1.plot(steps, p95s, color=COLORS["buy"],    linewidth=1, linestyle="--", label="P95")
        ax1.axhline(p50s[0] if p50s else 100_000, color="gray", linewidth=0.7, linestyle=":")
        ax1.set_xlabel("Trade #")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No equity band data", ha="center", va="center")
    ax1.set_title("Equity Confidence Band")

    # ── 2. P(ruin) gauge ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    p_ruin = mc["p_ruin"]
    ruin_color = COLORS["avoid"] if p_ruin > 0.10 else COLORS["watch"] if p_ruin > 0.03 else COLORS["buy"]
    ax2.pie([p_ruin, 1 - p_ruin],
            labels=[f"Ruin\n{p_ruin:.1%}", f"Safe\n{1-p_ruin:.1%}"],
            colors=[ruin_color, "#ecf0f1"],
            startangle=90, textprops={"fontsize": 9},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    ax2.set_title(f"P(Ruin >40%)")

    # ── 3. Final portfolio distribution (p5/p50/p95 bars) ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    labels_p = ["P5", "Median", "P95"]
    vals_p   = [mc["p5_final"], mc["p50_final"], mc["p95_final"]]
    clrs_p   = [COLORS["avoid"], COLORS["equity"], COLORS["buy"]]
    bars = ax3.bar(labels_p, vals_p, color=clrs_p, width=0.5)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar, v in zip(bars, vals_p):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_p)*0.01,
                 f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax3.set_title("Final Portfolio Outcomes")

    # ── 4. Sharpe distribution bars ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    labels_s = ["P5 Sharpe", "Median", "P95 Sharpe"]
    vals_s   = [mc["p5_sharpe"], mc["p50_sharpe"], mc["p95_sharpe"]]
    clrs_s   = [COLORS["avoid"], COLORS["equity"], COLORS["buy"]]
    bars_s = ax4.bar(labels_s, vals_s, color=clrs_s, width=0.5)
    ax4.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Floor (0.5)")
    ax4.axhline(0.0, color="black", linewidth=0.5)
    ax4.legend(fontsize=8)
    for bar, v in zip(bars_s, vals_s):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax4.set_title("Simulated Sharpe Ratio")

    # ── 5. Risk metrics table ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ttr = mc["median_time_to_ruin"]
    sev = mc["ruin_severity"]
    rows = [
        ("Median CAGR",            f"{mc['median_cagr']:.2%}"),
        ("P95 Max Drawdown",       f"{mc['p95_max_drawdown']:.2%}"),
        ("P95 Max Consec. Losses", str(mc["p95_max_consec"])),
        ("Optimal Kelly Fraction", f"{mc['kelly_fraction']:.3f}"),
        ("Median Trade at Ruin",   str(ttr) if ttr is not None else "N/A"),
        ("Mean Equity at Ruin",    f"${sev:,.0f}" if sev is not None else "N/A"),
        ("P5/P50 Win Rate",        f"{mc['p5_win_rate']:.1%} / {mc['p50_win_rate']:.1%}"),
    ]
    tbl = ax5.table(cellText=rows, colLabels=["Metric", "Value"],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax5.set_title("Risk Metrics", pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_name = re.sub(r"[^\w\-]", "_", ticker)
    out_path  = os.path.join(save_dir, f"mc_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Resolve report path
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = _latest_report("reports")

    print(f"\nParsing: {report_path}")
    parser   = ReportParser(report_path)
    save_dir = str(Path(report_path).parent)

    # Overview dashboard
    print("\nGenerating overview dashboard...")
    plot_overview(parser, save_dir)

    # Per-ticker figures — only tickers that actually traded
    backtests   = [b for b in parser.backtests() if b["trade_count"] > 0]
    diagnostics = {d["label"]: d for d in parser.diagnostics()}

    if not backtests:
        print("\nNo backtested tickers with trades found — only overview generated.")
    else:
        print(f"\nGenerating {len(backtests)} ticker figure(s) (skipping tickers with 0 trades)...")
        for bt in backtests:
            diag = diagnostics.get(bt["label"])
            plot_ticker(bt, diag, save_dir)

    # Monte Carlo figures
    mc_results = parser.monte_carlos()
    if mc_results:
        print(f"\nGenerating {len(mc_results)} Monte Carlo figure(s)...")
        for mc in mc_results:
            plot_monte_carlo(mc, save_dir)
    else:
        print("\nNo Monte Carlo data found in report.")

    print(f"\nDone. All PNGs saved to: {save_dir}/")


if __name__ == "__main__":
    main()
