# Alpha Research Framework — Quant-Level Improvement Suggestions

Scope: MFT alpha-seeker and evaluator. Goal is maximising true-positive alpha
identification and minimising false positives — NOT necessarily maximising
trading P&L.

---

## 1. Fixes Already Implemented (This Session)

| Component | Fix | Why |
|---|---|---|
| `diagnostics_engine` | Permutation test gate (perm_p ≥ 0.10 → reject) | Non-parametric complement to Lo t-stat; catches temporal luck |
| `diagnostics_engine` | Rolling Sharpe stability metric (60-day windows) | Flags regime-only strategies that fail outside their regime |
| `diagnostics_engine` | p-value gate (Lo 2002 autocorr-corrected) | Ensures Sharpe is statistically distinguishable from noise |
| `diagnostics_engine` | Bootstrap Sharpe CI lower bound gate (p5 > 0) | Ensures Sharpe is real across resampled histories |
| `diagnostics_engine` | Permutation + bootstrap are skipped when underpowered | Prevents false rejection of low-trade-count strategies |
| `backtester` | `avg_win`, `avg_loss`, `payoff_ratio` in summary | Directly measures the "noise trading" signature |
| `backtester` | `exit_reason_breakdown` in summary | Shows if signal is too short-lived (> 60% alpha_reversal) |
| `backtester` | `avg_holding_days` in summary | Quantifies signal duration vs assumed holding horizon |
| `alpha_engine` | Signal component correlation matrix | Detects redundancy — warns when avg pairwise corr > 0.70 |
| `alpha_engine` | IC by horizon (1, 2, 5, 10 days) | Measures how quickly signal predictive power decays |
| `alpha_engine` | `compute_signal_diagnostics()` method | Callable diagnostic layer separate from the main pipeline |

---

## 2. Critical Structural Issues (Not Yet Fixed)

### 2.1 Cross-Sectional Design Mismatch

**Problem**: The `AlphaEngine` computes cross-sectional (CS) signals correctly —
it ranks stocks relative to each other. But the `Backtester` runs each ticker
independently in a long-only mode. This means:

- You are using a **relative** signal (this stock vs peers) to drive an **absolute** position (just buy this stock)
- The alpha is designed to be market-neutral but the execution is directional
- This structurally dilutes the edge

**Fix Options**:

Option A — True Long-Short Portfolio (recommended if you ever want to trade):
```python
# PortfolioOptimizer should rank all tickers by alpha_signal on the same day
# Long top 20%, short bottom 20%, dollar-neutral

def build_long_short(signal_scores: dict[str, float], capital: float) -> dict[str, float]:
    scores = pd.Series(signal_scores).dropna()
    ranks  = scores.rank(pct=True)
    long   = ranks[ranks > 0.80]
    short  = ranks[ranks < 0.20]
    # Dollar neutral: long notional = short notional
    n = max(len(long), len(short), 1)
    weights = pd.Series(0.0, index=scores.index)
    weights[long.index]  = +1.0 / n
    weights[short.index] = -1.0 / n
    return (weights * capital).to_dict()
```

Option B — Per-ticker time-series alpha (current approach):
- Remove the cross-sectional normalization from `AlphaEngine`
- Each ticker's signal should be computed independently
- Remove `pct_rank`, `_cs_zscore`, and peer comparisons
- Use only time-series features (RSI, momentum, ATR deviation)

**You cannot do both.** Pick one.

---

### 2.2 Signal Redundancy — Orthogonalization

**Problem**: The 4 components (cs_mr, residual, vol_spike, mom_2d) are likely
correlated. The `alpha_engine` now measures this. If `avg_pairwise_corr > 0.70`,
you effectively have 1 factor repeated 4 times.

**Fix**: Orthogonalize via PCA residuals before combining:

```python
from sklearn.decomposition import PCA

# Stack components into (T × N_tickers, 4) matrix
X = np.column_stack([comp.values.flatten() for comp in components.values()])
mask = ~np.any(np.isnan(X), axis=1)
X_clean = X[mask]

# Fit PCA — keep components with eigenvalue > 1 (Kaiser criterion)
pca = PCA()
pca.fit(X_clean)
n_factors = np.sum(pca.explained_variance_ratio_.cumsum() < 0.90) + 1

# Use PCA scores as orthogonal alpha components
X_orth = pca.transform(X_clean)[:, :n_factors]
```

This ensures each combined component adds truly independent information.

---

### 2.3 Event-Driven Regime: Labeling vs Modeling

**Problem**: "Event-Driven" currently means "earnings blackout flag = True" (from
`ohlcv_fetcher`). This is correct regime *detection* but the strategy assigned to
it (`AlphaCombined`) does not actually use any event-specific data.

**What "Event-Driven" should mean**:

| Feature | Current | Needed |
|---|---|---|
| Event detection | Earnings proximity flag | ✅ Already have this |
| Event signal | ATR + volume spike | ❌ No actual event alpha |
| Post-earnings drift | Not modeled | ❌ Missing |
| Surprise direction | Not available | ❌ Missing |

**Fix**: Add at minimum two event features to `ohlcv_fetcher.compute_features()`:

```python
# 1. Earnings gap magnitude: close on day after earnings vs close day before
#    Positive gap = beat expectations (historically drifts +3–5% over 5 days)
#    Negative gap = miss (drifts -3–5% over 5 days)
earnings_gap = close.shift(-1) / close.shift(1) - 1  # +/-gap on announcement

# 2. Post-earnings drift (PEAD) signal
#    Based on Rendleman et al. (1982): drift continues for ~60 days post-earnings
#    Signal: sign(earnings_gap) * normalized_surprise_magnitude
pead_signal = np.sign(earnings_gap) * earnings_gap.abs().clip(upper=0.10) / 0.10
```

Without this, the "Event-Driven" strategy has no informational edge over
the base `AlphaCombined` — it's the same strategy with a different label.

---

### 2.4 Walk-Forward: Rolling Windows vs Static Splits

**Problem**: Current WF uses 3 static splits (60/40, 70/30, 80/20). This is
better than a single split but still has two weaknesses:
1. The OOS periods do not overlap — each split's OOS tests a different regime
2. No test of whether performance is *consistent* over time

**Fix**: Rolling walk-forward (anchored expanding window):

```python
def rolling_walk_forward(returns: pd.Series, min_is: int = 252, oos: int = 63) -> list[dict]:
    """
    Anchored expanding window: IS grows from min_is → end, OOS is always
    the next `oos` bars.  This gives ~(len - min_is) // oos test windows.
    Example: 5 years data, 1y IS min, 3m OOS → ~16 non-overlapping OOS periods.
    """
    results = []
    n = len(returns)
    for end_is in range(min_is, n - oos, oos):
        is_ret  = returns.iloc[:end_is]
        oos_ret = returns.iloc[end_is:end_is + oos]
        results.append({
            "is_sharpe":   sharpe(is_ret),
            "oos_sharpe":  sharpe(oos_ret),
            "window_end":  returns.index[end_is],
        })
    return results
```

Accept only if median OOS Sharpe > 0 across all rolling windows.

---

### 2.5 Execution Cost Model: Volume Participation Rate

**Problem**: Current slippage is ADV-tiered (5–75 bps by liquidity). This is
directionally correct but doesn't account for trade *size* relative to ADV.
A $100K position in a $50M ADV stock is very different from a $10M position.

**Fix**: Add a volume participation rate to scale slippage:

```python
def compute_slippage_bps(adv_shares: float, trade_shares: float, spread_bps: float = 5.0) -> float:
    """
    Total cost = spread (always paid) + market impact (scales with participation).
    Almgren-Chriss simplified: impact ∝ sqrt(participation_rate).
    """
    participation = trade_shares / max(adv_shares, 1.0)
    impact_bps = 10.0 * math.sqrt(participation)   # 10bps at 1% participation
    return spread_bps + impact_bps
```

For MFT at 1% portfolio risk per trade with $100K capital, position size is
~$1K / ATR×price. For large-caps this is usually < 0.1% of ADV — minimal impact.
But for small-caps this can easily reach 5–10% of ADV → 30–70 bps of impact.

---

### 2.6 Parameter Sensitivity Sweep

**Problem**: Strategy params (lookback, ATR multipliers, thresholds) are set by
`StrategySelector` using deterministic rules. There is no test of whether
performance is sensitive to small param changes — a hallmark of overfitting.

**Fix**: Add a parameter sweep to `StrategySelector` or a new
`parameter_sensitivity.py` module:

```python
def param_sensitivity_sweep(
    ticker: str,
    strategy: dict,
    ohlcv: pd.DataFrame,
    param_grid: dict[str, list],   # e.g. {"stop_loss_atr": [1.0, 1.5, 2.0, 2.5]}
) -> dict:
    """
    Vary each parameter independently (one-at-a-time), run backtest, measure
    Sharpe change.  If Sharpe degrades > 50% from a ±20% param change → overfit.
    """
    from backtester import Backtester
    from diagnostics_engine import DiagnosticsEngine

    bt = Backtester()
    base_sharpe = DiagnosticsEngine._sharpe(bt.run(ticker, strategy, ohlcv)["returns"])

    results = {}
    for param, values in param_grid.items():
        sharpes = []
        for val in values:
            modified = copy.deepcopy(strategy)
            modified["adjusted_params"][param] = val
            ret = bt.run(ticker, modified, ohlcv)["returns"]
            sharpes.append(DiagnosticsEngine._sharpe(ret))
        results[param] = {
            "values":   values,
            "sharpes":  sharpes,
            "range":    max(sharpes) - min(sharpes),
            "stable":   (max(sharpes) - min(sharpes)) < 0.3,
        }
    return results
```

Reject strategies where any parameter has sensitivity > 0.30 Sharpe units
across a ±30% range.

---

### 2.7 Benchmark Comparison: Matched Holding Period

**Problem**: The report compares strategy returns against "SPY exposure-adjusted"
which is computed as SPY return over only the days the strategy was in position.
This creates a biased comparison because:
- The strategy cherry-picks entry timing (not random)
- SPY over the same windows is influenced by the strategy's entry logic

**Correct benchmarks**:

1. **Buy-and-hold SPY** over the full backtest period — the simplest and fairest
2. **Random entry baseline** — same number of trades, same average holding days,
   random entry dates: this is what `_permutation_test` approximates
3. **Sector ETF** — compare AAPL strategy vs XLK (tech ETF), not SPY

Add this to `backtester._summarize()`:

```python
def _benchmark_returns(ohlcv, trade_log, benchmark_close):
    """Compute buy-and-hold return over the same period as the backtest."""
    if not trade_log:
        return 0.0
    start = min(t["entry_date"] for t in trade_log)
    end   = max(t["exit_date"]  for t in trade_log)
    bm = benchmark_close.loc[start:end]
    if len(bm) < 2:
        return 0.0
    return float(bm.iloc[-1] / bm.iloc[0] - 1)
```

---

## 3. Alpha Signal Quality: What Real Edge Looks Like

These metrics are now computed by the updated system. Use them to diagnose
whether signals have genuine predictive power:

| Metric | Noise Level | Marginal Edge | Real Edge |
|---|---|---|---|
| Sharpe (full period) | < 0.3 | 0.3–0.7 | > 1.0 |
| p-value (Lo 2002) | > 0.20 | 0.05–0.20 | < 0.05 |
| Permutation p-value | > 0.15 | 0.05–0.15 | < 0.05 |
| Bootstrap Sharpe p5 | ≤ 0 | 0–0.2 | > 0.3 |
| IC (1-day horizon) | < 0.01 | 0.01–0.03 | > 0.05 |
| Payoff ratio (avg_win/avg_loss) | < 1.0 | 1.0–1.5 | > 1.5 |
| Rolling pct positive windows | < 50% | 50–70% | > 70% |
| Avg pairwise signal corr | > 0.80 | 0.40–0.80 | < 0.40 |

The current system is now hardened against false positives via the first four.
The IC, payoff ratio, and signal correlation are reported but not gated — they
provide diagnostic context for why a strategy was rejected or should be improved.

---

## 4. What To Build Next (Priority Order)

1. **Parameter sensitivity sweep** — medium effort, high diagnostic value.
   Implement `parameter_sensitivity.py` using the pattern above.
   Add sweep results to the diagnostic report per ticker.

2. **Rolling walk-forward** — medium effort, highest validation improvement.
   Replaces static 3-split WF with ~16 non-overlapping OOS windows over 5 years.
   Accept only if median OOS Sharpe > 0 across all windows.

3. **Post-earnings drift feature** — adds real event alpha to the "Event-Driven"
   regime. Without it the regime label is misleading.

4. **Signal orthogonalization (PCA)** — reduces the 4 correlated components to
   ~2 truly independent factors. Will improve Sharpe if signals overlap > 0.7.

5. **Long-short portfolio construction** — only relevant if you eventually want
   to trade. For the evaluator role, this is lower priority than validation quality.

---

## 5. What Is Already Working Well

| Feature | Assessment |
|---|---|
| Signal shift(1) everywhere | ✅ No lookahead — correctly implemented |
| ADV-tiered slippage | ✅ Directionally correct; volume participation is the next upgrade |
| Earnings blackout in regime + strategy gating | ✅ Prevents trading into catalyst events |
| Multiple WF splits (60/40, 70/30, 80/20) | ✅ Better than single split |
| Lo (2002) autocorr-corrected t-stat | ✅ Correctly handles return autocorrelation |
| Block bootstrap CI (block=20) | ✅ Preserves volatility clustering structure |
| Kelly fraction gate (negative Kelly → reject) | ✅ Catches provably losing strategies |
| Profit factor bypass for win-rate floor | ✅ Correctly handles trend-following strategies |
| Sector diversity enforcement | ✅ Prevents concentration in single sector |

The foundation is solid. The improvements above sharpen the **false-positive
rejection layer** — which is exactly what an alpha evaluator should prioritize.
