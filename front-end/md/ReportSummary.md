# MFT Alpha Finder & Evaluator — Report 2026-03-28

## Executive Summary

**Run date:** 2026-03-28  
**News window:** 2026-03-21 → 2026-03-28  
**Articles analysed:** 305  
**Overall market bias:** NEUTRAL  

**Buy candidates (0):** None  
**Watch (0):** None  
**Avoid (0):** None  

**Top themes:** Semiconductor supply chain constraints, Cybersecurity sector vulnerability, Stagflation risk resurgence

**Key risks:** Taiwan semiconductor capacity maxed out disrupting global chip supply, Anthropic AI data leak exposing cybersecurity sector weaknesses

## Macro Environment

**Market bias:** NEUTRAL

**Favoured sectors:**   
**Avoid sectors:**   
**Active macro risks:**   

> Insufficient data to form a macro view.

## Shortlisted Tickers

Ticker  Verdict  Reasoning

## Regime Classification

Ticker  Regime         Hurst  ATR/Price  Near Earnings
MSFT    Trending-Down  0.665  2.44%      Yes
AAPL    Trending-Down  0.630  2.26%      Yes
BA      Trending-Down  0.650  3.59%      Yes
JPM     Trending-Down  0.643  2.57%      Yes
AMZN    Trending-Down  0.587  2.91%      Yes
GOOGL   Trending-Down  0.587  2.86%      Yes
GS      Trending-Down  0.631  3.53%      Yes
CVX     Trending-Up    0.588  1.99%      Yes
JNJ     Trending-Down  0.654  1.72%      Yes
TSLA    Trending-Down  0.631  3.73%      Yes
MRK     Trending-Down  0.606  2.19%      Yes
WMT     Trending-Down  0.597  2.29%      Yes
CAT     Trending-Down  0.641  3.43%      Yes
SPY     Trending-Down  0.600  1.55%      No
TSM     Trending-Down  0.574  3.79%      Yes

## Strategy Parameters

### MSFT — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to align with MSFT's current Trending-Down regime, leveraging the Hurst exponent's moderate trend persistence (0.665) and volatility (2.445% ATR/price) by setting a conservative stop-loss (1.5 ATR) and a wider trailing stop (2.0 ATR) to accommodate directional movement, while the 7-day holding limit balances risk exposure with the neutral market bias and volume dynamics.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal 0.467 <= threshold 0.55
```

### AAPL — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to capitalize on AAPL's current Trending-Down regime, with a Hurst exponent of 0.63 indicating persistent trend strength, a stop loss and trailing stop set at 1.5x and 2.0x ATR (aligned with the 2.255% volatility) to manage risk while allowing room for the downtrend to unfold, and a 7-day holding limit reflecting the regime's expected duration and the neutral market bias.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.889 <= threshold 0.55
```

### BA — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage risk in a moderately volatile, trending-down regime (Hurst 0.65, ATR/price 3.585%), with a stop loss and trailing stop set at 2.0x and 2.5x ATR to accommodate volatility while maintaining a 7-day holding period, and reversal thresholds aligned to capture potential short-term mean reversion within the downtrend.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (2.0 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.5 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       2.0
trailing_stop_atr   2.5
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)
- High ATR 3.59%: stop_loss_atr=2.0, trailing=2.5

#### Current Entry Signal (as of run date)

**Status: ACTIVE — entry condition met on latest bar**
```
alpha_signal 0.630 > threshold 0.55
```

> NOTE: **Trade setup suppressed — this ticker FAILED diagnostic floors.**  
> The entry signal fired but the backtest has no demonstrated edge. Do not trade.

### JPM — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage risk in JPM's current Trending-Down regime by using a moderate stop-loss (1.5 ATR) and trailing stop (2.0 ATR) aligned with its volatility (2.57% ATR/price), while the alpha_threshold (0.55) and reversal_threshold (-0.5) target sustained downward momentum indicated by the Hurst exponent (0.643), and the 7-day holding limit balances the neutral market bias with the need to capture trend persistence.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal 0.076 <= threshold 0.55
```

### AMZN — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage AMZN's moderate volatility (ATR/price: 2.907%) with a balanced stop-loss (1.5 ATR) and trailing stop (2.0 ATR) to protect against adverse moves while allowing room for the Trending-Down regime's directional bias, while the alpha threshold (0.55) and 7-day holding period align with the Hurst exponent (0.587) suggesting weak persistence and the neutral market bias, ensuring timely exits without overcommitting to a fading trend.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.461 <= threshold 0.55
```

### GOOGL — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage GOOGL's moderate volatility (ATR/price 2.857%) with a balanced stop-loss (1.5 ATR) and trailing stop (2.0 ATR) to protect against downside risk in the Trending-Down regime, while the alpha threshold (0.55) and 7-day holding limit align with the Hurst exponent's subdiffusive persistence (0.587) and neutral market bias, ensuring disciplined exits without overreacting to transient price fluctuations.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: ACTIVE — entry condition met on latest bar**
```
alpha_signal 1.610 > threshold 0.55
```

> NOTE: **Trade setup suppressed — this ticker FAILED diagnostic floors.**  
> The entry signal fired but the backtest has no demonstrated edge. Do not trade.

### GS — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are tailored to the Trending-Down regime and GS's volatility profile, with a Hurst exponent of 0.631 indicating persistent trend behavior, an ATR/price of 3.53% justifying a stop-loss and trailing-stop ATR of 2.0 and 2.5 to manage volatility, an alpha_threshold of 0.55 and reversal_threshold of -0.5 aligning with moderate trend strength and risk of reversal, a 7-day holding limit to avoid overexposure in a neutral-volume, low-liquidity environment.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (2.0 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.5 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       2.0
trailing_stop_atr   2.5
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)
- High ATR 3.53%: stop_loss_atr=2.0, trailing=2.5

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.275 <= threshold 0.55
```

### CVX — Momentum

**Regime:** Trending-Up  
**Reasoning:** The parameters are calibrated to capitalize on CVX's moderate volatility (ATR/price: 1.99%) and slight trend persistence (Hurst: 0.588) by using a 10-day momentum lookback for timely entries, a 2.0 ATR trailing stop to lock in gains during the Trending-Up regime, and a 20-day MA exit to manage risk, while the 1.2 volume multiplier and 20-day max holding period align with the neutral market bias and avoid overexposure.

#### Strategy Mechanics

**Why it works:** Momentum strategies exploit the empirical tendency of assets with high Hurst exponents (H > 0.55) to persist in their current direction. Requiring a volume surge at breakout filters false breakouts driven by thin liquidity, keeping the signal anchored to genuine institutional participation. ATR-based stops let volatility scale the exit distance, avoiding premature stops in volatile regimes while still capping loss per trade at ~1% of capital.

**Order type:** Market order at next session open.

**Entry (both conditions required):**
- Price breakout: Close > 10-day rolling high (prior session close)
- Volume confirmation: Volume > 1.2× 20-day average volume

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Hard stop loss** — Close < entry price − 1.5 × ATR₁₄
2. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
3. **MA exit** — Close < 20-day simple moving average
4. **Max holding** — Force exit after 20 trading days

#### Adjusted Parameters

Parameter          Value
entry_lookback     10
volume_multiplier  1.2
trailing_stop_atr  2.0
ma_exit_period     20
stop_loss_atr      1.5
max_holding_days   20
momentum_lookback  252

#### Current Entry Signal (as of run date)

**Status: INACTIVE — volume confirmation not met**
```
Close 211.15 > 10d high 207.79 | Volume 13,894,100 <= 1.2× avg 18,253,698
```

### JNJ — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage JNJ's moderate volatility (ATR/price: 1.719%) within a Trending-Down regime, using a tighter stop-loss (1.5 ATR) to protect against reversals indicated by the Hurst exponent (0.654) and a trailing stop (2.0 ATR) to lock in gains as the trend persists, while the 7-day holding limit aligns with the neutral market bias and volume profile (0.87) to avoid overexposure.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

> **LLM Alpha Hypothesis:** Disagrees with regime-rule selection. Suggests **Momentum** instead. Reason: _The regime is Trending-Down with Hurst > 0.55, aligning with Momentum's edge in trend persistence, despite the negative 20d return, as trend-following strategies can capitalize on continued directional moves._

#### Current Entry Signal (as of run date)

**Status: ACTIVE — entry condition met on latest bar**
```
alpha_signal 0.559 > threshold 0.55
```

> NOTE: **Trade setup suppressed — this ticker FAILED diagnostic floors.**  
> The entry signal fired but the backtest has no demonstrated edge. Do not trade.

### TSLA — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to align with TSLA's current Trending-Down regime, leveraging the Hurst exponent (0.631) indicating persistent trend strength, ATR/price (3.726%) to set conservative stop-loss (2.0 ATR) and trailing-stop (2.5 ATR) levels that account for volatility, and a 7-day holding limit to manage risk in a neutral market with moderate volume imbalance (1.04).

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (2.0 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.5 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       2.0
trailing_stop_atr   2.5
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)
- High ATR 3.73%: stop_loss_atr=2.0, trailing=2.5

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal 0.297 <= threshold 0.55
```

### MRK — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are tailored to MRK's current Trending-Down regime and moderate volatility (ATR/price: 2.19%), with a 0.55 alpha_threshold to capture persistent trend momentum, -0.5 reversal_threshold to exit weakening downtrends, ATR-based stops (1.5x/2.0x) to manage volatility, and a 7-day holding limit to align with neutral market bias and avoid overexposure in a low-volume (0.80 ratio) environment.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.202 <= threshold 0.55
```

### WMT — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage WMT's moderate volatility (ATR/price 2.286%) with a balanced stop-loss (1.5x ATR) and trailing stop (2.0x ATR), aligning with the Trending-Down regime's need for disciplined risk management, while the alpha_threshold (0.55) and reversal_threshold (-0.5) reflect sensitivity to weak trend persistence (Hurst 0.597) and neutral market conditions, ensuring timely exits within a 7-day holding window.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.035 <= threshold 0.55
```

### CAT — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage risk in CAT's moderate volatility (ATR/price 3.43%) and trending-down regime, with a conservative stop-loss (2.0 ATR) and trailing stop (2.5 ATR) to protect against reversals, a 7-day holding limit to align with the Hurst exponent's trending persistence (0.641), and thresholds that balance the neutral market bias with the strategy's directional focus.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (2.0 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.5 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       2.0
trailing_stop_atr   2.5
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)
- High ATR 3.43%: stop_loss_atr=2.0, trailing=2.5

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.523 <= threshold 0.55
```

### SPY — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are tailored to the Trending-Down regime and SPY's volatility profile, with a moderate alpha_threshold (0.55) and reversal_threshold (-0.5) to capture sustained downward momentum, stop_loss_atr (1.5) and trailing_stop_atr (2.0) aligned with the ATR/price (1.545%) to manage risk within the current volatility, and a 7-day holding limit to capitalize on short-term trend persistence amid neutral market bias and a Hurst exponent (0.6) suggesting moderate trend strength.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (1.5 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.0 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       1.5
trailing_stop_atr   2.0
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)

> **LLM Alpha Hypothesis:** Disagrees with regime-rule selection. Suggests **Momentum** instead. Reason: _The high Hurst exponent (0.60) and trending-down regime favor trend persistence, making Momentum strategies more suitable than AlphaCombined's diversified, market-neutral approach._

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.418 <= threshold 0.55
```

### TSM — AlphaCombined

**Regime:** Trending-Down  
**Reasoning:** The parameters are calibrated to manage TSM's moderate volatility (ATR/price: 3.79%) with a 2.0 ATR stop loss and 2.5 ATR trailing stop, aligning with the Trending-Down regime's need for disciplined risk control, while the alpha threshold (0.55) and 7-day holding limit reflect the neutral market bias and Hurst exponent (0.574) suggesting a weakly persistent trend requiring timely exits.

#### Strategy Mechanics

**Why it works:** AlphaCombined blends four cross-sectional signals — cross-sectional mean-reversion (40%), market-neutral residual reversion (30%), volume-spike exhaustion (20%), and 2-day momentum (10%) — all lagged by one bar to prevent look-ahead bias. Because each signal is z-scored cross-sectionally (ranked across all tickers in the universe each day), the combined alpha is market-neutral by construction and adapts to whichever tickers are relatively mis-priced on any given day. This multi-factor approach produces 200+ trades over a 2-year backtest window, providing the statistical power required for walk-forward validation.

**Order type:** Market order at next session open.

**Entry condition:**
- Cross-sectional alpha signal > 0.55 (normalised z-score threshold)

**Position sizing:** 1% portfolio risk ÷ (2.0 × ATR₁₄) = shares to buy

**Exit rules (checked in priority order each day):**
1. **Trailing stop** — Close < highest close since entry − 2.5 × ATR₁₄
2. **Alpha reversal** — alpha signal drops below -0.5 (signal exhaustion)
3. **Max holding** — Force exit after 7 trading days

#### Adjusted Parameters

Parameter           Value
alpha_threshold     0.55
reversal_threshold  -0.5
stop_loss_atr       2.0
trailing_stop_atr   2.5
max_holding_days    7

**LLM adjustments:**

- Trending-Down: alpha_threshold=0.55, max_holding_days=7 (only trade strongest idiosyncratic bounce signals)
- High ATR 3.79%: stop_loss_atr=2.0, trailing=2.5

#### Current Entry Signal (as of run date)

**Status: INACTIVE — entry condition not met**
```
alpha_signal -0.316 <= threshold 0.55
```


## Diagnostic Results

### MSFT — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.160 below regime floor 0.4 (OOS 0.592 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.160
Sortino Ratio           0.168
Calmar Ratio            0.448
CAGR                    5.31%
Annualised Volatility   5.05%
Max Drawdown            11.85%
Max DD Recovery (days)  549
VaR 95% (daily)         -0.503%
CVaR 95% (daily)        -0.802%

#### Trade Statistics

Metric                             Value
Trade Count                        316
Win Rate                           53.5%
Entry Efficiency (% reaching +1R)  21.5%
Avg Win                            888.28
Avg Loss                           -801.30
Payoff Ratio (avg_win / avg_loss)  1.109
Avg Holding Days                   3.5
Profit Factor                      1.274
Max Consecutive Losses             6

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.956  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  54.8%  Consistent
Rolling Sharpe Std Dev               1.881  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      0.239   47.86%
Out-of-Sample  -0.038  13.42%
Degradation    —       14.6%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  255
stop_loss       41
max_holding     12
trailing_stop   8

### AAPL — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.197 below regime floor 0.4 (OOS 0.244 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.197
Sortino Ratio           0.220
Calmar Ratio            0.703
CAGR                    5.57%
Annualised Volatility   5.41%
Max Drawdown            7.93%
Max DD Recovery (days)  457
VaR 95% (daily)         -0.490%
CVaR 95% (daily)        -0.819%

#### Trade Statistics

Metric                             Value
Trade Count                        315
Win Rate                           55.9%
Entry Efficiency (% reaching +1R)  22.2%
Avg Win                            928.52
Avg Loss                           -909.91
Payoff Ratio (avg_win / avg_loss)  1.020
Avg Holding Days                   3.5
Profit Factor                      1.292
Max Consecutive Losses             6

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.452  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  52.1%  Consistent
Rolling Sharpe Std Dev               1.822  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      0.378   57.80%
Out-of-Sample  -0.361  8.88%
Degradation    —       41.1%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  259
stop_loss       42
trailing_stop   11
max_holding     3

### BA — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.335 below regime floor 0.4 (OOS -0.357 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.335
Sortino Ratio           -0.362
Calmar Ratio            0.211
CAGR                    2.60%
Annualised Volatility   5.36%
Max Drawdown            12.32%
Max DD Recovery (days)  610
VaR 95% (daily)         -0.483%
CVaR 95% (daily)        -0.834%

#### Trade Statistics

Metric                             Value
Trade Count                        332
Win Rate                           53.0%
Entry Efficiency (% reaching +1R)  23.5%
Avg Win                            743.03
Avg Loss                           -806.25
Payoff Ratio (avg_win / avg_loss)  0.922
Avg Holding Days                   3.6
Profit Factor                      1.040
Max Consecutive Losses             7

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.651  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  47.3%  Regime-dependent
Rolling Sharpe Std Dev               2.389  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.257  23.39%
Out-of-Sample  -0.507  4.67%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  261
stop_loss       54
trailing_stop   9
max_holding     8

### JPM — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.208 below regime floor 0.4 (OOS -0.299 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.208
Sortino Ratio           -0.222
Calmar Ratio            0.240
CAGR                    3.31%
Annualised Volatility   5.33%
Max Drawdown            13.76%
Max DD Recovery (days)  1078
VaR 95% (daily)         -0.501%
CVaR 95% (daily)        -0.859%

#### Trade Statistics

Metric                             Value
Trade Count                        331
Win Rate                           51.1%
Entry Efficiency (% reaching +1R)  26.6%
Avg Win                            805.70
Avg Loss                           -769.09
Payoff Ratio (avg_win / avg_loss)  1.048
Avg Holding Days                   3.6
Profit Factor                      1.093
Max Consecutive Losses             6

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.893  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  43.2%  Regime-dependent
Rolling Sharpe Std Dev               2.364  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.560  10.28%
Out-of-Sample  0.595   25.46%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  254
stop_loss       54
trailing_stop   9
max_holding     14

### AMZN — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.128 below regime floor 0.4 (OOS 0.364 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.128
Sortino Ratio           0.138
Calmar Ratio            0.841
CAGR                    5.18%
Annualised Volatility   5.49%
Max Drawdown            6.16%
Max DD Recovery (days)  422
VaR 95% (daily)         -0.508%
CVaR 95% (daily)        -0.835%

#### Trade Statistics

Metric                             Value
Trade Count                        290
Win Rate                           53.8%
Entry Efficiency (% reaching +1R)  26.9%
Avg Win                            886.85
Avg Loss                           -787.73
Payoff Ratio (avg_win / avg_loss)  1.126
Avg Holding Days                   4.0
Profit Factor                      1.311
Max Consecutive Losses             7

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.067  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  49.1%  Regime-dependent
Rolling Sharpe Std Dev               1.978  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      0.004   35.57%
Out-of-Sample  0.489   22.14%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  222
stop_loss       43
trailing_stop   12
max_holding     13

### GOOGL — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.372 below regime floor 0.4 (OOS 0.496 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.372
Sortino Ratio           0.399
Calmar Ratio            1.139
CAGR                    6.43%
Annualised Volatility   5.01%
Max Drawdown            5.65%
Max DD Recovery (days)  143
VaR 95% (daily)         -0.495%
CVaR 95% (daily)        -0.778%

#### Trade Statistics

Metric                             Value
Trade Count                        298
Win Rate                           56.7%
Entry Efficiency (% reaching +1R)  27.9%
Avg Win                            940.36
Avg Loss                           -853.96
Payoff Ratio (avg_win / avg_loss)  1.101
Avg Holding Days                   3.8
Profit Factor                      1.443
Max Consecutive Losses             6

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.207  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  52.9%  Consistent
Rolling Sharpe Std Dev               1.860  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      0.364   55.27%
Out-of-Sample  0.403   20.04%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  243
max_holding     9
stop_loss       37
trailing_stop   9

### GS — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.466 below regime floor 0.4 (OOS 0.108 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.466
Sortino Ratio           -0.506
Calmar Ratio            0.182
CAGR                    2.01%
Annualised Volatility   5.11%
Max Drawdown            11.03%
Max DD Recovery (days)  1070
VaR 95% (daily)         -0.515%
CVaR 95% (daily)        -0.836%

#### Trade Statistics

Metric                             Value
Trade Count                        306
Win Rate                           50.3%
Entry Efficiency (% reaching +1R)  23.9%
Avg Win                            634.14
Avg Loss                           -674.04
Payoff Ratio (avg_win / avg_loss)  0.941
Avg Holding Days                   3.8
Profit Factor                      0.953
Max Consecutive Losses             8

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.435  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  43.5%  Regime-dependent
Rolling Sharpe Std Dev               1.833  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.751  3.19%
Out-of-Sample  0.248   18.20%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  232
trailing_stop   13
stop_loss       46
max_holding     15

### CVX — Momentum [FAIL]

**Reject reason:** Sharpe ratio -0.270 below regime floor 0.45 (OOS -0.198 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.270
Sortino Ratio           -0.156
Calmar Ratio            0.951
CAGR                    3.71%
Annualised Volatility   3.02%
Max Drawdown            3.90%
Max DD Recovery (days)  576
VaR 95% (daily)         -0.175%
CVaR 95% (daily)        -0.509%

#### Trade Statistics

Metric                             Value
Trade Count                        37
Win Rate                           32.4%
Entry Efficiency (% reaching +1R)  45.9%
Avg Win                            1898.85
Avg Loss                           -953.72
Payoff Ratio (avg_win / avg_loss)  1.991
Avg Holding Days                   11.4
Profit Factor                      0.956
Max Consecutive Losses             8

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.278  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  32.3%  Regime-dependent
Rolling Sharpe Std Dev               1.992  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.165  31.61%
Out-of-Sample  -0.564  9.27%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason    Count
ma_exit        14
trailing_stop  11
max_holding    6
stop_loss      6

### JNJ — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.253 below regime floor 0.4 (OOS -0.160 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.253
Sortino Ratio           -0.276
Calmar Ratio            0.307
CAGR                    3.20%
Annualised Volatility   4.87%
Max Drawdown            10.41%
Max DD Recovery (days)  1255
VaR 95% (daily)         -0.447%
CVaR 95% (daily)        -0.782%

#### Trade Statistics

Metric                             Value
Trade Count                        340
Win Rate                           53.2%
Entry Efficiency (% reaching +1R)  20.9%
Avg Win                            677.47
Avg Loss                           -716.15
Payoff Ratio (avg_win / avg_loss)  0.946
Avg Holding Days                   3.4
Profit Factor                      1.077
Max Consecutive Losses             8

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.665  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  47.1%  Regime-dependent
Rolling Sharpe Std Dev               2.090  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.194  26.86%
Out-of-Sample  -0.406  7.93%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  273
trailing_stop   4
max_holding     13
stop_loss       50

### TSLA — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.351 below regime floor 0.4 (OOS -0.345 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.351
Sortino Ratio           -0.390
Calmar Ratio            0.227
CAGR                    2.56%
Annualised Volatility   5.23%
Max Drawdown            11.27%
Max DD Recovery (days)  907
VaR 95% (daily)         -0.504%
CVaR 95% (daily)        -0.801%

#### Trade Statistics

Metric                             Value
Trade Count                        345
Win Rate                           51.9%
Entry Efficiency (% reaching +1R)  22.6%
Avg Win                            720.75
Avg Loss                           -744.04
Payoff Ratio (avg_win / avg_loss)  0.969
Avg Holding Days                   3.3
Profit Factor                      1.045
Max Consecutive Losses             5

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.550  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  43.9%  Regime-dependent
Rolling Sharpe Std Dev               2.213  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.219  25.30%
Out-of-Sample  -0.651  2.71%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
stop_loss       50
alpha_reversal  282
trailing_stop   7
max_holding     6

### MRK — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.045 below regime floor 0.4 (OOS -0.057 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.045
Sortino Ratio           -0.052
Calmar Ratio            0.601
CAGR                    4.21%
Annualised Volatility   5.28%
Max Drawdown            7.01%
Max DD Recovery (days)  292
VaR 95% (daily)         -0.448%
CVaR 95% (daily)        -0.773%

#### Trade Statistics

Metric                             Value
Trade Count                        337
Win Rate                           55.2%
Entry Efficiency (% reaching +1R)  21.4%
Avg Win                            745.85
Avg Loss                           -776.82
Payoff Ratio (avg_win / avg_loss)  0.960
Avg Holding Days                   3.5
Profit Factor                      1.183
Max Consecutive Losses             7

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.150  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  45.7%  Regime-dependent
Rolling Sharpe Std Dev               1.935  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      0.084   39.92%
Out-of-Sample  -0.366  7.88%
Degradation    —       20.3%

#### Exit Reason Breakdown

Exit Reason     Count
stop_loss       44
alpha_reversal  275
trailing_stop   7
max_holding     11

### WMT — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.376 below regime floor 0.4 (OOS 0.396 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.376
Sortino Ratio           0.380
Calmar Ratio            0.956
CAGR                    6.51%
Annualised Volatility   5.14%
Max Drawdown            6.81%
Max DD Recovery (days)  319
VaR 95% (daily)         -0.430%
CVaR 95% (daily)        -0.752%

#### Trade Statistics

Metric                             Value
Trade Count                        328
Win Rate                           57.9%
Entry Efficiency (% reaching +1R)  21.0%
Avg Win                            773.25
Avg Loss                           -718.03
Payoff Ratio (avg_win / avg_loss)  1.077
Avg Holding Days                   3.4
Profit Factor                      1.483
Max Consecutive Losses             5

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.340  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  57.9%  Consistent
Rolling Sharpe Std Dev               1.854  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.099  31.12%
Out-of-Sample  1.412   43.09%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  278
stop_loss       33
max_holding     8
trailing_stop   9

### CAT — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.251 below regime floor 0.4 (OOS 0.352 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.251
Sortino Ratio           0.278
Calmar Ratio            0.439
CAGR                    5.87%
Annualised Volatility   5.39%
Max Drawdown            13.38%
Max DD Recovery (days)  486
VaR 95% (daily)         -0.467%
CVaR 95% (daily)        -0.811%

#### Trade Statistics

Metric                             Value
Trade Count                        310
Win Rate                           54.8%
Entry Efficiency (% reaching +1R)  27.4%
Avg Win                            891.08
Avg Loss                           -810.15
Payoff Ratio (avg_win / avg_loss)  1.100
Avg Holding Days                   3.7
Profit Factor                      1.336
Max Consecutive Losses             6

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.981  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  55.8%  Consistent
Rolling Sharpe Std Dev               2.240  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.062  32.36%
Out-of-Sample  1.016   33.59%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
stop_loss       50
alpha_reversal  246
trailing_stop   6
max_holding     8

### SPY — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio -0.016 below regime floor 0.4 (OOS 0.230 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            -0.016
Sortino Ratio           -0.016
Calmar Ratio            0.438
CAGR                    4.32%
Annualised Volatility   5.95%
Max Drawdown            9.87%
Max DD Recovery (days)  468
VaR 95% (daily)         -0.565%
CVaR 95% (daily)        -1.000%

#### Trade Statistics

Metric                             Value
Trade Count                        361
Win Rate                           54.0%
Entry Efficiency (% reaching +1R)  26.6%
Avg Win                            841.16
Avg Loss                           -870.32
Payoff Ratio (avg_win / avg_loss)  0.966
Avg Holding Days                   3.4
Profit Factor                      1.135
Max Consecutive Losses             8

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.373  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  51.6%  Consistent
Rolling Sharpe Std Dev               2.190  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.041  32.84%
Out-of-Sample  0.049   14.83%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  281
max_holding     15
stop_loss       50
trailing_stop   15

### TSM — AlphaCombined [FAIL]

**Reject reason:** Sharpe ratio 0.085 below regime floor 0.4 (OOS 0.414 cannot rescue a failed full-period Sharpe)  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.085
Sortino Ratio           0.092
Calmar Ratio            0.536
CAGR                    4.93%
Annualised Volatility   5.49%
Max Drawdown            9.20%
Max DD Recovery (days)  516
VaR 95% (daily)         -0.543%
CVaR 95% (daily)        -0.859%

#### Trade Statistics

Metric                             Value
Trade Count                        333
Win Rate                           52.9%
Entry Efficiency (% reaching +1R)  25.8%
Avg Win                            817.34
Avg Loss                           -727.59
Payoff Ratio (avg_win / avg_loss)  1.123
Avg Holding Days                   3.4
Profit Factor                      1.259
Max Consecutive Losses             7

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.554  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  53.0%  Consistent
Rolling Sharpe Std Dev               2.088  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      -0.188  25.87%
Out-of-Sample  0.764   28.50%
Degradation    —       0.0%

#### Exit Reason Breakdown

Exit Reason     Count
alpha_reversal  276
stop_loss       41
trailing_stop   10
max_holding     6

### AlphaCombined_Portfolio — AlphaCombined [FAIL]

**Reject reason:** Max drawdown 50.7% exceeds regime floor 25%  

#### Core Metrics

Metric                  Value
Sharpe Ratio            0.784
Sortino Ratio           0.808
Calmar Ratio            0.449
CAGR                    22.76%
Annualised Volatility   24.20%
Max Drawdown            50.72%
Max DD Recovery (days)  849
VaR 95% (daily)         -2.408%
CVaR 95% (daily)        -4.298%

#### Trade Statistics

Metric                             Value
Trade Count                        4139
Win Rate                           53.8%
Entry Efficiency (% reaching +1R)  16.7%
Avg Win                            2252.13
Avg Loss                           -2265.70
Payoff Ratio (avg_win / avg_loss)  0.994
Avg Holding Days                   3.4
Profit Factor                      1.156
Max Consecutive Losses             20

#### Alpha Quality Diagnostics

Metric                               Value  Interpretation
Permutation p-value (Calmar)         0.950  < 0.10 = temporal structure present
Rolling Sharpe (% positive windows)  67.4%  Consistent
Rolling Sharpe Std Dev               2.619  Lower = more stable

#### Walk-Forward Validation

Period         Sharpe  Total Return
In-Sample      0.522   175.77%
Out-of-Sample  1.552   180.87%
Degradation    —       20.7%

#### Exit Reason Breakdown

Exit Reason      Count
alpha_reversal   3059
max_holding      434
stop_loss        555
trailing_stop    88
end_of_backtest  3


## Backtest Results

**SPY Buy-and-Hold (full window):** 267.23%  
_Note: each ticker also shows an exposure-adjusted SPY return — SPY compounded only on days the strategy was invested. This is the fair apples-to-apples comparison._

### MSFT — AlphaCombined

**Net Return (after slippage):** 32.33%  **vs SPY (exposure-adj): -92.71%** (underperform)  
**Gross Return (pre-cost):** 45.86%  
**Total Slippage Cost:** $13,529.30  
**Trade Count:** 316  
**Win Rate:** 53.5%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-03  43.82    2016-05-05  43.92   2     665.9   +64.54    alpha_reversal
2016-05-10  44.91    2016-05-12  45.30   2     752.5   +290.49   alpha_reversal
2016-05-24  45.73    2016-05-31  46.94   4     738.7   +888.66   alpha_reversal
2016-06-03  45.91    2016-06-07  46.14   2     816.2   +186.63   alpha_reversal
2016-06-08  46.13    2016-06-13  44.40   3     883.1   -1526.60  stop_loss
2016-06-14  44.17    2016-06-23  45.97   7     801.9   +1441.67  alpha_reversal
2016-06-24  44.17    2016-06-28  43.78   2     759.6   -295.92   alpha_reversal
2016-06-29  44.80    2016-07-06  45.50   4     707.7   +494.76   alpha_reversal
2016-07-20  49.56    2016-07-21  49.42   1     704.2   -103.50   alpha_reversal
2016-08-11  51.68    2016-08-19  51.35   6     1025.6  -344.44   alpha_reversal
2016-09-06  51.39    2016-09-09  50.09   3     1175.0  -1526.19  stop_loss
2016-09-12  50.89    2016-09-13  50.38   1     1003.9  -516.24   alpha_reversal
2016-09-14  50.18    2016-09-16  51.02   2     989.3   +823.15   alpha_reversal
2016-09-21  51.52    2016-09-22  51.52   1     922.6   +1.81     alpha_reversal
2016-10-04  51.06    2016-10-06  51.45   2     1021.6  +403.05   alpha_reversal
2016-10-21  53.22    2016-10-24  54.36   1     821.1   +936.75   alpha_reversal
2016-11-08  53.94    2016-11-10  52.31   2     868.2   -1416.12  stop_loss
2016-11-11  52.65    2016-11-22  54.83   7     709.0   +1550.31  alpha_reversal
2016-11-25  54.36    2016-11-28  54.38   1     768.8   +13.41    alpha_reversal
2016-12-01  53.16    2016-12-13  56.50   8     762.8   +2546.10  alpha_reversal
2016-12-19  57.13    2016-12-20  57.00   1     775.4   -99.93    alpha_reversal
2016-12-21  57.06    2016-12-22  57.01   1     811.0   -38.98    alpha_reversal
2016-12-27  56.83    2016-12-28  56.51   1     842.5   -267.02   alpha_reversal
2016-12-29  56.49    2017-01-13  56.25   10    893.7   -210.80   max_holding
2017-01-19  55.95    2017-01-23  56.48   2     949.5   +509.11   alpha_reversal
2017-02-02  56.73    2017-02-16  58.23   10    881.6   +1327.10  max_holding
2017-02-17  58.38    2017-03-06  58.01   10    1062.5  -397.67   max_holding
2017-03-15  58.50    2017-03-16  58.34   1     1132.7  -178.71   alpha_reversal
2017-03-17  58.61    2017-03-20  58.60   1     1166.1  -5.17     alpha_reversal
2017-04-10  59.21    2017-04-24  60.95   9     1146.9  +2002.39  alpha_reversal
2017-05-09  62.38    2017-05-11  61.79   2     1081.2  -633.42   alpha_reversal
2017-05-12  61.78    2017-05-17  61.26   3     1054.5  -553.55   trailing_stop
2017-05-22  62.20    2017-05-23  62.34   1     877.2   +128.62   alpha_reversal
2017-06-01  63.70    2017-06-05  65.61   2     915.4   +1753.27  alpha_reversal
2017-06-12  63.41    2017-06-19  64.33   5     681.0   +630.63   alpha_reversal
2017-06-21  63.85    2017-06-29  62.17   6     708.6   -1190.25  trailing_stop
2017-07-05  62.77    2017-07-06  62.24   1     656.2   -344.98   alpha_reversal
2017-07-19  67.11    2017-07-21  66.98   2     719.9   -94.08    alpha_reversal
2017-07-24  66.88    2017-08-07  65.72   10    741.1   -856.83   max_holding
2017-08-08  66.14    2017-08-09  65.78   1     763.7   -272.37   alpha_reversal
2017-08-22  66.83    2017-08-23  66.36   1     757.2   -354.63   alpha_reversal
2017-08-24  66.40    2017-08-28  66.46   2     786.8   +48.32    alpha_reversal
2017-09-05  67.24    2017-09-14  68.23   7     766.8   +760.21   alpha_reversal
2017-09-15  68.80    2017-09-18  68.59   1     797.3   -164.00   alpha_reversal
2017-09-19  68.91    2017-09-25  66.86   4     814.2   -1675.87  stop_loss
2017-09-26  66.92    2017-09-29  67.98   3     772.0   +814.90   alpha_reversal
2017-10-04  68.23    2017-10-05  69.33   1     835.9   +919.40   alpha_reversal
2017-11-22  76.30    2017-11-28  77.85   3     761.0   +1177.39  alpha_reversal
2017-11-30  77.28    2017-12-01  77.28   1     742.3   +3.93     alpha_reversal
2017-12-05  74.91    2017-12-06  75.92   1     594.9   +604.75   alpha_reversal
2017-12-08  77.27    2017-12-13  78.28   3     577.0   +585.17   alpha_reversal
2017-12-19  78.80    2017-12-26  78.33   4     584.2   -276.40   alpha_reversal
2017-12-27  78.69    2017-12-28  78.62   1     661.5   -45.97    alpha_reversal
2018-01-11  80.87    2018-01-26  86.27   10    766.3   +4140.94  alpha_reversal
2018-01-30  85.14    2018-02-01  86.45   2     591.0   +773.62   alpha_reversal
2018-02-02  84.26    2018-02-05  80.71   1     502.6   -1784.90  stop_loss
2018-02-08  78.05    2018-02-12  81.75   2     331.1   +1225.37  alpha_reversal
2018-02-20  85.53    2018-02-21  84.31   1     335.1   -408.47   alpha_reversal
2018-02-22  84.61    2018-02-23  86.67   1     349.9   +721.72   alpha_reversal
2018-02-28  86.49    2018-03-02  85.74   2     370.6   -277.87   alpha_reversal
2018-03-07  86.58    2018-03-08  87.02   1     381.4   +167.32   alpha_reversal
2018-03-14  86.57    2018-03-19  85.60   3     389.4   -378.13   alpha_reversal
2018-03-22  82.82    2018-03-27  82.44   3     400.2   -151.15   alpha_reversal
2018-03-28  82.45    2018-04-02  81.57   2     296.3   -261.97   alpha_reversal
2018-04-04  85.17    2018-04-09  83.64   3     270.0   -411.11   alpha_reversal
2018-04-12  86.32    2018-04-13  85.77   1     290.7   -159.00   alpha_reversal
2018-04-16  86.86    2018-04-17  88.53   1     307.1   +511.08   alpha_reversal
2018-05-01  87.63    2018-05-04  87.69   3     300.9   +18.01    alpha_reversal
2018-05-09  89.42    2018-05-15  89.68   4     335.7   +87.55    alpha_reversal
2018-05-18  89.27    2018-05-21  90.33   1     420.7   +445.25   alpha_reversal
2018-05-23  91.40    2018-05-24  90.98   1     431.8   -179.33   alpha_reversal
2018-06-01  93.37    2018-06-04  94.09   1     453.5   +326.97   alpha_reversal
2018-06-08  94.15    2018-06-22  92.93   10    496.5   -607.38   max_holding
2018-06-25  91.15    2018-07-03  91.67   6     492.2   +255.79   alpha_reversal
2018-07-19  96.72    2018-07-23  99.92   2     506.8   +1625.31  alpha_reversal
2018-07-31  98.27    2018-08-06  100.07  4     403.5   +725.86   alpha_reversal
2018-08-13  100.25   2018-08-15  100.02  2     495.2   -111.80   alpha_reversal
2018-08-17  100.05   2018-08-29  104.07  8     492.7   +1982.89  alpha_reversal
2018-09-05  100.89   2018-09-12  103.78  5     498.6   +1441.20  alpha_reversal
2018-09-20  105.62   2018-09-24  106.53  2     474.5   +434.83   alpha_reversal
2018-10-05  104.28   2018-10-10  98.63   3     473.8   -2677.46  stop_loss
2018-10-11  98.49    2018-10-15  99.96   2     344.9   +507.59   alpha_reversal
2018-10-17  102.96   2018-10-19  100.95  2     311.7   -625.62   alpha_reversal
2018-10-25  100.72   2018-10-26  99.37   1     245.9   -330.87   alpha_reversal
2018-10-30  96.47    2018-11-07  104.01  6     221.9   +1675.19  alpha_reversal
2018-11-21  96.30    2018-11-23  96.17   1     243.1   -32.47    alpha_reversal
2018-11-26  99.44    2018-11-28  103.68  2     250.0   +1059.93  alpha_reversal
2018-12-10  100.49   2018-12-11  101.32  1     244.4   +203.52   alpha_reversal
2018-12-18  97.11    2018-12-19  96.75   1     253.3   -90.76    alpha_reversal
2018-12-20  94.81    2018-12-24  87.83   2     231.7   -1617.54  stop_loss
2018-12-28  93.76    2019-01-09  97.29   7     208.1   +733.69   alpha_reversal
2019-01-11  96.01    2019-01-18  100.50  5     251.0   +1125.88  alpha_reversal
2019-01-25  100.09   2019-01-29  96.05   2     316.4   -1280.41  stop_loss
2019-01-30  99.36    2019-01-31  97.44   1     305.6   -586.30   alpha_reversal
2019-02-01  95.99    2019-02-11  98.20   6     305.6   +674.89   alpha_reversal
2019-02-12  99.83    2019-02-15  100.97  3     353.9   +403.88   alpha_reversal
2019-02-20  100.50   2019-02-22  103.98  2     410.0   +1426.43  alpha_reversal
2019-03-06  104.82   2019-03-07  103.44  1     465.8   -642.40   alpha_reversal
2019-04-05  112.45   2019-04-08  112.38  1     485.0   -36.35    alpha_reversal
2019-04-10  112.73   2019-04-12  113.33  2     512.4   +307.17   alpha_reversal
2019-04-17  114.22   2019-04-22  115.97  2     569.6   +997.15   alpha_reversal
2019-04-25  121.14   2019-04-26  121.71  1     452.0   +258.69   alpha_reversal
2019-05-03  120.90   2019-05-07  117.62  2     430.4   -1415.30  stop_loss
2019-05-08  117.72   2019-05-13  115.58  3     394.1   -844.05   alpha_reversal
2019-05-15  118.64   2019-05-17  120.45  2     338.5   +612.59   alpha_reversal
2019-06-04  115.95   2019-06-10  124.71  4     331.3   +2903.11  alpha_reversal
2019-06-26  126.09   2019-06-28  125.99  2     354.6   -34.68    alpha_reversal
2019-07-01  127.74   2019-07-03  129.28  2     372.2   +575.52   alpha_reversal
2019-07-08  128.94   2019-07-19  128.49  9     407.4   -182.77   alpha_reversal
2019-08-01  129.98   2019-08-02  128.75  1     358.5   -437.66   alpha_reversal
2019-08-23  126.00   2019-08-28  127.92  3     283.8   +545.41   alpha_reversal
2019-08-29  130.47   2019-08-30  130.09  1     301.7   -113.36   alpha_reversal
2019-09-04  130.00   2019-09-18  130.71  10    315.2   +223.71   alpha_reversal
2019-09-24  129.77   2019-09-27  129.97  3     338.3   +67.85    alpha_reversal
2019-10-08  128.15   2019-10-09  130.45  1     336.4   +772.70   alpha_reversal
2019-10-15  133.72   2019-10-18  129.66  3     369.7   -1500.59  stop_loss
2019-10-21  130.76   2019-10-24  132.05  3     364.8   +472.09   alpha_reversal
2019-11-06  136.08   2019-11-11  137.87  3     393.4   +707.49   alpha_reversal
2019-12-02  141.74   2019-12-16  147.26  10    476.4   +2629.92  alpha_reversal
2019-12-18  146.31   2019-12-20  149.04  2     526.2   +1437.53  alpha_reversal
2019-12-26  150.39   2019-12-27  150.51  1     552.1   +68.61    alpha_reversal
2020-01-28  156.82   2020-01-30  163.60  2     377.8   +2559.37  alpha_reversal
2020-02-06  174.04   2020-02-07  174.12  1     273.0   +19.72    alpha_reversal
2020-02-12  175.07   2020-02-13  173.94  1     241.0   -270.34   alpha_reversal
2020-02-24  162.41   2020-02-27  150.18  3     191.5   -2342.46  stop_loss
2020-02-28  153.97   2020-03-02  164.05  1     150.4   +1516.02  alpha_reversal
2020-03-05  158.02   2020-03-06  153.40  1     136.3   -629.54   alpha_reversal
2020-03-11  146.01   2020-03-12  132.03  1     117.5   -1642.23  stop_loss
2020-03-26  148.36   2020-04-03  146.05  6     83.5    -193.11   alpha_reversal
2020-04-06  157.07   2020-04-13  157.14  4     92.1    +6.52     alpha_reversal
2020-04-16  168.26   2020-04-17  169.57  1     113.5   +148.97   alpha_reversal
2020-04-21  159.49   2020-04-22  164.75  1     119.7   +628.78   alpha_reversal
2020-05-19  174.52   2020-05-26  172.87  4     163.1   -269.14   alpha_reversal
2020-05-27  173.27   2020-06-10  187.41  10    178.2   +2519.15  alpha_reversal
2020-06-12  178.92   2020-06-15  179.89  1     173.2   +166.92   alpha_reversal
2020-06-29  189.12   2020-06-30  193.76  1     176.4   +818.27   alpha_reversal
2020-07-13  197.34   2020-07-23  192.83  8     170.7   -769.77   alpha_reversal
2020-07-29  194.47   2020-08-03  206.16  3     171.8   +2008.38  alpha_reversal
2020-08-05  202.94   2020-08-07  202.30  2     161.2   -103.29   alpha_reversal
2020-08-11  193.83   2020-08-18  201.35  5     163.0   +1226.69  alpha_reversal
2020-09-04  204.68   2020-09-08  193.41  1     145.3   -1636.68  stop_loss
2020-09-09  201.85   2020-09-10  196.00  1     127.3   -744.65   alpha_reversal
2020-09-15  199.45   2020-09-16  195.70  1     128.4   -482.61   alpha_reversal
2020-09-17  193.85   2020-09-22  197.96  3     131.7   +541.32   alpha_reversal
2020-09-30  200.94   2020-10-02  196.78  2     142.0   -589.39   alpha_reversal
2020-10-05  200.98   2020-10-13  212.69  6     147.7   +1730.06  alpha_reversal
2020-10-16  209.85   2020-10-22  205.09  4     166.5   -792.76   alpha_reversal
2020-11-03  197.21   2020-11-05  213.10  2     160.0   +2542.41  alpha_reversal
2020-11-10  201.58   2020-11-24  204.64  10    138.1   +421.74   max_holding
2020-11-25  204.85   2020-11-27  205.95  1     178.5   +195.75   alpha_reversal
2020-12-01  207.09   2020-12-14  204.96  9     190.0   -404.75   alpha_reversal
2020-12-16  210.03   2020-12-17  209.96  1     218.2   -16.58    alpha_reversal
2020-12-21  213.20   2020-12-22  214.28  1     214.7   +231.57   alpha_reversal
2020-12-24  213.36   2020-12-28  215.26  1     227.1   +431.74   alpha_reversal
2020-12-30  212.33   2021-01-06  203.10  4     231.8   -2140.42  stop_loss
2021-01-07  209.09   2021-01-21  215.27  9     203.1   +1255.81  alpha_reversal
2021-02-03  232.75   2021-02-04  231.57  1     160.8   -189.78   alpha_reversal
2021-02-05  231.99   2021-02-08  232.01  1     172.4   +4.57     alpha_reversal
2021-02-11  234.18   2021-02-12  234.43  1     188.0   +45.94    alpha_reversal
2021-02-19  231.34   2021-02-22  224.91  1     209.3   -1345.16  stop_loss
2021-02-23  223.95   2021-03-01  227.24  4     189.6   +625.02   alpha_reversal
2021-03-04  217.67   2021-03-08  218.09  2     169.9   +70.58    alpha_reversal
2021-03-09  224.44   2021-03-10  222.91  1     159.9   -244.37   alpha_reversal
2021-03-15  225.43   2021-03-23  227.86  6     165.9   +403.43   alpha_reversal
2021-03-30  222.59   2021-04-01  232.43  2     177.0   +1743.36  alpha_reversal
2021-04-07  239.91   2021-04-08  242.89  1     175.9   +523.02   alpha_reversal
2021-04-15  249.13   2021-04-19  248.15  2     197.5   -193.09   alpha_reversal
2021-04-20  247.94   2021-04-26  250.85  4     207.9   +604.52   alpha_reversal
2021-05-03  241.80   2021-05-12  229.22  7     203.3   -2556.49  stop_loss
2021-05-13  233.32   2021-05-26  241.76  9     180.2   +1520.81  alpha_reversal
2021-05-27  239.90   2021-06-01  237.82  2     210.4   -436.69   alpha_reversal
2021-06-02  237.97   2021-06-10  247.28  6     218.1   +2031.74  alpha_reversal
2021-06-16  247.67   2021-06-18  249.39  2     233.5   +402.37   alpha_reversal
2021-06-21  252.72   2021-06-22  255.23  1     223.3   +561.76   alpha_reversal
2021-06-28  258.58   2021-06-29  260.90  1     246.3   +570.93   alpha_reversal
2021-07-12  266.85   2021-07-14  271.58  2     248.6   +1173.99  alpha_reversal
2021-08-02  274.07   2021-08-04  275.42  2     233.8   +315.79   alpha_reversal
2021-08-05  278.59   2021-08-06  278.26  1     244.1   -82.08    alpha_reversal
2021-08-09  277.45   2021-08-12  278.59  3     253.3   +290.11   alpha_reversal
2021-08-30  292.69   2021-09-09  286.29  7     220.9   -1413.24  stop_loss
2021-09-20  283.73   2021-09-28  273.07  6     199.3   -2125.68  trailing_stop
2021-09-29  273.80   2021-10-07  283.98  6     184.7   +1879.41  alpha_reversal
2021-10-13  285.67   2021-10-19  296.87  4     178.4   +1997.12  alpha_reversal
2021-11-16  327.32   2021-11-17  327.22  1     191.5   -20.14    alpha_reversal
2021-12-01  318.81   2021-12-13  327.49  8     150.5   +1305.62  alpha_reversal
2021-12-15  323.23   2021-12-16  313.50  1     124.9   -1215.35  alpha_reversal
2021-12-17  312.75   2021-12-27  330.43  5     117.7   +2080.84  alpha_reversal
2021-12-29  330.28   2021-12-30  327.41  1     135.0   -387.01   alpha_reversal
2021-12-31  324.84   2022-01-05  305.27  3     143.6   -2809.40  stop_loss
2022-01-06  303.16   2022-01-11  303.92  3     127.6   +96.77    alpha_reversal
2022-01-12  307.40   2022-01-13  294.10  1     125.4   -1668.73  stop_loss
2022-01-14  299.61   2022-01-18  292.03  1     114.9   -871.73   alpha_reversal
2022-01-19  292.97   2022-01-20  291.01  1     110.8   -217.45   alpha_reversal
2022-02-02  302.76   2022-02-11  284.68  7     95.6    -1728.31  stop_loss
2022-02-14  284.93   2022-02-23  270.99  6     99.9    -1393.02  trailing_stop
2022-03-04  280.54   2022-03-08  266.72  2     99.1    -1369.76  stop_loss
2022-03-09  279.23   2022-03-23  289.57  10    90.2    +933.30   max_holding
2022-03-24  294.33   2022-04-01  299.18  6     103.5   +501.83   alpha_reversal
2022-04-06  289.87   2022-04-11  275.82  3     114.5   -1609.92  stop_loss
2022-04-12  272.99   2022-04-19  275.85  4     109.8   +314.10   alpha_reversal
2022-05-05  268.44   2022-05-11  251.92  4     89.5    -1478.11  stop_loss
2022-05-12  247.14   2022-05-26  257.70  10    87.6    +924.41   alpha_reversal
2022-06-07  264.36   2022-06-08  262.07  1     103.1   -236.04   alpha_reversal
2022-06-09  256.88   2022-06-13  234.78  2     107.7   -2379.37  stop_loss
2022-06-14  237.18   2022-06-17  240.01  3     103.5   +292.40   alpha_reversal
2022-07-11  256.60   2022-07-12  245.84  1     123.1   -1324.32  stop_loss
2022-07-13  245.17   2022-07-27  260.45  10    116.1   +1774.76  max_holding
2022-08-08  271.94   2022-08-10  280.24  2     121.1   +1004.64  alpha_reversal
2022-08-11  278.44   2022-08-16  283.68  3     126.8   +663.90   alpha_reversal
2022-08-17  283.21   2022-08-22  269.75  3     142.4   -1917.38  stop_loss
2022-08-24  268.12   2022-08-29  257.59  3     154.7   -1628.12  stop_loss
2022-08-31  254.19   2022-09-13  244.73  8     145.0   -1371.91  trailing_stop
2022-09-14  245.20   2022-09-20  235.47  4     132.0   -1284.79  stop_loss
2022-10-03  234.04   2022-10-05  242.02  2     127.3   +1016.44  alpha_reversal
2022-10-07  227.72   2022-10-13  227.49  4     119.2   -27.14    alpha_reversal
2022-10-31  225.67   2022-11-03  208.08  3     103.3   -1816.75  stop_loss
2022-11-08  222.50   2022-11-10  235.98  2     104.0   +1402.41  alpha_reversal
2022-11-18  235.17   2022-11-22  238.64  2     112.3   +390.17   alpha_reversal
2022-11-28  235.69   2022-12-01  248.05  3     130.4   +1610.98  alpha_reversal
2022-12-08  241.19   2022-12-13  250.22  3     128.9   +1164.27  alpha_reversal
2022-12-22  232.21   2023-01-05  216.52  8     122.6   -1924.31  stop_loss
2023-01-06  219.29   2023-01-19  225.89  8     120.7   +796.33   alpha_reversal
2023-02-09  257.01   2023-02-14  265.08  3     111.9   +902.82   alpha_reversal
2023-02-16  256.21   2023-02-24  243.33  5     117.2   -1510.15  stop_loss
2023-02-27  244.49   2023-02-28  243.53  1     127.5   -123.25   alpha_reversal
2023-03-01  240.69   2023-03-07  248.15  4     134.8   +1004.76  alpha_reversal
2023-03-09  246.61   2023-03-10  242.72  1     144.6   -562.35   alpha_reversal
2023-03-21  267.58   2023-03-22  265.86  1     121.6   -209.39   alpha_reversal
2023-03-28  269.00   2023-04-12  276.79  10    124.4   +969.53   max_holding
2023-04-17  282.26   2023-04-20  279.35  3     134.8   -392.14   alpha_reversal
2023-05-08  301.66   2023-05-16  304.38  6     132.1   +358.80   alpha_reversal
2023-06-01  325.76   2023-06-06  326.51  3     133.3   +100.11   alpha_reversal
2023-06-07  316.75   2023-06-15  340.62  6     133.8   +3194.55  alpha_reversal
2023-06-20  331.12   2023-06-23  327.82  3     128.2   -422.48   alpha_reversal
2023-06-27  327.71   2023-06-29  327.85  2     127.5   +18.13    alpha_reversal
2023-06-30  333.56   2023-07-06  333.94  3     134.0   +51.05    alpha_reversal
2023-07-10  325.02   2023-07-18  351.77  6     132.7   +3548.29  alpha_reversal
2023-07-31  329.03   2023-08-07  323.02  5     101.7   -611.40   alpha_reversal
2023-08-09  315.62   2023-08-11  314.11  2     114.8   -173.32   alpha_reversal
2023-08-14  317.39   2023-08-15  314.94  1     124.0   -303.97   alpha_reversal
2023-09-01  322.60   2023-09-05  327.07  1     140.9   +630.13   alpha_reversal
2023-09-13  329.86   2023-09-14  332.12  1     150.6   +340.10   alpha_reversal
2023-09-18  322.99   2023-09-21  313.33  3     151.0   -1459.64  stop_loss
2023-09-22  311.17   2023-10-02  315.55  6     140.4   +615.76   alpha_reversal
2023-10-18  324.02   2023-10-20  320.33  2     135.9   -502.51   alpha_reversal
2023-11-14  363.44   2023-11-21  366.57  5     134.6   +420.70   alpha_reversal
2023-11-27  372.38   2023-11-28  376.03  1     138.7   +505.82   alpha_reversal
2023-11-30  372.68   2023-12-04  362.71  2     143.6   -1432.28  stop_loss
2023-12-05  366.39   2023-12-18  366.16  9     130.9   -31.21    alpha_reversal
2023-12-28  369.11   2023-12-29  369.49  1     168.2   +63.55    alpha_reversal
2024-01-08  368.53   2024-01-10  376.10  2     158.4   +1199.43  alpha_reversal
2024-01-23  392.34   2024-01-25  397.81  2     153.0   +837.77   alpha_reversal
2024-02-07  407.24   2024-02-08  406.89  1     127.2   -44.27    alpha_reversal
2024-02-13  399.64   2024-02-28  401.36  10    126.8   +217.67   max_holding
2024-03-06  396.21   2024-03-15  409.92  7     130.6   +1790.41  alpha_reversal
2024-03-18  411.22   2024-04-02  414.86  10    117.7   +429.05   alpha_reversal
2024-04-04  411.77   2024-04-10  416.65  4     135.6   +662.34   alpha_reversal
2024-04-18  398.36   2024-04-29  395.97  7     127.1   -303.27   alpha_reversal
2024-05-01  389.16   2024-05-09  405.88  6     101.3   +1693.80  alpha_reversal
2024-05-14  410.47   2024-05-15  417.23  1     123.4   +834.13   alpha_reversal
2024-05-20  419.88   2024-05-21  423.10  1     130.1   +420.18   alpha_reversal
2024-05-31  409.80   2024-06-03  407.80  1     120.8   -241.23   alpha_reversal
2024-06-06  419.07   2024-06-11  426.69  3     124.9   +952.57   alpha_reversal
2024-06-25  445.16   2024-06-26  445.91  1     141.7   +106.05   alpha_reversal
2024-07-01  450.86   2024-07-03  454.40  2     135.4   +478.39   alpha_reversal
2024-07-12  447.72   2024-07-18  434.28  4     129.5   -1741.49  stop_loss
2024-07-19  431.49   2024-07-24  422.97  3     120.2   -1025.13  alpha_reversal
2024-08-06  394.48   2024-08-13  408.28  5     87.2    +1204.33  alpha_reversal
2024-08-19  416.86   2024-08-21  419.03  2     110.6   +239.00   alpha_reversal
2024-08-22  410.95   2024-09-04  403.97  8     113.8   -794.69   alpha_reversal
2024-09-05  403.87   2024-09-11  417.94  4     116.3   +1636.06  alpha_reversal
2024-09-19  433.83   2024-09-20  430.02  1     112.3   -428.06   alpha_reversal
2024-09-23  428.71   2024-10-01  415.62  6     117.3   -1536.15  stop_loss
2024-10-03  411.93   2024-10-17  411.70  10    127.6   -29.84    max_holding
2024-11-06  415.53   2024-11-15  410.00  7     114.1   -631.44   alpha_reversal
2024-11-21  409.12   2024-12-02  426.64  6     112.4   +1969.68  alpha_reversal
2024-12-23  431.29   2024-12-31  417.25  5     113.0   -1586.84  trailing_stop
2025-01-02  414.78   2025-01-10  414.73  5     112.4   -5.43     alpha_reversal
2025-01-15  422.44   2025-01-23  442.21  5     108.4   +2143.07  alpha_reversal
2025-02-04  408.62   2025-02-07  405.62  3     94.2    -282.85   alpha_reversal
2025-02-12  405.32   2025-02-25  394.68  8     110.2   -1173.16  trailing_stop
2025-03-11  377.75   2025-03-12  380.17  1     93.1    +225.16   alpha_reversal
2025-03-14  385.80   2025-03-17  385.55  1     95.6    -23.59    alpha_reversal
2025-03-19  385.07   2025-03-20  383.71  1     101.3   -137.43   alpha_reversal
2025-03-21  388.48   2025-03-24  389.90  1     101.8   +144.21   alpha_reversal
2025-03-27  387.81   2025-03-28  375.73  1     114.3   -1379.76  alpha_reversal
2025-04-01  379.48   2025-04-04  356.93  3     104.3   -2352.37  stop_loss
2025-04-08  352.04   2025-04-10  378.26  2     77.7    +2036.25  alpha_reversal
2025-04-15  382.99   2025-04-21  356.21  3     68.3    -1830.04  stop_loss
2025-04-22  364.22   2025-04-28  387.99  4     70.2    +1669.78  alpha_reversal
2025-05-13  445.95   2025-05-23  447.36  8     81.2    +114.16   alpha_reversal
2025-06-11  470.13   2025-06-12  475.87  1     130.9   +751.54   alpha_reversal
2025-06-23  483.43   2025-06-24  487.04  1     124.8   +449.39   alpha_reversal
2025-06-30  494.78   2025-07-01  488.96  1     134.5   -782.80   alpha_reversal
2025-07-03  496.21   2025-07-14  499.86  6     133.0   +486.39   alpha_reversal
2025-07-15  503.15   2025-07-16  502.45  1     143.0   -100.30   alpha_reversal
2025-07-17  509.00   2025-07-22  502.10  3     143.1   -986.84   alpha_reversal
2025-07-23  503.20   2025-07-31  530.15  6     142.0   +3827.23  alpha_reversal
2025-08-13  517.83   2025-08-20  502.55  5     97.8    -1494.32  stop_loss
2025-08-21  502.40   2025-09-04  505.61  9     107.5   +345.05   alpha_reversal
2025-09-08  496.38   2025-09-10  498.05  2     112.8   +187.75   alpha_reversal
2025-09-23  507.37   2025-09-25  504.68  2     120.7   -325.64   alpha_reversal
2025-10-15  511.56   2025-10-23  518.15  6     115.0   +757.63   alpha_reversal
2025-11-06  495.29   2025-11-17  505.14  7     94.7    +932.51   alpha_reversal
2025-11-19  485.35   2025-12-03  476.40  9     86.5    -773.71   alpha_reversal
2025-12-04  479.98   2025-12-10  477.23  4     89.5    -246.31   alpha_reversal
2025-12-11  482.61   2025-12-17  474.80  4     92.7    -724.10   alpha_reversal
2025-12-23  485.98   2025-12-24  486.67  1     108.0   +73.54    alpha_reversal
2025-12-29  486.23   2025-12-31  482.28  2     122.3   -483.87   alpha_reversal
2026-01-05  472.01   2026-01-14  458.10  7     121.4   -1687.44  trailing_stop
2026-01-15  455.85   2026-01-21  442.88  3     108.5   -1406.85  stop_loss
2026-01-22  450.34   2026-01-26  468.97  2     100.6   +1875.01  alpha_reversal
2026-02-03  410.48   2026-02-18  398.49  10    69.1    -828.10   max_holding
2026-02-20  397.43   2026-03-03  403.73  7     82.4    +518.93   alpha_reversal
2026-03-12  402.06   2026-03-20  381.68  6     96.9    -1975.36  stop_loss
2026-03-23  383.19   2026-03-26  365.79  3     105.2   -1831.56  stop_loss

**Best 3 trades:**

- 2018-01-26: P&L = **+4140.94** (alpha_reversal)
- 2025-07-31: P&L = **+3827.23** (alpha_reversal)
- 2023-07-18: P&L = **+3548.29** (alpha_reversal)

**Worst 3 trades:**

- 2022-01-05: P&L = **-2809.40** (stop_loss)
- 2018-10-10: P&L = **-2677.46** (stop_loss)
- 2021-05-12: P&L = **-2556.49** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  99,902.61
2017-03-23  104,550.27
2017-09-20  105,009.86
2018-03-21  111,696.48
2018-09-18  117,234.65
2019-03-20  117,596.64
2019-09-17  120,640.22
2020-03-17  124,821.94
2020-09-14  128,628.99
2021-03-15  130,501.01
2021-09-10  135,979.97
2022-03-10  129,266.63
2022-09-08  124,935.16
2023-03-09  125,581.28
2023-09-07  131,794.65
2024-03-07  133,406.41
2024-09-05  137,206.49
2025-03-07  137,304.98
2025-09-05  138,266.67
2026-03-06  136,136.02

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -1.63%
2017-03-23  -0.81%
2017-09-20  -3.18%
2018-03-21  -1.03%
2018-09-18  -0.02%
2019-03-20  -0.56%
2019-09-17  -0.82%
2020-03-17  -2.65%
2020-09-14  -1.83%
2021-03-15  -1.57%
2021-09-10  -1.05%
2022-03-10  -7.60%
2022-09-08  -10.70%
2023-03-09  -10.24%
2023-09-07  -5.79%
2024-03-07  -4.64%
2024-09-05  -1.93%
2025-03-07  -1.86%
2025-09-05  -1.17%
2026-03-06  -2.97%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  47.86%
Out-of-Sample (30%)  2023-03-24  2026-03-27  13.42%

#### Return Distribution

Return Bin          Count
-2.283% to -1.834%  1
-1.834% to -1.386%  6
-1.386% to -0.938%  23
-0.938% to -0.490%  101
-0.490% to -0.042%  333
-0.042% to 0.407%   1838
0.407% to 0.855%    175
0.855% to 1.303%    28
1.303% to 1.751%    8
1.751% to 2.199%    3

### AAPL — AlphaCombined

**Net Return (after slippage):** 36.94%  **vs SPY (exposure-adj): -105.17%** (underperform)  
**Gross Return (pre-cost):** 49.91%  
**Total Slippage Cost:** $12,964.77  
**Trade Count:** 315  
**Win Rate:** 55.9%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-02  21.22    2016-05-10  21.28   6     1224.1  +70.68    alpha_reversal
2016-05-11  21.09    2016-05-18  21.54   5     1439.1  +641.57   alpha_reversal
2016-05-31  22.77    2016-06-06  22.46   4     1548.6  -469.05   alpha_reversal
2016-06-09  22.72    2016-06-10  22.51   1     1603.8  -335.93   alpha_reversal
2016-06-20  21.68    2016-06-27  20.96   5     1646.6  -1183.18  stop_loss
2016-06-30  21.79    2016-07-05  21.63   2     1665.4  -267.64   alpha_reversal
2016-07-06  21.78    2016-07-08  22.02   2     1750.3  +420.31   alpha_reversal
2016-07-14  22.52    2016-07-20  22.77   4     1981.9  +483.49   alpha_reversal
2016-07-25  22.19    2016-07-28  23.76   3     2100.8  +3302.58  alpha_reversal
2016-08-12  24.80    2016-08-15  25.07   1     2026.7  +553.06   alpha_reversal
2016-08-18  25.00    2016-08-22  24.85   2     2220.5  -345.31   alpha_reversal
2016-08-23  24.95    2016-08-29  24.46   4     2262.8  -1108.25  stop_loss
2016-08-30  24.30    2016-09-08  24.16   6     2236.3  -300.08   trailing_stop
2016-09-09  23.64    2016-09-14  25.59   3     1971.4  +3853.68  alpha_reversal
2016-09-26  25.87    2016-09-27  25.90   1     1452.9  +32.30    alpha_reversal
2016-09-30  25.91    2016-10-05  25.89   3     1558.9  -40.37    alpha_reversal
2016-10-19  26.84    2016-10-26  26.47   5     1676.9  -632.50   alpha_reversal
2016-10-27  26.24    2016-11-01  25.53   3     1648.4  -1171.81  stop_loss
2016-11-02  25.58    2016-11-10  24.81   6     1643.3  -1263.65  stop_loss
2016-11-11  24.98    2016-11-17  25.31   4     1346.3  +437.37   alpha_reversal
2016-11-18  25.36    2016-11-21  25.72   1     1321.9  +474.59   alpha_reversal
2016-11-25  25.75    2016-11-28  25.68   1     1484.9  -113.41   alpha_reversal
2016-11-30  25.46    2016-12-05  25.11   3     1534.0  -536.85   alpha_reversal
2016-12-06  25.33    2016-12-14  26.51   6     1614.0  +1905.58  alpha_reversal
2016-12-30  26.68    2017-01-10  27.41   6     1948.9  +1423.77  alpha_reversal
2017-01-25  28.08    2017-01-26  28.07   1     2349.9  -33.49    alpha_reversal
2017-02-01  29.66    2017-02-02  29.58   1     1714.1  -137.60   alpha_reversal
2017-03-09  32.09    2017-03-14  32.13   3     2118.2  +83.86    alpha_reversal
2017-03-15  32.50    2017-03-17  32.36   2     2296.7  -324.11   alpha_reversal
2017-03-20  32.73    2017-03-21  32.32   1     2354.0  -958.52   alpha_reversal
2017-03-22  32.72    2017-03-23  32.57   1     2079.3  -308.33   alpha_reversal
2017-03-31  33.24    2017-04-04  33.46   2     1943.2  +434.02   alpha_reversal
2017-04-07  33.17    2017-04-13  32.60   4     2065.8  -1162.01  stop_loss
2017-04-17  32.82    2017-04-28  33.21   9     2016.8  +782.32   alpha_reversal
2017-05-04  33.90    2017-05-08  35.37   2     1848.1  +2705.64  alpha_reversal
2017-05-18  35.44    2017-05-31  35.46   8     1390.7  +21.74    alpha_reversal
2017-06-01  35.59    2017-06-02  36.08   1     1708.9  +839.60   alpha_reversal
2017-06-06  35.88    2017-06-09  34.58   3     1651.9  -2156.54  stop_loss
2017-06-12  33.79    2017-06-26  33.85   10    1157.0  +68.36    alpha_reversal
2017-06-27  33.39    2017-07-03  33.31   4     1187.8  -103.05   alpha_reversal
2017-07-05  33.48    2017-07-06  33.13   1     1214.5  -424.04   alpha_reversal
2017-07-21  34.91    2017-07-26  35.62   3     1432.0  +1010.32  alpha_reversal
2017-07-28  34.73    2017-08-02  36.47   3     1312.7  +2282.27  alpha_reversal
2017-09-07  37.61    2017-09-20  36.37   9     1289.1  -1607.48  stop_loss
2017-09-21  35.78    2017-10-02  35.84   7     1107.4  +68.78    alpha_reversal
2017-10-03  36.03    2017-10-10  36.33   5     1248.8  +368.23   alpha_reversal
2017-10-13  36.62    2017-10-16  37.25   1     1466.1  +933.66   alpha_reversal
2017-10-20  36.45    2017-10-30  38.85   6     1401.0  +3366.94  alpha_reversal
2017-11-14  40.11    2017-11-24  40.92   7     1203.8  +973.59   alpha_reversal
2017-11-29  39.67    2017-12-12  40.15   9     1204.7  +577.66   alpha_reversal
2017-12-13  40.33    2017-12-18  41.26   3     1211.3  +1126.72  alpha_reversal
2017-12-21  40.97    2017-12-22  40.93   1     1306.5  -53.50    alpha_reversal
2017-12-27  39.94    2017-12-29  39.58   2     1311.2  -472.44   alpha_reversal
2018-01-03  40.32    2018-01-18  41.92   10    1335.5  +2141.71  alpha_reversal
2018-01-22  41.43    2018-01-24  40.74   2     1458.4  -1008.50  alpha_reversal
2018-01-25  40.06    2018-01-30  39.05   3     1277.7  -1288.14  stop_loss
2018-01-31  39.19    2018-02-02  37.53   2     1224.1  -2031.79  stop_loss
2018-02-05  36.63    2018-02-12  38.21   5     962.1   +1513.51  alpha_reversal
2018-03-07  41.14    2018-03-09  42.26   2     861.2   +965.60   alpha_reversal
2018-03-19  41.20    2018-03-22  39.65   3     970.9   -1510.45  stop_loss
2018-03-23  38.77    2018-04-06  39.54   9     924.0   +710.52   alpha_reversal
2018-04-12  40.93    2018-04-13  41.03   1     790.5   +77.18    alpha_reversal
2018-04-19  40.62    2018-04-20  38.91   1     872.3   -1485.51  stop_loss
2018-04-23  38.84    2018-05-01  39.71   6     820.1   +711.46   alpha_reversal
2018-05-14  44.39    2018-05-17  44.08   3     854.6   -271.59   alpha_reversal
2018-05-18  43.96    2018-05-29  44.29   6     938.2   +310.39   alpha_reversal
2018-05-31  44.09    2018-06-13  44.95   9     1125.7  +966.66   alpha_reversal
2018-06-14  45.02    2018-06-19  43.77   3     1272.5  -1589.97  stop_loss
2018-06-20  44.00    2018-06-25  42.94   3     1215.5  -1294.00  stop_loss
2018-07-11  44.33    2018-07-16  45.00   3     1085.0  +726.87   alpha_reversal
2018-07-17  45.17    2018-07-18  44.88   1     1148.4  -336.06   alpha_reversal
2018-07-19  45.27    2018-08-01  47.50   9     1165.9  +2591.00  alpha_reversal
2018-09-10  51.70    2018-09-14  52.95   4     842.7   +1054.81  alpha_reversal
2018-09-19  51.71    2018-09-25  52.56   4     713.1   +607.50   alpha_reversal
2018-09-27  53.26    2018-09-28  53.40   1     727.4   +97.20    alpha_reversal
2018-10-18  51.15    2018-10-24  50.88   4     593.0   -160.76   alpha_reversal
2018-10-29  50.25    2018-11-02  49.08   4     492.8   -579.62   trailing_stop
2018-11-05  47.73    2018-11-12  46.09   5     437.0   -717.83   trailing_stop
2018-11-14  44.38    2018-11-20  42.01   4     455.6   -1082.16  trailing_stop
2018-11-21  42.00    2018-12-06  41.47   9     458.2   -243.27   alpha_reversal
2018-12-12  40.18    2018-12-14  39.28   2     523.5   -470.84   alpha_reversal
2018-12-21  35.81    2018-12-31  37.44   5     515.0   +838.46   alpha_reversal
2019-01-04  35.23    2019-01-11  36.15   5     480.1   +442.38   alpha_reversal
2019-01-14  35.64    2019-01-16  36.78   2     586.5   +666.84   alpha_reversal
2019-01-23  36.57    2019-01-24  36.25   1     681.3   -222.21   alpha_reversal
2019-01-25  37.48    2019-01-31  39.51   4     691.5   +1398.79  alpha_reversal
2019-02-12  40.78    2019-02-13  40.57   1     796.0   -167.16   alpha_reversal
2019-02-14  40.76    2019-02-15  40.63   1     847.9   -111.35   alpha_reversal
2019-02-19  40.79    2019-02-20  41.01   1     907.9   +201.06   alpha_reversal
2019-02-22  41.27    2019-02-25  41.53   1     988.4   +256.09   alpha_reversal
2019-03-20  44.90    2019-03-21  46.51   1     1020.0  +1639.21  alpha_reversal
2019-03-25  45.04    2019-03-28  44.99   3     843.7   -42.00    alpha_reversal
2019-04-12  47.46    2019-04-16  47.50   2     948.6   +40.94    alpha_reversal
2019-04-17  48.47    2019-04-23  49.46   3     962.1   +951.03   alpha_reversal
2019-04-26  48.75    2019-05-02  49.86   4     1061.5  +1175.50  alpha_reversal
2019-05-10  47.23    2019-05-13  44.44   1     699.5   -1951.38  stop_loss
2019-05-14  45.19    2019-05-23  42.99   7     610.5   -1342.32  stop_loss
2019-05-28  42.69    2019-05-31  41.89   3     649.6   -518.94   alpha_reversal
2019-06-13  46.51    2019-06-18  47.49   3     697.2   +685.01   alpha_reversal
2019-06-20  47.78    2019-07-03  48.92   9     750.6   +853.30   alpha_reversal
2019-07-08  47.91    2019-07-16  48.94   6     856.8   +877.52   alpha_reversal
2019-07-17  48.71    2019-07-19  48.48   2     1012.2  -233.38   alpha_reversal
2019-07-25  49.59    2019-07-30  49.96   3     1055.3  +392.15   alpha_reversal
2019-07-31  51.03    2019-08-01  49.88   1     881.0   -1016.89  alpha_reversal
2019-08-02  48.87    2019-08-05  46.27   1     731.9   -1906.34  stop_loss
2019-08-09  48.33    2019-08-13  50.20   2     660.5   +1234.24  alpha_reversal
2019-08-28  49.42    2019-09-11  53.71   9     628.6   +2696.13  alpha_reversal
2019-09-16  52.88    2019-09-17  53.01   1     704.4   +98.14    alpha_reversal
2019-09-20  52.35    2019-09-24  52.29   2     754.3   -48.55    alpha_reversal
2019-10-16  56.35    2019-10-18  56.79   2     776.4   +336.75   alpha_reversal
2019-10-30  58.49    2019-11-01  61.45   2     819.7   +2425.13  alpha_reversal
2019-11-20  63.47    2019-11-29  64.39   6     869.8   +795.63   alpha_reversal
2019-12-02  63.71    2019-12-03  62.51   1     898.4   -1076.73  alpha_reversal
2019-12-04  63.12    2019-12-05  63.99   1     829.4   +715.05   alpha_reversal
2019-12-10  64.75    2019-12-11  65.24   1     790.8   +385.12   alpha_reversal
2019-12-23  68.49    2019-12-26  69.85   2     833.7   +1129.99  alpha_reversal
2020-01-22  76.62    2020-01-23  76.91   1     718.1   +209.69   alpha_reversal
2020-01-28  76.62    2020-01-29  78.14   1     602.4   +919.05   alpha_reversal
2020-02-03  74.44    2020-02-11  77.19   6     493.7   +1356.06  alpha_reversal
2020-02-12  79.10    2020-02-18  77.04   3     501.9   -1033.57  alpha_reversal
2020-02-19  78.23    2020-02-20  77.35   1     519.0   -456.68   alpha_reversal
2020-02-21  75.68    2020-02-24  72.01   1     500.5   -1835.07  stop_loss
2020-02-25  69.64    2020-02-27  66.06   2     390.6   -1400.71  stop_loss
2020-02-28  66.08    2020-03-03  69.87   2     318.3   +1205.84  alpha_reversal
2020-03-23  54.24    2020-03-31  61.41   6     194.6   +1395.81  alpha_reversal
2020-04-01  58.24    2020-04-03  58.30   2     217.8   +13.62    alpha_reversal
2020-04-06  63.45    2020-04-07  62.65   1     226.9   -181.01   alpha_reversal
2020-04-08  64.32    2020-04-13  65.99   2     237.1   +395.93   alpha_reversal
2020-04-24  68.41    2020-04-28  67.28   2     309.3   -349.12   alpha_reversal
2020-05-19  75.91    2020-05-26  76.70   4     377.0   +299.13   alpha_reversal
2020-05-27  77.11    2020-06-02  78.30   4     426.0   +506.72   alpha_reversal
2020-06-05  80.36    2020-06-10  85.44   3     481.4   +2449.14  alpha_reversal
2020-06-22  86.99    2020-06-24  87.19   2     397.7   +80.04    alpha_reversal
2020-06-25  88.44    2020-07-09  92.75   9     385.3   +1661.37  alpha_reversal
2020-07-17  93.40    2020-07-21  93.96   2     393.0   +219.32   alpha_reversal
2020-07-22  94.32    2020-07-23  89.93   1     400.0   -1752.98  stop_loss
2020-07-24  89.80    2020-07-31  102.93  5     345.4   +4534.02  alpha_reversal
2020-08-31  125.34   2020-09-01  130.21  1     273.1   +1327.99  alpha_reversal
2020-09-03  117.42   2020-09-08  109.48  2     202.6   -1608.25  stop_loss
2020-09-09  113.96   2020-09-18  103.68  7     167.2   -1719.81  stop_loss
2020-09-21  106.93   2020-09-25  108.95  4     168.2   +341.17   alpha_reversal
2020-09-30  112.49   2020-10-01  113.33  1     190.2   +159.49   alpha_reversal
2020-10-05  113.16   2020-10-13  117.51  6     201.5   +876.51   alpha_reversal
2020-10-14  117.72   2020-10-15  117.13  1     209.5   -122.25   alpha_reversal
2020-10-22  112.43   2020-10-27  113.15  3     229.3   +163.35   alpha_reversal
2020-11-04  111.66   2020-11-16  116.94  8     217.1   +1146.41  alpha_reversal
2020-11-18  114.85   2020-11-20  114.06  2     260.6   -204.69   alpha_reversal
2020-11-23  110.78   2020-12-01  119.29  5     274.5   +2336.38  alpha_reversal
2020-12-04  118.95   2020-12-08  120.90  2     312.6   +610.04   alpha_reversal
2020-12-10  119.92   2020-12-16  124.24  4     305.2   +1319.05  alpha_reversal
2020-12-21  124.77   2020-12-22  128.19  1     301.7   +1032.81  alpha_reversal
2020-12-31  129.11   2021-01-06  123.06  3     284.7   -1722.40  stop_loss
2021-01-07  127.39   2021-01-14  125.31  5     249.9   -520.09   alpha_reversal
2021-01-15  123.71   2021-01-22  135.18  4     271.4   +3113.92  alpha_reversal
2021-02-02  131.35   2021-02-09  132.41  5     227.4   +240.58   alpha_reversal
2021-02-10  131.94   2021-02-18  126.27  5     271.6   -1537.69  stop_loss
2021-02-23  122.65   2021-03-01  124.40  4     255.5   +448.71   alpha_reversal
2021-03-05  118.32   2021-03-16  122.24  7     230.1   +902.40   alpha_reversal
2021-03-18  117.45   2021-03-23  119.29  3     240.9   +443.13   alpha_reversal
2021-03-24  117.03   2021-03-26  118.00  2     263.8   +256.73   alpha_reversal
2021-03-31  119.03   2021-04-05  122.56  2     287.2   +1014.47  alpha_reversal
2021-04-19  131.40   2021-04-20  129.58  1     332.7   -604.07   alpha_reversal
2021-05-05  124.83   2021-05-12  119.72  5     323.7   -1654.26  trailing_stop
2021-05-18  121.87   2021-05-25  123.75  5     322.3   +605.11   alpha_reversal
2021-05-26  123.82   2021-06-09  123.97  9     356.9   +53.28    alpha_reversal
2021-06-24  130.23   2021-06-29  132.94  3     436.3   +1185.61  alpha_reversal
2021-07-09  141.65   2021-07-12  140.91  1     402.3   -296.31   alpha_reversal
2021-07-16  142.90   2021-07-30  142.24  10    354.0   -233.49   alpha_reversal
2021-08-03  143.84   2021-08-04  143.30  1     347.2   -188.71   alpha_reversal
2021-08-05  143.55   2021-08-06  142.72  1     367.9   -304.32   alpha_reversal
2021-08-09  142.82   2021-08-12  145.41  3     393.5   +1019.93  alpha_reversal
2021-08-25  145.04   2021-08-31  148.28  4     375.9   +1219.46  alpha_reversal
2021-09-01  149.09   2021-09-02  150.06  1     370.9   +357.64   alpha_reversal
2021-09-13  146.20   2021-09-20  139.60  5     340.7   -2248.87  stop_loss
2021-09-21  140.22   2021-09-27  141.97  4     324.8   +569.85   alpha_reversal
2021-09-28  138.73   2021-10-08  139.56  8     335.0   +277.43   alpha_reversal
2021-10-11  139.61   2021-10-21  145.99  8     326.9   +2083.77  alpha_reversal
2021-11-05  148.11   2021-11-10  144.67  3     372.3   -1278.60  alpha_reversal
2021-11-12  146.84   2021-11-16  147.69  2     390.8   +328.63   alpha_reversal
2021-12-02  160.33   2021-12-07  167.42  3     250.7   +1778.93  alpha_reversal
2021-12-14  170.67   2021-12-16  168.48  2     215.7   -473.49   alpha_reversal
2021-12-17  167.55   2021-12-27  176.37  5     193.5   +1706.55  alpha_reversal
2021-12-29  175.62   2021-12-30  174.29  1     230.5   -306.43   alpha_reversal
2021-12-31  173.85   2022-01-06  168.22  4     246.3   -1384.38  trailing_stop
2022-01-20  161.06   2022-01-31  170.94  7     236.2   +2334.21  alpha_reversal
2022-02-02  172.15   2022-02-03  169.10  1     205.4   -625.98   alpha_reversal
2022-02-07  168.27   2022-02-09  172.63  2     216.3   +942.33   alpha_reversal
2022-02-17  165.55   2022-02-23  156.76  3     235.6   -2071.98  stop_loss
2022-02-28  161.86   2022-03-08  154.18  6     206.9   -1589.75  stop_loss
2022-03-09  159.74   2022-03-11  151.53  2     196.1   -1609.90  stop_loss
2022-03-15  152.03   2022-03-22  165.32  5     189.6   +2520.00  alpha_reversal
2022-04-04  174.92   2022-04-05  171.44  1     231.5   -806.61   alpha_reversal
2022-04-06  168.44   2022-04-08  166.57  2     227.9   -426.61   alpha_reversal
2022-04-11  162.48   2022-04-13  166.87  2     232.9   +1022.66  alpha_reversal
2022-04-18  161.81   2022-04-20  163.77  2     230.6   +450.55   alpha_reversal
2022-04-21  163.14   2022-04-26  153.55  3     231.9   -2222.41  stop_loss
2022-05-06  154.40   2022-05-09  149.13  1     172.2   -908.20   alpha_reversal
2022-05-10  151.68   2022-05-11  143.68  1     172.5   -1381.18  alpha_reversal
2022-05-12  139.95   2022-05-19  134.70  5     158.3   -831.10   trailing_stop
2022-05-20  135.07   2022-05-27  146.76  5     156.0   +1821.99  alpha_reversal
2022-06-02  148.45   2022-06-03  142.58  1     171.2   -1004.16  alpha_reversal
2022-06-09  140.03   2022-06-13  129.34  2     179.4   -1918.53  stop_loss
2022-06-16  127.68   2022-06-21  133.25  2     178.6   +994.91   alpha_reversal
2022-06-23  135.74   2022-06-24  138.93  1     191.6   +611.06   alpha_reversal
2022-06-27  139.07   2022-06-28  134.79  1     202.8   -867.66   alpha_reversal
2022-06-29  136.68   2022-06-30  134.09  1     199.2   -517.58   alpha_reversal
2022-07-01  136.39   2022-07-06  140.17  2     200.2   +755.94   alpha_reversal
2022-08-03  163.09   2022-08-10  166.21  5     227.5   +709.12   alpha_reversal
2022-08-11  165.64   2022-08-16  169.93  3     248.3   +1065.80  alpha_reversal
2022-08-17  171.60   2022-08-18  171.03  1     267.1   -150.73   alpha_reversal
2022-08-24  164.70   2022-08-29  158.49  3     291.3   -1807.30  trailing_stop
2022-08-30  156.22   2022-09-08  151.69  6     257.3   -1164.74  alpha_reversal
2022-09-09  154.71   2022-09-13  151.09  2     252.8   -915.35   trailing_stop
2022-09-14  152.68   2022-09-21  150.97  5     214.9   -368.29   alpha_reversal
2022-09-29  140.07   2022-10-06  142.83  5     187.4   +516.75   alpha_reversal
2022-10-07  137.72   2022-10-11  136.49  2     193.9   -238.09   alpha_reversal
2022-10-12  136.00   2022-10-20  140.82  6     210.1   +1013.69  alpha_reversal
2022-10-21  144.78   2022-10-25  149.61  2     195.6   +945.42   alpha_reversal
2022-11-02  142.58   2022-11-09  132.67  5     174.7   -1730.16  stop_loss
2022-11-10  144.62   2022-11-18  148.83  6     158.5   +666.08   alpha_reversal
2022-11-22  147.88   2022-11-23  148.61  1     186.5   +135.74   alpha_reversal
2022-11-28  142.01   2022-12-09  139.85  9     195.8   -424.61   alpha_reversal
2022-12-13  143.25   2022-12-14  140.88  1     204.0   -482.73   alpha_reversal
2022-12-15  134.41   2022-12-28  123.99  8     192.2   -2003.82  stop_loss
2022-12-30  127.94   2023-01-11  131.32  7     210.1   +708.87   alpha_reversal
2023-01-13  132.70   2023-01-17  133.73  1     225.7   +232.03   alpha_reversal
2023-02-15  153.19   2023-02-17  150.30  2     237.1   -685.65   alpha_reversal
2023-03-01  143.31   2023-03-07  149.36  4     267.0   +1616.28  alpha_reversal
2023-03-20  155.23   2023-03-21  156.93  1     254.2   +431.49   alpha_reversal
2023-03-28  155.48   2023-03-30  159.96  2     267.7   +1200.82  alpha_reversal
2023-04-03  163.88   2023-04-04  163.18  1     289.1   -201.17   alpha_reversal
2023-04-05  161.50   2023-04-10  159.64  2     293.4   -547.44   alpha_reversal
2023-04-11  158.58   2023-04-18  164.01  5     292.9   +1590.00  alpha_reversal
2023-04-24  163.05   2023-04-25  161.35  1     332.2   -564.62   alpha_reversal
2023-05-16  169.93   2023-05-17  170.37  1     333.0   +147.14   alpha_reversal
2023-05-18  172.88   2023-05-25  170.67  5     336.4   -741.92   alpha_reversal
2023-05-30  175.10   2023-06-02  178.52  3     350.3   +1200.00  alpha_reversal
2023-06-06  176.98   2023-06-21  181.49  10    315.7   +1423.62  alpha_reversal
2023-07-05  188.95   2023-07-07  188.12  2     343.4   -285.08   alpha_reversal
2023-07-10  186.27   2023-07-17  191.39  5     336.4   +1723.04  alpha_reversal
2023-07-19  192.68   2023-07-20  190.54  1     329.9   -704.75   alpha_reversal
2023-07-24  190.36   2023-07-26  191.89  2     313.7   +481.99   alpha_reversal
2023-08-10  175.76   2023-08-15  175.31  3     278.0   -125.74   alpha_reversal
2023-09-07  175.59   2023-09-20  173.37  9     242.8   -539.17   alpha_reversal
2023-10-11  177.81   2023-10-12  178.53  1     283.2   +204.25   alpha_reversal
2023-10-17  175.19   2023-10-25  169.03  6     281.7   -1733.29  stop_loss
2023-10-27  166.35   2023-10-30  168.23  1     285.2   +535.85   alpha_reversal
2023-11-14  185.61   2023-11-21  188.59  5     289.9   +863.86   alpha_reversal
2023-11-22  189.44   2023-11-24  187.92  1     330.0   -499.98   alpha_reversal
2023-11-27  187.93   2023-11-29  187.33  2     344.4   -207.78   alpha_reversal
2023-11-30  188.09   2023-12-06  190.25  4     354.1   +763.55   alpha_reversal
2023-12-12  192.80   2023-12-14  195.98  2     322.3   +1021.96  alpha_reversal
2023-12-18  193.97   2024-01-02  183.64  9     327.5   -3384.25  stop_loss
2024-01-03  182.45   2024-01-11  183.59  6     310.5   +354.93   alpha_reversal
2024-02-09  187.24   2024-02-12  185.37  1     285.2   -533.64   alpha_reversal
2024-02-13  183.46   2024-02-28  179.69  10    288.5   -1087.30  max_holding
2024-02-29  179.21   2024-03-04  173.43  2     299.5   -1729.50  stop_loss
2024-03-05  168.67   2024-03-15  170.98  8     262.8   +606.34   alpha_reversal
2024-03-18  172.24   2024-03-20  176.97  2     266.5   +1260.79  alpha_reversal
2024-03-22  170.81   2024-04-04  167.21  8     248.0   -892.27   alpha_reversal
2024-04-09  168.22   2024-04-10  166.18  1     318.9   -650.66   alpha_reversal
2024-04-19  163.59   2024-04-26  167.69  5     274.9   +1125.82  alpha_reversal
2024-05-13  184.94   2024-05-14  185.90  1     238.2   +227.61   alpha_reversal
2024-05-23  185.54   2024-05-28  188.44  2     279.1   +809.10   alpha_reversal
2024-06-07  195.48   2024-06-10  191.54  1     308.7   -1214.73  alpha_reversal
2024-06-11  205.66   2024-06-12  211.33  1     232.6   +1318.00  alpha_reversal
2024-06-21  206.00   2024-06-28  208.90  5     190.6   +552.37   alpha_reversal
2024-07-23  223.39   2024-07-24  216.75  1     179.5   -1192.15  alpha_reversal
2024-08-07  208.31   2024-08-08  211.57  1     131.8   +428.89   alpha_reversal
2024-08-20  225.14   2024-08-21  224.81  1     173.8   -58.09    alpha_reversal
2024-08-22  223.18   2024-08-27  226.43  3     181.0   +588.78   alpha_reversal
2024-09-03  221.43   2024-09-06  219.27  3     185.5   -400.16   alpha_reversal
2024-09-11  221.32   2024-09-19  227.26  6     184.5   +1097.02  alpha_reversal
2024-09-23  225.10   2024-10-07  220.13  10    181.9   -904.22   trailing_stop
2024-10-08  224.41   2024-10-22  234.20  10    193.3   +1893.45  alpha_reversal
2024-10-25  230.01   2024-11-01  221.34  5     216.7   -1879.03  trailing_stop
2024-11-07  226.11   2024-11-20  227.64  9     218.9   +335.69   alpha_reversal
2024-11-22  228.74   2024-11-26  233.67  2     233.0   +1148.75  alpha_reversal
2024-11-29  236.16   2024-12-02  238.17  1     250.5   +503.61   alpha_reversal
2024-12-09  245.53   2024-12-10  246.30  1     256.2   +196.88   alpha_reversal
2025-01-03  242.16   2025-01-10  235.45  4     217.3   -1458.67  stop_loss
2025-01-13  233.24   2025-01-21  221.32  5     192.0   -2289.49  stop_loss
2025-01-22  222.73   2025-01-28  236.85  4     168.0   +2373.01  alpha_reversal
2025-02-06  232.07   2025-02-07  226.28  1     148.2   -858.10   alpha_reversal
2025-02-10  226.78   2025-02-11  231.49  1     151.4   +714.31   alpha_reversal
2025-03-06  234.43   2025-03-07  237.91  1     153.7   +535.91   alpha_reversal
2025-03-11  219.99   2025-03-13  208.67  2     133.8   -1514.98  stop_loss
2025-03-14  212.67   2025-03-24  219.66  6     133.7   +935.01   alpha_reversal
2025-03-27  222.99   2025-03-28  216.85  1     156.9   -964.20   alpha_reversal
2025-04-04  187.66   2025-04-08  171.59  2     118.2   -1898.92  stop_loss
2025-04-09  198.09   2025-04-15  201.16  4     83.0    +255.32   alpha_reversal
2025-05-07  195.50   2025-05-21  201.38  10    114.5   +673.35   max_holding
2025-05-23  194.77   2025-05-29  199.24  3     142.8   +638.18   alpha_reversal
2025-05-30  200.34   2025-06-03  202.55  2     153.6   +339.58   alpha_reversal
2025-06-05  200.12   2025-06-20  200.29  10    172.2   +29.04    max_holding
2025-06-25  201.05   2025-06-26  200.29  1     209.6   -159.10   alpha_reversal
2025-06-27  200.57   2025-07-01  207.09  2     217.3   +1415.72  alpha_reversal
2025-07-14  208.09   2025-07-16  209.42  2     216.0   +286.55   alpha_reversal
2025-07-18  210.64   2025-07-22  213.64  2     237.5   +712.12   alpha_reversal
2025-07-24  213.22   2025-07-29  210.52  3     253.7   -683.56   alpha_reversal
2025-08-06  212.71   2025-08-07  219.25  1     194.0   +1269.26  alpha_reversal
2025-08-19  230.24   2025-08-28  232.00  7     188.8   +333.18   alpha_reversal
2025-09-10  226.47   2025-09-17  238.42  5     196.0   +2340.69  alpha_reversal
2025-09-30  254.27   2025-10-01  254.84  1     200.4   +112.98   alpha_reversal
2025-10-07  256.12   2025-10-10  244.68  3     213.1   -2437.70  stop_loss
2025-12-04  280.58   2025-12-17  271.45  9     167.7   -1531.06  stop_loss
2025-12-18  272.07   2025-12-19  273.28  1     183.5   +221.27   alpha_reversal
2025-12-23  272.24   2025-12-30  272.69  4     194.5   +86.92    alpha_reversal
2026-01-06  262.25   2026-01-16  255.16  8     206.6   -1462.96  stop_loss
2026-01-20  246.59   2026-01-27  257.90  5     186.5   +2108.47  alpha_reversal
2026-02-10  273.82   2026-02-12  261.60  2     160.5   -1960.44  stop_loss
2026-02-13  255.91   2026-02-19  260.45  3     139.7   +634.36   alpha_reversal
2026-02-20  264.71   2026-02-23  266.05  1     140.9   +188.08   alpha_reversal
2026-03-04  262.65   2026-03-11  260.68  5     143.7   -283.34   alpha_reversal
2026-03-12  255.89   2026-03-25  252.49  9     150.6   -511.31   alpha_reversal

**Best 3 trades:**

- 2020-07-31: P&L = **+4534.02** (alpha_reversal)
- 2016-09-14: P&L = **+3853.68** (alpha_reversal)
- 2017-10-30: P&L = **+3366.94** (alpha_reversal)

**Worst 3 trades:**

- 2024-01-02: P&L = **-3384.25** (stop_loss)
- 2025-10-10: P&L = **-2437.70** (stop_loss)
- 2025-01-21: P&L = **-2289.49** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  105,315.92
2017-03-23  104,186.61
2017-09-20  106,867.34
2018-03-21  113,086.71
2018-09-18  114,687.24
2019-03-20  115,417.47
2019-09-17  119,090.96
2020-03-17  122,699.29
2020-09-14  131,363.26
2021-03-15  138,214.87
2021-09-10  141,429.09
2022-03-10  140,604.50
2022-09-08  135,826.15
2023-03-09  134,792.79
2023-09-07  139,798.88
2024-03-07  133,535.90
2024-09-05  136,667.11
2025-03-07  137,786.75
2025-09-05  139,435.02
2026-03-06  136,989.63

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.02%
2017-03-23  -1.60%
2017-09-20  -1.50%
2018-03-21  -2.43%
2018-09-18  -1.05%
2019-03-20  -0.83%
2019-09-17  -0.41%
2020-03-17  -3.25%
2020-09-14  -1.47%
2021-03-15  -0.58%
2021-09-10  -0.10%
2022-03-10  -3.13%
2022-09-08  -6.42%
2023-03-09  -7.14%
2023-09-07  -3.69%
2024-03-07  -8.00%
2024-09-05  -5.85%
2025-03-07  -5.07%
2025-09-05  -3.94%
2026-03-06  -5.62%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  57.80%
Out-of-Sample (30%)  2023-03-24  2026-03-27  8.88%

#### Return Distribution

Return Bin          Count
-2.136% to -1.619%  3
-1.619% to -1.101%  15
-1.101% to -0.584%  81
-0.584% to -0.067%  349
-0.067% to 0.450%   1879
0.450% to 0.967%    145
0.967% to 1.485%    32
1.485% to 2.002%    7
2.002% to 2.519%    2
2.519% to 3.036%    3

### BA — AlphaCombined

**Net Return (after slippage):** 5.00%  **vs SPY (exposure-adj): -141.49%** (underperform)  
**Gross Return (pre-cost):** 14.80%  
**Total Slippage Cost:** $9,803.91  
**Trade Count:** 332  
**Win Rate:** 53.0%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size   P&L       Exit Reason
2016-05-03  119.98   2016-05-09  119.51  4     278.7  -131.77   alpha_reversal
2016-05-10  122.00   2016-05-11  121.30  1     294.2  -204.95   alpha_reversal
2016-05-20  116.30   2016-05-27  117.86  5     282.6  +438.83   alpha_reversal
2016-05-31  115.17   2016-06-06  120.30  4     294.5  +1510.78  alpha_reversal
2016-06-16  118.11   2016-06-20  121.08  2     318.5  +944.15   alpha_reversal
2016-06-22  120.30   2016-06-23  121.81  1     327.5  +492.33   alpha_reversal
2016-06-27  112.02   2016-07-08  118.65  8     270.1  +1790.11  alpha_reversal
2016-07-13  118.79   2016-07-18  121.70  3     283.2  +824.00   alpha_reversal
2016-07-25  121.37   2016-07-27  124.00  2     330.3  +869.69   alpha_reversal
2016-07-29  122.03   2016-08-01  121.50  1     312.0  -166.08   alpha_reversal
2016-08-02  120.15   2016-08-04  119.67  2     323.9  -154.11   alpha_reversal
2016-08-05  120.27   2016-08-09  120.96  2     355.8  +246.05   alpha_reversal
2016-08-18  124.27   2016-08-19  123.63  1     431.2  -275.63   alpha_reversal
2016-08-22  124.26   2016-08-23  123.36  1     439.7  -398.28   alpha_reversal
2016-08-24  122.70   2016-08-31  119.04  5     425.2  -1553.52  stop_loss
2016-09-01  119.58   2016-09-07  121.98  3     420.1  +1008.37  alpha_reversal
2016-09-08  122.34   2016-09-09  118.20  1     438.6  -1816.16  stop_loss
2016-09-12  119.78   2016-09-13  118.41  1     356.5  -488.60   alpha_reversal
2016-09-14  117.53   2016-09-28  121.60  10    359.2  +1464.12  alpha_reversal
2016-10-03  121.86   2016-10-04  121.62  1     404.3  -97.59    alpha_reversal
2016-10-24  126.53   2016-10-25  127.85  1     374.3  +493.13   alpha_reversal
2016-11-03  128.89   2016-11-11  137.63  6     314.6  +2748.78  alpha_reversal
2016-11-18  135.76   2016-11-21  136.24  1     308.1  +149.46   alpha_reversal
2016-11-22  138.70   2016-11-25  139.04  2     309.5  +106.23   alpha_reversal
2016-11-29  140.66   2016-11-30  139.52  1     328.1  -374.47   alpha_reversal
2016-12-05  141.14   2016-12-13  145.17  6     340.4  +1371.65  alpha_reversal
2016-12-14  143.29   2016-12-19  144.73  3     305.4  +440.13   alpha_reversal
2016-12-28  144.80   2017-01-09  146.71  7     338.0  +646.49   alpha_reversal
2017-01-24  148.93   2017-01-25  155.09  1     373.0  +2298.54  alpha_reversal
2017-02-03  150.64   2017-02-08  153.11  3     311.9  +767.73   alpha_reversal
2017-02-09  153.70   2017-02-13  157.05  2     334.5  +1121.01  alpha_reversal
2017-03-03  170.45   2017-03-08  169.86  3     324.9  -188.98   alpha_reversal
2017-03-09  168.94   2017-03-20  167.67  7     338.4  -430.39   alpha_reversal
2017-03-21  164.63   2017-03-22  165.41  1     331.8  +261.72   alpha_reversal
2017-03-27  164.76   2017-03-28  165.77  1     344.0  +348.46   alpha_reversal
2017-04-03  165.27   2017-04-05  165.51  2     377.2  +89.29    alpha_reversal
2017-04-19  166.91   2017-04-20  167.58  1     365.1  +246.21   alpha_reversal
2017-04-21  168.76   2017-04-24  170.16  1     381.8  +535.14   alpha_reversal
2017-04-27  171.42   2017-05-01  170.47  2     374.3  -354.48   alpha_reversal
2017-05-02  171.62   2017-05-03  171.41  1     374.7  -81.78    alpha_reversal
2017-05-05  173.09   2017-05-09  174.70  2     385.4  +617.75   alpha_reversal
2017-05-11  173.41   2017-05-17  168.38  4     344.5  -1733.98  stop_loss
2017-05-19  170.41   2017-05-25  176.18  4     315.6  +1821.64  alpha_reversal
2017-05-30  176.18   2017-05-31  176.71  1     345.3  +183.08   alpha_reversal
2017-06-02  179.34   2017-06-09  178.97  5     323.7  -119.00   alpha_reversal
2017-07-14  196.57   2017-07-18  196.12  2     332.1  -149.69   alpha_reversal
2017-07-19  198.81   2017-07-24  199.83  3     347.6  +356.51   alpha_reversal
2017-08-07  226.48   2017-08-10  220.64  3     204.0  -1191.29  stop_loss
2017-08-22  227.37   2017-08-25  223.49  3     225.5  -876.01   alpha_reversal
2017-08-28  224.93   2017-08-29  227.85  1     236.0  +687.12   alpha_reversal
2017-08-31  227.29   2017-09-06  222.28  3     237.9  -1189.98  stop_loss
2017-09-07  224.11   2017-09-11  227.94  2     237.1  +908.48   alpha_reversal
2017-09-13  229.44   2017-09-14  232.34  1     225.0  +651.94   alpha_reversal
2017-09-27  242.10   2017-10-11  247.69  10    210.6  +1178.28  alpha_reversal
2017-10-16  246.34   2017-10-19  245.42  3     248.2  -228.06   alpha_reversal
2017-10-20  251.08   2017-10-23  248.53  1     227.6  -581.16   alpha_reversal
2017-10-24  252.27   2017-10-25  244.83  1     219.5  -1631.93  stop_loss
2017-10-26  245.88   2017-11-03  247.99  6     188.2  +396.00   alpha_reversal
2017-11-13  250.21   2017-11-14  249.33  1     199.8  -175.54   alpha_reversal
2017-11-15  250.63   2017-11-16  251.18  1     203.1  +111.61   alpha_reversal
2017-11-20  252.32   2017-11-27  252.97  4     219.6  +143.35   alpha_reversal
2017-11-28  255.52   2017-12-01  258.49  3     244.2  +726.29   alpha_reversal
2017-12-04  265.04   2017-12-05  262.46  1     182.5  -470.79   alpha_reversal
2017-12-12  276.45   2017-12-13  277.98  1     154.3  +236.65   alpha_reversal
2017-12-22  281.37   2017-12-26  281.34  1     173.9  -5.84     alpha_reversal
2018-01-02  283.03   2018-01-08  295.42  4     205.6  +2548.37  alpha_reversal
2018-01-17  334.68   2018-01-18  324.01  1     117.0  -1248.17  stop_loss
2018-01-19  322.02   2018-01-31  337.54  8     109.2  +1696.37  alpha_reversal
2018-02-14  330.43   2018-02-15  341.21  1     70.3   +758.48   alpha_reversal
2018-02-21  337.63   2018-02-23  341.40  2     76.5   +288.35   alpha_reversal
2018-03-02  330.25   2018-03-12  329.46  6     79.2   -62.54    alpha_reversal
2018-03-13  324.51   2018-03-22  305.94  7     79.3   -1473.43  stop_loss
2018-04-05  322.33   2018-04-09  308.68  2     70.4   -960.27   alpha_reversal
2018-04-10  320.83   2018-04-17  322.31  5     67.2   +100.07   alpha_reversal
2018-04-20  324.51   2018-04-24  314.98  2     79.6   -757.83   alpha_reversal
2018-04-25  328.52   2018-04-26  328.12  1     71.6   -28.31    alpha_reversal
2018-05-01  315.76   2018-05-15  329.12  10    77.3   +1033.00  alpha_reversal
2018-05-17  331.39   2018-05-18  337.88  1     103.3  +670.10   alpha_reversal
2018-06-06  357.80   2018-06-11  356.84  3     105.1  -100.25   alpha_reversal
2018-06-13  350.37   2018-06-19  328.16  4     111.5  -2477.21  stop_loss
2018-06-20  330.00   2018-06-27  317.28  5     100.6  -1278.80  stop_loss
2018-07-10  334.30   2018-07-13  337.46  3     103.4  +326.67   alpha_reversal
2018-07-23  340.18   2018-08-06  333.40  10    123.2  -836.08   trailing_stop
2018-08-07  337.77   2018-08-13  327.11  4     114.7  -1222.57  stop_loss
2018-08-14  328.19   2018-08-17  334.88  3     125.5  +839.50   alpha_reversal
2018-08-24  338.10   2018-09-07  337.67  9     119.9  -52.12    alpha_reversal
2018-09-10  330.82   2018-09-13  343.64  3     123.8  +1586.65  alpha_reversal
2018-09-14  348.18   2018-09-19  353.08  3     115.8  +566.27   alpha_reversal
2018-09-27  355.53   2018-09-28  359.53  1     124.8  +499.94   alpha_reversal
2018-10-11  346.55   2018-10-23  338.41  8     93.1   -757.93   trailing_stop
2018-10-30  338.61   2018-10-31  343.06  1     67.1   +298.21   alpha_reversal
2018-11-01  351.35   2018-11-06  354.28  3     66.7   +195.78   alpha_reversal
2018-11-08  360.46   2018-11-09  358.71  1     70.4   -123.21   alpha_reversal
2018-11-14  335.13   2018-11-19  311.70  3     69.3   -1624.81  stop_loss
2018-11-20  308.86   2018-11-29  332.70  6     60.5   +1441.23  alpha_reversal
2018-12-07  314.23   2018-12-10  316.95  1     56.5   +154.00   alpha_reversal
2018-12-12  317.60   2018-12-18  318.62  4     57.8   +58.54    alpha_reversal
2018-12-27  308.32   2019-01-02  314.49  3     57.4   +354.24   alpha_reversal
2019-01-07  318.98   2019-01-11  342.74  4     62.3   +1480.92  alpha_reversal
2019-01-17  349.10   2019-01-23  348.29  3     75.7   -61.74    alpha_reversal
2019-01-25  354.07   2019-01-30  376.56  3     80.7   +1815.71  alpha_reversal
2019-02-01  376.65   2019-02-04  385.57  1     79.5   +708.84   alpha_reversal
2019-02-14  400.42   2019-02-25  416.66  6     91.7   +1489.37  alpha_reversal
2019-03-05  420.26   2019-03-11  390.45  4     101.8  -3034.21  stop_loss
2019-03-12  366.80   2019-03-26  361.52  10    58.4   -308.39   alpha_reversal
2019-03-27  365.63   2019-03-28  365.49  1     67.9   -9.57     alpha_reversal
2019-04-04  386.78   2019-04-05  382.56  1     75.4   -318.23   alpha_reversal
2019-04-08  365.93   2019-04-22  366.20  9     71.8   +19.31    alpha_reversal
2019-04-23  365.44   2019-04-25  373.65  2     92.1   +755.95   alpha_reversal
2019-04-29  370.36   2019-05-07  348.69  6     93.2   -2019.47  stop_loss
2019-05-08  351.50   2019-05-13  331.20  3     93.3   -1894.95  stop_loss
2019-05-14  337.10   2019-05-20  346.33  4     84.4   +779.22   alpha_reversal
2019-06-12  341.02   2019-06-19  361.81  5     93.2   +1938.69  alpha_reversal
2019-06-20  368.39   2019-06-24  367.15  2     90.3   -112.15   alpha_reversal
2019-06-28  357.71   2019-07-08  344.69  5     96.2   -1251.61  stop_loss
2019-07-09  346.97   2019-07-17  362.76  6     105.0  +1657.65  alpha_reversal
2019-07-25  342.06   2019-08-01  328.17  5     88.7   -1231.50  stop_loss
2019-08-13  329.14   2019-08-14  316.52  1     92.5   -1166.78  stop_loss
2019-08-21  336.19   2019-08-22  350.09  1     94.3   +1310.87  alpha_reversal
2019-09-06  358.94   2019-09-12  371.06  4     92.7   +1123.75  alpha_reversal
2019-09-24  377.38   2019-09-25  381.50  1     99.1   +407.85   alpha_reversal
2019-10-02  363.25   2019-10-04  371.13  2     97.7   +769.41   alpha_reversal
2019-10-11  370.73   2019-10-18  339.81  5     109.1  -3373.52  stop_loss
2019-10-21  327.36   2019-10-30  341.85  7     86.0   +1245.50  alpha_reversal
2019-10-31  336.11   2019-11-08  348.75  6     90.3   +1140.99  alpha_reversal
2019-11-11  364.97   2019-11-12  360.55  1     84.9   -374.98   alpha_reversal
2019-11-19  365.01   2019-11-21  364.09  2     88.8   -81.78    alpha_reversal
2019-11-22  369.33   2019-11-27  365.64  3     95.3   -351.54   alpha_reversal
2019-11-29  364.20   2019-12-02  352.90  1     108.2  -1222.37  stop_loss
2019-12-05  343.81   2019-12-12  344.07  5     105.3  +27.63    alpha_reversal
2019-12-13  339.82   2019-12-16  324.90  1     101.9  -1519.52  stop_loss
2019-12-17  325.23   2019-12-24  330.86  5     91.8   +517.68   alpha_reversal
2019-12-27  328.35   2019-12-31  323.67  2     99.6   -466.24   alpha_reversal
2020-01-02  331.51   2020-01-08  329.25  4     105.6  -239.58   alpha_reversal
2020-01-09  334.52   2020-01-17  322.07  6     98.5   -1225.84  stop_loss
2020-01-21  311.67   2020-01-27  314.57  4     92.7   +268.75   alpha_reversal
2020-02-04  316.22   2020-02-05  327.44  1     93.9   +1053.74  alpha_reversal
2020-02-14  340.66   2020-02-24  317.74  5     94.4   -2162.63  stop_loss
2020-03-02  289.41   2020-03-05  260.24  3     68.9   -2011.08  stop_loss
2020-03-06  262.46   2020-03-09  227.06  1     59.4   -2101.34  stop_loss
2020-03-12  154.92   2020-03-16  129.55  2     39.3   -997.24   trailing_stop
2020-03-17  124.20   2020-03-24  127.62  5     33.4   +114.02   alpha_reversal
2020-03-31  149.21   2020-04-01  130.63  1     31.6   -586.22   alpha_reversal
2020-04-03  124.58   2020-04-16  134.17  8     33.7   +323.19   alpha_reversal
2020-04-17  154.08   2020-04-24  128.92  5     38.5   -969.95   alpha_reversal
2020-05-06  121.92   2020-05-19  130.37  9     54.6   +461.73   alpha_reversal
2020-06-01  151.47   2020-06-04  184.21  3     74.3   +2432.52  alpha_reversal
2020-06-11  170.08   2020-06-15  190.84  2     45.9   +952.45   alpha_reversal
2020-06-22  188.61   2020-06-30  183.21  6     50.0   -270.07   alpha_reversal
2020-07-01  180.41   2020-07-06  187.82  2     53.3   +394.43   alpha_reversal
2020-07-08  180.17   2020-07-09  173.19  1     60.0   -418.31   alpha_reversal
2020-07-10  178.53   2020-07-24  173.67  10    62.0   -301.27   max_holding
2020-08-04  165.15   2020-08-06  172.11  2     84.0   +584.90   alpha_reversal
2020-08-10  179.50   2020-08-12  175.35  2     88.5   -367.08   alpha_reversal
2020-08-13  174.82   2020-08-27  174.11  10    85.5   -60.26    max_holding
2020-08-28  175.89   2020-09-08  161.00  6     100.2  -1492.43  stop_loss
2020-09-10  157.77   2020-09-14  165.27  2     97.4   +730.69   alpha_reversal
2020-09-16  167.54   2020-09-18  161.06  2     98.7   -640.28   alpha_reversal
2020-09-25  156.11   2020-09-28  166.00  1     95.1   +940.44   alpha_reversal
2020-10-06  159.62   2020-10-15  164.16  7     89.9   +407.86   alpha_reversal
2020-11-03  153.73   2020-11-09  179.27  4     116.7  +2981.94  alpha_reversal
2020-11-12  176.81   2020-11-13  187.02  1     92.1   +939.71   alpha_reversal
2020-11-19  205.77   2020-11-20  199.52  1     74.9   -468.18   alpha_reversal
2020-11-30  210.82   2020-12-04  232.59  4     77.4   +1686.62  alpha_reversal
2020-12-14  228.73   2020-12-24  217.04  8     75.7   -884.61   alpha_reversal
2020-12-28  216.20   2020-12-30  216.56  2     93.0   +33.80    alpha_reversal
2021-01-04  202.82   2021-01-19  210.60  10    97.1   +755.69   max_holding
2021-02-08  212.06   2021-02-10  211.81  2     116.3  -28.15    alpha_reversal
2021-02-11  210.77   2021-02-16  217.07  2     116.3  +733.11   alpha_reversal
2021-02-18  208.58   2021-02-22  212.77  2     119.5  +500.49   alpha_reversal
2021-02-23  212.23   2021-02-25  216.34  2     110.9  +456.44   alpha_reversal
2021-03-16  255.34   2021-03-18  255.93  2     68.6   +40.74    alpha_reversal
2021-03-22  251.36   2021-03-31  254.59  7     71.6   +231.80   alpha_reversal
2021-04-05  259.49   2021-04-06  255.04  1     78.7   -349.88   alpha_reversal
2021-04-07  252.71   2021-04-20  233.94  9     84.2   -1579.83  stop_loss
2021-04-21  236.04   2021-04-28  235.34  5     94.1   -65.49    alpha_reversal
2021-05-05  228.29   2021-05-14  228.36  7     106.0  +6.54     alpha_reversal
2021-05-18  227.66   2021-05-21  234.70  3     104.6  +736.06   alpha_reversal
2021-06-07  252.79   2021-06-08  252.63  1     119.6  -18.27    alpha_reversal
2021-06-10  248.46   2021-06-18  237.23  6     119.9  -1347.25  stop_loss
2021-06-23  243.69   2021-06-25  248.26  2     132.3  +604.04   alpha_reversal
2021-06-28  240.08   2021-07-09  239.47  8     127.1  -77.53    alpha_reversal
2021-07-12  238.41   2021-07-13  228.09  1     125.5  -1295.23  stop_loss
2021-07-14  224.56   2021-07-19  206.89  3     118.3  -2090.46  stop_loss
2021-07-20  217.26   2021-07-27  222.16  5     103.2  +505.79   alpha_reversal
2021-08-03  229.20   2021-08-04  226.52  1     107.2  -288.02   alpha_reversal
2021-08-05  230.05   2021-08-09  232.15  2     108.7  +228.15   alpha_reversal
2021-08-13  234.58   2021-08-17  222.11  2     119.1  -1485.33  stop_loss
2021-08-18  219.11   2021-09-01  218.01  10    114.2  -125.42   max_holding
2021-09-02  220.94   2021-09-08  211.27  3     127.2  -1229.24  stop_loss
2021-09-13  214.59   2021-09-20  209.40  5     127.0  -659.40   alpha_reversal
2021-09-22  217.09   2021-09-23  220.99  1     128.6  +501.50   alpha_reversal
2021-10-11  226.56   2021-10-12  223.46  1     123.2  -382.52   alpha_reversal
2021-10-15  217.15   2021-10-27  206.51  8     129.1  -1373.35  stop_loss
2021-11-03  213.49   2021-11-04  212.92  1     142.6  -80.31    alpha_reversal
2021-11-05  224.57   2021-11-08  222.57  1     133.1  -266.76   alpha_reversal
2021-11-22  210.00   2021-11-26  199.11  3     105.2  -1145.78  stop_loss
2021-11-29  198.60   2021-12-01  188.10  2     95.3   -1001.18  alpha_reversal
2021-12-10  205.16   2021-12-16  190.69  4     93.2   -1348.81  stop_loss
2021-12-17  192.73   2021-12-27  203.07  5     92.8   +959.28   alpha_reversal
2021-12-28  206.23   2021-12-29  203.56  1     97.2   -260.13   alpha_reversal
2021-12-30  202.81   2022-01-05  212.96  4     104.3  +1058.44  alpha_reversal
2022-02-02  207.62   2022-02-07  211.81  3     83.7   +350.78   alpha_reversal
2022-02-15  217.84   2022-02-17  213.47  2     88.3   -385.35   alpha_reversal
2022-02-24  198.53   2022-03-03  188.76  5     86.5   -844.97   trailing_stop
2022-03-04  180.93   2022-03-09  178.47  3     81.8   -201.32   alpha_reversal
2022-03-10  178.48   2022-03-17  190.09  5     76.7   +890.93   alpha_reversal
2022-03-22  191.14   2022-03-31  191.40  7     83.7   +22.49    alpha_reversal
2022-04-04  191.28   2022-04-06  178.63  2     100.0  -1264.18  stop_loss
2022-04-07  178.06   2022-04-13  182.78  4     98.9   +466.79   alpha_reversal
2022-05-03  153.66   2022-05-06  148.83  3     92.2   -445.28   alpha_reversal
2022-05-11  129.45   2022-05-20  120.64  7     84.5   -744.88   alpha_reversal
2022-05-23  124.13   2022-06-02  140.43  7     85.0   +1384.51  alpha_reversal
2022-06-10  127.06   2022-06-13  115.80  1     105.3  -1185.66  stop_loss
2022-06-14  122.22   2022-06-15  133.65  1     95.0   +1086.00  alpha_reversal
2022-06-27  138.79   2022-07-05  137.63  5     97.8   -113.23   alpha_reversal
2022-07-07  140.04   2022-07-13  143.88  4     101.4  +389.06   alpha_reversal
2022-08-05  165.12   2022-08-15  170.38  6     109.9  +578.18   alpha_reversal
2022-09-02  151.90   2022-09-13  147.24  6     120.5  -561.51   trailing_stop
2022-09-14  149.33   2022-09-15  149.71  1     121.6  +45.04    alpha_reversal
2022-09-19  144.95   2022-09-21  143.22  2     118.4  -205.36   alpha_reversal
2022-09-26  127.40   2022-09-29  125.27  3     115.4  -246.55   alpha_reversal
2022-09-30  121.14   2022-10-07  129.73  5     112.8  +968.09   alpha_reversal
2022-11-17  172.87   2022-11-28  171.74  6     113.4  -127.27   alpha_reversal
2022-12-02  182.96   2022-12-05  185.01  1     119.6  +244.61   alpha_reversal
2022-12-29  189.00   2023-01-03  195.29  2     120.7  +758.77   alpha_reversal
2023-01-10  206.79   2023-01-25  212.57  10    115.3  +666.62   alpha_reversal
2023-02-03  206.11   2023-02-07  214.65  2     120.9  +1032.06  alpha_reversal
2023-02-09  212.10   2023-02-13  215.54  2     123.2  +424.72   alpha_reversal
2023-02-21  205.62   2023-02-23  208.03  2     126.6  +304.32   alpha_reversal
2023-02-27  200.56   2023-03-01  204.45  2     125.8  +488.88   alpha_reversal
2023-03-09  201.34   2023-03-13  203.27  2     126.4  +243.57   alpha_reversal
2023-03-16  203.29   2023-03-29  207.87  9     100.0  +457.32   alpha_reversal
2023-04-06  211.48   2023-04-13  213.48  4     122.7  +246.41   alpha_reversal
2023-04-17  205.13   2023-04-25  202.09  6     121.0  -368.43   alpha_reversal
2023-05-02  203.35   2023-05-10  200.74  6     133.1  -347.76   alpha_reversal
2023-05-11  201.94   2023-05-12  200.60  1     135.7  -181.97   alpha_reversal
2023-05-17  206.97   2023-05-18  207.14  1     137.4  +22.40    alpha_reversal
2023-05-30  204.79   2023-06-01  207.86  2     147.7  +452.38   alpha_reversal
2023-06-07  212.04   2023-06-08  218.00  1     130.7  +779.39   alpha_reversal
2023-06-16  220.10   2023-06-22  205.51  3     132.3  -1930.30  stop_loss
2023-06-23  205.51   2023-06-27  209.33  2     128.3  +489.26   alpha_reversal
2023-07-03  211.03   2023-07-06  212.51  2     154.3  +229.58   alpha_reversal
2023-07-10  213.42   2023-07-11  218.65  1     159.6  +835.30   alpha_reversal
2023-07-14  213.23   2023-07-27  233.63  9     159.0  +3244.94  alpha_reversal
2023-08-04  231.48   2023-08-11  235.60  5     137.4  +567.11   alpha_reversal
2023-08-17  224.65   2023-08-23  228.47  4     142.2  +542.30   alpha_reversal
2023-08-24  217.42   2023-08-28  226.95  2     141.0  +1343.74  alpha_reversal
2023-08-29  227.36   2023-09-06  217.84  5     138.6  -1319.52  stop_loss
2023-09-08  211.38   2023-09-20  202.27  8     138.8  -1263.75  stop_loss
2023-09-21  200.05   2023-09-26  195.54  3     163.1  -735.05   alpha_reversal
2023-09-29  191.78   2023-10-04  186.64  3     170.9  -878.36   alpha_reversal
2023-10-05  186.38   2023-10-11  195.97  4     161.9  +1552.77  alpha_reversal
2023-10-16  185.02   2023-10-24  182.27  6     148.2  -408.02   alpha_reversal
2023-11-07  191.51   2023-11-09  193.23  2     153.3  +264.85   alpha_reversal
2023-11-13  204.64   2023-11-14  207.37  1     147.1  +400.74   alpha_reversal
2023-12-26  262.92   2024-01-02  251.63  4     139.4  -1573.57  stop_loss
2024-01-04  245.06   2024-01-08  228.89  2     133.3  -2156.32  stop_loss
2024-01-09  225.87   2024-01-16  200.42  4     107.4  -2732.71  stop_loss
2024-01-17  203.16   2024-01-30  200.34  9     96.8   -273.08   trailing_stop
2024-02-09  209.30   2024-02-13  204.36  2     103.5  -512.19   alpha_reversal
2024-02-15  205.43   2024-03-01  199.90  10    114.0  -630.94   max_holding
2024-03-04  200.64   2024-03-06  200.90  2     136.3  +35.33    alpha_reversal
2024-03-07  203.13   2024-03-11  192.39  2     146.8  -1575.95  stop_loss
2024-03-12  184.33   2024-03-21  187.61  7     127.2  +416.39   alpha_reversal
2024-03-22  188.94   2024-03-25  191.31  1     135.4  +320.96   alpha_reversal
2024-04-02  188.13   2024-04-09  178.03  5     145.7  -1472.07  stop_loss
2024-04-10  174.72   2024-04-19  169.74  7     144.5  -720.10   alpha_reversal
2024-05-07  176.80   2024-05-09  181.16  2     128.0  +558.01   alpha_reversal
2024-05-13  178.53   2024-05-15  176.90  2     139.5  -227.13   alpha_reversal
2024-05-16  183.05   2024-05-21  184.69  3     135.0  +220.83   alpha_reversal
2024-05-22  186.37   2024-05-23  172.12  1     143.2  -2040.06  stop_loss
2024-05-24  174.61   2024-06-04  188.53  6     121.9  +1696.67  alpha_reversal
2024-06-11  185.59   2024-06-14  177.18  3     131.5  -1105.70  stop_loss
2024-06-17  178.48   2024-06-27  182.42  7     130.9  +515.55   alpha_reversal
2024-07-03  184.40   2024-07-08  185.75  2     136.4  +183.48   alpha_reversal
2024-07-10  183.82   2024-07-11  183.82  1     140.9  -0.54     alpha_reversal
2024-07-15  179.20   2024-07-17  184.75  2     145.9  +809.37   alpha_reversal
2024-07-23  186.57   2024-07-24  179.98  1     128.8  -849.12   alpha_reversal
2024-08-07  163.32   2024-08-21  173.34  10    100.8  +1010.27  max_holding
2024-08-23  175.05   2024-09-03  160.94  6     122.3  -1725.32  stop_loss
2024-09-04  163.30   2024-09-18  155.03  10    116.5  -963.55   max_holding
2024-09-19  154.67   2024-09-24  155.73  3     126.3  +134.46   alpha_reversal
2024-09-25  152.30   2024-10-02  152.81  5     132.9  +68.77    alpha_reversal
2024-10-03  150.60   2024-10-08  154.57  3     142.9  +568.37   alpha_reversal
2024-10-09  149.44   2024-10-18  154.92  7     146.3  +801.56   alpha_reversal
2024-10-29  153.06   2024-10-30  154.21  1     138.0  +159.56   alpha_reversal
2024-11-01  154.67   2024-11-05  150.92  2     132.0  -494.05   alpha_reversal
2024-11-06  147.23   2024-11-13  139.90  5     122.4  -897.42   trailing_stop
2024-11-14  138.21   2024-11-20  146.01  4     125.7  +979.89   alpha_reversal
2024-11-22  149.36   2024-11-25  153.02  1     132.1  +483.51   alpha_reversal
2024-12-04  158.36   2024-12-05  156.59  1     143.4  -253.50   alpha_reversal
2024-12-06  154.01   2024-12-10  164.02  2     142.4  +1425.40  alpha_reversal
2024-12-31  177.09   2025-01-02  171.78  1     126.5  -671.17   alpha_reversal
2025-01-03  169.98   2025-01-08  171.67  3     121.4  +205.14   alpha_reversal
2025-01-10  172.09   2025-01-13  170.48  1     126.5  -202.49   alpha_reversal
2025-01-14  167.10   2025-01-24  175.97  7     124.0  +1100.06  alpha_reversal
2025-02-03  175.96   2025-02-07  181.40  4     112.8  +613.98   alpha_reversal
2025-02-11  180.53   2025-02-26  172.95  10    124.8  -945.78   trailing_stop
2025-02-27  173.92   2025-02-28  174.54  1     121.0  +75.71    alpha_reversal
2025-03-05  163.24   2025-03-10  148.08  3     110.0  -1667.92  stop_loss
2025-03-11  154.14   2025-03-12  158.72  1     99.9   +457.98   alpha_reversal
2025-03-19  172.71   2025-03-20  172.74  1     106.0  +3.95     alpha_reversal
2025-04-02  168.64   2025-04-03  150.83  1     109.0  -1940.70  stop_loss
2025-04-07  138.93   2025-04-09  160.74  2     78.8   +1719.14  alpha_reversal
2025-04-10  155.60   2025-04-11  156.76  1     68.5   +79.67    alpha_reversal
2025-04-30  183.33   2025-05-01  182.80  1     88.6   -47.23    alpha_reversal
2025-05-02  185.55   2025-05-06  185.87  2     95.1   +29.90    alpha_reversal
2025-05-07  185.65   2025-05-09  194.75  2     107.1  +974.78   alpha_reversal
2025-05-19  205.35   2025-05-21  203.11  2     117.6  -263.98   alpha_reversal
2025-05-28  201.60   2025-05-29  208.08  1     134.0  +867.58   alpha_reversal
2025-06-06  210.91   2025-06-12  203.65  4     136.5  -990.65   trailing_stop
2025-06-13  200.42   2025-06-30  209.43  10    115.4  +1038.83  max_holding
2025-07-22  228.59   2025-07-28  236.29  4     123.2  +948.00   alpha_reversal
2025-08-05  224.97   2025-08-06  224.93  1     114.0  -5.13     alpha_reversal
2025-08-19  225.11   2025-08-21  224.35  2     120.0  -91.73    alpha_reversal
2025-08-25  226.98   2025-08-28  236.04  3     124.0  +1123.26  alpha_reversal
2025-09-05  229.72   2025-09-11  219.88  4     124.6  -1226.63  stop_loss
2025-09-12  216.05   2025-09-24  214.99  8     125.1  -132.06   alpha_reversal
2025-09-30  215.94   2025-10-08  225.21  6     123.6  +1145.56  alpha_reversal
2025-10-10  210.84   2025-10-20  216.71  6     124.8  +733.29   alpha_reversal
2025-11-04  198.15   2025-11-18  189.54  10    127.6  -1099.16  stop_loss
2025-11-20  179.47   2025-12-03  202.44  8     134.2  +3082.88  alpha_reversal
2025-12-04  201.97   2025-12-08  206.17  2     119.6  +502.02   alpha_reversal
2025-12-10  198.82   2025-12-15  205.40  3     128.7  +846.43   alpha_reversal
2025-12-29  217.36   2025-12-30  218.39  1     158.6  +163.73   alpha_reversal
2026-02-02  233.14   2026-02-06  242.91  4     112.3  +1097.07  alpha_reversal
2026-02-12  239.47   2026-02-13  242.84  1     101.0  +340.14   alpha_reversal
2026-02-19  233.83   2026-03-03  224.01  8     103.2  -1012.97  alpha_reversal
2026-03-04  227.42   2026-03-09  224.89  3     109.4  -277.50   alpha_reversal
2026-03-10  217.87   2026-03-12  204.66  2     94.3   -1245.79  stop_loss
2026-03-13  209.99   2026-03-20  195.02  5     95.2   -1425.78  stop_loss

**Best 3 trades:**

- 2023-07-27: P&L = **+3244.94** (alpha_reversal)
- 2025-12-03: P&L = **+3082.88** (alpha_reversal)
- 2020-11-09: P&L = **+2981.94** (alpha_reversal)

**Worst 3 trades:**

- 2019-10-18: P&L = **-3373.52** (stop_loss)
- 2019-03-11: P&L = **-3034.21** (stop_loss)
- 2024-01-16: P&L = **-2732.71** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  104,302.20
2017-03-23  113,712.77
2017-09-20  114,462.19
2018-03-21  118,009.71
2018-09-18  113,914.09
2019-03-20  117,004.26
2019-09-17  116,244.96
2020-03-17  105,546.77
2020-09-14  107,085.26
2021-03-15  114,492.09
2021-09-10  106,893.26
2022-03-10  101,806.56
2022-09-08  103,588.41
2023-03-09  106,657.85
2023-09-07  111,976.27
2024-03-07  103,051.07
2024-09-05  98,947.30
2025-03-07  100,273.37
2025-09-05  103,491.41
2026-03-06  108,349.36

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -2.11%
2017-03-23  -0.34%
2017-09-20  -0.90%
2018-03-21  -0.78%
2018-09-18  -4.22%
2019-03-20  -2.52%
2019-09-17  -3.15%
2020-03-17  -12.06%
2020-09-14  -10.78%
2021-03-15  -4.61%
2021-09-10  -10.94%
2022-03-10  -15.18%
2022-09-08  -13.69%
2023-03-09  -11.14%
2023-09-07  -6.71%
2024-03-07  -14.14%
2024-09-05  -17.56%
2025-03-07  -16.46%
2025-09-05  -13.78%
2026-03-06  -9.73%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  23.39%
Out-of-Sample (30%)  2023-03-24  2026-03-27  4.67%

#### Return Distribution

Return Bin          Count
-2.427% to -1.908%  5
-1.908% to -1.390%  6
-1.390% to -0.872%  30
-0.872% to -0.353%  164
-0.353% to 0.165%   1920
0.165% to 0.684%    315
0.684% to 1.202%    61
1.202% to 1.720%    9
1.720% to 2.239%    3
2.239% to 2.757%    3

### JPM — AlphaCombined

**Net Return (after slippage):** 11.57%  **vs SPY (exposure-adj): -72.20%** (underperform)  
**Gross Return (pre-cost):** 24.05%  
**Total Slippage Cost:** $12,476.79  
**Trade Count:** 331  
**Win Rate:** 51.1%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-06  47.48    2016-05-17  47.48   7     747.4   -0.94     alpha_reversal
2016-05-27  50.43    2016-06-13  48.72   10    778.0   -1333.04  stop_loss
2016-06-15  47.76    2016-06-22  48.28   5     746.7   +389.81   alpha_reversal
2016-06-27  44.40    2016-06-29  47.12   2     574.4   +1562.12  alpha_reversal
2016-06-30  47.89    2016-07-15  49.80   10    576.6   +1099.90  alpha_reversal
2016-07-19  49.60    2016-07-27  49.92   6     676.2   +213.10   alpha_reversal
2016-07-29  49.69    2016-08-08  51.29   6     911.1   +1460.56  alpha_reversal
2016-08-15  51.05    2016-08-25  51.27   8     1042.6  +229.95   alpha_reversal
2016-09-07  52.16    2016-09-15  51.71   6     1185.2  -540.00   alpha_reversal
2016-09-19  51.41    2016-09-21  51.86   2     1016.0  +460.23   alpha_reversal
2016-09-22  52.34    2016-09-26  51.04   2     1066.1  -1387.61  stop_loss
2016-09-30  51.72    2016-10-03  51.61   1     870.8   -99.07    alpha_reversal
2016-10-17  52.55    2016-10-24  53.83   5     840.3   +1072.41  alpha_reversal
2016-10-26  54.09    2016-10-27  54.11   1     951.5   +22.91    alpha_reversal
2016-11-04  53.01    2016-11-08  54.73   2     944.6   +1625.79  alpha_reversal
2016-11-22  61.44    2016-11-30  62.66   5     609.8   +744.19   alpha_reversal
2016-12-13  66.31    2016-12-20  67.63   5     625.7   +824.07   alpha_reversal
2016-12-29  67.20    2017-01-03  68.18   2     706.4   +692.41   alpha_reversal
2017-01-06  67.75    2017-01-17  65.66   6     730.8   -1525.49  stop_loss
2017-01-18  66.04    2017-01-30  67.61   8     626.3   +987.42   alpha_reversal
2017-02-03  68.58    2017-02-10  68.37   5     634.5   -133.26   alpha_reversal
2017-02-13  69.35    2017-02-15  71.20   2     679.3   +1255.54  alpha_reversal
2017-02-27  71.14    2017-02-28  71.22   1     759.4   +59.40    alpha_reversal
2017-03-08  71.75    2017-03-14  71.92   4     661.0   +108.44   alpha_reversal
2017-03-20  70.83    2017-03-21  68.68   1     697.7   -1496.98  stop_loss
2017-03-22  68.86    2017-03-23  68.68   1     595.2   -106.45   alpha_reversal
2017-03-24  68.67    2017-03-31  69.03   5     602.9   +219.23   alpha_reversal
2017-04-04  69.08    2017-04-13  66.71   7     619.6   -1467.99  stop_loss
2017-04-17  67.93    2017-04-25  69.76   6     584.4   +1068.92  alpha_reversal
2017-04-28  68.84    2017-05-04  68.85   4     595.4   +6.09     alpha_reversal
2017-05-08  68.92    2017-05-09  68.57   1     678.3   -234.34   alpha_reversal
2017-05-10  69.18    2017-05-11  68.89   1     687.4   -194.24   alpha_reversal
2017-05-12  68.77    2017-05-16  69.25   2     710.9   +338.87   alpha_reversal
2017-05-18  66.43    2017-05-31  64.93   8     657.9   -984.88   trailing_stop
2017-06-01  65.72    2017-06-09  68.74   6     661.2   +1994.96  alpha_reversal
2017-06-19  69.68    2017-06-27  69.60   6     688.2   -58.80    alpha_reversal
2017-07-12  73.60    2017-07-25  73.76   9     693.1   +108.76   alpha_reversal
2017-07-27  72.83    2017-08-01  73.94   3     719.9   +794.34   alpha_reversal
2017-08-04  74.51    2017-08-07  74.72   1     745.5   +157.77   alpha_reversal
2017-08-14  73.58    2017-08-16  73.19   2     736.8   -288.43   alpha_reversal
2017-08-17  72.12    2017-08-21  72.03   2     720.6   -63.41    alpha_reversal
2017-08-22  72.84    2017-08-23  72.86   1     742.5   +10.82    alpha_reversal
2017-08-25  73.11    2017-08-28  72.80   1     764.0   -231.94   alpha_reversal
2017-08-29  72.48    2017-08-31  72.24   2     762.5   -182.50   alpha_reversal
2017-09-01  72.95    2017-09-05  71.14   1     769.4   -1395.40  stop_loss
2017-09-06  71.69    2017-09-13  72.44   5     705.6   +532.70   alpha_reversal
2017-09-25  74.88    2017-09-28  75.81   3     688.7   +638.13   alpha_reversal
2017-10-09  77.15    2017-10-19  78.43   8     734.9   +942.04   alpha_reversal
2017-10-24  80.76    2017-10-26  81.33   2     693.8   +398.77   alpha_reversal
2017-10-31  80.51    2017-11-02  81.21   2     713.2   +501.38   alpha_reversal
2017-11-06  80.65    2017-11-07  78.94   1     735.5   -1252.80  stop_loss
2017-11-08  78.13    2017-11-16  78.72   6     666.3   +390.09   alpha_reversal
2017-11-22  78.93    2017-11-29  82.92   4     712.4   +2842.69  alpha_reversal
2017-12-12  85.50    2017-12-18  85.51   4     520.1   +1.28     alpha_reversal
2017-12-20  84.94    2017-12-22  85.90   2     543.3   +522.82   alpha_reversal
2017-12-27  85.80    2017-12-28  86.17   1     570.8   +211.14   alpha_reversal
2018-01-02  86.38    2018-01-11  89.07   7     612.1   +1643.81  alpha_reversal
2018-01-12  90.63    2018-01-16  90.22   1     624.8   -257.46   alpha_reversal
2018-01-17  90.89    2018-01-25  92.98   6     587.5   +1226.01  alpha_reversal
2018-02-06  90.18    2018-02-07  90.70   1     374.6   +195.01   alpha_reversal
2018-02-09  88.52    2018-02-12  89.79   1     324.7   +414.80   alpha_reversal
2018-02-20  92.27    2018-02-21  92.57   1     365.8   +107.36   alpha_reversal
2018-03-05  92.55    2018-03-07  92.20   2     354.5   -126.81   alpha_reversal
2018-03-08  92.30    2018-03-09  94.86   1     377.0   +964.97   alpha_reversal
2018-03-13  93.51    2018-03-16  92.77   3     381.6   -284.03   alpha_reversal
2018-03-23  86.08    2018-04-09  89.17   10    374.2   +1155.46  max_holding
2018-04-13  89.17    2018-04-25  88.83   8     298.2   -101.25   alpha_reversal
2018-04-30  87.95    2018-05-08  89.64   6     391.2   +663.84   alpha_reversal
2018-05-18  89.85    2018-05-29  85.56   6     498.5   -2138.52  trailing_stop
2018-05-30  87.60    2018-06-13  88.82   10    432.8   +528.35   max_holding
2018-06-15  87.23    2018-06-19  86.86   2     482.6   -178.49   alpha_reversal
2018-06-21  86.92    2018-06-27  83.38   4     510.8   -1805.89  stop_loss
2018-07-02  84.95    2018-07-17  89.73   10    460.3   +2199.10  alpha_reversal
2018-08-01  94.02    2018-08-02  94.32   1     482.0   +146.51   alpha_reversal
2018-08-10  94.07    2018-08-24  93.13   10    543.8   -514.84   max_holding
2018-08-27  94.87    2018-08-28  94.31   1     621.8   -346.75   alpha_reversal
2018-08-29  94.10    2018-09-12  91.83   9     648.5   -1472.43  stop_loss
2018-09-13  92.28    2018-09-14  92.17   1     646.5   -70.11    alpha_reversal
2018-09-17  92.54    2018-09-19  95.51   2     661.8   +1970.16  alpha_reversal
2018-09-24  94.88    2018-09-28  91.63   4     596.4   -1935.53  stop_loss
2018-10-01  92.26    2018-10-04  94.26   3     558.3   +1116.91  alpha_reversal
2018-10-12  87.54    2018-10-24  84.46   8     410.1   -1263.35  trailing_stop
2018-10-26  84.66    2018-10-30  87.25   2     399.4   +1037.39  alpha_reversal
2018-11-02  88.72    2018-11-06  89.62   2     375.1   +340.97   alpha_reversal
2018-11-12  89.18    2018-11-14  87.77   2     400.3   -565.92   alpha_reversal
2018-11-19  90.72    2018-11-20  88.68   1     377.3   -768.56   alpha_reversal
2018-11-23  87.30    2018-11-28  90.72   3     386.1   +1320.86  alpha_reversal
2018-11-29  90.09    2018-12-04  87.69   3     403.5   -970.17   trailing_stop
2018-12-06  86.10    2018-12-11  82.08   3     356.1   -1434.30  stop_loss
2018-12-12  82.69    2018-12-13  82.69   1     320.1   -0.28     alpha_reversal
2018-12-17  81.05    2018-12-21  77.01   4     344.3   -1390.43  stop_loss
2018-12-24  75.42    2018-12-26  78.47   1     319.1   +972.75   alpha_reversal
2018-12-31  79.91    2019-01-02  81.21   1     317.1   +412.89   alpha_reversal
2019-01-09  82.85    2019-01-18  86.22   7     333.5   +1124.33  alpha_reversal
2019-01-24  84.78    2019-01-25  85.23   1     392.2   +176.93   alpha_reversal
2019-02-01  85.72    2019-02-05  85.56   2     450.1   -71.96    alpha_reversal
2019-02-06  85.61    2019-02-11  83.16   3     489.3   -1195.40  stop_loss
2019-02-12  84.67    2019-02-19  86.71   4     478.7   +977.63   alpha_reversal
2019-02-22  86.65    2019-03-08  84.92   10    519.0   -896.29   max_holding
2019-03-13  86.14    2019-03-18  88.37   3     583.8   +1297.29  alpha_reversal
2019-03-21  84.89    2019-03-22  82.24   1     553.6   -1466.25  stop_loss
2019-03-25  81.64    2019-04-03  86.85   7     495.5   +2582.16  alpha_reversal
2019-04-08  87.85    2019-04-09  87.11   1     570.5   -419.71   alpha_reversal
2019-04-10  87.59    2019-04-12  92.38   2     582.9   +2791.38  alpha_reversal
2019-05-09  93.56    2019-05-13  90.92   2     513.6   -1357.94  stop_loss
2019-05-16  92.56    2019-05-21  92.81   3     455.2   +116.70   alpha_reversal
2019-05-24  91.23    2019-05-28  90.15   1     490.0   -529.05   alpha_reversal
2019-06-10  91.75    2019-06-17  90.73   5     475.9   -486.38   alpha_reversal
2019-06-18  92.06    2019-06-21  90.91   3     500.7   -574.26   alpha_reversal
2019-06-24  90.35    2019-07-01  94.43   5     492.4   +2008.99  alpha_reversal
2019-07-05  95.04    2019-07-08  94.42   1     510.2   -313.11   alpha_reversal
2019-07-09  94.92    2019-07-10  94.55   1     535.1   -198.47   alpha_reversal
2019-07-11  95.55    2019-07-17  95.36   4     556.7   -104.39   alpha_reversal
2019-07-22  95.69    2019-07-23  97.34   1     543.5   +898.25   alpha_reversal
2019-07-31  97.14    2019-08-01  94.48   1     579.9   -1540.89  stop_loss
2019-08-02  94.57    2019-08-05  91.66   1     508.0   -1475.85  stop_loss
2019-08-08  92.00    2019-08-14  87.67   4     414.1   -1791.08  stop_loss
2019-08-15  88.10    2019-08-20  89.77   3     364.9   +611.95   alpha_reversal
2019-08-21  90.11    2019-08-22  90.95   1     393.7   +330.12   alpha_reversal
2019-08-28  89.43    2019-09-05  94.01   5     396.7   +1813.10  alpha_reversal
2019-09-23  99.55    2019-09-25  98.72   2     438.8   -366.70   alpha_reversal
2019-10-01  96.76    2019-10-02  94.74   1     448.2   -905.76   alpha_reversal
2019-10-04  96.75    2019-10-10  96.31   4     420.9   -186.23   alpha_reversal
2019-10-31  105.45   2019-11-14  108.44  10    458.6   +1374.78  max_holding
2019-11-18  110.26   2019-11-22  110.29  4     532.0   +17.63    alpha_reversal
2019-11-26  111.14   2019-11-29  111.11  2     573.9   -20.18    alpha_reversal
2019-12-02  110.98   2019-12-05  112.21  3     614.1   +755.23   alpha_reversal
2019-12-16  115.93   2019-12-17  116.52  1     484.8   +287.22   alpha_reversal
2019-12-20  115.85   2019-12-30  116.90  5     514.4   +543.41   alpha_reversal
2020-01-09  116.76   2020-01-15  116.03  4     527.1   -383.60   alpha_reversal
2020-01-16  116.60   2020-01-24  113.00  5     493.2   -1773.65  stop_loss
2020-01-27  112.16   2020-02-03  113.19  5     442.7   +453.87   alpha_reversal
2020-02-10  117.01   2020-02-11  117.12  1     420.0   +43.54    alpha_reversal
2020-02-13  117.13   2020-02-24  112.16  6     445.7   -2215.88  stop_loss
2020-02-26  107.58   2020-02-27  103.00  1     337.6   -1546.23  stop_loss
2020-02-28  98.64    2020-03-06  91.73   5     269.3   -1861.89  trailing_stop
2020-03-09  79.38    2020-03-12  74.73   3     164.0   -763.43   trailing_stop
2020-03-13  88.27    2020-03-16  74.99   1     129.7   -1722.52  stop_loss
2020-03-25  77.93    2020-03-31  76.41   4     99.5    -151.31   alpha_reversal
2020-04-01  71.67    2020-04-08  80.86   5     108.4   +997.20   alpha_reversal
2020-04-20  78.72    2020-04-23  76.65   3     124.3   -257.05   alpha_reversal
2020-05-05  78.97    2020-05-13  72.06   6     174.4   -1205.85  stop_loss
2020-05-14  75.12    2020-05-15  73.66   1     189.8   -277.89   alpha_reversal
2020-05-18  77.64    2020-05-27  86.93   6     192.6   +1788.40  alpha_reversal
2020-06-10  91.04    2020-06-11  83.36   1     191.7   -1472.03  stop_loss
2020-06-12  85.72    2020-06-26  79.40   10    176.4   -1116.08  trailing_stop
2020-06-29  79.83    2020-07-13  84.55   9     196.1   +926.32   alpha_reversal
2020-07-20  84.33    2020-07-24  85.10   4     252.7   +193.15   alpha_reversal
2020-07-27  83.99    2020-07-29  86.31   2     283.3   +658.13   alpha_reversal
2020-08-03  83.29    2020-08-12  89.13   7     306.0   +1787.04  alpha_reversal
2020-08-13  88.73    2020-08-14  88.67   1     295.4   -15.98    alpha_reversal
2020-08-19  85.42    2020-08-28  88.98   7     312.8   +1116.13  alpha_reversal
2020-09-02  88.10    2020-09-08  86.52   3     346.4   -549.40   alpha_reversal
2020-09-16  86.41    2020-09-21  82.52   3     311.5   -1211.01  stop_loss
2020-09-22  81.71    2020-10-02  84.76   8     308.8   +942.70   alpha_reversal
2020-10-19  87.30    2020-10-21  86.84   2     341.5   -158.14   alpha_reversal
2020-10-22  90.00    2020-10-28  84.37   4     336.9   -1897.13  stop_loss
2020-11-05  91.28    2020-11-06  89.98   1     275.3   -359.47   alpha_reversal
2020-11-09  102.26   2020-11-10  101.83  1     209.5   -90.99    alpha_reversal
2020-11-17  101.57   2020-12-02  106.65  10    230.1   +1169.00  max_holding
2020-12-04  107.02   2020-12-09  105.78  3     263.6   -325.33   alpha_reversal
2020-12-11  104.59   2020-12-22  106.33  7     295.8   +514.55   alpha_reversal
2021-01-22  117.88   2021-01-27  112.54  3     266.8   -1423.78  stop_loss
2021-02-08  123.47   2021-02-09  122.86  1     264.0   -162.73   alpha_reversal
2021-02-10  123.05   2021-02-11  122.58  1     280.4   -130.73   alpha_reversal
2021-02-12  124.45   2021-02-16  127.32  1     285.3   +818.23   alpha_reversal
2021-02-26  129.67   2021-03-02  132.04  2     255.3   +605.21   alpha_reversal
2021-03-03  134.72   2021-03-04  132.52  1     247.9   -546.06   alpha_reversal
2021-03-05  132.96   2021-03-08  134.59  1     224.2   +364.95   alpha_reversal
2021-03-10  136.68   2021-03-19  136.55  7     213.2   -27.24    alpha_reversal
2021-03-22  133.01   2021-03-25  134.27  3     199.2   +250.56   alpha_reversal
2021-04-05  136.15   2021-04-19  135.15  10    229.2   -228.06   max_holding
2021-04-20  132.29   2021-04-22  130.48  2     261.6   -474.67   alpha_reversal
2021-04-26  133.43   2021-04-27  133.91  1     253.2   +120.89   alpha_reversal
2021-05-21  144.16   2021-05-24  144.79  1     256.8   +163.09   alpha_reversal
2021-05-27  145.66   2021-05-28  145.41  1     261.9   -63.63    alpha_reversal
2021-06-01  147.16   2021-06-02  147.02  1     272.4   -37.66    alpha_reversal
2021-06-08  146.23   2021-06-10  142.01  2     297.1   -1253.35  stop_loss
2021-06-11  142.06   2021-06-15  137.39  2     282.2   -1316.72  stop_loss
2021-06-16  138.50   2021-06-17  134.36  1     258.7   -1068.89  stop_loss
2021-06-18  131.10   2021-07-01  139.07  9     233.3   +1861.52  alpha_reversal
2021-07-06  136.74   2021-07-08  134.41  2     281.4   -657.30   alpha_reversal
2021-07-21  136.25   2021-07-27  134.86  4     208.6   -290.32   alpha_reversal
2021-07-28  135.22   2021-07-30  135.16  2     233.0   -14.89    alpha_reversal
2021-08-03  136.28   2021-08-11  143.51  6     239.0   +1727.27  alpha_reversal
2021-08-17  139.95   2021-08-23  139.54  4     254.5   -105.87   alpha_reversal
2021-08-31  142.57   2021-09-10  140.12  7     278.1   -681.04   alpha_reversal
2021-09-15  140.98   2021-09-16  140.77  1     282.8   -57.48    alpha_reversal
2021-09-17  140.55   2021-09-20  136.21  1     282.4   -1226.66  stop_loss
2021-09-21  136.36   2021-09-28  147.89  5     257.3   +2966.14  alpha_reversal
2021-10-22  154.04   2021-10-25  153.13  1     244.1   -221.25   alpha_reversal
2021-11-01  152.27   2021-11-15  149.21  10    252.5   -771.46   max_holding
2021-11-16  148.28   2021-11-19  144.16  3     294.8   -1216.36  stop_loss
2021-11-22  147.38   2021-11-23  150.75  1     272.7   +919.88   alpha_reversal
2021-11-29  144.58   2021-12-13  141.47  10    227.8   -708.42   max_holding
2021-12-16  143.85   2021-12-17  140.43  1     226.5   -773.22   alpha_reversal
2021-12-20  138.04   2021-12-23  140.88  3     210.9   +598.30   alpha_reversal
2021-12-30  142.11   2022-01-05  147.60  4     257.8   +1414.29  alpha_reversal
2022-01-21  130.88   2022-01-27  130.95  4     203.1   +15.54    alpha_reversal
2022-02-03  134.14   2022-02-08  140.54  3     190.4   +1218.76  alpha_reversal
2022-02-25  133.49   2022-03-01  122.97  2     182.7   -1920.90  stop_loss
2022-03-02  125.65   2022-03-07  116.45  3     165.1   -1518.79  stop_loss
2022-03-08  115.74   2022-03-15  119.39  5     160.5   +585.97   alpha_reversal
2022-03-21  125.98   2022-03-28  126.95  5     174.8   +170.19   alpha_reversal
2022-03-29  127.36   2022-04-05  121.06  5     194.3   -1224.78  stop_loss
2022-04-07  119.13   2022-04-11  120.75  2     209.7   +338.60   alpha_reversal
2022-04-20  119.58   2022-04-22  115.13  2     217.0   -965.73   alpha_reversal
2022-04-26  111.80   2022-05-06  112.32  8     212.0   +111.05   alpha_reversal
2022-05-17  111.04   2022-05-18  109.03  1     206.3   -414.35   alpha_reversal
2022-05-20  106.64   2022-05-24  114.72  2     209.4   +1692.48  alpha_reversal
2022-06-02  119.96   2022-06-03  118.17  1     205.3   -367.50   alpha_reversal
2022-06-06  117.90   2022-06-08  116.21  2     214.3   -361.79   alpha_reversal
2022-06-09  113.88   2022-06-10  108.54  1     225.6   -1205.21  stop_loss
2022-06-16  103.08   2022-07-01  103.54  10    205.0   +94.28    max_holding
2022-07-07  105.18   2022-07-13  102.50  4     220.5   -590.61   alpha_reversal
2022-07-20  105.01   2022-08-03  104.06  10    216.6   -207.22   max_holding
2022-08-04  103.02   2022-08-10  108.44  4     264.8   +1434.99  alpha_reversal
2022-08-18  111.52   2022-08-22  106.86  2     280.8   -1309.54  stop_loss
2022-08-23  105.91   2022-08-29  104.77  4     274.0   -312.54   alpha_reversal
2022-08-30  104.90   2022-08-31  104.17  1     270.1   -196.56   alpha_reversal
2022-09-19  108.33   2022-09-20  106.09  1     248.3   -556.79   alpha_reversal
2022-09-22  101.96   2022-09-27  96.95   3     239.9   -1202.32  stop_loss
2022-09-28  99.01    2022-10-05  102.01  5     227.7   +683.92   alpha_reversal
2022-11-11  125.16   2022-11-17  122.48  4     230.6   -617.11   alpha_reversal
2022-11-18  123.81   2022-11-21  122.95  1     248.9   -212.48   alpha_reversal
2022-11-22  124.92   2022-11-23  126.12  1     258.9   +312.18   alpha_reversal
2022-12-02  125.03   2022-12-07  121.52  3     258.9   -907.86   alpha_reversal
2022-12-08  122.92   2022-12-13  123.91  3     241.8   +238.41   alpha_reversal
2022-12-16  119.60   2022-12-20  120.77  2     241.0   +283.01   alpha_reversal
2023-01-20  125.88   2023-01-25  129.51  3     243.3   +884.53   alpha_reversal
2023-01-31  130.42   2023-02-06  132.12  4     269.5   +456.55   alpha_reversal
2023-02-22  129.12   2023-02-27  132.34  3     271.3   +874.30   alpha_reversal
2023-03-03  133.87   2023-03-07  129.05  2     286.9   -1384.62  stop_loss
2023-03-08  128.41   2023-03-09  121.34  1     278.7   -1971.43  stop_loss
2023-03-10  124.54   2023-03-13  122.19  1     211.4   -498.61   alpha_reversal
2023-03-14  125.45   2023-03-15  119.40  1     201.4   -1217.52  stop_loss
2023-03-16  121.84   2023-03-24  116.28  6     175.0   -972.57   stop_loss
2023-03-27  119.74   2023-03-29  120.22  2     170.3   +82.68    alpha_reversal
2023-03-31  121.43   2023-04-03  121.17  1     195.1   -50.93    alpha_reversal
2023-04-04  119.67   2023-04-06  119.60  2     201.9   -14.68    alpha_reversal
2023-04-10  120.11   2023-04-11  120.58  1     224.5   +105.75   alpha_reversal
2023-04-28  129.83   2023-05-02  130.34  2     236.7   +120.28   alpha_reversal
2023-05-08  128.73   2023-05-16  126.03  6     219.0   -593.32   alpha_reversal
2023-05-17  130.03   2023-05-31  127.33  9     228.2   -616.42   alpha_reversal
2023-06-01  129.21   2023-06-09  132.30  6     253.0   +781.52   alpha_reversal
2023-06-13  133.38   2023-06-14  132.75  1     276.8   -174.58   alpha_reversal
2023-06-15  134.39   2023-06-20  133.73  2     265.5   -175.19   alpha_reversal
2023-06-29  134.71   2023-06-30  136.46  1     269.1   +471.18   alpha_reversal
2023-08-02  146.95   2023-08-04  147.39  2     266.5   +116.95   alpha_reversal
2023-08-07  148.24   2023-08-14  146.21  5     275.9   -559.64   alpha_reversal
2023-08-18  140.87   2023-09-01  138.70  10    266.9   -579.72   max_holding
2023-09-06  137.08   2023-09-15  140.58  7     279.7   +978.89   alpha_reversal
2023-09-18  141.01   2023-09-19  140.69  1     302.4   -96.90    alpha_reversal
2023-10-02  135.95   2023-10-11  139.09  7     273.9   +857.57   alpha_reversal
2023-10-30  130.91   2023-11-07  137.05  6     250.8   +1539.99  alpha_reversal
2023-11-08  137.86   2023-11-09  137.32  1     294.2   -160.94   alpha_reversal
2023-11-14  141.41   2023-11-17  145.43  3     293.9   +1183.66  alpha_reversal
2023-11-28  146.26   2023-12-01  149.26  3     380.7   +1139.92  alpha_reversal
2023-12-06  148.90   2023-12-13  153.27  5     358.7   +1568.17  alpha_reversal
2023-12-26  160.41   2023-12-27  161.21  1     341.3   +273.35   alpha_reversal
2024-01-23  161.97   2024-01-30  168.78  5     270.6   +1842.70  alpha_reversal
2024-02-02  167.48   2024-02-05  167.09  1     268.3   -103.99   alpha_reversal
2024-02-09  167.74   2024-02-12  168.32  1     295.5   +171.14   alpha_reversal
2024-03-04  178.93   2024-03-06  181.48  2     301.4   +768.60   alpha_reversal
2024-03-15  182.40   2024-03-18  184.48  1     264.9   +550.23   alpha_reversal
2024-04-18  174.74   2024-04-22  182.42  2     200.9   +1543.97  alpha_reversal
2024-04-30  184.85   2024-05-02  184.59  2     221.2   -57.91    alpha_reversal
2024-05-03  183.67   2024-05-13  191.40  6     213.3   +1649.65  alpha_reversal
2024-05-21  192.35   2024-05-22  191.00  1     205.2   -278.58   alpha_reversal
2024-05-23  189.85   2024-05-31  195.16  5     211.9   +1125.15  alpha_reversal
2024-06-04  192.01   2024-06-12  184.47  6     220.7   -1664.44  stop_loss
2024-06-13  186.70   2024-06-21  189.06  5     203.3   +479.03   alpha_reversal
2024-06-24  191.74   2024-07-01  197.87  5     207.0   +1270.10  alpha_reversal
2024-07-05  198.53   2024-07-18  203.36  9     219.7   +1060.60  alpha_reversal
2024-07-25  202.29   2024-07-26  205.54  1     202.1   +657.81   alpha_reversal
2024-07-30  208.61   2024-08-01  201.40  2     198.4   -1430.47  stop_loss
2024-08-05  188.94   2024-08-08  197.62  3     154.7   +1343.45  alpha_reversal
2024-08-21  208.04   2024-08-22  209.80  1     191.4   +336.48   alpha_reversal
2024-09-06  205.96   2024-09-10  199.07  2     187.2   -1289.80  trailing_stop
2024-09-11  200.89   2024-09-23  204.77  8     144.0   +558.21   alpha_reversal
2024-09-27  204.06   2024-10-03  198.76  4     187.2   -993.62   alpha_reversal
2024-10-04  206.02   2024-10-10  207.39  4     173.8   +238.61   alpha_reversal
2024-10-22  218.60   2024-10-23  217.69  1     179.8   -163.64   alpha_reversal
2024-10-28  219.94   2024-11-07  230.33  8     187.2   +1943.93  alpha_reversal
2024-11-08  231.14   2024-11-13  234.98  3     130.0   +499.61   alpha_reversal
2024-11-20  234.85   2024-12-05  239.19  10    145.5   +632.16   max_holding
2024-12-11  237.53   2024-12-18  224.47  5     176.2   -2301.33  stop_loss
2024-12-19  227.22   2024-12-20  231.51  1     164.8   +707.59   alpha_reversal
2025-01-07  238.40   2025-01-15  247.16  5     174.0   +1522.72  alpha_reversal
2025-02-03  261.58   2025-02-05  264.86  2     164.2   +539.33   alpha_reversal
2025-02-11  269.60   2025-02-12  269.78  1     159.9   +28.96    alpha_reversal
2025-02-20  261.57   2025-02-25  252.10  3     154.0   -1457.72  stop_loss
2025-02-26  253.72   2025-02-27  253.72  1     137.3   +0.14     alpha_reversal
2025-03-05  246.60   2025-03-10  227.44  3     118.5   -2270.39  stop_loss
2025-03-11  224.65   2025-03-19  234.19  6     105.6   +1007.67  alpha_reversal
2025-03-21  236.89   2025-03-24  242.95  1     115.3   +698.63   alpha_reversal
2025-04-02  241.00   2025-04-03  223.98  1     120.1   -2043.39  stop_loss
2025-04-07  211.53   2025-04-08  213.71  1     83.7    +182.75   alpha_reversal
2025-04-17  228.81   2025-04-21  225.66  1     77.1    -243.23   alpha_reversal
2025-04-22  232.39   2025-05-02  248.84  8     80.6    +1325.51  alpha_reversal
2025-05-07  246.01   2025-05-21  257.24  10    104.4   +1173.08  max_holding
2025-05-23  257.17   2025-05-27  261.43  1     127.0   +540.66   alpha_reversal
2025-05-29  260.78   2025-05-30  260.16  1     137.8   -86.14    alpha_reversal
2025-06-02  261.07   2025-06-03  262.40  1     147.3   +195.22   alpha_reversal
2025-06-05  258.40   2025-06-13  261.10  6     156.0   +420.98   alpha_reversal
2025-06-30  285.98   2025-07-01  286.18  1     158.5   +32.79    alpha_reversal
2025-07-09  280.66   2025-07-21  288.12  8     143.6   +1070.44  alpha_reversal
2025-07-22  288.86   2025-07-23  293.85  1     145.6   +726.39   alpha_reversal
2025-07-29  294.42   2025-07-31  293.34  2     159.0   -172.73   alpha_reversal
2025-08-01  286.82   2025-08-05  288.51  2     144.3   +244.37   alpha_reversal
2025-08-06  288.78   2025-08-12  289.98  4     143.5   +171.71   alpha_reversal
2025-08-13  287.97   2025-08-15  287.64  2     140.4   -45.96    alpha_reversal
2025-08-20  289.66   2025-08-27  296.35  5     147.9   +988.19   alpha_reversal
2025-09-08  290.33   2025-09-17  308.69  7     145.6   +2673.81  alpha_reversal
2025-09-30  312.65   2025-10-02  304.54  2     155.6   -1262.67  stop_loss
2025-10-03  307.30   2025-10-10  299.39  5     151.6   -1198.99  stop_loss
2025-10-20  301.15   2025-10-28  303.84  6     113.7   +305.19   alpha_reversal
2025-11-12  319.13   2025-11-13  307.94  1     125.4   -1404.25  stop_loss
2025-11-14  302.40   2025-11-20  296.89  4     114.5   -630.47   alpha_reversal
2025-11-24  296.81   2025-12-05  313.47  8     108.9   +1814.57  alpha_reversal
2025-12-08  313.95   2025-12-09  299.01  1     126.8   -1895.00  stop_loss
2025-12-10  308.87   2025-12-11  315.80  1     102.5   +709.94   alpha_reversal
2025-12-19  315.94   2025-12-22  321.48  1     115.7   +640.28   alpha_reversal
2026-01-02  324.18   2026-01-13  310.74  7     139.2   -1869.68  trailing_stop
2026-01-22  303.78   2026-02-03  314.69  8     110.8   +1209.16  alpha_reversal
2026-02-11  310.98   2026-02-18  308.63  4     93.9    -220.58   alpha_reversal
2026-02-20  310.95   2026-02-23  297.52  1     100.2   -1345.08  stop_loss
2026-02-25  303.45   2026-02-27  300.15  2     92.9    -306.58   alpha_reversal
2026-03-02  297.71   2026-03-12  282.75  8     92.3    -1380.99  stop_loss
2026-03-13  283.58   2026-03-19  287.83  4     93.8    +397.99   alpha_reversal
2026-03-23  290.05   2026-03-25  295.27  2     102.8   +536.23   alpha_reversal

**Best 3 trades:**

- 2021-09-28: P&L = **+2966.14** (alpha_reversal)
- 2017-11-29: P&L = **+2842.69** (alpha_reversal)
- 2019-04-12: P&L = **+2791.38** (alpha_reversal)

**Worst 3 trades:**

- 2024-12-18: P&L = **-2301.33** (stop_loss)
- 2025-03-10: P&L = **-2270.39** (stop_loss)
- 2020-02-24: P&L = **-2215.88** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  103,513.78
2017-03-23  106,205.84
2017-09-20  106,315.92
2018-03-21  115,395.12
2018-09-18  113,676.83
2019-03-20  113,716.04
2019-09-17  114,611.31
2020-03-17  106,340.91
2020-09-14  108,761.68
2021-03-15  106,901.46
2021-09-10  104,794.14
2022-03-10  103,998.86
2022-09-08  100,764.74
2023-03-09  97,661.95
2023-09-07  93,523.26
2024-03-07  103,832.37
2024-09-05  110,417.44
2025-03-07  109,793.87
2025-09-05  114,799.14
2026-03-06  111,259.17

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.15%
2017-03-23  -1.52%
2017-09-20  -1.60%
2018-03-21  -0.26%
2018-09-18  -3.54%
2019-03-20  -3.51%
2019-09-17  -2.75%
2020-03-17  -9.76%
2020-09-14  -7.71%
2021-03-15  -9.29%
2021-09-10  -11.08%
2022-03-10  -11.75%
2022-09-08  -14.50%
2023-03-09  -17.13%
2023-09-07  -20.64%
2024-03-07  -11.89%
2024-09-05  -6.31%
2025-03-07  -6.83%
2025-09-05  -2.59%
2026-03-06  -5.59%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  10.28%
Out-of-Sample (30%)  2023-03-24  2026-03-27  25.46%

#### Return Distribution

Return Bin          Count
-1.944% to -1.318%  15
-1.318% to -0.691%  56
-0.691% to -0.065%  411
-0.065% to 0.562%   1910
0.562% to 1.188%    112
1.188% to 1.815%    9
1.815% to 2.442%    2
2.442% to 3.068%    0
3.068% to 3.695%    0
3.695% to 4.321%    1

### AMZN — AlphaCombined

**Net Return (after slippage):** 32.79%  **vs SPY (exposure-adj): -88.81%** (underperform)  
**Gross Return (pre-cost):** 42.55%  
**Total Slippage Cost:** $9,762.39  
**Trade Count:** 290  
**Win Rate:** 53.8%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-24  35.23    2016-05-25  35.40   1     885.5   +152.47   alpha_reversal
2016-05-31  36.16    2016-06-01  35.95   1     1001.2  -203.81   alpha_reversal
2016-06-08  36.35    2016-06-09  36.36   1     1127.9  +15.95    alpha_reversal
2016-06-16  35.89    2016-06-22  35.51   4     1181.9  -450.54   alpha_reversal
2016-06-23  36.12    2016-06-24  34.93   1     1159.1  -1381.13  stop_loss
2016-07-13  37.15    2016-07-25  36.96   8     1038.5  -195.30   alpha_reversal
2016-07-27  36.85    2016-07-29  37.92   2     1184.5  +1266.92  alpha_reversal
2016-08-16  38.22    2016-08-29  38.55   9     1486.0  +481.63   alpha_reversal
2016-09-12  38.59    2016-09-19  38.74   5     1387.2  +196.75   alpha_reversal
2016-10-10  42.11    2016-10-11  41.53   1     1318.1  -760.98   alpha_reversal
2016-10-17  40.67    2016-10-25  41.74   6     1251.1  +1339.00  alpha_reversal
2016-10-27  40.94    2016-10-28  38.80   1     1146.3  -2455.30  stop_loss
2016-10-31  39.51    2016-11-02  38.26   2     919.0   -1150.54  stop_loss
2016-11-03  38.37    2016-11-10  37.10   5     846.9   -1075.76  trailing_stop
2016-11-11  36.97    2016-11-21  38.98   6     648.3   +1304.12  alpha_reversal
2016-11-30  37.55    2016-12-12  37.99   8     755.3   +332.11   alpha_reversal
2016-12-13  38.74    2016-12-20  38.54   5     846.7   -164.81   alpha_reversal
2016-12-22  38.34    2016-12-29  38.24   4     977.1   -95.55    alpha_reversal
2017-01-03  37.70    2017-01-06  39.78   3     992.8   +2062.39  alpha_reversal
2017-01-19  40.47    2017-01-20  40.40   1     1083.9  -82.30    alpha_reversal
2017-01-23  40.91    2017-01-24  41.10   1     1089.7  +203.77   alpha_reversal
2017-02-06  40.40    2017-02-13  41.81   5     975.1   +1368.47  alpha_reversal
2017-02-27  42.45    2017-03-06  42.31   5     1192.5  -171.57   alpha_reversal
2017-03-08  42.55    2017-03-09  42.63   1     1365.1  +112.50   alpha_reversal
2017-03-13  42.75    2017-03-14  42.61   1     1508.7  -219.78   alpha_reversal
2017-03-16  42.69    2017-03-17  42.59   1     1594.3  -156.47   alpha_reversal
2017-03-20  42.87    2017-03-21  42.14   1     1684.2  -1231.13  stop_loss
2017-03-22  42.42    2017-03-29  43.69   5     1466.1  +1861.83  alpha_reversal
2017-04-07  44.77    2017-04-11  45.10   2     1128.5  +371.36   alpha_reversal
2017-04-13  44.26    2017-04-19  44.94   3     1133.4  +772.87   alpha_reversal
2017-04-24  45.39    2017-04-28  46.23   4     1176.9  +980.55   alpha_reversal
2017-05-08  47.48    2017-05-15  47.87   5     1056.6  +421.38   alpha_reversal
2017-05-16  48.33    2017-05-17  47.21   1     1144.9  -1274.55  stop_loss
2017-06-12  48.27    2017-06-19  49.73   5     751.2   +1099.82  alpha_reversal
2017-06-27  48.86    2017-07-11  49.68   9     749.6   +613.34   alpha_reversal
2017-07-28  51.03    2017-07-31  49.36   1     704.0   -1170.83  stop_loss
2017-08-01  49.83    2017-08-10  47.82   7     668.5   -1345.29  stop_loss
2017-08-11  48.42    2017-08-18  47.90   5     731.3   -383.34   alpha_reversal
2017-08-22  48.37    2017-09-06  48.37   10    803.3   -2.71     max_holding
2017-09-11  48.92    2017-09-14  49.59   3     884.6   +586.72   alpha_reversal
2017-09-15  49.36    2017-09-21  48.21   4     917.1   -1060.01  stop_loss
2017-09-22  47.78    2017-10-06  49.45   10    928.2   +1555.03  max_holding
2017-10-19  49.36    2017-10-30  55.51   7     958.4   +5903.45  alpha_reversal
2017-11-21  57.00    2017-11-28  59.65   4     845.1   +2237.15  alpha_reversal
2017-11-29  58.09    2017-12-07  57.96   6     700.9   -92.54    alpha_reversal
2017-12-13  58.24    2017-12-19  59.34   4     726.1   +801.38   alpha_reversal
2017-12-20  58.91    2018-01-02  59.42   7     799.3   +407.93   alpha_reversal
2018-01-17  64.78    2018-01-24  67.84   5     689.0   +2107.73  alpha_reversal
2018-01-25  68.93    2018-01-26  70.07   1     607.5   +689.77   alpha_reversal
2018-02-02  71.53    2018-02-05  69.47   1     418.8   -866.13   alpha_reversal
2018-02-06  72.18    2018-02-07  70.80   1     320.7   -440.84   alpha_reversal
2018-02-12  69.35    2018-02-21  74.11   6     276.2   +1315.55  alpha_reversal
2018-03-16  78.62    2018-03-23  74.74   5     474.9   -1843.96  stop_loss
2018-03-26  77.83    2018-03-27  74.82   1     405.0   -1221.68  stop_loss
2018-03-28  71.61    2018-04-02  68.57   2     321.9   -978.97   alpha_reversal
2018-04-03  69.64    2018-04-04  70.49   1     278.5   +238.34   alpha_reversal
2018-04-05  72.62    2018-04-10  71.78   3     277.0   -235.13   alpha_reversal
2018-04-16  72.11    2018-04-17  75.15   1     332.6   +1012.18  alpha_reversal
2018-05-09  80.44    2018-05-23  80.05   10    360.6   -139.63   max_holding
2018-06-07  84.51    2018-06-12  84.90   3     576.5   +223.55   alpha_reversal
2018-06-18  86.23    2018-06-19  86.70   1     671.3   +310.81   alpha_reversal
2018-06-22  85.83    2018-06-25  83.12   1     584.4   -1584.02  stop_loss
2018-06-26  84.60    2018-07-03  84.66   5     500.5   +29.47    alpha_reversal
2018-07-24  91.51    2018-07-30  88.92   4     450.4   -1167.10  trailing_stop
2018-07-31  88.92    2018-08-03  91.12   3     351.7   +774.56   alpha_reversal
2018-08-06  92.43    2018-08-07  93.08   1     380.3   +244.80   alpha_reversal
2018-08-20  93.88    2018-08-29  99.86   7     422.8   +2525.37  alpha_reversal
2018-09-06  97.96    2018-09-20  97.17   10    415.3   -331.47   max_holding
2018-09-21  95.80    2018-09-25  98.68   2     365.5   +1052.49  alpha_reversal
2018-10-03  97.69    2018-10-05  94.44   2     367.7   -1195.76  stop_loss
2018-10-08  93.27    2018-10-10  87.72   2     324.4   -1800.30  stop_loss
2018-10-11  86.01    2018-10-15  88.00   2     280.9   +559.59   alpha_reversal
2018-10-17  91.63    2018-10-18  88.49   1     270.1   -848.27   alpha_reversal
2018-10-29  76.98    2018-11-07  87.73   7     180.3   +1938.20  alpha_reversal
2018-11-15  81.01    2018-11-20  74.74   3     198.2   -1244.26  stop_loss
2018-11-21  75.87    2018-11-27  79.03   3     194.9   +615.33   alpha_reversal
2018-12-14  79.64    2018-12-20  73.00   4     208.9   -1384.92  stop_loss
2018-12-21  68.91    2018-12-26  73.51   2     192.8   +887.12   alpha_reversal
2018-12-28  73.94    2018-12-31  75.06   1     185.7   +208.53   alpha_reversal
2019-01-11  82.07    2019-01-18  84.77   5     225.2   +607.67   alpha_reversal
2019-01-23  82.04    2019-02-01  81.27   7     257.3   -198.44   alpha_reversal
2019-02-04  81.71    2019-02-12  81.86   6     258.3   +39.58    alpha_reversal
2019-02-13  82.04    2019-02-26  81.78   8     303.9   -79.59    alpha_reversal
2019-03-11  83.57    2019-03-18  87.06   5     402.8   +1406.10  alpha_reversal
2019-04-03  91.08    2019-04-08  92.45   3     465.6   +636.16   alpha_reversal
2019-04-12  92.20    2019-04-18  93.04   4     572.0   +479.88   alpha_reversal
2019-04-22  94.41    2019-04-23  96.14   1     584.8   +1010.34  alpha_reversal
2019-05-07  96.10    2019-05-09  94.95   2     443.1   -510.42   alpha_reversal
2019-05-13  91.18    2019-05-23  90.73   8     400.7   -180.68   trailing_stop
2019-05-31  88.80    2019-06-03  84.59   1     422.5   -1776.85  stop_loss
2019-06-04  86.52    2019-06-11  93.14   5     361.2   +2390.15  alpha_reversal
2019-06-26  94.94    2019-06-27  95.17   1     448.5   +102.01   alpha_reversal
2019-07-01  96.16    2019-07-03  96.90   2     469.0   +348.94   alpha_reversal
2019-07-12  100.60   2019-07-15  101.00  1     495.5   +197.55   alpha_reversal
2019-07-17  99.65    2019-07-26  97.10   7     526.4   -1340.82  stop_loss
2019-07-29  95.67    2019-08-01  92.72   3     459.8   -1356.71  stop_loss
2019-08-02  91.21    2019-08-05  88.21   1     413.9   -1239.79  stop_loss
2019-08-06  89.44    2019-08-07  89.63   1     379.7   +71.76    alpha_reversal
2019-08-09  90.42    2019-08-19  90.76   6     383.0   +128.85   alpha_reversal
2019-08-21  91.22    2019-08-22  90.19   1     407.1   -421.21   alpha_reversal
2019-08-23  87.52    2019-09-04  89.99   7     392.9   +967.04   alpha_reversal
2019-09-06  91.72    2019-09-18  90.83   8     450.4   -402.53   alpha_reversal
2019-09-19  91.12    2019-09-20  89.66   1     519.1   -756.49   alpha_reversal
2019-09-23  89.31    2019-09-26  86.95   3     493.8   -1165.93  stop_loss
2019-09-27  86.32    2019-10-02  85.62   3     433.6   -302.17   alpha_reversal
2019-10-08  85.32    2019-10-15  88.32   5     462.8   +1391.59  alpha_reversal
2019-10-16  88.92    2019-10-17  89.33   1     502.9   +207.87   alpha_reversal
2019-10-23  88.15    2019-10-25  88.02   2     500.5   -65.11    alpha_reversal
2019-10-28  88.90    2019-11-11  88.54   10    439.2   -158.21   max_holding
2019-11-12  88.94    2019-11-15  86.93   3     583.0   -1173.77  stop_loss
2019-11-18  87.67    2019-11-26  89.80   6     567.2   +1209.20  alpha_reversal
2019-12-03  88.54    2019-12-12  87.97   7     555.6   -316.55   alpha_reversal
2019-12-19  89.66    2019-12-24  89.42   3     685.3   -166.55   alpha_reversal
2019-12-26  93.49    2019-12-27  93.44   1     612.7   -25.71    alpha_reversal
2020-01-13  94.61    2020-01-27  91.37   9     539.9   -1749.73  stop_loss
2020-03-05  96.25    2020-03-06  95.01   1     235.2   -292.32   alpha_reversal
2020-03-13  89.29    2020-03-16  84.42   1     185.1   -903.29   alpha_reversal
2020-03-26  97.82    2020-04-03  95.28   6     159.7   -405.95   alpha_reversal
2020-04-13  108.50   2020-04-14  114.11  1     193.5   +1085.56  alpha_reversal
2020-04-20  119.74   2020-04-21  116.35  1     173.5   -588.61   alpha_reversal
2020-05-13  118.46   2020-05-14  119.38  1     195.7   +181.54   alpha_reversal
2020-05-19  122.53   2020-05-20  124.83  1     212.9   +491.09   alpha_reversal
2020-05-26  121.15   2020-06-03  123.86  6     224.6   +607.51   alpha_reversal
2020-06-05  124.21   2020-06-10  132.31  3     255.1   +2064.50  alpha_reversal
2020-06-12  127.31   2020-06-18  132.63  4     203.9   +1084.11  alpha_reversal
2020-06-29  134.09   2020-07-02  144.44  3     229.7   +2379.25  alpha_reversal
2020-07-14  154.28   2020-07-21  156.84  5     148.8   +380.68   alpha_reversal
2020-07-22  155.07   2020-07-27  152.68  3     136.6   -326.28   alpha_reversal
2020-08-10  157.49   2020-08-18  165.54  6     151.6   +1221.51  alpha_reversal
2020-09-03  168.48   2020-09-08  157.41  2     163.8   -1813.61  stop_loss
2020-09-09  163.51   2020-09-16  153.83  5     133.2   -1290.28  stop_loss
2020-09-17  150.51   2020-09-22  156.37  3     132.0   +773.41   alpha_reversal
2020-09-30  157.52   2020-10-13  172.10  9     140.1   +2043.21  alpha_reversal
2020-10-19  160.44   2020-10-26  160.27  5     132.6   -22.40    alpha_reversal
2020-11-04  162.14   2020-11-10  151.68  4     130.7   -1367.24  trailing_stop
2020-11-11  156.95   2020-11-18  155.20  5     125.5   -220.00   alpha_reversal
2020-11-19  155.93   2020-11-20  154.89  1     154.6   -160.25   alpha_reversal
2020-11-23  155.00   2020-11-30  158.32  4     164.3   +546.33   alpha_reversal
2020-12-03  159.42   2020-12-14  157.77  7     186.7   -307.45   alpha_reversal
2020-12-16  162.13   2020-12-17  161.72  1     208.2   -84.50    alpha_reversal
2020-12-21  160.39   2020-12-23  159.18  2     215.1   -259.22   alpha_reversal
2020-12-24  158.71   2020-12-29  166.02  2     239.7   +1750.59  alpha_reversal
2020-12-31  162.93   2021-01-06  156.84  3     233.2   -1419.83  stop_loss
2021-01-07  158.19   2021-01-21  165.27  9     214.6   +1519.46  alpha_reversal
2021-02-09  165.33   2021-02-19  162.41  7     197.8   -577.30   alpha_reversal
2021-02-22  159.12   2021-02-25  152.78  3     213.2   -1350.53  stop_loss
2021-03-03  150.33   2021-03-08  147.52  3     189.7   -531.54   alpha_reversal
2021-03-09  153.22   2021-03-12  154.40  3     165.3   +194.76   alpha_reversal
2021-03-15  154.16   2021-03-17  156.71  2     175.7   +447.41   alpha_reversal
2021-03-31  154.78   2021-04-08  164.88  5     201.1   +2031.47  alpha_reversal
2021-04-20  166.82   2021-04-22  165.37  2     227.3   -329.24   alpha_reversal
2021-04-23  167.13   2021-04-27  170.79  2     227.9   +833.82   alpha_reversal
2021-05-05  163.61   2021-05-12  157.52  5     197.5   -1202.81  stop_loss
2021-05-19  161.67   2021-05-21  160.07  2     200.7   -320.46   alpha_reversal
2021-05-24  162.33   2021-05-25  162.87  1     216.0   +116.73   alpha_reversal
2021-05-28  161.23   2021-06-03  159.27  3     242.7   -476.44   alpha_reversal
2021-06-04  160.39   2021-06-09  163.98  3     268.8   +963.33   alpha_reversal
2021-06-25  170.16   2021-06-30  171.92  3     262.3   +462.73   alpha_reversal
2021-07-01  171.73   2021-07-07  184.74  3     285.2   +3708.71  alpha_reversal
2021-07-16  178.77   2021-07-30  166.30  10    227.3   -2836.04  trailing_stop
2021-08-04  167.82   2021-08-17  162.02  9     196.8   -1141.74  trailing_stop
2021-08-23  163.38   2021-08-27  167.40  4     243.2   +978.27   alpha_reversal
2021-09-14  172.59   2021-09-17  173.04  3     269.3   +122.04   alpha_reversal
2021-09-20  167.87   2021-09-30  164.17  8     239.9   -887.73   trailing_stop
2021-10-04  159.57   2021-10-13  164.13  7     232.9   +1062.79  alpha_reversal
2021-10-20  170.84   2021-10-22  166.69  2     261.3   -1082.69  alpha_reversal
2021-11-03  169.28   2021-11-05  175.86  2     208.9   +1374.02  alpha_reversal
2021-11-12  176.35   2021-11-16  176.95  2     181.7   +109.20   alpha_reversal
2021-11-17  177.54   2021-11-18  184.71  1     195.9   +1405.02  alpha_reversal
2021-12-03  169.57   2021-12-17  169.93  10    166.5   +59.65    max_holding
2021-12-21  170.50   2021-12-27  169.58  3     162.2   -148.81   alpha_reversal
2021-12-28  170.75   2022-01-06  163.17  7     179.2   -1357.49  stop_loss
2022-01-07  162.64   2022-01-18  158.84  6     198.2   -752.44   alpha_reversal
2022-01-21  142.71   2022-01-25  139.92  2     169.5   -474.25   alpha_reversal
2022-01-26  138.94   2022-02-03  138.78  6     147.3   -24.43    trailing_stop
2022-02-10  159.08   2022-02-11  153.22  1     120.4   -706.16   alpha_reversal
2022-03-09  139.35   2022-03-11  145.45  2     129.1   +788.17   alpha_reversal
2022-03-18  161.33   2022-03-22  164.81  2     128.9   +447.98   alpha_reversal
2022-03-31  163.08   2022-04-04  168.26  2     158.8   +823.01   alpha_reversal
2022-04-06  158.84   2022-04-11  151.05  3     157.0   -1222.49  stop_loss
2022-04-12  150.86   2022-04-19  158.04  4     159.8   +1146.47  alpha_reversal
2022-04-22  144.42   2022-04-29  124.22  5     147.0   -2970.10  stop_loss
2022-05-04  125.99   2022-05-06  114.72  2     115.9   -1306.85  stop_loss
2022-05-09  108.84   2022-05-13  113.00  4     110.4   +458.88   alpha_reversal
2022-05-16  110.87   2022-05-19  107.27  3     115.9   -417.47   alpha_reversal
2022-05-24  104.15   2022-06-01  121.62  5     121.3   +2119.32  alpha_reversal
2022-06-08  121.24   2022-06-10  109.60  2     138.3   -1610.65  stop_loss
2022-06-15  107.72   2022-06-21  108.63  3     132.5   +119.53   alpha_reversal
2022-06-28  107.45   2022-06-30  106.16  2     130.6   -169.33   alpha_reversal
2022-07-01  109.61   2022-07-07  116.27  3     135.0   +898.84   alpha_reversal
2022-07-11  111.81   2022-07-15  113.49  4     146.0   +246.31   alpha_reversal
2022-08-10  142.76   2022-08-19  138.16  7     142.1   -653.93   alpha_reversal
2022-09-06  126.17   2022-09-13  126.76  5     169.0   +98.63    trailing_stop
2022-09-14  128.61   2022-09-21  118.48  5     167.2   -1694.49  stop_loss
2022-10-05  121.01   2022-10-10  113.61  3     161.0   -1190.70  stop_loss
2022-10-11  112.27   2022-10-13  112.47  2     161.4   +33.51    alpha_reversal
2022-10-14  106.95   2022-10-21  119.26  5     149.4   +1838.08  alpha_reversal
2022-11-02  92.17    2022-11-11  100.74  7     133.4   +1143.44  alpha_reversal
2022-11-18  94.19    2022-11-29  92.37   6     150.3   -272.49   alpha_reversal
2022-11-30  96.59    2022-12-06  88.21   4     178.5   -1496.45  stop_loss
2022-12-12  90.60    2022-12-13  92.44   1     214.7   +396.85   alpha_reversal
2022-12-19  84.96    2023-01-04  85.10   10    209.5   +28.28    max_holding
2023-01-05  83.16    2023-01-12  95.22   5     244.7   +2951.70  alpha_reversal
2023-02-08  100.10   2023-02-15  101.11  5     181.5   +183.24   alpha_reversal
2023-02-21  94.63    2023-02-23  95.77   2     209.6   +239.99   alpha_reversal
2023-03-02  92.18    2023-03-09  92.20   5     244.4   +6.80     alpha_reversal
2023-03-13  92.48    2023-03-14  94.83   1     235.9   +555.78   alpha_reversal
2023-03-21  100.66   2023-04-04  103.90  10    230.6   +746.75   max_holding
2023-04-06  102.11   2023-04-10  102.12  1     272.0   +2.14     alpha_reversal
2023-04-12  97.88    2023-04-18  102.25  4     272.1   +1188.98  alpha_reversal
2023-04-19  104.35   2023-04-20  103.76  1     269.4   -160.03   alpha_reversal
2023-05-26  120.17   2023-05-30  121.60  1     245.3   +350.50   alpha_reversal
2023-06-08  124.31   2023-06-21  124.77  8     227.6   +103.66   alpha_reversal
2023-06-22  130.22   2023-06-23  129.27  1     232.9   -221.23   alpha_reversal
2023-06-30  130.43   2023-07-06  128.30  3     242.7   -516.80   alpha_reversal
2023-07-07  129.84   2023-07-10  127.07  1     259.3   -720.56   alpha_reversal
2023-07-11  128.84   2023-07-24  128.74  9     255.5   -27.79    trailing_stop
2023-07-25  129.19   2023-08-03  128.85  7     256.1   -89.38    alpha_reversal
2023-08-14  140.64   2023-08-16  135.00  2     211.0   -1189.33  stop_loss
2023-08-21  134.75   2023-09-05  137.20  10    224.0   +549.58   max_holding
2023-09-15  140.46   2023-09-20  135.22  3     230.8   -1208.79  stop_loss
2023-09-21  129.39   2023-10-03  124.66  8     212.1   -1004.91  alpha_reversal
2023-10-06  128.02   2023-10-11  131.76  3     206.7   +773.02   alpha_reversal
2023-11-09  140.67   2023-11-24  146.67  10    212.0   +1271.52  max_holding
2023-11-29  146.39   2023-12-13  148.77  10    235.8   +559.35   max_holding
2023-12-14  147.49   2023-12-18  153.99  2     232.5   +1511.29  alpha_reversal
2023-12-20  152.20   2023-12-21  153.76  1     230.5   +361.20   alpha_reversal
2023-12-26  153.49   2023-12-29  151.86  3     261.0   -423.49   alpha_reversal
2024-01-05  145.31   2024-01-12  154.54  5     269.6   +2488.77  alpha_reversal
2024-01-23  156.10   2024-01-25  157.67  2     273.3   +429.98   alpha_reversal
2024-02-13  168.72   2024-02-28  173.07  10    205.0   +891.73   max_holding
2024-02-29  176.85   2024-03-04  177.49  2     233.3   +149.98   alpha_reversal
2024-03-07  176.91   2024-03-08  175.26  1     233.0   -383.60   alpha_reversal
2024-03-11  172.05   2024-03-14  178.66  3     226.4   +1497.59  alpha_reversal
2024-03-18  174.57   2024-03-19  175.81  1     232.2   +289.04   alpha_reversal
2024-03-22  178.96   2024-03-25  179.62  1     241.3   +159.44   alpha_reversal
2024-03-27  179.92   2024-04-02  180.60  3     248.3   +168.80   alpha_reversal
2024-04-16  183.41   2024-04-19  174.54  3     228.7   -2028.58  stop_loss
2024-04-22  177.32   2024-04-30  174.91  6     209.6   -504.21   alpha_reversal
2024-05-09  189.59   2024-05-23  180.96  10    174.8   -1509.16  stop_loss
2024-05-28  182.24   2024-05-29  181.93  1     210.1   -65.56    alpha_reversal
2024-06-03  178.43   2024-06-04  179.25  1     212.5   +174.53   alpha_reversal
2024-06-06  185.09   2024-06-07  184.21  1     218.3   -193.09   alpha_reversal
2024-06-17  184.15   2024-06-21  188.99  3     227.7   +1100.63  alpha_reversal
2024-06-25  186.43   2024-06-26  193.51  1     222.6   +1576.23  alpha_reversal
2024-07-01  197.30   2024-07-02  199.90  1     186.6   +485.45   alpha_reversal
2024-07-05  200.10   2024-07-10  199.69  3     194.8   -79.84    alpha_reversal
2024-07-15  192.82   2024-07-18  183.66  3     200.3   -1834.23  stop_loss
2024-07-19  183.22   2024-07-29  183.11  6     174.9   -19.79    alpha_reversal
2024-08-09  167.02   2024-08-21  180.02  8     123.4   +1604.01  alpha_reversal
2024-08-23  177.13   2024-09-04  173.24  7     162.0   -629.23   alpha_reversal
2024-09-16  184.98   2024-09-18  186.34  2     168.1   +227.70   alpha_reversal
2024-09-19  189.96   2024-09-23  193.78  2     172.4   +658.13   alpha_reversal
2024-09-30  186.42   2024-10-11  188.73  9     185.2   +426.41   alpha_reversal
2024-10-15  187.78   2024-10-22  189.61  5     206.9   +376.85   alpha_reversal
2024-11-12  209.01   2024-11-15  202.51  3     176.3   -1147.11  trailing_stop
2024-11-18  201.80   2024-12-02  210.60  9     157.4   +1385.36  alpha_reversal
2024-12-27  223.86   2025-01-10  218.83  8     158.5   -797.59   alpha_reversal
2025-01-14  217.87   2025-01-28  238.03  9     160.7   +3240.47  alpha_reversal
2025-01-30  234.76   2025-02-03  237.30  2     155.4   +395.33   alpha_reversal
2025-02-12  229.04   2025-02-21  216.47  6     153.7   -1932.67  stop_loss
2025-02-24  212.82   2025-02-26  214.24  2     157.7   +224.93   alpha_reversal
2025-03-10  194.64   2025-03-12  198.79  2     117.0   +486.00   alpha_reversal
2025-03-18  192.92   2025-03-26  201.03  6     125.0   +1014.44  alpha_reversal
2025-03-27  201.46   2025-03-28  192.62  1     138.7   -1225.30  alpha_reversal
2025-04-01  192.27   2025-04-03  178.32  2     128.8   -1796.06  stop_loss
2025-04-04  171.09   2025-04-08  170.57  2     102.9   -52.57    alpha_reversal
2025-04-09  191.20   2025-04-10  181.13  1     78.1    -786.61   alpha_reversal
2025-04-15  179.68   2025-04-28  187.61  8     81.4    +645.30   alpha_reversal
2025-05-07  188.80   2025-05-12  208.54  3     107.9   +2129.17  alpha_reversal
2025-05-16  205.69   2025-05-30  204.91  9     116.5   -91.51    alpha_reversal
2025-06-04  207.33   2025-06-05  207.81  1     152.6   +72.08    alpha_reversal
2025-06-20  209.79   2025-06-27  223.19  5     163.7   +2191.92  alpha_reversal
2025-07-01  220.57   2025-07-08  219.25  4     167.6   -221.25   alpha_reversal
2025-07-11  225.13   2025-07-14  225.58  1     185.3   +82.40    alpha_reversal
2025-07-23  228.40   2025-07-29  230.89  4     221.5   +551.55   alpha_reversal
2025-08-11  221.41   2025-08-15  230.91  4     169.5   +1611.15  alpha_reversal
2025-08-25  228.05   2025-08-28  231.48  3     173.2   +594.18   alpha_reversal
2025-09-02  225.45   2025-09-05  232.21  3     182.3   +1232.53  alpha_reversal
2025-09-11  230.06   2025-09-22  227.52  7     176.8   -450.66   alpha_reversal
2025-09-23  220.82   2025-10-07  221.67  10    172.4   +146.36   max_holding
2025-10-13  220.18   2025-10-24  224.10  9     165.3   +647.81   alpha_reversal
2025-11-13  237.70   2025-11-18  222.44  3     126.1   -1924.92  stop_loss
2025-11-19  222.80   2025-12-02  234.30  8     125.2   +1439.89  alpha_reversal
2025-12-04  229.22   2025-12-17  221.16  9     141.5   -1141.17  trailing_stop
2025-12-30  232.65   2025-12-31  230.70  1     205.1   -398.30   alpha_reversal
2026-01-05  233.18   2026-01-07  241.44  2     179.8   +1485.53  alpha_reversal
2026-02-11  204.18   2026-02-20  210.00  6     115.1   +670.25   alpha_reversal
2026-03-03  208.83   2026-03-04  216.71  1     136.8   +1077.42  alpha_reversal
2026-03-12  209.63   2026-03-26  207.44  10    145.6   -320.18   alpha_reversal

**Best 3 trades:**

- 2017-10-30: P&L = **+5903.45** (alpha_reversal)
- 2021-07-07: P&L = **+3708.71** (alpha_reversal)
- 2025-01-28: P&L = **+3240.47** (alpha_reversal)

**Worst 3 trades:**

- 2022-04-29: P&L = **-2970.10** (stop_loss)
- 2021-07-30: P&L = **-2836.04** (trailing_stop)
- 2016-10-28: P&L = **-2455.30** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  99,882.95
2017-03-23  98,960.17
2017-09-20  100,926.90
2018-03-21  114,353.82
2018-09-18  111,940.57
2019-03-20  112,550.94
2019-09-17  111,789.69
2020-03-17  107,387.42
2020-09-14  112,634.38
2021-03-15  112,972.42
2021-09-10  116,221.69
2022-03-10  115,884.98
2022-09-08  114,234.52
2023-03-09  115,895.77
2023-09-07  116,456.77
2024-03-07  122,235.83
2024-09-05  122,064.85
2025-03-07  125,122.66
2025-09-05  131,581.25
2026-03-06  133,112.28

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.28%
2017-03-23  -1.97%
2017-09-20  -2.95%
2018-03-21  -0.10%
2018-09-18  -2.20%
2019-03-20  -1.67%
2019-09-17  -3.50%
2020-03-17  -7.30%
2020-09-14  -2.78%
2021-03-15  -2.48%
2021-09-10  -3.66%
2022-03-10  -3.94%
2022-09-08  -5.31%
2023-03-09  -3.93%
2023-09-07  -3.46%
2024-03-07  -0.22%
2024-09-05  -1.62%
2025-03-07  -1.50%
2025-09-05  -0.46%
2026-03-06  -0.01%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  35.57%
Out-of-Sample (30%)  2023-03-24  2026-03-27  22.14%

#### Return Distribution

Return Bin          Count
-2.591% to -1.724%  4
-1.724% to -0.857%  36
-0.857% to 0.009%   802
0.009% to 0.876%    1632
0.876% to 1.743%    40
1.743% to 2.609%    1
2.609% to 3.476%    0
3.476% to 4.343%    0
4.343% to 5.209%    0
5.209% to 6.076%    1

### GOOGL — AlphaCombined

**Net Return (after slippage):** 48.76%  **vs SPY (exposure-adj): -141.14%** (underperform)  
**Gross Return (pre-cost):** 60.41%  
**Total Slippage Cost:** $11,650.86  
**Trade Count:** 298  
**Win Rate:** 56.7%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-03  35.15    2016-05-05  35.42   2     938.6   +258.73   alpha_reversal
2016-05-16  36.23    2016-05-31  37.12   10    1059.2  +935.50   max_holding
2016-06-02  36.93    2016-06-15  36.29   9     1201.9  -763.99   alpha_reversal
2016-06-16  35.93    2016-06-17  34.91   1     1257.9  -1292.10  stop_loss
2016-06-20  35.03    2016-06-24  33.96   4     1147.0  -1230.11  trailing_stop
2016-06-27  33.79    2016-07-12  36.31   10    1024.2  +2573.26  max_holding
2016-07-14  36.51    2016-07-15  36.46   1     1248.7  -56.09    alpha_reversal
2016-07-26  37.59    2016-07-28  37.96   2     1410.7  +519.65   alpha_reversal
2016-08-09  40.06    2016-08-11  40.06   2     1268.3  -5.51     alpha_reversal
2016-08-12  40.04    2016-08-25  39.22   9     1401.0  -1149.76  stop_loss
2016-08-26  39.36    2016-09-07  40.05   7     1583.8  +1097.17  alpha_reversal
2016-09-12  39.63    2016-09-13  39.09   1     1458.5  -787.92   alpha_reversal
2016-09-14  39.22    2016-09-16  39.55   2     1400.6  +466.46   alpha_reversal
2016-09-19  39.46    2016-09-21  39.90   2     1370.4  +600.73   alpha_reversal
2016-10-03  39.71    2016-10-05  39.71   2     1395.9  +3.40     alpha_reversal
2016-10-06  39.84    2016-10-07  39.69   1     1478.6  -232.57   alpha_reversal
2016-10-10  40.39    2016-10-13  39.85   3     1395.1  -754.02   alpha_reversal
2016-10-26  40.79    2016-10-31  40.14   3     1273.7  -822.11   alpha_reversal
2016-11-01  39.96    2016-11-03  38.77   2     1113.8  -1330.21  stop_loss
2016-11-04  38.75    2016-11-11  38.25   5     1018.0  -511.23   trailing_stop
2016-11-14  37.37    2016-11-22  38.91   6     768.6   +1181.95  alpha_reversal
2016-11-25  38.71    2016-11-29  39.13   2     865.3   +361.53   alpha_reversal
2016-12-07  39.27    2016-12-09  40.12   2     838.8   +714.58   alpha_reversal
2016-12-19  40.31    2016-12-20  40.41   1     955.5   +89.37    alpha_reversal
2016-12-22  40.17    2017-01-04  40.04   7     1055.3  -142.27   alpha_reversal
2017-01-05  40.34    2017-01-06  40.90   1     1134.4  +639.69   alpha_reversal
2017-01-11  41.17    2017-01-19  40.86   5     1189.3  -372.55   alpha_reversal
2017-01-20  41.09    2017-01-24  42.11   2     1378.7  +1403.00  alpha_reversal
2017-01-27  41.93    2017-01-30  40.83   1     1212.7  -1325.09  stop_loss
2017-01-31  40.69    2017-02-07  41.10   5     1118.4  +455.63   alpha_reversal
2017-02-16  41.78    2017-02-17  41.96   1     1440.4  +252.56   alpha_reversal
2017-02-27  42.16    2017-02-28  41.88   1     1543.8  -427.73   alpha_reversal
2017-03-06  42.04    2017-03-08  42.31   2     1549.2  +424.05   alpha_reversal
2017-03-21  42.18    2017-03-24  41.39   3     1494.7  -1174.26  stop_loss
2017-03-27  41.60    2017-04-04  42.26   6     1280.2  +838.91   alpha_reversal
2017-04-10  41.76    2017-04-13  41.64   3     1390.9  -162.86   alpha_reversal
2017-05-11  47.43    2017-05-12  47.34   1     1176.0  -99.46    alpha_reversal
2017-06-12  47.72    2017-06-22  48.41   8     869.1   +596.53   alpha_reversal
2017-06-23  48.92    2017-06-26  48.18   1     897.8   -666.91   alpha_reversal
2017-06-27  47.04    2017-07-03  45.57   4     822.3   -1205.57  trailing_stop
2017-07-05  46.25    2017-07-11  47.26   4     738.8   +744.68   alpha_reversal
2017-07-26  47.89    2017-08-03  46.61   6     821.3   -1057.39  stop_loss
2017-08-04  46.93    2017-08-17  45.98   9     835.6   -790.09   alpha_reversal
2017-08-22  46.66    2017-08-24  46.44   2     962.7   -212.38   alpha_reversal
2017-08-28  46.05    2017-08-30  46.77   2     1007.3  +727.48   alpha_reversal
2017-09-05  46.71    2017-09-08  46.66   3     1000.5  -50.18    alpha_reversal
2017-09-11  46.80    2017-09-21  46.97   8     1028.9  +169.12   alpha_reversal
2017-09-25  46.35    2017-09-27  47.58   2     1057.7  +1294.11  alpha_reversal
2017-10-05  48.88    2017-10-12  49.85   5     1014.3  +979.11   alpha_reversal
2017-10-20  49.87    2017-10-23  48.85   1     1095.5  -1115.04  stop_loss
2017-10-24  49.04    2017-10-30  51.21   4     1027.4  +2222.94  alpha_reversal
2017-11-07  52.21    2017-11-08  52.45   1     851.5   +204.58   alpha_reversal
2017-11-13  51.66    2017-11-24  52.37   8     894.3   +632.93   alpha_reversal
2017-11-29  51.47    2017-12-04  50.15   3     855.2   -1125.35  stop_loss
2017-12-05  50.59    2017-12-06  51.19   1     759.3   +455.38   alpha_reversal
2017-12-22  53.03    2018-01-09  55.16   10    925.3   +1965.78  alpha_reversal
2018-01-11  55.17    2018-01-22  57.70   6     1050.2  +2654.52  alpha_reversal
2018-01-25  58.65    2018-01-26  58.86   1     929.8   +195.28   alpha_reversal
2018-02-02  55.53    2018-02-05  52.66   1     737.9   -2118.73  stop_loss
2018-02-06  53.80    2018-02-08  49.95   2     562.5   -2169.27  stop_loss
2018-02-09  51.91    2018-02-12  52.27   1     461.9   +165.84   alpha_reversal
2018-02-13  52.30    2018-02-21  55.20   5     491.6   +1426.85  alpha_reversal
2018-03-19  54.58    2018-03-22  52.20   3     503.0   -1197.32  stop_loss
2018-03-23  50.93    2018-03-27  49.91   2     472.7   -483.51   alpha_reversal
2018-03-28  49.87    2018-04-02  50.19   2     407.4   +130.13   alpha_reversal
2018-04-03  50.54    2018-04-06  50.06   3     396.7   -191.68   alpha_reversal
2018-04-09  50.61    2018-04-10  51.37   1     407.6   +310.93   alpha_reversal
2018-04-12  51.47    2018-04-13  51.35   1     442.9   -50.22    alpha_reversal
2018-04-16  51.90    2018-04-20  53.40   4     465.5   +696.18   alpha_reversal
2018-04-24  50.74    2018-05-08  52.47   10    458.1   +793.07   alpha_reversal
2018-05-09  54.03    2018-05-23  53.83   10    519.1   -104.95   max_holding
2018-05-25  53.79    2018-06-01  56.26   4     667.5   +1648.77  alpha_reversal
2018-06-11  56.61    2018-06-19  58.42   6     684.2   +1242.83  alpha_reversal
2018-06-27  55.42    2018-07-03  55.33   4     636.6   -56.08    alpha_reversal
2018-07-05  56.63    2018-07-09  57.86   2     586.6   +722.41   alpha_reversal
2018-07-11  58.12    2018-07-12  59.54   1     639.9   +908.00   alpha_reversal
2018-07-19  59.49    2018-07-24  62.36   3     650.9   +1866.26  alpha_reversal
2018-08-01  61.17    2018-08-10  62.08   7     563.6   +510.85   alpha_reversal
2018-08-17  60.32    2018-08-30  62.18   9     636.0   +1178.23  alpha_reversal
2018-09-04  60.10    2018-09-07  58.37   3     697.0   -1206.84  stop_loss
2018-09-10  58.30    2018-09-24  58.47   10    649.6   +107.03   max_holding
2018-10-05  57.94    2018-10-10  54.13   3     633.8   -2413.98  stop_loss
2018-10-16  56.22    2018-10-23  55.26   5     520.8   -498.30   alpha_reversal
2018-10-29  51.34    2018-11-01  53.83   3     351.2   +874.05   alpha_reversal
2018-11-02  53.16    2018-11-15  53.09   9     365.2   -27.37    alpha_reversal
2018-11-26  52.39    2018-11-27  52.16   1     447.3   -104.58   alpha_reversal
2018-12-14  52.18    2018-12-17  50.84   1     467.3   -627.96   alpha_reversal
2018-12-20  50.78    2018-12-26  51.94   3     420.9   +484.98   alpha_reversal
2019-01-02  52.33    2019-01-03  50.83   1     397.1   -595.66   alpha_reversal
2019-01-07  53.38    2019-01-18  54.88   9     390.4   +586.41   alpha_reversal
2019-01-25  54.65    2019-01-28  53.52   1     523.8   -590.72   alpha_reversal
2019-01-29  53.09    2019-02-05  57.09   5     534.8   +2140.05  alpha_reversal
2019-02-11  54.68    2019-02-15  55.49   4     577.7   +469.79   alpha_reversal
2019-02-19  55.89    2019-02-20  55.54   1     622.4   -217.42   alpha_reversal
2019-02-21  54.79    2019-03-01  56.93   6     631.9   +1353.23  alpha_reversal
2019-03-18  58.97    2019-03-21  61.27   3     752.1   +1729.37  alpha_reversal
2019-03-27  58.45    2019-04-03  60.01   5     701.8   +1099.88  alpha_reversal
2019-04-08  59.95    2019-04-11  59.95   3     841.3   +4.22     alpha_reversal
2019-05-01  58.21    2019-05-08  58.03   5     656.4   -120.83   alpha_reversal
2019-05-15  58.09    2019-05-16  58.71   1     570.5   +354.29   alpha_reversal
2019-05-28  56.54    2019-05-29  55.51   1     649.2   -667.97   alpha_reversal
2019-06-04  52.32    2019-06-14  53.84   8     579.3   +883.14   alpha_reversal
2019-06-25  53.96    2019-07-03  55.66   6     714.0   +1214.57  alpha_reversal
2019-07-18  56.92    2019-07-24  56.49   4     923.9   -396.48   alpha_reversal
2019-07-25  56.36    2019-07-29  61.55   2     900.0   +4673.25  alpha_reversal
2019-08-07  58.34    2019-08-15  57.96   6     601.7   -231.65   alpha_reversal
2019-08-23  57.23    2019-08-29  59.19   4     624.3   +1222.46  alpha_reversal
2019-09-03  58.03    2019-09-13  61.46   8     678.2   +2329.74  alpha_reversal
2019-09-23  61.26    2019-09-24  60.39   1     846.9   -738.63   alpha_reversal
2019-09-25  61.82    2019-09-26  61.57   1     747.5   -181.41   alpha_reversal
2019-09-30  60.59    2019-10-02  58.38   2     766.0   -1687.43  stop_loss
2019-10-03  59.01    2019-10-10  59.95   5     703.8   +657.59   alpha_reversal
2019-10-14  60.42    2019-10-15  61.57   1     781.3   +900.44   alpha_reversal
2019-10-22  61.58    2019-10-29  62.49   5     844.5   +762.54   alpha_reversal
2019-10-30  62.55    2019-11-08  64.88   7     804.2   +1874.98  alpha_reversal
2019-11-12  64.36    2019-11-18  65.42   4     912.4   +964.73   alpha_reversal
2019-11-20  64.59    2019-12-03  64.17   8     908.5   -379.27   alpha_reversal
2019-12-11  66.69    2019-12-12  66.84   1     987.6   +141.72   alpha_reversal
2019-12-16  67.51    2019-12-31  66.39   10    1023.2  -1149.79  max_holding
2020-01-02  67.91    2020-01-07  69.15   3     1070.7  +1330.00  alpha_reversal
2020-01-31  71.09    2020-02-04  71.64   2     757.3   +420.27   alpha_reversal
2020-02-05  71.75    2020-02-10  74.78   3     613.8   +1860.81  alpha_reversal
2020-02-11  74.92    2020-02-13  75.01   2     620.4   +55.94    alpha_reversal
2020-03-05  65.23    2020-03-09  60.26   2     385.7   -1916.94  stop_loss
2020-03-11  60.08    2020-03-12  55.09   1     323.2   -1610.92  stop_loss
2020-03-13  60.25    2020-03-16  53.18   1     282.3   -1993.52  stop_loss
2020-03-17  55.47    2020-03-18  54.09   1     248.4   -344.62   alpha_reversal
2020-03-24  56.07    2020-04-01  54.63   6     235.3   -338.71   alpha_reversal
2020-04-02  55.42    2020-04-09  59.80   5     258.8   +1134.18  alpha_reversal
2020-04-13  60.05    2020-04-14  62.71   1     292.1   +776.16   alpha_reversal
2020-04-16  62.39    2020-04-20  62.51   2     305.1   +37.23    alpha_reversal
2020-04-21  60.14    2020-04-24  63.28   3     325.4   +1019.89  alpha_reversal
2020-05-19  68.19    2020-06-02  71.49   9     402.6   +1327.62  alpha_reversal
2020-06-04  70.17    2020-06-16  71.69   8     506.3   +771.83   alpha_reversal
2020-06-19  70.68    2020-06-23  72.56   2     507.5   +953.74   alpha_reversal
2020-06-29  69.32    2020-07-06  74.33   4     456.1   +2285.36  alpha_reversal
2020-07-16  75.16    2020-07-21  77.12   3     460.5   +901.25   alpha_reversal
2020-07-22  77.64    2020-07-23  75.18   1     467.1   -1149.95  alpha_reversal
2020-08-05  73.38    2020-08-13  75.17   6     465.9   +833.11   alpha_reversal
2020-08-17  75.23    2020-08-18  77.11   1     529.3   +997.57   alpha_reversal
2020-09-04  78.45    2020-09-14  74.79   5     401.6   -1472.15  stop_loss
2020-09-15  76.16    2020-09-18  71.92   3     384.2   -1629.55  stop_loss
2020-09-21  70.96    2020-09-30  72.64   7     368.0   +620.62   alpha_reversal
2020-10-01  73.82    2020-10-02  72.15   1     424.2   -710.49   alpha_reversal
2020-10-05  73.57    2020-10-06  71.92   1     431.4   -711.85   alpha_reversal
2020-10-07  72.40    2020-10-13  77.67   4     433.0   +2285.03  alpha_reversal
2020-11-09  87.39    2020-11-23  85.63   10    319.3   -563.78   max_holding
2020-11-24  87.52    2020-12-07  90.06   8     406.6   +1035.26  alpha_reversal
2020-12-09  88.21    2020-12-14  86.85   3     439.8   -596.79   alpha_reversal
2020-12-15  87.38    2020-12-28  87.93   8     460.1   +253.57   alpha_reversal
2020-12-31  86.96    2021-01-15  85.63   10    506.8   -672.51   max_holding
2021-02-09  102.97   2021-02-12  103.84  3     301.3   +262.26   alpha_reversal
2021-03-09  101.23   2021-03-22  100.65  9     289.6   -168.11   alpha_reversal
2021-03-25  100.84   2021-03-30  101.43  3     333.1   +197.75   alpha_reversal
2021-04-13  111.85   2021-04-14  111.12  1     381.2   -279.17   alpha_reversal
2021-04-15  113.38   2021-04-20  112.96  3     383.7   -162.15   alpha_reversal
2021-05-10  113.71   2021-05-12  109.06  2     354.4   -1647.44  stop_loss
2021-05-13  110.59   2021-05-18  112.14  3     338.9   +524.12   alpha_reversal
2021-05-19  112.70   2021-05-20  114.34  1     348.6   +573.29   alpha_reversal
2021-06-01  118.14   2021-06-02  117.50  1     405.7   -260.82   alpha_reversal
2021-06-03  116.48   2021-06-08  118.88  3     407.1   +978.90   alpha_reversal
2021-06-14  121.50   2021-06-15  120.36  1     475.1   -540.88   alpha_reversal
2021-06-21  120.87   2021-06-23  121.07  2     461.0   +88.03    alpha_reversal
2021-06-24  121.56   2021-07-02  124.17  6     499.6   +1304.85  alpha_reversal
2021-08-09  135.86   2021-08-10  135.62  1     426.3   -102.68   alpha_reversal
2021-08-11  135.23   2021-08-16  137.11  3     445.5   +836.47   alpha_reversal
2021-09-03  142.63   2021-09-10  139.65  4     458.1   -1365.66  stop_loss
2021-09-13  141.24   2021-09-15  143.17  2     450.2   +872.38   alpha_reversal
2021-09-20  137.65   2021-09-28  134.65  6     390.5   -1172.16  trailing_stop
2021-09-29  133.32   2021-10-07  138.01  6     352.7   +1656.06  alpha_reversal
2021-10-12  135.40   2021-10-14  139.92  2     329.7   +1492.10  alpha_reversal
2021-10-15  140.28   2021-10-19  141.99  2     349.3   +598.22   alpha_reversal
2021-11-02  144.31   2021-11-09  147.62  5     282.2   +934.37   alpha_reversal
2021-11-10  144.77   2021-11-18  148.54  6     295.2   +1111.66  alpha_reversal
2021-12-06  142.05   2021-12-09  146.36  3     279.0   +1200.50  alpha_reversal
2021-12-15  145.31   2021-12-16  143.19  1     279.1   -592.87   alpha_reversal
2021-12-17  140.63   2021-12-28  145.41  6     274.6   +1312.30  alpha_reversal
2021-12-29  145.53   2022-01-05  136.58  5     317.6   -2841.89  stop_loss
2022-01-06  136.69   2022-01-11  138.52  3     300.6   +551.51   alpha_reversal
2022-01-12  140.34   2022-01-13  137.38  1     286.1   -846.60   alpha_reversal
2022-01-24  129.80   2022-01-25  125.83  1     241.5   -957.57   alpha_reversal
2022-01-26  128.24   2022-01-27  127.88  1     227.5   -80.58    alpha_reversal
2022-02-10  137.55   2022-02-15  135.42  3     199.7   -425.69   alpha_reversal
2022-03-04  130.89   2022-03-18  134.94  10    221.0   +895.23   max_holding
2022-03-22  138.79   2022-03-28  140.23  4     217.9   +312.64   alpha_reversal
2022-03-31  138.00   2022-04-04  141.73  2     263.2   +982.47   alpha_reversal
2022-04-06  135.50   2022-04-11  127.70  3     265.3   -2067.80  stop_loss
2022-04-12  126.73   2022-04-19  128.88  4     252.6   +542.60   alpha_reversal
2022-05-03  116.43   2022-05-06  114.74  3     221.2   -373.86   alpha_reversal
2022-05-20  108.07   2022-06-02  116.60  8     212.6   +1814.03  alpha_reversal
2022-06-10  110.31   2022-06-13  105.47  1     224.4   -1085.69  alpha_reversal
2022-06-30  108.12   2022-07-11  114.67  6     222.5   +1456.82  alpha_reversal
2022-07-13  110.50   2022-07-26  104.11  9     228.9   -1462.36  trailing_stop
2022-08-02  114.24   2022-08-03  117.05  1     224.0   +629.44   alpha_reversal
2022-08-08  116.40   2022-08-15  121.02  5     245.7   +1135.64  alpha_reversal
2022-08-17  118.63   2022-08-19  116.19  2     286.8   -699.20   alpha_reversal
2022-08-24  112.81   2022-08-29  108.47  3     307.7   -1337.34  trailing_stop
2022-08-31  107.39   2022-09-02  106.91  2     292.7   -138.79   alpha_reversal
2022-09-09  109.80   2022-09-13  103.41  2     301.8   -1926.63  stop_loss
2022-09-14  104.19   2022-09-21  98.42   5     290.2   -1675.56  stop_loss
2022-10-04  100.86   2022-10-13  98.20   7     285.9   -760.06   alpha_reversal
2022-10-18  99.99    2022-10-24  101.63  4     266.9   +436.32   alpha_reversal
2022-10-31  93.78    2022-11-02  86.21   2     246.3   -1863.94  stop_loss
2022-11-03  82.79    2022-11-17  97.50   10    230.6   +3394.50  alpha_reversal
2022-11-21  94.86    2022-12-06  96.14   10    264.7   +336.95   max_holding
2022-12-09  92.12    2022-12-19  87.67   6     316.6   -1407.08  trailing_stop
2022-12-28  85.36    2023-01-09  87.25   7     318.7   +604.58   alpha_reversal
2023-01-10  87.74    2023-01-19  92.24   6     322.9   +1453.75  alpha_reversal
2023-01-26  96.77    2023-02-03  103.87  6     315.0   +2236.64  alpha_reversal
2023-02-09  94.28    2023-02-24  88.36   10    223.2   -1322.09  trailing_stop
2023-03-28  100.25   2023-04-04  103.81  5     303.1   +1078.33  alpha_reversal
2023-04-11  104.54   2023-04-14  107.92  3     321.7   +1088.81  alpha_reversal
2023-04-18  103.70   2023-04-25  102.95  5     322.5   -241.24   alpha_reversal
2023-05-01  106.37   2023-05-04  103.78  3     338.3   -877.62   alpha_reversal
2023-06-08  121.20   2023-06-12  122.57  2     285.9   +390.46   alpha_reversal
2023-06-15  124.13   2023-06-21  119.50  3     309.3   -1430.30  stop_loss
2023-06-22  122.20   2023-06-23  121.28  1     298.5   -276.12   alpha_reversal
2023-06-27  117.42   2023-06-29  118.06  2     296.6   +191.58   alpha_reversal
2023-06-30  118.78   2023-07-05  120.69  2     309.0   +591.31   alpha_reversal
2023-07-07  118.56   2023-07-14  124.33  5     323.5   +1866.37  alpha_reversal
2023-08-07  130.52   2023-08-11  128.43  4     274.2   -571.24   alpha_reversal
2023-08-14  130.32   2023-08-15  128.65  1     300.3   -500.50   alpha_reversal
2023-09-05  134.72   2023-09-06  133.29  1     302.4   -433.38   alpha_reversal
2023-09-07  134.22   2023-09-08  135.19  1     305.0   +297.66   alpha_reversal
2023-09-13  135.66   2023-09-14  136.90  1     332.6   +413.25   alpha_reversal
2023-09-18  137.15   2023-09-19  136.84  1     342.9   -104.78   alpha_reversal
2023-09-21  129.44   2023-10-03  131.28  8     325.3   +599.63   alpha_reversal
2023-10-11  139.47   2023-10-17  138.51  4     309.8   -298.05   alpha_reversal
2023-10-30  123.50   2023-11-07  129.83  6     247.3   +1565.54  alpha_reversal
2023-11-14  132.59   2023-11-22  137.29  6     311.0   +1460.22  alpha_reversal
2023-11-27  135.36   2023-11-29  133.82  2     342.6   -528.68   alpha_reversal
2023-11-30  131.51   2023-12-08  133.82  6     328.5   +757.83   alpha_reversal
2023-12-11  132.26   2023-12-21  139.20  8     285.9   +1983.30  alpha_reversal
2023-12-28  139.15   2024-01-04  135.20  4     339.0   -1337.57  alpha_reversal
2024-01-05  134.68   2024-01-10  141.04  3     349.2   +2220.68  alpha_reversal
2024-01-23  145.91   2024-01-25  150.55  2     348.9   +1619.85  alpha_reversal
2024-02-07  144.42   2024-02-09  147.70  2     294.2   +966.62   alpha_reversal
2024-02-13  144.02   2024-02-14  144.67  1     309.9   +201.17   alpha_reversal
2024-02-16  139.44   2024-02-26  136.37  5     299.9   -918.77   trailing_stop
2024-02-27  137.81   2024-03-04  132.19  4     309.2   -1737.72  stop_loss
2024-03-05  131.65   2024-03-13  138.57  6     303.4   +2101.59  alpha_reversal
2024-03-20  147.59   2024-03-21  146.32  1     271.6   -347.00   alpha_reversal
2024-03-22  149.61   2024-03-25  148.77  1     269.0   -226.92   alpha_reversal
2024-03-26  149.51   2024-03-27  149.56  1     279.1   +13.63    alpha_reversal
2024-05-09  168.65   2024-05-10  167.18  1     230.7   -338.44   alpha_reversal
2024-06-06  175.37   2024-06-12  176.45  4     298.7   +321.94   alpha_reversal
2024-06-20  175.14   2024-06-24  177.87  2     295.9   +805.59   alpha_reversal
2024-06-25  182.82   2024-06-26  182.49  1     285.0   -94.49    alpha_reversal
2024-07-15  185.31   2024-07-17  179.65  2     289.1   -1634.54  stop_loss
2024-07-29  168.42   2024-08-02  165.40  4     228.7   -689.90   alpha_reversal
2024-08-06  157.25   2024-08-08  160.81  2     198.4   +705.40   alpha_reversal
2024-08-09  162.60   2024-08-21  164.60  8     199.9   +400.03   alpha_reversal
2024-08-22  162.73   2024-08-26  164.90  2     234.2   +510.34   alpha_reversal
2024-08-28  161.78   2024-09-04  155.27  4     253.6   -1651.57  stop_loss
2024-09-05  156.21   2024-09-06  149.78  1     243.0   -1562.06  stop_loss
2024-09-09  147.93   2024-09-17  158.33  6     218.6   +2272.83  alpha_reversal
2024-09-26  161.88   2024-09-30  164.82  2     287.7   +845.47   alpha_reversal
2024-10-03  164.99   2024-10-04  166.02  1     292.5   +300.57   alpha_reversal
2024-10-08  163.52   2024-10-11  162.22  3     290.5   -376.57   alpha_reversal
2024-10-17  162.08   2024-10-23  161.76  4     296.7   -92.29    alpha_reversal
2024-11-07  179.80   2024-11-15  171.41  6     237.8   -1994.85  stop_loss
2024-11-21  166.75   2024-12-03  170.27  7     209.7   +738.29   alpha_reversal
2024-12-27  191.97   2025-01-03  190.81  4     192.5   -222.66   alpha_reversal
2025-01-14  188.88   2025-01-27  190.83  8     196.4   +383.00   alpha_reversal
2025-01-28  194.50   2025-02-03  200.20  4     191.2   +1090.61  alpha_reversal
2025-02-10  185.70   2025-02-20  183.62  7     168.7   -351.93   alpha_reversal
2025-02-26  172.02   2025-03-05  172.14  5     207.1   +24.15    alpha_reversal
2025-03-17  163.80   2025-03-26  164.41  7     192.1   +115.87   alpha_reversal
2025-03-27  161.76   2025-03-28  153.72  1     201.7   -1621.71  stop_loss
2025-03-31  154.18   2025-04-01  156.45  1     188.7   +427.57   alpha_reversal
2025-04-03  150.27   2025-04-08  144.13  3     189.2   -1162.77  alpha_reversal
2025-04-16  152.88   2025-04-28  159.97  7     151.9   +1078.51  alpha_reversal
2025-04-30  158.33   2025-05-05  163.56  3     169.3   +885.24   alpha_reversal
2025-05-07  150.93   2025-05-15  163.31  6     162.2   +2007.96  alpha_reversal
2025-06-03  165.69   2025-06-10  178.11  5     189.9   +2359.20  alpha_reversal
2025-06-13  174.36   2025-06-17  175.47  2     212.8   +234.51   alpha_reversal
2025-06-18  173.02   2025-06-20  166.18  1     230.7   -1576.94  stop_loss
2025-06-23  164.90   2025-06-30  175.75  5     209.3   +2269.41  alpha_reversal
2025-07-01  175.53   2025-07-03  179.04  2     209.8   +735.30   alpha_reversal
2025-08-12  202.98   2025-08-15  203.34  3     239.5   +85.14    alpha_reversal
2025-08-19  201.22   2025-08-22  205.52  3     237.8   +1023.87  alpha_reversal
2025-09-03  230.26   2025-09-04  231.66  1     176.0   +247.30   alpha_reversal
2025-09-23  251.44   2025-09-29  243.60  4     185.8   -1457.67  alpha_reversal
2025-09-30  242.89   2025-10-03  244.89  3     179.3   +359.15   alpha_reversal
2025-10-06  250.22   2025-10-09  241.08  3     176.7   -1614.28  stop_loss
2025-10-10  236.37   2025-10-15  250.56  3     176.7   +2508.63  alpha_reversal
2025-10-22  251.47   2025-10-23  252.61  1     157.2   +178.56   alpha_reversal
2025-11-13  278.33   2025-11-18  283.75  3     127.7   +692.25   alpha_reversal
2025-11-19  292.56   2025-11-20  288.91  1     106.2   -387.22   alpha_reversal
2025-12-04  317.35   2025-12-17  296.36  9     103.4   -2169.76  stop_loss
2025-12-26  313.45   2025-12-30  313.47  2     129.2   +3.40     alpha_reversal
2026-01-02  315.09   2026-01-08  325.05  4     136.5   +1359.52  alpha_reversal
2026-01-20  321.93   2026-01-28  335.61  6     130.8   +1788.55  alpha_reversal
2026-02-11  310.90   2026-02-23  311.12  7     98.0    +21.43    alpha_reversal
2026-02-25  312.84   2026-02-26  307.01  1     113.4   -660.60   alpha_reversal
2026-03-03  303.52   2026-03-10  306.89  5     109.6   +369.25   alpha_reversal
2026-03-16  305.71   2026-03-17  310.76  1     129.0   +651.49   alpha_reversal
2026-03-19  307.28   2026-03-24  290.29  3     136.7   -2322.98  stop_loss

**Best 3 trades:**

- 2019-07-29: P&L = **+4673.25** (alpha_reversal)
- 2022-11-17: P&L = **+3394.50** (alpha_reversal)
- 2018-01-22: P&L = **+2654.52** (alpha_reversal)

**Worst 3 trades:**

- 2022-01-05: P&L = **-2841.89** (stop_loss)
- 2018-10-10: P&L = **-2413.98** (stop_loss)
- 2026-03-24: P&L = **-2322.98** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  101,166.00
2017-03-23  99,964.94
2017-09-20  98,455.24
2018-03-21  103,935.37
2018-09-18  110,542.55
2019-03-20  113,147.76
2019-09-17  123,851.12
2020-03-17  123,155.34
2020-09-14  130,909.76
2021-03-15  130,654.07
2021-09-10  130,475.01
2022-03-10  133,354.93
2022-09-08  133,858.59
2023-03-09  133,365.98
2023-09-07  134,221.99
2024-03-07  141,897.24
2024-09-05  141,259.30
2025-03-07  142,332.84
2025-09-05  149,441.29
2026-03-06  149,492.31

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.05%
2017-03-23  -1.44%
2017-09-20  -2.93%
2018-03-21  -2.70%
2018-09-18  -1.62%
2019-03-20  0.00%
2019-09-17  -0.02%
2020-03-17  -4.43%
2020-09-14  -1.12%
2021-03-15  -1.31%
2021-09-10  -1.44%
2022-03-10  -3.51%
2022-09-08  -3.15%
2023-03-09  -3.50%
2023-09-07  -2.88%
2024-03-07  -2.21%
2024-09-05  -2.65%
2025-03-07  -1.91%
2025-09-05  -0.01%
2026-03-06  -1.05%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  55.27%
Out-of-Sample (30%)  2023-03-24  2026-03-27  20.04%

#### Return Distribution

Return Bin          Count
-1.947% to -1.331%  8
-1.331% to -0.715%  44
-0.715% to -0.100%  340
-0.100% to 0.516%   1981
0.516% to 1.132%    131
1.132% to 1.748%    10
1.748% to 2.363%    1
2.363% to 2.979%    0
2.979% to 3.595%    0
3.595% to 4.211%    1

### GS — AlphaCombined

**Net Return (after slippage):** -4.80%  **vs SPY (exposure-adj): -79.15%** (underperform)  
**Gross Return (pre-cost):** 12.42%  
**Total Slippage Cost:** $17,218.66  
**Trade Count:** 306  
**Win Rate:** 50.3%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size   P&L       Exit Reason
2016-05-06  129.47   2016-05-19  125.83  9     236.0  -857.86   alpha_reversal
2016-05-20  125.93   2016-05-24  128.18  2     222.3  +500.92   alpha_reversal
2016-06-03  127.40   2016-06-10  122.42  5     232.8  -1158.30  trailing_stop
2016-06-13  121.64   2016-06-22  120.99  7     251.7  -161.89   alpha_reversal
2016-06-27  114.17   2016-07-08  122.82  8     210.1  +1817.13  alpha_reversal
2016-07-22  131.28   2016-07-25  131.09  1     252.2  -45.56    alpha_reversal
2016-07-28  131.37   2016-08-02  127.46  3     281.1  -1099.93  stop_loss
2016-08-05  132.65   2016-08-09  133.49  2     287.8  +241.04   alpha_reversal
2016-08-15  135.48   2016-08-17  135.31  2     318.2  -54.95    alpha_reversal
2016-08-24  135.28   2016-08-26  135.55  2     350.8  +97.15    alpha_reversal
2016-09-19  136.55   2016-09-21  136.95  2     290.2  +115.94   alpha_reversal
2016-09-22  138.04   2016-09-26  132.40  2     302.1  -1703.33  stop_loss
2016-09-28  134.29   2016-09-29  130.33  1     282.7  -1118.83  stop_loss
2016-09-30  132.50   2016-10-03  132.07  1     248.2  -106.42   alpha_reversal
2016-10-04  133.32   2016-10-07  139.25  3     257.5  +1527.82  alpha_reversal
2016-11-03  144.77   2016-11-08  149.16  3     299.9  +1317.18  alpha_reversal
2016-11-28  172.82   2016-11-30  180.36  2     204.2  +1540.16  alpha_reversal
2016-12-19  197.02   2016-12-20  199.94  1     163.9  +477.51   alpha_reversal
2016-12-22  197.89   2016-12-27  198.68  2     174.3  +137.55   alpha_reversal
2016-12-30  197.34   2017-01-03  198.69  1     191.1  +257.92   alpha_reversal
2017-01-06  201.83   2017-01-17  193.89  6     180.0  -1428.97  stop_loss
2017-01-18  193.08   2017-01-31  188.61  9     167.2  -748.03   trailing_stop
2017-02-01  190.10   2017-02-06  197.38  3     178.8  +1300.99  alpha_reversal
2017-02-08  195.92   2017-02-10  199.63  2     175.5  +651.53   alpha_reversal
2017-02-17  206.34   2017-02-21  207.07  1     191.3  +138.27   alpha_reversal
2017-02-23  207.01   2017-02-28  204.56  3     210.0  -515.41   alpha_reversal
2017-03-08  206.77   2017-03-17  201.16  7     208.1  -1167.16  stop_loss
2017-03-20  200.07   2017-03-21  192.14  1     207.2  -1644.31  stop_loss
2017-03-22  190.93   2017-04-05  187.73  10    183.3  -585.33   max_holding
2017-04-06  188.92   2017-04-11  187.80  3     179.7  -201.16   alpha_reversal
2017-04-13  184.52   2017-04-18  177.78  2     190.1  -1281.60  stop_loss
2017-04-19  176.90   2017-05-03  186.62  10    166.1  +1615.55  max_holding
2017-05-09  184.89   2017-05-17  176.24  6     198.7  -1718.55  stop_loss
2017-05-18  177.78   2017-05-25  183.45  5     184.6  +1046.99  alpha_reversal
2017-05-26  184.70   2017-05-30  180.72  1     192.1  -763.88   alpha_reversal
2017-06-01  178.26   2017-06-12  183.61  7     172.3  +922.19   alpha_reversal
2017-06-21  184.46   2017-06-30  183.60  7     191.5  -164.10   alpha_reversal
2017-07-10  187.23   2017-07-11  187.78  1     182.5  +99.36    alpha_reversal
2017-07-17  190.07   2017-07-18  184.77  1     193.1  -1024.24  stop_loss
2017-07-19  184.77   2017-07-31  186.44  8     185.3  +308.82   alpha_reversal
2017-08-03  185.70   2017-08-07  192.72  2     228.3  +1601.84  alpha_reversal
2017-08-14  188.49   2017-08-16  186.67  2     203.3  -370.95   alpha_reversal
2017-08-17  183.57   2017-08-21  182.68  2     200.5  -178.06   alpha_reversal
2017-08-22  185.36   2017-08-25  184.07  3     207.5  -267.44   alpha_reversal
2017-08-28  182.68   2017-09-05  180.81  5     216.2  -405.50   trailing_stop
2017-09-06  182.04   2017-09-13  188.10  5     197.5  +1195.69  alpha_reversal
2017-09-18  189.28   2017-09-19  190.05  1     203.2  +155.94   alpha_reversal
2017-09-25  191.55   2017-09-29  196.92  4     219.4  +1178.45  alpha_reversal
2017-10-05  204.69   2017-10-06  204.25  1     216.7  -95.82    alpha_reversal
2017-10-11  201.65   2017-10-17  196.01  4     233.6  -1317.88  stop_loss
2017-10-18  201.34   2017-10-19  199.25  1     189.2  -396.63   alpha_reversal
2017-10-24  203.68   2017-11-07  199.10  10    186.7  -855.82   max_holding
2017-11-08  200.69   2017-11-16  198.73  6     186.9  -366.62   alpha_reversal
2017-11-17  198.01   2017-11-29  201.01  7     193.8  +582.76   alpha_reversal
2017-12-12  215.03   2017-12-13  212.84  1     173.0  -379.75   alpha_reversal
2017-12-14  213.20   2017-12-18  216.55  2     172.3  +578.07   alpha_reversal
2017-12-20  212.95   2017-12-29  212.17  6     170.1  -131.89   alpha_reversal
2018-01-04  214.33   2018-01-17  211.25  8     181.1  -557.11   alpha_reversal
2018-01-18  209.44   2018-01-23  216.61  3     165.9  +1190.99  alpha_reversal
2018-01-24  221.71   2018-01-25  224.06  1     156.8  +368.02   alpha_reversal
2018-02-20  221.05   2018-03-01  214.46  7     108.0  -712.44   trailing_stop
2018-03-08  222.89   2018-03-09  226.14  1     117.4  +381.96   alpha_reversal
2018-03-14  221.29   2018-03-16  223.49  2     120.6  +265.96   alpha_reversal
2018-03-20  220.25   2018-03-21  218.69  1     128.5  -200.42   alpha_reversal
2018-03-23  205.24   2018-03-29  210.35  4     119.3  +608.75   alpha_reversal
2018-04-02  206.99   2018-04-06  208.77  4     111.1  +197.12   alpha_reversal
2018-04-13  214.17   2018-04-24  202.52  7     108.1  -1258.66  stop_loss
2018-04-26  200.92   2018-05-10  203.31  10    119.9  +287.34   max_holding
2018-05-14  204.12   2018-05-15  201.74  1     150.1  -355.82   alpha_reversal
2018-05-17  200.09   2018-05-29  189.61  7     165.1  -1730.10  stop_loss
2018-05-30  192.45   2018-06-12  194.97  9     171.9  +433.81   alpha_reversal
2018-06-13  196.37   2018-06-14  195.83  1     193.2  -104.94   alpha_reversal
2018-06-18  194.32   2018-06-25  185.68  5     199.7  -1726.17  stop_loss
2018-07-02  187.49   2018-07-03  184.71  1     176.5  -492.26   alpha_reversal
2018-07-05  185.38   2018-07-17  193.62  8     180.2  +1485.50  alpha_reversal
2018-07-18  194.20   2018-07-19  192.46  1     183.0  -317.96   alpha_reversal
2018-08-06  198.14   2018-08-10  192.44  4     222.5  -1266.82  stop_loss
2018-08-13  190.52   2018-08-20  197.61  5     211.3  +1499.18  alpha_reversal
2018-08-24  197.45   2018-08-28  203.14  2     226.7  +1290.04  alpha_reversal
2018-08-29  203.40   2018-09-06  197.21  5     212.5  -1315.87  stop_loss
2018-09-10  195.40   2018-09-20  199.63  8     214.7  +907.14   alpha_reversal
2018-09-24  196.24   2018-09-27  191.51  3     213.3  -1009.05  stop_loss
2018-09-28  188.94   2018-10-10  180.70  8     213.9  -1762.30  trailing_stop
2018-10-11  179.45   2018-10-15  180.98  2     179.2  +274.86   alpha_reversal
2018-10-26  178.93   2018-10-30  184.39  2     133.9  +731.49   alpha_reversal
2018-11-06  192.28   2018-11-09  187.23  3     139.1  -702.79   alpha_reversal
2018-11-13  172.77   2018-11-20  160.90  5     114.4  -1358.60  stop_loss
2018-11-21  162.28   2018-12-04  155.61  8     117.1  -781.02   trailing_stop
2018-12-06  155.74   2018-12-14  145.87  6     123.7  -1220.76  stop_loss
2018-12-17  142.14   2018-12-24  132.01  5     125.7  -1273.27  stop_loss
2018-12-28  137.92   2019-01-03  143.12  3     113.6  +590.23   alpha_reversal
2019-01-09  149.29   2019-01-10  148.60  1     122.6  -85.23    alpha_reversal
2019-01-11  149.68   2019-01-15  151.90  2     129.1  +286.12   alpha_reversal
2019-02-01  166.27   2019-02-15  167.60  10    135.3  +178.98   max_holding
2019-02-21  166.12   2019-02-28  166.75  5     159.0  +99.48    alpha_reversal
2019-03-07  163.74   2019-03-11  166.13  2     168.1  +400.98   alpha_reversal
2019-03-20  165.06   2019-04-03  170.27  10    169.2  +882.47   alpha_reversal
2019-04-16  171.45   2019-05-01  173.55  10    168.6  +355.27   max_holding
2019-05-09  171.27   2019-05-10  171.28  1     177.6  +2.46     alpha_reversal
2019-05-15  166.82   2019-05-28  160.59  8     171.4  -1068.60  trailing_stop
2019-05-29  160.45   2019-05-31  155.40  2     185.6  -938.48   stop_loss
2019-06-05  160.79   2019-06-07  161.63  2     177.5  +150.08   alpha_reversal
2019-06-10  165.63   2019-06-12  161.98  2     174.4  -636.99   alpha_reversal
2019-06-13  163.35   2019-06-14  163.21  1     180.4  -26.61    alpha_reversal
2019-06-18  166.37   2019-06-24  168.17  4     178.8  +322.75   alpha_reversal
2019-07-03  175.80   2019-07-05  177.03  1     187.1  +230.58   alpha_reversal
2019-07-11  180.33   2019-07-12  182.18  1     181.0  +333.98   alpha_reversal
2019-07-30  188.91   2019-07-31  187.45  1     190.1  -277.40   alpha_reversal
2019-08-07  175.55   2019-08-14  166.53  5     145.5  -1312.54  stop_loss
2019-08-15  167.39   2019-08-16  169.81  1     134.8  +326.75   alpha_reversal
2019-08-21  171.23   2019-08-22  172.37  1     145.7  +165.98   alpha_reversal
2019-08-28  171.01   2019-08-29  174.32  1     147.0  +487.72   alpha_reversal
2019-09-04  173.20   2019-09-09  181.64  3     149.8  +1264.04  alpha_reversal
2019-09-18  186.38   2019-09-19  184.43  1     153.7  -300.80   alpha_reversal
2019-09-20  183.52   2019-09-30  177.57  6     161.0  -957.04   stop_loss
2019-10-01  174.03   2019-10-09  168.68  6     165.1  -883.56   alpha_reversal
2019-10-10  171.61   2019-10-14  176.36  2     165.6  +787.63   alpha_reversal
2019-10-25  183.94   2019-11-08  191.01  10    162.8  +1151.03  alpha_reversal
2019-11-11  188.06   2019-11-15  188.73  4     174.3  +116.73   alpha_reversal
2019-11-19  188.93   2019-11-22  188.75  3     200.8  -34.50    alpha_reversal
2019-12-02  187.92   2019-12-03  182.89  1     218.2  -1097.45  stop_loss
2019-12-04  186.45   2019-12-06  193.55  2     191.4  +1358.78  alpha_reversal
2019-12-20  197.67   2019-12-24  198.12  2     192.9  +86.71    alpha_reversal
2019-12-27  199.16   2020-01-02  201.92  3     219.7  +605.50   alpha_reversal
2020-01-16  215.62   2020-01-17  214.96  1     175.0  -114.60   alpha_reversal
2020-01-30  210.79   2020-02-03  205.96  2     150.8  -729.01   alpha_reversal
2020-02-07  205.50   2020-02-21  198.73  9     147.4  -997.61   stop_loss
2020-02-24  193.88   2020-02-25  187.52  1     145.7  -926.53   stop_loss
2020-02-26  186.31   2020-02-27  177.25  1     133.5  -1210.25  stop_loss
2020-02-28  174.41   2020-03-06  167.20  5     116.4  -839.89   trailing_stop
2020-03-09  150.12   2020-03-11  149.03  2     81.8   -89.74    alpha_reversal
2020-03-13  153.91   2020-03-16  134.09  1     60.6   -1201.51  stop_loss
2020-03-19  129.86   2020-03-26  143.74  5     47.4   +657.62   alpha_reversal
2020-03-27  137.55   2020-03-31  134.03  2     48.9   -172.35   alpha_reversal
2020-04-01  126.22   2020-04-08  153.42  5     53.0   +1441.52  alpha_reversal
2020-04-24  153.76   2020-04-27  159.13  1     74.2   +397.76   alpha_reversal
2020-05-01  153.85   2020-05-15  149.01  10    80.3   -388.69   max_holding
2020-05-18  158.00   2020-05-19  154.18  1     90.3   -345.55   alpha_reversal
2020-05-20  157.62   2020-05-27  181.77  4     93.7   +2262.77  alpha_reversal
2020-06-04  187.78   2020-06-11  169.36  5     89.1   -1642.40  trailing_stop
2020-06-18  178.19   2020-06-23  178.65  3     81.9   +38.02    alpha_reversal
2020-06-29  169.12   2020-07-13  182.23  9     81.3   +1065.12  alpha_reversal
2020-07-22  180.07   2020-08-05  178.42  10    97.6   -161.13   max_holding
2020-08-07  182.06   2020-08-10  182.66  1     130.6  +78.94    alpha_reversal
2020-08-17  177.51   2020-08-24  180.88  5     132.9  +448.07   alpha_reversal
2020-08-31  180.17   2020-09-08  177.71  5     146.5  -360.07   alpha_reversal
2020-09-09  177.84   2020-09-14  176.71  3     126.8  -142.94   alpha_reversal
2020-09-15  174.13   2020-09-21  170.27  4     135.3  -522.13   alpha_reversal
2020-09-22  168.52   2020-09-25  171.10  3     123.8  +320.24   alpha_reversal
2020-09-28  175.07   2020-10-01  174.26  3     111.9  -90.22    alpha_reversal
2020-10-02  175.80   2020-10-05  177.12  1     115.8  +152.38   alpha_reversal
2020-10-06  176.85   2020-10-15  183.08  7     118.8  +741.15   alpha_reversal
2020-10-26  176.89   2020-10-28  166.56  2     131.4  -1357.47  stop_loss
2020-10-30  166.25   2020-11-04  173.67  3     126.1  +935.57   alpha_reversal
2020-11-05  178.49   2020-11-10  190.87  3     121.0  +1497.75  alpha_reversal
2020-11-23  201.24   2020-12-08  210.77  10    121.6  +1158.53  max_holding
2020-12-21  227.23   2020-12-22  220.70  1     112.4  -734.29   alpha_reversal
2020-12-30  229.41   2020-12-31  232.72  1     114.1  +376.78   alpha_reversal
2021-01-26  249.14   2021-02-02  253.24  5     86.9   +356.43   alpha_reversal
2021-02-08  265.40   2021-02-10  268.52  2     89.4   +278.46   alpha_reversal
2021-02-12  270.86   2021-02-16  275.29  1     96.0   +425.96   alpha_reversal
2021-03-08  296.66   2021-03-15  306.58  5     72.6   +719.77   alpha_reversal
2021-03-16  303.34   2021-03-19  304.94  3     72.7   +116.33   alpha_reversal
2021-03-23  294.51   2021-03-31  289.70  6     74.2   -357.38   alpha_reversal
2021-04-01  290.85   2021-04-07  289.30  3     75.6   -116.99   alpha_reversal
2021-04-08  293.96   2021-04-13  290.30  3     79.8   -291.60   alpha_reversal
2021-04-23  301.24   2021-04-28  308.40  3     79.4   +568.68   alpha_reversal
2021-05-17  327.79   2021-05-18  322.98  1     80.3   -386.19   alpha_reversal
2021-05-20  319.38   2021-05-21  324.48  1     78.6   +401.11   alpha_reversal
2021-06-10  333.07   2021-06-17  321.35  5     83.8   -982.40   stop_loss
2021-06-18  310.70   2021-06-24  327.79  4     76.0   +1298.55  alpha_reversal
2021-06-29  331.89   2021-06-30  337.37  1     83.0   +454.99   alpha_reversal
2021-07-06  329.44   2021-07-13  334.22  5     86.1   +411.92   alpha_reversal
2021-07-20  324.89   2021-07-26  334.15  4     67.0   +619.82   alpha_reversal
2021-07-29  336.29   2021-07-30  333.24  1     81.8   -249.09   alpha_reversal
2021-08-02  336.86   2021-08-04  335.89  2     80.4   -77.71    alpha_reversal
2021-08-17  360.71   2021-08-31  369.37  10    78.9   +682.63   max_holding
2021-09-07  367.25   2021-09-16  354.33  7     95.3   -1231.18  stop_loss
2021-09-17  350.37   2021-09-20  337.76  1     87.2   -1099.73  stop_loss
2021-09-21  336.39   2021-09-28  347.92  5     79.4   +915.53   alpha_reversal
2021-09-30  338.35   2021-10-07  348.92  5     74.9   +791.18   alpha_reversal
2021-11-01  372.96   2021-11-09  362.08  6     85.4   -929.41   trailing_stop
2021-11-10  357.23   2021-11-12  361.68  2     78.3   +348.95   alpha_reversal
2021-11-16  360.78   2021-11-18  349.59  2     87.4   -978.18   stop_loss
2021-11-19  346.73   2021-11-23  362.96  2     82.3   +1336.11  alpha_reversal
2021-11-30  341.00   2021-12-03  343.68  3     71.2   +190.57   alpha_reversal
2021-12-09  356.74   2021-12-15  350.12  4     69.7   -461.35   alpha_reversal
2021-12-20  334.36   2022-01-04  365.90  10    64.6   +2036.27  alpha_reversal
2022-01-25  307.31   2022-01-27  306.23  2     58.2   -62.90    alpha_reversal
2022-02-03  322.90   2022-02-08  332.33  3     60.9   +573.96   alpha_reversal
2022-02-15  327.46   2022-02-18  310.73  3     69.9   -1169.88  stop_loss
2022-02-24  306.09   2022-03-07  290.75  7     69.8   -1070.31  trailing_stop
2022-03-24  304.31   2022-04-06  285.66  9     74.5   -1389.33  trailing_stop
2022-04-07  284.32   2022-04-11  289.73  2     84.2   +454.85   alpha_reversal
2022-05-03  284.79   2022-05-10  275.55  5     71.8   -663.46   alpha_reversal
2022-05-17  283.26   2022-05-18  277.05  1     68.5   -425.06   alpha_reversal
2022-05-19  278.94   2022-05-20  277.12  1     70.8   -128.98   alpha_reversal
2022-05-25  285.00   2022-05-26  292.32  1     67.6   +494.49   alpha_reversal
2022-05-27  297.38   2022-06-01  292.49  2     70.0   -342.25   alpha_reversal
2022-06-06  291.86   2022-06-07  292.18  1     74.1   +24.12    alpha_reversal
2022-06-08  286.55   2022-06-10  260.84  2     76.5   -1967.08  stop_loss
2022-06-13  257.99   2022-06-17  254.27  4     69.6   -259.05   alpha_reversal
2022-06-21  259.35   2022-06-29  275.62  6     69.6   +1132.72  alpha_reversal
2022-07-07  271.91   2022-07-14  255.90  5     74.4   -1190.35  stop_loss
2022-08-02  298.57   2022-08-09  305.91  5     79.9   +587.32   alpha_reversal
2022-08-24  311.37   2022-08-29  303.46  3     96.0   -759.51   alpha_reversal
2022-09-15  304.25   2022-09-16  298.69  1     84.1   -467.80   alpha_reversal
2022-09-19  301.54   2022-09-22  286.52  3     80.1   -1203.22  stop_loss
2022-09-23  277.05   2022-10-05  282.93  8     71.2   +418.74   alpha_reversal
2022-10-07  276.23   2022-10-13  281.17  4     69.0   +340.32   alpha_reversal
2022-11-21  348.83   2022-11-22  351.49  1     80.0   +212.23   alpha_reversal
2022-12-01  354.45   2022-12-05  342.43  2     86.2   -1036.08  stop_loss
2022-12-07  332.38   2022-12-12  334.72  3     81.4   +190.59   alpha_reversal
2022-12-15  323.07   2022-12-30  316.48  10    74.8   -493.04   max_holding
2023-01-03  319.73   2023-01-04  320.46  1     93.5   +67.79    alpha_reversal
2023-01-23  322.43   2023-01-31  337.15  6     67.0   +985.74   alpha_reversal
2023-02-15  345.70   2023-02-21  332.83  3     77.2   -993.30   stop_loss
2023-03-01  322.03   2023-03-09  317.54  6     80.2   -359.88   alpha_reversal
2023-03-13  293.46   2023-03-27  295.68  10    69.1   +153.20   max_holding
2023-03-29  298.30   2023-03-30  298.10  1     68.9   -14.21    alpha_reversal
2023-04-04  300.10   2023-04-10  301.64  3     75.6   +116.71   alpha_reversal
2023-05-02  310.07   2023-05-04  298.21  2     91.6   -1086.91  stop_loss
2023-05-05  304.16   2023-05-19  302.75  10    88.2   -124.80   max_holding
2023-05-23  300.96   2023-05-30  307.09  4     99.5   +610.99   alpha_reversal
2023-06-01  296.53   2023-06-05  300.99  2     94.9   +423.83   alpha_reversal
2023-06-06  306.27   2023-06-08  313.77  2     96.2   +721.02   alpha_reversal
2023-06-15  318.40   2023-06-16  316.43  1     93.6   -184.76   alpha_reversal
2023-06-20  309.95   2023-06-22  298.91  2     93.3   -1029.63  stop_loss
2023-06-23  294.94   2023-06-30  301.68  5     94.8   +638.36   alpha_reversal
2023-07-03  306.10   2023-07-05  299.35  1     97.6   -658.72   alpha_reversal
2023-07-31  333.52   2023-08-02  328.39  2     87.3   -447.71   alpha_reversal
2023-08-10  319.02   2023-08-16  307.82  4     85.6   -959.01   stop_loss
2023-08-17  306.95   2023-08-31  309.07  10    87.6   +185.80   max_holding
2023-09-01  309.39   2023-09-13  315.93  7     98.5   +643.54   alpha_reversal
2023-09-28  307.32   2023-10-03  288.71  3     99.1   -1843.54  stop_loss
2023-10-04  291.63   2023-10-17  291.76  9     89.9   +12.05    alpha_reversal
2023-10-24  282.73   2023-10-27  273.42  3     92.9   -864.39   alpha_reversal
2023-11-17  320.54   2023-11-20  320.14  1     88.6   -35.02    alpha_reversal
2023-11-28  319.08   2023-12-04  332.22  4     104.5  +1373.70  alpha_reversal
2024-01-02  369.96   2024-01-05  367.45  3     95.2   -238.69   alpha_reversal
2024-01-24  361.48   2024-01-31  365.14  5     79.9   +292.70   alpha_reversal
2024-02-02  369.54   2024-02-07  367.66  3     76.9   -144.53   alpha_reversal
2024-02-09  366.11   2024-02-22  371.28  8     86.0   +444.89   alpha_reversal
2024-02-27  372.35   2024-02-28  376.51  1     79.7   +331.66   alpha_reversal
2024-03-01  372.39   2024-03-15  370.79  10    80.0   -127.78   max_holding
2024-03-18  368.81   2024-03-20  379.66  2     80.2   +870.45   alpha_reversal
2024-03-21  396.96   2024-03-22  389.57  1     72.2   -533.55   alpha_reversal
2024-05-13  435.20   2024-05-20  443.31  5     80.1   +649.75   alpha_reversal
2024-05-21  451.37   2024-05-22  442.77  1     80.8   -693.92   alpha_reversal
2024-05-23  439.60   2024-06-05  444.78  8     78.3   +405.32   alpha_reversal
2024-06-07  439.14   2024-06-17  433.70  6     78.4   -425.84   alpha_reversal
2024-06-24  446.07   2024-06-25  440.64  1     71.3   -386.83   alpha_reversal
2024-06-26  440.05   2024-07-01  446.69  3     73.3   +486.34   alpha_reversal
2024-07-25  474.66   2024-07-26  480.76  1     57.6   +351.63   alpha_reversal
2024-08-05  443.10   2024-08-12  467.73  5     40.9   +1006.12  alpha_reversal
2024-08-22  480.49   2024-08-23  490.77  1     53.8   +553.06   alpha_reversal
2024-08-27  489.67   2024-09-03  472.39  4     56.3   -973.02   stop_loss
2024-09-04  476.43   2024-09-10  452.69  4     52.8   -1252.64  stop_loss
2024-09-11  457.52   2024-09-20  483.03  7     44.2   +1127.82  alpha_reversal
2024-09-23  483.00   2024-09-25  475.96  2     50.0   -352.03   alpha_reversal
2024-10-01  475.97   2024-10-14  506.59  9     54.4   +1664.45  alpha_reversal
2024-10-24  509.31   2024-10-30  508.19  4     61.1   -68.14    alpha_reversal
2024-11-08  572.19   2024-11-11  583.72  1     41.1   +473.80   alpha_reversal
2024-11-19  564.54   2024-11-27  586.72  6     44.5   +987.64   alpha_reversal
2024-12-03  587.54   2024-12-04  583.08  1     49.2   -219.34   alpha_reversal
2024-12-05  582.38   2024-12-09  578.61  2     51.1   -192.79   alpha_reversal
2024-12-10  570.90   2024-12-18  535.89  6     53.3   -1866.58  stop_loss
2024-12-19  540.61   2024-12-20  551.32  1     45.7   +489.82   alpha_reversal
2024-12-27  562.26   2024-12-30  558.58  1     48.1   -177.38   alpha_reversal
2025-01-02  561.08   2025-01-03  564.99  1     49.3   +192.58   alpha_reversal
2025-01-27  617.98   2025-02-06  641.04  8     42.8   +987.17   alpha_reversal
2025-02-11  631.61   2025-02-20  625.49  6     46.8   -286.15   trailing_stop
2025-02-21  610.49   2025-02-25  598.86  2     40.1   -466.77   alpha_reversal
2025-02-26  602.85   2025-03-03  592.38  3     38.3   -400.37   alpha_reversal
2025-03-04  569.91   2025-03-10  520.35  4     32.1   -1589.02  stop_loss
2025-03-11  521.16   2025-03-19  545.48  6     28.5   +692.75   alpha_reversal
2025-03-27  548.12   2025-04-02  551.12  4     32.3   +96.81    alpha_reversal
2025-04-04  461.72   2025-04-16  488.43  8     24.7   +660.98   alpha_reversal
2025-04-28  535.85   2025-04-29  537.24  1     25.3   +35.28    alpha_reversal
2025-05-01  543.13   2025-05-12  578.76  7     27.6   +983.83   alpha_reversal
2025-05-20  594.80   2025-05-27  602.63  4     35.5   +277.89   alpha_reversal
2025-05-30  591.79   2025-06-11  613.94  8     38.5   +851.89   alpha_reversal
2025-07-02  705.56   2025-07-03  711.81  1     42.8   +267.69   alpha_reversal
2025-07-09  686.51   2025-07-17  694.27  6     42.9   +332.54   alpha_reversal
2025-07-22  690.30   2025-07-25  717.03  3     45.1   +1205.94  alpha_reversal
2025-07-29  721.42   2025-07-30  718.77  1     46.9   -124.28   alpha_reversal
2025-08-04  715.55   2025-08-05  709.09  1     42.1   -271.99   alpha_reversal
2025-08-08  710.86   2025-08-11  707.45  1     44.3   -151.28   alpha_reversal
2025-08-12  732.65   2025-08-15  718.74  3     42.1   -586.06   alpha_reversal
2025-08-18  720.41   2025-08-19  709.55  1     42.6   -462.44   alpha_reversal
2025-08-22  731.19   2025-08-27  737.38  3     40.4   +250.28   alpha_reversal
2025-09-03  723.87   2025-09-10  761.01  5     44.2   +1640.65  alpha_reversal
2025-09-15  779.56   2025-09-16  776.79  1     39.9   -110.68   alpha_reversal
2025-09-25  787.49   2025-09-26  793.58  1     40.0   +243.36   alpha_reversal
2025-10-01  778.32   2025-10-14  762.18  9     40.0   -645.31   alpha_reversal
2025-10-20  756.33   2025-10-28  783.27  6     30.4   +819.19   alpha_reversal
2025-11-03  778.33   2025-11-05  784.26  2     33.9   +200.91   alpha_reversal
2025-11-17  768.46   2025-11-19  777.24  2     27.8   +244.07   alpha_reversal
2025-11-21  766.95   2025-12-01  801.83  5     26.6   +927.89   alpha_reversal
2025-12-15  885.82   2025-12-16  873.67  1     29.0   -352.16   alpha_reversal
2025-12-29  888.40   2026-01-05  942.53  4     32.8   +1775.70  alpha_reversal
2026-01-26  927.91   2026-01-29  934.26  3     27.0   +171.63   alpha_reversal
2026-02-05  886.63   2026-02-11  938.71  4     22.8   +1188.39  alpha_reversal
2026-02-20  918.33   2026-02-23  886.75  1     21.4   -675.22   alpha_reversal
2026-03-02  862.56   2026-03-04  866.38  2     19.3   +73.66    alpha_reversal
2026-03-06  822.24   2026-03-20  812.72  10    18.9   -180.32   alpha_reversal

**Best 3 trades:**

- 2020-05-27: P&L = **+2262.77** (alpha_reversal)
- 2022-01-04: P&L = **+2036.27** (alpha_reversal)
- 2016-07-08: P&L = **+1817.13** (alpha_reversal)

**Worst 3 trades:**

- 2022-06-10: P&L = **-1967.08** (stop_loss)
- 2024-12-18: P&L = **-1866.58** (stop_loss)
- 2023-10-03: P&L = **-1843.54** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  99,352.03
2017-03-23  98,400.75
2017-09-20  98,295.72
2018-03-21  97,855.69
2018-09-18  94,473.14
2019-03-20  90,309.05
2019-09-17  90,598.41
2020-03-17  85,322.30
2020-09-14  88,521.41
2021-03-15  92,780.14
2021-09-10  94,232.66
2022-03-10  94,040.17
2022-09-08  89,608.59
2023-03-09  87,296.17
2023-09-07  85,078.29
2024-03-07  85,582.74
2024-09-05  86,602.52
2025-03-07  86,717.42
2025-09-05  90,183.39
2026-03-06  95,367.35

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -1.11%
2017-03-23  -3.35%
2017-09-20  -3.46%
2018-03-21  -3.89%
2018-09-18  -7.21%
2019-03-20  -11.30%
2019-09-17  -11.02%
2020-03-17  -16.20%
2020-09-14  -13.06%
2021-03-15  -8.87%
2021-09-10  -7.45%
2022-03-10  -7.64%
2022-09-08  -11.99%
2023-03-09  -14.26%
2023-09-07  -16.44%
2024-03-07  -15.94%
2024-09-05  -14.94%
2025-03-07  -14.83%
2025-09-05  -11.42%
2026-03-06  -6.33%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  3.19%
Out-of-Sample (30%)  2023-03-24  2026-03-27  18.20%

#### Return Distribution

Return Bin          Count
-1.986% to -1.623%  3
-1.623% to -1.260%  12
-1.260% to -0.897%  22
-0.897% to -0.535%  84
-0.535% to -0.172%  252
-0.172% to 0.191%   1767
0.191% to 0.554%    242
0.554% to 0.917%    99
0.917% to 1.280%    28
1.280% to 1.642%    7

### CVX — Momentum

**Net Return (after slippage):** -1.06%  **vs SPY (exposure-adj): +17.44%** (outperform)  
**Gross Return (pre-cost):** 0.33%  
**Total Slippage Cost:** $1,389.65  
**Trade Count:** 37  
**Win Rate:** 32.4%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size   P&L       Exit Reason
2017-05-15  72.78    2017-05-17  71.97   2     647.1  -524.82   ma_exit
2017-05-19  73.29    2017-05-25  72.25   4     651.7  -679.39   ma_exit
2017-06-09  73.21    2017-06-21  71.83   8     678.7  -940.77   trailing_stop
2017-07-26  72.33    2017-08-16  74.18   15    704.2  +1301.36  trailing_stop
2017-09-06  77.69    2017-10-04  81.64   20    693.2  +2732.74  max_holding
2017-11-30  83.47    2017-12-29  87.73   20    694.0  +2957.54  max_holding
2018-01-11  93.00    2018-01-29  90.04   11    617.9  -1828.47  stop_loss
2018-03-09  83.05    2018-03-22  80.47   9     344.6  -887.10   ma_exit
2018-04-18  87.72    2018-05-16  91.70   20    369.6  +1469.14  max_holding
2018-07-27  90.03    2018-08-02  88.36   4     413.9  -690.29   ma_exit
2018-09-21  87.40    2018-10-10  88.37   13    534.4  +519.53   trailing_stop
2019-03-01  89.80    2019-03-25  90.34   16    489.1  +265.35   ma_exit
2019-05-09  89.18    2019-05-23  87.50   10    407.0  -682.66   ma_exit
2019-06-21  92.84    2019-07-17  92.16   17    504.8  -342.90   ma_exit
2019-09-10  91.48    2019-09-26  90.12   12    432.7  -588.04   trailing_stop
2019-11-04  91.27    2019-11-19  88.20   11    431.5  -1324.88  stop_loss
2019-12-20  90.74    2020-01-07  90.18   10    544.5  -305.06   ma_exit
2021-03-10  90.48    2021-03-18  84.51   6     275.1  -1643.54  stop_loss
2021-05-17  90.03    2021-05-19  84.79   2     316.5  -1659.00  stop_loss
2021-09-28  86.15    2021-10-26  95.00   20    342.5  +3033.44  max_holding
2022-01-04  102.38   2022-02-02  114.05  20    357.2  +4169.55  max_holding
2022-02-07  116.81   2022-02-22  112.69  10    249.7  -1030.32  ma_exit
2022-02-25  119.60   2022-03-15  134.71  12    219.6  +3319.86  trailing_stop
2022-05-16  147.40   2022-06-13  143.59  19    145.6  -554.89   trailing_stop
2022-08-23  140.41   2022-09-01  134.68  7     190.2  -1088.76  trailing_stop
2022-09-14  141.52   2022-09-16  135.47  2     193.4  -1169.26  stop_loss
2022-11-14  161.70   2022-11-28  155.64  9     168.1  -1018.53  ma_exit
2023-01-17  157.65   2023-01-30  152.01  9     198.7  -1122.06  trailing_stop
2023-04-03  149.76   2023-04-26  146.12  16    181.6  -661.97   trailing_stop
2023-07-31  145.63   2023-08-15  141.56  11    238.8  -972.92   ma_exit
2023-09-05  149.55   2023-10-02  149.47  19    264.7  -20.56    ma_exit
2024-11-13  148.78   2024-12-04  149.77  14    262.5  +259.60   ma_exit
2025-01-13  147.10   2025-01-24  147.24  8     245.3  +33.55    trailing_stop
2025-03-19  157.06   2025-04-03  149.32  11    203.8  -1577.41  trailing_stop
2025-10-23  153.41   2025-10-30  150.28  5     252.7  -790.66   ma_exit
2026-01-05  162.34   2026-01-06  154.95  1     235.0  -1738.81  stop_loss
2026-01-14  165.70   2026-02-12  180.54  20    183.6  +2724.49  max_holding

**Best 3 trades:**

- 2022-02-02: P&L = **+4169.55** (max_holding)
- 2022-03-15: P&L = **+3319.86** (trailing_stop)
- 2021-10-26: P&L = **+3033.44** (max_holding)

**Worst 3 trades:**

- 2018-01-29: P&L = **-1828.47** (stop_loss)
- 2026-01-06: P&L = **-1738.81** (stop_loss)
- 2021-05-19: P&L = **-1659.00** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  100,000.00
2017-03-23  100,000.00
2017-09-20  101,334.80
2018-03-21  102,959.98
2018-09-18  102,909.95
2019-03-20  104,364.52
2019-09-17  103,311.89
2020-03-17  100,451.29
2020-09-14  100,451.29
2021-03-15  100,188.67
2021-09-10  97,148.75
2022-03-10  109,001.42
2022-09-08  104,997.63
2023-03-09  101,687.79
2023-09-07  100,075.90
2024-03-07  100,032.34
2024-09-05  100,032.34
2025-03-07  100,325.49
2025-09-05  98,748.09
2026-03-06  98,943.11

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  0.00%
2017-03-23  0.00%
2017-09-20  0.00%
2018-03-21  -2.19%
2018-09-18  -2.24%
2019-03-20  -0.85%
2019-09-17  -1.85%
2020-03-17  -4.57%
2020-09-14  -4.57%
2021-03-15  -4.82%
2021-09-10  -7.71%
2022-03-10  0.00%
2022-09-08  -3.69%
2023-03-09  -6.72%
2023-09-07  -8.20%
2024-03-07  -8.24%
2024-09-05  -8.24%
2025-03-07  -7.97%
2025-09-05  -9.42%
2026-03-06  -9.24%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  31.61%
Out-of-Sample (30%)  2023-03-24  2026-03-27  9.27%

#### Return Distribution

Return Bin          Count
-2.002% to -1.616%  2
-1.616% to -1.230%  4
-1.230% to -0.844%  10
-0.844% to -0.457%  38
-0.457% to -0.071%  116
-0.071% to 0.315%   2264
0.315% to 0.702%    55
0.702% to 1.088%    23
1.088% to 1.474%    1
1.474% to 1.861%    3

### JNJ — AlphaCombined

**Net Return (after slippage):** 8.75%  **vs SPY (exposure-adj): -79.86%** (underperform)  
**Gross Return (pre-cost):** 26.18%  
**Total Slippage Cost:** $17,429.89  
**Trade Count:** 340  
**Win Rate:** 53.2%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size   P&L       Exit Reason
2016-05-05  85.78    2016-05-06  85.59   1     780.9  -149.94   alpha_reversal
2016-05-09  86.42    2016-05-10  87.05   1     789.4  +501.13   alpha_reversal
2016-05-18  86.32    2016-05-26  86.32   6     795.5  +1.48     alpha_reversal
2016-05-27  86.53    2016-05-31  86.16   1     843.8  -311.66   alpha_reversal
2016-06-01  86.32    2016-06-02  87.54   1     829.3  +1012.69  alpha_reversal
2016-06-20  89.19    2016-06-23  89.75   3     736.6  +407.42   alpha_reversal
2016-06-24  88.50    2016-06-28  90.40   2     713.8  +1355.79  alpha_reversal
2016-07-08  94.02    2016-07-11  93.99   1     687.6  -22.56    alpha_reversal
2016-07-13  94.14    2016-07-14  94.18   1     731.2  +31.84    alpha_reversal
2016-07-18  94.25    2016-07-20  95.68   2     751.0  +1077.71  alpha_reversal
2016-07-21  95.78    2016-07-22  95.60   1     727.2  -136.34   alpha_reversal
2016-07-25  95.59    2016-08-02  95.46   6     759.5  -95.79    alpha_reversal
2016-08-04  94.80    2016-08-15  93.52   7     839.0  -1073.79  trailing_stop
2016-08-16  92.10    2016-08-18  92.30   2     850.1  +175.25   alpha_reversal
2016-08-19  92.39    2016-09-02  91.84   10    829.1  -459.49   max_holding
2016-09-08  92.05    2016-09-12  91.72   2     944.7  -312.33   alpha_reversal
2016-09-13  90.61    2016-09-21  91.52   6     812.5  +739.43   alpha_reversal
2016-09-26  90.75    2016-09-27  91.76   1     829.6  +844.19   alpha_reversal
2016-10-07  91.87    2016-10-10  92.21   1     770.0  +261.21   alpha_reversal
2016-10-17  91.29    2016-10-18  88.83   1     708.9  -1745.34  stop_loss
2016-10-19  88.29    2016-11-01  88.78   9     624.0  +305.18   alpha_reversal
2016-11-11  91.28    2016-11-21  89.13   6     543.8  -1169.34  stop_loss
2016-11-22  87.46    2016-12-05  86.76   8     576.4  -407.73   alpha_reversal
2016-12-06  86.93    2016-12-13  89.82   5     658.0  +1895.89  alpha_reversal
2016-12-21  89.46    2016-12-23  89.87   2     611.4  +253.34   alpha_reversal
2016-12-28  89.29    2016-12-30  89.29   2     655.1  -2.62     alpha_reversal
2017-01-05  90.66    2017-01-11  88.92   4     705.2  -1228.06  stop_loss
2017-01-12  88.92    2017-01-24  86.62   7     679.0  -1565.38  stop_loss
2017-01-25  87.51    2017-02-01  87.75   5     648.2  +159.31   alpha_reversal
2017-02-07  88.04    2017-02-08  87.89   1     750.7  -112.61   alpha_reversal
2017-02-09  88.50    2017-02-14  90.18   3     772.2  +1296.16  alpha_reversal
2017-02-23  94.41    2017-02-24  95.75   1     699.0  +931.98   alpha_reversal
2017-03-20  100.01   2017-03-21  99.27   1     677.7  -501.27   alpha_reversal
2017-03-23  98.32    2017-03-28  98.03   3     666.9  -190.40   alpha_reversal
2017-03-29  97.55    2017-04-04  97.27   4     688.4  -196.02   alpha_reversal
2017-04-05  97.46    2017-04-06  97.56   1     745.7  +72.79    alpha_reversal
2017-04-10  97.10    2017-04-13  97.51   3     800.5  +328.27   alpha_reversal
2017-04-19  94.78    2017-05-03  96.21   10    672.0  +963.85   max_holding
2017-05-04  96.80    2017-05-05  96.36   1     854.5  -375.99   alpha_reversal
2017-05-08  96.25    2017-05-15  99.07   5     862.9  +2434.58  alpha_reversal
2017-05-25  100.25   2017-05-31  100.72  3     766.7  +356.68   alpha_reversal
2017-06-09  103.40   2017-06-12  103.52  1     768.0  +95.54    alpha_reversal
2017-06-20  105.51   2017-06-21  105.95  1     756.0  +329.92   alpha_reversal
2017-06-29  104.27   2017-07-05  104.95  3     664.7  +452.75   alpha_reversal
2017-07-07  104.19   2017-07-19  106.18  8     673.7  +1342.46  alpha_reversal
2017-07-25  103.67   2017-08-04  104.59  8     557.6  +511.43   alpha_reversal
2017-08-07  104.44   2017-08-10  104.52  3     611.6  +46.65    alpha_reversal
2017-08-28  104.22   2017-08-30  103.59  2     662.4  -419.74   alpha_reversal
2017-08-31  104.72   2017-09-01  103.55  1     666.8  -776.00   alpha_reversal
2017-09-05  102.76   2017-09-08  103.51  3     647.7  +486.33   alpha_reversal
2017-09-13  104.78   2017-09-15  106.26  2     580.0  +856.00   alpha_reversal
2017-09-21  104.23   2017-10-05  105.26  10    524.7  +542.52   alpha_reversal
2017-10-25  112.62   2017-10-26  112.07  1     487.0  -266.49   alpha_reversal
2017-10-30  110.75   2017-11-06  110.45  5     458.5  -137.74   alpha_reversal
2017-11-07  110.57   2017-11-08  111.69  1     542.8  +604.98   alpha_reversal
2017-11-10  110.41   2017-11-14  110.24  2     532.0  -88.11    alpha_reversal
2017-11-15  110.04   2017-11-28  111.34  8     556.9  +721.03   alpha_reversal
2017-11-30  110.90   2017-12-04  110.53  2     591.9  -216.24   alpha_reversal
2017-12-05  111.17   2017-12-06  112.16  1     552.2  +548.94   alpha_reversal
2017-12-19  112.85   2017-12-28  111.77  6     545.5  -590.75   alpha_reversal
2017-12-29  111.21   2018-01-10  114.48  7     621.1  +2029.86  alpha_reversal
2018-01-24  113.08   2018-01-25  114.82  1     479.9  +834.88   alpha_reversal
2018-01-29  114.36   2018-01-31  109.88  2     484.1  -2168.46  stop_loss
2018-02-01  111.45   2018-02-05  103.68  2     419.3  -3257.43  stop_loss
2018-02-06  104.93   2018-02-08  100.48  2     268.7  -1196.94  stop_loss
2018-02-13  103.44   2018-02-28  103.94  10    258.2  +127.61   max_holding
2018-03-07  103.37   2018-03-14  105.89  5     329.6  +831.15   alpha_reversal
2018-03-22  102.04   2018-03-23  100.11  1     318.4  -613.44   alpha_reversal
2018-04-12  104.48   2018-04-13  104.53  1     322.1  +15.34    alpha_reversal
2018-04-17  104.57   2018-04-24  100.98  5     327.4  -1173.82  stop_loss
2018-05-01  100.94   2018-05-14  100.88  9     374.0  -22.77    alpha_reversal
2018-05-15  100.23   2018-05-17  99.11   2     391.3  -440.06   alpha_reversal
2018-05-22  98.46    2018-05-29  96.26   4     436.3  -959.21   alpha_reversal
2018-05-30  97.62    2018-06-08  100.01  7     430.5  +1030.40  alpha_reversal
2018-06-12  98.89    2018-06-18  97.81   4     495.2  -536.00   alpha_reversal
2018-06-22  99.13    2018-06-25  98.83   1     504.2  -151.56   alpha_reversal
2018-07-16  100.62   2018-07-18  103.03  2     492.0  +1183.95  alpha_reversal
2018-07-19  101.63   2018-07-24  104.29  3     428.8  +1138.78  alpha_reversal
2018-07-26  105.13   2018-07-27  106.05  1     437.5  +401.93   alpha_reversal
2018-08-08  105.85   2018-08-10  105.41  2     524.7  -233.16   alpha_reversal
2018-08-13  105.09   2018-08-16  106.80  3     558.1  +958.16   alpha_reversal
2018-08-28  109.12   2018-08-31  109.31  3     524.4  +100.27   alpha_reversal
2018-09-04  108.63   2018-09-06  111.08  2     576.4  +1410.79  alpha_reversal
2018-09-17  113.70   2018-09-18  114.06  1     569.9  +203.47   alpha_reversal
2018-09-20  115.34   2018-09-24  114.00  2     596.7  -800.01   alpha_reversal
2018-09-25  112.63   2018-10-05  112.89  8     538.9  +140.51   alpha_reversal
2018-10-12  108.75   2018-10-17  113.18  3     432.5  +1915.28  alpha_reversal
2018-10-23  112.86   2018-10-24  111.56  1     416.0  -539.81   alpha_reversal
2018-10-29  111.74   2018-10-30  114.23  1     359.7  +893.86   alpha_reversal
2018-11-01  114.40   2018-11-09  117.95  6     362.9  +1289.67  alpha_reversal
2018-11-23  115.54   2018-11-26  115.46  1     372.5  -30.84    alpha_reversal
2018-11-27  117.09   2018-12-03  119.41  4     370.2  +860.71   alpha_reversal
2018-12-04  119.38   2018-12-12  120.14  5     369.6  +278.89   alpha_reversal
2018-12-17  105.58   2018-12-24  100.33  5     239.8  -1259.25  stop_loss
2018-12-31  105.50   2019-01-15  105.65  10    238.7  +35.25    max_holding
2019-01-17  105.54   2019-01-29  106.44  7     321.6  +291.78   alpha_reversal
2019-02-05  108.63   2019-02-07  107.85  2     371.7  -292.32   alpha_reversal
2019-02-08  108.24   2019-02-25  111.97  10    412.4  +1539.67  max_holding
2019-02-27  111.82   2019-03-01  113.74  2     519.9  +1002.01  alpha_reversal
2019-03-14  113.58   2019-03-25  112.31  7     561.9  -715.16   alpha_reversal
2019-03-26  114.04   2019-03-27  114.03  1     549.8  -3.90     alpha_reversal
2019-04-02  113.33   2019-04-11  111.16  7     579.4  -1256.43  stop_loss
2019-04-12  111.91   2019-04-17  113.88  3     606.7  +1199.02  alpha_reversal
2019-04-22  113.43   2019-04-23  115.02  1     478.3  +759.75   alpha_reversal
2019-04-30  116.20   2019-05-02  116.15  2     489.7  -24.67    alpha_reversal
2019-05-03  116.87   2019-05-07  115.07  2     492.2  -883.00   alpha_reversal
2019-05-09  114.17   2019-05-20  113.80  7     470.7  -173.66   alpha_reversal
2019-05-21  113.67   2019-05-23  114.93  2     472.8  +595.42   alpha_reversal
2019-05-29  108.82   2019-06-11  115.70  9     377.2  +2596.99  alpha_reversal
2019-06-18  116.19   2019-06-25  119.40  5     444.8  +1424.94  alpha_reversal
2019-06-28  115.41   2019-07-03  117.66  3     453.8  +1022.08  alpha_reversal
2019-07-08  116.81   2019-07-09  117.05  1     489.6  +121.15   alpha_reversal
2019-07-11  116.09   2019-07-12  111.17  1     502.7  -2476.22  stop_loss
2019-07-15  111.62   2019-07-19  107.87  4     406.4  -1525.54  stop_loss
2019-07-22  106.59   2019-07-30  109.33  6     406.0  +1112.78  alpha_reversal
2019-08-22  108.77   2019-08-23  105.73  1     492.1  -1495.37  stop_loss
2019-08-26  106.69   2019-08-28  107.31  2     457.4  +286.91   alpha_reversal
2019-08-29  107.05   2019-09-04  107.46  3     411.6  +168.77   alpha_reversal
2019-09-06  107.03   2019-09-16  108.03  6     463.3  +464.36   alpha_reversal
2019-09-20  109.90   2019-09-23  109.87  1     484.2  -16.84    alpha_reversal
2019-09-26  107.56   2019-10-02  110.08  4     481.0  +1211.84  alpha_reversal
2019-10-10  107.74   2019-10-16  112.73  4     435.9  +2174.37  alpha_reversal
2019-10-21  106.87   2019-11-04  108.61  10    342.1  +593.94   max_holding
2019-11-05  108.87   2019-11-19  112.43  10    385.4  +1372.37  max_holding
2019-11-27  115.79   2019-11-29  115.46  1     470.8  -157.27   alpha_reversal
2019-12-19  122.18   2019-12-20  122.65  1     534.1  +253.21   alpha_reversal
2019-12-27  122.51   2020-01-13  122.35  10    566.8  -93.19    max_holding
2020-01-23  124.85   2020-01-24  124.55  1     516.8  -155.63   alpha_reversal
2020-01-27  125.04   2020-01-28  125.54  1     519.5  +262.28   alpha_reversal
2020-02-07  127.68   2020-02-12  126.88  3     494.3  -395.19   alpha_reversal
2020-02-13  126.16   2020-02-24  123.31  6     555.6  -1586.24  stop_loss
2020-02-25  122.37   2020-02-27  117.57  2     477.0  -2287.69  stop_loss
2020-02-28  113.76   2020-03-09  115.30  6     338.7  +522.55   alpha_reversal
2020-03-12  106.09   2020-03-16  107.44  2     183.5  +247.26   alpha_reversal
2020-03-20  101.42   2020-03-31  110.82  7     139.2  +1307.74  alpha_reversal
2020-04-27  130.52   2020-05-06  125.14  7     188.0  -1011.41  alpha_reversal
2020-05-07  124.85   2020-05-11  126.01  2     235.4  +273.03   alpha_reversal
2020-05-13  124.46   2020-05-18  127.20  3     260.3  +713.36   alpha_reversal
2020-05-20  124.93   2020-06-04  124.86  10    265.4  -18.08    max_holding
2020-06-05  125.47   2020-06-08  124.89  1     299.0  -172.37   alpha_reversal
2020-06-09  124.33   2020-06-11  119.87  2     302.6  -1348.23  stop_loss
2020-06-12  121.08   2020-06-22  122.02  6     261.4  +244.17   alpha_reversal
2020-06-23  121.69   2020-06-26  117.27  3     301.7  -1333.39  stop_loss
2020-06-29  118.43   2020-06-30  119.67  1     303.0  +374.11   alpha_reversal
2020-07-02  120.08   2020-07-06  121.67  1     326.4  +519.11   alpha_reversal
2020-07-10  121.27   2020-07-16  127.00  4     339.6  +1947.30  alpha_reversal
2020-07-28  125.07   2020-08-04  125.28  5     374.6  +77.49    alpha_reversal
2020-08-05  126.41   2020-08-10  125.97  3     403.3  -177.92   alpha_reversal
2020-08-11  125.19   2020-08-18  127.72  5     412.2  +1042.68  alpha_reversal
2020-08-26  130.59   2020-08-27  131.05  1     424.2  +191.71   alpha_reversal
2020-08-31  131.55   2020-09-03  128.11  3     441.2  -1516.92  stop_loss
2020-09-04  127.41   2020-09-10  125.85  3     321.9  -504.19   alpha_reversal
2020-09-17  126.19   2020-09-21  124.30  2     337.1  -640.32   alpha_reversal
2020-09-23  123.85   2020-09-24  123.93  1     309.3  +22.65    alpha_reversal
2020-10-05  127.10   2020-10-19  123.63  10    333.9  -1160.61  trailing_stop
2020-10-20  123.95   2020-10-26  123.33  4     346.5  -215.06   alpha_reversal
2020-10-30  117.57   2020-11-05  119.72  4     332.7  +716.16   alpha_reversal
2020-11-18  126.37   2020-11-30  124.80  7     338.0  -530.38   alpha_reversal
2020-12-16  129.23   2020-12-18  133.28  2     393.3  +1591.30  alpha_reversal
2020-12-23  131.19   2020-12-28  132.14  2     360.0  +340.99   alpha_reversal
2021-01-05  136.72   2021-01-06  137.86  1     365.8  +420.18   alpha_reversal
2021-01-13  136.33   2021-01-25  143.17  7     378.7  +2591.12  alpha_reversal
2021-02-01  140.49   2021-02-10  143.89  7     280.4  +952.18   alpha_reversal
2021-02-12  143.83   2021-02-16  142.38  1     332.8  -481.26   alpha_reversal
2021-02-17  143.04   2021-02-19  140.58  2     344.5  -845.49   alpha_reversal
2021-02-22  140.64   2021-02-26  137.54  4     350.4  -1086.37  alpha_reversal
2021-03-01  138.42   2021-03-04  132.86  3     323.2  -1797.77  stop_loss
2021-03-05  135.62   2021-03-15  139.24  6     302.2  +1092.20  alpha_reversal
2021-03-18  139.42   2021-03-25  140.58  5     356.3  +414.19   alpha_reversal
2021-04-01  141.47   2021-04-16  140.82  10    370.2  -241.94   max_holding
2021-04-27  141.77   2021-05-05  145.01  6     374.0  +1209.61  alpha_reversal
2021-05-20  148.63   2021-05-21  148.38  1     411.5  -100.41   alpha_reversal
2021-05-26  147.81   2021-06-01  144.57  3     441.7  -1430.99  stop_loss
2021-06-02  145.30   2021-06-10  145.92  6     408.8  +254.83   alpha_reversal
2021-06-14  144.57   2021-06-15  143.66  1     413.1  -377.17   alpha_reversal
2021-06-16  143.75   2021-06-28  143.25  8     420.0  -210.76   alpha_reversal
2021-06-29  143.40   2021-06-30  143.88  1     465.4  +221.89   alpha_reversal
2021-07-13  147.98   2021-07-21  148.03  6     471.7  +20.87    alpha_reversal
2021-08-05  151.85   2021-08-06  151.19  1     479.7  -315.80   alpha_reversal
2021-08-09  151.86   2021-08-10  151.76  1     496.0  -49.30    alpha_reversal
2021-08-19  156.11   2021-08-20  156.72  1     478.4  +288.86   alpha_reversal
2021-08-24  154.24   2021-08-31  152.10  5     480.1  -1027.35  alpha_reversal
2021-09-01  152.79   2021-09-07  151.39  3     489.5  -685.42   alpha_reversal
2021-09-08  151.17   2021-09-09  147.66  1     470.7  -1655.09  stop_loss
2021-09-10  146.84   2021-09-20  143.92  6     424.1  -1239.60  stop_loss
2021-09-23  144.98   2021-09-30  141.89  5     406.3  -1258.25  stop_loss
2021-10-04  140.02   2021-10-18  140.67  10    364.8  +237.42   max_holding
2021-10-29  143.24   2021-11-01  143.22  1     367.4  -7.44     alpha_reversal
2021-11-04  144.75   2021-11-11  143.26  5     333.7  -499.75   alpha_reversal
2021-11-12  145.12   2021-11-15  143.66  1     332.5  -483.43   alpha_reversal
2021-11-26  140.92   2021-11-29  141.27  1     362.9  +125.41   alpha_reversal
2021-12-03  141.08   2021-12-06  144.09  1     332.2  +999.07   alpha_reversal
2021-12-13  149.11   2021-12-14  150.59  1     313.7  +463.62   alpha_reversal
2021-12-20  148.49   2021-12-23  148.78  3     300.8  +88.37    alpha_reversal
2022-01-12  150.30   2022-01-20  146.13  5     358.7  -1494.12  stop_loss
2022-02-02  152.93   2022-02-11  148.31  7     287.0  -1328.04  stop_loss
2022-02-14  146.59   2022-02-24  140.74  7     309.5  -1809.20  trailing_stop
2022-02-25  147.89   2022-02-28  146.46  1     252.5  -358.72   alpha_reversal
2022-03-14  152.95   2022-03-16  155.32  2     242.6  +574.05   alpha_reversal
2022-03-17  157.51   2022-03-22  155.75  3     236.3  -415.71   alpha_reversal
2022-03-23  155.32   2022-03-28  158.27  3     254.1  +749.93   alpha_reversal
2022-03-30  159.99   2022-03-31  157.73  1     285.5  -645.25   alpha_reversal
2022-04-01  158.74   2022-04-06  162.18  3     281.8  +968.49   alpha_reversal
2022-05-03  158.83   2022-05-10  157.61  5     255.9  -313.89   alpha_reversal
2022-05-16  158.65   2022-05-17  159.15  1     257.6  +128.80   alpha_reversal
2022-05-20  157.67   2022-05-25  160.89  3     255.0  +821.04   alpha_reversal
2022-05-26  160.90   2022-06-09  155.59  9     264.3  -1403.59  stop_loss
2022-06-21  155.12   2022-06-23  160.92  2     245.2  +1422.29  alpha_reversal
2022-07-06  159.86   2022-07-07  159.88  1     238.7  +4.62     alpha_reversal
2022-07-11  159.91   2022-07-12  157.51  1     264.3  -634.07   alpha_reversal
2022-07-13  157.30   2022-07-15  159.64  2     269.9  +632.14   alpha_reversal
2022-07-28  156.19   2022-08-11  149.71  10    267.7  -1734.36  stop_loss
2022-08-12  148.21   2022-08-16  150.09  2     295.6  +557.18   alpha_reversal
2022-08-19  151.80   2022-08-22  151.12  1     306.2  -209.39   alpha_reversal
2022-08-25  150.86   2022-08-29  146.98  2     327.2  -1267.73  stop_loss
2022-08-30  146.61   2022-09-07  147.95  5     327.5  +436.38   alpha_reversal
2022-09-13  145.62   2022-09-14  148.48  1     298.8  +853.76   alpha_reversal
2022-10-04  149.49   2022-10-07  144.46  3     282.3  -1421.86  stop_loss
2022-10-26  155.44   2022-10-27  155.38  1     272.2  -17.74    alpha_reversal
2022-11-08  156.91   2022-11-10  157.32  2     277.9  +114.30   alpha_reversal
2022-11-14  155.17   2022-11-17  157.68  3     245.7  +615.51   alpha_reversal
2022-12-12  161.56   2022-12-14  163.14  2     338.8  +535.70   alpha_reversal
2022-12-19  159.41   2022-12-27  161.03  5     330.1  +531.58   alpha_reversal
2022-12-28  160.49   2022-12-30  160.32  2     364.1  -61.73    alpha_reversal
2023-01-03  161.88   2023-01-05  162.27  2     366.0  +143.41   alpha_reversal
2023-01-09  159.51   2023-01-18  154.06  6     341.6  -1858.87  stop_loss
2023-01-30  147.17   2023-02-09  146.48  8     292.2  -202.11   alpha_reversal
2023-02-15  144.78   2023-02-21  144.42  3     321.1  -114.54   alpha_reversal
2023-02-28  140.23   2023-03-06  142.19  4     326.3  +640.26   alpha_reversal
2023-03-09  138.38   2023-03-13  139.91  2     350.0  +533.82   alpha_reversal
2023-03-17  139.43   2023-03-27  140.13  6     334.8  +234.89   alpha_reversal
2023-03-29  140.28   2023-03-31  141.68  2     368.4  +517.42   alpha_reversal
2023-04-24  149.76   2023-04-25  150.99  1     323.1  +394.63   alpha_reversal
2023-05-02  151.00   2023-05-09  147.21  5     357.1  -1352.90  stop_loss
2023-05-10  147.91   2023-05-18  144.86  6     375.1  -1142.30  stop_loss
2023-05-19  145.40   2023-05-22  144.47  1     384.9  -357.25   alpha_reversal
2023-05-25  142.35   2023-06-01  142.33  4     378.9  -8.55     alpha_reversal
2023-06-02  144.71   2023-06-16  151.25  10    376.5  +2462.83  max_holding
2023-06-28  150.23   2023-06-30  152.44  2     410.9  +907.13   alpha_reversal
2023-07-05  150.09   2023-07-07  146.66  2     415.7  -1425.46  stop_loss
2023-07-10  147.05   2023-07-17  146.50  5     414.5  -228.89   alpha_reversal
2023-08-01  155.70   2023-08-02  156.48  1     264.4  +207.21   alpha_reversal
2023-08-07  159.58   2023-08-08  159.49  1     256.2  -21.98    alpha_reversal
2023-08-10  158.72   2023-08-14  159.73  2     258.3  +261.09   alpha_reversal
2023-08-15  159.43   2023-08-18  158.86  3     262.0  -150.32   alpha_reversal
2023-08-21  154.28   2023-09-01  148.87  9     244.9  -1324.39  stop_loss
2023-09-05  149.20   2023-09-08  148.94  3     267.4  -69.65    alpha_reversal
2023-09-20  151.28   2023-09-21  149.96  1     303.2  -397.45   alpha_reversal
2023-09-26  147.66   2023-10-02  143.93  4     333.5  -1246.69  stop_loss
2023-10-11  145.03   2023-10-18  141.68  5     346.8  -1160.23  stop_loss
2023-10-24  140.43   2023-10-26  138.22  2     329.9  -728.78   alpha_reversal
2023-10-30  136.53   2023-11-07  139.98  6     313.6  +1083.08  alpha_reversal
2023-11-08  139.61   2023-11-21  141.31  9     355.1  +602.78   alpha_reversal
2023-11-24  142.74   2023-11-27  141.46  1     370.3  -475.26   alpha_reversal
2023-12-08  144.54   2023-12-14  146.69  4     364.3  +782.06   alpha_reversal
2023-12-18  145.49   2023-12-26  146.00  5     350.0  +178.18   alpha_reversal
2024-01-12  152.00   2024-01-16  150.10  1     389.5  -740.29   alpha_reversal
2024-01-29  149.16   2024-02-01  148.08  3     376.4  -408.03   alpha_reversal
2024-02-02  146.59   2024-02-07  147.72  3     375.5  +426.07   alpha_reversal
2024-02-08  146.39   2024-02-21  149.51  8     363.9  +1132.54  alpha_reversal
2024-02-27  151.82   2024-02-28  152.21  1     378.0  +145.65   alpha_reversal
2024-03-01  152.90   2024-03-04  150.60  1     404.5  -930.65   alpha_reversal
2024-03-07  149.83   2024-03-11  151.91  2     374.5  +776.62   alpha_reversal
2024-03-14  150.15   2024-03-20  146.75  4     351.8  -1196.29  stop_loss
2024-03-21  146.89   2024-03-28  149.04  5     371.9  +800.41   alpha_reversal
2024-04-02  148.76   2024-04-03  145.34  1     407.8  -1393.96  stop_loss
2024-04-04  143.83   2024-04-11  140.19  5     359.7  -1309.04  stop_loss
2024-04-29  138.47   2024-05-02  141.25  3     323.5  +900.19   alpha_reversal
2024-05-07  140.26   2024-05-08  140.34  1     295.7  +22.62    alpha_reversal
2024-05-22  145.94   2024-05-23  142.18  1     340.9  -1280.16  stop_loss
2024-05-28  137.27   2024-06-03  140.32  4     304.3  +929.36   alpha_reversal
2024-06-06  139.21   2024-06-07  139.70  1     322.9  +157.50   alpha_reversal
2024-06-12  138.25   2024-06-17  138.62  3     332.2  +124.47   alpha_reversal
2024-06-20  140.50   2024-06-21  141.28  1     346.0  +270.21   alpha_reversal
2024-06-26  139.59   2024-07-01  139.09  3     363.9  -182.15   alpha_reversal
2024-07-03  138.51   2024-07-08  138.18  2     361.5  -122.15   alpha_reversal
2024-07-09  139.81   2024-07-12  142.36  3     362.5  +923.67   alpha_reversal
2024-08-01  152.84   2024-08-02  155.90  1     274.2  +838.35   alpha_reversal
2024-08-14  150.67   2024-08-15  151.10  1     262.8  +112.68   alpha_reversal
2024-08-16  151.54   2024-08-19  151.62  1     278.3  +21.29    alpha_reversal
2024-08-28  157.03   2024-08-29  157.17  1     322.3  +45.02    alpha_reversal
2024-09-06  157.47   2024-09-09  159.45  1     321.9  +636.22   alpha_reversal
2024-09-12  157.72   2024-09-16  159.81  2     312.1  +652.62   alpha_reversal
2024-09-19  157.89   2024-09-25  153.70  4     329.8  -1383.96  stop_loss
2024-09-26  154.61   2024-10-10  153.61  10    341.6  -340.49   max_holding
2024-10-28  154.81   2024-11-04  151.44  5     325.6  -1097.49  stop_loss
2024-11-05  151.69   2024-11-11  148.37  4     341.5  -1133.63  stop_loss
2024-11-12  146.22   2024-11-18  148.12  4     319.4  +604.33   alpha_reversal
2024-11-20  146.67   2024-11-21  148.81  1     306.0  +655.02   alpha_reversal
2024-11-25  149.23   2024-11-26  149.06  1     300.0  -50.53    alpha_reversal
2024-11-27  150.06   2024-11-29  149.54  1     305.3  -160.65   alpha_reversal
2024-12-02  149.48   2024-12-04  145.16  2     325.4  -1407.92  stop_loss
2024-12-05  144.38   2024-12-12  141.08  5     322.2  -1065.89  stop_loss
2024-12-13  141.58   2024-12-18  139.64  3     320.5  -623.46   alpha_reversal
2024-12-27  140.07   2025-01-02  138.93  3     325.3  -368.73   alpha_reversal
2025-01-03  139.24   2025-01-06  138.59  1     348.2  -226.51   alpha_reversal
2025-01-07  141.21   2025-01-08  137.25  1     330.1  -1307.73  stop_loss
2025-01-10  137.18   2025-01-14  139.64  2     292.6  +719.11   alpha_reversal
2025-01-29  145.96   2025-01-30  147.47  1     250.7  +379.46   alpha_reversal
2025-02-04  148.22   2025-02-11  150.62  5     263.8  +632.66   alpha_reversal
2025-02-13  151.85   2025-02-21  157.82  5     288.3  +1721.99  alpha_reversal
2025-02-27  159.37   2025-02-28  160.47  1     267.5  +292.99   alpha_reversal
2025-03-13  158.65   2025-03-14  158.32  1     248.9  -83.05    alpha_reversal
2025-03-17  158.51   2025-03-19  158.49  2     260.6  -3.28     alpha_reversal
2025-03-20  158.68   2025-03-24  158.78  2     279.5  +29.05    alpha_reversal
2025-03-25  156.73   2025-03-27  158.63  2     281.7  +533.79   alpha_reversal
2025-04-02  151.22   2025-04-04  149.01  2     226.3  -500.74   alpha_reversal
2025-04-08  146.01   2025-04-17  153.12  7     183.9  +1308.99  alpha_reversal
2025-04-22  153.55   2025-04-23  151.09  1     171.8  -422.37   alpha_reversal
2025-04-24  150.81   2025-04-29  151.61  3     178.5  +143.21   alpha_reversal
2025-05-02  151.96   2025-05-05  150.72  1     204.3  -253.48   alpha_reversal
2025-05-06  150.36   2025-05-13  144.34  5     215.3  -1294.90  trailing_stop
2025-05-14  142.46   2025-05-20  149.42  4     206.7  +1437.50  alpha_reversal
2025-06-04  150.42   2025-06-06  152.04  2     277.7  +451.17   alpha_reversal
2025-06-12  153.80   2025-06-16  152.23  2     309.1  -484.07   alpha_reversal
2025-06-17  149.60   2025-07-02  152.56  10    301.3  +894.68   alpha_reversal
2025-07-03  153.16   2025-07-09  153.27  3     321.9  +35.95    alpha_reversal
2025-07-31  161.73   2025-08-04  167.75  2     269.1  +1618.91  alpha_reversal
2025-08-13  171.23   2025-08-14  171.36  1     279.3  +34.37    alpha_reversal
2025-08-19  174.55   2025-08-20  175.40  1     288.6  +244.05   alpha_reversal
2025-08-26  174.54   2025-08-28  173.33  2     296.6  -356.47   alpha_reversal
2025-08-29  175.21   2025-09-02  175.91  1     309.3  +217.80   alpha_reversal
2025-09-08  176.16   2025-09-18  172.06  8     312.9  -1282.37  stop_loss
2025-09-19  174.24   2025-09-22  172.11  1     290.4  -618.65   alpha_reversal
2025-09-23  174.63   2025-09-26  177.54  3     285.7  +833.47   alpha_reversal
2025-10-07  186.80   2025-10-09  188.78  2     245.2  +484.72   alpha_reversal
2025-10-22  190.80   2025-10-23  190.15  1     251.9  -165.01   alpha_reversal
2025-10-24  188.29   2025-11-05  183.76  8     252.5  -1145.21  stop_loss
2025-11-06  184.90   2025-11-07  184.32  1     261.7  -151.77   alpha_reversal
2025-11-10  186.32   2025-11-11  191.49  1     257.6  +1331.59  alpha_reversal
2025-12-01  204.36   2025-12-09  198.80  6     242.7  -1347.82  stop_loss
2025-12-10  205.55   2025-12-12  210.36  2     212.6  +1021.73  alpha_reversal
2025-12-19  205.38   2026-01-02  206.15  8     194.5  +149.57   alpha_reversal
2026-01-06  203.81   2026-01-13  212.41  5     220.4  +1896.23  alpha_reversal
2026-02-11  239.71   2026-02-12  243.14  1     193.3  +662.97   alpha_reversal
2026-02-17  242.16   2026-02-19  245.48  2     194.2  +644.06   alpha_reversal
2026-02-23  244.66   2026-02-24  246.16  1     191.2  +285.75   alpha_reversal
2026-02-26  243.59   2026-03-02  248.44  2     197.5  +956.66   alpha_reversal
2026-03-04  245.42   2026-03-09  242.47  3     186.5  -551.04   alpha_reversal
2026-03-12  242.16   2026-03-20  235.25  6     173.9  -1201.17  stop_loss

**Best 3 trades:**

- 2019-06-11: P&L = **+2596.99** (alpha_reversal)
- 2021-01-25: P&L = **+2591.12** (alpha_reversal)
- 2023-06-16: P&L = **+2462.83** (max_holding)

**Worst 3 trades:**

- 2018-02-05: P&L = **-3257.43** (stop_loss)
- 2019-07-12: P&L = **-2476.22** (stop_loss)
- 2020-02-27: P&L = **-2287.69** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  102,740.86
2017-03-23  101,922.98
2017-09-20  108,274.87
2018-03-21  106,593.68
2018-09-18  108,939.26
2019-03-20  113,862.84
2019-09-17  115,430.36
2020-03-17  117,376.14
2020-09-14  117,984.32
2021-03-15  118,974.90
2021-09-10  115,238.75
2022-03-10  108,705.21
2022-09-08  108,376.15
2023-03-09  108,109.59
2023-09-07  107,459.52
2024-03-07  105,806.06
2024-09-05  106,272.74
2025-03-07  103,400.13
2025-09-05  106,951.25
2026-03-06  109,570.23

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -1.24%
2017-03-23  -2.02%
2017-09-20  -0.06%
2018-03-21  -5.07%
2018-09-18  -2.98%
2019-03-20  -0.46%
2019-09-17  -2.94%
2020-03-17  -2.94%
2020-09-14  -2.44%
2021-03-15  -2.55%
2021-09-10  -5.61%
2022-03-10  -10.97%
2022-09-08  -11.23%
2023-03-09  -11.45%
2023-09-07  -11.99%
2024-03-07  -13.34%
2024-09-05  -12.96%
2025-03-07  -15.31%
2025-09-05  -12.40%
2026-03-06  -10.26%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  26.86%
Out-of-Sample (30%)  2023-03-24  2026-03-27  7.93%

#### Return Distribution

Return Bin          Count
-2.225% to -1.783%  2
-1.783% to -1.341%  8
-1.341% to -0.899%  22
-0.899% to -0.457%  90
-0.457% to -0.015%  748
-0.015% to 0.427%   1470
0.427% to 0.869%    142
0.869% to 1.311%    27
1.311% to 1.753%    4
1.753% to 2.195%    3

### TSLA — AlphaCombined

**Net Return (after slippage):** 5.50%  **vs SPY (exposure-adj): -194.24%** (underperform)  
**Gross Return (pre-cost):** 11.28%  
**Total Slippage Cost:** $5,772.72  
**Trade Count:** 345  
**Win Rate:** 51.9%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-02  16.13    2016-05-04  14.83   2     1055.1  -1369.71  stop_loss
2016-05-05  14.11    2016-05-16  13.88   7     939.9   -216.16   alpha_reversal
2016-05-31  14.89    2016-06-01  14.63   1     1338.7  -347.30   alpha_reversal
2016-06-02  14.60    2016-06-08  15.69   4     1409.2  +1534.36  alpha_reversal
2016-06-13  14.53    2016-06-14  14.32   1     1304.6  -271.92   alpha_reversal
2016-06-15  14.52    2016-06-22  13.10   5     1286.6  -1822.51  stop_loss
2016-06-23  13.10    2016-07-08  14.44   10    1198.0  +1611.18  alpha_reversal
2016-07-13  14.84    2016-07-25  15.33   8     1330.8  +643.55   alpha_reversal
2016-07-27  15.24    2016-07-29  15.64   2     1461.2  +591.14   alpha_reversal
2016-08-04  15.38    2016-08-05  15.33   1     1478.4  -79.87    alpha_reversal
2016-08-08  15.08    2016-08-10  15.04   2     1557.2  -76.40    alpha_reversal
2016-08-11  15.00    2016-08-15  15.03   2     1633.9  +49.53    alpha_reversal
2016-08-16  14.91    2016-08-19  14.99   3     1776.2  +138.04   alpha_reversal
2016-08-23  15.00    2016-08-24  14.83   1     2037.4  -331.93   alpha_reversal
2016-08-26  14.67    2016-08-30  14.08   2     2111.7  -1248.11  stop_loss
2016-08-31  14.14    2016-09-01  13.38   1     2067.5  -1577.67  stop_loss
2016-09-02  13.19    2016-09-12  13.21   5     1805.8  +38.76    alpha_reversal
2016-09-13  13.08    2016-09-15  13.35   2     1749.4  +486.54   alpha_reversal
2016-09-28  13.76    2016-09-29  13.37   1     1947.9  -749.76   alpha_reversal
2016-09-30  13.61    2016-10-04  14.09   2     1879.0  +898.44   alpha_reversal
2016-10-06  13.41    2016-10-19  13.56   9     1709.4  +268.68   alpha_reversal
2016-10-21  13.35    2016-10-25  13.48   2     1715.3  +234.28   alpha_reversal
2016-10-27  13.61    2016-10-28  13.32   1     1709.5  -483.44   alpha_reversal
2016-11-02  12.54    2016-11-09  12.66   5     1614.5  +199.23   alpha_reversal
2016-11-10  12.36    2016-11-25  13.10   10    1473.5  +1091.25  alpha_reversal
2016-12-01  12.13    2016-12-13  13.20   8     1552.6  +1664.42  alpha_reversal
2016-12-16  13.51    2016-12-19  13.51   1     1756.5  +4.38     alpha_reversal
2017-01-06  15.27    2017-01-09  15.41   1     1573.5  +213.98   alpha_reversal
2017-01-12  15.31    2017-01-17  15.70   2     1716.4  +658.78   alpha_reversal
2017-02-02  16.78    2017-02-03  16.75   1     1714.6  -53.89    alpha_reversal
2017-02-06  17.19    2017-02-07  17.16   1     1754.5  -64.06    alpha_reversal
2017-02-17  18.16    2017-02-21  18.48   1     1465.2  +477.19   alpha_reversal
2017-02-22  18.24    2017-02-23  17.06   1     1386.6  -1644.03  stop_loss
2017-02-24  17.14    2017-03-03  16.76   5     1238.0  -469.15   alpha_reversal
2017-03-08  16.47    2017-03-15  17.04   5     1402.5  +804.93   alpha_reversal
2017-03-24  17.55    2017-03-27  18.01   1     1407.5  +637.43   alpha_reversal
2017-04-18  20.03    2017-04-20  20.16   2     1238.8  +161.75   alpha_reversal
2017-04-21  20.38    2017-05-01  21.51   6     1275.2  +1438.06  alpha_reversal
2017-05-05  20.57    2017-05-10  21.67   3     1093.8  +1207.05  alpha_reversal
2017-05-16  21.14    2017-05-30  22.33   9     1163.1  +1377.38  alpha_reversal
2017-06-12  23.95    2017-06-13  25.05   1     971.3   +1073.17  alpha_reversal
2017-06-28  24.76    2017-07-03  23.50   3     974.2   -1232.84  stop_loss
2017-07-05  21.82    2017-07-17  21.29   8     826.7   -432.26   alpha_reversal
2017-07-18  21.89    2017-07-21  21.88   3     835.3   -9.37     alpha_reversal
2017-07-24  22.85    2017-07-25  22.63   1     902.7   -196.25   alpha_reversal
2017-08-01  21.32    2017-08-04  23.78   3     872.4   +2152.14  alpha_reversal
2017-08-16  24.21    2017-08-17  23.45   1     962.1   -727.80   alpha_reversal
2017-08-21  22.54    2017-08-30  23.53   7     941.7   +940.08   alpha_reversal
2017-09-01  23.71    2017-09-12  24.17   6     1135.4  +529.15   alpha_reversal
2017-09-13  24.43    2017-09-14  25.16   1     1089.2  +801.51   alpha_reversal
2017-09-20  24.94    2017-09-22  23.39   2     1078.2  -1666.39  stop_loss
2017-09-25  23.01    2017-10-04  23.66   7     960.8   +619.43   alpha_reversal
2017-10-10  23.72    2017-10-13  23.69   3     1005.5  -25.18    alpha_reversal
2017-10-16  23.39    2017-10-18  23.96   2     1110.6  +643.74   alpha_reversal
2017-10-20  23.02    2017-10-25  21.71   3     1128.3  -1474.00  stop_loss
2017-10-26  21.76    2017-11-02  19.94   5     1094.5  -1986.31  stop_loss
2017-11-03  20.42    2017-11-14  20.57   7     935.8   +143.65   alpha_reversal
2017-11-21  21.20    2017-11-28  21.16   4     1007.1  -38.78    alpha_reversal
2017-11-30  20.60    2017-12-07  20.74   5     1088.1  +150.88   alpha_reversal
2017-12-08  21.02    2017-12-11  21.92   1     1168.7  +1048.54  alpha_reversal
2017-12-19  22.08    2017-12-27  20.77   5     1105.9  -1458.41  stop_loss
2017-12-28  21.03    2018-01-09  22.23   7     1198.8  +1438.99  alpha_reversal
2018-01-10  22.33    2018-01-11  22.52   1     1098.6  +206.06   alpha_reversal
2018-01-26  22.87    2018-02-05  22.20   6     1102.8  -739.43   trailing_stop
2018-02-07  23.01    2018-02-08  21.00   1     1015.0  -2036.80  stop_loss
2018-02-09  20.71    2018-02-20  22.31   6     790.0   +1265.48  alpha_reversal
2018-02-22  23.09    2018-02-23  23.46   1     875.6   +322.87   alpha_reversal
2018-02-26  23.84    2018-02-27  23.39   1     921.9   -416.95   alpha_reversal
2018-03-02  22.35    2018-03-13  22.78   7     875.1   +372.29   alpha_reversal
2018-03-15  21.72    2018-03-23  20.09   6     836.1   -1358.58  stop_loss
2018-03-28  17.19    2018-04-02  16.82   2     733.0   -271.45   alpha_reversal
2018-04-03  17.84    2018-04-04  19.12   1     660.7   +842.70   alpha_reversal
2018-04-13  20.03    2018-04-20  19.34   5     659.5   -457.06   alpha_reversal
2018-04-23  18.90    2018-04-25  18.70   2     739.0   -145.94   alpha_reversal
2018-04-26  19.04    2018-04-30  19.58   2     799.3   +433.25   alpha_reversal
2018-05-04  19.62    2018-05-18  18.45   10    793.0   -928.05   trailing_stop
2018-05-21  18.98    2018-05-29  18.91   5     904.9   -61.18    alpha_reversal
2018-06-08  21.19    2018-06-11  22.13   1     878.9   +827.05   alpha_reversal
2018-06-25  22.21    2018-06-28  23.32   3     719.7   +795.49   alpha_reversal
2018-07-02  22.35    2018-07-03  20.71   1     671.7   -1098.61  stop_loss
2018-07-06  20.60    2018-07-16  20.66   6     648.6   +38.51    alpha_reversal
2018-07-17  21.52    2018-07-24  19.82   5     736.5   -1255.52  stop_loss
2018-07-25  20.59    2018-08-01  20.05   5     729.9   -399.23   alpha_reversal
2018-08-10  23.71    2018-08-13  23.75   1     581.6   +21.87    alpha_reversal
2018-08-15  22.59    2018-08-17  20.36   2     606.3   -1354.53  stop_loss
2018-08-20  20.57    2018-08-27  21.27   5     573.0   +401.74   alpha_reversal
2018-08-29  20.34    2018-09-05  18.71   4     702.3   -1149.96  stop_loss
2018-09-06  18.74    2018-09-17  19.65   7     756.3   +685.83   alpha_reversal
2018-09-19  19.94    2018-09-25  20.06   4     660.0   +73.48    alpha_reversal
2018-10-01  20.72    2018-10-02  20.06   1     565.9   -376.71   alpha_reversal
2018-10-04  18.80    2018-10-08  16.70   2     575.5   -1210.04  stop_loss
2018-10-09  17.53    2018-10-11  16.81   2     559.7   -403.98   alpha_reversal
2018-11-06  22.75    2018-11-09  23.36   3     575.9   +349.53   alpha_reversal
2018-11-23  21.73    2018-11-29  22.73   4     667.2   +667.49   alpha_reversal
2018-11-30  23.38    2018-12-03  23.89   1     716.6   +365.73   alpha_reversal
2018-12-10  24.36    2018-12-11  24.44   1     677.0   +56.15    alpha_reversal
2018-12-13  25.13    2018-12-14  24.37   1     725.8   -554.08   alpha_reversal
2018-12-19  22.21    2018-12-24  19.68   3     672.8   -1699.62  stop_loss
2018-12-26  21.75    2018-12-31  22.18   3     579.9   +246.68   alpha_reversal
2019-01-02  20.69    2019-01-10  22.99   6     528.8   +1217.04  alpha_reversal
2019-01-15  22.97    2019-01-16  23.06   1     605.2   +51.43    alpha_reversal
2019-01-22  19.94    2019-01-31  20.46   7     577.9   +300.38   alpha_reversal
2019-02-11  20.87    2019-02-21  19.41   7     726.5   -1061.22  stop_loss
2019-02-22  19.66    2019-02-28  21.31   4     906.5   +1502.61  alpha_reversal
2019-03-04  19.03    2019-03-15  18.35   9     786.9   -535.64   alpha_reversal
2019-03-18  17.97    2019-03-27  18.31   7     856.7   +289.44   alpha_reversal
2019-04-03  19.46    2019-04-04  17.84   1     1037.4  -1681.33  stop_loss
2019-04-05  18.34    2019-04-22  17.51   10    885.5   -736.65   max_holding
2019-04-23  17.60    2019-04-25  16.50   2     1021.9  -1125.88  stop_loss
2019-04-26  15.68    2019-05-03  16.99   5     951.3   +1245.89  alpha_reversal
2019-05-13  15.14    2019-05-17  14.06   4     966.1   -1043.33  trailing_stop
2019-05-20  13.70    2019-05-24  12.70   4     916.1   -911.74   alpha_reversal
2019-05-28  12.59    2019-06-07  13.63   8     892.4   +928.31   alpha_reversal
2019-06-10  14.20    2019-06-11  14.47   1     931.7   +248.76   alpha_reversal
2019-06-21  14.80    2019-06-24  14.90   1     981.5   +101.89   alpha_reversal
2019-06-26  14.63    2019-06-28  14.89   2     1049.1  +277.56   alpha_reversal
2019-07-03  15.67    2019-07-05  15.53   1     1056.9  -143.32   alpha_reversal
2019-07-08  15.36    2019-07-15  16.89   5     1134.3  +1733.12  alpha_reversal
2019-07-19  17.22    2019-07-22  17.04   1     1330.7  -244.58   alpha_reversal
2019-07-23  17.35    2019-07-24  17.65   1     1339.3  +397.10   alpha_reversal
2019-07-26  15.21    2019-08-02  15.61   5     1034.4  +418.51   alpha_reversal
2019-08-12  15.27    2019-08-15  14.37   3     1134.8  -1028.30  trailing_stop
2019-08-16  14.67    2019-08-29  14.77   9     1061.8  +109.67   alpha_reversal
2019-08-30  15.05    2019-09-03  14.99   1     1188.0  -65.36    alpha_reversal
2019-09-05  15.31    2019-09-06  15.16   1     1191.2  -187.29   alpha_reversal
2019-09-09  15.46    2019-09-10  15.69   1     1247.7  +292.49   alpha_reversal
2019-09-13  16.35    2019-09-18  16.22   3     1251.6  -163.06   alpha_reversal
2019-09-19  16.45    2019-09-20  16.03   1     1358.9  -563.83   alpha_reversal
2019-09-25  15.25    2019-09-27  16.13   2     1178.1  +1036.32  alpha_reversal
2019-09-30  16.07    2019-10-02  16.20   2     1116.2  +150.16   alpha_reversal
2019-10-04  15.44    2019-10-08  16.00   2     1096.0  +612.59   alpha_reversal
2019-10-10  16.32    2019-10-11  16.52   1     1118.6  +216.53   alpha_reversal
2019-10-22  17.05    2019-10-23  16.97   1     1217.2  -93.74    alpha_reversal
2019-10-24  19.99    2019-10-25  21.86   1     907.7   +1702.70  alpha_reversal
2019-11-04  21.18    2019-11-07  22.36   3     851.4   +1007.17  alpha_reversal
2019-11-19  23.98    2019-11-20  23.47   1     977.9   -499.12   alpha_reversal
2019-11-25  22.43    2019-12-03  22.40   5     889.4   -28.24    alpha_reversal
2019-12-06  22.40    2019-12-10  23.24   2     1096.6  +921.74   alpha_reversal
2020-01-03  29.55    2020-01-06  30.09   1     771.0   +415.46   alpha_reversal
2020-01-10  31.89    2020-01-13  34.97   1     671.8   +2069.48  alpha_reversal
2020-01-17  34.05    2020-01-22  37.95   2     573.1   +2235.73  alpha_reversal
2020-02-11  51.65    2020-02-14  53.31   3     193.7   +321.08   alpha_reversal
2020-02-27  45.29    2020-02-28  44.51   1     180.3   -140.41   alpha_reversal
2020-03-03  49.73    2020-03-04  49.94   1     167.5   +36.21    alpha_reversal
2020-03-11  42.30    2020-03-13  36.42   2     178.9   -1051.82  alpha_reversal
2020-03-16  29.69    2020-03-20  28.49   4     156.6   -187.64   alpha_reversal
2020-03-23  28.97    2020-03-24  33.65   1     158.8   +743.59   alpha_reversal
2020-03-27  34.31    2020-04-01  32.09   3     169.6   -376.49   alpha_reversal
2020-04-02  30.31    2020-04-06  34.40   2     184.3   +753.10   alpha_reversal
2020-05-08  54.66    2020-05-13  52.70   3     204.6   -399.24   alpha_reversal
2020-05-18  54.27    2020-05-22  54.43   4     226.6   +36.79    alpha_reversal
2020-05-26  54.62    2020-05-27  54.65   1     271.8   +9.79     alpha_reversal
2020-05-29  55.69    2020-06-04  57.60   4     282.6   +537.42   alpha_reversal
2020-06-05  59.07    2020-06-11  64.82   4     291.9   +1678.36  alpha_reversal
2020-06-15  66.09    2020-06-16  65.44   1     227.2   -147.76   alpha_reversal
2020-06-23  66.82    2020-06-24  64.02   1     271.7   -759.28   alpha_reversal
2020-07-14  101.17   2020-07-15  103.02  1     114.2   +210.67   alpha_reversal
2020-07-28  98.48    2020-08-05  98.95   6     101.7   +47.80    alpha_reversal
2020-08-06  99.35    2020-08-13  108.01  5     125.0   +1082.29  alpha_reversal
2020-08-25  134.96   2020-08-26  143.47  1     106.3   +905.62   alpha_reversal
2020-08-31  166.19   2020-09-01  158.27  1     88.9    -703.74   alpha_reversal
2020-09-03  135.73   2020-09-08  110.01  2     73.2    -1881.93  stop_loss
2020-09-09  122.15   2020-09-11  124.18  2     59.8    +120.99   alpha_reversal
2020-09-23  126.85   2020-09-29  139.62  4     59.3    +757.25   alpha_reversal
2020-09-30  143.07   2020-10-01  149.31  1     67.2    +419.34   alpha_reversal
2020-10-07  141.84   2020-10-12  147.36  3     75.1    +414.75   alpha_reversal
2020-10-27  141.63   2020-10-28  135.27  1     108.2   -687.75   alpha_reversal
2020-11-05  146.10   2020-11-12  137.18  5     106.2   -946.78   alpha_reversal
2020-11-13  136.23   2020-11-18  162.13  3     112.9   +2924.03  alpha_reversal
2020-12-03  197.89   2020-12-08  216.52  3     78.2    +1455.70  alpha_reversal
2020-12-10  209.13   2020-12-11  203.23  1     66.2    -390.59   alpha_reversal
2020-12-16  207.69   2020-12-18  231.55  2     68.9    +1642.66  alpha_reversal
2021-01-19  281.66   2021-01-20  283.34  1     60.0    +101.04   alpha_reversal
2021-02-04  283.47   2021-02-08  287.66  2     60.8    +254.78   alpha_reversal
2021-02-10  268.41   2021-02-22  238.05  7     63.6    -1931.86  stop_loss
2021-02-23  233.06   2021-03-03  217.62  6     56.2    -868.41   trailing_stop
2021-03-04  207.25   2021-03-10  222.58  4     49.5    +759.04   alpha_reversal
2021-03-11  233.32   2021-03-16  225.51  3     42.3    -330.35   alpha_reversal
2021-03-17  234.05   2021-03-26  206.13  7     45.3    -1266.07  stop_loss
2021-03-29  203.87   2021-04-01  220.47  3     51.3    +852.29   alpha_reversal
2021-04-05  230.47   2021-04-07  223.54  2     52.5    -363.23   alpha_reversal
2021-04-08  228.05   2021-04-09  225.56  1     58.7    -145.99   alpha_reversal
2021-04-12  234.11   2021-04-14  243.95  2     62.4    +614.03   alpha_reversal
2021-04-15  246.41   2021-04-16  246.47  1     59.7    +3.80     alpha_reversal
2021-04-30  236.60   2021-05-03  228.19  1     64.4    -541.91   alpha_reversal
2021-05-04  224.65   2021-05-11  205.63  5     65.8    -1251.67  stop_loss
2021-05-12  196.73   2021-05-26  206.27  10    66.0    +630.23   max_holding
2021-06-01  208.07   2021-06-03  190.85  2     81.3    -1399.77  stop_loss
2021-06-04  199.78   2021-06-11  203.20  5     79.2    +270.36   alpha_reversal
2021-06-14  206.00   2021-06-15  199.69  1     90.1    -568.93   alpha_reversal
2021-06-16  201.72   2021-06-17  205.43  1     93.4    +346.22   alpha_reversal
2021-06-28  229.69   2021-06-29  226.81  1     90.3    -260.29   alpha_reversal
2021-07-01  226.09   2021-07-07  214.78  3     98.1    -1109.35  stop_loss
2021-07-08  217.71   2021-07-20  220.06  8     88.7    +207.94   alpha_reversal
2021-08-09  238.04   2021-08-10  236.54  1     94.4    -141.07   alpha_reversal
2021-08-11  236.06   2021-08-17  221.79  4     100.4   -1432.23  stop_loss
2021-08-18  229.78   2021-08-19  224.38  1     88.2    -476.56   alpha_reversal
2021-08-20  226.87   2021-08-25  236.95  3     90.0    +906.87   alpha_reversal
2021-09-21  246.58   2021-09-24  258.00  3     105.1   +1200.05  alpha_reversal
2021-10-04  260.64   2021-10-05  260.07  1     100.7   -57.75    alpha_reversal
2021-10-06  261.05   2021-10-07  264.40  1     102.8   +345.18   alpha_reversal
2021-10-27  346.13   2021-10-28  358.83  1     61.7    +783.98   alpha_reversal
2021-11-09  341.34   2021-11-17  362.82  6     40.7    +874.79   alpha_reversal
2021-12-07  350.76   2021-12-09  334.43  2     35.8    -584.46   alpha_reversal
2021-12-10  339.18   2021-12-16  308.82  4     37.2    -1129.66  stop_loss
2021-12-17  311.01   2021-12-20  299.83  1     37.4    -418.59   alpha_reversal
2021-12-21  313.00   2021-12-23  355.49  2     38.3    +1627.85  alpha_reversal
2021-12-31  352.44   2022-01-04  383.01  2     41.3    +1262.82  alpha_reversal
2022-01-05  362.89   2022-01-18  343.33  8     34.9    -683.43   alpha_reversal
2022-02-01  310.57   2022-02-04  307.62  3     33.2    -97.89    alpha_reversal
2022-02-08  307.49   2022-02-09  310.51  1     36.7    +110.91   alpha_reversal
2022-02-14  292.07   2022-02-15  307.32  1     38.9    +593.33   alpha_reversal
2022-02-24  267.06   2022-03-01  287.98  3     38.0    +794.25   alpha_reversal
2022-03-16  280.22   2022-03-22  331.16  4     43.5    +2215.23  alpha_reversal
2022-04-01  361.71   2022-04-04  381.63  1     48.7    +970.52   alpha_reversal
2022-04-11  325.47   2022-04-19  342.55  5     45.7    +779.67   alpha_reversal
2022-04-27  293.98   2022-04-28  292.36  1     41.5    -67.54    alpha_reversal
2022-04-29  290.40   2022-05-04  317.38  3     40.0    +1078.10  alpha_reversal
2022-05-13  256.66   2022-05-20  221.19  5     37.1    -1317.58  stop_loss
2022-05-23  225.08   2022-06-02  258.20  7     38.6    +1278.47  alpha_reversal
2022-06-06  238.40   2022-06-09  239.59  3     41.3    +49.07    alpha_reversal
2022-06-24  245.83   2022-07-05  232.95  6     47.4    -610.81   alpha_reversal
2022-07-07  244.67   2022-07-08  250.64  1     53.9    +322.16   alpha_reversal
2022-07-26  258.99   2022-08-01  297.13  4     59.8    +2279.49  alpha_reversal
2022-08-08  290.57   2022-08-22  289.77  10    56.0    -44.83    max_holding
2022-08-23  296.60   2022-08-24  296.95  1     63.6    +22.05    alpha_reversal
2022-08-26  288.23   2022-09-02  270.07  5     64.9    -1177.69  alpha_reversal
2022-09-06  274.56   2022-09-07  283.56  1     65.8    +592.57   alpha_reversal
2022-09-23  275.47   2022-09-27  282.80  2     66.8    +489.63   alpha_reversal
2022-09-30  265.38   2022-10-03  242.28  1     62.2    -1436.75  stop_loss
2022-10-04  249.56   2022-10-05  240.69  1     57.1    -506.40   alpha_reversal
2022-10-06  238.25   2022-10-11  216.39  3     57.4    -1255.18  stop_loss
2022-10-12  217.35   2022-10-26  224.53  10    60.1    +431.39   max_holding
2022-10-27  225.20   2022-10-28  228.41  1     57.3    +183.54   alpha_reversal
2022-11-03  215.42   2022-11-04  207.37  1     60.8    -489.79   alpha_reversal
2022-11-07  197.18   2022-11-21  167.79  10    58.3    -1712.20  stop_loss
2022-11-22  169.99   2022-11-29  180.74  4     66.1    +709.93   alpha_reversal
2022-12-08  173.53   2022-12-12  167.74  2     76.5    -442.93   alpha_reversal
2022-12-13  161.03   2022-12-16  150.15  3     71.6    -778.36   alpha_reversal
2022-12-19  149.94   2022-12-22  125.29  3     74.5    -1836.36  stop_loss
2022-12-23  123.21   2022-12-29  121.76  3     73.3    -106.45   alpha_reversal
2023-01-04  113.70   2023-01-05  110.28  1     72.1    -245.92   alpha_reversal
2023-02-14  209.35   2023-02-15  214.13  1     62.9    +300.76   alpha_reversal
2023-02-17  208.41   2023-02-21  197.27  1     63.0    -702.02   alpha_reversal
2023-02-27  207.73   2023-02-28  205.61  1     64.4    -137.03   alpha_reversal
2023-03-01  202.87   2023-03-08  181.91  5     67.0    -1405.09  stop_loss
2023-03-09  173.01   2023-03-13  174.39  2     68.2    +94.49    alpha_reversal
2023-03-27  191.91   2023-03-28  189.10  1     73.7    -207.16   alpha_reversal
2023-03-29  193.98   2023-03-31  207.36  2     77.2    +1032.63  alpha_reversal
2023-04-04  192.68   2023-04-19  180.50  10    76.5    -931.29   max_holding
2023-04-25  160.75   2023-05-04  161.12  7     88.2    +32.57    alpha_reversal
2023-05-10  168.62   2023-05-12  167.90  2     100.9   -73.48    alpha_reversal
2023-05-15  166.43   2023-05-17  173.77  2     100.4   +736.85   alpha_reversal
2023-05-24  182.99   2023-05-26  193.07  2     102.6   +1034.01  alpha_reversal
2023-06-16  260.67   2023-06-20  274.31  1     79.2    +1080.78  alpha_reversal
2023-06-22  264.74   2023-06-23  256.47  1     68.9    -569.97   alpha_reversal
2023-06-27  250.34   2023-06-30  261.64  3     65.9    +745.24   alpha_reversal
2023-07-07  274.57   2023-07-12  271.85  3     69.5    -188.61   alpha_reversal
2023-07-25  265.41   2023-08-01  260.94  5     64.8    -289.66   alpha_reversal
2023-08-02  254.24   2023-08-04  253.73  2     69.8    -35.17    alpha_reversal
2023-08-07  251.58   2023-08-15  232.84  6     70.0    -1310.39  stop_loss
2023-08-16  225.71   2023-08-22  233.07  4     75.7    +557.48   alpha_reversal
2023-09-05  256.62   2023-09-06  251.79  1     67.7    -326.45   alpha_reversal
2023-09-11  273.72   2023-09-12  267.35  1     61.9    -394.06   alpha_reversal
2023-09-19  266.63   2023-09-20  262.46  1     68.9    -287.52   alpha_reversal
2023-09-25  247.11   2023-09-26  244.00  1     69.0    -214.84   alpha_reversal
2023-09-27  240.62   2023-10-02  251.47  3     69.9    +758.85   alpha_reversal
2023-10-10  263.75   2023-10-17  254.72  5     71.3    -644.00   alpha_reversal
2023-10-25  212.53   2023-10-30  197.26  3     66.6    -1017.06  alpha_reversal
2023-10-31  200.94   2023-11-14  237.29  10    66.8    +2428.15  alpha_reversal
2023-11-24  235.57   2023-11-27  235.96  1     70.7    +27.87    alpha_reversal
2023-11-30  240.20   2023-12-08  243.72  6     72.5    +255.22   alpha_reversal
2023-12-12  237.13   2023-12-20  247.02  6     82.9    +819.23   alpha_reversal
2023-12-26  256.74   2023-12-27  261.31  1     85.7    +391.89   alpha_reversal
2024-01-02  248.54   2024-01-09  234.84  5     86.8    -1189.34  stop_loss
2024-01-10  234.06   2024-01-12  218.78  2     93.6    -1429.54  stop_loss
2024-01-16  220.02   2024-01-25  182.54  7     87.6    -3284.74  stop_loss
2024-01-30  191.69   2024-02-02  187.82  3     79.5    -307.65   alpha_reversal
2024-02-06  185.19   2024-02-09  193.47  3     79.8    +660.86   alpha_reversal
2024-02-13  184.11   2024-02-22  197.31  6     88.4    +1166.45  alpha_reversal
2024-02-26  199.50   2024-03-04  188.05  5     89.2    -1021.43  alpha_reversal
2024-03-05  180.83   2024-03-14  162.42  7     84.1    -1548.37  stop_loss
2024-03-15  163.65   2024-03-21  172.73  4     86.9    +789.52   alpha_reversal
2024-03-22  170.92   2024-03-26  177.58  2     91.2    +607.81   alpha_reversal
2024-04-01  175.31   2024-04-05  164.82  4     94.9    -995.16   alpha_reversal
2024-04-08  173.07   2024-04-10  171.67  2     87.3    -121.50   alpha_reversal
2024-04-16  157.19   2024-04-22  141.98  4     90.5    -1376.13  stop_loss
2024-05-07  177.90   2024-05-17  177.37  8     75.4    -39.79    alpha_reversal
2024-05-21  186.69   2024-05-22  180.02  1     87.4    -583.04   alpha_reversal
2024-05-30  178.88   2024-06-13  182.38  10    91.9    +321.68   alpha_reversal
2024-07-12  248.35   2024-07-19  239.08  5     56.8    -526.49   alpha_reversal
2024-07-29  232.22   2024-08-01  216.75  3     52.6    -813.40   alpha_reversal
2024-08-02  207.77   2024-08-06  200.54  2     51.9    -375.74   alpha_reversal
2024-08-08  198.94   2024-08-22  210.55  10    51.1    +593.92   alpha_reversal
2024-08-26  213.32   2024-09-05  230.05  7     61.3    +1026.10  alpha_reversal
2024-09-09  216.38   2024-09-10  226.06  1     59.3    +574.38   alpha_reversal
2024-09-13  230.41   2024-09-16  226.67  1     63.7    -238.05   alpha_reversal
2024-09-17  227.98   2024-09-20  238.13  3     66.5    +674.48   alpha_reversal
2024-09-23  250.12   2024-09-24  254.14  1     65.3    +262.52   alpha_reversal
2024-09-27  260.59   2024-09-30  261.50  1     71.1    +64.58    alpha_reversal
2024-10-03  240.78   2024-10-09  240.93  4     65.2    +9.72     alpha_reversal
2024-10-10  238.89   2024-10-11  217.69  1     69.2    -1466.98  stop_loss
2024-10-14  219.27   2024-10-25  269.06  9     63.2    +3146.46  alpha_reversal
2024-11-01  249.10   2024-11-07  296.76  4     66.1    +3149.76  alpha_reversal
2024-11-21  339.81   2024-11-25  338.42  2     43.2    -60.04    alpha_reversal
2024-11-26  338.40   2024-12-05  369.31  6     42.0    +1299.10  alpha_reversal
2024-12-19  436.39   2024-12-31  403.64  7     32.3    -1056.84  trailing_stop
2025-01-02  379.47   2025-01-10  394.54  5     31.1    +468.58   alpha_reversal
2025-01-15  428.43   2025-01-16  413.61  1     30.0    -445.15   alpha_reversal
2025-01-17  426.71   2025-01-29  388.91  7     30.2    -1143.24  stop_loss
2025-02-04  392.41   2025-02-05  377.98  1     34.3    -495.42   alpha_reversal
2025-02-06  374.51   2025-02-11  328.34  3     35.4    -1635.47  stop_loss
2025-02-12  336.68   2025-02-20  354.22  5     35.5    +622.84   alpha_reversal
2025-02-26  290.95   2025-03-10  222.04  8     36.5    -2517.47  stop_loss
2025-03-11  230.70   2025-03-12  247.97  1     33.5    +579.41   alpha_reversal
2025-03-14  250.10   2025-03-25  288.00  7     35.5    +1344.55  alpha_reversal
2025-04-11  252.44   2025-04-15  253.98  2     28.7    +44.34    alpha_reversal
2025-05-05  280.40   2025-05-12  318.22  5     37.4    +1413.80  alpha_reversal
2025-05-20  343.99   2025-05-21  334.45  1     43.0    -409.75   alpha_reversal
2025-05-27  363.07   2025-05-28  356.72  1     42.9    -272.71   alpha_reversal
2025-06-03  344.44   2025-06-05  284.56  2     45.6    -2731.71  stop_loss
2025-06-06  295.29   2025-06-13  325.15  5     36.6    +1092.91  alpha_reversal
2025-06-18  322.21   2025-06-24  340.30  3     40.2    +726.56   alpha_reversal
2025-06-25  327.71   2025-07-07  293.79  7     37.9    -1286.79  stop_loss
2025-07-08  297.96   2025-07-14  316.74  4     42.0    +789.47   alpha_reversal
2025-07-30  319.20   2025-08-04  309.11  3     49.7    -502.03   alpha_reversal
2025-08-06  320.07   2025-08-08  329.49  2     53.0    +498.57   alpha_reversal
2025-08-14  335.75   2025-08-19  329.15  3     55.5    -366.51   alpha_reversal
2025-08-20  324.06   2025-08-26  351.49  4     55.8    +1531.84  alpha_reversal
2025-09-02  329.52   2025-09-08  346.23  4     58.5    +977.19   alpha_reversal
2025-10-03  430.04   2025-10-10  413.28  5     39.2    -657.67   trailing_stop
2025-10-15  435.37   2025-10-16  428.54  1     37.7    -257.41   alpha_reversal
2025-10-31  456.79   2025-11-03  468.14  1     37.9    +430.35   alpha_reversal
2025-11-10  445.45   2025-11-13  401.79  3     34.4    -1501.55  stop_loss
2025-11-14  404.55   2025-11-18  401.05  2     32.8    -114.86   alpha_reversal
2025-12-02  429.45   2025-12-05  454.77  3     37.9    +958.52   alpha_reversal
2025-12-12  459.19   2025-12-15  475.07  1     42.1    +667.95   alpha_reversal
2025-12-24  485.64   2025-12-30  454.20  3     40.3    -1268.51  stop_loss
2025-12-31  449.94   2026-01-15  438.35  10    43.7    -506.91   max_holding
2026-01-16  437.72   2026-01-23  448.84  4     47.6    +528.76   alpha_reversal
2026-02-03  422.17   2026-02-05  397.01  2     45.9    -1154.50  stop_loss
2026-02-09  417.53   2026-02-12  416.86  3     43.4    -28.93    alpha_reversal
2026-02-24  409.58   2026-03-05  405.35  7     47.4    -200.69   alpha_reversal
2026-03-06  396.93   2026-03-10  399.04  2     49.7    +105.06   alpha_reversal
2026-03-19  380.49   2026-03-24  382.84  3     53.0    +124.55   alpha_reversal

**Best 3 trades:**

- 2024-11-07: P&L = **+3149.76** (alpha_reversal)
- 2024-10-25: P&L = **+3146.46** (alpha_reversal)
- 2020-11-18: P&L = **+2924.03** (alpha_reversal)

**Worst 3 trades:**

- 2024-01-25: P&L = **-3284.74** (stop_loss)
- 2025-06-05: P&L = **-2731.71** (stop_loss)
- 2025-03-10: P&L = **-2517.47** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  97,751.53
2017-03-23  100,802.79
2017-09-20  108,508.55
2018-03-21  104,377.03
2018-09-18  100,458.01
2019-03-20  99,674.25
2019-09-17  98,277.74
2020-03-17  106,349.58
2020-09-14  108,177.44
2021-03-15  112,193.40
2021-09-10  106,624.97
2022-03-10  111,246.34
2022-09-08  117,615.20
2023-03-09  108,670.08
2023-09-07  110,057.86
2024-03-07  106,592.83
2024-09-05  103,742.66
2025-03-07  106,433.82
2025-09-05  108,648.43
2026-03-06  105,263.21

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -2.80%
2017-03-23  -1.29%
2017-09-20  -0.03%
2018-03-21  -3.83%
2018-09-18  -7.44%
2019-03-20  -8.16%
2019-09-17  -9.45%
2020-03-17  -2.01%
2020-09-14  -2.23%
2021-03-15  -1.89%
2021-09-10  -6.76%
2022-03-10  -2.72%
2022-09-08  -1.39%
2023-03-09  -8.89%
2023-09-07  -7.73%
2024-03-07  -10.63%
2024-09-05  -13.02%
2025-03-07  -10.77%
2025-09-05  -8.91%
2026-03-06  -11.75%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  25.30%
Out-of-Sample (30%)  2023-03-24  2026-03-27  2.71%

#### Return Distribution

Return Bin          Count
-2.036% to -1.545%  8
-1.545% to -1.055%  10
-1.055% to -0.565%  77
-0.565% to -0.075%  389
-0.075% to 0.415%   1830
0.415% to 0.905%    162
0.905% to 1.396%    29
1.396% to 1.886%    8
1.886% to 2.376%    2
2.376% to 2.866%    1

### MRK — AlphaCombined

**Net Return (after slippage):** 21.43%  **vs SPY (exposure-adj): -214.04%** (underperform)  
**Gross Return (pre-cost):** 36.03%  
**Total Slippage Cost:** $14,605.74  
**Trade Count:** 337  
**Win Rate:** 55.2%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-02  38.81    2016-05-05  37.92   3     1177.1  -1044.25  stop_loss
2016-05-06  37.62    2016-05-11  38.08   3     1115.3  +513.25   alpha_reversal
2016-05-12  37.98    2016-05-17  38.08   3     1186.1  +121.26   alpha_reversal
2016-05-18  38.37    2016-05-19  38.27   1     1176.8  -119.38   alpha_reversal
2016-05-24  39.02    2016-05-25  39.66   1     1170.5  +750.35   alpha_reversal
2016-05-27  39.64    2016-06-02  39.84   3     1219.0  +242.29   alpha_reversal
2016-06-06  40.12    2016-06-15  39.65   7     1297.5  -617.96   alpha_reversal
2016-06-20  39.72    2016-06-23  40.77   3     1129.9  +1185.08  alpha_reversal
2016-06-30  40.76    2016-07-05  41.00   2     1050.0  +246.67   alpha_reversal
2016-07-12  42.17    2016-07-26  41.25   10    1138.2  -1045.58  stop_loss
2016-07-27  41.46    2016-08-05  45.14   7     1258.5  +4626.93  alpha_reversal
2016-08-15  44.80    2016-08-23  44.94   6     924.0   +128.44   alpha_reversal
2016-08-26  44.47    2016-09-09  44.17   9     1079.9  -322.77   alpha_reversal
2016-09-12  44.71    2016-09-13  44.01   1     1105.5  -773.55   alpha_reversal
2016-09-15  44.46    2016-09-22  44.87   5     1000.6  +411.49   alpha_reversal
2016-09-27  44.59    2016-09-29  44.08   2     1046.6  -538.45   alpha_reversal
2016-09-30  44.48    2016-10-03  44.51   1     986.2   +33.40    alpha_reversal
2016-10-05  44.76    2016-10-10  45.50   3     1020.1  +753.35   alpha_reversal
2016-10-12  43.95    2016-10-14  44.24   2     877.6   +261.38   alpha_reversal
2016-10-17  43.85    2016-10-21  43.57   4     878.7   -238.73   alpha_reversal
2016-10-24  43.30    2016-10-26  43.34   2     967.2   +40.79    alpha_reversal
2016-10-27  43.68    2016-10-28  41.89   1     850.9   -1521.52  stop_loss
2016-10-31  41.85    2016-11-02  41.89   2     780.5   +34.04    alpha_reversal
2016-11-03  41.64    2016-11-07  42.76   2     822.7   +914.72   alpha_reversal
2016-11-17  44.69    2016-12-01  43.26   9     762.9   -1087.83  stop_loss
2016-12-06  42.96    2016-12-15  44.75   7     851.7   +1521.01  alpha_reversal
2016-12-20  43.45    2017-01-04  43.14   9     858.2   -265.10   alpha_reversal
2017-01-05  43.17    2017-01-12  44.63   5     1014.0  +1484.02  alpha_reversal
2017-01-25  43.87    2017-02-01  44.55   5     851.1   +585.54   alpha_reversal
2017-02-13  46.52    2017-02-24  47.47   8     998.0   +948.91   alpha_reversal
2017-02-28  47.31    2017-03-01  47.50   1     1151.5  +226.42   alpha_reversal
2017-03-13  46.40    2017-03-21  46.18   6     1144.1  -251.48   alpha_reversal
2017-03-23  45.77    2017-03-30  45.85   5     1232.7  +95.03    alpha_reversal
2017-03-31  45.96    2017-04-03  45.87   1     1399.2  -135.04   alpha_reversal
2017-04-04  46.06    2017-04-05  45.94   1     1473.1  -174.26   alpha_reversal
2017-04-06  45.75    2017-04-12  45.56   4     1458.2  -266.88   alpha_reversal
2017-04-17  45.43    2017-04-18  45.00   1     1513.9  -648.55   alpha_reversal
2017-04-21  44.77    2017-05-02  45.31   7     1489.9  +805.46   alpha_reversal
2017-05-16  46.14    2017-05-17  45.53   1     1278.1  -788.63   alpha_reversal
2017-05-22  46.32    2017-05-23  46.65   1     1191.2  +383.89   alpha_reversal
2017-05-30  46.93    2017-06-07  46.29   6     1338.0  -855.63   trailing_stop
2017-06-08  45.72    2017-06-15  46.00   5     1328.3  +375.73   alpha_reversal
2017-06-19  46.40    2017-06-21  47.65   2     1275.4  +1593.54  alpha_reversal
2017-06-29  46.88    2017-07-05  46.71   3     1215.0  -216.13   alpha_reversal
2017-07-06  45.98    2017-07-20  45.82   10    1177.9  -191.33   max_holding
2017-07-26  45.03    2017-07-28  46.67   2     1446.3  +2367.05  alpha_reversal
2017-08-01  46.58    2017-08-02  46.18   1     1222.0  -483.90   alpha_reversal
2017-08-07  45.79    2017-08-18  44.76   9     1296.0  -1332.97  stop_loss
2017-09-05  46.36    2017-09-12  47.65   5     1509.1  +1951.41  alpha_reversal
2017-09-14  48.53    2017-09-15  48.51   1     1473.4  -28.25    alpha_reversal
2017-09-19  48.40    2017-09-26  47.48   5     1522.5  -1402.11  stop_loss
2017-09-27  47.39    2017-10-11  46.77   10    1571.5  -984.71   alpha_reversal
2017-10-12  46.85    2017-10-20  46.84   6     1637.4  -16.65    alpha_reversal
2017-10-24  46.32    2017-10-26  45.45   2     1468.3  -1273.78  stop_loss
2017-10-27  42.75    2017-10-30  40.11   1     1027.9  -2704.36  stop_loss
2017-10-31  40.43    2017-11-09  41.21   7     823.9   +643.31   alpha_reversal
2017-11-13  40.44    2017-11-20  39.67   5     943.1   -729.64   alpha_reversal
2017-11-21  39.83    2017-11-24  39.85   2     1002.6  +18.90    alpha_reversal
2017-11-27  40.04    2017-11-30  40.53   3     1121.1  +547.00   alpha_reversal
2017-12-01  41.01    2017-12-06  39.85   3     1151.5  -1330.55  stop_loss
2017-12-07  40.20    2017-12-08  40.75   1     1003.8  +548.45   alpha_reversal
2017-12-19  41.62    2017-12-20  41.49   1     1113.9  -145.18   alpha_reversal
2017-12-21  41.90    2017-12-29  41.61   5     1153.2  -329.68   alpha_reversal
2018-01-03  41.56    2018-01-17  45.87   9     1344.3  +5798.98  alpha_reversal
2018-01-18  45.25    2018-01-22  45.29   2     913.5   +39.74    alpha_reversal
2018-01-24  45.29    2018-01-31  43.81   5     927.6   -1365.90  trailing_stop
2018-02-01  44.31    2018-02-05  41.71   2     909.0   -2365.88  stop_loss
2018-02-07  41.36    2018-02-09  40.57   2     695.4   -548.14   alpha_reversal
2018-02-12  41.03    2018-02-27  40.46   10    668.9   -378.60   max_holding
2018-02-28  40.13    2018-03-05  40.23   3     797.9   +74.20    alpha_reversal
2018-03-06  40.19    2018-03-09  40.77   3     823.0   +478.12   alpha_reversal
2018-03-28  41.13    2018-03-29  40.63   1     877.3   -441.80   alpha_reversal
2018-04-02  39.78    2018-04-04  40.68   2     844.3   +766.23   alpha_reversal
2018-04-06  39.84    2018-04-10  42.14   2     809.5   +1857.78  alpha_reversal
2018-05-02  42.78    2018-05-16  44.57   10    782.6   +1396.71  max_holding
2018-05-22  43.64    2018-05-30  44.55   5     833.5   +753.25   alpha_reversal
2018-06-01  45.22    2018-06-04  46.26   1     902.2   +941.77   alpha_reversal
2018-06-12  46.62    2018-06-13  46.69   1     980.4   +64.01    alpha_reversal
2018-06-15  46.67    2018-06-18  45.85   1     1001.5  -829.68   alpha_reversal
2018-06-19  45.92    2018-06-26  45.64   5     983.3   -274.28   alpha_reversal
2018-07-03  45.68    2018-07-05  46.33   1     1054.1  +688.78   alpha_reversal
2018-07-10  46.88    2018-07-16  47.05   4     1084.1  +185.54   alpha_reversal
2018-07-17  47.05    2018-07-19  46.99   2     1178.5  -73.14    alpha_reversal
2018-07-20  47.04    2018-07-26  48.12   4     1242.3  +1332.98  alpha_reversal
2018-07-27  47.77    2018-07-31  49.51   2     1130.2  +1968.08  alpha_reversal
2018-08-08  50.18    2018-08-13  50.15   3     1110.3  -30.63    alpha_reversal
2018-08-15  50.69    2018-08-16  51.75   1     1108.7  +1177.31  alpha_reversal
2018-08-23  51.90    2018-09-06  52.15   9     1120.4  +287.20   alpha_reversal
2018-09-20  53.62    2018-09-24  53.57   2     1137.3  -60.95    alpha_reversal
2018-10-04  53.95    2018-10-09  54.75   3     1094.5  +868.74   alpha_reversal
2018-10-16  53.95    2018-10-19  54.76   3     787.8   +631.22   alpha_reversal
2018-10-25  53.00    2018-10-29  54.08   2     652.1   +700.78   alpha_reversal
2018-11-06  55.54    2018-11-08  57.13   2     588.1   +933.13   alpha_reversal
2018-11-15  56.70    2018-11-16  57.56   1     643.7   +557.89   alpha_reversal
2018-11-26  57.15    2018-11-27  57.78   1     699.9   +436.74   alpha_reversal
2018-11-29  59.02    2018-11-30  60.05   1     719.7   +736.48   alpha_reversal
2018-12-19  56.28    2018-12-24  54.23   3     594.6   -1220.64  stop_loss
2019-01-03  56.48    2019-01-17  57.62   10    545.1   +617.33   max_holding
2019-01-22  57.85    2019-01-24  55.77   2     663.8   -1384.00  stop_loss
2019-01-25  55.65    2019-02-04  58.58   6     622.9   +1826.41  alpha_reversal
2019-02-08  59.14    2019-02-11  58.46   1     694.0   -469.46   alpha_reversal
2019-02-15  60.89    2019-02-21  60.84   3     721.6   -32.91    alpha_reversal
2019-02-26  61.60    2019-02-27  61.44   1     812.9   -124.39   alpha_reversal
2019-02-28  62.02    2019-03-06  61.55   4     831.7   -387.52   alpha_reversal
2019-03-11  61.70    2019-03-18  62.42   5     832.9   +603.78   alpha_reversal
2019-03-19  62.91    2019-03-21  63.65   2     918.3   +675.04   alpha_reversal
2019-04-01  63.98    2019-04-05  62.27   4     961.4   -1647.46  stop_loss
2019-04-08  62.18    2019-04-15  60.26   5     930.5   -1785.71  stop_loss
2019-04-16  59.57    2019-04-17  56.72   1     885.8   -2526.66  stop_loss
2019-04-18  56.21    2019-04-29  58.91   6     719.7   +1942.07  alpha_reversal
2019-05-13  59.27    2019-05-14  59.40   1     708.9   +93.98    alpha_reversal
2019-05-29  60.95    2019-05-31  60.78   2     726.6   -127.89   alpha_reversal
2019-06-03  61.65    2019-06-11  63.67   6     746.8   +1512.65  alpha_reversal
2019-06-14  64.01    2019-06-17  64.34   1     793.6   +268.00   alpha_reversal
2019-06-21  65.39    2019-06-25  65.84   2     770.1   +348.20   alpha_reversal
2019-06-27  64.82    2019-06-28  64.77   1     731.9   -36.11    alpha_reversal
2019-07-02  66.12    2019-07-11  62.57   6     737.7   -2618.67  trailing_stop
2019-07-12  61.65    2019-07-19  62.87   5     585.7   +714.93   alpha_reversal
2019-07-23  63.43    2019-07-30  64.32   5     662.2   +592.24   alpha_reversal
2019-08-21  67.20    2019-08-22  66.98   1     634.5   -135.72   alpha_reversal
2019-08-23  65.68    2019-08-27  66.06   2     624.9   +238.94   alpha_reversal
2019-08-28  66.87    2019-08-29  66.98   1     647.3   +76.74    alpha_reversal
2019-09-03  67.00    2019-09-04  66.47   1     690.9   -366.49   alpha_reversal
2019-09-05  66.57    2019-09-09  64.47   2     688.9   -1445.27  stop_loss
2019-09-10  63.16    2019-09-19  65.29   7     552.3   +1175.83  alpha_reversal
2019-09-26  65.29    2019-09-27  64.47   1     671.4   -550.18   alpha_reversal
2019-10-08  64.83    2019-10-10  65.13   2     622.2   +187.08   alpha_reversal
2019-10-17  65.22    2019-10-22  63.22   3     737.7   -1482.05  stop_loss
2019-10-23  64.50    2019-10-30  67.04   5     635.3   +1613.77  alpha_reversal
2019-11-06  64.80    2019-11-14  65.74   6     577.4   +546.28   alpha_reversal
2019-11-25  66.70    2019-11-27  68.12   2     725.2   +1028.65  alpha_reversal
2019-12-12  69.72    2019-12-16  69.90   2     880.2   +166.03   alpha_reversal
2019-12-18  70.23    2019-12-19  70.45   1     914.8   +200.77   alpha_reversal
2019-12-27  71.71    2019-12-30  71.27   1     980.6   -431.09   alpha_reversal
2019-12-31  71.28    2020-01-07  69.83   4     1010.8  -1456.85  trailing_stop
2020-01-08  69.43    2020-01-16  71.38   6     850.1   +1658.10  alpha_reversal
2020-01-17  71.29    2020-01-23  69.33   3     897.6   -1757.45  stop_loss
2020-01-24  67.38    2020-01-28  67.52   2     769.3   +110.80   alpha_reversal
2020-01-29  68.37    2020-02-03  68.38   3     742.2   +7.39     alpha_reversal
2020-02-06  67.15    2020-02-12  65.18   4     605.3   -1192.09  stop_loss
2020-02-13  64.24    2020-02-25  62.84   7     611.6   -853.19   alpha_reversal
2020-03-02  63.77    2020-03-04  64.96   2     461.4   +548.59   alpha_reversal
2020-03-20  56.39    2020-03-23  52.41   1     241.5   -959.23   alpha_reversal
2020-03-25  53.90    2020-04-03  60.19   7     234.6   +1474.51  alpha_reversal
2020-04-08  64.49    2020-04-17  65.88   6     244.8   +339.74   alpha_reversal
2020-04-22  63.20    2020-04-24  64.28   2     289.0   +312.51   alpha_reversal
2020-05-04  60.73    2020-05-06  60.88   2     328.7   +47.51    alpha_reversal
2020-05-08  60.37    2020-05-14  63.19   4     370.9   +1046.33  alpha_reversal
2020-05-19  61.28    2020-05-27  61.21   5     406.2   -24.88    alpha_reversal
2020-06-02  63.88    2020-06-04  64.37   2     460.5   +228.68   alpha_reversal
2020-06-05  65.00    2020-06-11  61.06   4     490.4   -1932.36  stop_loss
2020-06-15  58.95    2020-06-17  60.70   2     439.5   +767.94   alpha_reversal
2020-06-18  60.70    2020-06-22  61.06   2     450.2   +162.53   alpha_reversal
2020-06-23  61.42    2020-06-26  59.82   3     464.1   -741.19   alpha_reversal
2020-07-09  61.08    2020-07-13  61.54   2     536.1   +248.80   alpha_reversal
2020-07-15  63.27    2020-07-28  63.40   9     563.5   +76.45    alpha_reversal
2020-08-06  64.55    2020-08-14  66.42   6     543.8   +1016.29  alpha_reversal
2020-08-25  68.24    2020-08-26  68.06   1     678.6   -121.89   alpha_reversal
2020-08-27  68.35    2020-09-03  68.02   5     703.4   -232.71   alpha_reversal
2020-09-04  67.89    2020-09-10  66.49   3     589.2   -822.81   alpha_reversal
2020-09-23  66.29    2020-10-01  65.43   6     549.3   -472.21   alpha_reversal
2020-10-02  64.82    2020-10-16  63.98   10    578.3   -487.01   max_holding
2020-10-21  62.63    2020-10-23  63.98   2     687.3   +926.43   alpha_reversal
2020-11-02  61.54    2020-11-05  64.48   3     611.0   +1798.80  alpha_reversal
2020-11-06  64.47    2020-11-20  64.47   10    509.6   +3.91     max_holding
2020-11-24  64.29    2020-12-01  65.36   4     526.2   +560.84   alpha_reversal
2020-12-04  65.73    2020-12-10  66.51   4     609.7   +473.01   alpha_reversal
2020-12-15  65.07    2020-12-23  64.43   6     622.7   -402.61   alpha_reversal
2020-12-31  66.14    2021-01-04  65.40   1     695.6   -517.98   alpha_reversal
2021-01-15  67.42    2021-01-26  64.82   6     580.2   -1505.99  stop_loss
2021-01-27  62.32    2021-01-29  62.25   2     590.7   -36.79    alpha_reversal
2021-02-10  60.55    2021-02-18  60.91   5     647.7   +232.84   alpha_reversal
2021-02-22  60.59    2021-02-23  60.21   1     705.6   -264.99   alpha_reversal
2021-03-01  58.52    2021-03-03  59.21   2     703.5   +481.67   alpha_reversal
2021-03-05  59.13    2021-03-09  59.89   2     703.8   +532.56   alpha_reversal
2021-03-10  60.45    2021-03-16  62.59   4     682.1   +1460.00  alpha_reversal
2021-03-26  63.12    2021-03-30  62.71   2     673.8   -278.59   alpha_reversal
2021-04-05  62.81    2021-04-19  63.28   10    719.1   +341.60   max_holding
2021-04-23  63.52    2021-04-29  60.04   4     835.8   -2913.32  stop_loss
2021-05-17  65.15    2021-05-21  64.52   4     700.8   -439.60   alpha_reversal
2021-05-25  63.21    2021-06-01  61.26   4     719.7   -1405.96  stop_loss
2021-06-02  61.88    2021-06-04  63.29   2     760.3   +1066.48  alpha_reversal
2021-06-08  61.89    2021-06-10  65.02   2     765.3   +2396.82  alpha_reversal
2021-06-15  65.27    2021-06-16  65.98   1     721.9   +518.72   alpha_reversal
2021-06-23  65.16    2021-06-25  66.49   2     768.2   +1021.82  alpha_reversal
2021-06-29  66.57    2021-06-30  66.98   1     790.2   +328.54   alpha_reversal
2021-07-07  67.73    2021-07-08  67.28   1     872.8   -389.85   alpha_reversal
2021-07-12  66.86    2021-07-14  66.85   2     875.9   -5.73     alpha_reversal
2021-07-21  65.86    2021-07-28  67.47   5     819.8   +1315.92  alpha_reversal
2021-08-05  65.17    2021-08-13  66.08   6     875.4   +794.98   alpha_reversal
2021-08-24  67.01    2021-09-01  65.61   6     902.0   -1256.78  stop_loss
2021-09-02  66.49    2021-09-03  66.54   1     921.7   +49.89    alpha_reversal
2021-09-08  64.97    2021-09-10  63.26   2     863.4   -1476.39  stop_loss
2021-09-13  63.09    2021-09-21  62.54   6     822.7   -452.22   alpha_reversal
2021-09-28  63.78    2021-09-30  65.27   2     767.1   +1144.32  alpha_reversal
2021-10-12  69.23    2021-10-22  70.52   8     497.8   +640.46   alpha_reversal
2021-11-08  71.92    2021-11-17  71.78   7     368.0   -52.04    alpha_reversal
2021-11-26  68.86    2021-11-29  65.08   1     425.2   -1607.02  stop_loss
2021-11-30  65.16    2021-12-01  64.69   1     394.6   -186.89   alpha_reversal
2021-12-02  64.28    2021-12-14  64.67   8     400.9   +158.69   alpha_reversal
2021-12-22  66.88    2021-12-23  66.44   1     504.5   -224.03   alpha_reversal
2021-12-27  67.24    2021-12-28  67.38   1     539.1   +77.28    alpha_reversal
2022-01-03  67.50    2022-01-06  69.15   3     615.1   +1016.07  alpha_reversal
2022-02-09  67.20    2022-02-14  67.08   3     525.7   -62.98    alpha_reversal
2022-02-15  68.33    2022-02-22  66.63   4     542.1   -921.52   alpha_reversal
2022-02-23  66.60    2022-02-24  64.49   1     608.6   -1284.58  stop_loss
2022-02-25  67.02    2022-03-04  68.28   5     508.7   +639.77   alpha_reversal
2022-03-17  69.94    2022-03-21  69.97   2     559.4   +20.31    alpha_reversal
2022-03-23  70.64    2022-03-24  71.17   1     588.9   +312.87   alpha_reversal
2022-03-29  72.27    2022-03-30  72.93   1     647.6   +428.94   alpha_reversal
2022-04-05  74.17    2022-04-06  75.20   1     622.5   +642.56   alpha_reversal
2022-04-19  76.01    2022-04-27  74.71   6     553.4   -717.90   alpha_reversal
2022-05-16  81.79    2022-05-17  82.31   1     431.5   +224.42   alpha_reversal
2022-05-26  81.78    2022-06-08  79.20   8     437.1   -1130.46  trailing_stop
2022-06-10  77.24    2022-06-14  75.40   2     492.8   -905.25   alpha_reversal
2022-06-16  75.82    2022-06-17  75.51   1     481.2   -152.41   alpha_reversal
2022-06-21  78.63    2022-06-23  82.09   2     436.6   +1512.24  alpha_reversal
2022-07-01  82.55    2022-07-07  82.99   3     416.5   +184.92   alpha_reversal
2022-07-08  82.87    2022-07-12  83.52   2     435.2   +282.40   alpha_reversal
2022-07-19  82.50    2022-07-27  81.41   6     454.6   -495.84   alpha_reversal
2022-08-03  78.26    2022-08-10  79.59   5     452.3   +598.27   alpha_reversal
2022-08-11  79.43    2022-08-15  80.84   2     437.1   +616.69   alpha_reversal
2022-08-16  80.92    2022-08-18  81.56   2     467.4   +300.02   alpha_reversal
2022-08-24  80.40    2022-08-30  77.52   4     526.4   -1512.49  stop_loss
2022-09-01  77.84    2022-09-07  77.51   3     519.4   -170.20   alpha_reversal
2022-09-12  78.74    2022-09-15  78.03   3     539.3   -384.41   alpha_reversal
2022-10-21  86.14    2022-10-25  87.89   2     457.2   +799.59   alpha_reversal
2022-11-02  89.50    2022-11-09  91.38   5     419.7   +789.27   alpha_reversal
2022-11-11  88.20    2022-11-15  89.59   2     383.1   +531.33   alpha_reversal
2022-11-16  89.98    2022-11-17  92.03   1     383.6   +786.78   alpha_reversal
2022-11-25  96.79    2022-11-28  97.55   1     435.0   +329.64   alpha_reversal
2022-12-19  99.19    2022-12-20  99.34   1     463.8   +67.40    alpha_reversal
2022-12-30  100.56   2023-01-03  100.63  1     529.8   +37.90    alpha_reversal
2023-01-05  103.00   2023-01-06  103.98  1     513.8   +505.36   alpha_reversal
2023-01-10  100.43   2023-01-20  99.55   7     448.0   -397.89   alpha_reversal
2023-01-23  99.59    2023-01-26  96.77   3     464.2   -1311.46  stop_loss
2023-01-27  95.51    2023-01-31  97.26   2     429.2   +748.00   alpha_reversal
2023-02-16  96.54    2023-02-21  98.76   2     465.6   +1034.28  alpha_reversal
2023-02-28  96.29    2023-03-02  96.89   2     440.0   +264.45   alpha_reversal
2023-03-06  100.70   2023-03-07  100.75  1     429.8   +22.90    alpha_reversal
2023-03-14  97.63    2023-03-16  97.87   2     362.1   +86.84    alpha_reversal
2023-03-17  95.01    2023-03-21  96.23   2     337.8   +411.37   alpha_reversal
2023-03-22  95.28    2023-03-27  97.49   3     355.0   +785.12   alpha_reversal
2023-03-30  96.67    2023-03-31  97.00   1     393.5   +130.60   alpha_reversal
2023-04-05  102.14   2023-04-06  102.42  1     385.1   +108.15   alpha_reversal
2023-04-20  104.20   2023-04-21  105.19  1     438.6   +434.24   alpha_reversal
2023-05-10  107.60   2023-05-17  104.63  5     449.4   -1335.08  stop_loss
2023-05-18  104.04   2023-05-22  104.39  2     440.5   +150.98   alpha_reversal
2023-05-23  103.38   2023-05-25  102.39  2     430.9   -425.59   alpha_reversal
2023-05-26  101.37   2023-06-01  101.14  3     419.1   -95.96    alpha_reversal
2023-06-02  102.69   2023-06-06  100.30  2     417.7   -998.71   alpha_reversal
2023-06-08  100.69   2023-06-22  104.38  9     366.2   +1351.90  alpha_reversal
2023-07-03  105.04   2023-07-07  100.05  3     403.7   -2017.33  stop_loss
2023-07-10  101.03   2023-07-13  97.59   3     408.6   -1406.30  stop_loss
2023-07-14  98.62    2023-07-21  101.32  5     426.5   +1151.86  alpha_reversal
2023-07-25  98.80    2023-08-03  97.04   7     431.9   -756.13   alpha_reversal
2023-08-07  97.47    2023-08-09  97.38   2     413.7   -36.50    alpha_reversal
2023-08-10  96.99    2023-08-14  99.96   2     442.7   +1314.33  alpha_reversal
2023-08-16  99.90    2023-08-17  99.97   1     469.3   +34.98    alpha_reversal
2023-08-29  101.06   2023-08-30  101.16  1     433.8   +43.79    alpha_reversal
2023-09-01  100.92   2023-09-06  97.74   2     462.0   -1467.03  stop_loss
2023-09-07  99.17    2023-09-08  100.09  1     432.5   +397.76   alpha_reversal
2023-09-12  100.16   2023-09-13  98.95   1     452.9   -548.39   alpha_reversal
2023-09-18  99.74    2023-09-19  99.21   1     471.5   -247.45   alpha_reversal
2023-09-21  98.73    2023-09-27  96.05   4     497.7   -1332.32  stop_loss
2023-10-02  94.86    2023-10-10  95.75   6     505.2   +446.98   alpha_reversal
2023-10-11  95.70    2023-10-13  96.12   2     528.8   +218.20   alpha_reversal
2023-10-19  92.90    2023-10-23  95.51   2     495.6   +1291.34  alpha_reversal
2023-11-03  95.63    2023-11-17  94.03   10    441.3   -706.98   max_holding
2023-11-20  94.98    2023-11-27  93.73   4     474.9   -593.69   alpha_reversal
2023-11-28  92.67    2023-11-30  94.70   2     500.9   +1018.21  alpha_reversal
2023-12-11  96.55    2023-12-19  99.12   6     499.5   +1286.23  alpha_reversal
2023-12-27  100.61   2023-12-28  101.24  1     517.6   +328.57   alpha_reversal
2024-01-12  110.53   2024-01-16  110.29  1     482.2   -116.10   alpha_reversal
2024-01-18  110.45   2024-01-19  110.66  1     482.6   +103.95   alpha_reversal
2024-01-22  111.28   2024-02-02  117.66  9     505.0   +3225.08  alpha_reversal
2024-02-09  116.89   2024-02-14  117.11  3     476.1   +108.34   alpha_reversal
2024-02-27  120.22   2024-02-28  119.32  1     517.4   -466.69   alpha_reversal
2024-02-29  118.47   2024-03-04  115.47  2     512.9   -1540.74  stop_loss
2024-03-05  114.44   2024-03-18  113.75  9     457.1   -315.97   alpha_reversal
2024-03-19  114.58   2024-03-25  117.38  4     435.8   +1219.70  alpha_reversal
2024-04-09  118.81   2024-04-16  117.14  5     391.4   -651.47   alpha_reversal
2024-05-03  119.56   2024-05-10  121.83  5     405.8   +920.74   alpha_reversal
2024-05-13  121.23   2024-05-23  122.79  8     429.7   +672.39   alpha_reversal
2024-05-28  118.23   2024-06-05  121.26  6     428.5   +1297.88  alpha_reversal
2024-06-13  121.30   2024-06-21  123.18  5     406.0   +761.54   alpha_reversal
2024-06-28  116.77   2024-07-12  120.39  9     298.8   +1080.27  alpha_reversal
2024-07-17  118.75   2024-07-25  118.59  6     340.3   -53.21    alpha_reversal
2024-08-13  107.90   2024-08-23  109.87  8     301.9   +596.16   alpha_reversal
2024-09-10  108.78   2024-09-19  111.21  7     349.5   +845.95   alpha_reversal
2024-09-20  111.26   2024-09-26  107.28  4     397.1   -1580.89  stop_loss
2024-09-27  107.96   2024-10-02  106.32  3     418.2   -683.80   alpha_reversal
2024-10-03  104.62   2024-10-17  104.11  10    419.0   -214.71   max_holding
2024-10-18  103.22   2024-10-23  100.91  3     470.0   -1082.88  alpha_reversal
2024-10-24  100.54   2024-10-29  98.40   3     480.6   -1028.37  alpha_reversal
2024-11-06  95.65    2024-11-12  93.51   4     421.1   -899.19   trailing_stop
2024-11-13  93.53    2024-11-19  91.58   4     419.6   -819.47   alpha_reversal
2024-11-20  92.53    2024-11-22  94.08   2     396.4   +617.65   alpha_reversal
2024-11-25  96.06    2024-11-26  96.40   1     369.6   +125.79   alpha_reversal
2024-12-02  95.54    2024-12-10  95.81   6     396.0   +108.68   alpha_reversal
2024-12-12  96.14    2024-12-13  96.76   1     379.6   +233.62   alpha_reversal
2024-12-17  95.77    2024-12-19  95.16   2     392.2   -240.05   alpha_reversal
2024-12-23  95.11    2024-12-24  95.09   1     393.0   -7.30     alpha_reversal
2024-12-30  94.16    2025-01-02  94.83   2     448.0   +300.52   alpha_reversal
2025-01-03  94.89    2025-01-14  95.29   6     481.6   +189.16   alpha_reversal
2025-01-16  96.39    2025-01-17  93.63   1     455.7   -1255.31  stop_loss
2025-01-21  92.12    2025-01-28  93.18   5     426.2   +453.85   alpha_reversal
2025-01-29  94.07    2025-02-03  95.42   3     416.6   +562.33   alpha_reversal
2025-02-10  82.70    2025-02-21  85.58   8     338.9   +976.65   alpha_reversal
2025-03-13  90.65    2025-03-18  91.35   3     356.2   +249.87   alpha_reversal
2025-03-19  90.77    2025-03-21  89.80   2     382.4   -370.33   alpha_reversal
2025-03-24  89.12    2025-03-25  84.75   1     404.8   -1769.58  stop_loss
2025-03-26  85.06    2025-03-31  86.57   3     363.4   +547.37   alpha_reversal
2025-04-03  83.40    2025-04-04  78.57   1     367.7   -1775.40  stop_loss
2025-04-08  76.22    2025-04-23  75.94   10    292.3   -81.46    max_holding
2025-05-02  80.30    2025-05-06  76.23   2     307.1   -1250.72  stop_loss
2025-05-09  73.34    2025-05-14  70.86   3     312.7   -776.93   trailing_stop
2025-05-15  72.21    2025-05-21  74.24   4     281.9   +572.45   alpha_reversal
2025-06-03  74.47    2025-06-05  74.86   2     383.7   +149.08   alpha_reversal
2025-06-23  78.16    2025-06-30  77.11   5     411.5   -432.98   alpha_reversal
2025-07-07  78.89    2025-07-08  79.26   1     422.3   +160.05   alpha_reversal
2025-07-18  77.97    2025-07-25  82.52   5     420.5   +1912.85  alpha_reversal
2025-08-05  78.78    2025-08-06  77.38   1     369.3   -514.72   alpha_reversal
2025-08-12  78.30    2025-08-18  82.04   4     396.7   +1483.76  alpha_reversal
2025-08-27  81.98    2025-09-08  81.91   7     429.9   -26.85    alpha_reversal
2025-09-09  82.50    2025-09-15  79.70   4     479.6   -1342.85  stop_loss
2025-09-16  79.85    2025-09-25  76.34   7     488.0   -1714.24  stop_loss
2025-09-26  77.36    2025-10-01  88.66   3     511.2   +5779.29  alpha_reversal
2025-10-09  86.16    2025-10-15  82.74   4     400.4   -1369.86  stop_loss
2025-10-16  82.64    2025-10-27  86.57   7     426.1   +1675.14  alpha_reversal
2025-11-17  91.44    2025-11-18  94.86   1     360.6   +1233.33  alpha_reversal
2025-12-03  100.71   2025-12-09  95.31   4     305.3   -1646.37  stop_loss
2025-12-10  96.13    2025-12-15  99.47   3     311.3   +1040.89  alpha_reversal
2026-01-23  107.44   2026-01-28  106.06  3     322.9   -444.68   alpha_reversal
2026-02-27  122.97   2026-03-02  120.46  1     275.4   -692.29   alpha_reversal
2026-03-03  119.01   2026-03-05  115.16  2     276.6   -1064.73  alpha_reversal
2026-03-09  116.31   2026-03-23  115.62  10    260.2   -178.20   max_holding

**Best 3 trades:**

- 2018-01-17: P&L = **+5798.98** (alpha_reversal)
- 2025-10-01: P&L = **+5779.29** (alpha_reversal)
- 2016-08-05: P&L = **+4626.93** (alpha_reversal)

**Worst 3 trades:**

- 2021-04-29: P&L = **-2913.32** (stop_loss)
- 2017-10-30: P&L = **-2704.36** (stop_loss)
- 2019-07-11: P&L = **-2618.67** (trailing_stop)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  104,324.72
2017-03-23  107,174.52
2017-09-20  109,437.72
2018-03-21  104,226.80
2018-09-18  113,996.90
2019-03-20  118,320.52
2019-09-17  114,568.92
2020-03-17  113,522.74
2020-09-14  114,408.94
2021-03-15  116,890.27
2021-09-10  116,887.28
2022-03-10  115,745.27
2022-09-08  115,784.38
2023-03-09  119,607.50
2023-09-07  117,051.58
2024-03-07  120,413.02
2024-09-05  125,473.36
2025-03-07  122,075.60
2025-09-05  120,429.47
2026-03-06  121,605.71

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.89%
2017-03-23  -0.66%
2017-09-20  -0.25%
2018-03-21  -5.00%
2018-09-18  -0.03%
2019-03-20  -0.42%
2019-09-17  -3.67%
2020-03-17  -4.55%
2020-09-14  -3.80%
2021-03-15  -1.72%
2021-09-10  -2.25%
2022-03-10  -3.20%
2022-09-08  -3.17%
2023-03-09  -0.17%
2023-09-07  -3.73%
2024-03-07  -1.29%
2024-09-05  -0.01%
2025-03-07  -3.73%
2025-09-05  -5.03%
2026-03-06  -4.10%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  39.92%
Out-of-Sample (30%)  2023-03-24  2026-03-27  7.88%

#### Return Distribution

Return Bin          Count
-2.512% to -1.723%  4
-1.723% to -0.933%  22
-0.933% to -0.144%  355
-0.144% to 0.646%   2055
0.646% to 1.435%    72
1.435% to 2.225%    4
2.225% to 3.015%    2
3.015% to 3.804%    1
3.804% to 4.594%    0
4.594% to 5.383%    1

### WMT — AlphaCombined

**Net Return (after slippage):** 47.83%  **vs SPY (exposure-adj): -121.93%** (underperform)  
**Gross Return (pre-cost):** 63.55%  
**Total Slippage Cost:** $15,718.29  
**Trade Count:** 328  
**Win Rate:** 57.9%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-02  18.89    2016-05-05  18.76   3     2323.9  -290.39   alpha_reversal
2016-05-11  18.69    2016-05-13  18.26   2     2131.5  -920.96   alpha_reversal
2016-05-16  18.58    2016-05-18  17.76   2     1960.6  -1618.75  stop_loss
2016-05-19  19.48    2016-05-20  19.65   1     1403.2  +233.10   alpha_reversal
2016-05-31  19.92    2016-06-13  19.83   9     1806.2  -162.96   alpha_reversal
2016-07-05  20.59    2016-07-06  20.76   1     2389.0  +407.67   alpha_reversal
2016-07-08  20.79    2016-07-14  20.73   4     2517.0  -151.38   alpha_reversal
2016-07-18  20.79    2016-07-25  20.74   5     2743.7  -126.44   alpha_reversal
2016-07-27  20.64    2016-08-05  20.74   7     3191.0  +329.01   alpha_reversal
2016-08-09  20.70    2016-08-11  20.90   2     2892.1  +562.49   alpha_reversal
2016-08-12  20.94    2016-08-15  20.76   1     2716.4  -495.25   alpha_reversal
2016-08-16  20.66    2016-08-18  21.04   2     2635.5  +997.74   alpha_reversal
2016-08-22  20.60    2016-08-25  20.16   3     2323.1  -1021.28  stop_loss
2016-08-26  20.16    2016-09-02  20.53   5     2239.9  +817.36   alpha_reversal
2016-09-13  20.25    2016-09-15  20.50   2     2262.9  +556.48   alpha_reversal
2016-09-20  20.40    2016-09-22  20.46   2     2427.9  +156.73   alpha_reversal
2016-09-27  20.50    2016-09-29  20.03   2     2595.1  -1228.79  stop_loss
2016-10-04  20.34    2016-10-06  19.64   2     2416.7  -1684.52  stop_loss
2016-10-07  19.47    2016-10-17  19.32   6     2094.1  -325.33   alpha_reversal
2016-10-24  19.61    2016-10-27  19.77   3     2429.7  +392.65   alpha_reversal
2016-11-15  20.24    2016-11-17  19.59   2     2457.2  -1601.15  stop_loss
2016-11-18  19.43    2016-11-28  20.16   5     2085.9  +1524.57  alpha_reversal
2016-12-06  19.94    2016-12-20  20.48   10    2390.9  +1288.66  alpha_reversal
2016-12-22  19.86    2017-01-09  19.59   10    2330.8  -631.14   max_holding
2017-01-11  19.56    2017-01-13  19.14   2     2592.9  -1085.87  stop_loss
2017-01-17  19.53    2017-01-18  19.42   1     2344.6  -253.04   alpha_reversal
2017-01-19  19.30    2017-01-27  18.72   6     2395.0  -1384.85  stop_loss
2017-01-30  18.96    2017-02-08  19.34   7     2376.3  +896.85   alpha_reversal
2017-02-13  19.34    2017-02-21  20.37   5     2396.9  +2468.96  alpha_reversal
2017-03-01  20.11    2017-03-03  19.97   2     2033.0  -284.36   alpha_reversal
2017-03-06  19.95    2017-03-09  20.07   3     2133.6  +257.83   alpha_reversal
2017-03-14  20.34    2017-03-15  20.27   1     2428.9  -147.06   alpha_reversal
2017-03-16  20.25    2017-03-21  20.08   3     2525.5  -442.87   alpha_reversal
2017-03-24  20.02    2017-03-28  20.20   2     2740.8  +504.15   alpha_reversal
2017-04-07  20.96    2017-04-10  20.99   1     2553.5  +63.87    alpha_reversal
2017-04-25  21.58    2017-05-04  21.93   7     3047.4  +1063.53  alpha_reversal
2017-05-12  21.92    2017-05-16  21.72   2     3293.1  -643.52   alpha_reversal
2017-05-17  21.74    2017-05-18  22.42   1     3073.7  +2084.23  alpha_reversal
2017-05-30  22.62    2017-06-06  22.82   5     2822.9  +572.90   alpha_reversal
2017-06-07  22.91    2017-06-14  23.11   5     2534.6  +491.68   alpha_reversal
2017-06-19  21.85    2017-06-28  22.13   7     1674.3  +452.45   alpha_reversal
2017-07-06  21.85    2017-07-07  21.78   1     2113.4  -131.69   alpha_reversal
2017-07-10  21.20    2017-07-18  22.04   6     2009.9  +1683.61  alpha_reversal
2017-07-19  21.96    2017-07-21  22.02   2     2275.9  +134.32   alpha_reversal
2017-08-18  23.10    2017-08-22  23.29   2     2159.2  +396.27   alpha_reversal
2017-08-25  22.90    2017-09-06  23.30   7     2071.3  +826.55   alpha_reversal
2017-09-14  23.21    2017-09-20  23.43   4     2263.2  +487.54   alpha_reversal
2017-09-22  23.17    2017-09-27  23.07   3     2369.8  -220.38   alpha_reversal
2017-09-28  23.00    2017-10-06  22.99   6     2383.5  -20.12    alpha_reversal
2017-10-09  23.46    2017-10-10  24.48   1     2164.7  +2216.97  alpha_reversal
2017-10-31  25.43    2017-11-01  25.59   1     1949.9  +307.91   alpha_reversal
2017-11-16  29.02    2017-11-17  28.36   1     1327.0  -868.69   alpha_reversal
2017-11-20  28.39    2017-11-21  28.09   1     1239.5  -381.42   alpha_reversal
2017-11-27  28.14    2017-12-04  28.23   5     1389.8  +118.63   alpha_reversal
2017-12-07  28.34    2017-12-19  28.90   8     1603.9  +902.32   alpha_reversal
2017-12-21  28.71    2017-12-26  29.01   2     1758.3  +515.33   alpha_reversal
2018-01-03  29.12    2018-01-18  30.51   10    1988.3  +2763.07  max_holding
2018-02-01  30.90    2018-02-05  29.28   2     1571.5  -2544.84  stop_loss
2018-02-06  29.55    2018-02-08  29.26   2     1118.3  -320.92   alpha_reversal
2018-02-09  29.10    2018-02-14  29.75   3     940.0   +613.39   alpha_reversal
2018-02-21  26.80    2018-03-07  25.67   10    788.7   -893.25   trailing_stop
2018-03-08  25.90    2018-03-14  25.80   4     1008.5  -100.30   alpha_reversal
2018-03-16  26.27    2018-03-19  25.73   1     1099.5  -585.39   alpha_reversal
2018-03-28  25.85    2018-03-29  26.18   1     1141.3  +373.54   alpha_reversal
2018-04-03  25.57    2018-04-05  25.84   2     1119.2  +304.04   alpha_reversal
2018-04-09  25.42    2018-04-20  25.60   9     1207.9  +218.14   alpha_reversal
2018-05-03  25.40    2018-05-09  24.44   4     1516.0  -1452.68  trailing_stop
2018-05-10  24.51    2018-05-17  25.02   5     1322.5  +672.55   alpha_reversal
2018-05-18  24.79    2018-06-04  25.30   10    1341.7  +673.97   max_holding
2018-06-07  25.18    2018-06-08  24.98   1     1612.3  -322.28   alpha_reversal
2018-06-11  24.99    2018-06-20  24.76   7     1701.9  -390.26   alpha_reversal
2018-07-03  25.03    2018-07-05  25.04   1     1760.8  +23.73    alpha_reversal
2018-07-09  25.47    2018-07-11  25.62   2     1737.3  +264.45   alpha_reversal
2018-07-13  26.00    2018-07-16  25.95   1     1787.5  -78.21    alpha_reversal
2018-07-17  26.14    2018-07-18  26.08   1     1831.8  -112.95   alpha_reversal
2018-07-19  26.00    2018-07-23  25.95   2     1860.2  -97.92    alpha_reversal
2018-07-24  26.07    2018-07-31  26.42   5     1986.0  +695.15   alpha_reversal
2018-08-16  29.41    2018-08-17  29.14   1     1335.4  -353.47   alpha_reversal
2018-08-24  28.31    2018-08-31  28.55   5     1422.6  +345.34   alpha_reversal
2018-09-13  28.36    2018-09-18  28.42   3     1786.8  +114.32   alpha_reversal
2018-09-20  28.55    2018-09-24  28.27   2     1926.2  -531.15   alpha_reversal
2018-09-25  28.35    2018-10-02  28.34   5     1944.9  -26.14    alpha_reversal
2018-10-04  28.09    2018-10-05  27.79   1     1964.5  -581.77   alpha_reversal
2018-10-16  28.57    2018-10-17  28.76   1     1518.8  +295.92   alpha_reversal
2018-10-19  28.97    2018-10-22  28.93   1     1511.5  -48.26    alpha_reversal
2018-10-29  29.76    2018-10-30  30.51   1     1383.8  +1038.72  alpha_reversal
2018-11-01  29.99    2018-11-02  30.18   1     1327.9  +260.80   alpha_reversal
2018-11-14  30.27    2018-11-16  29.10   2     1365.4  -1603.03  stop_loss
2018-11-19  28.85    2018-11-26  28.34   4     1202.7  -618.61   alpha_reversal
2018-11-27  28.34    2018-12-06  28.38   6     1255.7  +56.88    alpha_reversal
2018-12-12  27.91    2018-12-20  26.14   6     1235.1  -2190.92  stop_loss
2018-12-21  26.12    2018-12-31  27.90   5     1114.4  +1980.02  alpha_reversal
2019-01-10  28.47    2019-01-15  28.82   3     1172.9  +419.72   alpha_reversal
2019-01-28  29.10    2019-01-29  28.96   1     1360.9  -182.21   alpha_reversal
2019-01-30  28.42    2019-02-01  28.11   2     1369.5  -424.42   alpha_reversal
2019-02-04  28.41    2019-02-07  28.97   3     1296.9  +724.41   alpha_reversal
2019-02-11  28.84    2019-02-12  29.04   1     1449.7  +292.51   alpha_reversal
2019-02-21  29.79    2019-03-06  29.43   9     1285.9  -473.44   alpha_reversal
2019-03-13  29.69    2019-03-19  30.06   4     1648.1  +620.99   alpha_reversal
2019-03-26  29.63    2019-04-05  29.76   8     1855.9  +230.00   alpha_reversal
2019-04-26  30.60    2019-05-07  30.50   7     1690.8  -168.79   alpha_reversal
2019-05-09  30.16    2019-05-13  30.24   2     1606.5  +121.78   alpha_reversal
2019-05-16  30.70    2019-05-17  30.53   1     1346.3  -224.68   alpha_reversal
2019-05-20  30.76    2019-05-21  30.61   1     1329.2  -201.79   alpha_reversal
2019-05-22  30.97    2019-05-23  30.83   1     1375.0  -196.57   alpha_reversal
2019-06-14  33.05    2019-06-17  33.04   1     1565.8  -9.06     alpha_reversal
2019-06-20  33.43    2019-06-24  33.67   2     1669.9  +409.23   alpha_reversal
2019-06-27  33.36    2019-07-03  34.00   4     1695.4  +1082.69  alpha_reversal
2019-07-08  34.15    2019-07-09  34.17   1     1715.5  +24.53    alpha_reversal
2019-07-17  34.72    2019-07-19  34.48   2     1911.7  -471.37   alpha_reversal
2019-07-23  33.96    2019-08-01  33.11   7     1785.4  -1525.17  stop_loss
2019-08-02  33.15    2019-08-05  32.03   1     1551.9  -1733.07  stop_loss
2019-08-06  32.50    2019-08-07  32.75   1     1340.7  +333.86   alpha_reversal
2019-08-15  34.31    2019-08-16  34.37   1     969.3   +55.21    alpha_reversal
2019-08-22  34.07    2019-08-28  34.29   4     1013.1  +215.10   alpha_reversal
2019-09-06  34.93    2019-09-13  35.72   5     1184.4  +931.36   alpha_reversal
2019-09-20  35.62    2019-09-24  36.01   2     1420.8  +563.10   alpha_reversal
2019-10-02  35.36    2019-10-09  36.18   5     1455.2  +1192.36  alpha_reversal
2019-10-16  36.36    2019-10-22  36.37   4     1490.4  +18.36    alpha_reversal
2019-10-23  36.34    2019-10-29  35.63   4     1658.2  -1169.89  stop_loss
2019-10-30  35.96    2019-11-13  36.80   10    1650.3  +1386.40  alpha_reversal
2019-11-15  36.19    2019-12-02  36.28   10    1248.0  +110.49   alpha_reversal
2019-12-12  36.63    2019-12-23  36.37   7     1694.6  -440.02   alpha_reversal
2019-12-24  36.55    2019-12-26  36.52   1     1713.3  -57.37    alpha_reversal
2019-12-27  36.58    2019-12-30  36.48   1     1817.2  -171.93   alpha_reversal
2019-12-31  36.35    2020-01-03  36.02   2     1838.1  -600.30   alpha_reversal
2020-01-06  35.98    2020-01-15  35.22   7     1821.0  -1384.10  stop_loss
2020-01-16  35.45    2020-01-22  35.47   3     1842.5  +47.30    alpha_reversal
2020-01-24  34.98    2020-01-28  35.63   2     1741.0  +1125.35  alpha_reversal
2020-01-30  35.66    2020-01-31  34.98   1     1622.2  -1093.65  stop_loss
2020-02-04  35.25    2020-02-10  35.21   4     1492.7  -61.70    alpha_reversal
2020-02-11  35.29    2020-02-14  36.02   3     1522.0  +1104.27  alpha_reversal
2020-02-20  36.00    2020-02-21  36.23   1     1446.5  +341.33   alpha_reversal
2020-03-02  35.44    2020-03-03  34.50   1     866.5   -817.01   alpha_reversal
2020-03-10  36.64    2020-03-11  34.96   1     637.3   -1066.98  stop_loss
2020-03-20  35.01    2020-03-24  35.30   2     327.6   +95.11    alpha_reversal
2020-03-25  33.61    2020-04-02  36.41   6     335.1   +940.15   alpha_reversal
2020-04-09  37.42    2020-04-14  39.59   2     406.3   +882.55   alpha_reversal
2020-04-20  39.89    2020-04-22  40.38   2     462.4   +228.47   alpha_reversal
2020-04-24  39.76    2020-04-30  37.30   4     520.6   -1281.16  stop_loss
2020-05-01  37.76    2020-05-13  38.13   8     565.6   +210.23   alpha_reversal
2020-05-26  38.22    2020-06-09  37.41   10    748.1   -607.36   max_holding
2020-06-15  36.43    2020-06-22  37.51   5     894.1   +959.57   alpha_reversal
2020-06-24  37.12    2020-06-29  36.70   3     1000.2  -419.42   alpha_reversal
2020-07-02  36.78    2020-07-08  38.36   3     1074.6  +1692.81  alpha_reversal
2020-07-09  39.42    2020-07-10  40.28   1     838.5   +724.29   alpha_reversal
2020-07-20  40.57    2020-07-29  40.28   7     840.5   -236.15   alpha_reversal
2020-07-30  40.15    2020-08-13  40.81   10    991.8   +654.83   max_holding
2020-08-25  40.47    2020-08-28  43.42   3     965.1   +2849.44  alpha_reversal
2020-09-14  42.54    2020-09-22  42.81   6     581.3   +153.41   alpha_reversal
2020-09-29  42.49    2020-10-02  43.49   3     778.8   +776.88   alpha_reversal
2020-10-05  43.93    2020-10-06  43.53   1     767.3   -311.57   alpha_reversal
2020-10-23  44.57    2020-10-26  44.00   1     986.9   -560.18   alpha_reversal
2020-11-05  44.45    2020-11-12  45.88   5     936.8   +1338.55  alpha_reversal
2020-11-30  47.34    2020-12-01  47.24   1     923.0   -86.52    alpha_reversal
2020-12-02  46.63    2020-12-14  45.24   8     924.5   -1284.30  stop_loss
2020-12-16  45.22    2020-12-18  45.34   2     1061.3  +123.47   alpha_reversal
2020-12-21  45.39    2020-12-28  45.11   4     1121.9  -312.28   alpha_reversal
2020-12-30  44.83    2021-01-05  45.28   3     1163.5  +515.29   alpha_reversal
2021-01-06  45.60    2021-01-15  44.93   7     1105.3  -743.93   trailing_stop
2021-01-19  44.59    2021-01-25  45.42   4     1095.8  +907.65   alpha_reversal
2021-02-01  43.31    2021-02-04  44.28   3     987.1   +956.87   alpha_reversal
2021-02-11  44.78    2021-02-17  45.73   3     1111.9  +1048.56  alpha_reversal
2021-02-24  41.42    2021-03-03  39.63   5     930.8   -1663.43  stop_loss
2021-03-04  39.66    2021-03-11  41.05   5     909.3   +1263.33  alpha_reversal
2021-03-12  41.70    2021-03-22  41.29   6     916.4   -379.02   alpha_reversal
2021-04-05  43.54    2021-04-06  43.70   1     1035.9  +171.42   alpha_reversal
2021-04-08  43.62    2021-04-12  43.61   2     1102.2  -17.12    alpha_reversal
2021-04-14  43.50    2021-04-21  44.05   5     1256.6  +682.29   alpha_reversal
2021-04-27  43.21    2021-05-03  44.33   4     1351.6  +1518.52  alpha_reversal
2021-05-10  44.14    2021-05-11  43.70   1     1362.2  -601.85   alpha_reversal
2021-05-12  42.61    2021-05-17  43.50   3     1230.8  +1084.63  alpha_reversal
2021-05-27  44.42    2021-06-04  44.42   5     1186.9  +6.78     alpha_reversal
2021-06-08  43.83    2021-06-16  42.95   6     1412.2  -1247.10  trailing_stop
2021-06-21  42.76    2021-07-01  43.63   8     1393.8  +1215.02  alpha_reversal
2021-07-13  44.07    2021-07-14  44.33   1     1365.9  +354.75   alpha_reversal
2021-07-22  44.28    2021-07-23  44.60   1     1367.0  +436.08   alpha_reversal
2021-07-29  44.59    2021-07-30  44.64   1     1454.2  +76.37    alpha_reversal
2021-08-24  46.85    2021-09-07  46.29   9     1210.5  -676.88   alpha_reversal
2021-09-14  45.40    2021-09-21  44.95   5     1304.0  -583.79   alpha_reversal
2021-09-23  44.92    2021-09-30  43.81   5     1319.2  -1464.93  stop_loss
2021-10-04  42.70    2021-10-11  43.86   5     1149.2  +1323.51  alpha_reversal
2021-10-15  44.22    2021-10-20  45.89   3     1294.4  +2160.14  alpha_reversal
2021-11-03  47.25    2021-11-12  46.44   7     1332.9  -1076.85  trailing_stop
2021-11-30  44.25    2021-12-01  43.11   1     1060.5  -1210.26  stop_loss
2021-12-03  43.26    2021-12-06  43.69   1     981.3   +417.14   alpha_reversal
2021-12-08  43.15    2021-12-10  44.51   2     1008.8  +1366.75  alpha_reversal
2021-12-20  43.97    2021-12-21  44.06   1     849.7   +75.28    alpha_reversal
2021-12-27  44.47    2021-12-29  45.04   2     971.8   +554.85   alpha_reversal
2021-12-30  45.23    2021-12-31  45.66   1     1056.5  +459.04   alpha_reversal
2022-01-04  44.86    2022-01-06  45.29   2     1050.5  +456.81   alpha_reversal
2022-01-12  45.31    2022-01-20  44.57   5     1084.5  -808.91   alpha_reversal
2022-01-26  42.88    2022-01-28  43.40   2     928.4   +478.78   alpha_reversal
2022-01-31  44.17    2022-02-04  43.97   4     874.9   -171.14   alpha_reversal
2022-02-07  43.58    2022-02-14  42.27   5     937.1   -1226.76  stop_loss
2022-03-03  44.00    2022-03-04  45.07   1     896.4   +959.21   alpha_reversal
2022-03-08  43.83    2022-03-10  45.01   2     822.7   +973.90   alpha_reversal
2022-03-14  45.50    2022-03-15  46.01   1     846.1   +423.47   alpha_reversal
2022-03-17  45.99    2022-03-18  46.08   1     861.6   +77.76    alpha_reversal
2022-03-21  45.74    2022-03-28  46.25   5     874.4   +450.34   alpha_reversal
2022-04-13  49.86    2022-04-14  49.76   1     831.8   -78.34    alpha_reversal
2022-04-19  49.99    2022-04-20  50.57   1     876.4   +505.95   alpha_reversal
2022-04-28  49.54    2022-05-02  48.15   2     894.4   -1242.86  stop_loss
2022-05-03  48.36    2022-05-10  47.43   5     842.4   -784.72   alpha_reversal
2022-05-11  46.98    2022-05-17  41.76   4     738.2   -3853.56  stop_loss
2022-05-23  39.02    2022-06-07  39.23   10    523.6   +107.77   max_holding
2022-06-08  38.92    2022-06-13  37.97   3     661.4   -633.47   alpha_reversal
2022-06-16  38.39    2022-06-17  37.61   1     710.9   -553.91   alpha_reversal
2022-06-23  39.34    2022-06-24  39.34   1     707.5   -5.33     alpha_reversal
2022-06-27  39.50    2022-06-28  38.91   1     747.4   -445.36   alpha_reversal
2022-06-29  38.80    2022-07-05  39.51   3     756.0   +530.73   alpha_reversal
2022-07-13  39.90    2022-07-14  40.64   1     910.8   +673.20   alpha_reversal
2022-07-19  41.24    2022-07-26  38.78   5     932.1   -2284.88  trailing_stop
2022-07-27  40.29    2022-08-03  41.49   5     720.2   +866.36   alpha_reversal
2022-08-04  39.97    2022-08-12  42.22   6     684.8   +1545.92  alpha_reversal
2022-08-24  43.14    2022-08-30  42.31   4     814.3   -682.61   alpha_reversal
2022-09-19  42.94    2022-09-22  42.60   3     888.9   -304.98   alpha_reversal
2022-10-06  42.09    2022-10-11  42.37   3     827.9   +226.90   alpha_reversal
2022-10-18  42.88    2022-10-21  43.69   3     759.3   +612.41   alpha_reversal
2022-11-01  45.29    2022-11-08  45.60   5     890.5   +272.48   alpha_reversal
2022-12-01  49.03    2022-12-07  47.62   4     892.5   -1257.91  stop_loss
2022-12-14  47.06    2022-12-16  45.76   2     886.7   -1155.84  stop_loss
2022-12-19  45.82    2022-12-20  46.17   1     885.4   +314.19   alpha_reversal
2022-12-23  46.13    2022-12-27  46.10   1     920.9   -30.65    alpha_reversal
2023-01-03  46.08    2023-01-18  45.15   10    1006.8  -937.07   trailing_stop
2023-01-19  44.55    2023-01-23  45.72   2     975.4   +1147.74  alpha_reversal
2023-01-24  45.89    2023-02-07  45.19   10    883.4   -618.18   max_holding
2023-02-08  44.99    2023-02-10  46.07   2     953.8   +1027.13  alpha_reversal
2023-02-28  45.60    2023-03-09  43.97   7     842.5   -1377.86  stop_loss
2023-03-17  44.91    2023-03-20  45.35   1     1011.1  +442.76   alpha_reversal
2023-03-22  44.95    2023-03-28  46.22   4     1037.4  +1318.95  alpha_reversal
2023-04-05  48.22    2023-04-06  48.54   1     1074.0  +338.85   alpha_reversal
2023-04-11  48.35    2023-04-14  47.79   3     1123.7  -629.37   alpha_reversal
2023-04-17  48.17    2023-04-19  48.28   2     1185.9  +129.93   alpha_reversal
2023-05-01  48.84    2023-05-02  48.66   1     1327.5  -239.97   alpha_reversal
2023-05-03  48.53    2023-05-05  49.03   2     1302.3  +660.50   alpha_reversal
2023-05-08  49.39    2023-05-12  49.45   4     1297.3  +82.66    alpha_reversal
2023-05-26  47.35    2023-06-01  47.63   3     1133.7  +308.95   alpha_reversal
2023-06-05  48.45    2023-06-06  48.39   1     1133.9  -62.23    alpha_reversal
2023-06-07  48.51    2023-06-12  49.79   3     1178.0  +1503.31  alpha_reversal
2023-06-20  49.86    2023-06-23  50.23   3     1214.3  +449.51   alpha_reversal
2023-06-27  49.96    2023-07-05  51.08   5     1287.1  +1441.10  alpha_reversal
2023-07-12  50.14    2023-07-17  50.03   3     1244.9  -138.81   alpha_reversal
2023-07-19  50.00    2023-07-25  51.42   4     1340.8  +1904.02  alpha_reversal
2023-08-01  51.46    2023-08-03  51.45   2     1356.8  -4.04     alpha_reversal
2023-08-11  52.32    2023-08-14  51.88   1     1331.3  -587.59   alpha_reversal
2023-09-05  52.02    2023-09-07  53.00   2     1132.3  +1115.96  alpha_reversal
2023-09-18  53.04    2023-09-21  52.50   3     1337.9  -721.62   alpha_reversal
2023-10-09  50.58    2023-10-16  52.27   5     1022.7  +1729.00  alpha_reversal
2023-11-03  53.44    2023-11-16  50.59   9     1139.7  -3246.17  trailing_stop
2023-11-24  50.65    2023-12-07  49.62   9     914.5   -945.82   trailing_stop
2023-12-08  49.14    2023-12-18  50.43   6     985.6   +1269.86  alpha_reversal
2023-12-21  50.43    2023-12-22  50.98   1     1074.7  +592.83   alpha_reversal
2023-12-27  51.43    2023-12-28  51.28   1     1138.1  -173.33   alpha_reversal
2023-12-29  51.36    2024-01-02  51.84   1     1236.2  +596.33   alpha_reversal
2024-01-05  51.05    2024-01-11  52.43   4     1221.1  +1686.24  alpha_reversal
2024-01-12  52.55    2024-01-16  52.66   1     1307.1  +148.29   alpha_reversal
2024-01-22  52.90    2024-01-23  52.98   1     1329.5  +107.10   alpha_reversal
2024-01-25  53.05    2024-01-29  53.71   2     1291.7  +856.32   alpha_reversal
2024-02-01  54.83    2024-02-02  55.18   1     1205.9  +428.39   alpha_reversal
2024-02-09  55.14    2024-02-20  57.23   6     1318.9  +2751.58  alpha_reversal
2024-02-28  58.26    2024-02-29  57.22   1     985.8   -1029.49  alpha_reversal
2024-03-08  58.75    2024-03-12  59.95   2     1010.5  +1213.33  alpha_reversal
2024-03-14  59.84    2024-03-20  60.00   4     1032.7  +170.91   alpha_reversal
2024-03-26  59.34    2024-04-02  57.95   4     1178.2  -1628.07  stop_loss
2024-04-09  58.63    2024-04-10  59.38   1     1190.6  +898.30   alpha_reversal
2024-04-12  58.97    2024-04-15  58.71   1     1113.7  -294.76   alpha_reversal
2024-04-18  58.11    2024-04-25  58.98   5     1124.1  +980.85   alpha_reversal
2024-04-29  59.07    2024-04-30  58.14   1     1044.4  -972.21   alpha_reversal
2024-05-09  59.47    2024-05-17  63.55   6     1028.6  +4195.50  alpha_reversal
2024-06-13  65.63    2024-06-14  65.88   1     867.8   +216.06   alpha_reversal
2024-06-24  67.80    2024-06-25  66.27   1     915.5   -1393.88  stop_loss
2024-06-26  67.20    2024-06-27  66.73   1     800.0   -376.17   alpha_reversal
2024-06-28  66.63    2024-07-03  67.08   3     821.8   +373.42   alpha_reversal
2024-07-15  68.50    2024-07-24  69.40   7     876.0   +792.58   alpha_reversal
2024-07-25  68.90    2024-07-31  67.47   4     977.7   -1393.69  stop_loss
2024-08-01  68.67    2024-08-02  67.30   1     910.8   -1253.26  alpha_reversal
2024-08-05  66.51    2024-08-15  71.94   8     772.9   +4195.62  alpha_reversal
2024-08-29  75.41    2024-09-09  76.24   6     776.8   +645.94   alpha_reversal
2024-09-18  77.99    2024-09-23  79.19   3     730.1   +878.75   alpha_reversal
2024-09-30  79.68    2024-10-02  79.29   2     722.2   -285.36   alpha_reversal
2024-10-03  79.37    2024-10-04  79.79   1     749.9   +317.54   alpha_reversal
2024-10-09  79.34    2024-10-16  80.07   5     752.8   +548.85   alpha_reversal
2024-10-17  79.82    2024-10-24  81.86   5     801.5   +1634.90  alpha_reversal
2024-10-28  81.66    2024-10-29  80.54   1     849.2   -948.33   alpha_reversal
2024-10-30  80.32    2024-11-01  81.02   2     866.6   +613.85   alpha_reversal
2024-11-07  82.74    2024-11-11  83.02   2     742.0   +201.97   alpha_reversal
2024-11-12  83.87    2024-11-20  85.94   6     721.3   +1496.73  alpha_reversal
2024-12-13  93.21    2024-12-16  93.73   1     613.0   +318.38   alpha_reversal
2024-12-23  89.35    2025-01-06  90.33   8     533.3   +521.40   alpha_reversal
2025-01-08  90.79    2025-01-10  91.88   1     595.3   +651.75   alpha_reversal
2025-01-15  90.33    2025-01-28  96.12   8     602.9   +3489.95  alpha_reversal
2025-02-12  102.47   2025-02-14  102.79  2     572.8   +184.70   alpha_reversal
2025-02-27  95.72    2025-02-28  97.43   1     403.6   +687.08   alpha_reversal
2025-03-10  86.85    2025-03-24  86.68   10    360.3   -63.23    max_holding
2025-03-26  84.50    2025-03-27  84.83   1     425.1   +140.98   alpha_reversal
2025-03-31  87.06    2025-04-01  88.00   1     427.5   +403.26   alpha_reversal
2025-04-22  94.06    2025-04-23  94.08   1     295.6   +4.42     alpha_reversal
2025-04-28  94.43    2025-04-29  95.15   1     322.1   +231.26   alpha_reversal
2025-05-07  98.01    2025-05-08  96.52   1     402.4   -597.50   alpha_reversal
2025-05-09  96.15    2025-05-13  95.22   2     423.7   -394.16   alpha_reversal
2025-05-23  95.77    2025-05-27  96.91   1     419.2   +476.14   alpha_reversal
2025-05-28  96.67    2025-06-02  99.08   3     445.9   +1077.30  alpha_reversal
2025-06-05  97.38    2025-06-12  94.18   5     481.9   -1544.87  stop_loss
2025-06-13  93.88    2025-06-20  95.46   4     503.5   +792.81   alpha_reversal
2025-06-26  95.45    2025-06-30  97.11   2     545.2   +900.87   alpha_reversal
2025-07-01  97.66    2025-07-07  98.66   3     545.2   +547.82   alpha_reversal
2025-07-10  94.30    2025-07-22  95.19   8     553.8   +492.24   alpha_reversal
2025-07-24  96.03    2025-07-29  97.65   3     662.0   +1073.78  alpha_reversal
2025-08-06  102.75   2025-08-07  102.41  1     590.4   -201.35   alpha_reversal
2025-08-13  100.39   2025-08-20  102.10  5     583.2   +995.33   alpha_reversal
2025-08-27  95.74    2025-09-03  98.98   4     565.3   +1836.70  alpha_reversal
2025-09-17  103.90   2025-09-18  103.13  1     584.7   -450.69   alpha_reversal
2025-09-19  101.96   2025-09-23  102.05  2     559.7   +48.81    alpha_reversal
2025-09-24  102.36   2025-09-25  102.58  1     593.5   +128.33   alpha_reversal
2025-10-09  101.41   2025-10-13  101.65  2     563.8   +139.27   alpha_reversal
2025-10-21  105.84   2025-10-28  102.70  5     472.0   -1483.00  stop_loss
2025-10-29  102.09   2025-11-05  101.01  5     492.5   -535.64   alpha_reversal
2025-11-06  101.32   2025-11-18  100.93  8     521.8   -203.48   alpha_reversal
2025-12-11  115.11   2025-12-15  116.49  2     402.1   +557.33   alpha_reversal
2025-12-22  112.43   2025-12-29  112.24  4     445.8   -81.22    alpha_reversal
2025-12-31  111.24   2026-01-05  112.42  2     543.0   +643.76   alpha_reversal
2026-01-16  119.51   2026-01-20  118.41  1     406.6   -450.08   alpha_reversal
2026-01-23  117.55   2026-02-04  127.67  8     400.3   +4053.92  alpha_reversal
2026-02-26  124.23   2026-03-02  126.78  2     292.8   +746.41   alpha_reversal
2026-03-03  127.71   2026-03-04  127.48  1     304.3   -69.20    alpha_reversal
2026-03-05  123.12   2026-03-13  126.20  6     293.6   +904.01   alpha_reversal
2026-03-17  124.89   2026-03-20  118.96  3     341.4   -2022.94  stop_loss

**Best 3 trades:**

- 2024-08-15: P&L = **+4195.62** (alpha_reversal)
- 2024-05-17: P&L = **+4195.50** (alpha_reversal)
- 2026-02-04: P&L = **+4053.92** (alpha_reversal)

**Worst 3 trades:**

- 2022-05-17: P&L = **-3853.56** (stop_loss)
- 2023-11-16: P&L = **-3246.17** (trailing_stop)
- 2018-02-05: P&L = **-2544.84** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  99,298.02
2017-03-23  97,033.70
2017-09-20  105,046.12
2018-03-21  106,521.90
2018-09-18  107,424.78
2019-03-20  106,409.40
2019-09-17  105,282.65
2020-03-17  104,308.66
2020-09-14  110,989.65
2021-03-15  112,908.18
2021-09-10  115,747.68
2022-03-10  118,449.01
2022-09-08  113,046.98
2023-03-09  110,983.87
2023-09-07  119,029.86
2024-03-07  123,049.35
2024-09-05  129,158.33
2025-03-07  139,731.98
2025-09-05  145,903.77
2026-03-06  149,073.81

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.70%
2017-03-23  -2.97%
2017-09-20  0.00%
2018-03-21  -3.50%
2018-09-18  -2.68%
2019-03-20  -3.60%
2019-09-17  -4.62%
2020-03-17  -5.50%
2020-09-14  -0.03%
2021-03-15  -0.56%
2021-09-10  -0.67%
2022-03-10  -0.38%
2022-09-08  -5.66%
2023-03-09  -7.38%
2023-09-07  -0.67%
2024-03-07  -0.86%
2024-09-05  -0.17%
2025-03-07  -0.01%
2025-09-05  -0.02%
2026-03-06  -0.14%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  31.12%
Out-of-Sample (30%)  2023-03-24  2026-03-27  43.09%

#### Return Distribution

Return Bin          Count
-4.168% to -3.407%  1
-3.407% to -2.646%  1
-2.646% to -1.885%  1
-1.885% to -1.123%  7
-1.123% to -0.362%  154
-0.362% to 0.399%   2156
0.399% to 1.160%    184
1.160% to 1.922%    6
1.922% to 2.683%    4
2.683% to 3.444%    2

### CAT — AlphaCombined

**Net Return (after slippage):** 38.06%  **vs SPY (exposure-adj): -138.88%** (underperform)  
**Gross Return (pre-cost):** 58.16%  
**Total Slippage Cost:** $20,100.92  
**Trade Count:** 310  
**Win Rate:** 54.8%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size   P&L       Exit Reason
2016-05-06  58.63    2016-05-09  56.45   1     480.7  -1045.52  stop_loss
2016-05-10  57.95    2016-05-17  56.67   5     450.5  -576.74   alpha_reversal
2016-05-19  55.48    2016-05-24  56.70   3     463.8  +562.64   alpha_reversal
2016-05-27  57.51    2016-06-01  57.64   2     522.2  +69.12    alpha_reversal
2016-06-02  58.83    2016-06-06  60.95   2     543.3  +1149.30  alpha_reversal
2016-06-17  60.68    2016-06-20  60.96   1     538.0  +149.31   alpha_reversal
2016-06-23  62.51    2016-06-24  58.25   1     565.1  -2409.81  stop_loss
2016-06-27  57.04    2016-07-07  59.86   7     453.3  +1275.15  alpha_reversal
2016-07-12  63.77    2016-07-18  64.27   4     456.6  +229.10   alpha_reversal
2016-07-20  64.30    2016-07-22  63.92   2     548.0  -207.18   alpha_reversal
2016-07-25  63.49    2016-07-27  67.72   2     563.6  +2383.52  alpha_reversal
2016-08-03  66.19    2016-08-09  66.70   4     541.3  +277.14   alpha_reversal
2016-08-10  66.69    2016-08-12  66.83   2     599.7  +89.11    alpha_reversal
2016-08-19  67.65    2016-08-30  66.42   7     625.1  -768.99   alpha_reversal
2016-08-31  66.12    2016-09-08  67.27   5     705.5  +810.03   alpha_reversal
2016-09-16  66.20    2016-09-19  65.93   1     620.2  -166.93   alpha_reversal
2016-09-20  65.92    2016-09-29  70.43   7     642.3  +2899.44  alpha_reversal
2016-10-11  70.57    2016-10-18  70.23   5     551.2  -188.71   alpha_reversal
2016-10-19  71.00    2016-10-25  68.63   4     599.1  -1423.40  stop_loss
2016-10-26  68.48    2016-11-02  65.89   5     535.8  -1387.84  stop_loss
2016-11-17  75.52    2016-11-29  76.39   7     393.9  +343.71   alpha_reversal
2016-12-06  77.51    2016-12-07  79.07   1     446.9  +696.74   alpha_reversal
2016-12-08  78.36    2016-12-16  75.21   6     451.9  -1425.21  stop_loss
2016-12-19  75.46    2016-12-23  76.62   4     455.7  +527.32   alpha_reversal
2017-01-05  75.70    2017-01-17  76.01   7     533.1  +166.22   alpha_reversal
2017-01-27  81.24    2017-01-30  79.28   1     492.7  -967.78   alpha_reversal
2017-02-02  76.96    2017-02-13  80.68   7     460.9  +1714.84  alpha_reversal
2017-02-17  81.13    2017-02-21  80.35   1     495.9  -384.99   alpha_reversal
2017-02-22  80.59    2017-02-23  78.26   1     514.8  -1200.26  stop_loss
2017-02-24  78.36    2017-03-02  77.29   4     466.1  -500.61   trailing_stop
2017-03-03  78.07    2017-03-09  74.86   4     396.3  -1272.50  stop_loss
2017-03-10  75.76    2017-03-21  75.71   7     376.4  -19.97    alpha_reversal
2017-03-24  75.63    2017-04-06  78.48   9     417.9  +1192.98  alpha_reversal
2017-04-20  78.33    2017-04-25  86.24   3     429.4  +3394.32  alpha_reversal
2017-05-03  84.03    2017-05-16  84.52   9     397.9  +196.08   alpha_reversal
2017-05-17  82.87    2017-05-18  82.77   1     455.7  -45.36    alpha_reversal
2017-05-23  85.76    2017-05-24  85.85   1     441.1  +40.99    alpha_reversal
2017-05-31  87.25    2017-06-12  87.60   8     466.9  +165.38   alpha_reversal
2017-06-14  86.65    2017-06-16  88.87   2     479.9  +1062.29  alpha_reversal
2017-06-20  88.58    2017-06-21  85.45   1     488.5  -1526.85  stop_loss
2017-06-22  85.93    2017-06-26  86.10   2     454.5  +75.86    alpha_reversal
2017-06-28  88.09    2017-07-06  87.96   5     438.2  -55.41    alpha_reversal
2017-07-14  90.01    2017-07-25  95.28   7     476.1  +2508.29  alpha_reversal
2017-08-02  94.26    2017-08-04  95.12   2     469.7  +403.87   alpha_reversal
2017-08-14  94.78    2017-08-21  94.50   5     561.8  -157.80   alpha_reversal
2017-08-29  96.70    2017-08-30  97.78   1     575.5  +626.11   alpha_reversal
2017-09-01  98.59    2017-09-06  97.31   2     573.0  -732.51   alpha_reversal
2017-09-07  98.16    2017-09-11  98.88   2     557.7  +400.92   alpha_reversal
2017-09-14  100.42   2017-09-15  100.96  1     531.6  +286.90   alpha_reversal
2017-09-25  103.62   2017-09-27  103.58  2     565.1  -23.00    alpha_reversal
2017-10-02  103.96   2017-10-11  106.98  7     620.0  +1872.26  alpha_reversal
2017-11-09  113.48   2017-11-13  114.25  2     471.5  +362.59   alpha_reversal
2017-11-20  114.80   2017-11-24  114.97  3     440.8  +75.95    alpha_reversal
2017-11-27  115.03   2017-12-01  118.43  4     490.6  +1664.86  alpha_reversal
2017-12-12  120.26   2017-12-14  122.41  2     413.4  +890.12   alpha_reversal
2018-01-03  131.88   2018-01-09  139.27  4     400.0  +2957.11  alpha_reversal
2018-01-17  141.28   2018-01-26  140.45  7     346.7  -290.66   alpha_reversal
2018-01-29  136.95   2018-01-31  136.85  2     243.0  -25.63    alpha_reversal
2018-02-01  136.67   2018-02-05  127.01  2     238.5  -2302.99  stop_loss
2018-02-20  131.04   2018-02-23  136.54  3     174.5  +959.48   alpha_reversal
2018-03-02  123.31   2018-03-09  133.04  5     179.7  +1748.47  alpha_reversal
2018-03-13  129.46   2018-03-19  127.91  4     182.6  -283.69   alpha_reversal
2018-03-27  123.82   2018-04-02  120.97  3     180.1  -513.89   alpha_reversal
2018-04-25  122.92   2018-04-26  123.62  1     172.5  +120.84   alpha_reversal
2018-04-27  122.52   2018-05-09  128.94  8     183.9  +1181.13  alpha_reversal
2018-05-22  132.24   2018-05-24  133.24  2     233.0  +233.78   alpha_reversal
2018-05-25  131.95   2018-05-31  128.35  3     244.9  -879.93   alpha_reversal
2018-06-05  129.54   2018-06-12  133.07  5     274.1  +966.47   alpha_reversal
2018-06-14  129.65   2018-06-19  121.08  3     288.0  -2469.21  stop_loss
2018-06-20  121.18   2018-06-25  115.40  3     263.3  -1520.97  stop_loss
2018-06-26  114.75   2018-07-11  115.55  10    259.6  +208.04   max_holding
2018-07-13  119.16   2018-07-16  116.67  1     261.3  -651.76   alpha_reversal
2018-07-17  117.64   2018-07-27  121.19  8     265.2  +942.32   alpha_reversal
2018-07-31  122.49   2018-08-01  117.77  1     243.1  -1146.38  alpha_reversal
2018-08-02  117.56   2018-08-13  114.78  7     234.5  -651.19   trailing_stop
2018-08-14  114.93   2018-08-20  117.88  4     267.1  +790.20   alpha_reversal
2018-08-21  119.24   2018-08-22  118.70  1     258.3  -140.59   alpha_reversal
2018-08-23  116.52   2018-09-07  120.14  10    269.7  +975.89   alpha_reversal
2018-09-12  122.90   2018-09-13  123.72  1     321.0  +264.98   alpha_reversal
2018-09-17  124.55   2018-09-18  126.73  1     310.2  +674.34   alpha_reversal
2018-09-27  129.88   2018-10-04  133.25  5     309.1  +1044.33  alpha_reversal
2018-10-11  120.08   2018-10-18  115.44  5     247.6  -1147.84  trailing_stop
2018-10-19  112.57   2018-10-23  101.79  2     231.5  -2495.81  stop_loss
2018-10-24  96.30    2018-11-05  108.15  8     177.2  +2098.78  alpha_reversal
2018-11-15  110.94   2018-11-16  111.18  1     180.8  +43.44    alpha_reversal
2018-11-26  106.98   2018-11-27  106.60  1     196.8  -75.73    alpha_reversal
2018-11-28  112.10   2018-12-03  118.87  3     194.8  +1319.83  alpha_reversal
2018-12-12  107.47   2018-12-14  108.45  2     173.7  +170.76   alpha_reversal
2018-12-26  106.95   2018-12-27  108.37  1     172.1  +244.45   alpha_reversal
2018-12-31  108.93   2019-01-02  108.12  1     179.6  -145.10   alpha_reversal
2019-01-03  104.16   2019-01-11  113.01  6     177.1  +1567.27  alpha_reversal
2019-01-14  112.98   2019-01-15  111.81  1     197.8  -232.48   alpha_reversal
2019-01-16  112.86   2019-01-28  107.09  7     207.9  -1201.07  trailing_stop
2019-01-29  109.16   2019-02-12  114.23  10    182.5  +925.11   alpha_reversal
2019-02-25  122.00   2019-02-26  118.80  1     263.8  -843.33   alpha_reversal
2019-03-01  118.60   2019-03-08  113.10  5     263.2  -1449.48  stop_loss
2019-03-11  114.84   2019-03-20  114.53  7     286.9  -90.52    alpha_reversal
2019-03-21  115.66   2019-03-22  111.74  1     333.2  -1307.94  stop_loss
2019-03-25  113.35   2019-03-26  113.52  1     312.7  +53.03    alpha_reversal
2019-03-28  114.20   2019-03-29  116.66  1     329.1  +809.06   alpha_reversal
2019-04-03  120.15   2019-04-05  120.85  2     319.3  +225.77   alpha_reversal
2019-04-09  117.64   2019-04-17  122.90  6     336.3  +1771.18  alpha_reversal
2019-04-24  119.55   2019-05-07  115.79  9     325.9  -1223.98  stop_loss
2019-05-09  113.88   2019-05-13  108.54  2     278.2  -1485.13  stop_loss
2019-05-14  110.64   2019-05-23  105.91  7     248.7  -1177.55  stop_loss
2019-06-13  110.38   2019-06-17  110.29  2     340.0  -30.81    alpha_reversal
2019-07-03  117.69   2019-07-16  120.49  8     344.4  +963.29   alpha_reversal
2019-07-23  120.79   2019-07-24  115.15  1     342.3  -1932.04  stop_loss
2019-07-25  117.83   2019-07-26  116.03  1     273.1  -491.10   alpha_reversal
2019-07-29  117.61   2019-07-30  116.05  1     286.3  -444.67   alpha_reversal
2019-07-31  115.17   2019-08-01  110.68  1     281.6  -1264.46  stop_loss
2019-08-02  108.93   2019-08-09  104.21  5     254.2  -1200.50  stop_loss
2019-08-12  102.09   2019-08-23  99.56   9     254.4  -642.51   alpha_reversal
2019-08-26  100.08   2019-09-03  102.16  5     275.8  +573.13   alpha_reversal
2019-09-18  114.64   2019-09-24  109.98  4     280.3  -1307.14  stop_loss
2019-09-25  110.74   2019-09-30  110.26  3     285.6  -138.00   alpha_reversal
2019-10-01  107.03   2019-10-03  104.78  2     286.9  -644.90   alpha_reversal
2019-10-08  103.18   2019-10-09  104.26  1     297.7  +320.63   alpha_reversal
2019-10-10  107.31   2019-10-14  112.06  2     295.6  +1404.61  alpha_reversal
2019-11-12  129.00   2019-11-20  124.51  6     269.6  -1212.53  stop_loss
2019-11-21  126.38   2019-11-22  126.58  1     283.1  +58.02    alpha_reversal
2019-11-27  128.43   2019-12-03  123.22  3     307.1  -1599.98  stop_loss
2019-12-04  123.52   2019-12-06  125.56  2     291.3  +594.34   alpha_reversal
2019-12-16  128.14   2019-12-23  130.65  5     301.2  +754.81   alpha_reversal
2019-12-26  130.66   2019-12-30  129.78  2     341.7  -299.61   alpha_reversal
2020-01-08  130.18   2020-01-22  126.36  9     349.7  -1336.75  stop_loss
2020-01-23  126.72   2020-01-27  120.25  2     341.0  -2207.31  stop_loss
2020-01-29  120.53   2020-01-31  116.37  2     307.2  -1279.62  stop_loss
2020-02-03  115.20   2020-02-06  121.60  3     270.9  +1732.85  alpha_reversal
2020-02-07  118.39   2020-02-14  122.25  5     250.7  +967.00   alpha_reversal
2020-02-28  110.29   2020-03-02  113.05  1     224.9  +619.92   alpha_reversal
2020-03-06  107.78   2020-03-09  92.20   1     204.7  -3188.32  stop_loss
2020-03-10  94.53    2020-03-11  89.21   1     146.2  -778.61   alpha_reversal
2020-03-12  81.90    2020-03-19  91.26   5     128.3  +1201.33  alpha_reversal
2020-03-25  92.92    2020-04-01  98.65   5     98.8   +566.57   alpha_reversal
2020-04-13  101.32   2020-04-20  102.46  5     99.4   +113.09   alpha_reversal
2020-04-21  98.41    2020-04-22  98.92   1     112.8  +57.48    alpha_reversal
2020-05-04  96.50    2020-05-15  96.49   9     131.1  -1.84     alpha_reversal
2020-05-22  100.76   2020-06-02  110.26  6     161.5  +1533.78  alpha_reversal
2020-06-12  110.33   2020-06-17  114.65  3     143.3  +619.40   alpha_reversal
2020-06-18  114.30   2020-06-24  109.13  4     147.3  -761.73   alpha_reversal
2020-07-08  114.79   2020-07-09  112.41  1     189.0  -448.90   alpha_reversal
2020-07-10  114.68   2020-07-14  122.38  2     196.0  +1509.49  alpha_reversal
2020-07-21  122.63   2020-07-27  126.01  4     204.9  +691.60   alpha_reversal
2020-08-06  121.30   2020-08-11  128.39  3     231.7  +1643.05  alpha_reversal
2020-08-18  124.89   2020-09-01  131.56  10    241.5  +1610.27  max_holding
2020-09-15  134.12   2020-09-18  137.27  3     197.7  +621.83   alpha_reversal
2020-09-21  131.17   2020-09-23  130.05  2     176.7  -197.48   alpha_reversal
2020-09-24  131.00   2020-09-25  131.43  1     177.3  +76.57    alpha_reversal
2020-09-30  134.62   2020-10-05  138.26  3     187.1  +680.94   alpha_reversal
2020-11-05  148.10   2020-11-16  156.96  7     139.3  +1233.79  alpha_reversal
2020-11-17  156.11   2020-11-19  156.70  2     150.1  +87.84    alpha_reversal
2020-11-23  158.72   2020-11-24  160.32  1     171.5  +275.12   alpha_reversal
2020-11-27  158.99   2020-11-30  157.32  1     183.6  -306.23   alpha_reversal
2020-12-01  157.34   2020-12-07  161.96  4     190.5  +880.80   alpha_reversal
2020-12-08  162.40   2020-12-09  163.19  1     188.4  +148.87   alpha_reversal
2020-12-17  162.70   2020-12-21  163.27  2     201.0  +114.99   alpha_reversal
2020-12-22  160.35   2020-12-24  162.73  2     201.1  +478.66   alpha_reversal
2020-12-29  160.41   2021-01-04  165.08  3     218.0  +1016.90  alpha_reversal
2021-01-14  179.26   2021-01-25  170.69  6     181.6  -1556.64  stop_loss
2021-02-10  180.50   2021-02-12  180.39  2     175.0  -20.06    alpha_reversal
2021-02-16  184.76   2021-02-17  184.32  1     175.9  -77.77    alpha_reversal
2021-02-18  182.51   2021-02-22  198.67  2     180.5  +2917.21  alpha_reversal
2021-03-03  196.20   2021-03-11  200.22  6     148.4  +597.72   alpha_reversal
2021-03-12  209.06   2021-03-15  210.23  1     133.7  +156.09   alpha_reversal
2021-03-22  206.34   2021-03-23  198.85  1     133.0  -996.16   alpha_reversal
2021-04-01  212.47   2021-04-06  210.14  2     125.4  -293.15   alpha_reversal
2021-04-07  210.35   2021-04-13  208.96  4     131.8  -182.71   alpha_reversal
2021-04-14  212.33   2021-04-15  211.85  1     145.8  -69.83    alpha_reversal
2021-04-16  213.04   2021-04-19  211.73  1     154.9  -202.89   alpha_reversal
2021-04-20  207.81   2021-04-26  211.01  4     155.2  +497.16   alpha_reversal
2021-05-07  221.00   2021-05-10  222.04  1     143.8  +149.75   alpha_reversal
2021-05-21  217.56   2021-05-27  220.82  4     147.2  +480.14   alpha_reversal
2021-06-01  222.62   2021-06-03  223.17  2     163.6  +88.97    alpha_reversal
2021-06-08  220.24   2021-06-09  214.76  1     163.6  -896.88   alpha_reversal
2021-06-10  207.01   2021-06-16  198.75  4     148.4  -1225.12  stop_loss
2021-06-17  192.08   2021-06-25  197.97  6     139.8  +823.89   alpha_reversal
2021-06-28  198.33   2021-07-01  198.34  3     144.5  +0.93     alpha_reversal
2021-07-02  199.70   2021-07-06  195.42  1     159.6  -683.19   alpha_reversal
2021-07-09  199.38   2021-07-12  200.05  1     156.4  +103.76   alpha_reversal
2021-07-14  194.08   2021-07-19  186.86  3     162.1  -1171.31  stop_loss
2021-07-21  194.29   2021-08-04  188.18  10    157.8  -963.10   max_holding
2021-08-05  191.02   2021-08-10  196.97  3     161.0  +957.68   alpha_reversal
2021-08-20  188.95   2021-09-03  193.57  10    160.8  +742.56   max_holding
2021-09-07  191.98   2021-09-17  183.80  8     194.7  -1593.97  stop_loss
2021-09-20  175.93   2021-09-29  182.07  7     174.7  +1071.68  alpha_reversal
2021-10-06  176.89   2021-10-20  187.88  10    177.0  +1945.66  max_holding
2021-11-04  188.74   2021-11-09  194.12  3     185.7  +998.53   alpha_reversal
2021-11-16  190.45   2021-11-26  183.87  7     183.1  -1204.52  stop_loss
2021-11-30  179.25   2021-12-06  186.23  4     160.5  +1120.16  alpha_reversal
2021-12-09  189.24   2021-12-15  187.40  4     166.8  -306.80   alpha_reversal
2021-12-20  181.20   2021-12-29  191.82  6     163.1  +1732.19  alpha_reversal
2021-12-31  191.66   2022-01-03  191.52  1     201.2  -28.65    alpha_reversal
2022-01-20  201.80   2022-01-24  199.26  2     163.6  -415.78   alpha_reversal
2022-02-03  186.86   2022-02-08  187.33  3     131.2  +62.02    alpha_reversal
2022-02-15  189.49   2022-02-17  181.05  2     156.7  -1322.73  stop_loss
2022-02-25  174.26   2022-03-03  181.15  4     154.0  +1061.67  alpha_reversal
2022-03-17  206.53   2022-03-18  205.38  1     118.9  -136.34   alpha_reversal
2022-03-21  208.40   2022-03-22  207.05  1     123.5  -166.20   alpha_reversal
2022-03-23  206.96   2022-03-28  206.25  3     131.1  -93.19    alpha_reversal
2022-03-29  205.91   2022-03-31  207.16  2     144.1  +179.22   alpha_reversal
2022-04-04  205.70   2022-04-05  200.83  1     148.6  -722.61   alpha_reversal
2022-04-08  202.18   2022-04-11  200.86  1     156.7  -206.12   alpha_reversal
2022-05-19  193.53   2022-05-25  195.53  4     114.5  +228.54   alpha_reversal
2022-05-26  199.36   2022-05-31  201.64  2     117.0  +266.08   alpha_reversal
2022-06-16  182.33   2022-06-23  166.91  4     119.4  -1840.95  stop_loss
2022-06-24  173.62   2022-07-05  162.32  6     110.8  -1252.93  stop_loss
2022-07-06  161.29   2022-07-19  168.84  9     116.2  +877.42   alpha_reversal
2022-07-25  171.36   2022-07-26  170.48  1     136.9  -121.54   alpha_reversal
2022-08-09  175.34   2022-08-10  179.40  1     143.9  +584.30   alpha_reversal
2022-08-16  186.17   2022-08-17  184.30  1     157.1  -293.43   alpha_reversal
2022-09-02  170.44   2022-09-14  173.69  7     155.5  +505.90   alpha_reversal
2022-09-19  171.65   2022-09-21  161.75  2     155.9  -1543.48  stop_loss
2022-09-22  160.75   2022-09-26  152.97  2     147.3  -1146.34  stop_loss
2022-09-27  153.11   2022-09-29  156.06  2     145.6  +429.31   alpha_reversal
2022-10-13  172.62   2022-10-14  167.62  1     136.6  -683.18   alpha_reversal
2022-10-18  173.00   2022-10-20  169.83  2     135.1  -429.26   alpha_reversal
2022-11-10  220.56   2022-11-15  222.15  3     112.1  +177.70   alpha_reversal
2022-11-16  220.53   2022-11-22  224.67  4     118.2  +489.51   alpha_reversal
2022-11-25  223.64   2022-11-30  223.87  3     133.0  +29.99    alpha_reversal
2022-12-01  223.64   2022-12-05  220.23  2     138.6  -471.50   alpha_reversal
2022-12-06  216.61   2022-12-08  218.67  2     138.4  +284.78   alpha_reversal
2022-12-12  221.14   2022-12-16  220.38  4     142.5  -108.80   alpha_reversal
2023-01-03  226.66   2023-01-18  239.41  10    148.9  +1897.89  max_holding
2023-01-20  238.07   2023-01-25  245.90  3     145.3  +1137.85  alpha_reversal
2023-02-08  237.27   2023-02-13  236.11  3     129.1  -149.66   alpha_reversal
2023-02-14  232.97   2023-02-24  224.71  7     131.1  -1082.48  trailing_stop
2023-02-27  228.79   2023-03-02  239.94  3     134.5  +1500.07  alpha_reversal
2023-03-13  214.22   2023-03-27  207.42  10    115.9  -788.42   max_holding
2023-04-05  203.57   2023-04-17  214.31  7     120.4  +1292.04  alpha_reversal
2023-05-08  205.89   2023-05-22  204.97  10    139.3  -127.92   alpha_reversal
2023-05-23  203.89   2023-06-05  212.82  8     147.4  +1315.87  alpha_reversal
2023-06-23  224.72   2023-06-27  230.25  2     130.8  +723.24   alpha_reversal
2023-07-06  230.61   2023-07-07  234.55  1     149.6  +589.22   alpha_reversal
2023-07-17  246.79   2023-07-19  252.60  2     148.7  +864.57   alpha_reversal
2023-08-21  262.74   2023-08-31  270.27  8     128.6  +968.65   alpha_reversal
2023-09-08  271.92   2023-09-11  271.16  1     143.3  -109.55   alpha_reversal
2023-09-12  271.47   2023-09-15  268.37  3     145.7  -451.50   alpha_reversal
2023-09-18  270.63   2023-09-19  268.87  1     140.1  -246.86   alpha_reversal
2023-10-05  251.23   2023-10-18  249.21  9     136.1  -275.07   trailing_stop
2023-10-19  243.61   2023-10-27  230.53  6     116.4  -1522.31  stop_loss
2023-11-06  230.72   2023-11-16  239.86  8     109.8  +1003.17  alpha_reversal
2023-11-21  241.50   2023-11-28  239.17  4     133.4  -310.57   alpha_reversal
2023-12-22  281.15   2023-12-27  288.09  2     135.3  +938.67   alpha_reversal
2023-12-29  286.29   2024-01-03  274.73  2     143.3  -1656.09  stop_loss
2024-01-12  281.20   2024-01-17  269.25  2     138.8  -1658.18  stop_loss
2024-01-18  272.96   2024-01-25  291.99  5     132.4  +2519.57  alpha_reversal
2024-02-09  308.52   2024-02-22  312.69  8     109.1  +454.94   alpha_reversal
2024-03-12  328.05   2024-03-15  336.85  3     128.2  +1126.84  alpha_reversal
2024-03-26  346.68   2024-03-27  354.01  1     128.1  +938.31   alpha_reversal
2024-04-12  355.67   2024-04-24  354.20  8     105.0  -154.69   alpha_reversal
2024-05-01  323.23   2024-05-08  335.67  5     80.8   +1005.64  alpha_reversal
2024-05-22  347.51   2024-05-29  329.44  4     105.0  -1896.90  stop_loss
2024-05-30  331.21   2024-06-04  319.17  3     103.0  -1240.91  stop_loss
2024-06-05  321.64   2024-06-13  317.99  6     100.8  -368.35   alpha_reversal
2024-06-17  314.76   2024-06-21  319.43  3     106.1  +495.82   alpha_reversal
2024-06-24  322.18   2024-06-25  319.00  1     112.8  -358.39   alpha_reversal
2024-06-27  319.70   2024-06-28  324.56  1     120.5  +585.06   alpha_reversal
2024-07-02  319.99   2024-07-05  319.93  2     119.5  -7.71     alpha_reversal
2024-07-08  321.26   2024-07-09  317.70  1     123.4  -439.89   alpha_reversal
2024-07-10  320.70   2024-07-15  336.90  3     125.1  +2027.76  alpha_reversal
2024-07-24  329.41   2024-07-29  337.06  3     100.3  +767.23   alpha_reversal
2024-07-31  339.37   2024-08-01  324.33  1     95.2   -1432.48  stop_loss
2024-08-15  338.83   2024-08-27  343.51  8     82.1   +384.53   alpha_reversal
2024-08-28  341.34   2024-08-30  348.38  2     105.7  +744.72   alpha_reversal
2024-09-04  330.11   2024-09-17  346.02  9     99.3   +1580.02  alpha_reversal
2024-09-23  363.85   2024-09-25  370.05  2     96.2   +596.35   alpha_reversal
2024-10-04  389.26   2024-10-07  389.62  1     93.4   +33.35    alpha_reversal
2024-10-09  388.38   2024-10-11  393.31  2     92.4   +455.11   alpha_reversal
2024-10-15  380.34   2024-10-29  380.47  10    92.9   +12.32    alpha_reversal
2024-11-05  377.16   2024-11-07  400.80  2     100.7  +2380.77  alpha_reversal
2024-11-08  387.00   2024-11-15  377.10  5     79.4   -786.77   alpha_reversal
2024-11-20  375.32   2024-11-26  400.42  4     96.0   +2409.51  alpha_reversal
2024-12-03  392.79   2024-12-12  373.87  7     97.0   -1836.20  stop_loss
2024-12-17  369.71   2024-12-18  357.28  1     104.8  -1303.19  stop_loss
2024-12-19  354.53   2024-12-23  358.75  2     94.4   +398.27   alpha_reversal
2024-12-24  361.62   2025-01-10  344.63  10    96.8   -1645.20  stop_loss
2025-01-13  356.63   2025-01-14  364.82  1     102.9  +843.42   alpha_reversal
2025-02-06  360.85   2025-02-14  348.18  6     86.6   -1097.89  alpha_reversal
2025-02-18  349.54   2025-02-21  335.09  3     98.6   -1424.65  stop_loss
2025-02-24  334.65   2025-02-26  337.59  2     93.6   +275.13   alpha_reversal
2025-03-05  334.23   2025-03-06  336.05  1     83.8   +152.12   alpha_reversal
2025-03-13  329.11   2025-03-17  337.65  2     83.0   +708.20   alpha_reversal
2025-03-19  334.36   2025-04-02  329.79  10    91.0   -415.69   alpha_reversal
2025-04-04  284.45   2025-04-21  281.95  10    73.3   -183.85   max_holding
2025-05-06  318.38   2025-05-12  339.19  4     76.6   +1593.13  alpha_reversal
2025-06-02  341.97   2025-06-12  357.42  8     101.0  +1559.46  alpha_reversal
2025-06-13  354.25   2025-06-17  354.17  2     119.3  -10.02    alpha_reversal
2025-06-18  356.98   2025-06-23  362.64  2     119.8  +677.50   alpha_reversal
2025-07-07  388.44   2025-07-09  398.23  2     114.6  +1122.17  alpha_reversal
2025-07-22  415.44   2025-07-23  424.94  1     113.5  +1079.00  alpha_reversal
2025-08-11  406.83   2025-08-14  414.92  3     90.6   +733.47   alpha_reversal
2025-08-18  410.91   2025-08-20  417.99  2     92.0   +651.06   alpha_reversal
2025-08-22  433.84   2025-08-25  429.63  1     86.5   -364.68   alpha_reversal
2025-08-26  429.45   2025-08-28  432.22  2     92.6   +256.56   alpha_reversal
2025-09-02  414.30   2025-09-05  420.46  3     91.0   +560.60   alpha_reversal
2025-09-09  416.34   2025-09-18  464.07  7     99.4   +4743.40  alpha_reversal
2025-09-24  467.82   2025-09-25  460.85  1     88.0   -613.22   alpha_reversal
2025-10-22  513.22   2025-10-30  581.21  6     66.5   +4523.18  alpha_reversal
2025-11-10  570.09   2025-11-11  566.04  1     52.5   -212.49   alpha_reversal
2025-11-12  572.25   2025-11-18  545.06  4     54.7   -1487.50  stop_loss
2025-12-02  581.69   2025-12-12  595.90  8     56.7   +805.85   alpha_reversal
2025-12-15  588.97   2025-12-17  560.02  2     51.7   -1497.21  stop_loss
2025-12-18  565.07   2025-12-24  581.82  4     49.9   +835.84   alpha_reversal
2025-12-29  577.84   2026-01-06  621.01  5     61.3   +2648.82  alpha_reversal
2026-02-05  678.99   2026-02-06  725.47  1     41.1   +1909.43  alpha_reversal
2026-02-19  761.29   2026-03-03  721.46  8     37.5   -1491.94  stop_loss
2026-03-09  705.52   2026-03-13  693.30  4     31.9   -389.76   alpha_reversal
2026-03-16  700.48   2026-03-17  701.30  1     34.7   +28.36    alpha_reversal
2026-03-19  689.34   2026-03-25  718.32  4     36.6   +1059.38  alpha_reversal

**Best 3 trades:**

- 2025-09-18: P&L = **+4743.40** (alpha_reversal)
- 2025-10-30: P&L = **+4523.18** (alpha_reversal)
- 2017-04-25: P&L = **+3394.32** (alpha_reversal)

**Worst 3 trades:**

- 2020-03-09: P&L = **-3188.32** (stop_loss)
- 2018-10-23: P&L = **-2495.81** (stop_loss)
- 2018-06-19: P&L = **-2469.21** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  102,708.77
2017-03-23  99,396.26
2017-09-20  107,232.30
2018-03-21  114,837.18
2018-09-18  113,260.60
2019-03-20  112,986.76
2019-09-17  106,148.65
2020-03-17  99,501.31
2020-09-14  106,941.69
2021-03-15  114,098.95
2021-09-10  110,616.50
2022-03-10  114,350.78
2022-09-08  111,966.30
2023-03-09  112,491.35
2023-09-07  117,328.61
2024-03-07  116,014.82
2024-09-05  117,848.49
2025-03-07  119,233.87
2025-09-05  127,239.10
2026-03-06  137,364.95

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  0.00%
2017-03-23  -5.13%
2017-09-20  -0.09%
2018-03-21  -0.92%
2018-09-18  -2.35%
2019-03-20  -3.16%
2019-09-17  -9.02%
2020-03-17  -14.72%
2020-09-14  -8.34%
2021-03-15  -2.21%
2021-09-10  -5.19%
2022-03-10  -1.99%
2022-09-08  -4.03%
2023-03-09  -3.58%
2023-09-07  -0.16%
2024-03-07  -1.78%
2024-09-05  -1.06%
2025-03-07  -4.54%
2025-09-05  0.00%
2026-03-06  -1.26%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  32.36%
Out-of-Sample (30%)  2023-03-24  2026-03-27  33.59%

#### Return Distribution

Return Bin          Count
-3.069% to -2.455%  1
-2.455% to -1.842%  2
-1.842% to -1.228%  8
-1.228% to -0.614%  73
-0.614% to -0.000%  749
-0.000% to 0.613%   1578
0.613% to 1.227%    89
1.227% to 1.841%    9
1.841% to 2.455%    4
2.455% to 3.069%    3

### SPY — AlphaCombined

**Net Return (after slippage):** 19.55%  **vs SPY (exposure-adj): -109.86%** (underperform)  
**Gross Return (pre-cost):** 45.10%  
**Total Slippage Cost:** $25,545.13  
**Trade Count:** 361  
**Win Rate:** 54.0%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size   P&L       Exit Reason
2016-05-02  176.78   2016-05-03  175.06  1     407.4  -698.22   alpha_reversal
2016-05-04  174.26   2016-05-10  177.01  4     396.9  +1090.23  alpha_reversal
2016-05-18  174.18   2016-05-19  173.40  1     374.6  -291.03   alpha_reversal
2016-05-20  174.67   2016-05-27  178.53  5     375.8  +1450.03  alpha_reversal
2016-06-01  178.73   2016-06-02  179.10  1     431.7  +157.53   alpha_reversal
2016-06-13  177.19   2016-06-17  176.28  4     455.4  -411.63   alpha_reversal
2016-06-28  173.62   2016-07-01  179.18  3     305.7  +1700.55  alpha_reversal
2016-07-08  181.70   2016-07-12  183.48  2     310.0  +552.27   alpha_reversal
2016-07-14  184.66   2016-07-19  184.54  3     347.7  -43.41    alpha_reversal
2016-07-20  185.49   2016-08-03  184.53  10    400.2  -385.08   max_holding
2016-08-08  186.31   2016-08-16  186.05  6     468.8  -123.29   alpha_reversal
2016-08-17  186.58   2016-08-18  186.81  1     542.9  +125.80   alpha_reversal
2016-08-22  186.72   2016-08-24  185.95  2     586.9  -450.18   alpha_reversal
2016-08-29  186.57   2016-09-02  186.40  4     554.8  -98.73    alpha_reversal
2016-09-08  186.70   2016-09-09  182.05  1     585.7  -2724.11  stop_loss
2016-09-12  184.85   2016-09-13  182.01  1     421.5  -1196.95  stop_loss
2016-09-14  182.12   2016-09-20  183.09  4     383.9  +371.83   alpha_reversal
2016-10-03  185.30   2016-10-07  184.48  4     385.2  -315.88   alpha_reversal
2016-10-12  183.52   2016-10-13  182.74  1     386.7  -303.16   alpha_reversal
2016-10-18  183.52   2016-10-25  183.73  5     394.1  +83.24    alpha_reversal
2016-10-26  183.55   2016-11-01  181.02  4     449.8  -1135.98  stop_loss
2016-11-02  180.11   2016-11-08  183.68  4     421.9  +1505.65  alpha_reversal
2016-11-17  188.06   2016-11-23  189.34  4     387.0  +495.00   alpha_reversal
2016-12-13  195.59   2016-12-14  193.78  1     468.8  -847.78   alpha_reversal
2016-12-15  194.77   2016-12-16  194.20  1     438.0  -251.95   alpha_reversal
2016-12-19  194.82   2016-12-27  195.26  5     456.8  +202.77   alpha_reversal
2016-12-30  193.09   2017-01-17  195.24  10    512.5  +1104.13  max_holding
2017-01-27  197.79   2017-02-08  197.82  8     546.0  +19.28    alpha_reversal
2017-02-17  203.07   2017-02-22  203.90  2     545.4  +449.39   alpha_reversal
2017-02-23  204.24   2017-02-27  204.61  2     563.7  +210.87   alpha_reversal
2017-03-01  207.12   2017-03-02  205.61  1     514.6  -777.15   alpha_reversal
2017-03-07  204.72   2017-03-14  204.43  5     537.9  -156.49   alpha_reversal
2017-03-21  202.78   2017-03-30  204.79  7     488.9  +985.57   alpha_reversal
2017-04-06  204.26   2017-04-07  203.85  1     460.4  -189.79   alpha_reversal
2017-04-20  204.17   2017-04-25  206.75  3     442.9  +1141.78  alpha_reversal
2017-05-01  207.07   2017-05-02  206.94  1     487.6  -62.88    alpha_reversal
2017-05-09  207.73   2017-05-11  207.47  2     563.4  -146.29   alpha_reversal
2017-05-12  207.33   2017-05-17  204.39  3     588.7  -1734.38  trailing_stop
2017-05-18  205.41   2017-05-19  206.54  1     482.8  +545.21   alpha_reversal
2017-05-30  209.52   2017-06-13  211.95  10    561.9  +1367.75  max_holding
2017-06-22  211.71   2017-07-03  210.95  7     552.7  -420.14   alpha_reversal
2017-07-12  212.73   2017-07-14  213.87  2     465.4  +529.33   alpha_reversal
2017-07-18  214.17   2017-07-19  215.11  1     519.3  +490.37   alpha_reversal
2017-07-24  215.18   2017-08-03  215.08  8     574.5  -53.53    alpha_reversal
2017-08-04  215.69   2017-08-07  215.88  1     646.2  +119.55   alpha_reversal
2017-08-09  215.55   2017-08-10  212.30  1     637.5  -2074.98  stop_loss
2017-08-11  212.82   2017-08-14  214.72  1     559.5  +1060.24  alpha_reversal
2017-08-18  211.59   2017-08-29  213.25  7     482.8  +797.70   alpha_reversal
2017-09-12  217.99   2017-09-19  218.79  5     496.3  +393.47   alpha_reversal
2017-09-22  218.54   2017-09-28  219.12  4     640.9  +370.45   alpha_reversal
2017-10-04  221.80   2017-10-11  223.21  5     685.2  +963.51   alpha_reversal
2017-10-13  223.37   2017-10-17  223.60  2     755.7  +175.23   alpha_reversal
2017-10-26  223.96   2017-10-30  224.72  2     701.5  +536.81   alpha_reversal
2017-11-10  226.12   2017-11-14  225.58  2     626.3  -338.86   alpha_reversal
2017-11-16  226.58   2017-11-21  227.56  3     543.7  +528.81   alpha_reversal
2017-11-22  227.58   2017-11-27  227.77  2     592.7  +108.96   alpha_reversal
2017-12-04  231.42   2017-12-18  235.94  10    443.4  +2005.00  max_holding
2017-12-21  235.63   2017-12-22  235.34  1     492.3  -146.23   alpha_reversal
2017-12-27  235.40   2017-12-28  235.65  1     555.4  +138.06   alpha_reversal
2018-01-03  238.18   2018-01-09  241.52  4     540.1  +1805.18  alpha_reversal
2018-01-11  243.15   2018-01-18  245.57  4     532.3  +1284.85  alpha_reversal
2018-01-19  246.93   2018-01-22  248.69  1     469.5  +825.88   alpha_reversal
2018-01-25  249.48   2018-01-30  247.87  3     449.0  -720.20   trailing_stop
2018-01-31  248.24   2018-02-02  242.32  2     391.1  -2315.95  stop_loss
2018-02-05  232.42   2018-02-07  235.48  2     257.9  +788.51   alpha_reversal
2018-02-08  226.87   2018-02-12  233.43  2     191.6  +1255.78  alpha_reversal
2018-02-13  234.24   2018-02-14  237.17  1     176.9  +517.17   alpha_reversal
2018-02-21  237.81   2018-03-01  235.50  6     184.1  -424.43   trailing_stop
2018-03-08  241.37   2018-03-09  245.33  1     197.2  +780.08   alpha_reversal
2018-03-13  243.68   2018-03-21  238.86  6     204.0  -983.95   alpha_reversal
2018-03-22  233.12   2018-03-27  230.17  3     211.4  -622.60   alpha_reversal
2018-04-02  227.64   2018-04-13  234.19  9     167.5  +1098.24  alpha_reversal
2018-04-16  236.36   2018-04-17  238.65  1     180.6  +413.54   alpha_reversal
2018-04-20  235.72   2018-04-30  233.63  6     201.6  -421.36   alpha_reversal
2018-05-04  235.20   2018-05-08  235.76  2     210.0  +117.55   alpha_reversal
2018-05-11  241.24   2018-05-14  241.11  1     241.0  -30.45    alpha_reversal
2018-05-18  239.89   2018-06-04  242.81  10    280.4  +817.01   max_holding
2018-06-07  245.23   2018-06-21  243.31  10    318.7  -612.51   alpha_reversal
2018-06-26  241.21   2018-07-02  241.20  4     349.6  -3.65     alpha_reversal
2018-07-05  242.55   2018-07-10  247.44  3     319.8  +1565.36  alpha_reversal
2018-07-17  249.09   2018-07-19  248.42  2     360.2  -239.91   alpha_reversal
2018-07-23  248.85   2018-07-24  249.85  1     406.9  +407.80   alpha_reversal
2018-08-02  250.79   2018-08-06  252.54  2     373.2  +651.48   alpha_reversal
2018-08-09  253.17   2018-08-13  250.28  2     433.0  -1250.68  stop_loss
2018-08-16  252.27   2018-08-21  254.05  3     373.5  +661.27   alpha_reversal
2018-08-23  253.81   2018-09-04  257.12  7     420.1  +1391.84  alpha_reversal
2018-09-10  255.86   2018-09-11  256.45  1     447.5  +262.78   alpha_reversal
2018-09-18  258.36   2018-09-21  260.23  3     448.8  +840.39   alpha_reversal
2018-09-24  259.63   2018-10-05  256.51  9     466.8  -1452.57  stop_loss
2018-10-08  256.77   2018-10-10  248.03  2     388.9  -3399.33  stop_loss
2018-10-11  242.81   2018-10-15  244.55  2     263.3  +459.42   alpha_reversal
2018-10-17  250.20   2018-10-22  245.10  3     238.6  -1216.67  stop_loss
2018-10-23  244.09   2018-10-24  236.46  1     216.0  -1648.41  stop_loss
2018-10-29  235.40   2018-11-06  245.20  6     158.9  +1557.17  alpha_reversal
2018-11-09  247.80   2018-11-14  240.81  3     179.8  -1256.22  stop_loss
2018-11-26  238.64   2018-11-28  244.71  2     176.8  +1073.48  alpha_reversal
2018-11-30  245.91   2018-12-07  234.90  4     187.4  -2063.44  stop_loss
2018-12-10  235.58   2018-12-11  235.40  1     150.2  -27.32    alpha_reversal
2018-12-14  232.37   2018-12-19  223.93  3     154.0  -1299.56  stop_loss
2018-12-20  220.51   2018-12-24  210.07  2     137.8  -1437.91  stop_loss
2019-01-02  224.49   2019-01-08  230.18  4     124.2  +706.11   alpha_reversal
2019-01-15  233.62   2019-01-16  233.95  1     153.3  +50.77    alpha_reversal
2019-01-30  240.11   2019-02-05  244.82  4     180.8  +851.37   alpha_reversal
2019-02-07  242.41   2019-02-12  245.71  3     212.7  +703.61   alpha_reversal
2019-02-22  250.48   2019-03-07  246.53  9     266.1  -1051.82  trailing_stop
2019-03-08  246.28   2019-03-19  254.27  7     291.3  +2326.37  alpha_reversal
2019-03-25  251.49   2019-03-29  254.34  4     274.8  +782.10   alpha_reversal
2019-04-01  257.61   2019-04-03  257.89  2     274.1  +75.04    alpha_reversal
2019-04-12  261.52   2019-04-23  263.70  6     351.0  +767.97   alpha_reversal
2019-04-25  263.22   2019-05-01  262.74  4     390.0  -186.88   alpha_reversal
2019-05-02  262.44   2019-05-07  259.25  3     367.1  -1170.44  trailing_stop
2019-05-08  259.15   2019-05-13  252.88  3     286.8  -1796.37  stop_loss
2019-05-14  255.42   2019-05-22  257.18  6     219.8  +385.13   alpha_reversal
2019-05-23  254.29   2019-05-28  252.24  2     222.8  -455.74   alpha_reversal
2019-06-12  259.92   2019-06-14  260.45  2     233.7  +122.37   alpha_reversal
2019-06-17  260.80   2019-06-18  263.27  1     265.2  +654.40   alpha_reversal
2019-06-25  263.33   2019-06-28  265.10  3     280.6  +494.92   alpha_reversal
2019-07-09  269.16   2019-07-10  270.17  1     306.9  +311.73   alpha_reversal
2019-07-17  269.65   2019-07-31  269.10  10    353.9  -194.64   max_holding
2019-08-01  267.03   2019-08-05  256.79  2     285.8  -2926.17  stop_loss
2019-08-06  260.65   2019-08-07  260.54  1     220.5  -23.52    alpha_reversal
2019-08-15  257.80   2019-08-22  264.52  5     175.5  +1178.89  alpha_reversal
2019-08-23  257.98   2019-08-29  264.72  4     173.7  +1169.92  alpha_reversal
2019-09-03  263.31   2019-09-05  269.46  2     190.2  +1168.14  alpha_reversal
2019-09-06  269.93   2019-09-09  269.80  1     202.1  -27.11    alpha_reversal
2019-09-24  269.20   2019-10-01  266.54  5     256.7  -682.73   alpha_reversal
2019-10-02  262.09   2019-10-04  267.55  2     224.2  +1223.15  alpha_reversal
2019-10-09  265.01   2019-10-11  269.30  2     215.3  +923.52   alpha_reversal
2019-10-15  271.94   2019-10-16  271.23  1     222.0  -157.23   alpha_reversal
2019-10-17  272.30   2019-10-18  270.84  1     240.5  -351.87   alpha_reversal
2019-10-23  272.85   2019-10-24  273.02  1     267.3  +46.14    alpha_reversal
2019-10-30  276.72   2019-11-04  279.38  3     295.1  +784.75   alpha_reversal
2019-11-06  279.42   2019-11-18  283.61  8     327.9  +1374.93  alpha_reversal
2019-11-20  282.75   2019-11-27  286.75  5     372.7  +1490.21  alpha_reversal
2019-12-03  281.64   2019-12-06  286.20  3     366.9  +1670.97  alpha_reversal
2019-12-10  285.27   2019-12-13  288.43  3     379.7  +1199.79  alpha_reversal
2020-01-03  294.79   2020-01-17  303.21  10    394.3  +3319.70  max_holding
2020-01-27  295.78   2020-02-07  303.44  9     331.3  +2534.53  alpha_reversal
2020-02-19  309.35   2020-02-20  307.77  1     304.0  -479.98   alpha_reversal
2020-02-24  294.80   2020-02-25  285.58  1     231.2  -2131.77  stop_loss
2020-02-26  284.81   2020-02-27  271.75  1     185.1  -2417.71  stop_loss
2020-02-28  270.88   2020-03-09  250.48  6     144.9  -2954.10  trailing_stop
2020-03-23  205.05   2020-03-31  236.82  6     55.8   +1773.70  alpha_reversal
2020-04-01  226.38   2020-04-03  228.03  2     59.6   +98.15    alpha_reversal
2020-04-06  243.59   2020-04-07  243.60  1     61.4   +0.29     alpha_reversal
2020-04-14  261.00   2020-04-15  255.20  1     68.9   -399.44   alpha_reversal
2020-04-16  256.69   2020-04-17  263.36  1     73.5   +490.07   alpha_reversal
2020-04-21  251.12   2020-04-23  256.41  2     78.7   +416.77   alpha_reversal
2020-04-24  260.25   2020-04-27  263.74  1     86.3   +301.12   alpha_reversal
2020-04-29  269.67   2020-05-04  260.54  3     91.8   -837.61   alpha_reversal
2020-05-07  264.58   2020-05-08  268.69  1     105.3  +432.69   alpha_reversal
2020-05-14  262.09   2020-05-26  274.79  7     106.3  +1350.05  alpha_reversal
2020-06-05  293.70   2020-06-08  296.95  1     136.5  +443.96   alpha_reversal
2020-06-10  293.38   2020-06-11  276.19  1     147.1  -2528.76  stop_loss
2020-06-12  279.78   2020-06-15  282.11  1     118.9  +276.97   alpha_reversal
2020-07-17  297.19   2020-07-20  299.29  1     146.9  +308.87   alpha_reversal
2020-08-12  311.71   2020-08-18  312.51  4     193.7  +154.19   alpha_reversal
2020-08-20  312.49   2020-08-21  313.28  1     227.4  +180.83   alpha_reversal
2020-08-31  322.67   2020-09-03  318.73  3     251.9  -992.55   trailing_stop
2020-09-04  316.45   2020-09-08  307.49  1     166.3  -1489.46  stop_loss
2020-09-09  313.88   2020-09-10  308.12  1     145.8  -839.73   alpha_reversal
2020-09-11  308.59   2020-09-23  298.93  8     137.4  -1326.58  trailing_stop
2020-09-30  310.59   2020-10-06  310.32  4     130.5  -35.70    alpha_reversal
2020-10-16  322.09   2020-10-26  314.45  6     146.9  -1122.77  alpha_reversal
2020-10-27  313.68   2020-10-28  302.66  1     149.1  -1643.92  stop_loss
2020-10-29  306.04   2020-10-30  302.55  1     130.3  -455.31   alpha_reversal
2020-11-02  306.24   2020-11-05  324.50  3     128.0  +2336.75  alpha_reversal
2020-11-06  324.75   2020-11-10  328.02  2     120.8  +395.19   alpha_reversal
2020-11-18  330.43   2020-11-30  335.46  7     128.5  +645.95   alpha_reversal
2020-12-04  343.02   2020-12-09  339.89  3     170.4  -532.04   alpha_reversal
2020-12-10  340.12   2020-12-24  343.34  10    181.3  +583.96   alpha_reversal
2021-01-05  345.86   2021-01-07  352.74  2     176.4  +1214.15  alpha_reversal
2021-01-12  352.79   2021-01-14  352.15  2     172.8  -110.80   alpha_reversal
2021-01-15  349.93   2021-01-22  356.26  4     180.4  +1142.17  alpha_reversal
2021-01-27  348.73   2021-01-29  344.34  2     171.6  -752.61   alpha_reversal
2021-02-01  350.42   2021-02-12  365.34  9     149.5  +2229.66  alpha_reversal
2021-02-17  365.47   2021-02-24  364.53  5     191.2  -180.16   alpha_reversal
2021-02-26  354.27   2021-03-02  359.66  2     155.4  +838.50   alpha_reversal
2021-03-04  350.86   2021-03-08  355.18  2     137.4  +593.58   alpha_reversal
2021-03-09  360.61   2021-03-10  362.49  1     126.5  +238.11   alpha_reversal
2021-03-17  370.01   2021-03-23  363.60  4     145.3  -930.54   alpha_reversal
2021-03-31  370.35   2021-04-08  381.36  5     158.0  +1739.54  alpha_reversal
2021-04-15  388.61   2021-04-16  389.52  1     203.3  +184.84   alpha_reversal
2021-04-28  390.04   2021-05-12  378.46  10    219.9  -2546.52  trailing_stop
2021-05-13  383.39   2021-05-14  388.88  1     167.9  +923.18   alpha_reversal
2021-05-17  388.28   2021-05-25  390.43  6     168.0  +361.30   alpha_reversal
2021-05-26  391.60   2021-05-27  391.41  1     178.4  -33.20    alpha_reversal
2021-06-02  392.78   2021-06-03  390.93  1     204.8  -378.57   alpha_reversal
2021-06-04  394.90   2021-06-09  393.62  3     203.6  -260.96   alpha_reversal
2021-06-10  395.84   2021-06-14  396.99  2     227.1  +259.94   alpha_reversal
2021-06-17  394.31   2021-06-18  388.60  1     240.5  -1372.79  stop_loss
2021-06-21  394.56   2021-06-22  396.27  1     215.0  +368.28   alpha_reversal
2021-06-24  398.53   2021-06-30  400.91  4     227.6  +540.37   alpha_reversal
2021-07-09  408.30   2021-07-13  407.96  2     229.9  -78.72    alpha_reversal
2021-07-20  404.12   2021-07-21  406.99  1     194.7  +557.71   alpha_reversal
2021-07-22  408.25   2021-07-23  412.03  1     204.8  +775.81   alpha_reversal
2021-07-29  413.11   2021-08-09  414.09  7     214.9  +209.15   alpha_reversal
2021-08-10  415.02   2021-08-11  415.63  1     254.8  +156.79   alpha_reversal
2021-08-18  411.73   2021-08-20  415.24  2     243.3  +852.24   alpha_reversal
2021-09-13  418.67   2021-09-14  416.00  1     251.0  -671.51   alpha_reversal
2021-09-17  415.14   2021-09-20  407.81  1     224.5  -1645.51  stop_loss
2021-09-21  407.83   2021-09-28  407.51  5     187.3  -60.52    trailing_stop
2021-09-29  408.60   2021-10-04  402.74  3     173.8  -1019.78  alpha_reversal
2021-10-05  407.34   2021-10-08  411.40  3     147.0  +597.51   alpha_reversal
2021-10-11  408.83   2021-10-18  420.17  5     147.2  +1668.22  alpha_reversal
2021-10-25  428.45   2021-10-26  428.41  1     179.8  -7.73     alpha_reversal
2021-11-10  436.04   2021-11-12  439.03  2     210.5  +630.09   alpha_reversal
2021-11-23  440.34   2021-11-24  441.07  1     221.9  +162.95   alpha_reversal
2021-11-29  436.96   2021-11-30  428.03  1     181.4  -1619.55  stop_loss
2021-12-01  423.70   2021-12-03  426.02  2     146.4  +339.71   alpha_reversal
2021-12-06  431.50   2021-12-16  438.26  8     130.4  +882.10   alpha_reversal
2021-12-17  434.03   2021-12-22  440.97  3     129.3  +897.41   alpha_reversal
2021-12-29  450.65   2021-12-30  448.96  1     140.3  -237.86   alpha_reversal
2021-12-31  448.27   2022-01-07  439.46  5     151.2  -1332.62  stop_loss
2022-01-11  443.36   2022-01-13  437.99  2     137.7  -738.58   alpha_reversal
2022-01-19  426.37   2022-01-21  412.96  2     127.2  -1705.70  stop_loss
2022-01-24  415.13   2022-01-25  409.65  1     99.4   -544.28   alpha_reversal
2022-01-26  409.03   2022-01-31  424.21  3     89.8   +1363.29  alpha_reversal
2022-02-02  431.65   2022-02-09  431.40  5     89.3   -22.52    alpha_reversal
2022-03-01  405.82   2022-03-08  392.47  5     85.5   -1141.94  trailing_stop
2022-03-09  403.40   2022-03-23  419.75  10    80.6   +1318.62  alpha_reversal
2022-03-24  426.50   2022-03-25  428.15  1     94.0   +155.51   alpha_reversal
2022-04-18  414.65   2022-04-19  420.92  1     114.5  +718.35   alpha_reversal
2022-04-21  414.73   2022-04-22  402.95  1     110.7  -1304.69  stop_loss
2022-04-25  405.69   2022-04-26  393.55  1     102.1  -1239.69  stop_loss
2022-05-02  392.41   2022-05-03  393.81  1     86.5   +121.53   alpha_reversal
2022-05-06  389.43   2022-05-11  371.46  3     78.9   -1418.48  stop_loss
2022-05-17  386.58   2022-05-18  370.62  1     77.2   -1231.12  stop_loss
2022-05-20  368.88   2022-05-24  372.54  2     71.3   +261.16   alpha_reversal
2022-05-25  376.21   2022-05-26  383.34  1     74.8   +533.26   alpha_reversal
2022-06-02  395.16   2022-06-07  393.21  3     78.0   -152.46   alpha_reversal
2022-06-09  380.06   2022-06-10  368.67  1     83.8   -954.76   alpha_reversal
2022-06-13  355.03   2022-06-16  346.78  3     77.5   -639.90   alpha_reversal
2022-06-27  369.49   2022-07-07  369.50  7     81.3   +0.87     alpha_reversal
2022-07-11  365.34   2022-07-25  375.75  10    93.4   +971.92   max_holding
2022-07-27  381.32   2022-08-01  390.18  3     99.4   +880.67   alpha_reversal
2022-08-05  393.14   2022-08-16  408.17  7     112.0  +1682.48  alpha_reversal
2022-08-23  392.08   2022-08-25  398.49  2     127.3  +816.02   alpha_reversal
2022-08-29  382.84   2022-09-02  372.58  4     122.6  -1256.67  stop_loss
2022-09-06  371.55   2022-09-09  386.22  3     115.3  +1691.28  alpha_reversal
2022-09-13  373.77   2022-09-21  359.95  6     104.0  -1438.10  stop_loss
2022-09-22  357.28   2022-09-29  346.02  5     105.8  -1190.97  stop_loss
2022-09-30  341.01   2022-10-10  343.38  6     95.2   +225.33   trailing_stop
2022-10-11  341.55   2022-10-14  341.10  3     92.1   -41.12    alpha_reversal
2022-10-17  350.22   2022-10-24  361.36  5     83.5   +930.79   alpha_reversal
2022-10-28  371.41   2022-11-02  357.55  3     92.2   -1278.23  stop_loss
2022-11-03  354.22   2022-11-08  364.35  3     93.1   +943.20   alpha_reversal
2022-11-15  380.46   2022-11-23  383.82  6     92.3   +310.90   alpha_reversal
2022-11-28  377.99   2022-12-01  388.55  3     118.2  +1248.43  alpha_reversal
2022-12-15  372.00   2022-12-28  360.90  8     109.3  -1212.91  stop_loss
2023-01-04  368.07   2023-01-19  372.38  10    115.0  +495.47   max_holding
2023-01-25  383.99   2023-02-02  399.35  6     125.7  +1929.85  alpha_reversal
2023-02-07  398.22   2023-02-08  393.47  1     123.9  -588.15   alpha_reversal
2023-02-15  397.06   2023-02-16  391.20  1     129.9  -761.15   alpha_reversal
2023-02-21  382.78   2023-02-24  379.80  3     128.6  -383.14   alpha_reversal
2023-02-27  381.47   2023-03-07  381.61  6     131.6  +17.91    alpha_reversal
2023-03-08  382.61   2023-03-09  375.18  1     143.5  -1067.19  alpha_reversal
2023-03-10  370.14   2023-03-15  373.00  3     128.2  +366.41   alpha_reversal
2023-03-16  379.92   2023-03-22  377.14  4     114.5  -318.02   alpha_reversal
2023-03-29  386.41   2023-03-30  388.29  1     122.5  +229.51   alpha_reversal
2023-04-03  395.66   2023-04-04  393.07  1     132.6  -343.27   alpha_reversal
2023-04-05  392.43   2023-04-10  393.97  2     139.3  +214.63   alpha_reversal
2023-04-20  396.55   2023-04-25  390.58  3     168.2  -1005.10  alpha_reversal
2023-04-26  389.31   2023-04-28  400.05  2     172.8  +1855.99  alpha_reversal
2023-05-02  395.55   2023-05-09  395.24  5     164.2  -50.71    alpha_reversal
2023-05-10  397.49   2023-05-11  396.40  1     165.3  -180.14   alpha_reversal
2023-05-17  399.78   2023-05-23  398.28  4     175.1  -262.01   alpha_reversal
2023-06-01  406.12   2023-06-07  410.27  4     170.8  +707.66   alpha_reversal
2023-06-20  422.47   2023-06-23  418.22  3     173.1  -736.51   alpha_reversal
2023-06-27  421.50   2023-06-30  427.94  3     180.7  +1164.01  alpha_reversal
2023-07-06  424.87   2023-07-07  423.38  1     193.4  -289.42   alpha_reversal
2023-07-28  441.55   2023-08-02  434.55  3     196.9  -1377.45  stop_loss
2023-08-03  433.74   2023-08-11  430.23  6     196.6  -690.62   alpha_reversal
2023-08-18  421.82   2023-08-21  424.14  1     170.4  +395.34   alpha_reversal
2023-09-06  431.21   2023-09-11  432.93  3     171.4  +295.20   alpha_reversal
2023-09-18  430.22   2023-09-19  428.90  1     190.0  -250.99   alpha_reversal
2023-09-22  417.41   2023-09-28  415.15  4     175.6  -396.39   alpha_reversal
2023-10-02  414.39   2023-10-09  418.80  5     163.3  +720.36   alpha_reversal
2023-10-18  417.21   2023-10-20  408.05  2     141.6  -1296.11  stop_loss
2023-10-23  407.75   2023-10-27  397.87  4     135.3  -1337.26  trailing_stop
2023-10-30  403.03   2023-11-07  423.30  6     133.3  +2702.94  alpha_reversal
2023-11-13  426.88   2023-11-17  436.73  4     150.7  +1483.34  alpha_reversal
2023-11-22  441.27   2023-11-24  441.10  1     172.3  -29.25    alpha_reversal
2023-11-28  441.18   2023-12-11  447.58  9     196.8  +1259.35  alpha_reversal
2023-12-18  459.56   2023-12-20  455.49  2     197.7  -804.25   alpha_reversal
2023-12-21  460.27   2024-01-03  456.01  7     184.3  -785.80   trailing_stop
2024-01-04  454.99   2024-01-11  463.36  5     200.3  +1676.50  alpha_reversal
2024-01-31  470.18   2024-02-05  479.12  3     195.1  +1743.25  alpha_reversal
2024-02-13  481.09   2024-02-22  493.66  6     182.1  +2289.53  alpha_reversal
2024-02-27  493.60   2024-02-28  492.45  1     183.7  -210.37   alpha_reversal
2024-02-29  494.72   2024-03-14  500.91  10    190.6  +1179.52  max_holding
2024-03-15  497.97   2024-03-19  503.21  2     173.9  +911.38   alpha_reversal
2024-03-21  510.05   2024-04-04  500.63  9     173.8  -1636.78  stop_loss
2024-04-05  506.37   2024-04-09  506.73  2     170.3  +61.71    alpha_reversal
2024-04-16  491.81   2024-04-19  483.15  3     142.9  -1237.35  stop_loss
2024-04-22  488.09   2024-04-30  489.81  6     135.8  +233.18   alpha_reversal
2024-05-17  517.13   2024-05-21  518.48  2     168.2  +226.52   alpha_reversal
2024-06-06  522.22   2024-06-11  523.93  3     165.1  +282.68   alpha_reversal
2024-06-21  533.55   2024-06-26  534.00  3     178.8  +79.71    alpha_reversal
2024-07-01  534.37   2024-07-02  537.43  1     187.7  +574.19   alpha_reversal
2024-07-18  541.54   2024-07-24  529.81  4     161.4  -1893.51  stop_loss
2024-07-25  527.58   2024-07-30  530.56  3     133.8  +399.56   alpha_reversal
2024-08-02  522.18   2024-08-05  506.46  1     103.5  -1626.81  stop_loss
2024-08-06  511.64   2024-08-07  507.72  1     87.3   -343.03   alpha_reversal
2024-08-09  522.27   2024-08-13  530.60  2     84.4   +703.82   alpha_reversal
2024-08-16  543.16   2024-08-19  547.80  1     94.3   +437.93   alpha_reversal
2024-08-21  549.34   2024-08-22  544.48  1     101.5  -492.73   alpha_reversal
2024-08-27  550.26   2024-08-28  546.52  1     108.5  -405.73   alpha_reversal
2024-09-09  535.42   2024-09-11  542.72  2     102.1  +746.15   alpha_reversal
2024-09-19  559.49   2024-09-20  557.97  1     104.2  -158.88   alpha_reversal
2024-09-23  559.92   2024-10-01  558.33  6     113.2  -180.02   alpha_reversal
2024-10-02  559.12   2024-10-03  557.54  1     127.4  -201.35   alpha_reversal
2024-10-04  563.17   2024-10-09  566.70  3     130.9  +461.09   alpha_reversal
2024-10-22  573.34   2024-10-25  568.56  3     154.1  -735.92   alpha_reversal
2024-11-01  561.27   2024-11-07  584.83  4     139.4  +3286.01  alpha_reversal
2024-11-15  575.73   2024-11-19  579.62  2     135.9  +528.78   alpha_reversal
2024-11-29  592.24   2024-12-02  592.71  1     149.2  +69.90    alpha_reversal
2024-12-06  597.41   2024-12-09  593.74  1     179.8  -659.93   alpha_reversal
2024-12-18  576.25   2024-12-23  585.89  3     148.3  +1430.79  alpha_reversal
2024-12-26  593.04   2024-12-30  579.52  2     131.6  -1779.72  stop_loss
2025-01-13  573.36   2025-01-28  595.58  10    108.0  +2399.87  max_holding
2025-01-30  596.69   2025-02-05  595.28  4     117.3  -164.74   alpha_reversal
2025-02-18  603.05   2025-02-19  603.86  1     135.5  +110.59   alpha_reversal
2025-02-24  588.96   2025-02-27  576.40  3     131.1  -1648.10  stop_loss
2025-02-28  585.98   2025-03-03  575.14  1     110.0  -1192.08  alpha_reversal
2025-03-04  568.90   2025-03-06  564.24  2     95.0   -442.32   alpha_reversal
2025-03-07  567.97   2025-03-10  552.29  1     87.0   -1364.53  stop_loss
2025-03-11  548.24   2025-03-25  568.65  10    78.1   +1592.91  max_holding
2025-03-27  560.93   2025-03-31  552.77  2     89.8   -732.67   alpha_reversal
2025-04-04  499.80   2025-04-10  518.38  4     63.8   +1185.58  alpha_reversal
2025-04-11  528.15   2025-04-28  544.33  10    40.1   +648.17   max_holding
2025-05-06  552.74   2025-05-20  585.84  10    60.9   +2014.83  max_holding
2025-05-22  576.77   2025-06-06  592.05  10    84.1   +1285.24  max_holding
2025-06-09  593.18   2025-06-10  595.95  1     104.4  +288.94   alpha_reversal
2025-06-16  596.15   2025-06-24  601.38  5     112.5  +588.26   alpha_reversal
2025-07-09  619.12   2025-07-11  618.07  2     142.8  -150.66   alpha_reversal
2025-07-14  619.86   2025-07-15  616.60  1     153.2  -500.16   alpha_reversal
2025-07-29  630.23   2025-08-01  616.18  3     181.1  -2544.42  stop_loss
2025-08-04  626.17   2025-08-15  637.71  9     139.0  +1602.85  alpha_reversal
2025-08-19  634.75   2025-08-27  640.87  6     159.0  +973.94   alpha_reversal
2025-09-03  638.64   2025-09-11  651.77  6     153.9  +2020.48  alpha_reversal
2025-09-17  653.96   2025-09-19  659.61  2     168.9  +954.52   alpha_reversal
2025-09-24  657.69   2025-10-01  664.33  5     169.3  +1125.70  alpha_reversal
2025-10-06  668.14   2025-10-07  665.00  1     177.1  -556.57   alpha_reversal
2025-10-10  649.65   2025-10-20  667.17  6     142.6  +2497.68  alpha_reversal
2025-10-22  664.35   2025-10-29  683.16  5     118.7  +2232.19  alpha_reversal
2025-11-14  668.46   2025-11-18  656.01  2     111.7  -1390.60  stop_loss
2025-11-19  659.21   2025-11-20  648.51  1     105.2  -1125.24  alpha_reversal
2025-11-21  655.63   2025-11-24  664.61  1     89.2   +801.14   alpha_reversal
2025-12-02  678.01   2025-12-03  679.68  1     100.7  +167.94   alpha_reversal
2025-12-09  679.51   2025-12-17  667.26  6     120.1  -1470.67  trailing_stop
2025-12-18  672.98   2025-12-19  678.40  1     113.7  +616.19   alpha_reversal
2025-12-29  686.32   2026-01-02  680.97  3     136.6  -730.82   alpha_reversal
2026-01-09  692.53   2026-01-20  675.40  6     150.3  -2574.22  stop_loss
2026-01-21  683.88   2026-01-22  686.76  1     126.7  +365.47   alpha_reversal
2026-01-29  692.50   2026-02-05  675.44  5     132.1  -2252.81  stop_loss
2026-02-11  690.42   2026-02-23  680.19  7     107.6  -1101.01  alpha_reversal
2026-02-27  684.46   2026-03-06  670.21  5     102.4  -1459.64  stop_loss
2026-03-09  676.76   2026-03-10  675.00  1     84.5   -148.89   alpha_reversal
2026-03-12  664.58   2026-03-20  648.25  6     86.1   -1405.56  stop_loss
2026-03-23  655.71   2026-03-24  652.85  1     81.7   -233.32   alpha_reversal

**Best 3 trades:**

- 2020-01-17: P&L = **+3319.70** (max_holding)
- 2024-11-07: P&L = **+3286.01** (alpha_reversal)
- 2023-11-07: P&L = **+2702.94** (alpha_reversal)

**Worst 3 trades:**

- 2018-10-10: P&L = **-3399.33** (stop_loss)
- 2020-03-09: P&L = **-2954.10** (trailing_stop)
- 2019-08-05: P&L = **-2926.17** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  99,025.62
2017-03-23  99,385.17
2017-09-20  102,056.52
2018-03-21  109,235.56
2018-09-18  113,358.95
2019-03-20  107,132.34
2019-09-17  107,462.09
2020-03-17  112,854.41
2020-09-14  112,531.49
2021-03-15  116,052.66
2021-09-10  117,380.51
2022-03-10  113,010.59
2022-09-08  113,468.55
2023-03-09  112,295.41
2023-09-07  111,684.95
2024-03-07  120,210.12
2024-09-05  116,463.05
2025-03-07  118,308.48
2025-09-05  123,706.47
2026-03-06  121,375.53

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -4.42%
2017-03-23  -4.07%
2017-09-20  -1.49%
2018-03-21  -2.08%
2018-09-18  -0.41%
2019-03-20  -6.43%
2019-09-17  -6.14%
2020-03-17  -7.06%
2020-09-14  -7.33%
2021-03-15  -4.43%
2021-09-10  -3.33%
2022-03-10  -6.93%
2022-09-08  -6.55%
2023-03-09  -7.52%
2023-09-07  -8.02%
2024-03-07  -1.00%
2024-09-05  -4.09%
2025-03-07  -3.23%
2025-09-05  -0.27%
2026-03-06  -7.72%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  32.84%
Out-of-Sample (30%)  2023-03-24  2026-03-27  14.83%

#### Return Distribution

Return Bin          Count
-2.802% to -2.241%  4
-2.241% to -1.680%  8
-1.680% to -1.120%  17
-1.120% to -0.559%  101
-0.559% to 0.002%   772
0.002% to 0.563%    1442
0.563% to 1.124%    155
1.124% to 1.685%    15
1.685% to 2.246%    1
2.246% to 2.806%    1

### TSM — AlphaCombined

**Net Return (after slippage):** 29.62%  **vs SPY (exposure-adj): -37.30%** (underperform)  
**Gross Return (pre-cost):** 40.15%  
**Total Slippage Cost:** $10,526.96  
**Trade Count:** 333  
**Win Rate:** 52.9%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L       Exit Reason
2016-05-02  18.20    2016-05-11  17.83   7     1632.3  -618.64   alpha_reversal
2016-05-12  17.67    2016-05-17  17.62   3     1923.5  -93.02    alpha_reversal
2016-05-18  18.07    2016-05-19  17.85   1     1907.4  -429.80   alpha_reversal
2016-05-20  18.37    2016-05-23  18.63   1     1803.6  +465.34   alpha_reversal
2016-06-01  19.40    2016-06-02  19.33   1     1943.8  -142.14   alpha_reversal
2016-06-03  19.62    2016-06-06  19.55   1     2028.7  -133.22   alpha_reversal
2016-06-07  20.03    2016-06-08  20.21   1     2007.6  +375.94   alpha_reversal
2016-07-06  20.85    2016-07-08  21.29   2     1667.2  +735.46   alpha_reversal
2016-07-13  21.44    2016-07-15  21.51   2     1746.6  +129.52   alpha_reversal
2016-07-18  22.04    2016-07-21  21.89   3     1728.5  -258.35   alpha_reversal
2016-08-12  23.31    2016-08-15  23.41   1     1901.1  +182.82   alpha_reversal
2016-08-18  23.25    2016-08-19  22.81   1     2031.0  -888.38   alpha_reversal
2016-08-22  22.64    2016-08-30  22.80   6     1936.8  +295.54   alpha_reversal
2016-09-13  22.71    2016-09-20  23.81   5     1598.8  +1759.27  alpha_reversal
2016-09-27  24.24    2016-09-30  24.36   3     1502.0  +190.92   alpha_reversal
2016-10-05  24.34    2016-10-07  24.83   2     1678.5  +828.16   alpha_reversal
2016-10-12  24.99    2016-10-13  24.32   1     1629.1  -1104.70  stop_loss
2016-10-14  24.44    2016-10-17  24.35   1     1350.2  -129.77   alpha_reversal
2016-10-20  24.76    2016-10-24  25.02   2     1501.1  +381.30   alpha_reversal
2016-10-27  24.53    2016-11-01  24.46   3     1661.4  -120.14   alpha_reversal
2016-11-09  24.39    2016-11-10  23.45   1     1498.7  -1409.27  stop_loss
2016-11-11  23.86    2016-11-14  22.98   1     1261.3  -1115.06  stop_loss
2016-11-15  23.28    2016-11-16  23.78   1     1192.0  +598.87   alpha_reversal
2016-11-18  23.50    2016-11-23  23.81   3     1229.1  +382.31   alpha_reversal
2016-11-25  23.65    2016-12-01  22.74   4     1401.6  -1272.29  stop_loss
2016-12-02  23.01    2016-12-06  23.50   2     1336.2  +661.03   alpha_reversal
2016-12-13  24.21    2016-12-19  23.31   4     1377.3  -1229.01  stop_loss
2016-12-20  23.37    2016-12-29  23.20   6     1436.6  -239.51   alpha_reversal
2017-01-05  23.76    2017-01-06  23.60   1     1585.1  -252.27   alpha_reversal
2017-01-09  23.95    2017-01-10  23.95   1     1608.5  -0.08     alpha_reversal
2017-01-13  23.47    2017-01-17  23.59   1     1513.3  +181.46   alpha_reversal
2017-01-19  23.62    2017-01-24  24.55   3     1696.0  +1567.48  alpha_reversal
2017-02-03  24.69    2017-02-06  24.59   1     1815.9  -189.45   alpha_reversal
2017-02-07  24.33    2017-02-14  24.95   5     1859.3  +1154.31  alpha_reversal
2017-02-15  25.31    2017-02-16  25.74   1     1687.4  +723.39   alpha_reversal
2017-02-22  25.57    2017-02-27  25.11   3     1721.3  -784.31   alpha_reversal
2017-02-28  25.09    2017-03-14  24.86   10    1799.6  -417.79   alpha_reversal
2017-03-22  26.06    2017-03-23  25.93   1     1667.4  -216.08   alpha_reversal
2017-03-30  26.37    2017-03-31  26.16   1     1749.2  -366.52   alpha_reversal
2017-04-03  26.22    2017-04-04  26.17   1     1786.7  -89.53    alpha_reversal
2017-04-05  26.23    2017-04-10  25.90   3     1859.0  -611.41   alpha_reversal
2017-04-11  25.83    2017-04-19  25.18   5     1869.5  -1224.57  stop_loss
2017-04-20  25.35    2017-05-02  27.02   8     1860.2  +3123.60  alpha_reversal
2017-05-16  27.93    2017-05-25  28.33   7     1957.5  +787.30   alpha_reversal
2017-05-30  28.53    2017-05-31  28.16   1     2206.9  -801.20   alpha_reversal
2017-06-01  28.61    2017-06-06  29.02   3     2019.6  +810.87   alpha_reversal
2017-06-12  28.10    2017-06-20  28.78   6     1637.7  +1114.96  alpha_reversal
2017-07-05  28.67    2017-07-12  29.35   5     1529.8  +1050.76  alpha_reversal
2017-07-18  29.53    2017-07-27  29.76   7     1661.9  +374.66   alpha_reversal
2017-07-31  29.60    2017-08-01  29.48   1     1770.6  -227.15   alpha_reversal
2017-08-02  29.69    2017-08-07  30.26   3     1773.5  +997.53   alpha_reversal
2017-08-09  30.02    2017-08-10  29.06   1     1738.9  -1653.95  stop_loss
2017-08-15  29.98    2017-08-16  29.89   1     1573.2  -150.65   alpha_reversal
2017-08-17  29.46    2017-08-21  29.59   2     1608.4  +203.96   alpha_reversal
2017-08-29  30.36    2017-08-30  30.37   1     1800.8  +19.40    alpha_reversal
2017-09-06  30.68    2017-09-12  31.14   4     1821.8  +828.16   alpha_reversal
2017-09-14  30.77    2017-09-19  31.50   3     1968.1  +1428.58  alpha_reversal
2017-09-21  31.48    2017-09-22  31.06   1     1815.2  -758.77   alpha_reversal
2017-09-25  30.52    2017-10-02  31.36   5     1649.5  +1387.69  alpha_reversal
2017-10-05  31.66    2017-10-06  31.63   1     1760.5  -55.71    alpha_reversal
2017-10-26  33.92    2017-11-01  34.62   4     1665.6  +1176.40  alpha_reversal
2017-11-09  34.28    2017-11-14  34.43   3     1565.7  +229.65   alpha_reversal
2017-11-15  34.20    2017-11-17  34.60   2     1638.3  +658.14   alpha_reversal
2017-11-24  35.39    2017-11-27  33.79   1     1646.7  -2631.39  stop_loss
2017-11-28  33.70    2017-11-29  32.41   1     1386.4  -1779.84  stop_loss
2017-11-30  32.60    2017-12-08  32.08   6     1227.0  -645.44   alpha_reversal
2017-12-11  32.41    2017-12-15  32.49   4     1242.6  +92.61    alpha_reversal
2017-12-19  32.39    2017-12-26  32.03   4     1535.7  -542.30   alpha_reversal
2017-12-27  32.17    2017-12-28  32.68   1     1594.9  +814.44   alpha_reversal
2018-01-10  34.00    2018-01-17  35.47   4     1258.9  +1851.96  alpha_reversal
2018-01-26  37.64    2018-02-05  35.32   6     1025.1  -2373.92  stop_loss
2018-02-08  33.23    2018-02-13  34.85   3     778.9   +1268.19  alpha_reversal
2018-02-14  35.44    2018-02-15  35.79   1     759.3   +266.59   alpha_reversal
2018-02-20  36.01    2018-02-27  35.87   5     784.9   -105.71   alpha_reversal
2018-02-28  35.69    2018-03-02  35.37   2     847.2   -267.12   alpha_reversal
2018-03-29  36.03    2018-04-02  34.95   1     844.3   -905.31   alpha_reversal
2018-04-04  36.17    2018-04-13  35.08   7     749.8   -822.64   alpha_reversal
2018-04-16  35.29    2018-04-19  32.51   3     792.1   -2203.78  stop_loss
2018-04-20  32.07    2018-05-04  31.74   10    706.8   -231.91   alpha_reversal
2018-05-07  31.65    2018-05-10  32.71   3     880.7   +928.22   alpha_reversal
2018-05-16  32.76    2018-05-18  31.60   2     1008.6  -1169.48  stop_loss
2018-05-21  32.49    2018-05-23  32.61   2     944.8   +109.20   alpha_reversal
2018-06-08  32.05    2018-06-18  31.46   6     1118.9  -661.62   alpha_reversal
2018-06-19  30.92    2018-06-25  31.13   4     1156.3  +235.58   alpha_reversal
2018-07-03  31.05    2018-07-09  32.25   3     1164.4  +1393.31  alpha_reversal
2018-07-12  31.84    2018-07-13  32.20   1     1171.8  +422.25   alpha_reversal
2018-08-07  35.19    2018-08-15  34.48   6     1198.9  -849.63   trailing_stop
2018-08-17  34.25    2018-08-27  35.41   6     1222.4  +1417.08  alpha_reversal
2018-09-07  38.30    2018-09-10  38.04   1     990.7   -257.52   alpha_reversal
2018-09-12  37.50    2018-09-21  37.56   7     994.6   +64.46    alpha_reversal
2018-09-24  38.31    2018-09-27  38.16   3     994.6   -148.32   alpha_reversal
2018-10-03  37.51    2018-10-04  36.17   1     974.9   -1308.16  stop_loss
2018-10-09  34.90    2018-10-10  33.55   1     841.3   -1141.10  stop_loss
2018-10-11  33.23    2018-10-17  33.73   4     776.6   +391.29   alpha_reversal
2018-10-18  32.89    2018-10-24  31.10   4     720.9   -1289.74  stop_loss
2018-10-25  31.91    2018-10-31  32.48   4     689.3   +395.22   alpha_reversal
2018-11-06  33.05    2018-11-08  32.99   2     727.5   -42.63    alpha_reversal
2018-11-09  32.49    2018-11-14  32.26   3     785.6   -179.55   alpha_reversal
2018-11-19  30.76    2018-11-23  30.66   3     748.1   -80.40    alpha_reversal
2018-11-26  31.44    2018-11-27  31.37   1     787.7   -51.61    alpha_reversal
2018-11-28  32.44    2018-12-07  31.13   6     801.1   -1043.52  trailing_stop
2018-12-11  31.32    2018-12-12  31.87   1     750.5   +411.61   alpha_reversal
2018-12-14  31.17    2018-12-18  31.34   2     793.9   +130.92   alpha_reversal
2019-01-02  31.17    2019-01-03  29.29   1     785.1   -1470.17  stop_loss
2019-01-04  29.84    2019-01-07  30.03   1     718.1   +137.74   alpha_reversal
2019-01-09  30.39    2019-01-14  30.49   3     767.0   +74.78    alpha_reversal
2019-01-17  30.97    2019-01-18  31.41   1     800.6   +350.62   alpha_reversal
2019-02-04  32.38    2019-02-06  33.03   2     857.8   +557.26   alpha_reversal
2019-02-11  32.68    2019-02-12  32.94   1     956.1   +254.04   alpha_reversal
2019-02-14  32.48    2019-02-25  33.74   6     1033.1  +1305.15  alpha_reversal
2019-03-01  33.61    2019-03-04  33.38   1     1256.6  -299.31   alpha_reversal
2019-03-07  32.97    2019-03-18  33.97   7     1289.7  +1298.88  alpha_reversal
2019-03-20  34.01    2019-03-21  34.99   1     1317.9  +1292.19  alpha_reversal
2019-03-26  34.48    2019-04-04  35.84   7     1197.6  +1633.08  alpha_reversal
2019-04-10  36.16    2019-04-11  35.89   1     1463.2  -389.69   alpha_reversal
2019-04-12  36.34    2019-04-16  36.67   2     1479.9  +488.74   alpha_reversal
2019-04-23  38.76    2019-04-24  38.71   1     1234.5  -58.35    alpha_reversal
2019-04-26  37.50    2019-05-03  38.12   5     1131.8  +700.53   alpha_reversal
2019-05-13  34.98    2019-05-20  32.96   5     899.6   -1818.36  trailing_stop
2019-05-21  33.54    2019-05-28  32.10   4     817.1   -1176.72  stop_loss
2019-05-29  32.16    2019-05-30  32.43   1     870.4   +231.73   alpha_reversal
2019-06-06  32.80    2019-06-07  33.09   1     903.5   +263.08   alpha_reversal
2019-06-13  33.67    2019-06-14  32.26   1     919.3   -1292.70  stop_loss
2019-06-17  32.14    2019-06-24  35.19   5     889.4   +2716.10  alpha_reversal
2019-07-01  36.46    2019-07-02  35.91   1     803.1   -439.26   alpha_reversal
2019-07-03  35.54    2019-07-15  36.81   7     828.6   +1057.35  alpha_reversal
2019-07-29  38.72    2019-07-30  37.95   1     1012.0  -778.52   alpha_reversal
2019-07-31  37.56    2019-08-05  35.09   3     940.9   -2321.21  stop_loss
2019-08-12  36.09    2019-08-19  36.78   5     800.8   +549.09   alpha_reversal
2019-08-23  36.08    2019-08-28  36.66   3     809.8   +469.75   alpha_reversal
2019-09-13  39.42    2019-09-16  39.26   1     934.4   -151.97   alpha_reversal
2019-09-17  39.76    2019-09-18  39.65   1     966.9   -115.04   alpha_reversal
2019-09-19  39.85    2019-09-20  38.89   1     1008.2  -960.85   alpha_reversal
2019-09-23  39.11    2019-09-24  39.12   1     1003.0  +5.24     alpha_reversal
2019-09-30  41.25    2019-10-01  41.84   1     859.8   +505.75   alpha_reversal
2019-10-09  43.20    2019-10-10  43.38   1     841.6   +150.18   alpha_reversal
2019-10-17  44.15    2019-10-25  45.33   6     780.0   +919.94   alpha_reversal
2019-11-01  46.24    2019-11-04  47.42   1     971.0   +1151.69  alpha_reversal
2019-11-11  46.68    2019-11-19  47.66   6     932.8   +915.78   alpha_reversal
2019-11-25  47.71    2019-11-29  47.07   3     1005.6  -645.29   alpha_reversal
2019-12-02  47.02    2019-12-05  48.23   3     1015.0  +1230.18  alpha_reversal
2019-12-20  51.77    2019-12-24  51.57   2     803.4   -163.50   alpha_reversal
2019-12-26  52.06    2019-12-30  51.65   2     874.9   -357.93   alpha_reversal
2019-12-31  51.92    2020-01-02  53.60   1     901.2   +1514.13  alpha_reversal
2020-01-06  51.29    2020-01-17  52.30   9     775.3   +783.93   alpha_reversal
2020-01-21  52.05    2020-01-27  49.33   4     694.8   -1884.69  stop_loss
2020-01-31  48.20    2020-02-05  51.71   3     569.7   +1996.81  alpha_reversal
2020-02-19  51.51    2020-02-24  48.32   3     533.8   -1705.06  stop_loss
2020-03-12  43.51    2020-03-16  40.09   2     359.5   -1232.13  trailing_stop
2020-03-17  43.38    2020-03-18  39.61   1     283.0   -1065.85  stop_loss
2020-03-20  40.17    2020-03-23  40.79   1     254.8   +157.39   alpha_reversal
2020-04-01  41.96    2020-04-03  42.41   2     295.9   +134.27   alpha_reversal
2020-04-06  45.08    2020-04-07  44.81   1     307.6   -83.16    alpha_reversal
2020-04-08  45.35    2020-04-17  48.39   6     327.6   +994.80   alpha_reversal
2020-04-24  47.57    2020-04-28  47.55   2     382.4   -7.84     alpha_reversal
2020-04-30  47.93    2020-05-08  47.68   6     413.6   -101.81   alpha_reversal
2020-05-20  46.77    2020-06-03  47.82   9     458.3   +482.51   alpha_reversal
2020-06-08  50.13    2020-06-09  51.02   1     527.4   +467.91   alpha_reversal
2020-06-17  51.06    2020-06-18  50.98   1     505.2   -39.55    alpha_reversal
2020-06-19  50.27    2020-06-23  51.54   2     529.5   +670.56   alpha_reversal
2020-08-06  73.18    2020-08-20  69.74   10    253.0   -870.82   max_holding
2020-08-31  72.03    2020-09-02  74.61   2     361.7   +932.97   alpha_reversal
2020-09-04  71.72    2020-09-16  75.48   7     317.6   +1194.34  alpha_reversal
2020-09-17  74.83    2020-09-18  73.22   1     294.0   -472.70   alpha_reversal
2020-09-23  71.19    2020-09-28  72.80   3     308.6   +499.03   alpha_reversal
2020-10-05  77.23    2020-10-07  79.33   2     330.6   +692.50   alpha_reversal
2020-10-28  76.60    2020-10-29  77.48   1     377.6   +329.49   alpha_reversal
2020-11-02  78.30    2020-11-03  80.03   1     389.0   +672.49   alpha_reversal
2020-11-10  80.08    2020-11-16  90.60   4     363.1   +3817.98  alpha_reversal
2020-11-20  87.09    2020-12-02  90.85   7     295.6   +1109.91  alpha_reversal
2020-12-15  96.01    2020-12-17  95.23   2     290.7   -227.34   alpha_reversal
2020-12-18  95.67    2020-12-22  95.26   2     317.2   -129.19   alpha_reversal
2020-12-23  95.19    2020-12-28  97.29   2     338.8   +709.93   alpha_reversal
2020-12-30  99.91    2020-12-31  99.94   1     328.2   +9.35     alpha_reversal
2021-01-21  123.08   2021-01-22  118.36  1     162.0   -765.32   alpha_reversal
2021-01-27  111.69   2021-02-02  117.72  4     162.2   +977.99   alpha_reversal
2021-02-04  117.57   2021-02-05  117.11  1     175.5   -80.14    alpha_reversal
2021-02-08  121.61   2021-02-10  122.53  2     179.5   +165.76   alpha_reversal
2021-02-23  118.44   2021-03-03  112.63  6     187.9   -1091.48  trailing_stop
2021-03-04  106.05   2021-03-10  104.33  4     167.5   -287.95   alpha_reversal
2021-03-11  110.62   2021-03-16  109.39  3     157.0   -194.36   alpha_reversal
2021-03-18  105.44   2021-03-23  105.69  3     172.9   +43.80    alpha_reversal
2021-03-24  100.34   2021-03-31  108.81  5     181.0   +1533.96  alpha_reversal
2021-04-21  108.11   2021-04-27  111.56  4     220.9   +762.73   alpha_reversal
2021-04-28  110.19   2021-05-10  103.32  8     242.7   -1668.09  stop_loss
2021-05-11  103.92   2021-05-19  103.40  6     251.7   -130.33   alpha_reversal
2021-05-24  104.64   2021-05-25  105.13  1     245.1   +120.92   alpha_reversal
2021-05-28  108.08   2021-06-01  109.03  1     278.9   +264.98   alpha_reversal
2021-06-04  110.06   2021-06-07  108.76  1     291.2   -377.65   alpha_reversal
2021-06-08  106.75   2021-06-18  106.93  8     293.4   +53.49    alpha_reversal
2021-06-21  105.51   2021-06-23  107.17  2     302.3   +501.37   alpha_reversal
2021-06-25  107.72   2021-06-29  111.03  2     321.1   +1062.61  alpha_reversal
2021-07-08  108.96   2021-07-13  114.42  3     335.7   +1832.97  alpha_reversal
2021-07-21  108.96   2021-07-27  104.85  4     293.6   -1205.87  stop_loss
2021-07-28  106.37   2021-08-02  107.92  3     306.3   +473.80   alpha_reversal
2021-08-09  109.28   2021-08-10  107.46  1     373.5   -682.41   alpha_reversal
2021-08-11  107.22   2021-08-17  102.61  4     344.2   -1587.85  stop_loss
2021-08-18  102.10   2021-08-20  99.85   2     347.6   -783.30   alpha_reversal
2021-08-23  102.57   2021-08-26  109.06  3     323.9   +2102.41  alpha_reversal
2021-09-16  112.12   2021-09-20  106.36  2     369.9   -2130.09  stop_loss
2021-09-22  107.55   2021-09-28  103.76  4     354.4   -1342.85  stop_loss
2021-09-29  103.61   2021-10-12  101.26  9     358.9   -842.50   alpha_reversal
2021-10-25  105.48   2021-11-02  105.81  6     371.1   +122.62   alpha_reversal
2021-11-03  105.83   2021-11-05  109.23  2     405.8   +1383.37  alpha_reversal
2021-11-16  109.60   2021-11-18  114.42  2     326.9   +1573.95  alpha_reversal
2021-12-07  113.54   2021-12-13  107.96  4     241.2   -1346.94  stop_loss
2021-12-14  107.94   2021-12-29  113.05  10    262.4   +1339.70  max_holding
2021-12-31  112.14   2022-01-04  124.21  2     299.0   +3611.35  alpha_reversal
2022-01-25  114.16   2022-02-07  113.89  9     172.4   -46.95    alpha_reversal
2022-02-15  116.06   2022-02-22  108.86  4     188.4   -1356.79  stop_loss
2022-02-25  103.67   2022-03-02  102.06  3     181.0   -291.75   alpha_reversal
2022-03-03  101.40   2022-03-07  92.45   2     190.7   -1705.91  stop_loss
2022-03-08  93.21    2022-03-09  97.72   1     180.6   +815.59   alpha_reversal
2022-03-10  96.90    2022-03-18  99.85   6     182.7   +539.42   alpha_reversal
2022-03-28  99.58    2022-04-05  95.90   6     241.7   -890.11   trailing_stop
2022-04-07  94.14    2022-04-19  93.05   7     259.8   -284.58   alpha_reversal
2022-04-21  91.46    2022-04-26  86.80   3     262.8   -1226.19  stop_loss
2022-04-28  89.18    2022-05-03  87.60   3     252.0   -397.37   alpha_reversal
2022-05-04  89.90    2022-05-05  86.22   1     256.1   -943.12   alpha_reversal
2022-05-09  81.76    2022-05-18  84.70   7     239.8   +705.17   alpha_reversal
2022-05-20  85.02    2022-05-24  83.01   2     250.5   -504.03   alpha_reversal
2022-05-25  84.67    2022-05-31  89.16   3     265.8   +1193.71  alpha_reversal
2022-06-02  90.23    2022-06-03  87.73   1     279.1   -696.36   alpha_reversal
2022-06-09  85.09    2022-06-13  80.04   2     288.0   -1452.71  stop_loss
2022-06-14  81.66    2022-06-15  83.39   1     284.3   +492.75   alpha_reversal
2022-06-23  79.20    2022-07-01  72.42   6     288.2   -1952.54  trailing_stop
2022-07-05  71.66    2022-07-12  74.40   5     271.7   +744.63   alpha_reversal
2022-07-25  81.25    2022-08-04  83.16   8     304.1   +581.67   alpha_reversal
2022-08-15  86.21    2022-08-16  84.60   1     298.0   -479.82   alpha_reversal
2022-08-22  80.25    2022-08-24  80.47   2     317.9   +70.18    alpha_reversal
2022-08-25  82.45    2022-08-26  79.70   1     343.2   -941.71   alpha_reversal
2022-08-30  77.95    2022-09-01  76.79   2     334.8   -388.17   alpha_reversal
2022-09-02  76.17    2022-09-19  73.99   10    325.5   -706.68   max_holding
2022-09-26  69.13    2022-09-28  68.23   2     392.0   -353.40   alpha_reversal
2022-09-29  65.59    2022-10-05  70.45   4     364.7   +1769.81  alpha_reversal
2022-10-19  60.27    2022-11-02  57.24   10    280.8   -850.81   max_holding
2022-11-25  77.07    2022-12-06  75.25   7     280.5   -509.75   alpha_reversal
2022-12-13  76.23    2022-12-16  72.58   3     329.9   -1203.83  stop_loss
2022-12-21  73.45    2022-12-22  71.61   1     357.3   -658.37   alpha_reversal
2022-12-27  70.77    2023-01-10  77.31   9     371.2   +2427.90  alpha_reversal
2023-01-27  88.84    2023-02-10  90.72   10    272.2   +511.89   alpha_reversal
2023-02-13  91.51    2023-02-14  93.18   1     275.6   +459.77   alpha_reversal
2023-02-16  86.48    2023-03-02  84.69   9     244.0   -436.55   alpha_reversal
2023-03-07  84.60    2023-03-09  84.54   2     290.2   -16.26    alpha_reversal
2023-03-27  86.77    2023-03-29  88.69   2     295.6   +567.99   alpha_reversal
2023-03-30  88.51    2023-04-03  88.77   2     320.1   +84.93    alpha_reversal
2023-04-04  88.22    2023-04-10  85.12   3     334.9   -1038.35  alpha_reversal
2023-04-11  85.41    2023-04-21  81.63   8     325.9   -1233.68  stop_loss
2023-04-26  78.72    2023-05-05  81.25   7     327.4   +825.68   alpha_reversal
2023-06-05  93.85    2023-06-13  102.12  6     255.8   +2116.46  alpha_reversal
2023-06-16  100.51   2023-06-26  96.12   5     258.1   -1131.41  stop_loss
2023-06-27  98.11    2023-07-05  96.97   5     273.6   -313.23   alpha_reversal
2023-07-06  95.55    2023-07-10  95.80   2     299.3   +74.86    alpha_reversal
2023-07-11  97.33    2023-07-20  93.96   7     304.9   -1025.11  trailing_stop
2023-07-27  95.49    2023-07-31  95.20   2     287.6   -82.67    alpha_reversal
2023-08-01  94.58    2023-08-08  90.73   5     308.0   -1185.34  stop_loss
2023-08-09  90.33    2023-08-17  87.99   6     311.8   -728.74   alpha_reversal
2023-08-25  89.48    2023-08-28  90.26   1     313.1   +242.60   alpha_reversal
2023-08-31  89.93    2023-09-07  86.46   4     345.5   -1198.68  stop_loss
2023-09-08  86.16    2023-09-20  84.25   8     323.9   -617.87   trailing_stop
2023-09-21  82.47    2023-09-27  82.46   4     358.7   -1.87     alpha_reversal
2023-10-10  87.54    2023-10-11  88.80   1     382.8   +480.06   alpha_reversal
2023-10-30  83.15    2023-11-07  89.19   6     305.1   +1844.55  alpha_reversal
2023-11-08  88.88    2023-11-13  93.06   3     346.7   +1448.02  alpha_reversal
2023-11-27  93.92    2023-12-11  97.43   10    370.3   +1301.84  max_holding
2023-12-14  100.42   2023-12-19  100.72  3     394.8   +121.17   alpha_reversal
2023-12-20  97.11    2023-12-26  101.29  3     379.5   +1586.19  alpha_reversal
2024-01-02  98.55    2024-01-09  98.80   5     398.8   +99.92    alpha_reversal
2024-01-30  112.66   2024-02-06  115.76  5     283.7   +881.45   alpha_reversal
2024-02-16  122.98   2024-02-27  124.70  6     223.3   +383.98   alpha_reversal
2024-02-28  123.65   2024-03-04  134.07  3     229.9   +2397.17  alpha_reversal
2024-03-11  134.94   2024-03-13  137.83  2     142.4   +411.60   alpha_reversal
2024-03-14  135.53   2024-03-25  136.54  7     146.3   +147.56   alpha_reversal
2024-03-26  135.32   2024-04-02  136.53  4     173.8   +210.00   alpha_reversal
2024-04-03  138.40   2024-04-04  135.98  1     182.4   -440.85   alpha_reversal
2024-04-24  129.60   2024-04-26  134.66  2     163.9   +829.29   alpha_reversal
2024-05-08  139.96   2024-05-09  139.03  1     180.6   -167.69   alpha_reversal
2024-05-10  145.47   2024-05-13  142.55  1     175.7   -513.14   alpha_reversal
2024-05-14  148.09   2024-05-15  151.48  1     175.2   +593.43   alpha_reversal
2024-05-20  149.65   2024-05-21  149.62  1     186.7   -6.11     alpha_reversal
2024-06-03  151.02   2024-06-06  157.80  3     185.2   +1255.99  alpha_reversal
2024-06-21  170.08   2024-07-01  168.32  6     143.1   -252.22   alpha_reversal
2024-07-02  171.78   2024-07-03  178.24  1     148.4   +958.77   alpha_reversal
2024-07-12  183.17   2024-07-17  167.21  3     130.9   -2088.26  stop_loss
2024-07-30  151.09   2024-08-02  146.37  3     110.1   -519.80   trailing_stop
2024-08-15  170.08   2024-08-19  171.37  2     102.5   +131.73   alpha_reversal
2024-08-21  167.62   2024-08-28  165.26  5     116.9   -275.24   alpha_reversal
2024-08-29  165.38   2024-09-06  153.17  5     119.4   -1457.27  stop_loss
2024-09-18  164.15   2024-09-24  178.39  4     130.4   +1856.78  alpha_reversal
2024-09-30  170.42   2024-10-07  180.88  5     130.7   +1366.37  alpha_reversal
2024-10-11  187.24   2024-10-17  201.79  4     146.3   +2127.71  alpha_reversal
2024-10-25  199.64   2024-10-28  190.85  1     127.8   -1123.24  alpha_reversal
2024-10-29  193.26   2024-11-12  188.00  10    124.7   -655.86   max_holding
2024-11-13  183.17   2024-11-19  185.94  4     121.4   +336.11   alpha_reversal
2024-11-21  187.67   2024-11-22  186.34  1     135.1   -179.04   alpha_reversal
2024-11-25  181.62   2024-12-03  194.98  5     137.6   +1838.56  alpha_reversal
2024-12-11  190.96   2024-12-16  198.87  3     139.6   +1104.17  alpha_reversal
2024-12-26  199.90   2024-12-30  197.07  2     131.9   -372.78   alpha_reversal
2025-01-02  198.44   2025-01-07  207.92  3     137.0   +1298.74  alpha_reversal
2025-01-08  203.89   2025-01-10  204.92  1     121.8   +124.88   alpha_reversal
2025-01-28  199.25   2025-02-04  200.67  5     88.5    +126.00   alpha_reversal
2025-02-13  198.69   2025-02-25  186.23  7     105.2   -1309.65  stop_loss
2025-02-28  177.72   2025-03-05  181.23  3     109.7   +384.98   alpha_reversal
2025-03-06  173.11   2025-03-10  167.82  2     105.9   -559.81   alpha_reversal
2025-03-19  171.72   2025-03-21  174.48  2     123.1   +339.75   alpha_reversal
2025-03-26  171.46   2025-04-02  168.22  5     129.7   -420.33   alpha_reversal
2025-04-04  145.08   2025-04-10  149.21  4     117.1   +483.42   alpha_reversal
2025-04-29  162.55   2025-04-30  164.57  1     112.8   +227.83   alpha_reversal
2025-05-20  191.18   2025-06-03  195.09  9     139.3   +545.65   alpha_reversal
2025-06-18  211.76   2025-06-24  218.08  3     169.4   +1070.40  alpha_reversal
2025-07-01  222.85   2025-07-07  227.08  3     161.4   +682.12   alpha_reversal
2025-07-08  226.01   2025-07-10  227.67  2     155.6   +257.73   alpha_reversal
2025-07-11  228.53   2025-07-14  226.58  1     160.8   -312.44   alpha_reversal
2025-07-23  238.38   2025-07-31  239.42  6     143.3   +149.04   alpha_reversal
2025-08-07  240.65   2025-08-08  239.63  1     127.4   -130.33   alpha_reversal
2025-08-11  240.12   2025-08-12  242.06  1     134.3   +260.44   alpha_reversal
2025-08-14  239.04   2025-08-19  230.58  3     140.7   -1190.77  alpha_reversal
2025-08-20  226.74   2025-08-26  236.54  4     134.9   +1322.44  alpha_reversal
2025-08-29  228.99   2025-09-08  244.94  5     143.3   +2284.50  alpha_reversal
2025-09-22  271.27   2025-09-23  281.02  1     125.6   +1224.70  alpha_reversal
2025-09-30  277.89   2025-10-01  286.74  1     118.8   +1050.72  alpha_reversal
2025-10-03  290.73   2025-10-06  300.59  1     110.2   +1086.40  alpha_reversal
2025-10-21  293.04   2025-10-30  301.40  7     83.5    +698.70   alpha_reversal
2025-11-06  287.79   2025-11-17  280.32  7     96.4    -720.47   alpha_reversal
2025-11-18  276.52   2025-11-20  275.84  2     94.2    -64.45    alpha_reversal
2025-11-24  283.22   2025-12-01  285.96  4     86.7    +237.47   alpha_reversal
2025-12-02  290.63   2025-12-09  301.59  5     93.2    +1022.10  alpha_reversal
2025-12-12  291.37   2025-12-17  276.04  3     93.2    -1428.23  stop_loss
2025-12-18  284.02   2025-12-23  295.97  3     91.5    +1092.69  alpha_reversal
2026-01-22  326.61   2026-01-29  338.43  5     88.1    +1040.60  alpha_reversal
2026-02-18  361.42   2026-02-23  368.82  3     60.4    +446.77   alpha_reversal
2026-03-04  356.61   2026-03-10  345.94  4     65.7    -700.87   alpha_reversal
2026-03-13  337.53   2026-03-16  339.10  1     64.9    +102.32   alpha_reversal

**Best 3 trades:**

- 2020-11-16: P&L = **+3817.98** (alpha_reversal)
- 2022-01-04: P&L = **+3611.35** (alpha_reversal)
- 2017-05-02: P&L = **+3123.60** (alpha_reversal)

**Worst 3 trades:**

- 2017-11-27: P&L = **-2631.39** (stop_loss)
- 2018-02-05: P&L = **-2373.92** (stop_loss)
- 2019-08-05: P&L = **-2321.21** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  101,380.32
2017-03-23  99,591.45
2017-09-20  105,184.63
2018-03-21  103,770.12
2018-09-18  100,832.84
2019-03-20  99,468.91
2019-09-17  100,446.96
2020-03-17  102,569.18
2020-09-14  104,694.10
2021-03-15  111,146.69
2021-09-10  113,494.51
2022-03-10  113,268.45
2022-09-08  107,233.59
2023-03-09  107,894.12
2023-09-07  103,872.11
2024-03-07  113,781.78
2024-09-05  113,411.25
2025-03-07  119,644.37
2025-09-05  124,013.50
2026-03-06  128,993.44

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -0.06%
2017-03-23  -2.76%
2017-09-20  -0.03%
2018-03-21  -3.82%
2018-09-18  -6.54%
2019-03-20  -7.81%
2019-09-17  -6.90%
2020-03-17  -4.93%
2020-09-14  -2.96%
2021-03-15  -1.89%
2021-09-10  -1.48%
2022-03-10  -2.46%
2022-09-08  -7.65%
2023-03-09  -7.09%
2023-09-07  -10.55%
2024-03-07  -2.01%
2024-09-05  -2.92%
2025-03-07  -1.42%
2025-09-05  0.00%
2026-03-06  -0.97%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  25.87%
Out-of-Sample (30%)  2023-03-24  2026-03-27  28.50%

#### Return Distribution

Return Bin          Count
-2.388% to -1.921%  2
-1.921% to -1.454%  7
-1.454% to -0.987%  20
-0.987% to -0.519%  110
-0.519% to -0.052%  334
-0.052% to 0.415%   1816
0.415% to 0.882%    177
0.882% to 1.349%    40
1.349% to 1.817%    6
1.817% to 2.284%    4

### AlphaCombined_Portfolio — AlphaCombined

**Net Return (after slippage):** 670.40%  **vs SPY B&H: +403.18%**  
**Gross Return (pre-cost):** 1129.23%  
**Total Slippage Cost:** $454,799.22  
**Trade Count:** 4139  
**Win Rate:** 53.8%  

#### Trade Log

Entry Date  Entry $  Exit Date   Exit $  Days  Size    P&L        Exit Reason
2016-05-09  86.42    2016-05-10  87.05   1     790.6   +501.88    alpha_reversal
2016-05-10  122.00   2016-05-11  121.30  1     294.6   -205.22    alpha_reversal
2016-05-09  37.97    2016-05-11  38.08   2     1129.3  +123.41    alpha_reversal
2016-05-09  17.96    2016-05-11  17.83   2     1846.6  -245.79    alpha_reversal
2016-05-11  18.69    2016-05-13  18.26   2     2146.7  -927.52    alpha_reversal
2016-05-09  13.93    2016-05-16  13.88   5     974.3   -54.47     alpha_reversal
2016-05-10  47.82    2016-05-17  47.48   5     766.5   -260.90    alpha_reversal
2016-05-12  37.98    2016-05-17  38.08   3     1194.5  +122.12    alpha_reversal
2016-05-10  57.92    2016-05-17  56.69   5     457.5   -559.54    alpha_reversal
2016-05-12  17.67    2016-05-17  17.62   3     1938.9  -93.76     alpha_reversal
2016-05-11  21.09    2016-05-18  21.54   5     1445.3  +644.33    alpha_reversal
2016-05-09  128.31   2016-05-18  130.16  7     238.0   +440.23    max_holding
2016-05-16  18.58    2016-05-18  17.76   2     1968.6  -1625.36   stop_loss
2016-05-18  38.37    2016-05-19  38.27   1     1175.6  -119.25    alpha_reversal
2016-05-18  174.18   2016-05-19  173.40  1     365.1   -283.69    alpha_reversal
2016-05-18  18.07    2016-05-19  17.85   1     1880.0  -423.62    alpha_reversal
2016-05-19  19.48    2016-05-20  19.65   1     1411.5  +234.47    alpha_reversal
2016-05-20  125.87   2016-05-24  128.25  2     217.5   +517.90    alpha_reversal
2016-05-19  55.46    2016-05-24  56.73   3     460.8   +584.84    alpha_reversal
2016-05-24  35.23    2016-05-25  35.40   1     861.3   +148.31    alpha_reversal
2016-05-24  39.02    2016-05-25  39.66   1     1150.6  +737.63    alpha_reversal
2016-05-18  86.32    2016-05-26  86.32   6     788.6   +1.47      alpha_reversal
2016-05-20  116.30   2016-05-27  117.86  5     275.1   +427.24    alpha_reversal
2016-05-20  174.67   2016-05-27  178.53  5     365.1   +1408.98   alpha_reversal
2016-05-27  86.53    2016-05-31  86.16   1     838.1   -309.58    alpha_reversal
2016-05-31  36.16    2016-06-01  35.95   1     1010.6  -205.72    alpha_reversal
2016-05-24  36.37    2016-06-01  37.10   5     1031.1  +751.13    alpha_reversal
2016-05-31  14.89    2016-06-01  14.63   1     1370.9  -355.66    alpha_reversal
2016-05-27  57.48    2016-06-01  57.67   2     526.2   +99.93     alpha_reversal
2016-06-01  86.32    2016-06-02  87.54   1     840.0   +1025.74   alpha_reversal
2016-05-27  39.64    2016-06-02  39.84   3     1212.5  +240.99    alpha_reversal
2016-06-01  178.73   2016-06-02  179.10  1     429.7   +156.79    alpha_reversal
2016-06-01  19.40    2016-06-02  19.33   1     1978.1  -144.65    alpha_reversal
2016-06-02  22.28    2016-06-06  22.46   2     1579.0  +292.09    alpha_reversal
2016-05-31  115.17   2016-06-06  120.30  4     297.5   +1525.74   alpha_reversal
2016-06-02  58.80    2016-06-06  60.98   2     561.5   +1221.60   alpha_reversal
2016-06-03  19.62    2016-06-06  19.55   1     2093.6  -137.48    alpha_reversal
2016-05-27  50.43    2016-06-08  50.24   7     775.5   -146.57    max_holding
2016-06-02  14.60    2016-06-08  15.69   4     1467.1  +1597.46   alpha_reversal
2016-06-07  20.03    2016-06-08  20.21   1     2133.4  +399.49    alpha_reversal
2016-06-08  36.35    2016-06-09  36.36   1     1186.1  +16.77     alpha_reversal
2016-05-31  19.92    2016-06-09  19.99   7     1868.9  +125.70    max_holding
2016-06-09  22.72    2016-06-10  22.51   1     1713.6  -358.93    alpha_reversal
2016-06-03  127.33   2016-06-10  122.48  5     239.1   -1159.92   trailing_stop
2016-06-08  46.13    2016-06-13  44.40   3     916.4   -1584.16   stop_loss
2016-06-02  36.93    2016-06-13  36.28   7     1200.5  -781.53    max_holding
2016-06-10  20.03    2016-06-13  19.83   1     2514.6  -481.67    alpha_reversal
2016-06-10  49.20    2016-06-14  47.80   2     839.9   -1179.43   stop_loss
2016-06-06  40.12    2016-06-15  39.65   7     1345.4  -640.77    alpha_reversal
2016-06-16  35.93    2016-06-17  34.91   1     1265.8  -1300.22   stop_loss
2016-06-13  177.19   2016-06-17  176.28  4     460.6   -416.37    alpha_reversal
2016-06-16  118.11   2016-06-20  121.08  2     316.7   +939.03    alpha_reversal
2016-06-15  47.76    2016-06-22  48.28   5     769.7   +401.80    alpha_reversal
2016-06-20  35.72    2016-06-22  35.51   2     1109.5  -228.69    alpha_reversal
2016-06-13  121.57   2016-06-22  121.05  7     264.2   -137.87    alpha_reversal
2016-06-15  14.52    2016-06-22  13.10   5     1317.4  -1866.04   stop_loss
2016-06-14  44.17    2016-06-23  45.97   7     825.8   +1484.64   alpha_reversal
2016-06-22  120.30   2016-06-23  121.81  1     320.3   +481.41    alpha_reversal
2016-06-20  89.19    2016-06-23  89.75   3     731.0   +404.32    alpha_reversal
2016-06-20  39.72    2016-06-23  40.77   3     1134.9  +1190.28   alpha_reversal
2016-06-23  36.12    2016-06-24  34.93   1     1169.7  -1393.68   stop_loss
2016-06-20  35.03    2016-06-24  33.96   4     1160.3  -1244.32   trailing_stop
2016-06-23  62.48    2016-06-24  58.27   1     574.7   -2416.08   stop_loss
2016-06-20  21.68    2016-06-27  20.96   5     1637.3  -1176.51   stop_loss
2016-06-24  88.50    2016-06-28  90.40   2     699.1   +1327.92   alpha_reversal
2016-06-27  44.40    2016-06-29  47.12   2     555.4   +1510.50   alpha_reversal
2016-06-28  173.62   2016-07-01  179.18  3     293.1   +1630.28   alpha_reversal
2016-06-30  21.79    2016-07-05  21.63   2     1663.7  -267.36    alpha_reversal
2016-06-23  13.10    2016-07-05  14.26   7     1238.7  +1434.87   max_holding
2016-06-30  40.76    2016-07-05  41.00   2     1024.9  +240.78    alpha_reversal
2016-06-29  44.80    2016-07-06  45.50   4     680.1   +475.48    alpha_reversal
2016-07-05  20.59    2016-07-06  20.76   1     2497.6  +426.19    alpha_reversal
2016-06-27  112.02   2016-07-07  115.99  7     251.0   +995.33    max_holding
2016-06-27  33.79    2016-07-07  35.06   7     1002.0  +1263.39   max_holding
2016-06-27  57.01    2016-07-07  59.89   7     443.5   +1273.52   alpha_reversal
2016-07-06  21.78    2016-07-08  22.02   2     1815.6  +436.02    alpha_reversal
2016-06-28  116.49   2016-07-08  122.88  7     207.4   +1326.52   alpha_reversal
2016-07-06  20.85    2016-07-08  21.29   2     1719.8  +758.66    alpha_reversal
2016-07-08  94.02    2016-07-11  93.99   1     721.3   -23.66     alpha_reversal
2016-07-08  181.70   2016-07-12  183.48  2     324.6   +578.32    alpha_reversal
2016-07-05  46.25    2016-07-14  49.75   7     571.4   +1999.69   max_holding
2016-07-13  94.14    2016-07-14  94.18   1     776.6   +33.81     alpha_reversal
2016-07-11  20.85    2016-07-14  20.73   3     2814.2  -343.54    alpha_reversal
2016-07-08  35.61    2016-07-15  36.46   5     1216.0  +1032.55   alpha_reversal
2016-07-13  21.44    2016-07-15  21.51   2     1903.7  +141.17    alpha_reversal
2016-07-13  118.79   2016-07-18  121.70  3     294.9   +858.03    alpha_reversal
2016-07-12  63.74    2016-07-18  64.31   4     499.9   +282.86    alpha_reversal
2016-07-14  184.66   2016-07-19  184.54  3     372.3   -46.48     alpha_reversal
2016-07-14  22.52    2016-07-20  22.77   4     2188.2  +533.81    alpha_reversal
2016-07-20  49.56    2016-07-21  49.42   1     784.5   -115.30    alpha_reversal
2016-07-12  42.17    2016-07-21  41.55   7     1220.4  -750.15    max_holding
2016-07-18  22.04    2016-07-21  21.89   3     1950.6  -291.54    alpha_reversal
2016-07-13  37.15    2016-07-22  37.22   7     1155.3  +85.85     max_holding
2016-07-21  95.78    2016-07-22  95.60   1     794.7   -149.00    alpha_reversal
2016-07-13  14.84    2016-07-22  14.81   7     1465.8  -47.14     max_holding
2016-07-21  64.52    2016-07-22  63.95   1     616.3   -352.53    alpha_reversal
2016-07-22  131.21   2016-07-25  131.16  1     283.6   -14.02     alpha_reversal
2016-07-18  20.79    2016-07-25  20.74   5     3177.0  -146.41    alpha_reversal
2016-07-25  121.37   2016-07-27  124.00  2     350.2   +922.19    alpha_reversal
2016-07-19  49.60    2016-07-27  49.92   6     752.4   +237.10    alpha_reversal
2016-07-25  63.46    2016-07-27  67.75   2     635.6   +2730.05   alpha_reversal
2016-07-25  22.19    2016-07-28  23.76   3     2369.0  +3724.09   alpha_reversal
2016-07-27  36.85    2016-07-29  37.92   2     1367.1  +1462.29   alpha_reversal
2016-07-27  15.24    2016-07-29  15.64   2     1655.7  +669.82    alpha_reversal
2016-07-20  185.49   2016-07-29  185.33  7     439.5   -70.22     max_holding
2016-07-29  122.03   2016-08-01  121.50  1     349.9   -186.28    alpha_reversal
2016-07-28  131.31   2016-08-02  127.52  3     335.6   -1269.69   stop_loss
2016-07-25  95.59    2016-08-02  95.46   6     819.9   -103.41    alpha_reversal
2016-08-03  120.39   2016-08-04  119.67  1     383.1   -276.69    alpha_reversal
2016-08-04  15.38    2016-08-05  15.33   1     1763.8  -95.28     alpha_reversal
2016-07-27  41.46    2016-08-05  45.14   7     1419.4  +5218.21   alpha_reversal
2016-07-27  20.64    2016-08-05  20.74   7     3704.6  +381.96    alpha_reversal
2016-07-29  49.69    2016-08-08  51.29   6     1068.1  +1712.22   alpha_reversal
2016-08-05  120.27   2016-08-09  120.96  2     401.1   +277.34    alpha_reversal
2016-08-05  132.58   2016-08-09  133.55  2     348.1   +337.82    alpha_reversal
2016-08-05  66.58    2016-08-09  66.73   2     691.0   +104.32    alpha_reversal
2016-08-08  15.08    2016-08-10  15.04   2     1971.3  -96.71     alpha_reversal
2016-08-10  66.65    2016-08-12  66.87   2     751.1   +161.74    alpha_reversal
2016-08-12  24.80    2016-08-15  25.07   1     2518.2  +687.18    alpha_reversal
2016-08-04  94.80    2016-08-15  93.52   7     968.9   -1240.01   trailing_stop
2016-08-11  15.00    2016-08-15  15.03   2     2080.1  +63.06     alpha_reversal
2016-08-12  20.94    2016-08-15  20.76   1     3526.4  -642.92    alpha_reversal
2016-08-12  23.31    2016-08-15  23.41   1     2427.4  +233.42    alpha_reversal
2016-08-09  186.42   2016-08-16  186.05  5     599.2   -224.14    alpha_reversal
2016-08-15  135.41   2016-08-17  135.38  2     411.7   -15.36     alpha_reversal
2016-08-16  92.10    2016-08-18  92.30   2     1050.8  +216.63    alpha_reversal
2016-08-16  20.66    2016-08-18  21.04   2     3418.8  +1294.27   alpha_reversal
2016-08-17  186.58   2016-08-18  186.81  1     667.3   +154.60    alpha_reversal
2016-08-17  51.34    2016-08-19  51.35   2     1377.1  +2.94      alpha_reversal
2016-08-16  14.91    2016-08-19  14.99   3     2247.1  +174.63    alpha_reversal
2016-08-18  23.25    2016-08-19  22.81   1     2599.2  -1136.91   alpha_reversal
2016-08-18  25.00    2016-08-22  24.85   2     2723.2  -423.49    alpha_reversal
2016-08-22  124.26   2016-08-23  123.36  1     525.3   -475.85    alpha_reversal
2016-08-12  40.04    2016-08-23  39.48   7     1770.5  -988.76    max_holding
2016-08-16  44.60    2016-08-23  44.94   5     1163.4  +392.18    alpha_reversal
2016-08-15  51.05    2016-08-24  51.17   7     1294.9  +165.02    max_holding
2016-08-23  15.00    2016-08-24  14.83   1     2545.2  -414.65    alpha_reversal
2016-08-22  186.72   2016-08-24  185.95  2     722.0   -553.82    alpha_reversal
2016-08-16  38.22    2016-08-25  37.94   7     1899.7  -530.17    max_holding
2016-08-22  20.60    2016-08-25  20.16   3     2984.0  -1311.87   stop_loss
2016-08-24  135.21   2016-08-26  135.62  2     445.5   +183.71    alpha_reversal
2016-08-23  24.95    2016-08-29  24.46   4     2790.6  -1366.76   stop_loss
2016-08-19  92.39    2016-08-30  91.96   7     1034.7  -453.94    max_holding
2016-08-26  14.67    2016-08-30  14.08   2     2603.0  -1538.52   stop_loss
2016-08-23  67.75    2016-08-30  66.45   5     802.4   -1043.35   alpha_reversal
2016-08-23  22.72    2016-08-30  22.80   5     2511.0  +182.97    alpha_reversal
2016-08-24  122.70   2016-08-31  119.04  5     505.6   -1847.35   stop_loss
2016-08-31  14.14    2016-09-01  13.38   1     2453.8  -1872.44   stop_loss
2016-08-24  39.37    2016-09-02  39.50   7     2011.1  +246.81    max_holding
2016-08-26  20.16    2016-09-02  20.53   5     2826.3  +1031.35   alpha_reversal
2016-08-29  186.57   2016-09-02  186.40  4     659.1   -117.28    alpha_reversal
2016-09-01  119.58   2016-09-07  121.98  3     472.6   +1134.46   alpha_reversal
2016-08-30  24.30    2016-09-08  24.16   6     2680.2  -359.65    trailing_stop
2016-09-01  65.68    2016-09-08  67.30   4     793.0   +1283.11   alpha_reversal
2016-09-06  51.39    2016-09-09  50.09   3     1354.6  -1759.54   stop_loss
2016-09-08  44.74    2016-09-09  44.17   1     1327.1  -762.81    alpha_reversal
2016-08-31  91.95    2016-09-12  91.72   7     971.7   -223.94    alpha_reversal
2016-09-02  13.19    2016-09-12  13.21   5     2147.8  +46.10     alpha_reversal
2016-09-12  50.89    2016-09-13  50.38   1     1170.4  -601.88    alpha_reversal
2016-09-12  119.78   2016-09-13  118.41  1     400.2   -548.44    alpha_reversal
2016-09-12  39.63    2016-09-13  39.09   1     1678.3  -906.65    alpha_reversal
2016-09-12  44.71    2016-09-13  44.01   1     1224.3  -856.68    alpha_reversal
2016-09-12  184.85   2016-09-13  182.01  1     489.3   -1389.47   stop_loss
2016-09-09  23.64    2016-09-14  25.59   3     2270.5  +4438.19   alpha_reversal
2016-09-07  52.16    2016-09-15  51.71   6     1346.1  -613.31    alpha_reversal
2016-09-13  20.25    2016-09-15  20.50   2     2594.4  +637.99    alpha_reversal
2016-09-14  50.18    2016-09-16  51.02   2     1114.7  +927.48    alpha_reversal
2016-09-14  39.22    2016-09-16  39.55   2     1623.8  +540.77    alpha_reversal
2016-09-12  38.59    2016-09-19  38.74   5     1615.4  +229.12    alpha_reversal
2016-09-14  182.12   2016-09-20  183.09  4     451.5   +437.38    alpha_reversal
2016-09-13  22.71    2016-09-20  23.81   5     1791.2  +1970.95   alpha_reversal
2016-09-19  51.41    2016-09-21  51.86   2     1158.5  +524.77    alpha_reversal
2016-09-20  39.68    2016-09-21  39.90   1     1634.5  +360.50    alpha_reversal
2016-09-19  136.49   2016-09-21  137.02  2     344.3   +184.61    alpha_reversal
2016-09-13  90.61    2016-09-21  91.52   6     907.0   +825.36    alpha_reversal
2016-09-21  51.52    2016-09-22  51.52   1     1110.2  +2.18      alpha_reversal
2016-09-15  44.46    2016-09-22  44.87   5     1111.8  +457.21    alpha_reversal
2016-09-20  20.40    2016-09-22  20.46   2     2884.8  +186.22    alpha_reversal
2016-09-15  117.62   2016-09-26  120.07  7     412.9   +1014.56   max_holding
2016-09-23  135.60   2016-09-29  130.39  4     369.3   -1922.15   stop_loss
2016-09-28  13.76    2016-09-29  13.37   1     2465.7  -949.03    alpha_reversal
2016-09-27  20.50    2016-09-29  20.03   2     3234.5  -1531.54   stop_loss
2016-09-26  66.43    2016-09-29  70.47   3     803.1   +3246.58   alpha_reversal
2016-09-27  24.24    2016-09-30  24.36   3     1833.1  +233.01    alpha_reversal
2016-09-30  51.72    2016-10-03  51.61   1     1044.8  -118.87    alpha_reversal
2016-09-30  132.43   2016-10-03  132.13  1     315.0   -93.41     alpha_reversal
2016-09-30  44.48    2016-10-03  44.51   1     1165.0  +39.45     alpha_reversal
2016-10-03  121.86   2016-10-04  121.62  1     475.6   -114.80    alpha_reversal
2016-10-03  14.25    2016-10-04  14.09   1     2179.4  -363.61    alpha_reversal
2016-10-04  25.90    2016-10-05  25.89   1     1903.8  -27.50     alpha_reversal
2016-10-03  39.71    2016-10-05  39.71   2     1692.8  +4.12      alpha_reversal
2016-10-04  51.06    2016-10-06  51.45   2     1254.4  +494.89    alpha_reversal
2016-10-04  20.34    2016-10-06  19.64   2     3011.1  -2098.81   stop_loss
2016-10-06  39.84    2016-10-07  39.69   1     1792.2  -281.90    alpha_reversal
2016-10-04  133.25   2016-10-07  139.32  3     327.1   +1985.07   alpha_reversal
2016-10-03  185.30   2016-10-07  184.48  4     477.1   -391.19    alpha_reversal
2016-10-05  24.34    2016-10-07  24.83   2     2018.3  +995.82    alpha_reversal
2016-10-07  91.87    2016-10-10  92.21   1     908.6   +308.23    alpha_reversal
2016-10-05  44.76    2016-10-10  45.50   3     1200.3  +886.43    alpha_reversal
2016-10-10  42.11    2016-10-11  41.53   1     1621.1  -935.87    alpha_reversal
2016-10-10  40.39    2016-10-13  39.85   3     1697.8  -917.63    alpha_reversal
2016-10-12  183.52   2016-10-13  182.74  1     482.2   -378.05    alpha_reversal
2016-10-12  24.99    2016-10-13  24.32   1     1958.4  -1327.99   stop_loss
2016-10-12  43.95    2016-10-14  44.24   2     1033.3  +307.75    alpha_reversal
2016-10-06  13.41    2016-10-17  12.92   7     2141.1  -1033.10   max_holding
2016-10-07  19.47    2016-10-17  19.32   6     2656.3  -412.68    alpha_reversal
2016-10-14  24.44    2016-10-17  24.35   1     1609.9  -154.73    alpha_reversal
2016-10-17  91.29    2016-10-18  88.83   1     824.6   -2029.95   stop_loss
2016-10-11  70.54    2016-10-18  70.27   5     648.0   -176.20    alpha_reversal
2016-10-18  13.28    2016-10-19  13.56   1     2100.9  +596.47    alpha_reversal
2016-10-17  43.85    2016-10-21  43.57   4     1004.0  -272.75    alpha_reversal
2016-10-21  53.22    2016-10-24  54.36   1     962.6   +1098.25   alpha_reversal
2016-10-17  52.55    2016-10-24  53.83   5     994.5   +1269.18   alpha_reversal
2016-10-20  24.76    2016-10-24  25.02   2     1744.5  +443.13    alpha_reversal
2016-10-17  40.67    2016-10-25  41.74   6     1524.4  +1631.58   alpha_reversal
2016-10-21  13.35    2016-10-25  13.48   2     2054.3  +280.59    alpha_reversal
2016-10-19  70.97    2016-10-25  68.66   4     673.9   -1553.93   stop_loss
2016-10-18  183.52   2016-10-25  183.73  5     468.5   +98.95     alpha_reversal
2016-10-19  26.84    2016-10-26  26.47   5     1862.7  -702.57    alpha_reversal
2016-10-24  43.30    2016-10-26  43.34   2     1106.8  +46.68     alpha_reversal
2016-10-26  54.09    2016-10-27  54.11   1     1105.9  +26.63     alpha_reversal
2016-10-27  40.94    2016-10-28  38.80   1     1368.6  -2931.29   stop_loss
2016-10-19  88.29    2016-10-28  88.77   7     714.9   +344.14    max_holding
2016-10-27  13.61    2016-10-28  13.32   1     2083.5  -589.23    alpha_reversal
2016-10-27  43.68    2016-10-28  41.89   1     975.6   -1744.37   stop_loss
2016-10-26  40.79    2016-10-31  40.14   3     1523.9  -983.61    alpha_reversal
2016-10-27  26.24    2016-11-01  25.53   3     1888.3  -1342.36   stop_loss
2016-10-26  183.55   2016-11-01  181.02  4     547.6   -1383.04   stop_loss
2016-10-27  24.53    2016-11-01  24.46   3     1962.3  -141.89    alpha_reversal
2016-10-31  39.51    2016-11-02  38.26   2     1078.6  -1350.23   stop_loss
2016-10-31  41.85    2016-11-02  41.89   2     863.3   +37.66     alpha_reversal
2016-10-26  68.44    2016-11-02  65.92   5     623.1   -1572.15   stop_loss
2016-11-01  39.96    2016-11-03  38.77   2     1263.2  -1508.66   stop_loss
2016-11-04  53.01    2016-11-08  54.73   2     977.7   +1682.86   alpha_reversal
2016-11-07  149.02   2016-11-08  149.24  1     303.2   +64.26     alpha_reversal
2016-11-02  180.11   2016-11-08  183.68  4     469.2   +1674.63   alpha_reversal
2016-11-02  12.54    2016-11-09  12.66   5     1810.7  +223.43    alpha_reversal
2016-11-08  53.94    2016-11-10  52.31   2     915.8   -1493.81   stop_loss
2016-11-02  25.58    2016-11-10  24.81   6     1765.0  -1357.19   stop_loss
2016-11-03  38.37    2016-11-10  37.10   5     946.8   -1202.65   trailing_stop
2016-11-08  40.29    2016-11-10  38.68   2     1047.4  -1687.39   stop_loss
2016-11-09  24.39    2016-11-10  23.45   1     1631.5  -1534.19   stop_loss
2016-11-03  128.89   2016-11-11  137.63  6     325.1   +2840.19   alpha_reversal
2016-11-11  23.86    2016-11-14  22.98   1     1336.5  -1181.56   stop_loss
2016-11-15  23.28    2016-11-16  23.78   1     1263.1  +634.58    alpha_reversal
2016-11-11  24.98    2016-11-17  25.31   4     1358.2  +441.23    alpha_reversal
2016-11-15  20.24    2016-11-17  19.59   2     2670.5  -1740.18   stop_loss
2016-11-18  25.36    2016-11-21  25.72   1     1340.7  +481.33    alpha_reversal
2016-11-18  135.76   2016-11-21  136.24  1     298.3   +144.72    alpha_reversal
2016-11-11  36.97    2016-11-21  38.98   6     717.4   +1443.03   alpha_reversal
2016-11-10  12.36    2016-11-21  12.30   7     1571.9  -106.36    max_holding
2016-11-11  52.65    2016-11-22  54.83   7     732.7   +1602.17   alpha_reversal
2016-11-11  38.29    2016-11-22  38.91   7     879.4   +543.90    alpha_reversal
2016-11-16  89.65    2016-11-22  87.38   4     579.2   -1318.43   stop_loss
2016-11-17  188.06   2016-11-23  189.34  4     407.7   +521.41    alpha_reversal
2016-11-18  23.50    2016-11-23  23.81   3     1286.4  +400.12    alpha_reversal
2016-11-22  12.75    2016-11-25  13.10   2     1619.0  +570.56    alpha_reversal
2016-11-25  54.36    2016-11-28  54.38   1     818.1   +14.27     alpha_reversal
2016-11-25  25.75    2016-11-28  25.68   1     1552.5  -118.57    alpha_reversal
2016-11-18  19.43    2016-11-28  20.16   5     2290.7  +1674.24   alpha_reversal
2016-11-25  38.71    2016-11-29  39.13   2     945.5   +395.03    alpha_reversal
2016-11-17  44.69    2016-11-29  44.28   7     776.3   -316.57    max_holding
2016-11-17  75.48    2016-11-29  76.43   7     403.2   +382.50    alpha_reversal
2016-11-29  140.66   2016-11-30  139.52  1     334.7   -382.08    alpha_reversal
2016-11-28  61.28    2016-11-30  62.66   2     668.3   +925.40    alpha_reversal
2016-11-28  172.73   2016-11-30  180.45  2     222.7   +1719.12   alpha_reversal
2016-11-25  23.65    2016-12-01  22.74   4     1521.4  -1381.01   stop_loss
2016-11-30  25.46    2016-12-05  25.11   3     1644.4  -575.47    alpha_reversal
2016-11-23  87.72    2016-12-05  86.76   7     598.6   -576.69    alpha_reversal
2016-12-02  23.01    2016-12-06  23.50   2     1508.5  +746.30    alpha_reversal
2016-12-06  77.47    2016-12-07  79.10   1     482.4   +789.91    alpha_reversal
2016-11-30  37.55    2016-12-09  38.41   7     863.5   +748.20    max_holding
2016-12-07  39.27    2016-12-09  40.12   2     939.2   +800.14    alpha_reversal
2016-12-01  53.16    2016-12-12  55.78   7     848.1   +2214.75   max_holding
2016-12-01  12.13    2016-12-12  12.82   7     1763.9  +1218.62   max_holding
2016-12-05  141.14   2016-12-13  145.17  6     351.0   +1414.17   alpha_reversal
2016-12-06  86.93    2016-12-13  89.82   5     719.0   +2071.68   alpha_reversal
2016-12-06  25.33    2016-12-14  26.51   6     1734.8  +2048.30   alpha_reversal
2016-12-13  195.59   2016-12-14  193.78  1     567.1   -1025.50   alpha_reversal
2016-12-06  42.96    2016-12-15  44.75   7     911.4   +1627.75   alpha_reversal
2016-12-06  19.94    2016-12-15  20.27   7     2734.0  +896.65    max_holding
2016-12-08  78.32    2016-12-16  75.24   6     491.2   -1511.70   stop_loss
2016-12-15  194.77   2016-12-16  194.20  1     550.1   -316.46    alpha_reversal
2016-12-14  143.29   2016-12-19  144.73  3     343.1   +494.48    alpha_reversal
2016-12-16  13.51    2016-12-19  13.51   1     2154.4  +5.37      alpha_reversal
2016-12-13  24.21    2016-12-19  23.31   4     1667.4  -1487.88   stop_loss
2016-12-19  57.13    2016-12-20  57.00   1     909.6   -117.23    alpha_reversal
2016-12-13  66.31    2016-12-20  67.63   5     700.3   +922.39    alpha_reversal
2016-12-13  38.74    2016-12-20  38.54   5     1026.6  -199.82    alpha_reversal
2016-12-19  196.92   2016-12-20  200.04  1     199.0   +619.01    alpha_reversal
2016-12-21  89.46    2016-12-23  89.87   2     727.0   +301.22    alpha_reversal
2016-12-19  75.43    2016-12-23  76.66   4     550.5   +678.81    alpha_reversal
2016-12-22  197.79   2016-12-27  198.78  2     210.1   +207.46    alpha_reversal
2016-12-19  194.82   2016-12-27  195.26  5     569.0   +252.60    alpha_reversal
2016-12-27  56.83    2016-12-28  56.51   1     999.5   -316.80    alpha_reversal
2016-12-22  38.34    2016-12-29  38.24   4     1227.2  -120.01    alpha_reversal
2016-12-20  23.37    2016-12-29  23.20   6     1789.6  -298.37    alpha_reversal
2016-12-28  89.29    2016-12-30  89.29   2     784.1   -3.14      alpha_reversal
2016-12-20  43.45    2016-12-30  42.24   7     1003.3  -1216.92   max_holding
2016-12-29  67.20    2017-01-03  68.18   2     818.8   +802.55    alpha_reversal
2016-12-30  197.24   2017-01-03  198.78  1     231.4   +358.08    alpha_reversal
2016-12-23  40.08    2017-01-04  40.04   6     1342.3  -55.76     alpha_reversal
2016-12-22  19.86    2017-01-04  19.69   7     2915.6  -498.52    max_holding
2017-01-03  37.70    2017-01-06  39.78   3     1251.1  +2598.81   alpha_reversal
2017-01-05  40.34    2017-01-06  40.90   1     1390.4  +784.05    alpha_reversal
2017-01-05  23.76    2017-01-06  23.60   1     1981.0  -315.28    alpha_reversal
2016-12-28  144.80   2017-01-09  146.71  7     381.4   +729.49    alpha_reversal
2016-12-29  56.49    2017-01-10  56.18   7     1064.3  -327.42    max_holding
2016-12-30  26.68    2017-01-10  27.41   6     2293.1  +1675.17   alpha_reversal
2017-01-05  90.66    2017-01-11  88.92   4     837.1   -1457.67   stop_loss
2016-12-30  193.09   2017-01-11  195.98  7     633.1   +1828.32   max_holding
2017-01-05  43.17    2017-01-12  44.63   5     1189.3  +1740.63   alpha_reversal
2017-01-05  19.76    2017-01-13  19.14   6     3160.0  -1936.71   stop_loss
2017-01-06  67.75    2017-01-17  65.66   6     834.5   -1742.00   stop_loss
2017-01-06  201.73   2017-01-17  193.99  6     222.3   -1720.56   stop_loss
2017-01-12  15.31    2017-01-17  15.70   2     2168.8  +832.46    alpha_reversal
2017-01-05  75.66    2017-01-17  76.05   7     639.7   +247.98    alpha_reversal
2017-01-13  23.47    2017-01-17  23.59   1     1978.6  +237.26    alpha_reversal
2017-01-17  19.53    2017-01-18  19.42   1     3053.3  -329.53    alpha_reversal
2017-01-19  40.47    2017-01-20  40.40   1     1366.3  -103.75    alpha_reversal
2017-01-11  56.75    2017-01-23  56.48   7     1154.3  -303.63    alpha_reversal
2017-01-23  40.91    2017-01-24  41.10   1     1370.3  +256.24    alpha_reversal
2017-01-20  41.09    2017-01-24  42.11   2     1723.1  +1753.47   alpha_reversal
2017-01-12  88.92    2017-01-24  86.62   7     852.5   -1965.34   stop_loss
2017-01-19  23.62    2017-01-24  24.55   3     2170.5  +2006.00   alpha_reversal
2017-01-24  148.93   2017-01-25  155.09  1     423.2   +2607.77   alpha_reversal
2017-01-25  28.08    2017-01-26  28.07   1     2813.6  -40.10     alpha_reversal
2017-01-18  66.04    2017-01-27  68.32   7     744.4   +1700.09   max_holding
2017-01-18  192.99   2017-01-27  194.98  7     209.1   +417.32    max_holding
2017-01-19  19.30    2017-01-27  18.72   6     3131.2  -1810.49   stop_loss
2017-01-27  41.93    2017-01-30  40.83   1     1566.6  -1711.79   stop_loss
2017-01-27  81.20    2017-01-30  79.32   1     625.8   -1179.00   alpha_reversal
2017-01-25  87.51    2017-02-01  87.75   5     836.7   +205.66    alpha_reversal
2017-01-25  43.87    2017-02-01  44.55   5     1041.4  +716.45    alpha_reversal
2017-02-01  29.66    2017-02-02  29.58   1     2052.8  -164.79    alpha_reversal
2017-02-02  16.78    2017-02-03  16.75   1     2147.8  -67.51     alpha_reversal
2017-02-01  190.01   2017-02-06  197.48  3     227.5   +1699.96   alpha_reversal
2017-01-31  40.69    2017-02-07  41.10   5     1416.0  +576.88    alpha_reversal
2017-02-06  17.19    2017-02-07  17.16   1     2227.1  -81.31     alpha_reversal
2017-02-06  152.11   2017-02-08  153.11  2     363.0   +361.63    alpha_reversal
2017-02-07  88.04    2017-02-08  87.89   1     970.1   -145.53    alpha_reversal
2017-01-30  18.96    2017-02-08  19.34   7     3223.0  +1216.41   alpha_reversal
2017-02-07  197.76   2017-02-08  197.82  1     710.1   +43.51     alpha_reversal
2017-02-03  68.58    2017-02-10  68.37   5     759.3   -159.48    alpha_reversal
2017-02-08  195.82   2017-02-10  199.73  2     226.1   +884.26    alpha_reversal
2017-02-02  56.73    2017-02-13  58.06   7     1083.9  +1445.72   max_holding
2017-02-09  153.70   2017-02-13  157.05  2     388.1   +1300.78   alpha_reversal
2017-02-06  40.40    2017-02-13  41.81   5     1250.5  +1754.89   alpha_reversal
2017-02-02  76.92    2017-02-13  80.72   7     581.3   +2208.59   alpha_reversal
2017-02-09  88.50    2017-02-14  90.18   3     1009.7  +1694.89   alpha_reversal
2017-02-07  24.33    2017-02-14  24.95   5     2431.0  +1509.25   alpha_reversal
2017-02-13  69.35    2017-02-15  71.20   2     859.2   +1588.03   alpha_reversal
2017-02-15  25.31    2017-02-16  25.74   1     2411.5  +1033.80   alpha_reversal
2017-02-16  41.78    2017-02-17  41.96   1     2053.2  +360.00    alpha_reversal
2017-02-17  206.24   2017-02-21  207.17  1     272.7   +253.47    alpha_reversal
2017-02-17  18.16    2017-02-21  18.48   1     2085.8  +679.32    alpha_reversal
2017-02-13  19.34    2017-02-21  20.37   5     3432.1  +3535.25   alpha_reversal
2017-02-22  18.24    2017-02-23  17.06   1     2025.4  -2401.39   stop_loss
2017-02-13  46.52    2017-02-23  47.25   7     1279.8  +932.19    max_holding
2017-02-22  80.55    2017-02-23  78.30   1     749.9   -1688.85   stop_loss
2017-02-23  94.41    2017-02-24  95.75   1     1026.9  +1369.17   alpha_reversal
2017-02-23  204.24   2017-02-27  204.61  2     822.8   +307.76    alpha_reversal
2017-02-22  25.57    2017-02-27  25.11   3     2542.2  -1158.36   alpha_reversal
2017-02-16  58.29    2017-02-28  57.75   7     1398.0  -762.83    max_holding
2017-02-27  71.14    2017-02-28  71.22   1     1039.8  +81.34     alpha_reversal
2017-02-27  42.16    2017-02-28  41.88   1     2257.3  -625.45    alpha_reversal
2017-02-23  206.91   2017-02-28  204.66  3     308.1   -692.95    alpha_reversal
2017-02-28  47.31    2017-03-01  47.50   1     1551.8  +305.15    alpha_reversal
2017-02-24  78.32    2017-03-02  77.33   4     678.8   -676.20    trailing_stop
2017-03-01  207.12   2017-03-02  205.61  1     743.5   -1122.74   alpha_reversal
2017-02-24  17.14    2017-03-03  16.76   5     1815.9  -688.13    alpha_reversal
2017-03-01  20.11    2017-03-03  19.97   2     3015.0  -421.71    alpha_reversal
2017-03-02  42.47    2017-03-06  42.31   2     1764.2  -277.65    alpha_reversal
2017-03-06  42.04    2017-03-08  42.31   2     2186.5  +598.48    alpha_reversal
2017-03-08  42.55    2017-03-09  42.63   1     1923.0  +158.48    alpha_reversal
2017-03-06  19.95    2017-03-09  20.07   3     3103.6  +375.05    alpha_reversal
2017-03-03  78.03    2017-03-09  74.89   4     558.5   -1750.72   stop_loss
2017-02-28  25.09    2017-03-09  24.68   7     2595.3  -1077.94   max_holding
2017-03-09  32.09    2017-03-14  32.13   3     2851.3  +112.89    alpha_reversal
2017-03-08  71.75    2017-03-14  71.92   4     869.4   +142.64    alpha_reversal
2017-03-13  42.75    2017-03-14  42.61   1     2097.5  -305.56    alpha_reversal
2017-03-08  16.47    2017-03-15  17.04   5     1994.8  +1144.87   alpha_reversal
2017-03-14  20.34    2017-03-15  20.27   1     3480.3  -210.72    alpha_reversal
2017-03-15  58.50    2017-03-16  58.34   1     1512.9  -238.68    alpha_reversal
2017-03-15  32.50    2017-03-17  32.36   2     3038.1  -428.75    alpha_reversal
2017-03-09  168.94   2017-03-20  167.67  7     422.7   -537.53    alpha_reversal
2017-03-13  204.95   2017-03-20  199.77  5     295.5   -1528.14   stop_loss
2017-03-20  32.73    2017-03-21  32.32   1     3129.5  -1274.30   alpha_reversal
2017-03-20  70.83    2017-03-21  68.68   1     903.7   -1938.95   stop_loss
2017-03-20  42.87    2017-03-21  42.14   1     2344.9  -1714.15   stop_loss
2017-03-20  100.01   2017-03-21  99.27   1     913.4   -675.57    alpha_reversal
2017-03-13  46.40    2017-03-21  46.18   6     1489.9  -327.49    alpha_reversal
2017-03-16  20.25    2017-03-21  20.08   3     3642.2  -638.70    alpha_reversal
2017-03-10  75.72    2017-03-21  75.75   7     529.8   +12.02     alpha_reversal
2017-03-21  164.63   2017-03-22  165.41  1     400.1   +315.60    alpha_reversal
2017-03-22  32.72    2017-03-23  32.57   1     2618.1  -388.22    alpha_reversal
2017-03-22  68.86    2017-03-23  68.68   1     738.3   -132.04    alpha_reversal
2017-03-21  42.18    2017-03-24  41.39   3     1975.0  -1551.60   stop_loss
2017-03-21  192.43   2017-03-27  186.03  4     246.1   -1574.27   stop_loss
2017-03-24  17.55    2017-03-27  18.01   1     1811.9  +820.61    alpha_reversal
2017-03-27  164.76   2017-03-28  165.77  1     392.6   +397.66    alpha_reversal
2017-03-23  98.32    2017-03-28  98.03   3     859.0   -245.25    alpha_reversal
2017-03-24  20.02    2017-03-28  20.20   2     3665.4  +674.24    alpha_reversal
2017-03-22  42.42    2017-03-29  43.69   5     1951.6  +2478.45   alpha_reversal
2017-03-29  46.04    2017-03-30  45.85   1     1603.5  -293.96    alpha_reversal
2017-03-21  202.78   2017-03-30  204.79  7     647.5   +1305.34   alpha_reversal
2017-03-29  69.44    2017-03-31  69.03   2     725.5   -295.54    alpha_reversal
2017-03-30  26.37    2017-03-31  26.16   1     2342.3  -490.81    alpha_reversal
2017-03-31  45.96    2017-04-03  45.87   1     1734.8  -167.43    alpha_reversal
2017-03-27  41.60    2017-04-04  42.26   6     1668.0  +1093.03   alpha_reversal
2017-03-29  97.55    2017-04-04  97.27   4     895.1   -254.88    alpha_reversal
2017-04-03  26.22    2017-04-04  26.17   1     2384.2  -119.47    alpha_reversal
2017-04-03  165.27   2017-04-05  165.51  2     438.3   +103.76    alpha_reversal
2017-04-04  46.06    2017-04-05  45.94   1     1831.3  -216.63    alpha_reversal
2017-03-28  189.40   2017-04-06  188.64  7     220.7   -167.39    max_holding
2017-04-05  97.46    2017-04-06  97.56   1     977.9   +95.46     alpha_reversal
2017-03-28  76.25    2017-04-06  78.52   7     554.0   +1260.85   alpha_reversal
2017-04-06  204.26   2017-04-07  203.85  1     616.0   -253.91    alpha_reversal
2017-04-07  20.96    2017-04-10  20.99   1     3512.7  +87.86     alpha_reversal
2017-04-05  26.23    2017-04-10  25.90   3     2494.4  -820.35    alpha_reversal
2017-04-07  44.77    2017-04-11  45.10   2     1500.6  +493.82    alpha_reversal
2017-04-10  189.03   2017-04-11  187.89  1     259.8   -295.58    alpha_reversal
2017-04-07  45.67    2017-04-12  45.56   3     1887.8  -195.31    alpha_reversal
2017-04-07  33.17    2017-04-13  32.60   4     2650.2  -1490.74   stop_loss
2017-04-04  69.08    2017-04-13  66.71   7     770.9   -1826.35   stop_loss
2017-04-10  41.76    2017-04-13  41.64   3     1854.6  -217.14    alpha_reversal
2017-04-10  97.10    2017-04-13  97.51   3     1054.7  +432.51    alpha_reversal
2017-04-17  45.43    2017-04-18  45.00   1     1845.4  -790.55    alpha_reversal
2017-04-13  44.26    2017-04-19  44.94   3     1453.5  +991.13    alpha_reversal
2017-04-12  25.74    2017-04-19  25.18   4     2456.5  -1374.08   stop_loss
2017-04-10  59.21    2017-04-20  59.12   7     1469.0  -126.74    max_holding
2017-04-19  166.91   2017-04-20  167.58  1     413.5   +278.88    alpha_reversal
2017-04-18  20.03    2017-04-20  20.16   2     1588.6  +207.43    alpha_reversal
2017-04-17  67.93    2017-04-25  69.76   6     724.5   +1325.13   alpha_reversal
2017-04-20  78.30    2017-04-25  86.28   3     551.9   +4407.74   alpha_reversal
2017-04-20  204.17   2017-04-25  206.75  3     571.9   +1474.52   alpha_reversal
2017-04-17  32.82    2017-04-26  33.21   7     2536.7  +1001.60   max_holding
2017-04-18  178.05   2017-04-27  186.30  7     220.6   +1820.50   max_holding
2017-04-25  45.40    2017-04-28  46.23   3     1555.0  +1279.25   alpha_reversal
2017-04-19  94.78    2017-04-28  96.32   7     858.6   +1325.32   max_holding
2017-04-27  171.42   2017-05-01  170.47  2     447.7   -424.04    alpha_reversal
2017-04-27  20.59    2017-05-01  21.51   2     1864.6  +1725.94   alpha_reversal
2017-04-20  25.35    2017-05-01  26.60   7     2472.1  +3107.46   max_holding
2017-04-21  44.77    2017-05-02  45.31   7     1815.9  +981.65    alpha_reversal
2017-05-01  207.07   2017-05-02  206.94  1     689.7   -88.95     alpha_reversal
2017-05-02  171.62   2017-05-03  171.41  1     478.5   -104.44    alpha_reversal
2017-04-28  68.84    2017-05-04  68.85   4     782.5   +8.00      alpha_reversal
2017-04-25  21.58    2017-05-04  21.93   7     4077.8  +1423.11   alpha_reversal
2017-05-02  96.60    2017-05-05  96.36   3     1154.1  -282.49    alpha_reversal
2017-05-04  33.90    2017-05-08  35.37   2     2609.0  +3819.50   alpha_reversal
2017-05-08  68.92    2017-05-09  68.57   1     973.0   -336.16    alpha_reversal
2017-05-05  20.57    2017-05-10  21.67   3     1573.8  +1736.73   alpha_reversal
2017-05-09  62.38    2017-05-11  61.79   2     1543.2  -904.03    alpha_reversal
2017-05-10  69.18    2017-05-11  68.89   1     986.2   -278.64    alpha_reversal
2017-05-09  207.73   2017-05-11  207.47  2     845.0   -219.38    alpha_reversal
2017-05-11  47.43    2017-05-12  47.34   1     1786.0  -151.05    alpha_reversal
2017-05-03  83.99    2017-05-12  83.22   7     562.9   -433.29    max_holding
2017-05-08  47.48    2017-05-15  47.87   5     1559.5  +621.98    alpha_reversal
2017-05-08  96.25    2017-05-15  99.07   5     1279.5  +3610.05   alpha_reversal
2017-05-12  68.77    2017-05-16  69.25   2     1024.0  +488.09    alpha_reversal
2017-05-12  21.92    2017-05-16  21.72   2     5070.6  -990.88    alpha_reversal
2017-05-12  61.78    2017-05-17  61.26   3     1514.0  -794.76    trailing_stop
2017-05-11  173.41   2017-05-17  168.38  4     456.6   -2298.22   stop_loss
2017-05-16  48.33    2017-05-17  47.21   1     1728.8  -1924.59   stop_loss
2017-05-09  184.80   2017-05-17  176.33  6     308.1   -2609.17   stop_loss
2017-05-16  46.14    2017-05-17  45.53   1     1868.1  -1152.63   alpha_reversal
2017-05-12  207.33   2017-05-17  204.39  3     882.7   -2600.46   trailing_stop
2017-05-17  21.74    2017-05-18  22.42   1     4592.0  +3113.73   alpha_reversal
2017-05-18  205.41   2017-05-19  206.54  1     714.4   +806.83    alpha_reversal
2017-05-23  85.71    2017-05-24  85.89   1     625.9   +111.86    alpha_reversal
2017-05-19  170.41   2017-05-25  176.18  4     409.0   +2361.05   alpha_reversal
2017-05-18  177.69   2017-05-25  183.55  5     276.1   +1616.33   alpha_reversal
2017-05-16  21.14    2017-05-25  21.11   7     1742.8  -57.74     max_holding
2017-05-16  27.93    2017-05-25  28.33   7     3026.2  +1217.15   alpha_reversal
2017-05-18  35.44    2017-05-30  35.67   7     1871.1  +424.46    max_holding
2017-05-18  66.43    2017-05-30  66.32   7     893.4   -101.69    max_holding
2017-05-26  184.61   2017-05-30  180.81  1     302.4   -1147.38   alpha_reversal
2017-05-30  176.18   2017-05-31  176.71  1     459.8   +243.84    alpha_reversal
2017-05-25  100.25   2017-05-31  100.72  3     1108.9  +515.86    alpha_reversal
2017-05-30  28.53    2017-05-31  28.16   1     3319.1  -1204.97   alpha_reversal
2017-06-01  35.59    2017-06-02  36.08   1     2424.9  +1191.38   alpha_reversal
2017-05-30  22.62    2017-06-06  22.82   5     4291.4  +870.93    alpha_reversal
2017-06-01  28.61    2017-06-06  29.02   3     3052.7  +1225.66   alpha_reversal
2017-05-30  46.93    2017-06-07  46.29   6     1912.4  -1222.95   trailing_stop
2017-05-30  209.52   2017-06-08  211.28  7     856.5   +1513.13   max_holding
2017-06-06  35.88    2017-06-09  34.58   3     2344.0  -3060.10   stop_loss
2017-06-02  179.34   2017-06-09  178.97  5     429.0   -157.72    alpha_reversal
2017-05-31  65.00    2017-06-09  68.74   7     965.6   +3608.49   alpha_reversal
2017-05-31  87.20    2017-06-09  87.52   7     685.4   +217.78    max_holding
2017-06-01  178.17   2017-06-12  183.70  7     271.1   +1500.28   alpha_reversal
2017-06-09  103.40   2017-06-12  103.52  1     1135.1  +141.20    alpha_reversal
2017-06-12  23.95    2017-06-13  25.05   1     1449.0  +1600.90   alpha_reversal
2017-06-09  22.99    2017-06-14  23.11   3     3679.0  +426.15    alpha_reversal
2017-06-08  45.72    2017-06-15  46.00   5     1934.4  +547.15    alpha_reversal
2017-06-14  86.61    2017-06-16  88.91   2     734.0   +1689.23   alpha_reversal
2017-06-12  63.41    2017-06-19  64.33   5     990.0   +916.85    alpha_reversal
2017-06-12  48.27    2017-06-19  49.73   5     1146.5  +1678.40   alpha_reversal
2017-06-12  28.10    2017-06-20  28.78   6     2549.9  +1735.99   alpha_reversal
2017-06-12  33.79    2017-06-21  33.86   7     1708.1  +120.73    max_holding
2017-06-12  47.72    2017-06-21  48.50   7     1352.8  +1060.60   max_holding
2017-06-20  105.51   2017-06-21  105.95  1     1178.9  +514.46    alpha_reversal
2017-06-19  46.40    2017-06-21  47.65   2     1978.6  +2472.00   alpha_reversal
2017-06-20  88.54    2017-06-21  85.50   1     762.1   -2315.52   stop_loss
2017-06-12  211.13   2017-06-21  211.59  7     881.3   +405.81    max_holding
2017-06-23  33.99    2017-06-26  33.85   1     1911.9  -269.06    alpha_reversal
2017-06-23  48.92    2017-06-26  48.18   1     1500.5  -1114.59   alpha_reversal
2017-06-22  85.89    2017-06-26  86.14   2     736.8   +186.37    alpha_reversal
2017-06-22  68.85    2017-06-27  69.60   3     1188.2  +885.61    alpha_reversal
2017-06-19  21.85    2017-06-28  22.13   7     2721.2  +735.33    alpha_reversal
2017-06-21  63.85    2017-06-29  62.17   6     1091.5  -1833.30   trailing_stop
2017-06-28  47.68    2017-06-30  46.08   2     1319.2  -2111.46   stop_loss
2017-06-21  184.36   2017-06-30  183.69  7     329.2   -221.50    alpha_reversal
2017-06-28  24.76    2017-07-03  23.50   3     1534.9  -1942.40   stop_loss
2017-06-22  211.71   2017-07-03  210.95  7     919.6   -699.11    alpha_reversal
2017-06-29  104.27   2017-07-05  104.95  3     1049.6  +714.85    alpha_reversal
2017-06-29  46.88    2017-07-05  46.71   3     1886.0  -335.50    alpha_reversal
2017-07-05  62.77    2017-07-06  62.24   1     996.6   -523.95    alpha_reversal
2017-07-05  33.48    2017-07-06  33.13   1     1863.9  -650.76    alpha_reversal
2017-06-28  88.05    2017-07-06  88.01   5     711.8   -27.35     alpha_reversal
2017-07-06  21.85    2017-07-07  21.78   1     3353.3  -208.95    alpha_reversal
2017-06-29  48.82    2017-07-11  49.68   7     1168.6  +1005.90   alpha_reversal
2017-07-03  45.62    2017-07-11  47.26   5     1228.3  +2018.23   alpha_reversal
2017-07-10  187.14   2017-07-11  187.87  1     302.4   +221.31    alpha_reversal
2017-07-05  28.67    2017-07-12  29.35   5     2428.4  +1668.00   alpha_reversal
2017-07-05  21.82    2017-07-14  21.84   7     1275.7  +30.83     max_holding
2017-07-12  212.73   2017-07-14  213.87  2     758.4   +862.57    alpha_reversal
2017-07-06  45.98    2017-07-17  45.58   7     1768.9  -712.25    max_holding
2017-07-14  196.57   2017-07-18  196.12  2     478.0   -215.47    alpha_reversal
2017-07-17  189.97   2017-07-18  184.86  1     331.2   -1694.24   alpha_reversal
2017-07-07  104.19   2017-07-18  105.59  7     1022.4  +1435.14   max_holding
2017-07-10  21.20    2017-07-18  22.04   6     3188.4  +2670.83   alpha_reversal
2017-07-18  214.17   2017-07-19  215.11  1     862.6   +814.48    alpha_reversal
2017-07-12  73.60    2017-07-21  72.24   7     1065.2  -1449.90   max_holding
2017-07-18  21.89    2017-07-21  21.88   3     1317.2  -14.78     alpha_reversal
2017-07-20  22.01    2017-07-21  22.02   1     3861.0  +60.24     alpha_reversal
2017-07-19  198.81   2017-07-24  199.83  3     508.2   +521.21    alpha_reversal
2017-07-24  72.62    2017-07-25  73.76   1     1196.8  +1358.90   alpha_reversal
2017-07-24  22.85    2017-07-25  22.63   1     1445.9  -314.35    alpha_reversal
2017-07-14  89.97    2017-07-25  95.33   7     760.5   +4076.79   alpha_reversal
2017-07-21  34.91    2017-07-26  35.62   3     2302.8  +1624.73   alpha_reversal
2017-07-18  45.48    2017-07-27  46.36   7     2022.3  +1792.45   max_holding
2017-07-18  29.53    2017-07-27  29.76   7     2705.9  +610.01    alpha_reversal
2017-07-19  184.68   2017-07-28  185.11  7     324.1   +138.69    max_holding
2017-07-28  51.03    2017-07-31  49.36   1     1201.9  -1999.03   stop_loss
2017-07-27  72.83    2017-08-01  73.94   3     1177.4  +1299.17   alpha_reversal
2017-07-31  29.60    2017-08-01  29.48   1     2995.6  -384.31    alpha_reversal
2017-07-24  66.88    2017-08-02  65.59   7     1169.8  -1501.08   max_holding
2017-07-28  34.73    2017-08-02  36.47   3     2193.1  +3812.93   alpha_reversal
2017-07-24  215.18   2017-08-02  215.50  7     949.2   +308.34    max_holding
2017-07-26  47.89    2017-08-03  46.61   6     1442.1  -1856.72   stop_loss
2017-07-25  103.67   2017-08-03  104.72  7     879.0   +923.63    max_holding
2017-08-02  21.74    2017-08-04  23.78   2     1454.0  +2973.84   alpha_reversal
2017-08-02  94.21    2017-08-04  95.17   2     789.1   +753.15    alpha_reversal
2017-08-04  74.51    2017-08-07  74.72   1     1232.1  +260.73    alpha_reversal
2017-08-04  190.41   2017-08-07  192.81  1     400.9   +962.45    alpha_reversal
2017-08-04  215.69   2017-08-07  215.88  1     1154.7  +213.63    alpha_reversal
2017-08-02  29.69    2017-08-07  30.26   3     3067.7  +1725.47   alpha_reversal
2017-08-04  66.04    2017-08-09  65.78   3     1292.9  -331.81    alpha_reversal
2017-08-07  226.48   2017-08-10  220.64  3     321.2   -1876.01   stop_loss
2017-08-01  49.83    2017-08-10  47.82   7     1150.8  -2315.73   stop_loss
2017-08-07  104.44   2017-08-10  104.52  3     1036.0  +79.01     alpha_reversal
2017-08-09  215.55   2017-08-10  212.30  1     1155.5  -3761.21   stop_loss
2017-08-09  30.02    2017-08-10  29.06   1     3072.8  -2922.62   stop_loss
2017-08-11  212.82   2017-08-14  214.72  1     974.8   +1847.10   alpha_reversal
2017-08-04  46.93    2017-08-15  46.50   7     1509.7  -647.73    max_holding
2017-08-14  73.58    2017-08-16  73.19   2     1186.9  -464.65    alpha_reversal
2017-08-14  188.40   2017-08-16  186.76  2     360.1   -589.50    alpha_reversal
2017-08-07  45.79    2017-08-16  45.64   7     2175.4  -321.24    max_holding
2017-08-15  29.98    2017-08-16  29.89   1     2677.9  -256.44    alpha_reversal
2017-08-16  46.85    2017-08-17  45.98   1     1729.2  -1504.56   alpha_reversal
2017-08-16  24.21    2017-08-17  23.45   1     1565.0  -1183.91   alpha_reversal
2017-08-16  48.93    2017-08-18  47.90   2     1332.8  -1378.04   alpha_reversal
2017-08-17  72.12    2017-08-21  72.03   2     1161.1  -102.17    alpha_reversal
2017-08-17  183.48   2017-08-21  182.77  2     352.5   -248.50    alpha_reversal
2017-08-14  94.73    2017-08-21  94.55   5     914.5   -170.33    alpha_reversal
2017-08-17  29.46    2017-08-21  29.59   2     2674.2  +339.10    alpha_reversal
2017-08-18  23.10    2017-08-22  23.29   2     3540.1  +649.69    alpha_reversal
2017-08-22  66.83    2017-08-23  66.36   1     1218.7  -570.79    alpha_reversal
2017-08-22  72.84    2017-08-23  72.86   1     1167.8  +17.02     alpha_reversal
2017-08-22  46.66    2017-08-24  46.44   2     1665.6  -367.44    alpha_reversal
2017-08-22  227.37   2017-08-25  223.49  3     333.9   -1296.95   alpha_reversal
2017-08-22  185.27   2017-08-25  184.16  3     359.7   -397.10    alpha_reversal
2017-08-24  66.40    2017-08-28  66.46   2     1271.3  +78.09     alpha_reversal
2017-08-25  73.11    2017-08-28  72.80   1     1190.3  -361.35    alpha_reversal
2017-08-28  224.93   2017-08-29  227.85  1     348.2   +1013.72   alpha_reversal
2017-08-18  211.59   2017-08-29  213.25  7     810.7   +1339.53   alpha_reversal
2017-08-28  46.05    2017-08-30  46.77   2     1723.2  +1244.56   alpha_reversal
2017-08-28  104.22   2017-08-30  103.59  2     1022.7  -648.06    alpha_reversal
2017-08-21  22.54    2017-08-30  23.53   7     1498.1  +1495.50   alpha_reversal
2017-08-29  96.65    2017-08-30  97.83   1     906.4   +1074.17   alpha_reversal
2017-08-29  30.36    2017-08-30  30.37   1     2962.5  +31.92     alpha_reversal
2017-08-29  72.48    2017-08-31  72.24   2     1192.9  -285.51    alpha_reversal
2017-08-22  48.37    2017-08-31  49.01   7     1345.9  +856.41    max_holding
2017-08-31  104.72   2017-09-01  103.55  1     1071.5  -1246.89   alpha_reversal
2017-09-01  72.95    2017-09-05  71.14   1     1242.5  -2253.26   stop_loss
2017-08-28  182.59   2017-09-05  180.90  5     370.7   -628.03    trailing_stop
2017-08-31  227.29   2017-09-06  222.28  3     359.7   -1799.16   stop_loss
2017-08-25  22.90    2017-09-06  23.30   7     3340.0  +1332.85   alpha_reversal
2017-09-05  48.29    2017-09-07  48.95   2     1454.5  +962.00    alpha_reversal
2017-09-05  46.71    2017-09-08  46.66   3     1725.7  -86.56     alpha_reversal
2017-09-05  102.76   2017-09-08  103.51  3     1023.3  +768.34    alpha_reversal
2017-09-07  224.11   2017-09-11  227.94  2     353.9   +1355.65   alpha_reversal
2017-09-07  98.11    2017-09-11  98.93   2     886.9   +724.96    alpha_reversal
2017-09-05  23.32    2017-09-12  24.17   5     1787.3  +1525.59   alpha_reversal
2017-09-06  30.68    2017-09-12  31.14   4     2982.0  +1355.55   alpha_reversal
2017-09-06  71.69    2017-09-13  72.44   5     1114.9  +841.69    alpha_reversal
2017-09-06  181.95   2017-09-13  188.19  5     340.5   +2124.52   alpha_reversal
2017-09-05  67.24    2017-09-14  68.23   7     1257.0  +1246.22   alpha_reversal
2017-09-13  229.44   2017-09-14  232.34  1     346.2   +1002.92   alpha_reversal
2017-09-11  48.92    2017-09-14  49.59   3     1502.1  +996.25    alpha_reversal
2017-09-13  24.43    2017-09-14  25.16   1     1800.3  +1324.78   alpha_reversal
2017-09-14  48.53    2017-09-15  48.51   1     2453.2  -47.04     alpha_reversal
2017-09-14  100.37   2017-09-15  101.01  1     907.8   +581.30    alpha_reversal
2017-09-15  68.80    2017-09-18  68.59   1     1379.2  -283.67    alpha_reversal
2017-09-07  37.61    2017-09-18  36.97   7     2002.6  -1283.88   max_holding
2017-09-12  217.99   2017-09-19  218.79  5     848.1   +672.37    alpha_reversal
2017-09-14  30.77    2017-09-19  31.50   3     3463.9  +2514.38   alpha_reversal
2017-09-11  46.80    2017-09-20  46.97   7     1795.5  +294.22    max_holding
2017-09-15  23.41    2017-09-20  23.43   3     3843.1  +44.25     alpha_reversal
2017-09-19  37.02    2017-09-21  35.74   2     2037.9  -2611.22   stop_loss
2017-09-15  49.36    2017-09-21  48.21   4     1648.9  -1905.79   stop_loss
2017-09-20  24.94    2017-09-22  23.39   2     1838.7  -2841.72   stop_loss
2017-09-21  31.48    2017-09-22  31.06   1     3116.5  -1302.72   alpha_reversal
2017-09-19  68.91    2017-09-25  66.86   4     1402.6  -2886.93   stop_loss
2017-09-19  48.40    2017-09-26  47.48   5     2521.2  -2321.86   stop_loss
2017-09-25  46.35    2017-09-27  47.58   2     1865.0  +2281.85   alpha_reversal
2017-09-22  23.17    2017-09-27  23.07   3     4011.0  -373.01    alpha_reversal
2017-09-25  74.88    2017-09-28  75.81   3     1124.3  +1041.74   alpha_reversal
2017-09-22  218.54   2017-09-28  219.12  4     1116.2  +645.20    alpha_reversal
2017-09-26  66.92    2017-09-29  67.98   3     1291.7  +1363.43   alpha_reversal
2017-09-25  191.46   2017-09-29  197.02  4     387.4   +2156.03   alpha_reversal
2017-09-22  35.43    2017-10-02  35.84   6     1868.3  +769.71    alpha_reversal
2017-09-21  104.23   2017-10-02  103.70  7     875.2   -457.81    max_holding
2017-09-29  30.91    2017-10-02  31.36   1     2919.2  +1302.38   alpha_reversal
2017-09-22  47.78    2017-10-03  47.83   7     1667.6  +87.04     max_holding
2017-09-25  23.01    2017-10-04  23.66   7     1560.7  +1006.10   alpha_reversal
2017-10-04  68.23    2017-10-05  69.33   1     1439.6  +1583.39   alpha_reversal
2017-09-27  242.10   2017-10-06  244.98  7     315.1   +908.91    max_holding
2017-10-05  204.59   2017-10-06  204.35  1     397.8   -94.58     alpha_reversal
2017-09-27  47.39    2017-10-06  47.33   7     2519.0  -156.24    max_holding
2017-09-28  23.00    2017-10-06  22.99   6     3961.7  -33.44     alpha_reversal
2017-10-05  31.66    2017-10-06  31.63   1     3038.7  -96.17     alpha_reversal
2017-10-04  48.30    2017-10-09  49.52   3     1677.2  +2059.73   alpha_reversal
2017-10-03  36.03    2017-10-10  36.33   5     2102.1  +619.85    alpha_reversal
2017-10-09  23.46    2017-10-10  24.48   1     3826.8  +3919.24   alpha_reversal
2017-10-10  247.46   2017-10-11  247.69  1     377.3   +89.00     alpha_reversal
2017-10-09  47.22    2017-10-11  46.77   2     2679.6  -1207.05   alpha_reversal
2017-10-02  103.90   2017-10-11  107.03  7     1033.2  +3229.11   alpha_reversal
2017-10-04  221.80   2017-10-11  223.21  5     1211.2  +1703.21   alpha_reversal
2017-10-05  48.88    2017-10-12  49.85   5     1857.7  +1793.20   alpha_reversal
2017-10-10  23.72    2017-10-13  23.69   3     1738.7  -43.53     alpha_reversal
2017-10-13  36.62    2017-10-16  37.25   1     2670.1  +1700.37   alpha_reversal
2017-10-11  201.55   2017-10-17  196.11  4     446.3   -2429.05   stop_loss
2017-10-13  223.37   2017-10-17  223.60  2     1427.8  +331.07    alpha_reversal
2017-10-09  77.15    2017-10-18  78.34   7     1258.6  +1492.64   max_holding
2017-10-16  23.39    2017-10-18  23.96   2     2036.5  +1180.46   alpha_reversal
2017-10-16  246.34   2017-10-19  245.42  3     422.9   -388.59    alpha_reversal
2017-10-18  201.24   2017-10-19  199.35  1     379.1   -718.59    alpha_reversal
2017-10-12  46.85    2017-10-20  46.84   6     2982.4  -30.33     alpha_reversal
2017-10-20  251.08   2017-10-23  248.53  1     387.6   -989.50    alpha_reversal
2017-10-20  49.87    2017-10-23  48.85   1     2137.6  -2175.76   stop_loss
2017-10-20  23.02    2017-10-25  21.71   3     2051.2  -2679.58   stop_loss
2017-10-25  112.62   2017-10-26  112.07  1     865.1   -473.40    alpha_reversal
2017-10-24  46.32    2017-10-26  45.45   2     2646.5  -2295.88   stop_loss
2017-10-20  36.45    2017-10-30  38.85   6     2544.0  +6113.93   alpha_reversal
2017-10-19  49.36    2017-10-30  55.51   7     1851.9  +11407.10  alpha_reversal
2017-10-24  49.04    2017-10-30  51.21   4     1994.3  +4314.82   alpha_reversal
2017-10-27  42.75    2017-10-30  40.11   1     1822.1  -4793.90   stop_loss
2017-10-26  223.96   2017-10-30  224.72  2     1272.4  +973.72    alpha_reversal
2017-10-31  80.51    2017-11-02  81.21   2     1355.9  +953.14    alpha_reversal
2017-10-24  203.58   2017-11-02  205.07  7     369.6   +551.12    max_holding
2017-10-26  21.76    2017-11-02  19.94   5     1952.0  -3542.56   stop_loss
2017-10-26  245.88   2017-11-03  247.99  6     317.0   +666.86    alpha_reversal
2017-11-03  110.82   2017-11-06  110.45  1     962.8   -350.11    alpha_reversal
2017-11-06  80.65    2017-11-07  78.94   1     1382.4  -2354.88   stop_loss
2017-11-07  52.21    2017-11-08  52.45   1     1687.8  +405.50    alpha_reversal
2017-11-07  110.57   2017-11-08  111.69  1     1010.4  +1126.10   alpha_reversal
2017-10-31  40.43    2017-11-09  41.21   7     1642.5  +1282.46   alpha_reversal
2017-11-09  113.42   2017-11-13  114.31  2     884.4   +780.83    alpha_reversal
2017-11-13  250.21   2017-11-14  249.33  1     359.8   -316.19    alpha_reversal
2017-11-06  202.46   2017-11-14  197.06  6     415.1   -2238.95   stop_loss
2017-11-10  110.41   2017-11-14  110.24  2     998.5   -165.37    alpha_reversal
2017-11-03  20.42    2017-11-14  20.57   7     1829.0  +280.76    alpha_reversal
2017-11-10  226.12   2017-11-14  225.58  2     1230.9  -666.03    alpha_reversal
2017-11-09  34.28    2017-11-14  34.43   3     2995.8  +439.42    alpha_reversal
2017-11-15  250.63   2017-11-16  251.18  1     362.9   +199.50    alpha_reversal
2017-11-08  78.13    2017-11-16  78.72   6     1250.3  +731.98    alpha_reversal
2017-11-16  29.02    2017-11-17  28.36   1     2518.5  -1648.73   alpha_reversal
2017-11-15  34.20    2017-11-17  34.60   2     3099.3  +1245.04   alpha_reversal
2017-11-16  40.49    2017-11-20  39.67   2     2014.1  -1661.72   alpha_reversal
2017-11-16  226.58   2017-11-21  227.56  3     1067.2  +1037.91   alpha_reversal
2017-11-13  51.66    2017-11-22  52.14   7     1793.7  +860.47    max_holding
2017-11-14  40.11    2017-11-24  40.92   7     2215.8  +1792.14   alpha_reversal
2017-11-21  39.83    2017-11-24  39.85   2     1958.7  +36.92     alpha_reversal
2017-11-20  252.32   2017-11-27  252.97  4     393.2   +256.63    alpha_reversal
2017-11-24  228.11   2017-11-27  227.77  1     1206.8  -412.45    alpha_reversal
2017-11-24  35.39    2017-11-27  33.79   1     3135.7  -5010.83   stop_loss
2017-11-22  76.30    2017-11-28  77.85   3     1462.1  +2262.07   alpha_reversal
2017-11-21  57.00    2017-11-28  59.65   4     1577.9  +4177.01   alpha_reversal
2017-11-17  109.17   2017-11-28  111.34  6     1100.0  +2381.49   alpha_reversal
2017-11-21  21.20    2017-11-28  21.16   4     1937.7  -74.62     alpha_reversal
2017-11-22  78.93    2017-11-29  82.92   4     1337.5  +5336.77   alpha_reversal
2017-11-17  197.91   2017-11-29  201.11  7     409.2   +1312.28   alpha_reversal
2017-11-28  33.70    2017-11-29  32.41   1     2753.3  -3534.61   stop_loss
2017-11-27  40.04    2017-11-30  40.53   3     2233.2  +1089.60   alpha_reversal
2017-11-30  77.28    2017-12-01  77.28   1     1475.7  +7.81      alpha_reversal
2017-11-28  255.52   2017-12-01  258.49  3     434.9   +1293.27   alpha_reversal
2017-11-27  114.97   2017-12-01  118.48  4     920.9   +3232.55   alpha_reversal
2017-11-29  51.47    2017-12-04  50.15   3     1784.9  -2348.67   stop_loss
2017-11-30  110.90   2017-12-04  110.53  2     1144.5  -418.10    alpha_reversal
2017-11-27  28.14    2017-12-04  28.23   5     2694.0  +229.96    alpha_reversal
2017-12-04  265.04   2017-12-05  262.46  1     347.2   -895.64    alpha_reversal
2017-12-05  74.91    2017-12-06  75.92   1     1199.8  +1219.71   alpha_reversal
2017-12-05  50.59    2017-12-06  51.19   1     1602.0  +960.79    alpha_reversal
2017-12-01  41.01    2017-12-06  39.85   3     2379.6  -2749.62   stop_loss
2017-11-29  58.09    2017-12-07  57.96   6     1362.5  -179.88    alpha_reversal
2017-11-30  20.60    2017-12-07  20.74   5     2202.6  +305.42    alpha_reversal
2017-11-29  39.67    2017-12-08  39.61   7     2235.8  -146.15    max_holding
2017-12-07  40.20    2017-12-08  40.75   1     2094.9  +1144.66   alpha_reversal
2017-11-30  32.60    2017-12-08  32.08   6     2528.5  -1330.09   alpha_reversal
2017-12-08  21.02    2017-12-11  21.92   1     2380.1  +2135.44   alpha_reversal
2017-12-08  77.27    2017-12-13  78.28   3     1150.0  +1166.27   alpha_reversal
2017-12-12  276.45   2017-12-13  277.98  1     292.0   +447.76    alpha_reversal
2017-12-12  214.93   2017-12-13  212.95  1     384.3   -761.48    alpha_reversal
2017-12-04  231.42   2017-12-13  233.47  7     913.5   +1875.42   max_holding
2017-12-12  120.20   2017-12-14  122.47  2     801.6   +1823.02   alpha_reversal
2017-12-11  32.41    2017-12-15  32.49   4     2606.5  +194.26    alpha_reversal
2017-12-13  40.33    2017-12-18  41.26   3     2320.5  +2158.49   alpha_reversal
2017-12-12  85.50    2017-12-18  85.51   4     1012.0  +2.50      alpha_reversal
2017-12-07  28.34    2017-12-18  28.64   7     3229.1  +966.53    max_holding
2017-12-13  58.24    2017-12-19  59.34   4     1432.0  +1580.56   alpha_reversal
2017-12-19  41.62    2017-12-20  41.49   1     2434.2  -317.25    alpha_reversal
2017-12-21  40.97    2017-12-22  40.93   1     2569.4  -105.21    alpha_reversal
2017-12-20  84.94    2017-12-22  85.90   2     1103.5  +1061.98   alpha_reversal
2017-12-14  232.75   2017-12-22  235.34  6     1067.6  +2758.08   alpha_reversal
2017-12-19  78.80    2017-12-26  78.33   4     1210.5  -572.70    alpha_reversal
2017-12-22  281.37   2017-12-26  281.34  1     342.1   -11.49     alpha_reversal
2017-12-21  28.71    2017-12-26  29.01   2     3689.9  +1081.44   alpha_reversal
2017-12-19  32.39    2017-12-26  32.03   4     3359.6  -1186.36   alpha_reversal
2017-12-19  22.08    2017-12-27  20.77   5     2349.0  -3097.77   stop_loss
2017-12-27  78.69    2017-12-28  78.62   1     1400.4  -97.32     alpha_reversal
2017-12-27  85.80    2017-12-28  86.17   1     1167.9  +432.00    alpha_reversal
2017-12-19  112.85   2017-12-28  111.77  6     1116.1  -1208.63   alpha_reversal
2017-12-27  235.40   2017-12-28  235.65  1     1174.2  +291.85    alpha_reversal
2017-12-27  32.17    2017-12-28  32.68   1     3501.6  +1788.07   alpha_reversal
2017-12-27  39.94    2017-12-29  39.58   2     2614.6  -942.04    alpha_reversal
2017-12-20  212.84   2017-12-29  212.28  6     393.7   -221.56    alpha_reversal
2017-12-21  41.90    2017-12-29  41.61   5     2520.0  -720.45    alpha_reversal
2017-12-20  58.91    2018-01-02  59.42   7     1621.6  +827.58    alpha_reversal
2017-12-22  53.03    2018-01-04  54.31   7     2047.5  +2621.47   max_holding
2018-01-02  283.03   2018-01-08  295.42  4     403.3   +4999.16   alpha_reversal
2017-12-28  21.03    2018-01-09  22.23   7     2567.4  +3081.79   alpha_reversal
2018-01-03  131.81   2018-01-09  139.34  4     802.2   +6039.56   alpha_reversal
2017-12-29  111.21   2018-01-10  114.48  7     1275.6  +4168.93   alpha_reversal
2018-01-02  86.38    2018-01-11  89.07   7     1229.3  +3301.31   alpha_reversal
2018-01-10  22.33    2018-01-11  22.52   1     2548.2  +477.99    alpha_reversal
2018-01-03  40.32    2018-01-12  41.41   7     2639.4  +2893.44   max_holding
2018-01-03  41.56    2018-01-12  43.38   7     2948.1  +5371.14   max_holding
2018-01-03  29.12    2018-01-12  29.51   7     4154.0  +1604.62   max_holding
2018-01-12  90.63    2018-01-16  90.22   1     1393.5  -574.20    alpha_reversal
2018-01-04  214.22   2018-01-16  215.36  7     424.1   +485.22    max_holding
2018-01-11  33.78    2018-01-17  35.47   3     3117.0  +5278.16   alpha_reversal
2018-01-17  41.93    2018-01-18  41.92   1     3157.3  -14.16     alpha_reversal
2018-01-17  334.68   2018-01-18  324.01  1     259.9   -2772.58   stop_loss
2018-01-11  243.15   2018-01-18  245.57  4     1227.2  +2962.28   alpha_reversal
2018-01-11  55.17    2018-01-22  57.70   6     2515.6  +6358.61   alpha_reversal
2018-01-19  246.93   2018-01-22  248.69  1     1135.1  +1996.67   alpha_reversal
2018-01-18  209.33   2018-01-23  216.72  3     449.9   +3325.09   alpha_reversal
2018-01-22  41.43    2018-01-24  40.74   2     3331.4  -2303.80   alpha_reversal
2018-01-17  64.78    2018-01-24  67.84   5     1604.5  +4908.40   alpha_reversal
2018-01-17  90.89    2018-01-25  92.98   6     1349.1  +2815.31   alpha_reversal
2018-01-24  221.60   2018-01-25  224.17  1     447.6   +1150.40   alpha_reversal
2018-01-24  113.08   2018-01-25  114.82  1     1201.6  +2090.42   alpha_reversal
2018-01-17  82.76    2018-01-26  86.27   7     1515.1  +5322.08   alpha_reversal
2018-01-17  141.21   2018-01-26  140.52  7     781.3   -545.00    alpha_reversal
2018-01-25  40.06    2018-01-30  39.05   3     3103.2  -3128.64   stop_loss
2018-01-19  322.02   2018-01-30  321.67  7     250.5   -85.38     max_holding
2018-01-25  249.48   2018-01-30  247.87  3     1160.1  -1860.95   trailing_stop
2018-01-29  114.36   2018-01-31  109.88  2     1249.7  -5598.42   stop_loss
2018-01-24  45.29    2018-01-31  43.81   5     2389.0  -3517.57   trailing_stop
2018-01-29  136.88   2018-01-31  136.92  2     613.7   +19.29     alpha_reversal
2018-01-31  87.23    2018-02-01  86.45   1     1412.5  -1094.74   alpha_reversal
2018-01-31  39.19    2018-02-02  37.53   2     3068.9  -5093.77   stop_loss
2018-01-31  248.24   2018-02-02  242.32  2     983.6   -5825.04   stop_loss
2018-02-02  84.26    2018-02-05  80.71   1     1227.4  -4358.69   stop_loss
2018-02-02  71.53    2018-02-05  69.47   1     988.9   -2045.06   alpha_reversal
2018-02-02  55.53    2018-02-05  52.66   1     1861.8  -5345.85   stop_loss
2018-02-01  111.45   2018-02-05  103.68  2     1045.7  -8123.79   stop_loss
2018-01-26  22.87    2018-02-05  22.20   6     3017.2  -2023.16   trailing_stop
2018-02-01  44.31    2018-02-05  41.71   2     2333.1  -6072.55   stop_loss
2018-02-01  30.90    2018-02-05  29.28   2     3909.9  -6331.48   stop_loss
2018-02-01  136.60   2018-02-05  127.08  2     570.9   -5436.59   stop_loss
2018-01-30  36.95    2018-02-05  35.32   4     2808.8  -4562.13   stop_loss
2018-02-06  90.18    2018-02-07  90.70   1     719.9   +374.80    alpha_reversal
2018-02-06  72.18    2018-02-07  70.80   1     621.1   -853.76    alpha_reversal
2018-02-05  232.42   2018-02-07  235.48  2     538.2   +1645.70   alpha_reversal
2018-02-06  104.93   2018-02-08  100.48  2     551.7   -2457.44   stop_loss
2018-02-06  29.55    2018-02-08  29.26   2     2275.2  -652.91    alpha_reversal
2018-02-07  41.36    2018-02-09  40.57   2     1455.1  -1146.87   alpha_reversal
2018-02-08  78.05    2018-02-12  81.75   2     659.9   +2441.99   alpha_reversal
2018-02-05  36.63    2018-02-12  38.21   5     2236.3  +3517.98   alpha_reversal
2018-02-09  88.52    2018-02-12  89.79   1     617.4   +788.80    alpha_reversal
2018-02-08  50.00    2018-02-12  52.27   2     1061.8  +2412.59   alpha_reversal
2018-02-08  226.87   2018-02-12  233.43  2     385.3   +2526.20   alpha_reversal
2018-02-08  33.23    2018-02-13  34.85   3     1650.3  +2686.87   alpha_reversal
2018-02-09  29.10    2018-02-14  29.75   3     1891.1  +1234.03   alpha_reversal
2018-02-13  234.24   2018-02-14  237.17  1     368.8   +1078.29   alpha_reversal
2018-02-14  330.43   2018-02-15  341.21  1     138.5   +1493.04   alpha_reversal
2018-02-14  35.44    2018-02-15  35.79   1     1702.6  +597.80    alpha_reversal
2018-02-09  20.71    2018-02-20  22.31   6     1661.8  +2661.94   alpha_reversal
2018-02-20  85.53    2018-02-21  84.31   1     704.0   -858.13    alpha_reversal
2018-02-20  92.27    2018-02-21  92.57   1     749.3   +219.92    alpha_reversal
2018-02-12  69.35    2018-02-21  74.11   6     546.0   +2600.37   alpha_reversal
2018-02-13  52.30    2018-02-21  55.20   5     1091.4  +3167.59   alpha_reversal
2018-02-22  84.61    2018-02-23  86.67   1     762.3   +1572.22   alpha_reversal
2018-02-22  23.09    2018-02-23  23.46   1     2032.4  +749.40    alpha_reversal
2018-02-21  130.70   2018-02-23  136.61  2     386.5   +2283.41   alpha_reversal
2018-02-14  103.21   2018-02-26  105.69  7     585.7   +1451.17   max_holding
2018-02-15  41.44    2018-02-27  40.46   7     1619.7  -1588.15   max_holding
2018-02-20  36.01    2018-02-27  35.87   5     1791.3  -241.27    alpha_reversal
2018-02-20  220.94   2018-03-01  214.56  7     258.8   -1650.62   trailing_stop
2018-02-21  237.81   2018-03-01  235.50  6     407.0   -938.25    trailing_stop
2018-02-28  86.49    2018-03-02  85.74   2     816.0   -611.90    alpha_reversal
2018-02-21  26.80    2018-03-02  25.97   7     1771.6  -1472.62   max_holding
2018-02-28  35.69    2018-03-02  35.37   2     2011.6  -634.29    alpha_reversal
2018-02-28  40.13    2018-03-05  40.23   3     1901.3  +176.81    alpha_reversal
2018-03-05  92.55    2018-03-07  92.20   2     746.3   -266.94    alpha_reversal
2018-03-07  86.58    2018-03-08  87.02   1     824.5   +361.70    alpha_reversal
2018-03-07  41.14    2018-03-09  42.26   2     1842.7  +2066.06   alpha_reversal
2018-03-08  92.30    2018-03-09  94.86   1     795.3   +2035.75   alpha_reversal
2018-03-08  222.77   2018-03-09  226.25  1     291.7   +1014.72   alpha_reversal
2018-03-06  40.19    2018-03-09  40.77   3     1918.9  +1114.83   alpha_reversal
2018-03-02  123.25   2018-03-09  133.11  5     384.1   +3787.13   alpha_reversal
2018-03-08  241.37   2018-03-09  245.33  1     436.2   +1725.39   alpha_reversal
2018-03-02  330.25   2018-03-12  329.46  6     163.4   -129.02    alpha_reversal
2018-03-02  22.35    2018-03-13  22.78   7     2041.5  +868.53    alpha_reversal
2018-03-07  103.37   2018-03-14  105.89  5     753.1   +1898.99   alpha_reversal
2018-03-05  26.35    2018-03-14  25.80   7     2074.3  -1139.20   alpha_reversal
2018-03-13  93.51    2018-03-16  92.77   3     836.6   -622.71    alpha_reversal
2018-03-14  221.18   2018-03-16  223.60  2     313.9   +762.15    alpha_reversal
2018-03-14  86.57    2018-03-19  85.60   3     884.1   -858.62    alpha_reversal
2018-03-20  220.14   2018-03-21  218.80  1     333.8   -447.13    alpha_reversal
2018-03-13  243.68   2018-03-21  238.86  6     471.0   -2272.38   alpha_reversal
2018-03-13  324.51   2018-03-22  305.94  7     170.3   -3163.09   stop_loss
2018-03-19  54.58    2018-03-22  52.20   3     1230.0  -2927.49   stop_loss
2018-03-21  40.26    2018-03-23  38.73   2     2201.2  -3360.36   stop_loss
2018-03-16  78.62    2018-03-23  74.74   5     1059.6  -4113.90   stop_loss
2018-03-22  102.04   2018-03-23  100.11  1     734.0   -1414.11   alpha_reversal
2018-03-15  21.72    2018-03-23  20.09   6     2034.6  -3306.03   stop_loss
2018-03-22  82.82    2018-03-27  82.44   3     902.2   -340.76    alpha_reversal
2018-03-26  77.83    2018-03-27  74.82   1     842.2   -2540.75   stop_loss
2018-03-23  50.93    2018-03-27  49.91   2     1094.4  -1119.45   alpha_reversal
2018-03-22  233.12   2018-03-27  230.17  3     475.7   -1400.82   alpha_reversal
2018-03-26  40.61    2018-03-28  39.09   2     1747.8  -2652.32   stop_loss
2018-03-23  205.14   2018-03-29  210.45  4     290.6   +1542.90   alpha_reversal
2018-03-28  41.13    2018-03-29  40.63   1     1897.8  -955.74    alpha_reversal
2018-03-28  82.45    2018-04-02  81.57   2     606.0   -535.77    alpha_reversal
2018-03-28  71.61    2018-04-02  68.57   2     653.4   -1987.39   alpha_reversal
2018-03-28  49.87    2018-04-02  50.19   2     896.9   +286.48    alpha_reversal
2018-03-28  17.19    2018-04-02  16.82   2     1596.3  -591.16    alpha_reversal
2018-03-29  36.03    2018-04-02  34.95   1     1839.3  -1972.17   alpha_reversal
2018-03-23  86.08    2018-04-04  89.19   7     785.9   +2445.81   max_holding
2018-04-03  69.64    2018-04-04  70.49   1     559.7   +479.03    alpha_reversal
2018-04-03  17.84    2018-04-04  19.12   1     1415.6  +1805.66   alpha_reversal
2018-04-02  39.78    2018-04-04  40.68   2     1815.9  +1648.09   alpha_reversal
2018-04-04  51.09    2018-04-06  50.06   2     864.5   -890.87    alpha_reversal
2018-04-02  206.89   2018-04-06  208.87  4     252.6   +500.72    alpha_reversal
2018-04-04  85.17    2018-04-09  83.64   3     536.8   -817.39    alpha_reversal
2018-04-05  322.33   2018-04-09  308.68  2     137.3   -1873.62   alpha_reversal
2018-04-05  72.62    2018-04-10  71.78   3     571.6   -485.14    alpha_reversal
2018-04-09  50.61    2018-04-10  51.37   1     894.3   +682.13    alpha_reversal
2018-04-06  39.84    2018-04-10  42.14   2     1759.5  +4037.97   alpha_reversal
2018-04-02  227.64   2018-04-11  232.97  7     344.4   +1834.91   max_holding
2018-04-12  86.32    2018-04-13  85.77   1     604.6   -330.75    alpha_reversal
2018-04-12  40.93    2018-04-13  41.03   1     1610.1  +157.20    alpha_reversal
2018-04-12  51.47    2018-04-13  51.35   1     994.9   -112.82    alpha_reversal
2018-04-12  104.48   2018-04-13  104.53  1     700.9   +33.38     alpha_reversal
2018-04-06  34.92    2018-04-13  35.08   5     1664.2  +256.70    alpha_reversal
2018-04-16  86.86    2018-04-17  88.53   1     639.8   +1064.65   alpha_reversal
2018-04-10  320.83   2018-04-17  322.31  5     130.5   +194.20    alpha_reversal
2018-04-16  72.11    2018-04-17  75.15   1     696.9   +2120.44   alpha_reversal
2018-04-09  25.42    2018-04-18  25.77   7     2530.2  +896.25    max_holding
2018-04-18  34.51    2018-04-19  32.51   1     1810.0  -3620.21   stop_loss
2018-04-19  40.62    2018-04-20  38.91   1     1808.4  -3079.82   stop_loss
2018-04-16  51.90    2018-04-20  53.40   4     1046.2  +1564.63   alpha_reversal
2018-04-13  20.03    2018-04-20  19.34   5     1459.5  -1011.43   alpha_reversal
2018-04-19  25.89    2018-04-20  25.60   1     3134.4  -920.45    alpha_reversal
2018-04-20  324.51   2018-04-24  314.98  2     156.8   -1493.34   alpha_reversal
2018-04-17  212.14   2018-04-24  202.62  5     255.8   -2434.90   stop_loss
2018-04-17  104.57   2018-04-24  100.98  5     722.8   -2591.48   stop_loss
2018-04-16  89.10    2018-04-25  88.83   7     611.0   -163.00    alpha_reversal
2018-04-23  18.90    2018-04-25  18.70   2     1624.6  -320.80    alpha_reversal
2018-04-25  328.52   2018-04-26  328.12  1     137.7   -54.45     alpha_reversal
2018-04-25  122.86   2018-04-26  123.68  1     333.1   +274.49    alpha_reversal
2018-04-26  19.04    2018-04-30  19.58   2     1705.1  +924.18    alpha_reversal
2018-04-24  232.51   2018-04-30  233.63  4     397.5   +444.82    alpha_reversal
2018-04-23  38.84    2018-05-01  39.71   6     1671.0  +1449.66   alpha_reversal
2018-04-20  32.07    2018-05-01  32.02   7     1612.9  -78.21     max_holding
2018-04-24  50.74    2018-05-03  50.87   7     1003.7  +131.17    max_holding
2018-05-01  87.63    2018-05-04  87.69   3     601.7   +36.01     alpha_reversal
2018-04-26  200.82   2018-05-07  198.20  7     271.9   -711.05    max_holding
2018-04-30  87.95    2018-05-08  89.64   6     742.6   +1260.17   alpha_reversal
2018-04-27  122.46   2018-05-08  126.74  7     355.3   +1521.21   max_holding
2018-05-04  235.20   2018-05-08  235.76  2     428.7   +240.04    alpha_reversal
2018-05-03  25.40    2018-05-09  24.44   4     3160.3  -3028.31   trailing_stop
2018-05-07  31.65    2018-05-10  32.71   3     1973.9  +2080.45   alpha_reversal
2018-05-02  98.93    2018-05-11  101.82  7     783.1   +2266.26   max_holding
2018-05-02  42.78    2018-05-11  44.52   7     1645.9  +2863.85   max_holding
2018-05-03  316.86   2018-05-14  331.49  7     150.7   +2204.95   max_holding
2018-05-09  89.42    2018-05-15  89.68   4     682.8   +178.08    alpha_reversal
2018-05-09  202.19   2018-05-15  201.85  4     325.3   -111.95    alpha_reversal
2018-05-04  19.62    2018-05-15  18.94   7     1708.7  -1161.82   max_holding
2018-05-14  44.39    2018-05-17  44.08   3     1750.2  -556.21    alpha_reversal
2018-05-16  44.61    2018-05-17  44.06   1     1771.7  -977.67    alpha_reversal
2018-05-10  24.51    2018-05-17  25.02   5     2786.3  +1416.92   alpha_reversal
2018-05-17  331.39   2018-05-18  337.88  1     205.5   +1333.84   alpha_reversal
2018-05-09  80.44    2018-05-18  78.68   7     734.4   -1293.27   max_holding
2018-05-09  54.03    2018-05-18  53.02   7     1128.1  -1140.67   max_holding
2018-05-16  32.76    2018-05-18  31.60   2     2323.2  -2693.69   stop_loss
2018-05-18  89.27    2018-05-21  90.33   1     874.6   +925.64    alpha_reversal
2018-05-21  32.49    2018-05-23  32.61   2     2173.4  +251.20    alpha_reversal
2018-05-23  91.40    2018-05-24  90.98   1     883.1   -366.72    alpha_reversal
2018-05-21  53.78    2018-05-24  53.80   3     1339.7  +23.60     alpha_reversal
2018-05-16  19.11    2018-05-25  18.58   7     1992.0  -1050.83   max_holding
2018-05-18  43.96    2018-05-29  44.29   6     1934.3  +639.95    alpha_reversal
2018-05-18  89.85    2018-05-29  85.56   6     990.9   -4250.65   trailing_stop
2018-05-21  79.31    2018-05-29  80.60   5     993.5   +1282.20   alpha_reversal
2018-05-17  199.99   2018-05-29  189.70  7     391.9   -4030.94   stop_loss
2018-05-22  98.46    2018-05-29  96.26   4     955.6   -2100.96   alpha_reversal
2018-05-22  43.64    2018-05-30  44.55   5     1767.3  +1597.10   alpha_reversal
2018-05-18  24.79    2018-05-30  24.91   7     2898.1  +340.15    max_holding
2018-05-18  239.89   2018-05-30  240.78  7     588.5   +524.22    max_holding
2018-05-25  131.88   2018-05-31  128.42  3     480.8   -1664.76   alpha_reversal
2018-05-25  53.79    2018-06-01  56.26   4     1466.4  +3622.23   alpha_reversal
2018-06-01  93.37    2018-06-04  94.09   1     893.0   +643.93    alpha_reversal
2018-05-31  239.55   2018-06-05  242.98  3     576.7   +1980.87   alpha_reversal
2018-05-30  87.60    2018-06-08  89.74   7     824.2   +1765.17   max_holding
2018-05-30  192.35   2018-06-08  195.71  7     393.8   +1321.15   max_holding
2018-05-30  97.62    2018-06-08  100.01  7     911.7   +2182.13   alpha_reversal
2018-06-07  25.18    2018-06-08  24.98   1     3395.9  -678.78    alpha_reversal
2018-05-31  44.09    2018-06-11  45.08   7     2214.4  +2178.21   max_holding
2018-06-06  357.80   2018-06-11  356.84  3     203.5   -194.11    alpha_reversal
2018-06-08  21.19    2018-06-11  22.13   1     1974.6  +1858.13   alpha_reversal
2018-06-07  84.51    2018-06-12  84.90   3     1174.2  +455.36    alpha_reversal
2018-06-11  196.49   2018-06-12  195.07  1     479.1   -680.67    alpha_reversal
2018-06-05  129.48   2018-06-12  133.13  5     534.0   +1953.29   alpha_reversal
2018-06-12  45.37    2018-06-13  44.95   1     2524.2  -1054.56   alpha_reversal
2018-06-12  46.62    2018-06-13  46.69   1     2097.2  +136.91    alpha_reversal
2018-06-13  196.27   2018-06-14  195.93  1     473.0   -164.23    alpha_reversal
2018-06-12  98.89    2018-06-18  97.81   4     1110.7  -1202.20   alpha_reversal
2018-06-15  46.67    2018-06-18  45.85   1     2149.0  -1780.27   alpha_reversal
2018-06-08  32.05    2018-06-18  31.46   6     2594.0  -1533.82   alpha_reversal
2018-06-08  94.15    2018-06-19  93.34   7     1002.7  -808.94    max_holding
2018-06-14  45.02    2018-06-19  43.77   3     2636.8  -3294.66   stop_loss
2018-06-13  350.37   2018-06-19  328.16  4     224.8   -4994.24   stop_loss
2018-06-15  87.23    2018-06-19  86.86   2     982.3   -363.30    alpha_reversal
2018-06-11  56.61    2018-06-19  58.42   6     1506.2  +2736.11   alpha_reversal
2018-06-14  129.59   2018-06-19  121.14  3     584.0   -4933.49   stop_loss
2018-06-11  24.99    2018-06-20  24.76   7     3729.5  -855.20    alpha_reversal
2018-06-18  245.61   2018-06-21  243.31  3     786.5   -1812.04   alpha_reversal
2018-06-20  44.00    2018-06-25  42.94   3     2377.0  -2530.59   stop_loss
2018-06-22  85.83    2018-06-25  83.12   1     1133.5  -3072.32   stop_loss
2018-06-18  194.23   2018-06-25  185.77  5     489.5   -4138.45   stop_loss
2018-06-22  99.13    2018-06-25  98.83   1     1049.5  -315.47    alpha_reversal
2018-06-20  121.12   2018-06-25  115.46  3     506.0   -2863.00   stop_loss
2018-06-19  30.92    2018-06-25  31.13   4     2561.8  +521.93    alpha_reversal
2018-06-20  330.00   2018-06-27  317.28  5     193.0   -2452.60   stop_loss
2018-06-21  86.92    2018-06-27  83.38   4     965.8   -3414.76   stop_loss
2018-06-25  22.21    2018-06-28  23.32   3     1430.2  +1580.67   alpha_reversal
2018-06-26  241.21   2018-07-02  241.20  4     648.1   -6.77      alpha_reversal
2018-06-26  84.60    2018-07-03  84.66   5     928.4   +54.66     alpha_reversal
2018-06-27  55.42    2018-07-03  55.33   4     1179.0  -103.88    alpha_reversal
2018-07-02  187.40   2018-07-03  184.80  1     374.1   -973.51    alpha_reversal
2018-07-03  45.68    2018-07-05  46.33   1     1929.6  +1260.84   alpha_reversal
2018-07-03  25.03    2018-07-05  25.04   1     3279.6  +44.20     alpha_reversal
2018-06-26  114.69   2018-07-06  114.47  7     472.6   -106.10    max_holding
2018-07-05  56.63    2018-07-09  57.86   2     1090.0  +1342.47   alpha_reversal
2018-07-03  31.05    2018-07-09  32.25   3     2334.0  +2792.87   alpha_reversal
2018-07-09  25.47    2018-07-11  25.62   2     3276.4  +498.74    alpha_reversal
2018-07-02  84.95    2018-07-12  86.77   7     809.2   +1467.60   max_holding
2018-07-11  58.12    2018-07-12  59.54   1     1212.9  +1721.05   alpha_reversal
2018-07-10  334.30   2018-07-13  337.46  3     186.9   +590.06    alpha_reversal
2018-07-03  20.73    2018-07-13  21.25   7     1220.4  +626.09    max_holding
2018-07-12  31.84    2018-07-13  32.20   1     2421.6  +872.57    alpha_reversal
2018-07-11  44.33    2018-07-16  45.00   3     1999.1  +1339.27   alpha_reversal
2018-07-05  185.29   2018-07-16  194.07  7     381.8   +3354.91   max_holding
2018-07-10  46.88    2018-07-16  47.05   4     2024.9  +346.56    alpha_reversal
2018-07-13  26.00    2018-07-16  25.95   1     3491.5  -152.76    alpha_reversal
2018-07-09  119.29   2018-07-16  116.73  5     453.4   -1161.80   alpha_reversal
2018-07-16  89.89    2018-07-17  89.73   1     789.0   -122.14    alpha_reversal
2018-07-17  45.17    2018-07-18  44.88   1     2200.0  -643.83    alpha_reversal
2018-07-16  100.62   2018-07-18  103.03  2     1016.4  +2446.20   alpha_reversal
2018-07-17  26.14    2018-07-18  26.08   1     3657.3  -225.52    alpha_reversal
2018-07-18  194.10   2018-07-19  192.56  1     408.8   -631.27    alpha_reversal
2018-07-17  47.05    2018-07-19  46.99   2     2298.6  -142.66    alpha_reversal
2018-07-17  249.09   2018-07-19  248.42  2     688.8   -458.73    alpha_reversal
2018-07-19  96.72    2018-07-23  99.92   2     975.8   +3129.57   alpha_reversal
2018-07-19  26.00    2018-07-23  25.95   2     3732.0  -196.46    alpha_reversal
2018-07-23  60.08    2018-07-24  62.36   1     1329.9  +3028.21   alpha_reversal
2018-07-19  101.63   2018-07-24  104.29  3     875.4   +2324.76   alpha_reversal
2018-07-17  21.52    2018-07-24  19.82   5     1517.4  -2586.72   stop_loss
2018-07-17  117.58   2018-07-26  121.27  7     507.1   +1871.30   max_holding
2018-07-19  45.27    2018-07-30  44.76   7     2255.7  -1149.49   max_holding
2018-07-24  91.51    2018-07-30  88.92   4     887.1   -2298.64   trailing_stop
2018-07-24  26.07    2018-07-31  26.42   5     4085.8  +1430.11   alpha_reversal
2018-07-23  340.18   2018-08-01  339.35  7     235.9   -195.96    max_holding
2018-07-25  20.59    2018-08-01  20.05   5     1565.4  -856.23    alpha_reversal
2018-07-31  122.43   2018-08-01  117.83  1     473.7   -2177.18   alpha_reversal
2018-08-01  89.90    2018-08-03  91.12   2     731.8   +889.45    alpha_reversal
2018-08-02  250.79   2018-08-06  252.54  2     721.4   +1259.34   alpha_reversal
2018-08-06  92.43    2018-08-07  93.08   1     752.6   +484.49    alpha_reversal
2018-08-01  61.17    2018-08-10  62.08   7     1118.5  +1013.76   alpha_reversal
2018-08-06  198.04   2018-08-10  192.54  4     508.6   -2796.29   stop_loss
2018-08-08  105.85   2018-08-10  105.41  2     1077.1  -478.64    alpha_reversal
2018-08-10  23.71    2018-08-13  23.75   1     1235.2  +46.45     alpha_reversal
2018-08-02  117.50   2018-08-13  114.84  7     454.9   -1210.38   trailing_stop
2018-08-09  253.17   2018-08-13  250.28  2     842.4   -2432.93   stop_loss
2018-08-03  335.53   2018-08-14  327.86  7     216.4   -1659.82   max_holding
2018-08-07  35.19    2018-08-15  34.48   6     2597.9  -1841.02   trailing_stop
2018-08-13  105.09   2018-08-16  106.80  3     1136.2  +1950.83   alpha_reversal
2018-08-15  50.69    2018-08-16  51.75   1     2079.3  +2207.87   alpha_reversal
2018-08-16  334.81   2018-08-17  334.88  1     201.9   +14.41     alpha_reversal
2018-08-15  22.59    2018-08-17  20.36   2     1255.9  -2805.96   stop_loss
2018-08-16  29.41    2018-08-17  29.14   1     2655.4  -702.89    alpha_reversal
2018-08-15  112.40   2018-08-20  117.94  3     485.4   +2691.16   alpha_reversal
2018-08-10  94.07    2018-08-21  93.65   7     1025.9  -437.99    max_holding
2018-08-16  252.27   2018-08-21  254.05  3     717.2   +1269.89   alpha_reversal
2018-08-21  119.18   2018-08-22  118.76  1     491.6   -209.09    alpha_reversal
2018-08-20  20.57    2018-08-27  21.27   5     1196.4  +838.74    alpha_reversal
2018-08-17  34.25    2018-08-27  35.41   6     2564.9  +2973.49   alpha_reversal
2018-08-17  100.05   2018-08-28  102.44  7     923.6   +2207.16   max_holding
2018-08-23  93.26    2018-08-28  94.31   3     1173.3  +1234.03   alpha_reversal
2018-08-17  60.32    2018-08-28  61.75   7     1224.3  +1747.33   max_holding
2018-08-27  203.63   2018-08-28  203.24  1     457.2   -181.24    alpha_reversal
2018-08-20  93.88    2018-08-29  99.86   7     808.2   +4827.03   alpha_reversal
2018-08-28  109.12   2018-08-31  109.31  3     1086.0  +207.66    alpha_reversal
2018-08-24  28.31    2018-08-31  28.55   5     2835.8  +688.40    alpha_reversal
2018-08-23  51.90    2018-09-04  51.08   7     2100.5  -1719.41   max_holding
2018-08-23  116.46   2018-09-04  117.58  7     516.6   +576.93    max_holding
2018-08-23  253.81   2018-09-04  257.12  7     801.3   +2654.79   alpha_reversal
2018-08-24  338.10   2018-09-05  335.15  7     228.3   -673.07    max_holding
2018-08-29  20.34    2018-09-05  18.71   4     1578.3  -2584.53   stop_loss
2018-08-29  203.30   2018-09-06  197.31  5     500.2   -2996.71   stop_loss
2018-09-04  108.63   2018-09-06  111.08  2     1223.3  +2994.27   alpha_reversal
2018-09-05  51.63    2018-09-06  52.15   1     2529.2  +1314.39   alpha_reversal
2018-09-04  60.10    2018-09-07  58.37   3     1416.9  -2453.22   stop_loss
2018-08-29  94.10    2018-09-10  92.34   7     1251.9  -2201.84   max_holding
2018-09-07  38.30    2018-09-10  38.04   1     2196.0  -570.82    alpha_reversal
2018-09-10  255.86   2018-09-11  256.45  1     880.5   +517.01    alpha_reversal
2018-09-06  101.12   2018-09-12  103.78  4     980.8   +2607.08   alpha_reversal
2018-09-12  122.84   2018-09-13  123.79  1     643.7   +610.79    alpha_reversal
2018-09-10  51.70    2018-09-14  52.95   4     1667.4  +2087.03   alpha_reversal
2018-09-11  93.02    2018-09-14  92.17   3     1306.9  -1108.53   alpha_reversal
2018-09-06  97.96    2018-09-17  95.35   7     835.6   -2181.51   max_holding
2018-09-06  18.74    2018-09-17  19.65   7     1712.9  +1553.30   alpha_reversal
2018-09-17  113.70   2018-09-18  114.06  1     1177.4  +420.40    alpha_reversal
2018-09-13  28.36    2018-09-18  28.42   3     3750.3  +239.95    alpha_reversal
2018-09-17  92.54    2018-09-19  95.51   2     1322.7  +3937.76   alpha_reversal
2018-09-10  58.30    2018-09-19  58.20   7     1305.0  -127.13    max_holding
2018-09-10  195.31   2018-09-19  198.20  7     502.7   +1454.07   max_holding
2018-09-18  258.36   2018-09-21  260.23  3     897.4   +1680.56   alpha_reversal
2018-09-12  37.50    2018-09-21  37.56   7     2213.6  +143.47    alpha_reversal
2018-09-20  105.62   2018-09-24  106.53  2     939.3   +860.75    alpha_reversal
2018-09-20  115.34   2018-09-24  114.00  2     1271.5  -1704.74   alpha_reversal
2018-09-20  53.62    2018-09-24  53.57   2     2315.2  -124.08    alpha_reversal
2018-09-20  28.55    2018-09-24  28.27   2     4162.0  -1147.70   alpha_reversal
2018-09-19  51.71    2018-09-25  52.56   4     1410.2  +1201.39   alpha_reversal
2018-09-18  97.10    2018-09-25  98.68   5     723.0   +1140.30   alpha_reversal
2018-09-24  58.52    2018-09-25  59.18   1     1382.8  +901.31    alpha_reversal
2018-09-19  19.94    2018-09-25  20.06   4     1524.7  +169.75    alpha_reversal
2018-09-24  38.31    2018-09-27  38.16   3     2277.0  -339.58    alpha_reversal
2018-09-27  53.26    2018-09-28  53.40   1     1483.9  +198.29    alpha_reversal
2018-09-27  355.53   2018-09-28  359.53  1     257.4   +1030.71   alpha_reversal
2018-09-24  94.88    2018-09-28  91.63   4     1212.8  -3936.35   stop_loss
2018-09-25  195.80   2018-09-28  188.66  3     534.7   -3820.30   stop_loss
2018-10-01  20.72    2018-10-02  20.06   1     1285.2  -855.57    alpha_reversal
2018-09-25  28.35    2018-10-02  28.34   5     4280.2  -57.53     alpha_reversal
2018-09-24  259.63   2018-10-03  259.99  7     947.0   +345.01    max_holding
2018-10-01  92.26    2018-10-04  94.26   3     1123.1  +2246.95   alpha_reversal
2018-09-25  112.63   2018-10-04  113.09  7     1171.5  +543.16    max_holding
2018-09-27  129.81   2018-10-04  133.32  5     642.0   +2253.82   alpha_reversal
2018-10-03  37.51    2018-10-04  36.17   1     2196.4  -2947.26   stop_loss
2018-10-04  28.09    2018-10-05  27.79   1     4239.0  -1255.35   alpha_reversal
2018-10-04  18.80    2018-10-08  16.70   2     1324.8  -2785.37   stop_loss
2018-10-04  53.95    2018-10-09  54.75   3     2214.7  +1757.86   alpha_reversal
2018-10-04  95.52    2018-10-10  87.72   4     725.0   -5654.70   stop_loss
2018-10-09  56.82    2018-10-10  54.13   1     1253.9  -3365.65   stop_loss
2018-10-01  189.77   2018-10-10  180.79  7     516.0   -4630.03   stop_loss
2018-10-05  256.77   2018-10-10  248.03  3     803.8   -7026.05   stop_loss
2018-10-09  34.90    2018-10-10  33.55   1     1918.8  -2602.46   stop_loss
2018-10-09  17.53    2018-10-11  16.81   2     1277.3  -921.97    alpha_reversal
2018-10-11  98.49    2018-10-15  99.96   2     612.8   +901.76    alpha_reversal
2018-10-11  86.01    2018-10-15  88.00   2     521.4   +1038.85   alpha_reversal
2018-10-12  180.11   2018-10-15  181.07  1     375.5   +358.86    alpha_reversal
2018-10-11  242.81   2018-10-15  244.55  2     489.4   +853.99    alpha_reversal
2018-10-12  108.75   2018-10-17  113.18  3     812.6   +3598.30   alpha_reversal
2018-10-11  33.23    2018-10-17  33.73   4     1601.1  +806.73    alpha_reversal
2018-10-17  91.63    2018-10-18  88.49   1     504.3   -1583.90   alpha_reversal
2018-10-11  120.02   2018-10-18  115.50  5     440.8   -1991.00   trailing_stop
2018-10-16  53.95    2018-10-19  54.76   3     1417.1  +1135.50   alpha_reversal
2018-10-11  346.55   2018-10-22  344.14  7     166.1   -399.43    max_holding
2018-10-19  28.97    2018-10-22  28.93   1     2957.3  -94.43     alpha_reversal
2018-10-12  87.54    2018-10-23  86.07   7     727.7   -1075.33   max_holding
2018-10-19  112.52   2018-10-23  101.84  2     426.6   -4554.30   stop_loss
2018-10-19  32.81    2018-10-24  31.10   3     1527.2  -2614.97   stop_loss
2018-10-23  338.75   2018-10-25  351.67  2     163.3   +2110.74   alpha_reversal
2018-10-25  100.72   2018-10-26  99.37   1     427.6   -575.33    alpha_reversal
2018-10-26  53.33    2018-10-29  54.08   1     1119.5  +829.93    alpha_reversal
2018-10-26  84.66    2018-10-30  87.25   2     709.2   +1842.21   alpha_reversal
2018-10-26  178.84   2018-10-30  184.49  2     288.4   +1627.60   alpha_reversal
2018-10-29  111.74   2018-10-30  114.23  1     660.4   +1641.26   alpha_reversal
2018-10-29  29.76    2018-10-30  30.51   1     2626.0  +1971.12   alpha_reversal
2018-10-30  338.61   2018-10-31  343.06  1     119.1   +529.54    alpha_reversal
2018-10-25  31.91    2018-10-31  32.48   4     1423.5  +816.22    alpha_reversal
2018-10-29  51.34    2018-11-01  53.83   3     654.4   +1628.81   alpha_reversal
2018-10-29  50.25    2018-11-02  49.08   4     860.8   -1012.47   trailing_stop
2018-11-01  29.99    2018-11-02  30.18   1     2619.8  +514.51    alpha_reversal
2018-10-24  96.25    2018-11-02  107.67  7     324.1   +3700.47   max_holding
2018-11-01  351.35   2018-11-06  354.28  3     123.0   +361.21    alpha_reversal
2018-11-02  88.72    2018-11-06  89.62   2     692.4   +629.43    alpha_reversal
2018-10-29  235.40   2018-11-06  245.20  6     300.1   +2941.37   alpha_reversal
2018-10-30  96.47    2018-11-07  104.01  6     391.5   +2955.90   alpha_reversal
2018-10-29  76.98    2018-11-07  87.73   7     330.7   +3554.52   alpha_reversal
2018-11-06  33.05    2018-11-08  32.99   2     1625.7  -95.27     alpha_reversal
2018-11-08  360.46   2018-11-09  358.71  1     139.2   -243.39    alpha_reversal
2018-11-06  192.18   2018-11-09  187.32  3     319.6   -1553.52   alpha_reversal
2018-11-01  114.40   2018-11-09  117.95  6     696.7   +2475.92   alpha_reversal
2018-11-06  22.75    2018-11-09  23.36   3     1264.4  +767.45    alpha_reversal
2018-11-05  47.73    2018-11-12  46.09   5     820.9   -1348.45   trailing_stop
2018-11-02  53.16    2018-11-13  51.94   7     708.1   -863.15    max_holding
2018-11-12  89.18    2018-11-14  87.77   2     787.9   -1113.91   alpha_reversal
2018-11-09  247.80   2018-11-14  240.81  3     376.3   -2628.76   stop_loss
2018-11-09  32.49    2018-11-14  32.26   3     1819.3  -415.81    alpha_reversal
2018-11-14  52.32    2018-11-15  53.09   1     840.8   +642.41    alpha_reversal
2018-11-15  56.70    2018-11-16  57.56   1     1217.6  +1055.26   alpha_reversal
2018-11-15  110.89   2018-11-16  111.24  1     355.1   +124.78    alpha_reversal
2018-11-14  335.13   2018-11-19  311.70  3     136.6   -3200.88   stop_loss
2018-11-14  44.38    2018-11-20  42.01   4     899.5   -2136.67   trailing_stop
2018-11-15  81.01    2018-11-20  74.74   3     391.9   -2459.66   stop_loss
2018-11-13  172.69   2018-11-20  160.98  5     275.3   -3222.78   stop_loss
2018-11-16  29.13    2018-11-20  28.05   2     2504.6  -2706.22   stop_loss
2018-11-21  96.30    2018-11-23  96.17   1     436.9   -58.37     alpha_reversal
2018-11-19  30.76    2018-11-23  30.66   3     1676.4  -180.18    alpha_reversal
2018-11-23  115.54   2018-11-26  115.46  1     695.8   -57.60     alpha_reversal
2018-11-21  28.08    2018-11-26  28.34   2     2303.0  +607.60    alpha_reversal
2018-11-21  75.87    2018-11-27  79.03   3     368.6   +1163.78   alpha_reversal
2018-11-26  57.15    2018-11-27  57.78   1     1241.1  +774.50    alpha_reversal
2018-11-26  31.44    2018-11-27  31.37   1     1684.3  -110.36    alpha_reversal
2018-11-26  99.44    2018-11-28  103.68  2     449.0   +1903.49   alpha_reversal
2018-11-23  87.30    2018-11-28  90.72   3     711.1   +2432.63   alpha_reversal
2018-11-26  238.64   2018-11-28  244.71  2     344.7   +2092.96   alpha_reversal
2018-11-20  308.86   2018-11-29  332.70  6     116.7   +2781.98   alpha_reversal
2018-11-23  21.73    2018-11-29  22.73   4     1409.0  +1409.59   alpha_reversal
2018-11-21  42.00    2018-12-03  43.87   7     848.1   +1582.93   max_holding
2018-11-21  162.20   2018-12-03  161.88  7     265.6   -86.38     max_holding
2018-11-29  119.24   2018-12-03  119.41  2     714.7   +124.94    alpha_reversal
2018-11-30  23.38    2018-12-03  23.89   1     1595.4  +814.24    alpha_reversal
2018-11-28  112.04   2018-12-03  118.93  3     372.4   +2565.45   alpha_reversal
2018-11-29  90.09    2018-12-04  87.69   3     774.7   -1862.68   trailing_stop
2018-11-27  28.34    2018-12-06  28.38   6     2511.1  +113.74    alpha_reversal
2018-11-28  32.44    2018-12-07  31.13   6     1781.5  -2320.71   trailing_stop
2018-12-07  314.23   2018-12-10  316.95  1     111.5   +303.80    alpha_reversal
2018-12-06  86.10    2018-12-11  82.08   3     703.8   -2834.63   stop_loss
2018-12-10  24.36    2018-12-11  24.44   1     1510.1  +125.24    alpha_reversal
2018-12-06  240.73   2018-12-11  235.40  3     340.7   -1815.57   alpha_reversal
2018-12-04  119.38   2018-12-12  120.14  5     737.6   +556.58    alpha_reversal
2018-12-12  82.69    2018-12-13  82.69   1     622.4   -0.54      alpha_reversal
2018-12-12  40.18    2018-12-14  39.28   2     1015.1  -913.02    alpha_reversal
2018-12-04  155.85   2018-12-14  145.94  7     300.0   -2971.38   stop_loss
2018-12-13  25.13    2018-12-14  24.37   1     1589.2  -1213.25   alpha_reversal
2018-12-12  107.42   2018-12-14  108.51  2     333.6   +363.92    alpha_reversal
2018-12-14  52.18    2018-12-17  50.84   1     936.7   -1258.79   alpha_reversal
2018-12-12  317.60   2018-12-18  318.62  4     110.5   +111.98    alpha_reversal
2018-12-18  97.11    2018-12-19  96.75   1     459.2   -164.54    alpha_reversal
2018-12-14  79.64    2018-12-20  73.00   4     410.0   -2718.43   stop_loss
2018-12-12  27.91    2018-12-20  26.14   6     2559.5  -4540.17   stop_loss
2018-12-18  227.56   2018-12-20  220.29  2     303.0   -2205.28   stop_loss
2018-12-18  80.66    2018-12-21  77.01   3     651.1   -2379.14   stop_loss
2018-12-20  94.81    2018-12-24  87.83   2     420.3   -2934.02   stop_loss
2018-12-17  142.07   2018-12-24  132.07  5     297.0   -2967.00   stop_loss
2018-12-17  105.58   2018-12-24  100.33  5     452.2   -2374.60   stop_loss
2018-12-19  22.21    2018-12-24  19.68   3     1440.4  -3638.97   stop_loss
2018-12-19  56.28    2018-12-24  54.23   3     1065.4  -2187.22   stop_loss
2018-12-24  75.42    2018-12-26  78.47   1     569.9   +1737.31   alpha_reversal
2018-12-21  68.91    2018-12-26  73.51   2     353.4   +1625.99   alpha_reversal
2018-12-21  215.99   2018-12-26  220.68  2     251.0   +1178.76   alpha_reversal
2018-12-26  106.90   2018-12-27  108.42  1     286.8   +438.29    alpha_reversal
2018-12-21  35.81    2018-12-31  37.44   5     934.1   +1520.91   alpha_reversal
2018-12-28  73.94    2018-12-31  75.06   1     322.3   +361.94    alpha_reversal
2018-12-26  21.75    2018-12-31  22.18   3     1129.2  +480.34    alpha_reversal
2018-12-21  26.12    2018-12-31  27.90   5     2165.6  +3847.64   alpha_reversal
2018-12-27  308.32   2019-01-02  314.49  3     96.2    +593.83    alpha_reversal
2018-12-31  108.87   2019-01-02  108.18  1     311.0   -217.49    alpha_reversal
2019-01-02  52.33    2019-01-03  50.83   1     726.5   -1089.82   alpha_reversal
2018-12-31  141.25   2019-01-03  143.19  2     253.8   +491.67    alpha_reversal
2019-01-02  31.17    2019-01-03  29.29   1     1601.9  -2999.77   stop_loss
2019-01-04  29.84    2019-01-07  30.03   1     1460.7  +280.19    alpha_reversal
2018-12-31  94.86    2019-01-09  97.29   6     357.6   +866.94    alpha_reversal
2019-01-09  149.22   2019-01-10  148.67  1     268.7   -146.76    alpha_reversal
2018-12-31  105.50   2019-01-10  105.94  7     414.3   +179.61    max_holding
2019-01-02  20.69    2019-01-10  22.99   6     1072.6  +2468.49   alpha_reversal
2019-01-04  35.23    2019-01-11  36.15   5     828.3   +763.17    alpha_reversal
2019-01-03  104.11   2019-01-11  113.07  6     306.6   +2747.38   alpha_reversal
2019-01-03  56.48    2019-01-14  55.92   7     917.5   -520.29    max_holding
2019-01-09  30.39    2019-01-14  30.49   3     1567.1  +152.80    alpha_reversal
2019-01-11  149.61   2019-01-15  151.98  2     287.8   +681.48    alpha_reversal
2019-01-10  28.47    2019-01-15  28.82   3     2208.7  +790.42    alpha_reversal
2019-01-14  112.93   2019-01-15  111.86  1     344.1   -365.78    alpha_reversal
2019-01-14  35.64    2019-01-16  36.78   2     1044.9  +1188.00   alpha_reversal
2019-01-07  53.38    2019-01-16  54.00   7     704.4   +436.92    max_holding
2019-01-15  22.97    2019-01-16  23.06   1     1234.7  +104.93    alpha_reversal
2019-01-15  233.62   2019-01-16  233.95  1     298.2   +98.77     alpha_reversal
2019-01-11  96.01    2019-01-18  100.50  5     427.9   +1919.22   alpha_reversal
2019-01-09  82.85    2019-01-18  86.22   7     582.2   +1962.73   alpha_reversal
2019-01-11  82.07    2019-01-18  84.77   5     405.2   +1093.34   alpha_reversal
2019-01-17  349.10   2019-01-23  348.29  3     133.5   -108.86    alpha_reversal
2019-01-24  84.78    2019-01-25  85.23   1     724.4   +326.77    alpha_reversal
2019-01-25  54.65    2019-01-28  53.52   1     1012.0  -1141.26   alpha_reversal
2019-01-16  56.92    2019-01-28  55.57   7     1060.6  -1426.35   max_holding
2019-01-16  112.81   2019-01-28  107.14  7     367.7   -2084.33   trailing_stop
2019-01-25  100.09   2019-01-29  96.05   2     563.6   -2280.89   stop_loss
2019-01-17  105.54   2019-01-29  106.44  7     589.2   +534.56    alpha_reversal
2019-01-28  29.10    2019-01-29  28.96   1     2664.5  -356.76    alpha_reversal
2019-01-25  354.07   2019-01-30  376.56  3     145.8   +3279.32   alpha_reversal
2019-01-30  99.36    2019-01-31  97.44   1     533.4   -1023.52   alpha_reversal
2019-01-25  37.48    2019-01-31  39.51   4     1272.2  +2573.57   alpha_reversal
2019-01-22  19.94    2019-01-31  20.46   7     1220.5  +634.42    alpha_reversal
2019-01-29  79.73    2019-02-01  81.27   3     508.1   +780.99    alpha_reversal
2019-01-30  28.42    2019-02-01  28.11   2     2674.2  -828.77    alpha_reversal
2019-01-31  56.78    2019-02-04  58.58   2     1202.7  +2168.36   alpha_reversal
2019-02-01  85.72    2019-02-05  85.56   2     826.1   -132.08    alpha_reversal
2019-01-30  54.48    2019-02-05  57.09   4     1015.4  +2656.50   alpha_reversal
2019-01-30  240.11   2019-02-05  244.82  4     357.7   +1684.27   alpha_reversal
2019-02-04  32.38    2019-02-06  33.03   2     1874.9  +1218.01   alpha_reversal
2019-02-05  108.63   2019-02-07  107.85  2     707.6   -556.47    alpha_reversal
2019-02-04  28.41    2019-02-07  28.97   3     2595.7  +1449.90   alpha_reversal
2019-01-29  109.11   2019-02-07  110.94  7     323.9   +592.57    max_holding
2019-02-01  95.99    2019-02-11  98.20   6     550.6   +1216.00   alpha_reversal
2019-02-06  85.61    2019-02-11  83.16   3     925.9   -2262.25   stop_loss
2019-02-08  59.14    2019-02-11  58.46   1     1272.7  -860.89    alpha_reversal
2019-02-04  81.71    2019-02-12  81.86   6     484.4   +74.21     alpha_reversal
2019-02-01  166.19   2019-02-12  164.29  7     315.9   -599.59    max_holding
2019-02-11  28.84    2019-02-12  29.04   1     2950.2  +595.28    alpha_reversal
2019-02-08  110.89   2019-02-12  114.29  2     431.1   +1463.70   alpha_reversal
2019-02-07  242.41   2019-02-12  245.71  3     440.1   +1455.75   alpha_reversal
2019-02-11  32.68    2019-02-12  32.94   1     2127.1  +565.18    alpha_reversal
2019-02-12  40.78    2019-02-13  40.57   1     1489.3  -312.76    alpha_reversal
2019-02-12  99.83    2019-02-15  100.97  3     655.6   +748.20    alpha_reversal
2019-02-14  40.76    2019-02-15  40.63   1     1612.6  -211.78    alpha_reversal
2019-02-11  54.68    2019-02-15  55.49   4     1135.5  +923.42    alpha_reversal
2019-02-12  84.67    2019-02-19  86.71   4     918.9   +1876.75   alpha_reversal
2019-02-19  40.79    2019-02-20  41.01   1     1739.9  +385.31    alpha_reversal
2019-02-19  55.89    2019-02-20  55.54   1     1250.5  -436.78    alpha_reversal
2019-02-15  167.85   2019-02-20  167.76  2     365.0   -30.40     alpha_reversal
2019-02-08  108.24   2019-02-20  111.36  7     803.4   +2504.81   max_holding
2019-02-11  20.87    2019-02-21  19.41   7     1580.1  -2308.28   stop_loss
2019-02-15  60.89    2019-02-21  60.84   3     1345.7  -61.37     alpha_reversal
2019-02-20  100.50   2019-02-22  103.98  2     780.0   +2713.74   alpha_reversal
2019-02-22  41.27    2019-02-25  41.53   1     1930.7  +500.24    alpha_reversal
2019-02-14  400.42   2019-02-25  416.66  6     169.4   +2750.80   alpha_reversal
2019-02-14  32.48    2019-02-25  33.74   6     2326.8  +2939.61   alpha_reversal
2019-02-15  80.44    2019-02-26  81.78   6     635.0   +851.79    alpha_reversal
2019-02-25  121.94   2019-02-26  118.86  1     522.0   -1606.07   alpha_reversal
2019-02-26  61.60    2019-02-27  61.44   1     1596.6  -244.32    alpha_reversal
2019-02-21  166.04   2019-02-28  166.83  5     397.5   +314.92    alpha_reversal
2019-02-22  19.66    2019-02-28  21.31   4     2071.2  +3433.02   alpha_reversal
2019-02-21  54.79    2019-03-01  56.93   6     1285.8  +2753.62   alpha_reversal
2019-02-28  112.45   2019-03-01  113.74  1     1037.7  +1342.21   alpha_reversal
2019-02-21  29.79    2019-03-04  29.30   7     2689.9  -1320.65   max_holding
2019-03-01  33.61    2019-03-04  33.38   1     3036.9  -723.37    alpha_reversal
2019-02-22  86.65    2019-03-05  85.83   7     1030.5  -845.27    max_holding
2019-02-22  250.48   2019-03-05  250.12  7     565.6   -202.48    max_holding
2019-02-28  62.02    2019-03-06  61.55   4     1648.7  -768.18    alpha_reversal
2019-03-06  104.82   2019-03-07  103.44  1     925.5   -1276.46   alpha_reversal
2019-03-08  412.85   2019-03-11  390.45  1     180.5   -4043.35   stop_loss
2019-03-07  84.97    2019-03-11  86.02   2     1121.3  +1180.41   alpha_reversal
2019-03-08  165.76   2019-03-11  166.21  1     422.0   +191.34    alpha_reversal
2019-03-04  19.03    2019-03-13  19.25   7     1873.4  +413.76    max_holding
2019-03-13  86.14    2019-03-18  88.37   3     1195.6  +2656.77   alpha_reversal
2019-03-12  83.70    2019-03-18  87.06   4     856.7   +2884.73   alpha_reversal
2019-03-11  61.70    2019-03-18  62.42   5     1630.2  +1181.76   alpha_reversal
2019-03-07  32.97    2019-03-18  33.97   7     3059.0  +3080.75   alpha_reversal
2019-03-13  29.69    2019-03-19  30.06   4     3593.2  +1353.90   alpha_reversal
2019-03-08  246.28   2019-03-19  254.27  7     647.3   +5169.64   alpha_reversal
2019-03-12  114.24   2019-03-20  114.58  6     606.8   +207.76    alpha_reversal
2019-03-20  44.90    2019-03-21  46.51   1     2182.1  +3506.85   alpha_reversal
2019-03-12  366.80   2019-03-21  363.79  7     115.0   -346.47    max_holding
2019-03-18  58.97    2019-03-21  61.27   3     1589.5  +3654.77   alpha_reversal
2019-03-19  62.91    2019-03-21  63.65   2     1867.5  +1372.83   alpha_reversal
2019-03-20  34.01    2019-03-21  34.99   1     3274.2  +3210.32   alpha_reversal
2019-03-21  115.60   2019-03-22  111.79  1     753.4   -2871.30   stop_loss
2019-03-14  113.58   2019-03-25  112.31  7     1134.5  -1443.95   alpha_reversal
2019-03-22  353.86   2019-03-26  361.52  2     143.5   +1099.00   alpha_reversal
2019-03-25  113.29   2019-03-26  113.57  1     712.2   +201.56    alpha_reversal
2019-03-26  114.04   2019-03-27  114.03  1     1236.9  -8.77      alpha_reversal
2019-03-18  17.97    2019-03-27  18.31   7     2034.6  +687.40    alpha_reversal
2019-03-26  44.57    2019-03-28  44.99   2     1694.4  +704.07    alpha_reversal
2019-03-27  365.63   2019-03-28  365.49  1     148.7   -20.96     alpha_reversal
2019-03-20  164.98   2019-03-29  162.83  7     462.7   -990.65    max_holding
2019-03-28  114.15   2019-03-29  116.72  1     757.0   +1948.33   alpha_reversal
2019-03-25  251.49   2019-03-29  254.34  4     652.2   +1856.16   alpha_reversal
2019-03-22  82.32    2019-04-02  86.68   7     1173.5  +5107.98   max_holding
2019-03-27  58.45    2019-04-03  60.01   5     1580.3  +2476.81   alpha_reversal
2019-04-03  19.46    2019-04-04  17.84   1     2780.6  -4506.38   stop_loss
2019-03-26  29.63    2019-04-04  29.54   7     4453.8  -413.50    max_holding
2019-03-28  34.42    2019-04-04  35.84   5     3181.2  +4500.98   alpha_reversal
2019-04-04  386.78   2019-04-05  382.56  1     172.7   -729.23    alpha_reversal
2019-04-01  63.98    2019-04-05  62.27   4     2100.0  -3598.71   stop_loss
2019-04-03  120.09   2019-04-05  120.91  2     758.6   +627.86    alpha_reversal
2019-04-05  112.45   2019-04-08  112.38  1     1100.8  -82.50     alpha_reversal
2019-04-03  91.08    2019-04-08  92.45   3     1095.7  +1496.95   alpha_reversal
2019-04-08  87.85    2019-04-09  87.11   1     1307.1  -961.71    alpha_reversal
2019-04-08  59.95    2019-04-11  59.95   3     1942.9  +9.75      alpha_reversal
2019-04-02  113.33   2019-04-11  111.16  7     1351.8  -2931.44   stop_loss
2019-04-10  112.73   2019-04-12  113.33  2     1149.2  +688.89    alpha_reversal
2019-04-10  87.59    2019-04-12  92.38   2     1343.3  +6432.50   alpha_reversal
2019-04-08  62.18    2019-04-15  60.26   5     2099.9  -4029.82   stop_loss
2019-04-05  18.34    2019-04-16  18.21   7     2403.6  -300.32    max_holding
2019-04-12  36.34    2019-04-16  36.67   2     3885.2  +1283.11   alpha_reversal
2019-04-08  365.93   2019-04-17  368.49  7     162.5   +416.31    max_holding
2019-04-12  111.91   2019-04-17  113.88  3     1447.6  +2860.94   alpha_reversal
2019-04-16  59.57    2019-04-17  56.72   1     2021.5  -5766.22   stop_loss
2019-04-09  117.58   2019-04-17  122.96  6     786.5   +4236.90   alpha_reversal
2019-04-12  92.20    2019-04-18  93.04   4     1353.7  +1135.61   alpha_reversal
2019-04-17  114.22   2019-04-22  115.97  2     1279.7  +2240.24   alpha_reversal
2019-04-22  94.41    2019-04-23  96.14   1     1388.7  +2399.31   alpha_reversal
2019-04-22  113.43   2019-04-23  115.02  1     1138.0  +1807.58   alpha_reversal
2019-04-12  261.52   2019-04-23  263.70  6     870.7   +1904.86   alpha_reversal
2019-04-18  18.23    2019-04-24  17.24   3     2848.0  -2822.56   stop_loss
2019-04-23  365.44   2019-04-25  373.65  2     213.7   +1753.71   alpha_reversal
2019-04-25  121.14   2019-04-26  121.71  1     1038.9  +594.62    alpha_reversal
2019-04-16  171.36   2019-04-26  172.24  7     487.6   +429.29    max_holding
2019-04-18  56.21    2019-04-29  58.91   6     1705.8  +4602.90   alpha_reversal
2019-04-26  48.75    2019-05-02  49.86   4     2478.2  +2744.48   alpha_reversal
2019-04-30  116.20   2019-05-02  116.15  2     1203.2  -60.60     alpha_reversal
2019-05-01  173.81   2019-05-03  176.01  2     551.1   +1208.35   alpha_reversal
2019-04-25  16.52    2019-05-03  16.99   6     2858.3  +1362.20   alpha_reversal
2019-04-24  119.49   2019-05-03  120.52  7     777.6   +803.48    max_holding
2019-04-29  37.40    2019-05-03  38.12   4     3212.4  +2289.88   alpha_reversal
2019-05-03  120.90   2019-05-07  117.62  2     1023.6  -3365.69   stop_loss
2019-04-29  370.36   2019-05-07  348.69  6     219.6   -4759.30   stop_loss
2019-05-03  116.87   2019-05-07  115.07  2     1226.3  -2199.88   alpha_reversal
2019-04-26  30.60    2019-05-07  30.50   7     4375.9  -436.86    alpha_reversal
2019-05-02  262.44   2019-05-07  259.25  3     957.8   -3053.93   trailing_stop
2019-05-01  58.21    2019-05-08  58.03   5     1607.3  -295.87    alpha_reversal
2019-05-06  118.65   2019-05-08  114.36  2     741.9   -3180.77   stop_loss
2019-05-09  171.18   2019-05-10  171.37  1     527.0   +97.55     alpha_reversal
2019-05-08  117.72   2019-05-13  115.58  3     921.2   -1972.88   alpha_reversal
2019-05-10  47.23    2019-05-13  44.44   1     1594.1  -4446.98   stop_loss
2019-05-08  351.50   2019-05-13  331.20  3     223.1   -4529.96   stop_loss
2019-05-09  30.16    2019-05-13  30.24   2     4099.0  +310.71    alpha_reversal
2019-05-09  113.82   2019-05-13  108.59  2     667.1   -3487.25   stop_loss
2019-05-08  259.15   2019-05-13  252.88  3     725.3   -4543.69   stop_loss
2019-05-13  91.01    2019-05-14  91.64   1     1055.0  +666.53    alpha_reversal
2019-05-13  59.27    2019-05-14  59.40   1     1609.2  +213.34    alpha_reversal
2019-05-15  58.09    2019-05-16  58.71   1     1265.8  +786.02    alpha_reversal
2019-05-13  15.14    2019-05-17  14.06   4     2585.9  -2792.56   trailing_stop
2019-05-16  30.70    2019-05-17  30.53   1     3217.7  -536.96    alpha_reversal
2019-05-14  337.10   2019-05-20  346.33  4     188.7   +1742.15   alpha_reversal
2019-05-09  114.17   2019-05-20  113.80  7     1128.4  -416.34    alpha_reversal
2019-05-13  34.98    2019-05-20  32.96   5     2207.0  -4461.08   trailing_stop
2019-05-16  92.56    2019-05-21  92.81   3     997.9   +255.84    alpha_reversal
2019-05-20  30.76    2019-05-21  30.61   1     3158.3  -479.49    alpha_reversal
2019-05-13  91.18    2019-05-22  92.94   7     915.3   +1609.03   max_holding
2019-05-14  255.42   2019-05-22  257.18  6     528.5   +926.18    alpha_reversal
2019-05-14  45.19    2019-05-23  42.99   7     1317.6  -2897.29   stop_loss
2019-05-22  114.19   2019-05-23  114.93  1     1062.1  +778.21    alpha_reversal
2019-05-22  30.97    2019-05-23  30.83   1     3233.5  -462.26    alpha_reversal
2019-05-14  110.59   2019-05-23  105.96  7     565.0   -2613.82   stop_loss
2019-05-15  166.74   2019-05-24  163.69  7     475.4   -1450.17   max_holding
2019-05-20  13.70    2019-05-24  12.70   4     2402.1  -2390.54   alpha_reversal
2019-05-24  91.23    2019-05-28  90.15   1     1036.9  -1119.57   alpha_reversal
2019-05-21  33.54    2019-05-28  32.10   4     2000.0  -2880.38   stop_loss
2019-05-28  56.54    2019-05-29  55.51   1     1359.3  -1398.72   alpha_reversal
2019-05-28  42.69    2019-05-31  41.89   3     1353.7  -1081.47   alpha_reversal
2019-05-31  88.80    2019-06-03  84.59   1     871.3   -3664.48   stop_loss
2019-05-28  12.59    2019-06-06  13.72   7     2247.2  +2554.73   max_holding
2019-05-29  160.37   2019-06-07  161.71  7     484.4   +648.71    alpha_reversal
2019-05-29  108.82   2019-06-07  114.69  7     783.2   +4595.32   max_holding
2019-06-06  32.80    2019-06-07  33.09   1     2105.6  +613.10    alpha_reversal
2019-06-04  115.95   2019-06-10  124.71  4     652.6   +5718.79   alpha_reversal
2019-06-04  86.52    2019-06-11  93.14   5     744.9   +4929.33   alpha_reversal
2019-06-10  14.20    2019-06-11  14.47   1     2372.8  +633.55    alpha_reversal
2019-06-03  61.65    2019-06-11  63.67   6     1504.8  +3047.81   alpha_reversal
2019-06-10  165.55   2019-06-12  162.06  2     477.4   -1665.13   alpha_reversal
2019-06-04  52.32    2019-06-13  54.08   7     1174.4  +2064.45   max_holding
2019-06-12  259.92   2019-06-14  260.45  2     559.2   +292.79    alpha_reversal
2019-06-10  91.75    2019-06-17  90.73   5     1012.2  -1034.61   alpha_reversal
2019-06-14  64.01    2019-06-17  64.34   1     1736.1  +586.26    alpha_reversal
2019-06-14  33.05    2019-06-17  33.04   1     3761.2  -21.77     alpha_reversal
2019-06-13  110.33   2019-06-17  110.35  2     782.4   +15.43     alpha_reversal
2019-06-13  46.51    2019-06-18  47.49   3     1525.6  +1498.88   alpha_reversal
2019-06-17  260.80   2019-06-18  263.27  1     638.6   +1575.65   alpha_reversal
2019-06-12  341.02   2019-06-19  361.81  5     207.9   +4323.10   alpha_reversal
2019-06-18  92.06    2019-06-21  90.91   3     1114.6  -1278.41   alpha_reversal
2019-06-20  368.39   2019-06-24  367.15  2     203.9   -253.29    alpha_reversal
2019-06-18  166.28   2019-06-24  168.25  4     513.8   +1013.32   alpha_reversal
2019-06-21  14.80    2019-06-24  14.90   1     2645.1  +274.61    alpha_reversal
2019-06-20  33.43    2019-06-24  33.67   2     4125.7  +1011.06   alpha_reversal
2019-06-14  32.29    2019-06-24  35.19   6     2187.9  +6345.66   alpha_reversal
2019-06-21  117.73   2019-06-25  119.40  2     1058.4  +1759.01   alpha_reversal
2019-06-21  65.39    2019-06-25  65.84   2     1720.1  +777.73    alpha_reversal
2019-06-26  126.09   2019-06-28  125.99  2     799.5   -78.19     alpha_reversal
2019-06-26  14.63    2019-06-28  14.89   2     2942.9  +778.61    alpha_reversal
2019-06-27  64.82    2019-06-28  64.77   1     1698.2  -83.79     alpha_reversal
2019-06-25  263.33   2019-06-28  265.10  3     716.6   +1263.68   alpha_reversal
2019-06-20  47.78    2019-07-01  48.23   7     1692.9  +765.83    max_holding
2019-06-24  90.35    2019-07-01  94.43   5     1120.5  +4571.53   alpha_reversal
2019-07-01  127.74   2019-07-03  129.28  2     845.2   +1307.09   alpha_reversal
2019-07-01  96.16    2019-07-03  96.90   2     1139.5  +847.72    alpha_reversal
2019-06-25  53.96    2019-07-03  55.66   6     1668.9  +2839.07   alpha_reversal
2019-06-28  115.41   2019-07-03  117.66  3     1045.9  +2355.42   alpha_reversal
2019-06-27  33.36    2019-07-03  34.00   4     4326.3  +2762.88   alpha_reversal
2019-07-03  15.67    2019-07-05  15.53   1     3115.1  -422.40    alpha_reversal
2019-06-28  357.71   2019-07-08  344.69  5     225.4   -2932.91   stop_loss
2019-07-05  95.04    2019-07-08  94.42   1     1266.0  -776.89    alpha_reversal
2019-07-08  116.81   2019-07-09  117.05  1     1173.3  +290.33    alpha_reversal
2019-07-08  34.15    2019-07-09  34.17   1     4544.4  +64.98     alpha_reversal
2019-07-09  94.92    2019-07-10  94.55   1     1312.1  -486.71    alpha_reversal
2019-07-09  269.16   2019-07-10  270.17  1     818.9   +831.82    alpha_reversal
2019-07-11  180.24   2019-07-12  182.27  1     576.3   +1167.94   alpha_reversal
2019-07-11  116.09   2019-07-12  111.17  1     1206.6  -5942.74   stop_loss
2019-07-12  100.60   2019-07-15  101.00  1     1228.9  +489.94    alpha_reversal
2019-07-08  15.36    2019-07-15  16.89   5     3332.2  +5091.10   alpha_reversal
2019-07-03  117.63   2019-07-15  121.02  7     899.4   +3051.19   max_holding
2019-07-03  35.54    2019-07-15  36.81   7     2354.2  +3003.99   alpha_reversal
2019-07-08  47.91    2019-07-16  48.94   6     2112.8  +2163.94   alpha_reversal
2019-07-08  128.94   2019-07-17  128.16  7     971.4   -755.56    max_holding
2019-07-09  346.97   2019-07-17  362.76  6     261.0   +4118.96   alpha_reversal
2019-07-11  95.55    2019-07-17  95.36   4     1370.9  -257.07    alpha_reversal
2019-07-17  48.71    2019-07-19  48.48   2     2522.6  -581.64    alpha_reversal
2019-07-15  111.62   2019-07-19  107.87  4     981.1   -3682.90   stop_loss
2019-07-11  62.63    2019-07-19  62.87   6     1509.2  +360.14    alpha_reversal
2019-07-17  34.72    2019-07-19  34.48   2     5291.1  -1304.64   alpha_reversal
2019-07-19  17.22    2019-07-22  17.04   1     3955.9  -727.07    alpha_reversal
2019-07-18  56.92    2019-07-24  56.49   4     2364.2  -1014.52   alpha_reversal
2019-07-23  17.35    2019-07-24  17.65   1     3968.6  +1176.69   alpha_reversal
2019-07-23  120.73   2019-07-24  115.20  1     894.5   -4943.79   stop_loss
2019-07-17  99.65    2019-07-26  97.10   7     1358.6  -3460.83   stop_loss
2019-07-25  117.77   2019-07-26  116.09  1     714.5   -1201.12   alpha_reversal
2019-07-17  269.65   2019-07-26  273.25  7     982.8   +3532.02   max_holding
2019-07-25  56.36    2019-07-29  61.55   2     2227.5  +11566.55  alpha_reversal
2019-07-25  49.59    2019-07-30  49.96   3     2573.8  +956.42    alpha_reversal
2019-07-23  106.76   2019-07-30  109.33  5     1063.1  +2737.75   alpha_reversal
2019-07-23  63.43    2019-07-30  64.32   5     1678.0  +1500.83   alpha_reversal
2019-07-29  117.55   2019-07-30  116.11  1     779.7   -1119.89   alpha_reversal
2019-07-29  38.72    2019-07-30  37.95   1     2928.5  -2252.86   alpha_reversal
2019-07-30  272.35   2019-07-31  269.10  1     1026.7  -3335.51   stop_loss
2019-07-31  51.03    2019-08-01  49.88   1     2233.2  -2577.51   alpha_reversal
2019-07-25  342.06   2019-08-01  328.17  5     218.8   -3038.82   stop_loss
2019-07-31  97.14    2019-08-01  94.48   1     1486.4  -3949.32   stop_loss
2019-07-29  95.67    2019-08-01  92.72   3     1152.9  -3401.97   stop_loss
2019-07-23  33.96    2019-08-01  33.11   7     4864.4  -4155.31   stop_loss
2019-07-31  115.11   2019-08-01  110.73  1     774.8   -3391.34   stop_loss
2019-08-01  129.98   2019-08-02  128.75  1     877.9   -1071.80   alpha_reversal
2019-07-26  15.21    2019-08-02  15.61   5     2966.6  +1200.23   alpha_reversal
2019-08-02  94.57    2019-08-05  91.66   1     1209.4  -3513.76   stop_loss
2019-08-02  91.21    2019-08-05  88.21   1     1008.0  -3019.21   stop_loss
2019-08-02  33.15    2019-08-05  32.03   1     4050.3  -4523.11   stop_loss
2019-08-01  267.03   2019-08-05  256.79  2     735.6   -7530.45   stop_loss
2019-07-31  37.56    2019-08-05  35.09   3     2729.6  -6733.94   stop_loss
2019-08-06  89.44    2019-08-07  89.63   1     852.8   +161.15    alpha_reversal
2019-08-06  32.50    2019-08-07  32.75   1     3230.2  +804.40    alpha_reversal
2019-08-06  260.65   2019-08-07  260.54  1     530.1   -56.55     alpha_reversal
2019-08-09  48.33    2019-08-13  50.20   2     1440.4  +2691.54   alpha_reversal
2019-08-13  329.14   2019-08-14  316.52  1     203.9   -2573.21   stop_loss
2019-08-08  92.00    2019-08-14  87.67   4     914.2   -3953.86   stop_loss
2019-08-07  105.59   2019-08-14  100.62  5     582.8   -2896.75   stop_loss
2019-08-12  15.27    2019-08-15  14.37   3     2869.9  -2600.56   trailing_stop
2019-08-12  171.86   2019-08-16  169.90  4     402.6   -789.43    alpha_reversal
2019-08-15  34.31    2019-08-16  34.37   1     2249.5  +128.14    alpha_reversal
2019-08-09  90.42    2019-08-19  90.76   6     862.7   +290.21    alpha_reversal
2019-08-12  36.09    2019-08-19  36.78   5     2017.0  +1383.09   alpha_reversal
2019-08-15  88.10    2019-08-20  89.77   3     796.4   +1335.69   alpha_reversal
2019-08-21  336.19   2019-08-22  350.09  1     202.0   +2809.17   alpha_reversal
2019-08-21  90.11    2019-08-22  90.95   1     853.7   +715.88    alpha_reversal
2019-08-21  91.22    2019-08-22  90.19   1     890.4   -921.26    alpha_reversal
2019-08-21  171.14   2019-08-22  172.45  1     400.6   +525.32    alpha_reversal
2019-08-21  67.20    2019-08-22  66.98   1     1338.2  -286.26    alpha_reversal
2019-08-22  108.77   2019-08-23  105.73  1     1047.8  -3184.25   stop_loss
2019-08-21  102.88   2019-08-23  99.61   2     625.7   -2042.41   alpha_reversal
2019-08-23  65.68    2019-08-27  66.06   2     1317.8  +503.85    alpha_reversal
2019-08-23  126.00   2019-08-28  127.92  3     582.1   +1118.64   alpha_reversal
2019-08-26  106.69   2019-08-28  107.31  2     964.7   +605.10    alpha_reversal
2019-08-23  33.75    2019-08-28  34.29   3     2405.3  +1301.69   alpha_reversal
2019-08-23  36.08    2019-08-28  36.66   3     1952.8  +1132.81   alpha_reversal
2019-08-23  57.23    2019-08-29  59.19   4     1280.5  +2507.31   alpha_reversal
2019-08-21  14.73    2019-08-29  14.77   6     2777.8  +121.99    alpha_reversal
2019-08-28  66.87    2019-08-29  66.98   1     1363.2  +161.61    alpha_reversal
2019-08-23  257.98   2019-08-29  264.72  4     398.9   +2687.07   alpha_reversal
2019-08-29  130.47   2019-08-30  130.09  1     614.6   -230.92    alpha_reversal
2019-08-26  100.03   2019-09-03  102.21  5     630.8   +1374.86   alpha_reversal
2019-08-23  87.52    2019-09-04  89.99   7     872.7   +2147.85   alpha_reversal
2019-08-29  107.05   2019-09-04  107.46  3     891.6   +365.59    alpha_reversal
2019-09-03  67.00    2019-09-04  66.47   1     1500.0  -795.62    alpha_reversal
2019-09-03  263.31   2019-09-05  269.46  2     452.2   +2777.43   alpha_reversal
2019-09-05  15.31    2019-09-06  15.16   1     3086.7  -485.33    alpha_reversal
2019-08-28  49.42    2019-09-09  51.45   7     1314.5  +2663.14   max_holding
2019-09-04  173.12   2019-09-09  181.73  3     427.6   +3683.16   alpha_reversal
2019-09-05  66.57    2019-09-09  64.47   2     1518.7  -3186.19   stop_loss
2019-09-06  269.93   2019-09-09  269.80  1     482.8   -64.76     alpha_reversal
2019-09-09  15.46    2019-09-10  15.69   1     3349.3  +785.15    alpha_reversal
2019-09-06  358.94   2019-09-12  371.06  4     207.3   +2511.51   alpha_reversal
2019-09-03  58.03    2019-09-12  61.21   7     1403.1  +4468.18   max_holding
2019-09-04  130.00   2019-09-13  129.58  7     659.9   -278.80    max_holding
2019-09-06  34.93    2019-09-13  35.72   5     2914.8  +2292.03   alpha_reversal
2019-09-06  107.03   2019-09-16  108.03  6     1036.9  +1039.22   alpha_reversal
2019-09-13  39.42    2019-09-16  39.26   1     2504.4  -407.31    alpha_reversal
2019-09-16  52.88    2019-09-17  53.01   1     1596.7  +222.45    alpha_reversal
2019-09-17  129.78   2019-09-18  130.71  1     808.3   +757.05    alpha_reversal
2019-09-09  91.61    2019-09-18  90.83   7     1060.6  -833.34    alpha_reversal
2019-09-18  186.29   2019-09-19  184.52  1     458.7   -812.91    alpha_reversal
2019-09-10  63.16    2019-09-19  65.29   7     1262.9  +2688.71   alpha_reversal
2019-09-19  91.12    2019-09-20  89.66   1     1257.3  -1832.40   alpha_reversal
2019-09-19  16.45    2019-09-20  16.03   1     3733.0  -1548.84   alpha_reversal
2019-09-20  109.90   2019-09-23  109.87  1     1134.7  -39.46     alpha_reversal
2019-09-20  52.35    2019-09-24  52.29   2     1725.2  -111.03    alpha_reversal
2019-09-23  61.26    2019-09-24  60.39   1     1839.2  -1603.98   alpha_reversal
2019-09-20  35.62    2019-09-24  36.01   2     3629.4  +1438.47   alpha_reversal
2019-09-18  114.58   2019-09-24  110.03  4     714.2   -3250.26   stop_loss
2019-09-24  377.38   2019-09-25  381.50  1     229.1   +943.11    alpha_reversal
2019-09-23  99.55    2019-09-25  98.72   2     1029.8  -860.52    alpha_reversal
2019-09-23  89.31    2019-09-26  86.95   3     1197.6  -2827.61   stop_loss
2019-09-25  61.82    2019-09-26  61.57   1     1611.7  -391.18    alpha_reversal
2019-09-24  129.77   2019-09-27  129.97  3     752.2   +150.84    alpha_reversal
2019-09-25  15.25    2019-09-27  16.13   2     3204.6  +2818.86   alpha_reversal
2019-09-26  65.29    2019-09-27  64.47   1     1528.6  -1252.71   alpha_reversal
2019-09-20  183.43   2019-09-30  177.66  6     482.2   -2779.87   stop_loss
2019-09-27  110.67   2019-09-30  110.31  1     768.2   -272.84    alpha_reversal
2019-09-24  269.20   2019-10-01  266.54  5     634.0   -1686.03   alpha_reversal
2019-10-01  96.76    2019-10-02  94.74   1     1023.6  -2068.58   alpha_reversal
2019-09-27  86.32    2019-10-02  85.62   3     1036.7  -722.57    alpha_reversal
2019-09-30  60.59    2019-10-02  58.38   2     1644.9  -3623.34   stop_loss
2019-09-26  107.56   2019-10-02  110.08  4     1092.9  +2753.60   alpha_reversal
2019-09-30  16.07    2019-10-02  16.20   2     2955.7  +397.64    alpha_reversal
2019-10-01  106.98   2019-10-03  104.84  2     715.0   -1531.32   alpha_reversal
2019-10-02  262.09   2019-10-04  267.55  2     537.5   +2932.02   alpha_reversal
2019-10-04  15.44    2019-10-08  16.00   2     2822.9  +1577.83   alpha_reversal
2019-10-08  128.15   2019-10-09  130.45  1     715.3   +1643.19   alpha_reversal
2019-10-01  173.94   2019-10-09  168.76  6     482.1   -2497.76   alpha_reversal
2019-10-08  103.13   2019-10-09  104.31  1     740.8   +874.71    alpha_reversal
2019-10-04  96.75    2019-10-10  96.31   4     944.9   -418.07    alpha_reversal
2019-10-03  59.01    2019-10-10  59.95   5     1485.9  +1388.27   alpha_reversal
2019-10-09  43.20    2019-10-10  43.38   1     2181.5  +389.30    alpha_reversal
2019-10-10  16.32    2019-10-11  16.52   1     2925.2  +566.26    alpha_reversal
2019-10-09  265.01   2019-10-11  269.30  2     516.3   +2214.35   alpha_reversal
2019-10-10  171.52   2019-10-14  176.45  2     486.7   +2399.14   alpha_reversal
2019-10-10  107.26   2019-10-14  112.12  2     736.0   +3578.58   alpha_reversal
2019-10-11  86.64    2019-10-15  88.32   2     1150.4  +1938.95   alpha_reversal
2019-10-10  107.74   2019-10-16  112.73  4     971.6   +4846.18   alpha_reversal
2019-10-15  271.94   2019-10-16  271.23  1     552.4   -391.20    alpha_reversal
2019-10-16  88.92    2019-10-17  89.33   1     1229.8  +508.36    alpha_reversal
2019-10-15  133.72   2019-10-18  129.66  3     816.7   -3315.18   stop_loss
2019-10-16  56.35    2019-10-18  56.79   2     1768.0  +766.79    alpha_reversal
2019-10-11  370.73   2019-10-18  339.81  5     241.9   -7478.97   stop_loss
2019-10-17  272.30   2019-10-18  270.84  1     610.3   -892.81    alpha_reversal
2019-10-18  65.91    2019-10-22  63.22   2     1716.4  -4624.10   stop_loss
2019-10-17  36.49    2019-10-22  36.37   3     3965.7  -458.27    alpha_reversal
2019-10-22  17.05    2019-10-23  16.97   1     3239.0  -249.44    alpha_reversal
2019-10-21  130.76   2019-10-24  132.05  3     803.7   +1040.20   alpha_reversal
2019-10-23  88.15    2019-10-25  88.02   2     1172.0  -152.46    alpha_reversal
2019-10-24  19.99    2019-10-25  21.86   1     2378.6  +4461.66   alpha_reversal
2019-10-17  44.15    2019-10-25  45.33   6     2151.6  +2537.56   alpha_reversal
2019-10-22  61.58    2019-10-29  62.49   5     1822.7  +1645.91   alpha_reversal
2019-10-23  36.34    2019-10-29  35.63   4     4023.0  -2838.25   stop_loss
2019-10-21  327.36   2019-10-30  341.85  7     199.8   +2894.67   alpha_reversal
2019-10-21  106.87   2019-10-30  110.78  7     763.2   +2986.46   max_holding
2019-10-23  64.50    2019-10-30  67.04   5     1455.8  +3698.35   alpha_reversal
2019-10-30  58.49    2019-11-01  61.45   2     1829.6  +5413.12   alpha_reversal
2019-11-01  46.24    2019-11-04  47.42   1     2706.2  +3209.87   alpha_reversal
2019-10-28  88.90    2019-11-06  89.74   7     1059.3  +895.27    max_holding
2019-11-04  21.18    2019-11-07  22.36   3     2367.2  +2800.17   alpha_reversal
2019-10-31  336.11   2019-11-08  348.75  6     216.1   +2731.43   alpha_reversal
2019-10-30  62.55    2019-11-08  64.88   7     1752.8  +4086.69   alpha_reversal
2019-10-31  183.12   2019-11-08  191.10  6     512.6   +4094.18   alpha_reversal
2019-10-30  35.96    2019-11-08  36.33   7     4301.6  +1598.75   max_holding
2019-11-06  136.08   2019-11-11  137.87  3     927.2   +1667.41   alpha_reversal
2019-10-31  105.45   2019-11-11  109.56  7     1118.6  +4598.53   max_holding
2019-11-11  364.97   2019-11-12  360.55  1     220.5   -974.21    alpha_reversal
2019-11-04  108.72   2019-11-13  109.47  7     889.5   +674.88    max_holding
2019-11-06  64.80    2019-11-14  65.74   6     1434.3  +1356.94   alpha_reversal
2019-11-08  89.34    2019-11-15  86.93   5     1451.1  -3493.81   stop_loss
2019-11-11  187.96   2019-11-15  188.82  4     592.2   +508.15    alpha_reversal
2019-11-06  279.42   2019-11-15  283.40  7     857.1   +3414.34   max_holding
2019-11-12  64.36    2019-11-18  65.42   4     2226.7  +2354.34   alpha_reversal
2019-11-11  46.68    2019-11-19  47.66   6     2805.0  +2753.88   alpha_reversal
2019-11-19  23.98    2019-11-20  23.47   1     2974.5  -1518.15   alpha_reversal
2019-11-12  128.94   2019-11-20  124.57  6     780.1   -3410.04   stop_loss
2019-11-19  365.01   2019-11-21  364.09  2     237.8   -219.10    alpha_reversal
2019-11-15  112.65   2019-11-21  113.79  4     1114.2  +1268.38   alpha_reversal
2019-11-13  108.45   2019-11-22  110.29  7     1353.9  +2490.56   alpha_reversal
2019-11-19  188.83   2019-11-22  188.85  3     690.0   +11.77     alpha_reversal
2019-11-21  126.31   2019-11-22  126.65  1     838.7   +277.97    alpha_reversal
2019-11-18  87.67    2019-11-26  89.80   6     1596.6  +3403.57   alpha_reversal
2019-11-15  36.19    2019-11-26  36.25   7     3550.3  +217.15    max_holding
2019-11-22  369.33   2019-11-27  365.64  3     254.6   -939.08    alpha_reversal
2019-11-25  66.70    2019-11-27  68.12   2     1962.8  +2784.16   alpha_reversal
2019-11-20  282.75   2019-11-27  286.75  5     1040.2  +4159.40   alpha_reversal
2019-11-20  63.47    2019-11-29  64.39   6     2240.3  +2049.37   alpha_reversal
2019-11-27  115.79   2019-11-29  115.46  1     1229.1  -410.58    alpha_reversal
2019-11-25  47.71    2019-11-29  47.07   3     3050.8  -1957.66   alpha_reversal
2019-11-29  364.20   2019-12-02  352.90  1     303.5   -3428.08   stop_loss
2019-11-20  64.59    2019-12-02  63.88   7     2254.6  -1598.31   max_holding
2019-11-29  36.26    2019-12-02  36.28   1     4314.2  +92.96     alpha_reversal
2019-12-02  63.71    2019-12-03  62.51   1     2359.3  -2827.52   alpha_reversal
2019-12-02  187.82   2019-12-03  182.98  1     764.3   -3701.82   stop_loss
2019-11-25  22.43    2019-12-03  22.40   5     2732.9  -86.77     alpha_reversal
2019-11-29  127.52   2019-12-03  123.28  2     968.8   -4105.87   stop_loss
2019-12-04  63.12    2019-12-05  63.99   1     2090.6  +1802.25   alpha_reversal
2019-12-02  110.98   2019-12-05  112.21  3     1707.6  +2100.15   alpha_reversal
2019-12-02  47.02    2019-12-05  48.23   3     3142.2  +3808.11   alpha_reversal
2019-12-04  186.36   2019-12-06  193.65  2     655.9   +4780.24   alpha_reversal
2019-12-04  123.46   2019-12-06  125.62  2     865.7   +1874.46   alpha_reversal
2019-12-03  281.64   2019-12-06  286.20  3     1002.2  +4564.40   alpha_reversal
2019-12-06  22.40    2019-12-10  23.24   2     3437.0  +2888.82   alpha_reversal
2019-12-10  64.75    2019-12-11  65.24   1     2103.9  +1024.63   alpha_reversal
2019-12-03  141.51   2019-12-12  145.09  7     1190.7  +4262.23   max_holding
2019-12-05  343.81   2019-12-12  344.07  5     283.4   +74.39     alpha_reversal
2019-12-03  88.54    2019-12-12  87.97   7     1574.6  -897.16    alpha_reversal
2019-12-11  66.69    2019-12-12  66.84   1     2578.9  +370.08    alpha_reversal
2019-12-10  285.27   2019-12-13  288.43  3     1094.7  +3459.26   alpha_reversal
2019-12-13  339.82   2019-12-16  324.90  1     296.4   -4420.49   stop_loss
2019-12-12  69.72    2019-12-16  69.90   2     2514.6  +474.34    alpha_reversal
2019-12-18  70.23    2019-12-19  70.45   1     2606.0  +571.94    alpha_reversal
2019-12-18  146.31   2019-12-20  149.04  2     1409.3  +3850.37   alpha_reversal
2019-12-12  36.63    2019-12-23  36.37   7     5253.5  -1364.10   alpha_reversal
2019-12-16  128.08   2019-12-23  130.71  5     966.1   +2546.35   alpha_reversal
2019-12-17  325.23   2019-12-24  330.86  5     270.4   +1524.03   alpha_reversal
2019-12-19  89.66    2019-12-24  89.42   3     2061.1  -500.91    alpha_reversal
2019-12-20  197.57   2019-12-24  198.22  2     716.0   +463.57    alpha_reversal
2019-12-20  51.77    2019-12-24  51.57   2     2611.3  -531.48    alpha_reversal
2019-12-23  68.49    2019-12-26  69.85   2     2290.3  +3104.44   alpha_reversal
2019-12-16  67.51    2019-12-26  67.53   7     2692.0  +54.49     max_holding
2019-12-24  36.55    2019-12-26  36.52   1     5439.1  -182.14    alpha_reversal
2019-12-26  150.39   2019-12-27  150.51  1     1490.8  +185.24    alpha_reversal
2019-12-26  93.49    2019-12-27  93.44   1     1899.0  -79.69     alpha_reversal
2019-12-20  115.85   2019-12-30  116.90  5     1499.4  +1583.84   alpha_reversal
2019-12-27  71.71    2019-12-30  71.27   1     2869.6  -1261.51   alpha_reversal
2019-12-27  36.58    2019-12-30  36.48   1     5815.5  -550.21    alpha_reversal
2019-12-26  130.60   2019-12-30  129.85  2     1119.4  -835.76    alpha_reversal
2019-12-26  52.06    2019-12-30  51.65   2     2891.5  -1183.01   alpha_reversal
2019-12-27  328.35   2019-12-31  323.67  2     300.5   -1406.30   alpha_reversal
2019-12-27  199.06   2020-01-02  202.02  3     827.4   +2446.25   alpha_reversal
2019-12-31  36.35    2020-01-03  36.02   2     5828.8  -1903.60   alpha_reversal
2020-01-03  29.55    2020-01-06  30.09   1     2560.9  +1379.93   alpha_reversal
2019-12-27  67.21    2020-01-07  69.15   6     3104.7  +6019.30   alpha_reversal
2019-12-31  71.28    2020-01-07  69.83   4     2937.2  -4233.28   trailing_stop
2020-01-02  331.51   2020-01-08  329.25  4     316.3   -717.69    alpha_reversal
2019-12-27  122.51   2020-01-08  121.73  7     1604.0  -1260.44   max_holding
2020-01-10  31.89    2020-01-13  34.97   1     2217.6  +6831.48   alpha_reversal
2020-01-03  294.79   2020-01-14  299.10  7     1162.3  +5008.23   max_holding
2020-01-07  35.65    2020-01-15  35.22   6     5812.3  -2480.23   alpha_reversal
2020-01-06  51.29    2020-01-15  52.13   7     2522.1  +2122.36   max_holding
2020-01-08  69.43    2020-01-16  71.38   6     2514.3  +4904.07   alpha_reversal
2020-01-16  215.51   2020-01-17  215.07  1     672.9   -295.83    alpha_reversal
2020-01-16  52.50    2020-01-17  52.30   1     2356.0  -481.20    alpha_reversal
2020-01-15  328.01   2020-01-21  311.36  3     330.2   -5498.32   stop_loss
2020-01-09  122.21   2020-01-21  125.35  7     1494.8  +4687.72   alpha_reversal
2020-01-17  34.05    2020-01-22  37.95   2     1942.2  +7577.23   alpha_reversal
2020-01-16  35.45    2020-01-22  35.47   3     6272.2  +161.03    alpha_reversal
2020-01-10  128.75   2020-01-22  126.43  7     1184.0  -2757.36   trailing_stop
2020-01-17  71.29    2020-01-23  69.33   3     2740.7  -5366.42   stop_loss
2020-01-16  116.60   2020-01-24  113.00  5     1498.5  -5388.88   stop_loss
2020-01-23  124.85   2020-01-24  124.55  1     1540.2  -463.80    alpha_reversal
2020-01-22  307.33   2020-01-27  314.57  3     288.6   +2090.74   alpha_reversal
2020-01-15  93.15    2020-01-27  91.37   7     1714.9  -3046.14   trailing_stop
2020-01-23  126.66   2020-01-27  120.31  2     1176.8  -7471.68   stop_loss
2020-01-21  52.05    2020-01-27  49.33   4     2339.8  -6346.56   stop_loss
2020-01-24  67.38    2020-01-28  67.52   2     2334.5  +336.26    alpha_reversal
2020-01-24  34.98    2020-01-28  35.63   2     5793.8  +3745.11   alpha_reversal
2020-01-28  76.62    2020-01-29  78.14   1     1622.9  +2475.99   alpha_reversal
2020-01-28  156.82   2020-01-30  163.60  2     1004.4  +6803.67   alpha_reversal
2020-01-30  35.66    2020-01-31  34.98   1     5319.4  -3586.32   stop_loss
2020-01-29  120.47   2020-01-31  116.43  2     1040.3  -4210.37   stop_loss
2020-01-27  112.16   2020-02-03  113.19  5     1362.9  +1397.13   alpha_reversal
2020-01-29  68.37    2020-02-03  68.38   3     2197.3  +21.88     alpha_reversal
2020-02-04  316.22   2020-02-05  327.44  1     286.5   +3214.01   alpha_reversal
2020-01-27  295.78   2020-02-05  304.04  7     952.8   +7864.16   max_holding
2020-01-31  48.20    2020-02-05  51.71   3     1868.3  +6548.59   alpha_reversal
2020-02-03  115.14   2020-02-06  121.66  3     930.4   +6061.80   alpha_reversal
2020-02-06  174.04   2020-02-07  174.12  1     763.8   +55.15     alpha_reversal
2020-02-05  71.75    2020-02-10  74.78   3     1666.2  +5051.27   alpha_reversal
2020-02-04  35.25    2020-02-10  35.21   4     4855.1  -200.70    alpha_reversal
2020-02-03  74.44    2020-02-11  77.19   6     1342.4  +3686.89   alpha_reversal
2020-02-10  117.01   2020-02-11  117.12  1     1338.8  +138.79    alpha_reversal
2020-02-11  127.76   2020-02-12  126.88  1     1645.1  -1453.45   alpha_reversal
2020-02-06  67.15    2020-02-12  65.18   4     1886.6  -3715.86   stop_loss
2020-02-12  175.07   2020-02-13  173.94  1     701.8   -787.30    alpha_reversal
2020-02-11  74.92    2020-02-13  75.01   2     1800.6  +162.35    alpha_reversal
2020-02-11  51.65    2020-02-14  53.31   3     673.7   +1116.66   alpha_reversal
2020-02-11  35.29    2020-02-14  36.02   3     5424.5  +3935.64   alpha_reversal
2020-02-07  118.34   2020-02-14  122.31  5     905.4   +3600.54   alpha_reversal
2020-02-12  79.10    2020-02-18  77.04   3     1484.4  -3057.01   alpha_reversal
2020-02-07  205.40   2020-02-19  204.61  7     593.2   -464.40    max_holding
2020-02-19  78.23    2020-02-20  77.35   1     1546.8  -1361.15   alpha_reversal
2020-02-20  36.00    2020-02-21  36.23   1     5074.0  +1197.29   alpha_reversal
2020-02-21  75.68    2020-02-24  72.01   1     1489.8  -5462.67   stop_loss
2020-02-18  339.05   2020-02-24  317.74  4     322.4   -6870.05   stop_loss
2020-02-13  117.13   2020-02-24  112.16  6     1430.6  -7112.52   stop_loss
2020-02-13  126.16   2020-02-24  123.31  6     1694.9  -4838.84   stop_loss
2020-02-19  51.51    2020-02-24  48.32   3     1885.4  -6022.09   stop_loss
2020-02-21  199.03   2020-02-25  187.61  2     638.2   -7285.67   stop_loss
2020-02-13  64.24    2020-02-25  62.84   7     1974.6  -2754.79   alpha_reversal
2020-02-24  294.80   2020-02-25  285.58  1     669.0   -6167.47   stop_loss
2020-02-24  162.41   2020-02-27  150.18  3     557.7   -6821.07   stop_loss
2020-02-25  69.64    2020-02-27  66.06   2     1087.6  -3900.15   stop_loss
2020-02-26  107.58   2020-02-27  103.00  1     980.5   -4490.94   stop_loss
2020-02-26  186.22   2020-02-27  177.33  1     490.7   -4360.38   stop_loss
2020-02-25  122.37   2020-02-27  117.57  2     1343.7  -6443.99   stop_loss
2020-02-26  284.81   2020-02-27  271.75  1     510.3   -6666.24   stop_loss
2020-02-28  153.97   2020-03-02  164.05  1     351.2   +3540.79   alpha_reversal
2020-02-28  66.08    2020-03-03  69.87   2     768.4   +2911.02   alpha_reversal
2020-03-02  118.45   2020-03-03  114.59  1     791.4   -3056.35   stop_loss
2020-03-02  35.44    2020-03-03  34.50   1     2422.1  -2283.78   alpha_reversal
2020-03-03  49.73    2020-03-04  49.94   1     462.2   +99.90     alpha_reversal
2020-03-02  63.77    2020-03-04  64.96   2     1212.4  +1441.43   alpha_reversal
2020-03-02  289.41   2020-03-05  260.24  3     184.9   -5394.66   stop_loss
2020-03-05  158.02   2020-03-06  153.40  1     317.3   -1465.95   alpha_reversal
2020-02-28  98.64    2020-03-06  91.73   5     713.6   -4933.55   trailing_stop
2020-03-05  96.25    2020-03-06  95.01   1     629.5   -782.22    alpha_reversal
2020-02-28  174.33   2020-03-06  167.28  5     390.4   -2750.11   trailing_stop
2020-03-06  262.46   2020-03-09  227.06  1     157.9   -5591.26   stop_loss
2020-03-04  121.38   2020-03-09  115.30  3     636.3   -3862.55   stop_loss
2020-03-06  107.72   2020-03-09  92.25   1     559.9   -8666.58   stop_loss
2020-02-28  270.88   2020-03-09  250.48  6     366.9   -7481.56   trailing_stop
2020-03-10  160.07   2020-03-11  149.10  1     231.7   -2541.12   alpha_reversal
2020-03-10  36.64    2020-03-11  34.96   1     1542.3  -2582.21   stop_loss
2020-03-10  94.49    2020-03-11  89.25   1     375.0   -1963.42   alpha_reversal
2020-03-09  79.38    2020-03-12  74.73   3     414.6   -1929.37   trailing_stop
2020-03-09  60.32    2020-03-12  55.09   3     747.3   -3906.32   trailing_stop
2020-03-11  42.30    2020-03-13  36.42   2     419.9   -2469.02   alpha_reversal
2020-03-12  154.92   2020-03-16  129.55  2     91.5    -2320.60   trailing_stop
2020-03-13  88.27    2020-03-16  74.99   1     290.5   -3859.03   stop_loss
2020-03-13  89.29    2020-03-16  84.42   1     413.9   -2019.44   alpha_reversal
2020-03-13  60.25    2020-03-16  53.18   1     546.0   -3856.23   stop_loss
2020-03-13  153.83   2020-03-16  134.15  1     169.6   -3337.49   stop_loss
2020-03-12  106.09   2020-03-16  107.44  2     379.3   +511.08    alpha_reversal
2020-03-12  43.51    2020-03-16  40.09   2     838.5   -2873.56   trailing_stop
2020-03-17  55.47    2020-03-18  54.09   1     447.5   -620.83    alpha_reversal
2020-03-17  43.38    2020-03-18  39.61   1     612.2   -2305.49   stop_loss
2020-03-12  81.86    2020-03-19  91.31   5     315.1   +2976.76   alpha_reversal
2020-03-16  29.69    2020-03-20  28.49   4     330.4   -395.95    alpha_reversal
2020-03-20  56.39    2020-03-23  52.41   1     471.3   -1871.84   alpha_reversal
2020-03-20  40.17    2020-03-23  40.79   1     556.0   +343.48    alpha_reversal
2020-03-17  124.20   2020-03-24  127.62  5     70.2    +239.68    alpha_reversal
2020-03-23  28.97    2020-03-24  33.65   1     330.9   +1549.35   alpha_reversal
2020-03-20  35.01    2020-03-24  35.30   2     695.8   +202.00    alpha_reversal
2020-03-19  129.80   2020-03-26  143.81  5     121.6   +1704.19   alpha_reversal
2020-03-23  54.24    2020-03-31  61.41   6     351.4   +2520.08   alpha_reversal
2020-03-25  77.93    2020-03-31  76.41   4     207.7   -315.87    alpha_reversal
2020-03-27  137.48   2020-03-31  134.09  2     127.2   -431.13    alpha_reversal
2020-03-20  101.42   2020-03-31  110.82  7     263.1   +2472.56   alpha_reversal
2020-03-23  205.05   2020-03-31  236.82  6     108.7   +3452.30   alpha_reversal
2020-03-31  149.21   2020-04-01  130.63  1     67.5    -1255.05   alpha_reversal
2020-03-26  57.70    2020-04-01  54.63   4     424.1   -1303.03   alpha_reversal
2020-03-27  34.31    2020-04-01  32.09   3     354.4   -786.64    alpha_reversal
2020-03-25  33.61    2020-04-02  36.41   6     712.6   +1999.04   alpha_reversal
2020-03-30  152.28   2020-04-03  146.05  4     152.7   -950.85    alpha_reversal
2020-04-01  58.24    2020-04-03  58.30   2     406.2   +25.40     alpha_reversal
2020-03-26  97.82    2020-04-03  95.28   6     330.2   -839.19    alpha_reversal
2020-03-25  53.90    2020-04-03  60.19   7     462.7   +2907.95   alpha_reversal
2020-04-01  226.38   2020-04-03  228.03  2     118.5   +195.26    alpha_reversal
2020-04-01  41.96    2020-04-03  42.41   2     663.8   +301.18    alpha_reversal
2020-04-02  30.31    2020-04-06  34.40   2     394.0   +1609.79   alpha_reversal
2020-04-06  63.45    2020-04-07  62.65   1     423.7   -337.91    alpha_reversal
2020-04-06  243.59   2020-04-07  243.60  1     124.8   +0.58      alpha_reversal
2020-04-06  45.08    2020-04-07  44.81   1     704.9   -190.57    alpha_reversal
2020-04-01  71.67    2020-04-08  80.86   5     235.0   +2161.18   alpha_reversal
2020-04-01  126.15   2020-04-08  153.50  5     141.3   +3864.08   alpha_reversal
2020-04-08  59.89    2020-04-09  59.80   1     517.4   -42.00     alpha_reversal
2020-04-06  157.07   2020-04-13  157.14  4     171.2   +12.13     alpha_reversal
2020-04-08  64.32    2020-04-13  65.99   2     445.4   +743.66    alpha_reversal
2020-04-13  108.50   2020-04-14  114.11  1     433.1   +2430.35   alpha_reversal
2020-04-13  60.05    2020-04-14  62.71   1     566.0   +1503.86   alpha_reversal
2020-04-09  37.42    2020-04-14  39.59   2     920.8   +2000.23   alpha_reversal
2020-04-03  124.58   2020-04-15  145.91  7     73.5    +1566.98   max_holding
2020-04-14  261.00   2020-04-15  255.20  1     147.3   -854.55    alpha_reversal
2020-04-16  168.26   2020-04-17  169.57  1     224.1   +294.21    alpha_reversal
2020-04-08  64.49    2020-04-17  65.88   6     512.6   +711.41    alpha_reversal
2020-04-08  45.35    2020-04-17  48.39   6     769.0   +2335.42   alpha_reversal
2020-04-13  101.27   2020-04-20  102.51  5     237.3   +294.03    alpha_reversal
2020-04-20  39.89    2020-04-22  40.38   2     1086.0  +536.61    alpha_reversal
2020-04-21  98.36    2020-04-22  98.97   1     280.3   +170.53    alpha_reversal
2020-04-20  78.72    2020-04-23  76.65   3     289.3   -598.33    alpha_reversal
2020-04-21  251.12   2020-04-23  256.41  2     171.1   +906.71    alpha_reversal
2020-04-17  154.08   2020-04-24  128.92  5     90.1    -2267.97   alpha_reversal
2020-04-21  60.14    2020-04-24  63.28   3     653.3   +2047.47   alpha_reversal
2020-04-22  63.20    2020-04-24  64.28   2     631.1   +682.49    alpha_reversal
2020-04-24  153.69   2020-04-27  159.21  1     213.0   +1175.71   alpha_reversal
2020-04-24  260.25   2020-04-27  263.74  1     188.2   +656.60    alpha_reversal
2020-04-24  68.41    2020-04-28  67.28   2     624.1   -704.30    alpha_reversal
2020-04-28  39.32    2020-04-30  37.30   2     1319.9  -2664.57   stop_loss
2020-04-29  269.67   2020-05-04  260.54  3     200.5   -1829.83   alpha_reversal
2020-04-27  130.52   2020-05-06  125.14  7     400.0   -2151.31   alpha_reversal
2020-05-04  60.73    2020-05-06  60.88   2     715.7   +103.46    alpha_reversal
2020-05-07  264.58   2020-05-08  268.69  1     225.7   +927.48    alpha_reversal
2020-05-01  46.10    2020-05-08  47.68   5     997.7   +1581.51   alpha_reversal
2020-05-07  124.85   2020-05-11  126.01  2     491.8   +570.44    alpha_reversal
2020-05-01  153.77   2020-05-12  153.71  7     228.7   -13.34     max_holding
2020-05-01  37.76    2020-05-12  38.15   7     1343.0  +528.17    max_holding
2020-05-05  78.97    2020-05-13  72.06   6     404.4   -2795.61   stop_loss
2020-05-08  54.66    2020-05-13  52.70   3     468.3   -913.57    alpha_reversal
2020-05-08  60.37    2020-05-14  63.19   4     794.8   +2242.02   alpha_reversal
2020-05-05  97.52    2020-05-14  94.99   7     333.0   -842.61    max_holding
2020-05-06  121.92   2020-05-15  119.94  7     129.7   -256.84    max_holding
2020-05-14  75.12    2020-05-15  73.66   1     441.1   -645.93    alpha_reversal
2020-05-18  135.51   2020-05-19  130.37  1     152.4   -782.03    alpha_reversal
2020-05-18  157.92   2020-05-19  154.25  1     254.8   -935.34    alpha_reversal
2020-05-18  54.27    2020-05-22  54.43   4     521.3   +84.62     alpha_reversal
2020-05-19  174.52   2020-05-26  172.87  4     320.2   -528.43    alpha_reversal
2020-05-20  77.38    2020-05-26  76.70   3     759.5   -518.57    alpha_reversal
2020-05-14  262.09   2020-05-26  274.79  7     228.2   +2898.01   alpha_reversal
2020-05-18  77.64    2020-05-27  86.93   6     449.7   +4176.25   alpha_reversal
2020-05-20  157.54   2020-05-27  181.86  4     263.6   +6410.79   alpha_reversal
2020-05-26  54.62    2020-05-27  54.65   1     618.3   +22.28     alpha_reversal
2020-05-19  61.28    2020-05-27  61.21   5     857.9   -52.54     alpha_reversal
2020-05-19  68.19    2020-05-29  71.05   7     787.7   +2254.50   max_holding
2020-05-20  46.77    2020-06-01  45.99   7     1092.3  -848.41    max_holding
2020-05-27  77.11    2020-06-02  78.30   4     844.9   +1004.91   alpha_reversal
2020-05-21  124.11   2020-06-02  126.15  7     556.4   +1138.25   max_holding
2020-05-22  100.71   2020-06-02  110.31  6     392.8   +3772.80   alpha_reversal
2020-05-26  121.15   2020-06-03  123.86  6     505.9   +1368.12   alpha_reversal
2020-05-29  55.69    2020-06-04  57.60   4     684.1   +1301.12   alpha_reversal
2020-06-02  63.88    2020-06-04  64.37   2     1036.5  +514.72    alpha_reversal
2020-05-26  38.22    2020-06-04  37.64   7     1728.9  -998.67    max_holding
2020-05-27  173.27   2020-06-05  178.23  7     350.9   +1740.17   max_holding
2020-06-05  125.47   2020-06-08  124.89  1     676.1   -389.74    alpha_reversal
2020-06-05  293.70   2020-06-08  296.95  1     314.3   +1022.45   alpha_reversal
2020-06-09  180.88   2020-06-10  187.41  1     438.1   +2856.97   alpha_reversal
2020-06-05  80.36    2020-06-10  85.44   3     1035.0  +5265.29   alpha_reversal
2020-06-05  124.21   2020-06-10  132.31  3     629.1   +5092.21   alpha_reversal
2020-06-05  37.51    2020-06-10  37.35   3     2161.3  -347.48    alpha_reversal
2020-06-10  91.04    2020-06-11  83.36   1     495.1   -3802.63   stop_loss
2020-06-04  187.69   2020-06-11  169.44  5     265.6   -4846.16   trailing_stop
2020-06-09  124.33   2020-06-11  119.87  2     686.8   -3060.11   stop_loss
2020-06-05  65.00    2020-06-11  61.06   4     1134.1  -4469.19   stop_loss
2020-06-10  293.38   2020-06-11  276.19  1     354.5   -6093.70   stop_loss
2020-06-12  178.92   2020-06-15  179.89  1     352.1   +339.29    alpha_reversal
2020-06-11  170.08   2020-06-15  190.84  2     120.4   +2500.46   alpha_reversal
2020-06-04  70.17    2020-06-15  70.42   7     1061.2  +264.31    max_holding
2020-06-12  279.78   2020-06-15  282.11  1     269.7   +628.26    alpha_reversal
2020-06-15  58.95    2020-06-17  60.70   2     1011.9  +1768.02   alpha_reversal
2020-06-12  110.27   2020-06-17  114.70  3     364.4   +1615.53   alpha_reversal
2020-06-15  128.70   2020-06-18  132.63  3     480.9   +1891.93   alpha_reversal
2020-06-17  51.06    2020-06-18  50.98   1     1300.7  -101.82    alpha_reversal
2020-06-12  121.08   2020-06-22  122.02  6     579.1   +540.97    alpha_reversal
2020-06-18  60.70    2020-06-22  61.06   2     1052.7  +380.06    alpha_reversal
2020-06-15  36.43    2020-06-22  37.51   5     2240.7  +2404.88   alpha_reversal
2020-06-22  71.97    2020-06-23  72.56   1     1072.5  +630.95    alpha_reversal
2020-06-18  178.10   2020-06-23  178.74  3     251.3   +161.46    alpha_reversal
2020-06-19  50.27    2020-06-23  51.54   2     1373.0  +1738.63   alpha_reversal
2020-06-22  86.99    2020-06-24  87.19   2     839.0   +168.84    alpha_reversal
2020-06-15  86.91    2020-06-24  81.17   7     431.5   -2475.80   max_holding
2020-06-23  66.82    2020-06-24  64.02   1     678.4   -1895.57   alpha_reversal
2020-06-18  114.25   2020-06-24  109.19  4     385.0   -1948.45   alpha_reversal
2020-06-23  121.69   2020-06-26  117.27  3     700.6   -3096.04   stop_loss
2020-06-23  61.42    2020-06-26  59.82   3     1099.9  -1756.58   alpha_reversal
2020-06-24  37.12    2020-06-29  36.70   3     2554.5  -1071.18   alpha_reversal
2020-06-29  189.12   2020-06-30  193.76  1     363.3   +1685.11   alpha_reversal
2020-06-24  176.78   2020-06-30  183.21  4     133.2   +856.27    alpha_reversal
2020-06-29  118.43   2020-06-30  119.67  1     687.5   +848.77    alpha_reversal
2020-06-29  134.09   2020-07-02  144.44  3     540.4   +5596.85   alpha_reversal
2020-07-01  180.41   2020-07-06  187.82  2     131.0   +969.81    alpha_reversal
2020-06-29  69.32    2020-07-06  74.33   4     934.5   +4681.83   alpha_reversal
2020-07-02  120.08   2020-07-06  121.67  1     760.3   +1209.24   alpha_reversal
2020-07-02  36.78    2020-07-08  38.36   3     2766.7  +4358.51   alpha_reversal
2020-07-02  88.26    2020-07-09  92.75   4     846.0   +3797.49   alpha_reversal
2020-07-08  180.17   2020-07-09  173.19  1     153.8   -1072.86   alpha_reversal
2020-06-29  79.83    2020-07-09  79.04   7     493.3   -390.69    max_holding
2020-06-29  169.04   2020-07-09  171.80  7     244.3   +675.21    max_holding
2020-07-08  114.73   2020-07-09  112.47  1     523.6   -1184.06   alpha_reversal
2020-07-09  39.42    2020-07-10  40.28   1     2236.3  +1931.66   alpha_reversal
2020-07-09  61.08    2020-07-13  61.54   2     1339.2  +621.51    alpha_reversal
2020-07-14  101.17   2020-07-15  103.02  1     302.0   +557.34    alpha_reversal
2020-07-17  297.19   2020-07-20  299.29  1     368.1   +773.89    alpha_reversal
2020-07-20  95.37    2020-07-21  93.96   1     870.5   -1227.66   alpha_reversal
2020-07-10  178.53   2020-07-21  178.54  7     163.3   +1.87      max_holding
2020-07-14  154.28   2020-07-21  156.84  5     373.3   +955.18    alpha_reversal
2020-07-16  75.16    2020-07-21  77.12   3     1011.7  +1979.93   alpha_reversal
2020-07-13  197.34   2020-07-22  201.60  7     379.6   +1616.59   max_holding
2020-07-22  94.32    2020-07-23  89.93   1     902.2   -3954.24   stop_loss
2020-07-22  77.64    2020-07-23  75.18   1     1033.8  -2544.79   alpha_reversal
2020-07-20  84.33    2020-07-24  85.10   4     687.8   +525.65    alpha_reversal
2020-07-22  155.07   2020-07-27  152.68  3     347.1   -829.21    alpha_reversal
2020-07-22  123.57   2020-07-27  126.07  3     591.4   +1477.92   alpha_reversal
2020-07-21  62.83    2020-07-28  63.40   5     1562.3  +896.30    alpha_reversal
2020-07-22  179.88   2020-07-29  165.93  5     196.3   -2739.23   alpha_reversal
2020-07-28  84.35    2020-07-29  86.31   1     795.6   +1558.73   alpha_reversal
2020-07-20  40.57    2020-07-29  40.28   7     2241.5  -629.80    alpha_reversal
2020-07-24  89.80    2020-07-31  102.93  5     772.2   +10136.93  alpha_reversal
2020-07-22  179.98   2020-07-31  172.79  7     320.9   -2309.59   max_holding
2020-07-29  194.47   2020-08-03  206.16  3     385.6   +4507.25   alpha_reversal
2020-07-28  125.07   2020-08-04  125.28  5     901.6   +186.50    alpha_reversal
2020-07-28  98.48    2020-08-05  98.95   6     267.9   +125.91    alpha_reversal
2020-08-04  165.15   2020-08-06  172.11  2     232.1   +1616.03   alpha_reversal
2020-08-03  174.21   2020-08-10  182.75  5     404.4   +3455.40   alpha_reversal
2020-08-05  126.41   2020-08-10  125.97  3     1009.4  -445.28    alpha_reversal
2020-07-30  40.15    2020-08-10  40.65   7     2631.2  +1321.87   max_holding
2020-08-06  121.24   2020-08-11  128.45  3     669.2   +4828.51   alpha_reversal
2020-08-10  179.50   2020-08-12  175.35  2     244.8   -1015.26   alpha_reversal
2020-08-04  82.82    2020-08-12  89.13   6     893.9   +5645.62   alpha_reversal
2020-08-10  74.26    2020-08-13  75.17   3     1098.7  +998.34    alpha_reversal
2020-08-06  99.35    2020-08-13  108.01  5     344.5   +2982.69   alpha_reversal
2020-08-13  88.73    2020-08-14  88.67   1     855.1   -46.24     alpha_reversal
2020-08-06  64.55    2020-08-14  66.42   6     1421.3  +2656.17   alpha_reversal
2020-08-12  40.69    2020-08-14  41.04   2     2838.9  +982.83    alpha_reversal
2020-08-11  193.83   2020-08-18  201.35  5     381.4   +2871.21   alpha_reversal
2020-08-10  157.49   2020-08-18  165.54  6     397.1   +3198.95   alpha_reversal
2020-08-17  75.23    2020-08-18  77.11   1     1292.6  +2436.08   alpha_reversal
2020-08-11  125.19   2020-08-18  127.72  5     1054.1  +2666.81   alpha_reversal
2020-08-14  311.16   2020-08-18  312.51  2     581.4   +784.95    alpha_reversal
2020-08-07  72.74    2020-08-18  72.10   7     756.6   -480.92    max_holding
2020-08-20  312.49   2020-08-21  313.28  1     656.4   +521.89    alpha_reversal
2020-08-13  174.82   2020-08-24  178.18  7     248.3   +835.21    max_holding
2020-08-17  177.42   2020-08-24  180.97  5     481.6   +1709.52   alpha_reversal
2020-08-25  134.96   2020-08-26  143.47  1     325.0   +2767.52   alpha_reversal
2020-08-26  130.59   2020-08-27  131.05  1     1187.3  +536.58    alpha_reversal
2020-08-18  124.83   2020-08-27  129.21  7     761.1   +3334.56   max_holding
2020-08-19  85.42    2020-08-28  88.98   7     960.6   +3427.80   alpha_reversal
2020-08-25  40.47    2020-08-28  43.42   3     2992.5  +8835.28   alpha_reversal
2020-08-31  125.34   2020-09-01  130.21  1     733.1   +3564.73   alpha_reversal
2020-08-31  166.19   2020-09-01  158.27  1     284.5   -2253.22   alpha_reversal
2020-08-31  72.03    2020-09-02  74.61   2     1240.3  +3199.04   alpha_reversal
2020-08-31  131.55   2020-09-03  128.11  3     1302.5  -4477.89   stop_loss
2020-08-31  67.91    2020-09-03  68.02   3     2271.3  +243.39    alpha_reversal
2020-09-01  325.71   2020-09-03  318.73  2     774.0   -5401.85   trailing_stop
2020-09-04  204.68   2020-09-08  193.41  1     387.1   -4361.54   stop_loss
2020-08-27  174.29   2020-09-08  161.00  7     306.2   -4068.79   stop_loss
2020-09-02  88.10    2020-09-08  86.52   3     1126.8  -1787.08   alpha_reversal
2020-08-31  180.08   2020-09-08  177.80  5     583.1   -1328.83   alpha_reversal
2020-09-03  135.73   2020-09-08  110.01  2     235.8   -6064.65   stop_loss
2020-09-04  316.45   2020-09-08  307.49  1     508.0   -4548.56   stop_loss
2020-09-09  201.85   2020-09-10  196.00  1     321.6   -1881.97   alpha_reversal
2020-09-04  127.41   2020-09-10  125.85  3     948.4   -1485.59   alpha_reversal
2020-09-04  67.89    2020-09-10  66.49   3     1785.1  -2492.93   alpha_reversal
2020-09-09  313.88   2020-09-10  308.12  1     422.6   -2433.33   alpha_reversal
2020-09-04  164.81   2020-09-11  155.73  4     436.5   -3963.25   stop_loss
2020-09-09  122.15   2020-09-11  124.18  2     180.9   +366.10    alpha_reversal
2020-09-10  157.77   2020-09-14  165.27  2     297.9   +2233.61   alpha_reversal
2020-09-04  78.45    2020-09-14  74.79   5     1059.2  -3882.88   stop_loss
2020-09-09  177.75   2020-09-14  176.80  3     467.6   -444.37    alpha_reversal
2020-09-15  199.45   2020-09-16  195.70  1     312.4   -1174.30   alpha_reversal
2020-09-04  71.72    2020-09-16  75.48   7     1063.8  +3999.88   alpha_reversal
2020-09-16  167.54   2020-09-18  161.06  2     287.5   -1864.49   alpha_reversal
2020-09-15  76.16    2020-09-18  71.92   3     918.7   -3896.40   stop_loss
2020-09-15  134.06   2020-09-18  137.34  3     578.5   +1898.38   alpha_reversal
2020-09-17  74.83    2020-09-18  73.22   1     880.4   -1415.78   alpha_reversal
2020-09-10  110.24   2020-09-21  106.82  7     399.2   -1364.81   max_holding
2020-09-16  86.41    2020-09-21  82.52   3     893.1   -3471.81   stop_loss
2020-09-15  174.04   2020-09-21  170.36  4     478.5   -1764.15   alpha_reversal
2020-09-17  126.19   2020-09-21  124.30  2     902.4   -1713.95   alpha_reversal
2020-09-15  314.23   2020-09-21  302.94  4     393.1   -4437.50   stop_loss
2020-09-17  193.85   2020-09-22  197.96  3     324.5   +1334.05   alpha_reversal
2020-09-14  155.23   2020-09-22  156.37  6     359.9   +412.14    alpha_reversal
2020-09-14  42.54    2020-09-22  42.81   6     1639.2  +432.54    alpha_reversal
2020-09-21  131.11   2020-09-23  130.12  2     496.4   -490.02    alpha_reversal
2020-09-23  123.85   2020-09-24  123.93  1     790.7   +57.90     alpha_reversal
2020-09-22  168.43   2020-09-25  171.19  3     421.6   +1161.86   alpha_reversal
2020-09-24  130.93   2020-09-25  131.50  1     494.6   +278.50    alpha_reversal
2020-09-25  156.11   2020-09-28  166.00  1     267.6   +2646.49   alpha_reversal
2020-09-23  71.19    2020-09-28  72.80   3     880.3   +1423.68   alpha_reversal
2020-09-23  126.85   2020-09-29  139.62  4     164.4   +2099.70   alpha_reversal
2020-09-21  70.96    2020-09-30  72.64   7     870.2   +1467.67   alpha_reversal
2020-09-30  112.49   2020-10-01  113.33  1     448.3   +375.90    alpha_reversal
2020-09-22  81.71    2020-10-01  83.96   7     858.8   +1937.63   max_holding
2020-09-28  174.98   2020-10-01  174.35  3     384.8   -243.00    alpha_reversal
2020-09-30  143.07   2020-10-01  149.31  1     190.5   +1187.95   alpha_reversal
2020-09-23  66.29    2020-10-01  65.43   6     1440.1  -1238.04   alpha_reversal
2020-09-30  200.94   2020-10-02  196.78  2     338.8   -1406.67   alpha_reversal
2020-09-29  42.49    2020-10-02  43.49   3     2152.0  +2146.59   alpha_reversal
2020-10-02  175.71   2020-10-05  177.21  1     405.8   +605.79    alpha_reversal
2020-09-30  134.55   2020-10-05  138.33  3     537.3   +2029.13   alpha_reversal
2020-10-05  73.57    2020-10-06  71.92   1     1039.8  -1715.82   alpha_reversal
2020-10-05  43.93    2020-10-06  43.53   1     2138.7  -868.38    alpha_reversal
2020-09-30  310.59   2020-10-06  310.32  4     364.0   -99.55     alpha_reversal
2020-10-05  77.23    2020-10-07  79.33   2     984.1   +2061.69   alpha_reversal
2020-09-30  157.52   2020-10-09  164.25  7     380.1   +2560.08   max_holding
2020-10-07  141.84   2020-10-12  147.36  3     213.8   +1180.72   alpha_reversal
2020-10-05  113.16   2020-10-13  117.51  6     480.8   +2091.73   alpha_reversal
2020-10-07  72.40    2020-10-13  77.67   4     1049.3  +5537.43   alpha_reversal
2020-10-02  64.82    2020-10-13  64.68   7     1569.4  -214.89    max_holding
2020-10-05  127.10   2020-10-14  126.87  7     887.4   -211.53    max_holding
2020-10-14  117.72   2020-10-15  117.13  1     517.7   -302.05    alpha_reversal
2020-10-07  164.69   2020-10-15  164.16  6     263.3   -140.72    alpha_reversal
2020-10-19  87.30    2020-10-21  86.84   2     1019.5  -472.07    alpha_reversal
2020-10-21  62.63    2020-10-23  63.98   2     1959.3  +2640.91   alpha_reversal
2020-10-19  160.44   2020-10-26  160.27  5     372.6   -62.92     alpha_reversal
2020-10-20  123.95   2020-10-26  123.33  4     965.6   -599.38    alpha_reversal
2020-10-23  44.57    2020-10-26  44.00   1     2882.7  -1636.27   alpha_reversal
2020-10-16  322.09   2020-10-26  314.45  6     430.1   -3286.88   alpha_reversal
2020-10-22  112.43   2020-10-27  113.15  3     565.0   +402.56    alpha_reversal
2020-10-22  90.00    2020-10-28  84.37   4     1005.8  -5663.32   stop_loss
2020-10-26  176.80   2020-10-28  166.64  2     480.6   -4883.88   stop_loss
2020-10-27  141.63   2020-10-28  135.27  1     316.2   -2010.32   alpha_reversal
2020-10-29  306.04   2020-10-30  302.55  1     372.6   -1301.38   alpha_reversal
2020-11-03  197.21   2020-11-05  213.10  2     380.6   +6048.92   alpha_reversal
2020-10-30  117.57   2020-11-05  119.72  4     884.3   +1903.56   alpha_reversal
2020-11-02  61.54    2020-11-05  64.48   3     1640.1  +4828.23   alpha_reversal
2020-11-02  306.24   2020-11-05  324.50  3     365.7   +6678.95   alpha_reversal
2020-11-05  91.28    2020-11-06  89.98   1     809.6   -1057.26   alpha_reversal
2020-11-05  157.17   2020-11-09  179.27  2     346.9   +7666.66   alpha_reversal
2020-11-09  102.26   2020-11-10  101.83  1     657.8   -285.69    alpha_reversal
2020-11-04  162.14   2020-11-10  151.68  4     348.1   -3641.96   trailing_stop
2020-11-05  178.40   2020-11-10  190.97  3     427.1   +5366.20   alpha_reversal
2020-11-05  146.10   2020-11-12  137.18  5     306.5   -2733.69   alpha_reversal
2020-11-05  44.45    2020-11-12  45.88   5     2697.4  +3854.15   alpha_reversal
2020-11-05  115.62   2020-11-16  116.94  7     512.7   +675.81    alpha_reversal
2020-11-05  148.03   2020-11-16  157.04  7     412.0   +3711.85   alpha_reversal
2020-11-10  80.08    2020-11-16  90.60   4     1133.3  +11917.95  alpha_reversal
2020-11-11  156.95   2020-11-18  155.20  5     368.7   -646.21    alpha_reversal
2020-11-13  136.23   2020-11-18  162.13  3     350.4   +9075.68   alpha_reversal
2020-11-09  64.58    2020-11-18  64.31   7     1315.9  -348.61    max_holding
2020-11-10  201.58   2020-11-19  203.26  7     349.7   +585.73    max_holding
2020-11-17  156.03   2020-11-19  156.77  2     483.0   +358.21    alpha_reversal
2020-11-18  114.85   2020-11-20  114.06  2     692.0   -543.54    alpha_reversal
2020-11-19  205.77   2020-11-20  199.52  1     241.7   -1511.33   alpha_reversal
2020-11-19  155.93   2020-11-20  154.89  1     490.5   -508.40    alpha_reversal
2020-11-13  87.93    2020-11-24  87.43   7     906.1   -455.07    max_holding
2020-11-23  158.64   2020-11-24  160.40  1     561.6   +990.50    alpha_reversal
2020-11-23  201.25   2020-11-27  205.95  3     468.9   +2202.79   alpha_reversal
2020-11-17  101.57   2020-11-27  105.93  7     764.0   +3334.21   max_holding
2020-11-23  155.00   2020-11-30  158.32  4     518.7   +1725.04   alpha_reversal
2020-11-20  125.50   2020-11-30  124.80  5     1085.0  -763.05    alpha_reversal
2020-11-27  158.91   2020-11-30  157.40  1     609.9   -920.80    alpha_reversal
2020-11-19  331.82   2020-11-30  335.46  6     432.0   +1569.64   alpha_reversal
2020-11-23  110.78   2020-12-01  119.29  5     743.4   +6327.71   alpha_reversal
2020-11-24  64.29    2020-12-01  65.36   4     1621.3  +1727.93   alpha_reversal
2020-11-30  47.34    2020-12-01  47.24   1     3000.9  -281.30    alpha_reversal
2020-11-20  87.09    2020-12-02  90.85   7     954.5   +3584.34   alpha_reversal
2020-11-23  201.14   2020-12-03  207.90  7     483.3   +3265.87   max_holding
2020-11-30  210.82   2020-12-04  232.59  4     253.7   +5526.15   alpha_reversal
2020-12-01  157.26   2020-12-07  162.04  4     650.7   +3113.49   alpha_reversal
2020-12-07  120.41   2020-12-08  120.90  1     910.4   +447.98    alpha_reversal
2020-11-30  103.12   2020-12-09  105.78  7     845.7   +2255.57   alpha_reversal
2020-12-04  211.74   2020-12-09  214.39  3     533.8   +1414.09   alpha_reversal
2020-12-08  162.32   2020-12-09  163.27  1     665.7   +634.52    alpha_reversal
2020-12-08  343.31   2020-12-09  339.89  1     637.8   -2180.77   alpha_reversal
2020-12-04  65.73    2020-12-10  66.51   4     2017.2  +1564.96   alpha_reversal
2020-12-10  209.13   2020-12-11  203.23  1     230.6   -1360.51   alpha_reversal
2020-12-02  46.63    2020-12-11  45.66   7     3077.5  -2984.50   max_holding
2020-12-03  205.21   2020-12-14  204.96  7     571.5   -139.09    alpha_reversal
2020-12-03  159.42   2020-12-14  157.77  7     617.9   -1017.45   alpha_reversal
2020-12-09  88.21    2020-12-14  86.85   3     1313.9  -1783.03   alpha_reversal
2020-12-10  119.92   2020-12-16  124.24  4     884.1   +3821.46   alpha_reversal
2020-12-16  210.03   2020-12-17  209.96  1     639.3   -48.59     alpha_reversal
2020-12-15  96.01    2020-12-17  95.23   2     1001.7  -783.28    alpha_reversal
2020-12-17  132.64   2020-12-18  133.28  1     1244.8  +790.56    alpha_reversal
2020-12-16  207.69   2020-12-18  231.55  2     239.5   +5713.95   alpha_reversal
2020-12-16  45.22    2020-12-18  45.34   2     3728.2  +433.71    alpha_reversal
2020-12-17  162.62   2020-12-21  163.36  2     709.6   +521.66    alpha_reversal
2020-12-10  340.12   2020-12-21  342.28  7     647.1   +1397.65   max_holding
2020-12-21  213.20   2020-12-22  214.28  1     645.3   +696.12    alpha_reversal
2020-12-21  124.77   2020-12-22  128.19  1     874.9   +2995.09   alpha_reversal
2020-12-14  103.48   2020-12-22  106.33  6     1061.8  +3017.29   alpha_reversal
2020-12-21  227.12   2020-12-22  220.81  1     487.4   -3075.50   alpha_reversal
2020-12-21  160.39   2020-12-23  159.18  2     749.6   -903.43    alpha_reversal
2020-12-15  65.07    2020-12-23  64.43   6     2051.6  -1326.54   alpha_reversal
2020-12-15  229.61   2020-12-24  217.04  7     262.1   -3295.96   alpha_reversal
2020-12-15  87.38    2020-12-24  85.95   7     1360.4  -1934.03   max_holding
2020-12-22  160.27   2020-12-24  162.81  2     730.6   +1857.08   alpha_reversal
2020-12-22  342.05   2020-12-24  343.34  2     599.4   +776.62    alpha_reversal
2020-12-24  213.36   2020-12-28  215.26  1     687.1   +1306.45   alpha_reversal
2020-12-23  131.19   2020-12-28  132.14  2     1225.2  +1160.42   alpha_reversal
2020-12-21  45.39    2020-12-28  45.11   4     3998.4  -1112.91   alpha_reversal
2020-12-23  95.19    2020-12-28  97.29   2     1211.3  +2538.00   alpha_reversal
2020-12-24  158.71   2020-12-29  166.02  2     837.4   +6115.49   alpha_reversal
2020-12-28  216.20   2020-12-30  216.56  2     330.3   +120.12    alpha_reversal
2020-12-30  229.30   2020-12-31  232.83  1     512.7   +1810.95   alpha_reversal
2020-12-31  66.14    2021-01-04  65.40   1     2434.5  -1812.74   alpha_reversal
2020-12-29  160.33   2021-01-04  165.16  3     798.9   +3856.09   alpha_reversal
2020-12-30  44.83    2021-01-05  45.28   3     4272.0  +1892.03   alpha_reversal
2020-12-30  212.33   2021-01-06  203.10  4     712.0   -6575.53   stop_loss
2021-01-05  136.72   2021-01-06  137.86  1     1272.5  +1461.55   alpha_reversal
2021-01-05  345.86   2021-01-07  352.74  2     656.7   +4521.02   alpha_reversal
2020-12-31  86.96    2021-01-12  86.12   7     1575.9  -1325.00   max_holding
2021-01-04  202.82   2021-01-13  207.11  7     354.1   +1517.49   max_holding
2021-01-04  159.41   2021-01-13  158.22  7     775.7   -927.59    max_holding
2021-01-05  127.48   2021-01-14  125.31  7     793.8   -1721.63   alpha_reversal
2021-01-12  352.79   2021-01-14  352.15  2     633.7   -406.20    alpha_reversal
2021-01-06  45.60    2021-01-15  44.93   7     4045.8  -2723.13   trailing_stop
2021-01-07  209.09   2021-01-19  207.11  7     635.4   -1257.52   max_holding
2021-01-19  281.66   2021-01-20  283.34  1     213.7   +359.96    alpha_reversal
2021-01-14  210.01   2021-01-21  207.31  4     384.0   -1040.06   alpha_reversal
2021-01-15  155.29   2021-01-21  165.27  3     824.3   +8223.52   alpha_reversal
2021-01-15  85.72    2021-01-21  93.39   3     1422.8  +10917.20  alpha_reversal
2021-01-15  123.71   2021-01-22  135.18  4     820.5   +9413.23   alpha_reversal
2021-01-15  349.93   2021-01-22  356.26  4     655.2   +4148.01   alpha_reversal
2021-01-21  123.08   2021-01-22  118.36  1     610.5   -2883.90   alpha_reversal
2021-01-13  136.33   2021-01-25  143.17  7     1314.4  +8993.37   alpha_reversal
2021-01-19  44.59    2021-01-25  45.42   4     4015.0  +3325.72   alpha_reversal
2021-01-15  67.42    2021-01-26  64.82   6     2041.9  -5300.05   stop_loss
2021-01-22  117.88   2021-01-27  112.54  3     1076.4  -5745.23   stop_loss
2021-01-21  175.37   2021-01-27  164.65  4     705.2   -7554.84   stop_loss
2021-01-27  62.32    2021-01-29  62.25   2     2242.3  -139.66    alpha_reversal
2021-01-28  351.72   2021-01-29  344.34  1     623.2   -4602.47   stop_loss
2021-01-26  249.02   2021-02-02  253.37  5     426.9   +1857.48   alpha_reversal
2021-01-29  111.49   2021-02-02  117.72  2     605.9   +3775.15   alpha_reversal
2021-02-03  232.75   2021-02-04  231.57  1     524.5   -618.92    alpha_reversal
2021-02-01  43.31    2021-02-04  44.28   3     3747.6  +3632.93   alpha_reversal
2021-02-04  117.57   2021-02-05  117.11  1     673.1   -307.33    alpha_reversal
2021-02-05  231.99   2021-02-08  232.01  1     567.0   +15.03     alpha_reversal
2021-02-04  283.47   2021-02-08  287.66  2     228.7   +958.58    alpha_reversal
2021-02-02  131.35   2021-02-09  132.41  5     695.3   +735.46    alpha_reversal
2021-02-08  123.47   2021-02-09  122.86  1     1076.2  -663.31    alpha_reversal
2021-02-08  212.06   2021-02-10  211.81  2     445.2   -107.70    alpha_reversal
2021-02-01  140.49   2021-02-10  143.89  7     980.1   +3328.56   alpha_reversal
2021-02-01  350.42   2021-02-10  362.96  7     563.3   +7062.15   max_holding
2021-02-08  121.61   2021-02-10  122.53  2     690.0   +637.09    alpha_reversal
2021-02-10  123.05   2021-02-11  122.58  1     1147.1  -534.83    alpha_reversal
2021-02-11  234.18   2021-02-12  234.43  1     635.1   +155.17    alpha_reversal
2021-02-09  102.97   2021-02-12  103.84  3     1001.3  +871.71    alpha_reversal
2021-02-10  180.41   2021-02-12  180.48  2     690.5   +45.44     alpha_reversal
2021-02-12  211.09   2021-02-16  217.07  1     472.4   +2827.97   alpha_reversal
2021-02-12  124.45   2021-02-16  127.32  1     1197.3  +3434.14   alpha_reversal
2021-02-12  270.72   2021-02-16  275.43  1     465.6   +2192.48   alpha_reversal
2021-02-12  143.83   2021-02-16  142.38  1     1210.9  -1751.21   alpha_reversal
2021-02-11  44.78    2021-02-17  45.73   3     4380.9  +4131.51   alpha_reversal
2021-02-16  184.66   2021-02-17  184.41  1     718.2   -184.92    alpha_reversal
2021-02-10  131.94   2021-02-18  126.27  5     847.3   -4796.65   stop_loss
2021-02-11  60.43    2021-02-18  60.91   4     2575.9  +1238.46   alpha_reversal
2021-02-09  165.33   2021-02-19  162.41  7     742.7   -2167.87   alpha_reversal
2021-02-17  143.04   2021-02-19  140.58  2     1277.5  -3135.70   alpha_reversal
2021-02-18  208.58   2021-02-22  212.77  2     473.6   +1984.14   alpha_reversal
2021-02-10  268.41   2021-02-22  238.05  7     243.1   -7380.78   stop_loss
2021-02-22  360.48   2021-02-24  364.53  2     761.4   +3083.70   alpha_reversal
2021-02-23  212.23   2021-02-25  216.34  2     428.5   +1763.71   alpha_reversal
2021-02-22  159.12   2021-02-25  152.78  3     832.9   -5276.38   stop_loss
2021-02-22  140.64   2021-02-26  137.54  4     1300.2  -4030.89   alpha_reversal
2021-02-22  225.14   2021-03-01  227.24  5     664.6   +1399.36   alpha_reversal
2021-02-23  122.65   2021-03-01  124.40  4     820.5   +1441.05   alpha_reversal
2021-02-26  129.67   2021-03-02  132.04  2     1056.0  +2502.82   alpha_reversal
2021-02-26  354.27   2021-03-02  359.66  2     592.5   +3197.45   alpha_reversal
2021-02-23  233.06   2021-03-03  217.62  6     220.9   -3410.76   trailing_stop
2021-03-01  58.52    2021-03-03  59.21   2     2692.4  +1843.31   alpha_reversal
2021-02-24  41.42    2021-03-03  39.63   5     3612.7  -6456.53   stop_loss
2021-02-24  121.38   2021-03-03  112.63  5     735.3   -6430.44   stop_loss
2021-03-01  138.42   2021-03-04  132.86  3     1185.6  -6595.81   stop_loss
2021-03-04  217.67   2021-03-08  218.09  2     559.3   +232.35    alpha_reversal
2021-03-05  132.96   2021-03-08  134.59  1     892.0   +1451.72   alpha_reversal
2021-03-03  150.33   2021-03-08  147.52  3     744.6   -2085.96   alpha_reversal
2021-03-04  350.86   2021-03-08  355.18  2     505.2   +2182.58   alpha_reversal
2021-03-09  224.44   2021-03-10  222.91  1     520.1   -795.06    alpha_reversal
2021-03-04  207.25   2021-03-10  222.58  4     188.5   +2888.75   alpha_reversal
2021-03-09  360.61   2021-03-10  362.49  1     464.8   +874.71    alpha_reversal
2021-03-05  110.83   2021-03-10  104.33  3     625.7   -4066.17   alpha_reversal
2021-03-04  39.66    2021-03-11  41.05   5     3444.0  +4784.82   alpha_reversal
2021-03-04  194.15   2021-03-11  200.32  5     537.0   +3317.22   alpha_reversal
2021-03-09  153.22   2021-03-12  154.40  3     623.5   +734.64    alpha_reversal
2021-03-08  296.51   2021-03-15  306.73  5     333.9   +3410.77   alpha_reversal
2021-03-05  135.62   2021-03-15  139.24  6     1086.3  +3926.06   alpha_reversal
2021-03-12  208.95   2021-03-15  210.33  1     508.3   +700.12    alpha_reversal
2021-03-05  118.32   2021-03-16  122.24  7     708.2   +2777.39   alpha_reversal
2021-03-11  233.32   2021-03-16  225.51  3     160.3   -1250.75   alpha_reversal
2021-03-12  60.85    2021-03-16  62.59   2     2699.9  +4697.76   alpha_reversal
2021-03-11  110.62   2021-03-16  109.39  3     609.6   -754.68    alpha_reversal
2021-03-15  154.16   2021-03-17  156.71  2     673.5   +1715.38   alpha_reversal
2021-03-16  255.34   2021-03-18  255.93  2     265.8   +158.00    alpha_reversal
2021-03-09  101.23   2021-03-18  100.19  7     944.4   -985.86    max_holding
2021-03-10  136.68   2021-03-19  136.55  7     847.1   -108.25    alpha_reversal
2021-03-16  303.19   2021-03-19  305.09  3     348.0   +662.70    alpha_reversal
2021-03-16  41.48    2021-03-22  41.29   4     3815.4  -712.07    alpha_reversal
2021-03-15  225.43   2021-03-23  227.86  6     550.7   +1338.97   alpha_reversal
2021-03-22  120.24   2021-03-23  119.29  1     809.9   -767.49    alpha_reversal
2021-03-22  206.24   2021-03-23  198.95  1     521.6   -3801.59   alpha_reversal
2021-03-17  370.01   2021-03-23  363.60  4     561.5   -3595.45   alpha_reversal
2021-03-18  105.44   2021-03-23  105.69  3     695.9   +176.34    alpha_reversal
2021-03-22  133.01   2021-03-25  134.27  3     835.6   +1050.93   alpha_reversal
2021-03-18  139.42   2021-03-25  140.58  5     1340.5  +1558.47   alpha_reversal
2021-03-24  117.03   2021-03-26  118.00  2     836.4   +814.11    alpha_reversal
2021-03-17  234.05   2021-03-26  206.13  7     182.0   -5080.24   stop_loss
2021-03-29  101.50   2021-03-30  101.43  1     1161.5  -78.69     alpha_reversal
2021-03-26  63.12    2021-03-30  62.71   2     2524.6  -1043.87   alpha_reversal
2021-03-22  251.36   2021-03-31  254.59  7     280.2   +906.96    alpha_reversal
2021-03-23  294.37   2021-03-31  289.85  6     358.1   -1619.30   alpha_reversal
2021-03-24  100.34   2021-03-31  108.81  5     717.4   +6079.19   alpha_reversal
2021-03-31  226.35   2021-04-01  232.43  1     575.3   +3500.71   alpha_reversal
2021-03-29  203.87   2021-04-01  220.47  3     204.0   +3387.41   alpha_reversal
2021-03-31  119.03   2021-04-05  122.56  2     903.5   +3190.78   alpha_reversal
2021-04-05  259.49   2021-04-06  255.04  1     310.8   -1382.39   alpha_reversal
2021-04-05  43.54    2021-04-06  43.70   1     4166.2  +689.43    alpha_reversal
2021-04-01  212.37   2021-04-06  210.24  2     499.3   -1061.78   alpha_reversal
2021-04-01  290.70   2021-04-07  289.45  3     365.3   -459.11    alpha_reversal
2021-04-05  230.47   2021-04-07  223.54  2     213.8   -1479.31   alpha_reversal
2021-04-07  239.91   2021-04-08  242.89  1     598.9   +1780.72   alpha_reversal
2021-03-31  154.78   2021-04-08  164.88  5     778.2   +7860.25   alpha_reversal
2021-03-31  370.35   2021-04-08  381.36  5     600.2   +6607.50   alpha_reversal
2021-04-08  228.05   2021-04-09  225.56  1     243.0   -604.34    alpha_reversal
2021-04-08  43.62    2021-04-12  43.61   2     4484.2  -69.67     alpha_reversal
2021-04-08  293.81   2021-04-13  290.45  3     396.9   -1333.83   alpha_reversal
2021-04-05  141.99   2021-04-13  138.42  6     1422.2  -5077.64   stop_loss
2021-04-07  210.24   2021-04-13  209.06  4     525.7   -618.38    alpha_reversal
2021-04-13  111.85   2021-04-14  111.12  1     1359.3  -995.47    alpha_reversal
2021-04-12  234.11   2021-04-14  243.95  2     261.9   +2578.47   alpha_reversal
2021-04-05  62.81    2021-04-14  62.25   7     2789.2  -1538.69   max_holding
2021-04-14  212.22   2021-04-15  211.95  1     593.5   -158.37    alpha_reversal
2021-04-07  252.71   2021-04-16  248.06  7     332.4   -1545.77   max_holding
2021-04-15  246.41   2021-04-16  246.47  1     245.7   +15.62     alpha_reversal
2021-04-15  388.61   2021-04-16  389.52  1     797.1   +724.70    alpha_reversal
2021-04-16  212.93   2021-04-19  211.84  1     628.8   -689.79    alpha_reversal
2021-04-19  131.40   2021-04-20  129.58  1     1084.9  -1969.69   alpha_reversal
2021-04-15  113.38   2021-04-20  112.96  3     1350.4  -570.72    alpha_reversal
2021-04-14  138.94   2021-04-20  144.50  4     1431.2  +7950.12   alpha_reversal
2021-04-14  43.50    2021-04-21  44.05   5     5101.4  +2769.89   alpha_reversal
2021-04-20  132.29   2021-04-22  130.48  2     1113.1  -2019.70   alpha_reversal
2021-04-20  166.82   2021-04-22  165.37  2     895.1   -1296.59   alpha_reversal
2021-04-21  250.17   2021-04-26  250.85  3     734.4   +499.59    alpha_reversal
2021-04-20  207.70   2021-04-26  211.12  4     638.4   +2178.90   alpha_reversal
2021-04-23  167.13   2021-04-27  170.79  2     913.6   +3342.56   alpha_reversal
2021-04-21  108.11   2021-04-27  111.56  4     910.9   +3144.56   alpha_reversal
2021-04-19  244.27   2021-04-28  235.34  7     390.1   -3483.32   alpha_reversal
2021-04-30  236.60   2021-05-03  228.19  1     270.1   -2271.94   alpha_reversal
2021-04-27  43.21    2021-05-03  44.33   4     5564.1  +6251.11   alpha_reversal
2021-04-27  141.77   2021-05-05  145.01  6     1467.8  +4747.70   alpha_reversal
2021-04-28  390.04   2021-05-07  394.06  7     877.6   +3524.74   max_holding
2021-04-28  110.19   2021-05-07  107.48  7     999.1   -2711.20   max_holding
2021-05-07  220.89   2021-05-10  222.15  1     606.8   +766.14    alpha_reversal
2021-05-04  224.65   2021-05-11  205.63  5     279.7   -5318.12   stop_loss
2021-05-10  44.14    2021-05-11  43.70   1     5642.4  -2493.01   alpha_reversal
2021-05-03  241.80   2021-05-12  229.22  7     711.0   -8940.53   stop_loss
2021-05-05  124.83   2021-05-12  119.72  5     1092.1  -5581.19   trailing_stop
2021-05-05  163.61   2021-05-12  157.52  5     802.4   -4887.28   stop_loss
2021-05-05  228.29   2021-05-14  228.36  7     442.9   +27.32     alpha_reversal
2021-05-12  378.84   2021-05-14  388.88  2     682.7   +6860.73   alpha_reversal
2021-05-12  42.61    2021-05-17  43.50   3     4840.7  +4265.80   alpha_reversal
2021-05-17  113.56   2021-05-18  112.14  1     1227.9  -1749.11   alpha_reversal
2021-05-10  103.42   2021-05-19  103.40  7     1091.2  -22.48     alpha_reversal
2021-05-18  227.66   2021-05-21  234.70  3     427.9   +3011.91   alpha_reversal
2021-05-19  161.67   2021-05-21  160.07  2     803.9   -1283.66   alpha_reversal
2021-05-20  319.22   2021-05-21  324.64  1     391.5   +2123.96   alpha_reversal
2021-05-20  148.63   2021-05-21  148.38  1     1571.8  -383.57    alpha_reversal
2021-05-12  196.73   2021-05-21  193.53  7     271.2   -867.28    max_holding
2021-05-17  65.15    2021-05-21  64.52   4     2802.0  -1757.79   alpha_reversal
2021-05-21  144.16   2021-05-24  144.79  1     1115.7  +708.50    alpha_reversal
2021-05-18  121.87   2021-05-25  123.75  5     1077.8  +2023.32   alpha_reversal
2021-05-17  388.28   2021-05-25  390.43  6     671.5   +1444.48   alpha_reversal
2021-05-24  104.64   2021-05-25  105.13  1     1011.8  +499.22    alpha_reversal
2021-05-19  233.94   2021-05-26  241.76  5     644.4   +5034.10   alpha_reversal
2021-05-24  202.25   2021-05-27  210.18  3     289.4   +2294.69   alpha_reversal
2021-05-21  217.45   2021-05-27  220.93  4     599.9   +2088.80   alpha_reversal
2021-05-26  391.60   2021-05-27  391.41  1     724.7   -134.82    alpha_reversal
2021-05-27  145.66   2021-05-28  145.41  1     1154.4  -280.52    alpha_reversal
2021-05-27  239.90   2021-06-01  237.82  2     746.2   -1549.06   alpha_reversal
2021-05-27  147.58   2021-06-01  144.57  2     1744.9  -5255.90   stop_loss
2021-05-25  63.21    2021-06-01  61.26   4     2927.1  -5718.48   stop_loss
2021-05-28  108.08   2021-06-01  109.03  1     1182.8  +1123.58   alpha_reversal
2021-06-01  147.16   2021-06-02  147.02  1     1207.6  -166.98    alpha_reversal
2021-06-01  118.14   2021-06-02  117.50  1     1480.6  -951.95    alpha_reversal
2021-05-28  161.23   2021-06-03  159.27  3     1004.5  -1972.17   alpha_reversal
2021-06-01  222.51   2021-06-03  223.28  2     665.9   +510.47    alpha_reversal
2021-06-02  392.78   2021-06-03  390.93  1     816.7   -1510.00   alpha_reversal
2021-06-02  61.88    2021-06-04  63.29   2     3120.5  +4376.99   alpha_reversal
2021-05-27  44.42    2021-06-04  44.42   5     4852.6  +27.71     alpha_reversal
2021-05-26  123.82   2021-06-07  122.77  7     1210.7  -1271.41   max_holding
2021-06-04  110.06   2021-06-07  108.76  1     1203.1  -1560.18   alpha_reversal
2021-06-07  252.79   2021-06-08  252.63  1     486.7   -74.33     alpha_reversal
2021-06-03  116.48   2021-06-08  118.88  3     1448.1  +3482.00   alpha_reversal
2021-06-04  160.39   2021-06-09  163.98  3     1080.7  +3873.44   alpha_reversal
2021-06-04  394.90   2021-06-09  393.62  3     817.4   -1047.55   alpha_reversal
2021-06-02  237.97   2021-06-10  247.28  6     763.8   +7116.74   alpha_reversal
2021-06-08  146.23   2021-06-10  142.01  2     1282.9  -5412.45   stop_loss
2021-06-02  145.30   2021-06-10  145.92  6     1588.8  +990.33    alpha_reversal
2021-06-08  61.89    2021-06-10  65.02   2     3124.9  +9786.97   alpha_reversal
2021-06-02  201.81   2021-06-11  203.20  7     342.5   +475.17    alpha_reversal
2021-06-10  395.84   2021-06-14  396.99  2     944.9   +1081.70   alpha_reversal
2021-06-11  142.06   2021-06-15  137.39  2     1283.4  -5988.66   stop_loss
2021-06-14  121.50   2021-06-15  120.36  1     1749.3  -1991.58   alpha_reversal
2021-06-14  144.57   2021-06-15  143.66  1     1663.7  -1518.98   alpha_reversal
2021-06-14  206.00   2021-06-15  199.69  1     395.9   -2499.22   alpha_reversal
2021-06-15  65.27    2021-06-16  65.98   1     2917.0  +2096.10   alpha_reversal
2021-06-08  43.83    2021-06-16  42.95   6     5665.4  -5002.99   trailing_stop
2021-06-10  206.90   2021-06-16  198.85  4     630.9   -5081.73   stop_loss
2021-06-16  138.50   2021-06-17  134.36  1     1165.6  -4815.72   stop_loss
2021-06-10  332.90   2021-06-17  321.51  5     423.3   -4824.17   stop_loss
2021-06-16  201.72   2021-06-17  205.43  1     403.1   +1494.06   alpha_reversal
2021-06-16  247.67   2021-06-18  249.39  2     815.6   +1405.34   alpha_reversal
2021-06-10  248.46   2021-06-18  237.23  6     500.5   -5622.57   stop_loss
2021-06-17  394.31   2021-06-18  388.60  1     943.1   -5383.52   stop_loss
2021-06-09  107.86   2021-06-18  106.93  7     1258.5  -1172.90   alpha_reversal
2021-06-21  394.56   2021-06-22  396.27  1     833.0   +1426.84   alpha_reversal
2021-06-21  105.51   2021-06-23  107.17  2     1194.5  +1980.99   alpha_reversal
2021-06-18  310.55   2021-06-24  327.96  4     371.4   +6465.16   alpha_reversal
2021-06-23  243.69   2021-06-25  248.26  2     522.9   +2386.57   alpha_reversal
2021-06-16  143.75   2021-06-25  143.42  7     1658.4  -556.92    max_holding
2021-06-23  65.16    2021-06-25  66.49   2     2922.3  +3887.12   alpha_reversal
2021-06-17  191.98   2021-06-25  198.07  6     567.6   +3456.25   alpha_reversal
2021-06-28  258.58   2021-06-29  260.90  1     839.8   +1946.60   alpha_reversal
2021-06-24  130.23   2021-06-29  132.94  3     1400.4  +3805.34   alpha_reversal
2021-06-18  131.10   2021-06-29  136.47  7     1015.9  +5461.45   max_holding
2021-06-28  229.69   2021-06-29  226.81  1     382.0   -1100.88   alpha_reversal
2021-06-25  107.72   2021-06-29  111.03  2     1317.5  +4360.06   alpha_reversal
2021-06-25  170.16   2021-06-30  171.92  3     1035.0  +1825.68   alpha_reversal
2021-06-29  66.57    2021-06-30  66.98   1     3166.2  +1316.46   alpha_reversal
2021-06-21  42.76    2021-06-30  44.16   7     5383.9  +7559.46   max_holding
2021-06-28  198.23   2021-07-01  198.44  3     593.3   +121.50    alpha_reversal
2021-06-24  121.56   2021-07-02  124.17  6     1713.3  +4475.28   alpha_reversal
2021-07-02  199.60   2021-07-06  195.51  1     697.6   -2847.55   alpha_reversal
2021-07-01  171.73   2021-07-07  184.74  3     1199.8  +15600.18  alpha_reversal
2021-07-01  226.09   2021-07-07  214.78  3     438.5   -4960.16   stop_loss
2021-06-29  235.88   2021-07-09  239.47  7     508.8   +1827.87   alpha_reversal
2021-07-09  141.65   2021-07-12  140.91  1     1433.5  -1055.69   alpha_reversal
2021-07-12  238.41   2021-07-13  228.09  1     555.7   -5736.78   stop_loss
2021-07-06  329.27   2021-07-13  334.39  5     452.0   +2311.71   alpha_reversal
2021-07-09  408.30   2021-07-13  407.96  2     1000.6  -342.63    alpha_reversal
2021-07-08  108.96   2021-07-13  114.42  3     1475.9  +8057.95   alpha_reversal
2021-07-12  266.85   2021-07-14  271.58  2     916.2   +4326.73   alpha_reversal
2021-07-12  66.86    2021-07-14  66.85   2     3722.2  -24.34     alpha_reversal
2021-07-13  44.07    2021-07-14  44.33   1     5858.3  +1521.53   alpha_reversal
2021-07-14  224.56   2021-07-19  206.89  3     539.1   -9528.40   stop_loss
2021-07-08  217.71   2021-07-19  215.30  7     410.9   -991.53    max_holding
2021-07-15  193.78   2021-07-19  186.95  2     758.1   -5172.60   stop_loss
2021-07-13  147.98   2021-07-21  148.03  6     1969.7  +87.14     alpha_reversal
2021-07-20  404.12   2021-07-21  406.99  1     836.6   +2396.71   alpha_reversal
2021-07-22  44.28    2021-07-23  44.60   1     5849.0  +1865.87   alpha_reversal
2021-07-21  332.51   2021-07-26  334.31  3     353.0   +636.08    alpha_reversal
2021-07-16  142.90   2021-07-27  143.12  7     1291.4  +294.10    max_holding
2021-07-20  217.26   2021-07-27  222.16  5     466.1   +2283.81   alpha_reversal
2021-07-21  136.25   2021-07-27  134.86  4     988.4   -1375.68   alpha_reversal
2021-07-16  178.77   2021-07-27  181.23  7     970.9   +2386.56   max_holding
2021-07-21  65.86    2021-07-28  67.47   5     3445.7  +5530.70   alpha_reversal
2021-07-29  336.12   2021-07-30  333.41  1     439.4   -1190.87   alpha_reversal
2021-07-29  44.59    2021-07-30  44.64   1     6344.0  +333.18    alpha_reversal
2021-07-21  194.19   2021-07-30  190.33  7     705.0   -2719.87   max_holding
2021-07-23  107.96   2021-08-02  107.92  6     1315.1  -56.90     alpha_reversal
2021-08-02  274.07   2021-08-04  275.42  2     860.9   +1162.85   alpha_reversal
2021-08-03  143.84   2021-08-04  143.30  1     1255.7  -682.57    alpha_reversal
2021-08-05  278.59   2021-08-06  278.26  1     897.6   -301.81    alpha_reversal
2021-08-05  143.55   2021-08-06  142.72  1     1333.9  -1103.27   alpha_reversal
2021-08-05  230.05   2021-08-09  232.15  2     500.7   +1050.94   alpha_reversal
2021-08-09  151.86   2021-08-10  151.76  1     2111.7  -209.91    alpha_reversal
2021-08-09  238.04   2021-08-10  236.54  1     441.7   -659.84    alpha_reversal
2021-08-09  109.28   2021-08-10  107.46  1     1645.2  -3005.60   alpha_reversal
2021-08-10  415.02   2021-08-11  415.63  1     1101.8  +677.96    alpha_reversal
2021-08-09  277.45   2021-08-12  278.59  3     929.4   +1064.39   alpha_reversal
2021-08-09  142.82   2021-08-12  145.41  3     1425.8  +3695.40   alpha_reversal
2021-08-04  167.82   2021-08-13  164.62  7     852.7   -2731.93   max_holding
2021-08-05  65.17    2021-08-13  66.08   6     3718.5  +3376.92   alpha_reversal
2021-08-12  136.14   2021-08-16  137.11  2     1726.2  +1673.90   alpha_reversal
2021-08-11  236.06   2021-08-17  221.79  4     466.6   -6656.21   stop_loss
2021-08-11  107.22   2021-08-17  102.61  4     1515.2  -6990.90   stop_loss
2021-08-16  229.17   2021-08-18  218.89  2     542.2   -5575.82   stop_loss
2021-08-16  165.03   2021-08-18  159.98  2     1031.6  -5210.36   stop_loss
2021-08-18  229.78   2021-08-19  224.38  1     401.8   -2169.82   alpha_reversal
2021-08-19  156.11   2021-08-20  156.72  1     1954.6  +1180.13   alpha_reversal
2021-08-18  411.73   2021-08-20  415.24  2     1009.4  +3536.25   alpha_reversal
2021-08-18  102.10   2021-08-20  99.85   2     1498.1  -3376.31   alpha_reversal
2021-08-17  139.95   2021-08-23  139.54  4     1224.8  -509.45    alpha_reversal
2021-08-20  226.87   2021-08-25  236.95  3     410.6   +4139.18   alpha_reversal
2021-08-23  102.57   2021-08-26  109.06  3     1402.0  +9100.54   alpha_reversal
2021-08-24  165.37   2021-08-27  167.40  3     1033.4  +2093.70   alpha_reversal
2021-08-25  145.04   2021-08-31  148.28  4     1296.1  +4204.39   alpha_reversal
2021-08-20  212.78   2021-08-31  219.39  7     517.2   +3421.03   max_holding
2021-08-24  154.24   2021-08-31  152.10  5     1951.4  -4175.43   alpha_reversal
2021-08-20  188.85   2021-08-31  194.12  7     702.1   +3700.38   max_holding
2021-08-24  67.01    2021-09-01  65.61   6     3638.1  -5069.20   stop_loss
2021-08-24  46.85    2021-09-02  46.69   7     5013.5  -770.52    max_holding
2021-09-02  66.49    2021-09-03  66.54   1     3892.7  +210.70    alpha_reversal
2021-09-01  152.79   2021-09-07  151.39  3     2100.3  -2941.08   alpha_reversal
2021-09-02  220.94   2021-09-08  211.27  3     587.6   -5679.81   stop_loss
2021-09-01  290.99   2021-09-09  286.29  5     836.4   -3932.87   alpha_reversal
2021-09-08  151.17   2021-09-09  147.66  1     1974.4  -6941.82   stop_loss
2021-08-31  142.57   2021-09-10  140.12  7     1332.1  -3262.18   alpha_reversal
2021-09-03  142.63   2021-09-10  139.65  4     1733.6  -5168.23   stop_loss
2021-09-08  64.97    2021-09-10  63.26   2     3578.0  -6118.46   stop_loss
2021-09-02  195.16   2021-09-10  188.73  5     826.0   -5308.82   stop_loss
2021-09-13  418.67   2021-09-14  416.00  1     982.9   -2629.73   alpha_reversal
2021-09-13  141.24   2021-09-15  143.17  2     1586.6  +3074.23   alpha_reversal
2021-09-15  140.98   2021-09-16  140.77  1     1233.7  -250.72    alpha_reversal
2021-09-07  367.07   2021-09-16  354.51  7     501.4   -6296.60   stop_loss
2021-09-15  173.88   2021-09-17  173.04  2     1026.1  -858.77    alpha_reversal
2021-09-15  189.58   2021-09-17  183.89  2     822.2   -4682.25   stop_loss
2021-09-13  146.20   2021-09-20  139.60  5     1107.2  -7309.35   stop_loss
2021-09-13  214.59   2021-09-20  209.40  5     546.2   -2835.63   alpha_reversal
2021-09-17  140.55   2021-09-20  136.21  1     1223.3  -5313.17   stop_loss
2021-09-17  350.19   2021-09-20  337.93  1     421.8   -5172.83   stop_loss
2021-09-10  146.84   2021-09-20  143.92  6     1733.2  -5066.22   stop_loss
2021-09-17  415.14   2021-09-20  407.81  1     861.8   -6316.92   stop_loss
2021-09-16  112.12   2021-09-20  106.36  2     1478.1  -8512.80   stop_loss
2021-09-13  63.09    2021-09-21  62.54   6     3236.0  -1778.83   alpha_reversal
2021-09-14  45.40    2021-09-21  44.95   5     5178.5  -2318.43   alpha_reversal
2021-09-21  140.22   2021-09-27  141.97  4     951.0   +1668.47   alpha_reversal
2021-09-21  284.22   2021-09-28  273.07  5     608.8   -6786.57   trailing_stop
2021-09-20  137.65   2021-09-28  134.65  6     1286.3  -3861.28   trailing_stop
2021-09-21  336.22   2021-09-28  348.09  5     349.8   +4152.18   alpha_reversal
2021-09-21  407.83   2021-09-28  407.51  5     656.7   -212.18    trailing_stop
2021-09-27  107.81   2021-09-28  103.76  1     1404.5  -5686.34   stop_loss
2021-09-20  167.87   2021-09-29  164.97  7     891.9   -2583.88   max_holding
2021-09-20  175.84   2021-09-29  182.16  7     672.9   +4248.70   alpha_reversal
2021-09-23  144.98   2021-09-30  141.89  5     1437.4  -4451.37   stop_loss
2021-09-28  63.78    2021-09-30  65.27   2     2626.5  +3918.18   alpha_reversal
2021-09-23  44.92    2021-09-30  43.81   5     4621.2  -5131.51   stop_loss
2021-09-30  164.33   2021-10-04  159.41  2     825.5   -4065.38   stop_loss
2021-09-29  408.60   2021-10-04  402.74  3     596.0   -3496.84   alpha_reversal
2021-09-29  273.80   2021-10-07  283.98  6     541.8   +5513.55   alpha_reversal
2021-09-28  138.73   2021-10-07  139.94  7     954.8   +1154.37   max_holding
2021-09-29  133.32   2021-10-07  138.01  6     1057.1  +4963.99   alpha_reversal
2021-10-06  347.14   2021-10-07  349.09  1     287.4   +560.32    alpha_reversal
2021-10-06  261.05   2021-10-07  264.40  1     363.6   +1220.72   alpha_reversal
2021-10-05  407.34   2021-10-08  411.40  3     491.5   +1997.86   alpha_reversal
2021-09-29  103.61   2021-10-08  102.04  7     1286.3  -2017.83   max_holding
2021-10-04  42.70    2021-10-11  43.86   5     3887.4  +4477.14   alpha_reversal
2021-10-11  226.56   2021-10-12  223.46  1     455.4   -1413.88   alpha_reversal
2021-10-05  161.13   2021-10-13  164.13  6     764.8   +2295.49   alpha_reversal
2021-10-04  140.02   2021-10-13  139.87  7     1244.3  -195.96    max_holding
2021-10-06  176.80   2021-10-15  183.22  7     609.4   +3908.94   max_holding
2021-10-12  407.82   2021-10-18  420.17  4     525.6   +6487.09   alpha_reversal
2021-10-13  285.67   2021-10-19  296.87  4     522.5   +5849.76   alpha_reversal
2021-10-15  140.28   2021-10-19  141.99  2     1047.4  +1793.79   alpha_reversal
2021-10-11  139.61   2021-10-20  145.77  7     921.0   +5672.99   max_holding
2021-10-12  69.23    2021-10-21  70.54   7     1683.8  +2195.42   max_holding
2021-10-19  185.35   2021-10-21  186.09  2     705.2   +518.56    alpha_reversal
2021-10-20  170.84   2021-10-22  166.69  2     949.3   -3934.14   alpha_reversal
2021-10-15  217.15   2021-10-26  209.71  7     485.0   -3609.92   max_holding
2021-10-27  346.13   2021-10-28  358.83  1     238.9   +3035.21   alpha_reversal
2021-10-29  143.24   2021-11-01  143.22  1     1370.8  -27.75     alpha_reversal
2021-10-25  105.48   2021-11-02  105.81  6     1435.1  +474.16    alpha_reversal
2021-11-03  213.49   2021-11-04  212.92  1     573.3   -322.87    alpha_reversal
2021-11-03  169.28   2021-11-05  175.86  2     763.8   +5023.51   alpha_reversal
2021-11-03  105.83   2021-11-05  109.23  2     1567.1  +5341.52   alpha_reversal
2021-11-05  148.11   2021-11-10  144.67  3     1104.8  -3794.28   alpha_reversal
2021-11-01  152.27   2021-11-10  150.16  7     1002.2  -2109.82   max_holding
2021-11-04  144.75   2021-11-11  143.26  5     1245.3  -1865.06   alpha_reversal
2021-11-04  365.05   2021-11-12  361.87  6     340.6   -1086.21   alpha_reversal
2021-11-03  47.25    2021-11-12  46.44   7     4800.6  -3878.33   trailing_stop
2021-11-11  436.18   2021-11-12  439.03  1     801.1   +2285.23   alpha_reversal
2021-11-12  145.12   2021-11-15  143.66  1     1250.7  -1818.64   alpha_reversal
2021-11-12  146.84   2021-11-16  147.69  2     1177.3  +990.14    alpha_reversal
2021-11-12  176.35   2021-11-16  176.95  2     660.2   +396.69    alpha_reversal
2021-11-09  341.34   2021-11-17  362.82  6     161.6   +3471.01   alpha_reversal
2021-11-08  71.92    2021-11-17  71.78   7     1345.4  -190.25    alpha_reversal
2021-11-11  150.30   2021-11-18  146.07  5     1114.0  -4718.02   stop_loss
2021-11-17  177.54   2021-11-18  184.71  1     705.8   +5061.60   alpha_reversal
2021-11-10  144.77   2021-11-18  148.54  6     933.1   +3513.98   alpha_reversal
2021-11-16  109.60   2021-11-18  114.42  2     1244.1  +5989.54   alpha_reversal
2021-11-16  360.60   2021-11-19  346.21  3     393.2   -5658.56   stop_loss
2021-11-22  147.38   2021-11-23  150.75  1     1121.3  +3782.60   alpha_reversal
2021-11-23  440.34   2021-11-24  441.07  1     821.0   +602.95    alpha_reversal
2021-11-22  210.00   2021-11-26  199.11  3     430.9   -4694.18   stop_loss
2021-11-16  190.35   2021-11-26  183.96  7     678.5   -4337.01   stop_loss
2021-11-26  140.92   2021-11-29  141.27  1     1387.7  +479.56    alpha_reversal
2021-11-26  68.86    2021-11-29  65.08   1     1541.7  -5826.88   stop_loss
2021-11-29  436.96   2021-11-30  428.03  1     648.8   -5793.79   stop_loss
2021-11-29  198.60   2021-12-01  188.10  2     390.6   -4102.23   alpha_reversal
2021-11-30  65.16    2021-12-01  64.69   1     1417.8  -671.44    alpha_reversal
2021-12-01  338.57   2021-12-03  343.85  2     300.5   +1585.39   alpha_reversal
2021-12-01  423.70   2021-12-03  426.02  2     517.7   +1201.23   alpha_reversal
2021-12-03  141.08   2021-12-06  144.09  1     1213.8  +3649.92   alpha_reversal
2021-12-01  43.15    2021-12-06  43.69   3     3578.1  +1937.54   alpha_reversal
2021-11-30  179.16   2021-12-06  186.32  4     597.6   +4279.91   alpha_reversal
2021-12-02  160.33   2021-12-07  167.42  3     724.7   +5143.07   alpha_reversal
2021-11-29  144.58   2021-12-08  143.97  7     918.5   -560.61    max_holding
2021-12-06  142.05   2021-12-09  146.36  3     843.4   +3628.81   alpha_reversal
2021-12-07  350.76   2021-12-09  334.43  2     138.9   -2267.68   alpha_reversal
2021-12-01  318.81   2021-12-10  330.52  7     451.3   +5281.84   max_holding
2021-12-08  43.15    2021-12-10  44.51   2     3720.9  +5041.45   alpha_reversal
2021-12-02  64.28    2021-12-13  63.80   7     1406.3  -664.71    max_holding
2021-12-07  113.54   2021-12-13  107.96  4     915.1   -5110.28   stop_loss
2021-12-03  169.57   2021-12-14  169.01  7     574.3   -325.80    max_holding
2021-12-09  143.89   2021-12-15  141.49  4     879.2   -2111.16   alpha_reversal
2021-12-09  356.57   2021-12-15  350.30  4     317.3   -1988.92   alpha_reversal
2021-12-09  189.15   2021-12-15  187.49  4     626.5   -1034.76   alpha_reversal
2021-12-06  431.50   2021-12-15  442.16  7     473.9   +5054.32   max_holding
2021-12-15  323.23   2021-12-16  313.50  1     387.2   -3767.42   alpha_reversal
2021-12-14  170.67   2021-12-16  168.48  2     650.8   -1428.53   alpha_reversal
2021-12-15  145.31   2021-12-16  143.19  1     872.9   -1853.90   alpha_reversal
2021-12-10  339.18   2021-12-16  308.82  4     147.2   -4469.43   stop_loss
2021-12-16  143.85   2021-12-17  140.43  1     923.0   -3150.76   alpha_reversal
2021-12-15  173.40   2021-12-20  167.00  3     600.0   -3843.62   alpha_reversal
2021-12-17  311.01   2021-12-20  299.83  1     144.2   -1612.21   alpha_reversal
2021-12-20  43.97    2021-12-21  44.06   1     2991.2  +265.01    alpha_reversal
2021-12-13  197.50   2021-12-22  201.59  7     389.3   +1592.33   max_holding
2021-12-20  138.04   2021-12-23  140.88  3     846.7   +2401.49   alpha_reversal
2021-12-22  148.65   2021-12-23  148.78  1     1134.3  +152.43    alpha_reversal
2021-12-21  313.00   2021-12-23  355.49  2     146.2   +6211.51   alpha_reversal
2021-12-22  66.88    2021-12-23  66.44   1     1786.6  -793.35    alpha_reversal
2021-12-14  107.94   2021-12-23  112.37  7     1019.5  +4512.21   max_holding
2021-12-17  312.75   2021-12-27  330.43  5     358.1   +6332.06   alpha_reversal
2021-12-17  167.55   2021-12-27  176.37  5     569.5   +5023.02   alpha_reversal
2021-12-20  140.52   2021-12-28  145.41  5     845.4   +4138.64   alpha_reversal
2021-12-27  67.24    2021-12-28  67.38   1     2023.3  +290.03    alpha_reversal
2021-12-28  206.23   2021-12-29  203.56  1     415.7   -1111.97   alpha_reversal
2021-12-27  44.47    2021-12-29  45.04   2     3632.6  +2074.01   alpha_reversal
2021-12-20  181.11   2021-12-29  191.92  6     591.5   +6391.81   alpha_reversal
2021-12-29  330.28   2021-12-30  327.41  1     425.3   -1219.71   alpha_reversal
2021-12-29  175.62   2021-12-30  174.29  1     704.7   -937.00    alpha_reversal
2021-12-20  334.19   2021-12-30  346.35  7     284.0   +3454.18   max_holding
2021-12-29  450.65   2021-12-30  448.96  1     535.1   -907.06    alpha_reversal
2021-12-30  45.23    2021-12-31  45.66   1     4048.4  +1758.95   alpha_reversal
2021-12-31  344.03   2022-01-04  366.08  2     394.3   +8694.96   alpha_reversal
2022-01-03  400.13   2022-01-04  383.01  1     151.6   -2595.01   alpha_reversal
2021-12-27  114.36   2022-01-04  124.21  6     1031.9  +10163.15  alpha_reversal
2021-12-31  324.84   2022-01-05  305.27  3     461.8   -9034.62   stop_loss
2021-12-30  202.81   2022-01-05  212.96  4     456.7   +4636.06   alpha_reversal
2021-12-30  142.11   2022-01-05  147.60  4     1102.8  +6049.24   alpha_reversal
2021-12-29  145.53   2022-01-05  136.58  5     1013.1  -9065.22   stop_loss
2021-12-31  173.85   2022-01-06  168.22  4     767.9   -4316.66   trailing_stop
2021-12-28  170.75   2022-01-06  163.17  7     661.6   -5010.67   stop_loss
2022-01-04  44.86    2022-01-06  45.29   2     4071.6  +1770.54   alpha_reversal
2022-01-06  303.16   2022-01-11  303.92  3     428.8   +325.12    alpha_reversal
2022-01-06  136.69   2022-01-11  138.52  3     1001.7  +1837.67   alpha_reversal
2022-01-12  307.40   2022-01-13  294.10  1     416.1   -5536.43   stop_loss
2022-01-12  140.34   2022-01-13  137.38  1     957.7   -2834.01   alpha_reversal
2022-01-11  443.36   2022-01-13  437.99  2     538.7   -2890.34   alpha_reversal
2022-01-05  362.89   2022-01-14  349.70  7     145.2   -1915.46   max_holding
2022-01-14  299.61   2022-01-18  292.03  1     376.5   -2855.25   alpha_reversal
2022-01-10  161.57   2022-01-18  158.84  5     734.8   -2005.02   alpha_reversal
2022-01-19  292.97   2022-01-20  291.01  1     359.7   -705.87    alpha_reversal
2022-01-12  150.30   2022-01-20  146.13  5     1431.9  -5963.69   stop_loss
2022-01-12  45.31    2022-01-20  44.57   5     4162.9  -3105.00   alpha_reversal
2022-01-19  426.37   2022-01-21  412.96  2     481.1   -6450.85   stop_loss
2022-01-21  142.71   2022-01-25  139.92  2     621.7   -1739.75   alpha_reversal
2022-01-24  129.80   2022-01-25  125.83  1     752.1   -2982.09   alpha_reversal
2022-01-24  415.13   2022-01-25  409.65  1     367.2   -2011.57   alpha_reversal
2022-01-21  130.88   2022-01-27  130.95  4     818.5   +62.63     alpha_reversal
2022-01-26  128.24   2022-01-27  127.88  1     702.2   -248.70    alpha_reversal
2022-01-25  307.16   2022-01-27  306.38  2     252.1   -195.13    alpha_reversal
2022-01-26  42.88    2022-01-28  43.40   2     3264.0  +1683.31   alpha_reversal
2022-01-26  409.03   2022-01-31  424.21  3     328.2   +4981.51   alpha_reversal
2022-02-02  172.15   2022-02-03  169.10  1     594.4   -1811.59   alpha_reversal
2022-01-26  138.94   2022-02-03  138.78  6     525.8   -87.20     trailing_stop
2022-02-02  302.04   2022-02-04  307.62  2     129.9   +725.00    alpha_reversal
2022-02-03  44.54    2022-02-04  43.97   1     3301.0  -1876.24   alpha_reversal
2022-01-26  114.72   2022-02-04  112.68  7     614.4   -1248.91   max_holding
2022-02-02  207.62   2022-02-07  211.81  3     341.0   +1428.78   alpha_reversal
2022-02-03  134.14   2022-02-08  140.54  3     751.7   +4810.84   alpha_reversal
2022-02-03  186.77   2022-02-08  187.43  3     477.9   +315.28    alpha_reversal
2022-02-07  168.27   2022-02-09  172.63  2     622.3   +2710.84   alpha_reversal
2022-02-08  307.49   2022-02-09  310.51  1     140.7   +425.48    alpha_reversal
2022-02-08  425.60   2022-02-09  431.40  1     339.8   +1969.73   alpha_reversal
2022-02-10  159.08   2022-02-11  153.22  1     444.0   -2604.48   alpha_reversal
2022-02-02  152.93   2022-02-11  148.31  7     1072.0  -4960.52   stop_loss
2022-02-09  67.20    2022-02-14  67.08   3     1901.2  -227.76    alpha_reversal
2022-02-07  43.58    2022-02-14  42.27   5     3313.4  -4337.35   stop_loss
2022-02-10  137.55   2022-02-15  135.42  3     637.1   -1358.02   alpha_reversal
2022-02-08  294.16   2022-02-17  281.10  7     304.5   -3976.15   trailing_stop
2022-02-15  217.84   2022-02-17  213.47  2     354.2   -1546.42   alpha_reversal
2022-02-15  327.29   2022-02-18  310.89  3     300.0   -4922.19   stop_loss
2022-02-15  68.33    2022-02-22  66.63   4     1908.3  -3244.02   alpha_reversal
2022-02-15  116.06   2022-02-22  108.86  4     671.8   -4837.95   stop_loss
2022-02-17  165.55   2022-02-23  156.76  3     664.0   -5838.98   stop_loss
2022-02-14  146.59   2022-02-24  140.74  7     1169.3  -6835.40   trailing_stop
2022-02-23  66.60    2022-02-24  64.49   1     2032.1  -4288.97   stop_loss
2022-02-25  147.89   2022-02-28  146.46  1     874.0   -1241.56   alpha_reversal
2022-02-25  133.49   2022-03-01  122.97  2     644.9   -6780.91   stop_loss
2022-02-24  267.06   2022-03-01  287.98  3     131.2   +2745.27   alpha_reversal
2022-02-25  103.67   2022-03-02  102.06  3     596.9   -962.17    alpha_reversal
2022-02-28  205.44   2022-03-03  188.76  3     315.9   -5270.63   stop_loss
2022-02-25  174.17   2022-03-03  181.24  4     513.1   +3628.68   alpha_reversal
2022-02-25  67.02    2022-03-04  68.28   5     1668.2  +2098.07   alpha_reversal
2022-03-03  44.00    2022-03-04  45.07   1     2816.1  +3013.31   alpha_reversal
2022-03-02  125.65   2022-03-07  116.45  3     585.1   -5383.21   stop_loss
2022-02-24  305.94   2022-03-07  290.89  7     285.1   -4288.03   trailing_stop
2022-03-03  101.40   2022-03-07  92.45   2     617.3   -5522.60   stop_loss
2022-03-04  280.54   2022-03-08  266.72  2     279.7   -3867.46   stop_loss
2022-02-28  161.86   2022-03-08  154.18  6     546.0   -4194.72   stop_loss
2022-03-04  180.93   2022-03-09  178.47  3     296.5   -729.39    alpha_reversal
2022-03-08  93.21    2022-03-09  97.72   1     564.4   +2548.80   alpha_reversal
2022-03-09  159.74   2022-03-11  151.53  2     487.2   -3999.44   stop_loss
2022-03-09  139.35   2022-03-11  145.45  2     394.1   +2405.41   alpha_reversal
2022-03-04  130.89   2022-03-15  128.07  7     612.9   -1725.71   max_holding
2022-03-14  45.50    2022-03-15  46.01   1     2512.6  +1257.51   alpha_reversal
2022-03-14  152.95   2022-03-16  155.32  2     785.0   +1857.15   alpha_reversal
2022-03-10  178.48   2022-03-17  190.09  5     266.1   +3091.37   alpha_reversal
2022-03-09  279.23   2022-03-18  290.48  7     244.7   +2754.69   max_holding
2022-03-17  45.99    2022-03-18  46.08   1     2581.8  +233.02    alpha_reversal
2022-03-09  403.40   2022-03-18  420.43  7     249.9   +4256.60   max_holding
2022-03-10  96.90    2022-03-18  99.85   6     569.8   +1682.27   alpha_reversal
2022-03-15  152.03   2022-03-22  165.32  5     476.7   +6336.51   alpha_reversal
2022-03-17  157.51   2022-03-22  155.75  3     770.1   -1354.87   alpha_reversal
2022-03-16  280.22   2022-03-22  331.16  4     138.0   +7030.62   alpha_reversal
2022-03-21  208.29   2022-03-22  207.16  1     394.7   -449.27    alpha_reversal
2022-03-22  425.65   2022-03-23  419.75  1     296.6   -1750.70   alpha_reversal
2022-03-21  125.98   2022-03-28  126.95  5     613.2   +597.00    alpha_reversal
2022-03-17  132.81   2022-03-28  140.23  7     573.3   +4252.13   alpha_reversal
2022-03-23  155.32   2022-03-28  158.27  3     879.3   +2594.63   alpha_reversal
2022-03-22  45.60    2022-03-28  46.25   4     2834.1  +1846.09   alpha_reversal
2022-03-23  206.86   2022-03-28  206.36  3     432.8   -218.32    alpha_reversal
2022-03-29  72.27    2022-03-30  72.93   1     2142.1  +1418.86   alpha_reversal
2022-03-22  294.29   2022-03-31  298.10  7     276.4   +1054.53   max_holding
2022-03-22  191.14   2022-03-31  191.40  7     302.6   +81.32     alpha_reversal
2022-03-30  159.99   2022-03-31  157.73  1     1000.0  -2260.26   alpha_reversal
2022-03-29  205.81   2022-03-31  207.26  2     485.7   +704.20    alpha_reversal
2022-03-31  138.00   2022-04-04  141.73  2     756.7   +2824.20   alpha_reversal
2022-03-24  304.16   2022-04-04  296.60  7     297.0   -2243.19   max_holding
2022-04-01  361.71   2022-04-04  381.63  1     165.3   +3292.93   alpha_reversal
2022-04-04  174.92   2022-04-05  171.44  1     625.8   -2180.91   alpha_reversal
2022-03-29  127.36   2022-04-05  121.06  5     715.8   -4511.13   stop_loss
2022-04-04  205.59   2022-04-05  200.93  1     506.1   -2358.98   alpha_reversal
2022-03-28  99.58    2022-04-05  95.90   6     815.3   -3002.86   trailing_stop
2022-04-04  191.28   2022-04-06  178.63  2     374.7   -4737.52   stop_loss
2022-04-05  293.06   2022-04-06  285.81  1     347.4   -2519.04   alpha_reversal
2022-04-01  158.74   2022-04-06  162.18  3     995.5   +3421.54   alpha_reversal
2022-04-06  168.44   2022-04-08  166.57  2     606.4   -1135.39   alpha_reversal
2022-04-06  289.87   2022-04-11  275.82  3     329.5   -4631.66   stop_loss
2022-04-06  135.50   2022-04-11  127.70  3     728.9   -5680.00   stop_loss
2022-04-08  290.73   2022-04-11  289.87  1     330.4   -284.13    alpha_reversal
2022-04-08  202.08   2022-04-11  200.96  1     513.9   -572.20    alpha_reversal
2022-04-11  162.48   2022-04-13  166.87  2     605.8   +2660.39   alpha_reversal
2022-04-08  175.29   2022-04-13  182.78  3     374.0   +2801.35   alpha_reversal
2022-04-13  49.86    2022-04-14  49.76   1     2550.8  -240.23    alpha_reversal
2022-04-12  272.99   2022-04-19  275.85  4     306.2   +875.72    alpha_reversal
2022-04-11  151.20   2022-04-19  158.04  5     506.0   +3460.80   alpha_reversal
2022-04-12  126.73   2022-04-19  128.88  4     683.1   +1467.19   alpha_reversal
2022-04-11  325.47   2022-04-19  342.55  5     144.2   +2461.16   alpha_reversal
2022-04-18  414.65   2022-04-19  420.92  1     365.5   +2292.45   alpha_reversal
2022-04-07  94.14    2022-04-19  93.05   7     858.2   -939.85    alpha_reversal
2022-04-18  161.81   2022-04-20  163.77  2     593.5   +1159.47   alpha_reversal
2022-04-19  49.99    2022-04-20  50.57   1     2748.4  +1586.67   alpha_reversal
2022-04-20  119.58   2022-04-22  115.13  2     790.8   -3519.07   alpha_reversal
2022-04-21  414.73   2022-04-22  402.95  1     363.0   -4277.53   stop_loss
2022-04-21  163.14   2022-04-26  153.55  3     615.0   -5893.91   stop_loss
2022-04-25  405.69   2022-04-26  393.55  1     331.6   -4026.18   stop_loss
2022-04-21  91.46    2022-04-26  86.80   3     882.3   -4117.37   stop_loss
2022-04-19  76.01    2022-04-27  74.71   6     1767.2  -2292.73   alpha_reversal
2022-04-27  293.98   2022-04-28  292.36  1     128.5   -208.95    alpha_reversal
2022-05-02  392.41   2022-05-03  393.81  1     271.4   +381.21    alpha_reversal
2022-04-28  89.18    2022-05-03  87.60   3     800.5   -1262.20   alpha_reversal
2022-04-29  290.40   2022-05-04  317.38  3     122.8   +3313.59   alpha_reversal
2022-05-04  89.90    2022-05-05  86.22   1     822.0   -3027.12   alpha_reversal
2022-05-03  153.66   2022-05-06  148.83  3     320.0   -1546.02   alpha_reversal
2022-04-29  108.47   2022-05-06  112.32  5     745.3   +2869.23   alpha_reversal
2022-05-04  125.99   2022-05-06  114.72  2     359.1   -4049.25   stop_loss
2022-05-03  116.43   2022-05-06  114.74  3     584.6   -987.96    alpha_reversal
2022-05-06  154.40   2022-05-09  149.13  1     433.4   -2285.44   alpha_reversal
2022-05-03  158.83   2022-05-10  157.61  5     823.7   -1010.44   alpha_reversal
2022-05-03  48.36    2022-05-10  47.43   5     2514.5  -2342.24   alpha_reversal
2022-05-05  268.44   2022-05-11  251.92  4     246.0   -4061.76   stop_loss
2022-05-06  389.43   2022-05-11  371.46  3     244.3   -4390.60   stop_loss
2022-05-09  108.84   2022-05-13  113.00  4     340.6   +1415.04   alpha_reversal
2022-05-16  158.65   2022-05-17  159.15  1     791.7   +395.87    alpha_reversal
2022-05-16  81.79    2022-05-17  82.31   1     1248.6  +649.40    alpha_reversal
2022-05-17  111.04   2022-05-18  109.03  1     678.1   -1361.86   alpha_reversal
2022-05-17  283.12   2022-05-18  277.19  1     249.8   -1479.22   alpha_reversal
2022-05-17  386.58   2022-05-18  370.62  1     233.8   -3730.76   stop_loss
2022-05-09  81.76    2022-05-18  84.70   7     756.7   +2224.94   alpha_reversal
2022-05-12  139.95   2022-05-19  134.70  5     384.6   -2018.94   trailing_stop
2022-05-11  129.45   2022-05-20  120.64  7     283.0   -2494.77   alpha_reversal
2022-05-19  278.80   2022-05-20  277.26  1     255.1   -393.87    alpha_reversal
2022-05-13  256.66   2022-05-20  221.19  5     107.7   -3819.53   stop_loss
2022-05-12  247.14   2022-05-23  252.61  7     229.3   +1252.81   max_holding
2022-05-20  368.88   2022-05-24  372.54  2     210.2   +769.30    alpha_reversal
2022-05-20  85.02    2022-05-24  83.01   2     734.6   -1478.17   alpha_reversal
2022-05-19  193.44   2022-05-25  195.63  4     335.5   +734.69    alpha_reversal
2022-05-25  254.67   2022-05-26  257.70  1     236.3   +713.91    alpha_reversal
2022-05-25  284.86   2022-05-26  292.47  1     239.4   +1820.40   alpha_reversal
2022-05-25  376.21   2022-05-26  383.34  1     220.5   +1573.25   alpha_reversal
2022-05-20  135.07   2022-05-27  146.76  5     376.7   +4401.01   alpha_reversal
2022-05-26  199.26   2022-05-31  201.74  2     339.2   +839.03    alpha_reversal
2022-05-27  87.82    2022-05-31  89.16   1     820.9   +1103.10   alpha_reversal
2022-05-24  104.15   2022-06-01  121.62  5     351.1   +6134.80   alpha_reversal
2022-05-20  108.07   2022-06-01  112.90  7     524.0   +2532.46   max_holding
2022-05-27  297.24   2022-06-01  292.64  2     253.4   -1165.14   alpha_reversal
2022-05-23  124.13   2022-06-02  140.43  7     275.0   +4482.32   alpha_reversal
2022-05-23  225.08   2022-06-02  258.20  7     109.5   +3628.76   alpha_reversal
2022-05-23  39.02    2022-06-02  40.54   7     1498.9  +2281.47   max_holding
2022-06-02  148.45   2022-06-03  142.58  1     423.5   -2484.18   alpha_reversal
2022-06-02  119.96   2022-06-03  118.17  1     689.4   -1234.39   alpha_reversal
2022-06-02  90.23    2022-06-03  87.73   1     887.7   -2214.64   alpha_reversal
2022-06-06  291.71   2022-06-07  292.33  1     280.5   +173.28    alpha_reversal
2022-05-26  160.90   2022-06-07  159.74  7     785.6   -914.47    max_holding
2022-05-26  81.78    2022-06-07  80.08   7     1231.5  -2095.30   max_holding
2022-06-07  264.36   2022-06-08  262.07  1     278.3   -637.24    alpha_reversal
2022-06-06  117.90   2022-06-08  116.21  2     722.2   -1219.43   alpha_reversal
2022-06-09  113.88   2022-06-10  108.54  1     752.6   -4021.59   stop_loss
2022-06-08  121.24   2022-06-10  109.60  2     414.3   -4825.27   stop_loss
2022-06-08  286.41   2022-06-10  260.97  2     285.8   -7269.36   stop_loss
2022-06-09  256.88   2022-06-13  234.78  2     287.4   -6348.64   stop_loss
2022-06-09  140.03   2022-06-13  129.34  2     446.8   -4777.49   stop_loss
2022-06-10  127.06   2022-06-13  115.80  1     354.5   -3992.51   stop_loss
2022-06-07  39.27    2022-06-13  37.97   4     1937.0  -2514.95   alpha_reversal
2022-06-09  85.09    2022-06-13  80.04   2     894.1   -4509.77   stop_loss
2022-06-10  77.24    2022-06-14  75.40   2     1398.0  -2568.35   alpha_reversal
2022-06-14  122.22   2022-06-15  133.65  1     287.6   +3288.25   alpha_reversal
2022-06-14  81.66    2022-06-15  83.39   1     788.0   +1365.98   alpha_reversal
2022-06-13  355.03   2022-06-16  346.78  3     218.7   -1804.46   alpha_reversal
2022-06-14  237.18   2022-06-17  240.01  3     250.0   +706.42    alpha_reversal
2022-06-13  257.86   2022-06-17  254.40  4     241.4   -837.11    alpha_reversal
2022-06-16  75.82    2022-06-17  75.51   1     1291.9  -409.15    alpha_reversal
2022-06-16  38.39    2022-06-17  37.61   1     1928.9  -1502.99   alpha_reversal
2022-06-15  107.72   2022-06-21  108.63  3     358.8   +323.54    alpha_reversal
2022-06-21  155.12   2022-06-23  160.92  2     683.1   +3963.05   alpha_reversal
2022-06-21  78.63    2022-06-23  82.09   2     1160.1  +4018.63   alpha_reversal
2022-06-16  182.24   2022-06-23  166.99  4     323.0   -4925.84   stop_loss
2022-06-23  39.34    2022-06-24  39.34   1     1957.1  -14.74     alpha_reversal
2022-06-27  139.07   2022-06-28  134.79  1     452.3   -1934.70   alpha_reversal
2022-06-21  259.22   2022-06-29  275.75  6     235.8   +3898.40   alpha_reversal
2022-06-29  136.68   2022-06-30  134.09  1     444.2   -1154.09   alpha_reversal
2022-06-21  105.27   2022-06-30  102.24  7     620.7   -1879.71   max_holding
2022-06-28  107.45   2022-06-30  106.16  2     351.9   -456.29    alpha_reversal
2022-06-23  79.20    2022-07-01  72.42   6     806.2   -5462.94   trailing_stop
2022-06-27  138.79   2022-07-05  137.63  5     294.4   -340.97    alpha_reversal
2022-06-24  245.83   2022-07-05  232.95  6     125.4   -1614.60   alpha_reversal
2022-06-29  38.80    2022-07-05  39.51   3     2079.6  +1459.97   alpha_reversal
2022-06-24  173.54   2022-07-05  162.40  6     304.4   -3390.13   stop_loss
2022-07-01  136.39   2022-07-06  140.17  2     448.6   +1694.24   alpha_reversal
2022-07-01  109.61   2022-07-07  116.27  3     364.9   +2428.93   alpha_reversal
2022-07-06  159.86   2022-07-07  159.88  1     643.3   +12.45     alpha_reversal
2022-07-01  82.55    2022-07-07  82.99   3     1097.8  +487.40    alpha_reversal
2022-06-27  369.49   2022-07-07  369.50  7     228.3   +2.44      alpha_reversal
2022-07-07  244.67   2022-07-08  250.64  1     140.2   +837.26    alpha_reversal
2022-06-30  108.12   2022-07-11  114.67  6     506.6   +3316.73   alpha_reversal
2022-07-11  256.60   2022-07-12  245.84  1     292.9   -3152.34   stop_loss
2022-07-11  159.91   2022-07-12  157.51  1     729.2   -1749.39   alpha_reversal
2022-07-08  82.87    2022-07-12  83.52   2     1130.8  +733.72    alpha_reversal
2022-07-05  71.66    2022-07-12  74.40   5     746.5   +2045.89   alpha_reversal
2022-07-07  140.04   2022-07-13  143.88  4     296.5   +1137.83   alpha_reversal
2022-07-07  105.18   2022-07-13  102.50  4     644.6   -1726.68   alpha_reversal
2022-07-07  271.77   2022-07-14  256.03  5     245.7   -3866.89   stop_loss
2022-07-13  39.90    2022-07-14  40.64   1     2439.2  +1802.83   alpha_reversal
2022-07-11  111.81   2022-07-15  113.49  4     386.4   +651.97    alpha_reversal
2022-07-13  157.30   2022-07-15  159.64  2     742.4   +1738.54   alpha_reversal
2022-07-06  161.21   2022-07-15  162.05  7     313.0   +262.62    max_holding
2022-07-12  362.11   2022-07-21  378.81  7     259.4   +4331.90   max_holding
2022-07-13  245.17   2022-07-22  252.33  7     280.5   +2008.06   max_holding
2022-07-14  109.52   2022-07-25  106.58  7     519.4   -1527.99   max_holding
2022-07-25  42.02    2022-07-26  38.78   1     2732.1  -8836.44   stop_loss
2022-07-25  171.28   2022-07-26  170.56  1     378.0   -271.00    alpha_reversal
2022-07-19  82.50    2022-07-27  81.41   6     1182.9  -1290.23   alpha_reversal
2022-07-20  105.01   2022-07-29  105.66  7     647.5   +418.34    max_holding
2022-07-26  258.99   2022-08-01  297.13  4     158.7   +6051.90   alpha_reversal
2022-08-02  114.24   2022-08-03  117.05  1     507.0   +1424.64   alpha_reversal
2022-07-27  40.29    2022-08-03  41.49   5     1924.3  +2314.80   alpha_reversal
2022-07-25  81.25    2022-08-03  81.37   7     861.9   +100.24    max_holding
2022-07-28  156.19   2022-08-08  152.45  7     719.8   -2691.37   max_holding
2022-08-02  298.42   2022-08-09  306.07  5     270.6   +2069.80   alpha_reversal
2022-08-08  271.94   2022-08-10  280.24  2     292.9   +2429.66   alpha_reversal
2022-08-05  162.55   2022-08-10  166.21  3     533.5   +1951.34   alpha_reversal
2022-08-02  103.08   2022-08-10  108.44  6     782.6   +4191.48   alpha_reversal
2022-08-03  78.26    2022-08-10  79.59   5     1191.0  +1575.35   alpha_reversal
2022-08-09  152.58   2022-08-12  148.06  3     835.1   -3777.42   stop_loss
2022-08-04  39.97    2022-08-12  42.22   6     1878.5  +4240.69   alpha_reversal
2022-08-05  165.12   2022-08-15  170.38  6     330.5   +1739.26   alpha_reversal
2022-08-08  116.40   2022-08-15  121.02  5     560.5   +2590.67   alpha_reversal
2022-08-11  79.43    2022-08-15  80.84   2     1189.9  +1678.70   alpha_reversal
2022-08-12  283.19   2022-08-16  283.68  2     320.2   +157.63    alpha_reversal
2022-08-11  165.64   2022-08-16  169.93  3     571.3   +2452.69   alpha_reversal
2022-08-15  148.92   2022-08-16  150.09  1     900.7   +1059.97   alpha_reversal
2022-08-05  393.14   2022-08-16  408.17  7     309.9   +4656.34   alpha_reversal
2022-08-15  86.21    2022-08-16  84.60   1     883.7   -1422.63   alpha_reversal
2022-08-08  290.57   2022-08-17  303.84  7     144.6   +1919.39   max_holding
2022-08-17  171.60   2022-08-18  171.03  1     635.7   -358.72    alpha_reversal
2022-08-16  80.92    2022-08-18  81.56   2     1306.1  +838.36    alpha_reversal
2022-08-12  143.62   2022-08-19  138.16  5     408.2   -2229.23   alpha_reversal
2022-08-17  118.63   2022-08-19  116.19  2     697.0   -1699.45   alpha_reversal
2022-08-17  283.21   2022-08-22  269.75  3     365.3   -4917.65   stop_loss
2022-08-18  111.52   2022-08-22  106.86  2     909.3   -4240.72   stop_loss
2022-08-19  151.80   2022-08-22  151.12  1     921.1   -629.80    alpha_reversal
2022-08-19  296.81   2022-08-24  296.95  3     173.3   +23.07     alpha_reversal
2022-08-22  80.25    2022-08-24  80.47   2     933.7   +206.14    alpha_reversal
2022-08-23  392.08   2022-08-25  398.49  2     360.2   +2308.46   alpha_reversal
2022-08-25  82.45    2022-08-26  79.70   1     1015.4  -2786.37   alpha_reversal
2022-08-24  268.12   2022-08-29  257.59  3     389.0   -4093.73   stop_loss
2022-08-24  164.70   2022-08-29  158.49  3     670.2   -4158.47   trailing_stop
2022-08-23  105.91   2022-08-29  104.77  4     863.8   -985.46    alpha_reversal
2022-08-24  112.81   2022-08-29  108.47  3     726.1   -3155.43   trailing_stop
2022-08-26  306.17   2022-08-29  303.61  1     328.3   -840.87    alpha_reversal
2022-08-25  150.86   2022-08-29  146.98  2     957.3   -3709.47   stop_loss
2022-08-24  80.40    2022-08-30  77.52   4     1431.1  -4111.81   stop_loss
2022-08-24  43.14    2022-08-30  42.31   4     2286.5  -1916.71   alpha_reversal
2022-08-30  104.90   2022-08-31  104.17  1     808.3   -588.21    alpha_reversal
2022-08-31  78.47    2022-09-01  76.79   1     923.2   -1557.24   alpha_reversal
2022-08-31  107.39   2022-09-02  106.91  2     645.6   -306.08    alpha_reversal
2022-08-26  288.23   2022-09-02  270.07  5     176.6   -3206.84   alpha_reversal
2022-08-29  382.84   2022-09-02  372.58  4     325.7   -3339.49   stop_loss
2022-08-30  146.61   2022-09-07  147.95  5     916.8   +1221.45   alpha_reversal
2022-09-06  274.56   2022-09-07  283.56  1     161.5   +1453.81   alpha_reversal
2022-09-01  77.84    2022-09-07  77.51   3     1323.8  -433.76    alpha_reversal
2022-08-30  156.22   2022-09-08  151.69  6     567.6   -2569.16   alpha_reversal
2022-09-06  371.55   2022-09-09  386.22  3     294.3   +4318.62   alpha_reversal
2022-08-31  254.19   2022-09-12  258.97  7     342.5   +1635.96   max_holding
2022-09-09  154.71   2022-09-13  151.09  2     533.7   -1932.88   trailing_stop
2022-09-02  151.90   2022-09-13  147.24  6     344.3   -1604.42   trailing_stop
2022-09-06  126.17   2022-09-13  126.76  5     427.1   +249.23    trailing_stop
2022-09-09  109.80   2022-09-13  103.41  2     646.5   -4127.54   stop_loss
2022-09-13  145.62   2022-09-14  148.48  1     786.7   +2247.62   alpha_reversal
2022-09-02  170.36   2022-09-14  173.78  7     404.6   +1385.59   alpha_reversal
2022-09-02  76.17    2022-09-14  74.92   7     869.7   -1080.47   max_holding
2022-09-14  149.33   2022-09-15  149.71  1     339.0   +125.61    alpha_reversal
2022-09-12  78.74    2022-09-15  78.03   3     1363.4  -971.88    alpha_reversal
2022-09-15  304.10   2022-09-16  298.84  1     270.3   -1421.96   alpha_reversal
2022-09-14  245.20   2022-09-20  235.47  4     303.7   -2956.25   stop_loss
2022-09-14  152.68   2022-09-21  150.97  5     454.4   -778.97    alpha_reversal
2022-09-19  144.95   2022-09-21  143.22  2     330.4   -572.96    alpha_reversal
2022-09-14  128.61   2022-09-21  118.48  5     419.5   -4251.44   stop_loss
2022-09-14  104.19   2022-09-21  98.42   5     627.5   -3623.52   stop_loss
2022-09-19  171.56   2022-09-21  161.83  2     397.0   -3864.08   stop_loss
2022-09-14  375.20   2022-09-21  359.95  5     269.9   -4116.55   stop_loss
2022-09-19  301.39   2022-09-22  286.67  3     256.7   -3779.50   stop_loss
2022-09-19  42.94    2022-09-22  42.60   3     2245.7  -770.50    alpha_reversal
2022-09-22  160.67   2022-09-26  153.05  2     347.5   -2649.10   stop_loss
2022-09-26  69.13    2022-09-28  68.23   2     948.4   -854.96    alpha_reversal
2022-09-26  127.40   2022-09-29  125.27  3     294.7   -629.62    alpha_reversal
2022-09-22  357.28   2022-09-29  346.02  5     244.9   -2757.18   stop_loss
2022-09-30  265.38   2022-10-03  242.28  1     133.7   -3089.82   stop_loss
2022-10-03  234.04   2022-10-05  242.02  2     263.6   +2103.84   alpha_reversal
2022-09-26  97.91    2022-10-05  102.01  7     604.0   +2478.96   alpha_reversal
2022-09-26  270.17   2022-10-05  283.08  7     209.8   +2707.24   alpha_reversal
2022-10-04  249.56   2022-10-05  240.69  1     122.7   -1089.05   alpha_reversal
2022-09-29  65.59    2022-10-05  70.45   4     870.7   +4225.29   alpha_reversal
2022-09-29  140.07   2022-10-06  142.83  5     358.6   +988.69    alpha_reversal
2022-09-30  121.14   2022-10-07  129.73  5     281.1   +2412.81   alpha_reversal
2022-10-04  149.49   2022-10-07  144.46  3     648.4   -3266.04   stop_loss
2022-10-06  120.36   2022-10-10  113.61  2     392.7   -2649.55   stop_loss
2022-09-30  341.01   2022-10-10  343.38  6     216.8   +513.30    trailing_stop
2022-10-07  137.72   2022-10-11  136.49  2     376.7   -462.49    alpha_reversal
2022-10-06  238.25   2022-10-11  216.39  3     129.7   -2834.46   stop_loss
2022-10-06  42.09    2022-10-11  42.37   3     1926.2  +527.92    alpha_reversal
2022-10-07  227.72   2022-10-13  227.49  4     252.8   -57.53     alpha_reversal
2022-10-05  100.65   2022-10-13  98.20   6     559.9   -1371.76   alpha_reversal
2022-10-07  276.10   2022-10-13  281.31  4     206.7   +1077.27   alpha_reversal
2022-10-11  341.55   2022-10-14  341.10  3     211.6   -94.42     alpha_reversal
2022-10-13  140.57   2022-10-20  140.82  5     370.3   +93.43     alpha_reversal
2022-10-18  172.92   2022-10-20  169.91  2     316.8   -952.31    alpha_reversal
2022-10-14  106.95   2022-10-21  119.26  5     345.2   +4247.77   alpha_reversal
2022-10-12  217.35   2022-10-21  214.33  7     134.2   -404.63    max_holding
2022-10-18  42.88    2022-10-21  43.69   3     1721.3  +1388.41   alpha_reversal
2022-10-18  99.99    2022-10-24  101.63  4     527.8   +862.94    alpha_reversal
2022-10-17  350.22   2022-10-24  361.36  5     191.6   +2134.51   alpha_reversal
2022-10-21  144.78   2022-10-25  149.61  2     367.5   +1776.84   alpha_reversal
2022-10-26  155.44   2022-10-27  155.38  1     669.8   -43.66     alpha_reversal
2022-10-25  222.53   2022-10-28  228.41  3     129.1   +758.18    alpha_reversal
2022-10-19  60.27    2022-10-28  58.65   7     664.9   -1077.68   max_holding
2022-10-31  93.78    2022-11-02  86.21   2     502.1   -3800.06   stop_loss
2022-10-28  371.41   2022-11-02  357.55  3     217.7   -3018.98   stop_loss
2022-10-31  225.67   2022-11-03  208.08  3     221.2   -3890.42   stop_loss
2022-11-03  215.42   2022-11-04  207.37  1     133.9   -1077.70   alpha_reversal
2022-11-01  45.29    2022-11-08  45.60   5     2076.8  +635.49    alpha_reversal
2022-11-03  354.22   2022-11-08  364.35  3     212.6   +2153.41   alpha_reversal
2022-11-01  58.30    2022-11-08  61.50   5     801.2   +2560.10   alpha_reversal
2022-11-08  222.50   2022-11-10  235.98  2     216.0   +2912.28   alpha_reversal
2022-11-02  92.17    2022-11-11  100.74  7     313.5   +2687.54   alpha_reversal
2022-11-03  82.79    2022-11-14  94.87   7     457.8   +5530.21   max_holding
2022-11-11  88.20    2022-11-15  89.59   2     864.6   +1199.28   alpha_reversal
2022-11-10  220.45   2022-11-15  222.26  3     269.2   +486.55    alpha_reversal
2022-11-07  136.80   2022-11-16  146.37  7     315.8   +3023.05   max_holding
2022-11-07  197.18   2022-11-16  186.83  7     128.2   -1326.89   max_holding
2022-11-11  125.16   2022-11-17  122.48  4     604.7   -1617.97   alpha_reversal
2022-11-14  155.17   2022-11-17  157.68  3     613.9   +1537.79   alpha_reversal
2022-11-16  89.98    2022-11-17  92.03   1     891.1   +1827.50   alpha_reversal
2022-11-18  123.81   2022-11-21  122.95  1     690.1   -589.20    alpha_reversal
2022-11-18  235.17   2022-11-22  238.64  2     250.1   +869.40    alpha_reversal
2022-11-16  220.42   2022-11-22  224.79  4     296.0   +1292.19   alpha_reversal
2022-11-22  147.88   2022-11-23  148.61  1     378.0   +275.08    alpha_reversal
2022-11-22  124.92   2022-11-23  126.12  1     720.1   +868.38    alpha_reversal
2022-11-15  380.46   2022-11-23  383.82  6     223.0   +751.16    alpha_reversal
2022-11-17  172.87   2022-11-28  171.74  6     300.9   -337.69    alpha_reversal
2022-11-25  96.79    2022-11-28  97.55   1     1022.8  +775.10    alpha_reversal
2022-11-21  92.51    2022-11-29  92.37   5     365.0   -48.35     alpha_reversal
2022-11-18  180.28   2022-11-29  180.74  6     154.0   +70.74     alpha_reversal
2022-11-25  223.53   2022-11-30  223.98  3     338.0   +151.85    alpha_reversal
2022-11-28  235.69   2022-12-01  248.05  3     293.2   +3623.54   alpha_reversal
2022-11-28  377.99   2022-12-01  388.55  3     292.5   +3089.45   alpha_reversal
2022-12-02  182.96   2022-12-05  185.01  1     332.2   +679.70    alpha_reversal
2022-12-01  223.52   2022-12-05  220.34  2     357.6   -1136.90   alpha_reversal
2022-11-30  96.59    2022-12-06  88.21   4     438.0   -3671.41   stop_loss
2022-12-02  351.29   2022-12-06  334.64  2     282.4   -4700.50   stop_loss
2022-11-28  75.01    2022-12-06  75.25   6     731.9   +180.49    alpha_reversal
2022-11-28  142.01   2022-12-07  138.65  7     401.0   -1350.91   max_holding
2022-11-28  95.31    2022-12-07  94.11   7     624.2   -746.32    trailing_stop
2022-12-01  49.03    2022-12-07  47.62   4     2214.0  -3120.32   stop_loss
2022-12-06  216.51   2022-12-08  218.78  2     351.3   +799.26    alpha_reversal
2022-12-07  332.22   2022-12-12  334.89  3     254.6   +680.87    alpha_reversal
2022-12-08  173.53   2022-12-12  167.74  2     182.2   -1055.16   alpha_reversal
2022-12-08  241.19   2022-12-13  250.22  3     279.5   +2524.30   alpha_reversal
2022-12-08  122.92   2022-12-13  123.91  3     668.4   +659.07    alpha_reversal
2022-12-13  143.25   2022-12-14  140.88  1     413.7   -978.93    alpha_reversal
2022-12-12  161.56   2022-12-14  163.14  2     852.8   +1348.24   alpha_reversal
2022-12-13  161.03   2022-12-16  150.15  3     173.4   -1886.18   alpha_reversal
2022-12-12  221.03   2022-12-16  220.49  4     354.6   -192.56    alpha_reversal
2022-12-13  76.23    2022-12-16  72.58   3     850.2   -3102.67   stop_loss
2022-12-09  92.12    2022-12-19  87.67   6     654.5   -2908.55   trailing_stop
2022-12-19  120.31   2022-12-20  120.77  1     673.3   +311.02    alpha_reversal
2022-12-19  45.82    2022-12-20  46.17   1     2125.5  +754.27    alpha_reversal
2022-12-19  149.94   2022-12-22  125.29  3     176.6   -4355.03   stop_loss
2022-12-21  73.45    2022-12-22  71.61   1     908.8   -1674.69   alpha_reversal
2022-12-15  322.91   2022-12-27  315.33  7     234.2   -1772.71   max_holding
2022-12-19  159.41   2022-12-27  161.03  5     809.8   +1304.05   alpha_reversal
2022-12-23  46.13    2022-12-27  46.10   1     2163.8  -72.02     alpha_reversal
2022-12-15  372.00   2022-12-27  365.45  7     264.7   -1734.31   max_holding
2022-12-16  132.45   2022-12-28  123.99  7     397.7   -3366.04   stop_loss
2022-12-23  123.21   2022-12-29  121.76  3     173.4   -251.86    alpha_reversal
2022-12-28  160.49   2022-12-30  160.32  2     853.6   -144.72    alpha_reversal
2022-12-29  189.00   2023-01-03  195.29  2     301.0   +1892.78   alpha_reversal
2022-12-30  100.56   2023-01-03  100.63  1     1145.0  +81.90     alpha_reversal
2022-12-28  314.63   2023-01-04  320.62  4     254.5   +1522.83   alpha_reversal
2022-12-28  228.65   2023-01-05  216.52  5     271.6   -3294.67   trailing_stop
2023-01-03  161.88   2023-01-05  162.27  2     863.7   +338.38    alpha_reversal
2023-01-04  113.70   2023-01-05  110.28  1     169.0   -576.79    alpha_reversal
2022-12-27  70.77    2023-01-06  74.26   7     920.8   +3219.49   max_holding
2022-12-28  81.86    2023-01-09  87.32   7     529.9   +2890.74   max_holding
2022-12-28  85.36    2023-01-09  87.25   7     627.9   +1191.27   alpha_reversal
2022-12-28  361.26   2023-01-09  371.64  7     261.9   +2716.42   max_holding
2022-12-30  127.94   2023-01-11  131.32  7     405.7   +1368.85   alpha_reversal
2023-01-03  46.08    2023-01-12  46.42   7     2329.5  +796.25    max_holding
2023-01-03  226.55   2023-01-12  241.66  7     352.5   +5326.83   max_holding
2023-01-13  132.70   2023-01-17  133.73  1     462.8   +475.78    alpha_reversal
2023-01-06  219.29   2023-01-18  229.66  7     248.6   +2579.56   max_holding
2023-01-10  159.12   2023-01-18  154.06  5     843.9   -4270.03   stop_loss
2023-01-13  46.62    2023-01-18  45.15   2     2517.1  -3707.71   stop_loss
2023-01-11  90.82    2023-01-19  92.24   5     652.6   +930.53    alpha_reversal
2023-01-10  206.79   2023-01-20  206.66  7     296.2   -40.51     max_holding
2023-01-10  100.43   2023-01-20  99.55   7     1001.3  -889.26    alpha_reversal
2023-01-10  374.62   2023-01-20  379.32  7     277.9   +1307.15   max_holding
2023-01-19  44.55    2023-01-23  45.72   2     2377.2  +2797.30   alpha_reversal
2023-01-20  125.88   2023-01-25  129.51  3     665.0   +2417.32   alpha_reversal
2023-01-18  239.77   2023-01-25  246.02  5     361.1   +2257.99   alpha_reversal
2023-01-24  98.36    2023-01-27  95.42   3     1001.6  -2946.03   stop_loss
2023-01-23  322.27   2023-01-31  337.32  6     206.6   +3109.44   alpha_reversal
2023-01-25  383.99   2023-02-02  399.35  6     308.6   +4740.36   alpha_reversal
2023-01-26  96.77    2023-02-03  103.87  6     660.8   +4692.10   alpha_reversal
2023-01-31  130.42   2023-02-06  132.12  4     743.3   +1259.27   alpha_reversal
2023-01-26  45.63    2023-02-06  45.09   7     2232.1  -1196.50   max_holding
2023-02-03  206.11   2023-02-07  214.65  2     327.3   +2794.76   alpha_reversal
2023-01-30  147.17   2023-02-08  148.48  7     745.2   +979.25    max_holding
2023-02-07  398.22   2023-02-08  393.47  1     312.6   -1484.12   alpha_reversal
2023-01-30  88.68    2023-02-08  89.68   7     717.4   +721.18    max_holding
2023-02-08  44.99    2023-02-10  46.07   2     2495.8  +2687.82   alpha_reversal
2023-02-09  212.10   2023-02-13  215.54  2     339.6   +1170.15   alpha_reversal
2023-02-08  237.15   2023-02-13  236.23  3     335.1   -309.14    alpha_reversal
2023-02-09  257.01   2023-02-14  265.08  3     259.3   +2092.94   alpha_reversal
2023-02-08  100.10   2023-02-15  101.11  5     456.5   +460.77    alpha_reversal
2023-02-14  209.35   2023-02-15  214.13  1     168.5   +805.26    alpha_reversal
2023-02-15  397.06   2023-02-16  391.20  1     337.6   -1977.90   alpha_reversal
2023-02-15  153.19   2023-02-17  150.30  2     524.6   -1517.02   alpha_reversal
2023-02-09  94.28    2023-02-21  90.99   7     481.5   -1582.25   max_holding
2023-02-15  144.78   2023-02-21  144.42  3     885.2   -315.75    alpha_reversal
2023-02-17  208.41   2023-02-21  197.27  1     167.0   -1860.39   alpha_reversal
2023-02-16  96.54    2023-02-21  98.76   2     1170.7  +2600.77   alpha_reversal
2023-02-15  236.99   2023-02-22  226.46  4     349.7   -3681.89   stop_loss
2023-02-21  382.78   2023-02-24  379.80  3     331.2   -986.61    alpha_reversal
2023-02-23  130.15   2023-02-27  132.34  2     792.2   +1733.37   alpha_reversal
2023-02-21  246.95   2023-02-28  243.53  5     278.4   -952.16    alpha_reversal
2023-02-27  207.73   2023-02-28  205.61  1     169.5   -360.37    alpha_reversal
2023-02-16  86.48    2023-02-28  82.82   7     665.5   -2431.29   max_holding
2023-02-27  200.56   2023-03-01  204.45  2     341.2   +1326.55   alpha_reversal
2023-02-24  88.44    2023-03-02  91.20   4     617.6   +1702.40   alpha_reversal
2023-02-28  96.29    2023-03-02  96.89   2     1064.1  +639.50    alpha_reversal
2023-03-01  84.19    2023-03-02  84.69   1     714.0   +354.19    alpha_reversal
2023-02-28  140.23   2023-03-06  142.19  4     876.9   +1720.76   alpha_reversal
2023-03-01  240.69   2023-03-07  248.15  4     309.6   +2307.25   alpha_reversal
2023-03-01  143.31   2023-03-07  149.36  4     573.6   +3472.30   alpha_reversal
2023-03-03  133.87   2023-03-07  129.05  2     824.2   -3977.38   stop_loss
2023-03-06  100.70   2023-03-07  100.75  1     1048.9  +55.89     alpha_reversal
2023-02-27  381.47   2023-03-07  381.61  6     336.6   +45.81     alpha_reversal
2023-03-01  202.87   2023-03-08  181.91  5     175.0   -3668.86   stop_loss
2023-03-08  128.41   2023-03-09  121.34  1     821.9   -5813.46   stop_loss
2023-03-02  92.18    2023-03-09  92.20   5     606.1   +16.86     alpha_reversal
2023-03-01  321.87   2023-03-09  317.70  6     263.1   -1096.29   alpha_reversal
2023-02-28  45.60    2023-03-09  43.97   7     2163.8  -3538.85   stop_loss
2023-03-08  382.61   2023-03-09  375.18  1     367.4   -2731.52   alpha_reversal
2023-03-09  246.61   2023-03-10  242.72  1     334.0   -1298.81   alpha_reversal
2023-03-09  201.34   2023-03-13  203.27  2     343.6   +662.38    alpha_reversal
2023-03-10  124.54   2023-03-13  122.19  1     596.7   -1407.42   alpha_reversal
2023-03-09  138.38   2023-03-13  139.91  2     916.5   +1398.02   alpha_reversal
2023-03-09  173.01   2023-03-13  174.39  2     177.6   +246.23    alpha_reversal
2023-03-13  92.48    2023-03-14  94.83   1     559.5   +1318.32   alpha_reversal
2023-03-14  125.45   2023-03-15  119.40  1     573.2   -3465.62   stop_loss
2023-03-10  370.14   2023-03-15  373.00  3     314.6   +899.56    alpha_reversal
2023-03-17  44.91    2023-03-20  45.35   1     2508.3  +1098.38   alpha_reversal
2023-03-17  95.01    2023-03-21  96.23   2     776.8   +946.07    alpha_reversal
2023-03-21  267.58   2023-03-22  265.86  1     268.7   -462.82    alpha_reversal
2023-03-13  293.31   2023-03-22  291.31  7     217.7   -435.77    max_holding
2023-03-13  214.12   2023-03-22  209.38  7     284.9   -1349.16   max_holding
2023-03-16  379.92   2023-03-22  377.14  4     279.7   -777.23    alpha_reversal
2023-03-16  121.84   2023-03-24  116.28  6     502.1   -2790.86   stop_loss
2023-03-16  203.29   2023-03-27  200.47  7     257.4   -726.37    max_holding
2023-03-17  139.43   2023-03-27  140.13  6     848.1   +595.02    alpha_reversal
2023-03-22  95.28    2023-03-27  97.49   3     817.1   +1806.93   alpha_reversal
2023-03-23  292.70   2023-03-28  295.74  3     196.7   +598.14    alpha_reversal
2023-03-27  191.91   2023-03-28  189.10  1     183.9   -516.84    alpha_reversal
2023-03-22  44.95    2023-03-28  46.22   4     2573.8  +3272.54   alpha_reversal
2023-03-27  86.77    2023-03-29  88.69   2     748.6   +1438.28   alpha_reversal
2023-03-28  155.48   2023-03-30  159.96  2     540.9   +2425.90   alpha_reversal
2023-03-29  298.16   2023-03-30  298.25  1     218.1   +20.07     alpha_reversal
2023-03-29  386.41   2023-03-30  388.29  1     301.5   +565.07    alpha_reversal
2023-03-29  193.98   2023-03-31  207.36  2     196.6   +2630.32   alpha_reversal
2023-03-30  96.67    2023-03-31  97.00   1     912.8   +302.95    alpha_reversal
2023-03-31  121.43   2023-04-03  121.17  1     576.8   -150.57    alpha_reversal
2023-03-31  89.03    2023-04-03  88.77   1     844.0   -220.37    alpha_reversal
2023-04-03  163.88   2023-04-04  163.18  1     601.7   -418.65    alpha_reversal
2023-03-28  100.25   2023-04-04  103.81  5     620.8   +2208.80   alpha_reversal
2023-04-03  395.66   2023-04-04  393.07  1     334.4   -865.50    alpha_reversal
2023-03-27  98.09    2023-04-05  101.05  7     559.2   +1655.36   alpha_reversal
2023-04-04  119.67   2023-04-06  119.60  2     601.7   -43.74     alpha_reversal
2023-04-05  48.22    2023-04-06  48.54   1     2726.1  +860.07    alpha_reversal
2023-04-05  161.50   2023-04-10  159.64  2     612.7   -1143.18   alpha_reversal
2023-04-06  102.11   2023-04-10  102.12  1     664.0   +5.24      alpha_reversal
2023-04-04  299.95   2023-04-10  301.79  3     246.7   +455.20    alpha_reversal
2023-04-05  392.43   2023-04-10  393.97  2     355.2   +547.32    alpha_reversal
2023-04-04  88.22    2023-04-10  85.12   3     877.9   -2721.51   alpha_reversal
2023-04-10  120.11   2023-04-11  120.58  1     675.6   +318.23    alpha_reversal
2023-04-03  280.73   2023-04-13  282.99  7     302.7   +686.49    alpha_reversal
2023-04-11  212.43   2023-04-13  213.48  2     339.2   +358.51    alpha_reversal
2023-04-11  104.54   2023-04-14  107.92  3     680.5   +2303.27   alpha_reversal
2023-04-04  192.68   2023-04-14  184.91  7     199.1   -1546.93   max_holding
2023-04-11  48.35    2023-04-14  47.79   3     2826.5  -1583.10   alpha_reversal
2023-04-05  203.47   2023-04-17  214.42  7     308.3   +3373.96   alpha_reversal
2023-04-11  158.58   2023-04-18  164.01  5     613.4   +3329.19   alpha_reversal
2023-04-12  97.88    2023-04-18  102.25  4     660.3   +2885.49   alpha_reversal
2023-04-17  48.17    2023-04-19  48.28   2     3001.9  +328.90    alpha_reversal
2023-04-17  282.26   2023-04-20  279.35  3     305.0   -887.23    alpha_reversal
2023-04-19  104.35   2023-04-20  103.76  1     669.5   -397.74    alpha_reversal
2023-04-11  85.41    2023-04-20  85.38   7     862.2   -32.38     max_holding
2023-04-20  104.20   2023-04-21  105.19  1     1061.9  +1051.29   alpha_reversal
2023-04-17  205.13   2023-04-25  202.09  6     320.1   -974.40    alpha_reversal
2023-04-18  103.70   2023-04-25  102.95  5     700.1   -523.67    alpha_reversal
2023-04-20  396.55   2023-04-25  390.58  3     438.8   -2621.92   alpha_reversal
2023-04-26  389.31   2023-04-28  400.05  2     450.1   +4833.56   alpha_reversal
2023-05-01  48.84    2023-05-02  48.66   1     3478.7  -628.85    alpha_reversal
2023-05-02  309.92   2023-05-04  298.36  2     308.9   -3569.24   stop_loss
2023-04-25  160.75   2023-05-04  161.12  7     237.8   +87.75     alpha_reversal
2023-04-26  78.72    2023-05-05  81.25   7     893.9   +2254.55   alpha_reversal
2023-05-02  151.00   2023-05-09  147.21  5     959.2   -3634.19   stop_loss
2023-05-03  392.84   2023-05-09  395.24  4     418.4   +1006.74   alpha_reversal
2023-05-08  197.36   2023-05-10  200.74  2     376.8   +1274.10   alpha_reversal
2023-05-10  397.49   2023-05-11  396.40  1     426.0   -464.25    alpha_reversal
2023-05-11  201.94   2023-05-12  200.60  1     370.3   -496.68    alpha_reversal
2023-05-10  128.18   2023-05-16  126.03  4     687.8   -1482.02   alpha_reversal
2023-05-16  169.93   2023-05-17  170.37  1     708.4   +313.01    alpha_reversal
2023-05-08  303.54   2023-05-17  306.64  7     306.5   +951.71    max_holding
2023-05-15  166.43   2023-05-17  173.77  2     268.8   +1972.76   alpha_reversal
2023-05-10  107.60   2023-05-17  104.63  5     1078.8  -3204.45   stop_loss
2023-05-10  147.91   2023-05-18  144.86  6     1009.0  -3072.95   stop_loss
2023-05-09  204.54   2023-05-18  205.51  7     367.1   +357.12    max_holding
2023-05-19  145.40   2023-05-22  144.47  1     1028.1  -954.22    alpha_reversal
2023-05-18  104.04   2023-05-22  104.39  2     1048.9  +359.52    alpha_reversal
2023-05-22  403.21   2023-05-23  398.28  1     457.8   -2254.03   alpha_reversal
2023-05-23  169.43   2023-05-25  170.67  2     740.0   +918.73    alpha_reversal
2023-05-23  103.38   2023-05-25  102.39  2     1023.9  -1011.33   alpha_reversal
2023-05-17  130.03   2023-05-26  128.48  7     698.8   -1080.83   max_holding
2023-05-24  182.99   2023-05-26  193.07  2     266.2   +2683.37   alpha_reversal
2023-05-23  300.80   2023-05-30  307.25  4     329.9   +2125.19   alpha_reversal
2023-05-25  142.35   2023-06-01  142.33  4     1008.5  -22.76     alpha_reversal
2023-05-26  101.37   2023-06-01  101.14  3     996.9   -228.25    alpha_reversal
2023-05-26  47.35    2023-06-01  47.63   3     2860.7  +779.58    alpha_reversal
2023-06-01  177.85   2023-06-02  178.52  1     729.1   +489.01    alpha_reversal
2023-05-23  203.79   2023-06-02  216.91  7     373.5   +4899.91   max_holding
2023-06-01  296.38   2023-06-05  301.15  2     313.7   +1495.20   alpha_reversal
2023-06-01  325.76   2023-06-06  326.51  3     304.8   +228.80    alpha_reversal
2023-06-05  103.23   2023-06-06  100.30  1     1032.3  -3024.20   stop_loss
2023-06-05  48.45    2023-06-06  48.39   1     2948.7  -161.84    alpha_reversal
2023-06-06  412.10   2023-06-07  410.27  1     450.4   -826.73    alpha_reversal
2023-06-07  212.04   2023-06-08  218.00  1     355.8   +2122.29   alpha_reversal
2023-06-06  306.12   2023-06-08  313.93  2     324.9   +2536.44   alpha_reversal
2023-06-08  121.20   2023-06-12  122.57  2     623.6   +851.68    alpha_reversal
2023-06-02  144.71   2023-06-13  148.04  7     1015.4  +3378.66   max_holding
2023-06-05  93.85    2023-06-13  102.12  6     704.4   +5827.33   alpha_reversal
2023-06-13  133.38   2023-06-14  132.75  1     866.4   -546.34    alpha_reversal
2023-06-07  316.75   2023-06-15  340.62  6     310.3   +7408.54   alpha_reversal
2023-06-06  176.98   2023-06-15  183.52  7     678.5   +4431.58   max_holding
2023-06-08  124.31   2023-06-20  125.72  7     562.8   +790.71    max_holding
2023-06-16  260.67   2023-06-20  274.31  1     227.5   +3104.24   alpha_reversal
2023-06-08  100.69   2023-06-20  101.17  7     911.0   +445.31    max_holding
2023-06-20  182.71   2023-06-21  181.49  1     786.8   -958.75    alpha_reversal
2023-06-16  220.10   2023-06-22  205.51  3     388.4   -5668.52   stop_loss
2023-06-20  309.79   2023-06-22  299.06  2     337.4   -3621.30   stop_loss
2023-06-22  130.22   2023-06-23  129.27  1     617.9   -586.82    alpha_reversal
2023-06-20  122.15   2023-06-23  121.28  3     740.0   -647.84    alpha_reversal
2023-06-22  264.74   2023-06-23  256.47  1     192.4   -1591.00   alpha_reversal
2023-06-20  49.86    2023-06-23  50.23   3     3399.3  +1258.32   alpha_reversal
2023-06-20  422.47   2023-06-23  418.22  3     490.3   -2086.02   alpha_reversal
2023-06-16  100.51   2023-06-26  96.12   5     749.5   -3285.06   stop_loss
2023-06-23  205.51   2023-06-27  209.33  2     376.6   +1435.65   alpha_reversal
2023-06-23  224.61   2023-06-27  230.37  2     354.6   +2041.70   alpha_reversal
2023-06-27  327.71   2023-06-29  327.85  2     301.6   +42.87     alpha_reversal
2023-06-28  119.25   2023-06-29  118.06  1     677.3   -805.85    alpha_reversal
2023-06-29  134.71   2023-06-30  136.46  1     876.9   +1535.71   alpha_reversal
2023-06-27  293.62   2023-06-30  301.83  3     346.1   +2842.86   alpha_reversal
2023-06-28  150.23   2023-06-30  152.44  2     1155.6  +2551.07   alpha_reversal
2023-06-27  250.34   2023-06-30  261.64  3     181.7   +2053.96   alpha_reversal
2023-06-27  421.50   2023-06-30  427.94  3     493.1   +3176.79   alpha_reversal
2023-06-30  118.78   2023-07-05  120.69  2     715.2   +1368.49   alpha_reversal
2023-07-03  305.94   2023-07-05  299.50  1     356.1   -2294.66   alpha_reversal
2023-06-27  49.96    2023-07-05  51.08   5     3413.2  +3821.44   alpha_reversal
2023-06-27  98.11    2023-07-05  96.97   5     779.1   -891.84    alpha_reversal
2023-06-30  333.56   2023-07-06  333.94  3     319.8   +121.80    alpha_reversal
2023-07-03  211.03   2023-07-06  212.51  2     462.1   +687.67    alpha_reversal
2023-06-30  130.43   2023-07-06  128.30  3     632.1   -1345.90   alpha_reversal
2023-07-05  188.95   2023-07-07  188.12  2     789.9   -655.74    alpha_reversal
2023-07-05  150.09   2023-07-07  146.66  2     1199.0  -4111.09   stop_loss
2023-07-03  105.04   2023-07-07  100.05  3     1071.6  -5355.07   stop_loss
2023-07-06  230.49   2023-07-07  234.67  1     417.2   +1740.89   alpha_reversal
2023-07-06  424.87   2023-07-07  423.38  1     544.3   -814.49    alpha_reversal
2023-07-07  129.84   2023-07-10  127.07  1     703.5   -1954.63   alpha_reversal
2023-07-06  95.55    2023-07-10  95.80   2     890.3   +222.70    alpha_reversal
2023-07-07  274.57   2023-07-12  271.85  3     196.6   -533.48    alpha_reversal
2023-07-10  101.03   2023-07-13  97.59   3     1069.6  -3681.57   stop_loss
2023-07-10  115.55   2023-07-14  124.33  4     737.0   +6468.58   alpha_reversal
2023-07-11  185.74   2023-07-17  191.39  4     754.0   +4256.20   alpha_reversal
2023-07-10  147.05   2023-07-17  146.50  5     1177.8  -650.36    alpha_reversal
2023-07-12  50.14    2023-07-17  50.03   3     3297.0  -367.63    alpha_reversal
2023-07-11  325.65   2023-07-18  351.77  5     321.8   +8404.21   alpha_reversal
2023-07-11  128.84   2023-07-20  129.90  7     674.9   +709.08    max_holding
2023-07-11  97.33    2023-07-20  93.96   7     875.6   -2943.55   trailing_stop
2023-07-14  98.62    2023-07-21  101.32  5     1139.0  +3076.20   alpha_reversal
2023-07-21  130.06   2023-07-24  128.74  1     670.2   -890.91    alpha_reversal
2023-07-14  213.23   2023-07-25  214.01  7     451.4   +354.96    max_holding
2023-07-19  50.00    2023-07-25  51.42   4     3721.5  +5284.55   alpha_reversal
2023-07-24  190.36   2023-07-26  191.89  2     729.7   +1121.07   alpha_reversal
2023-07-27  95.49    2023-07-31  95.20   2     889.2   -255.62    alpha_reversal
2023-07-25  265.41   2023-08-01  260.94  5     188.1   -841.58    alpha_reversal
2023-07-31  333.35   2023-08-02  328.55  2     332.1   -1592.83   alpha_reversal
2023-08-01  155.70   2023-08-02  156.48  1     802.8   +629.22    alpha_reversal
2023-07-28  441.55   2023-08-02  434.55  3     572.2   -4003.69   stop_loss
2023-07-25  129.19   2023-08-03  128.85  7     707.7   -247.02    alpha_reversal
2023-08-01  51.46    2023-08-03  51.45   2     3769.5  -11.21     alpha_reversal
2023-08-02  146.95   2023-08-04  147.39  2     924.4   +405.65    alpha_reversal
2023-08-02  254.24   2023-08-04  253.73  2     206.0   -103.84    alpha_reversal
2023-07-31  329.03   2023-08-07  323.02  5     253.9   -1526.73   alpha_reversal
2023-08-07  159.58   2023-08-08  159.49  1     759.5   -65.14     alpha_reversal
2023-08-01  94.58    2023-08-08  90.73   5     949.8   -3655.66   stop_loss
2023-08-09  315.62   2023-08-11  314.11  2     278.3   -419.97    alpha_reversal
2023-08-04  231.48   2023-08-11  235.60  5     401.6   +1657.33   alpha_reversal
2023-08-07  130.52   2023-08-11  128.43  4     651.8   -1357.97   alpha_reversal
2023-08-03  433.74   2023-08-11  430.23  6     567.3   -1993.07   alpha_reversal
2023-08-11  146.05   2023-08-14  146.21  1     909.5   +142.17    alpha_reversal
2023-08-10  158.72   2023-08-14  159.73  2     756.8   +765.08    alpha_reversal
2023-08-11  98.71    2023-08-14  99.96   1     1195.3  +1494.85   alpha_reversal
2023-08-11  52.32    2023-08-14  51.88   1     3583.1  -1581.48   alpha_reversal
2023-08-14  317.39   2023-08-15  314.94  1     299.0   -732.63    alpha_reversal
2023-08-11  175.82   2023-08-15  175.31  2     651.0   -333.07    alpha_reversal
2023-08-14  130.32   2023-08-15  128.65  1     704.2   -1173.76   alpha_reversal
2023-08-08  249.82   2023-08-15  232.84  5     208.9   -3547.92   stop_loss
2023-08-14  140.64   2023-08-16  135.00  2     571.1   -3219.83   stop_loss
2023-08-10  318.86   2023-08-16  307.98  4     316.3   -3441.98   stop_loss
2023-08-16  99.90    2023-08-17  99.97   1     1209.3  +90.13     alpha_reversal
2023-08-09  90.33    2023-08-17  87.99   6     942.4   -2202.51   alpha_reversal
2023-08-15  159.43   2023-08-18  158.86  3     757.4   -434.56    alpha_reversal
2023-08-18  421.82   2023-08-21  424.14  1     462.4   +1072.74   alpha_reversal
2023-08-16  225.71   2023-08-22  233.07  4     210.5   +1549.40   alpha_reversal
2023-08-24  217.42   2023-08-28  226.95  2     384.6   +3664.64   alpha_reversal
2023-08-17  306.80   2023-08-28  305.04  7     313.1   -550.14    max_holding
2023-08-25  89.48    2023-08-28  90.26   1     912.2   +706.69    alpha_reversal
2023-08-21  134.75   2023-08-30  135.00  7     584.9   +149.22    max_holding
2023-08-29  101.06   2023-08-30  101.16  1     1131.8  +114.25    alpha_reversal
2023-08-21  262.61   2023-08-30  271.56  7     334.6   +2995.12   max_holding
2023-08-22  153.05   2023-08-31  149.98  7     694.8   -2132.40   max_holding
2023-08-23  139.36   2023-09-01  138.70  7     851.2   -560.82    max_holding
2023-09-01  322.60   2023-09-05  327.07  1     333.3   +1490.54   alpha_reversal
2023-08-29  227.36   2023-09-06  217.84  5     378.1   -3600.41   stop_loss
2023-09-05  256.62   2023-09-06  251.79  1     190.8   -920.33    alpha_reversal
2023-09-01  100.92   2023-09-06  97.74   2     1206.9  -3832.83   stop_loss
2023-08-31  89.93    2023-09-07  86.46   4     1020.3  -3540.08   stop_loss
2023-09-07  134.22   2023-09-08  135.19  1     688.0   +671.50    alpha_reversal
2023-09-01  149.02   2023-09-08  148.94  4     745.5   -55.71     alpha_reversal
2023-09-07  99.17    2023-09-08  100.09  1     1118.7  +1028.91   alpha_reversal
2023-09-08  271.78   2023-09-11  271.29  1     367.6   -181.18    alpha_reversal
2023-09-06  431.21   2023-09-11  432.93  3     463.8   +798.56    alpha_reversal
2023-08-31  309.53   2023-09-12  312.23  7     356.3   +962.31    max_holding
2023-09-11  273.72   2023-09-12  267.35  1     169.1   -1077.51   alpha_reversal
2023-09-12  100.16   2023-09-13  98.95   1     1162.3  -1407.24   alpha_reversal
2023-09-13  135.66   2023-09-14  136.90  1     745.3   +925.92    alpha_reversal
2023-09-06  137.08   2023-09-15  140.58  7     916.5   +3207.62   alpha_reversal
2023-09-12  271.33   2023-09-15  268.50  3     374.6   -1059.98   alpha_reversal
2023-09-07  175.59   2023-09-18  175.82  7     525.9   +120.72    max_holding
2023-09-08  211.38   2023-09-19  204.38  7     370.9   -2595.62   max_holding
2023-09-18  99.74    2023-09-19  99.21   1     1222.9  -641.74    alpha_reversal
2023-09-18  270.49   2023-09-19  269.00  1     363.9   -542.92    alpha_reversal
2023-09-18  430.22   2023-09-19  428.90  1     513.2   -678.07    alpha_reversal
2023-09-08  86.16    2023-09-19  85.10   7     938.6   -993.21    max_holding
2023-09-15  140.46   2023-09-20  135.22  3     602.8   -3157.22   stop_loss
2023-09-19  266.63   2023-09-20  262.46  1     188.8   -788.22    alpha_reversal
2023-09-18  322.99   2023-09-21  313.33  3     346.3   -3348.37   stop_loss
2023-09-20  151.28   2023-09-21  149.96  1     830.5   -1088.58   alpha_reversal
2023-09-18  53.04    2023-09-21  52.50   3     3409.3  -1838.81   alpha_reversal
2023-09-20  202.47   2023-09-26  195.54  4     433.9   -3006.44   alpha_reversal
2023-09-25  247.11   2023-09-26  244.00  1     181.3   -564.82    alpha_reversal
2023-09-22  417.41   2023-09-28  415.15  4     450.8   -1017.80   alpha_reversal
2023-09-22  311.17   2023-10-02  315.55  6     309.0   +1355.03   alpha_reversal
2023-09-21  129.39   2023-10-02  129.40  7     534.7   +0.33      max_holding
2023-09-26  147.66   2023-10-02  143.93  4     885.6   -3310.05   stop_loss
2023-09-27  240.62   2023-10-02  251.47  3     181.9   +1974.19   alpha_reversal
2023-09-22  129.25   2023-10-03  131.28  7     700.1   +1422.47   alpha_reversal
2023-10-02  300.83   2023-10-03  288.85  1     326.9   -3917.13   stop_loss
2023-09-29  191.78   2023-10-04  186.64  3     439.8   -2260.02   alpha_reversal
2023-10-02  414.39   2023-10-09  418.80  5     414.2   +1826.72   alpha_reversal
2023-10-02  94.86    2023-10-10  95.75   6     1239.5  +1096.66   alpha_reversal
2023-10-05  186.38   2023-10-11  195.97  4     413.0   +3960.04   alpha_reversal
2023-10-02  135.95   2023-10-11  139.09  7     821.8   +2573.35   alpha_reversal
2023-10-06  128.02   2023-10-11  131.76  3     503.4   +1882.68   alpha_reversal
2023-10-10  87.54    2023-10-11  88.80   1     1042.5  +1307.40   alpha_reversal
2023-10-11  177.81   2023-10-12  178.53  1     571.6   +412.30    alpha_reversal
2023-10-04  291.48   2023-10-13  291.85  7     296.3   +109.41    max_holding
2023-10-11  95.70    2023-10-13  96.12   2     1322.6  +545.69    alpha_reversal
2023-10-09  50.58    2023-10-16  52.27   5     2405.9  +4067.28   alpha_reversal
2023-10-05  251.10   2023-10-16  260.46  7     325.0   +3041.49   max_holding
2023-10-11  139.47   2023-10-17  138.51  4     662.2   -637.19    alpha_reversal
2023-10-10  263.75   2023-10-17  254.72  5     181.7   -1640.83   alpha_reversal
2023-10-11  145.03   2023-10-18  141.68  5     948.1   -3171.77   stop_loss
2023-10-18  324.02   2023-10-20  320.33  2     307.3   -1135.92   alpha_reversal
2023-10-18  417.21   2023-10-20  408.05  2     370.1   -3388.18   stop_loss
2023-10-19  92.90    2023-10-23  95.51   2     1254.4  +3268.31   alpha_reversal
2023-10-16  185.02   2023-10-24  182.27  6     391.0   -1076.54   alpha_reversal
2023-10-17  175.19   2023-10-25  169.03  6     604.1   -3716.21   stop_loss
2023-10-24  140.43   2023-10-26  138.22  2     917.2   -2026.20   alpha_reversal
2023-10-24  282.59   2023-10-27  273.56  3     320.4   -2893.09   alpha_reversal
2023-10-19  243.49   2023-10-27  230.65  6     294.0   -3775.31   stop_loss
2023-10-23  407.75   2023-10-27  397.87  4     356.3   -3521.45   trailing_stop
2023-10-27  166.35   2023-10-30  168.23  1     591.2   +1110.72   alpha_reversal
2023-10-25  212.53   2023-10-30  197.26  3     175.3   -2676.41   alpha_reversal
2023-10-30  130.91   2023-11-07  137.05  6     725.2   +4452.87   alpha_reversal
2023-10-30  123.50   2023-11-07  129.83  6     505.9   +3202.39   alpha_reversal
2023-10-30  136.53   2023-11-07  139.98  6     833.4   +2878.03   alpha_reversal
2023-10-30  83.15    2023-11-07  89.19   6     805.4   +4868.99   alpha_reversal
2023-11-08  191.93   2023-11-09  193.23  1     417.7   +546.05    alpha_reversal
2023-11-08  137.86   2023-11-09  137.32  1     875.8   -479.07    alpha_reversal
2023-10-31  200.94   2023-11-09  209.88  7     168.9   +1509.28   max_holding
2023-11-08  88.88    2023-11-13  93.06   3     949.6   +3966.77   alpha_reversal
2023-11-10  214.76   2023-11-14  237.29  2     185.1   +4171.01   alpha_reversal
2023-11-03  95.63    2023-11-14  94.42   7     1030.1  -1250.30   max_holding
2023-11-03  53.44    2023-11-14  54.36   7     2600.1  +2381.75   max_holding
2023-11-06  230.61   2023-11-15  242.76  7     262.1   +3185.37   max_holding
2023-11-14  141.41   2023-11-17  145.43  3     893.2   +3596.85   alpha_reversal
2023-11-08  139.61   2023-11-17  138.95  7     976.9   -643.83    max_holding
2023-11-13  426.88   2023-11-17  436.73  4     389.9   +3837.59   alpha_reversal
2023-11-09  140.67   2023-11-20  146.06  7     533.3   +2872.54   max_holding
2023-11-20  371.23   2023-11-21  366.57  1     304.4   -1420.02   alpha_reversal
2023-11-14  185.61   2023-11-21  188.59  5     617.8   +1841.09   alpha_reversal
2023-11-20  140.32   2023-11-21  141.31  1     1057.0  +1047.68   alpha_reversal
2023-11-14  132.59   2023-11-22  137.29  6     670.4   +3147.86   alpha_reversal
2023-11-22  189.44   2023-11-24  187.92  1     745.6   -1129.45   alpha_reversal
2023-11-22  441.27   2023-11-24  441.10  1     480.4   -81.56     alpha_reversal
2023-11-24  142.74   2023-11-27  141.46  1     1108.0  -1422.13   alpha_reversal
2023-11-24  235.57   2023-11-27  235.96  1     202.0   +79.64     alpha_reversal
2023-11-15  93.75    2023-11-27  93.73   7     1183.0  -23.41     alpha_reversal
2023-11-27  372.38   2023-11-28  376.03  1     334.4   +1219.29   alpha_reversal
2023-11-21  241.38   2023-11-28  239.29  4     362.3   -756.52    alpha_reversal
2023-11-27  187.93   2023-11-29  187.33  2     785.6   -473.97    alpha_reversal
2023-11-27  135.36   2023-11-29  133.82  2     784.3   -1210.13   alpha_reversal
2023-11-29  93.55    2023-11-30  94.70   1     1372.1  +1583.51   alpha_reversal
2023-11-28  146.26   2023-12-01  149.26  3     1225.9  +3670.75   alpha_reversal
2023-11-28  318.92   2023-12-04  332.39  4     395.7   +5328.03   alpha_reversal
2023-11-24  50.65    2023-12-05  50.49   7     2477.8  -390.58    max_holding
2023-11-30  188.09   2023-12-06  190.25  4     802.3   +1730.10   alpha_reversal
2023-11-27  93.92    2023-12-06  94.44   7     1089.7  +570.82    max_holding
2023-11-28  441.18   2023-12-07  443.93  7     545.2   +1502.73   max_holding
2023-11-29  146.39   2023-12-08  147.35  7     638.5   +608.54    max_holding
2023-11-30  131.51   2023-12-08  133.82  6     748.6   +1727.08   alpha_reversal
2023-11-30  240.20   2023-12-08  243.72  6     205.5   +723.12    alpha_reversal
2023-12-04  363.07   2023-12-13  367.85  7     323.0   +1542.79   max_holding
2023-12-12  192.80   2023-12-14  195.98  2     765.8   +2428.12   alpha_reversal
2023-12-08  144.54   2023-12-14  146.69  4     1142.5  +2452.27   alpha_reversal
2023-12-14  359.91   2023-12-18  366.16  2     332.4   +2075.16   alpha_reversal
2023-12-14  147.49   2023-12-18  153.99  2     666.1   +4329.24   alpha_reversal
2023-12-08  49.14    2023-12-18  50.43   6     2814.7  +3626.50   alpha_reversal
2023-12-13  99.08    2023-12-19  99.12   4     1352.3  +54.90     alpha_reversal
2023-12-18  99.91    2023-12-19  100.72  1     1275.5  +1035.26   alpha_reversal
2023-12-11  132.26   2023-12-20  137.14  7     683.3   +3330.50   max_holding
2023-12-12  237.13   2023-12-20  247.02  6     246.9   +2441.70   alpha_reversal
2023-12-18  459.56   2023-12-20  455.49  2     596.4   -2426.05   alpha_reversal
2023-12-20  152.20   2023-12-21  153.76  1     678.3   +1062.91   alpha_reversal
2023-12-21  50.43    2023-12-22  50.98   1     3237.2  +1785.82   alpha_reversal
2023-12-18  145.49   2023-12-26  146.00  5     1134.2  +577.38    alpha_reversal
2023-12-20  97.11    2023-12-26  101.29  3     1230.8  +5143.88   alpha_reversal
2023-12-26  160.41   2023-12-27  161.21  1     1199.4  +960.52    alpha_reversal
2023-12-26  256.74   2023-12-27  261.31  1     272.3   +1244.72   alpha_reversal
2023-12-18  193.97   2023-12-28  191.49  7     792.3   -1964.11   max_holding
2023-12-27  100.61   2023-12-28  101.24  1     1585.7  +1006.56   alpha_reversal
2023-12-27  51.43    2023-12-28  51.28   1     3505.1  -533.80    alpha_reversal
2023-12-28  369.11   2023-12-29  369.49  1     468.0   +176.82    alpha_reversal
2023-12-26  153.49   2023-12-29  151.86  3     779.3   -1264.52   alpha_reversal
2023-12-26  262.92   2024-01-02  251.63  4     445.6   -5029.42   stop_loss
2023-12-29  51.36    2024-01-02  51.84   1     3785.7  +1826.15   alpha_reversal
2023-12-29  286.15   2024-01-03  274.87  2     443.1   -4997.29   stop_loss
2023-12-21  460.27   2024-01-03  456.01  7     568.8   -2424.86   trailing_stop
2023-12-28  139.15   2024-01-04  135.20  4     870.4   -3434.24   alpha_reversal
2024-01-02  369.77   2024-01-05  367.63  3     397.8   -850.83    alpha_reversal
2024-01-04  245.06   2024-01-08  228.89  2     425.7   -6887.11   stop_loss
2024-01-03  238.57   2024-01-09  234.84  4     268.2   -999.51    alpha_reversal
2024-01-02  98.55    2024-01-09  98.80   5     1292.6  +323.90    alpha_reversal
2024-01-08  137.77   2024-01-10  141.04  2     835.6   +2734.32   alpha_reversal
2024-01-03  182.45   2024-01-11  183.59  6     809.1   +924.98    alpha_reversal
2024-01-05  51.05    2024-01-11  52.43   4     3566.4  +4924.81   alpha_reversal
2024-01-04  454.99   2024-01-11  463.36  5     610.3   +5106.97   alpha_reversal
2024-01-05  145.31   2024-01-12  154.54  5     788.1   +7274.59   alpha_reversal
2024-01-10  234.06   2024-01-12  218.78  2     286.7   -4380.30   stop_loss
2024-01-09  225.87   2024-01-16  200.42  4     338.6   -8618.77   stop_loss
2024-01-12  152.00   2024-01-16  150.10  1     1314.2  -2497.59   alpha_reversal
2024-01-12  110.53   2024-01-16  110.29  1     1438.6  -346.40    alpha_reversal
2024-01-12  52.55    2024-01-16  52.66   1     3862.1  +438.17    alpha_reversal
2024-01-12  281.06   2024-01-17  269.39  2     428.4   -4999.06   stop_loss
2024-01-23  392.34   2024-01-25  397.81  2     393.3   +2152.63   alpha_reversal
2024-01-23  156.10   2024-01-25  157.67  2     764.7   +1202.95   alpha_reversal
2024-01-23  145.91   2024-01-25  150.55  2     834.7   +3874.71   alpha_reversal
2024-01-16  220.02   2024-01-25  182.54  7     274.2   -10278.21  stop_loss
2024-01-18  272.82   2024-01-25  292.14  5     395.7   +7643.78   alpha_reversal
2024-01-17  203.16   2024-01-26  205.37  7     317.7   +700.85    max_holding
2024-01-25  53.05    2024-01-29  53.71   2     3603.4  +2388.82   alpha_reversal
2024-01-23  161.97   2024-01-30  168.78  5     904.1   +6155.89   alpha_reversal
2024-01-24  361.30   2024-01-31  365.32  5     319.1   +1284.23   alpha_reversal
2024-01-29  149.16   2024-02-01  148.08  3     1224.7  -1327.83   alpha_reversal
2024-01-30  191.69   2024-02-02  187.82  3     263.1   -1018.26   alpha_reversal
2024-01-25  111.93   2024-02-02  117.66  6     1391.8  +7979.90   alpha_reversal
2024-02-01  54.83    2024-02-02  55.18   1     3508.2  +1246.27   alpha_reversal
2024-02-02  167.48   2024-02-05  167.09  1     916.2   -355.13    alpha_reversal
2024-01-31  470.18   2024-02-05  479.12  3     597.9   +5343.10   alpha_reversal
2024-02-02  369.35   2024-02-07  367.84  3     318.4   -481.02    alpha_reversal
2024-02-02  146.59   2024-02-07  147.72  3     1257.2  +1426.41   alpha_reversal
2024-02-07  407.24   2024-02-08  406.89  1     350.7   -122.07    alpha_reversal
2024-02-07  144.42   2024-02-09  147.70  2     751.3   +2468.55   alpha_reversal
2024-02-06  185.19   2024-02-09  193.47  3     274.8   +2275.71   alpha_reversal
2024-02-09  187.24   2024-02-12  185.37  1     760.3   -1422.54   alpha_reversal
2024-02-09  167.74   2024-02-12  168.32  1     1050.3  +608.34    alpha_reversal
2024-02-09  209.30   2024-02-13  204.36  2     363.5   -1798.25   alpha_reversal
2024-02-13  144.02   2024-02-14  144.67  1     792.6   +514.44    alpha_reversal
2024-02-09  116.89   2024-02-14  117.11  3     1447.5  +329.43    alpha_reversal
2024-02-08  146.39   2024-02-20  148.73  7     1261.7  +2951.96   max_holding
2024-02-09  55.14    2024-02-20  57.23   6     4027.5  +8402.22   alpha_reversal
2024-02-09  365.93   2024-02-21  369.38  7     373.4   +1287.74   max_holding
2024-02-13  184.11   2024-02-22  197.31  6     304.9   +4024.18   alpha_reversal
2024-02-13  481.09   2024-02-22  493.66  6     572.9   +7203.06   alpha_reversal
2024-02-13  399.64   2024-02-23  403.93  7     354.2   +1521.73   max_holding
2024-02-13  183.46   2024-02-23  180.78  7     780.3   -2090.82   max_holding
2024-02-16  139.44   2024-02-26  136.37  5     767.6   -2351.57   trailing_stop
2024-02-15  205.43   2024-02-27  201.30  7     405.6   -1676.62   max_holding
2024-02-16  122.98   2024-02-27  124.70  6     741.8   +1275.48   alpha_reversal
2024-02-16  169.59   2024-02-28  173.07  7     645.9   +2246.93   max_holding
2024-02-27  372.16   2024-02-28  376.70  1     362.9   +1645.21   alpha_reversal
2024-02-27  151.82   2024-02-28  152.21  1     1375.7  +530.03    alpha_reversal
2024-02-27  120.22   2024-02-28  119.32  1     1646.0  -1484.74   alpha_reversal
2024-02-27  493.60   2024-02-28  492.45  1     597.9   -684.61    alpha_reversal
2024-02-26  401.58   2024-02-29  407.18  3     394.6   +2211.14   alpha_reversal
2024-02-26  179.62   2024-03-04  173.43  5     894.0   -5526.79   stop_loss
2024-02-29  176.85   2024-03-04  177.49  2     752.3   +483.59    alpha_reversal
2024-02-27  137.81   2024-03-04  132.19  4     838.2   -4710.17   stop_loss
2024-03-01  152.90   2024-03-04  150.60  1     1491.5  -3432.06   alpha_reversal
2024-02-26  199.50   2024-03-04  188.05  5     322.4   -3692.23   alpha_reversal
2024-02-29  118.47   2024-03-04  115.47  2     1662.3  -4993.41   stop_loss
2024-02-28  123.65   2024-03-04  134.07  3     808.1   +8426.03   alpha_reversal
2024-03-04  200.64   2024-03-06  200.90  2     513.5   +133.12    alpha_reversal
2024-03-04  178.93   2024-03-06  181.48  2     1135.3  +2895.01   alpha_reversal
2024-03-07  176.91   2024-03-08  175.26  1     730.6   -1202.70   alpha_reversal
2024-03-07  203.13   2024-03-11  192.39  2     545.8   -5861.09   stop_loss
2024-03-07  149.83   2024-03-11  151.91  2     1356.4  +2812.78   alpha_reversal
2024-03-01  372.20   2024-03-12  371.91  7     368.2   -108.79    max_holding
2024-03-05  131.65   2024-03-13  138.57  6     816.0   +5651.94   alpha_reversal
2024-03-11  134.94   2024-03-13  137.83  2     474.4   +1371.23   alpha_reversal
2024-03-05  168.67   2024-03-14  171.35  7     746.5   +2003.69   max_holding
2024-03-11  172.05   2024-03-14  178.66  3     699.0   +4623.37   alpha_reversal
2024-03-05  180.83   2024-03-14  162.42  7     299.5   -5514.50   stop_loss
2024-03-05  114.44   2024-03-14  112.88  7     1449.3  -2264.31   max_holding
2024-03-05  493.84   2024-03-14  500.91  7     589.5   +4164.47   max_holding
2024-03-06  396.21   2024-03-15  409.92  7     374.7   +5137.58   alpha_reversal
2024-03-12  327.89   2024-03-15  337.01  3     418.7   +3819.89   alpha_reversal
2024-03-15  182.40   2024-03-18  184.48  1     1005.3  +2088.55   alpha_reversal
2024-03-15  113.94   2024-03-18  113.75  1     1397.0  -263.79    alpha_reversal
2024-03-18  174.57   2024-03-19  175.81  1     752.9   +937.21    alpha_reversal
2024-03-15  497.97   2024-03-19  503.21  2     575.9   +3017.57   alpha_reversal
2024-03-18  172.24   2024-03-20  176.97  2     788.8   +3731.52   alpha_reversal
2024-03-15  371.35   2024-03-20  379.85  3     363.2   +3087.77   alpha_reversal
2024-03-14  150.15   2024-03-20  146.75  4     1295.5  -4405.41   stop_loss
2024-03-14  59.84    2024-03-20  60.00   4     3198.1  +529.29    alpha_reversal
2024-03-12  184.33   2024-03-21  187.61  7     475.0   +1555.19   alpha_reversal
2024-03-20  147.59   2024-03-21  146.32  1     771.2   -985.24    alpha_reversal
2024-03-15  163.65   2024-03-21  172.73  4     325.6   +2957.10   alpha_reversal
2024-03-22  188.94   2024-03-25  191.31  1     545.1   +1291.92   alpha_reversal
2024-03-22  178.96   2024-03-25  179.62  1     800.5   +528.90    alpha_reversal
2024-03-22  149.61   2024-03-25  148.77  1     770.9   -650.21    alpha_reversal
2024-03-19  114.58   2024-03-25  117.38  4     1459.8  +4085.44   alpha_reversal
2024-03-14  135.53   2024-03-25  136.54  7     498.5   +502.65    alpha_reversal
2024-03-22  170.92   2024-03-26  177.58  2     352.8   +2351.68   alpha_reversal
2024-03-18  411.22   2024-03-27  414.85  7     348.8   +1267.70   max_holding
2024-03-26  149.51   2024-03-27  149.56  1     812.2   +39.66     alpha_reversal
2024-03-21  146.89   2024-03-28  149.04  5     1436.8  +3092.13   alpha_reversal
2024-03-28  414.57   2024-04-02  414.86  2     437.0   +128.69    alpha_reversal
2024-03-27  179.92   2024-04-02  180.60  3     841.5   +571.98    alpha_reversal
2024-03-26  59.34    2024-04-02  57.95   4     3960.8  -5473.03   stop_loss
2024-03-21  510.05   2024-04-02  506.26  7     588.7   -2229.99   max_holding
2024-03-26  135.32   2024-04-02  136.53  4     635.8   +768.24    alpha_reversal
2024-03-22  170.81   2024-04-03  168.04  7     749.6   -2080.61   max_holding
2024-04-02  148.76   2024-04-03  145.34  1     1625.7  -5556.75   stop_loss
2024-04-03  138.40   2024-04-04  135.98  1     651.1   -1573.39   alpha_reversal
2024-04-01  175.31   2024-04-05  164.82  4     376.1   -3945.20   alpha_reversal
2024-04-02  188.13   2024-04-09  178.03  5     602.7   -6088.76   stop_loss
2024-04-05  506.37   2024-04-09  506.73  2     575.1   +208.34    alpha_reversal
2024-04-04  411.77   2024-04-10  416.65  4     411.4   +2009.60   alpha_reversal
2024-04-09  168.22   2024-04-10  166.18  1     954.0   -1946.37   alpha_reversal
2024-04-08  173.07   2024-04-10  171.67  2     333.2   -463.91    alpha_reversal
2024-04-09  58.63    2024-04-10  59.38   1     3851.1  +2905.63   alpha_reversal
2024-04-04  143.83   2024-04-11  140.19  5     1402.9  -5105.63   stop_loss
2024-04-12  58.97    2024-04-15  58.71   1     3554.8  -940.81    alpha_reversal
2024-04-09  118.81   2024-04-16  117.14  5     1286.7  -2141.36   alpha_reversal
2024-04-10  174.72   2024-04-19  169.74  7     570.2   -2840.91   alpha_reversal
2024-04-16  183.41   2024-04-19  174.54  3     726.6   -6444.50   stop_loss
2024-04-16  491.81   2024-04-19  483.15  3     468.4   -4055.50   stop_loss
2024-04-18  174.74   2024-04-22  182.42  2     754.1   +5794.56   alpha_reversal
2024-04-16  157.19   2024-04-22  141.98  4     337.8   -5137.44   stop_loss
2024-04-12  355.49   2024-04-23  354.11  7     351.2   -485.02    max_holding
2024-04-24  58.71    2024-04-25  58.98   1     2907.9  +797.91    alpha_reversal
2024-04-19  163.59   2024-04-26  167.69  5     802.6   +3287.06   alpha_reversal
2024-04-24  129.60   2024-04-26  134.66  2     543.7   +2751.33   alpha_reversal
2024-04-19  393.28   2024-04-29  395.97  6     365.8   +983.41    alpha_reversal
2024-04-22  177.32   2024-04-30  174.91  6     660.1   -1588.38   alpha_reversal
2024-04-29  59.07    2024-04-30  58.14   1     3244.2  -3020.04   alpha_reversal
2024-04-22  488.09   2024-04-30  489.81  6     435.1   +747.26    alpha_reversal
2024-04-30  184.85   2024-05-02  184.59  2     806.8   -211.26    alpha_reversal
2024-04-29  138.47   2024-05-02  141.25  3     1207.7  +3360.34   alpha_reversal
2024-05-07  140.26   2024-05-08  140.34  1     1092.4  +83.55     alpha_reversal
2024-05-01  323.07   2024-05-08  335.83  5     262.2   +3348.60   alpha_reversal
2024-05-01  389.16   2024-05-09  405.88  6     286.8   +4795.91   alpha_reversal
2024-05-07  176.80   2024-05-09  181.16  2     493.3   +2151.43   alpha_reversal
2024-05-08  139.96   2024-05-09  139.03  1     611.4   -567.72    alpha_reversal
2024-05-03  119.56   2024-05-10  121.83  5     1302.0  +2954.46   alpha_reversal
2024-05-03  183.67   2024-05-13  191.40  6     777.1   +6009.83   alpha_reversal
2024-05-10  145.47   2024-05-13  142.55  1     610.0   -1781.48   alpha_reversal
2024-05-13  184.94   2024-05-14  185.90  1     701.4   +670.32    alpha_reversal
2024-05-14  410.47   2024-05-15  417.23  1     363.3   +2455.39   alpha_reversal
2024-05-13  178.53   2024-05-15  176.90  2     552.7   -899.69    alpha_reversal
2024-05-07  177.90   2024-05-16  174.75  7     279.3   -878.80    max_holding
2024-05-09  59.47    2024-05-17  63.55   6     3300.8  +13464.06  alpha_reversal
2024-05-13  434.98   2024-05-20  443.53  5     377.6   +3228.92   alpha_reversal
2024-05-20  419.88   2024-05-21  423.10  1     394.7   +1274.58   alpha_reversal
2024-05-16  183.05   2024-05-21  184.69  3     544.5   +890.91    alpha_reversal
2024-05-20  149.65   2024-05-21  149.62  1     683.8   -22.39     alpha_reversal
2024-05-21  192.35   2024-05-22  191.00  1     807.1   -1095.72   alpha_reversal
2024-05-13  186.66   2024-05-22  183.04  7     612.6   -2220.48   max_holding
2024-05-21  186.69   2024-05-22  180.02  1     354.9   -2368.69   alpha_reversal
2024-05-13  121.23   2024-05-22  122.80  7     1434.9  +2258.86   max_holding
2024-05-22  186.37   2024-05-23  172.12  1     601.9   -8576.42   stop_loss
2024-05-22  145.94   2024-05-23  142.18  1     1369.6  -5142.87   stop_loss
2024-05-23  185.54   2024-05-28  188.44  2     863.7   +2504.06   alpha_reversal
2024-05-23  181.14   2024-05-29  181.93  3     681.0   +536.93    alpha_reversal
2024-05-22  347.33   2024-05-29  329.61  4     370.3   -6564.58   stop_loss
2024-05-31  409.80   2024-06-03  407.80  1     351.9   -702.80    alpha_reversal
2024-05-28  137.27   2024-06-03  140.32  4     1204.2  +3677.88   alpha_reversal
2024-05-24  174.61   2024-06-04  188.53  6     501.9   +6985.26   alpha_reversal
2024-06-03  178.43   2024-06-04  179.25  1     711.8   +584.52    alpha_reversal
2024-05-23  439.38   2024-06-04  438.85  7     375.1   -198.75    max_holding
2024-05-30  331.05   2024-06-04  319.33  3     354.1   -4149.33   stop_loss
2024-05-28  118.23   2024-06-05  121.26  6     1436.1  +4350.30   alpha_reversal
2024-06-03  151.02   2024-06-06  157.80  3     653.4   +4431.09   alpha_reversal
2024-06-06  139.21   2024-06-07  139.70  1     1280.6  +624.55    alpha_reversal
2024-06-07  195.48   2024-06-10  191.54  1     944.7   -3716.95   alpha_reversal
2024-05-30  178.88   2024-06-10  173.70  7     357.2   -1848.86   max_holding
2024-06-06  419.07   2024-06-11  426.69  3     374.0   +2852.94   alpha_reversal
2024-06-07  521.58   2024-06-11  523.93  2     585.6   +1374.53   alpha_reversal
2024-06-11  205.66   2024-06-12  211.33  1     714.6   +4048.85   alpha_reversal
2024-06-04  192.01   2024-06-12  184.47  6     839.5   -6330.57   stop_loss
2024-06-10  173.86   2024-06-12  176.45  2     840.6   +2173.07   alpha_reversal
2024-06-11  170.75   2024-06-13  182.38  2     412.2   +4795.70   alpha_reversal
2024-06-05  321.48   2024-06-13  318.15  6     359.4   -1198.04   alpha_reversal
2024-06-13  65.63    2024-06-14  65.88   1     2864.0  +713.06    alpha_reversal
2024-06-07  438.92   2024-06-17  433.92  6     379.4   -1896.35   alpha_reversal
2024-06-12  138.25   2024-06-17  138.62  3     1326.8  +497.14    alpha_reversal
2024-06-13  186.70   2024-06-21  189.06  5     793.4   +1869.20   alpha_reversal
2024-06-17  184.15   2024-06-21  188.99  3     799.2   +3862.75   alpha_reversal
2024-06-20  140.50   2024-06-21  141.28  1     1390.1  +1085.52   alpha_reversal
2024-06-13  121.30   2024-06-21  123.18  5     1389.0  +2605.62   alpha_reversal
2024-06-17  314.60   2024-06-21  319.59  3     385.5   +1923.23   alpha_reversal
2024-06-13  180.79   2024-06-25  175.01  7     544.9   -3148.60   max_holding
2024-06-24  445.84   2024-06-25  440.86  1     357.9   -1784.28   alpha_reversal
2024-06-24  67.80    2024-06-25  66.27   1     3083.8  -4695.49   stop_loss
2024-06-24  322.02   2024-06-25  319.16  1     419.0   -1196.98   alpha_reversal
2024-06-25  186.43   2024-06-26  193.51  1     786.7   +5570.17   alpha_reversal
2024-06-25  182.82   2024-06-26  182.49  1     847.5   -281.02    alpha_reversal
2024-06-21  533.55   2024-06-26  534.00  3     647.0   +288.38    alpha_reversal
2024-06-26  178.59   2024-06-27  182.42  1     568.8   +2178.36   alpha_reversal
2024-06-26  67.20    2024-06-27  66.73   1     2689.6  -1264.69   alpha_reversal
2024-06-24  206.65   2024-06-28  208.90  4     590.1   +1329.50   alpha_reversal
2024-06-27  319.54   2024-06-28  324.72  1     444.4   +2301.27   alpha_reversal
2024-06-24  191.74   2024-07-01  197.87  5     831.6   +5102.82   alpha_reversal
2024-06-26  439.83   2024-07-01  446.91  3     365.1   +2584.17   alpha_reversal
2024-06-28  138.96   2024-07-01  139.09  1     1494.2  +189.82    alpha_reversal
2024-06-21  170.08   2024-07-01  168.32  6     530.9   -935.44    alpha_reversal
2024-07-01  197.30   2024-07-02  199.90  1     662.5   +1723.34   alpha_reversal
2024-07-01  534.37   2024-07-02  537.43  1     690.0   +2110.41   alpha_reversal
2024-07-01  450.86   2024-07-03  454.40  2     419.1   +1480.98   alpha_reversal
2024-06-28  66.63    2024-07-03  67.08   3     2787.6  +1266.69   alpha_reversal
2024-07-02  171.78   2024-07-03  178.24  1     564.5   +3647.09   alpha_reversal
2024-07-03  322.62   2024-07-05  320.09  1     471.3   -1190.40   alpha_reversal
2024-07-03  184.40   2024-07-08  185.75  2     605.3   +814.12    alpha_reversal
2024-07-03  138.51   2024-07-08  138.18  2     1530.2  -517.09    alpha_reversal
2024-07-08  321.10   2024-07-09  317.86  1     474.0   -1538.51   alpha_reversal
2024-06-28  116.77   2024-07-10  119.96  7     1031.8  +3282.51   max_holding
2024-07-10  183.82   2024-07-11  183.82  1     627.5   -2.39      alpha_reversal
2024-07-10  320.54   2024-07-15  337.07  3     484.4   +8009.69   alpha_reversal
2024-07-15  179.20   2024-07-17  184.75  2     654.5   +3631.34   alpha_reversal
2024-07-15  185.31   2024-07-17  179.65  2     901.6   -5097.20   stop_loss
2024-07-12  183.17   2024-07-17  167.21  3     501.8   -8007.87   stop_loss
2024-07-12  447.72   2024-07-18  434.28  4     416.4   -5597.93   stop_loss
2024-07-15  192.82   2024-07-18  183.66  3     729.7   -6682.72   stop_loss
2024-07-12  248.35   2024-07-19  239.08  5     244.9   -2271.26   alpha_reversal
2024-07-23  223.39   2024-07-24  216.75  1     566.0   -3758.64   alpha_reversal
2024-07-23  186.57   2024-07-24  179.98  1     552.7   -3643.92   alpha_reversal
2024-07-15  68.50    2024-07-24  69.40   7     3103.6  +2807.94   alpha_reversal
2024-07-18  541.54   2024-07-24  529.81  4     585.6   -6869.16   stop_loss
2024-07-17  118.75   2024-07-25  118.59  6     1237.9  -193.57    alpha_reversal
2024-07-25  202.29   2024-07-26  205.54  1     775.8   +2525.55   alpha_reversal
2024-07-23  186.50   2024-07-29  183.11  4     620.6   -2106.71   alpha_reversal
2024-07-24  329.25   2024-07-29  337.23  3     364.1   +2906.18   alpha_reversal
2024-07-25  527.58   2024-07-30  530.56  3     477.3   +1425.55   alpha_reversal
2024-07-25  68.90    2024-07-31  67.47   4     3229.7  -4603.67   stop_loss
2024-07-30  208.61   2024-08-01  201.40  2     762.8   -5499.92   stop_loss
2024-07-29  232.22   2024-08-01  216.75  3     214.2   -3312.29   alpha_reversal
2024-07-29  168.42   2024-08-02  165.40  4     677.5   -2043.77   alpha_reversal
2024-08-01  152.84   2024-08-02  155.90  1     1080.8  +3304.63   alpha_reversal
2024-08-01  68.67    2024-08-02  67.30   1     2979.1  -4099.33   alpha_reversal
2024-07-30  151.09   2024-08-02  146.37  3     407.8   -1925.69   trailing_stop
2024-08-02  522.18   2024-08-05  506.46  1     358.0   -5626.20   stop_loss
2024-08-06  511.64   2024-08-07  507.72  1     300.6   -1180.67   alpha_reversal
2024-08-05  188.94   2024-08-08  197.62  3     578.9   +5026.31   alpha_reversal
2024-08-06  157.25   2024-08-08  160.81  2     563.4   +2002.65   alpha_reversal
2024-08-05  442.88   2024-08-12  467.96  5     192.9   +4837.21   alpha_reversal
2024-08-06  394.48   2024-08-13  408.28  5     256.2   +3537.62   alpha_reversal
2024-08-14  150.67   2024-08-15  151.10  1     1029.1  +441.23    alpha_reversal
2024-08-07  163.32   2024-08-16  179.90  7     405.3   +6718.64   max_holding
2024-08-16  151.54   2024-08-19  151.62  1     1107.1  +84.71     alpha_reversal
2024-08-08  198.94   2024-08-19  222.61  7     203.8   +4824.32   max_holding
2024-08-15  170.08   2024-08-19  171.37  2     373.2   +479.69    alpha_reversal
2024-08-09  167.02   2024-08-20  178.79  7     414.9   +4881.85   max_holding
2024-08-20  225.14   2024-08-21  224.81  1     545.9   -182.48    alpha_reversal
2024-08-12  161.22   2024-08-21  164.60  7     588.0   +1982.81   alpha_reversal
2024-08-20  172.19   2024-08-22  172.67  2     474.9   +231.53    alpha_reversal
2024-08-21  549.34   2024-08-22  544.48  1     375.8   -1824.85   alpha_reversal
2024-08-22  480.25   2024-08-23  491.02  1     268.3   +2888.84   alpha_reversal
2024-08-15  106.88   2024-08-23  109.87  6     1031.5  +3087.70   alpha_reversal
2024-08-22  162.73   2024-08-26  164.90  2     715.0   +1558.34   alpha_reversal
2024-08-15  338.66   2024-08-26  343.72  7     291.6   +1476.95   max_holding
2024-08-22  223.18   2024-08-27  226.43  3     577.8   +1879.37   alpha_reversal
2024-08-27  550.26   2024-08-28  546.52  1     412.0   -1541.25   alpha_reversal
2024-08-21  167.62   2024-08-28  165.26  5     444.2   -1046.37   alpha_reversal
2024-08-28  157.03   2024-08-29  157.17  1     1346.8  +188.17    alpha_reversal
2024-08-28  341.16   2024-08-30  348.56  2     399.5   +2952.82   alpha_reversal
2024-08-22  410.95   2024-09-03  404.50  7     358.5   -2311.28   max_holding
2024-08-23  175.05   2024-09-03  160.94  6     525.4   -7411.85   stop_loss
2024-08-27  489.43   2024-09-03  472.63  4     285.0   -4786.11   stop_loss
2024-08-23  177.13   2024-09-04  173.24  7     571.6   -2220.92   alpha_reversal
2024-08-28  161.78   2024-09-04  155.27  4     787.6   -5129.89   stop_loss
2024-08-26  213.32   2024-09-05  230.05  7     263.0   +4401.64   alpha_reversal
2024-09-03  221.43   2024-09-06  219.27  3     599.7   -1293.85   alpha_reversal
2024-09-05  156.21   2024-09-06  149.78  1     726.9   -4673.00   stop_loss
2024-08-29  165.38   2024-09-06  153.17  5     462.1   -5641.55   stop_loss
2024-09-06  205.96   2024-09-10  199.07  2     722.0   -4973.22   trailing_stop
2024-09-04  476.19   2024-09-10  452.92  4     257.1   -5982.06   stop_loss
2024-09-09  216.38   2024-09-10  226.06  1     237.7   +2300.26   alpha_reversal
2024-09-05  403.87   2024-09-11  417.94  4     358.1   +5038.76   alpha_reversal
2024-09-09  535.42   2024-09-11  542.72  2     364.3   +2661.57   alpha_reversal
2024-09-04  163.30   2024-09-13  156.69  7     505.7   -3342.59   max_holding
2024-09-12  157.72   2024-09-16  159.81  2     1209.8  +2530.16   alpha_reversal
2024-09-05  326.82   2024-09-16  340.59  7     364.9   +5024.08   max_holding
2024-09-09  147.93   2024-09-17  158.33  6     650.1   +6758.10   alpha_reversal
2024-09-16  184.98   2024-09-18  186.34  2     566.3   +766.95    alpha_reversal
2024-09-16  215.02   2024-09-19  227.26  3     556.9   +6820.08   alpha_reversal
2024-09-10  108.78   2024-09-19  111.21  7     1132.9  +2742.55   alpha_reversal
2024-09-11  200.89   2024-09-20  204.43  7     543.4   +1922.34   max_holding
2024-09-11  457.29   2024-09-20  483.27  7     213.0   +5531.38   alpha_reversal
2024-09-17  227.98   2024-09-20  238.13  3     271.7   +2757.29   alpha_reversal
2024-09-19  559.49   2024-09-20  557.97  1     387.2   -590.70    alpha_reversal
2024-09-18  77.99    2024-09-23  79.19   3     2404.4  +2894.00   alpha_reversal
2024-09-16  155.63   2024-09-24  155.73  6     487.7   +50.87     alpha_reversal
2024-09-23  250.12   2024-09-24  254.14  1     277.8   +1116.30   alpha_reversal
2024-09-18  164.15   2024-09-24  178.39  4     493.7   +7028.36   alpha_reversal
2024-09-23  482.76   2024-09-25  476.20  2     257.0   -1686.39   alpha_reversal
2024-09-19  157.89   2024-09-25  153.70  4     1327.7  -5571.76   stop_loss
2024-09-20  111.26   2024-09-26  107.28  4     1401.9  -5581.59   stop_loss
2024-09-26  161.88   2024-09-30  164.82  2     910.2   +2674.95   alpha_reversal
2024-09-23  428.71   2024-10-01  415.62  6     377.5   -4942.31   stop_loss
2024-09-23  559.92   2024-10-01  558.33  6     433.4   -689.47    alpha_reversal
2024-09-23  225.10   2024-10-02  225.19  7     589.4   +48.83     max_holding
2024-09-25  152.30   2024-10-02  152.81  5     617.5   +319.51    alpha_reversal
2024-09-27  107.96   2024-10-02  106.32  3     1487.3  -2432.04   alpha_reversal
2024-09-30  79.68    2024-10-02  79.29   2     2473.9  -977.47    alpha_reversal
2024-09-27  204.06   2024-10-03  198.76  4     757.2   -4018.93   alpha_reversal
2024-10-02  559.12   2024-10-03  557.54  1     477.2   -754.03    alpha_reversal
2024-10-03  164.99   2024-10-04  166.02  1     888.0   +912.50    alpha_reversal
2024-09-26  154.61   2024-10-07  152.67  7     1445.4  -2796.25   max_holding
2024-09-30  170.42   2024-10-07  180.88  5     509.6   +5328.18   alpha_reversal
2024-10-03  150.60   2024-10-08  154.57  3     636.1   +2529.96   alpha_reversal
2024-10-03  240.78   2024-10-09  240.93  4     269.0   +40.11     alpha_reversal
2024-10-04  563.17   2024-10-09  566.70  3     486.8   +1714.41   alpha_reversal
2024-10-04  206.02   2024-10-10  207.39  4     692.3   +950.28    alpha_reversal
2024-10-01  185.22   2024-10-10  186.56  7     681.9   +909.67    max_holding
2024-10-01  475.73   2024-10-10  488.40  7     278.0   +3520.41   max_holding
2024-10-09  161.01   2024-10-11  162.22  2     857.4   +1037.79   alpha_reversal
2024-10-10  238.89   2024-10-11  217.69  1     293.7   -6226.07   stop_loss
2024-10-09  388.19   2024-10-11  393.50  2     336.7   +1790.57   alpha_reversal
2024-10-03  411.93   2024-10-14  414.09  7     407.9   +879.77    max_holding
2024-10-03  104.62   2024-10-14  104.05  7     1464.5  -833.86    max_holding
2024-10-08  152.98   2024-10-16  157.22  6     1619.7  +6867.22   alpha_reversal
2024-10-11  79.04    2024-10-16  80.07   3     2646.9  +2713.33   alpha_reversal
2024-10-08  224.41   2024-10-17  230.52  7     617.7   +3774.61   max_holding
2024-10-11  187.24   2024-10-17  201.79  4     558.9   +8130.57   alpha_reversal
2024-10-09  149.44   2024-10-18  154.92  7     649.4   +3557.17   alpha_reversal
2024-10-15  105.90   2024-10-18  103.11  3     1618.6  -4516.49   stop_loss
2024-10-17  412.11   2024-10-22  422.36  3     442.8   +4538.18   alpha_reversal
2024-10-15  187.78   2024-10-22  189.61  5     742.3   +1352.01   alpha_reversal
2024-10-17  162.08   2024-10-23  161.76  4     947.6   -294.77    alpha_reversal
2024-10-14  219.27   2024-10-23  213.54  7     270.5   -1548.95   max_holding
2024-10-22  101.26   2024-10-23  100.91  1     1753.3  -609.88    alpha_reversal
2024-10-17  79.82    2024-10-24  81.86   5     2792.6  +5696.09   alpha_reversal
2024-10-15  380.15   2024-10-24  380.22  7     340.3   +24.13     max_holding
2024-10-23  568.10   2024-10-25  568.56  2     583.6   +270.31    alpha_reversal
2024-10-25  199.64   2024-10-28  190.85  1     512.2   -4500.46   alpha_reversal
2024-10-24  100.54   2024-10-29  98.40   3     1826.6  -3908.85   alpha_reversal
2024-10-28  81.66    2024-10-29  80.54   1     3028.5  -3382.06   alpha_reversal
2024-10-29  153.06   2024-10-30  154.21  1     647.9   +749.20    alpha_reversal
2024-10-25  230.01   2024-11-01  221.34  5     739.4   -6410.49   trailing_stop
2024-10-30  80.32    2024-11-01  81.02   2     3040.0  +2153.44   alpha_reversal
2024-11-01  154.67   2024-11-05  150.92  2     601.7   -2251.96   alpha_reversal
2024-10-30  153.86   2024-11-07  149.99  6     1427.1  -5518.44   stop_loss
2024-11-04  242.96   2024-11-07  296.76  3     284.4   +15298.11  alpha_reversal
2024-11-05  376.97   2024-11-07  401.00  2     379.2   +9110.48   alpha_reversal
2024-11-01  561.27   2024-11-07  584.83  4     547.9   +12910.15  alpha_reversal
2024-10-29  193.26   2024-11-07  197.23  7     491.7   +1953.53   max_holding
2024-11-08  571.91   2024-11-11  584.02  1     228.5   +2767.09   alpha_reversal
2024-11-07  82.74    2024-11-11  83.02   2     2609.4  +710.26    alpha_reversal
2024-11-06  95.65    2024-11-12  93.51   4     1574.3  -3361.31   trailing_stop
2024-11-06  147.23   2024-11-13  139.90  5     560.4   -4110.04   trailing_stop
2024-11-08  231.14   2024-11-13  234.98  3     574.1   +2205.78   alpha_reversal
2024-11-08  148.93   2024-11-14  145.34  4     1551.8  -5577.43   stop_loss
2024-11-06  415.53   2024-11-15  410.00  7     379.4   -2099.27   alpha_reversal
2024-11-08  386.81   2024-11-15  377.28  5     315.0   -2999.87   alpha_reversal
2024-11-07  226.11   2024-11-18  226.67  7     729.4   +407.95    max_holding
2024-11-15  147.53   2024-11-18  148.12  1     1404.0  +827.57    alpha_reversal
2024-11-13  93.53    2024-11-19  91.58   4     1691.4  -3302.88   alpha_reversal
2024-11-18  578.08   2024-11-19  579.62  1     554.2   +849.76    alpha_reversal
2024-11-12  188.19   2024-11-19  185.94  5     507.9   -1141.21   alpha_reversal
2024-11-14  138.21   2024-11-20  146.01  4     621.6   +4846.81   alpha_reversal
2024-11-12  83.87    2024-11-20  85.94   6     2663.8  +5527.69   alpha_reversal
2024-11-20  146.67   2024-11-21  148.81  1     1405.9  +3009.63   alpha_reversal
2024-11-20  92.53    2024-11-22  94.08   2     1581.6  +2464.32   alpha_reversal
2024-11-21  187.67   2024-11-22  186.34  1     565.3   -748.92    alpha_reversal
2024-11-21  339.81   2024-11-25  338.42  2     191.6   -266.21    alpha_reversal
2024-11-25  149.23   2024-11-26  149.06  1     1399.0  -235.66    alpha_reversal
2024-11-25  96.06    2024-11-26  96.40   1     1497.7  +509.73    alpha_reversal
2024-11-20  375.13   2024-11-26  400.62  4     379.6   +9676.97   alpha_reversal
2024-11-18  201.80   2024-11-27  205.64  7     611.9   +2347.28   max_holding
2024-11-27  150.06   2024-11-29  149.54  1     1459.6  -768.03    alpha_reversal
2024-11-21  409.12   2024-12-02  426.64  6     399.7   +7002.35   alpha_reversal
2024-11-29  207.99   2024-12-02  210.60  1     665.0   +1736.15   alpha_reversal
2024-11-29  592.24   2024-12-02  592.71  1     621.0   +290.96    alpha_reversal
2024-11-21  166.75   2024-12-03  170.27  7     722.1   +2542.01   alpha_reversal
2024-11-25  181.62   2024-12-03  194.98  5     578.4   +7725.49   alpha_reversal
2024-12-03  587.24   2024-12-04  583.37  1     282.0   -1091.60   alpha_reversal
2024-12-02  149.48   2024-12-04  145.16  2     1582.9  -6848.59   stop_loss
2024-12-04  158.36   2024-12-05  156.59  1     745.9   -1318.27   alpha_reversal
2024-11-26  338.40   2024-12-05  369.31  6     186.9   +5777.72   alpha_reversal
2024-11-27  243.64   2024-12-09  237.56  7     694.3   -4214.43   alpha_reversal
2024-12-05  582.09   2024-12-09  578.90  2     292.5   -934.10    alpha_reversal
2024-12-06  597.41   2024-12-09  593.74  1     771.6   -2831.95   alpha_reversal
2024-12-09  157.12   2024-12-10  164.02  1     701.4   +4838.95   alpha_reversal
2024-12-02  95.54    2024-12-10  95.81   6     1669.4  +458.19    alpha_reversal
2024-12-05  144.38   2024-12-12  141.08  5     1592.6  -5268.94   stop_loss
2024-12-03  392.60   2024-12-12  374.05  7     397.7   -7374.96   stop_loss
2024-12-12  96.14    2024-12-13  96.76   1     1596.3  +982.32    alpha_reversal
2024-12-11  190.96   2024-12-16  198.87  3     606.7   +4797.90   alpha_reversal
2024-12-11  577.96   2024-12-17  559.96  4     313.1   -5636.30   stop_loss
2024-12-13  234.03   2024-12-18  224.47  3     815.2   -7792.28   stop_loss
2024-12-13  141.58   2024-12-18  139.64  3     1570.7  -3055.81   alpha_reversal
2024-12-17  369.53   2024-12-18  357.46  1     426.8   -5151.06   stop_loss
2024-12-17  95.77    2024-12-19  95.16   2     1622.5  -993.07    alpha_reversal
2024-12-19  227.22   2024-12-20  231.51  1     729.5   +3132.68   alpha_reversal
2024-12-18  536.69   2024-12-20  551.60  2     264.8   +3947.17   alpha_reversal
2024-12-19  354.36   2024-12-23  358.93  2     375.2   +1717.22   alpha_reversal
2024-12-18  576.25   2024-12-23  585.89  3     601.3   +5800.80   alpha_reversal
2024-12-23  95.11    2024-12-24  95.09   1     1596.8  -29.66     alpha_reversal
2024-12-27  561.98   2024-12-30  558.86  1     274.1   -856.33    alpha_reversal
2024-12-26  593.04   2024-12-30  579.52  2     542.2   -7329.75   stop_loss
2024-12-26  199.90   2024-12-30  197.07  2     551.4   -1558.17   alpha_reversal
2024-12-23  431.29   2024-12-31  417.25  5     401.5   -5637.97   trailing_stop
2024-12-19  436.39   2024-12-31  403.64  7     140.8   -4611.11   trailing_stop
2024-12-31  177.09   2025-01-02  171.78  1     604.7   -3207.71   alpha_reversal
2024-12-27  140.07   2025-01-02  138.93  3     1596.7  -1810.09   alpha_reversal
2024-12-30  94.16    2025-01-02  94.83   2     1844.8  +1237.62   alpha_reversal
2024-12-27  191.97   2025-01-03  190.81  4     678.5   -784.99    alpha_reversal
2024-12-23  89.35    2025-01-03  89.69   7     1951.3  +654.74    max_holding
2025-01-03  139.24   2025-01-06  138.59  1     1631.1  -1061.00   alpha_reversal
2024-12-24  361.44   2025-01-06  357.77  7     395.2   -1451.21   max_holding
2025-01-02  198.44   2025-01-07  207.92  3     547.1   +5185.94   alpha_reversal
2025-01-06  170.87   2025-01-08  171.67  2     585.1   +473.22    alpha_reversal
2024-12-27  223.86   2025-01-08  222.02  7     637.4   -1174.66   max_holding
2025-01-02  414.78   2025-01-10  414.73  5     393.7   -19.02     alpha_reversal
2025-01-03  242.16   2025-01-10  235.45  4     743.5   -4991.14   stop_loss
2025-01-03  410.65   2025-01-10  394.54  4     130.3   -2098.08   alpha_reversal
2025-01-08  90.79    2025-01-10  91.88   1     2106.0  +2305.78   alpha_reversal
2025-01-08  203.89   2025-01-10  204.92  1     482.9   +495.17    alpha_reversal
2025-01-10  172.09   2025-01-13  170.48  1     592.6   -948.96    alpha_reversal
2025-01-10  137.18   2025-01-14  139.64  2     1384.6  +3403.09   alpha_reversal
2025-01-03  94.89    2025-01-14  95.29   6     1884.0  +740.07    alpha_reversal
2025-01-13  356.45   2025-01-14  365.01  1     402.7   +3444.83   alpha_reversal
2025-01-16  96.39    2025-01-17  93.63   1     1798.8  -4954.92   stop_loss
2025-01-13  233.24   2025-01-21  221.32  5     660.6   -7876.11   stop_loss
2025-01-15  422.44   2025-01-23  442.21  5     380.0   +7512.96   alpha_reversal
2025-01-14  167.10   2025-01-24  175.97  7     582.2   +5163.20   alpha_reversal
2025-01-15  223.46   2025-01-27  235.30  7     618.8   +7326.74   max_holding
2025-01-15  194.75   2025-01-27  190.83  7     648.1   -2537.59   alpha_reversal
2025-01-22  222.73   2025-01-28  236.85  4     580.2   +8193.02   alpha_reversal
2025-01-21  92.12    2025-01-28  93.18   5     1654.2  +1761.65   alpha_reversal
2025-01-17  90.93    2025-01-28  96.12   6     2133.7  +11084.46  alpha_reversal
2025-01-17  426.71   2025-01-29  388.91  7     131.5   -4972.34   stop_loss
2025-01-17  589.33   2025-01-29  592.91  7     431.2   +1543.15   max_holding
2025-01-30  234.76   2025-02-03  237.30  2     616.0   +1567.07   alpha_reversal
2025-01-28  194.50   2025-02-03  200.20  4     664.0   +3788.44   alpha_reversal
2025-01-29  94.07    2025-02-03  95.42   3     1726.8  +2330.94   alpha_reversal
2025-01-28  199.25   2025-02-04  200.67  5     371.2   +528.43    alpha_reversal
2025-01-27  617.67   2025-02-05  628.91  7     236.4   +2657.80   max_holding
2025-02-04  392.41   2025-02-05  377.98  1     160.4   -2313.41   alpha_reversal
2025-01-30  596.69   2025-02-05  595.28  4     483.3   -678.60    alpha_reversal
2025-02-04  408.62   2025-02-07  405.62  3     345.5   -1037.28   alpha_reversal
2025-02-06  232.07   2025-02-07  226.28  1     549.3   -3179.80   alpha_reversal
2025-02-10  226.78   2025-02-11  231.49  1     559.7   +2641.56   alpha_reversal
2025-02-04  148.22   2025-02-11  150.62  5     1332.2  +3195.37   alpha_reversal
2025-02-06  374.51   2025-02-11  328.34  3     166.2   -7674.87   stop_loss
2025-02-12  102.47   2025-02-14  102.79  2     2075.1  +669.07    alpha_reversal
2025-02-06  360.67   2025-02-14  348.35  6     363.4   -4477.63   alpha_reversal
2025-02-18  603.05   2025-02-19  603.86  1     556.8   +454.37    alpha_reversal
2025-02-10  185.70   2025-02-20  183.62  7     597.1   -1245.40   alpha_reversal
2025-02-14  644.27   2025-02-20  625.81  3     267.9   -4946.47   trailing_stop
2025-02-12  336.68   2025-02-20  354.22  5     167.1   +2932.28   alpha_reversal
2025-02-10  82.70    2025-02-20  83.84   7     1413.0  +1612.60   max_holding
2025-02-11  180.53   2025-02-21  177.06  7     620.2   -2151.38   max_holding
2025-02-12  229.04   2025-02-21  216.47  6     609.7   -7665.01   stop_loss
2025-02-18  150.86   2025-02-21  157.82  3     1469.1  +10221.03  alpha_reversal
2025-02-18  349.37   2025-02-21  335.26  3     409.2   -5775.11   stop_loss
2025-02-12  405.32   2025-02-24  400.73  7     400.4   -1838.77   max_holding
2025-02-20  261.57   2025-02-25  252.10  3     684.8   -6483.55   stop_loss
2025-02-21  610.18   2025-02-25  599.16  2     221.9   -2447.18   alpha_reversal
2025-02-13  198.69   2025-02-25  186.23  7     439.2   -5469.33   stop_loss
2025-02-24  212.82   2025-02-26  214.24  2     619.6   +883.90    alpha_reversal
2025-02-26  253.72   2025-02-27  253.72  1     589.8   +0.61      alpha_reversal
2025-02-24  588.96   2025-02-27  576.40  3     529.2   -6651.19   stop_loss
2025-02-27  159.37   2025-02-28  160.47  1     1238.5  +1356.25   alpha_reversal
2025-02-27  95.72    2025-02-28  97.43   1     1385.3  +2358.47   alpha_reversal
2025-02-26  602.55   2025-03-03  592.68  3     207.9   -2051.91   alpha_reversal
2025-02-28  585.98   2025-03-03  575.14  1     434.7   -4713.46   alpha_reversal
2025-02-27  167.81   2025-03-05  172.14  4     673.5   +2915.54   alpha_reversal
2025-02-28  177.72   2025-03-05  181.23  3     436.7   +1532.81   alpha_reversal
2025-03-04  568.90   2025-03-06  564.24  2     373.9   -1741.27   alpha_reversal
2025-03-06  234.43   2025-03-07  237.91  1     528.5   +1843.36   alpha_reversal
2025-03-05  581.32   2025-03-07  548.04  2     171.4   -5704.79   stop_loss
2025-02-26  290.95   2025-03-07  262.54  7     162.3   -4609.23   max_holding
2025-03-05  163.24   2025-03-10  148.08  3     507.8   -7701.12   stop_loss
2025-03-05  246.60   2025-03-10  227.44  3     499.7   -9574.06   stop_loss
2025-03-06  173.11   2025-03-10  167.82  2     416.6   -2202.72   alpha_reversal
2025-03-11  377.75   2025-03-12  380.17  1     299.8   +725.38    alpha_reversal
2025-03-10  194.64   2025-03-12  198.79  2     415.7   +1726.71   alpha_reversal
2025-03-11  230.70   2025-03-12  247.97  1     141.4   +2442.13   alpha_reversal
2025-03-11  219.99   2025-03-13  208.67  2     429.5   -4864.24   stop_loss
2025-03-13  158.65   2025-03-14  158.32  1     1065.0  -355.29    alpha_reversal
2025-03-14  385.80   2025-03-17  385.55  1     307.5   -75.88     alpha_reversal
2025-03-13  328.95   2025-03-17  337.82  2     307.8   +2730.11   alpha_reversal
2025-03-13  90.65    2025-03-18  91.35   3     1291.0  +905.49    alpha_reversal
2025-03-11  224.65   2025-03-19  234.19  6     430.2   +4103.90   alpha_reversal
2025-03-10  521.13   2025-03-19  545.75  7     145.0   +3570.70   alpha_reversal
2025-03-17  158.51   2025-03-19  158.49  2     1114.8  -14.02     alpha_reversal
2025-03-10  86.85    2025-03-19  85.29   7     1146.4  -1787.17   max_holding
2025-03-19  385.07   2025-03-20  383.71  1     328.2   -445.33    alpha_reversal
2025-03-19  90.77    2025-03-21  89.80   2     1416.9  -1372.14   alpha_reversal
2025-03-19  171.72   2025-03-21  174.48  2     467.1   +1289.32   alpha_reversal
2025-03-21  388.48   2025-03-24  389.90  1     334.1   +473.47    alpha_reversal
2025-03-14  212.67   2025-03-24  219.66  6     434.1   +3035.69   alpha_reversal
2025-03-20  158.68   2025-03-24  158.78  2     1220.2  +126.83    alpha_reversal
2025-03-17  238.13   2025-03-25  288.00  6     150.0   +7481.81   alpha_reversal
2025-03-24  89.12    2025-03-25  84.75   1     1508.9  -6595.67   stop_loss
2025-03-21  85.27    2025-03-25  83.97   2     1311.8  -1697.33   alpha_reversal
2025-03-18  192.92   2025-03-26  201.03  6     442.7   +3591.25   alpha_reversal
2025-03-17  163.80   2025-03-26  164.41  7     596.4   +359.77    alpha_reversal
2025-03-17  559.32   2025-03-26  561.86  7     300.6   +765.07    alpha_reversal
2025-03-25  156.73   2025-03-27  158.63  2     1238.9  +2347.88   alpha_reversal
2025-03-26  84.50    2025-03-27  84.83   1     1393.1  +462.01    alpha_reversal
2025-03-27  387.81   2025-03-28  375.73  1     381.0   -4600.06   alpha_reversal
2025-03-27  222.99   2025-03-28  216.85  1     524.4   -3222.21   alpha_reversal
2025-03-27  201.46   2025-03-28  192.62  1     502.0   -4436.45   alpha_reversal
2025-03-27  161.76   2025-03-28  153.72  1     649.1   -5219.40   stop_loss
2025-03-19  334.19   2025-03-28  325.05  7     342.3   -3128.44   max_holding
2025-03-26  85.06    2025-03-31  86.57   3     1383.8  +2084.51   alpha_reversal
2025-03-27  560.93   2025-03-31  552.77  2     349.3   -2850.51   alpha_reversal
2025-03-31  154.18   2025-04-01  156.45  1     590.4   +1337.93   alpha_reversal
2025-03-31  87.06    2025-04-01  88.00   1     1353.8  +1277.05   alpha_reversal
2025-03-27  547.85   2025-04-02  551.40  4     171.2   +607.12    alpha_reversal
2025-03-26  171.46   2025-04-02  168.22  5     498.3   -1615.25   alpha_reversal
2025-04-02  168.64   2025-04-03  150.83  1     481.9   -8582.40   stop_loss
2025-04-02  241.00   2025-04-03  223.98  1     481.6   -8196.57   stop_loss
2025-04-01  192.27   2025-04-03  178.32  2     451.8   -6300.63   stop_loss
2025-04-01  379.48   2025-04-04  356.93  3     337.1   -7601.41   stop_loss
2025-04-02  151.22   2025-04-04  149.01  2     965.4   -2136.08   alpha_reversal
2025-04-03  83.40    2025-04-04  78.57   1     1274.3  -6153.05   stop_loss
2025-04-04  187.66   2025-04-08  171.59  2     356.4   -5726.77   stop_loss
2025-04-07  211.53   2025-04-08  213.71  1     311.2   +679.42    alpha_reversal
2025-04-04  171.09   2025-04-08  170.57  2     342.0   -174.72    alpha_reversal
2025-04-03  150.27   2025-04-08  144.13  3     560.4   -3444.40   alpha_reversal
2025-04-07  138.93   2025-04-09  160.74  2     323.4   +7052.53   alpha_reversal
2025-04-08  352.04   2025-04-10  378.26  2     233.7   +6126.92   alpha_reversal
2025-04-04  499.80   2025-04-10  518.38  4     218.1   +4050.29   alpha_reversal
2025-04-04  145.08   2025-04-10  149.21  4     396.3   +1636.86   alpha_reversal
2025-04-10  155.60   2025-04-11  156.76  1     279.1   +324.83    alpha_reversal
2025-04-09  198.09   2025-04-15  201.16  4     243.4   +748.65    alpha_reversal
2025-04-04  461.48   2025-04-15  497.33  7     117.4   +4206.90   max_holding
2025-04-11  252.44   2025-04-15  253.98  2     110.8   +171.38    alpha_reversal
2025-04-07  276.40   2025-04-16  286.06  7     223.6   +2160.26   max_holding
2025-04-08  146.01   2025-04-17  153.12  7     700.7   +4987.36   alpha_reversal
2025-04-08  76.22    2025-04-17  75.23   7     967.7   -960.36    max_holding
2025-04-15  382.99   2025-04-21  356.21  3     207.9   -5565.71   stop_loss
2025-04-22  153.55   2025-04-23  151.09  1     687.4   -1689.78   alpha_reversal
2025-04-22  94.06    2025-04-23  94.08   1     882.8   +13.21     alpha_reversal
2025-04-11  528.15   2025-04-23  529.09  7     139.0   +129.94    max_holding
2025-04-15  179.68   2025-04-25  188.90  7     274.4   +2529.02   max_holding
2025-04-22  364.22   2025-04-28  387.99  4     219.5   +5219.30   alpha_reversal
2025-04-16  152.88   2025-04-28  159.97  7     453.5   +3219.46   alpha_reversal
2025-04-28  535.58   2025-04-29  537.51  1     124.1   +239.70    alpha_reversal
2025-04-24  150.81   2025-04-29  151.61  3     714.4   +573.14    alpha_reversal
2025-04-28  94.43    2025-04-29  95.15   1     983.7   +706.28    alpha_reversal
2025-04-29  162.55   2025-04-30  164.57  1     406.2   +820.27    alpha_reversal
2025-04-30  183.33   2025-05-01  182.80  1     381.0   -203.12    alpha_reversal
2025-04-22  232.39   2025-05-01  243.30  7     312.0   +3401.97   max_holding
2025-04-30  158.33   2025-05-05  163.56  3     515.2   +2694.86   alpha_reversal
2025-05-02  151.96   2025-05-05  150.72  1     848.6   -1053.12   alpha_reversal
2025-05-02  80.30    2025-05-06  76.23   2     1120.5  -4563.78   stop_loss
2025-05-07  98.01    2025-05-08  96.52   1     1234.3  -1832.92   alpha_reversal
2025-05-07  185.65   2025-05-09  194.75  2     462.0   +4203.81   alpha_reversal
2025-05-07  188.80   2025-05-12  208.54  3     376.7   +7433.47   alpha_reversal
2025-05-01  542.86   2025-05-12  579.05  7     137.2   +4964.96   alpha_reversal
2025-05-06  275.49   2025-05-12  318.22  4     158.3   +6762.73   alpha_reversal
2025-05-06  318.22   2025-05-12  339.36  4     276.4   +5842.37   alpha_reversal
2025-05-06  150.36   2025-05-13  144.34  5     900.1   -5413.35   trailing_stop
2025-05-09  96.15    2025-05-13  95.22   2     1312.5  -1221.00   alpha_reversal
2025-05-09  73.34    2025-05-14  70.86   3     1151.7  -2861.47   trailing_stop
2025-05-07  150.93   2025-05-15  163.31  6     492.0   +6090.06   alpha_reversal
2025-05-06  552.74   2025-05-15  583.48  7     219.2   +6736.75   max_holding
2025-05-07  195.50   2025-05-16  210.51  7     366.6   +5505.54   max_holding
2025-05-07  246.01   2025-05-16  263.67  7     410.7   +7252.79   max_holding
2025-05-14  142.46   2025-05-20  149.42  4     907.3   +6311.25   alpha_reversal
2025-05-19  205.35   2025-05-21  203.11  2     553.0   -1241.16   alpha_reversal
2025-05-20  343.99   2025-05-21  334.45  1     190.6   -1817.88   alpha_reversal
2025-05-15  72.21    2025-05-21  74.24   4     1097.4  +2228.19   alpha_reversal
2025-05-13  445.95   2025-05-22  452.01  7     274.3   +1661.41   max_holding
2025-05-20  206.34   2025-05-23  194.58  3     490.5   -5766.48   stop_loss
2025-05-20  262.08   2025-05-27  261.43  4     520.5   -336.40    alpha_reversal
2025-05-20  594.51   2025-05-27  602.93  4     190.8   +1607.06   alpha_reversal
2025-05-27  363.07   2025-05-28  356.72  1     189.8   -1205.25   alpha_reversal
2025-05-27  199.70   2025-05-29  199.24  2     507.2   -232.64    alpha_reversal
2025-05-28  201.60   2025-05-29  208.08  1     635.1   +4112.67   alpha_reversal
2025-05-29  260.78   2025-05-30  260.16  1     593.9   -371.38    alpha_reversal
2025-05-21  201.22   2025-05-30  204.91  6     484.3   +1785.45   alpha_reversal
2025-05-20  191.18   2025-05-30  190.86  7     560.0   -178.86    max_holding
2025-05-30  200.34   2025-06-03  202.55  2     542.0   +1198.54   alpha_reversal
2025-06-02  261.07   2025-06-03  262.40  1     637.0   +844.46    alpha_reversal
2025-05-22  576.77   2025-06-03  589.04  7     332.7   +4082.59   max_holding
2025-06-04  207.33   2025-06-05  207.81  1     592.3   +279.81    alpha_reversal
2025-06-03  344.44   2025-06-05  284.56  2     204.6   -12254.39  stop_loss
2025-06-03  74.47    2025-06-05  74.86   2     1578.1  +613.10    alpha_reversal
2025-06-03  165.69   2025-06-10  178.11  5     636.7   +7908.19   alpha_reversal
2025-05-30  591.49   2025-06-10  605.09  7     208.9   +2840.97   max_holding
2025-06-05  586.62   2025-06-10  595.95  3     382.6   +3568.24   alpha_reversal
2025-06-02  341.80   2025-06-11  359.76  7     401.4   +7208.10   max_holding
2025-06-11  470.13   2025-06-12  475.87  1     472.4   +2712.03   alpha_reversal
2025-06-06  210.91   2025-06-12  203.65  4     639.8   -4642.98   trailing_stop
2025-06-05  258.40   2025-06-13  261.10  6     682.3   +1840.96   alpha_reversal
2025-06-06  295.29   2025-06-13  325.15  5     165.9   +4954.58   alpha_reversal
2025-06-09  96.87    2025-06-13  93.79   4     1701.6  -5251.35   stop_loss
2025-06-05  200.12   2025-06-16  197.72  7     615.5   -1478.67   max_holding
2025-06-13  174.36   2025-06-17  175.47  2     722.3   +796.11    alpha_reversal
2025-06-13  354.08   2025-06-17  354.35  2     483.7   +130.71    alpha_reversal
2025-06-18  173.02   2025-06-20  166.18  1     780.7   -5335.63   stop_loss
2025-06-17  93.69    2025-06-20  95.46   2     1849.6  +3261.68   alpha_reversal
2025-06-18  196.08   2025-06-23  200.79  2     727.2   +3422.71   alpha_reversal
2025-06-23  483.43   2025-06-24  487.04  1     451.7   +1626.56   alpha_reversal
2025-06-16  596.15   2025-06-24  601.38  5     452.1   +2363.95   alpha_reversal
2025-06-20  207.81   2025-06-24  218.08  2     670.9   +6893.78   alpha_reversal
2025-06-13  200.42   2025-06-25  198.80  7     568.5   -920.75    max_holding
2025-06-25  201.05   2025-06-26  200.29  1     782.2   -593.69    alpha_reversal
2025-06-20  209.79   2025-06-27  223.19  5     647.0   +8665.50   alpha_reversal
2025-06-17  149.60   2025-06-27  149.48  7     1433.7  -172.17    max_holding
2025-06-23  164.90   2025-06-30  175.75  5     717.7   +7783.17   alpha_reversal
2025-06-23  78.16    2025-06-30  77.11   5     1741.4  -1832.42   alpha_reversal
2025-06-26  95.45    2025-06-30  97.11   2     1970.4  +3256.02   alpha_reversal
2025-06-30  494.78   2025-07-01  488.96  1     506.1   -2945.71   alpha_reversal
2025-06-27  200.57   2025-07-01  207.09  2     809.3   +5273.14   alpha_reversal
2025-06-30  285.98   2025-07-01  286.18  1     731.0   +151.21    alpha_reversal
2025-06-26  202.68   2025-07-02  211.92  4     656.1   +6064.00   alpha_reversal
2025-07-01  175.53   2025-07-03  179.04  2     750.6   +2630.41   alpha_reversal
2025-07-02  705.21   2025-07-03  712.17  1     255.3   +1777.02   alpha_reversal
2025-06-27  323.79   2025-07-07  293.79  5     196.2   -5886.03   stop_loss
2025-07-01  97.66    2025-07-07  98.66   3     2036.0  +2045.66   alpha_reversal
2025-07-01  220.57   2025-07-08  219.25  4     691.9   -913.19    alpha_reversal
2025-07-07  78.89    2025-07-08  79.26   1     1916.9  +726.50    alpha_reversal
2025-07-03  153.16   2025-07-09  153.27  3     1645.4  +183.80    alpha_reversal
2025-07-07  388.25   2025-07-09  398.43  2     496.8   +5058.54   alpha_reversal
2025-07-08  226.01   2025-07-10  227.67  2     680.9   +1128.17   alpha_reversal
2025-07-09  619.12   2025-07-11  618.07  2     621.4   -655.50    alpha_reversal
2025-07-03  496.21   2025-07-14  499.86  6     520.6   +1904.19   alpha_reversal
2025-07-11  225.13   2025-07-14  225.58  1     784.8   +348.98    alpha_reversal
2025-07-08  297.96   2025-07-14  316.74  4     212.4   +3989.16   alpha_reversal
2025-07-11  228.53   2025-07-14  226.58  1     710.1   -1379.39   alpha_reversal
2025-07-15  503.15   2025-07-16  502.45  1     569.1   -399.33    alpha_reversal
2025-07-14  208.09   2025-07-16  209.42  2     850.1   +1127.66   alpha_reversal
2025-07-09  686.17   2025-07-17  694.61  6     255.1   +2154.42   alpha_reversal
2025-07-09  280.66   2025-07-18  288.41  7     685.4   +5311.48   max_holding
2025-07-10  94.30    2025-07-21  95.01   7     2105.5  +1495.23   max_holding
2025-07-17  509.00   2025-07-22  502.10  3     570.7   -3936.99   alpha_reversal
2025-07-18  210.64   2025-07-22  213.64  2     942.9   +2826.81   alpha_reversal
2025-07-22  288.86   2025-07-23  293.85  1     712.9   +3556.52   alpha_reversal
2025-07-22  415.23   2025-07-23  425.16  1     504.8   +5010.61   alpha_reversal
2025-07-22  689.96   2025-07-25  717.39  3     277.1   +7599.36   alpha_reversal
2025-07-18  77.97    2025-07-25  82.52   5     1980.0  +9007.28   alpha_reversal
2025-07-22  228.59   2025-07-28  236.29  4     670.4   +5160.74   alpha_reversal
2025-07-24  213.22   2025-07-29  210.52  3     1028.0  -2769.86   alpha_reversal
2025-07-24  96.03    2025-07-29  97.65   3     2612.9  +4238.39   alpha_reversal
2025-07-29  721.06   2025-07-30  719.13  1     298.2   -575.82    alpha_reversal
2025-07-23  503.20   2025-07-31  530.15  6     578.8   +15601.45  alpha_reversal
2025-07-29  294.42   2025-07-31  293.34  2     812.0   -882.22    alpha_reversal
2025-07-29  630.23   2025-08-01  616.18  3     859.3   -12073.06  stop_loss
2025-07-30  319.20   2025-08-04  309.11  3     273.4   -2759.39   alpha_reversal
2025-08-01  286.82   2025-08-05  288.51  2     761.4   +1289.65   alpha_reversal
2025-08-05  78.78    2025-08-06  77.38   1     1812.6  -2526.39   alpha_reversal
2025-08-06  102.75   2025-08-07  102.41  1     2400.6  -818.74    alpha_reversal
2025-08-06  320.07   2025-08-08  329.49  2     293.0   +2758.94   alpha_reversal
2025-08-07  240.65   2025-08-08  239.63  1     607.4   -621.57    alpha_reversal
2025-08-08  710.51   2025-08-11  707.80  1     283.6   -767.89    alpha_reversal
2025-08-07  284.41   2025-08-12  289.98  3     718.8   +4002.03   alpha_reversal
2025-08-11  240.12   2025-08-12  242.06  1     642.5   +1246.33   alpha_reversal
2025-08-04  626.17   2025-08-13  639.15  7     672.4   +8722.97   max_holding
2025-08-13  171.23   2025-08-14  171.36  1     1538.6  +189.34    alpha_reversal
2025-08-11  406.62   2025-08-14  415.12  3     421.6   +3585.09   alpha_reversal
2025-08-13  287.97   2025-08-15  287.64  2     725.5   -237.53    alpha_reversal
2025-08-11  221.41   2025-08-15  230.91  4     772.6   +7342.77   alpha_reversal
2025-08-12  202.98   2025-08-15  203.34  3     949.4   +337.59    alpha_reversal
2025-08-12  732.29   2025-08-15  719.10  3     272.7   -3596.87   alpha_reversal
2025-08-12  78.30    2025-08-18  82.04   4     1962.3  +7339.71   alpha_reversal
2025-08-18  335.33   2025-08-19  329.15  1     328.0   -2027.57   alpha_reversal
2025-08-14  239.04   2025-08-19  230.58  3     692.3   -5859.32   alpha_reversal
2025-08-19  174.55   2025-08-20  175.40  1     1653.5  +1398.04   alpha_reversal
2025-08-13  100.39   2025-08-20  102.10  5     2398.4  +4093.38   alpha_reversal
2025-08-19  201.22   2025-08-22  205.52  3     982.1   +4229.51   alpha_reversal
2025-08-18  514.37   2025-08-26  499.71  6     445.2   -6526.78   stop_loss
2025-08-21  320.27   2025-08-26  351.49  3     333.7   +10418.91  alpha_reversal
2025-08-20  226.74   2025-08-26  236.54  4     680.3   +6667.45   alpha_reversal
2025-08-22  730.82   2025-08-27  737.75  3     276.9   +1917.57   alpha_reversal
2025-08-19  634.75   2025-08-27  640.87  6     793.5   +4860.24   alpha_reversal
2025-08-19  230.24   2025-08-28  232.00  7     830.6   +1466.07   alpha_reversal
2025-08-26  234.95   2025-08-28  236.04  2     712.9   +780.29    alpha_reversal
2025-08-26  228.82   2025-08-28  231.48  2     835.0   +2221.01   alpha_reversal
2025-08-26  174.54   2025-08-28  173.33  2     1682.1  -2021.80   alpha_reversal
2025-08-26  429.24   2025-08-28  432.44  2     452.9   +1449.28   alpha_reversal
2025-08-29  175.21   2025-09-02  175.91  1     1840.6  +1296.05   alpha_reversal
2025-08-27  95.74    2025-09-03  98.98   4     2458.0  +7985.75   alpha_reversal
2025-08-27  504.89   2025-09-04  505.61  5     518.0   +372.77    alpha_reversal
2025-09-03  230.26   2025-09-04  231.66  1     750.7   +1054.93   alpha_reversal
2025-09-02  414.10   2025-09-05  420.67  3     457.5   +3008.29   alpha_reversal
2025-09-03  334.26   2025-09-08  346.23  3     341.6   +4088.30   alpha_reversal
2025-08-28  81.14    2025-09-08  81.91   6     2293.6  +1780.16   alpha_reversal
2025-08-29  228.99   2025-09-08  244.94  5     744.5   +11869.14  alpha_reversal
2025-09-08  496.38   2025-09-10  498.05  2     529.6   +881.10    alpha_reversal
2025-09-03  723.51   2025-09-10  761.39  5     312.8   +11849.59  alpha_reversal
2025-09-05  229.72   2025-09-11  219.88  4     777.5   -7653.93   stop_loss
2025-09-03  638.64   2025-09-11  651.77  6     804.9   +10566.74  alpha_reversal
2025-09-09  82.50    2025-09-15  79.70   4     2660.8  -7450.05   stop_loss
2025-09-15  779.17   2025-09-16  777.17  1     297.4   -593.58    alpha_reversal
2025-09-10  226.47   2025-09-17  238.42  5     938.1   +11204.96  alpha_reversal
2025-09-08  290.33   2025-09-17  308.69  7     822.9   +15112.74  alpha_reversal
2025-09-09  416.13   2025-09-18  464.30  7     520.7   +25087.15  alpha_reversal
2025-09-10  173.84   2025-09-19  174.07  7     2003.4  +443.65    max_holding
2025-09-17  653.96   2025-09-19  659.61  2     945.0   +5338.92   alpha_reversal
2025-09-11  230.06   2025-09-22  227.52  7     902.7   -2300.82   alpha_reversal
2025-09-12  216.05   2025-09-23  216.23  7     834.6   +153.44    max_holding
2025-09-22  102.45   2025-09-23  102.05  1     2859.6  -1146.79   alpha_reversal
2025-09-22  271.27   2025-09-23  281.02  1     735.5   +7170.09   alpha_reversal
2025-09-16  79.85    2025-09-25  76.34   7     2769.0  -9727.67   stop_loss
2025-09-24  102.36   2025-09-25  102.58  1     2999.1  +648.47    alpha_reversal
2025-09-25  787.09   2025-09-26  793.97  1     321.4   +2211.53   alpha_reversal
2025-09-23  174.63   2025-09-26  177.54  3     1983.0  +5785.80   alpha_reversal
2025-09-23  251.44   2025-09-29  243.60  4     906.5   -7113.10   alpha_reversal
2025-09-30  254.27   2025-10-01  254.84  1     1027.6  +579.42    alpha_reversal
2025-09-26  77.36    2025-10-01  88.66   3     3205.4  +36235.89  alpha_reversal
2025-09-24  657.69   2025-10-01  664.33  5     986.6   +6558.71   alpha_reversal
2025-09-30  277.89   2025-10-01  286.74  1     686.6   +6074.73   alpha_reversal
2025-09-30  312.65   2025-10-02  304.54  2     963.0   -7814.95   stop_loss
2025-09-23  220.82   2025-10-02  222.30  7     959.2   +1418.06   max_holding
2025-09-30  242.89   2025-10-03  244.89  3     880.9   +1764.54   alpha_reversal
2025-10-06  668.14   2025-10-07  665.00  1     1073.7  -3374.17   alpha_reversal
2025-09-30  215.94   2025-10-08  225.21  6     879.7   +8153.95   alpha_reversal
2025-10-06  250.22   2025-10-09  241.08  3     919.7   -8399.72   stop_loss
2025-10-07  256.12   2025-10-10  244.68  3     1159.3  -13261.22  stop_loss
2025-10-01  777.93   2025-10-10  756.23  7     317.4   -6888.47   max_holding
2025-10-03  430.04   2025-10-10  413.28  5     279.4   -4683.97   trailing_stop
2025-10-10  101.48   2025-10-13  101.65  1     2833.4  +502.34    alpha_reversal
2025-10-09  86.16    2025-10-15  82.74   4     2503.6  -8564.99   stop_loss
2025-10-15  435.37   2025-10-16  428.54  1     260.2   -1777.71   alpha_reversal
2025-10-10  210.84   2025-10-20  216.71  6     912.1   +5359.90   alpha_reversal
2025-10-13  659.62   2025-10-20  667.17  5     779.3   +5883.65   alpha_reversal
2025-10-13  220.18   2025-10-22  217.84  7     936.5   -2190.55   max_holding
2025-10-15  511.56   2025-10-23  518.15  6     619.6   +4080.42   alpha_reversal
2025-10-22  251.47   2025-10-23  252.61  1     782.1   +888.54    alpha_reversal
2025-10-22  190.80   2025-10-23  190.15  1     1758.8  -1152.08   alpha_reversal
2025-10-16  82.64    2025-10-27  86.57   7     2572.8  +10113.75  alpha_reversal
2025-10-20  301.15   2025-10-28  303.84  6     730.6   +1960.84   alpha_reversal
2025-10-20  755.96   2025-10-28  783.66  6     246.9   +6841.52   alpha_reversal
2025-10-21  105.84   2025-10-28  102.70  5     2411.8  -7577.63   stop_loss
2025-10-22  664.35   2025-10-29  683.16  5     682.0   +12824.39  alpha_reversal
2025-10-22  512.97   2025-10-30  581.50  6     376.2   +25783.32  alpha_reversal
2025-10-21  293.04   2025-10-30  301.40  7     486.4   +4069.08   alpha_reversal
2025-10-27  188.19   2025-11-03  184.01  5     1847.7  -7722.46   stop_loss
2025-10-31  456.79   2025-11-03  468.14  1     282.5   +3205.57   alpha_reversal
2025-11-03  777.94   2025-11-05  784.65  2     295.3   +1982.10   alpha_reversal
2025-10-30  101.86   2025-11-05  101.01  4     2684.5  -2304.18   alpha_reversal
2025-11-06  184.90   2025-11-07  184.32  1     1981.8  -1149.38   alpha_reversal
2025-11-10  186.32   2025-11-11  191.49  1     1951.2  +10084.51  alpha_reversal
2025-11-10  569.80   2025-11-11  566.32  1     306.9   -1068.32   alpha_reversal
2025-11-04  198.15   2025-11-13  194.48  7     976.4   -3579.66   max_holding
2025-11-12  319.13   2025-11-13  307.94  1     874.1   -9784.53   stop_loss
2025-11-10  445.45   2025-11-13  401.79  3     253.2   -11056.83  stop_loss
2025-11-06  495.29   2025-11-17  505.14  7     542.5   +5342.06   alpha_reversal
2025-11-06  287.79   2025-11-17  280.32  7     596.4   -4457.68   alpha_reversal
2025-11-13  237.70   2025-11-18  222.44  3     755.6   -11530.77  stop_loss
2025-11-13  278.33   2025-11-18  283.75  3     675.3   +3660.75   alpha_reversal
2025-11-14  404.55   2025-11-18  401.05  2     240.1   -841.07    alpha_reversal
2025-11-17  91.44    2025-11-18  94.86   1     2296.1  +7853.98   alpha_reversal
2025-11-14  102.11   2025-11-18  100.93  2     2850.1  -3383.34   alpha_reversal
2025-11-12  571.97   2025-11-18  545.33  4     324.0   -8630.53   stop_loss
2025-11-14  668.46   2025-11-18  656.01  2     662.0   -8239.00   stop_loss
2025-11-14  302.40   2025-11-20  296.89  4     783.1   -4311.97   alpha_reversal
2025-11-19  659.21   2025-11-20  648.51  1     613.6   -6563.16   alpha_reversal
2025-11-21  655.63   2025-11-24  664.61  1     517.1   +4645.96   alpha_reversal
2025-11-19  485.35   2025-12-01  485.39  7     469.6   +20.39     max_holding
2025-11-19  222.80   2025-12-01  233.76  7     730.9   +8012.01   max_holding
2025-11-21  766.56   2025-12-01  802.23  5     215.6   +7690.72   alpha_reversal
2025-11-24  283.22   2025-12-01  285.96  4     510.6   +1398.37   alpha_reversal
2025-11-20  179.47   2025-12-02  205.28  7     989.6   +25538.33  max_holding
2025-12-02  678.01   2025-12-03  679.68  1     617.1   +1029.14   alpha_reversal
2025-11-24  296.81   2025-12-04  314.52  7     719.2   +12739.87  max_holding
2025-12-02  429.45   2025-12-05  454.77  3     283.3   +7172.74   alpha_reversal
2025-12-04  201.97   2025-12-08  206.17  2     898.7   +3771.03   alpha_reversal
2025-12-08  313.95   2025-12-09  299.01  1     903.6   -13500.87  stop_loss
2025-12-01  204.36   2025-12-09  198.80  6     1755.4  -9747.89   stop_loss
2025-12-03  100.71   2025-12-09  95.31   4     1951.4  -10524.10  stop_loss
2025-12-02  290.63   2025-12-09  301.59  5     579.1   +6348.18   alpha_reversal
2025-12-04  479.98   2025-12-10  477.23  4     512.3   -1410.63   alpha_reversal
2025-12-10  308.87   2025-12-11  315.80  1     716.5   +4961.75   alpha_reversal
2025-12-02  581.40   2025-12-11  623.84  7     336.3   +14270.14  max_holding
2025-12-10  205.55   2025-12-12  210.36  2     1603.8  +7706.67   alpha_reversal
2025-12-10  198.82   2025-12-15  205.40  3     955.8   +6287.21   alpha_reversal
2025-12-04  229.22   2025-12-15  222.43  7     871.1   -5919.73   max_holding
2025-12-04  317.35   2025-12-15  307.85  7     558.7   -5307.15   max_holding
2025-12-12  459.19   2025-12-15  475.07  1     320.8   +5095.76   alpha_reversal
2025-12-10  96.13    2025-12-15  99.47   3     2006.0  +6707.99   alpha_reversal
2025-12-11  115.11   2025-12-15  116.49  2     2230.0  +3090.99   alpha_reversal
2025-12-15  885.38   2025-12-16  874.11  1     253.0   -2849.96   alpha_reversal
2025-12-11  482.61   2025-12-17  474.80  4     528.3   -4127.42   alpha_reversal
2025-12-09  277.06   2025-12-17  271.45  6     1075.5  -6032.73   alpha_reversal
2025-12-15  588.68   2025-12-17  560.30  2     317.3   -9004.01   stop_loss
2025-12-09  679.51   2025-12-17  667.26  6     726.5   -8897.94   trailing_stop
2025-12-15  287.07   2025-12-23  295.97  6     602.7   +5359.91   alpha_reversal
2025-12-23  485.98   2025-12-24  486.67  1     623.5   +424.59    alpha_reversal
2025-12-16  306.51   2025-12-24  313.71  6     614.8   +4429.41   alpha_reversal
2025-12-18  564.79   2025-12-24  582.11  4     298.2   +5163.30   alpha_reversal
2025-12-22  112.43   2025-12-29  112.24  4     2466.1  -449.29    alpha_reversal
2025-12-23  272.24   2025-12-30  272.69  4     1122.4  +501.57    alpha_reversal
2025-12-26  313.45   2025-12-30  313.47  2     711.4   +18.73     alpha_reversal
2025-12-24  485.64   2025-12-30  454.20  3     301.9   -9491.26   stop_loss
2025-12-29  486.23   2025-12-31  482.28  2     719.3   -2845.88   alpha_reversal
2025-12-30  232.65   2025-12-31  230.70  1     1279.5  -2484.32   alpha_reversal
2025-12-19  205.38   2025-12-31  205.75  7     1463.8  +543.59    max_holding
2025-12-29  686.32   2026-01-02  680.97  3     859.2   -4597.23   alpha_reversal
2025-12-29  887.95   2026-01-05  943.00  4     287.0   +15801.98  alpha_reversal
2025-12-31  111.24   2026-01-05  112.42  2     3010.0  +3568.32   alpha_reversal
2025-12-29  577.55   2026-01-06  621.32  5     371.0   +16239.19  alpha_reversal
2026-01-05  233.18   2026-01-07  241.44  2     1098.7  +9078.36   alpha_reversal
2026-01-02  315.09   2026-01-08  325.05  4     738.3   +7355.41   alpha_reversal
2025-12-31  449.94   2026-01-12  448.74  7     327.3   -395.84    max_holding
2026-01-06  203.81   2026-01-13  212.41  5     1689.3  +14536.85  alpha_reversal
2026-01-05  472.01   2026-01-14  458.10  7     699.7   -9729.08   trailing_stop
2026-01-06  262.25   2026-01-15  257.84  7     1215.1  -5353.79   max_holding
2026-01-09  692.53   2026-01-20  675.40  6     990.2   -16961.03  stop_loss
2026-01-15  455.85   2026-01-21  442.88  3     678.1   -8796.12   stop_loss
2026-01-21  683.88   2026-01-22  686.76  1     824.9   +2379.71   alpha_reversal
2026-01-16  437.72   2026-01-23  448.84  4     378.6   +4209.14   alpha_reversal
2026-01-22  450.34   2026-01-26  468.97  2     612.4   +11412.26  alpha_reversal
2026-01-20  246.59   2026-01-27  257.90  5     1151.9  +13024.34  alpha_reversal
2026-01-20  321.93   2026-01-28  335.61  6     742.2   +10146.82  alpha_reversal
2026-01-23  107.44   2026-01-28  106.06  3     2153.1  -2965.47   alpha_reversal
2026-01-26  927.44   2026-01-29  934.73  3     238.9   +1740.72   alpha_reversal
2026-01-22  326.61   2026-01-29  338.43  5     562.1   +6640.07   alpha_reversal
2026-01-22  303.78   2026-02-02  307.99  7     805.5   +3386.63   max_holding
2026-02-02  693.86   2026-02-05  675.44  3     892.0   -16435.59  stop_loss
2026-02-02  233.14   2026-02-06  242.91  4     903.9   +8832.47   alpha_reversal
2026-02-05  678.65   2026-02-06  725.84  1     260.7   +12299.83  alpha_reversal
2026-02-05  886.19   2026-02-11  939.18  4     209.2   +11085.46  alpha_reversal
2026-02-03  410.48   2026-02-12  400.72  7     440.0   -4291.50   max_holding
2026-02-10  273.82   2026-02-12  261.60  2     1009.5  -12334.24  stop_loss
2026-02-11  239.71   2026-02-12  243.14  1     1584.6  +5433.69   alpha_reversal
2026-02-05  397.41   2026-02-12  416.86  5     355.0   +6905.50   alpha_reversal
2026-02-12  239.47   2026-02-13  242.84  1     807.0   +2718.76   alpha_reversal
2026-02-12  302.79   2026-02-18  308.63  3     682.4   +3981.17   alpha_reversal
2026-02-13  255.91   2026-02-19  260.45  3     898.3   +4079.79   alpha_reversal
2026-02-17  242.16   2026-02-19  245.48  2     1578.8  +5237.20   alpha_reversal
2026-02-11  204.18   2026-02-20  210.00  6     765.6   +4458.00   alpha_reversal
2026-02-20  264.71   2026-02-23  266.05  1     918.5   +1225.92   alpha_reversal
2026-02-20  310.95   2026-02-23  297.52  1     790.3   -10609.02  stop_loss
2026-02-12  308.94   2026-02-23  311.12  6     569.5   +1240.39   alpha_reversal
2026-02-20  917.87   2026-02-23  887.19  1     200.7   -6157.13   alpha_reversal
2026-02-11  690.42   2026-02-23  680.19  7     768.6   -7862.67   alpha_reversal
2026-02-18  361.42   2026-02-23  368.82  3     413.2   +3054.46   alpha_reversal
2026-02-23  244.66   2026-02-24  246.16  1     1551.5  +2318.91   alpha_reversal
2026-02-25  312.84   2026-02-26  307.01  1     665.2   -3875.37   alpha_reversal
2026-02-18  398.89   2026-02-27  392.54  7     502.4   -3187.94   max_holding
2026-02-25  303.45   2026-02-27  300.15  2     730.9   -2413.37   alpha_reversal
2026-02-19  233.83   2026-03-02  229.63  7     843.9   -3545.91   max_holding
2026-02-26  243.59   2026-03-02  248.44  2     1587.1  +7687.70   alpha_reversal
2026-02-27  122.97   2026-03-02  120.46  1     1952.6  -4908.89   alpha_reversal
2026-02-26  124.23   2026-03-02  126.78  2     1738.6  +4431.70   alpha_reversal
2026-03-03  208.83   2026-03-04  216.71  1     909.9   +7167.88   alpha_reversal
2026-03-02  862.13   2026-03-04  866.82  2     176.2   +825.59    alpha_reversal
2026-03-03  127.71   2026-03-04  127.48  1     1793.8  -407.90    alpha_reversal
2026-02-23  756.85   2026-03-04  731.60  7     249.3   -6292.97   max_holding
2026-02-27  402.71   2026-03-05  405.35  4     410.1   +1080.88   alpha_reversal
2026-03-03  119.01   2026-03-05  115.16  2     1980.6  -7624.23   alpha_reversal
2026-03-04  683.61   2026-03-06  670.21  2     675.0   -9039.64   stop_loss
2026-03-06  231.23   2026-03-09  224.89  1     802.2   -5084.69   alpha_reversal
2026-03-04  245.42   2026-03-09  242.47  3     1496.3  -4419.99   alpha_reversal
2026-03-03  303.52   2026-03-10  306.89  5     641.7   +2161.42   alpha_reversal
2026-03-06  396.93   2026-03-10  399.04  2     412.6   +871.42    alpha_reversal
2026-03-04  356.61   2026-03-10  345.94  4     443.6   -4734.72   alpha_reversal
2026-03-04  262.65   2026-03-11  260.68  5     916.5   -1806.97   alpha_reversal
2026-03-02  297.71   2026-03-11  287.38  7     717.9   -7418.00   max_holding
2026-03-10  217.87   2026-03-12  204.66  2     748.5   -9888.35   stop_loss
2026-03-05  123.12   2026-03-13  126.20  6     1721.3  +5299.45   alpha_reversal
2026-03-09  705.17   2026-03-13  693.64  4     198.3   -2286.18   alpha_reversal
2026-03-13  337.53   2026-03-16  339.10  1     419.4   +661.09    alpha_reversal
2026-03-16  305.71   2026-03-17  310.76  1     717.4   +3624.06   alpha_reversal
2026-03-06  821.83   2026-03-17  806.64  7     173.3   -2633.17   max_holding
2026-03-16  700.13   2026-03-17  701.65  1     211.8   +321.70    alpha_reversal
2026-03-13  283.58   2026-03-19  287.83  4     706.7   +2999.34   alpha_reversal
2026-03-12  402.06   2026-03-20  381.68  6     600.6   -12242.05  stop_loss
2026-03-13  209.99   2026-03-20  195.02  5     746.1   -11170.67  stop_loss
2026-03-19  809.90   2026-03-20  813.12  1     190.7   +613.90    alpha_reversal
2026-03-16  243.31   2026-03-20  235.25  4     1341.2  -10809.11  stop_loss
2026-03-11  115.41   2026-03-20  114.12  7     1895.2  -2445.44   max_holding
2026-03-17  124.89   2026-03-20  118.96  3     1910.1  -11318.83  stop_loss
2026-03-12  664.58   2026-03-20  648.25  6     592.1   -9670.14   stop_loss
2026-03-12  255.89   2026-03-23  251.36  7     924.7   -4182.76   max_holding
2026-03-12  209.63   2026-03-23  210.03  7     912.2   +365.04    max_holding
2026-03-19  307.28   2026-03-24  290.29  3     761.8   -12942.76  stop_loss
2026-03-19  380.49   2026-03-24  382.84  3     423.7   +994.89    alpha_reversal
2026-03-23  655.71   2026-03-24  652.85  1     532.9   -1521.11   alpha_reversal
2026-03-23  115.74   2026-03-25  119.31  2     1958.2  +6995.49   alpha_reversal
2026-03-19  688.99   2026-03-25  718.68  4     224.6   +6667.18   alpha_reversal
2026-03-23  383.19   2026-03-26  365.79  3     615.5   -10713.24  stop_loss
2026-03-23  198.51   2026-03-27  190.42  4     689.0   -5570.07   end_of_backtest
2026-03-23  235.54   2026-03-27  240.33  4     1247.5  +5978.28   end_of_backtest
2026-03-23  120.78   2026-03-27  122.83  4     1768.8  +3622.84   end_of_backtest

**Best 3 trades:**

- 2025-10-01: P&L = **+36235.89** (alpha_reversal)
- 2025-10-30: P&L = **+25783.32** (alpha_reversal)
- 2025-12-02: P&L = **+25538.33** (max_holding)

**Worst 3 trades:**

- 2026-01-20: P&L = **-16961.03** (stop_loss)
- 2026-02-05: P&L = **-16435.59** (stop_loss)
- 2025-12-09: P&L = **-13500.87** (stop_loss)

#### Equity Curve

Date        Portfolio Value
2016-03-28  100,000.00
2016-09-22  122,717.28
2017-03-23  131,322.36
2017-09-20  185,108.24
2018-03-21  251,810.50
2018-09-18  226,804.61
2019-03-20  247,175.72
2019-09-17  270,531.71
2020-03-17  221,875.04
2020-09-14  312,981.31
2021-03-15  441,223.74
2021-09-10  459,677.09
2022-03-10  353,274.21
2022-09-08  286,774.04
2023-03-09  276,922.78
2023-09-07  299,299.32
2024-03-07  383,310.12
2024-09-05  427,043.82
2025-03-07  461,820.54
2025-09-05  648,861.20
2026-03-06  864,116.02

#### Drawdown Curve

Date        Drawdown
2016-03-28  0.00%
2016-09-22  -4.31%
2017-03-23  -11.95%
2017-09-20  0.00%
2018-03-21  -13.12%
2018-09-18  -21.74%
2019-03-20  -14.71%
2019-09-17  -9.52%
2020-03-17  -41.02%
2020-09-14  -16.80%
2021-03-15  -3.01%
2021-09-10  -9.72%
2022-03-10  -30.62%
2022-09-08  -43.68%
2023-03-09  -45.61%
2023-09-07  -41.22%
2024-03-07  -24.72%
2024-09-05  -16.13%
2025-03-07  -11.13%
2025-09-05  0.00%
2026-03-06  -4.09%

#### Walk-Forward Returns (70% IS / 30% OOS)

Period               Start       End         Cumulative Return
In-Sample (70%)      2016-03-28  2023-03-23  175.77%
Out-of-Sample (30%)  2023-03-24  2026-03-27  180.87%

#### Return Distribution

Return Bin            Count
-16.803% to -14.164%  1
-14.164% to -11.525%  0
-11.525% to -8.885%   5
-8.885% to -6.246%    11
-6.246% to -3.607%    31
-3.607% to -0.967%    348
-0.967% to 1.672%     1876
1.672% to 4.311%      225
4.311% to 6.951%      17
6.951% to 9.590%      2


## Baseline Comparison

> Every strategy must be judged against what a passive, zero-effort investor would earn.
> A strategy that underperforms SPY buy-and-hold provides negative alpha even with a
> positive return. A strategy that underperforms a simple MA cross has no edge over
> basic trend-following.

### Reference Baselines

Baseline          Total Return  Sharpe  Notes
SPY buy-and-hold  267.23%       0.567   Fully invested, full window
SPY 50d MA cross  148.69%       0.490   Long when SPY > 50d MA, flat (earns RF) otherwise

### Strategy vs Baselines

Ticker                   Strategy       Net Return  Sharpe  vs SPY B&H     vs 50d MA cross
MSFT                     AlphaCombined  32.33%      0.160   FAIL -234.90%  FAIL -0.330
AAPL                     AlphaCombined  36.94%      0.197   FAIL -230.28%  FAIL -0.292
BA                       AlphaCombined  5.00%       -0.335  FAIL -262.23%  FAIL -0.825
JPM                      AlphaCombined  11.57%      -0.208  FAIL -255.65%  FAIL -0.697
AMZN                     AlphaCombined  32.79%      0.128   FAIL -234.43%  FAIL -0.362
GOOGL                    AlphaCombined  48.76%      0.372   FAIL -218.46%  FAIL -0.118
GS                       AlphaCombined  -4.80%      -0.466  FAIL -272.02%  FAIL -0.956
CVX                      Momentum       -1.06%      -0.270  FAIL -268.28%  FAIL -0.760
JNJ                      AlphaCombined  8.75%       -0.253  FAIL -258.47%  FAIL -0.743
TSLA                     AlphaCombined  5.50%       -0.351  FAIL -261.72%  FAIL -0.841
MRK                      AlphaCombined  21.43%      -0.045  FAIL -245.80%  FAIL -0.535
WMT                      AlphaCombined  47.83%      0.376   FAIL -219.40%  FAIL -0.113
CAT                      AlphaCombined  38.06%      0.251   FAIL -229.16%  FAIL -0.239
SPY                      AlphaCombined  19.55%      -0.016  FAIL -247.67%  FAIL -0.505
TSM                      AlphaCombined  29.62%      0.085   FAIL -237.60%  FAIL -0.405
AlphaCombined_Portfolio  AlphaCombined  670.40%     0.784   PASS +403.18%  PASS +0.295

## Portfolio Construction

### Cross-Sectional Momentum Ranking (12-1 Month)

Rank  Ticker  12-1m Return  Status
1     CAT     +123.8%       Allocated
2     TSM     +119.2%       Allocated
3     GOOGL   +86.7%        Allocated
4     GS      +64.4%        Allocated
5     JNJ     +54.7%        Allocated
6     TSLA    +50.2%        Allocated
7     WMT     +47.0%        Allocated
8     MRK     +39.3%        Allocated
9     BA      +28.5%        Allocated
10    JPM     +24.4%        Filtered
11    AAPL    +23.8%        Filtered
12    SPY     +22.3%        Filtered
13    CVX     +14.6%        Filtered
14    MSFT    +3.8%         Filtered
15    AMZN    +3.4%         Filtered

### Volatility-Parity Allocations

Ticker  Weight  $ Allocated  Sharpe  CS Rank  Rationale
JNJ     11.9%   $11,865      -0.253  5        vol-parity 11.9%; CS rank 5/16; 12-1m mom +54.7%
GOOGL   11.5%   $11,540      0.372   3        vol-parity 11.5%; CS rank 3/16; 12-1m mom +86.7%
GS      11.3%   $11,321      -0.466  4        vol-parity 11.3%; CS rank 4/16; 12-1m mom +64.4%
WMT     11.2%   $11,240      0.376   7        vol-parity 11.2%; CS rank 7/16; 12-1m mom +47.0%
TSLA    11.1%   $11,048      -0.351  6        vol-parity 11.0%; CS rank 6/16; 12-1m mom +50.2%
MRK     10.9%   $10,942      -0.045  8        vol-parity 10.9%; CS rank 8/16; 12-1m mom +39.3%
BA      10.8%   $10,793      -0.335  9        vol-parity 10.8%; CS rank 9/16; 12-1m mom +28.5%
CAT     10.7%   $10,718      0.251   1        vol-parity 10.7%; CS rank 1/16; 12-1m mom +123.8%
TSM     10.5%   $10,534      0.085   2        vol-parity 10.5%; CS rank 2/16; 12-1m mom +119.2%

### Rejected by Portfolio Filter

- **JPM**: CS momentum filter: rank 10 / 16, 12-1m return +24.4% — below top-60% cutoff
- **AAPL**: CS momentum filter: rank 11 / 16, 12-1m return +23.8% — below top-60% cutoff
- **SPY**: CS momentum filter: rank 12 / 16, 12-1m return +22.3% — below top-60% cutoff
- **CVX**: CS momentum filter: rank 13 / 16, 12-1m return +14.6% — below top-60% cutoff
- **MSFT**: CS momentum filter: rank 14 / 16, 12-1m return +3.8% — below top-60% cutoff
- **AMZN**: CS momentum filter: rank 15 / 16, 12-1m return +3.4% — below top-60% cutoff
- **AlphaCombined_Portfolio**: CS momentum filter: rank 17 / 16, 12-1m return +0.0% — below top-60% cutoff

### Portfolio-Level Risk Metrics

Metric            Value
Portfolio Sharpe  -0.088
Annualised Vol    2.4%
VaR (95%)         0.22%
CVaR (95%)        0.35%
Max Drawdown      3.2%

### Statistical Significance (per Ticker)

_t-stat uses Lo (2002) autocorrelation correction. Bootstrap CI is 90% block-bootstrap (block=20 days). p-val < 0.05 = statistically significant at 95% confidence. Rolling stable = % of 60-day windows with positive Sharpe ≥ 50%. Perm p-val = Calmar-based permutation test (lower = more order-dependent return path)._

Ticker                   Sharpe  t-stat  p-value  Bootstrap 90% CI  Rolling Stable?  Perm p-val  Significant?
MSFT                     0.160   0.52    0.301    [-0.31, 0.68]     PASS 55%         0.956       FAIL
AAPL                     0.197   0.64    0.262    [-0.29, 0.69]     PASS 52%         0.452       FAIL
BA                       -0.335  -1.03   0.849    [-0.90, 0.21]     WARNING 47%      0.651       FAIL
JPM                      -0.208  -0.64   0.739    [-0.72, 0.30]     WARNING 43%      0.893       FAIL
AMZN                     0.128   0.40    0.346    [-0.38, 0.63]     WARNING 49%      0.067       FAIL
GOOGL                    0.372   1.20    0.115    [-0.10, 0.84]     PASS 53%         0.207       FAIL
GS                       -0.466  -1.50   0.933    [-0.93, 0.03]     WARNING 44%      0.435       FAIL
CVX                      -0.270  -0.84   0.800    [-0.82, 0.23]     WARNING 32%      0.278       FAIL
JNJ                      -0.253  -0.79   0.784    [-0.77, 0.33]     WARNING 47%      0.665       FAIL
TSLA                     -0.351  -1.09   0.863    [-0.89, 0.16]     WARNING 44%      0.550       FAIL
MRK                      -0.045  -0.14   0.555    [-0.62, 0.44]     WARNING 46%      0.150       FAIL
WMT                      0.376   1.21    0.113    [-0.15, 0.90]     PASS 58%         0.340       FAIL
CAT                      0.251   0.77    0.221    [-0.25, 0.79]     PASS 56%         0.981       FAIL
SPY                      -0.016  -0.05   0.520    [-0.53, 0.52]     PASS 52%         0.373       FAIL
TSM                      0.085   0.27    0.393    [-0.44, 0.57]     PASS 53%         0.554       FAIL
AlphaCombined_Portfolio  0.784   1.97    0.024    [0.14, 1.51]      PASS 67%         0.950       PASS


## Monte Carlo Stress Test

### MSFT

#### Outcome Distribution (10,000 simulations)

Metric               P5       Median   P95
Final Portfolio ($)  102,166  132,800  162,679
Sharpe Ratio         -1.935   -0.546   0.931
Win Rate             49.4%    53.5%    57.9%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.05%
P95 Max Drawdown             20.07%
Median CAGR                  2.88%
P95 Max Consecutive Losses   9
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  270
Mean Portfolio at Ruin      $69,654

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
16       94,748   101,697     108,654
33       93,606   103,530     113,225
49       93,014   105,155     117,046
66       92,979   107,189     120,925
83       92,946   108,914     124,235
99       93,211   110,417     127,327
116      93,590   112,206     130,345
133      93,766   113,915     133,171
149      94,512   115,642     135,916
166      95,248   117,238     138,940
182      95,699   118,806     141,762
199      96,354   120,429     144,466
216      96,985   122,023     147,209
232      97,461   123,863     149,797
249      98,735   125,774     152,287
266      99,438   127,482     155,020
282      100,221  129,105     157,290
299      101,248  130,842     160,188
316      102,166  132,800     162,679

### AAPL

#### Outcome Distribution (10,000 simulations)

Metric               P5       Median   P95
Final Portfolio ($)  106,242  137,191  166,699
Sharpe Ratio         -2.068   -0.709   0.664
Win Rate             51.7%    55.9%    60.0%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.01%
P95 Max Drawdown             19.07%
Median CAGR                  3.21%
P95 Max Consecutive Losses   9
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  185
Mean Portfolio at Ruin      $83,863

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
16       94,901   101,889     108,660
33       93,814   103,998     113,758
49       93,546   105,881     117,734
66       93,462   107,762     121,920
82       93,731   109,613     125,328
99       94,124   111,517     128,747
116      94,947   113,677     132,197
132      95,924   115,435     135,394
149      96,532   117,411     138,983
165      97,513   119,241     141,908
182      98,118   121,270     144,933
198      98,667   123,295     147,830
215      99,548   125,292     150,762
232      100,665  127,281     153,549
248      101,345  129,214     156,340
265      102,386  131,223     159,389
281      103,799  133,210     162,149
298      105,250  135,093     164,366
315      106,242  137,191     166,699

### AMZN

#### Outcome Distribution (10,000 simulations)

Metric               P5       Median   P95
Final Portfolio ($)  102,136  132,668  165,214
Sharpe Ratio         -2.479   -0.914   0.697
Win Rate             49.0%    53.8%    59.0%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.02%
P95 Max Drawdown             19.43%
Median CAGR                  2.87%
P95 Max Consecutive Losses   10
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  250
Mean Portfolio at Ruin      $54,422

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
15       94,785   101,528     109,041
30       93,470   103,182     113,889
45       92,693   104,846     117,771
61       92,415   106,713     121,531
76       92,494   108,429     125,032
91       92,885   110,081     128,028
106      93,383   111,910     131,413
122      93,818   113,744     134,464
137      94,115   115,370     137,487
152      94,937   117,082     140,149
167      95,848   118,699     143,189
183      96,574   120,697     146,274
198      97,273   122,466     149,326
213      97,906   124,116     151,812
228      98,608   125,770     154,327
244      99,319   127,468     157,316
259      100,367  129,083     160,065
274      101,266  130,885     162,470
290      102,136  132,668     165,214

### GOOGL

#### Outcome Distribution (10,000 simulations)

Metric               P5       Median   P95
Final Portfolio ($)  117,435  148,457  180,316
Sharpe Ratio         -2.742   -1.302   0.239
Win Rate             52.3%    56.7%    61.1%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.00%
P95 Max Drawdown             15.82%
Median CAGR                  4.03%
P95 Max Consecutive Losses   8
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  N/A
Mean Portfolio at Ruin      N/A

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
15       95,459   102,345     109,653
31       94,780   105,070     115,171
47       95,144   107,580     120,460
62       95,617   110,174     124,660
78       96,691   112,740     128,777
94       97,569   115,205     132,846
109      98,616   117,541     136,759
125      100,013  120,150     140,652
141      101,150  122,764     144,438
156      102,599  125,188     147,687
172      103,969  127,752     151,678
188      105,725  130,168     155,482
203      107,062  132,862     159,011
219      108,914  135,172     162,812
235      110,422  137,887     166,589
250      111,928  140,465     170,008
266      113,538  143,108     173,626
282      115,647  145,838     176,705
298      117,435  148,457     180,316

### WMT

#### Outcome Distribution (10,000 simulations)

Metric               P5       Median   P95
Final Portfolio ($)  119,930  147,662  175,810
Sharpe Ratio         -0.892   0.463    1.819
Win Rate             53.7%    57.9%    62.2%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.00%
P95 Max Drawdown             13.87%
Median CAGR                  3.98%
P95 Max Consecutive Losses   8
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  N/A
Mean Portfolio at Ruin      N/A

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
17       96,051   102,464     108,926
34       95,736   105,041     113,949
51       96,302   107,433     118,423
69       97,033   110,064     122,784
86       97,873   112,566     126,782
103      99,012   115,107     130,929
120      100,593  117,533     134,547
138      102,062  120,163     138,411
155      103,340  122,590     141,783
172      104,784  125,151     145,076
189      106,494  127,614     148,576
207      108,415  130,308     152,238
224      109,916  132,855     155,685
241      111,170  135,222     159,184
258      112,716  137,543     162,551
276      114,592  140,173     165,962
293      116,162  142,706     169,077
310      117,867  145,046     172,717
328      119,930  147,662     175,810

### CAT

#### Outcome Distribution (10,000 simulations)

Metric               P5       Median   P95
Final Portfolio ($)  106,583  138,022  169,156
Sharpe Ratio         -2.555   -1.167   0.299
Win Rate             50.3%    54.8%    59.4%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.00%
P95 Max Drawdown             18.80%
Median CAGR                  3.28%
P95 Max Consecutive Losses   9
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  N/A
Mean Portfolio at Ruin      N/A

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
16       94,875   101,900     109,272
32       93,988   103,787     114,245
48       93,733   105,675     118,435
65       93,491   107,756     122,539
81       93,723   109,812     126,448
97       94,268   111,680     130,055
114      94,530   113,867     133,435
130      95,317   115,922     136,557
146      95,939   117,985     139,202
163      96,898   119,985     142,653
179      97,652   121,964     145,560
195      98,789   123,720     148,770
212      99,714   125,833     151,662
228      100,399  127,612     155,093
244      101,917  129,656     157,962
261      102,802  131,631     160,940
277      103,879  133,721     164,045
293      104,953  135,734     166,452
310      106,583  138,022     169,156

### TSM

#### Outcome Distribution (10,000 simulations)

Metric               P5      Median   P95
Final Portfolio ($)  99,075  129,481  160,310
Sharpe Ratio         -2.179  -0.750   0.760
Win Rate             48.3%   52.9%    57.4%

#### Risk Metrics

Metric                       Value
P(Ruin) — equity falls >40%  0.03%
P95 Max Drawdown             20.69%
Median CAGR                  2.62%
P95 Max Consecutive Losses   10
Optimal Kelly Fraction       0.000

#### Ruin Analysis

Metric                      Value
Median Trade at First Ruin  212
Mean Portfolio at Ruin      $57,772

#### Equity Confidence Band

Trade #  P5 ($)   Median ($)  P95 ($)
0        100,000  100,000     100,000
17       94,719   101,466     108,559
35       93,119   103,111     113,084
52       92,662   104,509     116,961
70       92,292   106,011     120,359
87       92,478   107,619     123,423
105      92,696   109,339     126,805
122      93,078   110,877     129,366
140      93,132   112,407     132,575
157      93,619   113,833     135,032
175      93,955   115,660     137,909
192      94,469   117,172     140,611
210      94,682   118,905     143,134
227      95,137   120,339     145,882
245      95,801   121,925     148,194
262      96,399   123,390     150,716
280      96,940   124,749     153,318
297      97,752   126,398     155,822
315      98,354   127,952     157,910
333      99,075   129,481     160,310


## Historical Alpha Learning

_Only 0 historical runs recorded. A minimum of 10 is needed to compute per-regime/strategy statistics. Run the pipeline on more dates to accumulate the performance database._

## Execution Brief

**NYSE:** CLOSED — weekend — opens Monday 09:30 ET (in 23h 58m)  

**Warnings:**
- WARNING: NYSE CLOSED — weekend (opens Monday 09:30 ET (in 23h 58m)) — live quotes unavailable; all slippage estimates are ATR-based

### Active Signals — Enter at Next Open


### BA — ENTER NOW

Field                  Value
Entry price            $190.52
Stop loss              $176.86
Position size          73 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.6831/share → $49.86 total
Adjusted net risk      $1,050
Market impact          negligible
ADV (20d)              7,582,885 shares

**Note:** NYSE closed — queue order for next open; slippage is ATR-estimated

### GOOGL — ENTER NOW

Field                  Value
Entry price            $274.34
Stop loss              $262.59
Position size          85 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.7837/share → $66.61 total
Adjusted net risk      $1,067
Market impact          negligible
ADV (20d)              30,037,460 shares

**Note:** NYSE closed — queue order for next open; slippage is ATR-estimated

### JNJ — ENTER NOW

Field                  Value
Entry price            $240.45
Stop loss              $234.25
Position size          161 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.4133/share → $66.54 total
Adjusted net risk      $1,067
Market impact          negligible
ADV (20d)              8,220,515 shares

**Note:** NYSE closed — queue order for next open; slippage is ATR-estimated

---

### Pending Signals — Monitor Daily

_These tickers passed all 3 validation stages (backtest → diagnostics → Monte Carlo)_
_but have not yet triggered their entry signal. The setup below shows what the_
_trade will look like when conditions are met._


### MSFT — PENDING — conditions not yet met

Field                  Value
Entry price            $356.95
Stop loss              $343.87
Position size          76 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.8722/share → $66.29 total
Adjusted net risk      $1,066
Market impact          negligible
ADV (20d)              33,436,540 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### AAPL — PENDING — conditions not yet met

Field                  Value
Entry price            $248.92
Stop loss              $240.51
Position size          118 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.5611/share → $66.21 total
Adjusted net risk      $1,066
Market impact          negligible
ADV (20d)              41,772,895 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### JPM — PENDING — conditions not yet met

Field                  Value
Entry price            $282.98
Stop loss              $272.07
Position size          91 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.7274/share → $66.20 total
Adjusted net risk      $1,066
Market impact          negligible
ADV (20d)              11,410,310 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### AMZN — PENDING — conditions not yet met

Field                  Value
Entry price            $199.44
Stop loss              $190.75
Position size          115 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.5795/share → $66.65 total
Adjusted net risk      $1,067
Market impact          negligible
ADV (20d)              45,294,655 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### GS — PENDING — conditions not yet met

Field                  Value
Entry price            $803.29
Stop loss              $746.59
Position size          17 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $2.8350/share → $48.20 total
Adjusted net risk      $1,048
Market impact          negligible
ADV (20d)              2,605,245 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### CVX — PENDING — conditions not yet met

Field                  Value
Entry price            $208.00
Stop loss              $201.69
Position size          158 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.4203/share → $66.41 total
Adjusted net risk      $1,066
Market impact          negligible
ADV (20d)              15,211,415 shares

**Conditions to watch (enter when ALL are met):**

- Price must close **above $207.79** (N-day high breakout)
- Volume must exceed **18,253,698 shares** (volume confirmation)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### TSLA — PENDING — conditions not yet met

Field                  Value
Entry price            $362.01
Stop loss              $335.04
Position size          37 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $1.3484/share → $49.89 total
Adjusted net risk      $1,050
Market impact          negligible
ADV (20d)              60,700,350 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### MRK — PENDING — conditions not yet met

Field                  Value
Entry price            $119.69
Stop loss              $115.76
Position size          254 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.2622/share → $66.59 total
Adjusted net risk      $1,067
Market impact          negligible
ADV (20d)              10,714,090 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### WMT — PENDING — conditions not yet met

Field                  Value
Entry price            $122.95
Stop loss              $118.74
Position size          237 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.2809/share → $66.57 total
Adjusted net risk      $1,067
Market impact          negligible
ADV (20d)              21,535,920 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### CAT — PENDING — conditions not yet met

Field                  Value
Entry price            $695.75
Stop loss              $648.00
Position size          20 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $2.3872/share → $47.74 total
Adjusted net risk      $1,048
Market impact          negligible
ADV (20d)              2,774,070 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### SPY — PENDING — conditions not yet met

Field                  Value
Entry price            $634.72
Stop loss              $620.03
Position size          68 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $0.9798/share → $66.63 total
Adjusted net risk      $1,067
Market impact          negligible
ADV (20d)              98,273,705 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

### TSM — PENDING — conditions not yet met

Field                  Value
Entry price            $326.90
Stop loss              $302.11
Position size          40 shares
Dollar risk (1% rule)  $1,000
Slippage est.          $1.2395/share → $49.58 total
Adjusted net risk      $1,050
Market impact          negligible
ADV (20d)              14,027,955 shares

**Conditions to watch (enter when ALL are met):**

- Entry condition: **alpha_signal > 0.55** (cross-sectional alpha signal threshold)

_Setup above is projected at current ATR/price. Actual size/stop will be recalculated at entry bar._

**Note:** PENDING — conditions not yet met; use this setup when signal fires; NYSE closed — queue order for next open; slippage is ATR-estimated

---

### Portfolio Risk Summary

                Count  Dollar Risk  % of Portfolio
Active signals  3      $3,183       3.2%