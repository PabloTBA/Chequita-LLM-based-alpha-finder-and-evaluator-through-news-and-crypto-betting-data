
import pandas as pd
import numpy as np
import yfinance as yf
from report_generator import _sharpe_from_returns

# Reproduce the exact same data the pipeline passes
spy_ohlcv = yf.download("SPY", period="5y", auto_adjust=True)

spy_close   = spy_ohlcv["Close"].squeeze().astype(float)
spy_daily   = spy_close.pct_change().fillna(0.0)
spy_ma50    = spy_close.rolling(50).mean()
in_pos_mask = (spy_close > spy_ma50).shift(1).fillna(False)
daily_rf    = 0.045 / 252

# This is the fixed line
ma_rets = spy_daily.where(in_pos_mask, daily_rf)

print("B&H return  :", round(float((spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0]) * 100, 2), "%")
print("MA cross ret:", round(float((1 + ma_rets).prod() - 1) * 100, 2), "%")
print("MA Sharpe   :", round(_sharpe_from_returns(ma_rets), 3))
print("✓ No error")
