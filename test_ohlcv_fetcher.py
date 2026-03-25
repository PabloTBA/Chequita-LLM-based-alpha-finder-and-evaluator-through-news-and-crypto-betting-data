"""
Tests for OHLCVFetcher.
All compute_features tests use synthetic OHLCV fixtures — no real yfinance calls.
fetch() tests use a mocked yfinance.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from ohlcv_fetcher import OHLCVFetcher

# ── Fixtures ──────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "return_20d",
    "rsi_14",
    "atr_14",
    "atr_pct",
    "52w_high_prox",
    "52w_low_prox",
    "volume_ratio_30d",
}

def make_ohlcv(n: int = 300, close_start: float = 100.0, volume: float = 1_000_000.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with n rows of predictable data."""
    closes  = np.linspace(close_start, close_start + n - 1, n)
    highs   = closes + 1.0
    lows    = closes - 1.0
    opens   = closes - 0.5
    volumes = np.full(n, volume)
    idx     = pd.date_range(end="2026-03-18", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def make_alternating_ohlcv(n: int = 300) -> pd.DataFrame:
    """Close alternates up/down so RSI hovers near 50."""
    closes  = np.array([100.0 + (1 if i % 2 == 0 else -1) for i in range(n)], dtype=float)
    closes  = np.cumsum(np.where(np.arange(n) % 2 == 0, 1, -1)) + 100.0
    highs   = closes + 0.5
    lows    = closes - 0.5
    opens   = closes
    volumes = np.full(n, 1_000_000.0)
    idx     = pd.date_range(end="2026-03-18", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_compute_features_returns_all_required_keys(self):
        fetcher = OHLCVFetcher()
        df      = make_ohlcv()
        result  = fetcher.compute_features(df)
        assert REQUIRED_KEYS == set(result.keys())

    def test_all_values_are_finite_floats(self):
        fetcher = OHLCVFetcher()
        result  = fetcher.compute_features(make_ohlcv())
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float"
            assert np.isfinite(val),       f"{key} is not finite: {val}"


# ── Cycle 2: return_20d ───────────────────────────────────────────────────────

class TestReturn20d:
    def test_return_20d_formula(self):
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=300, close_start=100.0)
        closes  = df["Close"].values
        expected = (closes[-1] - closes[-21]) / closes[-21]
        result   = fetcher.compute_features(df)
        assert abs(result["return_20d"] - expected) < 1e-9


# ── Cycle 3: RSI(14) ─────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_all_up_prices_near_100(self):
        """Strictly rising prices → RSI should be very high (close to 100)."""
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=300, close_start=100.0)   # monotonically rising
        result  = fetcher.compute_features(df)
        assert result["rsi_14"] > 90.0

    def test_rsi_bounded_0_to_100(self):
        fetcher = OHLCVFetcher()
        for df in [make_ohlcv(), make_alternating_ohlcv()]:
            result = fetcher.compute_features(df)
            assert 0.0 <= result["rsi_14"] <= 100.0


# ── Cycle 4: ATR(14) ─────────────────────────────────────────────────────────

class TestATR:
    def test_atr_equals_constant_true_range(self):
        """When High-Low = 2 and no overnight gaps, ATR(14) = 2."""
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=300)   # High = Close+1, Low = Close-1, spread = 2
        result  = fetcher.compute_features(df)
        # True range = max(H-L, |H-prev_C|, |L-prev_C|)
        # For a smooth rising series: H-L=2, gaps are ~1 → TR dominated by H-L≈2
        assert 1.5 < result["atr_14"] < 3.0   # generous bounds for Wilder smoothing

    def test_atr_pct_equals_atr_over_last_close(self):
        fetcher  = OHLCVFetcher()
        df       = make_ohlcv(n=300)
        result   = fetcher.compute_features(df)
        expected = result["atr_14"] / df["Close"].iloc[-1]
        assert abs(result["atr_pct"] - expected) < 1e-9


# ── Cycle 5: volume ratio ─────────────────────────────────────────────────────

class TestVolumeRatio:
    def test_volume_ratio_equals_1_when_constant_volume(self):
        """If every day has the same volume, ratio = 1.0."""
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=300, volume=500_000.0)
        result  = fetcher.compute_features(df)
        assert abs(result["volume_ratio_30d"] - 1.0) < 1e-9

    def test_volume_ratio_double_when_last_day_2x(self):
        fetcher        = OHLCVFetcher()
        df             = make_ohlcv(n=300, volume=500_000.0)
        df.iloc[-1, df.columns.get_loc("Volume")] = 1_000_000.0
        result         = fetcher.compute_features(df)
        # last volume / mean(last 30 including today) ≠ exactly 2.0
        # but last volume / mean(prev 29 + today) is close to 2
        assert result["volume_ratio_30d"] > 1.5


# ── Cycle 6: 52-week proximity ───────────────────────────────────────────────

class TestFiftyTwoWeekProximity:
    def test_at_52w_high_prox_equals_1(self):
        """Last close is the 52-week high → 52w_high_prox = 1.0."""
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=300, close_start=100.0)   # monotonically rising
        result  = fetcher.compute_features(df)
        assert abs(result["52w_high_prox"] - 1.0) < 1e-6

    def test_52w_low_prox_gt_1_when_price_risen(self):
        """If price rose over the year, current price > 52w low → prox > 1."""
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=300, close_start=100.0)
        result  = fetcher.compute_features(df)
        assert result["52w_low_prox"] > 1.0


# ── Cycle 7: insufficient data guard ─────────────────────────────────────────

class TestInsufficientData:
    def test_fewer_than_30_rows_raises(self):
        fetcher = OHLCVFetcher()
        df      = make_ohlcv(n=20)
        with pytest.raises(ValueError, match="insufficient"):
            fetcher.compute_features(df)


# ── Cycle 8: fetch() with mocked yfinance ────────────────────────────────────

class TestFetch:
    def test_fetch_returns_dict_keyed_by_ticker(self):
        fetcher = OHLCVFetcher()
        # yf.download is called once per ticker, returns a DataFrame
        with patch("ohlcv_fetcher.yf.download", return_value=make_ohlcv()):
            result = fetcher.fetch(["AAPL", "MSFT"])
        assert set(result.keys()) == {"AAPL", "MSFT"}
        assert isinstance(result["AAPL"], pd.DataFrame)

    def test_fetch_empty_ticker_returns_none_no_crash(self):
        fetcher = OHLCVFetcher()
        with patch("ohlcv_fetcher.yf.download", return_value=pd.DataFrame()):
            result = fetcher.fetch(["FAKE"])
        assert result["FAKE"] is None
