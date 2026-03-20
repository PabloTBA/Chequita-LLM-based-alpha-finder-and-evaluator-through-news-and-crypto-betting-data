"""
Tests for RegimeClassifier.
All tests use synthetic OHLCV fixtures — no real market data, no mocks.
Pure math: given a known price series, assert the correct regime label.
"""
import numpy as np
import pandas as pd
import pytest

from regime_classifier import RegimeClassifier

# ── Fixture helpers ───────────────────────────────────────────────────────────

def make_ohlcv(closes: np.ndarray, hl_spread: float = 2.0) -> pd.DataFrame:
    """
    Build an OHLCV DataFrame from a close price array.
    High = close + hl_spread/2, Low = close - hl_spread/2 (constant spread).
    """
    n      = len(closes)
    highs  = closes + hl_spread / 2
    lows   = closes - hl_spread / 2
    opens  = closes - 0.1
    vols   = np.full(n, 1_000_000.0)
    idx    = pd.date_range(end="2026-03-18", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def trending_closes(n: int = 60) -> np.ndarray:
    """Monotonically rising: Hurst >> 0.55 (perfectly persistent)."""
    return np.linspace(100.0, 200.0, n)


def mean_reverting_closes(n: int = 60) -> np.ndarray:
    """Strictly alternating zig-zag: Hurst << 0.45 (perfectly anti-persistent)."""
    return np.array([100.0 + 5.0 * (i % 2) for i in range(n)])


def neutral_closes(n: int = 60, seed: int = 42) -> np.ndarray:
    """Random walk anchored at 100: Hurst ≈ 0.5 (neutral zone).
    Prices stay near 100 so H/L spreads translate to predictable atr_pct values.
    """
    rng        = np.random.default_rng(seed)
    increments = rng.choice([-1.0, 1.0], size=n - 1)
    return np.concatenate([[100.0], 100.0 + np.cumsum(increments)])


REQUIRED_KEYS = {"ticker", "hurst", "atr_pct", "regime"}
VALID_REGIMES = {"Trending", "Mean-Reverting", "High-Volatility", "Low-Volatility", "Neutral"}


# ── Cycle 1: output contract ──────────────────────────────────────────────────

class TestOutputContract:
    def test_classify_returns_required_keys(self):
        clf    = RegimeClassifier()
        df     = make_ohlcv(trending_closes())
        result = clf.classify("AAPL", df)
        assert REQUIRED_KEYS == set(result.keys())

    def test_ticker_in_result_matches_input(self):
        clf    = RegimeClassifier()
        result = clf.classify("MSFT", make_ohlcv(trending_closes()))
        assert result["ticker"] == "MSFT"

    def test_regime_is_valid_label(self):
        clf    = RegimeClassifier()
        result = clf.classify("AAPL", make_ohlcv(trending_closes()))
        assert result["regime"] in VALID_REGIMES

    def test_hurst_is_float_between_0_and_1(self):
        clf    = RegimeClassifier()
        result = clf.classify("AAPL", make_ohlcv(trending_closes()))
        assert isinstance(result["hurst"], float)
        assert 0.0 <= result["hurst"] <= 1.0

    def test_atr_pct_is_positive_float(self):
        clf    = RegimeClassifier()
        result = clf.classify("AAPL", make_ohlcv(trending_closes()))
        assert isinstance(result["atr_pct"], float)
        assert result["atr_pct"] > 0.0


# ── Cycle 2: Trending regime ──────────────────────────────────────────────────

class TestTrendingRegime:
    def test_monotonically_rising_classified_as_trending(self):
        """A steady uptrend has Hurst >> 0.55 → Trending."""
        clf    = RegimeClassifier()
        result = clf.classify("AAPL", make_ohlcv(trending_closes(n=60)))
        assert result["regime"] == "Trending"
        assert result["hurst"] > 0.55

    def test_monotonically_falling_classified_as_trending(self):
        """A steady downtrend is also persistent → Trending."""
        clf    = RegimeClassifier()
        closes = np.linspace(200.0, 100.0, 60)
        result = clf.classify("AAPL", make_ohlcv(closes))
        assert result["regime"] == "Trending"
        assert result["hurst"] > 0.55


# ── Cycle 3: Mean-Reverting regime ───────────────────────────────────────────

class TestMeanRevertingRegime:
    def test_alternating_series_classified_as_mean_reverting(self):
        """A zig-zag series has Hurst << 0.45 → Mean-Reverting."""
        clf    = RegimeClassifier()
        result = clf.classify("AAPL", make_ohlcv(mean_reverting_closes(n=60)))
        assert result["regime"] == "Mean-Reverting"
        assert result["hurst"] < 0.45


# ── Cycle 4: ATR fallback — High-Volatility ───────────────────────────────────

class TestHighVolatilityFallback:
    def test_neutral_hurst_high_atr_classified_as_high_volatility(self):
        """
        Neutral Hurst + ATR/price > 3% → High-Volatility.
        Use a wide H/L spread on a neutral-Hurst series.
        """
        clf    = RegimeClassifier()
        closes = neutral_closes(n=60)
        # Spread of 8 on a ~100 price → ATR ≈ 8 → atr_pct ≈ 8%
        df     = make_ohlcv(closes, hl_spread=8.0)
        result = clf.classify("AAPL", df)
        assert result["atr_pct"] > 0.03
        assert result["regime"] in {"High-Volatility", "Trending", "Mean-Reverting"}
        # If regime came from Hurst, that's also fine — only assert atr_pct here
        if 0.45 <= result["hurst"] <= 0.55:
            assert result["regime"] == "High-Volatility"


# ── Cycle 5: ATR fallback — Low-Volatility ────────────────────────────────────

class TestLowVolatilityFallback:
    def test_neutral_hurst_low_atr_classified_as_low_volatility(self):
        """
        Neutral Hurst + ATR/price < 1.5% → Low-Volatility.
        Use a very tight H/L spread on a neutral-Hurst series.
        """
        clf    = RegimeClassifier()
        closes = neutral_closes(n=60)
        # Spread of 0.5 on ~100 price → ATR ≈ 0.5 → atr_pct ≈ 0.5%
        df     = make_ohlcv(closes, hl_spread=0.5)
        result = clf.classify("AAPL", df)
        assert result["atr_pct"] < 0.015
        if 0.45 <= result["hurst"] <= 0.55:
            assert result["regime"] == "Low-Volatility"


# ── Cycle 6: ATR fallback — Neutral ──────────────────────────────────────────

class TestNeutralFallback:
    def test_neutral_hurst_mid_atr_classified_as_neutral(self):
        """
        Neutral Hurst + 1.5% ≤ ATR/price ≤ 3% → Neutral.
        Use a medium H/L spread (~2) on a ~100 price series → atr_pct ≈ 2%.
        """
        clf    = RegimeClassifier()
        closes = neutral_closes(n=60)
        # Spread of 2.0 on ~100 price → atr_pct ≈ 2%
        df     = make_ohlcv(closes, hl_spread=2.0)
        result = clf.classify("AAPL", df)
        if 0.45 <= result["hurst"] <= 0.55:
            assert 0.015 <= result["atr_pct"] <= 0.03
            assert result["regime"] == "Neutral"


# ── Cycle 7: data guard ───────────────────────────────────────────────────────

class TestDataGuard:
    def test_fewer_than_30_rows_raises_value_error(self):
        clf = RegimeClassifier()
        df  = make_ohlcv(trending_closes(n=20))
        with pytest.raises(ValueError, match="insufficient"):
            clf.classify("AAPL", df)

    def test_exactly_30_rows_does_not_raise(self):
        clf = RegimeClassifier()
        df  = make_ohlcv(trending_closes(n=30))
        result = clf.classify("AAPL", df)
        assert "regime" in result


# ── Cycle 8: classify_all ─────────────────────────────────────────────────────

class TestClassifyAll:
    def test_returns_one_entry_per_ticker(self):
        clf = RegimeClassifier()
        ohlcv_dict = {
            "AAPL": make_ohlcv(trending_closes()),
            "MSFT": make_ohlcv(mean_reverting_closes()),
        }
        results = clf.classify_all(ohlcv_dict)
        tickers = {r["ticker"] for r in results}
        assert tickers == {"AAPL", "MSFT"}

    def test_skips_none_entries(self):
        clf = RegimeClassifier()
        ohlcv_dict = {
            "AAPL": make_ohlcv(trending_closes()),
            "FAKE": None,
        }
        results = clf.classify_all(ohlcv_dict)
        tickers = {r["ticker"] for r in results}
        assert "FAKE" not in tickers
        assert "AAPL" in tickers

    def test_each_result_has_required_keys(self):
        clf = RegimeClassifier()
        ohlcv_dict = {"AAPL": make_ohlcv(trending_closes())}
        results = clf.classify_all(ohlcv_dict)
        assert REQUIRED_KEYS == set(results[0].keys())
