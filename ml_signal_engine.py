"""
MLSignalEngine
==============
Runs four complementary signal models and averages their probabilities into a
single ``ml_signal`` column (values in [0, 1]).

Models
------
1. Cross-sectional GBM    — one model trained on a panel of ALL tickers.
                             Each row is (ticker, date); features include
                             absolute values AND cross-sectional z-scores so
                             the model captures universe-relative ranking.

2. Regime-conditional GBM — per-ticker walk-forward GBM that appends two
                             regime flags (trending / high-vol) computed from
                             existing features.  The model implicitly learns
                             regime-specific entry patterns without requiring
                             the external RegimeClassifier.

3. Online adaptive SGD    — SGDClassifier updated via ``partial_fit`` on
                             every new bar (with _FORWARD-bar label lag).
                             Never goes stale between quarterly batch refits.

4. Calibrated ensemble    — GBM + LogisticRegression + RandomForest trained
                             in parallel; their raw probabilities are averaged.
                             Averaging across diverse model families reduces
                             variance and naturally improves calibration.

Final ml_signal = mean of available model probabilities across all four models.

Walk-forward properties (common to all models)
----------------------------------------------
  Minimum training bars : 252  (~1 year)
  Batch refit every     : 63 bars  (~quarterly)
  Forward target horizon: 5 bars
  All features          : shift(1)-lagged — zero look-ahead

Public interface
----------------
    engine = MLSignalEngine()
    enriched = engine.compute(ohlcv_dict)   # dict[str, DataFrame | None]
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# ── Constants ─────────────────────────────────────────────────────────────────
_MIN_TRAIN  = 252    # ~1 year minimum before any model fires
_REFIT      = 63     # quarterly batch refit
_FORWARD    = 5      # 5-day forward return target
_N_EST      = 200    # GBM trees
_DEPTH      = 4      # GBM max depth (shallow = less overfit)
_LR         = 0.05   # GBM learning rate
_MIN_LEAF   = 20     # GBM / RF min samples per leaf
_SUBSAMPLE  = 0.8    # row subsampling

# Feature column names (base 12, shift(1)-lagged)
_BASE = [
    "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_pct", "vol_ratio_20",
    "bb_position", "mom_12_1", "vol_z_20",
    "close_to_ma50", "close_to_ma200", "realized_vol_20",
]
# Cross-sectional z-scores appended for model 1
_CS   = [f"cs_{c}" for c in _BASE]
# Regime flags appended for model 2
_REG  = ["reg_trend", "reg_highvol"]


# ── Feature helpers ───────────────────────────────────────────────────────────

def _rsi(close: pd.Series, p: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """12 shift(1)-lagged features.  Zero look-ahead — safe for walk-forward."""
    c, h, lo, v = (df[col].astype(float) for col in ("Close", "High", "Low", "Volume"))
    tr    = pd.concat([(h - lo), (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    bm    = c.rolling(20).mean();  bs = c.rolling(20).std(ddof=1)
    vm    = v.rolling(20).mean()
    return pd.DataFrame({
        "ret_5d":          c.pct_change(5).shift(1),
        "ret_10d":         c.pct_change(10).shift(1),
        "ret_20d":         c.pct_change(20).shift(1),
        "rsi_14":          _rsi(c).shift(1),
        "atr_pct":         (atr / c).shift(1),
        "vol_ratio_20":    (v / vm.replace(0, np.nan)).shift(1),
        "bb_position":     ((c - (bm - bs)) / (2 * bs).replace(0, np.nan)).shift(1),
        "mom_12_1":        c.shift(22) / c.shift(253).replace(0, np.nan) - 1,
        "vol_z_20":        ((v - vm) / v.rolling(20).std(ddof=1).replace(0, np.nan)).shift(1),
        "close_to_ma50":   (c / c.rolling(50).mean().replace(0, np.nan) - 1).shift(1),
        "close_to_ma200":  (c / c.rolling(200).mean().replace(0, np.nan) - 1).shift(1),
        "realized_vol_20": c.pct_change().rolling(20).std(ddof=1).shift(1),
    }, index=df.index)[_BASE]


def _build_target(df: pd.DataFrame) -> pd.Series:
    """Binary: 1 if 5-day forward return > 0.  Used only during training."""
    c = df["Close"].astype(float)
    return (c.shift(-_FORWARD) / c - 1 > 0).astype(int)


def _add_regime_features(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Append two regime one-hot columns derived from existing features:
      reg_trend   — abs(ret_20d) is more than 1 std above its 63-bar mean
      reg_highvol — realized_vol_20 is > 1.5× its 63-bar rolling mean
    Both are computed from shift(1)-lagged features so zero look-ahead.
    """
    rv  = feat["realized_vol_20"].fillna(0)
    r20 = feat["ret_20d"].abs().fillna(0)
    rv_mean  = rv.rolling(63, min_periods=21).mean().replace(0, np.nan)
    r20_std  = r20.rolling(63, min_periods=21).std(ddof=1).replace(0, np.nan)
    return feat.assign(
        reg_trend   = (r20 / r20_std > 1.0).astype(float),
        reg_highvol = (rv  > 1.5 * rv_mean).astype(float),
    )


# ── Model factories ───────────────────────────────────────────────────────────

def _make_gbm() -> Any:
    if _HAS_LGBM:
        return _LGBMClassifier(
            n_estimators=_N_EST, max_depth=_DEPTH, learning_rate=_LR,
            min_child_samples=_MIN_LEAF, subsample=_SUBSAMPLE,
            subsample_freq=1, verbose=-1,
        )
    return GradientBoostingClassifier(
        n_estimators=_N_EST, max_depth=_DEPTH, learning_rate=_LR,
        min_samples_leaf=_MIN_LEAF, subsample=_SUBSAMPLE,
    )

def _make_lr() -> LogisticRegression:
    return LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")

def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=_MIN_LEAF,
        n_jobs=-1, random_state=42,   # use all CPU cores for tree building
    )

def _make_sgd() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.01,
        learning_rate="optimal", random_state=42,
    )


# ── Generic walk-forward utility ──────────────────────────────────────────────

def _walk_forward(
    X:          pd.DataFrame,
    y:          pd.Series,
    fit_fn:     Callable[[pd.DataFrame, pd.Series], Any],
    predict_fn: Callable[[Any, pd.DataFrame], float],
    min_train:  int = _MIN_TRAIN,
    refit:      int = _REFIT,
) -> pd.Series:
    """
    Expanding-window walk-forward: fit on [0 : i-_FORWARD], predict bar i.
    Refits every ``refit`` bars; requires ``min_train`` clean rows to start.
    Returns a pd.Series of probabilities (NaN where unavailable).
    """
    n, sig, model, last = len(X), pd.Series(np.nan, index=X.index, dtype=float), None, -refit
    for i in range(min_train, n):
        if (i - last) >= refit or model is None:
            end = i - _FORWARD
            if end < min_train:
                continue
            Xtr, ytr = X.iloc[:end], y.iloc[:end]
            ok = Xtr.notna().all(axis=1) & ytr.notna()
            if ok.sum() < min_train:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = fit_fn(Xtr.loc[ok], ytr.loc[ok])
                last = i
            except Exception:
                pass
        if model is None:
            continue
        row = X.iloc[[i]]
        if row.isna().any(axis=1).iloc[0]:
            continue
        try:
            sig.iloc[i] = predict_fn(model, row)
        except Exception:
            pass
    return sig


# ── MLSignalEngine ────────────────────────────────────────────────────────────

class MLSignalEngine:
    """
    Compute per-ticker ML signals by ensembling four complementary models.
    Final ml_signal = mean of available model probabilities.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def compute(
        self, ohlcv_dict: dict[str, pd.DataFrame | None]
    ) -> dict[str, pd.DataFrame | None]:
        """
        Enrich every ticker's OHLCV DataFrame with an ``ml_signal`` column.
        Returns a new mapping with the enriched DataFrames.
        """
        # Build base features for all valid tickers
        feat_map:   dict[str, pd.DataFrame] = {}
        target_map: dict[str, pd.Series]    = {}
        for ticker, df in ohlcv_dict.items():
            if df is None or df.empty:
                continue
            try:
                feat_map[ticker]   = _build_base_features(df)
                target_map[ticker] = _build_target(df)
            except Exception:
                pass

        # ── Model 1: cross-sectional (universe-level, one model for all) ──────
        cs_sigs = self._cs_signal(feat_map, target_map)

        # ── Models 2-4: per-ticker, run in parallel ───────────────────────────
        # Each ticker's regime/online/ensemble signals are independent so we
        # dispatch them concurrently.  ThreadPoolExecutor works here because
        # sklearn/numpy release the GIL during fit/predict.
        # CS signal (model 1) is already computed above (universe-level).
        valid_tickers = [t for t, df in ohlcv_dict.items()
                         if df is not None and not df.empty and t in feat_map]

        def _compute_ticker(ticker: str) -> tuple[str, pd.DataFrame]:
            df     = ohlcv_dict[ticker]
            feat   = feat_map[ticker]
            target = target_map[ticker]
            try:
                reg_sig = self._regime_signal(feat, target)
                onl_sig = self._online_signal(feat, target)
                ens_sig = self._ensemble_signal(feat, target)
                cs_sig  = cs_sigs.get(ticker)

                parts  = [s for s in (cs_sig, reg_sig, onl_sig, ens_sig) if s is not None]
                ml_sig = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=df.index)

                n_valid = int(ml_sig.notna().sum())
                if self.verbose:
                    print(f"  [MLSignal] {ticker}: {n_valid}/{len(df)} bars "
                          f"| CS={cs_sig is not None} Reg=✓ Onl=✓ Ens=✓")
                out = df.copy()
                out["ml_signal"] = ml_sig
            except Exception as exc:
                if self.verbose:
                    print(f"  [MLSignal] {ticker}: failed ({exc}), ml_signal=NaN")
                out = df.copy()
                out["ml_signal"] = np.nan
            return ticker, out

        result: dict[str, pd.DataFrame | None] = {
            t: df for t, df in ohlcv_dict.items()
            if df is None or df.empty or t not in feat_map
        }
        n_workers = min(len(valid_tickers), 4)  # cap at 4 to avoid memory pressure
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_compute_ticker, t): t for t in valid_tickers}
            for future in as_completed(futures):
                ticker, out = future.result()
                result[ticker] = out
        return result

    # ── Model 1: Cross-sectional GBM ─────────────────────────────────────────

    def _cs_signal(
        self,
        feat_map:   dict[str, pd.DataFrame],
        target_map: dict[str, pd.Series],
    ) -> dict[str, pd.Series]:
        """
        Universe panel walk-forward.

        At each refit bar t, build a stacked (ticker × date) training matrix
        where each row has:
          - the 12 base features for that (ticker, date)
          - 12 cross-sectional z-scores: how this ticker ranks vs the universe
            on each feature at that date

        A single GBM is trained on the panel and predicts all tickers at t.
        The CS z-scores make the signal market-neutral and regime-adaptive.
        """
        if not feat_map:
            return {}

        tickers    = list(feat_map.keys())
        all_cols   = _BASE + _CS

        # Compute CS z-scores: for each base feature, z-score across tickers at each date
        cs_map: dict[str, pd.DataFrame] = {t: pd.DataFrame(index=feat_map[t].index) for t in tickers}
        for col in _BASE:
            cross     = pd.DataFrame({t: feat_map[t][col] for t in tickers})
            col_mean  = cross.mean(axis=1)
            col_std   = cross.std(axis=1, ddof=1).replace(0, np.nan)
            z_cross   = cross.sub(col_mean, axis=0).div(col_std, axis=0)
            for t in tickers:
                cs_map[t][f"cs_{col}"] = z_cross[t]

        # Augmented features = base + CS z-scores
        aug: dict[str, pd.DataFrame] = {
            t: pd.concat([feat_map[t][_BASE], cs_map[t][_CS]], axis=1)
            for t in tickers
        }

        # Pre-build numpy panel for fast slicing:
        # arrays of (row_date_position, ticker, X_row, y_val)
        all_dates = sorted(set().union(*[set(feat_map[t].index) for t in tickers]))
        date_pos  = {d: i for i, d in enumerate(all_dates)}
        n_dates   = len(all_dates)

        if n_dates < _MIN_TRAIN:
            return {t: pd.Series(np.nan, index=feat_map[t].index) for t in tickers}

        rows_dp, rows_t, rows_X, rows_y = [], [], [], []
        for t in tickers:
            af  = aug[t][all_cols]
            tgt = target_map[t]
            common = af.index.intersection(tgt.index)
            for d in common:
                xrow = af.loc[d].values
                yval = tgt.loc[d]
                if pd.isna(yval):
                    continue
                rows_dp.append(date_pos[d])
                rows_t.append(t)
                rows_X.append(xrow)
                rows_y.append(float(yval))

        if not rows_dp:
            return {t: pd.Series(np.nan, index=feat_map[t].index) for t in tickers}

        arr_dp = np.array(rows_dp, dtype=np.int32)
        arr_X  = np.array(rows_X,  dtype=np.float64)
        arr_y  = np.array(rows_y,  dtype=np.float64)

        sigs: dict[str, pd.Series] = {
            t: pd.Series(np.nan, index=feat_map[t].index) for t in tickers
        }
        model, last = None, -_REFIT

        for bar_i in range(_MIN_TRAIN, n_dates):
            # Refit
            if (bar_i - last) >= _REFIT or model is None:
                cut   = bar_i - _FORWARD
                if cut < _MIN_TRAIN:
                    continue
                # Rolling 2-year window to bound training time
                window_lo = max(0, cut - 504)
                mask = (arr_dp >= window_lo) & (arr_dp < cut) & (~np.isnan(arr_X).any(axis=1))
                if mask.sum() < _MIN_TRAIN:
                    continue
                try:
                    m = _make_gbm()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(arr_X[mask], arr_y[mask])
                    model, last = m, bar_i
                except Exception:
                    pass

            if model is None:
                continue

            # Predict each ticker at bar_i
            pred_mask = (arr_dp == bar_i) & (~np.isnan(arr_X).any(axis=1))
            if not pred_mask.any():
                continue
            try:
                probs = model.predict_proba(arr_X[pred_mask])[:, 1]
                for prob, t in zip(probs, np.array(rows_t)[pred_mask]):
                    date = all_dates[bar_i]
                    if date in sigs[t].index:
                        sigs[t].loc[date] = float(prob)
            except Exception:
                pass

        return sigs

    # ── Model 2: Regime-conditional GBM ──────────────────────────────────────

    def _regime_signal(self, feat: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        Walk-forward GBM with two appended regime flags (reg_trend, reg_highvol).
        The model learns regime-specific entry patterns without needing the
        external RegimeClassifier — the flags are derived from existing features.
        """
        X = _add_regime_features(feat)[_BASE + _REG]
        return _walk_forward(
            X, target,
            fit_fn=lambda Xtr, ytr: _fit(Xtr, ytr, _make_gbm()),
            predict_fn=lambda m, row: float(m.predict_proba(row.values)[0, 1]),
        )

    # ── Model 3: Online adaptive SGD ─────────────────────────────────────────

    def _online_signal(self, feat: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        SGDClassifier with ``partial_fit`` — updates every bar with the new
        (feature, label) pair as it becomes available (_FORWARD-bar lag).
        Never goes stale: adapts to the current market regime continuously.

        Warm-up: batch fit on the first _MIN_TRAIN - _FORWARD bars.
        Online:  one partial_fit per bar thereafter.
        """
        n   = len(feat)
        sig = pd.Series(np.nan, index=feat.index, dtype=float)
        mdl = _make_sgd()

        # Warm-up batch
        Xw  = feat.iloc[: _MIN_TRAIN - _FORWARD]
        yw  = target.iloc[: _MIN_TRAIN - _FORWARD]
        ok  = Xw.notna().all(axis=1) & yw.notna()
        if ok.sum() < 32:
            return sig
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl.fit(Xw.loc[ok].values, yw.loc[ok].values)
        except Exception:
            return sig

        for i in range(_MIN_TRAIN, n):
            # Online update with resolved label (_FORWARD bars ago)
            j = i - _FORWARD
            xj = feat.iloc[[j]]
            yj = target.iloc[j]
            if not xj.isna().any(axis=1).iloc[0] and not pd.isna(yj):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mdl.partial_fit(xj.values, [int(yj)], classes=[0, 1])
                except Exception:
                    pass

            row = feat.iloc[[i]]
            if row.isna().any(axis=1).iloc[0]:
                continue
            try:
                sig.iloc[i] = float(mdl.predict_proba(row.values)[0, 1])
            except Exception:
                pass

        return sig

    # ── Model 4: Calibrated multi-model ensemble ─────────────────────────────

    def _ensemble_signal(self, feat: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        Walk-forward with three diverse base models trained simultaneously:
          GBM  (strong non-linear patterns)
          LR   (linear baseline; stable in low-data regimes)
          RF   (bagging; high variance reduction)

        Final probability = mean of the three raw probabilities.
        Averaging diverse model families improves calibration and reduces
        the variance of any single model's predictions.
        """
        def fit_fn(Xtr: pd.DataFrame, ytr: pd.Series) -> list:
            fitted = []
            for factory in (_make_gbm, _make_lr, _make_rf):
                fitted.append(_fit(Xtr, ytr, factory()))
            return fitted

        def predict_fn(models: list, row: pd.DataFrame) -> float:
            probs = [float(m.predict_proba(row.values)[0, 1]) for m in models]
            return float(np.mean(probs))

        return _walk_forward(feat[_BASE], target, fit_fn=fit_fn, predict_fn=predict_fn)


# ── Private helpers ───────────────────────────────────────────────────────────

def _fit(X: pd.DataFrame, y: pd.Series, model: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X.values, y.values)
    return model


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    tickers = ["AAPL", "MSFT", "NVDA"]
    print(f"Downloading {tickers} ...")
    raw = {t: yf.download(t, period="3y", auto_adjust=True, progress=False) for t in tickers}
    engine = MLSignalEngine(verbose=True)
    out    = engine.compute(raw)
    for t, df in out.items():
        valid = df["ml_signal"].dropna()
        if len(valid):
            print(f"  {t}: ml_signal [{valid.min():.3f}, {valid.max():.3f}] "
                  f"| {len(valid)}/{len(df)} bars with signal")
