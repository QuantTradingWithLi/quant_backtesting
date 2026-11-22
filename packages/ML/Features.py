"""
Feature engineering for ML strategies.

All transforms are look-ahead safe: features at time t are computed only from
data at or before t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.Signals import rsi


@dataclass
class FeatureEngineer:
    """
    Create a tabular feature matrix X and (optionally) a target y.
    The default target is next-day close-to-close return.
    """
    include_volume: bool = True
    ret_lags: tuple[int, ...] = (1, 5, 10, 20)
    ma_windows: tuple[int, ...] = (5, 10, 20, 60)
    rsi_window: int = 14
    vol_window: int = 20

    def _safe_div(self, a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)

    def transform(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Parameters
        ----------
        prices : DataFrame with at least a 'close' column

        Returns
        -------
        (X, y) : (DataFrame, Series)
            X: features aligned to prices.index
            y: next-day close-to-close return (NaN for the latest unknown label)
        """
        close = prices['close'].astype(float)
        # Basic returns
        ret1 = close.pct_change(fill_method=None)
        feats = {}

        # Lagged returns
        for L in self.ret_lags:
            feats[f"ret_{L}"] = close.pct_change(L, fill_method=None)

        # Moving averages and spreads
        for w in self.ma_windows:
            ma = close.rolling(w).mean()
            feats[f"ma_{w}"] = self._safe_div(close, ma) - 1.0  # relative distance to MA
        if 5 in self.ma_windows and 20 in self.ma_windows:
            feats["ma5_gt_ma20"] = (close.rolling(5).mean() > close.rolling(20).mean()).astype(int)

        # RSI
        feats[f"rsi_{self.rsi_window}"] = rsi(close, self.rsi_window)

        # Volatility
        feats[f"vol_{self.vol_window}"] = ret1.rolling(self.vol_window).std()

        # Volume (optional)
        if self.include_volume and 'volume' in prices.columns:
            v = prices['volume'].astype(float)
            v_ma = v.rolling(20).mean()
            scale = v_ma.rolling(20).std()
            volume_score = self._safe_div(v - v_ma, scale)
            feats["vol_z"] = volume_score.mask(scale.eq(0) & v.eq(v_ma), 0.0)

        X = pd.DataFrame(feats, index=prices.index)

        # Normalize RSI to [-1,1]
        rsi_col = f"rsi_{self.rsi_window}"
        if rsi_col in X.columns:
            X[rsi_col] = (X[rsi_col] - 50.0) / 50.0

        # Target: next-day close-to-close return
        y = ret1.shift(-1).replace([np.inf, -np.inf], np.nan)

        # Inference must not depend on whether tomorrow's label exists.
        X = X.replace([np.inf, -np.inf], np.nan)
        valid = X.notna().all(axis=1)
        X = X.loc[valid]
        y = y.loc[valid]

        return X, y
