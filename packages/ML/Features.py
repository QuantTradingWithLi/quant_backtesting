from __future__ import annotations

"""
Feature engineering for ML strategies.

All transforms are look-ahead safe: features at time t are computed only from
data at or before t.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
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
        prices : DataFrame with at least columns ['open','close']

        Returns
        -------
        (X, y) : (DataFrame, Series)
            X: features aligned to prices.index
            y: next-day close-to-close return (aligned to same index)
        """
        close = prices['close'].astype(float)
        openp = prices['open'].astype(float)

        # Basic returns
        ret1 = close.pct_change()
        feats = {}

        # Lagged returns
        for L in self.ret_lags:
            feats[f"ret_{L}"] = close.pct_change(L)

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
            feats["vol_z"] = (v - v_ma) / (v_ma.rolling(20).std())

        X = pd.DataFrame(feats, index=prices.index)

        # Normalize RSI to [-1,1]
        rsi_col = f"rsi_{self.rsi_window}"
        if rsi_col in X.columns:
            X[rsi_col] = (X[rsi_col] - 50.0) / 50.0

        # Target: next-day close-to-close return
        y = close.pct_change().shift(-1)

        # Drop rows with any NaNs (strict to ensure deterministic training)
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        return X, y
