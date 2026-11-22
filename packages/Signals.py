"""
Indicator and signal helpers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute a standard exponentially-weighted RSI.

    Parameters
    ----------
    series : pd.Series
        Price series (typically `close`).
    window : int
        RSI window.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError("RSI window must be a positive integer")
    delta = series.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask(roll_down.eq(0) & roll_up.gt(0), 100.0)
    out = out.mask(roll_down.eq(0) & roll_up.eq(0), 50.0)
    return out


@dataclass
class MASignal:
    """Binary trend signal: +1 if MA(fast) > MA(slow), -1 if MA(fast) < MA(slow), else 0."""
    fast: int = 20
    slow: int = 60

    def __post_init__(self):
        if not 0 < self.fast < self.slow:
            raise ValueError("MA windows must satisfy 0 < fast < slow")

    def __call__(self, close: pd.Series) -> pd.Series:
        ma_fast = close.rolling(self.fast).mean()
        ma_slow = close.rolling(self.slow).mean()
        sig = (ma_fast > ma_slow).astype(int) - (ma_fast < ma_slow).astype(int)
        return sig.rename(f"ma{self.fast}_{self.slow}")


@dataclass
class RSISignal:
    """Mean-reversion signal: +1 if RSI < low_th; -1 if RSI > high_th; else 0."""
    window: int = 14
    low_th: float = 30.0
    high_th: float = 70.0

    def __post_init__(self):
        if not 0 <= self.low_th < self.high_th <= 100:
            raise ValueError("RSI thresholds must satisfy 0 <= low < high <= 100")

    def __call__(self, close: pd.Series) -> pd.Series:
        r = rsi(close, self.window)
        sig = (r < self.low_th).astype(int) - (r > self.high_th).astype(int)
        return sig.rename(f"rsi{self.window}")
