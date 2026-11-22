"""
Backtesting engine for single-stock, mid-frequency quantitative trading strategies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.Constants import ExecutionType
from packages.Strategies import Strategy


@dataclass
class BackTestingEngine:
    market_data: pd.DataFrame
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    trade_on: ExecutionType = ExecutionType.NEXT_OPEN

    def run(self, strategy: Strategy) -> dict:
        """
        Execute the backtesting pipeline on the provided data file.

        Returns
        -------
        dict with keys:
            - results: pd.DataFrame (simulated returns/positions/equity)
            - summary: dict (metrics)
        """
        # Preserve optional features (notably volume) for strategy generation.
        prices = self.market_data
        positions = strategy.generate_positions(prices.copy())

        simulator = Simulator(
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            trade_on=self.trade_on,
        )

        results = simulator.run(prices=prices, positions=positions)

        summary = Metrics().summarize(results["net"])
        return {"results": results, "summary": summary}


def _to_bps(x: float) -> float:
    """Convert basis points to proportion."""
    return x / 10000.0


@dataclass
class Simulator:
    """
    Vectorized position P&L simulator.

    Conventions
    -----------
    - Signals are computed after the close; initial exposure is zero.
    - NEXT_OPEN executes yesterday's signal at today's open. Existing exposure
      earns the overnight gap; the new exposure earns the intraday return.
    - MOC is idealized close execution: today's signal only earns future returns.
    - Exposure is reset to the target at each return leg. Costs are a simplified
      per-side fraction of equity times target turnover, not a share-level ledger.
    - Open positions are marked to market, not liquidated at the end of the run.
    """
    commission_bps: float
    slippage_bps: float
    trade_on: ExecutionType

    def run(self, prices: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
        """
        Simulate a daily strategy.

        Parameters
        ----------
        prices : DataFrame
            Must include 'open' and 'close' columns.
        positions : Series
            Target positions in {-1,0,1} aligned to index.

        Returns
        -------
        DataFrame with columns: ret, pos, trades, gross, net, equity
        """
        if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be unique and sorted")
        if not prices.columns.is_unique:
            raise ValueError("price columns must be unique")
        missing = {"open", "close"} - set(prices.columns)
        if missing:
            raise ValueError(f"missing price columns: {', '.join(sorted(missing))}")
        df = prices[["open", "close"]].astype(float)
        if not np.isfinite(df.to_numpy()).all() or (df <= 0).any().any():
            raise ValueError("prices must be finite and strictly positive")
        if not isinstance(positions, pd.Series) or not positions.index.is_unique:
            raise ValueError("positions must be a Series with a unique index")
        sig = positions.reindex(df.index).fillna(0)
        if not sig.isin([-1, 0, 1]).all():
            raise ValueError("positions must contain only -1, 0, or 1")
        sig = sig.astype(int)
        if any(not np.isfinite(x) or x < 0 for x in (
            self.commission_bps, self.slippage_bps
        )):
            raise ValueError("commission and slippage must be finite and nonnegative")

        ret = df["close"].pct_change(fill_method=None).fillna(0.0)
        if self.trade_on == ExecutionType.NEXT_OPEN:
            pos = sig.shift(1, fill_value=0)
            previous_pos = pos.shift(1, fill_value=0)
            overnight = (df["open"] / df["close"].shift(1) - 1).fillna(0.0)
            intraday = df["close"] / df["open"] - 1
            opening_equity = 1 + previous_pos * overnight
            if (opening_equity < 0).any():
                raise ValueError("overnight strategy loss exceeds available capital")
            gross = opening_equity * (1 + pos * intraday) - 1
        elif self.trade_on == ExecutionType.MOC:
            pos = sig
            previous_pos = pos.shift(1, fill_value=0)
            gross = previous_pos * ret
        else:
            raise ValueError("Unknown execution type")

        trades = (pos - previous_pos).abs()
        costs = _to_bps(self.commission_bps + self.slippage_bps) * trades
        if self.trade_on == ExecutionType.NEXT_OPEN:
            # Fees are charged against equity at the opening execution.
            net = opening_equity * (1 + pos * intraday - costs) - 1
        else:
            net = gross - costs
        if (net < -1).any():
            raise ValueError("strategy loss exceeds available capital")

        out = pd.DataFrame({
            "ret": ret, "pos": pos, "trades": trades, "gross": gross, "net": net
        }, index=df.index)
        out["equity"] = (1 + out["net"]).cumprod()
        return out


@dataclass
class Metrics:
    """
    Compute standard backtest summary statistics.

    Parameters
    ----------
    freq_per_year : int
        Trading days per year (252 for daily).
    rf_rate : float
        Annualized risk-free rate, converted to a simple per-period rate.
    """
    freq_per_year: int = 252
    rf_rate: float = 0.0266

    def summarize(self, returns: pd.Series) -> dict:
        """
        Summarize a daily return series into a metrics dictionary.
        """
        r = returns.dropna().astype(float)
        if r.empty:
            return {}
        if self.freq_per_year <= 0 or not np.isfinite(self.rf_rate):
            raise ValueError("frequency must be positive and risk-free rate finite")
        if not np.isfinite(r.to_numpy()).all() or (r < -1).any():
            raise ValueError("returns must be finite and cannot be below -100%")
        ann_ret = (1 + r).prod() ** (self.freq_per_year / len(r)) - 1
        vol = r.std() * np.sqrt(self.freq_per_year)
        sharpe = ((r.mean() * self.freq_per_year - self.rf_rate) / vol
                  if vol > 0 else np.nan)
        equity = (1 + r).cumprod()
        # Include initial capital, so a loss on the first bar is not hidden.
        max_dd = (equity / equity.cummax().clip(lower=1.0) - 1).min()
        calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
        return {
            "days": len(r),
            "ann_return": float(ann_ret),
            "ann_vol": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "win_rate": float((r > 0).mean()),
            "total_return": float((1 + r).prod() - 1.0),
        }
