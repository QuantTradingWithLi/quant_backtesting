from __future__ import annotations

"""
Backtesting engine for single-stock, mid-frequency quantitative trading strategies.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
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
        prices = self.market_data[["open", "close"]]
        positions = strategy.generate_positions(prices)

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
    - Positions represent *target exposure* during the day.
    - If `trade_on="NEXT_OPEN"`, the position for day t is yesterday's signal.
    - Transaction costs are charged on absolute changes in position magnitude.
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
        df = prices[["open", "close"]].copy()
        sig = positions.reindex(df.index).fillna(0).astype(int)

        # Open-next execution: today's held position is yesterday's signal.
        if self.trade_on == ExecutionType.NEXT_OPEN:
            pos = sig.shift(1).fillna(0)
        elif self.trade_on == ExecutionType.MOC:
            pos = sig
        else:
            raise ValueError("Unknown execution type")

        # Close-to-close returns and gross strategy return
        ret = df["close"].pct_change().fillna(0.0)
        gross = pos * ret

        # Trading cost on position changes (0->1, 1->-1, etc.)
        trades = pos.diff().abs().fillna(0.0)
        costs = (_to_bps(self.commission_bps + self.slippage_bps)) * trades
        net = gross - costs

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
        Annualized risk-free rate for Sharpe (simple subtraction here).
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
        ann_ret = (1 + r).prod() ** (self.freq_per_year / len(r)) - 1
        vol = r.std() * np.sqrt(self.freq_per_year)
        sharpe = (ann_ret - self.rf_rate) / (vol if vol != 0 else np.nan)
        equity = (1 + r).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
        return {
            "days": int(len(r)),
            "ann_return": float(ann_ret),
            "ann_vol": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "win_rate": float((r > 0).mean()),
            "total_return": float((1 + r).prod() - 1.0),
        }
