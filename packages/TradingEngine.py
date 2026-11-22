"""
A minimal live-trading engine skeleton for demonstrating the modularization of trading strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from packages.Strategies import Strategy


class MarketDataProvider(Protocol):
    """Abstract provider of price history for a symbol."""

    def get_prices(self, symbol: str) -> pd.DataFrame: ...


class OrderExecutor(Protocol):
    """Abstract broker/executor that can accept target positions."""

    def submit_target_position(self, symbol: str, target: int) -> None: ...


@dataclass
class TradingEngine:
    """
    Minimal live-engine skeleton: fetch latest prices, compute target, submit order.
    Concrete implementations provide MarketDataProvider and OrderExecutor.
    """
    market_data_provider: MarketDataProvider
    order_executor: OrderExecutor
    symbol: str

    def step(self, strategy: Strategy) -> int:
        prices = self.market_data_provider.get_prices(self.symbol)
        if prices.empty:
            raise ValueError("market data is empty; no order submitted")
        if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be unique and sorted")
        positions = strategy.generate_positions(prices)
        if not positions.index.equals(prices.index):
            raise ValueError("strategy positions must align with market data")
        target = positions.iloc[-1]
        if pd.isna(target) or target not in (-1, 0, 1):
            raise ValueError("target position must be -1, 0, or 1")
        target = int(target)
        self.order_executor.submit_target_position(self.symbol, target)
        return target
