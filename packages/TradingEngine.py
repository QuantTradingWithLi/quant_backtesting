from __future__ import annotations

"""
A minimal live-trading engine skeleton for demonstrating the modularization of trading strategies.
"""
import pandas as pd
from typing import Protocol
from dataclasses import dataclass
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
        target = int(strategy.generate_positions(prices).iloc[-1])
        self.order_executor.submit_target_position(self.symbol, target)
        return target
