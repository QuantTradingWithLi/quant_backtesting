from unittest.mock import Mock

import pandas as pd
import pytest

from packages.Strategies import MAStrategy
from packages.TradingEngine import TradingEngine


def test_trading_engine_submits_valid_latest_target():
    provider = Mock()
    provider.get_prices.return_value = pd.DataFrame({"close": [100, 101]})
    executor = Mock()
    target = TradingEngine(provider, executor, "TEST").step(MAStrategy(fast=1, slow=2))
    assert target == 1
    executor.submit_target_position.assert_called_once_with("TEST", 1)


@pytest.mark.parametrize("bad_target", [0.5, 2, float("nan")])
def test_invalid_targets_do_not_reach_executor(bad_target):
    prices = pd.DataFrame({"close": [100]})
    provider = Mock()
    provider.get_prices.return_value = prices
    executor = Mock()
    strategy = Mock()
    strategy.generate_positions.return_value = pd.Series([bad_target], index=prices.index)
    with pytest.raises(ValueError):
        TradingEngine(provider, executor, "TEST").step(strategy)
    executor.submit_target_position.assert_not_called()


def test_empty_market_data_does_not_submit_order():
    provider = Mock()
    provider.get_prices.return_value = pd.DataFrame()
    executor = Mock()
    with pytest.raises(ValueError, match="empty"):
        TradingEngine(provider, executor, "TEST").step(MAStrategy())
    executor.submit_target_position.assert_not_called()
