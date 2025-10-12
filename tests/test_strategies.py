import unittest
import pandas as pd
import numpy as np
from packages.Strategies import MAStrategy, RSIStrategy, MultiFactorStrategy


class TestStrategies(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2020-01-01", periods=120, freq="D")
        close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)  # gentle uptrend
        openp = close * (1 - 0.001)
        self.prices = pd.DataFrame({"open": openp, "close": close})

    def test_ma_strategy_shape_and_dtype(self):
        s = MAStrategy(fast=5, slow=10).generate_positions(self.prices)
        self.assertEqual(len(s), len(self.prices))
        self.assertTrue(set(s.dropna().unique()).issubset({-1, 0, 1}))
        self.assertEqual(s.dtype.kind, 'i')

    def test_rsi_strategy_values(self):
        s = RSIStrategy().generate_positions(self.prices)
        self.assertTrue(set(s.dropna().unique()).issubset({-1, 0, 1}))

    def test_multi_factor_strategy_values(self):
        s = MultiFactorStrategy().generate_positions(self.prices)
        self.assertTrue(set(s.dropna().unique()).issubset({-1, 0, 1}))


if __name__ == "__main__":
    unittest.main()
