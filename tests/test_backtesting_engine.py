import unittest

import numpy as np
import pandas as pd

from packages.BackTestingEngine import BackTestingEngine
from packages.Strategies import MAStrategy


class TestBacktingEngine(unittest.TestCase):
    def test_end_to_end(self):
        # Use synthetic data without writing temporary files.
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=30, freq="D"),
            "open": np.linspace(100, 103, 30),
            "high": np.linspace(100.5, 103.5, 30),
            "low": np.linspace(99.5, 102.5, 30),
            "close": np.linspace(100, 103, 30),
            "volume": 1_000
        })
        bt = BackTestingEngine(market_data=df)
        out = bt.run(strategy=MAStrategy(fast=3, slow=5))
        self.assertIn("summary", out)
        self.assertIn("results", out)
        self.assertGreater(len(out["results"]), 0)
        # sanity: equity series ends with a finite number
        self.assertTrue(np.isfinite(out["results"]["equity"].iloc[-1]))

    def test_strategy_receives_volume(self):
        class VolumeStrategy(MAStrategy):
            def generate_positions(self, prices):
                self.seen_volume = "volume" in prices.columns
                return super().generate_positions(prices)

        prices = pd.DataFrame({
            "open": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1000, 2000],
        })
        strategy = VolumeStrategy(fast=1, slow=2)
        BackTestingEngine(prices).run(strategy)
        self.assertTrue(strategy.seen_volume)


if __name__ == "__main__":
    unittest.main()
