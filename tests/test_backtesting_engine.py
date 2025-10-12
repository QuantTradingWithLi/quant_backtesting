import unittest
import pandas as pd
import numpy as np
from packages.BackTestingEngine import BackTestingEngine
from packages.Strategies import MAStrategy


class TestBacktingEngine(unittest.TestCase):
    def test_end_to_end(self):
        # Use a tiny synthetic CSV to avoid relying on external files
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=30, freq="D"),
            "open": np.linspace(100, 103, 30),
            "high": np.linspace(100.5, 103.5, 30),
            "low": np.linspace(99.5, 102.5, 30),
            "close": np.linspace(100, 103, 30),
            "volume": 1_000
        })
        tmp = "_tmp.csv"
        df.to_csv(tmp, index=False)
        bt = BackTestingEngine(market_data=df)
        out = bt.run(strategy=MAStrategy(fast=3, slow=5))
        self.assertIn("summary", out)
        self.assertIn("results", out)
        self.assertGreater(len(out["results"]), 0)
        # sanity: equity series ends with a finite number
        self.assertTrue(np.isfinite(out["results"]["equity"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
