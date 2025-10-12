import unittest
import pandas as pd

from packages.BackTestingEngine import Simulator
from packages.Constants import ExecutionType


class TestSimulator(unittest.TestCase):
    def test_costs_apply_on_position_change(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.DataFrame({"open": [100, 100, 100, 100, 100], "close": [100, 101, 102, 101, 100]}, index=idx)
        # positions flip from 0 -> 1 -> -1 -> 0 to force trades
        pos = pd.Series([0, 1, -1, 0, 0], index=idx)
        sim = Simulator(commission_bps=10, slippage_bps=10, trade_on=ExecutionType.NEXT_OPEN)
        res = sim.run(prices, pos)
        # There are 3 trades magnitudes (0->1,1->-1,-1->0): total trades sum should be 3 (abs position changes)
        self.assertAlmostEqual(res["trades"].sum(), 4.0)
        # Net should be less than gross due to costs
        self.assertTrue((res["net"] <= res["gross"] + 1e-12).all())


if __name__ == "__main__":
    unittest.main()
