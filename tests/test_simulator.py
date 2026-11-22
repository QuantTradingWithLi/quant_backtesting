import unittest

import numpy as np
import pandas as pd

from packages.BackTestingEngine import Metrics, Simulator
from packages.Constants import ExecutionType


class TestSimulator(unittest.TestCase):
    def test_costs_apply_on_position_change(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.DataFrame({"open": [100, 100, 100, 100, 100], "close": [100, 101, 102, 101, 100]}, index=idx)
        # positions flip from 0 -> 1 -> -1 -> 0 to force trades
        pos = pd.Series([0, 1, -1, 0, 0], index=idx)
        sim = Simulator(commission_bps=10, slippage_bps=10, trade_on=ExecutionType.NEXT_OPEN)
        res = sim.run(prices, pos)
        # Entry, reversal and exit have turnover 1 + 2 + 1.
        self.assertAlmostEqual(res["trades"].sum(), 4.0)
        # Net should be less than gross due to costs
        self.assertTrue((res["net"] <= res["gross"] + 1e-12).all())

    def test_next_open_entry_does_not_earn_overnight_gap(self):
        prices = pd.DataFrame({"open": [100, 120], "close": [100, 132]})
        result = Simulator(0, 0, ExecutionType.NEXT_OPEN).run(
            prices, pd.Series([1, 1])
        )
        self.assertAlmostEqual(result["gross"].iloc[1], 0.1)

    def test_next_open_exit_keeps_overnight_return(self):
        prices = pd.DataFrame({
            "open": [100, 100, 120],
            "close": [100, 110, 60],
        })
        result = Simulator(0, 0, ExecutionType.NEXT_OPEN).run(
            prices, pd.Series([1, 0, 0])
        )
        self.assertAlmostEqual(result["gross"].iloc[2], 120 / 110 - 1)

    def test_next_open_compounds_overnight_and_intraday_returns(self):
        prices = pd.DataFrame({
            "open": [100, 100, 120],
            "close": [100, 110, 132],
        })
        result = Simulator(0, 0, ExecutionType.NEXT_OPEN).run(
            prices, pd.Series([1, 1, 1])
        )
        self.assertAlmostEqual(result["gross"].iloc[2], 132 / 110 - 1)

    def test_moc_cannot_earn_return_before_entry_and_charges_initial_cost(self):
        prices = pd.DataFrame({"open": [100, 100], "close": [100, 120]})
        result = Simulator(10, 0, ExecutionType.MOC).run(
            prices, pd.Series([1, 0])
        )
        self.assertEqual(result["trades"].tolist(), [1, 1])
        self.assertEqual(result["gross"].iloc[0], 0)
        self.assertAlmostEqual(result["gross"].iloc[1], 0.2)
        self.assertAlmostEqual(result["net"].iloc[0], -0.001)

    def test_rejects_invalid_inputs(self):
        prices = pd.DataFrame({"open": [100, 101], "close": [100, 101]})
        for positions in ([0, 0.5], [0, 2], [0, np.inf]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                Simulator(0, 0, ExecutionType.NEXT_OPEN).run(
                    prices, pd.Series(positions)
                )
        with self.assertRaises(ValueError):
            Simulator(-1, 0, ExecutionType.NEXT_OPEN).run(prices, pd.Series([0, 1]))
        with self.assertRaises(ValueError):
            Simulator(0, 0, ExecutionType.NEXT_OPEN).run(
                prices.iloc[::-1], pd.Series([0, 1])
            )
        prices.loc[1, "open"] = 0
        with self.assertRaises(ValueError):
            Simulator(0, 0, ExecutionType.NEXT_OPEN).run(prices, pd.Series([0, 1]))

    def test_short_cannot_recover_after_overnight_bankruptcy(self):
        prices = pd.DataFrame({
            "open": [100, 100, 300],
            "close": [100, 100, 900],
        })
        with self.assertRaisesRegex(ValueError, "overnight.*capital"):
            Simulator(0, 0, ExecutionType.NEXT_OPEN).run(
                prices, pd.Series([-1, -1, -1])
            )


class TestMetrics(unittest.TestCase):
    def test_initial_loss_counts_toward_drawdown(self):
        summary = Metrics(rf_rate=0).summarize(pd.Series([-0.1, 0.0]))
        self.assertAlmostEqual(summary["max_drawdown"], -0.1)

    def test_sharpe_uses_arithmetic_excess_returns(self):
        returns = pd.Series([0.01, -0.02, 0.03])
        summary = Metrics(rf_rate=0.02).summarize(returns)
        expected = (returns.mean() - 0.02 / 252) / returns.std() * np.sqrt(252)
        self.assertAlmostEqual(summary["sharpe"], expected)

    def test_empty_and_single_day_returns(self):
        self.assertEqual(Metrics().summarize(pd.Series(dtype=float)), {})
        summary = Metrics().summarize(pd.Series([0.01]))
        self.assertTrue(np.isnan(summary["sharpe"]))


if __name__ == "__main__":
    unittest.main()
