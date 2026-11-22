import unittest

import numpy as np
import pandas as pd

from packages.Signals import MASignal, RSISignal, rsi


class TestSignals(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        close = pd.Series(100 + np.sin(np.linspace(0, 10, len(idx))), index=idx)
        self.close = close

    def test_ma_signal(self):
        sig = MASignal(fast=5, slow=10)(self.close)
        self.assertEqual(len(sig), len(self.close))
        self.assertTrue(set(sig.dropna().unique()).issubset({-1, 0, 1}))

    def test_rsi_signal(self):
        r = rsi(self.close, 14)
        self.assertEqual(len(r), len(self.close))
        sig = RSISignal(window=14, low_th=30, high_th=70)(self.close)
        self.assertTrue(set(sig.dropna().unique()).issubset({-1, 0, 1}))

    def test_rsi_handles_one_sided_and_flat_prices(self):
        rising = pd.Series(np.arange(1.0, 21.0))
        falling = rising.iloc[::-1].reset_index(drop=True)
        flat = pd.Series([10.0] * 20)
        self.assertEqual(rsi(rising).iloc[-1], 100.0)
        self.assertEqual(rsi(falling).iloc[-1], 0.0)
        self.assertEqual(rsi(flat).iloc[-1], 50.0)
        self.assertEqual(RSISignal()(rising).iloc[-1], -1)


if __name__ == "__main__":
    unittest.main()
