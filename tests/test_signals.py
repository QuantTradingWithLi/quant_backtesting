import unittest
import pandas as pd
import numpy as np
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


if __name__ == "__main__":
    unittest.main()
