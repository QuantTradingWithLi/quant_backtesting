import unittest
import pandas as pd
from packages.DataPipeline import DataCleaner


class TestDataCleaner(unittest.TestCase):
    def test_swap_high_low_and_zero_prices(self):
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [10, 0, 11],
            "high": [9, 12, 12],  # first row high<low -> swap
            "low": [11, 10, 10],
            "close": [10, 12, 11],
            "volume": [100, 200, 300]
        })
        clean = DataCleaner(price_ret_clip=0.5).clean(df)  # wide clip for this test
        # After swap, high should bound >= open/close; low <= open/close
        self.assertGreaterEqual(clean.loc["2020-01-01", "high"], clean.loc["2020-01-01", "open"])
        self.assertLessEqual(clean.loc["2020-01-01", "low"], clean.loc["2020-01-01", "open"])
        # Zero open on 2020-01-02 should be treated as NaN then interpolated between 10 and 11 -> ~10.5
        self.assertTrue(abs(clean.loc["2020-01-02", "open"] - 10.5) < 1e-6)

    def test_return_outlier_clip(self):
        # Create a massive jump day that should be clipped/interpolated
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "open": [100, 100, 300, 100],
            "high": [101, 101, 301, 101],
            "low": [99, 99, 299, 99],
            "close": [100, 100, 300, 100],
            "volume": [1000, 1000, 1000, 1000]
        })
        clean = DataCleaner(price_ret_clip=0.2).clean(df)  # 200% jump should be removed
        ret = clean["close"].pct_change().abs()
        # All returns should be <= clip threshold after cleaning
        self.assertTrue((ret.fillna(0) <= 0.2 + 1e-9).all())


if __name__ == "__main__":
    unittest.main()
