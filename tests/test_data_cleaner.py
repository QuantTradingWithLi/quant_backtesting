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
        # Missing prices must use history, never tomorrow's price.
        self.assertEqual(clean.loc["2020-01-02", "open"], 10)

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

    def test_cleaning_is_prefix_invariant(self):
        raw = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "open": [100, 0, 102, 300, 103],
            "high": [101, 101, 103, 301, 104],
            "low": [99, 99, 101, 299, 102],
            "close": [100, 0, 102, 300, 103],
            "volume": [1000, None, 1200, 1300, 1400],
        })
        cleaner = DataCleaner()
        full = cleaner.clean(raw)
        for stop in range(1, len(raw) + 1):
            pd.testing.assert_frame_equal(
                cleaner.clean(raw.iloc[:stop]), full.iloc[:stop]
            )
        self.assertEqual(full["close"].iloc[4], 103)

    def test_normalizes_columns_before_date_lookup(self):
        raw = pd.DataFrame({
            " DATE ": ["2020-01-01"],
            "OPEN": [10],
            "HIGH": [11],
            "LOW": [9],
            "CLOSE": [10],
        })
        result = DataCleaner().clean(raw)
        self.assertEqual(result.index.name, "date")
        self.assertEqual(result["close"].iloc[0], 10)

    def test_drops_unrepairable_leading_prices(self):
        raw = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02"],
            "open": [0, 10],
            "high": [0, 11],
            "low": [0, 9],
            "close": [0, 10],
        })
        result = DataCleaner().clean(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index[0], pd.Timestamp("2020-01-02"))

    def test_missing_ohlc_columns_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing.*close"):
            DataCleaner().clean(pd.DataFrame({
                "date": ["2020-01-01"], "open": [10], "high": [11], "low": [9]
            }))

    def test_volume_spike_repair_is_causal_even_in_short_prefixes(self):
        raw = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=80),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        })
        raw.loc[45, "volume"] = 1_000_000
        cleaner = DataCleaner()
        full = cleaner.clean(raw)
        prefix = cleaner.clean(raw.iloc[:46])
        pd.testing.assert_frame_equal(prefix, full.iloc[:46])
        self.assertEqual(prefix["volume"].iloc[-1], 1000)

    def test_invalid_or_empty_data_returns_no_bars(self):
        raw = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        self.assertTrue(DataCleaner().clean(raw).empty)
        raw.loc[0] = ["invalid", 10, 11, 9, 10]
        self.assertTrue(DataCleaner().clean(raw).empty)


if __name__ == "__main__":
    unittest.main()
