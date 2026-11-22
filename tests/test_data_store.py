from pathlib import Path

import pandas as pd
import pytest

from packages.DataPipeline import DataCleaner, DataStore


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3),
        "open": [100, 0, 110],
        "high": [101, 105, 111],
        "low": [99, 100, 109],
        "close": [100, 105, 110],
        "volume": [1000, 1100, 1200],
    }).to_csv(path, index=False)
    return path


def test_csv_cache_round_trip_and_direct_parquet_read(csv_path, monkeypatch):
    store = DataStore(override_cache=False)
    expected = store.read(csv_path)

    def unexpected_csv_read(*args, **kwargs):
        raise AssertionError("a valid cache should not reread the CSV")

    monkeypatch.setattr(pd, "read_csv", unexpected_csv_read)
    pd.testing.assert_frame_equal(store.read(csv_path), expected)
    pd.testing.assert_frame_equal(
        DataStore().read(csv_path.with_suffix(".parquet")), expected
    )


def test_legacy_cache_and_changed_cleaner_settings_are_invalidated(csv_path):
    # Legacy Parquet lacks policy metadata and may have used future fills.
    pd.DataFrame({"close": [999]}).to_parquet(csv_path.with_suffix(".parquet"))
    store = DataStore(override_cache=False)
    result = store.read(csv_path)
    assert result["close"].tolist() == [100, 105, 110]

    changed = DataStore(DataCleaner(price_ret_clip=0.01), override_cache=False)
    assert changed.read(csv_path)["close"].tolist() == [100, 100, 100]


def test_changed_source_invalidates_cache(csv_path):
    store = DataStore(override_cache=False)
    store.read(csv_path)
    raw = pd.read_csv(csv_path)
    raw.loc[0, "volume"] = 987654
    raw.to_csv(csv_path, index=False)
    assert store.read(csv_path)["volume"].iloc[0] == 987654


def test_bulk_conversion_validates_targets_and_supports_empty_input(csv_path, tmp_path):
    store = DataStore()
    assert store.bulk_csv_to_parquet([]) == []
    with pytest.raises(ValueError, match="same length"):
        store.bulk_csv_to_parquet([str(csv_path)], [])
    with pytest.raises(ValueError, match="unique"):
        store.bulk_csv_to_parquet(
            [str(csv_path)] * 2, [str(tmp_path / "same.parquet")] * 2
        )
    with pytest.raises(ValueError):
        store.csv_to_parquet(str(csv_path), str(csv_path))


@pytest.mark.parametrize("workers", [1, 2])
def test_bulk_conversion_creates_nested_targets(csv_path, tmp_path, workers):
    target = str(tmp_path / "nested" / "prices.parquet")
    assert DataStore().bulk_csv_to_parquet(
        [str(csv_path)], [target], max_workers=workers
    ) == [target]
    assert Path(target).is_file()
    assert len(pd.read_parquet(target)) == 3
