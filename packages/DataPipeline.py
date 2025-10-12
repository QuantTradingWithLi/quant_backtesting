from __future__ import annotations

"""
Data ingestion and robust cleaning utilities.

Design principles
-----------------
- Deterministic, "pure" transforms: no I/O in the cleaner, just DataFrame -> DataFrame.
- Tolerant to vendor glitches: zeros, negative values, inverted high/low, missing bars.
- Avoid look-ahead bias: time-based interpolation and ffill/bfill are allowed, but we do not
  invent weekend/holiday bars (we only repair bars that exist).
- Explicit safeguards: drop invalid dates, cap implausible single-day returns, and re-enforce
  OHLC structural constraints after interpolation.

Key types
---------
- DataCleaner: core repair logic for a single-asset OHLCV daily series.
- DataStore: thin convenience layer for CSV/Parquet I/O (with graceful fallbacks).
"""
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, override
from concurrent.futures import ProcessPoolExecutor

OHLCV = ["open", "high", "low", "close", "volume"]


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric (invalids -> NaN)."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _repair_ohlc_row(row: pd.Series) -> pd.Series:
    """
    Enforce basic OHLC invariants row-wise.

    - If `high < low`, swap them.
    - Ensure `low <= open, close <= high` by expanding the bounds minimally.
    """
    h, l = row.get("high"), row.get("low")
    if pd.notna(h) and pd.notna(l) and h < l:
        row["high"], row["low"] = l, h
    # Tighten/expand bounds so open/close are inside [low, high].
    for k in ("open", "close"):
        v = row.get(k)
        if pd.notna(v):
            row["high"] = max(row.get("high", v), v)
            row["low"] = min(row.get("low", v), v)
    return row


def _interpolate_time(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Interpolate over time index; ffill/bfill as last resort.
    Assumes a DatetimeIndex with monotonically increasing timestamps.
    """
    df[cols] = df[cols].interpolate(method="time", limit_direction="both")
    df[cols] = df[cols].ffill().bfill()
    return df


@dataclass
class DataCleaner:
    """
    Cleans a single-asset OHLCV DataFrame.

    Parameters
    ----------
    fix_volume_outliers : bool
        Whether to apply robust de-spiking on volume.
    vol_clip_z : float
        Z-score threshold (via rolling-median MAD) for flagging volume outliers.
    price_ret_clip : float
        Absolute daily return threshold above which a bar is treated as a data error
        and repaired through interpolation.
    """
    fix_volume_outliers: bool = True
    vol_clip_z: float = 5.0
    price_ret_clip: float = 0.2

    def clean(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Clean a raw OHLCV DataFrame.

        Steps
        -----
        1) Parse dates, drop invalid/duplicates, sort.
        2) Coerce OHLCV numerics; set negatives -> NaN and treat zeros in prices as missing.
        3) Row-wise structural repair (swap high/low if reversed; enforce bounds).
        4) Time-aware interpolation for all numeric series; ffill/bfill last resort.
        5) Daily-return outlier handling: |ret| > `price_ret_clip` -> set OHLC to NaN and re-interpolate.
        6) Volume robust de-spiking using rolling-median and MAD-derived z-score.
        7) Final rounding/typing for volume.
        """
        df = df.copy()
        if date_col not in df.columns:
            raise ValueError("missing date column")
        # Parse/validate time; keep good rows only.
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        # Normalize column names to lowercase to prevent subtle mismatches.
        df.columns = [c.lower() for c in df.columns]

        # Keep only columns we recognize; extra columns are ignored.
        need = [date_col] + OHLCV
        df = df[[c for c in need if c in df.columns]]
        df = _coerce_numeric(df, [c for c in OHLCV if c in df.columns])
        df = df.drop_duplicates(subset=[date_col]).sort_values(date_col).set_index(date_col)

        # Basic sanity rules: negative values are invalid; zero prices treated as missing.
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df.loc[df[c] < 0, c] = np.nan
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df.loc[df[c] == 0, c] = np.nan

        # Row-wise structural repair prior to interpolation.
        df = df.apply(_repair_ohlc_row, axis=1)

        # Time-aware interpolation across all numeric columns.
        numeric_cols = [c for c in OHLCV if c in df.columns]
        df = _interpolate_time(df, numeric_cols)

        # Remove implausible price jumps by marking as NaN and interpolating again.
        if "close" in df.columns:
            ret = df["close"].pct_change().abs()
            bad = ret > self.price_ret_clip
            if bad.any():
                for c in ("open", "high", "low", "close"):
                    if c in df.columns:
                        df.loc[bad, c] = np.nan
                df = _interpolate_time(df, numeric_cols)
                df = df.apply(_repair_ohlc_row, axis=1)

        # Robust volume de-spiking (optional).
        if "volume" in df.columns and self.fix_volume_outliers and len(df) > 50:
            vol = df["volume"].astype(float)
            ma = vol.rolling(50, min_periods=20).median()
            mad = (vol - ma).abs().rolling(50, min_periods=20).median()
            # Convert MAD to a robust z-score (MAD * 1.4826 ~ std for Gaussian)
            z = (vol - ma) / (mad.replace(0, np.nan) * 1.4826)
            df.loc[z.abs() > self.vol_clip_z, "volume"] = ma
            df["volume"] = df["volume"].interpolate(method="time").ffill().bfill()
            df["volume"] = df["volume"].round()

        return df


@dataclass
class DataStore:
    """
    Light convenience I/O wrapper that uses DataCleaner internally.
    """
    data_cleaner: DataCleaner = field(default_factory=DataCleaner)
    override_cache: bool = True

    def read(self, path: str) -> pd.DataFrame:
        """
        Read a cleaned DataFrame from Parquet.
        If a file is missing or override_cache == True, write to a new parquet file.
        """
        parquet_path = os.path.splitext(path)[0] + ".parquet"
        if self.override_cache or not os.path.isfile(parquet_path):
            clean_data = self.data_cleaner.clean(pd.read_csv(path), "date")
            clean_data.to_parquet(parquet_path)
            return clean_data
        else:
            return pd.read_parquet(parquet_path)


    def csv_to_parquet(self, csv_path: str, parquet_path: str) -> str:
        """Convert a CSV at `csv_path` to a cleaned Parquet."""
        if self.override_cache or not os.path.isfile(parquet_path):
            raw = pd.read_csv(csv_path)
            df = self.data_cleaner.clean(raw, "date")
            df.to_parquet(parquet_path, index=True)

        return parquet_path

    def bulk_csv_to_parquet(
            self,
            csv_paths: List[str],
            parquet_paths: Optional[List[str]] = None,
            max_workers: int = 8
    ) -> List[str]:
        """Convert many CSVs in parallel. Returns list of created target paths."""
        if not parquet_paths:
            parquet_paths = [os.path.splitext(csv_path)[0] + ".parquet" for csv_path in csv_paths]

        out_paths: List[str] = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.csv_to_parquet, csv_paths[i], parquet_paths[i]) for i in range(len(csv_paths))]
            for f in futures:
                out_paths.append(f.result())

        return out_paths
