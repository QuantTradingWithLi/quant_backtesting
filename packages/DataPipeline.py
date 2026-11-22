"""Causal, single-asset OHLCV cleaning and CSV/Parquet storage."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]
OHLCV = [*OHLC, "volume"]


def _repair_ohlc(df: pd.DataFrame) -> None:
    """Expand bounds to contain every observed price, without row-wise apply."""
    bounds = df[OHLC]
    df["high"] = bounds.max(axis=1)
    df["low"] = bounds.min(axis=1)


@dataclass
class DataCleaner:
    """Repair existing bars using only information available at each timestamp.

    Missing prices are forward-filled; unrepairable leading bars are dropped.
    Jumps exceeding ``price_ret_clip`` relative to the last accepted close are
    treated as data errors. This heuristic can remove real market moves; use
    ``float("inf")`` to disable it for already validated/adjusted data.
    Volume de-spiking uses trailing medians and median absolute residuals.
    No weekend/holiday bars are created.
    """

    fix_volume_outliers: bool = True
    vol_clip_z: float = 5.0
    price_ret_clip: float = 0.2

    def clean(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Normalize, validate and causally repair raw OHLC(V) observations."""
        if np.isnan(self.price_ret_clip) or self.price_ret_clip <= 0:
            raise ValueError("price_ret_clip must be positive")
        if not np.isfinite(self.vol_clip_z) or self.vol_clip_z <= 0:
            raise ValueError("vol_clip_z must be finite and positive")

        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        date_col = date_col.strip().lower()
        if not df.columns.is_unique:
            raise ValueError("duplicate column names after normalization")
        missing = [c for c in [date_col, *OHLC] if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        columns = [c for c in OHLCV if c in df.columns]
        df = (
            df.dropna(subset=[date_col])
            .drop_duplicates(subset=[date_col], keep="first")
            .sort_values(date_col)
            .set_index(date_col)[columns]
        )
        df = df.apply(pd.to_numeric, errors="coerce").astype(float)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.mask(df < 0)
        df[OHLC] = df[OHLC].mask(df[OHLC] == 0).ffill()

        # Compare against the last accepted close, not a rejected spike. This
        # preserves a valid recovery bar immediately following corrupt data.
        bad = np.zeros(len(df), dtype=bool)
        previous = None
        for i, close in enumerate(df["close"].to_numpy()):
            if np.isnan(close):
                continue
            if previous is not None and abs(close / previous - 1) > self.price_ret_clip:
                bad[i] = True
            else:
                previous = close
        df.loc[bad, OHLC] = np.nan
        df[OHLC] = df[OHLC].ffill()
        _repair_ohlc(df)
        df = df.dropna(subset=OHLC)

        if "volume" in df.columns:
            # Missing initial volume represents no known activity, not a future fill.
            volume = df["volume"].ffill().fillna(0)
            if self.fix_volume_outliers:
                median = volume.rolling(50, min_periods=20).median()
                mad = (volume - median).abs().rolling(50, min_periods=20).median()
                z = (volume - median) / (mad.replace(0, np.nan) * 1.4826)
                spike = (z.abs() > self.vol_clip_z) | (
                    mad.eq(0) & volume.ne(median)
                )
                volume = volume.mask(spike, median)
            df["volume"] = volume.round()
        return df


@dataclass
class DataStore:
    """Read Parquet directly, or clean CSVs with configuration-aware caches."""

    data_cleaner: DataCleaner = field(default_factory=DataCleaner)
    override_cache: bool = True

    def _clean_csv(self, csv_path: Path, parquet_path: Path) -> pd.DataFrame:
        # Persist the policy with the cached frame. Legacy caches and caches
        # created with different settings must not bypass causal cleaning.
        source = csv_path.stat()
        policy = {
            "version": 1,
            "cleaner": asdict(self.data_cleaner),
            "source": str(csv_path.resolve()),
            "mtime_ns": source.st_mtime_ns,
            "size": source.st_size,
        }
        if (
            not self.override_cache
            and parquet_path.is_file()
            and parquet_path.stat().st_mtime_ns >= source.st_mtime_ns
        ):
            cached = pd.read_parquet(parquet_path)
            if cached.attrs.get("cleaning_policy") == policy:
                return cached

        cleaned = self.data_cleaner.clean(pd.read_csv(csv_path))
        cleaned.attrs["cleaning_policy"] = policy
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(parquet_path, index=True)
        return cleaned

    def read(self, path: str | Path) -> pd.DataFrame:
        """Read a Parquet file, or a CSV using a sibling Parquet cache."""
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() != ".csv":
            raise ValueError("data path must end in .csv or .parquet")
        return self._clean_csv(path, path.with_suffix(".parquet"))

    def csv_to_parquet(self, csv_path: str, parquet_path: str) -> str:
        """Convert a CSV to cleaned Parquet, creating parent directories."""
        if Path(csv_path).suffix.lower() != ".csv" or Path(parquet_path).suffix.lower() != ".parquet":
            raise ValueError("conversion requires a .csv source and a .parquet target")
        if Path(csv_path).resolve() == Path(parquet_path).resolve():
            raise ValueError("CSV source and Parquet target must be different")
        self._clean_csv(Path(csv_path), Path(parquet_path))
        return parquet_path

    def bulk_csv_to_parquet(
        self,
        csv_paths: list[str],
        parquet_paths: list[str] | None = None,
        max_workers: int = 8,
    ) -> list[str]:
        """Convert files in input order; reject mismatched or colliding targets."""
        if parquet_paths is None:
            parquet_paths = [str(Path(path).with_suffix(".parquet")) for path in csv_paths]
        if len(csv_paths) != len(parquet_paths):
            raise ValueError("CSV and Parquet path lists must have the same length")
        if len({Path(path).resolve() for path in parquet_paths}) != len(parquet_paths):
            raise ValueError("Parquet targets must be unique")
        if any(Path(path).suffix.lower() != ".csv" for path in csv_paths) or any(
            Path(path).suffix.lower() != ".parquet" for path in parquet_paths
        ):
            raise ValueError("conversion requires .csv sources and .parquet targets")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not csv_paths:
            return []
        if max_workers == 1:
            return [
                self.csv_to_parquet(source, target)
                for source, target in zip(csv_paths, parquet_paths)
            ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.csv_to_parquet, csv_paths, parquet_paths))
