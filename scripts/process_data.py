
"""
Command-line data pipeline runner.
Cleans and compresses raw data only.

Examples
--------
uv run python -m scripts.process_data --csv_dir spydata --out_dir processed_data --override_cache
"""
import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.DataPipeline import DataStore


def now_str(
        delta: timedelta = timedelta(0),
) -> str:
    return (datetime.now(UTC) + delta).strftime("%Y%m%d_%H%M%S")


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Run a data processing pipeline for CSV OHLCV data files.")
    ap.add_argument("--csv_dir", required=True, help="Path to CSV OHLCV data files")
    ap.add_argument("--out_dir", default="processed_data", help="Output directory")
    ap.add_argument("--override_cache", default=False, nargs="?", const=True,
                    type=_parse_bool, help="Override existing files (optional true/false)")
    ap.add_argument("--max_workers", default=8, type=int, help="Parallel conversion workers")
    args = ap.parse_args(argv)
    if args.max_workers <= 0:
        ap.error("--max_workers must be positive")
    source = Path(args.csv_dir)
    if not source.is_dir():
        ap.error("--csv_dir must be an existing directory")

    ds = DataStore(override_cache=args.override_cache)

    csv_paths = sorted(
        str(path) for path in source.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    parquet_paths = [str(destination / f"{Path(path).stem}.parquet") for path in csv_paths]

    print(f"{now_str()} Data Processing Started...")
    result = ds.bulk_csv_to_parquet(
        csv_paths=csv_paths, parquet_paths=parquet_paths, max_workers=args.max_workers
    )
    print(f"{now_str()} Data Processing Completed. Files Processed: {len(result)}")

if __name__ == "__main__":
    main()
