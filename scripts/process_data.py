
"""
Command-line data pipeline runner.
Cleans and compresses raw data only.

Examples
--------
python process_data.py --csv_dir "../spydata" --out_dir "../processed_data" --override_cache "true"
"""
import argparse, os
from datetime import datetime, timedelta, timezone
from packages.DataPipeline import DataStore


def now_str(
        delta: timedelta = timedelta(0),
) -> str:
    return (datetime.now().astimezone(timezone.utc) + delta).strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(description="Run a data processing pipeline for CSV OHLCV data files.")
    ap.add_argument("--csv_dir", required=True, help="Path to CSV OHLCV data files")
    ap.add_argument("--out_dir", default="../processed_data", help="Output directory")
    ap.add_argument("--override_cache", default=False, type=bool, choices=[True, False], help="Override existing files")
    args = ap.parse_args()

    ds = DataStore(override_cache=args.override_cache)

    files = os.listdir(args.csv_dir)
    csv_paths = [os.path.join(args.csv_dir, name) for name in files if name.endswith(".csv")]

    os.makedirs(args.out_dir, exist_ok=True)
    parquet_paths = [os.path.join(args.out_dir, os.path.splitext(os.path.split(csv_path)[1])[0] + ".parquet") for csv_path in csv_paths]

    print(f"{now_str()} Data Processing Started...")
    result = ds.bulk_csv_to_parquet(csv_paths=csv_paths, parquet_paths=parquet_paths, max_workers=20)
    print(f"{now_str()} Data Processing Completed. Files Processed: {len(result)}")

if __name__ == "__main__":
    main()
