import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from scripts import process_data, run_backtest


@pytest.fixture
def market_csv(tmp_path):
    path = tmp_path / "market.csv"
    close = 100 + np.arange(160) * 0.1 + np.sin(np.arange(160))
    pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=len(close)),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000,
    }).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("value,expected", [("False", False), ("true", True), ("0", False)])
def test_cache_boolean_is_parsed_correctly(value, expected):
    assert process_data._parse_bool(value) is expected


def test_processing_cli_respects_false_cache_flag(market_csv, tmp_path):
    destination = tmp_path / "processed"
    args = [
        "--csv_dir", str(market_csv.parent), "--out_dir", str(destination),
        "--max_workers", "1",
    ]
    process_data.main([*args, "--override_cache"])
    parquet = destination / "market.parquet"
    original_mtime = parquet.stat().st_mtime_ns
    process_data.main([*args, "--override_cache", "False"])
    assert parquet.stat().st_mtime_ns == original_mtime


@pytest.mark.parametrize("strategy_type,model", [
    ("STATISTICS", "MULTI_FACTOR"),
    ("MACHINE_LEARNING", "RIDGE_LINEAR"),
])
def test_backtest_cli_writes_results_and_strict_json(market_csv, tmp_path, strategy_type, model):
    output = tmp_path / strategy_type
    run_backtest.main([
        "--market_data", str(market_csv), "--outdir", str(output),
        "--strategy_type", strategy_type, "--strategy_model", model,
    ])
    summary_text = (output / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_text and "Infinity" not in summary_text
    assert json.loads(summary_text)["days"] == 160
    results = pd.read_parquet(output / "results.parquet")
    assert np.isfinite(results["equity"]).all()
    assert (output / "equity.png").is_file()
    assert (output / "drawdown.png").is_file()


def test_backtest_cli_accepts_parquet_and_null_statistics(market_csv, tmp_path):
    parquet = market_csv.with_suffix(".parquet")
    prices = pd.read_csv(market_csv).iloc[:1].set_index("date")
    prices.to_parquet(parquet)
    output = tmp_path / "one_day"
    run_backtest.main(["--market_data", str(parquet), "--outdir", str(output)])
    assert json.loads((output / "summary.json").read_text())["sharpe"] is None


def test_backtest_cli_rejects_model_strategy_mismatch():
    with pytest.raises(SystemExit) as exc:
        run_backtest.main([
            "--market_data", "unused.csv", "--strategy_type", "STATISTICS",
            "--strategy_model", "RIDGE_LINEAR",
        ])
    assert exc.value.code == 2


@pytest.mark.parametrize("script", ["process_data.py", "run_backtest.py"])
def test_scripts_can_run_directly_from_another_directory(script, tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
