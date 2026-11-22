"""
Command-line backtest runner with integrated data processing pipeline.

Examples
--------
uv run python -m scripts.run_backtest --market_data spydata/data.csv
"""
import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Support direct script execution as well as `python -m scripts.run_backtest`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.BackTestingEngine import BackTestingEngine
from packages.Constants import ExecutionType, MachineLearningModel, StatisticsModel, StrategyType
from packages.DataPipeline import DataStore
from packages.Strategies import MAStrategy, MLStrategy, MultiFactorStrategy, RSIStrategy


def plot_equity(result_df: pd.DataFrame, out_path: str) -> str:
    """
    Plot the equity curve from a backtest results DataFrame.
    """
    plt.figure(figsize=(10, 4))
    result_df["equity"].plot()
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_drawdown(result_df: pd.DataFrame, out_path: str) -> str:
    """
    Plot the drawdown series derived from the equity curve.
    """
    eq = result_df["equity"]
    dd = eq / eq.cummax().clip(lower=1.0) - 1
    plt.figure(figsize=(10, 4))
    dd.plot()
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def now_str(
        delta: timedelta = timedelta(0),
) -> str:
    return (datetime.now(UTC) + delta).strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Run a modular strategy backtest on OHLCV data.")
    ap.add_argument("--market_data", required=True, help="Path to CSV/Parquet with OHLCV data")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--commission_bps", type=float, default=1.0, help="Per-side commission in bps")
    ap.add_argument("--slippage_bps", type=float, default=1.0, help="Per-side slippage in bps")
    ap.add_argument("--strategy_type", type=StrategyType, choices=list(StrategyType),
                    default=StrategyType.STAT, help="Strategy type")
    ap.add_argument("--strategy_model", help="Model enum value (default: MULTI_FACTOR or RIDGE_LINEAR)")
    ap.add_argument("--trade_on", type=ExecutionType, choices=list(ExecutionType),
                    default=ExecutionType.NEXT_OPEN, help="Execution timing")
    ap.add_argument("--override_cache", action="store_true", help="Rebuild the cleaned CSV cache")
    args = ap.parse_args(argv)

    # Choose strategy implementation
    if args.strategy_type == StrategyType.STAT:
        factories = {
            StatisticsModel.MULTI_FACTOR: MultiFactorStrategy,
            StatisticsModel.MA: MAStrategy,
            StatisticsModel.RSI: RSIStrategy,
        }
        try:
            model = StatisticsModel(args.strategy_model or StatisticsModel.MULTI_FACTOR)
        except ValueError:
            ap.error(f"statistics model must be one of: {', '.join(StatisticsModel)}")
        strat = factories[model]()
    else:
        try:
            model = MachineLearningModel(args.strategy_model or MachineLearningModel.RIDGE)
        except ValueError:
            ap.error(f"ML model must be one of: {', '.join(MachineLearningModel)}")
        strat = MLStrategy(model=model)

    ds = DataStore(override_cache=args.override_cache)
    market_data = ds.read(args.market_data)
    if market_data.empty:
        ap.error("market data contains no valid bars")

    engine = BackTestingEngine(
        market_data=market_data,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        trade_on=args.trade_on,
    )

    print(f"{now_str()} Backtesting Started...")
    out = engine.run(strategy=strat)

    # Persist artifacts
    os.makedirs(args.outdir, exist_ok=True)
    # Undefined statistics are JSON null, not non-standard NaN tokens.
    summary = {
        key: value if math.isfinite(value) else None
        for key, value in out["summary"].items()
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    # Save results
    out["results"].to_parquet(os.path.join(args.outdir, "results.parquet"))

    # Charts
    plot_equity(out["results"], os.path.join(args.outdir, "equity.png"))
    plot_drawdown(out["results"], os.path.join(args.outdir, "drawdown.png"))

    print(f"{now_str()} Backtesting Completed.\nResults: \n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
