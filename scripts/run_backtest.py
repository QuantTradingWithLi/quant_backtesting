"""
Command-line backtest runner with integrated data processing pipeline.

Examples
--------
python run_backtest.py --market_data "../spydata/data.csv" --strategy_type "STATISTICS" --strategy_model "MULTI_FACTOR" --commission_bps 1 --slippage_bps 1
"""
import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from packages.Constants import StrategyType, StatisticsModel, ExecutionType
from packages.DataPipeline import DataStore
from packages.BackTestingEngine import BackTestingEngine
from packages.Strategies import MAStrategy, RSIStrategy, MultiFactorStrategy, MLStrategy


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
    dd = eq / eq.cummax() - 1
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
    return (datetime.now().astimezone(timezone.utc) + delta).strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(description="Run a modular strategy backtest on OHLCV data.")
    ap.add_argument("--market_data", required=True, help="Path to CSV/Parquet with OHLCV data")
    ap.add_argument("--outdir", default="../out", help="Output directory")
    ap.add_argument("--commission_bps", type=float, default=1.0, help="Per-side commission in bps")
    ap.add_argument("--slippage_bps", type=float, default=1.0, help="Per-side slippage in bps")
    ap.add_argument("--strategy_type", default=StrategyType.STAT, help="Strategy type")
    ap.add_argument("--strategy_model", default=StatisticsModel.MULTI_FACTOR, help="Strategy model")
    args = ap.parse_args()

    # Choose strategy implementation
    if args.strategy_type == StrategyType.STAT:
        if args.strategy_model == StatisticsModel.MULTI_FACTOR:
            strat = MultiFactorStrategy()
        elif args.strategy_model == StatisticsModel.MA:
            strat = MAStrategy()
        elif args.strategy_model == StatisticsModel.RSI:
            strat = RSIStrategy()
        else:
            raise ValueError("Unknown statistics model")
    elif args.strategy_type == StrategyType.ML:
        strat = MLStrategy(model=args.strategy_model)
    else:
        raise ValueError("Unknown strategy type")

    ds = DataStore(override_cache=False)
    market_data = ds.read(args.market_data)

    engine = BackTestingEngine(
        market_data=market_data,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        trade_on=ExecutionType.NEXT_OPEN,
    )

    print(f"{now_str()} Backtesting Started...")
    out = engine.run(strategy=strat)

    # Persist artifacts
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(out["summary"], f, indent=2)

    # Save results
    out["results"].to_parquet(os.path.join(args.outdir, "results.parquet"))

    # Charts
    plot_equity(out["results"], os.path.join(args.outdir, "equity.png"))
    plot_drawdown(out["results"], os.path.join(args.outdir, "drawdown.png"))

    print(f"{now_str()} Backtesting Completed.\nResults: \n{json.dumps(out["summary"], indent=2)}")


if __name__ == "__main__":
    main()
