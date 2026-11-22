# Quantitative Backtesting Framework

## Disclaimer
_All materials in this repository, including text, code, and documentation, were created entirely by me. This project is an independent work inspired by common practices in quantitative finance and software design. It is **not affiliated with, endorsed by, or derived from any proprietary materials** belonging to OPTrust or any other organization._

## Overview

A modular Python framework for single-asset, daily-frequency trading research.
The workflow covers OHLCV ingestion, signal generation, execution simulation,
walk-forward machine learning, and performance reporting.

## Current Status

The project is a working research and educational prototype, not a production
trading system.

| Component | Available functionality |
| --- | --- |
| Data pipeline | CSV ingestion, causal OHLC(V) cleaning, Parquet storage, configuration-aware caching, and multi-process batch conversion. |
| Rule-based strategies | Moving-average crossover, RSI mean reversion, and a weighted multi-factor strategy. |
| Machine learning | NumPy-based ridge regression, logistic regression, KNN regression, decision-stump regression, and an averaging ensemble with rolling walk-forward training. |
| Execution | Next-open and idealized market-on-close target exposure, long/flat/short signals, and commission/slippage in basis points. |
| Analytics | Total and annualized return, annualized volatility, Sharpe ratio, maximum drawdown, Calmar ratio, and win rate. |
| Reporting | JSON summaries, Parquet result series, equity curves, and drawdown charts. |
| Live integration | A minimal engine with market-data and order-execution interfaces; broker connectivity and persistent state are not provided. |

**Validation:** 63 automated tests cover indicators, strategies, execution,
metrics, ML, caching, CLI output, and process-pool conversion. Data processing
and both multi-factor and KNN backtests have also been exercised on the bundled
6,302-row dataset. These checks demonstrate functionality, not profitability.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Runtime libraries: NumPy, Pandas, PyArrow, and Matplotlib
- CPU execution; no GPU is required

## Getting Started

Run commands from the repository root:

```powershell
uv sync --locked
```

### Process market data

```powershell
uv run python -m scripts.process_data --csv_dir spydata --out_dir processed_data --override_cache --max_workers 8
```

The processing CLI converts CSV files into cleaned Parquet files. Worker count is
configurable. Omit `--override_cache` to reuse matching caches; explicit values
such as `--override_cache False` are also supported.

### Run a rule-based backtest

```powershell
uv run python -m scripts.run_backtest --market_data processed_data/data.parquet --strategy_type STATISTICS --strategy_model MULTI_FACTOR --outdir out/statistics
```

Other statistical model values are `MOVING_AVERAGE` and
`RELATIVE_STRENGTH_INDEX`.

### Run a machine-learning backtest

```powershell
uv run python -m scripts.run_backtest --market_data processed_data/data.parquet --strategy_type MACHINE_LEARNING --strategy_model K_NEAREST_NEIGHBORS_REGRESSOR --outdir out/ml
```

Supported ML model values:

- `RIDGE_LINEAR` (the ML default)
- `LOGISTIC_REGRESSION_VIA_BATCH_GRADIENT_DESCENT`
- `K_NEAREST_NEIGHBORS_REGRESSOR`
- `DECISION_STUMP_REGRESSOR`
- `ENSEMBLE_AVERAGE`

The runner accepts CSV or Parquet input. Execution defaults to `NEXT_OPEN`;
`--trade_on MARKET_ON_CLOSE` selects idealized close execution.
`--commission_bps` and `--slippage_bps` each default to 1 basis point per side.
Each run writes `summary.json`, `results.parquet`, `equity.png`, and
`drawdown.png` to its output directory.

### Run checks

```powershell
uv run pytest -q
uvx ruff check packages scripts tests
```

Direct script execution (`uv run python scripts/run_backtest.py ...`) and
`uv run python tests/run_tests.py -q` are also supported. The default sync
includes pytest; `uv sync --locked --no-dev` installs runtime dependencies only.

## Data and Execution Conventions

- CSV input requires `date`, `open`, `high`, `low`, and `close`; `volume` is
  optional. Column names are case-insensitive and surrounding whitespace is stripped.
- Cleaning uses past observations for missing prices, drops unrepairable leading
  bars, enforces OHLC bounds, and does not invent calendar bars.
- The default close-return filter treats moves greater than 20% relative to the
  last accepted close as data errors. Configure `DataCleaner` for the asset and
  vendor; `price_ret_clip=float("inf")` disables this filter.
- CSV caches record source path, modification time, size, and cleaner settings.
  Cache reuse requires matching metadata. Direct Parquet input is trusted as
  already processed.
- Strategies receive all market-data columns and emit close-time targets in
  `{-1, 0, 1}`. Price indices must be sorted and unique, and prices must be finite
  and positive.
- With `NEXT_OPEN`, yesterday's target executes at today's open. Existing
  exposure earns the overnight return; the new exposure earns the intraday
  return. The return legs compound.
- With `MARKET_ON_CLOSE`, existing exposure earns the close-to-close return and
  the new target is established at the close.
- Transaction costs apply to absolute target turnover, including initial entry.
  Positions are marked to market at the end of the backtest without forced liquidation.
- ML features use information available at each close. Training uses earlier
  rows with observed next-day labels; inference does not require a future label.
- Results contain `ret`, `pos`, `trades`, `gross`, `net`, and `equity`. `ret` is
  the underlying close-to-close return, `pos` is end-of-day exposure, and
  `trades` is absolute target turnover.
- Metrics assume 252 trading days per year. Sharpe uses arithmetic excess returns
  with a default annual risk-free rate of 2.66%; drawdown includes initial equity
  of 1.0. Undefined statistics are represented as `null` in JSON.

## Project Structure

```text
packages/
  BackTestingEngine.py   Backtest orchestration, simulation, and metrics
  DataPipeline.py        Cleaning, caching, and CSV/Parquet conversion
  Signals.py             Moving-average and RSI signals
  Strategies.py          Rule-based and ML strategy implementations
  TradingEngine.py       Live-integration interfaces and engine skeleton
  Constants.py           Execution, strategy, and model enums
  ML/                    Features, model implementations, and walk-forward training
scripts/                 Data-processing and backtesting CLIs
tests/                   Unit and integration tests
docs/                    Architecture and design documentation
spydata/                 Bundled sample market data
processed_data/          Processed example data
out/                     Example reports and backtest artifacts
```

See [Design & Rationale](docs/Design_Reflection_Report.md) for architectural
context and longer-term design ideas.

## Limitations

- **Accounting:** simulation uses target exposures reset at return-leg boundaries,
  not a share/cash ledger. Short borrowing, financing, margin, market impact,
  partial fills, and portfolio-level allocation are not modeled.
- **Close execution:** a completed-close signal may not be available before an
  actual auction cutoff. Market-on-close execution is an idealized research mode.
- **Data quality:** forward-filled prices do not guarantee tradability. The
  return filter can suppress genuine jumps or sustained repricing, and corporate
  actions require an appropriate external adjustment policy.
- **ML evaluation:** the prediction label is next-day close-to-close return,
  which is not identical to every execution model's P&L. Ensemble members mix
  probability and return scores without calibration. Feature scaling, parameter
  selection, and independent out-of-sample evaluation remain research responsibilities.
- **Scale and deployment:** Parquet reads load complete frames, KNN pairwise
  distances can consume substantial memory, and caches do not support concurrent
  writers. The live engine has no persistent order/model state, duplicate-order
  protection, or broker recovery.
