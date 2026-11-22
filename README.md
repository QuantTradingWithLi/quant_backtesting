# Quantitative Backtesting Framework Challenge

## Disclaimer
_All materials in this repository, including text, code, and documentation, were created entirely by me. This project is an independent work inspired by common practices in quantitative finance and software design. It is **not affiliated with, endorsed by, or derived from any proprietary materials** belonging to OPTrust or any other organization._

## Overview
This challenge is designed to demonstrate the ability to design and implement a modular, extensible backtesting framework for quantitative trading research. The project highlights skills in data handling, framework architecture, code modularity, algorithm implementation, and integration of machine learning concepts in systematic trading environments.

The exercise focuses on building the full lifecycle of a backtesting system, from data ingestion and signal generation to execution simulation and performance evaluation. The objective is to produce a flexible, production-ready foundation that can be extended to test various trading ideas.

## Project Scope
### Stage 1: Foundational Backtesting Framework
**Goal:** Build a complete backtesting framework from scratch capable of handling simple, rule-based (static) trading strategies.

#### Key Components
- **Data Processing and Storage**  
  - Efficiently load and manage large-scale OHLCV datasets (10k+ files).  
  - Implement an optimized storage and query mechanism for rapid access.  
  - Support parallel processing using multi-core and GPU resources.  
  - Provide documentation explaining performance design choices.

- **Signal Generation**  
  - Implement core trading signals such as:  
    - Trend-following using moving average (MA60/MA20) crossovers.  
    - Mean-reversion using RSI(14).  
  - Enable signal combination and strategy parameterization.

- **Execution and Simulation**  
  - Model position management, trade tracking, and transaction history.  
  - Handle realistic aspects such as slippage and transaction costs (optional extensions).

- **Performance Analytics**  
  - Compute essential performance metrics: cumulative return, Sharpe ratio, and drawdown.  
  - Provide summary reports and visualizations for results.

- **Software Design and Modularity**  
  - Apply OOP design patterns for extensibility.  
  - Ensure the system is modular, documented, and unit-testable.

#### Deliverables
- Source code implementing the full framework and sample strategies.  
- Design notes or diagrams describing architecture and data flow.  
- Technical documentation and comments following Python conventions.

---

### Stage 2: Integrating Machine Learning into Backtesting
**Goal:** Extend the discussion from rule-based trading to ML/AI-driven signal generation and evaluation.

#### Discussion Topics
- Architectural modifications required to incorporate predictive models.  
- Strategies for managing model retraining, versioning, and deployment.  
- Considerations for latency, reproducibility, and explainability in ML-based trading systems.  
- Monitoring, validation, and performance tracking for dynamic models.

#### Deliverables
- A written discussion outlining architectural proposals for ML/AI integration.  
- Design diagrams or conceptual flowcharts illustrating system extensions.

---

## Technical Environment
- **Languages & Libraries:** Python (NumPy, Pandas, PyTorch, TA-Lib, etc.)  
- **Hardware Context:** Assume access to a workstation with 24 CPU cores, 64GB RAM, and a 16GB GPU.  
- **Project Management:** Use virtual environments or dependency managers (e.g., `uv`, `poetry`, or `venv`).  

---

## Submission Guidelines
Participants are encouraged to submit:
- Source code (organized and documented).  
- Design documentation and diagrams.  
- Reflection notes summarizing design trade-offs and reasoning.

---

## Evaluation Criteria
- **Data engineering & performance optimization**  
- **Framework architecture & modularity**  
- **Code readability, documentation & testing**  
- **Quantitative modeling knowledge**  
- **Thoughtful discussion of ML/AI system design**

---

## Running The Code
Run these commands from the repository root with Python 3.12+ and `uv`:

```powershell
uv sync
uv run python -m scripts.process_data --csv_dir spydata --out_dir processed_data --override_cache --max_workers 8
uv run python -m scripts.run_backtest --market_data processed_data/data.parquet --outdir out
uv run python -m scripts.run_backtest --market_data processed_data/data.parquet --strategy_type MACHINE_LEARNING --strategy_model K_NEAREST_NEIGHBORS_REGRESSOR --outdir out
uv run pytest -q
uvx ruff check packages scripts tests
```

Direct script execution (`uv run python scripts/run_backtest.py ...`) and
`uv run python tests/run_tests.py` are also supported. `uv sync` includes the
development test dependency; use `uv sync --no-dev` for runtime dependencies only.
The implemented models use NumPy/Pandas, not PyTorch or GPU acceleration.

### Data and execution conventions
- CSV input requires `date`, `open`, `high`, `low`, and `close`; `volume` is optional.
  Column names are case-insensitive and whitespace is stripped.
- Cleaning is causal: missing prices use past observations only, leading
  unrepairable bars are dropped, and no calendar bars are invented. The default
  20% close-return filter is a data-quality heuristic, **not** a corporate-action
  adjustment; it can discard genuine moves. Configure `DataCleaner` appropriately
  for your data (use `price_ret_clip=float("inf")` to disable the filter).
- CSV caches record source metadata and cleaner settings. Legacy/incompatible
  caches are rebuilt automatically. Direct Parquet input is trusted as already
  processed; regenerate existing `processed_data/` files with the command above.
- Strategies receive all market-data columns and produce close-time target
  positions in `{-1, 0, 1}`. Prices must have a sorted, unique index.
- Default `NEXT_OPEN` execution applies yesterday's target at today's open:
  old exposure earns the overnight return and new exposure earns the intraday return.
  `--trade_on MARKET_ON_CLOSE` is idealized close execution and only earns returns
  after the signal's close; actual auction cutoffs are not modeled.
- This is a target-exposure return model, not a share/cash ledger. Costs apply to
  target turnover; shorts, financing, margin and corporate actions need further
  modeling. Final positions are marked to market, not forcibly liquidated.
- ML features retain the latest row even without a known next-day label.
  Walk-forward training uses earlier rows with observed labels only.
- Results include `ret`, `pos`, `trades`, `gross`, `net`, and `equity`. `ret` is the
  underlying close-to-close return, `pos` is end-of-day exposure, and `trades` is
  absolute target turnover. Summary JSON uses `null` for undefined statistics.

See [the code review](docs/Code_Review.md) for findings, fixes and remaining limitations.

---

## Closing Note
This challenge is intended purely as a demonstration of technical and architectural capabilities in quantitative systems development. It represents an independent and educational exploration of backtesting framework design principles.
