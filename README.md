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
- **Installing Dependencies: `uv sync`** 
- **Processing Data: `./scripts/process_data.py --csv_dir "../spydata" --out_dir "../processed_data" --override_cache True`**  
- **Running Backtest: `./scripts/run_backtest.py --market_data "../spydata/data.csv" --strategy_type "MACHINE_LEARNING" --strategy_model "K_NEAREST_NEIGHBORS_REGRESSOR"`**  
- **Running Unit Tests: `./tests/run_tests.py`**

---

## Closing Note
This challenge is intended purely as a demonstration of technical and architectural capabilities in quantitative systems development. It represents an independent and educational exploration of backtesting framework design principles.

