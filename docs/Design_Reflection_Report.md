
# Backtesting Platform: Design & Rationale

## Goals
- Build a modular, extensible backtesting framework from scratch (no third‑party backtesting libs).
- Handle raw OHLCV CSVs at scale (10k+ files) and convert to an efficient store for fast queries.
- Provide static signal implementations (MA(20/60), RSI(14)) plus a composite signal.
- Simulate execution with realistic frictions (commission, slippage, trade timing).
- Produce concise performance metrics and basic visuals.

## Architecture
**Packages**
- `BackTestingEngine`: orchestration (`BackTestingEngine`, `Simulator`, `Metrics`).
- `TradingEngine`: minimal live-trading engine skeleton (`MarketDataProvider`, `OrderExecutor`, `TradingEngine`).
- `DataPipeline`: cleaning + ingestion to Parquet (`DataCleaner`, `DataStore`).
- `Signals`: indicators & signals (`MASignal`, `RSISignal`).
- `Strategies`: modularized trading strategies (`MAStrategy`, `RSIStrategy`, `MultiFactorStrategy`, `MLStrategy`).
- `Constants`: pre-defined enums (`ExecutionType`, `StrategyType`, `StatisticsModel`, `MachineLearningModel`).


**Data Pipeline Rationale**
- CSV → Parquet conversion for 2–5x smaller footprint and much faster columnar reads.
- Cleaning rules tolerate common vendor glitches:
  - Parse/validate dates; drop invalid and duplicates.
  - Coerce numerics; negative prices/volumes → NaN.
  - Repair inverted `high`/`low`; enforce `low <= open,close <= high`.
  - Causal forward-fill for missing points; drop unrepairable leading bars.
  - Reject close-return outliers relative to the last accepted close (configurable heuristic).
  - Robust volume de‑spiking via rolling‑median MAD z‑score clipping (configurable).
- Parallel conversion via `ProcessPoolExecutor` supports 10k+ files.

**Signals**
- Trend: `MA(20)` vs `MA(60)` → {+1, 0, −1}.
- Mean‑reversion: `RSI(14)` with 30/70 thresholds → {+1, 0, −1}.
- Composite: convex combination (`0.6 * trend + 0.4 * meanrev`), then sign with a small dead‑band.

**Execution Model**
- Positions are the prior day’s end‑of‑day signal; trades execute at next day’s open (`NEXT_OPEN`), with per‑side bps commission + slippage.
- Old exposure earns the close-to-open gap; the new target earns the open-to-close return. These return legs compound.
- `MOC` is idealized close execution; the prior position earns the close-to-close return and the new target is set afterward.
- Positions are target exposures, not a share/cash ledger. Costs are proportional to absolute target changes, including the initial entry; no forced terminal liquidation.
- Results include daily net/gross returns, trades, position, equity curve.

**Metrics**
- Annualized compounded return/volatility, arithmetic excess-return Sharpe (default annual rf=2.66%), max drawdown, Calmar, win rate, total return.
- Drawdown includes initial equity of 1.0. Undefined statistics remain NaN internally and are serialized as JSON null.

**Performance & Scaling**
- Columnar Parquet enables lazy I/O and fast slicing across assets.
- Pandas/Numpy vectorization suffices for single‑asset daily bars. For multi‑asset intraday scale‑out, consider Polars/pyarrow, Ray/Dask, and numba where hotspots arise.
- With 24 cores/64GB RAM, conversion of 10k CSVs can be sharded 8–16 ways.

**Limitations & Next Steps**
- Cost model is simplified; extend with fixed fees, spread model, partial fills.
- Corporate actions/adjustments not modeled here; integrate with vendor adj factors.
- No portfolio optimizer; single‑asset example. Generalize to basket + risk targeting.
- Completed-close signals cannot guarantee actual market-on-close auction fills; use next-open execution for the default causal timing.
- Price jump filtering can discard genuine jumps/splits; validate and adjust vendor data before research.
- Parquet reads load a complete frame; lazy asset/date queries and GPU processing are not implemented.
- See [README.md](../README.md) for current capabilities, limitations, and validation commands.

## Stage 2 – ML/AI Integration Plan

### Objectives
- Extend from technical signals to **data-driven predictive models**.
- Keep ML strategies compatible with the same backtesting & live engine.
- Enable experimentation with supervised, unsupervised, and reinforcement learning.

### Architecture
- Add `MLStrategy` implementing `Strategy` interface.
- Separate **feature engineering** and **model** components.

```python
class MLStrategy(Strategy):
    def __init__(self, model, feature_engineer):
        self.model = model
        self.fe = feature_engineer

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        X = self.fe.transform(prices)
        preds = self.model.predict(X)
        return pd.Series(np.sign(preds), index=prices.index, name="MLStrategy")
```

### Feature Engineering
- Price-based: returns, vol, MA, RSI, MACD, Bollinger.
- Volume-based: OBV, VWAP, spikes.
- Optional: intraday/order book features, alternative data.

### Model Families
- Supervised: Logistic Regression, RF, Gradient Boosting, Neural Nets.
- Sequence models: LSTM/GRU, TCN, Transformer predictors.
- Probabilistic: Bayesian linear, HMM regimes.
- Reinforcement Learning: wrap Simulator as OpenAI Gym environment.

### Training & Validation
- Walk-forward CV (already supported).
- Keep feature-valid rows for inference even when their future label is unknown; only earlier observed labels enter a fit.
- Rolling train/test windows.
- Hyperparameter tuning with Optuna/sklearn.
- Evaluate by Sharpe, Sortino, drawdown, turnover.

### Backtest Integration
- MLStrategy produces positions -> Simulator -> Metrics.
- Identical pipeline as rule-based strategies.

### Live Deployment
- LiveTradingEngine calls `FeatureEngineer` + `model.predict` for latest data.
- Models serialized with joblib/ONNX.
- Add monitoring for drift and PnL.

### Risks & Safeguards
- Overfitting: mitigate with rolling OOS tests.
- Data snooping: strict train/test separation.
- Latency & robustness: benchmark inference pipeline.

---
This Stage-2 plan allows gradual adoption of ML/AI without breaking the clean modular design established in Stage-1.
