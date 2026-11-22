# Code review and refactor

## Scope

Reviewed the data pipeline, indicators, strategies, simulator, metrics, custom ML
models, walk-forward predictions, live-engine skeleton, CLIs and tests. Kept
existing public module/class names and enum values. Pre-existing `.idea/`
deletions and checked-in market data/results were not part of this refactor.

## Findings addressed

| Priority | Finding | Fix and regression coverage |
| --- | --- | --- |
| High | `NEXT_OPEN` ignored opening prices, granting new entries overnight returns and dropping overnight exit P&L. | Split overnight/intraday exposure and compound the legs. Entry-gap, exit-gap and continuous-long tests in `tests/test_simulator.py`. |
| High | MOC positions earned same-bar returns before entry, and initial entries escaped transaction costs. | Previous exposure earns close-to-close returns; new targets pay turnover costs at the close, including row zero. MOC regression test in `tests/test_simulator.py`. |
| High | Time interpolation/backfill used future data despite a no-lookahead claim. | Forward-only repairs, dropped unrepairable leading bars, last-accepted-close filtering, vectorized OHLC bounds. Prefix-invariance and recovery tests in `tests/test_data_cleaner.py`. |
| High | ML dropped feature rows without future labels, so the latest prediction was stale/absent. Constant-volume features also removed all samples. | Separate feature validity from label availability, handle flat-volume scores, train only on earlier known labels, and stop forward-filling missing predictions. Latest-row and prefix-invariance tests in `tests/test_ml.py`, including every model enum. |
| Medium | Backtesting stripped volume before strategy generation. | Pass a copy of the full frame to the strategy. Coverage in `tests/test_backtesting_engine.py`. |
| Medium | RSI was NaN for uninterrupted gains and flat prices. | Define gain-only RSI as 100 and flat RSI as 50; retain loss-only RSI of 0. Tests in `tests/test_signals.py`. |
| Medium | Drawdown ignored initial capital; Sharpe incorrectly substituted CAGR for mean excess returns. | Include equity 1.0 in the running peak and compute arithmetic excess-return Sharpe. Align chart drawdown. Tests in `tests/test_simulator.py`. |
| Medium | DataStore tried to read Parquet as CSV; stale caches bypassed changed sources or cleaning settings. | Dispatch by suffix and persist source/configuration metadata in Parquet attrs. Rebuild legacy or mismatched CSV caches. Coverage in `tests/test_data_store.py`. |
| Medium | Model assertions, singular ridge designs, constant stump refits, empty ensembles and invalid k/data caused opaque failures or stale predictions. | Shared finite/aligned data validation, explicit fitted-state errors, augmented least-squares ridge, constant stump fallback/reset, validated neighbors/ensemble. Tests in `tests/test_ml.py`. |
| Medium | `type=bool` parsed `"False"` as true; CLI paths/imports were fragile; JSON emitted non-standard NaN. | Explicit boolean parsing, repository-root examples, direct-script import support, validated strategy/model choices, configurable worker/execution settings, strict JSON. Coverage in `tests/test_cli.py`. |
| Medium | Live targets could be silently truncated from fractional values; empty data failed unclearly. | Validate chronology, alignment and target domain before submitting orders. Coverage in `tests/test_trading_engine.py`. |
| Low | Tests wrote persistent CSV artifacts; the old runner missed function-based tests; unused Torch increased installation weight. | Remove test file writes, use pytest consistently, move pytest to dev dependencies, remove unused Torch and refresh the lockfile. Add local pytest/Ruff configuration. |

## Behavior changes

- Historical performance will change. Previous results mixed lookahead cleaning,
  mismatched execution timing and incomplete metrics; they should not be used as
  reference results for the corrected engine.
- Reprocess CSV inputs to rebuild checked-in processed data before comparing new
  strategy results. Direct Parquet reads intentionally trust the supplied data.
- Missing aligned strategy targets remain neutral in backtests; explicit values
  outside `{-1, 0, 1}` now raise rather than being truncated.
- Source timestamps/size/path and cleaner settings govern CSV cache reuse.
  This is not a content-addressed cache and does not support concurrent writers.
- Invalid prices, negative fees, unsorted/duplicate indices and losses beyond
  available capital are rejected.

## Remaining limitations / follow-up

1. **Execution realism:** target exposures reset at return-leg boundaries. This
   is not share-level self-financing accounting, particularly for shorts.
   Borrow availability/costs, financing, margin, spread/impact, partial fills and
   forced terminal liquidation are outside this implementation.
2. **MOC feasibility:** a completed-close signal is not necessarily available
   before the auction cutoff. MOC remains an idealized research mode; use
   `NEXT_OPEN` unless the signal's real availability is established.
3. **Data policy:** causal filling is not equivalent to observing a tradable
   quote. The default 20% close-return filter can suppress genuine jumps and
   keep rejecting a persistent repricing. Corporate-action-adjusted data and
   an asset-appropriate cleaning policy are required.
4. **ML research:** the label is next-day close-to-close return, not necessarily
   the exact execution P&L being optimized. The ensemble mixes probability
   scores with return predictions without calibration. Frozen out-of-sample
   evaluation, train-only scaling/calibration and parameter selection remain
   research tasks; regression tests do not demonstrate trading profitability.
5. **Scale/deployment:** Parquet loading is eager, KNN is quadratic in query and
   training rows, and no GPU path exists. The live engine is a skeleton without
   persistent model/order state, duplicate-order protection or broker recovery.

## Validation

Use the repository's Python 3.12 environment:

```powershell
uv sync --locked
uv run pytest -q
uv run python tests/run_tests.py -q
uvx ruff check packages scripts tests
git diff --check
```

Tests exercise static and ML CLI output (Parquet, strict JSON and PNGs), direct
script invocation, Windows process-pool conversion, cache invalidation, model
edge cases, signal timing and causality. No external market downloads or live
broker orders are required.

Completed validation: **63 tests passed** through both test entry points; Ruff,
lockfile consistency and whitespace checks passed. A copy of the bundled CSV
also completed process-pool conversion and both multi-factor and KNN CLI
backtests across **6,302 rows**. Smoke-test artifacts were written outside the
repository; existing data, caches and output files were left unchanged. These
checks establish functionality, not strategy profitability.
