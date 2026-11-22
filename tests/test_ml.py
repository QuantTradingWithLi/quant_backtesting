import numpy as np
import pandas as pd
import pytest

from packages.Constants import MachineLearningModel
from packages.ML.Features import FeatureEngineer
from packages.ML.Models import (
    DecisionStumpRegressor,
    EnsembleAverage,
    KNNRegressor,
    LogisticRegressionGD,
    RidgeLinear,
)
from packages.ML.Walkforward import WalkForward
from packages.Strategies import MLStrategy


def prices_with_volume():
    index = pd.date_range("2020-01-01", periods=160)
    close = 100 + np.arange(len(index)) * 0.1 + np.sin(np.arange(len(index)))
    return pd.DataFrame({"open": close, "close": close, "volume": 1000}, index=index)


def test_features_keep_latest_row_without_fabricating_target():
    prices = prices_with_volume()
    X, y = FeatureEngineer(include_volume=False).transform(prices)
    assert X.index[-1] == prices.index[-1]
    assert pd.isna(y.iloc[-1])
    assert np.isfinite(X.to_numpy()).all()


def test_constant_volume_does_not_produce_infinite_features():
    X, _ = FeatureEngineer().transform(prices_with_volume())
    assert not X.empty
    assert np.isfinite(X.to_numpy()).all()
    assert (X["vol_z"] == 0).all()


def test_walk_forward_predicts_latest_row_and_only_fits_known_labels():
    prices = prices_with_volume()
    fit_dates = []

    class RecordingModel:
        def fit(self, X, y):
            assert y.notna().all()
            fit_dates.append(X.index[-1])
            return self

        def predict(self, X):
            assert fit_dates[-1] < X.index[0]
            return np.array([0.1])

    wf = WalkForward(
        RecordingModel, FeatureEngineer(include_volume=False),
        train_window=20, retrain_every=5, min_train=10, alpha=1
    )
    predictions = wf.predict(prices)
    assert predictions.iloc[-1] == 0.1
    assert predictions.iloc[:69].isna().all()
    prefix = wf.predict(prices.iloc[:-5])
    pd.testing.assert_series_equal(prefix, predictions.iloc[:-5])


@pytest.mark.parametrize("model", [
    RidgeLinear(), LogisticRegressionGD(), KNNRegressor(),
    DecisionStumpRegressor(), EnsembleAverage([RidgeLinear()]),
])
def test_models_validate_fitted_state_and_training_data(model):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0.1, -0.1, 0.2])
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)
    with pytest.raises(ValueError):
        model.fit(X.iloc[:0], y.iloc[:0])
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()
    with pytest.raises(ValueError):
        model.fit(X, y.iloc[::-1])


def test_stump_supports_constant_features_and_resets_on_refit():
    model = DecisionStumpRegressor()
    y = pd.Series([1.0, 2.0, 3.0])
    model.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), y)
    constant = pd.DataFrame({"a": [1.0, 1.0, 1.0]})
    model.fit(constant, y)
    np.testing.assert_allclose(model.predict(constant), [2.0, 2.0, 2.0])


def test_unregularized_ridge_handles_singular_features():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]})
    model = RidgeLinear(alpha=0).fit(X, pd.Series([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(model.predict(X), [2.0, 2.0, 2.0])


def test_knn_and_ensemble_reject_invalid_configuration():
    X = pd.DataFrame({"a": [1.0]})
    y = pd.Series([0.1])
    with pytest.raises(ValueError):
        KNNRegressor(k=0).fit(X, y)
    with pytest.raises(ValueError):
        EnsembleAverage([]).fit(X, y)


@pytest.mark.parametrize("model", list(MachineLearningModel))
def test_all_ml_strategies_produce_causal_positions(model):
    prices = prices_with_volume()
    strategy = MLStrategy(model=model, train_window=30, min_train=10)
    positions = strategy.generate_positions(prices)
    assert positions.index.equals(prices.index)
    assert set(positions.unique()).issubset({-1, 0, 1})
    assert positions.iloc[:69].eq(0).all()
    pd.testing.assert_series_equal(
        strategy.generate_positions(prices.iloc[:-5]), positions.iloc[:-5]
    )
