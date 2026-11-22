"""
Walk-forward training/inference to avoid look-ahead and capture drift.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from packages.ML.Features import FeatureEngineer
from packages.ML.Models import MLModel


class WalkForward:
    """
    Produces out-of-sample predictions using rolling fit windows.
    """
    def __init__(
            self,
            model_factory: Callable[[], MLModel],
            fe: FeatureEngineer,
            train_window: int,
            retrain_every: int,
            min_train: int,
            alpha: float | None = None
    ):
        self.model_factory = model_factory
        self.fe = fe or FeatureEngineer()
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.min_train = min_train
        self.alpha = alpha
        # alpha is retained for API compatibility; model_factory owns its use.
        if not 0 < min_train <= train_window:
            raise ValueError("windows must satisfy 0 < min_train <= train_window")
        if retrain_every <= 0:
            raise ValueError("retrain_every must be positive")

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be unique and sorted")
        X, y = self.fe.transform(prices)
        # Align indices
        dates = X.index

        # We'll produce predictions for every row in X using walk-forward fit.
        preds = pd.Series(index=dates, dtype=float)

        last_fit = None
        model = None

        for i in range(len(dates)):
            # Labels for earlier rows are known at today's close, but the
            # current row's next-day return must never enter training.
            train_start_loc = max(0, i - self.train_window)
            X_train = X.iloc[train_start_loc:i]
            y_train = y.iloc[train_start_loc:i]
            known = y_train.notna()
            X_train = X_train.loc[known]
            y_train = y_train.loc[known]
            if len(X_train) < self.min_train:
                continue  # insufficient data to predict yet

            # Retrain at cadence or if model missing
            if (model is None) or (last_fit is None) or ((i - last_fit) >= self.retrain_every):
                model = self.model_factory().fit(X_train, y_train)
                last_fit = i

            # Predict for dt (out-of-sample relative to train window)
            preds.iloc[i] = float(model.predict(X.iloc[[i]])[0])

        # Missing features mean no prediction, not a stale forward-filled score.
        return preds.reindex(prices.index)
