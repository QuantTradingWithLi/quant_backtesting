from __future__ import annotations

"""
Walk-forward training/inference to avoid look-ahead and capture drift.
"""
import pandas as pd
from packages.ML.Features import FeatureEngineer


class WalkForward:
    """
    Produces out-of-sample predictions using rolling fit windows.
    """
    def __init__(
            self,
            model_factory,
            fe: FeatureEngineer,
            train_window: int,
            retrain_every: int,
            min_train: int,
            alpha: float
    ):
        self.model_factory = model_factory
        self.fe = fe or FeatureEngineer()
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.min_train = min_train
        self.alpha = alpha

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        X, y = self.fe.transform(prices)
        # Align indices
        dates = X.index

        # We'll produce predictions for every row in X using walk-forward fit.
        preds = pd.Series(index=dates, dtype=float)

        last_fit = None
        model = None

        for i, dt in enumerate(dates):
            # Determine training window ending *before* dt
            train_end_loc = dates.get_loc(dt)
            train_start_loc = max(0, train_end_loc - self.train_window)
            X_train = X.iloc[train_start_loc:train_end_loc]
            y_train = y.iloc[train_start_loc:train_end_loc]
            if len(X_train) < self.min_train:
                continue  # insufficient data to predict yet

            # Retrain at cadence or if model missing
            if (model is None) or (last_fit is None) or ((i - last_fit) >= self.retrain_every):
                model = self.model_factory().fit(X_train, y_train)
                last_fit = i

            # Predict for dt (out-of-sample relative to train window)
            preds.iloc[i] = float(model.predict(X.iloc[[i]])[0])

        # Reindex to full price index (forward-fill last known prediction for convenience)
        preds = preds.reindex(prices.index).ffill()
        return preds
