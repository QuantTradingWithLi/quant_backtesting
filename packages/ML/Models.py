"""
Tiny ML models implemented without external dependencies.

All models share a minimal API:
    fit(X: DataFrame, y: Series) -> self
    predict(X: DataFrame) -> np.ndarray
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _training_arrays(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Shared model contract: nonempty, finite, uniquely named, aligned data."""
    if X.empty or not X.columns.is_unique:
        raise ValueError("training features must be nonempty with unique columns")
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("training features and targets must have matching unique indices")
    Xn = X.to_numpy(dtype=float)
    yn = y.to_numpy(dtype=float)
    if yn.ndim != 1 or not np.isfinite(Xn).all() or not np.isfinite(yn).all():
        raise ValueError("training features and targets must be finite")
    return Xn, yn


def _prediction_array(X: pd.DataFrame, columns: list[str] | None) -> np.ndarray:
    if columns is None:
        raise ValueError("Model is not fitted.")
    if not X.columns.is_unique or not set(columns).issubset(X.columns):
        raise ValueError("prediction data is missing required or unique feature columns")
    values = X[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction features must be finite")
    return values


class MLModel(ABC):
    """Abstract base for any machine learning model."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLModel:
        """
        Train model.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        """
        ...

    @property
    def name(self) -> str:
        """Human-friendly identifier (class name by default)."""
        return self.__class__.__name__


@dataclass
class RidgeLinear(MLModel):
    """
    Ridge-regularized linear regression solved in closed form.
    y_hat = X @ w + b
    """
    alpha: float = 1.0
    fit_intercept: bool = True
    w_: np.ndarray | None = None
    b_: float = 0.0
    cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLModel:
        """
        Train the ridge regression in closed form.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features). Column names are
            stored and must be present at prediction time.
        y : pd.Series
            Target values with shape (n_samples,). Should be numeric.

        Returns
        -------
        RidgeLinear
            The fitted model (self), for chaining.

        Notes
        -----
        - If `fit_intercept=True`, an intercept column of ones is appended and
          the intercept term is **not** penalized by ridge (`alpha`).
        - Solves an augmented least-squares system, avoiding the conditioning
          loss of normal equations and supporting singular designs at alpha=0.
        """
        Xn, yn = _training_arrays(X, y)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.cols_ = list(X.columns)
        if self.fit_intercept:
            X_aug = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        else:
            X_aug = Xn

        penalty = np.sqrt(self.alpha) * np.eye(X_aug.shape[1])
        if self.fit_intercept:
            penalty[-1, -1] = 0.0  # don't penalize intercept
        w_all = np.linalg.lstsq(
            np.vstack([X_aug, penalty]),
            np.concatenate([yn, np.zeros(X_aug.shape[1])]),
            rcond=None,
        )[0]

        if self.fit_intercept:
            self.w_ = w_all[:-1]
            self.b_ = float(w_all[-1])
        else:
            self.w_ = w_all
            self.b_ = 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the **same columns** (order-insensitive by name)
            used during training (`fit`). Extra columns are ignored; missing
            columns will raise a ValueError.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with continuous predictions.
        """
        if self.w_ is None:
            raise ValueError("Model is not fitted.")
        Xn = _prediction_array(X, self.cols_)
        return Xn @ self.w_ + self.b_


@dataclass
class LogisticRegressionGD(MLModel):
    """
    Binary logistic regression with L2 regularization, fitted by batch gradient descent.
    Outputs probability p(y=1|x). We return p - 0.5 as a centered score.
    """
    alpha: float = 1.0  # L2 penalty strength
    lr: float = 0.1  # learning rate
    epochs: int = 200
    fit_intercept: bool = True
    w_: np.ndarray | None = None
    b_: float = 0.0
    cols_: list[str] | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLModel:
        """
        Train logistic regression via batch gradient descent.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features). Column names are
            stored and must be present at prediction time.
        y : pd.Series
            Target values with shape (n_samples,). Values > 0 are treated as class 1,
            values <= 0 as class 0.

        Returns
        -------
        LogisticRegressionGD
            The fitted model (self), for chaining.

        Notes
        -----
        - Loss: binary cross-entropy + L2 regularization on `w` (not on intercept).
        - Gradients are averaged over the batch; we run `epochs` full passes.
        - This minimalist solver is for research/educational use and small datasets.
        """
        Xn, targets = _training_arrays(X, y)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(self.lr) or self.lr <= 0 or self.epochs <= 0:
            raise ValueError("learning rate and epochs must be positive")
        self.cols_ = list(X.columns)
        yn = (targets > 0).astype(float)  # label: 1 if positive target, else 0
        n, d = Xn.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0

        for _ in range(self.epochs):
            z = Xn @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
            p = self._sigmoid(z)
            # gradients
            grad_w = (Xn.T @ (p - yn)) / n + self.alpha * self.w_ / n
            self.w_ -= self.lr * grad_w
            if self.fit_intercept:
                grad_b = np.sum(p - yn) / n
                self.b_ -= self.lr * grad_b
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict centered probabilities for new samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the **same columns** used during training (`fit`).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with centered probabilities `p - 0.5`,
            where `p = P(y=1|x)`. Values in (-0.5, 0.5).
        """
        if self.w_ is None:
            raise ValueError("Model is not fitted.")
        Xn = _prediction_array(X, self.cols_)
        z = Xn @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
        p = self._sigmoid(z)
        return p - 0.5  # centered probability


@dataclass
class KNNRegressor(MLModel):
    """
    k-Nearest Neighbors regressor with uniform weights and Euclidean distance.
    This is a quick/simple reference model; not optimized for large datasets.
    """
    k: int = 5
    cols_: list[str] | None = None
    X_: np.ndarray | None = None
    y_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLModel:
        """
        Memorize the training set for nearest-neighbor queries.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features). Column names are
            stored and must be present at prediction time.
        y : pd.Series
            Target values with shape (n_samples,).

        Returns
        -------
        KNNRegressor
            The fitted model (self), for chaining.

        Notes
        -----
        - No actual optimization happens; we just store `X` and `y`.
        - Distance metric is Euclidean; neighbors use uniform weighting.
        """
        Xn, yn = _training_arrays(X, y)
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be a positive integer")
        self.cols_ = list(X.columns)
        self.X_ = Xn
        self.y_ = yn
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict by averaging the targets of the k closest training points.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the **same columns** used during training (`fit`).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with continuous predictions.

        Notes
        -----
        - Complexity is O(n_query * n_train * n_features); suitable for small sets.
        """
        if self.X_ is None or self.y_ is None:
            raise ValueError("Model is not fitted.")
        Xq = _prediction_array(X, self.cols_)
        # Compute pairwise distances (naive)
        dists = ((Xq[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2)
        k = min(self.k, self.X_.shape[0])
        idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        preds = self.y_[idx].mean(axis=1)
        return preds


@dataclass
class DecisionStumpRegressor(MLModel):
    """
    Single-feature, single-threshold regressor (decision stump).
    Splits each feature at its median and chooses the best split (MSE).
    """
    feature_: int | None = None
    threshold_: float | None = None
    left_value_: float | None = None
    right_value_: float | None = None
    cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLModel:
        """
        Fit a single-threshold regressor by scanning medians per feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features).
        y : pd.Series
            Target values with shape (n_samples,).

        Returns
        -------
        DecisionStumpRegressor
            The fitted model (self), for chaining.

        Notes
        -----
        - For each feature `j`, we split at its sample median and compute the
          mean target on each side; we pick the split with the lowest MSE.
        """
        Xn, yn = _training_arrays(X, y)
        self.cols_ = list(X.columns)
        # A constant model is the valid fallback when no split exists. Reset
        # every fit so a previous tree cannot leak into a constant-data refit.
        self.feature_ = 0
        self.threshold_ = np.inf
        self.left_value_ = self.right_value_ = float(yn.mean())
        best_mse = np.inf

        for j in range(Xn.shape[1]):
            thr = np.nanmedian(Xn[:, j])
            left = Xn[:, j] <= thr
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            lv = yn[left].mean()
            rv = yn[right].mean()
            mse = ((yn[left] - lv) ** 2).sum() + ((yn[right] - rv) ** 2).sum()
            if mse < best_mse:
                best_mse = mse
                self.feature_ = j
                self.threshold_ = thr
                self.left_value_ = lv
                self.right_value_ = rv
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the learned single-threshold rule.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the **same columns** used during training (`fit`).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with piecewise-constant predictions.
        """
        if self.feature_ is None:
            raise ValueError("Model is not fitted.")
        Xn = _prediction_array(X, self.cols_)
        left = Xn[:, self.feature_] <= self.threshold_
        out = np.where(left, self.left_value_, self.right_value_)
        return out


@dataclass
class EnsembleAverage(MLModel):
    """
    Simple averaging ensemble over a list of base models.
    """
    base_models: list[MLModel]
    fitted_models_: list[MLModel] = field(default_factory=list, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLModel:
        """
        Fit all base models independently on the same data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features).
        y : pd.Series
            Target values with shape (n_samples,).

        Returns
        -------
        EnsembleAverage
            The fitted ensemble (self), for chaining.

        Notes
        -----
        - Each base model must expose `fit(X, y)` and `predict(X)`.
        - The ensemble stores the fitted models internally.
        """
        _training_arrays(X, y)
        if not self.base_models:
            raise ValueError("ensemble must contain at least one model")
        self.fitted_models_ = []
        fitted = [model.fit(X, y) for model in self.base_models]
        self.fitted_models_ = fitted
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict by averaging the predictions from all fitted base models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the **same columns** expected by the base models.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with averaged predictions.
        """
        if not self.fitted_models_:
            raise ValueError("Model is not fitted.")
        preds = [m.predict(X) for m in self.fitted_models_]
        return np.mean(preds, axis=0)
