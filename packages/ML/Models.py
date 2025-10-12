from __future__ import annotations

"""
Tiny ML models implemented without external dependencies.

All models share a minimal API:
    fit(X: DataFrame, y: Series) -> self
    predict(X: DataFrame) -> np.ndarray
"""

import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
        - Uses `(X^T X + alpha I)^{-1} X^T y` with care to avoid penalizing
          the intercept dimension.
        """
        self.cols_ = list(X.columns)
        Xn = X.values.astype(float)
        yn = y.values.astype(float).reshape(-1, 1)
        if self.fit_intercept:
            X_aug = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        else:
            X_aug = Xn

        I = np.eye(X_aug.shape[1])
        if self.fit_intercept:
            I[-1, -1] = 0.0  # don't penalize intercept
        XtX = X_aug.T @ X_aug
        XtY = X_aug.T @ yn
        w_all = np.linalg.solve(XtX + self.alpha * I, XtY).ravel()

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
            columns will raise an assertion error.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with continuous predictions.
        """
        assert self.w_ is not None and self.cols_ is not None, "Model is not fitted."
        Xn = X[self.cols_].values.astype(float)
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
        self.cols_ = list(X.columns)
        Xn = X.values.astype(float)
        yn = (y.values > 0).astype(float)  # label: 1 if positive target, else 0
        n, d = Xn.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0 if self.fit_intercept else 0.0

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
        assert self.w_ is not None and self.cols_ is not None, "Model is not fitted."
        Xn = X[self.cols_].values.astype(float)
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
        self.cols_ = list(X.columns)
        self.X_ = X[self.cols_].values.astype(float)
        self.y_ = y.values.astype(float)
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
        assert self.X_ is not None and self.y_ is not None and self.cols_ is not None, "Model is not fitted."
        Xq = X[self.cols_].values.astype(float)
        # Compute pairwise distances (naive)
        dists = np.sqrt(((Xq[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        idx = np.argpartition(dists, kth=min(self.k, self.X_.shape[0] - 1), axis=1)[:, :self.k]
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
        self.cols_ = list(X.columns)
        Xn = X.values.astype(float)
        yn = y.values.astype(float)
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
        assert self.feature_ is not None and self.cols_ is not None, "Model is not fitted."
        Xn = X[self.cols_].values.astype(float)
        left = Xn[:, self.feature_] <= self.threshold_
        out = np.where(left, self.left_value_, self.right_value_)
        return out


@dataclass
class EnsembleAverage(MLModel):
    """
    Simple averaging ensemble over a list of base models.
    """
    base_models: List[object]

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
        self.fitted_models_ = []
        for m in self.base_models:
            self.fitted_models_.append(m.fit(X, y))
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
        preds = [m.predict(X) for m in self.fitted_models_]
        return np.mean(preds, axis=0)
