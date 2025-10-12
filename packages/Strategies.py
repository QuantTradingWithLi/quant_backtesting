from __future__ import annotations

"""
Strategy interface and reference implementations.

A Strategy receives historical prices and emits *target positions* in {-1, 0, 1}.
The interface is intentionally minimal so the same class can be used in
backtests (offline) and a trading engine (online).
"""
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from packages.Constants import MachineLearningModel
from packages.Signals import MASignal, RSISignal
from packages.ML.Walkforward import WalkForward
from packages.ML.Features import FeatureEngineer
from packages.ML.Models import RidgeLinear, LogisticRegressionGD, KNNRegressor, DecisionStumpRegressor, EnsembleAverage

class Strategy(ABC):
    """Abstract base for any strategy that turns prices into target positions."""

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        """
        Produce target positions aligned to the input index.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame with at least 'close' (and optionally 'open').

        Returns
        -------
        pd.Series (int)
            Values in {-1, 0, 1}.
        """
        ...

    @property
    def name(self) -> str:
        """Human-friendly identifier (class name by default)."""
        return self.__class__.__name__


@dataclass
class MAStrategy(Strategy):
    """Trend-following via MA cross (fast vs slow)."""
    fast: int = 20
    slow: int = 60

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return MASignal(self.fast, self.slow)(prices["close"]).astype(int).rename(self.name)


@dataclass
class RSIStrategy(Strategy):
    """Mean-reversion using RSI thresholds."""
    window: int = 14
    low_th: float = 30.0
    high_th: float = 70.0

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        return RSISignal(self.window, self.low_th, self.high_th)(prices["close"]).astype(int).rename(self.name)


@dataclass
class MultiFactorStrategy(Strategy):
    """
    Hybrid strategy combining MA(20/60) trend and RSI(14) mean-reversion.
    We take a weighted sum and quantize with a dead-band to {-1,0,1}.
    """
    w_trend: float = 0.6
    w_meanrev: float = 0.4
    fast: int = 20
    slow: int = 60
    rsi_window: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        ma = MASignal(self.fast, self.slow)(prices["close"]).astype(float)
        rsi = RSISignal(self.rsi_window, self.rsi_low, self.rsi_high)(prices["close"]).astype(float)
        s = self.w_trend * ma + self.w_meanrev * rsi
        s = s.where(s.abs() >= 0.25, 0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return s.astype(int).rename(self.name)


@dataclass
class MLStrategy(Strategy):
    """
    ML-driven strategy using walk-forward training on engineered features.
    Supports multiple model backends via `model_type`.
    """
    model: MachineLearningModel
    include_volume: bool = True
    train_window: int = 252 # ~ 1 year
    retrain_every: int = 20 # monthly
    min_train: int = 100
    alpha: float = 1.0      # ridge strength
    k: int = 5

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        def model_factory():
            if self.model == MachineLearningModel.RIDGE:
                return RidgeLinear(alpha=self.alpha)
            elif self.model == MachineLearningModel.LOGREG:
                return LogisticRegressionGD(alpha=self.alpha, lr=0.1, epochs=200)
            elif self.model == MachineLearningModel.KNN:
                return KNNRegressor(k=self.k)
            elif self.model == MachineLearningModel.STUMP:
                return DecisionStumpRegressor()
            elif self.model == MachineLearningModel.ENSEMBLE:
                return EnsembleAverage([
                    RidgeLinear(alpha=self.alpha),
                    LogisticRegressionGD(alpha=self.alpha, lr=0.1, epochs=200),
                    KNNRegressor(k=self.k),
                ])
            else:
                raise ValueError("Unknown machine learning model")

        wf = WalkForward(
            model_factory=model_factory,
            fe=FeatureEngineer(include_volume=self.include_volume),
            train_window=self.train_window,
            retrain_every=self.retrain_every,
            min_train=self.min_train,
            alpha=self.alpha
        )
        cols = [c for c in ["open", "close", "volume"] if c in prices.columns]
        preds = wf.predict(prices[cols])
        pos = preds.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        pos.name = self.name
        return pos.astype(int)
