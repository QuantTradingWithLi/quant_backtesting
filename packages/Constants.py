from enum import StrEnum


class ExecutionType(StrEnum):
    NEXT_OPEN = "NEXT_OPEN"
    MOC = "MARKET_ON_CLOSE"

class StrategyType(StrEnum):
    STAT = "STATISTICS"
    ML = "MACHINE_LEARNING"

class StatisticsModel(StrEnum):
    # Traditional Statistics Models
    MA = "MOVING_AVERAGE"
    RSI = "RELATIVE_STRENGTH_INDEX"
    MULTI_FACTOR = "MULTI_FACTOR"

class MachineLearningModel(StrEnum):
    # Machine Learning Models
    RIDGE = "RIDGE_LINEAR"
    LOGREG = "LOGISTIC_REGRESSION_VIA_BATCH_GRADIENT_DESCENT"
    KNN = "K_NEAREST_NEIGHBORS_REGRESSOR"
    STUMP = "DECISION_STUMP_REGRESSOR"
    ENSEMBLE = "ENSEMBLE_AVERAGE"
