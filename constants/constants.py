"""Constants and enums used across the workflow modules"""

from enum import Enum

class MetricType(str, Enum):
    """Available metrics for model evaluation
    
    Attributes:
        R2: R-squared score (coefficient of determination)
        MSE: Mean squared error
        RMSE: Root mean squared error
        MAE: Mean absolute error
        ALL: Compute all available metrics
    """
    R2 = "r2"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    ALL = "all"