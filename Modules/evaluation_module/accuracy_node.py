from typing import List, Optional
from pydantic import BaseModel, Field
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

from constants.constants import MetricType

class AccuracyInput(BaseModel):
    """Input parameters for model evaluation
    
    Attributes:
        y_true: Actual target values
        y_pred: Model predictions
        metrics: List of metrics to compute (default: all metrics)
    """
    y_true: pd.Series
    y_pred: pd.Series
    metrics: Optional[List[MetricType]] = Field(
        default=[MetricType.ALL],
        description="Metrics to compute. Use ALL to compute all available metrics."
    )
    model_config = {"arbitrary_types_allowed": True}

class AccuracyOutput(BaseModel):
    """Output containing computed metrics
    
    Attributes:
        r2_score: R-squared score (1 is perfect prediction)
        mse: Mean squared error
        rmse: Root mean squared error
        mae: Mean absolute error
    """
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None

class AccuracyNode:
    """Evaluation node for computing regression metrics.
    
    This node computes various regression evaluation metrics including:
    - R-squared (R²): Proportion of variance explained by the model
    - Mean Squared Error (MSE): Average squared difference between predictions and true values
    - Root Mean Squared Error (RMSE): Square root of MSE, in same units as target
    - Mean Absolute Error (MAE): Average absolute difference between predictions and true values
    
    Example:
        ```python
        # Compute all metrics
        input_data = AccuracyInput(
            y_true=actual_values,
            y_pred=model_predictions,
            metrics=[MetricType.ALL]
        )
        
        # Compute specific metrics
        input_data = AccuracyInput(
            y_true=actual_values,
            y_pred=model_predictions,
            metrics=[MetricType.R2, MetricType.RMSE]
        )
        
        results = AccuracyNode.execute(input_data)
        ```
    """
    
    @staticmethod
    def execute(inputs: AccuracyInput) -> AccuracyOutput:
        """Execute the evaluation computation
        
        Args:
            inputs: AccuracyInput containing true values, predictions and metrics to compute
            
        Returns:
            AccuracyOutput containing the computed metrics
            
        Raises:
            ValueError: If input series have different lengths or contain NaN values
        """
        # Input validation
        if len(inputs.y_true) != len(inputs.y_pred):
            raise ValueError("True values and predictions must have the same length")
        if inputs.y_true.isna().any() or inputs.y_pred.isna().any():
            raise ValueError("Input data contains NaN values")
            
        metrics_to_compute = set()
        for metric in inputs.metrics:
            if metric == MetricType.ALL:
                metrics_to_compute.update([m for m in MetricType if m != MetricType.ALL])
            else:
                metrics_to_compute.add(metric)
        
        result = AccuracyOutput()
        
        # Compute requested metrics
        if MetricType.R2 in metrics_to_compute:
            result.r2_score = r2_score(inputs.y_true, inputs.y_pred)
            
        if MetricType.MSE in metrics_to_compute:
            result.mse = mean_squared_error(inputs.y_true, inputs.y_pred)
            
        if MetricType.RMSE in metrics_to_compute:
            result.rmse = np.sqrt(mean_squared_error(inputs.y_true, inputs.y_pred))
            
        if MetricType.MAE in metrics_to_compute:
            result.mae = mean_absolute_error(inputs.y_true, inputs.y_pred)
            
        return result
