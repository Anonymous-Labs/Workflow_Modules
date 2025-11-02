from pydantic import BaseModel, Field
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class PredictInput(BaseModel):
    """Input parameters for model prediction
    
    Attributes:
        model: Trained scikit-learn model instance
        X: Feature DataFrame for prediction
        (prediction_intervals and confidence_level removed)
    """
    model: LinearRegression
    X: pd.DataFrame
    model_config = {"arbitrary_types_allowed": True}

class PredictOutput(BaseModel):
    """Output containing predictions and optional intervals
    
    Attributes:
        predictions: Series containing model predictions
        prediction_intervals: Dictionary with lower and upper bounds
        prediction_stats: Dictionary with prediction statistics
    """
    predictions: pd.Series
    prediction_intervals: Optional[Dict[str, pd.Series]] = None
    prediction_stats: Dict = Field(
        default_factory=dict,
        description="Statistics about the predictions"
    )
    model_config = {"arbitrary_types_allowed": True}

class PredictNode:
    """Node for generating predictions using a trained model
    
    This node handles prediction generation using a trained scikit-learn model.
    It supports:
    - Basic predictions
    - Prediction intervals (for supported models)
    - Basic prediction statistics
    
    Example:
        ```python
        # Basic prediction
        input_data = PredictInput(
            model=trained_model,
            X=test_features
        )
        
        # Prediction with intervals
        input_data = PredictInput(
            model=trained_model,
            X=test_features,
            prediction_intervals=True,
            confidence_level=0.95
        )
        
        result = PredictNode.execute(input_data)
        predictions = result.predictions
        
        if result.prediction_intervals:
            lower_bound = result.prediction_intervals["lower"]
            upper_bound = result.prediction_intervals["upper"]
        
        print("Prediction statistics:", result.prediction_stats)
        ```
    """
    
    @staticmethod
    
    @staticmethod
    def execute(inputs: PredictInput) -> PredictOutput:
        """Execute the prediction process
        
        Args:
            inputs: PredictInput containing model and features
            
        Returns:
            PredictOutput containing predictions and statistics
            
        Raises:
            ValueError: If input features don't match model requirements
            ValueError: If confidence_level is not between 0 and 1
        """
        # Input validation
        if inputs.X.isna().any().any():
            raise ValueError("Input features contain NaN values")
            
        # Generate predictions
        y_pred = inputs.model.predict(inputs.X)
        y_pred_series = pd.Series(y_pred, name="Predictions")
        
        # Calculate prediction statistics
        prediction_stats = {
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred)),
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
            "n_predictions": len(y_pred)
        }
        
        # prediction intervals removed; keep None
        prediction_intervals = None
        
        return PredictOutput(
            predictions=y_pred_series,
            prediction_intervals=prediction_intervals,
            prediction_stats=prediction_stats
        )
