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
        prediction_intervals: Whether to calculate prediction intervals
        confidence_level: Confidence level for prediction intervals (0-1)
    """
    model: LinearRegression
    X: pd.DataFrame
    prediction_intervals: bool = Field(
        default=False,
        description="Whether to calculate prediction intervals"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for prediction intervals"
    )
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
    def calculate_prediction_intervals(
        model: LinearRegression,
        X: pd.DataFrame,
        predictions: np.ndarray,
        confidence_level: float
    ) -> Dict[str, pd.Series]:
        """Calculate prediction intervals for linear regression
        
        Args:
            model: Trained LinearRegression model
            X: Feature DataFrame
            predictions: Model predictions
            confidence_level: Confidence level (0-1)
            
        Returns:
            Dictionary containing lower and upper prediction bounds
        """
        from scipy import stats
        
        # Get MSE from training
        mse = np.mean((model.predict(X) - predictions) ** 2)
        
        # Standard error of predictions
        X_new = X.values
        n = X_new.shape[0]
        p = X_new.shape[1]
        
        # Diagonal of hat matrix
        hat_matrix_diag = np.einsum('ij,ij->i', X_new.dot(np.linalg.inv(X_new.T.dot(X_new))), X_new)
        
        # Calculate prediction intervals
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, n-p)
        
        prediction_std = np.sqrt(mse * (1 + hat_matrix_diag))
        margin_of_error = t_value * prediction_std
        
        return {
            "lower": pd.Series(predictions - margin_of_error, name="Lower Bound"),
            "upper": pd.Series(predictions + margin_of_error, name="Upper Bound")
        }
    
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
        if inputs.prediction_intervals and not (0 < inputs.confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
            
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
        
        # Calculate prediction intervals if requested
        prediction_intervals = None
        if inputs.prediction_intervals:
            try:
                prediction_intervals = PredictNode.calculate_prediction_intervals(
                    inputs.model,
                    inputs.X,
                    y_pred,
                    inputs.confidence_level
                )
                prediction_stats["confidence_level"] = inputs.confidence_level
            except Exception as e:
                print(f"Warning: Could not calculate prediction intervals: {str(e)}")
        
        return PredictOutput(
            predictions=y_pred_series,
            prediction_intervals=prediction_intervals,
            prediction_stats=prediction_stats
        )
