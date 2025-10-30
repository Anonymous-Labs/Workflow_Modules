from pydantic import BaseModel, Field
from typing import Dict, Optional
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class LinearRegressionInput(BaseModel):
    """Input parameters for Linear Regression training
    
    Attributes:
        X_train: Training features DataFrame
        y_train: Training target Series
        fit_intercept: Whether to calculate the intercept (default: True)
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    fit_intercept: bool = Field(
        default=True,
        description="Whether to calculate the intercept for this model"
    )
    model_config = {"arbitrary_types_allowed": True}

class LinearRegressionOutput(BaseModel):
    """Output containing trained model and training statistics
    
    Attributes:
        model: Trained LinearRegression model
        model_info: Dictionary containing model coefficients and statistics
    """
    model: LinearRegression
    model_info: Dict = Field(
        default_factory=dict,
        description="Model coefficients and training statistics"
    )
    model_config = {"arbitrary_types_allowed": True}

class LinearRegressionNode:
    """Node for training a Linear Regression model
    
    This node implements linear regression using scikit-learn's LinearRegression.
    It provides model training, coefficient analysis, and basic model statistics.
    
    The linear regression model fits a linear equation to the data:
    y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    
    Where:
    - y is the target variable
    - β₀ is the intercept (if fit_intercept=True)
    - βᵢ are the coefficients
    - xᵢ are the features
    
    Example:
        ```python
        # Basic model training
        input_data = LinearRegressionInput(
            X_train=feature_df,
            y_train=target_series
        )
        
        # Training with custom parameters
        input_data = LinearRegressionInput(
            X_train=feature_df,
            y_train=target_series,
            fit_intercept=True,
            n_jobs=-1  # Use all CPU cores
        )
        
        result = LinearRegressionNode.execute(input_data)
        model = result.model
        print("Model coefficients:", result.model_info["coefficients"])
        print("R² score:", result.model_info["r2_score"])
        ```
    """
    
    @staticmethod
    def execute(inputs: LinearRegressionInput) -> LinearRegressionOutput:
        """Execute the model training process
        
        Args:
            inputs: LinearRegressionInput containing training data and parameters
            
        Returns:
            LinearRegressionOutput containing trained model and statistics
            
        Raises:
            ValueError: If input dimensions don't match or contain NaN values
            ValueError: If n_jobs is less than -1
        """
        # Input validation
        if inputs.X_train.shape[0] != len(inputs.y_train):
            raise ValueError("Number of samples in X_train and y_train must match")
        if inputs.X_train.isna().any().any() or inputs.y_train.isna().any():
            raise ValueError("Input data contains NaN values")
        
        # Initialize and train model
        model = LinearRegression(
            fit_intercept=inputs.fit_intercept
        )
        model.fit(inputs.X_train, inputs.y_train)
        
        # Collect model information
        model_info = {
            "coefficients": dict(zip(inputs.X_train.columns, model.coef_)),
            "intercept": model.intercept_ if inputs.fit_intercept else 0,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
            "n_features": inputs.X_train.shape[1],
            "n_samples": inputs.X_train.shape[0]
        }
        
        # Add feature importance based on absolute coefficient values
        coef_importance = np.abs(model.coef_)
        normalized_importance = coef_importance / np.sum(coef_importance)
        model_info["feature_importance"] = dict(
            zip(inputs.X_train.columns, normalized_importance)
        )
        
        return LinearRegressionOutput(model=model, model_info=model_info)
