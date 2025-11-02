from pydantic import BaseModel, Field
from typing import Dict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class PolynomialRegressionInput(BaseModel):
    """Input parameters for Polynomial Regression
    
    Attributes:
        X_train: Training features DataFrame
        y_train: Training target Series
        degree: Degree of the polynomial features (default: 2)
        fit_intercept: Whether to calculate the intercept (default: True)
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    degree: int = Field(default=2, description="Degree of the polynomial")
    fit_intercept: bool = Field(default=True, description="Include intercept term")
    model_config = {"arbitrary_types_allowed": True}


class PolynomialRegressionOutput(BaseModel):
    """Output containing trained model and polynomial transformation info"""
    model: LinearRegression
    poly_features: PolynomialFeatures
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class PolynomialRegressionNode:
    """Node for training a Polynomial Regression model
    
    Uses PolynomialFeatures to transform the input features and fits a LinearRegression model.
    """

    @staticmethod
    def execute(inputs: PolynomialRegressionInput) -> PolynomialRegressionOutput:
        if inputs.X_train.shape[0] != len(inputs.y_train):
            raise ValueError("Number of samples in X_train and y_train must match")

        # Transform features
        poly = PolynomialFeatures(degree=inputs.degree, include_bias=False)
        X_poly = poly.fit_transform(inputs.X_train)

        # Fit model
        model = LinearRegression(fit_intercept=inputs.fit_intercept)
        model.fit(X_poly, inputs.y_train)

        # Gather info
        model_info = {
            "degree": inputs.degree,
            "coefficients_shape": model.coef_.shape,
            "r2_score": model.score(X_poly, inputs.y_train),
            "intercept": model.intercept_,
            "n_samples": inputs.X_train.shape[0],
            "n_original_features": inputs.X_train.shape[1],
            "n_poly_features": X_poly.shape[1]
        }

        return PolynomialRegressionOutput(
            model=model, poly_features=poly, model_info=model_info
        )
