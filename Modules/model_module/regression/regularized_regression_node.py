from pydantic import BaseModel, Field
from typing import Dict, Literal
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import pandas as pd
import numpy as np

class RegularizedRegressionInput(BaseModel):
    """Input parameters for Ridge, Lasso, and Elastic Net Regression"""
    X_train: pd.DataFrame
    y_train: pd.Series
    regression_type: Literal["ridge", "lasso", "elasticnet"] = Field(
        default="ridge",
        description="Type of regression: 'ridge', 'lasso', or 'elasticnet'"
    )
    alpha: float = Field(default=1.0, description="Regularization strength")
    l1_ratio: float = Field(
        default=0.5, description="ElasticNet mixing ratio (ignored for ridge/lasso)"
    )
    fit_intercept: bool = Field(default=True)
    model_config = {"arbitrary_types_allowed": True}


class RegularizedRegressionOutput(BaseModel):
    """Output for Ridge/Lasso/ElasticNet model"""
    model: object
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class RegularizedRegressionNode:
    """Node for Ridge, Lasso, and ElasticNet Regression
    
    Supports switching between regression types easily.
    """

    @staticmethod
    def execute(inputs: RegularizedRegressionInput) -> RegularizedRegressionOutput:
        if inputs.X_train.shape[0] != len(inputs.y_train):
            raise ValueError("Number of samples in X_train and y_train must match")

        # Select model
        if inputs.regression_type == "ridge":
            model = Ridge(alpha=inputs.alpha, fit_intercept=inputs.fit_intercept)
        elif inputs.regression_type == "lasso":
            model = Lasso(alpha=inputs.alpha, fit_intercept=inputs.fit_intercept)
        else:
            model = ElasticNet(
                alpha=inputs.alpha,
                l1_ratio=inputs.l1_ratio,
                fit_intercept=inputs.fit_intercept
            )

        # Train model
        model.fit(inputs.X_train, inputs.y_train)

        # Info summary
        model_info = {
            "type": inputs.regression_type,
            "alpha": inputs.alpha,
            "l1_ratio": inputs.l1_ratio if inputs.regression_type == "elasticnet" else None,
            "coefficients": dict(zip(inputs.X_train.columns, model.coef_)),
            "intercept": model.intercept_,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
            "n_features": inputs.X_train.shape[1],
            "n_samples": inputs.X_train.shape[0],
        }

        return RegularizedRegressionOutput(model=model, model_info=model_info)
