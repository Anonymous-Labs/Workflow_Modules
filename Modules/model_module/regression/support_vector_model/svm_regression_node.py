from pydantic import BaseModel, Field
from typing import Dict
from sklearn.svm import SVR
import pandas as pd
import numpy as np

class SVMRegressionInput(BaseModel):
    """Input parameters for Support Vector Regression"""
    X_train: pd.DataFrame
    y_train: pd.Series
    kernel: str = Field(default="rbf", description="Kernel type: linear, poly, rbf, sigmoid")
    C: float = Field(default=1.0, description="Regularization parameter")
    epsilon: float = Field(default=0.1, description="Epsilon-tube within which no penalty is associated")
    model_config = {"arbitrary_types_allowed": True}


class SVMRegressionOutput(BaseModel):
    model: SVR
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class SVMRegressionNode:
    """Node for Support Vector Regression"""

    @staticmethod
    def execute(inputs: SVMRegressionInput) -> SVMRegressionOutput:
        if inputs.X_train.shape[0] != len(inputs.y_train):
            raise ValueError("Mismatched input lengths")

        model = SVR(kernel=inputs.kernel, C=inputs.C, epsilon=inputs.epsilon)
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "kernel": inputs.kernel,
            "C": inputs.C,
            "epsilon": inputs.epsilon,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
            "n_samples": inputs.X_train.shape[0],
        }

        return SVMRegressionOutput(model=model, model_info=info)
