from pydantic import BaseModel, Field
from typing import Dict, Literal
import numpy as np
import pandas as pd

class GradientDescentRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    learning_rate: float = Field(default=0.01)
    n_iterations: int = Field(default=1000)
    descent_type: Literal["batch", "stochastic", "mini-batch"] = Field(default="batch")
    batch_size: int = Field(default=32)
    model_config = {"arbitrary_types_allowed": True}


class GradientDescentRegressionOutput(BaseModel):
    weights: np.ndarray
    bias: float
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class GradientDescentRegressionNode:

    @staticmethod
    def execute(inputs: GradientDescentRegressionInput) -> GradientDescentRegressionOutput:
        X = np.c_[np.ones(inputs.X_train.shape[0]), inputs.X_train.values]
        y = inputs.y_train.values.reshape(-1, 1)
        m, n = X.shape

        theta = np.zeros((n, 1))

        for i in range(inputs.n_iterations):
            if inputs.descent_type == "batch":
                gradients = 2/m * X.T.dot(X.dot(theta) - y)
            elif inputs.descent_type == "stochastic":
                idx = np.random.randint(m)
                Xi = X[idx:idx+1]
                yi = y[idx:idx+1]
                gradients = 2 * Xi.T.dot(Xi.dot(theta) - yi)
            else:  # mini-batch
                idx = np.random.randint(0, m, inputs.batch_size)
                Xb = X[idx]
                yb = y[idx]
                gradients = 2/inputs.batch_size * Xb.T.dot(Xb.dot(theta) - yb)
            theta -= inputs.learning_rate * gradients

        weights, bias = theta[1:].flatten(), theta[0][0]
        info = {
            "descent_type": inputs.descent_type,
            "iterations": inputs.n_iterations,
            "learning_rate": inputs.learning_rate,
        }
        return GradientDescentRegressionOutput(weights=weights, bias=bias, model_info=info)
