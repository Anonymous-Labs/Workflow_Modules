from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd

class LinearRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    fit_intercept: bool = True
    normalize: bool = False
    model_config = {"arbitrary_types_allowed": True}

class LinearRegressionOutput(BaseModel):
    model: LinearRegression
    model_config = {"arbitrary_types_allowed": True}

class LinearRegressionNode:
    """Linear Regression model training node"""
    
    @staticmethod
    def execute(inputs: LinearRegressionInput) -> LinearRegressionOutput:
        model = LinearRegression(
            fit_intercept=inputs.fit_intercept,
            copy_X=True
        )
        model.fit(inputs.X_train, inputs.y_train)
        return LinearRegressionOutput(model=model)
