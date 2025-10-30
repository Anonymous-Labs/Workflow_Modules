from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

class PredictInput(BaseModel):
    model: LinearRegression
    X: pd.DataFrame
    model_config = {"arbitrary_types_allowed": True}

class PredictOutput(BaseModel):
    predictions: pd.Series
    model_config = {"arbitrary_types_allowed": True}

class PredictNode:
    """Predict values using trained model"""
    
    @staticmethod
    def execute(inputs: PredictInput) -> PredictOutput:
        y_pred = inputs.model.predict(inputs.X)
        y_pred_series = pd.Series(y_pred, name="Predictions")
        return PredictOutput(predictions=y_pred_series)
