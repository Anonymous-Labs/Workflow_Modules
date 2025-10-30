from pydantic import BaseModel
from sklearn.metrics import r2_score
import pandas as pd

class AccuracyInput(BaseModel):
    y_true: pd.Series
    y_pred: pd.Series
    model_config = {"arbitrary_types_allowed": True}

class AccuracyOutput(BaseModel):
    accuracy: float

class AccuracyNode:
    """Compute accuracy (R² Score)"""
    
    @staticmethod
    def execute(inputs: AccuracyInput) -> AccuracyOutput:
        acc = r2_score(inputs.y_true, inputs.y_pred)
        return AccuracyOutput(accuracy=acc * 100)
