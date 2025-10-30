from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
import pandas as pd

class TrainTestSplitInput(BaseModel):
    X: pd.DataFrame
    y: pd.Series
    test_size: float = Field(default=0.2)
    model_config = {"arbitrary_types_allowed": True}

class TrainTestSplitOutput(BaseModel):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model_config = {"arbitrary_types_allowed": True}

class TrainTestSplitNode:
    """Split data into train/test"""
    
    @staticmethod
    def execute(inputs: TrainTestSplitInput) -> TrainTestSplitOutput:
        X_train, X_test, y_train, y_test = train_test_split(
            inputs.X, inputs.y, test_size=inputs.test_size, random_state=42
        )
        return TrainTestSplitOutput(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
