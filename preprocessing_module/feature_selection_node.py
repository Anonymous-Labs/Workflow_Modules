from pydantic import BaseModel
import pandas as pd

class FeatureSelectionInput(BaseModel):
    dataframe: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    model_config = {"arbitrary_types_allowed": True}

class FeatureSelectionOutput(BaseModel):
    X: pd.DataFrame
    y: pd.Series
    model_config = {"arbitrary_types_allowed": True}

class FeatureSelectionNode:
    """Select features and target column"""
    
    @staticmethod
    def execute(inputs: FeatureSelectionInput) -> FeatureSelectionOutput:
        X = inputs.dataframe[inputs.feature_columns]
        y = inputs.dataframe[inputs.target_column]
        return FeatureSelectionOutput(X=X, y=y)
