from pydantic import BaseModel, Field
from typing import Dict
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

class DecisionTreeRegressionInput(BaseModel):
    """Input parameters for Decision Tree Regression"""
    X_train: pd.DataFrame
    y_train: pd.Series
    max_depth: int = Field(default=None, description="Maximum tree depth")
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class DecisionTreeRegressionOutput(BaseModel):
    model: DecisionTreeRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class DecisionTreeRegressionNode:

    @staticmethod
    def execute(inputs: DecisionTreeRegressionInput) -> DecisionTreeRegressionOutput:
        model = DecisionTreeRegressor(max_depth=inputs.max_depth, random_state=inputs.random_state)
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "max_depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return DecisionTreeRegressionOutput(model=model, model_info=info)
