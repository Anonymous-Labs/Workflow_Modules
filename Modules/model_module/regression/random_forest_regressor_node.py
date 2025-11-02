from pydantic import BaseModel, Field
from typing import Dict
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class RandomForestRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=None)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class RandomForestRegressionOutput(BaseModel):
    model: RandomForestRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class RandomForestRegressionNode:

    @staticmethod
    def execute(inputs: RandomForestRegressionInput) -> RandomForestRegressionOutput:
        model = RandomForestRegressor(
            n_estimators=inputs.n_estimators,
            max_depth=inputs.max_depth,
            random_state=inputs.random_state
        )
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "n_estimators": inputs.n_estimators,
            "max_depth": inputs.max_depth,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return RandomForestRegressionOutput(model=model, model_info=info)
