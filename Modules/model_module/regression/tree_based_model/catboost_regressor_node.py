from pydantic import BaseModel, Field
from typing import Dict
from catboost import CatBoostRegressor
import pandas as pd

class CatBoostRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    learning_rate: float = Field(default=0.1)
    iterations: int = Field(default=100)
    depth: int = Field(default=6)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class CatBoostRegressionOutput(BaseModel):
    model: CatBoostRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class CatBoostRegressionNode:

    @staticmethod
    def execute(inputs: CatBoostRegressionInput) -> CatBoostRegressionOutput:
        model = CatBoostRegressor(
            learning_rate=inputs.learning_rate,
            iterations=inputs.iterations,
            depth=inputs.depth,
            random_state=inputs.random_state,
            verbose=False
        )
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "learning_rate": inputs.learning_rate,
            "iterations": inputs.iterations,
            "depth": inputs.depth,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return CatBoostRegressionOutput(model=model, model_info=info)
