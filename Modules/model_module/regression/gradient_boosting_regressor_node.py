from pydantic import BaseModel, Field
from typing import Dict
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

class GradientBoostingRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    learning_rate: float = Field(default=0.1)
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=3)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class GradientBoostingRegressionOutput(BaseModel):
    model: GradientBoostingRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class GradientBoostingRegressionNode:

    @staticmethod
    def execute(inputs: GradientBoostingRegressionInput) -> GradientBoostingRegressionOutput:
        model = GradientBoostingRegressor(
            learning_rate=inputs.learning_rate,
            n_estimators=inputs.n_estimators,
            max_depth=inputs.max_depth,
            random_state=inputs.random_state
        )
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "learning_rate": inputs.learning_rate,
            "n_estimators": inputs.n_estimators,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return GradientBoostingRegressionOutput(model=model, model_info=info)
