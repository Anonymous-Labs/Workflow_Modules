from pydantic import BaseModel, Field
from typing import Dict
from lightgbm import LGBMRegressor
import pandas as pd

class LightGBMRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    learning_rate: float = Field(default=0.1)
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=-1)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class LightGBMRegressionOutput(BaseModel):
    model: LGBMRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class LightGBMRegressionNode:

    @staticmethod
    def execute(inputs: LightGBMRegressionInput) -> LightGBMRegressionOutput:
        model = LGBMRegressor(
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

        return LightGBMRegressionOutput(model=model, model_info=info)
