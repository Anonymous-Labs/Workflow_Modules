from pydantic import BaseModel, Field
from typing import Dict
from xgboost import XGBRegressor
import pandas as pd

class XGBoostRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    learning_rate: float = Field(default=0.1)
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=3)
    subsample: float = Field(default=1.0)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class XGBoostRegressionOutput(BaseModel):
    model: XGBRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class XGBoostRegressionNode:

    @staticmethod
    def execute(inputs: XGBoostRegressionInput) -> XGBoostRegressionOutput:
        model = XGBRegressor(
            learning_rate=inputs.learning_rate,
            n_estimators=inputs.n_estimators,
            max_depth=inputs.max_depth,
            subsample=inputs.subsample,
            random_state=inputs.random_state
        )
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "learning_rate": inputs.learning_rate,
            "n_estimators": inputs.n_estimators,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return XGBoostRegressionOutput(model=model, model_info=info)
