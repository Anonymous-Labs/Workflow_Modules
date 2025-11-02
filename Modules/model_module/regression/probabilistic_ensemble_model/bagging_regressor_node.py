from pydantic import BaseModel, Field
from typing import Dict
from sklearn.ensemble import BaggingRegressor
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class BaggingRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    n_estimators: int = Field(default=10)
    base_estimator_depth: int = Field(default=None)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class BaggingRegressionOutput(BaseModel):
    model: BaggingRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class BaggingRegressionNode:

    @staticmethod
    def execute(inputs: BaggingRegressionInput) -> BaggingRegressionOutput:
        base_estimator = DecisionTreeRegressor(max_depth=inputs.base_estimator_depth)
        model = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=inputs.n_estimators,
            random_state=inputs.random_state
        )
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "n_estimators": inputs.n_estimators,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return BaggingRegressionOutput(model=model, model_info=info)
