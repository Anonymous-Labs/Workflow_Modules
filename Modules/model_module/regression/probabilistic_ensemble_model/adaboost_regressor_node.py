from pydantic import BaseModel, Field
from typing import Dict
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

class AdaBoostRegressionInput(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    n_estimators: int = Field(default=50)
    learning_rate: float = Field(default=1.0)
    random_state: int = Field(default=42)
    model_config = {"arbitrary_types_allowed": True}


class AdaBoostRegressionOutput(BaseModel):
    model: AdaBoostRegressor
    model_info: Dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class AdaBoostRegressionNode:

    @staticmethod
    def execute(inputs: AdaBoostRegressionInput) -> AdaBoostRegressionOutput:
        model = AdaBoostRegressor(
            n_estimators=inputs.n_estimators,
            learning_rate=inputs.learning_rate,
            random_state=inputs.random_state
        )
        model.fit(inputs.X_train, inputs.y_train)

        info = {
            "n_estimators": inputs.n_estimators,
            "learning_rate": inputs.learning_rate,
            "r2_score": model.score(inputs.X_train, inputs.y_train),
        }

        return AdaBoostRegressionOutput(model=model, model_info=info)
