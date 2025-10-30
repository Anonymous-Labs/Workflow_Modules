"""
Example usage of the Workflow Modules using the California Housing dataset
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.preprocessing_module.feature_selection_node import (
    FeatureSelectionNode,
    FeatureSelectionInput
)
from Modules.preprocessing_module.train_test_split_node import (
    TrainTestSplitNode,
    TrainTestSplitInput
)
from Modules.model_module.linear_regression_node import (
    LinearRegressionNode,
    LinearRegressionInput
)
from Modules.evaluation_module.predict_node import (
    PredictNode,
    PredictInput
)
from Modules.evaluation_module.accuracy_node import (
    AccuracyNode,
    AccuracyInput
)
from constants.constants import MetricType
def main():
    print("Loading California Housing dataset...")
    # Load dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    df['target'] = housing.target
    
    print("\nDataset shape:", df.shape)
    print("Features:", housing.feature_names)
    print("\nSample of the dataset:")
    print(df.head())
    
    # 1. Feature Selection
    print("\n1. Selecting features...")
    feature_selection_input = FeatureSelectionInput(
        dataframe=df,
        target_column='target',
        feature_columns=housing.feature_names,
        validate_features=True
    )
    feature_selection_output = FeatureSelectionNode.execute(feature_selection_input)
    
    print("Feature statistics:")
    for key, value in feature_selection_output.feature_stats.items():
        print(f"{key}: {value}")
    
    # 2. Train-Test Split
    print("\n2. Splitting data into train and test sets...")
    split_input = TrainTestSplitInput(
        X=feature_selection_output.X,
        y=feature_selection_output.y,
        test_size=0.2,
        random_state=42
    )
    split_output = TrainTestSplitNode.execute(split_input)
    
    print("\nSplit information:")
    for key, value in split_output.split_info.items():
        if key != "feature_names":  # Skip printing long feature names list
            print(f"{key}: {value}")
    
    # 3. Train Linear Regression Model
    print("\n3. Training Linear Regression model...")
    model_input = LinearRegressionInput(
        X_train=split_output.X_train,
        y_train=split_output.y_train,
        fit_intercept=True
    )
    model_output = LinearRegressionNode.execute(model_input)
    
    print("\nModel information:")
    print("Coefficients:")
    for feature, coef in model_output.model_info['coefficients'].items():
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {model_output.model_info['intercept']:.4f}")
    print(f"R² Score (training): {model_output.model_info['r2_score']:.4f}")
    
    # 4. Make Predictions
    print("\n4. Making predictions on test set...")
    predict_input = PredictInput(
        model=model_output.model,
        X=split_output.X_test,
        prediction_intervals=True,
        confidence_level=0.95
    )
    predict_output = PredictNode.execute(predict_input)
    
    print("\nPrediction statistics:")
    for key, value in predict_output.prediction_stats.items():
        print(f"{key}: {value}")
    
    # 5. Evaluate Model
    print("\n5. Evaluating model performance...")
    accuracy_input = AccuracyInput(
        y_true=split_output.y_test,
        y_pred=predict_output.predictions,
        metrics=[MetricType.ALL]
    )
    accuracy_output = AccuracyNode.execute(accuracy_input)
    
    print("\nModel evaluation metrics:")
    print(f"R² Score: {accuracy_output.r2_score:.4f}")
    print(f"Mean Squared Error: {accuracy_output.mse:.4f}")
    print(f"Root Mean Squared Error: {accuracy_output.rmse:.4f}")
    print(f"Mean Absolute Error: {accuracy_output.mae:.4f}")
    
    # 6. Display sample predictions
    print("\n6. Sample predictions vs actual values:")
    sample_size = 5
    sample_indices = np.random.choice(len(split_output.y_test), sample_size, replace=False)
    
    print("\nSample predictions:")
    print("Actual Value | Predicted Value | Difference")
    print("-" * 45)
    for idx in sample_indices:
        actual = split_output.y_test.iloc[idx]
        predicted = predict_output.predictions.iloc[idx]
        diff = actual - predicted
        print(f"{actual:11.2f} | {predicted:14.2f} | {diff:10.2f}")

if __name__ == "__main__":
    main()