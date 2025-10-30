from pydantic import BaseModel, Field
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class TrainTestSplitInput(BaseModel):
    """Input parameters for train-test split
    
    Attributes:
        X: Feature DataFrame to split
        y: Target Series to split
        test_size: Proportion of dataset to include in test split (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        stratify: Whether to maintain ratio of samples for each class in target
        shuffle: Whether to shuffle data before splitting (default: True)
    """
    X: pd.DataFrame
    y: pd.Series
    test_size: float = Field(
        default=0.2,
        description="Proportion of dataset to include in test split",
        ge=0.0,
        le=1.0
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    stratify: Optional[pd.Series] = Field(
        default=None,
        description="Data to be used for stratified split"
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle data before splitting"
    )
    model_config = {"arbitrary_types_allowed": True}

class TrainTestSplitOutput(BaseModel):
    """Output containing split datasets and split information
    
    Attributes:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
        y_train: Training target Series
        y_test: Testing target Series
        split_info: Dictionary containing information about the split
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    split_info: Dict = Field(
        default_factory=dict,
        description="Information about the data split"
    )
    model_config = {"arbitrary_types_allowed": True}

class TrainTestSplitNode:
    """Node for splitting data into training and testing sets
    
    This node handles the division of data into training and testing sets
    with options for:
    - Stratified splitting (maintaining class ratios)
    - Random state control for reproducibility
    - Data shuffling
    - Split size configuration
    
    The node also provides statistics about the resulting split.
    
    Example:
        ```python
        # Basic split
        input_data = TrainTestSplitInput(
            X=features_df,
            y=target_series,
            test_size=0.2
        )
        
        # Stratified split for classification
        input_data = TrainTestSplitInput(
            X=features_df,
            y=target_series,
            test_size=0.2,
            stratify=target_series,  # For maintaining class ratios
            random_state=42
        )
        
        result = TrainTestSplitNode.execute(input_data)
        X_train, y_train = result.X_train, result.y_train
        X_test, y_test = result.X_test, result.y_test
        
        print("Split information:", result.split_info)
        ```
    """
    
    @staticmethod
    def calculate_split_statistics(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Calculate statistics about the train-test split
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            
        Returns:
            Dictionary containing split statistics
        """
        stats = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_ratio": len(X_train) / (len(X_train) + len(X_test)),
            "feature_names": X_train.columns.tolist(),
            "n_features": len(X_train.columns)
        }
        
        # Add target distribution statistics if categorical
        if y_train.dtype in ['object', 'category'] or y_train.nunique() < 10:
            stats["train_target_distribution"] = y_train.value_counts(normalize=True).to_dict()
            stats["test_target_distribution"] = y_test.value_counts(normalize=True).to_dict()
        
        # Add basic target statistics if numerical
        else:
            stats["train_target_stats"] = {
                "mean": float(y_train.mean()),
                "std": float(y_train.std()),
                "min": float(y_train.min()),
                "max": float(y_train.max())
            }
            stats["test_target_stats"] = {
                "mean": float(y_test.mean()),
                "std": float(y_test.std()),
                "min": float(y_test.min()),
                "max": float(y_test.max())
            }
        
        return stats
    
    @staticmethod
    def execute(inputs: TrainTestSplitInput) -> TrainTestSplitOutput:
        """Execute the train-test split
        
        Args:
            inputs: TrainTestSplitInput containing data and split parameters
            
        Returns:
            TrainTestSplitOutput containing split datasets and statistics
            
        Raises:
            ValueError: If test_size is not between 0 and 1
            ValueError: If input data contains NaN values
        """
        # Input validation
        if not 0 < inputs.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if inputs.X.isna().any().any() or inputs.y.isna().any():
            raise ValueError("Input data contains NaN values")
            
        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            inputs.X,
            inputs.y,
            test_size=inputs.test_size,
            random_state=inputs.random_state,
            stratify=inputs.stratify,
            shuffle=inputs.shuffle
        )
        
        # Calculate split statistics
        split_info = TrainTestSplitNode.calculate_split_statistics(
            X_train, X_test, y_train, y_test
        )
        
        # Add split parameters to info
        split_info["parameters"] = {
            "test_size": inputs.test_size,
            "random_state": inputs.random_state,
            "shuffle": inputs.shuffle,
            "stratified": inputs.stratify is not None
        }
        
        return TrainTestSplitOutput(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            split_info=split_info
        )
