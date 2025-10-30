from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class FeatureSelectionInput(BaseModel):
    """Input parameters for feature selection
    
    Attributes:
        dataframe: Input DataFrame containing all features and target
        target_column: Name of the target variable column
        feature_columns: List of feature column names to select
        validate_features: Whether to validate features for missing values and constant columns
    """
    dataframe: pd.DataFrame
    target_column: str
    feature_columns: List[str]
    validate_features: bool = Field(
        default=True,
        description="Validate features for missing values and constant columns"
    )
    model_config = {"arbitrary_types_allowed": True}

class FeatureSelectionOutput(BaseModel):
    """Output containing selected features and target
    
    Attributes:
        X: DataFrame containing selected feature columns
        y: Series containing target values
        feature_stats: Dictionary containing statistics about selected features
    """
    X: pd.DataFrame
    y: pd.Series
    feature_stats: Optional[Dict] = Field(
        default_factory=dict,
        description="Statistics about selected features"
    )
    model_config = {"arbitrary_types_allowed": True}

class FeatureSelectionNode:
    """Node for selecting and validating features for model training
    
    This node handles feature selection from the input DataFrame and performs
    basic validation of the selected features. It can detect and warn about:
    - Missing values in features
    - Constant or near-constant features
    - High correlation between features
    - Invalid feature names
    
    Example:
        ```python
        # Basic feature selection
        input_data = FeatureSelectionInput(
            dataframe=df,
            target_column="price",
            feature_columns=["size", "bedrooms", "location"]
        )
        
        # Feature selection with validation
        input_data = FeatureSelectionInput(
            dataframe=df,
            target_column="price",
            feature_columns=["size", "bedrooms", "location"],
            validate_features=True
        )
        
        result = FeatureSelectionNode.execute(input_data)
        X, y = result.X, result.y
        print("Feature statistics:", result.feature_stats)
        ```
    """
    
    @staticmethod
    def validate_features(df: pd.DataFrame) -> Dict:
        """Validate selected features and compute statistics
        
        Args:
            df: DataFrame containing features to validate
            
        Returns:
            Dictionary containing feature statistics and validation results
        """
        stats = {
            "n_features": len(df.columns),
            "missing_values": df.isna().sum().to_dict(),
            "constant_features": [],
            "high_correlation_pairs": []
        }
        
        # Check for constant features
        for col in df.columns:
            if df[col].nunique() == 1:
                stats["constant_features"].append(col)
                
        # Check for high correlation (if numeric features exist)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.95)
            high_corr_pairs = [(numeric_df.columns[i], numeric_df.columns[j])
                             for i, j in zip(*high_corr)
                             if i != j and i < j]
            stats["high_correlation_pairs"] = high_corr_pairs
            
        return stats
    
    @staticmethod
    def execute(inputs: FeatureSelectionInput) -> FeatureSelectionOutput:
        """Execute the feature selection process
        
        Args:
            inputs: FeatureSelectionInput containing DataFrame and selection criteria
            
        Returns:
            FeatureSelectionOutput containing selected features and target
            
        Raises:
            ValueError: If target or feature columns don't exist in DataFrame
            ValueError: If all selected features are invalid
        """
        # Validate column existence
        all_required_cols = set([inputs.target_column] + inputs.feature_columns)
        missing_cols = all_required_cols - set(inputs.dataframe.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        # Select features and target
        X = inputs.dataframe[inputs.feature_columns]
        y = inputs.dataframe[inputs.target_column]
        
        # Compute feature statistics if validation is requested
        feature_stats = {}
        if inputs.validate_features:
            feature_stats = FeatureSelectionNode.validate_features(X)
            
            # Warn about potential issues
            if feature_stats["constant_features"]:
                print(f"Warning: Found constant features: {feature_stats['constant_features']}")
            if feature_stats["high_correlation_pairs"]:
                print(f"Warning: Found highly correlated feature pairs: {feature_stats['high_correlation_pairs']}")
        
        return FeatureSelectionOutput(X=X, y=y, feature_stats=feature_stats)
