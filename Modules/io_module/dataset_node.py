from pydantic import BaseModel, FilePath, Field
from typing import Optional, Dict
import pandas as pd

class DatasetNodeInput(BaseModel):
    """Input parameters for dataset loading
    
    Attributes:
        file_path: Path to the CSV file to load
        index_col: Column to use as DataFrame index (optional)
        encoding: File encoding (default: 'utf-8')
        separator: Column separator in CSV file (default: ',')
    """
    file_path: FilePath
    model_config = {"arbitrary_types_allowed": True}

class DatasetNodeOutput(BaseModel):
    """Output containing the loaded dataset
    
    Attributes:
        dataframe: Loaded pandas DataFrame
        info: Dictionary containing dataset metadata
    """
    dataframe: pd.DataFrame
    info: Dict = Field(
        default_factory=dict,
        description="Dataset metadata including shape, columns, and data types"
    )
    model_config = {"arbitrary_types_allowed": True}

class DatasetNode:
    """Node for loading and basic preprocessing of CSV datasets
    
    This node handles loading CSV files into pandas DataFrames and provides
    basic information about the loaded dataset. It supports various file
    encodings and custom separators.
    
    Example:
        ```python
        # Load a standard CSV file
        input_data = DatasetNodeInput(
            file_path="data/mydata.csv"
        )
        
        # Load with custom settings
        input_data = DatasetNodeInput(
            file_path="data/custom.csv",
            index_col="ID",
            separator=";",
            encoding="latin1"
        )
        
        result = DatasetNode.execute(input_data)
        df = result.dataframe
        print(f"Loaded {result.info['rows']} rows and {result.info['columns']} columns")
        ```
    """
    
    @staticmethod
    def execute(inputs: DatasetNodeInput) -> DatasetNodeOutput:
        """Execute the dataset loading operation
        
        Args:
            inputs: DatasetNodeInput containing file path and loading parameters
            
        Returns:
            DatasetNodeOutput containing the loaded DataFrame and dataset info
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If the file format is invalid
        """
        try:
            # Load the dataset (use pandas defaults for encoding/separator/index)
            df = pd.read_csv(inputs.file_path)
            
            # Collect dataset information
            info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isna().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            return DatasetNodeOutput(dataframe=df, info=info)
            
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {str(e)}") from e
