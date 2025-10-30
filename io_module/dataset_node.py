from pydantic import BaseModel, FilePath
import pandas as pd

class DatasetNodeInput(BaseModel):
    file_path: FilePath
    model_config = {"arbitrary_types_allowed": True}     # Allow non-pydantic-native types like pandas.DataFrame

class DatasetNodeOutput(BaseModel):
    dataframe: pd.DataFrame
    model_config = {"arbitrary_types_allowed": True}

class DatasetNode:
    """Node to load CSV dataset"""
    
    @staticmethod
    def execute(inputs: DatasetNodeInput) -> DatasetNodeOutput:
        df = pd.read_csv(inputs.file_path)
        return DatasetNodeOutput(dataframe=df)
