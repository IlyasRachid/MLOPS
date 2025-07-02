import os
import sys

from src.exception import CustomException
import dill
import pandas as pd

from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def handle_missing_values(df: pd.DataFrame, target_col: str, text_cols: list = None, fill_value: str = "", fill_target: str = "unknown") -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Parameters:
    - df: pd.DataFrame - The input DataFrame.
    - target_col: str - The target column to fill with a specific value.
    - text_cols: list - List of text columns to fill with a specific value.
    - fill_value: str - Value to fill in text columns.
    - fill_target: str - Value to fill in the target column.
    
    Returns:
    - pd.DataFrame - DataFrame with missing values handled.
    """
    df = df.copy()

    # drop rows with missing values in the target column
    df = df.dropna(subset=[target_col], axis=0)

    # fill text columns with the specified fill value
    if text_cols:
        for col in text_cols:
            df[col] = df[col].fillna(fill_value)

    # fill the target column with the specified fill value
    df[target_col] = df[target_col].fillna(fill_target)

    return df.reset_index(drop=True)
