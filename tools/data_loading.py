"""
Data Loading Tools
Tools for loading CSV, Excel, and JSON data files.
"""

import pandas as pd
from typing import Union
import io
from utils.data_state import DataState


def load_csv(file_path_or_buffer: Union[str, io.BytesIO], file_name: str = "data.csv") -> str:
    """
    Load CSV file into DataState.

    Args:
        file_path_or_buffer: Path to CSV file or file buffer
        file_name: Name of the file

    Returns:
        Success message with basic info
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
        state = DataState()
        state.load_data(df, str(file_path_or_buffer), file_name)

        return f"Successfully loaded {file_name}: {df.shape[0]} rows × {df.shape[1]} columns"
    except Exception as e:
        return f"Error loading CSV: {str(e)}"


def load_excel(file_path_or_buffer: Union[str, io.BytesIO], file_name: str = "data.xlsx") -> str:
    """
    Load Excel file into DataState.

    Args:
        file_path_or_buffer: Path to Excel file or file buffer
        file_name: Name of the file

    Returns:
        Success message with basic info
    """
    try:
        df = pd.read_excel(file_path_or_buffer)
        state = DataState()
        state.load_data(df, str(file_path_or_buffer), file_name)

        return f"Successfully loaded {file_name}: {df.shape[0]} rows × {df.shape[1]} columns"
    except Exception as e:
        return f"Error loading Excel: {str(e)}"


def load_json(file_path_or_buffer: Union[str, io.BytesIO], file_name: str = "data.json") -> str:
    """
    Load JSON file into DataState.

    Args:
        file_path_or_buffer: Path to JSON file or file buffer
        file_name: Name of the file

    Returns:
        Success message with basic info
    """
    try:
        df = pd.read_json(file_path_or_buffer)
        state = DataState()
        state.load_data(df, str(file_path_or_buffer), file_name)

        return f"Successfully loaded {file_name}: {df.shape[0]} rows × {df.shape[1]} columns"
    except Exception as e:
        return f"Error loading JSON: {str(e)}"
