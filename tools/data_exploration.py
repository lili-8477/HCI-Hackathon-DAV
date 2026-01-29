"""
Data Exploration Tools
Tools for exploring and understanding the loaded dataset.
"""

import pandas as pd
from utils.data_state import DataState


def get_data_overview() -> str:
    """
    Get overview of the dataset: shape, columns, data types.

    Returns:
        Formatted string with dataset overview
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        overview = []
        overview.append(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        overview.append(f"\nColumns ({len(df.columns)}):")
        overview.append(", ".join(df.columns.tolist()))
        overview.append(f"\nData Types:")
        overview.append(df.dtypes.to_string())

        return "\n".join(overview)
    except Exception as e:
        return f"Error getting data overview: {str(e)}"


def check_missing_values() -> str:
    """
    Check for missing values in the dataset.

    Returns:
        Formatted string with missing value counts and percentages
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100

        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct.round(2)
        })

        # Filter to only columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if len(missing_df) == 0:
            return "No missing values found in the dataset."

        result = "Missing Values:\n"
        result += missing_df.to_string()
        return result
    except Exception as e:
        return f"Error checking missing values: {str(e)}"


def get_summary_statistics() -> str:
    """
    Get summary statistics for all numeric columns.

    Returns:
        Formatted string with descriptive statistics
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        # Get statistics for numeric columns
        stats = df.describe()

        if stats.empty:
            return "No numeric columns found in the dataset."

        result = "Summary Statistics (Numeric Columns):\n"
        result += stats.to_string()
        return result
    except Exception as e:
        return f"Error getting summary statistics: {str(e)}"


def get_column_info(column_name: str) -> str:
    """
    Get detailed information about a specific column.

    Args:
        column_name: Name of the column to analyze

    Returns:
        Formatted string with column details
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

        col = df[column_name]
        info = []
        info.append(f"Column: {column_name}")
        info.append(f"Data Type: {col.dtype}")
        info.append(f"Non-Null Count: {col.count()} / {len(col)}")
        info.append(f"Missing Values: {col.isnull().sum()} ({(col.isnull().sum() / len(col) * 100):.2f}%)")

        if pd.api.types.is_numeric_dtype(col):
            info.append(f"\nNumeric Statistics:")
            info.append(f"  Mean: {col.mean():.2f}")
            info.append(f"  Median: {col.median():.2f}")
            info.append(f"  Std Dev: {col.std():.2f}")
            info.append(f"  Min: {col.min()}")
            info.append(f"  Max: {col.max()}")
        else:
            info.append(f"\nUnique Values: {col.nunique()}")
            info.append(f"Most Common:")
            value_counts = col.value_counts().head(5)
            for val, count in value_counts.items():
                info.append(f"  {val}: {count}")

        return "\n".join(info)
    except Exception as e:
        return f"Error getting column info: {str(e)}"


def list_columns() -> str:
    """
    List all columns in the dataset.

    Returns:
        Comma-separated list of column names
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        return f"Columns ({len(df.columns)}): " + ", ".join(df.columns.tolist())
    except Exception as e:
        return f"Error listing columns: {str(e)}"
