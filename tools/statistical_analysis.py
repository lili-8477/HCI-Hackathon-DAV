"""
Statistical Analysis Tools
Tools for performing statistical analyses on the dataset.
"""

import pandas as pd
import numpy as np
from utils.data_state import DataState


def calculate_correlation(method: str = "pearson") -> str:
    """
    Calculate correlation matrix for numeric columns.

    Args:
        method: Correlation method - 'pearson', 'spearman', or 'kendall'

    Returns:
        Formatted correlation matrix
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return "No numeric columns found for correlation analysis."

        corr = numeric_df.corr(method=method)

        result = f"Correlation Matrix ({method}):\n"
        result += corr.to_string()
        return result
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"


def perform_groupby(group_column: str, agg_column: str, agg_func: str = "mean") -> str:
    """
    Perform group-by aggregation.

    Args:
        group_column: Column to group by
        agg_column: Column to aggregate
        agg_func: Aggregation function - 'mean', 'sum', 'count', 'min', 'max', 'std'

    Returns:
        Formatted aggregation results
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if group_column not in df.columns:
            return f"Column '{group_column}' not found."

        if agg_column not in df.columns:
            return f"Column '{agg_column}' not found."

        # Perform groupby
        if agg_func == "mean":
            result_df = df.groupby(group_column)[agg_column].mean().reset_index()
        elif agg_func == "sum":
            result_df = df.groupby(group_column)[agg_column].sum().reset_index()
        elif agg_func == "count":
            result_df = df.groupby(group_column)[agg_column].count().reset_index()
        elif agg_func == "min":
            result_df = df.groupby(group_column)[agg_column].min().reset_index()
        elif agg_func == "max":
            result_df = df.groupby(group_column)[agg_column].max().reset_index()
        elif agg_func == "std":
            result_df = df.groupby(group_column)[agg_column].std().reset_index()
        else:
            return f"Invalid aggregation function: {agg_func}"

        result_df.columns = [group_column, f"{agg_func}_{agg_column}"]

        result = f"Group-by Analysis: {agg_func} of {agg_column} by {group_column}\n"
        result += result_df.to_string(index=False)
        return result
    except Exception as e:
        return f"Error performing group-by: {str(e)}"


def detect_outliers(column_name: str, method: str = "iqr") -> str:
    """
    Detect outliers in a numeric column using IQR method.

    Args:
        column_name: Name of the column to check
        method: Detection method (currently only 'iqr' is supported)

    Returns:
        Summary of outliers detected
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found."

        col = df[column_name]

        if not pd.api.types.is_numeric_dtype(col):
            return f"Column '{column_name}' is not numeric."

        # IQR method
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = col[(col < lower_bound) | (col > upper_bound)]

        result = f"Outlier Detection for '{column_name}' (IQR Method):\n"
        result += f"Lower Bound: {lower_bound:.2f}\n"
        result += f"Upper Bound: {upper_bound:.2f}\n"
        result += f"Number of Outliers: {len(outliers)} ({len(outliers) / len(col) * 100:.2f}%)\n"

        if len(outliers) > 0:
            result += f"\nOutlier Values (first 10):\n"
            result += outliers.head(10).to_string()

        return result
    except Exception as e:
        return f"Error detecting outliers: {str(e)}"
