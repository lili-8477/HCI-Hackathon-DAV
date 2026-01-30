"""
Data Manipulation Tools
Tools for filtering, selecting, pivoting, and transforming data.
Results are saved as named intermediate tables in DataState.
"""

import numpy as np
import pandas as pd
from utils.data_state import DataState


def _preview(name: str, df: pd.DataFrame) -> str:
    """Return a standard preview string for a saved intermediate table."""
    preview = df.head(5).to_string()
    return (
        f"Saved as table '{name}' ({df.shape[0]} rows × {df.shape[1]} columns)\n\n"
        f"Preview (first 5 rows):\n{preview}\n\n"
        f"This is now the active dataset. Use 'use_table' to switch back to another table."
    )


def filter_rows(expression: str) -> str:
    """Filter rows using a pandas query expression."""
    try:
        state = DataState()
        df = state.get_dataframe()
        result = df.query(expression)
        name = state.next_table_name("filtered")
        state.save_intermediate(name, result)
        state.df = result
        return _preview(name, result)
    except Exception as e:
        return f"Error filtering rows: {e}"


def select_columns(columns: str) -> str:
    """Select specific columns. Input: comma-separated column names."""
    try:
        state = DataState()
        df = state.get_dataframe()
        cols = [c.strip() for c in columns.split(",")]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}. Available: {list(df.columns)}"
        result = df[cols]
        name = state.next_table_name("selected")
        state.save_intermediate(name, result)
        state.df = result
        return _preview(name, result)
    except Exception as e:
        return f"Error selecting columns: {e}"


def pivot_table(input_str: str) -> str:
    """Create a pivot table. Input: 'index,values,agg_func'."""
    try:
        state = DataState()
        df = state.get_dataframe()
        parts = [p.strip() for p in input_str.split(",")]
        if len(parts) != 3:
            return "Invalid input. Expected: 'index,values,agg_func' (e.g. 'region,revenue,sum')"
        index, values, agg_func = parts
        result = pd.pivot_table(df, index=index, values=values, aggfunc=agg_func)
        result = result.reset_index()
        name = state.next_table_name("pivot")
        state.save_intermediate(name, result)
        state.df = result
        return _preview(name, result)
    except Exception as e:
        return f"Error creating pivot table: {e}"


def transform_column(input_str: str) -> str:
    """Transform a column. Input: 'column,operation' where operation is log, normalize, round:N, upper, or lower."""
    try:
        state = DataState()
        df = state.get_dataframe()
        parts = [p.strip() for p in input_str.split(",")]
        if len(parts) != 2:
            return "Invalid input. Expected: 'column,operation' (e.g. 'price,log')"
        col, operation = parts
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"

        result = df.copy()
        if operation == "log":
            result[col] = np.log(result[col])
        elif operation == "normalize":
            min_val, max_val = result[col].min(), result[col].max()
            result[col] = (result[col] - min_val) / (max_val - min_val)
        elif operation.startswith("round:"):
            decimals = int(operation.split(":")[1])
            result[col] = result[col].round(decimals)
        elif operation == "upper":
            result[col] = result[col].astype(str).str.upper()
        elif operation == "lower":
            result[col] = result[col].astype(str).str.lower()
        else:
            return f"Unknown operation '{operation}'. Supported: log, normalize, round:N, upper, lower"

        name = state.next_table_name("transformed")
        state.save_intermediate(name, result)
        state.df = result
        return _preview(name, result)
    except Exception as e:
        return f"Error transforming column: {e}"


def use_table(name: str) -> str:
    """Set a named intermediate table as the active dataset."""
    try:
        state = DataState()
        name = name.strip()
        state.use_intermediate(name)
        df = state.get_dataframe()
        return f"Active dataset is now '{name}' ({df.shape[0]} rows × {df.shape[1]} columns)."
    except Exception as e:
        return f"Error: {e}"


def list_tables(_: str = "") -> str:
    """List all intermediate tables."""
    state = DataState()
    tables = state.list_intermediates()
    if not tables:
        return "No intermediate tables saved yet."
    lines = ["Intermediate tables:"]
    for t in tables:
        lines.append(f"  - {t['name']}: {t['rows']} rows × {t['columns']} columns")
    return "\n".join(lines)
