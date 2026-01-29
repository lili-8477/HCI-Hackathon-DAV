"""
Visualization Tools
Tools for creating various plots and visualizations.
Returns matplotlib figure objects for Streamlit display.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_state import DataState
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go


def plot_distribution(column_name: str, plot_type: str = "histogram"):
    """
    Create a distribution plot (histogram or KDE) for a numeric column.

    Args:
        column_name: Name of the column to plot
        plot_type: 'histogram' or 'kde'

    Returns:
        matplotlib figure object
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not numeric.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "histogram":
            df[column_name].hist(bins=30, ax=ax, edgecolor='black')
            ax.set_ylabel("Frequency")
        elif plot_type == "kde":
            df[column_name].plot(kind='kde', ax=ax)
            ax.set_ylabel("Density")
        else:
            raise ValueError("plot_type must be 'histogram' or 'kde'")

        ax.set_title(f"Distribution of {column_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column_name)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    except Exception as e:
        # Return error as text
        return f"Error creating distribution plot: {str(e)}"


def plot_correlation_heatmap():
    """
    Create a correlation heatmap for all numeric columns.

    Returns:
        matplotlib figure object
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation heatmap.")

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})

        ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error creating correlation heatmap: {str(e)}"


def plot_scatter(x_column: str, y_column: str, color_column: Optional[str] = None):
    """
    Create a scatter plot for two variables.

    Args:
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column for color coding

    Returns:
        plotly figure object
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found.")

        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found.")

        if color_column and color_column not in df.columns:
            raise ValueError(f"Column '{color_column}' not found.")

        fig = px.scatter(df, x=x_column, y=y_column, color=color_column,
                        title=f"{y_column} vs {x_column}",
                        labels={x_column: x_column, y_column: y_column})

        return fig
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


def plot_bar_chart(column_name: str, value_column: Optional[str] = None, agg_func: str = "count"):
    """
    Create a bar chart for categorical data.

    Args:
        column_name: Categorical column name
        value_column: Optional numeric column to aggregate
        agg_func: Aggregation function - 'count', 'sum', 'mean'

    Returns:
        matplotlib figure object
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if value_column is None:
            # Simple count
            df[column_name].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_ylabel("Count")
            title = f"Count by {column_name}"
        else:
            if value_column not in df.columns:
                raise ValueError(f"Column '{value_column}' not found.")

            # Aggregate by function
            if agg_func == "count":
                data = df.groupby(column_name)[value_column].count()
            elif agg_func == "sum":
                data = df.groupby(column_name)[value_column].sum()
            elif agg_func == "mean":
                data = df.groupby(column_name)[value_column].mean()
            else:
                raise ValueError("agg_func must be 'count', 'sum', or 'mean'")

            data.plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_ylabel(f"{agg_func.capitalize()} of {value_column}")
            title = f"{agg_func.capitalize()} of {value_column} by {column_name}"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column_name)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


def plot_time_series(date_column: str, value_column: str):
    """
    Create a time series line plot.

    Args:
        date_column: Column containing dates
        value_column: Column containing values to plot

    Returns:
        plotly figure object
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found.")

        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found.")

        # Convert to datetime if not already
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.sort_values(date_column)

        fig = px.line(df_copy, x=date_column, y=value_column,
                     title=f"{value_column} Over Time",
                     labels={date_column: "Date", value_column: value_column})

        return fig
    except Exception as e:
        return f"Error creating time series plot: {str(e)}"


def plot_box_plot(column_name: str, group_by: Optional[str] = None):
    """
    Create a box plot to show distribution and outliers.

    Args:
        column_name: Numeric column to plot
        group_by: Optional categorical column to group by

    Returns:
        matplotlib figure object
    """
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found.")

        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not numeric.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if group_by:
            if group_by not in df.columns:
                raise ValueError(f"Column '{group_by}' not found.")

            df.boxplot(column=column_name, by=group_by, ax=ax)
            plt.suptitle('')  # Remove automatic title
            ax.set_title(f"Box Plot of {column_name} by {group_by}", fontsize=14, fontweight='bold')
            ax.set_xlabel(group_by)
        else:
            df.boxplot(column=column_name, ax=ax)
            ax.set_title(f"Box Plot of {column_name}", fontsize=14, fontweight='bold')

        ax.set_ylabel(column_name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error creating box plot: {str(e)}"


def save_analysis(content: str, filename: str = "analysis_results.txt") -> str:
    """
    Save analysis findings to a text file.

    Args:
        content: Content to save
        filename: Name of the output file

    Returns:
        Success message
    """
    try:
        from config import OUTPUT_DIR
        import os

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = OUTPUT_DIR / filename

        with open(output_path, 'w') as f:
            f.write(content)

        return f"Analysis saved to {output_path}"
    except Exception as e:
        return f"Error saving analysis: {str(e)}"
