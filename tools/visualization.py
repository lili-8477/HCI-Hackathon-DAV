"""
Visualization Tools
Tools for creating various plots and visualizations.
Figures are stored in DataState for Streamlit to display after the agent responds.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_state import DataState
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
import io
import base64


def _store_matplotlib_fig(fig):
    """Save matplotlib figure as PNG bytes and store in DataState"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    state = DataState()
    state.add_figure("png_bytes", buf.getvalue())


def _store_plotly_fig(fig):
    """Save plotly figure as PNG bytes and store in DataState"""
    img_bytes = fig.to_image(format="png", width=900, height=600, scale=2)
    state = DataState()
    state.add_figure("png_bytes", img_bytes)


def plot_distribution(column_name: str, plot_type: str = "histogram"):
    """Create a distribution plot for a numeric column."""
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return f"Column '{column_name}' is not numeric."

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "histogram":
            df[column_name].hist(bins=30, ax=ax, edgecolor='black')
            ax.set_ylabel("Frequency")
        elif plot_type == "kde":
            df[column_name].plot(kind='kde', ax=ax)
            ax.set_ylabel("Density")
        else:
            return "plot_type must be 'histogram' or 'kde'"

        ax.set_title(f"Distribution of {column_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column_name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        _store_matplotlib_fig(fig)
        return f"Distribution plot ({plot_type}) for '{column_name}' has been generated and is displayed below."
    except Exception as e:
        return f"Error creating distribution plot: {str(e)}"


def plot_correlation_heatmap():
    """Create a correlation heatmap for all numeric columns."""
    try:
        state = DataState()
        df = state.get_dataframe()

        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return "No numeric columns found for correlation heatmap."

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})

        ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()

        _store_matplotlib_fig(fig)

        # Build a text summary of notable correlations
        summary = "Correlation heatmap has been generated and is displayed below.\n\nNotable correlations:\n"
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    summary += f"  {corr.columns[i]} vs {corr.columns[j]}: {val:.2f}\n"

        return summary
    except Exception as e:
        return f"Error creating correlation heatmap: {str(e)}"


def plot_scatter(x_column: str, y_column: str, color_column: Optional[str] = None):
    """Create a scatter plot for two variables."""
    try:
        state = DataState()
        df = state.get_dataframe()

        if x_column not in df.columns:
            return f"Column '{x_column}' not found. Available columns: {', '.join(df.columns)}"
        if y_column not in df.columns:
            return f"Column '{y_column}' not found. Available columns: {', '.join(df.columns)}"
        if color_column and color_column not in df.columns:
            return f"Column '{color_column}' not found."

        fig = px.scatter(df, x=x_column, y=y_column, color=color_column,
                        title=f"{y_column} vs {x_column}",
                        labels={x_column: x_column, y_column: y_column})

        _store_plotly_fig(fig)
        color_info = f" colored by '{color_column}'" if color_column else ""
        return f"Scatter plot of '{y_column}' vs '{x_column}'{color_info} has been generated and is displayed below."
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


def plot_bar_chart(column_name: str, value_column: Optional[str] = None, agg_func: str = "count"):
    """Create a bar chart for categorical data."""
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

        fig, ax = plt.subplots(figsize=(10, 6))

        if value_column is None:
            df[column_name].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_ylabel("Count")
            title = f"Count by {column_name}"
        else:
            if value_column not in df.columns:
                return f"Column '{value_column}' not found."

            if agg_func == "count":
                data = df.groupby(column_name)[value_column].count()
            elif agg_func == "sum":
                data = df.groupby(column_name)[value_column].sum()
            elif agg_func == "mean":
                data = df.groupby(column_name)[value_column].mean()
            else:
                return "agg_func must be 'count', 'sum', or 'mean'"

            data.plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_ylabel(f"{agg_func.capitalize()} of {value_column}")
            title = f"{agg_func.capitalize()} of {value_column} by {column_name}"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column_name)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        _store_matplotlib_fig(fig)
        return f"Bar chart '{title}' has been generated and is displayed below."
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


def plot_time_series(date_column: str, value_column: str):
    """Create a time series line plot."""
    try:
        state = DataState()
        df = state.get_dataframe()

        if date_column not in df.columns:
            return f"Column '{date_column}' not found. Available columns: {', '.join(df.columns)}"
        if value_column not in df.columns:
            return f"Column '{value_column}' not found."

        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.sort_values(date_column)

        fig = px.line(df_copy, x=date_column, y=value_column,
                     title=f"{value_column} Over Time",
                     labels={date_column: "Date", value_column: value_column})

        _store_plotly_fig(fig)
        return f"Time series plot of '{value_column}' over '{date_column}' has been generated and is displayed below."
    except Exception as e:
        return f"Error creating time series plot: {str(e)}"


def plot_pie_chart(column_name: str, value_column: Optional[str] = None, agg_func: str = "sum"):
    """Create a pie chart for categorical data."""
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

        if value_column and value_column not in df.columns:
            return f"Column '{value_column}' not found."

        fig, ax = plt.subplots(figsize=(10, 8))

        if value_column is None:
            data = df[column_name].value_counts()
            title = f"Distribution of {column_name}"
        else:
            if agg_func == "sum":
                data = df.groupby(column_name)[value_column].sum()
            elif agg_func == "mean":
                data = df.groupby(column_name)[value_column].mean()
            elif agg_func == "count":
                data = df.groupby(column_name)[value_column].count()
            else:
                return "agg_func must be 'sum', 'mean', or 'count'"
            title = f"{agg_func.capitalize()} of {value_column} by {column_name}"

        ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        plt.tight_layout()

        _store_matplotlib_fig(fig)
        return f"Pie chart '{title}' has been generated and is displayed below."
    except Exception as e:
        return f"Error creating pie chart: {str(e)}"


def plot_box_plot(column_name: str, group_by: Optional[str] = None):
    """Create a box plot to show distribution and outliers."""
    try:
        state = DataState()
        df = state.get_dataframe()

        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return f"Column '{column_name}' is not numeric."

        fig, ax = plt.subplots(figsize=(10, 6))

        if group_by:
            if group_by not in df.columns:
                return f"Column '{group_by}' not found."
            df.boxplot(column=column_name, by=group_by, ax=ax)
            plt.suptitle('')
            ax.set_title(f"Box Plot of {column_name} by {group_by}", fontsize=14, fontweight='bold')
            ax.set_xlabel(group_by)
        else:
            df.boxplot(column=column_name, ax=ax)
            ax.set_title(f"Box Plot of {column_name}", fontsize=14, fontweight='bold')

        ax.set_ylabel(column_name)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        _store_matplotlib_fig(fig)
        group_info = f" grouped by '{group_by}'" if group_by else ""
        return f"Box plot for '{column_name}'{group_info} has been generated and is displayed below."
    except Exception as e:
        return f"Error creating box plot: {str(e)}"


def save_analysis(content: str, filename: str = "analysis_results.txt") -> str:
    """Save analysis findings to a text file."""
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
