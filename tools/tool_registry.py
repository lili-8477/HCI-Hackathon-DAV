"""
Tool Registry
Registers all tools and makes them available to the LangChain agent.
"""

from langchain_core.tools import Tool
from . import data_cleaning, data_analysis, data_visualization


def get_all_tools():
    """
    Get all tools for the manager agent.

    Returns:
        List of LangChain Tool objects
    """
    tools = [
        # Data Exploration Tools
        Tool(
            name="load_data",
            func=lambda file_path: data_cleaning.load_data(file_path),
            description="Load a dataset from CSV, Excel, or JSON file. Input: file path as string."
        ),
        Tool(
            name="preprocess_data",
            func=lambda df: data_cleaning.preprocess_data(df),
            description="Clean the dataset: remove duplicates and handle missing values interactively. Input: pandas DataFrame."
        ),

        # Data Analysis Tools
        Tool(
            name="summary_statistics",
            func=lambda df: data_analysis.summary_stats(df),
            description="Get descriptive statistics for all numeric columns in the dataset. Input: pandas DataFrame."
        ),
        Tool(
            name="compute_pca",
            func=lambda df: data_analysis.compute_pca(df),
            description="Compute PCA on numeric columns and return first two components. Input: pandas DataFrame."
        ),
        Tool(
            name="generate_ollama_summary",
            func=lambda df: data_analysis.generate_summary_with_ollama(df),
            description="Generate natural language summary of the dataset using Ollama. Input: pandas DataFrame."
        ),

        # Visualization Tools
        Tool(
            name="visualize_continuous",
            func=lambda df_col: data_visualization.visualize_continuous(df_col[0], df_col[1]),
            description="Create histogram, KDE, boxplot for continuous column. Input: [DataFrame, column_name]"
        ),
        Tool(
            name="visualize_categorical",
            func=lambda df_col: data_visualization.visualize_categorical(df_col[0], df_col[1]),
            description="Create count plot/bar chart for categorical column. Input: [DataFrame, column_name]"
        ),
        Tool(
            name="correlation_heatmap",
            func=lambda df: data_visualization.correlation_heatmap(df),
            description="Create a correlation heatmap for numeric columns. Input: pandas DataFrame."
        ),
        Tool(
            name="visualize_pca",
            func=lambda pca_df: data_visualization.visualize_pca(pca_df),
            description="Visualize PCA scatter plot for first two components. Input: pandas DataFrame returned from compute_pca."
        )
    ]

    return tools
