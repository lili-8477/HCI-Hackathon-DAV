"""
Tool Registry
Registers all cleaning, analysis, and visualization tools for the LangChain agent.
"""

from langchain_core.tools import Tool
from . import data_cleaning as dc
from . import data_analysis as da
from . import data_visualization as dv

# --- Wrapper to show function and args before execution ---
def confirm(func, *args, **kwargs):
    """Show planned function call and ask for confirmation."""
    return dv.confirm_and_run(func, *args, **kwargs)

# --- Tool Registry ---
def get_all_tools():
    tools = [
        # Data Cleaning Tools
        Tool(
            name="load_data",
            func=lambda path: confirm(dc.load_data, path),
            description="Load dataset from CSV, Excel, or JSON. Input: file path."
        ),
        Tool(
            name="preprocess_data",
            func=lambda df: confirm(dc.preprocess_data, df),
            description="Clean dataset: remove duplicates, handle missing values interactively. Input: pandas DataFrame."
        ),
        Tool(
            name="select_columns",
            func=lambda df_coltype: confirm(dc.select_columns, df_coltype[0], col_type=df_coltype[1] if len(df_coltype) > 1 else "numeric"),
            description="Select columns by type. Input: [DataFrame, 'numeric'/'categorical']"
        ),

        # Data Analysis Tools
        Tool(
            name="summary_statistics",
            func=lambda df_cols=None: confirm(da.summary_stats, df_cols[0], columns=df_cols[1] if len(df_cols) > 1 else None),
            description="Return summary statistics for numeric columns. Input: [DataFrame, optional list of columns]"
        ),
        Tool(
            name="correlation_matrix",
            func=lambda df_method=None: confirm(da.compute_correlation, df_method[0], method=df_method[1] if len(df_method) > 1 else "pearson"),
            description="Compute correlation matrix. Input: [DataFrame, method='pearson/spearman/kendall']"
        ),
        Tool(
            name="compute_pca",
            func=lambda df_ncomp=None: confirm(da.compute_pca, df_ncomp[0], n_components=df_ncomp[1] if len(df_ncomp) > 1 else 2),
            description="Compute PCA. Input: [DataFrame, n_components=2]"
        ),

        # Visualization Tools
        Tool(
            name="plot_histogram",
            func=lambda df_col_opts: confirm(dv.plot_histogram,
                                            df_col_opts[0],
                                            df_col_opts[1],
                                            color=df_col_opts[2] if len(df_col_opts) > 2 else "skyblue",
                                            ablines=df_col_opts[3] if len(df_col_opts) > 3 else None,
                                            save=df_col_opts[4] if len(df_col_opts) > 4 else False),
            description="Histogram/KDE of a continuous column. Input: [DataFrame, column_name, color (optional), ablines list (optional), save (optional)]"
        ),
        Tool(
            name="plot_boxplot",
            func=lambda df_col_opts: confirm(dv.plot_boxplot,
                                            df_col_opts[0],
                                            df_col_opts[1],
                                            color=df_col_opts[2] if len(df_col_opts) > 2 else "skyblue",
                                            ablines=df_col_opts[3] if len(df_col_opts) > 3 else None,
                                            save=df_col_opts[4] if len(df_col_opts) > 4 else False),
            description="Boxplot of a continuous column. Input: [DataFrame, column_name, color (optional), ablines list (optional), save (optional)]"
        ),
        Tool(
            name="plot_scatter",
            func=lambda df_cols_opts: confirm(dv.plot_scatter,
                                             df_cols_opts[0],
                                             df_cols_opts[1],
                                             df_cols_opts[2] if len(df_cols_opts) > 2 else None,
                                             save=df_cols_opts[3] if len(df_cols_opts) > 3 else False),
            description="Scatter plot. Input: [DataFrame, x_col, y_col, optional color_col, save (optional)]"
        ),
        Tool(
            name="plot_bar",
            func=lambda df_cols_opts: confirm(dv.plot_bar,
                                             df_cols_opts[0],
                                             df_cols_opts[1] if len(df_cols_opts) > 1 else None,
                                             agg_func=df_cols_opts[2] if len(df_cols_opts) > 2 else "count",
                                             color=df_cols_opts[3] if len(df_cols_opts) > 3 else "orange",
                                             save=df_cols_opts[4] if len(df_cols_opts) > 4 else False),
            description="Bar plot for categorical column. Input: [DataFrame, col, value_col (optional), agg_func (optional), color (optional), save (optional)]"
        ),
        Tool(
            name="correlation_heatmap",
            func=lambda df_save=None: confirm(dv.correlation_heatmap,
                                              df_save[0],
                                              save=df_save[1] if len(df_save) > 1 else False),
            description="Generate correlation heatmap. Input: [DataFrame, save (optional)]"
        ),
        Tool(
            name="plot_pca",
            func=lambda df_save=None: confirm(dv.plot_pca,
                                              df_save[0],
                                              n_components=df_save[1] if len(df_save) > 1 else 2,
                                              save=df_save[2] if len(df_save) > 2 else False),
            description="Generate PCA scatter plot. Input: [DataFrame, n_components (optional), save (optional)]"
        ),
    ]
    return tools
