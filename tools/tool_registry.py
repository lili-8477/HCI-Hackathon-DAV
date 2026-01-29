"""
Tool Registry
Registers all cleaning, analysis, and visualization tools for the LangChain agent.
"""

from langchain_core.tools import Tool
from . import data_cleaning as dc
from . import data_analysis as da
from . import data_visualization as dv

def confirm(func, *args, **kwargs):
    return dv.confirm_and_run(func, *args, **kwargs)

def get_all_tools():
    tools = [
        Tool(
            name="load_data",
            func=lambda path: confirm(dc.load_data, path),
            description="Load dataset from CSV, Excel, or JSON. Input: file path."
        ),
        Tool(
            name="preprocess_data",
            func=lambda df: confirm(dc.preprocess_data, df),
            description="Remove duplicates and handle missing values interactively. Input: DataFrame."
        ),
        Tool(
            name="select_columns",
            func=lambda df_coltype: confirm(dc.select_columns, df_coltype[0], col_type=df_coltype[1] if len(df_coltype)>1 else "numeric"),
            description="Select columns by type. Input: [DataFrame, 'numeric'/'categorical']"
        ),
        Tool(
            name="summary_statistics",
            func=lambda df_cols=None: confirm(da.summary_stats, df_cols[0], columns=df_cols[1] if len(df_cols)>1 else None),
            description="Return summary statistics. Input: [DataFrame, optional columns]"
        ),
        Tool(
            name="correlation_matrix",
            func=lambda df_method=None: confirm(da.compute_correlation, df_method[0], method=df_method[1] if len(df_method)>1 else "pearson"),
            description="Correlation matrix. Input: [DataFrame, method]"
        ),
        Tool(
            name="compute_pca",
            func=lambda df_ncomp=None: confirm(da.compute_pca, df_ncomp[0], n_components=df_ncomp[1] if len(df_ncomp)>1 else 2),
            description="Compute PCA. Input: [DataFrame, n_components]"
        ),
        Tool(
            name="plot_histogram",
            func=lambda args: confirm(dv.plot_histogram, args[0], args[1], color=args[2] if len(args)>2 else "skyblue", ablines=args[3] if len(args)>3 else None, save=args[4] if len(args)>4 else False),
            description="Histogram/KDE plot. Input: [DataFrame, column, color, ablines, save]"
        ),
        Tool(
            name="plot_boxplot",
            func=lambda args: confirm(dv.plot_boxplot, args[0], args[1], color=args[2] if len(args)>2 else "skyblue", ablines=args[3] if len(args)>3 else None, save=args[4] if len(args)>4 else False),
            description="Boxplot. Input: [DataFrame, column, color, ablines, save]"
        ),
        Tool(
            name="plot_scatter",
            func=lambda args: confirm(dv.plot_scatter, args[0], args[1], args[2] if len(args)>2 else None, save=args[3] if len(args)>3 else False),
            description="Scatter plot. Input: [DataFrame, x_col, y_col, color_col, save]"
        ),
        Tool(
            name="plot_bar",
            func=lambda args: confirm(dv.plot_bar, args[0], args[1] if len(args)>1 else None, agg_func=args[2] if len(args)>2 else "count", color=args[3] if len(args)>3 else "orange", save=args[4] if len(args)>4 else False),
            description="Bar plot. Input: [DataFrame, col, value_col, agg_func, color, save]"
        ),
        Tool(
            name="correlation_heatmap",
            func=lambda args: confirm(dv.correlation_heatmap, args[0], save=args[1] if len(args)>1 else False),
            description="Correlation heatmap. Input: [DataFrame, save]"
        ),
        Tool(
            name="plot_pca",
            func=lambda args: confirm(dv.plot_pca, da.compute_pca(args[0], n_components=args[1] if len(args)>1 else 2), save=args[2] if len(args)>2 else False),
            description="Compute PCA and plot. Input: [DataFrame, n_components, save]"
        ),
        Tool(
            name="gene_heatmap",
            func=lambda args: confirm(dv.gene_heatmap, args[0], save=args[1] if len(args)>1 else False),
            description="Gene-style heatmap with clustering. Input: [DataFrame, save]"
        ),
    ]
    return tools
