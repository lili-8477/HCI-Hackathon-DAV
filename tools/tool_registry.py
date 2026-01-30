"""
Tool Registry
Registers all tools and makes them available to the LangChain agent.
"""

from langchain_core.tools import Tool
from . import data_exploration, statistical_analysis, data_manipulation, visualization


def get_all_tools():
    """
    Get all tools for the manager agent.

    Returns:
        List of LangChain Tool objects
    """
    tools = [
        # Data Exploration Tools
        Tool(
            name="get_data_overview",
            func=lambda _="": data_exploration.get_data_overview(),
            description="Get an overview of the loaded dataset including shape, columns, and data types. Use this to understand the structure of the data. No input required."
        ),
        Tool(
            name="check_missing_values",
            func=lambda _="": data_exploration.check_missing_values(),
            description="Check for missing values in the dataset. Shows count and percentage of missing values for each column. No input required."
        ),
        Tool(
            name="get_summary_statistics",
            func=lambda _="": data_exploration.get_summary_statistics(),
            description="Get descriptive statistics (mean, std, min, max, quartiles) for all numeric columns in the dataset. No input required."
        ),
        Tool(
            name="get_column_info",
            func=data_exploration.get_column_info,
            description="Get detailed information about a specific column including data type, missing values, and basic statistics. Input: column_name as string."
        ),
        Tool(
            name="list_columns",
            func=lambda _="": data_exploration.list_columns(),
            description="List all column names in the dataset. No input required."
        ),

        # Statistical Analysis Tools
        Tool(
            name="calculate_correlation",
            func=lambda method="pearson": statistical_analysis.calculate_correlation(method if method.strip() else "pearson"),
            description="Calculate correlation matrix for all numeric columns. Input: method ('pearson', 'spearman', or 'kendall'). Default is 'pearson' if no input provided."
        ),
        Tool(
            name="perform_groupby",
            func=lambda input_str: _parse_groupby_input(input_str),
            description="Perform group-by aggregation on the data. Input format: 'group_column,agg_column,agg_func' where agg_func can be 'mean', 'sum', 'count', 'min', 'max', or 'std'. Example: 'region,revenue,sum'"
        ),
        Tool(
            name="perform_ttest",
            func=lambda input_str: _parse_ttest_input(input_str),
            description="Perform a t-test on a numeric column. Input format: 'column,group_column' for independent two-sample t-test, 'column,group_column,paired' for paired t-test, or 'column,test_value,one_sample' for one-sample t-test. The group_column must have exactly 2 unique values for independent/paired tests. Example: 'score,treatment' or 'weight,0,one_sample'"
        ),
        Tool(
            name="detect_outliers",
            func=statistical_analysis.detect_outliers,
            description="Detect outliers in a numeric column using IQR method. Input: column_name as string."
        ),

        # Data Manipulation Tools
        Tool(
            name="filter_rows",
            func=data_manipulation.filter_rows,
            description="Filter rows using a pandas query expression. Input: query string (e.g. 'price > 100' or 'region == \"West\"'). Saves result as an intermediate table."
        ),
        Tool(
            name="select_columns",
            func=data_manipulation.select_columns,
            description="Select specific columns from the dataset. Input: comma-separated column names (e.g. 'name,price,region'). Saves result as an intermediate table."
        ),
        Tool(
            name="pivot_table",
            func=data_manipulation.pivot_table,
            description="Create a pivot table. Input format: 'index,values,agg_func' (e.g. 'region,revenue,sum'). Saves result as an intermediate table."
        ),
        Tool(
            name="transform_column",
            func=data_manipulation.transform_column,
            description="Transform a column's values. Input format: 'column,operation' where operation is 'log', 'normalize', 'round:N', 'upper', or 'lower'. Example: 'price,log'. Saves result as an intermediate table."
        ),
        Tool(
            name="use_table",
            func=data_manipulation.use_table,
            description="Set a named intermediate table as the active dataset for all tools. Input: table name (e.g. 'filtered_1'). Use 'list_tables' to see available tables."
        ),
        Tool(
            name="list_tables",
            func=lambda _="": data_manipulation.list_tables(),
            description="List all saved intermediate tables with their shapes. No input required."
        ),

        # Visualization Tools
        Tool(
            name="plot_distribution",
            func=lambda input_str: _parse_distribution_input(input_str),
            description="Create a histogram or KDE plot showing the distribution of a numeric column. Input format: 'column_name' or 'column_name,plot_type' where plot_type is 'histogram' or 'kde'. Example: 'price' or 'price,histogram'"
        ),
        Tool(
            name="plot_correlation_heatmap",
            func=lambda _: visualization.plot_correlation_heatmap(),
            description="Create a correlation heatmap showing relationships between all numeric columns. No input required."
        ),
        Tool(
            name="plot_scatter",
            func=lambda input_str: _parse_scatter_input(input_str),
            description="Create a scatter plot for two variables. Input format: 'x_column,y_column' or 'x_column,y_column,color_column'. Example: 'price,quantity' or 'price,quantity,region'"
        ),
        Tool(
            name="plot_bar_chart",
            func=lambda input_str: _parse_bar_chart_input(input_str),
            description="Create a bar chart for categorical data. Input format: 'column_name' for counts, or 'column_name,value_column,agg_func' for aggregations. Example: 'region' or 'region,revenue,sum'"
        ),
        Tool(
            name="plot_pie_chart",
            func=lambda input_str: _parse_pie_chart_input(input_str),
            description="Create a pie chart for categorical data. Input format: 'column_name' for value counts, or 'column_name,value_column,agg_func' for aggregations. Example: 'region' or 'region,revenue,sum'"
        ),
        Tool(
            name="plot_box_plot",
            func=lambda input_str: _parse_box_plot_input(input_str),
            description="Create a box plot showing distribution and outliers. Input format: 'column_name' or 'column_name,group_by'. Example: 'price' or 'price,region'"
        ),
        Tool(
            name="plot_time_series",
            func=lambda input_str: _parse_time_series_input(input_str),
            description="Create a time series line plot. Input format: 'date_column,value_column'. Example: 'date,revenue'"
        ),
    ]

    # Add RAG tools for code generation and memory
    try:
        from .rag_tools import get_rag_tools
        tools.extend(get_rag_tools())
    except ImportError as e:
        print(f"Warning: RAG tools not available: {e}")

    return tools


# Helper functions to parse tool inputs
def _parse_groupby_input(input_str: str):
    """Parse input for perform_groupby tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 2:
            return statistical_analysis.perform_groupby(parts[0], parts[1])
        elif len(parts) == 3:
            return statistical_analysis.perform_groupby(parts[0], parts[1], parts[2])
        else:
            return "Invalid input format. Expected: 'group_column,agg_column' or 'group_column,agg_column,agg_func'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_distribution_input(input_str: str):
    """Parse input for plot_distribution tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 1:
            return visualization.plot_distribution(parts[0])
        elif len(parts) == 2:
            return visualization.plot_distribution(parts[0], parts[1])
        else:
            return "Invalid input format. Expected: 'column_name' or 'column_name,plot_type'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_scatter_input(input_str: str):
    """Parse input for plot_scatter tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 2:
            return visualization.plot_scatter(parts[0], parts[1])
        elif len(parts) == 3:
            return visualization.plot_scatter(parts[0], parts[1], parts[2])
        else:
            return "Invalid input format. Expected: 'x_column,y_column' or 'x_column,y_column,color_column'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_bar_chart_input(input_str: str):
    """Parse input for plot_bar_chart tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 1:
            return visualization.plot_bar_chart(parts[0])
        elif len(parts) == 3:
            return visualization.plot_bar_chart(parts[0], parts[1], parts[2])
        else:
            return "Invalid input format. Expected: 'column_name' or 'column_name,value_column,agg_func'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_pie_chart_input(input_str: str):
    """Parse input for plot_pie_chart tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 1:
            return visualization.plot_pie_chart(parts[0])
        elif len(parts) == 3:
            return visualization.plot_pie_chart(parts[0], parts[1], parts[2])
        else:
            return "Invalid input format. Expected: 'column_name' or 'column_name,value_column,agg_func'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_box_plot_input(input_str: str):
    """Parse input for plot_box_plot tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 1:
            return visualization.plot_box_plot(parts[0])
        elif len(parts) == 2:
            return visualization.plot_box_plot(parts[0], parts[1])
        else:
            return "Invalid input format. Expected: 'column_name' or 'column_name,group_by'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_time_series_input(input_str: str):
    """Parse input for plot_time_series tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 2:
            return visualization.plot_time_series(parts[0], parts[1])
        else:
            return "Invalid input format. Expected: 'date_column,value_column'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def _parse_ttest_input(input_str: str):
    """Parse input for perform_ttest tool"""
    try:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 2:
            return statistical_analysis.perform_ttest(parts[0], parts[1])
        elif len(parts) == 3:
            return statistical_analysis.perform_ttest(parts[0], parts[1], parts[2])
        else:
            return "Invalid input format. Expected: 'column,group_column' or 'column,group_column,test_type'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"
