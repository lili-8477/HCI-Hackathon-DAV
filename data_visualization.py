#!/usr/bin/env python3
"""
Enhanced Data Visualization Module for LLM Integration & Streamlit
- Interactive filtering and color customization
- Multiple chart types with extensive options
- Export to various formats
- LLM-ready chart descriptions
- Theme support and style presets
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union, Any
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataVisualizer:
    """
    Comprehensive visualization class with LLM integration and Streamlit support.
    """
    
    # Color schemes
    COLOR_SCHEMES = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'pastel': ['#AEC6CF', '#FFB347', '#B19CD9', '#FFD1DC', '#CFCFC4',
                   '#FDFD96', '#C1E1C1', '#FAC898', '#B39EB5', '#FFB6B9'],
        'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#A8E6CF'],
        'corporate': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600',
                      '#7a5195', '#ef5675', '#ffa600', '#003f5c', '#665191'],
        'earth': ['#8B4513', '#CD853F', '#DEB887', '#D2691E', '#F4A460',
                  '#BC8F8F', '#A0522D', '#DAA520', '#B8860B', '#CD5C5C'],
        'ocean': ['#006994', '#0892A5', '#13ADB7', '#28C2D1', '#3CD5E8',
                  '#52E1F2', '#6FEDFD', '#9CFFFE', '#C8FFFF', '#E3FFFF'],
        'sunset': ['#FF6B35', '#F7931E', '#FDC830', '#F37335', '#C73E1D',
                   '#EB5E28', '#F06449', '#EF476F', '#FFD23F', '#EE964B'],
        'forest': ['#2D5016', '#3D6E23', '#4E8D2F', '#5FAC3C', '#70CB49',
                   '#81EA56', '#92FF63', '#A3FF70', '#B4FF7D', '#C5FF8A']
    }
    
    # Chart templates
    TEMPLATES = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn',
                 'simple_white', 'presentation', 'none']
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize visualizer with optional dataframe.
        
        Args:
            df: Input pandas DataFrame
        """
        self.df = df
        self.chart_history = []
        self.color_scheme = 'default'
        self.template = 'plotly_white'
        
    def set_color_scheme(self, scheme: str):
        """Set the color scheme for visualizations."""
        if scheme in self.COLOR_SCHEMES:
            self.color_scheme = scheme
        else:
            raise ValueError(f"Unknown scheme. Available: {list(self.COLOR_SCHEMES.keys())}")
    
    def set_template(self, template: str):
        """Set the Plotly template."""
        if template in self.TEMPLATES:
            self.template = template
        else:
            raise ValueError(f"Unknown template. Available: {self.TEMPLATES}")
    
    def get_colors(self, n: int = 1) -> List[str]:
        """Get n colors from current scheme."""
        colors = self.COLOR_SCHEMES[self.color_scheme]
        if n <= len(colors):
            return colors[:n]
        # Repeat colors if needed
        return (colors * ((n // len(colors)) + 1))[:n]
    
    def filter_data(self,
                   column: str,
                   condition: str,
                   value: Any) -> pd.DataFrame:
        """
        Filter dataframe based on condition.
        
        Args:
            column: Column to filter on
            condition: Comparison operator ('==', '!=', '>', '<', '>=', '<=', 'in', 'not in', 'contains')
            value: Value to compare against
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        filtered_df = self.df.copy()
        
        if condition == '==':
            filtered_df = filtered_df[filtered_df[column] == value]
        elif condition == '!=':
            filtered_df = filtered_df[filtered_df[column] != value]
        elif condition == '>':
            filtered_df = filtered_df[filtered_df[column] > value]
        elif condition == '<':
            filtered_df = filtered_df[filtered_df[column] < value]
        elif condition == '>=':
            filtered_df = filtered_df[filtered_df[column] >= value]
        elif condition == '<=':
            filtered_df = filtered_df[filtered_df[column] <= value]
        elif condition == 'in':
            filtered_df = filtered_df[filtered_df[column].isin(value)]
        elif condition == 'not in':
            filtered_df = filtered_df[~filtered_df[column].isin(value)]
        elif condition == 'contains':
            filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(value, na=False)]
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        return filtered_df
    
    def apply_filters(self, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply multiple filters to data.
        
        Args:
            filters: List of filter dictionaries with 'column', 'condition', 'value'
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = self.df.copy()
        
        for f in filters:
            column = f['column']
            condition = f['condition']
            value = f['value']
            
            if condition == '==':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif condition == '!=':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif condition == '>':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif condition == '<':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif condition == '>=':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif condition == '<=':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif condition == 'in':
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif condition == 'not in':
                filtered_df = filtered_df[~filtered_df[column].isin(value)]
            elif condition == 'contains':
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
        
        return filtered_df
    
    def plot_histogram(self,
                      column: str,
                      bins: int = 30,
                      color: Optional[str] = None,
                      title: Optional[str] = None,
                      show_kde: bool = True,
                      reference_lines: Optional[List[float]] = None,
                      filters: Optional[List[Dict]] = None,
                      **kwargs) -> go.Figure:
        """
        Create interactive histogram.
        
        Args:
            column: Column name to plot
            bins: Number of bins
            color: Custom color (overrides scheme)
            title: Chart title
            show_kde: Show kernel density estimate
            reference_lines: List of x-values for vertical reference lines
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Get color
        plot_color = color or self.get_colors(1)[0]
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=plot_df[column],
            nbinsx=bins,
            name='Distribution',
            marker_color=plot_color,
            opacity=0.7
        ))
        
        # Add KDE if requested
        if show_kde:
            kde_data = plot_df[column].dropna()
            if len(kde_data) > 1:
                from scipy import stats
                kde = stats.gaussian_kde(kde_data)
                x_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                kde_values = kde(x_range)
                
                # Scale KDE to match histogram
                hist, bin_edges = np.histogram(kde_data, bins=bins)
                bin_width = bin_edges[1] - bin_edges[0]
                kde_scaled = kde_values * len(kde_data) * bin_width
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_scaled,
                    name='KDE',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
        
        # Add reference lines
        if reference_lines:
            for value in reference_lines:
                fig.add_vline(
                    x=value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{value:.2f}"
                )
        
        # Update layout
        chart_title = title or f"Distribution of {column}"
        fig.update_layout(
            title=chart_title,
            xaxis_title=column,
            yaxis_title='Count',
            template=self.template,
            hovermode='x unified',
            **kwargs
        )
        
        if show_kde:
            fig.update_layout(
                yaxis2=dict(overlaying='y', side='right', title='Density')
            )
        
        self._log_chart('histogram', column, chart_title)
        
        return fig
    
    def plot_boxplot(self,
                    column: str,
                    group_by: Optional[str] = None,
                    color: Optional[str] = None,
                    title: Optional[str] = None,
                    orientation: str = 'v',
                    show_points: bool = False,
                    reference_lines: Optional[List[float]] = None,
                    filters: Optional[List[Dict]] = None,
                    **kwargs) -> go.Figure:
        """
        Create interactive boxplot.
        
        Args:
            column: Column to plot
            group_by: Optional column to group by
            color: Custom color
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal
            show_points: Show individual data points
            reference_lines: Reference lines (horizontal for vertical box, vertical for horizontal box)
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Create figure
        if group_by:
            fig = px.box(
                plot_df,
                y=column if orientation == 'v' else None,
                x=column if orientation == 'h' else None,
                color=group_by,
                orientation=orientation,
                points='all' if show_points else False,
                color_discrete_sequence=self.get_colors(plot_df[group_by].nunique()),
                template=self.template
            )
        else:
            plot_color = color or self.get_colors(1)[0]
            fig = go.Figure()
            
            if orientation == 'v':
                fig.add_trace(go.Box(
                    y=plot_df[column],
                    name=column,
                    marker_color=plot_color,
                    boxpoints='all' if show_points else 'outliers'
                ))
            else:
                fig.add_trace(go.Box(
                    x=plot_df[column],
                    name=column,
                    marker_color=plot_color,
                    boxpoints='all' if show_points else 'outliers',
                    orientation='h'
                ))
        
        # Add reference lines
        if reference_lines:
            for value in reference_lines:
                if orientation == 'v':
                    fig.add_hline(y=value, line_dash="dash", line_color="red")
                else:
                    fig.add_vline(x=value, line_dash="dash", line_color="red")
        
        # Update layout
        chart_title = title or f"Boxplot of {column}"
        if group_by:
            chart_title += f" by {group_by}"
        
        fig.update_layout(
            title=chart_title,
            template=self.template,
            **kwargs
        )
        
        self._log_chart('boxplot', f"{column} (grouped by {group_by})" if group_by else column, chart_title)
        
        return fig
    
    def plot_scatter(self,
                    x_column: str,
                    y_column: str,
                    color_by: Optional[str] = None,
                    size_by: Optional[str] = None,
                    title: Optional[str] = None,
                    trendline: Optional[str] = None,
                    marginal: Optional[str] = None,
                    filters: Optional[List[Dict]] = None,
                    **kwargs) -> go.Figure:
        """
        Create interactive scatter plot.
        
        Args:
            x_column: X-axis column
            y_column: Y-axis column
            color_by: Column to color points by
            size_by: Column to size points by
            title: Chart title
            trendline: Trendline type ('ols', 'lowess', 'rolling', 'expanding', 'ewm')
            marginal: Marginal plot type ('histogram', 'box', 'violin', 'rug')
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_by,
            size=size_by,
            trendline=trendline,
            marginal_x=marginal,
            marginal_y=marginal,
            color_discrete_sequence=self.get_colors(plot_df[color_by].nunique() if color_by else 1),
            template=self.template,
            **kwargs
        )
        
        # Update layout
        chart_title = title or f"{y_column} vs {x_column}"
        fig.update_layout(
            title=chart_title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            hovermode='closest'
        )
        
        self._log_chart('scatter', f"{x_column} vs {y_column}", chart_title)
        
        return fig
    
    def plot_bar(self,
                column: str,
                value_column: Optional[str] = None,
                aggregation: str = 'count',
                orientation: str = 'v',
                color: Optional[str] = None,
                title: Optional[str] = None,
                sort_by: Optional[str] = None,
                top_n: Optional[int] = None,
                filters: Optional[List[Dict]] = None,
                **kwargs) -> go.Figure:
        """
        Create interactive bar chart.
        
        Args:
            column: Category column
            value_column: Value column (for aggregation)
            aggregation: Aggregation function ('count', 'sum', 'mean', 'median', 'min', 'max')
            orientation: 'v' for vertical, 'h' for horizontal
            color: Custom color
            title: Chart title
            sort_by: Sort bars by 'value' or 'category'
            top_n: Show only top N categories
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Prepare data
        if value_column:
            if aggregation == 'count':
                grouped = plot_df.groupby(column).size().reset_index(name='count')
                y_col = 'count'
            else:
                grouped = plot_df.groupby(column)[value_column].agg(aggregation).reset_index()
                grouped.columns = [column, aggregation]
                y_col = aggregation
        else:
            grouped = plot_df[column].value_counts().reset_index()
            grouped.columns = [column, 'count']
            y_col = 'count'
        
        # Sort
        if sort_by == 'value':
            grouped = grouped.sort_values(y_col, ascending=False)
        elif sort_by == 'category':
            grouped = grouped.sort_values(column)
        
        # Top N
        if top_n:
            grouped = grouped.head(top_n)
        
        # Get color
        plot_color = color or self.get_colors(1)[0]
        
        # Create figure
        if orientation == 'v':
            fig = go.Figure(go.Bar(
                x=grouped[column],
                y=grouped[y_col],
                marker_color=plot_color,
                text=grouped[y_col],
                textposition='auto'
            ))
        else:
            fig = go.Figure(go.Bar(
                y=grouped[column],
                x=grouped[y_col],
                marker_color=plot_color,
                orientation='h',
                text=grouped[y_col],
                textposition='auto'
            ))
        
        # Update layout
        chart_title = title or f"{column} Distribution"
        if value_column:
            chart_title += f" ({aggregation} of {value_column})"
        
        fig.update_layout(
            title=chart_title,
            xaxis_title=column if orientation == 'v' else y_col,
            yaxis_title=y_col if orientation == 'v' else column,
            template=self.template,
            **kwargs
        )
        
        self._log_chart('bar', column, chart_title)
        
        return fig
    
    def plot_line(self,
                 x_column: str,
                 y_columns: Union[str, List[str]],
                 color_by: Optional[str] = None,
                 title: Optional[str] = None,
                 markers: bool = False,
                 fill: bool = False,
                 filters: Optional[List[Dict]] = None,
                 **kwargs) -> go.Figure:
        """
        Create interactive line chart.
        
        Args:
            x_column: X-axis column (usually time/date)
            y_columns: Y-axis column(s)
            color_by: Column to color lines by
            title: Chart title
            markers: Show markers on lines
            fill: Fill area under line
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Convert to list if single column
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        # Create figure
        fig = go.Figure()
        
        colors = self.get_colors(len(y_columns))
        
        for i, y_col in enumerate(y_columns):
            mode = 'lines+markers' if markers else 'lines'
            fill_mode = 'tonexty' if fill and i > 0 else 'tozeroy' if fill else None
            
            fig.add_trace(go.Scatter(
                x=plot_df[x_column],
                y=plot_df[y_col],
                name=y_col,
                mode=mode,
                line=dict(color=colors[i]),
                fill=fill_mode
            ))
        
        # Update layout
        chart_title = title or f"{', '.join(y_columns)} over {x_column}"
        fig.update_layout(
            title=chart_title,
            xaxis_title=x_column,
            yaxis_title='Value',
            template=self.template,
            hovermode='x unified',
            **kwargs
        )
        
        self._log_chart('line', f"{y_columns}", chart_title)
        
        return fig
    
    def plot_heatmap(self,
                    columns: Optional[List[str]] = None,
                    method: str = 'correlation',
                    title: Optional[str] = None,
                    colorscale: str = 'RdBu_r',
                    annotations: bool = True,
                    filters: Optional[List[Dict]] = None,
                    **kwargs) -> go.Figure:
        """
        Create interactive heatmap.
        
        Args:
            columns: Columns to include (default: all numeric)
            method: 'correlation' or 'values'
            title: Chart title
            colorscale: Plotly colorscale name
            annotations: Show value annotations
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Get columns
        if columns is None:
            columns = plot_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("Need at least 2 columns for heatmap")
        
        # Prepare data
        if method == 'correlation':
            data = plot_df[columns].corr()
            chart_title = title or "Correlation Heatmap"
            zmin, zmax = -1, 1
        else:
            data = plot_df[columns]
            chart_title = title or "Value Heatmap"
            zmin, zmax = None, None
        
        # Create figure
        fig = px.imshow(
            data,
            text_auto=annotations,
            color_continuous_scale=colorscale,
            zmin=zmin,
            zmax=zmax,
            template=self.template,
            **kwargs
        )
        
        fig.update_layout(title=chart_title)
        
        self._log_chart('heatmap', method, chart_title)
        
        return fig
    
    def plot_pie(self,
                column: str,
                values_column: Optional[str] = None,
                title: Optional[str] = None,
                hole: float = 0.0,
                top_n: Optional[int] = None,
                filters: Optional[List[Dict]] = None,
                **kwargs) -> go.Figure:
        """
        Create interactive pie/donut chart.
        
        Args:
            column: Category column
            values_column: Values column (if None, uses count)
            title: Chart title
            hole: Size of hole (0 for pie, 0.4 for donut)
            top_n: Show only top N categories
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Prepare data
        if values_column:
            grouped = plot_df.groupby(column)[values_column].sum().reset_index()
        else:
            grouped = plot_df[column].value_counts().reset_index()
            grouped.columns = [column, 'count']
            values_column = 'count'
        
        # Sort and get top N
        grouped = grouped.sort_values(values_column, ascending=False)
        if top_n:
            grouped = grouped.head(top_n)
        
        # Create figure
        fig = go.Figure(go.Pie(
            labels=grouped[column],
            values=grouped[values_column],
            hole=hole,
            marker=dict(colors=self.get_colors(len(grouped))),
            textposition='auto',
            textinfo='label+percent'
        ))
        
        # Update layout
        chart_title = title or f"Distribution of {column}"
        fig.update_layout(
            title=chart_title,
            template=self.template,
            **kwargs
        )
        
        self._log_chart('pie', column, chart_title)
        
        return fig
    
    def plot_violin(self,
                   column: str,
                   group_by: Optional[str] = None,
                   title: Optional[str] = None,
                   box: bool = True,
                   points: bool = False,
                   filters: Optional[List[Dict]] = None,
                   **kwargs) -> go.Figure:
        """
        Create interactive violin plot.
        
        Args:
            column: Column to plot
            group_by: Column to group by
            title: Chart title
            box: Show box plot inside violin
            points: Show individual points
            filters: List of filters to apply
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        # Apply filters
        plot_df = self.apply_filters(filters) if filters else self.df.copy()
        
        # Create figure
        fig = px.violin(
            plot_df,
            y=column,
            x=group_by,
            box=box,
            points='all' if points else False,
            color=group_by,
            color_discrete_sequence=self.get_colors(plot_df[group_by].nunique() if group_by else 1),
            template=self.template,
            **kwargs
        )
        
        # Update layout
        chart_title = title or f"Violin Plot of {column}"
        if group_by:
            chart_title += f" by {group_by}"
        
        fig.update_layout(title=chart_title)
        
        self._log_chart('violin', column, chart_title)
        
        return fig
    
    def plot_pca(self,
                pca_df: pd.DataFrame,
                color_by: Optional[str] = None,
                title: Optional[str] = None,
                show_loadings: bool = False,
                **kwargs) -> go.Figure:
        """
        Create PCA scatter plot.
        
        Args:
            pca_df: DataFrame with PC1, PC2 columns
            color_by: Column to color points by
            title: Chart title
            show_loadings: Show feature loading vectors
            **kwargs: Additional plotly arguments
            
        Returns:
            Plotly Figure object
        """
        # Create scatter
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color=color_by,
            color_discrete_sequence=self.get_colors(pca_df[color_by].nunique() if color_by else 1),
            template=self.template,
            **kwargs
        )
        
        # Update layout
        chart_title = title or "PCA Scatter Plot"
        fig.update_layout(
            title=chart_title,
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2'
        )
        
        self._log_chart('pca', 'PC1 vs PC2', chart_title)
        
        return fig
    
    def create_dashboard(self,
                        charts: List[Dict[str, Any]],
                        rows: int = 2,
                        cols: int = 2,
                        title: str = "Dashboard") -> go.Figure:
        """
        Create multi-chart dashboard.
        
        Args:
            charts: List of chart configurations
            rows: Number of rows
            cols: Number of columns
            title: Dashboard title
            
        Returns:
            Plotly Figure with subplots
        """
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[c.get('title', '') for c in charts]
        )
        
        # Add charts
        for i, chart_config in enumerate(charts):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            chart_type = chart_config['type']
            
            # Create individual chart
            if chart_type == 'histogram':
                temp_fig = self.plot_histogram(**chart_config.get('params', {}))
            elif chart_type == 'bar':
                temp_fig = self.plot_bar(**chart_config.get('params', {}))
            elif chart_type == 'scatter':
                temp_fig = self.plot_scatter(**chart_config.get('params', {}))
            # Add more types as needed
            
            # Add traces to subplot
            for trace in temp_fig.data:
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            title_text=title,
            template=self.template,
            showlegend=False,
            height=300 * rows
        )
        
        return fig
    
    def save_figure(self,
                   fig: go.Figure,
                   filename: str,
                   format: str = 'html',
                   width: Optional[int] = None,
                   height: Optional[int] = None):
        """
        Save figure to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: 'html', 'png', 'jpg', 'svg', 'pdf'
            width: Image width (for static formats)
            height: Image height (for static formats)
        """
        filepath = Path(filename)
        
        if format == 'html':
            fig.write_html(str(filepath))
        else:
            # Requires kaleido
            fig.write_image(
                str(filepath),
                format=format,
                width=width,
                height=height
            )
        
        print(f"Saved: {filepath}")
    
    def describe_chart(self, fig: go.Figure) -> str:
        """
        Generate natural language description of chart for LLM.
        
        Args:
            fig: Plotly figure
            
        Returns:
            Text description
        """
        layout = fig.layout
        
        description = f"Chart Type: {fig.data[0].type}\n"
        description += f"Title: {layout.title.text}\n"
        
        if hasattr(layout, 'xaxis') and layout.xaxis.title:
            description += f"X-axis: {layout.xaxis.title.text}\n"
        if hasattr(layout, 'yaxis') and layout.yaxis.title:
            description += f"Y-axis: {layout.yaxis.title.text}\n"
        
        description += f"Number of traces: {len(fig.data)}\n"
        
        return description
    
    def export_for_llm(self) -> Dict[str, Any]:
        """
        Export visualization history for LLM consumption.
        
        Returns:
            Dictionary with chart metadata
        """
        return {
            'chart_history': self.chart_history,
            'color_scheme': self.color_scheme,
            'template': self.template,
            'total_charts': len(self.chart_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_chart(self, chart_type: str, data_info: str, title: str):
        """Log chart creation."""
        self.chart_history.append({
            'type': chart_type,
            'data': data_info,
            'title': title,
            'timestamp': datetime.now().isoformat()
        })


if __name__ == "__main__":
    print("Data Visualization Module - Ready for LLM Integration & Streamlit")
    print("Import DataVisualizer class for full functionality")
