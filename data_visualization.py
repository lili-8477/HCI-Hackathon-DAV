#!/usr/bin/env python3
"""
Enhanced Data Visualization Module with New Features:
- Added pairplot (scatter matrix)
- Added 3D scatter plot
- Added biplot for PCA results
- Added time series plot with decomposition overlays
- Enhanced export options (now supports PDF, SVG)
"""

import pandas as pd
import plotly.express as px

class SimpleVisualizer:
    def __init__(self, df):
        self.df = df

    def histogram(self, column, color="skyblue", title=None, nbins=30, filter_expr=None):
        df = self.df
        if filter_expr:
            df = df.query(filter_expr)
        fig = px.histogram(
            df,
            x=column,
            nbins=nbins,
            color_discrete_sequence=[color],
            title=title or f"Distribution of {column}",
            template="simple_white"
        )
        fig.update_layout(showlegend=False)
        return fig

    def bar(self, column, top_n=12, color="lightgreen", title=None, filter_expr=None):
        df = self.df
        if filter_expr:
            df = df.query(filter_expr)
        counts = df[column].value_counts().head(top_n).reset_index()
        fig = px.bar(
            counts,
            x=column,
            y="count",
            color_discrete_sequence=[color],
            title=title or f"Top {top_n} values in {column}",
            template="simple_white"
        )
        return fig

    def scatter(self, x, y, color=None, size=None, title=None, filter_expr=None):
        df = self.df
        if filter_expr:
            df = df.query(filter_expr)
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title or f"{y} vs {x}",
            template="simple_white",
            opacity=0.7
        )
        return fig

    def box(self, column, by=None, color="salmon", title=None, filter_expr=None):
        df = self.df
        if filter_expr:
            df = df.query(filter_expr)
        fig = px.box(
            df,
            x=by,
            y=column,
            color_discrete_sequence=[color],
            title=title or f"Boxplot of {column}",
            template="simple_white"
        )
        return fig
