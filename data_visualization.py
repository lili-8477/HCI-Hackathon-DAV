#!/usr/bin/env python3
"""
Data Visualization Module (Interactive + Save + Zoom)
- Continuous: histogram, boxplot, scatter
- Categorical: bar plot, count plot
- Correlation heatmaps
- PCA plot
- Save plots optionally
- Zoom/pan with Plotly
"""

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def confirm_and_run(func, *args, **kwargs):
    """Show function and args before execution for user approval."""
    print("\n=== Pending Tool Execution ===")
    print(f"Function: {func.__name__}")
    print(f"Arguments: {args if args else ''} {kwargs if kwargs else ''}")
    proceed = input("Run this tool? (y/n): ").lower()
    if proceed == "y":
        return func(*args, **kwargs)
    else:
        print("Skipped.")
        return None

# --- Continuous ---
def plot_histogram(df, col, color="skyblue", ablines=None, save=False):
    plt.figure()
    sns.histplot(df[col], kde=True, color=color)
    plt.title(f"Histogram & KDE of {col}")
    if ablines:
        for x in ablines:
            plt.axvline(x=x, color='red', linestyle='--', lw=2)
    if save:
        plt.savefig(f"{col}_hist.png")
        print(f"Saved: {col}_hist.png")
    plt.show()

    fig = px.histogram(df, x=col, title=f"{col} Histogram (Interactive)", color_discrete_sequence=[color])
    if ablines:
        for x in ablines:
            fig.add_vline(x=x, line_color="red", line_dash="dash")
    fig.show()

def plot_boxplot(df, col, color="skyblue", ablines=None, save=False):
    plt.figure()
    sns.boxplot(y=df[col], color=color)
    plt.title(f"Boxplot of {col}")
    if ablines:
        for y in ablines:
            plt.axhline(y=y, color='red', linestyle='--', lw=2)
    if save:
        plt.savefig(f"{col}_box.png")
        print(f"Saved: {col}_box.png")
    plt.show()

def plot_scatter(df, x_col, y_col, color_col=None, save=False):
    plt.figure()
    sns.scatterplot(x=df[x_col], y=df[y_col], hue=df[color_col] if color_col else None)
    plt.title(f"Scatter plot: {x_col} vs {y_col}")
    if save:
        plt.savefig(f"{x_col}_vs_{y_col}_scatter.png")
        print(f"Saved: {x_col}_vs_{y_col}_scatter.png")
    plt.show()

    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    fig.show()

# --- Categorical ---
def plot_bar(df, col, agg_col=None, agg_func="count", color="orange", save=False):
    if agg_col:
        grouped = df.groupby(col)[agg_col].agg(agg_func).reset_index()
        plt.figure()
        sns.barplot(x=col, y=agg_col, data=grouped, color=color)
        if save:
            plt.savefig(f"{col}_bar.png")
            print(f"Saved: {col}_bar.png")
        plt.show()
        fig = px.bar(grouped, x=col, y=agg_col, title=f"{col} bar plot ({agg_func})", color_discrete_sequence=[color])
        fig.show()
    else:
        plt.figure()
        sns.countplot(x=df[col], color=color)
        if save:
            plt.savefig(f"{col}_count.png")
            print(f"Saved: {col}_count.png")
        plt.show()
        fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, color_discrete_sequence=[color])
        fig.show()

# --- Correlation ---
def correlation_heatmap(df, save=False):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation heatmap")
        return
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    if save:
        plt.savefig("correlation_heatmap.png")
        print("Saved: correlation_heatmap.png")
    plt.show()
    fig = px.imshow(df[numeric_cols].corr(), text_auto=True, title="Correlation Heatmap (Interactive)")
    fig.show()

# --- PCA ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def plot_pca(df, n_components=2, save=False):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for PCA")
        return
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    plt.figure()
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.title("PCA Plot")
    if save:
        plt.savefig("PCA_plot.png")
        print("Saved: PCA_plot.png")
    plt.show()
    fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA Scatter Plot (Interactive)")
    fig.show()
    return pca_df
