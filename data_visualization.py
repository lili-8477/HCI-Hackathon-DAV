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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from data_cleaning import load_data, preprocess_data
from data_analysis import compute_pca

def visualize_continuous(df, col):
    color = input(f"Choose color for continuous variable {col} (default 'skyblue'): ") or "skyblue"
    save = input("Save this plot? (y/n): ").lower() == "y"

    # Matplotlib histogram
    plt.figure()
    sns.histplot(df[col], kde=True, color=color)
    plt.title(f"Histogram & Density of {col}")
    if save:
        filename = f"{col}_hist.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.show()

    # Boxplot
    plt.figure()
    sns.boxplot(y=df[col], color=color)
    plt.title(f"Boxplot of {col}")
    if save:
        filename = f"{col}_box.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.show()

    # Plotly interactive
    fig = px.histogram(df, x=col, title=f"{col} Histogram (Interactive)", color_discrete_sequence=[color])
    fig.show()

def visualize_categorical(df, col):
    color = input(f"Choose color for categorical variable {col} (default 'orange'): ") or "orange"
    save = input("Save this plot? (y/n): ").lower() == "y"

    # Matplotlib countplot
    plt.figure()
    sns.countplot(x=df[col], color=color)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    if save:
        filename = f"{col}_count.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.show()

    # Plotly interactive
    fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col,
                 labels={'index': col, col: 'Count'}, color_discrete_sequence=[color],
                 title=f"{col} Countplot (Interactive)")
    fig.show()

def correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation heatmap")
        return
    save = input("Save correlation heatmap? (y/n): ").lower() == "y"

    # Static heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    if save:
        filename = "correlation_heatmap.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.show()

    # Plotly interactive
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap (Interactive)")
    fig.show()

def visualize_pca(pca_df):
    if pca_df is None:
        return
    save = input("Save PCA plot? (y/n): ").lower() == "y"

    # Static plot
    plt.figure()
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    plt.title("PCA Plot")
    if save:
        filename = "PCA_plot.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.show()

    # Plotly interactive
    fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA Scatter Plot (Interactive)")
    fig.show()

if __name__ == "__main__":
    file_path = input("Enter data file path: ")
    df = load_data(file_path)
    df = preprocess_data(df)
    pca_df, _ = compute_pca(df)

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in numeric_cols:
        visualize_continuous(df, col)

    for col in categorical_cols:
        visualize_categorical(df, col)

    correlation_heatmap(df)
    visualize_pca(pca_df)
