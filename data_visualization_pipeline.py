#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotnine import ggplot, aes, geom_histogram, theme_minimal
from data_analysis_pipeline import load_data, preprocess_data
import sys

def visualize_with_matplotlib(df):
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure()
        plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of {col} (Matplotlib)")
        plt.show()

def visualize_with_seaborn(df):
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, color='orange')
        plt.title(f"Distribution of {col} (Seaborn)")
        plt.show()

def visualize_with_plotly(df):
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col} (Plotly)")
        fig.show()

def visualize_with_ggplot(df):
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        p = (ggplot(df) + aes(x=col) + geom_histogram(bins=30, fill='steelblue', color='black') + theme_minimal())
        print(p)

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_visualization_pipeline.py <datafile>")
        sys.exit(1)
    file_path = sys.argv[1]
    df = load_data(file_path)
    df = preprocess_data(df)
    visualize_with_matplotlib(df)
    visualize_with_seaborn(df)
    visualize_with_plotly(df)
    visualize_with_ggplot(df)

if __name__ == "__main__":
    main()
