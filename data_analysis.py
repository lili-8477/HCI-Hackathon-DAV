#!/usr/bin/env python3
"""
Data Analysis Module
- Summary statistics
- PCA for numeric columns
- Optional Ollama summary
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def summary_stats(df, columns=None):
    """Descriptive statistics for numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include="number").columns
    return df[columns].describe()

def compute_correlation(df, method="pearson"):
    """Correlation matrix of numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    return df[numeric_cols].corr(method=method)

def compute_pca(df, n_components=2):
    """Compute PCA on numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        raise ValueError("Not enough numeric columns for PCA")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca_df
