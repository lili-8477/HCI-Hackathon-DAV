#!/usr/bin/env python3
"""
Data Analysis Module
- Summary statistics
- PCA for numeric columns
- Optional Ollama summary
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def summary_stats(df, columns=None):
    """Return descriptive statistics for selected columns or all numeric."""
    if columns is None:
        columns = df.select_dtypes(include="number").columns
    return df[columns].describe()

def compute_correlation(df, method="pearson"):
    """Return correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    return df[numeric_cols].corr(method=method)

def compute_pca(df, n_components=2):
    """Compute PCA on numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for PCA")
        return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca_df
