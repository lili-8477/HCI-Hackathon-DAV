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
import subprocess
from data_cleaning import load_data, preprocess_data

def summary_stats(df):
    print("\n=== Summary Statistics ===")
    print(df.describe(include='all').transpose())
    print("\n=== Column Types ===")
    print(df.dtypes)
    print("\n=== NA Counts ===")
    print(df.isna().sum())

def compute_pca(df, n_components=2):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for PCA")
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
    return pca_df, pca

def generate_summary_with_ollama(df):
    prompt = f"""
You are a data analysis assistant.
Summarize the dataset in plain language:
Columns: {', '.join(df.columns)}
First 5 rows:
{df.head().to_string(index=False)}
"""
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen3:8b"],
            input=prompt,
            text=True,
            capture_output=True
        )
        print("\n=== Ollama Summary ===")
        print(result.stdout)
    except Exception as e:
        print(f"Ollama failed: {e}")

if __name__ == "__main__":
    file_path = input("Enter data file path: ")
    df = load_data(file_path)
    df = preprocess_data(df)
    summary_stats(df)
    pca_df, _ = compute_pca(df)
    if pca_df is not None:
        print("\n=== PCA Head ===")
        print(pca_df.head())
    run_ollama = input("Run Ollama summary? (y/n) ")
    if run_ollama.lower() == "y":
        generate_summary_with_ollama(df)
