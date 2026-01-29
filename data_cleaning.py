#!/usr/bin/env python3
"""
Data Cleaning Module
- Handles duplicates
- Handles missing values interactively
"""
import pandas as pd

def load_data(file_path):
    """Load CSV, Excel, or JSON data."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV, Excel, or JSON.")
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def preprocess_data(df):
    """Remove duplicates and handle missing values interactively."""
    df = df.drop_duplicates()
    na_counts = df.isna().sum()
    if na_counts.sum() > 0:
        print("Columns with missing values:\n", na_counts[na_counts > 0])
        response = input("Fill missing values automatically? (y/n): ").lower()
        if response == "y":
            numeric_cols = df.select_dtypes(include="number").columns
            categorical_cols = df.select_dtypes(include="object").columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    return df

def select_columns(df, col_type="numeric"):
    """Let user select columns by type."""
    if col_type == "numeric":
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        cols = df.select_dtypes(include="object").columns.tolist()
    print(f"Available {col_type} columns: {cols}")
    selected = input("Enter comma-separated columns to use (or 'all'): ")
    if selected.lower() == "all":
        return cols
    return [c.strip() for c in selected.split(",")]
