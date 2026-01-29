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
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def handle_missing(df):
    """Interactively handle missing values."""
    na_counts = df.isna().sum()
    if na_counts.sum() > 0:
        print("\nColumns with missing values:")
        print(na_counts[na_counts > 0])
        response = input("Do you want to fill missing values automatically? (y/n) ")
        if response.lower() == "y":
            numeric_cols = df.select_dtypes(include="number").columns
            categorical_cols = df.select_dtypes(include="object").columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            df[categorical_cols] = df[categorical_cols].fillna("Unknown")
            print("Missing values filled automatically")
        else:
            print("Missing values left as NA")
    return df

def preprocess_data(df):
    """Remove duplicates and handle missing values."""
    df = df.drop_duplicates()
    df = handle_missing(df)
    return df

if __name__ == "__main__":
    file_path = input("Enter data file path: ")
    df = load_data(file_path)
    df = preprocess_data(df)
    print("Data cleaning complete.")
