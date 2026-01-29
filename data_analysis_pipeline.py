#!/usr/bin/env python3
import pandas as pd
import sys
import subprocess

def load_data(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV, Excel, or JSON.")
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    print("Preprocessing complete")
    return df

def generate_summary_with_ollama(df):
    prompt = f"""
You are a data analysis assistant.
Summarize the following dataset in plain language:
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
        print("\n=== Natural Language Summary ===")
        print(result.stdout)
    except Exception as e:
        print(f"Ollama summary failed: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_analysis_pipeline.py <datafile>")
        sys.exit(1)
    file_path = sys.argv[1]
    df = load_data(file_path)
    df = preprocess_data(df)
    print("\n=== Data Head ===")
    print(df.head())
    print("\n=== Summary Statistics ===")
    print(df.describe(include='all').transpose())
    generate_summary_with_ollama(df)

if __name__ == "__main__":
    main()
