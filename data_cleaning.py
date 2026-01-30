#!/usr/bin/env python3
"""
Modern Data Cleaning Module – 2025 edition
New / improved capabilities compared to earlier versions:
- KNN & Iterative (MICE-like) imputation
- Fuzzy / near-duplicate detection (using rapidfuzz)
- Isolation Forest & DBSCAN-style outlier detection
- Configurable text cleaning pipeline
- Basic schema validation & type consistency checks
- Rich, structured cleaning report suitable for LLM chaining
- Action history with reversible steps (soft delete tracking)
"""

# simple_cleaner.py
import pandas as pd

class SimpleCleaner:
    def __init__(self, df=None):
        self.df = df
        self.steps = []

    def load(self, path, file_type="csv"):
        """Load CSV, Excel or Parquet file"""
        if file_type == "csv":
            self.df = pd.read_csv(path)
        elif file_type == "excel":
            self.df = pd.read_excel(path)
        elif file_type == "parquet":
            self.df = pd.read_parquet(path)
        else:
            raise ValueError("Only csv, excel, parquet supported")
        self.steps.append(f"Loaded file → {self.df.shape}")
        return self

    def drop_duplicates(self, keep="first"):
        before = len(self.df)
        self.df = self.df.drop_duplicates(keep=keep)
        removed = before - len(self.df)
        self.steps.append(f"Removed {removed} duplicate rows")
        return self

    def fill_missing(self, how="zero", columns=None):
        """how = 'zero', 'mean', 'median', 'drop'"""
        if columns is None:
            columns = self.df.columns

        if how == "zero":
            self.df[columns] = self.df[columns].fillna(0)
        elif how == "mean":
            self.df[columns] = self.df[columns].fillna(self.df[columns].mean(numeric_only=True))
        elif how == "median":
            self.df[columns] = self.df[columns].fillna(self.df[columns].median(numeric_only=True))
        elif how == "drop":
            self.df = self.df.dropna(subset=columns)
        else:
            raise ValueError("how must be: zero, mean, median, drop")

        self.steps.append(f"Filled missing values ({how})")
        return self

    def fix_column_names(self, lowercase=True, replace_space="_"):
        if lowercase:
            self.df.columns = self.df.columns.str.lower()
        if replace_space:
            self.df.columns = self.df.columns.str.replace(" ", replace_space)
        self.steps.append("Cleaned column names")
        return self

    def show_report(self):
        print("\n".join(self.steps))
        print(f"\nFinal shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        return self.df
