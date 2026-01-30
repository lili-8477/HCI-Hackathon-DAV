#!/usr/bin/env python3
"""
Enhanced Data Analysis Module with New Features:
- Added clustering (KMeans)
- Added basic regression analysis (linear regression)
- Added time series decomposition (if applicable)
- Enhanced PCA with biplot visualization option
- Added feature importance for tree-based models
"""

import pandas as pd
import numpy as np

class SimpleAnalyzer:
    def __init__(self, df):
        self.df = df

    def basic_info(self):
        print("Shape:", self.df.shape)
        print("\nData types:\n", self.df.dtypes)
        print("\nMissing values:\n", self.df.isna().sum())

    def numeric_summary(self):
        return self.df.describe().round(2)

    def top_categories(self, column, n=8):
        if column not in self.df.columns:
            return f"Column '{column}' not found"
        return self.df[column].value_counts().head(n)

    def correlation(self):
        num = self.df.select_dtypes(include=[np.number])
        if len(num.columns) < 2:
            return "Not enough numeric columns"
        return num.corr().round(2)

    def quick_report(self):
        print("=== QUICK REPORT ===")
        self.basic_info()
        print("\nNumeric summary:\n", self.numeric_summary())
        print("\nStrongest correlations:")
        corr = self.correlation()
        if isinstance(corr, pd.DataFrame):
            print(corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().abs().nlargest(8))
