"""
Data Agent
Automatically profiles and analyzes datasets when loaded.
"""

import pandas as pd


class DataAgent:
    """Agent for automatic data profiling"""

    def analyze(self, df: pd.DataFrame) -> str:
        """
        Perform comprehensive data profiling.

        Args:
            df: DataFrame to analyze

        Returns:
            Formatted summary string
        """
        summary = []

        # Basic info
        summary.append("=" * 60)
        summary.append("DATA PROFILE")
        summary.append("=" * 60)

        summary.append(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Column list
        summary.append(f"\nColumns ({len(df.columns)}):")
        summary.append(", ".join(df.columns.tolist()))

        # Data types
        summary.append(f"\nData Types:")
        summary.append(df.dtypes.to_string())

        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            summary.append(f"\nMissing Values:")
            missing_df = pd.DataFrame({
                'Count': missing[missing > 0],
                'Percentage': (missing[missing > 0] / len(df) * 100).round(2)
            })
            summary.append(missing_df.to_string())
        else:
            summary.append(f"\nMissing Values: None")

        # Summary statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary.append(f"\nSummary Statistics (Numeric Columns):")
            summary.append(df[numeric_cols].describe().to_string())

        # Categorical columns info
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            summary.append(f"\nCategorical Columns ({len(cat_cols)}):")
            for col in cat_cols:
                n_unique = df[col].nunique()
                summary.append(f"  {col}: {n_unique} unique values")

        summary.append("\n" + "=" * 60)

        return "\n".join(summary)
