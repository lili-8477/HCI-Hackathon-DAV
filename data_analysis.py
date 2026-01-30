#!/usr/bin/env python3
"""
Enhanced Data Analysis Module for LLM Integration & Streamlit
- Comprehensive summary statistics with natural language descriptions
- Correlation analysis with interpretations
- PCA with variance explanation
- LLM-ready formatted outputs
- Streamlit-compatible visualizations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, List, Dict, Tuple, Union
import json


class DataAnalyzer:
    """
    Data analysis class optimized for LLM interpretation and Streamlit visualization.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with dataframe.
        
        Args:
            df: Input pandas DataFrame
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def get_dataset_overview(self) -> Dict[str, any]:
        """
        Get comprehensive dataset overview for LLM context.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            "shape": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            },
            "columns": {
                "numeric": self.numeric_cols,
                "categorical": self.categorical_cols,
                "total": self.df.columns.tolist()
            },
            "missing_values": self.df.isnull().sum().to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": self.df.dtypes.astype(str).to_dict()
        }
    
    def summary_stats(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate descriptive statistics with enhanced metrics.
        
        Args:
            columns: Specific columns to analyze (default: all numeric)
            
        Returns:
            DataFrame with comprehensive statistics
        """
        if columns is None:
            columns = self.numeric_cols
        
        if not columns:
            raise ValueError("No numeric columns found in dataset")
        
        stats = self.df[columns].describe()
        
        # Add additional statistics
        additional_stats = pd.DataFrame({
            col: {
                'missing': self.df[col].isnull().sum(),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'variance': self.df[col].var()
            } for col in columns
        })
        
        return pd.concat([stats, additional_stats])
    
    def get_summary_narrative(self, columns: Optional[List[str]] = None) -> str:
        """
        Generate natural language summary for LLM consumption.
        
        Args:
            columns: Specific columns to analyze
            
        Returns:
            Human-readable narrative of the data
        """
        if columns is None:
            columns = self.numeric_cols
        
        stats = self.summary_stats(columns)
        narratives = []
        
        for col in columns:
            mean_val = stats.loc['mean', col]
            std_val = stats.loc['std', col]
            min_val = stats.loc['min', col]
            max_val = stats.loc['max', col]
            missing = stats.loc['missing', col]
            
            narrative = (
                f"{col}: mean={mean_val:.2f}, std={std_val:.2f}, "
                f"range=[{min_val:.2f}, {max_val:.2f}], missing={int(missing)}"
            )
            narratives.append(narrative)
        
        return "\n".join(narratives)
    
    def compute_correlation(self, 
                          method: str = "pearson",
                          min_threshold: float = 0.0) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Compute correlation matrix with interpretable insights.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_threshold: Minimum absolute correlation to report
            
        Returns:
            Tuple of (correlation matrix, list of significant correlations)
        """
        if len(self.numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation")
        
        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        
        # Extract significant correlations
        significant_corrs = []
        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) >= min_threshold:
                    significant_corrs.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': round(corr_val, 3),
                        'strength': self._interpret_correlation(corr_val),
                        'direction': 'positive' if corr_val > 0 else 'negative'
                    })
        
        # Sort by absolute correlation
        significant_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return corr_matrix, significant_corrs
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very weak"
    
    def compute_pca(self, 
                   n_components: int = 2,
                   return_loadings: bool = True) -> Dict[str, any]:
        """
        Compute PCA with comprehensive output for interpretation.
        
        Args:
            n_components: Number of principal components
            return_loadings: Whether to include feature loadings
            
        Returns:
            Dictionary with PCA results and interpretations
        """
        if len(self.numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for PCA")
        
        # Handle missing values
        df_clean = self.df[self.numeric_cols].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No complete rows after removing missing values")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # Compute PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled_data)
        
        # Create results dataframe
        pca_df = pd.DataFrame(
            components,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=df_clean.index
        )
        
        # Prepare comprehensive output
        result = {
            'transformed_data': pca_df,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'total_variance_explained': sum(pca.explained_variance_ratio_),
            'n_components': n_components,
            'n_samples': len(df_clean),
            'n_features': len(self.numeric_cols)
        }
        
        if return_loadings:
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=self.numeric_cols
            )
            result['loadings'] = loadings
            result['top_contributors'] = self._get_top_contributors(loadings)
        
        return result
    
    def _get_top_contributors(self, loadings: pd.DataFrame, top_n: int = 3) -> Dict[str, List[Tuple]]:
        """Get top contributing features for each component."""
        contributors = {}
        for col in loadings.columns:
            top_features = loadings[col].abs().nlargest(top_n)
            contributors[col] = [
                (feat, round(loadings.loc[feat, col], 3))
                for feat in top_features.index
            ]
        return contributors
    
    def analyze_categorical(self, max_categories: int = 10) -> Dict[str, Dict]:
        """
        Analyze categorical variables.
        
        Args:
            max_categories: Maximum unique values to show
            
        Returns:
            Dictionary with categorical analysis
        """
        categorical_analysis = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            
            categorical_analysis[col] = {
                'unique_values': self.df[col].nunique(),
                'missing': self.df[col].isnull().sum(),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'top_categories': value_counts.head(max_categories).to_dict(),
                'mode': self.df[col].mode()[0] if not self.df[col].mode().empty else None
            }
        
        return categorical_analysis
    
    def generate_llm_report(self) -> str:
        """
        Generate comprehensive analysis report formatted for LLM consumption.
        
        Returns:
            Formatted text report
        """
        report_sections = []
        
        # Dataset Overview
        overview = self.get_dataset_overview()
        report_sections.append("=== DATASET OVERVIEW ===")
        report_sections.append(f"Rows: {overview['shape']['rows']}, Columns: {overview['shape']['columns']}")
        report_sections.append(f"Numeric columns: {', '.join(overview['columns']['numeric'])}")
        report_sections.append(f"Categorical columns: {', '.join(overview['columns']['categorical'])}")
        report_sections.append("")
        
        # Summary Statistics
        if self.numeric_cols:
            report_sections.append("=== NUMERIC SUMMARY ===")
            report_sections.append(self.get_summary_narrative())
            report_sections.append("")
        
        # Correlation Analysis
        if len(self.numeric_cols) >= 2:
            _, significant_corrs = self.compute_correlation(min_threshold=0.3)
            report_sections.append("=== SIGNIFICANT CORRELATIONS (|r| >= 0.3) ===")
            for corr in significant_corrs[:10]:  # Top 10
                report_sections.append(
                    f"{corr['variable_1']} <-> {corr['variable_2']}: "
                    f"{corr['correlation']} ({corr['strength']} {corr['direction']})"
                )
            report_sections.append("")
        
        # Categorical Analysis
        if self.categorical_cols:
            cat_analysis = self.analyze_categorical()
            report_sections.append("=== CATEGORICAL ANALYSIS ===")
            for col, stats in cat_analysis.items():
                report_sections.append(
                    f"{col}: {stats['unique_values']} unique values, "
                    f"mode={stats['mode']}, missing={stats['missing']}"
                )
            report_sections.append("")
        
        return "\n".join(report_sections)
    
    def export_for_llm(self) -> Dict[str, any]:
        """
        Export all analysis results in JSON-serializable format for LLM APIs.
        
        Returns:
            Dictionary with all analysis results
        """
        export_data = {
            'overview': self.get_dataset_overview(),
            'summary_narrative': self.get_summary_narrative() if self.numeric_cols else None,
            'full_report': self.generate_llm_report()
        }
        
        if len(self.numeric_cols) >= 2:
            corr_matrix, significant_corrs = self.compute_correlation()
            export_data['correlations'] = {
                'matrix': corr_matrix.to_dict(),
                'significant': significant_corrs
            }
        
        if self.categorical_cols:
            export_data['categorical_analysis'] = self.analyze_categorical()
        
        return export_data


# Standalone utility functions for backward compatibility
def summary_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Descriptive statistics for numeric columns."""
    analyzer = DataAnalyzer(df)
    return analyzer.summary_stats(columns)


def compute_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix of numeric columns."""
    analyzer = DataAnalyzer(df)
    corr_matrix, _ = analyzer.compute_correlation(method=method)
    return corr_matrix


def compute_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Compute PCA on numeric columns."""
    analyzer = DataAnalyzer(df)
    result = analyzer.compute_pca(n_components=n_components, return_loadings=False)
    return result['transformed_data']


if __name__ == "__main__":
    # Example usage
    print("Data Analysis Module - Ready for LLM Integration & Streamlit")
    print("Import DataAnalyzer class for full functionality")
