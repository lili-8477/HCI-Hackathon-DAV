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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, List, Dict, Tuple, Union, Any
import json

class DataAnalyzer:
    """
    Data analysis class with new features for clustering, regression, time series, and more.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Existing methods remain (get_dataset_overview, summary_stats, etc.)
    
    # NEW: Clustering
    def perform_clustering(self,
                           n_clusters: int = 3,
                           columns: Optional[List[str]] = None,
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Perform KMeans clustering on numeric columns.
        
        Args:
            n_clusters: Number of clusters
            columns: Specific columns to use (default: all numeric)
            random_state: For reproducibility
            
        Returns:
            Dictionary with cluster labels and centers
        """
        if columns is None:
            columns = self.numeric_cols
        
        if len(columns) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")
        
        # Prepare data
        df_clean = self.df[columns].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        result = {
            'cluster_labels': labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'n_samples': len(df_clean),
            'columns_used': columns
        }
        
        # Add cluster to original df (for indices without NaN)
        cluster_df = pd.DataFrame({'cluster': labels}, index=df_clean.index)
        
        return result
    
    # NEW: Regression Analysis
    def perform_regression(self,
                           target: str,
                           features: Optional[List[str]] = None,
                           model_type: str = 'linear') -> Dict[str, Any]:
        """
        Perform regression analysis.
        
        Args:
            target: Target column
            features: Predictor columns (default: all other numeric)
            model_type: 'linear' or 'random_forest'
            
        Returns:
            Dictionary with coefficients, metrics, and predictions
        """
        if features is None:
            features = [col for col in self.numeric_cols if col != target]
        
        if not features:
            raise ValueError("No features available for regression")
        
        # Prepare data
        df_clean = self.df[features + [target]].dropna()
        X = df_clean[features]
        y = df_clean[target]
        
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)
            result = {
                'coefficients': dict(zip(features, model.coef_)),
                'intercept': model.intercept_,
                'r2_score': r2_score(y, preds),
                'model_type': model_type
            }
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            result = {
                'feature_importances': dict(zip(features, model.feature_importances_)),
                'r2_score': r2_score(y, preds),
                'model_type': model_type
            }
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        result['n_samples'] = len(df_clean)
        result['predictions'] = preds.tolist()
        
        return result
    
    # NEW: Time Series Decomposition
    def time_series_decomposition(self,
                                 column: str,
                                 period: int = 12,
                                 model: str = 'additive') -> Dict[str, Any]:
        """
        Perform seasonal decomposition on a time series column.
        
        Args:
            column: Numeric column to decompose (assumes index is datetime)
            period: Seasonal period
            model: 'additive' or 'multiplicative'
            
        Returns:
            Dictionary with trend, seasonal, residual components
        """
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            raise ValueError("DataFrame index must be datetime for time series analysis")
        
        ts = self.df[column].dropna()
        if len(ts) < 2 * period:
            raise ValueError("Time series too short for decomposition")
        
        decomp = seasonal_decompose(ts, model=model, period=period)
        
        return {
            'trend': decomp.trend.dropna().tolist(),
            'seasonal': decomp.seasonal.dropna().tolist(),
            'residual': decomp.resid.dropna().tolist(),
            'observed': decomp.observed.dropna().tolist(),
            'indices': decomp.trend.dropna().index.strftime('%Y-%m-%d').tolist()
        }
    
    # ENHANCED: PCA with biplot option (for visualization integration)
    def compute_pca(self,
                   n_components: int = 2,
                   return_loadings: bool = True,
                   biplot: bool = False) -> Dict[str, Any]:
        """
        Compute PCA with optional biplot data.
        """
        result = super().compute_pca(n_components, return_loadings)  # Assuming inheritance or copy existing
        
        if biplot:
            # Add scaled loadings for biplot arrows
            if 'loadings' in result:
                scalings = np.sqrt(result['explained_variance'])  # Scale by sqrt(eigenvalues)
                result['biplot_loadings'] = (result['loadings'] * scalings).to_dict()
        
        return result

# ... rest of the class ...
