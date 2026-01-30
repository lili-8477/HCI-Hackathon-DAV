#!/usr/bin/env python3
"""
Enhanced Data Cleaning Module for LLM Integration & Streamlit
- Multi-format data loading (CSV, Excel, JSON, Parquet)
- Intelligent duplicate detection and handling
- Advanced missing value strategies
- Data type inference and conversion
- Outlier detection and handling
- Data validation and quality reports
- LLM-ready cleaning summaries
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive data cleaning class optimized for LLM interpretation and Streamlit.
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize cleaner with optional dataframe.
        
        Args:
            df: Input pandas DataFrame (optional)
        """
        self.df = df
        self.original_df = df.copy() if df is not None else None
        self.cleaning_log = []
        self.quality_issues = {}
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats with enhanced error handling.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                self.df = pd.read_csv(file_path, **kwargs)
                self._log_action(f"Loaded CSV file: {file_path.name}")
                
            elif extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path, **kwargs)
                self._log_action(f"Loaded Excel file: {file_path.name}")
                
            elif extension == '.json':
                self.df = pd.read_json(file_path, **kwargs)
                self._log_action(f"Loaded JSON file: {file_path.name}")
                
            elif extension == '.parquet':
                self.df = pd.read_parquet(file_path, **kwargs)
                self._log_action(f"Loaded Parquet file: {file_path.name}")
                
            elif extension == '.tsv':
                self.df = pd.read_csv(file_path, sep='\t', **kwargs)
                self._log_action(f"Loaded TSV file: {file_path.name}")
                
            else:
                raise ValueError(
                    f"Unsupported file type: {extension}. "
                    "Supported: .csv, .xlsx, .xls, .json, .parquet, .tsv"
                )
            
            # Store original copy
            self.original_df = self.df.copy()
            
            # Log basic info
            self._log_action(
                f"Dataset loaded: {len(self.df)} rows × {len(self.df.columns)} columns"
            )
            
            return self.df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Dictionary with quality metrics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_percentage': (self.df.duplicated().sum() / len(self.df)) * 100,
            'columns_info': {},
            'missing_data_summary': {},
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Analyze each column
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'missing_count': int(self.df[col].isnull().sum()),
                'missing_percentage': float((self.df[col].isnull().sum() / len(self.df)) * 100),
                'unique_values': int(self.df[col].nunique()),
                'unique_percentage': float((self.df[col].nunique() / len(self.df)) * 100)
            }
            
            # Add numeric-specific stats
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info.update({
                    'mean': float(self.df[col].mean()) if not self.df[col].isnull().all() else None,
                    'median': float(self.df[col].median()) if not self.df[col].isnull().all() else None,
                    'std': float(self.df[col].std()) if not self.df[col].isnull().all() else None,
                    'min': float(self.df[col].min()) if not self.df[col].isnull().all() else None,
                    'max': float(self.df[col].max()) if not self.df[col].isnull().all() else None,
                    'zeros_count': int((self.df[col] == 0).sum()),
                    'negative_count': int((self.df[col] < 0).sum()) if not self.df[col].isnull().all() else 0
                })
            
            report['columns_info'][col] = col_info
        
        # Missing data summary
        missing_cols = {
            col: int(count) 
            for col, count in self.df.isnull().sum().items() 
            if count > 0
        }
        report['missing_data_summary'] = missing_cols
        
        return report
    
    def detect_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect duplicate rows with detailed analysis.
        
        Args:
            subset: Columns to consider for duplicates (default: all columns)
            
        Returns:
            DataFrame containing duplicate rows
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        duplicates = self.df[self.df.duplicated(subset=subset, keep=False)]
        
        self._log_action(
            f"Found {len(duplicates)} duplicate rows "
            f"({(len(duplicates) / len(self.df) * 100):.2f}%)"
        )
        
        return duplicates.sort_values(by=subset if subset else self.df.columns.tolist())
    
    def remove_duplicates(self, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        
        self._log_action(
            f"Removed {removed} duplicate rows (kept='{keep}'). "
            f"Remaining: {len(self.df)} rows"
        )
        
        return self.df
    
    def handle_missing_values(self,
                             strategy: str = 'auto',
                             numeric_method: str = 'mean',
                             categorical_method: str = 'mode',
                             threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values with multiple strategies.
        
        Args:
            strategy: 'auto', 'drop', 'fill', or 'advanced'
            numeric_method: Method for numeric columns ('mean', 'median', 'mode', 'ffill', 'bfill', 'interpolate')
            categorical_method: Method for categorical columns ('mode', 'constant', 'ffill', 'bfill')
            threshold: Drop columns with missing % above this (0.0-1.0)
            
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        initial_missing = self.df.isnull().sum().sum()
        
        if initial_missing == 0:
            self._log_action("No missing values found")
            return self.df
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'drop':
            # Drop rows with any missing values
            initial_rows = len(self.df)
            self.df = self.df.dropna()
            self._log_action(
                f"Dropped {initial_rows - len(self.df)} rows with missing values"
            )
            
        elif strategy == 'auto' or strategy == 'fill':
            # Drop columns with too many missing values
            missing_pct = self.df.isnull().sum() / len(self.df)
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                self._log_action(
                    f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}"
                )
            
            # Fill numeric columns
            for col in numeric_cols:
                if col in self.df.columns and self.df[col].isnull().any():
                    if numeric_method == 'mean':
                        fill_value = self.df[col].mean()
                    elif numeric_method == 'median':
                        fill_value = self.df[col].median()
                    elif numeric_method == 'mode':
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                    elif numeric_method == 'ffill':
                        self.df[col] = self.df[col].fillna(method='ffill')
                        continue
                    elif numeric_method == 'bfill':
                        self.df[col] = self.df[col].fillna(method='bfill')
                        continue
                    elif numeric_method == 'interpolate':
                        self.df[col] = self.df[col].interpolate()
                        continue
                    else:
                        fill_value = 0
                    
                    self.df[col] = self.df[col].fillna(fill_value)
            
            # Fill categorical columns
            for col in categorical_cols:
                if col in self.df.columns and self.df[col].isnull().any():
                    if categorical_method == 'mode':
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    elif categorical_method == 'constant':
                        fill_value = 'Unknown'
                    elif categorical_method == 'ffill':
                        self.df[col] = self.df[col].fillna(method='ffill')
                        continue
                    elif categorical_method == 'bfill':
                        self.df[col] = self.df[col].fillna(method='bfill')
                        continue
                    else:
                        fill_value = 'Unknown'
                    
                    self.df[col] = self.df[col].fillna(fill_value)
            
            final_missing = self.df.isnull().sum().sum()
            self._log_action(
                f"Filled missing values: {initial_missing} → {final_missing} "
                f"(numeric: {numeric_method}, categorical: {categorical_method})"
            )
        
        return self.df
    
    def detect_outliers(self, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, pd.Series]:
        """
        Detect outliers in numeric columns.
        
        Args:
            columns: Columns to check (default: all numeric)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            Dictionary mapping column names to boolean Series indicating outliers
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers[col] = z_scores > threshold
        
        total_outliers = sum(mask.sum() for mask in outliers.values())
        self._log_action(f"Detected {total_outliers} outliers using {method} method")
        
        return outliers
    
    def handle_outliers(self,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       action: str = 'cap',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            columns: Columns to process
            method: Detection method ('iqr' or 'zscore')
            action: 'cap' (winsorize), 'remove', or 'flag'
            threshold: Detection threshold
            
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        outliers = self.detect_outliers(columns, method, threshold)
        
        for col, mask in outliers.items():
            if action == 'remove':
                initial_rows = len(self.df)
                self.df = self.df[~mask]
                self._log_action(f"Removed {initial_rows - len(self.df)} outlier rows from {col}")
                
            elif action == 'cap':
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    self._log_action(f"Capped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
            elif action == 'flag':
                flag_col = f'{col}_outlier_flag'
                self.df[flag_col] = mask
                self._log_action(f"Flagged {mask.sum()} outliers in {col}")
        
        return self.df
    
    def infer_and_convert_types(self, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Intelligently infer and convert data types.
        
        Args:
            date_columns: Columns to attempt datetime conversion
            
        Returns:
            DataFrame with optimized types
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        conversions = []
        
        # Convert object columns to categorical if beneficial
        for col in self.df.select_dtypes(include=['object']).columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                self.df[col] = self.df[col].astype('category')
                conversions.append(f"{col}: object → category")
        
        # Try to convert to numeric
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                # Try numeric conversion
                converted = pd.to_numeric(self.df[col], errors='coerce')
                if converted.notna().sum() / len(self.df) > 0.8:  # 80% success rate
                    self.df[col] = converted
                    conversions.append(f"{col}: object → numeric")
            except:
                pass
        
        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        conversions.append(f"{col}: → datetime")
                    except:
                        pass
        
        if conversions:
            self._log_action(f"Type conversions: {', '.join(conversions)}")
        
        return self.df
    
    def standardize_column_names(self, 
                                 style: str = 'snake_case',
                                 remove_special: bool = True) -> pd.DataFrame:
        """
        Standardize column names.
        
        Args:
            style: 'snake_case', 'camelCase', or 'lower'
            remove_special: Remove special characters
            
        Returns:
            DataFrame with standardized column names
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        old_names = self.df.columns.tolist()
        new_names = []
        
        for col in old_names:
            new_col = col
            
            if remove_special:
                # Remove special characters
                new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
            
            if style == 'snake_case':
                # Convert to snake_case
                new_col = new_col.lower().replace(' ', '_').replace('-', '_')
                # Remove consecutive underscores
                while '__' in new_col:
                    new_col = new_col.replace('__', '_')
                new_col = new_col.strip('_')
                
            elif style == 'camelCase':
                words = new_col.replace('_', ' ').replace('-', ' ').split()
                new_col = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
                
            elif style == 'lower':
                new_col = new_col.lower()
            
            new_names.append(new_col)
        
        self.df.columns = new_names
        
        if old_names != new_names:
            changes = [f"{old} → {new}" for old, new in zip(old_names, new_names) if old != new]
            self._log_action(f"Standardized column names ({style}): {len(changes)} changes")
        
        return self.df
    
    def remove_constant_columns(self) -> pd.DataFrame:
        """Remove columns with only one unique value."""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        
        if constant_cols:
            self.df = self.df.drop(columns=constant_cols)
            self._log_action(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
        
        return self.df
    
    def clean_text_columns(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean text in string columns.
        
        Args:
            columns: Columns to clean (default: all object columns)
            
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                # Strip whitespace
                self.df[col] = self.df[col].str.strip()
                # Remove multiple spaces
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
        
        self._log_action(f"Cleaned text in {len(columns)} columns")
        
        return self.df
    
    def get_cleaning_summary(self) -> str:
        """
        Generate natural language summary of cleaning operations.
        
        Returns:
            Human-readable cleaning summary
        """
        if not self.cleaning_log:
            return "No cleaning operations performed yet."
        
        summary = ["=== DATA CLEANING SUMMARY ===\n"]
        
        if self.original_df is not None:
            summary.append(
                f"Original dataset: {len(self.original_df)} rows × {len(self.original_df.columns)} columns"
            )
            summary.append(
                f"Current dataset: {len(self.df)} rows × {len(self.df.columns)} columns"
            )
            summary.append(
                f"Rows removed: {len(self.original_df) - len(self.df)} "
                f"({((len(self.original_df) - len(self.df)) / len(self.original_df) * 100):.2f}%)"
            )
            summary.append("")
        
        summary.append("Cleaning operations performed:")
        for i, action in enumerate(self.cleaning_log, 1):
            summary.append(f"{i}. {action}")
        
        return "\n".join(summary)
    
    def export_cleaned_data(self, 
                           output_path: Union[str, Path],
                           format: str = 'csv',
                           **kwargs) -> str:
        """
        Export cleaned data to file.
        
        Args:
            output_path: Path for output file
            format: 'csv', 'excel', 'json', or 'parquet'
            **kwargs: Additional arguments for export function
            
        Returns:
            Path to exported file
        """
        if self.df is None:
            raise ValueError("No data to export.")
        
        output_path = Path(output_path)
        
        if format == 'csv':
            self.df.to_csv(output_path, index=False, **kwargs)
        elif format == 'excel':
            self.df.to_excel(output_path, index=False, **kwargs)
        elif format == 'json':
            self.df.to_json(output_path, **kwargs)
        elif format == 'parquet':
            self.df.to_parquet(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self._log_action(f"Exported cleaned data to {output_path}")
        
        return str(output_path)
    
    def export_for_llm(self) -> Dict[str, Any]:
        """
        Export cleaning report in JSON format for LLM consumption.
        
        Returns:
            Dictionary with cleaning metadata and summary
        """
        return {
            'cleaning_summary': self.get_cleaning_summary(),
            'cleaning_log': self.cleaning_log,
            'data_quality_report': self.get_data_quality_report() if self.df is not None else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_action(self, action: str):
        """Log a cleaning action."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}"
        self.cleaning_log.append(log_entry)


# Standalone utility functions for backward compatibility
def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV, Excel, or JSON data."""
    cleaner = DataCleaner()
    return cleaner.load_data(file_path)


def preprocess_data(df: pd.DataFrame, auto_fill: bool = True) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    
    if auto_fill:
        cleaner.handle_missing_values(strategy='auto')
    
    return cleaner.df


def select_columns(df: pd.DataFrame, 
                  col_type: str = "numeric",
                  return_all: bool = False) -> List[str]:
    """Select columns by type."""
    if col_type == "numeric":
        cols = df.select_dtypes(include='number').columns.tolist()
    else:
        cols = df.select_dtypes(include='object').columns.tolist()
    
    if return_all:
        return cols
    
    return cols


if __name__ == "__main__":
    print("Data Cleaning Module - Ready for LLM Integration & Streamlit")
    print("Import DataCleaner class for full functionality")
