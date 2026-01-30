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

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Any, Literal
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from rapidfuzz import fuzz, process
import re
import json

warnings.filterwarnings("ignore", category=UserWarning)


class DataCleaner:
    """
    Modern data cleaning class with improved strategies and better reporting.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df.copy() if df is not None else None
        self.original_df = df.copy() if df is not None else None
        self.history: List[Dict[str, Any]] = []
        self.removed_rows_indices: set = set()   # for potential undo

    def load(self,
             source: Union[str, Path, pd.DataFrame],
             **kwargs) -> pd.DataFrame:
        """Unified load method supporting multiple formats"""
        if isinstance(source, (str, Path)):
            p = Path(source)
            ext = p.suffix.lower()
            loaders = {
                '.csv': pd.read_csv,
                '.tsv': lambda x: pd.read_csv(x, sep='\t'),
                '.xlsx': pd.read_excel,
                '.parquet': pd.read_parquet,
                '.json': pd.read_json,
            }
            if ext not in loaders:
                raise ValueError(f"Unsupported file type: {ext}")
            self.df = loaders[ext](p, **kwargs)
        elif isinstance(source, pd.DataFrame):
            self.df = source.copy()
        else:
            raise TypeError("source must be path or DataFrame")

        self.original_df = self.df.copy()
        self._record("load", f"Loaded shape {self.df.shape}")
        return self.df

    # ─── Duplicate handling ────────────────────────────────────────

    def find_duplicates(self,
                       subset: Optional[List[str]] = None,
                       fuzzy: bool = False,
                       fuzzy_threshold: float = 0.92,
                       fuzzy_column: Optional[str] = None) -> pd.DataFrame:
        """Exact + fuzzy duplicate detection"""
        if fuzzy and fuzzy_column:
            col = fuzzy_column
            dups = []
            values = self.df[col].astype(str).tolist()
            for i, val in enumerate(values):
                if i in dups:
                    continue
                matches = process.extract(val, values[i+1:], scorer=fuzz.token_sort_ratio)
                for match, score, idx in matches:
                    if score >= fuzzy_threshold * 100:
                        dups.append(i + 1 + idx)
            return self.df.iloc[dups]
        else:
            return self.df[self.df.duplicated(subset=subset, keep=False)]

    def drop_duplicates(self,
                       subset: Optional[List[str]] = None,
                       keep: Literal['first', 'last', False] = 'first',
                       fuzzy: bool = False,
                       fuzzy_column: Optional[str] = None) -> pd.DataFrame:
        initial_n = len(self.df)
        if fuzzy and fuzzy_column:
            to_drop = self.find_duplicates(fuzzy=True, fuzzy_column=fuzzy_column).index
            self.df = self.df[~self.df.index.isin(to_drop)]
        else:
            self.df = self.df.drop_duplicates(subset=subset, keep=keep)

        removed = initial_n - len(self.df)
        self._record("drop_duplicates", f"Removed {removed} rows (fuzzy={fuzzy})")
        return self.df

    # ─── Missing values ────────────────────────────────────────────

    def impute_missing(self,
                      strategy: Literal['drop_rows', 'drop_cols', 'mean', 'median', 'mode', 'constant', 'knn', 'mice', 'interpolate'] = 'mean',
                      columns: Optional[List[str]] = None,
                      knn_neighbors: int = 5,
                      mice_iterations: int = 10,
                      constant_value: Any = None,
                      col_drop_threshold: float = 0.75) -> pd.DataFrame:
        if columns is None:
            columns = self.df.columns

        # Drop columns first if requested
        if strategy == 'drop_cols' or col_drop_threshold > 0:
            miss_rate = self.df[columns].isna().mean()
            to_drop = miss_rate[miss_rate >= col_drop_threshold].index
            if to_drop.any():
                self.df.drop(columns=to_drop, inplace=True)
                self._record("drop_columns_high_missing", f"Dropped {len(to_drop)} columns")

        if strategy == 'drop_rows':
            self.df.dropna(subset=columns, inplace=True)
        elif strategy in ['mean', 'median', 'most_frequent']:
            imp = SimpleImputer(strategy=strategy)
            self.df[columns] = imp.fit_transform(self.df[columns])
        elif strategy == 'constant':
            self.df[columns] = self.df[columns].fillna(constant_value)
        elif strategy == 'knn':
            imp = KNNImputer(n_neighbors=knn_neighbors)
            self.df[columns] = imp.fit_transform(self.df[columns])
        elif strategy == 'mice':
            imp = IterativeImputer(max_iter=mice_iterations, random_state=42)
            self.df[columns] = imp.fit_transform(self.df[columns])
        elif strategy == 'interpolate':
            self.df[columns] = self.df[columns].interpolate(method='linear', limit_direction='both')

        self._record("impute_missing", f"Strategy: {strategy}, columns: {len(columns)}")
        return self.df

    # ─── Outliers ──────────────────────────────────────────────────

    def detect_and_handle_outliers(self,
                                  method: Literal['iqr', 'zscore', 'isolation_forest'] = 'iqr',
                                  threshold: float = 1.5,
                                  action: Literal['clip', 'remove', 'flag'] = 'clip',
                                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns

        for col in columns:
            ser = self.df[col].dropna()

            if method == 'iqr':
                q1, q3 = ser.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
                mask = (self.df[col] < lower) | (self.df[col] > upper)
            elif method == 'zscore':
                z = np.abs((ser - ser.mean()) / ser.std())
                mask = z > threshold
                mask = mask.reindex(self.df.index, fill_value=False)
            elif method == 'isolation_forest':
                iso = IsolationForest(contamination='auto', random_state=42)
                pred = iso.fit_predict(ser.values.reshape(-1, 1))
                mask = pred == -1
                mask = pd.Series(mask, index=ser.index).reindex(self.df.index, fill_value=False)

            outlier_count = mask.sum()

            if action == 'clip':
                self.df[col] = self.df[col].clip(lower, upper)
            elif action == 'remove':
                self.df = self.df[~mask]
            elif action == 'flag':
                self.df[f"{col}_is_outlier"] = mask.astype(int)

            if outlier_count > 0:
                self._record("outliers", f"{col}: {outlier_count} handled ({method} / {action})")

        return self.df

    # ─── Text cleaning ─────────────────────────────────────────────

    def clean_text_columns(self,
                          columns: Optional[List[str]] = None,
                          lowercase: bool = True,
                          remove_urls: bool = True,
                          remove_html: bool = True,
                          remove_punctuation: bool = True,
                          normalize_whitespace: bool = True) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'string']).columns

        for col in columns:
            s = self.df[col].astype(str)
            if remove_urls:
                s = s.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
            if remove_html:
                s = s.str.replace(r'<[^>]+>', '', regex=True)
            if lowercase:
                s = s.str.lower()
            if remove_p
