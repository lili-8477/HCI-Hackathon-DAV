"""
Data State Management
Singleton class to manage the current dataset state across all tools and agents.
"""

import pandas as pd
from typing import Optional


class DataState:
    """Singleton class to manage current dataset state"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.df = None
            cls._instance.file_path = None
            cls._instance.file_name = None
            cls._instance.pending_figures = []
            cls._instance.intermediate_tables = {}
            cls._instance._table_counters = {}
        return cls._instance

    def load_data(self, df: pd.DataFrame, file_path: str, file_name: str):
        """Load a new dataset into state"""
        self.df = df
        self.file_path = file_path
        self.file_name = file_name
        self.intermediate_tables = {}
        self._table_counters = {}
        self.save_intermediate("original", df.copy())

    def get_dataframe(self) -> pd.DataFrame:
        """Get the current dataframe"""
        if self.df is None:
            raise ValueError("No data loaded. Please upload a dataset first.")
        return self.df

    def is_data_loaded(self) -> bool:
        """Check if data is loaded"""
        return self.df is not None

    def get_file_info(self) -> dict:
        """Get information about the loaded file"""
        if not self.is_data_loaded():
            return {"loaded": False}
        return {
            "loaded": True,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "rows": self.df.shape[0],
            "columns": self.df.shape[1]
        }

    def add_figure(self, fig_type: str, fig):
        """Store a figure for later display in Streamlit"""
        self.pending_figures.append((fig_type, fig))

    def pop_figures(self) -> list:
        """Retrieve and clear all pending figures"""
        figs = self.pending_figures
        self.pending_figures = []
        return figs

    def next_table_name(self, prefix: str) -> str:
        """Generate an auto-incrementing table name like 'filtered_1'"""
        count = self._table_counters.get(prefix, 0) + 1
        self._table_counters[prefix] = count
        return f"{prefix}_{count}"

    def save_intermediate(self, name: str, df: pd.DataFrame):
        """Store a named intermediate table"""
        self.intermediate_tables[name] = df

    def get_intermediate(self, name: str) -> pd.DataFrame:
        """Retrieve an intermediate table by name"""
        if name not in self.intermediate_tables:
            available = list(self.intermediate_tables.keys())
            raise ValueError(f"Table '{name}' not found. Available tables: {available}")
        return self.intermediate_tables[name]

    def list_intermediates(self) -> list:
        """Return list of intermediate table info dicts"""
        return [
            {"name": name, "rows": df.shape[0], "columns": df.shape[1]}
            for name, df in self.intermediate_tables.items()
        ]

    def use_intermediate(self, name: str):
        """Copy an intermediate table into self.df so all existing tools operate on it"""
        self.df = self.get_intermediate(name).copy()

    def clear(self):
        """Clear the current dataset"""
        self.df = None
        self.file_path = None
        self.file_name = None
        self.pending_figures = []
        self.intermediate_tables = {}
        self._table_counters = {}
