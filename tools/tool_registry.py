#!/usr/bin/env python3
"""
Tool Registry for Data Processing Pipeline
Exposes DataCleaner, DataAnalyzer, DataVisualizer in a discoverable, uniform way
Designed for LLM agents, Streamlit apps, modular pipelines, auto-documentation
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
import inspect
import pandas as pd

# ─── Import your three main classes ──────────────────────────────────────────
from data_cleaner import DataCleaner     # assuming file names / module structure
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer


@dataclass
class ToolMetadata:
    """Standardized metadata for each exposed method"""
    category: str                   # "cleaning", "analysis", "visualization"
    name: str                       # method name
    description: str                # first line of docstring or custom
    full_doc: str                   # complete docstring
    signature: inspect.Signature
    required_args: List[str]
    optional_args: List[str]
    returns: str                    # short description of return type
    instance_required: bool = True  # whether you need an instance first


class DataToolRegistry:
    """
    Central registry of data processing tools based on:
      • DataCleaner
      • DataAnalyzer
      • DataVisualizer
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolMetadata] = {}
        self._register_all()
        
    def _register_tool(self, category: str, func: Callable, instance_required: bool = True):
        """Register one method with metadata"""
        name = func.__name__
        
        doc = (func.__doc__ or "").strip()
        first_line = doc.split("\n")[0].strip() if doc else "No description"
        
        sig = inspect.signature(func)
        
        required = []
        optional = []
        for param in sig.parameters.values():
            if param.default is param.empty and param.name != "self":
                required.append(param.name)
            elif param.name != "self":
                optional.append(param.name)
                
        returns = "DataFrame" if "-> pd.DataFrame" in str(func.__annotations__) else "various"
        
        meta = ToolMetadata(
            category=category,
            name=name,
            description=first_line,
            full_doc=doc,
            signature=sig,
            required_args=required,
            optional_args=optional,
            returns=returns,
            instance_required=instance_required
        )
        
        key = f"{category}.{name}"
        self.tools[key] = meta
    
    def _register_all(self):
        """Register selected high-value / commonly used methods"""
        
        # ─── Cleaning ────────────────────────────────────────────────────────────
        cleaner_methods = [
            "load_data",
            "get_data_quality_report",
            "detect_duplicates",
            "remove_duplicates",
            "handle_missing_values",
            "detect_outliers",
            "handle_outliers",
            "infer_and_convert_types",
            "standardize_column_names",
            "remove_constant_columns",
            "clean_text_columns",
            "get_cleaning_summary",
            "export_cleaned_data",
            "export_for_llm",
        ]
        
        for meth_name in cleaner_methods:
            meth = getattr(DataCleaner, meth_name, None)
            if meth and callable(meth):
                self._register_tool("cleaning", meth, instance_required=True)
        
        # ─── Analysis ────────────────────────────────────────────────────────────
        analyzer_methods = [
            "get_dataset_overview",
            "summary_stats",
            "get_summary_narrative",
            "compute_correlation",
            "compute_pca",
            "analyze_categorical",
            "generate_llm_report",
            "export_for_llm",
        ]
        
        for meth_name in analyzer_methods:
            meth = getattr(DataAnalyzer, meth_name, None)
            if meth and callable(meth):
                self._register_tool("analysis", meth, instance_required=True)
        
        # ─── Visualization ───────────────────────────────────────────────────────
        viz_methods = [
            "plot_histogram",
            "plot_boxplot",
            "plot_scatter",
            "plot_bar",
            "plot_line",
            "plot_heatmap",
            "plot_pie",
            "plot_violin",
            "plot_pca",
            "create_dashboard",
            "describe_chart",
            "export_for_llm",
        ]
        
        for meth_name in viz_methods:
            meth = getattr(DataVisualizer, meth_name, None)
            if meth and callable(meth):
                self._register_tool("visualization", meth, instance_required=True)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of available tools (optionally filtered by category)"""
        result = []
        for key, meta in sorted(self.tools.items()):
            if category is None or key.startswith(category + "."):
                result.append({
                    "tool": key,
                    "category": meta.category,
                    "name": meta.name,
                    "description": meta.description,
                    "required_args": meta.required_args,
                    "optional_args": meta.optional_args,
                    "returns": meta.returns
                })
        return result
    
    def get_tool_info(self, tool_key: str) -> Optional[ToolMetadata]:
        """Get full metadata for one tool"""
        return self.tools.get(tool_key)
    
    def create_pipeline(self) -> Dict[str, Any]:
        """Return a minimal pipeline-ready structure"""
        return {
            "cleaner": DataCleaner(),
            "analyzer": None,   # needs data
            "visualizer": None, # needs data
            "current_df": None
        }
    
    def print_registry_summary(self):
        """Pretty-print overview of registered tools"""
        from collections import defaultdict
        by_cat = defaultdict(list)
        for key in sorted(self.tools):
            cat, name = key.split(".", 1)
            by_cat[cat].append(name)
        
        print("┌──────────────────────────────────────────────────────────────┐")
        print("│                  Registered Data Tools                       │")
        print("├───────────────┬──────────────────────────────────────────────┤")
        for cat, methods in by_cat.items():
            print(f"│ {cat:13} │ {', '.join(methods[:3])} ... ({len(methods)}) │")
        print("└───────────────┴──────────────────────────────────────────────┘")
        print(f"Total tools registered: {len(self.tools)}")


# Singleton / global access
_tool_registry = DataToolRegistry()


def get_registry() -> DataToolRegistry:
    """Get the global tool registry instance"""
    return _tool_registry


if __name__ == "__main__":
    registry = get_registry()
    registry.print_registry_summary()
    
    print("\nExample: available cleaning tools:")
    for t in registry.list_tools("cleaning")[:6]:
        print(f"  • {t['tool']:35}  {t['description'][:58]}...")
