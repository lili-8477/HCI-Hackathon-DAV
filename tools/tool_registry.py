#!/usr/bin/env python3
"""
Central Tool Registry for the Data Processing Suite
Registers high-level callable methods from:
  • DataCleaner
  • DataAnalyzer
  • DataVisualizer

Version: 2025-style edition
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
import inspect
import pandas as pd

# ─── Import the three core classes ───────────────────────────────────────────
# Adjust import paths according to your actual file/module structure
from data_cleaner   import DataCleaner
from data_analyzer  import DataAnalyzer
from data_visualizer import DataVisualizer


@dataclass
class RegisteredTool:
    """Metadata for one registered method"""
    category: str                   # clean / analyze / visualize
    method_name: str
    short_description: str
    full_docstring: str
    required_parameters: List[str]
    optional_parameters: List[str]
    returns_dataframe: bool         # heuristic – helps agents decide chaining
    instance_required: bool = True  # almost always True for these classes


class ToolRegistry:
    """
    Centralized registry of data processing tools.
    Makes methods discoverable and callable in a uniform way.
    """

    def __init__(self):
        self.tools: Dict[str, RegisteredTool] = {}
        self._register_all()

    def _register(self,
                 category: str,
                 cls: type,
                 method_names: List[str]) -> None:
        """Register selected methods from a class"""
        for name in method_names:
            if not hasattr(cls, name):
                continue

            func = getattr(cls, name)
            if not callable(func):
                continue

            doc = (func.__doc__ or "").strip()
            short = doc.split('\n', 1)[0].strip() if doc else "(no description)"

            sig = inspect.signature(func)

            required = []
            optional = []
            for param in sig.parameters.values():
                if param.name == "self":
                    continue
                if param.default is param.empty:
                    required.append(param.name)
                else:
                    optional.append(param.name)

            returns_df = (
                "pd.DataFrame" in str(func.__annotations__.get("return", "")) or
                "-> pd.DataFrame" in doc or
                "Figure" in str(func.__annotations__.get("return", "")) or
                "plot" in name.lower()
            )

            tool_key = f"{category}.{name}"

            self.tools[tool_key] = RegisteredTool(
                category=category,
                method_name=name,
                short_description=short,
                full_docstring=doc,
                required_parameters=required,
                optional_parameters=optional,
                returns_dataframe=returns_df,
                instance_required=True
            )

    def _register_all(self):
        """Register the most useful / commonly chained methods"""

        # ─── Cleaning ────────────────────────────────────────────────────────
        self._register("clean", DataCleaner, [
            "load",
            "find_duplicates",
            "drop_duplicates",
            "impute_missing",
            "detect_and_handle_outliers",
            "clean_text_columns",
            "standardize_column_names",
            "infer_and_convert_dtypes",
            "get_cleaning_summary",
        ])

        # ─── Analysis ────────────────────────────────────────────────────────
        self._register("analyze", DataAnalyzer, [
            "get_dataset_overview",
            "summary_stats",
            "get_summary_narrative",
            "compute_correlation",
            "compute_pca",
            "perform_clustering",
            "perform_regression",
            "time_series_decomposition",
            "analyze_categorical",
            "generate_llm_report",
        ])

        # ─── Visualization ───────────────────────────────────────────────────
        self._register("visualize", DataVisualizer, [
            "plot_histogram",
            "plot_boxplot",
            "plot_scatter",
            "plot_bar",
            "plot_line",
            "plot_heatmap",
            "plot_pie",
            "plot_violin",
            "plot_pairplot",
            "plot_3d_scatter",
            "plot_pca_biplot",
            "plot_time_series_decomp",
            "create_dashboard",
            "save_figure",
        ])

    def list_tools(self,
                  category: Optional[str] = None,
                  returns_df_only: bool = False) -> List[Dict[str, Any]]:
        """List available tools – filterable"""
        result = []
        for key, tool in sorted(self.tools.items()):
            if category and not key.startswith(f"{category}."):
                continue
            if returns_df_only and not tool.returns_dataframe:
                continue

            result.append({
                "tool_key": key,
                "category": tool.category,
                "method": tool.method_name,
                "description": tool.short_description,
                "required": tool.required_parameters,
                "optional": tool.optional_parameters,
                "returns_dataframe": tool.returns_dataframe
            })
        return result

    def get_tool(self, tool_key: str) -> Optional[RegisteredTool]:
        """Get full metadata for one tool"""
        return self.tools.get(tool_key)

    def create_instances(self) -> Dict[str, Any]:
        """Helper – create fresh instances (useful for pipelines)"""
        return {
            "cleaner":    DataCleaner(),
            "analyzer":   None,   # usually needs data after cleaning
            "visualizer": None,   # usually needs data
        }

    def print_summary(self):
        """Quick console overview"""
        from collections import defaultdict
        groups = defaultdict(list)

        for key in sorted(self.tools):
            cat, name = key.split(".", 1)
            groups[cat].append(name)

        print("┌──────────────────────────────────────────────┐")
        print("│            Registered Data Tools             │")
        print("├──────────────┬───────────────────────────────┤")
        for cat, methods in groups.items():
            print(f"│ {cat:12} │ {len(methods):3d} methods                  │")
            print(f"│              │ {', '.join(methods[:4])} ... │")
        print("└──────────────┴───────────────────────────────┘")
        print(f"Total registered tools: {len(self.tools)}")


# Global singleton instance
global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Recommended way to access the registry"""
    return global_registry


if __name__ == "__main__":
    registry = get_tool_registry()
    registry.print_summary()

    print("\nExample: top 6 cleaning tools")
    for item in registry.list_tools("clean")[:6]:
        print(f"  • {item['tool_key']:38}  {item['description'][:68]}...")
