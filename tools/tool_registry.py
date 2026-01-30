#!/usr/bin/env python3
"""
Central Tool Registry for the Data Processing Suite
Registers high-level callable methods from:
  • data_cleaning
  • data_analysis
  • data_visualization

Version: 2025-style edition
"""

# simple_registry.py

class SimpleToolRegistry:
    def __init__(self):
        self.snippets = {
            # data_cleaning
            "load": "cleaner = SimpleCleaner()\ncleaner.load('data.csv')",
            "drop_duplicates": "cleaner.drop_duplicates()",
            "fill_missing_zero": "cleaner.fill_missing(how='zero')",
            "fill_missing_mean": "cleaner.fill_missing(how='mean')",
            "fix_names": "cleaner.fix_column_names()",
            "cleaner_report": "cleaner.show_report()",

            # data_analysis
            "info": "analyzer = SimpleAnalyzer(df)\nanalyzer.basic_info()",
            "stats": "analyzer.numeric_summary()",
            "top": "analyzer.top_categories('column_name', n=10)",
            "corr": "analyzer.correlation()",
            "quick": "analyzer.quick_report()",

            # data_visualization
            "hist": "viz = SimpleVisualizer(df)\nfig = viz.histogram('age', color='purple', nbins=25)\nfig.show()",
            "bar": "fig = viz.bar('city', top_n=10, color='orange')\nfig.show()",
            "scatter": "fig = viz.scatter('age', 'salary', color='department')\nfig.show()",
            "box": "fig = viz.box('salary', by='department')\nfig.show()",
        }

    def show_available(self):
        print("Available shortcuts:")
        for k in sorted(self.snippets):
            print(f"  {k:18} →  {self.snippets[k].splitlines()[0]} ...")

    def get_code(self, name):
        if name in self.snippets:
            print("\nCopy-paste this code:\n")
            print(self.snippets[name])
        else:
            print(f"No snippet called '{name}'")
            self.show_available()


# Quick demo usage
if __name__ == "__main__":
    reg = SimpleToolRegistry()
    reg.show_available()
    print("\nExample:")
    reg.get_code("hist")
