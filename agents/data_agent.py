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

        # Header with badges
        summary.append("### 📊 Data Profile")
        summary.append("")
        
        # Quick stats in a nice format
        summary.append(f"**Dataset Shape:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns")
        summary.append("")

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        summary.append(f"**Memory Usage:** `{memory_mb:.2f} MB`")
        summary.append("")
        
        summary.append("---")
        summary.append("")

        # Column list in a clean format
        summary.append(f"#### 📋 Columns ({len(df.columns)})")
        summary.append("")
        
        # Group columns by type
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if numeric_cols:
            summary.append(f"**🔢 Numeric ({len(numeric_cols)}):** {', '.join([f'`{col}`' for col in numeric_cols])}")
        if cat_cols:
            summary.append(f"**📝 Categorical ({len(cat_cols)}):** {', '.join([f'`{col}`' for col in cat_cols])}")
        if datetime_cols:
            summary.append(f"**📅 Datetime ({len(datetime_cols)}):** {', '.join([f'`{col}`' for col in datetime_cols])}")
        
        summary.append("")
        summary.append("---")
        summary.append("")

        # Missing values with better formatting
        missing = df.isnull().sum()
        if missing.sum() > 0:
            summary.append("#### ⚠️ Missing Values")
            summary.append("")
            missing_data = []
            for col in missing[missing > 0].index:
                count = missing[col]
                pct = (count / len(df) * 100)
                missing_data.append(f"- **`{col}`**: {count:,} missing ({pct:.1f}%)")
            summary.extend(missing_data)
        else:
            summary.append("#### ✅ Data Quality")
            summary.append("")
            summary.append("**No missing values detected!**")
        
        summary.append("")
        summary.append("---")
        summary.append("")

        # Summary statistics for numeric columns
        if len(numeric_cols) > 0:
            summary.append("#### 📈 Summary Statistics (Numeric)")
            summary.append("")
            
            stats_df = df[numeric_cols].describe().round(2)
            
            # Convert to markdown table
            summary.append("| Statistic | " + " | ".join([f"{col}" for col in stats_df.columns]) + " |")
            summary.append("|" + "---|" * (len(stats_df.columns) + 1))
            
            for idx in stats_df.index:
                row_data = [f"{stats_df.loc[idx, col]:,.2f}" if not pd.isna(stats_df.loc[idx, col]) else "N/A" 
                           for col in stats_df.columns]
                summary.append(f"| **{idx}** | " + " | ".join(row_data) + " |")
            
            summary.append("")

        # Categorical columns info with better formatting
        if len(cat_cols) > 0:
            summary.append("#### 🏷️ Categorical Insights")
            summary.append("")
            
            for col in cat_cols[:10]:  # Limit to first 10 to avoid cluttering
                n_unique = df[col].nunique()
                top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                top_count = (df[col] == top_value).sum()
                top_pct = (top_count / len(df) * 100)
                
                summary.append(f"**`{col}`**")
                summary.append(f"  - Unique values: `{n_unique:,}`")
                summary.append(f"  - Most common: `{top_value}` ({top_count:,} occurrences, {top_pct:.1f}%)")
                summary.append("")
            
            if len(cat_cols) > 10:
                summary.append(f"*...and {len(cat_cols) - 10} more categorical columns*")
                summary.append("")

        summary.append("---")
        summary.append("")
        summary.append("💡 **Ready to explore!** Ask me questions about your data.")

        return "\n".join(summary)
