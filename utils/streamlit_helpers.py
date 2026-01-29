"""
Streamlit Helper Functions
Utility functions for Streamlit UI components.
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def display_message(message: dict):
    """
    Display a chat message in Streamlit.

    Args:
        message: dict with 'role' and 'content' keys
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display figures if present
        if "figures" in message and message["figures"]:
            for fig_type, fig in message["figures"]:
                if fig_type == "matplotlib":
                    st.pyplot(fig)
                    plt.close(fig)  # Close to free memory
                elif fig_type == "plotly":
                    st.plotly_chart(fig, use_container_width=True)


def display_data_profile(profile_text: str):
    """
    Display data profile in an expander.

    Args:
        profile_text: Formatted profile string
    """
    with st.expander("📊 Data Profile", expanded=False):
        st.text(profile_text)


def display_sidebar_info(data_info: dict):
    """
    Display data information in sidebar.

    Args:
        data_info: Dictionary with file information
    """
    if data_info.get("loaded", False):
        st.sidebar.success("✅ Data Loaded")
        st.sidebar.markdown(f"**File:** {data_info['file_name']}")
        st.sidebar.markdown(f"**Rows:** {data_info['rows']:,}")
        st.sidebar.markdown(f"**Columns:** {data_info['columns']}")
    else:
        st.sidebar.info("No data loaded")


def display_example_queries():
    """Display example queries in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Example Queries")
    st.sidebar.markdown("""
    - Show me summary statistics
    - Are there any missing values?
    - Plot a correlation heatmap
    - Show the distribution of [column]
    - Create a scatter plot of [x] vs [y]
    - Which [category] has the highest [value]?
    - Detect outliers in [column]
    """)


def show_welcome_message():
    """Display welcome message when no data is loaded"""
    st.markdown("""
    # 📊 Data Analysis & Visualization Assistant

    Welcome! This tool helps you analyze and visualize your data using natural language queries.

    ## Getting Started

    1. **Upload your data** using the file uploader in the sidebar (CSV, Excel, or JSON)
    2. **Ask questions** about your data in natural language
    3. **Get insights** with automatic visualizations and analysis

    ## What You Can Do

    - Get summary statistics and data overviews
    - Check for missing values
    - Calculate correlations
    - Create various visualizations (histograms, scatter plots, heatmaps, etc.)
    - Perform group-by analyses
    - Detect outliers
    - And much more!

    Upload a file to begin! 👈
    """)
