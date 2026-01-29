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
        st.sidebar.markdown(f"**📄 {data_info['file_name']}**")
        
        # Display metrics in columns
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Rows", f"{data_info['rows']:,}")
        with col2:
            st.metric("Columns", data_info['columns'])
    else:
        st.sidebar.info("📂 No data loaded yet")


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
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="color: #00B4D8; font-size: 1.8rem; margin-bottom: 0.5rem;">
            Welcome to Your AI-Powered Analytics Platform
        </h2>
        <p style="color: #B8B8B8; font-size: 1.1rem; margin-bottom: 2rem;">
            Upload your data and start exploring insights with natural language queries
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #00B4D8; margin-top: 0;">📤 Upload Data</h3>
            <p style="color: #B8B8B8;">
                Support for CSV, Excel, and JSON formats. Simply drag and drop your file in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #F7B801; margin-top: 0;">💬 Ask Questions</h3>
            <p style="color: #B8B8B8;">
                Use natural language to query your data. Our AI understands your questions and provides insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #6366F1; margin-top: 0;">📊 Get Insights</h3>
            <p style="color: #B8B8B8;">
                Receive automatic visualizations, statistics, and analysis tailored to your questions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features section
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📊 **Summary Statistics** - Get instant overview of your data
        - 🔍 **Data Quality Checks** - Detect missing values and outliers
        - 📈 **Advanced Visualizations** - Heatmaps, distributions, scatter plots
        - 🎯 **Correlation Analysis** - Understand relationships in your data
        """)
    
    with col2:
        st.markdown("""
        - 📉 **Trend Analysis** - Explore patterns over time
        - 🔄 **Group-by Operations** - Aggregate and compare segments
        - 💾 **Export Results** - Download reports with all insights
        - 🤖 **AI-Powered** - Intelligent responses using advanced LLMs
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Get started by uploading a file in the sidebar!**")

