"""
Data Analysis & Visualization Assistant
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
from utils.data_state import DataState
from utils.streamlit_helpers import (
    display_sidebar_info, display_example_queries, show_welcome_message
)
from utils.export import build_export_zip
from agents.data_agent import DataAgent
from agents.manager_agent import ManagerAgent
from config import PAGE_TITLE, PAGE_ICON, LAYOUT


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'show_plot_dialog' not in st.session_state:
    st.session_state.show_plot_dialog = False

if 'latest_plots' not in st.session_state:
    st.session_state.latest_plots = []

if 'manager_agent' not in st.session_state:
    with st.spinner("Initializing AI agent..."):
        st.session_state.manager_agent = ManagerAgent()

if 'data_agent' not in st.session_state:
    st.session_state.data_agent = DataAgent()


# Sidebar: File Upload
st.sidebar.title("📁 Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=['csv', 'xlsx', 'xls', 'json'],
    help="Supported formats: CSV, Excel, JSON"
)

# Handle file upload
if uploaded_file is not None:
    if not st.session_state.data_loaded or st.session_state.get('last_file_name') != uploaded_file.name:
        try:
            with st.spinner("Loading data..."):
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == 'json':
                    df = pd.read_json(uploaded_file)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    st.stop()

                state = DataState()
                state.load_data(df, uploaded_file.name, uploaded_file.name)

                profile = st.session_state.data_agent.analyze(df)

                st.session_state.data_loaded = True
                st.session_state.last_file_name = uploaded_file.name

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Successfully loaded **{uploaded_file.name}**\n\n{profile}",
                    "plots": [],
                })

                st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

# Display data info in sidebar
state = DataState()
data_info = state.get_file_info()
display_sidebar_info(data_info)

# Display example queries
display_example_queries()

# Sidebar: Export
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Results")

if st.session_state.messages:
    zip_bytes = build_export_zip(st.session_state.messages)
    file_label = st.session_state.get('last_file_name', 'analysis')
    st.sidebar.download_button(
        label="Download Report (ZIP)",
        data=zip_bytes,
        file_name=f"{file_label}_report.zip",
        mime="application/zip",
    )
else:
    st.sidebar.info("Nothing to export yet.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit, LangChain, and Ollama")
st.sidebar.markdown("Model: qwen3:latest")


# Plot dialog popup
@st.dialog("📊 Visualization", width="large")
def show_plot_popup(plot_images):
    for img_bytes in plot_images:
        st.image(img_bytes, use_container_width=True)


# Check if we need to show the plot dialog
if st.session_state.show_plot_dialog and st.session_state.latest_plots:
    st.session_state.show_plot_dialog = False
    show_plot_popup(st.session_state.latest_plots)


# Main area
if not st.session_state.data_loaded:
    show_welcome_message()
else:
    # Table preview
    state = DataState()
    df = state.get_dataframe()

    with st.expander(f"🗂️ Table Preview  —  {df.shape[0]} rows × {df.shape[1]} columns", expanded=False):
        col1, col2 = st.columns([1, 3])
        with col1:
            preview_rows = st.slider("Rows to show", min_value=5, max_value=min(200, len(df)), value=10, step=5)
        with col2:
            selected_cols = st.multiselect("Filter columns", options=df.columns.tolist(), default=df.columns.tolist())

        if selected_cols:
            st.dataframe(df[selected_cols].head(preview_rows), use_container_width=True, height=350)
        else:
            st.info("Select at least one column to preview.")

    st.markdown("---")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show inline plots stored in message history
            for img_bytes in message.get("plots", []):
                st.image(img_bytes, use_container_width=True)

    # Chat input
    user_input = st.chat_input("Ask about your data...")

    if user_input:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "plots": [],
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response from manager agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.manager_agent.invoke(user_input)

            st.markdown(response["text"])

            # Collect plot images
            plot_images = [fig_data for _, fig_data in response["figures"]]

            # Show plots inline right below the response
            for img_bytes in plot_images:
                st.image(img_bytes, use_container_width=True)

        # Save to message history (with plots)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["text"],
            "plots": plot_images,
        })

        # Trigger popup dialog if there are plots
        if plot_images:
            st.session_state.latest_plots = plot_images
            st.session_state.show_plot_dialog = True
            st.rerun()
