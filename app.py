"""
Data Analysis & Visualization Assistant
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
from utils.data_state import DataState
from utils.streamlit_helpers import (
    display_message, display_data_profile, display_sidebar_info,
    display_example_queries, show_welcome_message
)
from agents.data_agent import DataAgent
from agents.manager_agent import ManagerAgent
from config import PAGE_TITLE, PAGE_ICON, LAYOUT
import io


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
    # Check if this is a new file
    if not st.session_state.data_loaded or st.session_state.get('last_file_name') != uploaded_file.name:
        try:
            with st.spinner("Loading data..."):
                # Read file based on extension
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

                # Load into data state
                state = DataState()
                state.load_data(df, uploaded_file.name, uploaded_file.name)

                # Profile the data
                profile = st.session_state.data_agent.analyze(df)

                # Update session state
                st.session_state.data_loaded = True
                st.session_state.last_file_name = uploaded_file.name

                # Add system message about data loading
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ Successfully loaded **{uploaded_file.name}**\n\n{profile}",
                    "figures": []
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

# Main area
if not st.session_state.data_loaded:
    # Show welcome message if no data loaded
    show_welcome_message()
else:
    # Display chat history
    for message in st.session_state.messages:
        display_message(message)

    # Chat input
    user_input = st.chat_input("Ask about your data...")

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "figures": []
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response from manager agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.manager_agent.invoke(user_input)

            # Display text response
            st.markdown(response["text"])

            # Display figures if any
            if response["figures"]:
                for fig_type, fig in response["figures"]:
                    if fig_type == "matplotlib":
                        st.pyplot(fig)
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    elif fig_type == "plotly":
                        st.plotly_chart(fig, use_container_width=True)

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["text"],
            "figures": response["figures"]
        })

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit, LangChain, and Ollama")
st.sidebar.markdown("Model: qwen3:8b")
