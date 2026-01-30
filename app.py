"""
Data Analysis & Visualization Assistant
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from utils.data_state import DataState
from utils.streamlit_helpers import (
    display_sidebar_info, display_example_queries, show_welcome_message
)
from utils.export import build_export_zip
from agents.data_agent import DataAgent
# from agents.manager_agent import ManagerAgent  # Legacy agent
from agents.langgraph_manager import LangGraphManager  # Graph agent
from config import PAGE_TITLE, PAGE_ICON, LAYOUT

# Initialize LangSmith tracing (if configured)
try:
    from utils.langsmith_config import setup_langsmith, get_langsmith_status
    langsmith_enabled = setup_langsmith("HCI-Hackathon-DAV")
except ImportError:
    langsmith_enabled = False


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Display logo and header
def display_header():
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 1.5rem 0; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 80px; height: 80px; margin-right: 1.5rem; filter: drop-shadow(0 4px 12px rgba(0, 180, 216, 0.4));">
            <div>
                <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #00B4D8 0%, #F7B801 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Data Analysis & Visualization Assistant
                </h1>
                <p style="margin: 0.3rem 0 0 0; color: #B8B8B8; font-size: 1rem;">
                    AI-Powered Analytics Platform
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.title("📊 Data Analysis & Visualization Assistant")

display_header()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'manager_agent' not in st.session_state:
    with st.spinner("Initializing AI agent..."):
        st.session_state.manager_agent = LangGraphManager()

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

# Sidebar: Sample Datasets
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Sample Datasets")
from utils.sample_datasets import SAMPLE_DATASETS
dataset_name = st.sidebar.selectbox("Choose a sample dataset", [""] + list(SAMPLE_DATASETS.keys()))
if dataset_name:
    st.sidebar.caption(SAMPLE_DATASETS[dataset_name]["desc"])
if st.sidebar.button("Load Sample", disabled=not dataset_name):
    with st.spinner("Generating sample data..."):
        df = SAMPLE_DATASETS[dataset_name]["fn"]()
        state = DataState()
        state.load_data(df, f"{dataset_name}.csv", f"{dataset_name}.csv")
        profile = st.session_state.data_agent.analyze(df)
        st.session_state.data_loaded = True
        st.session_state.last_file_name = f"{dataset_name}.csv"
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Loaded sample **{dataset_name}**\n\n{profile}",
            "plots": [],
        })
        st.rerun()

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
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <p style="color: #B8B8B8; font-size: 0.85rem; margin: 0.3rem 0;">
        ⚡ Powered by
    </p>
    <p style="color: #00B4D8; font-weight: 600; margin: 0.3rem 0;">
        Streamlit • LangChain • Ollama
    </p>
    <p style="color: #B8B8B8; font-size: 0.8rem; margin: 0.3rem 0;">
        🤖 Model: <span style="color: #F7B801;">qwen3:32b</span>
    </p>
</div>
""", unsafe_allow_html=True)


# Main area
if not st.session_state.data_loaded:
    show_welcome_message()
else:
    # Table preview
    state = DataState()
    df = state.get_dataframe()

    # Initialize operations log
    if "data_operations" not in st.session_state:
        st.session_state.data_operations = []

    with st.expander(f"🗂️ Table Preview  —  {df.shape[0]} rows × {df.shape[1]} columns", expanded=False):
        # Show applied operations log
        if st.session_state.data_operations:
            st.caption("Applied operations: " + " → ".join(st.session_state.data_operations))

        # Undo all button
        if st.session_state.data_operations:
            if st.button("↩ Undo All Changes", key="undo_all"):
                state.df = state.get_intermediate("original").copy()
                st.session_state.data_operations = []
                st.rerun()

        col1, col2 = st.columns([1, 3])
        with col1:
            preview_rows = st.slider("Rows to show", min_value=5, max_value=min(200, len(df)), value=10, step=5)
        with col2:
            selected_cols = st.multiselect("Filter columns", options=df.columns.tolist(), default=df.columns.tolist())

        if selected_cols:
            st.dataframe(df[selected_cols].head(preview_rows), use_container_width=True, height=350)
        else:
            st.info("Select at least one column to preview.")

        # --- Data manipulation tabs ---
        filter_tab, transform_tab, impute_tab = st.tabs(["🔍 Filter", "🔄 Transform", "🩹 Impute"])

        with filter_tab:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                filter_col = st.selectbox("Column", df.columns.tolist(), key="filter_col")
            with fc2:
                filter_op = st.selectbox("Condition", ["==", "!=", ">", "<", ">=", "<=", "contains", "isin"], key="filter_op")
            with fc3:
                filter_val = st.text_input("Value", key="filter_val", help="For 'isin', use comma-separated values")

            if st.button("Apply Filter", key="apply_filter"):
                if filter_val.strip() == "":
                    st.warning("Enter a value.")
                else:
                    try:
                        col_series = df[filter_col]
                        v = filter_val.strip()
                        if filter_op == "contains":
                            mask = col_series.astype(str).str.contains(v, case=False, na=False)
                        elif filter_op == "isin":
                            vals = [x.strip() for x in v.split(",")]
                            # try numeric cast
                            try:
                                vals = [float(x) for x in vals]
                            except ValueError:
                                pass
                            mask = col_series.isin(vals)
                        else:
                            # try numeric comparison
                            try:
                                v_cmp = float(v)
                            except ValueError:
                                v_cmp = v
                            ops = {"==": "__eq__", "!=": "__ne__", ">": "__gt__", "<": "__lt__", ">=": "__ge__", "<=": "__le__"}
                            mask = getattr(col_series, ops[filter_op])(v_cmp)
                        state.df = df[mask].reset_index(drop=True)
                        st.session_state.data_operations.append(f"Filter({filter_col} {filter_op} {filter_val})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Filter error: {e}")

        with transform_tab:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                tc1, tc2 = st.columns(2)
                with tc1:
                    trans_col = st.selectbox("Column", numeric_cols, key="trans_col")
                with tc2:
                    trans_op = st.selectbox("Operation", ["log", "normalize (0-1)", "standardize (z-score)", "round", "abs"], key="trans_op")

                if st.button("Apply Transform", key="apply_transform"):
                    try:
                        series = df[trans_col]
                        if trans_op == "log":
                            state.df[trans_col] = np.log1p(series)
                        elif trans_op == "normalize (0-1)":
                            mn, mx = series.min(), series.max()
                            state.df[trans_col] = (series - mn) / (mx - mn) if mx != mn else 0.0
                        elif trans_op == "standardize (z-score)":
                            state.df[trans_col] = (series - series.mean()) / series.std()
                        elif trans_op == "round":
                            state.df[trans_col] = series.round()
                        elif trans_op == "abs":
                            state.df[trans_col] = series.abs()
                        st.session_state.data_operations.append(f"{trans_op}({trans_col})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transform error: {e}")

        with impute_tab:
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if cols_with_nulls.empty:
                st.info("No missing values found.")
            else:
                st.dataframe(cols_with_nulls.rename("missing").to_frame(), use_container_width=True, height=150)
                ic1, ic2 = st.columns(2)
                with ic1:
                    imp_col = st.selectbox("Column", cols_with_nulls.index.tolist(), key="imp_col")
                with ic2:
                    imp_strategy = st.selectbox("Strategy", ["drop rows", "mean", "median", "mode", "forward fill", "backward fill", "custom value"], key="imp_strategy")

                custom_val = None
                if imp_strategy == "custom value":
                    custom_val = st.text_input("Custom fill value", key="imp_custom")

                if st.button("Apply Imputation", key="apply_impute"):
                    try:
                        if imp_strategy == "drop rows":
                            state.df = df.dropna(subset=[imp_col]).reset_index(drop=True)
                        elif imp_strategy == "mean":
                            state.df[imp_col] = df[imp_col].fillna(df[imp_col].mean())
                        elif imp_strategy == "median":
                            state.df[imp_col] = df[imp_col].fillna(df[imp_col].median())
                        elif imp_strategy == "mode":
                            state.df[imp_col] = df[imp_col].fillna(df[imp_col].mode().iloc[0])
                        elif imp_strategy == "forward fill":
                            state.df[imp_col] = df[imp_col].ffill()
                        elif imp_strategy == "backward fill":
                            state.df[imp_col] = df[imp_col].bfill()
                        elif imp_strategy == "custom value":
                            try:
                                fill = float(custom_val)
                            except (ValueError, TypeError):
                                fill = custom_val
                            state.df[imp_col] = df[imp_col].fillna(fill)
                        st.session_state.data_operations.append(f"Impute {imp_col}({imp_strategy})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Imputation error: {e}")

    st.markdown("---")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show executed code in foldable expanders (before text)
            for exe in message.get("tool_executions", []):
                with st.expander(f"Step {exe['step']}: {exe['label']}", expanded=False):
                    if exe.get("code"):
                        st.code(exe["code"], language="python")
                    if exe.get("result"):
                        st.text(exe["result"])
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
            from utils.streaming_handler import StreamlitStatusCallbackHandler
            with st.status("Thinking...", expanded=True) as status:
                handler = StreamlitStatusCallbackHandler(status)
                response = st.session_state.manager_agent.invoke(user_input, callbacks=[handler])
                status.update(label="Done", state="complete", expanded=False)

            # Show executed code in foldable expanders
            for exe in response.get("tool_executions", []):
                with st.expander(f"Step {exe['step']}: {exe['label']}", expanded=False):
                    if exe.get("code"):
                        st.code(exe["code"], language="python")
                    if exe.get("result"):
                        st.text(exe["result"])

            st.markdown(response["text"])

            # Collect plot images
            plot_images = [fig_data for _, fig_data in response["figures"]]

            # Show plots inline right below the response
            for img_bytes in plot_images:
                st.image(img_bytes, use_container_width=True)

        # Save to message history (with plots and tool executions)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["text"],
            "plots": plot_images,
            "tool_executions": response.get("tool_executions", []),
        })
