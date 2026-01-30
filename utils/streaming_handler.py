"""Streaming callback handler for displaying agent steps in Streamlit."""

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any


class StreamlitStatusCallbackHandler(BaseCallbackHandler):
    """Holds an st.status() widget so ManagerAgent can write step updates.

    With LangGraph the agent drives updates directly via graph.stream(),
    so this class simply stores the status container reference.
    The callback methods are no-ops to avoid NoSessionContext errors.
    """

    def __init__(self, status_container):
        self.status = status_container
