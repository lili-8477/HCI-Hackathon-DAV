"""
Manager Agent
User-facing orchestrator that processes natural language queries
and calls appropriate tools using LangChain.
"""

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, initialize_agent, AgentType
from tools.tool_registry import get_all_tools
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE


class ManagerAgent:
    """Manager agent for handling user queries"""

    def __init__(self):
        """Initialize the manager agent with LLM and tools"""
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE
        )

        self.tools = get_all_tools()

    def _get_data_context(self) -> str:
        """Get current data context for dynamic prompting"""
        from utils.data_state import DataState
        state = DataState()
        
        if not state.is_data_loaded():
            return """
DATA STATUS: No data loaded yet.
- Ask the user to upload a dataset first.
- Guide them to use the sidebar file uploader."""
        
        df = state.get_dataframe()
        file_info = state.get_file_info()
        
        # Get column info for context
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return f"""
DATA STATUS: ✅ Data is loaded and ready for analysis!

CURRENT DATASET:
- File: {file_info['file_name']}
- Shape: {file_info['rows']:,} rows × {file_info['columns']} columns
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}{'...' if len(cat_cols) > 10 else ''}

You MUST use the available tools to answer questions about this data.
The data IS loaded - proceed with analysis directly."""

    def _create_agent_executor(self):
        """Create the agent with tools and current data context"""
        
        data_context = self._get_data_context()
        
        # System prefix for the agent - includes dynamic data context
        system_prefix = f"""You are a data analysis assistant. You help users explore and analyze their dataset using the available tools.

{data_context}

IMPORTANT RULES:
1. The data context above tells you the current state. If data is loaded, proceed with analysis - DO NOT ask users to upload data again.
2. When suggesting next steps for analysis, consider the columns available and provide specific, actionable suggestions.
3. If the user's question is ambiguous, use get_data_overview or list_columns first to understand the data before answering.
4. When a tool requires no input, pass an empty string as input.
5. Always check available columns before making assumptions about column names.
6. Provide clear, concise answers based on tool outputs.
7. When the user asks about unique values or categories in a column, use get_column_info with the column name.
8. For questions like "what should I analyze next", suggest specific analyses based on the available columns."""

        # Create agent using initialize_agent
        agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
            agent_kwargs={"prefix": system_prefix}
        )

        return agent_executor

    def invoke(self, user_input: str):
        """
        Process user query and return results.

        Args:
            user_input: Natural language query from user

        Returns:
            dict with 'text' and 'figures' keys
        """
        try:
            from utils.data_state import DataState

            # Clear any previously pending figures
            state = DataState()
            state.pending_figures = []

            # Create agent with current data context (dynamic!)
            agent_executor = self._create_agent_executor()

            # Execute the agent
            result = agent_executor.invoke({"input": user_input})

            # Retrieve figures stored by visualization tools during execution
            figures = state.pop_figures()

            return {
                "text": result.get("output", "No response generated."),
                "figures": figures
            }
        except Exception as e:
            return {
                "text": f"Error processing query: {str(e)}",
                "figures": []
            }

