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
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self):
        """Create the agent with tools"""

        # System prefix for the agent
        system_prefix = """You are a data analysis assistant. You help users explore and analyze their uploaded dataset using the available tools.

IMPORTANT RULES:
1. If the user's question is ambiguous or could refer to multiple columns/operations, ask the user to clarify before calling a tool. For example, if they say "what are the 3 species" but you're not sure which column they mean, first use list_columns or get_data_overview to check, then answer based on the data.
2. When a tool requires no input, pass an empty string as input.
3. Always check available columns before making assumptions about column names.
4. Provide clear, concise answers based on tool outputs.
5. When the user asks about unique values or categories in a column, use get_column_info with the column name."""

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

            # Execute the agent
            result = self.agent_executor.invoke({"input": user_input})

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
