"""
LangGraph Manager Agent
A LangGraph-based implementation of the manager agent.
Enables graph visualization in LangSmith and better state management.
"""

from typing import Annotated, TypedDict, Union, List
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools.tool_registry import get_all_tools
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE


class AgentState(TypedDict):
    """The state of the agent in the graph"""
    messages: List[BaseMessage]
    # We can add other state keys here if needed
    data_context: str


class LangGraphManager:
    """Manager agent using LangGraph for orchestration"""

    def __init__(self):
        self.tools = get_all_tools()
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE
        ).bind_tools(self.tools)
        
        self.graph = self._build_graph()

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

    def _call_model(self, state: AgentState):
        """Call the model node"""
        messages = state['messages']
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """Determine if we should continue to tools or end"""
        messages = state['messages']
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        """Build the LangGraph state graph"""
        
        # Define the nodes
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define the edges
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    def invoke(self, user_input: str):
        """
        Process user query and return results.
        """
        try:
            from utils.data_state import DataState

            # Clear any previously pending figures
            state = DataState()
            state.pending_figures = []
            
            # Prepare dynamic system message
            data_context = self._get_data_context()
            system_message = f"""You are a data analysis assistant. You help users explore and analyze their dataset using the available tools.

{data_context}

IMPORTANT RULES:
1. The data context above tells you the current state.
2. When suggesting next steps, consider the columns available.
3. If the user's question is ambiguous, use get_data_overview or list_columns first.
4. Provide clear, concise answers based on tool outputs.
"""
            
            inputs = {
                "messages": [
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_input)
                ],
                "data_context": data_context
            }
            
            # Execute the graph
            result = self.graph.invoke(inputs)
            
            # Get the final response
            final_message = result['messages'][-1]
            content = final_message.content

            # Retrieve figures
            figures = state.pop_figures()

            return {
                "text": content,
                "figures": figures
            }

        except Exception as e:
            return {
                "text": f"Error processing query: {str(e)}",
                "figures": []
            }
