"""
Manager Agent
User-facing orchestrator that processes natural language queries
and calls appropriate tools using LangChain.
"""

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, initialize_agent, AgentType
from tools.tool_registry import get_all_tools
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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

        # Create agent using initialize_agent
        agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
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
            # Execute the agent
            result = self.agent_executor.invoke({"input": user_input})

            # Extract figures from the intermediate steps
            figures = self._extract_figures(result)

            return {
                "text": result.get("output", "No response generated."),
                "figures": figures
            }
        except Exception as e:
            return {
                "text": f"Error processing query: {str(e)}",
                "figures": []
            }

    def _extract_figures(self, result: dict) -> list:
        """
        Extract matplotlib or plotly figures from agent execution result.

        Args:
            result: Agent execution result

        Returns:
            List of figure objects
        """
        figures = []

        # Check intermediate steps for figures
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) >= 2:
                    observation = step[1]  # The tool output

                    # Check if it's a matplotlib figure
                    if isinstance(observation, plt.Figure):
                        figures.append(("matplotlib", observation))

                    # Check if it's a plotly figure
                    elif isinstance(observation, go.Figure):
                        figures.append(("plotly", observation))

        return figures
