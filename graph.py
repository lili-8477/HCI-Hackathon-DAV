"""
LangGraph Entry Point
This file exposes the compiled graph for LangGraph Studio and inspection.
"""

from agents.langgraph_manager import LangGraphManager

# Initialize the manager which builds the graph
manager = LangGraphManager()

# Expose the compiled graph
# LangGraph Studio looks for a variable named 'graph' or a compiled StateGraph
graph = manager.graph

# Optional: Add a main block to print the graph structure
if __name__ == "__main__":
    print("Graph initialized successfully.")
    try:
        # Save a visualization of the graph if needed
        pass
    except Exception as e:
        print(f"Error visualizing graph: {e}")
