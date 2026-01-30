"""
Visualize LangGraph
Generates a visualization of the agent's graph structure.
"""

from agents.langgraph_manager import LangGraphManager
from langchain_core.runnables.graph import MermaidDrawMethod

def visualize():
    print("Initializing Graph...")
    manager = LangGraphManager()
    graph = manager.graph
    
    print("\n=== ASCII Graph Structure ===")
    try:
        graph.get_graph().print_ascii()
    except Exception as e:
        print(f"Could not print ASCII: {e}")

    print("\n=== Generatng Mermaid PNG ===")
    try:
        # Generate PNG using mermaid API
        png_data = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        
        output_file = "agent_graph.png"
        with open(output_file, "wb") as f:
            f.write(png_data)
        
        print(f"✅ Graph visualization saved to: {output_file}")
    except Exception as e:
        print(f"❌ Could not generate PNG: {e}")
        print("Note: PNG generation requires internet access to mermaid.ink")

if __name__ == "__main__":
    visualize()
