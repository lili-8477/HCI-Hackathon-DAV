"""
RAG Tools
Tools for code generation and intermediate data management using RAG agent.
"""

from langchain_core.tools import Tool
from typing import Optional


# Global RAG agent instance (lazily initialized)
_rag_agent = None


def get_rag_agent():
    """Get or create the RAG agent instance"""
    global _rag_agent
    if _rag_agent is None:
        from agents.rag_agent import RAGAgent
        _rag_agent = RAGAgent()
    return _rag_agent


def generate_visualization_code(query: str) -> str:
    """
    Generate visualization code using RAG retrieval.
    
    Args:
        query: Description of the visualization needed
        
    Returns:
        Generated Python code with explanation
    """
    try:
        rag = get_rag_agent()
        result = rag.invoke(query)
        
        response = result["response"]
        
        # Add info about retrieved examples
        if result.get("retrieved_docs"):
            response += "\n\n---\n*Generated using similar code examples from the knowledge base.*"
        
        return response
    except Exception as e:
        return f"Error generating code: {str(e)}"


def store_dataframe(input_str: str) -> str:
    """
    Store current DataFrame or a subset with a name for later use.
    
    Args:
        input_str: Format 'name' or 'name,filter_expression'
                   e.g., 'filtered_data' or 'high_revenue,revenue > 1000'
    """
    try:
        from utils.data_state import DataState
        
        parts = [p.strip() for p in input_str.split(',', 1)]
        name = parts[0]
        
        state = DataState()
        df = state.get_dataframe()
        
        if len(parts) > 1:
            # Apply filter
            filter_expr = parts[1]
            try:
                filtered_df = df.query(filter_expr)
                rag = get_rag_agent()
                return rag.store_intermediate_data(name, filtered_df)
            except Exception as e:
                return f"Error applying filter '{filter_expr}': {str(e)}"
        else:
            rag = get_rag_agent()
            return rag.store_intermediate_data(name, df.copy())
            
    except Exception as e:
        return f"Error storing data: {str(e)}"


def list_stored_data(_: str = "") -> str:
    """
    List all stored intermediate DataFrames.
    """
    try:
        rag = get_rag_agent()
        return rag.list_intermediate_data()
    except Exception as e:
        return f"Error listing data: {str(e)}"


def get_stored_data_info(name: str) -> str:
    """
    Get information about a stored DataFrame.
    
    Args:
        name: Name of the stored DataFrame
    """
    try:
        rag = get_rag_agent()
        df = rag.get_intermediate_data(name)
        
        if df is None:
            return f"No data stored with name '{name}'. Use list_stored_data to see available data."
        
        info = f"**{name}**\n"
        info += f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        info += f"- Columns: {', '.join(df.columns.tolist())}\n"
        info += f"- Data types:\n"
        for col, dtype in df.dtypes.items():
            info += f"  - {col}: {dtype}\n"
        
        return info
    except Exception as e:
        return f"Error getting data info: {str(e)}"


def get_code_history(_: str = "") -> str:
    """
    Get all code generated in this session.
    """
    try:
        rag = get_rag_agent()
        history = rag.get_code_history()
        
        if not history:
            return "No code has been generated yet in this session."
        
        result = "**Generated Code History:**\n\n"
        for i, item in enumerate(history, 1):
            result += f"### {i}. {item['query'][:50]}...\n"
            result += f"```python\n{item['code'][:500]}{'...' if len(item['code']) > 500 else ''}\n```\n\n"
        
        return result
    except Exception as e:
        return f"Error getting code history: {str(e)}"


def clear_session_memory(what: str = "all") -> str:
    """
    Clear session memory.
    
    Args:
        what: 'all', 'conversation', 'intermediate', or 'code'
    """
    try:
        rag = get_rag_agent()
        return rag.clear_memory(what)
    except Exception as e:
        return f"Error clearing memory: {str(e)}"


def generate_and_run_code(query: str) -> str:
    """
    Generate Python code using the RAG agent and execute it.
    Use this when no other tool can handle the user's request.

    Args:
        query: Natural language description of the task

    Returns:
        Execution output (text result, confirmation of figures, etc.)
    """
    import re
    import io
    import sys
    import traceback

    try:
        rag = get_rag_agent()
        result = rag.invoke(query)
        code = result.get("code")

        if not code:
            # Try harder to extract code from the response
            pattern = r"```(?:python)?\n(.*?)```"
            matches = re.findall(pattern, result.get("response", ""), re.DOTALL)
            code = matches[0].strip() if matches else None

        if not code:
            return f"Code generation produced no executable code.\n\nResponse:\n{result.get('response', 'No response')}"

        # Build execution namespace with useful imports and the current dataframe
        from utils.data_state import DataState
        import pandas as pd
        import numpy as np

        state = DataState()
        df = state.get_dataframe()

        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "df": df,
            "DataState": DataState,
            "state": state,
        }

        # Try importing visualisation libraries
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            exec_globals["plt"] = plt
            exec_globals["sns"] = sns
        except ImportError:
            pass

        # Sanitize: remove any file-loading lines the LLM may have generated
        sanitized_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if re.match(r".*pd\.(read_csv|read_excel|read_json|read_parquet)\s*\(", stripped):
                continue  # skip file-loading lines
            if re.match(r".*open\s*\(", stripped) and ("csv" in stripped or "xls" in stripped):
                continue
            sanitized_lines.append(line)
        code = "\n".join(sanitized_lines)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            exec(code, exec_globals)
        finally:
            sys.stdout = old_stdout

        stdout_output = captured.getvalue()

        # Check if a matplotlib figure was created and save it to DataState
        try:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                state.add_figure("png_bytes", buf.getvalue())
                plt.close(fig)
        except Exception:
            pass

        # If the code updated df in the namespace, sync back to DataState
        if "df" in exec_globals and isinstance(exec_globals["df"], pd.DataFrame):
            if exec_globals["df"] is not df:
                state.df = exec_globals["df"]

        parts = []
        if stdout_output.strip():
            parts.append(stdout_output.strip())
        parts.append(f"Code executed successfully.\n\n```python\n{code}\n```")
        return "\n\n".join(parts)

    except Exception as e:
        return f"Code generation/execution failed: {traceback.format_exc()}"


def get_rag_tools():
    """
    Get all RAG-related tools for the manager agent.

    Returns:
        List of LangChain Tool objects
    """
    tools = [
        Tool(
            name="generate_code",
            func=generate_visualization_code,
            description="""Generate Python visualization code using the knowledge base.
Input: Natural language description of the plot/analysis needed.
Example: 'Create a grouped bar chart comparing sales by region and product category'
This tool retrieves similar code examples and generates customized code for your data."""
        ),
        Tool(
            name="generate_and_run_code",
            func=generate_and_run_code,
            description="""Generate AND execute Python code when no other tool can handle the user's request.
Use this as a fallback when the user asks for something that none of the other specific tools support —
for example custom calculations, advanced visualizations, machine learning, feature engineering,
or any analysis not covered by the predefined tools.
Input: Natural language description of what to do.
Example: 'Calculate the rolling 7-day average of the sales column and add it as a new column'
The generated code has access to the current DataFrame as 'df', pandas as 'pd', numpy as 'np',
matplotlib.pyplot as 'plt', and seaborn as 'sns'. Any figures created with matplotlib are
automatically captured. If the code produces a new DataFrame assigned to 'df', it becomes the
active dataset."""
        ),
        Tool(
            name="store_data",
            func=store_dataframe,
            description="""Store the current DataFrame or a filtered subset for later use.
Input format: 'name' or 'name,filter_expression'
Examples: 
- 'my_subset' (stores full DataFrame)
- 'high_sales,revenue > 1000' (stores filtered data)
The stored data can be referenced in code generation."""
        ),
        Tool(
            name="list_stored_data",
            func=lambda _="": list_stored_data(_),
            description="List all stored intermediate DataFrames. No input required."
        ),
        Tool(
            name="get_stored_data_info",
            func=get_stored_data_info,
            description="Get detailed information about a stored DataFrame. Input: name of the stored data."
        ),
        Tool(
            name="get_code_history",
            func=lambda _="": get_code_history(_),
            description="Get all code generated in this session. No input required."
        ),
        Tool(
            name="clear_memory",
            func=clear_session_memory,
            description="""Clear session memory. Input: 'all', 'conversation', 'intermediate', or 'code'.
- 'all' - Clear everything
- 'conversation' - Clear chat history
- 'intermediate' - Clear stored DataFrames  
- 'code' - Clear generated code history"""
        ),
    ]
    
    return tools
