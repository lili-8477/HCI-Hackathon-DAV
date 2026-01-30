"""
Manager Agent
User-facing orchestrator that processes natural language queries
using a LangGraph StateGraph with tool nodes.
"""

import re
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tools.tool_registry import get_all_tools
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from qwen3 model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_data_context() -> str:
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

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    intermediates = state.list_intermediates()
    intermediate_info = ""
    if intermediates:
        table_list = ", ".join(f"{t['name']} ({t['rows']}×{t['columns']})" for t in intermediates)
        intermediate_info = f"\n- Intermediate tables: {table_list}"

    return f"""
DATA STATUS: ✅ Data is loaded and ready for analysis!

CURRENT DATASET:
- File: {file_info['file_name']}
- Active data shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}{'...' if len(cat_cols) > 10 else ''}{intermediate_info}

You MUST use the available tools to answer questions about this data.
The data IS loaded - proceed with analysis directly.
When the user filters or manipulates data, the result automatically becomes the active dataset for subsequent tools (visualizations, statistics, etc.).
Use 'use_table' to switch between saved intermediate tables or back to the original data."""


SYSTEM_TEMPLATE = """/no_think
You are a data analysis assistant. You help users explore and analyze their dataset using the available tools.

{data_context}

IMPORTANT RULES:
1. The data context above tells you the current state. If data is loaded, proceed with analysis - DO NOT ask users to upload data again.
2. Only do exactly what the user asks. Do NOT proactively generate visualizations or plots unless the user explicitly requests one (e.g. "plot", "chart", "graph", "visualize", "show me a plot").
3. If the user's question is ambiguous, use get_data_overview or list_columns first to understand the data before answering.
4. When a tool requires no input, pass an empty string as input.
5. Always check available columns before making assumptions about column names.
6. Provide clear, concise answers based on tool outputs. Do NOT include any <think> tags or internal reasoning in your responses.
7. When the user asks about unique values or categories in a column, use get_column_info with the column name.
8. When suggesting next steps, keep suggestions text-only. Do NOT call visualization tools unless the user asked for a chart or plot.
9. When data has been filtered or manipulated, subsequent visualizations and analyses automatically use the filtered data. You do NOT need to call use_table first.
10. For pie charts, use the plot_pie_chart tool. For bar charts, use plot_bar_chart.
11. CRITICAL FALLBACK RULE: If the user's request cannot be handled well by any of the specific predefined tools (e.g. custom calculations, machine learning, advanced transformations, feature engineering, regression, clustering, or any non-standard analysis), you MUST use the 'generate_and_run_code' tool instead of forcing an ill-fitting tool. This tool generates Python code using the RAG knowledge base and executes it automatically. Always prefer the right tool for the job — use generate_and_run_code when nothing else fits."""


# -- Build the graph ----------------------------------------------------------

_tools = get_all_tools()
_llm = ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE).bind_tools(_tools)


def agent(state: MessagesState):
    """Call the LLM with the current messages."""
    messages = state["messages"]
    # Prepend system message with fresh data context on every invocation
    system_msg = SystemMessage(content=SYSTEM_TEMPLATE.format(data_context=_get_data_context()))
    response = _llm.invoke([system_msg] + messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


# -- ManagerAgent wrapper (keeps app.py interface unchanged) -------------------

class ManagerAgent:
    """Manager agent for handling user queries"""

    def __init__(self):
        self.conversation_history = []

    def invoke(self, user_input: str, callbacks=None):
        """
        Process user query and return results.

        Args:
            user_input: Natural language query from user
            callbacks: List containing a StreamlitStatusCallbackHandler (or None)

        Returns:
            dict with 'text' and 'figures' keys
        """
        try:
            from utils.data_state import DataState

            state = DataState()
            state.pending_figures = []

            # Extract the status handler if provided
            status_handler = None
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "status"):
                        status_handler = cb
                        break

            # Add user message to conversation history
            self.conversation_history.append(HumanMessage(content=user_input))

            # Stream the graph so we can report tool steps to the UI
            step = 0
            last_ai_message = None
            tool_executions = []  # Collect tool calls + results for display
            _pending_tool_calls = {}  # tool_call_id -> dict
            for event in graph.stream(
                {"messages": list(self.conversation_history)},
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                step += 1
                                tool_name = tc['name']
                                # Friendly labels for tools
                                tool_labels = {
                                    "filter_rows": "🔍 Filtering rows",
                                    "select_columns": "📋 Selecting columns",
                                    "pivot_table": "🔄 Creating pivot table",
                                    "transform_column": "⚙️ Transforming column",
                                    "use_table": "📌 Switching active table",
                                    "list_tables": "📑 Listing tables",
                                    "plot_bar_chart": "📊 Creating bar chart",
                                    "plot_pie_chart": "🥧 Creating pie chart",
                                    "plot_scatter": "📈 Creating scatter plot",
                                    "plot_distribution": "📉 Plotting distribution",
                                    "plot_box_plot": "📦 Creating box plot",
                                    "plot_correlation_heatmap": "🗺️ Creating heatmap",
                                    "plot_time_series": "📈 Plotting time series",
                                    "perform_groupby": "📊 Grouping data",
                                    "get_data_overview": "🔎 Inspecting data",
                                    "generate_and_run_code": "🧠 Generating & running custom code",
                                    "generate_code": "💡 Generating code",
                                }
                                label = tool_labels.get(tool_name, f"🛠️ {tool_name}")
                                args = tc.get("args", {})
                                arg_summary = ", ".join(f"{v}" for v in args.values()) if isinstance(args, dict) else str(args)
                                if len(arg_summary) > 120:
                                    arg_summary = arg_summary[:120] + "…"

                                # Build a Python-like representation of the call
                                if isinstance(args, dict) and args:
                                    formatted_args = ", ".join(f'{k}={repr(v)}' for k, v in args.items())
                                    code_repr = f"{tool_name}({formatted_args})"
                                else:
                                    code_repr = f"{tool_name}()"

                                tc_id = tc.get("id", str(step))
                                _pending_tool_calls[tc_id] = {
                                    "step": step,
                                    "label": label,
                                    "code": code_repr,
                                    "result": None,
                                }

                                if status_handler:
                                    status_handler.status.update(
                                        label=f"Step {step}: {label}",
                                        state="running",
                                    )
                                    status_handler.status.markdown(
                                        f"**Step {step}** — {label}\n\n`{arg_summary}`" if arg_summary else f"**Step {step}** — {label}"
                                    )
                        elif node_name == "tools":
                            # Match tool result back to its call
                            tc_id = getattr(msg, "tool_call_id", None)
                            content = getattr(msg, "content", "")
                            if tc_id and tc_id in _pending_tool_calls:
                                entry = _pending_tool_calls.pop(tc_id)
                                entry["result"] = content if isinstance(content, str) else str(content)
                                tool_executions.append(entry)
                            else:
                                # Fallback: no matching id
                                tool_executions.append({
                                    "step": step,
                                    "label": "🛠️ Tool",
                                    "code": "",
                                    "result": content if isinstance(content, str) else str(content),
                                })
                        elif node_name == "agent":
                            # Final agent response (no tool calls)
                            last_ai_message = msg
            # Flush any pending calls that never got a result
            for entry in _pending_tool_calls.values():
                tool_executions.append(entry)

            if status_handler:
                status_handler.status.update(label="Done", state="complete", expanded=False)

            text = ""
            if last_ai_message and hasattr(last_ai_message, "content"):
                text = _strip_think_tags(last_ai_message.content)
            text = text or "No response generated."

            # Save assistant response to conversation history
            from langchain_core.messages import AIMessage
            self.conversation_history.append(AIMessage(content=text))

            # Keep history from growing too large (last 20 exchanges)
            max_messages = 40
            if len(self.conversation_history) > max_messages:
                self.conversation_history = self.conversation_history[-max_messages:]

            figures = state.pop_figures()

            return {
                "text": text,
                "figures": figures,
                "tool_executions": tool_executions,
            }
        except Exception as e:
            return {
                "text": f"Error processing query: {str(e)}",
                "figures": [],
                "tool_executions": [],
            }
