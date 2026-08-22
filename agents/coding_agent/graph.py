"""Builds the compiled LangGraph for the coding agent."""

from langgraph.graph import StateGraph, END

from agents.coding_agent.model import call_model
from agents.coding_agent.state import AgentState
from agents.coding_agent.tools import call_tools


def should_continue(state: AgentState):
    """Router: Checks if the model wants to call a tool or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# --- Build the Graph ---

builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", call_tools)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")  # Loop back to model after tool execution

# Compiled coding agent graph.
graph = builder.compile()
