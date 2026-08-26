"""Tool node for the coding agent: runs tools the LLM requested."""

from langchain_core.messages import ToolMessage

from clyde.tools import tools_by_name as _initial
from clyde.trace import compact_trace

from clyde.agents.coding_agent.state import MessageState

# Tool registry. Seeded with the static tools at import; call `configure_tools`
# at startup to swap in a superset (static + MCP). `call_tools` reads this global
# at call time, so the rebind takes effect without rebuilding the graph.
_registry: dict[str, object] = dict(_initial)


def configure_tools(tool_list):
    """Replace the tool registry (called at startup with static + MCP tools)."""
    global _registry
    _registry = {t.name: t for t in tool_list}


def call_tools(state: MessageState):
    """Tool Node: Executes tools called by the LLM and feeds output back."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_fn = _registry[tool_call["name"]]
        output = tool_fn.invoke(
            tool_call["args"],
            config={"callbacks": [compact_trace]},
        )

        tool_results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": tool_results}
