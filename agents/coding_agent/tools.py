"""Tool node for the coding agent: runs tools the LLM requested."""

from langchain_core.messages import ToolMessage

from tools import tools_by_name
from trace import compact_trace

from agents.coding_agent.state import MessageState


def call_tools(state: MessageState):
    """Tool Node: Executes tools called by the LLM and feeds output back."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
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
