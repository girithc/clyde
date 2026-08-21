from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_fireworks import ChatFireworks
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools import tools, tools_by_name

# --- System prompt ---

system_prompt = (
    "You are an autonomous coding assistant. You can read local files and "
    "execute bash commands to inspect, write, test, or debug code."
)

# --- LLM via Fireworks ---

# You can use DeepSeek-Coder, Qwen2.5-Coder, or Llama models available on Fireworks
llm = ChatFireworks(
    model="accounts/fireworks/models/deepseek-coder-33b-instruct",
    temperature=0,
).bind_tools(tools)

# --- Graph State & Nodes ---

class AgentState(TypedDict):
    # 'add_messages' ensures new messages get appended to context history
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: AgentState):
    """LLM Node: Processes conversation history and outputs a response/tool call."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def call_tools(state: AgentState):
    """Tool Node: Executes tools called by the LLM and feeds output back."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        output = tool_fn.invoke(tool_call["args"])

        tool_results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": tool_results}


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
