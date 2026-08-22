"""LLM + model node for the coding agent."""

from llm import get_llm
from tools import tools

from agents.coding_agent.state import AgentState

# --- System prompt ---

system_prompt = (
    "You are an autonomous coding assistant. You can read local files and "
    "execute bash commands to inspect, write, test, or debug code."
)

# --- LLM via shared component (see llm.py) ---

llm = get_llm(temperature=0).bind_tools(tools)


def call_model(state: AgentState):
    """LLM Node: Processes conversation history and outputs a response/tool call."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
