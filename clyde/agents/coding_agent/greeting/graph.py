"""Compile the greeting graph: a single node that produces the greeting.

A 1-node MessageState graph so the greeting streams through the same
``stream_mode=["messages", "values"]`` machinery as a normal turn — the
greeting types out live, then the complete message lands.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from clyde.agents.coding_agent.greeting.nodes import greet_node
from clyde.agents.coding_agent.state import MessageState


def build_greet_graph():
    builder = StateGraph(MessageState)
    builder.add_node("greet", greet_node)
    builder.set_entry_point("greet")
    builder.add_edge("greet", END)
    return builder.compile()
