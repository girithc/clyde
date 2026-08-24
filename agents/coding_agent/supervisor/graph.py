"""Compile the supervisor: planner -> (direct | scout -> fan-out -> synthesize)."""

from langgraph.graph import END, StateGraph

from agents.coding_agent.state import SupervisorState
from agents.coding_agent.supervisor import nodes


def build_supervisor(worker):
    builder = StateGraph(SupervisorState)

    builder.add_node("planner", nodes.planner_node)
    builder.add_node("direct_answer", nodes.direct_answer_node)
    builder.add_node("scout", nodes.scout_node)
    builder.add_node("worker", worker)
    builder.add_node("synthesize", nodes.synthesize_node)

    builder.set_entry_point("planner")
    builder.add_conditional_edges("planner", nodes.route_planner, ["direct_answer", "scout"])
    builder.add_edge("direct_answer", END)
    builder.add_conditional_edges("scout", nodes.fan_out, ["worker"])
    builder.add_edge("worker", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
