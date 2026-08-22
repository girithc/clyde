"""Compile the worker subgraph: plan -> context -> executor loop -> verify."""

from langgraph.graph import END, StateGraph

from agents.coding_agent.state import WorkerState
from agents.coding_agent.worker import nodes


def build_worker():
    builder = StateGraph(WorkerState)

    builder.add_node("task_planner", nodes.task_planner_node)
    builder.add_node("select_context", nodes.select_context_node)
    builder.add_node("executor_model", nodes.executor_model)
    builder.add_node("executor_tools", nodes.executor_tools)
    builder.add_node("verify", nodes.verify_node)

    builder.set_entry_point("task_planner")
    builder.add_edge("task_planner", "select_context")
    builder.add_edge("select_context", "executor_model")
    builder.add_conditional_edges(
        "executor_model",
        nodes.worker_should_continue,
        {"executor_tools": "executor_tools", "verify": "verify"},
    )
    builder.add_edge("executor_tools", "executor_model")
    builder.add_conditional_edges(
        "verify",
        nodes.verify_route,
        {"executor_model": "executor_model", END: END},
    )

    return builder.compile()
