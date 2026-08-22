"""Nodes for the per-task worker subgraph.

Each worker is isolated: its transcript lives in `context` (the supervisor never
sees it), so parallel workers cannot read each other's tool output. Only the
final `results` value bridges back to the supervisor.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from llm import get_llm
from tools import get_file_structure, read_file

from agents.coding_agent.model import call_model
from agents.coding_agent.prompts import (
    CONTEXT_SELECT_PROMPT,
    TASK_PLANNER_PROMPT,
    VERIFY_PROMPT,
)
from agents.coding_agent.state import (
    Approach,
    RelevantPaths,
    TaskResult,
    VerifyVerdict,
    WorkerState,
)
from agents.coding_agent.tools import call_tools


_base_llm = get_llm(temperature=0)
_plan_model = _base_llm.with_structured_output(Approach, method="function_calling")
_context_model = _base_llm.with_structured_output(RelevantPaths, method="function_calling")
_verify_model = _base_llm.with_structured_output(VerifyVerdict, method="function_calling")


def task_planner_node(state: WorkerState):
    """Seed a step-by-step approach into the worker's context."""
    task = state["task"]
    approach = _plan_model.invoke(f"{TASK_PLANNER_PROMPT}\n\nTask: {task.description}")
    if approach is None or not approach.plan.strip():
        raise ValueError(
            f"task_planner: empty plan for task {task.id!r}; got {approach!r}"
        )
    return {"context": [SystemMessage(content=f"Plan for '{task.id}':\n{approach.plan}")]}


def select_context_node(state: WorkerState):
    """Pre-load the project tree and the files relevant to this task."""
    task = state["task"]
    tree = get_file_structure.invoke({})

    relevant = _context_model.invoke(
        f"{CONTEXT_SELECT_PROMPT}\n\nTask: {task.description}\n\nFile tree:\n{tree}"
    )
    if relevant is None or relevant.paths is None:
        raise ValueError(
            f"select_context: no paths for task {task.id!r}; got {relevant!r}"
        )

    messages = [SystemMessage(content=f"Project tree:\n{tree}")]
    for path in relevant.paths:
        contents = read_file.invoke({"file_path": path})
        messages.append(SystemMessage(content=f"--- {path} ---\n{contents}"))
    messages.append(
        HumanMessage(content=f"Task: {task.description}\n\nRequest: {state['request']}")
    )
    return {"context": messages}


def executor_model(state: WorkerState):
    """Run the shared agent node over this worker's isolated context."""
    result = call_model({"messages": state["context"]})
    return {"context": result["messages"]}


def executor_tools(state: WorkerState):
    """Run the shared tool node over this worker's isolated context."""
    result = call_tools({"messages": state["context"]})
    return {"context": result["messages"]}


def worker_should_continue(state: WorkerState):
    """Route the executor: more tools, or self-review."""
    last = state["context"][-1]
    if getattr(last, "tool_calls", None):
        return "executor_tools"
    return "verify"


def verify_node(state: WorkerState):
    """Self-review. Done -> publish result. Otherwise append feedback and loop."""
    task = state["task"]
    verdict = _verify_model.invoke(
        f"{VERIFY_PROMPT}\n\nTask: {task.description}\n\nTranscript:\n{_render(state['context'])}"
    )
    if verdict is None:
        raise ValueError(f"verify: no verdict for task {task.id!r}")

    if verdict.done:
        outcome = state["context"][-1].content if state["context"] else ""
        return {"results": [TaskResult(task_id=task.id, outcome=str(outcome))]}
    return {"context": [HumanMessage(content=f"Not done. Feedback: {verdict.feedback}")]}


def verify_route(state: WorkerState):
    """After self-review: loop back to the executor until a result is produced."""
    if state["results"]:
        return END
    return "executor_model"


def _render(messages) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", "message")
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)
