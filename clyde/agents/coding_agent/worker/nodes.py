"""Nodes for the per-task worker subgraph.

Each worker is isolated: its transcript lives in `context` (the supervisor never
sees it), so parallel workers cannot read each other's tool output. Only the
final `results` value bridges back to the supervisor.

The worker is NOT steered to inspect files. File/code access is just tools
(`read_file`, `get_file_structure`, `list_files`, `search_files`, ...) the
executor calls only when the task actually needs them. The worker seeds a plan
plus any scout-provided shared context, then runs the executor loop.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from clyde.llm import get_llm

from clyde.agents.coding_agent.model import call_model
from clyde.agents.coding_agent.prompts import TASK_PLANNER_PROMPT, VERIFY_PROMPT
from clyde.agents.coding_agent.state import Approach, TaskResult, VerifyVerdict, WorkerState
from clyde.agents.coding_agent.tools import call_tools


_base_llm = get_llm(temperature=0)
_plan_model = _base_llm.with_structured_output(Approach, method="function_calling")
_verify_model = _base_llm.with_structured_output(VerifyVerdict, method="function_calling")


def configure_model(provider, model_id):
    """Rebind the worker LLMs with a new provider + model (on model switch)."""
    global _base_llm, _plan_model, _verify_model
    from clyde.llm import set_model
    set_model(provider, model_id)
    _base_llm = get_llm(provider, model_id, temperature=0)
    _plan_model = _base_llm.with_structured_output(Approach, method="function_calling")
    _verify_model = _base_llm.with_structured_output(VerifyVerdict, method="function_calling")


def task_planner_node(state: WorkerState):
    """Seed the worker's context: scout shared data (if any), a plan, the task."""
    task = state["task"]
    approach = _plan_model.invoke(f"{TASK_PLANNER_PROMPT}\n\nTask: {task.description}")
    if approach is None or not approach.plan.strip():
        raise ValueError(
            f"task_planner: empty plan for task {task.id!r}; got {approach!r}"
        )

    messages = []
    # Scout's gathered data, if this request had shared grunt work.
    shared = state.get("shared_context") or []
    messages.extend(list(shared))
    messages.append(SystemMessage(content=f"Plan for '{task.id}':\n{approach.plan}"))
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
