"""Supervisor nodes: decompose the request, fan out to workers, synthesize."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from llm import get_llm

from agents.coding_agent.prompts import PLANNER_PROMPT, SYNTHESIZE_PROMPT
from agents.coding_agent.state import Plan, SupervisorState


_base_llm = get_llm(temperature=0)
_planner_model = _base_llm.with_structured_output(Plan, method="function_calling")


def _user_request(messages) -> str:
    """Return the latest user message; the planner appends its own AIMessage, so
    reading messages[-1] is not safe after it runs."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    raise RuntimeError("supervisor: no user request found in messages")


def planner_node(state: SupervisorState):
    """Decompose the request into independent tasks (raise on empty)."""
    request = _user_request(state["messages"])
    plan = _planner_model.invoke(f"{PLANNER_PROMPT}\n\nRequest:\n{request}")
    if plan is None or not plan.tasks:
        raise RuntimeError(f"planner: produced no tasks. Raw: {plan!r}")

    task_lines = "\n".join(f"- [{t.id}] {t.description}" for t in plan.tasks)
    return {
        "tasks": plan.tasks,
        "messages": [AIMessage(content=f"📋 Plan:\n{task_lines}")],
    }


def fan_out(state: SupervisorState):
    """Dispatch one worker per task, in parallel (raise on empty)."""
    tasks = state["tasks"]
    if not tasks:
        raise RuntimeError("fan_out: no tasks to dispatch")
    request = _user_request(state["messages"])
    return [Send("worker", {"task": task, "request": request}) for task in tasks]


def synthesize_node(state: SupervisorState):
    """Combine worker outcomes into one answer for the user."""
    results = state["results"]
    if not results:
        raise RuntimeError("synthesize: no worker results received")
    summary = "\n".join(f"- [{r.task_id}] {r.outcome}" for r in results)
    answer = _base_llm.invoke(
        [SystemMessage(content=SYNTHESIZE_PROMPT), HumanMessage(content=summary)]
    )
    return {"messages": [AIMessage(content=answer.content)]}
