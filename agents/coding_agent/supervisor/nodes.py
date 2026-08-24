"""Supervisor nodes: decompose the request, scout shared work, fan out, synthesize."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from llm import get_llm

from agents.coding_agent.model import call_model
from agents.coding_agent.prompts import PLANNER_PROMPT, SYNTHESIZE_PROMPT, system_prompt
from agents.coding_agent.state import Plan, SupervisorState
from agents.coding_agent.tools import call_tools


_base_llm = get_llm(temperature=0)
_planner_model = _base_llm.with_structured_output(Plan, method="function_calling")


def configure_model(provider, model_id):
    """Rebind the supervisor LLMs with a new provider + model (on model switch)."""
    global _base_llm, _planner_model
    from llm import set_model
    set_model(provider, model_id)
    _base_llm = get_llm(provider, model_id, temperature=0)
    _planner_model = _base_llm.with_structured_output(Plan, method="function_calling")

# Hard backstop so the scout's gather loop can never spin forever.
SCOUT_MAX_STEPS = 8


def _user_request(messages) -> str:
    """Return the latest user message; the planner appends its own AIMessage, so
    reading messages[-1] is not safe after it runs."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    raise RuntimeError("supervisor: no user request found in messages")


def planner_node(state: SupervisorState):
    """Decompose the request, or mark it trivial for a direct answer."""
    request = _user_request(state["messages"])
    plan = _planner_model.invoke(f"{PLANNER_PROMPT}\n\nRequest:\n{request}")
    if plan is None:
        raise RuntimeError(f"planner: produced no plan. Raw: {plan!r}")

    if plan.trivial:
        # No fan-out — the direct_answer node handles it in one LLM call.
        return {"trivial": True, "shared": [], "tasks": [], "messages": []}

    if not plan.tasks:
        raise RuntimeError(f"planner: non-trivial but produced no tasks. Raw: {plan!r}")

    shared_lines = "\n".join(f"- [{t.id}] {t.description}" for t in plan.shared)
    task_lines = "\n".join(f"- [{t.id}] {t.description}" for t in plan.tasks)
    plan_text = f"Plan:\nShared (scout, once):\n{shared_lines or '- (none)'}\nTasks:\n{task_lines}"
    return {
        "trivial": False,
        "shared": plan.shared,
        "tasks": plan.tasks,
        "messages": [AIMessage(content=plan_text)],
    }


def route_planner(state: SupervisorState) -> str:
    """Trivial -> direct answer; otherwise -> scout."""
    return "direct_answer" if state.get("trivial") else "scout"


def direct_answer_node(state: SupervisorState):
    """Answer a trivial request in one LLM call — no tools, no fan-out."""
    request = _user_request(state["messages"])
    answer = _base_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=request)]
    )
    return {"messages": [AIMessage(content=answer.content)]}


def scout_node(state: SupervisorState):
    """Run the shared grunt work once via a lightweight executor loop, then
    publish its transcript as `shared_context` for every task worker.

    Not a nested worker subgraph: grunt work needs no plan/select/verify
    overhead, and a plain loop avoids consuming the outer recursion budget.
    Reuses the shared `call_model` (trims context, bound tool-llm) and
    `call_tools` nodes. No-op when the planner emitted no shared tasks.
    """
    shared = state.get("shared") or []
    if not shared:
        return {"shared_context": []}

    desc = "\n".join(f"- [{t.id}] {t.description}" for t in shared)
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Shared grunt work — gather data the task workers will need:\n{desc}"
        ),
    ]
    for _ in range(SCOUT_MAX_STEPS):
        msgs.extend(call_model({"messages": msgs})["messages"])
        if not getattr(msgs[-1], "tool_calls", None):
            break
        msgs.extend(call_tools({"messages": msgs})["messages"])
    return {"shared_context": msgs}


def fan_out(state: SupervisorState):
    """Dispatch one worker per task, in parallel, seeded with scout output."""
    tasks = state["tasks"]
    if not tasks:
        raise RuntimeError("fan_out: no tasks to dispatch")
    request = _user_request(state["messages"])
    shared_context = state.get("shared_context") or []
    return [
        Send("worker", {"task": task, "request": request, "shared_context": shared_context})
        for task in tasks
    ]


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
