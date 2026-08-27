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

# Hard cap on verify-loop retries per task. Past this, the worker stops looping
# and publishes a best-effort result instead of spinning to the graph recursion
# limit and crashing the whole turn.
MAX_VERIFY_LOOPS = 5


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
    """Self-review. Done -> publish result. Otherwise append rich, strategy-
    changing feedback and loop. After MAX_VERIFY_LOOPS retries, stop and publish
    a best-effort result so a stuck worker can't spin to the recursion limit."""
    task = state["task"]
    steps = state.get("steps", 0) + 1  # this verify round

    # Cap: force-finalize instead of looping forever.
    if steps > MAX_VERIFY_LOOPS:
        outcome = _last_outcome(state["context"]) or "(no output produced)"
        return {
            "steps": steps,
            "results": [
                TaskResult(
                    task_id=task.id,
                    outcome=f"[best-effort, hit {MAX_VERIFY_LOOPS} verify retries]\n{outcome}",
                )
            ],
        }

    verdict = _verify_model.invoke(
        f"{VERIFY_PROMPT}\n\nTask: {task.description}\n\nTranscript:\n{_render(state['context'])}"
    )
    if verdict is None:
        raise ValueError(f"verify: no verdict for task {task.id!r}")

    if verdict.done:
        outcome = state["context"][-1].content if state["context"] else ""
        return {"steps": steps, "results": [TaskResult(task_id=task.id, outcome=str(outcome))]}

    # Not done: hand the model its prior attempts + the desired result and tell
    # it to try a DIFFERENT strategy, not repeat the same steps.
    feedback = _retry_feedback(task, state["request"], state["context"], verdict.feedback)
    return {"steps": steps, "context": [HumanMessage(content=feedback)]}


def _retry_feedback(task, request: str, context, verifier_feedback: str) -> str:
    """Build a strategy-pivoting prompt: restate the goal, enumerate what was
    already tried (so repetition is obvious), and require a different approach."""
    attempts = _summarize_attempts(context)
    return (
        f"You are not done yet. Reassess and try a DIFFERENT strategy — do NOT repeat "
        f"the steps below.\n\n"
        f"Task: {task.description}\n"
        f"Desired result: {request}\n"
        f"What you already tried (do not repeat):\n{attempts}\n\n"
        f"Verifier feedback: {verifier_feedback}\n\n"
        f"Pick a different approach or write the result directly. Do not call the "
        f"same tools with the same arguments again."
    )


def _summarize_attempts(context) -> str:
    """Compact list of distinct tool calls + short output previews so the model
    can see what it already did."""
    seen = []
    lines = []
    for m in context:
        calls = getattr(m, "tool_calls", None) or []
        for c in calls:
            key = (c.get("name"), str(c.get("args", "")))
            if key in seen:
                continue
            seen.append(key)
            lines.append(f"- {c['name']}({c.get('args', '')})")
        # ToolMessage: attach a short preview of the result to the last call line.
        if getattr(m, "type", None) == "tool":
            preview = str(getattr(m, "content", "")).strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            if lines:
                lines[-1] += f" → {preview}"
    return "\n".join(lines) if lines else "- (no tool calls yet)"


def _last_outcome(context) -> str:
    """Best-effort result text from the worker's transcript (last AI message)."""
    for m in reversed(context):
        content = getattr(m, "content", "")
        if getattr(m, "type", None) == "ai" and isinstance(content, str) and content.strip():
            return content
    return ""


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
