"""State and structured-output schemas for the coding agent.

Two graph states:

  SupervisorState — the outer planner -> workers -> synthesize graph; the REPL
      streams from this. `messages` carries only the user-facing transcript.
  WorkerState    — one isolated per-task subgraph. Its transcript lives in
      `context`, which the supervisor never sees, so parallel workers cannot
      read each other's tool output. Only `results` bridges back to the parent.
"""

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Task(BaseModel):
    """One independent unit of work emitted by the planner."""

    id: str = Field(description="Short kebab-case id, e.g. 'add-helpers'.")
    description: str = Field(description="One clear, self-contained instruction.")


class Plan(BaseModel):
    """The planner's decomposition of a request."""

    trivial: bool = Field(
        default=False,
        description=(
            "True when the request needs no tools or decomposition — a simple "
            "question, explanation, or single direct answer the model can give "
            "in one shot. The supervisor then answers directly with no fan-out."
        ),
    )
    shared: list[Task] = Field(
        default_factory=list,
        description=(
            "Grunt work needed by more than one task (file discovery, shared "
            "reads). Done once by the scout, not per task. Empty when no work "
            "is shared."
        ),
    )
    tasks: list[Task]


class TaskResult(BaseModel):
    """A worker's completed outcome, aggregated into the supervisor."""

    task_id: str
    outcome: str


class VerifyVerdict(BaseModel):
    """Per-task self-review: is the task done?"""

    done: bool
    feedback: str = Field(description="Actionable next step; empty when done.")


class Approach(BaseModel):
    """Per-task plan produced at the start of each worker."""

    plan: str = Field(description="Concise step-by-step approach for this task.")


class MessageState(TypedDict):
    """Minimal message-carrying state used by the shared executor nodes."""

    messages: Annotated[list[BaseMessage], add_messages]


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    trivial: bool
    shared: list[Task]
    shared_context: Annotated[list[BaseMessage], add_messages]
    tasks: Annotated[list[Task], add]
    results: Annotated[list[TaskResult], add]


class WorkerState(TypedDict):
    task: Task
    request: str
    shared_context: list[BaseMessage]
    context: Annotated[list[BaseMessage], add_messages]
    results: Annotated[list[TaskResult], add]
