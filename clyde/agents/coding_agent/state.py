"""State and structured-output schemas for the coding agent.

Two graph states:

  SupervisorState — the outer planner -> workers -> synthesize graph; the REPL
      streams from this. `messages` carries only the user-facing transcript.
  WorkerState    — one isolated per-task subgraph. Its transcript lives in
      `context`, which the supervisor never sees, so parallel workers cannot
      read each other's tool output. Only `results` bridges back to the parent.
"""

from operator import add
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Task(BaseModel):
    """One independent unit of work emitted by the planner."""

    id: str = Field(description="Short kebab-case id, e.g. 'add-helpers'.")
    description: str = Field(description="One clear, self-contained instruction.")


class ClarifyOption(BaseModel):
    """One choice the user can pick when Clyde asks for clarification."""

    label: str = Field(description="Short option label, e.g. 'Patch the bug'.")
    description: str = Field(
        default="",
        description="One-line subtitle explaining what this choice entails.",
    )
    recommended: bool = Field(
        default=False,
        description="True for the single option Clyde recommends. Put it first.",
    )


class ClarifyQuestion(BaseModel):
    """One clarifying question surfaced to the user before any work begins."""

    question: str = Field(description="The clarifying question to show the user.")
    options: list[ClarifyOption] = Field(
        description="Distinct, mutually exclusive choices. Recommended option first."
    )
    multi_select: bool = Field(
        default=False,
        description="True if the user may pick multiple options (reserved; UI is single-select).",
    )


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
    clarify: Optional[list[ClarifyQuestion]] = Field(
        default=None,
        description=(
            "A list of clarifying questions to surface to the user BEFORE any "
            "work, when the request is genuinely ambiguous and guessing wrong "
            "is costly (multiple valid interpretations, missing approach-defining "
            "constraints, or an irreversible fork). Like `tasks`, there can be "
            "one or many — each an independent question with its own options. Do "
            "not set when the request is trivial, clear, or a cheap default "
            "exists. Never re-set once the user has chosen."
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
    clarify: Optional[list[ClarifyQuestion]]
    clarify_choice: Optional[str]
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
