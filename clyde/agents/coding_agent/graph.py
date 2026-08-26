"""Compile the coding agent graph.

Top-level graph = supervisor that plans the request and fans it out to parallel
worker subgraphs, then synthesizes one answer. Compiled with an in-process
checkpointer so `interrupt()` (the clarify/ask step) can pause and resume a turn.
"""

from langgraph.checkpoint.memory import MemorySaver

from clyde.agents.coding_agent.supervisor.graph import build_supervisor
from clyde.agents.coding_agent.worker.graph import build_worker

# Shared in-process checkpointer; turns are keyed by thread_id (set per turn in the UI).
_checkpointer = MemorySaver()
graph = build_supervisor(build_worker(), checkpointer=_checkpointer)
