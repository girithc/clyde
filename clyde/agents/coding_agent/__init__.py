# Coding agent package.
#
# Structure:
#   state.py      - Task/TaskResult models + SupervisorState/WorkerState
#   prompts.py    - all prompt strings (incl. the exported system_prompt)
#   model.py      - shared executor LLM binding + call_model node
#   tools.py      - call_tools node
#   worker/       - per-task subgraph (plan + context + executor + verify)
#   supervisor/   - planner + fan-out + synthesize
#   graph.py      - compiles worker + supervisor into the public `graph`
#
# Exports the agent contract used by agents/__init__.py:
#   - graph          compiled langgraph graph ready to stream
#   - system_prompt  the agent's system message string
from clyde.agents.coding_agent.graph import graph
from clyde.agents.coding_agent.prompts import system_prompt

__all__ = ["graph", "system_prompt"]
