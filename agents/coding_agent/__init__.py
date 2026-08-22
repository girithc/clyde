# Coding agent package.
#
# Structure:
#   state.py  - shared AgentState TypedDict
#   model.py  - LLM, system prompt, call_model node
#   tools.py  - call_tools node
#   graph.py  - should_continue router + compiled graph
#
# Exports the agent contract used by agents/__init__.py:
#   - graph          compiled langgraph graph ready to stream
#   - system_prompt  the agent's system message string
from agents.coding_agent.graph import graph
from agents.coding_agent.model import system_prompt

__all__ = ["graph", "system_prompt"]
