# Agent registry.
#
# One file per agent type (coding, testing, resolver, qa, ...). When an agent's
# structure gets too complicated for a single file, turn that file into a
# sub-folder (e.g. agents/coding_agent/ with its own __init__.py).
#
# Each agent module exports:
#   - graph          compiled langgraph graph ready to stream
#   - system_prompt  the agent's system message string
from clyde.agents.coding_agent import graph as coding_graph, system_prompt as coding_prompt

# Default agent used by the REPL.
default_graph = coding_graph
default_system_prompt = coding_prompt
