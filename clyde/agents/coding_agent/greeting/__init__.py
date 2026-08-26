"""Proactive session greeting.

A 1-node graph that greets the user at session start, context-aware (time,
user name, project, last commit), LLM-generated and streamed into the transcript
like a normal turn. Falls back to a static greet if the LLM call fails.
"""

from clyde.agents.coding_agent.greeting.graph import build_greet_graph
from clyde.agents.coding_agent.greeting.nodes import greet_node
from clyde.agents.coding_agent.greeting.prompts import GREET_PROMPT, build_greet_prompt

__all__ = [
    "GREET_PROMPT",
    "build_greet_graph",
    "build_greet_prompt",
    "greet_node",
]
