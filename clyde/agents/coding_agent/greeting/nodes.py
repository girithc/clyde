"""Nodes for the proactive greeting graph.

A single node that calls the LLM (no tools, no fan-out) with a context-aware
greeting prompt and returns the greeting as an AIMessage. The app streams it
into the transcript at session start.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from clyde.llm import get_llm

from clyde.agents.coding_agent.greeting.prompts import GREET_PROMPT, build_greet_prompt
from clyde.agents.coding_agent.state import MessageState

# Plain LLM (current provider/model), no tools bound.
_llm = get_llm(temperature=0)


def greet_node(state: MessageState):
    """Generate a proactive greeting from session context."""
    prompt = build_greet_prompt()
    response = _llm.invoke(
        [SystemMessage(content=GREET_PROMPT), HumanMessage(content=prompt)]
    )
    return {"messages": [AIMessage(content=response.content)]}
