"""Shared graph state for the coding agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 'add_messages' ensures new messages get appended to context history.
    messages: Annotated[list[BaseMessage], add_messages]
