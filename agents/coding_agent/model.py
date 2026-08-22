"""Shared executor LLM binding + agent node used by every worker."""

from langchain_core.messages.utils import trim_messages

from llm import get_llm
from tools import tools

from agents.coding_agent.state import MessageState

# Tool-calling agent LLM shared by all workers.
llm = get_llm(temperature=0).bind_tools(tools)

# Drop oldest turns once context exceeds this many (approx) tokens.
MAX_CONTEXT_TOKENS = 24000


def _approx_tokens(messages) -> int:
    """~4 chars per token — good enough for trimming, no tokenizer needed."""
    return sum(len(m.content) for m in messages if isinstance(m.content, str)) // 4


def call_model(state: MessageState):
    """Agent node: trim the transcript, then invoke the model."""
    messages = trim_messages(
        state["messages"],
        max_tokens=MAX_CONTEXT_TOKENS,
        strategy="last",
        token_counter=_approx_tokens,
        include_system=True,
        start_on="human",
    )
    response = llm.invoke(messages)
    return {"messages": [response]}
