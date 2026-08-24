"""Shared executor LLM binding + agent node used by every worker."""

from langchain_core.messages.utils import trim_messages

from llm import get_llm
from tools import tools

from agents.coding_agent.state import MessageState

# Tool-calling agent LLM shared by all workers. Default binding uses the static
# tool set so bare imports work; call `configure_executor` at startup to inject
# extra tools (e.g. MCP tools) without rebuilding the compiled graph — `call_model`
# reads this module global at call time, so the rebind takes effect live.
llm = get_llm(temperature=0).bind_tools(tools)


def configure_executor(tool_list):
    """Rebind the executor LLM with a new tool set (called at startup)."""
    global llm
    llm = get_llm(temperature=0).bind_tools(tool_list)


def configure_model(provider, model_id, tool_list):
    """Rebind the executor LLM with a new provider + model (called on model switch)."""
    global llm
    from llm import set_model
    set_model(provider, model_id)
    llm = get_llm(provider, model_id, temperature=0).bind_tools(tool_list)


# Drop oldest turns once context exceeds this many (approx) tokens.
MAX_CONTEXT_TOKENS = 24000


def _approx_tokens(messages) -> int:
    """~4 chars per token — good enough for trimming, no tokenizer needed.

    Content-aware: multimodal messages carry a list of parts, so we sum the
    text-part lengths and charge a flat ~1000 tokens per image so images are
    not trimmed for free once context pressure builds.
    """
    total = 0
    for m in messages:
        c = m.content
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "image_url":
                    total += 1000
                else:
                    total += len(p.get("text", ""))
    return total // 4


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
