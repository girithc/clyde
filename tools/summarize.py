"""Summarize a large block of text down to the key facts.

Call this when a tool output (a big file listing, a long bash dump, a full file
read) is too verbose to reason over directly. Pass the bulky text and an
optional focus; get back a short, dense summary of just the relevant facts.

This is an opt-in compressor — the agent decides when an output is too big and
calls it, rather than every result being auto-summarized.
"""

from langchain_core.tools import tool

from llm import get_llm

# Plain LLM with no tools bound, so it can't recurse into another tool call.
_llm = get_llm(temperature=0)


def configure_model(provider, model_id):
    """Rebind the summarize tool's LLM with a new provider + model (on model switch)."""
    global _llm
    from llm import set_model
    set_model(provider, model_id)
    _llm = get_llm(provider, model_id, temperature=0)


@tool
def summarize(text: str, focus: str = "") -> str:
    """Condense `text` into the key facts relevant to `focus` (empty = general).

    Use to shrink a large tool output before reasoning over it. Returns a short
    summary, not the original text.
    """
    focus_line = f" Focus on: {focus}." if focus else ""
    prompt = (
        "Summarize the following into the key facts a coding assistant needs."
        f"{focus_line} Keep it short and dense — facts only, no preamble."
        f"\n\n{text}"
    )
    return _llm.invoke(prompt).content
