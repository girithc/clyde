"""Shared LLM component.

Every module that needs an LLM imports `get_llm` (or the ready `llm`) from here
instead of constructing its own ChatFireworks client. Defaults are read from the
project `.env`:

    FIREWORKS_API_KEY   - API key for the Fireworks backend
    FIREWORKS_MODEL_ID  - model id, e.g. accounts/fireworks/models/deepseek-v4-flash-0731
"""

import os

from dotenv import load_dotenv
from langchain_core.globals import set_debug, set_verbose
from langchain_fireworks import ChatFireworks

from trace import compact_trace

# Load .env before reading the keys so importers don't have to.
load_dotenv()

DEFAULT_MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash-0731"

# --- Tracing ---
# Raw JSON debug/verbose dumps are off; compact structured lines come from
# trace.CompactTraceHandler (attached to every LLM below) instead.
# Flip LANGCHAIN_DEBUG=true to bring back the full JSON dump for debugging.
_debug = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"
set_debug(_debug)
set_verbose(_debug)

# LangSmith remote tracing is intentionally off; traces stay local.


def get_llm(**kwargs) -> ChatFireworks:
    """Build a ChatFireworks LLM from env defaults, overridden by `kwargs`.

    `kwargs` are forwarded to ChatFireworks (e.g. temperature, model, max_tokens)
    and take precedence over the env defaults. A compact trace callback is
    attached automatically.
    """
    api_key = kwargs.pop("api_key", os.getenv("FIREWORKS_API_KEY"))
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set. Add it to your .env file.")
    model = kwargs.pop("model", os.getenv("FIREWORKS_MODEL_ID", DEFAULT_MODEL_ID))
    callbacks = kwargs.pop("callbacks", None)
    handler_list = [compact_trace] + (callbacks or [])
    return ChatFireworks(api_key=api_key, model=model, callbacks=handler_list, **kwargs)


# Ready-to-use default instance. Bind tools on the caller side:
#   from llm import llm
#   llm.bind_tools(tools)
llm = get_llm(temperature=0)
