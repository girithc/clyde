"""LLM factory: build a chat model for any registered provider, lazily.

We use langchain as the middleware so provider differences (auth, API shape)
are abstracted away. A provider's langchain connector is imported only when
``get_llm`` is called for that provider (via ``importlib``), so we don't require
every provider package to be installed at startup — only the one in use.

Callers do ``from llm import get_llm`` (or ``from llm import llm``) and stay
provider-agnostic; the factory routes by the current provider/model.
"""

from __future__ import annotations

import importlib
import os

from dotenv import load_dotenv
from langchain_core.globals import set_debug, set_verbose

from trace import compact_trace

# Load .env before reading keys so importers don't have to.
load_dotenv()

DEFAULT_MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash-0731"

# Current provider/model — updated by `set_model` and read as the default by
# `get_llm`, so every rebuilt LLM picks up the new provider/model.
CURRENT_PROVIDER = "fireworks"
CURRENT_MODEL_ID = os.getenv("FIREWORKS_MODEL_ID", DEFAULT_MODEL_ID)

# Raw JSON debug/verbose dumps are off; compact structured lines come from
# trace.CompactTraceHandler (attached to every LLM below) instead.
_debug = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"
set_debug(_debug)
set_verbose(_debug)

from llm.registry import PROVIDERS


def get_llm(
    provider: str = CURRENT_PROVIDER,
    model: str = CURRENT_MODEL_ID,
    **kwargs,
):
    """Build a chat model for `provider` (lazy-imported) with `model` and kwargs.

    `kwargs` are forwarded to the provider's chat class (e.g. temperature,
    max_tokens) and take precedence over the current defaults. A compact trace
    callback is attached automatically.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {list(PROVIDERS)}"
        )
    cfg = PROVIDERS[provider]
    try:
        module = importlib.import_module(cfg.module)
    except ImportError as e:
        raise RuntimeError(
            f"provider '{provider}' needs {cfg.module}; pip install it. ({e})"
        ) from e
    cls = getattr(module, cfg.cls)

    api_key = kwargs.pop("api_key", os.getenv(cfg.api_key_env))
    if not api_key:
        raise RuntimeError(
            f"{cfg.api_key_env} is not set. Add it to your .env file."
        )
    if cfg.base_url is not None:
        kwargs.setdefault("base_url", cfg.base_url)
    # streaming=True so token callbacks fire during .invoke(); langgraph's
    # stream_mode="messages" turns these into AIMessageChunks for live output.
    kwargs.setdefault("streaming", True)
    callbacks = kwargs.pop("callbacks", None)
    handler_list = [compact_trace] + (callbacks or [])
    return cls(api_key=api_key, model=model, callbacks=handler_list, **kwargs)


def set_model(provider: str, model_id: str) -> None:
    """Set the current provider + model and rebuild the ready instance."""
    global CURRENT_PROVIDER, CURRENT_MODEL_ID, llm
    CURRENT_PROVIDER = provider
    CURRENT_MODEL_ID = model_id
    llm = get_llm(temperature=0)


# Ready-to-use default instance. Bind tools on the caller side:
#   from llm import llm
#   llm.bind_tools(tools)
llm = get_llm(temperature=0)
