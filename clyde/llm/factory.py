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

from langchain_core.globals import set_debug, set_verbose

from clyde.config import load_config, save_config
from clyde.trace import compact_trace
from clyde.auth import get_key
from clyde.llm.registry import PROVIDERS

DEFAULT_PROVIDER = "fireworks"
DEFAULT_MODEL_ID = PROVIDERS[DEFAULT_PROVIDER].default_model

# Current provider/model — updated by `set_model` and read as the default by
# `get_llm`, so every rebuilt LLM picks up the new provider/model. Precedence on
# startup: persisted user choice > built-in default, so a model picked in one
# session carries into the next.
_cfg = load_config()
CURRENT_PROVIDER = _cfg.get("provider") or DEFAULT_PROVIDER
CURRENT_MODEL_ID = _cfg.get("model_id") or DEFAULT_MODEL_ID

# Raw JSON debug/verbose dumps are off; compact structured lines come from
# trace.CompactTraceHandler (attached to every LLM below) instead.
_debug = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"
set_debug(_debug)
set_verbose(_debug)


class NoCredentialsError(RuntimeError):
    """Raised when a provider has no API key in the OS keychain."""


def get_llm(
    provider: str = CURRENT_PROVIDER,
    model: str = CURRENT_MODEL_ID,
    **kwargs,
):
    """Build a chat model for `provider` (lazy-imported) with `model` and kwargs.

    `kwargs` are forwarded to the provider's chat class (e.g. temperature,
    max_tokens) and take precedence over the current defaults. A compact trace
    callback is attached automatically. The API key is read from the OS keychain
    (set via `clyde login`); there is no env-var fallback.
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

    api_key = kwargs.pop("api_key", get_key(provider))
    if not api_key:
        raise NoCredentialsError(
            f"No API key for provider '{provider}' in the keychain. "
            f"Run: clyde login {provider}"
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
    """Set the current provider + model, rebuild the ready instance, and persist.

    The choice is written to ``~/.clyde/config.json`` so it survives across
    sessions; the next process start loads it as the default.
    """
    global CURRENT_PROVIDER, CURRENT_MODEL_ID, llm
    CURRENT_PROVIDER = provider
    CURRENT_MODEL_ID = model_id
    llm = get_llm(temperature=0)
    save_config({"provider": provider, "model_id": model_id})


# Ready-to-use default instance. Bind tools on the caller side:
#   from clyde.llm import llm
#   llm.bind_tools(tools)
# Build with the startup provider/model. If the persisted choice is unusable
# (missing package or bad id), fall back to the built-in default. A missing key
# (NoCredentialsError) is expected before `clyde login` — leave `llm = None` so
# import never bricks; `clyde` checks for this and directs the user to log in.
try:
    llm = get_llm(temperature=0)
except NoCredentialsError:
    llm = None
except Exception:
    CURRENT_PROVIDER = DEFAULT_PROVIDER
    CURRENT_MODEL_ID = DEFAULT_MODEL_ID
    try:
        llm = get_llm(DEFAULT_PROVIDER, DEFAULT_MODEL_ID, temperature=0)
    except NoCredentialsError:
        llm = None
