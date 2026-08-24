"""LLM middleware: provider-abstracted chat model factory.

Re-exports the factory's stable objects (``get_llm``, ``set_model``, ``DEFAULT_MODEL_ID``,
``PROVIDERS``) as name bindings. The mutable current-state globals
(``CURRENT_MODEL_ID``', ``CURRENT_PROVIDER``', and the ready ``llm`` instance) are
NOT re-exported as names — a ``from llm import CURRENT_MODEL_ID`` would capture a
stale value at import time. Instead they're delegated through ``__getattr__`` so
callers that need the live value use attribute access (``llm.CURRENT_MODEL_ID``).
"""

from llm import factory as _factory
from llm.factory import DEFAULT_MODEL_ID, get_llm, set_model
from llm.registry import PROVIDERS

__all__ = ["DEFAULT_MODEL_ID", "PROVIDERS", "get_llm", "set_model"]


def __getattr__(name):
    # Delegate mutable state to the factory module so attribute access always
    # returns the latest value (a `from llm import <name>` would go stale).
    if name in ("CURRENT_MODEL_ID", "CURRENT_PROVIDER", "llm"):
        return getattr(_factory, name)
    raise AttributeError(f"module 'llm' has no attribute {name!r}")
