"""LLM middleware: provider-abstracted chat model factory.

Re-exports the factory's stable objects (``get_llm``, ``set_model``,
``DEFAULT_MODEL_ID``, ``PROVIDERS``) as name bindings. The factory itself is
imported LAZILY via ``__getattr__`` so that importing this package (or a
submodule like ``clyde.llm.registry``) does NOT eagerly build the LLM — that
keeps ``clyde.auth`` (which only needs ``PROVIDERS``) free of the LLM build and
breaks what would otherwise be an import cycle (auth ↔ factory).

The mutable current-state globals (``CURRENT_MODEL_ID``, ``CURRENT_PROVIDER``,
the ready ``llm`` instance) are also delegated through ``__getattr__`` so a
``from clyde.llm import CURRENT_MODEL_ID`` would capture a stale value at import
time; callers that need the live value use attribute access
(``clyde.llm.CURRENT_MODEL_ID``).
"""

from clyde.llm.registry import PROVIDERS

__all__ = ["DEFAULT_MODEL_ID", "PROVIDERS", "get_llm", "set_model"]

_FACTORY_NAMES = ("DEFAULT_MODEL_ID", "get_llm", "set_model",
                  "CURRENT_MODEL_ID", "CURRENT_PROVIDER", "llm",
                  "NoCredentialsError")


def __getattr__(name):
    if name in _FACTORY_NAMES:
        from clyde.llm import factory as _factory
        return getattr(_factory, name)
    raise AttributeError(f"module 'clyde.llm' has no attribute {name!r}")
