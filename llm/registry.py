"""Provider registry for the LLM factory.

Maps a provider name to its langchain connector: module path, chat class,
API-key env var, and (for OpenAI-compatible routers) an optional ``base_url``.
The factory lazy-imports a provider's module only when it's selected, so you
only need the langchain connector package for the provider you actually use.

To add a provider, append an entry here; the model panel picks it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Provider:
    module: str
    cls: str
    api_key_env: str
    base_url: str | None = None  # OpenAI-compatible providers (e.g. OpenRouter)


# provider name -> connector config
PROVIDERS: dict[str, Provider] = {
    "fireworks": Provider(
        "langchain_fireworks", "ChatFireworks", "FIREWORKS_API_KEY"
    ),
    "anthropic": Provider(
        "langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"
    ),
    "openai": Provider(
        "langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"
    ),
    "openrouter": Provider(
        "langchain_openai",
        "ChatOpenAI",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
    ),
}
