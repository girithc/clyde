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

# provider name -> substrings that mark a model id as vision-capable. Matched
# case-insensitively against the free-form model id the user types in the inline
# model-edit row. A model is vision-capable if any substring is in the id.
_VISION_PATTERNS: dict[str, tuple[str, ...]] = {
    # all Claude 3+ see images; claude-2 / claude-instant do not
    "anthropic": (
        "claude-3", "claude-sonnet", "claude-opus", "claude-haiku",
        "claude-3.5", "claude-3.7",
    ),
    "openai": (
        "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-4.1",
        "o1", "o3", "chatgpt-4o",
    ),
    # openrouter ids look like "anthropic/claude-3.5-sonnet" — match the union
    # of upstream markers (anthropic + openai + gemini) against the full id
    "openrouter": (
        "claude-3", "claude-sonnet", "claude-opus", "claude-haiku",
        "gpt-4o", "gpt-4-vision", "gpt-4.1", "o1", "o3", "gemini",
    ),
    # fireworks vision model ids contain "vision"
    # (e.g. accounts/fireworks/models/llama-3.2-90b-vision-instruct);
    # muse models (e.g. muse-glimmer-30b) are also multimodal.
    "fireworks": ("vision", "muse"),
}


def _pattern_vision(provider: str, model_id: str | None) -> bool:
    """True if the model id matches a known vision pattern for the provider."""
    if not provider or not model_id:
        return False
    patterns = _VISION_PATTERNS.get(provider)
    if not patterns:
        return False
    lowered = model_id.lower()
    return any(p in lowered for p in patterns)


# Cache of (provider, model_id) -> vision-capable, populated by probe_vision()
# for models the pattern list doesn't recognize (e.g. muse-glimmer-30b). Lives
# for the process lifetime so each unknown model is probed at most once.
_VISION_CACHE: dict[tuple[str, str], bool] = {}


def _vision_key(provider: str, model_id: str | None) -> tuple[str, str]:
    return ((provider or "").lower(), (model_id or "").lower())


def supports_vision(provider: str, model_id: str | None) -> bool:
    """True if the provider+model id is vision-capable (image input).

    Instant and never blocks: a known pattern match wins immediately, otherwise
    a cached probe result is used. Returns False for unknown providers or empty
    model ids. For models neither pattern nor cache resolves, call
    `probe_vision` (a blocking LLM yes/no question) to populate the cache; check
    `needs_vision_probe` first to avoid redundant probes.
    """
    if _pattern_vision(provider, model_id):
        return True
    return _VISION_CACHE.get(_vision_key(provider, model_id), False)


def needs_vision_probe(provider: str, model_id: str | None) -> bool:
    """True if vision capability is currently unknown (no pattern, no cache)."""
    if _pattern_vision(provider, model_id):
        return False
    return _vision_key(provider, model_id) not in _VISION_CACHE


def probe_vision(provider: str, model_id: str) -> bool:
    """Blocking: ask the current LLM whether `model_id` supports image input.

    Used as a fallback for models the pattern list doesn't recognize. Caches the
    answer so subsequent `supports_vision` calls are instant and the probe runs
    at most once per model per process. Returns False (and caches False) if the
    call fails or the answer is not a clear YES.
    """
    key = _vision_key(provider, model_id)
    prompt = (
        f'Does the {provider} model "{model_id}" support image/vision input '
        f"(multimodal image understanding)? "
        f"Answer with exactly one word: YES or NO. If unsure, answer NO."
    )
    try:
        from langchain_core.messages import HumanMessage

        from llm import get_llm

        probe_llm = get_llm(temperature=0)
        resp = probe_llm.invoke([HumanMessage(content=prompt)])
        text = (getattr(resp, "content", "") or "").strip().lower()
        result = text.startswith("yes")
    except Exception:
        result = False
    _VISION_CACHE[key] = bool(result)
    return bool(result)
