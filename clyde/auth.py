"""Credential storage for Clyde — API keys in the OS keychain, no plaintext.

Keys live in the OS keychain (macOS Keychain via the `keyring` library), keyed
by provider name under the service "clyde". There is no on-disk fallback and no
env-var fallback — install `clyde` and run `clyde login` to set keys.

CLI:
    clyde login            set keys for one or more providers (interactive)
    clyde logout [prov]    remove a provider's key (or --all)
    clyde auth             show which providers have keys + the active one
"""

from __future__ import annotations

import getpass
from typing import Iterable

import keyring

from clyde.llm.registry import PROVIDERS

SERVICE = "clyde"


# --- keychain primitives ---

def set_key(provider: str, api_key: str) -> None:
    """Store a provider's API key in the OS keychain."""
    keyring.set_password(SERVICE, provider, api_key)


def get_key(provider: str) -> str | None:
    """Return a provider's API key, or None if not set."""
    return keyring.get_password(SERVICE, provider)


def delete_key(provider: str) -> bool:
    """Remove a provider's key. Returns True if a key was removed."""
    if not has_key(provider):
        return False
    try:
        keyring.delete_password(SERVICE, provider)
    except keyring.PasswordDeleteError:
        return False
    return True


def has_key(provider: str) -> bool:
    return get_key(provider) is not None


def configured_providers() -> list[str]:
    """All registered providers that currently have a key in the keychain."""
    return [p for p in PROVIDERS if has_key(p)]


# --- interactive login ---

def _prompt_providers() -> list[str]:
    """Ask which providers to configure; returns a list of provider names."""
    names = list(PROVIDERS)
    print("Providers: " + ", ".join(names))
    choice = input("Configure which? (comma-separated, or 'all'): ").strip().lower()
    if choice in ("all", "*"):
        return names
    picked = [c.strip() for c in choice.split(",") if c.strip()]
    unknown = [p for p in picked if p not in PROVIDERS]
    if unknown:
        print(f"Unknown provider(s): {', '.join(unknown)}")
        return _prompt_providers()
    return picked or names


def login(providers: Iterable[str] | None = None) -> int:
    """Interactive: prompt for and store keys for one or more providers.

    Returns the number of keys saved. A provider is skipped (not saved) if the
    user submits an empty key at the prompt.
    """
    to_set = list(providers) if providers else _prompt_providers()
    saved = 0
    for provider in to_set:
        key = getpass.getpass(f"API key for {provider} (blank to skip): ")
        if not key:
            print(f"  skipped {provider}")
            continue
        set_key(provider, key)
        saved += 1
        print(f"  saved {provider}")
    print(f"\n{saved} provider key(s) saved to the keychain.")
    return saved


def logout(providers: Iterable[str] | None = None) -> int:
    """Remove keys. If `providers` is None or 'all', removes every configured key."""
    if providers is None:
        targets = configured_providers()
    else:
        targets = [p for p in providers if p in PROVIDERS]
    removed = 0
    for provider in targets:
        if delete_key(provider):
            print(f"  removed {provider}")
            removed += 1
    print(f"\n{removed} provider key(s) removed.")
    return removed


def auth_status() -> None:
    """Print which providers have keys and which is active."""
    from clyde.config import load_config
    cfg = load_config()
    active = cfg.get("provider")
    have = configured_providers()
    if not have:
        print("No API keys set. Run `clyde login` to add one.")
        return
    print("Configured providers:")
    for p in have:
        mark = " (active)" if p == active else ""
        print(f"  - {p}{mark}")
    if active and active not in have:
        print(f"\nActive provider is '{active}' but it has no key — run `clyde login {active}`.")
