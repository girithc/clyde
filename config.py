"""User-level config persistence for Clyde.

A small JSON file at ``~/.clyde/config.json`` holds user preferences that should
survive across sessions — currently the chosen provider + model id. All access
is best-effort: a missing/corrupt file is treated as empty, and write failures
are swallowed so a read-only home dir never crashes the app.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".clyde"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Read the user config, or return ``{}`` if missing/corrupt/unreadable."""
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    """Write the user config, creating the dir if needed. Best-effort."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass  # persistence is a convenience, never fatal
