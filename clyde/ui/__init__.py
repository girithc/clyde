"""Terminal UI for Clyde (textual TUI + rich renderables)."""

from clyde.ui.renderer import (
    agent_renderable,
    spacer_renderable,
    user_renderable,
    welcome_renderable,
)
from clyde.ui.app import ClydeApp
from clyde.ui.transcript import Transcript

__all__ = [
    "ClydeApp",
    "Transcript",
    "agent_renderable",
    "spacer_renderable",
    "user_renderable",
    "welcome_renderable",
]
