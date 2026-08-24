"""Terminal UI for Clyde (textual TUI + rich renderables)."""

from ui.renderer import (
    agent_renderable,
    spacer_renderable,
    user_renderable,
    welcome_renderable,
)
from ui.app import ClydeApp
from ui.transcript import Transcript

__all__ = [
    "ClydeApp",
    "Transcript",
    "agent_renderable",
    "spacer_renderable",
    "user_renderable",
    "welcome_renderable",
]
