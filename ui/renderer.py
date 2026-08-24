"""Rich renderables for the TUI transcript.

These return rich renderables (not print) so the textual ``RichLog`` can write
them directly. The agent's Markdown answer renders as formatted text (headers,
bold, GFM tables); trace lines render dimmed; user lines render bold.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text


def agent_renderable(text: str) -> Markdown:
    """The agent's Markdown answer — rendered headers/bold/tables/lists."""
    return Markdown(text or "")


def user_renderable(text: str) -> Text:
    """A bold user line."""
    return Text(f"> {text}", style="bold")


def dim_line(text: str) -> Text:
    """A dimmed trace line (🧠/✅/🔧/📥)."""
    return Text(text or "", style="dim")


def welcome_renderable() -> Text:
    """The init line shown when the app mounts."""
    return Text("Clyde initialized! Type 'exit' to quit.", style="cyan")


def spacer_renderable() -> Text:
    """A blank line used to add vertical spacing between transcript blocks."""
    return Text("")
