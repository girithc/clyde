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
    """The agent's Markdown answer — rendered headers/bold/tables/lists.

    Prose is tinted green so Clyde's responses read as a distinct voice from the
    user's cyan-bold lines; fenced code blocks keep their syntax theme (monokai).
    """
    return Markdown(text or "", style="green")


def user_renderable(text: str, images: list[str] | None = None) -> Text:
    """A cyan-bold user line, with any attached image filenames shown inline."""
    line = Text(f"> {text}", style="bold cyan")
    for name in images or []:
        line.append(f"  🖼 {name}", style="bold cyan")
    return line


def dim_line(text: str) -> Text:
    """A dimmed trace line (🧠/✅/🔧/📥)."""
    return Text(text or "", style="dim")


def welcome_renderable() -> Text:
    """The init line shown when the app mounts."""
    return Text("Clyde initialized! Type 'exit' to quit.", style="cyan")


def spacer_renderable() -> Text:
    """A blank line used to add vertical spacing between transcript blocks."""
    return Text("")
