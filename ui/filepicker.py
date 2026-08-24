"""Native macOS file picker for attaching an image to a prompt.

Uses AppleScript via ``osascript`` (the real macOS open-file dialog) so there's
no Python Tk dependency. Call ``pick_image_native`` from a worker thread — it
blocks until the user confirms or cancels, returning the path string or ``None``.
"""

from __future__ import annotations

import subprocess


def pick_image_native() -> str | None:
    """Open the native macOS file dialog restricted to image files.

    Blocks until the user selects a file or cancels. Returns the POSIX path
    string, or ``None`` if cancelled/closed. Safe to call off the UI thread.
    """
    script = (
        'set chosenFile to choose file of type {"public.image"} '
        'with prompt "Attach image" without multiple selections allowed\n'
        'return POSIX path of chosenFile'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # User cancelled → osascript exits non-zero with no stdout.
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip() or None
