import os
import re

from langchain_core.tools import tool

# Directories never worth searching.
_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build",
}


@tool
def search_files(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Regex-search file contents under `path` (recursive).

    Returns matching lines as `file:line: text`, capped at `max_results`.
    Skips binary files and common junk dirs (venv, .git, node_modules, ...).
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches: list[str] = []
    scanned = 0

    for root, dirs, files in os.walk(path):
        # Prune junk dirs in-place so os.walk skips them.
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]

        for name in files:
            file_path = os.path.join(root, name)
            if not _is_text_file(file_path):
                continue
            scanned += 1
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append(f"{file_path}:{lineno}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                matches.append(
                                    f"\n[...truncated at {max_results} matches]"
                                )
                                return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue

    if not matches:
        return f"No matches for /{pattern}/ in {path} ({scanned} files scanned)."
    return "\n".join(matches)


def _is_text_file(file_path: str) -> bool:
    """Cheap text-file check: skip files with a binary-ish extension or NUL bytes."""
    if file_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
                           ".gz", ".tar", ".pyc", ".so", ".dylib", ".class")):
        return False
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(2048)
        if b"\x00" in chunk:
            return False
    except OSError:
        return False
    return True
