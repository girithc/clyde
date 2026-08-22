import fnmatch
import os

from langchain_core.tools import tool

_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build",
}


@tool
def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files under `path` (recursive) whose path matches glob `pattern`.

    Returns one relative path per line. Skips common junk dirs.
    Example: list_files("src", "*.py") -> all .py files under src.
    """
    if not os.path.isdir(path):
        return f"Not a directory: {path}"

    results: list[str] = []

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            file_path = os.path.relpath(os.path.join(root, name), path)
            # Match against both the relative path and the bare filename.
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(name, pattern):
                results.append(os.path.join(path, file_path))

    if not results:
        return f"No files matching '{pattern}' under {path}."
    results.sort()
    return "\n".join(results)
