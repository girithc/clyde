import os

from langchain_core.tools import tool

_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build",
}
_SKIP_FILES = {".DS_Store"}


@tool
def get_file_structure(path: str = ".", max_depth: int = 3) -> str:
    """Return a depth-limited directory tree of the project under `path`.

    Useful for a quick overview of layout without reading every file.
    Skips common junk dirs (venv, .git, node_modules, ...).
    """
    if not os.path.isdir(path):
        return f"Not a directory: {path}"

    lines: list[str] = []

    def walk(dir_path: str, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return

        dirs = [e for e in entries
                if os.path.isdir(os.path.join(dir_path, e)) and e not in _SKIP_DIRS]
        files = [e for e in entries
                 if os.path.isfile(os.path.join(dir_path, e)) and e not in _SKIP_FILES]

        items = [(e, True) for e in dirs] + [(e, False) for e in files]
        for i, (name, is_dir) in enumerate(items):
            last = i == len(items) - 1
            connector = "└── " if last else "├── "
            suffix = "/" if is_dir else ""
            lines.append(f"{prefix}{connector}{name}{suffix}")
            if is_dir:
                extension = "    " if last else "│   "
                walk(os.path.join(dir_path, name), prefix + extension, depth + 1)

    # Label the root as `.` so paths read relative to the caller's cwd,
    # matching what read_file / execute_bash expect (no phantom parent dir).
    root_label = path if path != "." else "."
    lines.append(f"{root_label}/")
    walk(path, "", 1)
    return "\n".join(lines)
