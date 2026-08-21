from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Reads and returns the contents of a local file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"
