import os

from langchain_core.tools import tool

# Filenames that must never be read back into context — secrets can live here.
_SECRET_KEY_FILES = {"id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".cer")


@tool
def read_file(file_path: str) -> str:
    """Reads a local file. Refuses env files and private keys/certs."""
    name = os.path.basename(file_path).lower()
    if (
        name.startswith(".env")
        or name in _SECRET_KEY_FILES
        or name.endswith(_SECRET_SUFFIXES)
    ):
        return f"Refusing to read '{file_path}': it looks like a secrets/key file."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"
