import subprocess

from langchain_core.tools import tool


@tool
def execute_bash(command: str) -> str:
    """Executes a bash command in the local workspace terminal and returns output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        return output if output.strip() else "Command executed with no output."
    except Exception as e:
        return f"Execution error: {str(e)}"
