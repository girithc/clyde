import subprocess

from langchain_core.tools import tool

# Destructive commands the agent must not run without human involvement.
_DANGEROUS_PREFIXES = (
    "rm ", "sudo ", "mkfs", "dd ", "shutdown", "reboot", "halt",
    "git reset --hard", "git clean", "chmod -r", "chown -r",
)


@tool
def execute_bash(command: str, cwd: str | None = None, timeout: int = 120) -> str:
    """Executes a bash command and returns stdout/stderr plus the exit code.

    `cwd` sets the working directory; `timeout` (seconds) stops a hung command.
    Destructive commands (rm, sudo, git reset --hard, ...) are refused.
    """
    stripped = command.strip()
    if stripped.lower().startswith(_DANGEROUS_PREFIXES):
        return f"Refusing to run destructive command: {stripped}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as e:
        return f"Execution error: {str(e)}"

    output = result.stdout
    if result.stderr:
        output += f"\n[STDERR]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[EXIT CODE {result.returncode}]"
    return output if output.strip() else "Command executed with no output."
