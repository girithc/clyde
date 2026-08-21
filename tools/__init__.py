from langchain_core.tools import BaseTool

from tools.execute_bash import execute_bash
from tools.read_file import read_file

# Registry of all tools available to the agent.
tools: list[BaseTool] = [execute_bash, read_file]
tools_by_name = {tool.name: tool for tool in tools}
