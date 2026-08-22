from langchain_core.tools import BaseTool

from tools.execute_bash import execute_bash
from tools.get_file_structure import get_file_structure
from tools.list_files import list_files
from tools.read_file import read_file
from tools.search_files import search_files

# Registry of all tools available to the agent.
tools: list[BaseTool] = [
    execute_bash,
    read_file,
    search_files,
    list_files,
    get_file_structure,
]
tools_by_name = {tool.name: tool for tool in tools}
