from langchain_core.tools import BaseTool

from clyde.tools.execute_bash import execute_bash
from clyde.tools.get_file_structure import get_file_structure
from clyde.tools.list_files import list_files
from clyde.tools.mcp_manage import add_mcp, delete_mcp, get_mcp
from clyde.tools.read_file import read_file
from clyde.tools.search_files import search_files
from clyde.tools.search_mcp import search_mcp
from clyde.tools.summarize import summarize

# Registry of all tools available to the agent.
# The MCP management tools (get_mcp/add_mcp/delete_mcp/search_mcp) are built-in
# and always bound, so the agent can manage MCP servers at runtime; they survive
# rebinds. Flow: search_mcp (discover) -> add_mcp (register) -> get_mcp (list).
tools: list[BaseTool] = [
    execute_bash,
    read_file,
    search_files,
    list_files,
    get_file_structure,
    summarize,
    get_mcp,
    add_mcp,
    delete_mcp,
    search_mcp,
]
tools_by_name = {tool.name: tool for tool in tools}
