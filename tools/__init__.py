from langchain_core.tools import BaseTool

from tools.execute_bash import execute_bash
from tools.get_file_structure import get_file_structure
from tools.list_files import list_files
from tools.mcp_manage import add_mcp, delete_mcp, get_mcp
from tools.read_file import read_file
from tools.search_files import search_files
from tools.search_mcp import search_mcp
from tools.summarize import summarize

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
