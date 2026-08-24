from dotenv import load_dotenv

load_dotenv()  # Load FIREWORKS_API_KEY etc. before agents build the LLM

from langchain_core.messages import SystemMessage

from agents import default_graph as graph, default_system_prompt as system_prompt

import plugins.mcp as mcpmod
from plugins.mcp import McpManager
from plugins.skills import load_skills
from ui import ClydeApp


# --- Entry point ---

def main():
    # --- Startup: MCP manager + skills (before the screen takes over) ---
    manager = McpManager.from_config(".mcp.json")
    mcpmod.manager = manager  # management tools reach it via this singleton
    manager.start()
    manager._rebind()  # bind static + any pre-configured MCP tools

    skills = load_skills("./plugins")
    history = [SystemMessage(content=system_prompt)]

    app = ClydeApp(graph=graph, history=history, skills=skills)

    # Route trace lines into the TUI transcript (dimmed) instead of stdout.
    import trace as _trace
    _trace.compact_trace.set_sink(app.post_trace)

    try:
        app.run()
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
