"""Prompt strings for the coding agent's planner, workers, and synthesis."""

# Exported as the agent's `system_prompt` (see __init__.py) and seeded by main.py.
AGENT_SYSTEM_PROMPT = (
    "You are an autonomous coding assistant. You read local files and execute "
    "bash commands to inspect, write, test, or debug code."
)

PLANNER_PROMPT = (
    "Decompose the user's request into a small number of independent, "
    "parallelizable tasks. Each task must be self-contained: it must not depend "
    "on another task's output or edits. Each task needs a short kebab-case id "
    "and a one-line description."
)

TASK_PLANNER_PROMPT = (
    "Given one task, produce a concise step-by-step plan for how to complete it."
)

CONTEXT_SELECT_PROMPT = (
    "Given a task and a project file tree, list the file paths most relevant to "
    "completing the task. Only include paths that appear in the tree."
)

VERIFY_PROMPT = (
    "Review whether the task was fully completed by the prior work. Set done=true "
    "only when it is completely finished; otherwise set done=false and put a "
    "specific, actionable next step in feedback."
)

SYNTHESIZE_PROMPT = (
    "Combine the completed task outcomes into one clear final answer for the "
    "user. Reference each task by id."
)
