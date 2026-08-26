"""Prompt strings for the coding agent's planner, workers, and synthesis."""

import platform as _platform

# Exported as the agent's `system_prompt` (see __init__.py) and seeded by main.py.
AGENT_SYSTEM_PROMPT = (
    "You are an autonomous coding assistant. You read local files and execute "
    "bash commands to inspect, write, test, or debug code."
)

# Platform hint appended once at import time. Just the platform name — the model
# picks the correct (BSD/GNU) command syntax itself; we don't enumerate flags.
_OS = _platform.system()
_PLATFORM = {"Darwin": "macOS", "Windows": "Windows"}.get(_OS, _OS)
system_prompt = f"{AGENT_SYSTEM_PROMPT} You are running on {_PLATFORM}."

PLANNER_PROMPT = (
    "Decide whether the request is trivial, needs clarification, or needs "
    "decomposition. A request is trivial if it needs no tools and no parallel "
    "work — a general-knowledge question, an explanation, or a single direct "
    "answer you can give in one shot. For trivial requests, set trivial=true "
    "and emit no tasks. Otherwise decompose into a small number of independent, "
    "parallelizable tasks. Each task must be self-contained: it must not depend "
    "on another task's output or edits. Each task needs a short kebab-case id "
    "and a one-line description. If several tasks need the same upfront data "
    "(file tree, file sizes, shared file contents), factor that common grunt "
    "work into `shared` so it runs once instead of per task. Put only the "
    "common grunt work in `shared`; keep `tasks` as the independent deliverables.\n"
    "\n"
    "CLARIFICATION: set `clarify` (instead of tasks) ONLY when the request is "
    "genuinely ambiguous AND guessing wrong is costly — multiple materially "
    "different valid interpretations, a missing approach-defining constraint "
    "(target file, framework, scope), or an irreversible/destructive fork where "
    "the wrong guess is expensive. Do NOT set `clarify` when the request is "
    "trivial, clear and self-contained, or when a sensible default exists and "
    "being wrong is cheap/reversible — just proceed. Never set `clarify` again "
    "if the user has already chosen (a `[User's chosen option: ...]` line is "
    "present) — proceed with their choice. `clarify` is a LIST of independent "
    "questions, exactly like `tasks` is a list of independent units of work — "
    "emit one question per distinct ambiguity (often just one; only multiple "
    "when there are several genuinely independent things to settle). Each "
    "question needs 2-4 distinct, mutually exclusive options, each with a short "
    "label, a one-line description, and exactly one `recommended=true` placed "
    "first in that question's option list."
)

TASK_PLANNER_PROMPT = (
    "Given one task, produce a concise step-by-step plan for how to complete it."
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
