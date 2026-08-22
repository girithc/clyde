"""Compile the coding agent graph.

Top-level graph = supervisor that plans the request and fans it out to parallel
worker subgraphs, then synthesizes one answer.
"""

from agents.coding_agent.supervisor.graph import build_supervisor
from agents.coding_agent.worker.graph import build_worker

graph = build_supervisor(build_worker())
