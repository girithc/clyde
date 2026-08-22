"""Compact structured tracing for the REPL.

Replaces LangChain's default verbose/debug JSON dumps with one clean line per
event, printed to stdout so it sits alongside the REPL UI. Covers LLM calls and
tool runs; the graph node flow is already surfaced by main.py's own prints.
"""

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class CompactTraceHandler(BaseCallbackHandler):
    """One-line-per-event trace printer."""

    def __init__(self):
        self._llm_starts: dict[str, float] = {}
        self._tool_starts: dict[str, float] = {}

    # --- LLM ---

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_starts[str(run_id)] = time.time()
        n = len(prompts[0].splitlines()) if prompts else 0
        print(f"🧠 [LLM] start — {n} message-lines in context")

    def on_llm_end(self, response, *, run_id, **kwargs):
        started = self._llm_starts.pop(str(run_id), None)
        secs = f"{time.time() - started:.2f}s" if started else "?s"
        usage = (response.llm_output or {}).get("token_usage", {})
        out_toks = usage.get("completion_tokens", "?")
        finish = None
        try:
            gen = response.generations[0][0]
            finish = gen.generation_info.get("finish_reason") if gen else None
        except (IndexError, AttributeError):
            pass
        print(f"✅ [LLM] end — {secs}, {out_toks} out tokens, finish={finish}")

    # --- Tools ---

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._tool_starts[str(run_id)] = time.time()
        name = (serialized or {}).get("name", "tool")
        print(f"🔧 [Tool] start — {name}({input_str})")

    def on_tool_end(self, output, *, run_id, **kwargs):
        started = self._tool_starts.pop(str(run_id), None)
        secs = f"{(time.time() - started) * 1000:.0f}ms" if started else "?ms"
        preview = str(output).strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        print(f"📥 [Tool] end — {secs}: {preview}")


# Shared singleton — attach on the LLM in llm.py via callbacks=[...].
compact_trace = CompactTraceHandler()
