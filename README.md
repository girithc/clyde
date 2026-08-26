# Clyde

A terminal coding agent that combines the **observability, tracing, and customizability** of the DeepSeek harness with the **simplicity** of Claude Code — and makes it all available out of the box.

Everything Clyde can do is visible on the UI. There are no slash commands to memorize; you just talk to it.

---

## What is Clyde?

Clyde is an autonomous coding assistant that lives in your terminal. It reads your local files, runs bash commands, searches the web for MCP servers, and manages its own tools — all through a single scrolling conversation. Underneath, it runs a LangGraph supervisor/worker agent on top of LangChain, so it works with any model provider and any model.

## What inspired Clyde?

Two things, fused together:

- **The DeepSeek harness** — for its observability, structured tracing, and deep customizability. You should always be able to see what the agent is doing underneath, and you should be able to bend it to your workflow.
- **Claude Code** — for its simplicity. One input bar, one scrolling transcript, an inline "thinking" line that lives in the conversation like any other message. No modal panels, no cognitive overhead.

Clyde takes the transparency and tunability of the first and the ergonomics of the second: every capability is surfaced as a clickable visual shortcut on the input bar.

## Is it easy for new users?

Yes — by design.

- **No slash commands.** Nothing to memorize. You converse with Clyde in plain language and it does the rest, including managing its own plugins and MCP servers.
- **Everything is on the UI.** The bottom status bar shows the current trace mode, agent mode, and model — each clickable to change it. The `+` button attaches images. The model label opens an inline editor to switch providers and model ids. The bottom-right hint rotates between keybindings so you discover them naturally.
- **Visual shortcuts on the input bar.** Instead of typing `/mode` or `/trace`, you click the status chips or press a key. You can also just ask Clyde in conversation: *"turn on full trace"* or *"switch to plan mode"* and it can act on it.

## How do I install Clyde?

```bash
pipx install clyde-ai
```

`pipx` installs the `clyde` command in an isolated venv and puts it on your PATH. No clone, no venv, no `pip install -r requirements.txt`. (No `pipx`? `brew install pipx` on macOS, or `python -m pip install --user pipx`.)

## How do I uninstall Clyde?

```bash
pipx uninstall clyde-ai && uv cache clean clyde-ai
```

`pipx uninstall` removes the venv; `uv cache clean clyde-ai` clears clyde-ai's cached wheels + index entries from uv's shared cache so a later reinstall is truly fresh. (If you hit a stale version right after a release, install with `UV_NO_CACHE=1 pipx install clyde-ai` to bypass the cache for that one run.)

## How do I configure my API key?

Keys live in your **OS keychain** (macOS Keychain via `keyring`) — no `.env` file, no env-var fallback, no plaintext on disk. Set them with the built-in login command:

```bash
clyde login              # interactive: pick provider(s), paste key(s)
clyde login fireworks    # one-shot: set a single provider's key
clyde auth               # show which providers have keys + the active one
clyde logout fireworks   # remove a key (clyde logout --all clears every key)
```

Only the key for your active provider is required. The default provider is Fireworks; if its key is missing on startup, Clyde auto-switches to the first provider you *have* a key for. A provider's connector package is imported lazily — you only need the langchain package for the provider you actually use (e.g. `pipx inject clyde-ai langchain-anthropic` for Anthropic).

## How do I run Clyde?

```bash
clyde
```

A proactive, context-aware greeting streams in first, then you type at the input bar. Type `exit` or press `Ctrl+C` to quit.

## Which model providers does Clyde support?

Clyde is provider-agnostic via LangChain. Built-in providers (defined in `clyde/llm/registry.py`):

| Provider | Connector | Login name |
|----------|-----------|------------|
| Fireworks | `langchain_fireworks` | `fireworks` |
| Anthropic | `langchain_anthropic` | `anthropic` |
| OpenAI | `langchain_openai` | `openai` |
| OpenRouter | `langchain_openai` (OpenAI-compatible) | `openrouter` |

## Can I use any model?

Yes. You type a free-form model id (e.g. `accounts/fireworks/models/muse-glimmer-30b`, `claude-sonnet-5`, `gpt-4o`, `anthropic/claude-3.5-sonnet`). Click the model label in the status bar to open the inline editor, pick a provider from the dropdown, type the model id, and confirm. The choice is persisted to `~/.clyde/config.json` and survives across sessions. A provider's connector package is imported lazily — you only need the langchain package for the provider you actually use.

## Why LangChain and LangGraph?

For **uptime and reliability**. LangChain abstracts away provider differences (auth, API shape, streaming), so Clyde stays provider-agnostic and resilient. LangGraph gives the agent a durable, structured state machine — the supervisor plans, fans out to parallel workers, and synthesizes — with checkpointed state and a well-defined control flow instead of an ad-hoc loop. The compiled graph streams tokens live into the transcript.

## How does the agent work internally?

Clyde is a **supervisor + worker** graph (`agents/coding_agent/`):

1. **Planner** — decides whether the request is *trivial* (one direct answer, no tools) or needs decomposition. For non-trivial work it emits a small set of independent, parallelizable tasks plus optional `shared` grunt work.
2. **Scout** — runs the shared grunt work once (file discovery, shared reads) so each task doesn't repeat it. Hard backstop of 8 steps so it can never spin forever.
3. **Workers** — each task runs in an isolated worker subgraph (`plan → executor loop → verify`). Workers cannot read each other's transcripts; only the final result bridges back, so parallel work stays clean.
4. **Synthesize** — combines all task outcomes into one clear final answer.

Trivial requests skip all of this and answer directly in a single LLM call.

## What are the agent modes?

Clyde exposes an agent mode in the status bar — **auto / plan / ask** — cycle it with `Shift+Tab` or click the `mode:` chip. This is UI state that steers how the agent treats your request.

## What is trace mode?

Trace mode is the observability layer — your window into what Clyde is doing underneath with every request. Cycle it with `Ctrl+T` or click the `trace:` chip. Three levels:

- **none** — clean transcript, only user messages, the thinking line, and the answer.
- **minimal** — compact dimmed labels inline: `thinking`, `calling read_file`, `done 3.12s`.
- **full** — the complete structured trace line per event.

You opt into as much or as little visibility as you want, per request.

## What does the trace show?

A compact, one-line-per-event trace (instead of LangChain's default verbose JSON dumps):

- `[LLM] start — N message-lines in context`
- `[LLM] end — 3.12s · ↑ 294 in · ↓ 180 out · finish=stop`
- `[Tool] start — read_file({'file_path': ...})`
- `[Tool] end — 12ms: <output preview>`

Every LLM call reports wall-time, input/output tokens, and finish reason; every tool call reports duration and a preview. In the TUI these route into the transcript as dimmed text (never to stdout, which would corrupt the full-screen app).

## How does MCP management work?

MCP (Model Context Protocol) servers are managed **built-in and conversationally** — you never hand-edit config or restart. Clyde exposes four management tools that the agent calls on its own:

- `search_mcp(query)` — searches the internet (the curated `awesome-mcp-servers` directory) for servers matching a use case like *"notion"*, *"postgres"*, or *"github"*.
- `add_mcp(name, command, args, env, timeout)` — registers a stdio MCP server, writes it to `.mcp.json`, hot-starts a long-lived session, and binds its tools into the agent **immediately** — usable the same turn.
- `delete_mcp(name)` — closes the session, drops the tools, and removes the server from `.mcp.json`.
- `get_mcp()` — lists configured servers with live connection status and exposed tools.

So you just say: *"add a GitHub MCP server"*, and Clyde searches, registers, and starts it. Done.

## Why is built-in MCP removal a feature?

Because removing a plugin once it's added shouldn't be painful. In Claude Code, removing something like the caveman plugin once it was added was a tough, manual process. Clyde makes add **and** remove first-class and conversational — just tell Clyde *"remove the caveman plugin"* or *"delete the linkedin MCP server"* and it's gone, tools unbound and config cleaned up, with no restart. That customizability — easy in, easy out — is a core part of the philosophy.

## How do skills (plugins) work?

Skills are pure-prompt plugins: a markdown file (`SKILL.md` or `*.skill`) with YAML frontmatter:

```markdown
---
name: caveman
description: >
  Ultra-compressed mode. Use when user says "caveman mode", "be brief",
  or invokes /caveman.
---
Respond terse like smart caveman...
```

Clyde scans two locations — the skills bundled with the package (`clyde/skills`) and your user dir (`~/.clyde/skills`) — parses the frontmatter, and on each turn injects the body of any skill whose triggers match your input as a system message. Triggers are extracted heuristically from the description (quoted phrases plus the skill name), so authors phrase them as prose, not a structured list. Skills are pure text with no runtime — and because add/remove is conversational and built in, you're never stuck with one.

## Are there slash commands?

No. Clyde deliberately has no slash commands. The goal is to make **everything talkable** — you converse with Clyde to change behavior, manage tools, and put visual shortcuts on the input bar. Where a keybinding or clickable chip is more convenient (trace, mode, model, attach), it's surfaced on the UI; otherwise you just ask.

## How does image input work?

If your current model is vision-capable, a `+` attach button appears on the input bar. Click it to open the native macOS file picker (AppleScript via `osascript` — no Python Tk dependency) and attach an image. The image is base64-encoded and sent with your message. Vision capability is auto-detected from the model id via known patterns; for unrecognized models, Clyde asks the LLM in the background to confirm, and the button appears if the probe says yes.

## What is the proactive greeting?

On startup, Clyde streams a short, context-aware greeting generated from live session context: the time of day, your git user name, the project name (from the git remote or current directory), and your most recent commit. It's best-effort — if git isn't available, the missing pieces fall back to harmless defaults so the greet never fails. It types out live, then you start typing.

## What does the live thinking indicator show?

While a turn runs, an inline "thinking" line appears in the transcript (Claude Code style):

```
✻ Pondering… (33s · ↓ 2.0k tokens)
```

- A rotating spinner glyph.
- A rotating first word drawn from 50 gerunds (`Nesting`, `Pondering`, `Musing`…) cycling every ~2.5s.
- Elapsed wall-time.
- Output tokens accumulated from the LLM trace this turn.

When the turn finishes, the line settles in place to `Thought for 33s · ↓ 2.0k tokens` and stays until the next turn.

## What keybindings does Clyde have?

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+T` | Cycle trace mode (none / minimal / full) |
| `Shift+Tab` | Cycle agent mode (auto / plan / ask) |
| `Escape` | Cancel inline model edit |

The bottom-right hint rotates between these every few seconds so you discover them naturally. The status-bar chips (`trace:`, `mode:`, the model label) are also clickable.

## What safety guards are built in?

- **Destructive commands refused.** `execute_bash` blocks `rm`, `sudo`, `mkfs`, `dd`, `shutdown`, `reboot`, `git reset --hard`, `git clean`, and similar unless a human is involved.
- **Secrets never read into context.** `read_file` refuses `.env*` files, private keys (`id_rsa`, `id_ed25519`, …), and cert/key suffixes (`.pem`, `.key`, `.p12`, …).
- **Junk dirs skipped.** File search/listing/structure tools prune `venv`, `.git`, `node_modules`, `__pycache__`, `dist`, `build`, etc.
- **Context trimmed.** The executor trims oldest turns once context approaches ~24k tokens (approximated at ~4 chars/token; images charged a flat ~1000 tokens so they aren't trimmed for free).
- **Rebind never crashes a turn.** Adding/removing MCP tools rebinds the executor and tool registry under a lock; failures are traced and swallowed so a turn is never lost.

## Where is configuration stored?

- **OS keychain** — API keys (service `clyde`), set via `clyde login`. No plaintext on disk.
- `.mcp.json` — MCP server config, written/updated by the management tools at runtime (per-project, in CWD).
- `~/.clyde/config.json` — user preferences (chosen provider + model id), persists across sessions.
- `~/.clyde/skills` — your personal skills (markdown), loaded alongside the bundled ones.

## What does the project structure look like?

```
clyde/
├── __main__.py                 # entry point: `clyde` console script — MCP manager, skills, TUI
├── __init__.py                 # __version__ (single source of truth; pyproject reads it)
├── auth.py                     # OS-keychain credential storage + login/logout/auth CLI
├── config.py                   # ~/.clyde/config.json persistence
├── trace.py                    # compact one-line-per-event trace handler
├── llm/
│   ├── registry.py             # provider registry + vision detection/probe
│   └── factory.py              # lazy provider-agnostic LLM factory
├── agents/
│   └── coding_agent/
│       ├── graph.py            # supervisor(worker()) compile
│       ├── state.py            # Supervisor/Worker state + structured schemas
│       ├── prompts.py          # planner/worker/verify/synthesize prompts
│       ├── model.py            # shared executor LLM + context trimming
│       ├── tools.py            # tool node + registry (rebindable)
│       ├── supervisor/         # planner -> scout -> fan-out -> synthesize
│       ├── worker/             # plan -> executor loop -> verify (isolated)
│       └── greeting/           # proactive context-aware greeting graph
├── plugins/
│   ├── mcp.py                  # McpManager: long-lived sessions, hot add/remove
│   └── skills.py               # skill loader + trigger matcher
├── tools/
│   ├── execute_bash.py         # guarded shell execution
│   ├── read_file.py            # secrets-refusing file reader
│   ├── search_files.py         # regex content search
│   ├── list_files.py           # glob file listing
│   ├── get_file_structure.py   # depth-limited tree
│   ├── summarize.py            # opt-in output compressor
│   ├── search_mcp.py           # internet MCP discovery
│   └── mcp_manage.py           # add/delete/get MCP servers
└── ui/
    ├── app.py                  # textual TUI: transcript + input bar + status
    ├── transcript.py           # scrolling conversation
    ├── renderer.py             # rich renderables (markdown, dim, user lines)
    ├── thinking.py             # animated thinking indicator
    ├── trace_minimal.py        # compact trace labels
    ├── trace_full.py           # full trace lines
    └── filepicker.py           # native macOS image picker
```

## Why this architecture?

Clyde's architecture is a set of deliberate trade-offs. Here's what we chose and why.

### Why a supervisor/worker split instead of one loop?

A single agent loop is easy to build but bad at parallelism and context hygiene. We split into a **supervisor** (plan, fan out, synthesize) and **worker** subgraphs (plan, execute, verify) so independent tasks run in parallel and each worker keeps an isolated transcript. Workers cannot read each other's tool output — only the final `results` bridge back to the supervisor. That isolation prevents parallel workers from contaminating each other's reasoning and keeps the user-facing transcript clean: you see one synthesized answer, not five interleaved tool dumps.

### Why a planner with a trivial fast path?

Most requests are simple questions that don't need tools or decomposition. Forcing every request through fan-out would add latency and tokens for no gain. The planner emits a structured `Plan` with a `trivial` flag; trivial requests skip straight to a single direct-answer node. Non-trivial ones decompose into a small number of self-contained, parallelizable tasks. The fast path keeps simple interaction instant; the slow path only engages when there's real parallel work to do.

### Why a scout node for shared grunt work?

When several tasks need the same upfront data (a file tree, shared file contents), running that work per task is wasteful and can produce inconsistent views. The scout runs shared grunt work **once** and seeds every worker with the same `shared_context`. A hard backstop of 8 steps guarantees the scout's gather loop can never spin forever. The result: less redundant tool traffic, consistent inputs across workers, and a bounded worst case.

### Why structured output for plan/verify?

The planner, task planner, and verifier all use `with_structured_output(..., method="function_calling")` rather than free-text parsing. Function-calling gives us validated, typed objects (`Plan`, `Approach`, `VerifyVerdict`) directly — no regex, no "the model didn't follow format" failures. The verify step's `done`/`feedback` loop is what lets a worker self-correct: if a task isn't complete, the worker gets an actionable next step and loops back to the executor instead of declaring victory early.

### Why LangChain as the model middleware?

Provider APIs differ in auth, request shape, streaming semantics, and tool-calling format. Hand-rolling adapters for each is fragile and duplicates work that LangChain already maintains. LangChain normalizes all of that behind one `ChatModel` interface, so Clyde stays provider-agnostic and we get streaming, tool binding, and structured output for free. It's the reliability/uptime layer: when a provider changes their API, the connector updates and Clyde keeps working.

### Why LangGraph for the control flow?

An ad-hoc agent loop has no durable state, no clear edges, and no way to reason about control flow. LangGraph gives us a typed state machine (`SupervisorState`, `WorkerState`) with explicit nodes and edges, conditional routing, parallel fan-out via `Send`, and checkpointed state. The compiled graph streams tokens live (`stream_mode=["messages", "values"]`) straight into the transcript. We get a structure we can inspect, debug, and extend — add a node, wire an edge — instead of spaghetti.

### Why lazy provider imports?

The provider registry maps each provider to a connector module, but `get_llm` only imports that module when the provider is actually selected. You don't need every `langchain_*` package installed at startup — only the one you use. Import never bricks the app either: if the startup provider/model is unusable (a persisted choice whose package or key is now missing), the factory falls back to the built-in default.

### Why long-lived MCP sessions on a background asyncio loop?

MCP tools are async, but LangGraph's graph runs sync. We hold one long-lived session per server on a persistent background `asyncio` loop and wrap each MCP tool in a sync `StructuredTool` that dispatches via `run_coroutine_threadsafe`. Spawning a fresh subprocess per tool call would be slow and stateful-server-hostile; long-lived sessions make tool calls fast and let stateful servers (LinkedIn, browsers) work. Per-tool timeouts default to 200s because browser-backed servers need far more than a naive 60s.

### Why hot add/remove with rebind instead of restart?

Adding or removing an MCP server writes `.mcp.json`, opens/closes the session, and **rebinds** the executor LLM and tool registry under a lock — without rebuilding the compiled graph. `call_model` and `call_tools` read the module-global registry at call time, so a rebind takes effect between LLM invokes within the same turn. New tools are usable immediately; removed tools vanish immediately; no restart. Rebind is wrapped so it can never crash a turn — failures are traced and swallowed. This is what makes conversational MCP management practical.

### Why compact tracing via callbacks instead of debug JSON?

LangChain's built-in debug/verbose mode dumps raw JSON per event — unreadable and useless in a TUI. We attach a `CompactTraceHandler` (a `BaseCallbackHandler`) to every LLM that emits one clean line per event: LLM start/end with wall-time, token counts, and finish reason; tool start/end with duration and an output preview. The sink is configurable (default `print`); the TUI redirects it into the transcript as dimmed text so tracing never corrupts the full-screen app. The same callback feeds the thinking indicator's live token counter.

### Why a sync graph with async UI bridged by messages?

LangGraph's `graph.stream` is sync and blocking; Textual is async. We run each turn in a worker thread and post **non-blocking** messages (`_StreamChunk`, `_TraceLine`, `_TurnDone`) back to the UI thread, which awaits mounts into the transcript. This avoids blocking the event loop while keeping the UI fully responsive — you can submit a message mid-turn and it's surfaced inline immediately and processed as a seamless continuation right after the current turn, with no separate "new turn" framing.

### Why skills as pure prompt injection?

A skill is just markdown with YAML frontmatter — no runtime, no code execution, no plugin sandbox to maintain. On each turn we inject the body of any skill whose triggers match your input as a system message. Trigger phrases are extracted heuristically from the description (quoted strings plus the skill name), so authors write prose, not a schema. Pure text means adding/removing a skill is trivial and safe — which is the whole point of making plugin management conversational and reversible.

### Why context trimming with a flat image cost?

Context windows are finite and images are expensive. The executor trims oldest turns once the transcript approaches ~24k tokens, approximated at ~4 chars/token so we don't need a tokenizer in the hot path. Multimodal messages carry a list of parts; we sum text-part lengths and charge a flat ~1000 tokens per image so images aren't trimmed for free once pressure builds. This keeps long sessions responsive without silently dropping the context that matters.

### Why click-to-edit model switching with persistence?

Switching models shouldn't require editing a config file and restarting. The model label in the status bar opens an inline editor (provider dropdown + free-form model id), and the choice is written to `~/.clyde/config.json` so it carries into the next session. Vision capability is recomputed live on every switch, and the attach button appears or hides accordingly — the UI reflects the model's actual capabilities without the user needing to know them.

## What's on the roadmap?

Clyde's agent registry is already built for this — `agents/__init__.py` is structured as "one file (or sub-folder) per agent type (coding, testing, resolver, qa, …)," each exporting a compiled `graph` and `system_prompt`. The work below is extending that registry, not reinventing it.

### Multiple agent types

- [ ] **Testing agent** — generates and runs test suites from diffs/specs, reports failures with root-cause hints, and loops until green.
- [ ] **QA agent** — reviews changes for correctness, edge cases, and regressions before merge; adversarial self-review pass.
- [ ] **GTM (go-to-market) agent** — drafts release notes, changelogs, and README/docs updates from commit history and the diff.
- [ ] **Resolver agent** — triages and fixes failing CI / flaky tests / bug reports autonomously, then verifies.

### Stack-specific coding agents

Same supervisor/worker skeleton, stack-tuned system prompts, toolsets, and verification criteria.

- [ ] **iOS agent** — Swift / SwiftUI, Xcode toolchain (`xcodebuild`, simulators), Apple-platform file conventions.
- [ ] **Android agent** — Kotlin / Jetpack Compose, Gradle (`./gradlew`), emulator/ADB, Android project layout.
- [ ] **Frontend agent** — TypeScript / React / CSS, `npm`/`pnpm`/`vite`, component tree reasoning, browser-driven verification.

### Cloud agents

Run agents in the cloud 24×7 with high availability, instead of only locally in a terminal.

- [ ] **Modular cloud connectors** — pluggable adapters to deploy and run an agent on any cloud provider (AWS / GCP / Azure / fly.io / Render …) behind one interface.
- [ ] **Persistent cloud sessions** — long-running agent processes that survive disconnects; reconnect from any client and resume the same transcript/checkpoint.
- [ ] **High availability** — health checks, restart-on-failure, and the ability to run more than one instance so an agent is always reachable.
- [ ] **Local → cloud handoff** — start a task locally, push it to a cloud agent to keep running unattended, and pull the result back.

## What's the quickstart in one block?

```bash
pipx install clyde-ai
clyde login fireworks      # paste your Fireworks API key (stored in the OS keychain)
clyde
```

Then just talk to Clyde. Ask it to add an MCP server, switch models, turn on full trace, or write some code — no slash commands, no manual config.
