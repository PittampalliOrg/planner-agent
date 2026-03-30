# Contributing to Planner Agent

Thank you for your interest in contributing! Please read the [README](README.md) for background on what this project does and how it works.

---

## Getting Started

### Prerequisites

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)
- [Claude Code CLI](https://claude.ai/install.sh) (the SDK uses it as its runtime)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/planner-agent.git
cd planner-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY to your key
```

---

## Project Structure

The repository maps closely to the table in the README. Key modules:

| File | Purpose |
|------|---------|
| `main.py` | Entry point; argument parsing and top-level error handling |
| `planner_agent.py` | Core agent: MCP tools, `PlannerAgent` class, planning/execution prompts |
| `plan_manager.py` | Plan creation, persistence, and approval logic |
| `task_manager.py` | Task lifecycle (create, update, dependencies, status tracking) |
| `workflow_service.py` | FastAPI app exposing the agent as a Dapr workflow service |
| `streaming.py` | Publishes real-time execution events to Dapr pub/sub |
| `k8s/` | Kubernetes manifests (namespace, secret, PVC, deployment, job) |

---

## Development Workflow

### Run locally

```bash
# Interactive mode (prompts you for a feature request)
python main.py

# Direct mode
python main.py "Add streaming support for Dapr workflows"

# Target a specific repository
python main.py --cwd /path/to/repo "Add new API endpoint"
```

### Kubernetes-native development with DevSpace

DevSpace syncs your local source files into the running container and restarts `uvicorn` on changes:

```bash
devspace dev        # Start dev mode with live sync
devspace enter      # Open a shell in the container
devspace purge      # Clean up DevSpace resources
```

### Build and run the Docker image locally

```bash
# Build
docker build -t planner-agent:latest .

# Run the workflow service
docker run -it --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/plans:/app/plans \
  -p 8080:8080 \
  planner-agent:latest

# Run the CLI directly (override the default entrypoint)
docker run -it --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/plans:/app/plans \
  --entrypoint python \
  planner-agent:latest \
  main.py --cwd /app/workspace "Add user authentication"
```

---

## Branching & Commit Conventions

### Branch naming

```
feature/<short-description>
fix/<short-description>
docs/<short-description>
```

Examples: `feature/streaming-events`, `fix/task-dependency-order`, `docs/contributing-guide`

### Commit messages

- Use the **imperative mood** in the subject line
- Keep the subject line to **72 characters or fewer**
- Reference related issues where relevant

```
Add streaming support for Dapr workflows
Fix task blocked_by not updating on completion
Update k8s deployment resource limits
```

---

## Submitting a Pull Request

1. Fork the repository (or create a branch if you have access).
2. Make focused, atomic commits — one logical change per commit.
3. Ensure the project runs without errors before opening a PR (`python main.py` should start cleanly).
4. Open a PR with a description explaining **what** changed and **why**.
5. Reference any related issues using `Fixes #<issue>` or `Relates to #<issue>`.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) and match the patterns already in the codebase.
- Use type hints throughout; the codebase already uses `from __future__ import annotations` for forward references.
- Model data structures with `pydantic` (consistent with `plan_manager.py` and `task_manager.py`).
- Add concise docstrings to all new public functions and classes.

---

## Adding New MCP Tools

New tools extend what the agent can do during planning or execution.

1. Define the tool function using the `@tool` decorator from `claude_agent_sdk`:

   ```python
   @tool(
       "my_tool",
       "One-sentence description of what this tool does.",
       {"param_name": str},
   )
   async def my_tool(args: dict[str, Any]) -> dict[str, Any]:
       """Short docstring."""
       ...
   ```

2. Register the function in the `create_sdk_mcp_server()` call inside `PlannerAgent.__init__`:

   ```python
   self.mcp_server = create_sdk_mcp_server(
       name="planner",
       version="1.0.0",
       tools=[..., my_tool],
   )
   ```

3. Add the tool name to the `allowed_tools` list in the relevant `ClaudeAgentOptions` block (`run_planning_session`, `run_planning_only`, or `run_execution_only`).

4. Document the tool's purpose clearly in its description string — the model uses this to decide when to call it.

---

## Kubernetes / Dapr Changes

- All infrastructure changes belong under `k8s/`.
- Test infrastructure changes with `devspace dev` against a local cluster before submitting.
- Update `k8s/kustomization.yaml` if you add or remove manifest files.

---

## Reporting Issues

Use [GitHub Issues](../../issues) to report bugs or request features.

Please include:

- Steps to reproduce the problem
- Expected behaviour vs. actual behaviour
- Relevant log output (run with `PYTHONUNBUFFERED=1` to get unbuffered logs)

---

## License

By contributing, you agree that your contributions will be licensed under the **MIT License**, consistent with the rest of this project (see the README for details).
