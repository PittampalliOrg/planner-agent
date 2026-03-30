# Planner Agent

A Claude Agent SDK application that replicates Claude Code's **plan mode** functionality. This agent helps you plan and implement new features in your codebase through a structured workflow.

## Features

- **Codebase Exploration**: Automatically explores your codebase to understand structure and patterns
- **Implementation Planning**: Creates detailed, step-by-step implementation plans
- **Interactive Approval**: Presents plans for your review and approval before implementation
- **Task Management**: Converts approved plans into tracked tasks with dependencies
- **Sequential Implementation**: Works through tasks in order, respecting dependencies
- **Persistence**: Plans and tasks are saved to disk for review and continuation

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key**:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

3. **Ensure Claude Code CLI is installed** (the SDK uses it as the runtime):
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

## Usage

### Interactive Mode

Run without arguments to enter interactive mode:

```bash
python main.py
```

You'll be prompted to describe your feature request.

### Direct Prompt Mode

Provide your feature request directly:

```bash
python main.py "Add user authentication with JWT tokens"
```

### Specify Working Directory

Work on a specific git repository:

```bash
python main.py --cwd /path/to/your/repo "Add new API endpoint"
```

### Custom Plans Directory

Store plans in a custom location:

```bash
python main.py --plans-dir ./my-plans "Implement caching"
```

## CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `--version` | | Print the version and exit |
| `--verbose` | `-v` | Enable debug-level logging |
| `--cwd PATH` | | Working directory (git repository) to work in |
| `--plans-dir PATH` | | Directory to store plans and tasks |

### Print version

```bash
python main.py --version
```

### Run with verbose (debug) logging

```bash
python main.py --verbose "Add feature X"
```

### Combine flags

```bash
python main.py -v --cwd /path/to/repo "Add feature X"
```

## Workflow

The planner agent follows a structured workflow:

### Phase 1: Exploration
When you provide a feature request, the agent first explores your codebase to understand:
- Existing code structure and patterns
- Relevant files that may need modification
- Dependencies and architectural considerations

### Phase 2: Planning
The agent creates an implementation plan including:
- Summary of what will be implemented
- Context gathered during exploration
- Ordered implementation steps
- Critical files to be modified
- Architectural considerations

### Phase 3: Review
The plan is presented for your approval. You can:
- Request modifications
- Ask clarifying questions
- Approve to proceed

Type `approve`, `yes`, or `lgtm` to approve the plan.

### Phase 4: Task Creation
Once approved, the plan is converted to tasks with proper dependencies. Each step becomes a task that depends on the previous one.

### Phase 5: Implementation
The agent works through tasks sequentially:
1. Marks a task as in-progress
2. Implements the changes
3. Marks the task as completed
4. Moves to the next available task

## Docker

### Build the Image

```bash
docker build -t planner-agent:latest .
```

### Run Locally with Docker

```bash
# Interactive mode
docker run -it --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/plans:/app/plans \
  planner-agent:latest

# With a specific prompt
docker run -it --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v /path/to/your/repo:/app/workspace \
  -v $(pwd)/plans:/app/plans \
  planner-agent:latest \
  --cwd /app/workspace \
  "Add user authentication"
```

## Kubernetes Deployment

The `k8s/` directory contains Kubernetes manifests for deploying the planner agent.

### Prerequisites

1. A Kubernetes cluster (minikube, kind, EKS, GKE, etc.)
2. `kubectl` configured to access your cluster
3. Your container image pushed to a registry

### Quick Start

1. **Build and push the image**:
   ```bash
   docker build -t your-registry/planner-agent:latest .
   docker push your-registry/planner-agent:latest
   ```

2. **Update the image reference** in `k8s/kustomization.yaml`:
   ```yaml
   images:
     - name: planner-agent
       newName: your-registry/planner-agent
       newTag: latest
   ```

3. **Set your API key** in `k8s/secret.yaml`:
   ```bash
   # Encode your API key
   echo -n 'sk-ant-xxxxx' | base64
   # Update the ANTHROPIC_API_KEY value in k8s/secret.yaml
   ```

4. **Deploy**:
   ```bash
   kubectl apply -k k8s/
   ```

### Deployment Options

**Option A: Long-lived Deployment (default)**

Runs the agent as a persistent pod for interactive sessions:

```bash
# Deploy
kubectl apply -k k8s/

# Run interactive session
kubectl exec -it deployment/planner-agent -n planner-agent -- \
  python main.py --cwd /app/workspace "Add feature X"

# Check logs
kubectl logs deployment/planner-agent -n planner-agent
```

**Option B: One-off Job**

Edit `k8s/kustomization.yaml` to use `job.yaml` instead of `deployment.yaml`:

```yaml
resources:
  - namespace.yaml
  - secret.yaml
  - configmap.yaml
  - pvc.yaml
  - job.yaml  # Use job instead of deployment
```

Then edit `k8s/job.yaml` to set your feature request in the `args` section.

```bash
# Deploy the job
kubectl apply -k k8s/

# Watch progress
kubectl logs -f job/planner-agent-job -n planner-agent

# Clean up
kubectl delete job planner-agent-job -n planner-agent
```

### Kubernetes Files

| File | Description |
|------|-------------|
| `namespace.yaml` | Creates `planner-agent` namespace |
| `secret.yaml` | Stores Anthropic API key |
| `configmap.yaml` | Configuration settings |
| `pvc.yaml` | Persistent volumes for workspace and plans |
| `deployment.yaml` | Long-lived deployment for interactive use |
| `job.yaml` | One-off job for single planning tasks |
| `kustomization.yaml` | Kustomize configuration for easy deployment |

## Project Structure

```
planner-agent/
├── main.py              # Entry point
├── planner_agent.py     # Main agent implementation
├── plan_manager.py      # Plan creation and persistence
├── task_manager.py      # Task management with dependencies
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container build configuration
├── .dockerignore        # Docker build exclusions
├── .env.example         # Example environment file
├── .gitignore           # Git ignore patterns
├── README.md            # This file
└── k8s/                 # Kubernetes manifests
    ├── namespace.yaml
    ├── secret.yaml
    ├── configmap.yaml
    ├── pvc.yaml
    ├── deployment.yaml
    ├── job.yaml
    └── kustomization.yaml
```

## Output Files

The agent creates a `plans/` directory containing:

- `plan_*.json` - Plan data in JSON format
- `plan_*.md` - Human-readable plan in Markdown
- `tasks.json` - Task list with statuses and dependencies

## Task Management

Tasks support:
- **Status tracking**: `pending` → `in_progress` → `completed`
- **Dependencies**: `blocked_by` and `blocks` relationships
- **Metadata**: Arbitrary data attached to tasks
- **Persistence**: Saved to disk automatically

### Task Statuses
- `[ ]` pending - Not yet started
- `[~]` in_progress - Currently being worked on
- `[x]` completed - Done

## Customization

### Custom System Prompt

Modify `_get_planning_system_prompt()` in `planner_agent.py` to customize the agent's behavior.

### Additional Tools

Add more MCP tools by:
1. Defining a tool function with the `@tool` decorator
2. Adding it to the `create_sdk_mcp_server()` call
3. Including it in `allowed_tools`

### Hooks

Add hooks for custom behavior at various points:

```python
from claude_agent_sdk import HookMatcher

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [HookMatcher(matcher="Edit|Write", hooks=[my_hook])],
    }
)
```

## API Reference

### PlannerAgent

```python
from planner_agent import PlannerAgent

agent = PlannerAgent(
    cwd="/path/to/repo",      # Working directory
    plans_dir="./plans",      # Plans storage directory
)

# Run with a specific prompt
await agent.run_planning_session("Add user authentication")

# Or run interactively
await agent.run_interactive()
```

### TaskManager

```python
from task_manager import TaskManager, TaskStatus

tm = TaskManager("./plans/tasks.json")

# Create a task
task = tm.create_task(
    subject="Implement login endpoint",
    description="Create POST /api/login endpoint",
    active_form="Implementing login endpoint",
)

# Add dependency
tm.update_task(task.id, add_blocked_by=["1"])

# Update status
tm.update_task(task.id, status=TaskStatus.IN_PROGRESS)

# Complete task
tm.complete_task(task.id)

# Save/load
tm.save()
tm.load()
```

### PlanManager

```python
from plan_manager import PlanManager

pm = PlanManager("./plans")

# Create a plan
plan = pm.create_plan(
    title="User Authentication",
    summary="Add JWT-based authentication",
    steps=[
        {"title": "Create auth module", "description": "..."},
        {"title": "Add login endpoint", "description": "..."},
    ],
    critical_files=["src/auth.py", "src/routes.py"],
)

# Approve the plan
pm.approve_plan()

# Convert to tasks
tasks = pm.convert_plan_to_tasks(task_manager)

# Save
pm.save_plan()
```

## Troubleshooting

### "Claude Code not found"
Install Claude Code CLI:
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### "API key not set"
Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Permission errors
The agent uses `permission_mode="acceptEdits"` by default. For more control, modify the permission mode in `planner_agent.py`.

## Contributing

Contributions are welcome! Please follow the existing code style and add tests for new features.

## License

MIT License - see LICENSE file for details.
