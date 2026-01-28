# Planner Agent

A Dapr Workflow orchestrator that uses the Claude Agent SDK to plan and execute software engineering tasks with human-in-the-loop approval.

## Architecture

Three containers deployed as separate K8s Deployments, communicating via Dapr service invocation:

```
                         Dapr Service Invocation
┌─────────────────────────────────────────────────────────────┐
│            planner-orchestrator (Python/FastAPI)             │
│            Dapr app-id: planner-orchestrator                │
│                                                             │
│  unified_planner_workflow:                                  │
│    1. run_planning ──HTTP──► planner-agent-plan /plan       │
│    2. persist_tasks ──────► Dapr statestore (Redis)         │
│    3. wait_for_external_event (approval gate, 24h timeout)  │
│    4. run_execution ──HTTP──► planner-agent-exec /execute   │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐          ┌─────────────────────┐
│ planner-agent-plan  │          │ planner-agent-exec  │
│ (Node.js/Express)   │          │ (Node.js/Express)   │
│ Claude SDK plan mode│          │ Claude SDK bypass    │
│ Dapr app-id:        │          │ Dapr app-id:        │
│  planner-agent-plan │          │  planner-agent-exec  │
└─────────────────────┘          └─────────────────────┘
         │                                    │
         └──────── shared PVC (workspace) ────┘
```

Both agent containers use the same Docker image (`planner-sdk-agent`) but have different Dapr app-ids. The orchestrator invokes them on different HTTP endpoints (`/plan` vs `/execute`).

## Workflow Phases

| Phase | Progress | Activity | What happens |
|-------|----------|----------|-------------|
| Planning | 10% | `run_planning` | Orchestrator POSTs to `planner-agent-plan/plan` via Dapr. Claude SDK runs in `plan` mode, creates tasks using native `TaskCreate`. Server reads tasks from `~/.claude/tasks/`, returns them as JSON, then clears the directory. |
| Persist | 30% | `persist_tasks` | Tasks are saved to Dapr statestore under key `tasks:{workflow_id}`. |
| Approval | 50% | `wait_for_external_event` | Workflow pauses until a human approves or rejects via `POST /api/workflows/{id}/approve`. Times out after 24 hours. |
| Execution | 60% | `run_execution` | Orchestrator POSTs to `planner-agent-exec/execute` via Dapr, sending tasks in the request body. Server restores tasks to `~/.claude/tasks/workflow-tasks/` as numbered JSON files, then runs Claude SDK in `bypassPermissions` mode with a prompt listing all tasks and their dependencies. |

## Task Dependency Management

Tasks use Claude Code's native task system (`CLAUDE_CODE_ENABLE_TASKS=true`). Each task is a JSON file stored at `~/.claude/tasks/{list_id}/{n}.json`.

### Task schema

```json
{
  "id": "1",
  "subject": "Create module structure",
  "description": "Detailed implementation instructions...",
  "activeForm": "Creating module structure",
  "status": "pending",
  "blocks": ["2", "4"],
  "blockedBy": []
}
```

- `blocks` -- task IDs that cannot start until this task completes
- `blockedBy` -- task IDs that must complete before this task can start
- `status` -- `pending`, `in_progress`, or `completed`

### Task lifecycle across containers

1. **Planning agent creates tasks**: Claude SDK calls `TaskCreate` with subjects, descriptions, and dependency edges (`blocks`/`blockedBy`). Tasks are written as `~/.claude/tasks/{uuid}/1.json`, `2.json`, etc.
2. **Server reads and clears**: `server.ts` reads the single directory under `~/.claude/tasks/`, parses all JSON files sorted numerically, returns them in the HTTP response, then deletes the directory.
3. **Orchestrator persists**: The `persist_tasks` activity saves the task array to Dapr statestore (`statestore`) under `tasks:{workflow_id}`.
4. **Orchestrator serves tasks**: `GET /api/workflows/{id}/tasks` reads from statestore so humans can review the plan.
5. **Execution agent restores tasks**: `server.ts` receives tasks in the request body, writes them to `~/.claude/tasks/workflow-tasks/1.json`, `2.json`, etc., and sets `CLAUDE_CODE_TASK_LIST_ID=workflow-tasks` so the SDK can reference them.
6. **Execution agent works through tasks**: Claude SDK reads the task list, respects dependency ordering, marks tasks `in_progress`/`completed` via `TaskUpdate` as it implements each one.

### Dependency ordering

The planning agent creates a DAG of tasks. For example:

```
Task 1 (no deps) ──► Task 2 (blocked by 1) ──► Task 3 (blocked by 2)
         └──────────► Task 4 (blocked by 1) ──┘
                                               └──► Task 5 (blocked by 3, 4)
```

The execution agent receives the full task list with `blockedBy` fields in its prompt and respects the ordering when implementing.

## Project Structure

```
100-improve-planning-agent/
├── CLAUDE.md                           # This file
├── planner-orchestrator/               # Python/FastAPI Dapr workflow coordinator
│   ├── app.py                          # FastAPI app, lifecycle, HTTP endpoints
│   ├── workflows/
│   │   └── planner_workflow.py         # Dapr workflow: plan → persist → approve → execute
│   ├── activities/
│   │   ├── planning.py                 # HTTP call to planner-agent-plan via Dapr
│   │   ├── persist_tasks.py            # Save tasks to Dapr statestore
│   │   └── execution.py               # HTTP call to planner-agent-exec via Dapr
│   ├── Dockerfile                      # Python 3.12 slim
│   ├── requirements.txt                # fastapi, uvicorn, dapr, dapr-ext-workflow, requests
│   ├── dapr.yaml                       # Local dev Dapr config
│   └── resources/
│       └── statestore.yaml             # Redis statestore component (local dev)
├── planner-sdk-agent/                  # Node.js/Express Claude Agent SDK server
│   ├── src/
│   │   ├── server.ts                   # Express HTTP server: /plan, /execute, /health
│   │   └── index.ts                    # CLI entry point (local dev)
│   ├── Dockerfile                      # Node 20 slim
│   ├── package.json                    # @anthropic-ai/claude-agent-sdk, express
│   └── tsconfig.json
└── .gitignore
```

## API Endpoints

All endpoints are on the orchestrator (`planner-orchestrator:8080`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/workflows` | Start a new workflow. Body: `{"feature_request": "...", "cwd": "/path"}` |
| `GET` | `/api/workflows/{id}/status` | Get phase, progress, and message |
| `GET` | `/api/workflows/{id}/tasks` | Get tasks from statestore |
| `POST` | `/api/workflows/{id}/approve` | Approve or reject. Body: `{"approved": true}` |
| `GET` | `/health` | Health check |

## Docker Images

| Image | Dockerfile | Deployments |
|-------|-----------|-------------|
| `planner-orchestrator:latest` | `planner-orchestrator/Dockerfile` | `planner-orchestrator` |
| `planner-sdk-agent:latest` | `planner-sdk-agent/Dockerfile` | `planner-agent-plan`, `planner-agent-exec` |

Build and push to the local Gitea registry:

```bash
# From 100-improve-planning-agent/
docker build -t gitea.cnoe.localtest.me:8443/giteaadmin/planner-orchestrator:latest planner-orchestrator/
docker build -t gitea.cnoe.localtest.me:8443/giteaadmin/planner-sdk-agent:latest planner-sdk-agent/
docker push gitea.cnoe.localtest.me:8443/giteaadmin/planner-orchestrator:latest
docker push gitea.cnoe.localtest.me:8443/giteaadmin/planner-sdk-agent:latest
kubectl rollout restart deployment/planner-orchestrator deployment/planner-agent-plan deployment/planner-agent-exec -n planner-agent
```

## Dapr Service Invocation

Activities call agent services via Dapr HTTP service invocation with extended timeouts:

```
http://localhost:3500/v1.0/invoke/{app-id}/method/{endpoint}
```

- Planning: `dapr-app-timeout: 600` (10 minutes)
- Execution: `dapr-app-timeout: 1800` (30 minutes)

The `dapr-app-timeout` header overrides Dapr's default 60-second timeout, which is too short for Claude Agent SDK operations.

## E2E Test

```bash
# Port-forward the orchestrator
kubectl port-forward svc/planner-agent 8080:8080 -n planner-agent &

# Start a workflow
curl -s -X POST http://localhost:8080/api/workflows \
  -H "Content-Type: application/json" \
  -d '{"feature_request":"Create a hello world script","cwd":"/app/workspace"}'

# Check status (repeat until awaiting_approval)
curl -s http://localhost:8080/api/workflows/{id}/status

# Review tasks
curl -s http://localhost:8080/api/workflows/{id}/tasks

# Approve
curl -s -X POST http://localhost:8080/api/workflows/{id}/approve \
  -H "Content-Type: application/json" -d '{"approved": true}'

# Monitor until completed
curl -s http://localhost:8080/api/workflows/{id}/status
```
