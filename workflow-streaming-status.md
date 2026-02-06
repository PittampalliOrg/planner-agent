# Workflow Streaming Integration - Status Summary

**Date**: 2026-02-02
**Project**: planner-agent/100-improve-planning-agent

## Goal

Integrate the planner-dapr-agent's multi-step workflow into the ai-chatbot UI with real-time SSE streaming updates. The workflow phases are:

1. **Clone** - Clone GitHub repository with token auth
2. **Planning** - AI generates a plan with tasks using OpenAI
3. **Approval** - Human reviews and approves the plan (wait_for_external_event)
4. **Execution** - AI implements the tasks
5. **Testing** - Verify implementation

## Issues and Progress

### 1. SSE Streaming Through Dapr - SOLVED

**Problem**: Dapr service invocation buffers HTTP responses before forwarding, which breaks Server-Sent Events (SSE) streaming. The UI couldn't receive real-time updates.

**Solution**: Added direct Kubernetes service URL bypass for SSE streams in:
- `ai/main/app/(agent)/api/workflows/[instanceId]/stream/route.ts`

```typescript
// Direct Kubernetes service URL for SSE streaming (bypasses Dapr to avoid buffering)
const PLANNER_DAPR_AGENT_SERVICE_URL = process.env.PLANNER_DAPR_AGENT_SERVICE_URL ||
  "http://planner-dapr-agent.planner-agent.svc.cluster.local:8000";
```

The stream route now tries direct K8s service first, then falls back to Dapr, then polling.

**Verified**: Direct curl to `localhost:8000/workflows/{id}/stream` returns SSE data immediately.

---

### 2. DevSpace Hot Reload - PARTIALLY SOLVED → NOW FIXED

**Problem**: DevSpace used `python app.py` which required manual restarts.

**Initial Fix**: Changed to `uvicorn app:app --reload --reload-dir /app` in `devspace.yaml`.

**New Problem Discovered**: The `--reload-dir /app` watched the entire `/app` directory including `/app/workspace` where repositories get cloned. When the workflow cloned a repo, it triggered file changes that caused uvicorn to restart, killing the running workflow.

From the logs:
```
INFO: Clone completed: /app/workspace/planner-agent with 20 files
WARNING: WatchFiles detected changes in 'workspace/planner-agent/...' Reloading...
INFO: Shutting down
```

**Final Fix Applied**: Added `--reload-exclude` to exclude workspace directory:

```yaml
# devspace.yaml line 77
exec uvicorn app:app --host 0.0.0.0 --port ${UVICORN_PORT} --reload --reload-dir /app --reload-exclude '/app/workspace/*'
```

---

### 3. HTTP 504 Timeout on Workflow Creation - ROOT CAUSE FOUND

**Problem**: Creating workflows via Dapr service invocation returned HTTP 504 timeout even though the planner-dapr-agent received the request and started the workflow successfully.

**Root Cause**: The uvicorn hot reload was triggered by cloned repo files, causing the server to restart mid-request. The Dapr sidecar then timed out waiting for a response that never came.

**Status**: Should be fixed by excluding `/app/workspace/*` from hot reload. Deployment restarted.

---

### 4. Dapr Streaming Subscriptions

**Investigated**: Reviewed Diagrid blog about Dapr pub/sub subscription types.

**Finding**: Streaming subscriptions (pull-based, added in Dapr 1.15) are primarily available for the Go SDK and are in alpha. Python SDK support is limited/not yet available.

**Decision**: Continue using SSE direct streaming from the app endpoint rather than Dapr pub/sub streaming.

---

## Files Modified

| File | Changes |
|------|---------|
| `planner-dapr-agent/devspace.yaml` | Added `--reload-exclude '/app/workspace/*'` to prevent hot reload on cloned repos |
| `ai/main/app/(agent)/api/workflows/[instanceId]/stream/route.ts` | Added direct K8s service URL for SSE bypass |

## Next Steps

1. **Test the fix**: Submit a new workflow request and verify:
   - Workflow creation returns 200 (not 504)
   - SSE stream connects successfully
   - Clone phase doesn't trigger uvicorn restart
   - Planning phase completes and reaches approval gate

2. **Verify UI integration**: Check that the workflow view displays:
   - Real-time progress updates
   - Activity logs
   - Approval buttons when waiting for approval

3. **Session-workflow linking**: Ensure sessions properly link to workflow IDs in the database

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ai-chatbot (Next.js)                          │
│                                                                      │
│  /api/agent/sessions (POST)         /api/workflows/{id}/stream (GET)│
│         │                                    │                       │
│         │ Dapr invoke                        │ Direct K8s           │
│         ▼                                    ▼                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            planner-dapr-agent (Python/FastAPI)                │  │
│  │                                                               │  │
│  │  POST /workflow/dapr → schedule_new_workflow() → return       │  │
│  │                            │                                  │  │
│  │  GET /workflows/{id}/stream → SSE generator with Dapr sub     │  │
│  │                                                               │  │
│  │  Dapr Workflow (multi_step_workflow):                        │  │
│  │    clone → planning → wait_for_approval → execution → test   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Dapr Sidecar                              │   │
│  │  - Workflow state (Redis)                                    │   │
│  │  - Pub/Sub (workflow-events topic)                           │   │
│  │  - Service invocation                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```
