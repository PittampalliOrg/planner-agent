"""
Dapr Workflow Service for Planner Agent

Exposes HTTP endpoints for:
- Starting unified planning and execution workflows
- Workflow status queries
- Plan approval handling
- Repository cloning

Uses native Claude Code tools (TaskCreate, TaskList, TaskUpdate)
instead of custom plan/task managers.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Only import Dapr if available (optional for basic HTTP operation)
try:
    from dapr.ext.workflow import (
        WorkflowActivityContext,
        WorkflowRuntime,
    )
    from dapr.clients import DaprClient
    DAPR_AVAILABLE = True
except ImportError:
    DAPR_AVAILABLE = False
    WorkflowActivityContext = None
    WorkflowRuntime = None
    DaprClient = None

# Phoenix Arize observability imports
try:
    from phoenix.otel import register as phoenix_register
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    phoenix_register = None

# OpenInference Anthropic instrumentor for Claude SDK tracing
try:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    ANTHROPIC_INSTRUMENTOR_AVAILABLE = True
except ImportError:
    ANTHROPIC_INSTRUMENTOR_AVAILABLE = False
    AnthropicInstrumentor = None

from durable_agent import (
    is_durable_agents_available,
    get_durable_agent_status,
    get_workflow_runtime,
    get_workflow_client,
    stop_workflow_runtime,
    unified_workflow,
    DAPR_AGENTS_AVAILABLE,
)


# =============================================================================
# Configuration
# =============================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
PLANS_DIR = Path(os.getenv("PLANS_DIR", "/plans"))

# Phoenix Arize configuration
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://phoenix.phoenix-observability.svc.cluster.local:4317")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "planner-agent")

# Agent Registry configuration
AGENT_REGISTRY_STORE = os.getenv("AGENT_REGISTRY_STORE", "agentregistrystore")
AGENT_TEAM_NAME = os.getenv("AGENT_TEAM_NAME", "planner-agents")
AGENT_APP_ID = os.getenv("DAPR_APP_ID", "planner-agent")


# =============================================================================
# Pydantic Models
# =============================================================================

class CloneRequest(BaseModel):
    """Request model for repository cloning."""
    owner: str
    repo: str
    branch: str = "main"
    token: str | None = None


class CloneResponse(BaseModel):
    """Response model for repository cloning."""
    path: str
    success: bool
    fileCount: int = 0
    error: str | None = None


class WorkflowTargetRepository(BaseModel):
    """Target repository configuration for workflow."""
    owner: str
    repo: str
    branch: str = "main"
    token: str | None = None


class WorkflowOptions(BaseModel):
    """Options for workflow execution."""
    autoApprove: bool = False
    workingDirectory: str | None = None
    targetRepository: WorkflowTargetRepository | None = None


class WorkflowStartRequest(BaseModel):
    """Request model for starting a workflow."""
    prompt: str
    sessionId: str | None = None
    options: WorkflowOptions | None = None


class WorkflowStartResponse(BaseModel):
    """Response model for workflow start."""
    success: bool = True
    workflowId: str | None = None
    status: str | None = None
    error: str | None = None


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status query."""
    success: bool
    instance_id: str
    runtime_status: str | None = None
    custom_status: dict | None = None
    created_at: str | None = None
    last_updated_at: str | None = None
    error: str | None = None


class WorkflowApprovalRequest(BaseModel):
    """Request model for workflow approval event."""
    plan_id: str
    approved: bool
    reviewer: str | None = None
    reason: str | None = None


class WorkflowApprovalResponse(BaseModel):
    """Response model for workflow approval event."""
    success: bool
    instance_id: str
    plan_id: str
    error: str | None = None


class ClarificationResponseRequest(BaseModel):
    """Request model for responding to a clarification request."""
    plan_id: str
    clarification_index: int
    response: str


class ClarificationResponseResponse(BaseModel):
    """Response model for clarification response."""
    success: bool
    instance_id: str
    plan_id: str
    clarification_index: int
    error: str | None = None


class ToolInfo(BaseModel):
    """Information about a Claude Agent SDK tool."""
    name: str
    description: str


class ToolsResponse(BaseModel):
    """Response model for /api/tools endpoint."""
    tools: list[ToolInfo]
    agent_type: str
    capabilities: list[str]


class TaskSnapshotResponse(BaseModel):
    """Response model for a recorded task snapshot (native tool event data)."""
    local_seq: str
    tool_name: str | None = None
    tool_input: dict = {}
    updates: list[dict] = []
    workflow_id: str | None = None
    plan_id: str | None = None
    recorded_at: str | None = None


class TaskListResponse(BaseModel):
    """Response model for listing recorded task snapshots."""
    success: bool
    workflow_id: str
    tasks: list[TaskSnapshotResponse] = []
    total: int = 0
    error: str | None = None


class TaskGetResponse(BaseModel):
    """Response model for getting a single task snapshot."""
    success: bool
    task: TaskSnapshotResponse | None = None
    error: str | None = None


# =============================================================================
# Claude Agent SDK Tools Definition
# =============================================================================

CLAUDE_SDK_TOOLS: list[ToolInfo] = [
    ToolInfo(name="Read", description="Read files from filesystem"),
    ToolInfo(name="Write", description="Write/create files"),
    ToolInfo(name="Edit", description="Edit existing files with string replacement"),
    ToolInfo(name="Bash", description="Execute shell commands"),
    ToolInfo(name="Glob", description="Pattern-based file search"),
    ToolInfo(name="Grep", description="Content search with regex"),
    ToolInfo(name="TaskCreate", description="Create tasks for implementation steps"),
    ToolInfo(name="TaskList", description="List all tasks and their status"),
    ToolInfo(name="TaskUpdate", description="Update task status and dependencies"),
    ToolInfo(name="WebFetch", description="Fetch and analyze web content"),
    ToolInfo(name="WebSearch", description="Search the web"),
    ToolInfo(name="Task", description="Launch sub-agents for complex tasks"),
]

AGENT_CAPABILITIES = ["clone", "plan", "execute", "code-generation", "file-operations", "web-access"]


# =============================================================================
# Agent Registry Functions
# =============================================================================

async def register_agent_in_registry() -> bool:
    """Register this planner-agent in the Dapr agent registry."""
    if not DAPR_AVAILABLE or DaprClient is None:
        print("[AgentRegistry] Dapr not available, skipping registration")
        return False

    try:
        metadata = {
            "appId": AGENT_APP_ID,
            "teamName": AGENT_TEAM_NAME,
            "capabilities": AGENT_CAPABILITIES,
            "endpoints": [
                "/api/clone",
                "/api/workflows",
                "/api/workflow/{instance_id}/status",
                "/api/workflow/{instance_id}/approve",
                "/api/workflow/{instance_id}/clarify",
                "/api/tools",
            ],
            "status": "active",
            "registeredAt": datetime.utcnow().isoformat() + "Z",
            "description": "General-purpose planning and execution agent using Claude Agent SDK native tools",
            "version": "2.0.0",
        }

        with DaprClient() as client:
            key = f"agent:{AGENT_TEAM_NAME}:{AGENT_APP_ID}"
            client.save_state(
                store_name=AGENT_REGISTRY_STORE,
                key=key,
                value=json.dumps(metadata),
            )

            # Update team index
            index_key = f"team-index:{AGENT_TEAM_NAME}"
            try:
                index_response = client.get_state(
                    store_name=AGENT_REGISTRY_STORE,
                    key=index_key,
                )
                if index_response.data:
                    index = json.loads(index_response.data)
                else:
                    index = []
            except Exception:
                index = []

            if AGENT_APP_ID not in index:
                index.append(AGENT_APP_ID)
                client.save_state(
                    store_name=AGENT_REGISTRY_STORE,
                    key=index_key,
                    value=json.dumps(index),
                )

            print(f"[AgentRegistry] Registered agent: {AGENT_APP_ID} (team: {AGENT_TEAM_NAME})")
            return True

    except Exception as e:
        print(f"[AgentRegistry] Warning: Failed to register agent: {e}")
        return False


async def deregister_agent_from_registry() -> bool:
    """Deregister this planner-agent from the Dapr agent registry."""
    if not DAPR_AVAILABLE or DaprClient is None:
        return False

    try:
        with DaprClient() as client:
            key = f"agent:{AGENT_TEAM_NAME}:{AGENT_APP_ID}"

            response = client.get_state(
                store_name=AGENT_REGISTRY_STORE,
                key=key,
            )

            if response.data:
                metadata = json.loads(response.data)
                metadata["status"] = "inactive"
                metadata["lastHeartbeat"] = datetime.utcnow().isoformat() + "Z"

                client.save_state(
                    store_name=AGENT_REGISTRY_STORE,
                    key=key,
                    value=json.dumps(metadata),
                )

                print(f"[AgentRegistry] Deregistered agent: {AGENT_APP_ID}")
                return True

    except Exception as e:
        print(f"[AgentRegistry] Warning: Failed to deregister agent: {e}")
        return False

    return False


# =============================================================================
# FastAPI Lifespan
# =============================================================================

_workflow_runtime = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _workflow_runtime

    # Startup
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Workflow Service] Started. Workspace: {WORKSPACE_DIR}")

    # Initialize Phoenix Arize observability
    if PHOENIX_AVAILABLE and phoenix_register:
        try:
            tracer_provider = phoenix_register(
                project_name=PHOENIX_PROJECT_NAME,
                endpoint=PHOENIX_ENDPOINT,
                protocol="grpc",
            )

            if ANTHROPIC_INSTRUMENTOR_AVAILABLE and AnthropicInstrumentor:
                anthropic_instrumentor = AnthropicInstrumentor()
                anthropic_instrumentor.instrument(tracer_provider=tracer_provider)
                print(f"[Workflow Service] Anthropic instrumentor enabled")

            print(f"[Workflow Service] Phoenix observability enabled: {PHOENIX_ENDPOINT}")
        except Exception as e:
            print(f"[Workflow Service] Warning: Could not initialize Phoenix: {e}")
    else:
        print("[Workflow Service] Phoenix observability not available")

    # Start Dapr workflow runtime
    if DAPR_AVAILABLE and WorkflowRuntime:
        try:
            _workflow_runtime = get_workflow_runtime()
            if _workflow_runtime:
                _workflow_runtime.start()
                print("[Workflow Service] Dapr workflow runtime started")
            else:
                print("[Workflow Service] Warning: Could not get workflow runtime")
        except Exception as e:
            print(f"[Workflow Service] Warning: Could not start workflow runtime: {e}")
            _workflow_runtime = None
    else:
        print("[Workflow Service] Dapr not available - running in HTTP-only mode")

    # Register agent
    await register_agent_in_registry()

    print(f"[Workflow Service] Durable workflows available: {is_durable_agents_available()}")

    yield

    # Shutdown
    await deregister_agent_from_registry()
    await stop_workflow_runtime()
    print("[Workflow Service] Shutdown complete")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Planner Agent Workflow Service",
    description="HTTP endpoints for AI-powered planning and execution using native Claude Code tools",
    version="2.0.0",
    lifespan=lifespan,
)


# =============================================================================
# HTTP Endpoints
# =============================================================================

@app.post("/api/clone", response_model=CloneResponse)
async def clone_repository(request: CloneRequest) -> CloneResponse:
    """
    Clone a GitHub repository into /workspace.

    This is a standalone clone operation separate from the workflow.
    For durable cloning as part of a workflow, use /api/workflows with
    targetRepository in the options.
    """
    repo_url = f"https://github.com/{request.owner}/{request.repo}.git"
    clone_path = WORKSPACE_DIR / request.repo

    print(f"[Clone] Cloning {request.owner}/{request.repo} to {clone_path}")

    try:
        if clone_path.exists():
            print(f"[Clone] Removing existing directory: {clone_path}")
            shutil.rmtree(clone_path)

        cmd = [
            "git", "clone",
            "--depth", "1",
            "--single-branch",
            "--branch", request.branch,
        ]

        if request.token:
            auth_url = f"https://{request.token}@github.com/{request.owner}/{request.repo}.git"
            cmd.append(auth_url)
        else:
            cmd.append(repo_url)

        cmd.append(str(clone_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            print(f"[Clone] Failed: {error_msg}")
            return CloneResponse(
                path="",
                success=False,
                error=f"Clone failed: {error_msg}",
            )

        file_count = sum(1 for _ in clone_path.rglob("*") if _.is_file())
        print(f"[Clone] Success: {file_count} files cloned")

        return CloneResponse(
            path=str(clone_path),
            success=True,
            fileCount=file_count,
        )

    except subprocess.TimeoutExpired:
        return CloneResponse(
            path="",
            success=False,
            error="Clone timed out after 5 minutes",
        )
    except Exception as e:
        print(f"[Clone] Error: {e}")
        return CloneResponse(
            path="",
            success=False,
            error=f"Clone failed: {str(e)}",
        )


@app.post("/api/workflows", response_model=WorkflowStartResponse)
async def start_workflow(request: WorkflowStartRequest) -> WorkflowStartResponse:
    """
    Start a unified planning and execution workflow.

    This is the SINGLE ORCHESTRATOR endpoint. The workflow handles:
    0. Clone repository (if targetRepository provided in options)
    1. Planning - Claude uses TaskCreate to create implementation tasks
    2. Wait for approval (external event via /api/workflow/{id}/approve)
    3. Execution - Claude uses TaskList/TaskUpdate to execute tasks

    The clone happens INSIDE the workflow for durability.
    """
    print(f"[Workflows] Starting workflow")
    print(f"[Workflows] Prompt: {request.prompt[:100]}...")

    try:
        client = get_workflow_client()

        if client is None:
            return WorkflowStartResponse(
                success=False,
                error="Dapr workflow client not available. Ensure Dapr sidecar is running.",
            )

        instance_id = request.sessionId or f"workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        workflow_input_data: dict[str, Any] = {
            "feature_request": request.prompt,
            "plans_dir": str(PLANS_DIR),
            "auto_approve": request.options.autoApprove if request.options else False,
            "workflow_id": instance_id,
        }

        # Include repository info for workflow to clone
        if request.options and request.options.targetRepository:
            repo = request.options.targetRepository
            print(f"[Workflows] Target repository: {repo.owner}/{repo.repo}@{repo.branch}")
            workflow_input_data["repository"] = {
                "owner": repo.owner,
                "repo": repo.repo,
                "branch": repo.branch,
                "token": repo.token,
            }
        elif request.options and request.options.workingDirectory:
            cwd = request.options.workingDirectory
            cwd_path = Path(cwd)
            if not cwd_path.exists():
                return WorkflowStartResponse(
                    success=False,
                    error=f"Working directory not found: {cwd}",
                )
            workflow_input_data["cwd"] = cwd
        else:
            workflow_input_data["cwd"] = str(WORKSPACE_DIR)

        workflow_input = json.dumps(workflow_input_data)

        client.schedule_new_workflow(
            workflow=unified_workflow,
            input=workflow_input,
            instance_id=instance_id,
        )

        print(f"[Workflows] Workflow started: {instance_id}")

        return WorkflowStartResponse(
            success=True,
            workflowId=instance_id,
            status="pending",
        )

    except Exception as e:
        print(f"[Workflows] Error: {e}")
        import traceback
        traceback.print_exc()
        return WorkflowStartResponse(
            success=False,
            error=f"Failed to start workflow: {str(e)}",
        )


@app.get("/api/workflow/{instance_id}/status", response_model=WorkflowStatusResponse)
@app.get("/api/workflows/{instance_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(instance_id: str) -> WorkflowStatusResponse:
    """
    Get the current status of a workflow.

    Note: Supports both /api/workflow/ (singular) and /api/workflows/ (plural) paths.

    Returns:
    - runtime_status: PENDING, RUNNING, COMPLETED, FAILED, SUSPENDED, TERMINATED
    - custom_status: Application-defined status with phase, progress, message

    Phases: clone, planning, awaiting_approval, executing, completed, failed, rejected
    """
    try:
        client = get_workflow_client()

        if client is None:
            return WorkflowStatusResponse(
                success=False,
                instance_id=instance_id,
                error="Dapr workflow client not available",
            )

        state = client.get_workflow_state(instance_id=instance_id)

        if state is None:
            return WorkflowStatusResponse(
                success=False,
                instance_id=instance_id,
                error=f"Workflow {instance_id} not found",
            )

        # Parse custom status from serialized_custom_status
        custom_status = None
        # Try to_json() first which returns a dict with serialized_custom_status
        if hasattr(state, 'to_json'):
            state_dict = state.to_json()
            if isinstance(state_dict, dict) and state_dict.get('serialized_custom_status'):
                custom_status_str = state_dict['serialized_custom_status']
                try:
                    # May be double-encoded JSON, try parsing until we get a dict
                    parsed = json.loads(custom_status_str)
                    while isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    custom_status = parsed if isinstance(parsed, dict) else {"raw": str(parsed)}
                except (json.JSONDecodeError, TypeError):
                    custom_status = {"raw": str(custom_status_str)}
        # Fallback to properties if to_json doesn't have it
        if custom_status is None and hasattr(state, 'properties') and state.properties:
            custom_status_str = state.properties.get('dapr.workflow.custom_status')
            if custom_status_str:
                try:
                    parsed = json.loads(custom_status_str)
                    while isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    custom_status = parsed if isinstance(parsed, dict) else {"raw": str(parsed)}
                except json.JSONDecodeError:
                    custom_status = {"raw": custom_status_str}

        runtime_status = None
        if hasattr(state, 'runtime_status') and state.runtime_status:
            runtime_status = state.runtime_status.name if hasattr(state.runtime_status, 'name') else str(state.runtime_status)

        return WorkflowStatusResponse(
            success=True,
            instance_id=instance_id,
            runtime_status=runtime_status,
            custom_status=custom_status,
            created_at=state.created_at.isoformat() if hasattr(state, 'created_at') and state.created_at else None,
            last_updated_at=state.last_updated_at.isoformat() if hasattr(state, 'last_updated_at') and state.last_updated_at else None,
        )

    except Exception as e:
        print(f"[Workflow Status] Error: {e}")
        return WorkflowStatusResponse(
            success=False,
            instance_id=instance_id,
            error=f"Failed to get workflow status: {str(e)}",
        )


@app.post("/api/workflow/{instance_id}/approve", response_model=WorkflowApprovalResponse)
@app.post("/api/workflows/{instance_id}/approve", response_model=WorkflowApprovalResponse)
async def approve_workflow(
    instance_id: str,
    request: WorkflowApprovalRequest,
) -> WorkflowApprovalResponse:
    """
    Raise approval event to resume a workflow waiting for plan approval.

    This uses Dapr's native external event mechanism to resume a workflow
    that is paused at wait_for_external_event. The workflow will continue
    with execution if approved, or return a rejection result.

    Note: Supports both /api/workflow/ (singular) and /api/workflows/ (plural) paths.
    """
    print(f"[Workflow Approval] Raising approval event for workflow {instance_id}")
    print(f"[Workflow Approval] Plan: {request.plan_id}, Approved: {request.approved}")

    try:
        client = get_workflow_client()

        if client is None:
            return WorkflowApprovalResponse(
                success=False,
                instance_id=instance_id,
                plan_id=request.plan_id,
                error="Dapr workflow client not available",
            )

        # If plan_id is a default placeholder, look up the actual plan_id from workflow state
        actual_plan_id = request.plan_id
        if request.plan_id in ("plan_1", "plan-1", None, ""):
            print(f"[Workflow Approval] Default plan_id detected, looking up from workflow state")
            try:
                state = client.get_workflow_state(instance_id=instance_id)
                if state and hasattr(state, 'to_json'):
                    state_dict = state.to_json()
                    if isinstance(state_dict, dict) and state_dict.get('serialized_custom_status'):
                        custom_status_str = state_dict['serialized_custom_status']
                        parsed = json.loads(custom_status_str)
                        while isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        if isinstance(parsed, dict) and parsed.get('plan_id'):
                            actual_plan_id = parsed['plan_id']
                            print(f"[Workflow Approval] Found actual plan_id: {actual_plan_id}")
            except Exception as lookup_err:
                print(f"[Workflow Approval] Could not lookup plan_id: {lookup_err}")

        event_name = f"plan_approval_{actual_plan_id}"
        event_data = {
            "approved": request.approved,
            "reviewer": request.reviewer,
            "reason": request.reason,
        }

        client.raise_workflow_event(
            instance_id=instance_id,
            event_name=event_name,
            data=event_data,
        )

        print(f"[Workflow Approval] Event raised: {event_name}")

        return WorkflowApprovalResponse(
            success=True,
            instance_id=instance_id,
            plan_id=actual_plan_id,
        )

    except Exception as e:
        print(f"[Workflow Approval] Error: {e}")
        return WorkflowApprovalResponse(
            success=False,
            instance_id=instance_id,
            plan_id=request.plan_id,
            error=f"Failed to raise approval event: {str(e)}",
        )


@app.post("/api/workflow/{instance_id}/clarify", response_model=ClarificationResponseResponse)
async def respond_to_clarification(
    instance_id: str,
    request: ClarificationResponseRequest,
) -> ClarificationResponseResponse:
    """
    Respond to a clarification request to resume a paused planning workflow.

    When planning uses AskUserQuestion, the workflow pauses and waits for
    a clarification response via this endpoint. The planning phase will
    resume with the provided response.

    Args:
        instance_id: The workflow instance ID
        request: Contains plan_id, clarification_index, and the response text
    """
    print(f"[Workflow Clarification] Responding to clarification for workflow {instance_id}")
    print(f"[Workflow Clarification] Plan: {request.plan_id}, Index: {request.clarification_index}")

    try:
        client = get_workflow_client()

        if client is None:
            return ClarificationResponseResponse(
                success=False,
                instance_id=instance_id,
                plan_id=request.plan_id,
                clarification_index=request.clarification_index,
                error="Dapr workflow client not available",
            )

        # Raise the clarification event to resume the workflow
        event_name = f"clarification_{request.plan_id}_{request.clarification_index}"
        event_data = {
            "response": request.response,
        }

        client.raise_workflow_event(
            instance_id=instance_id,
            event_name=event_name,
            data=event_data,
        )

        print(f"[Workflow Clarification] Event raised: {event_name}")

        return ClarificationResponseResponse(
            success=True,
            instance_id=instance_id,
            plan_id=request.plan_id,
            clarification_index=request.clarification_index,
        )

    except Exception as e:
        print(f"[Workflow Clarification] Error: {e}")
        return ClarificationResponseResponse(
            success=False,
            instance_id=instance_id,
            plan_id=request.plan_id,
            clarification_index=request.clarification_index,
            error=f"Failed to respond to clarification: {str(e)}",
        )


@app.get("/api/workflow/{instance_id}/tasks", response_model=TaskListResponse)
@app.get("/api/workflows/{instance_id}/tasks", response_model=TaskListResponse)
async def list_workflow_tasks(instance_id: str) -> TaskListResponse:
    """
    List recorded task snapshots for a workflow.

    Returns snapshots of native Claude Code task tool events.
    The native SDK manages all task logic; these are just recordings.

    Note: Supports both /api/workflow/ (singular) and /api/workflows/ (plural) paths.
    """
    from task_persistence import TaskStore

    try:
        tasks_dir = WORKSPACE_DIR / "tasks" / instance_id

        if not tasks_dir.exists():
            return TaskListResponse(
                success=True,
                workflow_id=instance_id,
                tasks=[],
                total=0,
            )

        store = TaskStore(
            base_path=WORKSPACE_DIR / "tasks",
            workflow_id=instance_id,
        )

        snapshots = store.list_tasks()

        task_responses = []
        for snap in snapshots:
            task_responses.append(TaskSnapshotResponse(
                local_seq=snap.get("local_seq", ""),
                tool_name=snap.get("tool_name"),
                tool_input=snap.get("tool_input", {}),
                updates=snap.get("updates", []),
                workflow_id=snap.get("workflow_id"),
                plan_id=snap.get("plan_id"),
                recorded_at=snap.get("recorded_at"),
            ))

        return TaskListResponse(
            success=True,
            workflow_id=instance_id,
            tasks=task_responses,
            total=len(task_responses),
        )

    except Exception as e:
        print(f"[Task List] Error: {e}")
        return TaskListResponse(
            success=False,
            workflow_id=instance_id,
            error=f"Failed to list tasks: {str(e)}",
        )


@app.get("/api/workflow/{instance_id}/tasks/{task_id}", response_model=TaskGetResponse)
@app.get("/api/workflows/{instance_id}/tasks/{task_id}", response_model=TaskGetResponse)
async def get_task(instance_id: str, task_id: str) -> TaskGetResponse:
    """
    Get a single recorded task snapshot.

    Note: Supports both /api/workflow/ (singular) and /api/workflows/ (plural) paths.
    """
    from task_persistence import TaskStore

    try:
        tasks_dir = WORKSPACE_DIR / "tasks" / instance_id

        if not tasks_dir.exists():
            return TaskGetResponse(
                success=False,
                error=f"No tasks found for workflow {instance_id}",
            )

        store = TaskStore(
            base_path=WORKSPACE_DIR / "tasks",
            workflow_id=instance_id,
        )

        snap = store.get_task(task_id)
        if snap is None:
            return TaskGetResponse(
                success=False,
                error=f"Task {task_id} not found in workflow {instance_id}",
            )

        return TaskGetResponse(
            success=True,
            task=TaskSnapshotResponse(
                local_seq=snap.get("local_seq", ""),
                tool_name=snap.get("tool_name"),
                tool_input=snap.get("tool_input", {}),
                updates=snap.get("updates", []),
                workflow_id=snap.get("workflow_id"),
                plan_id=snap.get("plan_id"),
                recorded_at=snap.get("recorded_at"),
            ),
        )

    except Exception as e:
        print(f"[Task Get] Error: {e}")
        return TaskGetResponse(
            success=False,
            error=f"Failed to get task: {str(e)}",
        )


@app.get("/api/tools", response_model=ToolsResponse)
async def list_tools() -> ToolsResponse:
    """
    List all available Claude Agent SDK tools.

    This includes native Claude Code tools for file operations,
    search, and task management.
    """
    return ToolsResponse(
        tools=CLAUDE_SDK_TOOLS,
        agent_type="general-purpose",
        capabilities=AGENT_CAPABILITIES,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    durable_status = get_durable_agent_status()
    return {
        "status": "healthy",
        "dapr_available": DAPR_AVAILABLE,
        "dapr_workflow_available": durable_status["dapr_workflow_available"],
        "claude_cli_available": durable_status["claude_cli_available"],
        "phoenix_available": PHOENIX_AVAILABLE,
        "workspace": str(WORKSPACE_DIR),
        "agent_registry": {
            "app_id": AGENT_APP_ID,
            "team_name": AGENT_TEAM_NAME,
        },
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "planner-agent-workflow-service",
        "version": "2.0.0",
        "agent_type": "general-purpose",
        "durable_agents_available": is_durable_agents_available(),
        "endpoints": [
            "/api/clone",
            "/api/workflows",
            "/api/workflow/{instance_id}/status",
            "/api/workflow/{instance_id}/approve",
            "/api/workflow/{instance_id}/clarify",
            "/api/workflow/{instance_id}/tasks",
            "/api/workflow/{instance_id}/tasks/{task_id}",
            "/api/tools",
            "/health",
        ],
        "agent_registry": {
            "app_id": AGENT_APP_ID,
            "team_name": AGENT_TEAM_NAME,
            "capabilities": AGENT_CAPABILITIES,
        },
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "workflow_service:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
