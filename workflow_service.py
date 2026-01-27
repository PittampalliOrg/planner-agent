"""
Dapr Workflow Service for Planner Agent

Exposes HTTP endpoints and Dapr workflow activities for:
- Repository cloning
- Plan creation via Claude SDK
- Plan approval handling

This service is designed to integrate with the TypeScript workflow-patterns
(Next.js/Dapr) and provide planning capabilities via the Python planner-agent.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
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
    from dapr_agents.observability import DaprAgentsInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    phoenix_register = None
    DaprAgentsInstrumentor = None

# OpenInference Anthropic instrumentor for Claude SDK tracing
try:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    ANTHROPIC_INSTRUMENTOR_AVAILABLE = True
except ImportError:
    ANTHROPIC_INSTRUMENTOR_AVAILABLE = False
    AnthropicInstrumentor = None

from planner_agent import PlannerAgent
from durable_agent import (
    DurablePlanningWorkflow,
    is_durable_agents_available,
    get_durable_agent_status,
    create_durable_plan,
    execute_durable_plan,
    get_workflow_runtime,
    get_workflow_client,
    start_workflow_runtime,
    stop_workflow_runtime,
    planning_and_execution_workflow,
    DAPR_AGENTS_AVAILABLE,
)


# =============================================================================
# Configuration
# =============================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
PLANS_DIR = Path(os.getenv("PLANS_DIR", "/plans"))

# Phoenix Arize configuration - use GRPC endpoint (4317) for trace export
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


class PlanRequest(BaseModel):
    """Request model for plan creation."""
    cwd: str
    prompt: str


class PlanStepResponse(BaseModel):
    """Plan step in the response."""
    title: str
    description: str
    files_affected: list[str] = []
    complexity: str = "medium"


class PlanResponse(BaseModel):
    """Response model for plan creation."""
    id: str
    title: str
    summary: str
    steps: list[PlanStepResponse]
    critical_files: list[str] = []
    considerations: list[str] = []
    status: str = "draft"


class PlanCreateResponse(BaseModel):
    """Wrapper response for plan creation."""
    plan: PlanResponse | None = None
    error: str | None = None


class ExecuteRequest(BaseModel):
    """Request model for plan execution."""
    repo_path: str
    plan_id: str
    workflow_id: str


class ExecuteResponse(BaseModel):
    """Response model for plan execution."""
    success: bool
    tasks_completed: int = 0
    tasks_total: int = 0
    files_changed: list[str] = []
    error: str | None = None


class PlanStatusUpdateRequest(BaseModel):
    """Request model for plan status update."""
    status: str  # "approved" or "rejected"
    reviewer: str | None = None
    comments: str | None = None


class PlanStatusUpdateResponse(BaseModel):
    """Response model for plan status update."""
    success: bool
    plan_id: str
    status: str
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


class DurablePlanRequest(BaseModel):
    """Request model for durable plan creation."""
    cwd: str
    prompt: str
    session_id: str | None = None
    workflow_id: str | None = None  # UI workflow ID for streaming events
    use_durable: bool = True  # Whether to use DurableAgent (falls back if unavailable)


class DurablePlanResponse(BaseModel):
    """Response model for durable plan creation."""
    success: bool
    plan_id: str | None = None
    title: str | None = None
    summary: str | None = None
    steps_count: int = 0
    status: str | None = None
    durable_execution: bool = False  # Whether DurableAgent was used
    error: str | None = None


class DurableExecuteRequest(BaseModel):
    """Request model for durable plan execution."""
    cwd: str
    plan_id: str
    workflow_id: str
    session_id: str | None = None


class DurableExecuteResponse(BaseModel):
    """Response model for durable plan execution."""
    success: bool
    tasks_completed: int = 0
    tasks_total: int = 0
    files_changed: list[str] = []
    durable_execution: bool = False
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


class AgentMetadata(BaseModel):
    """Agent metadata for registry."""
    appId: str
    teamName: str
    capabilities: list[str]
    endpoints: list[str]
    status: str
    registeredAt: str
    description: str | None = None
    version: str | None = None


class WorkflowTargetRepository(BaseModel):
    """Target repository configuration for workflow."""
    owner: str
    repo: str
    branch: str = "main"
    token: str | None = None  # GitHub access token for private repos


class WorkflowOptions(BaseModel):
    """Options for workflow execution."""
    autoApprove: bool = False
    workingDirectory: str | None = None
    targetRepository: WorkflowTargetRepository | None = None


class WorkflowStartRequest(BaseModel):
    """Request model for starting a workflow (Next.js compatible)."""
    prompt: str
    sessionId: str | None = None
    options: WorkflowOptions | None = None


class WorkflowStartResponse(BaseModel):
    """Response model for workflow start (Next.js compatible)."""
    success: bool = True
    workflowId: str | None = None
    status: str | None = None
    error: str | None = None


# =============================================================================
# Claude Agent SDK Tools Definition
# =============================================================================

# All native Claude Agent SDK tools that planner-agent exposes
CLAUDE_SDK_TOOLS: list[ToolInfo] = [
    ToolInfo(name="Read", description="Read files from filesystem"),
    ToolInfo(name="Write", description="Write/create files"),
    ToolInfo(name="Edit", description="Edit existing files with string replacement"),
    ToolInfo(name="Bash", description="Execute shell commands"),
    ToolInfo(name="Glob", description="Pattern-based file search"),
    ToolInfo(name="Grep", description="Content search with regex"),
    ToolInfo(name="WebFetch", description="Fetch and analyze web content"),
    ToolInfo(name="WebSearch", description="Search the web"),
    ToolInfo(name="Task", description="Launch sub-agents for complex tasks"),
    ToolInfo(name="AskUserQuestion", description="Interactive clarification"),
]

# Agent capabilities
AGENT_CAPABILITIES = ["clone", "plan", "execute", "code-generation", "file-operations", "web-access"]


# =============================================================================
# Agent Registry Functions
# =============================================================================

async def register_agent_in_registry() -> bool:
    """
    Register this planner-agent in the Dapr agent registry.
    Called at startup to announce availability.

    Returns True if registration succeeded, False otherwise.
    """
    if not DAPR_AVAILABLE or DaprClient is None:
        print("[AgentRegistry] Dapr not available, skipping registration")
        return False

    try:
        from datetime import datetime
        import json

        metadata = {
            "appId": AGENT_APP_ID,
            "teamName": AGENT_TEAM_NAME,
            "capabilities": AGENT_CAPABILITIES,
            "endpoints": [
                "/api/clone",
                "/api/plan",
                "/api/execute",
                "/api/durable/plan",
                "/api/durable/execute",
                "/api/durable/plan-and-execute",
                "/api/plan/{plan_id}/approve",
                "/api/workflow/{instance_id}/approve",
                "/api/tools",
            ],
            "status": "active",
            "registeredAt": datetime.utcnow().isoformat() + "Z",
            "description": "General-purpose planning and execution agent using Claude Agent SDK",
            "version": "1.0.0",
        }

        with DaprClient() as client:
            # Save agent metadata
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
            print(f"[AgentRegistry] Capabilities: {AGENT_CAPABILITIES}")
            return True

    except Exception as e:
        print(f"[AgentRegistry] Warning: Failed to register agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def deregister_agent_from_registry() -> bool:
    """
    Deregister this planner-agent from the Dapr agent registry.
    Called at shutdown for graceful cleanup.

    Returns True if deregistration succeeded, False otherwise.
    """
    if not DAPR_AVAILABLE or DaprClient is None:
        return False

    try:
        from datetime import datetime
        import json

        with DaprClient() as client:
            key = f"agent:{AGENT_TEAM_NAME}:{AGENT_APP_ID}"

            # Get existing metadata
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

# Global workflow runtime reference for shutdown
_workflow_runtime = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _workflow_runtime

    # Startup
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Workflow Service] Started. Workspace: {WORKSPACE_DIR}, Plans: {PLANS_DIR}")

    # Initialize Phoenix Arize observability
    if PHOENIX_AVAILABLE and phoenix_register:
        try:
            # Use grpc protocol for OTLP export on port 4317
            tracer_provider = phoenix_register(
                project_name=PHOENIX_PROJECT_NAME,
                endpoint=PHOENIX_ENDPOINT,
                protocol="grpc",  # Use gRPC for OTLP export
            )

            # Instrument Dapr Agents operations (if available)
            if DaprAgentsInstrumentor:
                dapr_instrumentor = DaprAgentsInstrumentor()
                dapr_instrumentor.instrument(tracer_provider=tracer_provider)
                print(f"[Workflow Service] Dapr Agents instrumentor enabled")

            # Instrument Anthropic/Claude SDK calls for LLM tracing
            if ANTHROPIC_INSTRUMENTOR_AVAILABLE and AnthropicInstrumentor:
                anthropic_instrumentor = AnthropicInstrumentor()
                anthropic_instrumentor.instrument(tracer_provider=tracer_provider)
                print(f"[Workflow Service] Anthropic instrumentor enabled (Claude SDK tracing)")

            print(f"[Workflow Service] Phoenix observability enabled: {PHOENIX_ENDPOINT}")
        except Exception as e:
            print(f"[Workflow Service] Warning: Could not initialize Phoenix observability: {e}")
    else:
        print("[Workflow Service] Phoenix observability not available")

    # Start Dapr workflow runtime with Claude SDK native tools
    # This registers both HTTP activities AND Claude SDK workflow activities
    if DAPR_AVAILABLE and WorkflowRuntime:
        try:
            # Get the workflow runtime from durable_agent (includes Claude SDK activities)
            _workflow_runtime = get_workflow_runtime()

            if _workflow_runtime:
                # Also register HTTP-related activities (with http_ prefix to avoid collision)
                _workflow_runtime.register_activity(http_clone_repository_activity)
                _workflow_runtime.register_activity(http_create_plan_activity)
                _workflow_runtime.register_activity(http_execute_plan_activity)
                _workflow_runtime.start()  # Synchronous - do NOT await
                print("[Workflow Service] Dapr workflow runtime started (Claude SDK native tools)")
            else:
                print("[Workflow Service] Warning: Could not get workflow runtime")
        except Exception as e:
            print(f"[Workflow Service] Warning: Could not start Dapr workflow runtime: {e}")
            _workflow_runtime = None
    else:
        print("[Workflow Service] Dapr not available - running in HTTP-only mode")

    # Register agent in the Dapr agent registry
    await register_agent_in_registry()

    print(f"[Workflow Service] Durable workflows available: {is_durable_agents_available()}")
    print(f"[Workflow Service] Status: {get_durable_agent_status()}")

    yield

    # Shutdown - deregister from agent registry first
    await deregister_agent_from_registry()

    # Stop workflow runtime
    await stop_workflow_runtime()
    print("[Workflow Service] Workflow runtime shutdown complete")



# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Planner Agent Workflow Service",
    description="HTTP endpoints for repository cloning and AI-powered planning",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# HTTP Endpoints (called by TypeScript activities)
# =============================================================================

@app.post("/api/clone", response_model=CloneResponse)
async def clone_repository(request: CloneRequest) -> CloneResponse:
    """
    Clone a GitHub repository into /workspace.

    This endpoint is called by the TypeScript cloneRepositoryActivity.
    """
    repo_url = f"https://github.com/{request.owner}/{request.repo}.git"
    clone_path = WORKSPACE_DIR / request.repo

    print(f"[Clone] Cloning {request.owner}/{request.repo} to {clone_path}")

    try:
        # Remove existing if present
        if clone_path.exists():
            print(f"[Clone] Removing existing directory: {clone_path}")
            shutil.rmtree(clone_path)

        # Build clone command
        cmd = [
            "git", "clone",
            "--depth", "1",
            "--single-branch",
            "--branch", request.branch,
        ]

        # Add auth if token provided
        if request.token:
            auth_url = f"https://{request.token}@github.com/{request.owner}/{request.repo}.git"
            cmd.append(auth_url)
        else:
            cmd.append(repo_url)

        cmd.append(str(clone_path))

        # Execute clone
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            print(f"[Clone] Failed: {error_msg}")
            return CloneResponse(
                path="",
                success=False,
                error=f"Clone failed: {error_msg}",
            )

        # Count files
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


@app.post("/api/plan", response_model=PlanCreateResponse)
async def create_plan(request: PlanRequest) -> PlanCreateResponse:
    """
    Create an implementation plan using the Claude SDK planner agent.

    This endpoint is called by the TypeScript createPlanActivity.
    """
    print(f"[Plan] Creating plan for {request.cwd}")
    print(f"[Plan] Prompt: {request.prompt[:100]}...")

    try:
        # Verify the directory exists
        cwd_path = Path(request.cwd)
        if not cwd_path.exists():
            return PlanCreateResponse(
                plan=None,
                error=f"Directory not found: {request.cwd}",
            )

        # Create planner agent
        agent = PlannerAgent(
            cwd=request.cwd,
            plans_dir=str(PLANS_DIR),
        )

        # Run planning (exploration + plan creation, not implementation)
        await agent.run_planning_only(request.prompt)

        # Check if plan was created
        if not agent.plan_manager.current_plan:
            return PlanCreateResponse(
                plan=None,
                error="Failed to create plan - agent did not produce a plan",
            )

        # Convert plan to response format
        plan = agent.plan_manager.current_plan
        steps = [
            PlanStepResponse(
                title=s.title,
                description=s.description,
                files_affected=s.files_affected,
                complexity=s.estimated_complexity,
            )
            for s in plan.steps
        ]

        # Handle critical_files - may be strings or dicts
        critical_files = []
        for cf in plan.critical_files:
            if isinstance(cf, str):
                critical_files.append(cf)
            elif isinstance(cf, dict):
                # Extract path or convert to string
                critical_files.append(cf.get("path", str(cf)))
            else:
                critical_files.append(str(cf))

        # Handle considerations - may be strings or dicts
        considerations = []
        for c in plan.considerations:
            if isinstance(c, str):
                considerations.append(c)
            elif isinstance(c, dict):
                # Format as "title: description" or just the dict as string
                title = c.get("title", "")
                desc = c.get("description", "")
                if title and desc:
                    considerations.append(f"{title}: {desc}")
                elif title:
                    considerations.append(title)
                else:
                    considerations.append(str(c))
            else:
                considerations.append(str(c))

        return PlanCreateResponse(
            plan=PlanResponse(
                id=plan.id,
                title=plan.title,
                summary=plan.summary,
                steps=steps,
                critical_files=critical_files,
                considerations=considerations,
                status=plan.status,
            ),
        )

    except Exception as e:
        print(f"[Plan] Error: {e}")
        import traceback
        traceback.print_exc()
        return PlanCreateResponse(
            plan=None,
            error=f"Planning failed: {str(e)}",
        )


@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_plan(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute an approved implementation plan.

    This endpoint is called by the TypeScript executePlanActivity after
    the user approves a plan. It runs the implementation using Claude SDK.
    """
    print(f"[Execute] Starting execution for plan {request.plan_id}")
    print(f"[Execute] Repository: {request.repo_path}")
    print(f"[Execute] Workflow ID: {request.workflow_id}")

    try:
        # Verify the directory exists
        cwd_path = Path(request.repo_path)
        if not cwd_path.exists():
            return ExecuteResponse(
                success=False,
                error=f"Directory not found: {request.repo_path}",
            )

        # Create planner agent
        agent = PlannerAgent(
            cwd=request.repo_path,
            plans_dir=str(PLANS_DIR),
        )

        # Run execution
        result = await agent.run_execution_only(
            plan_id=request.plan_id,
            workflow_id=request.workflow_id,
        )

        print(f"[Execute] Completed: {result.get('tasks_completed', 0)}/{result.get('tasks_total', 0)} tasks")

        return ExecuteResponse(
            success=result.get("success", False),
            tasks_completed=result.get("tasks_completed", 0),
            tasks_total=result.get("tasks_total", 0),
            files_changed=result.get("files_changed", []),
            error=result.get("error"),
        )

    except Exception as e:
        print(f"[Execute] Error: {e}")
        import traceback
        traceback.print_exc()
        return ExecuteResponse(
            success=False,
            error=f"Execution failed: {str(e)}",
        )


@app.post("/api/durable/plan", response_model=DurablePlanResponse)
async def create_durable_plan_endpoint(request: DurablePlanRequest) -> DurablePlanResponse:
    """
    Create an implementation plan using the Dapr DurableAgent.

    This endpoint provides fault-tolerant plan creation with:
    - Durable state management (survives crashes/restarts)
    - Automatic retry mechanisms
    - Persistent conversation history

    Falls back to standard PlannerAgent if dapr-agents is not available.
    """
    print(f"[Durable Plan] Creating durable plan for {request.cwd}")
    print(f"[Durable Plan] Prompt: {request.prompt[:100]}...")
    print(f"[Durable Plan] Claude SDK native tools available: {is_durable_agents_available()}")

    try:
        # Verify the directory exists
        cwd_path = Path(request.cwd)
        if not cwd_path.exists():
            return DurablePlanResponse(
                success=False,
                error=f"Directory not found: {request.cwd}",
            )

        # Use durable workflow with Claude SDK native tools
        result = await create_durable_plan(
            cwd=request.cwd,
            feature_request=request.prompt,
            session_id=request.session_id,
            workflow_id=request.workflow_id,  # Pass UI workflow ID for streaming
        )

        if result.get("success"):
            return DurablePlanResponse(
                success=True,
                plan_id=result.get("plan_id"),
                title=result.get("title"),
                summary=result.get("summary"),
                steps_count=result.get("steps_count", len(result.get("steps", []))),
                status=result.get("status", "draft"),
                durable_execution=is_durable_agents_available(),
            )
        else:
            return DurablePlanResponse(
                success=False,
                error=result.get("error", "Unknown error"),
                durable_execution=is_durable_agents_available(),
            )

    except Exception as e:
        print(f"[Durable Plan] Error: {e}")
        import traceback
        traceback.print_exc()
        return DurablePlanResponse(
            success=False,
            error=f"Durable planning failed: {str(e)}",
        )


@app.post("/api/durable/execute", response_model=DurableExecuteResponse)
async def execute_durable_plan_endpoint(request: DurableExecuteRequest) -> DurableExecuteResponse:
    """
    Execute an approved plan using the Dapr DurableAgent.

    This endpoint provides fault-tolerant execution with:
    - Durable state management (survives crashes/restarts)
    - Automatic retry mechanisms
    - Progress streaming via Dapr pub/sub

    Falls back to standard PlannerAgent if dapr-agents is not available.
    """
    print(f"[Durable Execute] Starting durable execution for plan {request.plan_id}")
    print(f"[Durable Execute] CWD: {request.cwd}")
    print(f"[Durable Execute] Claude SDK native tools available: {is_durable_agents_available()}")

    try:
        # Verify the directory exists
        cwd_path = Path(request.cwd)
        if not cwd_path.exists():
            return DurableExecuteResponse(
                success=False,
                error=f"Directory not found: {request.cwd}",
            )

        # Use durable workflow with Claude SDK native tools
        result = await execute_durable_plan(
            cwd=request.cwd,
            plan_id=request.plan_id,
            workflow_id=request.workflow_id,
            session_id=request.session_id,
        )

        return DurableExecuteResponse(
            success=result.get("success", False),
            tasks_completed=result.get("tasks_completed", 0),
            tasks_total=result.get("tasks_total", 0),
            files_changed=result.get("files_changed", []),
            durable_execution=is_durable_agents_available(),
            error=result.get("error"),
        )

    except Exception as e:
        print(f"[Durable Execute] Error: {e}")
        import traceback
        traceback.print_exc()
        return DurableExecuteResponse(
            success=False,
            error=f"Durable execution failed: {str(e)}",
        )


@app.get("/api/tools", response_model=ToolsResponse)
async def list_tools() -> ToolsResponse:
    """
    List all available Claude Agent SDK tools.

    This endpoint exposes the tools that planner-agent makes available
    for task execution. These are the native Claude Agent SDK tools
    that enable general-purpose coding and file operations.
    """
    return ToolsResponse(
        tools=CLAUDE_SDK_TOOLS,
        agent_type="general-purpose",
        capabilities=AGENT_CAPABILITIES,
    )


# =============================================================================
# Plan Approval Endpoints
# =============================================================================

@app.post("/api/plan/{plan_id}/approve", response_model=PlanStatusUpdateResponse)
async def approve_plan(plan_id: str, request: PlanStatusUpdateRequest) -> PlanStatusUpdateResponse:
    """
    Update plan status to approved or rejected.

    This endpoint is called by the TypeScript workflow after the user
    approves a plan via the UI. It persists the approval status to storage
    before the execution activity runs.

    Args:
        plan_id: The plan ID to update
        request: Contains status ("approved" or "rejected"), optional reviewer and comments
    """
    from datetime import datetime
    from plan_manager import PlanManager

    print(f"[Plan Approval] Updating plan {plan_id} to status: {request.status}")

    try:
        plan_manager = PlanManager(str(PLANS_DIR))
        plan = plan_manager.load_plan(plan_id)

        if not plan:
            return PlanStatusUpdateResponse(
                success=False,
                plan_id=plan_id,
                status="",
                error=f"Plan {plan_id} not found",
            )

        # Update the plan status
        plan.status = request.status
        if request.status == "approved":
            plan.approved_at = datetime.now().isoformat()

        # Save the updated plan
        plan_manager.save_plan()

        print(f"[Plan Approval] Plan {plan_id} status updated to {request.status}")

        return PlanStatusUpdateResponse(
            success=True,
            plan_id=plan_id,
            status=request.status,
        )

    except FileNotFoundError:
        return PlanStatusUpdateResponse(
            success=False,
            plan_id=plan_id,
            status="",
            error=f"Plan file {plan_id} not found",
        )
    except Exception as e:
        print(f"[Plan Approval] Error: {e}")
        import traceback
        traceback.print_exc()
        return PlanStatusUpdateResponse(
            success=False,
            plan_id=plan_id,
            status="",
            error=f"Failed to update plan status: {str(e)}",
        )


@app.post("/api/workflow/{instance_id}/approve", response_model=WorkflowApprovalResponse)
async def approve_workflow_plan(
    instance_id: str,
    request: WorkflowApprovalRequest,
) -> WorkflowApprovalResponse:
    """
    Raise approval event to resume a workflow waiting for plan approval.

    This endpoint uses Dapr's native external event mechanism to resume
    a workflow that is paused at wait_for_external_event. The workflow
    will continue with execution if approved, or return a rejection result.

    Args:
        instance_id: The Dapr workflow instance ID
        request: Contains plan_id, approved status, optional reviewer and reason
    """
    from durable_agent import get_workflow_client

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

        # The event name matches what the workflow is waiting for
        event_name = f"plan_approval_{request.plan_id}"
        event_data = {
            "approved": request.approved,
            "reviewer": request.reviewer,
            "reason": request.reason,
        }

        # Raise the event to resume the workflow
        client.raise_workflow_event(
            instance_id=instance_id,
            event_name=event_name,
            data=event_data,
        )

        print(f"[Workflow Approval] Event raised: {event_name}")

        return WorkflowApprovalResponse(
            success=True,
            instance_id=instance_id,
            plan_id=request.plan_id,
        )

    except Exception as e:
        print(f"[Workflow Approval] Error: {e}")
        import traceback
        traceback.print_exc()
        return WorkflowApprovalResponse(
            success=False,
            instance_id=instance_id,
            plan_id=request.plan_id,
            error=f"Failed to raise approval event: {str(e)}",
        )


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status query."""
    success: bool
    instance_id: str
    runtime_status: str | None = None
    custom_status: dict | None = None
    created_at: str | None = None
    last_updated_at: str | None = None
    error: str | None = None


@app.get("/api/workflow/{instance_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(instance_id: str) -> WorkflowStatusResponse:
    """
    Get the current status of a Dapr workflow.

    Returns:
    - runtime_status: PENDING, RUNNING, COMPLETED, FAILED, SUSPENDED, TERMINATED
    - custom_status: Application-defined status with phase, progress, message

    This provides deterministic workflow state instead of inferring from events.
    """
    import json

    try:
        client = get_workflow_client()

        if client is None:
            return WorkflowStatusResponse(
                success=False,
                instance_id=instance_id,
                error="Dapr workflow client not available",
            )

        # Get workflow state from Dapr
        state = client.get_workflow_state(instance_id=instance_id)

        if state is None:
            return WorkflowStatusResponse(
                success=False,
                instance_id=instance_id,
                error=f"Workflow {instance_id} not found",
            )

        # Parse custom status from properties
        custom_status = None
        if hasattr(state, 'properties') and state.properties:
            custom_status_str = state.properties.get('dapr.workflow.custom_status')
            if custom_status_str:
                try:
                    custom_status = json.loads(custom_status_str)
                except json.JSONDecodeError:
                    custom_status = {"raw": custom_status_str}

        # Get runtime status name
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
        import traceback
        traceback.print_exc()
        return WorkflowStatusResponse(
            success=False,
            instance_id=instance_id,
            error=f"Failed to get workflow status: {str(e)}",
        )


@app.post("/api/durable/plan-and-execute", response_model=DurablePlanResponse)
async def create_plan_and_execute_endpoint(request: DurablePlanRequest) -> DurablePlanResponse:
    """
    Start a combined planning and execution workflow.

    This endpoint starts a workflow that:
    1. Explores the codebase
    2. Creates an implementation plan
    3. Waits for approval via external event
    4. Executes the plan when approved

    Returns the workflow instance ID. Use /api/workflow/{instance_id}/approve
    to approve or reject the plan once it's created.
    """
    from durable_agent import get_workflow_client, planning_and_execution_workflow

    print(f"[Durable Plan+Execute] Starting combined workflow for {request.cwd}")
    print(f"[Durable Plan+Execute] Prompt: {request.prompt[:100]}...")

    try:
        # Verify the directory exists
        cwd_path = Path(request.cwd)
        if not cwd_path.exists():
            return DurablePlanResponse(
                success=False,
                error=f"Directory not found: {request.cwd}",
            )

        client = get_workflow_client()

        if client is None:
            return DurablePlanResponse(
                success=False,
                error="Dapr workflow client not available",
            )

        from datetime import datetime
        import json

        instance_id = request.session_id or f"plan-exec-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        workflow_input = json.dumps({
            "cwd": request.cwd,
            "feature_request": request.prompt,
            "plans_dir": str(PLANS_DIR),
        })

        # Start the combined workflow
        client.schedule_new_workflow(
            workflow=planning_and_execution_workflow,
            input=workflow_input,
            instance_id=instance_id,
        )

        print(f"[Durable Plan+Execute] Workflow started: {instance_id}")

        return DurablePlanResponse(
            success=True,
            plan_id=instance_id,  # Use instance_id as reference
            title="Workflow started",
            summary=f"Combined planning and execution workflow started. Use /api/workflow/{instance_id}/approve to approve the plan.",
            status="pending",
            durable_execution=True,
        )

    except Exception as e:
        print(f"[Durable Plan+Execute] Error: {e}")
        import traceback
        traceback.print_exc()
        return DurablePlanResponse(
            success=False,
            error=f"Failed to start workflow: {str(e)}",
        )


@app.post("/api/workflows", response_model=WorkflowStartResponse)
async def start_workflow_endpoint(request: WorkflowStartRequest) -> WorkflowStartResponse:
    """
    Start a workflow (Next.js compatible endpoint).

    This is the SINGLE ORCHESTRATOR endpoint for starting workflows.
    The workflow handles all steps internally:
    0. Clone repository (if repository info provided)
    1. Explore the codebase
    2. Create an implementation plan
    3. Wait for approval (external event)
    4. Execute the approved plan

    The clone happens INSIDE the workflow for durability - if the workflow
    restarts, it can resume from the clone step.
    """
    from durable_agent import get_workflow_client, planning_and_execution_workflow
    from datetime import datetime
    import json

    print(f"[Workflows] Starting workflow")
    print(f"[Workflows] Prompt: {request.prompt[:100]}...")
    if request.sessionId:
        print(f"[Workflows] Session ID: {request.sessionId}")

    try:
        # Get workflow client
        client = get_workflow_client()

        if client is None:
            return WorkflowStartResponse(
                success=False,
                error="Dapr workflow client not available. Ensure Dapr sidecar is running.",
            )

        # Generate instance ID
        instance_id = request.sessionId or f"workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Build workflow input - repository info is passed to workflow for durable cloning
        workflow_input_data: dict[str, Any] = {
            "feature_request": request.prompt,
            "plans_dir": str(PLANS_DIR),
            "auto_approve": request.options.autoApprove if request.options else False,
            "workflow_id": instance_id,  # For streaming events
        }

        # Include repository info for workflow to clone (durable operation)
        if request.options and request.options.targetRepository:
            repo = request.options.targetRepository
            print(f"[Workflows] Target repository: {repo.owner}/{repo.repo}@{repo.branch}")
            has_token = repo.token is not None
            print(f"[Workflows] Auth token provided: {has_token}")
            workflow_input_data["repository"] = {
                "owner": repo.owner,
                "repo": repo.repo,
                "branch": repo.branch,
                "token": repo.token,  # Pass token for authenticated clone
            }
            # Don't set cwd - workflow will set it after clone
        elif request.options and request.options.workingDirectory:
            # Use provided working directory if no repo specified
            cwd = request.options.workingDirectory
            cwd_path = Path(cwd)
            if not cwd_path.exists():
                return WorkflowStartResponse(
                    success=False,
                    error=f"Working directory not found: {cwd}",
                )
            workflow_input_data["cwd"] = cwd
        else:
            # Default to workspace
            workflow_input_data["cwd"] = str(WORKSPACE_DIR)

        workflow_input = json.dumps(workflow_input_data)

        # Start the workflow
        client.schedule_new_workflow(
            workflow=planning_and_execution_workflow,
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    durable_status = get_durable_agent_status()
    return {
        "status": "healthy",
        "dapr_available": DAPR_AVAILABLE,
        "dapr_workflow_available": durable_status["dapr_workflow_available"],
        "claude_sdk_available": durable_status["claude_sdk_available"],
        "phoenix_available": PHOENIX_AVAILABLE,
        "phoenix_endpoint": PHOENIX_ENDPOINT if PHOENIX_AVAILABLE else None,
        "workspace": str(WORKSPACE_DIR),
        "plans_dir": str(PLANS_DIR),
        "agent_registry": {
            "app_id": AGENT_APP_ID,
            "team_name": AGENT_TEAM_NAME,
            "capabilities": AGENT_CAPABILITIES,
        },
    }


@app.get("/")
async def root():
    """Root endpoint."""
    durable_available = is_durable_agents_available()
    return {
        "service": "planner-agent-workflow-service",
        "version": "1.0.0",
        "agent_type": "general-purpose",
        "durable_agents_available": durable_available,
        "endpoints": [
            "/api/clone",
            "/api/plan",
            "/api/execute",
            "/api/durable/plan",
            "/api/durable/execute",
            "/api/durable/plan-and-execute",
            "/api/plan/{plan_id}/approve",
            "/api/workflow/{instance_id}/approve",
            "/api/workflow/{instance_id}/status",
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
# Dapr Workflow Activities (for direct Dapr integration)
# =============================================================================

def http_clone_repository_activity(ctx: WorkflowActivityContext, input_data: dict) -> dict:
    """
    Dapr activity: Clone repository (HTTP-style, for backward compatibility).

    This activity can be called directly by Dapr workflows if needed.
    """
    owner = input_data["owner"]
    repo = input_data["repo"]
    branch = input_data.get("branch", "main")
    token = input_data.get("token")

    repo_url = f"https://github.com/{owner}/{repo}.git"
    clone_path = WORKSPACE_DIR / repo

    if clone_path.exists():
        shutil.rmtree(clone_path)

    cmd = [
        "git", "clone",
        "--depth", "1",
        "--single-branch",
        "--branch", branch,
    ]

    if token:
        auth_url = f"https://{token}@github.com/{owner}/{repo}.git"
        cmd.append(auth_url)
    else:
        cmd.append(repo_url)

    cmd.append(str(clone_path))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    return {
        "success": result.returncode == 0,
        "path": str(clone_path) if result.returncode == 0 else "",
        "error": result.stderr if result.returncode != 0 else None,
    }


def http_create_plan_activity(ctx: WorkflowActivityContext, input_data: dict) -> dict:
    """
    Dapr activity: Create plan using PlannerAgent (HTTP-style, for backward compatibility).

    This activity can be called directly by Dapr workflows if needed.
    The durable_agent.create_plan_activity uses ClaudeSDKClient with native tools instead.
    """
    cwd = input_data["cwd"]
    prompt = input_data["prompt"]

    agent = PlannerAgent(cwd=cwd, plans_dir=str(PLANS_DIR))

    # Run async planning in sync context
    asyncio.run(agent.run_planning_only(prompt))

    if not agent.plan_manager.current_plan:
        return {"plan": None, "error": "Failed to create plan"}

    plan = agent.plan_manager.current_plan
    return {
        "plan": {
            "id": plan.id,
            "title": plan.title,
            "summary": plan.summary,
            "steps": [
                {
                    "title": s.title,
                    "description": s.description,
                    "files_affected": s.files_affected,
                    "complexity": s.estimated_complexity,
                }
                for s in plan.steps
            ],
            "critical_files": plan.critical_files,
            "considerations": plan.considerations,
            "status": plan.status,
        }
    }


def http_execute_plan_activity(ctx: WorkflowActivityContext, input_data: dict) -> dict:
    """
    Dapr activity: Execute an approved plan (HTTP-style, for backward compatibility).

    This activity can be called directly by Dapr workflows if needed.
    The durable_agent.execute_plan_activity uses ClaudeSDKClient with native tools instead.
    """
    repo_path = input_data["repo_path"]
    plan_id = input_data["plan_id"]
    workflow_id = input_data["workflow_id"]

    agent = PlannerAgent(cwd=repo_path, plans_dir=str(PLANS_DIR))

    # Run async execution in sync context
    result = asyncio.run(agent.run_execution_only(plan_id, workflow_id))

    return {
        "success": result.get("success", False),
        "tasks_completed": result.get("tasks_completed", 0),
        "tasks_total": result.get("tasks_total", 0),
        "files_changed": result.get("files_changed", []),
        "error": result.get("error"),
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
