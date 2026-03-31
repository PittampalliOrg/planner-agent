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
import json
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
    DAPR_AVAILABLE = True
except ImportError:
    DAPR_AVAILABLE = False
    WorkflowActivityContext = None
    WorkflowRuntime = None

from plan_manager import PlanManager
from planner_agent import PlannerAgent


# =============================================================================
# Configuration
# =============================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
PLANS_DIR = Path(os.getenv("PLANS_DIR", "/plans"))


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


class PlanSummary(BaseModel):
    """Summary of a saved plan."""
    id: str
    title: str
    status: str
    created_at: str


class PlanListResponse(BaseModel):
    """Response model for listing saved plans."""
    plans: list[PlanSummary]


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

    # Start Dapr workflow runtime if available
    # Note: WorkflowRuntime.start() is synchronous (starts a background thread)
    if DAPR_AVAILABLE and WorkflowRuntime:
        try:
            _workflow_runtime = WorkflowRuntime()
            _workflow_runtime.register_activity(clone_repository_activity)
            _workflow_runtime.register_activity(create_plan_activity)
            _workflow_runtime.register_activity(execute_plan_activity)
            _workflow_runtime.start()  # Synchronous - do NOT await
            print("[Workflow Service] Dapr workflow runtime started")
        except Exception as e:
            print(f"[Workflow Service] Warning: Could not start Dapr workflow runtime: {e}")
            _workflow_runtime = None
    else:
        print("[Workflow Service] Dapr not available - running in HTTP-only mode")

    yield

    # Shutdown
    if _workflow_runtime is not None:
        try:
            _workflow_runtime.shutdown()
            print("[Workflow Service] Dapr workflow runtime shutdown complete")
        except Exception as e:
            print(f"[Workflow Service] Warning: Error during workflow runtime shutdown: {e}")


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


@app.get("/api/plans", response_model=PlanListResponse)
async def list_plans() -> PlanListResponse:
    """Return a summary list of all saved plans from PLANS_DIR."""
    pm = PlanManager(PLANS_DIR)
    plan_ids = pm.list_plans()

    summaries: list[PlanSummary] = []
    for plan_id in plan_ids:
        json_path = PLANS_DIR / f"{plan_id}.json"
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            continue
        summaries.append(PlanSummary(
            id=data.get("id", plan_id),
            title=data.get("title", ""),
            status=data.get("status", "draft"),
            created_at=data.get("created_at", ""),
        ))

    return PlanListResponse(plans=summaries)


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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dapr_available": DAPR_AVAILABLE,
        "workspace": str(WORKSPACE_DIR),
        "plans_dir": str(PLANS_DIR),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "planner-agent-workflow-service",
        "version": "1.0.0",
        "endpoints": [
            "/api/clone",
            "/api/plan",
            "/api/plans",
            "/api/execute",
            "/health",
        ],
    }


# =============================================================================
# Dapr Workflow Activities (for direct Dapr integration)
# =============================================================================

def clone_repository_activity(ctx: WorkflowActivityContext, input_data: dict) -> dict:
    """
    Dapr activity: Clone repository.

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


def create_plan_activity(ctx: WorkflowActivityContext, input_data: dict) -> dict:
    """
    Dapr activity: Create plan using Claude SDK.

    This activity can be called directly by Dapr workflows if needed.
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


def execute_plan_activity(ctx: WorkflowActivityContext, input_data: dict) -> dict:
    """
    Dapr activity: Execute an approved plan.

    This activity can be called directly by Dapr workflows if needed.
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
