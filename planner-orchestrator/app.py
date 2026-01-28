"""FastAPI app for the planner orchestrator with Dapr workflow runtime lifecycle."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager

from dapr.clients import DaprClient
from dapr.ext.workflow import DaprWorkflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from workflows.planner_workflow import wfr, unified_planner_workflow
from activities.planning import run_planning
from activities.persist_tasks import persist_tasks
from activities.execution import run_execution
from activities.publish_event import publish_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATESTORE_NAME = "statestore"


# --- Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register activities and start/stop runtime.

    The workflow is already registered by the @wfr.workflow decorator at import time.
    Activities are plain functions and need explicit registration.
    """
    wfr.register_activity(run_planning)
    wfr.register_activity(persist_tasks)
    wfr.register_activity(run_execution)
    wfr.register_activity(publish_event)

    wfr.start()
    logger.info("Planner orchestrator workflow runtime started")
    yield
    wfr.shutdown()
    logger.info("Planner orchestrator workflow runtime stopped")


app = FastAPI(
    title="Planner Orchestrator",
    description="Dapr Workflow orchestrator for Claude Agent SDK planning and execution",
    lifespan=lifespan,
)


# --- Request / Response Models ---

class WorkflowStartRequest(BaseModel):
    feature_request: str = Field(..., description="Feature to plan and implement")
    cwd: str = Field(default="", description="Working directory for the agent")


class WorkflowStartResponse(BaseModel):
    workflow_id: str
    status: str = "started"


class WorkflowApprovalRequest(BaseModel):
    approved: bool = Field(..., description="Whether the plan is approved")
    reason: str = Field(default="", description="Optional reason for rejection")


class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    runtime_status: str
    phase: str | None = None
    progress: int | None = None
    message: str | None = None
    output: dict | None = None


# --- Endpoints ---

@app.post("/api/workflows", response_model=WorkflowStartResponse)
def start_workflow(request: WorkflowStartRequest):
    """Start a new planner workflow."""
    workflow_id = f"planner-{uuid.uuid4().hex[:12]}"

    workflow_input = {
        "feature_request": request.feature_request,
        "cwd": request.cwd,
    }

    try:
        client = DaprWorkflowClient()
        instance_id = client.schedule_new_workflow(
            workflow=unified_planner_workflow,
            input=workflow_input,
            instance_id=workflow_id,
        )
        logger.info(f"Workflow started: {instance_id}")
        return WorkflowStartResponse(workflow_id=instance_id)
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/{workflow_id}/approve")
def approve_workflow(workflow_id: str, request: WorkflowApprovalRequest):
    """Raise the approval event for a workflow waiting at the approval gate."""
    try:
        client = DaprWorkflowClient()
        client.raise_workflow_event(
            instance_id=workflow_id,
            event_name=f"plan_approval_{workflow_id}",
            data=request.model_dump(),
        )
        logger.info(f"Approval event raised for {workflow_id}: approved={request.approved}")
        return {"status": "event_raised", "workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Failed to raise approval event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/{workflow_id}/status", response_model=WorkflowStatusResponse)
def get_workflow_status(workflow_id: str):
    """Get the current status and phase of a workflow."""
    try:
        client = DaprWorkflowClient()
        state = client.get_workflow_state(instance_id=workflow_id)

        if state is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Get runtime status
        runtime_status = "UNKNOWN"
        if hasattr(state, "runtime_status") and state.runtime_status:
            runtime_status = (
                state.runtime_status.name
                if hasattr(state.runtime_status, "name")
                else str(state.runtime_status)
            )

        # Parse custom status via to_json() (serialized_custom_status may be double-encoded)
        phase = None
        progress = None
        message = None
        output = None

        if hasattr(state, "to_json"):
            state_dict = state.to_json()
            if isinstance(state_dict, dict):
                # Custom status
                custom_str = state_dict.get("serialized_custom_status")
                if custom_str:
                    try:
                        parsed = json.loads(custom_str)
                        while isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        if isinstance(parsed, dict):
                            phase = parsed.get("phase")
                            progress = parsed.get("progress")
                            message = parsed.get("message")
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Output
                output_str = state_dict.get("serialized_output")
                if output_str:
                    try:
                        parsed = json.loads(output_str)
                        while isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        output = parsed if isinstance(parsed, dict) else {"raw": str(parsed)}
                    except (json.JSONDecodeError, TypeError):
                        output = {"raw": str(output_str)}

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            runtime_status=runtime_status,
            phase=phase,
            progress=progress,
            message=message,
            output=output,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/{workflow_id}/tasks")
def get_workflow_tasks(workflow_id: str):
    """Get tasks for a workflow from the Dapr statestore."""
    try:
        with DaprClient() as client:
            state = client.get_state(store_name=STATESTORE_NAME, key=f"tasks:{workflow_id}")

        if not state.data:
            return {"workflow_id": workflow_id, "tasks": [], "count": 0}

        tasks = json.loads(state.data)
        return {"workflow_id": workflow_id, "tasks": tasks, "count": len(tasks)}
    except Exception as e:
        logger.error(f"Failed to get tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "planner-orchestrator"}
