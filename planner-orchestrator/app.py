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
from workflows.dapr_agent_workflow import dapr_agent_workflow
from activities.planning import run_planning
from activities.persist_tasks import persist_tasks
from activities.execution import run_execution
from activities.publish_event import publish_event
from activities.dapr_agent import run_dapr_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATESTORE_NAME = "statestore"
WORKFLOW_INDEX_KEY = "workflow_index"


def _get_workflow_index() -> list[str]:
    """Get list of all workflow IDs from index."""
    try:
        with DaprClient() as client:
            state = client.get_state(store_name=STATESTORE_NAME, key=WORKFLOW_INDEX_KEY)
            if state.data:
                return json.loads(state.data)
    except Exception as e:
        logger.warning(f"Failed to get workflow index: {e}")
    return []


def _add_to_workflow_index(workflow_id: str):
    """Add a workflow ID to the index."""
    try:
        index = _get_workflow_index()
        if workflow_id not in index:
            index.append(workflow_id)
            with DaprClient() as client:
                client.save_state(
                    store_name=STATESTORE_NAME,
                    key=WORKFLOW_INDEX_KEY,
                    value=json.dumps(index),
                )
    except Exception as e:
        logger.warning(f"Failed to update workflow index: {e}")


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
    wfr.register_activity(run_dapr_agent)

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
    feature_request: str | None = Field(default=None, description="Feature to plan and implement")
    prompt: str | None = Field(default=None, description="Alias for feature_request (chatbot compat)")
    cwd: str = Field(default="", description="Working directory for the agent")
    sessionId: str | None = Field(default=None, description="Chatbot session ID (optional)")
    options: dict | None = Field(default=None, description="Chatbot options (optional)")


class WorkflowStartResponse(BaseModel):
    workflow_id: str
    workflowId: str  # Alias for chatbot compatibility
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
    # Accept either feature_request or prompt (chatbot sends prompt)
    feature_request = request.feature_request or request.prompt
    if not feature_request:
        raise HTTPException(status_code=400, detail="feature_request or prompt is required")

    workflow_id = f"planner-{uuid.uuid4().hex[:12]}"

    workflow_input = {
        "feature_request": feature_request,
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
        _add_to_workflow_index(instance_id)
        return WorkflowStartResponse(workflow_id=instance_id, workflowId=instance_id)
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DaprAgentWorkflowRequest(BaseModel):
    """Request to start a DurableAgent workflow."""
    prompt: str = Field(..., description="Prompt to send to the DurableAgent")
    cwd: str = Field(default="", description="Working directory for the agent")


@app.post("/api/workflows/dapr-agent", response_model=WorkflowStartResponse)
def start_dapr_agent_workflow(request: DaprAgentWorkflowRequest):
    """Start a new DurableAgent workflow.

    This workflow invokes the planner-dapr-agent service which runs a DurableAgent
    with Anthropic Claude for planning tasks.
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    workflow_id = f"dapr-agent-{uuid.uuid4().hex[:12]}"

    workflow_input = {
        "prompt": request.prompt,
        "cwd": request.cwd,
    }

    try:
        client = DaprWorkflowClient()
        instance_id = client.schedule_new_workflow(
            workflow=dapr_agent_workflow,
            input=workflow_input,
            instance_id=workflow_id,
        )
        logger.info(f"DurableAgent workflow started: {instance_id}")
        _add_to_workflow_index(instance_id)
        return WorkflowStartResponse(workflow_id=instance_id, workflowId=instance_id)
    except Exception as e:
        logger.error(f"Failed to start DurableAgent workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows")
def list_workflows():
    """List all workflows with their current status."""
    workflow_ids = _get_workflow_index()
    workflows = []

    client = DaprWorkflowClient()
    for wf_id in workflow_ids:
        try:
            state = client.get_workflow_state(instance_id=wf_id)
            if state is None:
                continue

            runtime_status = "UNKNOWN"
            if hasattr(state, "runtime_status") and state.runtime_status:
                runtime_status = (
                    state.runtime_status.name
                    if hasattr(state.runtime_status, "name")
                    else str(state.runtime_status)
                )

            # Extract timestamps from Dapr workflow state
            # Ensure timestamps are in ISO format with UTC timezone suffix
            created_at = None
            updated_at = None
            if hasattr(state, "created_at") and state.created_at:
                ts = state.created_at.isoformat() if hasattr(state.created_at, "isoformat") else str(state.created_at)
                # Add Z suffix if not present to indicate UTC
                created_at = ts if ts.endswith("Z") or "+" in ts else f"{ts}Z"
            if hasattr(state, "last_updated_at") and state.last_updated_at:
                ts = state.last_updated_at.isoformat() if hasattr(state.last_updated_at, "isoformat") else str(state.last_updated_at)
                updated_at = ts if ts.endswith("Z") or "+" in ts else f"{ts}Z"

            # Parse custom status for phase/message
            phase = None
            message = None
            if hasattr(state, "to_json"):
                state_dict = state.to_json()
                if isinstance(state_dict, dict):
                    # Also try to get timestamps from JSON if not available as attributes
                    if not created_at:
                        created_at = state_dict.get("created_at") or state_dict.get("createdAt")
                    if not updated_at:
                        updated_at = state_dict.get("last_updated_at") or state_dict.get("lastUpdatedAt")

                    custom_str = state_dict.get("serialized_custom_status")
                    if custom_str:
                        try:
                            parsed = json.loads(custom_str)
                            while isinstance(parsed, str):
                                parsed = json.loads(parsed)
                            if isinstance(parsed, dict):
                                phase = parsed.get("phase")
                                message = parsed.get("message")
                        except (json.JSONDecodeError, TypeError):
                            pass

            workflows.append({
                "workflow_id": wf_id,
                "runtime_status": runtime_status,
                "phase": phase,
                "message": message,
                "created_at": created_at,
                "updated_at": updated_at,
            })
        except Exception as e:
            logger.warning(f"Failed to get status for {wf_id}: {e}")

    return {"workflows": workflows, "total": len(workflows)}


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


# --- Singular path aliases (chatbot sends /api/workflow/ instead of /api/workflows/) ---

@app.get("/api/workflow/{workflow_id}/status")
def get_workflow_status_singular(workflow_id: str):
    """Alias: singular /api/workflow/ for chatbot compatibility."""
    return get_workflow_status(workflow_id)


@app.post("/api/workflow/{workflow_id}/approve")
def approve_workflow_singular(workflow_id: str, request: WorkflowApprovalRequest):
    """Alias: singular /api/workflow/ for chatbot compatibility."""
    return approve_workflow(workflow_id, request)


@app.get("/api/workflow/{workflow_id}/tasks")
def get_workflow_tasks_singular(workflow_id: str):
    """Alias: singular /api/workflow/ for chatbot compatibility."""
    return get_workflow_tasks(workflow_id)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "planner-orchestrator"}
