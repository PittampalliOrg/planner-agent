#!/usr/bin/env python3
"""DurableAgent with AgentRunner.serve() using DaprChatClient."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from dapr.clients import DaprClient

from dapr_agents import DurableAgent, tool
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.agents.configs import AgentMemoryConfig, AgentStateConfig, AgentExecutionConfig
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.llm import OpenAIChatClient

from dapr_config import initialize_config_and_secrets, get_config, get_secret_value, is_dapr_enabled

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pub/sub configuration for ai-chatbot integration
# These are initialized after config provider setup
PUBSUB_NAME = "pubsub"  # Updated after init
PUBSUB_TOPIC = "workflow.stream"  # Updated after init
AGENT_ID = "planner-dapr-agent"

# Workflow index configuration (for ai-chatbot listing)
# Uses the same state store format as ai-chatbot's workflow-patterns index
# Must use ai-chatbot-statestore to write to the same Redis as ai-chatbot
WORKFLOW_INDEX_STORE = "ai-chatbot-statestore"  # Updated after init
WORKFLOW_INDEX_KEY = "workflow-patterns-index"
WORKFLOW_KEY_PREFIX = "workflow-pattern-"

DEFAULT_CWD = os.getenv("PLANNER_CWD", "/app/workspace")

# Task storage (reset per workflow via workflow input)
_task_counter = 0
_tasks: List[dict] = []


def register_workflow_in_index(
    workflow_id: str,
    workflow_name: str,
    message: str,
) -> bool:
    """Register a workflow in the workflow-patterns index for ai-chatbot listing."""
    now = datetime.now(timezone.utc).isoformat()

    entry = {
        "instanceId": workflow_id,
        "workflowName": workflow_name,
        "workflowType": "orchestrator",  # Use orchestrator type for agent workflows
        "status": "running",
        "input": {"message": message},
        "createdAt": now,
        "updatedAt": now,
    }

    try:
        with DaprClient() as client:
            # Save the workflow entry
            client.save_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
                value=json.dumps(entry),
            )

            # Get current index
            index_data = client.get_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=WORKFLOW_INDEX_KEY,
            )

            if index_data.data:
                try:
                    ids = json.loads(index_data.data.decode('utf-8'))
                except:
                    ids = []
            else:
                ids = []

            # Add to front of index if not already present
            if workflow_id not in ids:
                ids.insert(0, workflow_id)
                # Keep index manageable
                if len(ids) > 1000:
                    ids = ids[:1000]

                client.save_state(
                    store_name=WORKFLOW_INDEX_STORE,
                    key=WORKFLOW_INDEX_KEY,
                    value=json.dumps(ids),
                )

        logger.info(f"Registered workflow {workflow_id} in index")
        return True
    except Exception as e:
        logger.warning(f"Failed to register workflow in index: {e}")
        return False


def update_workflow_index_status(
    workflow_id: str,
    status: str,
    output: Optional[dict] = None,
    error: Optional[str] = None,
) -> bool:
    """Update workflow status in the index."""
    now = datetime.now(timezone.utc).isoformat()

    try:
        with DaprClient() as client:
            # Get existing entry
            entry_data = client.get_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
            )

            if entry_data.data:
                entry = json.loads(entry_data.data.decode('utf-8'))
            else:
                logger.warning(f"Workflow {workflow_id} not found in index")
                return False

            # Update entry
            entry["status"] = status
            entry["updatedAt"] = now
            if output:
                entry["output"] = output
            if error:
                entry["error"] = error
            if status in ("completed", "failed", "terminated"):
                entry["completedAt"] = now

            # Save updated entry
            client.save_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
                value=json.dumps(entry),
            )

        logger.info(f"Updated workflow {workflow_id} status to {status} in index")
        return True
    except Exception as e:
        logger.warning(f"Failed to update workflow index status: {e}")
        return False


def publish_workflow_event(
    workflow_id: str,
    event_type: str,
    data: dict,
    task_id: Optional[str] = None,
) -> bool:
    """Publish a workflow event to the Dapr pub/sub topic for ai-chatbot.

    Event types: initial, task_progress, execution_started, execution_completed, execution_failed
    """
    event = {
        "id": f"dapr-agent-{workflow_id}-{uuid.uuid4().hex[:8]}",
        "type": event_type,
        "workflowId": workflow_id,
        "agentId": AGENT_ID,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if task_id:
        event["taskId"] = task_id

    try:
        with DaprClient() as client:
            client.publish_event(
                pubsub_name=PUBSUB_NAME,
                topic_name=PUBSUB_TOPIC,
                data=json.dumps(event),
                data_content_type="application/json",
            )
        logger.info(f"Published {event_type} event for workflow {workflow_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to publish {event_type} event: {e}")
        return False


async def monitor_workflow_completion(
    workflow_id: str,
    message: str,
    poll_interval: float = 2.0,
    max_polls: int = 300,  # 10 minutes max
):
    """Background task to monitor workflow completion and publish events."""
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval)
        try:
            with DaprClient() as client:
                # Query workflow status via Dapr workflow API
                response = client.get_workflow(
                    instance_id=workflow_id,
                    workflow_component="dapr",
                )
                # Handle runtime_status - can be enum or string depending on SDK version
                if hasattr(response.runtime_status, 'name'):
                    status = response.runtime_status.name
                else:
                    status = str(response.runtime_status) if response.runtime_status else "UNKNOWN"

                # Normalize status string
                status = status.upper().replace("ORCHESTRATION_STATUS_", "")

                if status == "COMPLETED":
                    # Update index status
                    update_workflow_index_status(
                        workflow_id=workflow_id,
                        status="completed",
                        output={"message": message, "tasks": _tasks},
                    )
                    # Publish event for streaming
                    publish_workflow_event(
                        workflow_id=workflow_id,
                        event_type="execution_completed",
                        data={
                            "status": "completed",
                            "progress": 100,
                            "metadata": {"message": message, "tasks": _tasks},
                        },
                    )
                    logger.info(f"Workflow {workflow_id} completed")
                    return
                elif status in ("FAILED", "TERMINATED"):
                    # Update index status
                    update_workflow_index_status(
                        workflow_id=workflow_id,
                        status="failed",
                        error=f"Workflow {status.lower()}",
                    )
                    # Publish event for streaming
                    publish_workflow_event(
                        workflow_id=workflow_id,
                        event_type="execution_failed",
                        data={"error": f"Workflow {status.lower()}"},
                    )
                    logger.warning(f"Workflow {workflow_id} {status.lower()}")
                    return
        except Exception as e:
            logger.warning(f"Error polling workflow status: {e}")
            continue

    # Timeout - update index and publish failure
    update_workflow_index_status(
        workflow_id=workflow_id,
        status="failed",
        error="Workflow monitoring timed out",
    )
    publish_workflow_event(
        workflow_id=workflow_id,
        event_type="execution_failed",
        data={"error": "Workflow monitoring timed out"},
    )
    logger.warning(f"Workflow {workflow_id} monitoring timed out")


# Tool models
class TaskInput(BaseModel):
    subject: str = Field(description="Task title")
    description: str = Field(description="Task details")
    blocked_by: List[str] = Field(default_factory=list)


# Tools
@tool(args_model=TaskInput)
def create_task(subject: str, description: str, blocked_by: List[str] = None) -> dict:
    """Create a planning task."""
    global _task_counter, _tasks
    _task_counter += 1
    task = {
        "id": str(_task_counter),
        "subject": subject,
        "description": description,
        "status": "pending",
        "blockedBy": blocked_by or [],
        "blocks": [],
    }
    _tasks.append(task)
    for dep_id in (blocked_by or []):
        for t in _tasks:
            if t["id"] == dep_id:
                t["blocks"].append(str(_task_counter))
    logger.info(f"Created task {_task_counter}: {subject}")
    return {"id": str(_task_counter), "subject": subject, "status": "pending"}


@tool
def list_tasks() -> str:
    """List all tasks."""
    if not _tasks:
        return "No tasks."
    return "\n".join(f"[{t['id']}] {t['subject']}" for t in _tasks)


@tool
def list_directory(path: str = ".") -> dict:
    """List workspace files."""
    import glob
    full_path = os.path.join(DEFAULT_CWD, path)
    items = glob.glob(os.path.join(full_path, "*"))
    files = [os.path.relpath(p, DEFAULT_CWD) for p in items if os.path.isfile(p)]
    dirs = [os.path.relpath(p, DEFAULT_CWD) for p in items if os.path.isdir(p)]
    return {"files": files[:50], "directories": dirs[:20], "count": len(items)}


@tool
def read_file(file_path: str) -> dict:
    """Read a file."""
    full_path = os.path.join(DEFAULT_CWD, file_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return {"content": f.read()[:10000], "exists": True}
    return {"content": "", "exists": False}


@tool
def get_tasks_json() -> dict:
    """Get all tasks as JSON for the workflow response."""
    return {"tasks": _tasks, "count": len(_tasks)}


def create_agent(session_id: str = None):
    """Create a fresh DurableAgent instance with ReAct-like execution config."""
    # Use OpenAIChatClient - documented default for dapr-agents
    # OPENAI_API_KEY is provided via Azure Key Vault ExternalSecret
    llm = OpenAIChatClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    )

    agent = DurableAgent(
        name=f"Planner{uuid.uuid4().hex[:6]}",  # Unique name to avoid memory pollution
        role="Software Implementation Planner",
        goal="Create detailed implementation plans by calling the create_task tool for each task",
        instructions=[
            "You are a task planning assistant. Your job is to create implementation tasks using the create_task tool.",
            "IMPORTANT: You MUST use the create_task tool to create tasks. Do NOT just describe tasks in text.",
            "For each planning request, break it down into 3-8 specific implementation tasks.",
            "For EACH task, call create_task with: subject (brief title), description (detailed steps), blocked_by (list of task IDs that must complete first).",
            "Use blocked_by to define task dependencies (e.g., blocked_by=['1'] means task 1 must complete first).",
            "After creating ALL tasks, call get_tasks_json once to return the complete task list.",
            "Only after calling get_tasks_json, provide a brief summary of the plan you created.",
        ],
        tools=[create_task, list_tasks, get_tasks_json],
        llm=llm,
        # Execution config: force tool usage for multi-step execution
        execution=AgentExecutionConfig(
            max_iterations=15,  # Allow up to 15 reasoning loops
            tool_choice="required",  # Force tool usage for multi-step execution
        ),
        # State: workflow activity checkpoints (enables replay on failure)
        state=AgentStateConfig(
            store=StateStoreService(store_name="statestore")
        ),
    )
    return agent


class RunRequest(BaseModel):
    """Request model for workflow invocation."""
    message: str = Field(description="The planning request message")


async def startup_config():
    """Initialize configuration and secrets from Dapr (or env vars as fallback)."""
    global PUBSUB_NAME, PUBSUB_TOPIC, WORKFLOW_INDEX_STORE

    await initialize_config_and_secrets()

    # Update configuration values from Dapr/env vars
    PUBSUB_NAME = get_config("PUBSUB_NAME", "pubsub")
    PUBSUB_TOPIC = get_config("PUBSUB_TOPIC", "workflow.stream")
    WORKFLOW_INDEX_STORE = get_config("WORKFLOW_INDEX_STORE", "ai-chatbot-statestore")

    if is_dapr_enabled():
        logger.info("[ConfigProvider] Using Dapr for configuration and secrets")
    else:
        logger.info("[ConfigProvider] Using environment variables (Dapr not available)")


def main():
    port = int(os.getenv("PORT", "8000"))

    # Create FastAPI app with health endpoint FIRST
    app = FastAPI(
        title="Planner DurableAgent",
        description="DurableAgent for software engineering planning with workflow durability",
    )

    @app.on_event("startup")
    async def on_startup():
        """Initialize config provider on startup."""
        await startup_config()

    @app.get("/health")
    async def health():
        """Health check endpoint for Kubernetes."""
        return {"status": "healthy"}

    # Create the DurableAgent with a fresh session for each startup
    startup_session = f"planner-startup-{uuid.uuid4().hex[:8]}"
    agent = create_agent(session_id=startup_session)

    # Create runner - but we'll wrap the endpoint ourselves
    runner = AgentRunner(timeout_in_seconds=600)

    # Let AgentRunner register its internal workflow machinery on a different path
    runner.serve(
        agent,
        app=app,
        port=port,
        expose_entry=True,
        entry_path="/_internal/run",  # Internal path for AgentRunner
        status_path="/_internal/run/{instance_id}",
    )

    @app.post("/run")
    async def run_with_events(request: RunRequest, background_tasks: BackgroundTasks):
        """Start a workflow and publish events to ai-chatbot."""
        global _task_counter, _tasks
        # Reset task storage for new workflow
        _task_counter = 0
        _tasks = []

        # Start workflow via Dapr
        workflow_id = uuid.uuid4().hex
        try:
            with DaprClient() as client:
                client.start_workflow(
                    workflow_component="dapr",
                    workflow_name="agent_workflow",
                    instance_id=workflow_id,
                    input={"message": request.message},
                )
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to start workflow: {e}"},
            )

        # Register workflow in index for ai-chatbot listing
        register_workflow_in_index(
            workflow_id=workflow_id,
            workflow_name="agent_workflow",
            message=request.message,
        )

        # Publish initial event for real-time streaming
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="initial",
            data={
                "status": "started",
                "metadata": {"message": request.message},
            },
        )

        # Publish execution started event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_started",
            data={
                "status": "planning",
                "progress": 10,
                "metadata": {"phase": "planning"},
            },
        )

        # Start background monitoring for completion
        background_tasks.add_task(
            monitor_workflow_completion,
            workflow_id,
            request.message,
        )

        return {
            "instance_id": workflow_id,
            "status_url": f"/run/{workflow_id}",
        }

    @app.get("/run/{instance_id}")
    async def get_workflow_status(instance_id: str):
        """Get workflow status."""
        try:
            with DaprClient() as client:
                response = client.get_workflow(
                    instance_id=instance_id,
                    workflow_component="dapr",
                )
                # Handle runtime_status - can be enum or string depending on SDK version
                if hasattr(response.runtime_status, 'name'):
                    status = response.runtime_status.name
                else:
                    status = str(response.runtime_status) if response.runtime_status else "UNKNOWN"
                return {
                    "instance_id": instance_id,
                    "name": response.workflow_name,
                    "runtime_status": status,
                    "created_at": response.created_at.isoformat() if response.created_at else None,
                    "last_updated_at": response.last_updated_at.isoformat() if response.last_updated_at else None,
                    "serialized_input": response.serialized_input,
                    "serialized_output": response.serialized_output,
                    "serialized_custom_status": response.serialized_custom_status,
                }
        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={"error": f"Workflow not found: {e}"},
            )

    # Run uvicorn ourselves
    logger.info(f"Starting DurableAgent server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
