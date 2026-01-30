#!/usr/bin/env python3
"""DurableAgent with AgentRunner.serve() using DaprChatClient."""

import logging
import os
from typing import List

from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.agents.configs import AgentMemoryConfig, AgentStateConfig, AgentExecutionConfig
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.llm import OpenAIChatClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CWD = os.getenv("PLANNER_CWD", "/app/workspace")

# Task storage (reset per workflow via workflow input)
_task_counter = 0
_tasks: List[dict] = []


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


def create_agent():
    """Create a fresh DurableAgent instance."""
    # Use OpenAIChatClient - documented default for dapr-agents
    # OPENAI_API_KEY is provided via Azure Key Vault ExternalSecret
    llm = OpenAIChatClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    )

    agent = DurableAgent(
        name="PlannerAgent",
        role="Task Creator",
        goal="Create implementation tasks using the create_task tool",
        instructions=[
            "For each task needed, call create_task with subject and description",
            "Use blocked_by to set task dependencies (pass task IDs as strings)",
            "After creating all tasks, call get_tasks_json to return the task list",
        ],
        tools=[create_task, list_tasks, get_tasks_json],
        llm=llm,
        # Execution config
        execution=AgentExecutionConfig(
            tool_choice="auto",
            max_iterations=15,
        ),
        # State: workflow activity checkpoints (enables replay on failure)
        state=AgentStateConfig(
            store=StateStoreService(store_name="statestore")
        ),
        # Memory: conversation history persistence
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="memory-state",
                session_id="planner-session"
            )
        ),
    )
    return agent


def main():
    port = int(os.getenv("PORT", "8000"))

    # Create FastAPI app with health endpoint FIRST
    app = FastAPI(
        title="Planner DurableAgent",
        description="DurableAgent for software engineering planning with workflow durability",
    )

    @app.get("/health")
    async def health():
        """Health check endpoint for Kubernetes."""
        return {"status": "healthy"}

    # Create the DurableAgent
    agent = create_agent()

    # Create runner and serve - this registers workflows and adds /run endpoint
    # Since we pass our app, it won't auto-run uvicorn
    runner = AgentRunner(timeout_in_seconds=600)
    runner.serve(
        agent,
        app=app,  # Pass our app with /health already added
        port=port,
        expose_entry=True,
        entry_path="/run",
        status_path="/run/{instance_id}",
    )

    # Run uvicorn ourselves
    logger.info(f"Starting DurableAgent server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
