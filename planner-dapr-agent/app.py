#!/usr/bin/env python3
"""Planner Agent using OpenAI Agents SDK with Dapr durability.

This module provides the FastAPI application that exposes the planner agent.

Architecture:
- agent.py: Pure OpenAI Agents SDK agent definition (@function_tool, Agent)
- workflow_context.py: WorkflowContext for activity tracking and state management
- durable_runner.py: Interceptor-based durability (PR #827 pattern)
- app.py: FastAPI endpoints and workflow orchestration

Execution modes:
- durable=false: Standard Runner.run() with basic activity tracking
- durable=true: WorkflowContext-wrapped execution with full durability
"""

import asyncio
import functools
import glob
import inspect
import json
import logging
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import threading
from queue import Queue, Empty
from dotenv import load_dotenv
from dapr.clients import DaprClient

from agents import Agent, Runner, function_tool
from agents.run import RunHooks
from agents.tool import FunctionTool

# Try to import official DaprSession for conversation memory (OpenAI Agents SDK extension)
try:
    from agents.extensions.memory.dapr_session import DaprSession as OfficialDaprSession
    OFFICIAL_DAPR_SESSION_AVAILABLE = True
except ImportError:
    OFFICIAL_DAPR_SESSION_AVAILABLE = False
    OfficialDaprSession = None

# Try to import DaprOpenAIRunner for true workflow durability
try:
    from dapr_openai_runner import (
        DaprOpenAIRunner,
        build_runtime,
        get_or_create_runner,
        run_durable_agent,
        is_dapr_workflow_available,
        WorkflowResult,
    )
    DAPR_WORKFLOW_RUNNER_AVAILABLE = is_dapr_workflow_available()
except ImportError:
    DAPR_WORKFLOW_RUNNER_AVAILABLE = False
    DaprOpenAIRunner = None
    build_runtime = None
    get_or_create_runner = None
    run_durable_agent = None
    is_dapr_workflow_available = None
    WorkflowResult = None

from dapr_config import initialize_config_and_secrets, get_config, get_secret_value, is_dapr_enabled

# Import agent definition (clean OpenAI SDK pattern)
from agent import create_planner_agent

# Import multi-step workflow
from workflow_agent import run_workflow as run_multi_step_workflow, Plan, ExecutionResult, TestResult

# Import Dapr multi-step workflow (proper Dapr workflow with activities)
try:
    from dapr_multi_step_workflow import get_workflow_runtime, multi_step_workflow
    from dapr.ext.workflow import DaprWorkflowClient
    DAPR_MULTI_STEP_WORKFLOW_AVAILABLE = True
except ImportError as e:
    DAPR_MULTI_STEP_WORKFLOW_AVAILABLE = False
    get_workflow_runtime = None
    multi_step_workflow = None
    DaprWorkflowClient = None
    logger.warning(f"Dapr multi-step workflow not available: {e}")

# Import WorkflowContext for activity tracking (renamed from dapr_session)
from workflow_context import WorkflowContext, get_session_state

# Import interceptor framework (for backward compatibility)
from durable_runner import (
    WorkflowExecutionContext,
    DurableRunResult,
    get_workflow_context,
    set_workflow_context,
    DurableAgentRunner,
    ActivityTrackingInterceptor,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pub/sub configuration for ai-chatbot integration
PUBSUB_NAME = "pubsub"
PUBSUB_TOPIC = "workflow.stream"
AGENT_ID = "planner-dapr-agent"

# Event buffer for replaying events on SSE connect
# This solves the race condition where events are published before SSE stream connects
_event_buffer: dict[str, list[dict]] = {}
_event_buffer_lock = threading.Lock()
MAX_EVENTS_PER_WORKFLOW = 200  # Limit buffer size

def _buffer_event(workflow_id: str, event: dict) -> None:
    """Buffer an event for later replay on SSE connect."""
    with _event_buffer_lock:
        if workflow_id not in _event_buffer:
            _event_buffer[workflow_id] = []
        _event_buffer[workflow_id].append(event)
        # Trim if too large
        if len(_event_buffer[workflow_id]) > MAX_EVENTS_PER_WORKFLOW:
            _event_buffer[workflow_id] = _event_buffer[workflow_id][-MAX_EVENTS_PER_WORKFLOW:]

def _get_buffered_events(workflow_id: str) -> list[dict]:
    """Get buffered events for a workflow."""
    with _event_buffer_lock:
        return list(_event_buffer.get(workflow_id, []))

def _clear_event_buffer(workflow_id: str) -> None:
    """Clear buffered events for a workflow (after completion)."""
    with _event_buffer_lock:
        if workflow_id in _event_buffer:
            del _event_buffer[workflow_id]

# Workflow index configuration (for ai-chatbot listing)
WORKFLOW_INDEX_STORE = "ai-chatbot-statestore"
WORKFLOW_INDEX_KEY = "workflow-patterns-index"
WORKFLOW_KEY_PREFIX = "workflow-pattern-"

# Static default for PLANNER_CWD - actual value retrieved via get_workspace_dir()
_DEFAULT_CWD = "/app/workspace"


def get_workspace_dir() -> str:
    """Get the workspace directory from Dapr config or environment.

    Returns the configured workspace directory, falling back to the default.
    This function lazily retrieves the value after Dapr config is initialized.
    """
    return get_config("PLANNER_CWD", _DEFAULT_CWD)


# Backward compatibility alias
DEFAULT_CWD = _DEFAULT_CWD


# ============================================================================
# Activity Tracking Helpers (using interceptor framework)
# ============================================================================

def track_activity(
    name: str,
    status: str,
    input_data: Optional[dict] = None,
    output_data: Optional[dict] = None,
) -> None:
    """Track tool/activity execution for ai-chatbot visualization.

    This is a helper function that works with the WorkflowExecutionContext
    from durable_runner.py. Activity tracking is now primarily handled by
    the ActivityTrackingInterceptor, but this function is kept for backward
    compatibility and for tracking non-tool activities (like agent:run).

    Also publishes events to pub/sub for real-time SSE streaming.
    """
    ctx = get_workflow_context()
    if not ctx:
        logger.debug(f"No workflow context for activity: {name}")
        return

    now = datetime.now(timezone.utc).isoformat()

    # Find existing activity or create new one
    existing = None
    for activity in ctx.activities:
        if activity["activityName"] == name and activity["status"] == "running":
            existing = activity
            break

    if existing and status in ("completed", "failed"):
        # Update existing activity
        existing["status"] = status
        existing["endTime"] = now
        if existing.get("startTime"):
            start = datetime.fromisoformat(existing["startTime"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(now.replace("Z", "+00:00"))
            existing["durationMs"] = int((end - start).total_seconds() * 1000)
        if output_data:
            existing["output"] = output_data
        # Only persist on completion to avoid race conditions
        update_workflow_activities(ctx.workflow_id, ctx.activities)

        # Publish completion event for real-time streaming
        _publish_activity_event(ctx.workflow_id, name, status, input_data, output_data, existing.get("durationMs"))
    else:
        # Create new activity
        activity = {
            "activityName": name,
            "status": status,
            "startTime": now,
        }
        if input_data:
            activity["input"] = input_data
        if output_data:
            activity["output"] = output_data
        ctx.activities.append(activity)
        # Don't persist "running" status - wait for completion to avoid race conditions

        # Publish start event for real-time streaming
        _publish_activity_event(ctx.workflow_id, name, status, input_data, output_data)


def _publish_activity_event(
    workflow_id: str,
    name: str,
    status: str,
    input_data: Optional[dict] = None,
    output_data: Optional[dict] = None,
    duration_ms: Optional[int] = None,
) -> None:
    """Publish activity event to pub/sub for real-time SSE streaming."""
    # Determine event type based on activity name and status
    if name.startswith("llm:"):
        if status == "running":
            event_type = "llm_start"
            data = {"llm_call": name, "input": input_data}
        else:
            event_type = "llm_end"
            data = {"llm_call": name, "output": output_data, "durationMs": duration_ms}
    elif name.startswith("tool:") or name in ("create_task", "list_tasks", "get_tasks_json", "read_file", "write_file", "list_directory", "run_shell_command", "search_code"):
        tool_name = name.replace("tool:", "") if name.startswith("tool:") else name
        if status == "running":
            event_type = "tool_call"
            data = {"toolName": tool_name, "toolInput": input_data}
        else:
            event_type = "tool_result"
            data = {"toolName": tool_name, "toolOutput": output_data, "status": status, "durationMs": duration_ms}
    elif name.startswith("agent:"):
        if status == "running":
            event_type = "agent_started"
            data = {"agent": name, "input": input_data}
        else:
            event_type = "agent_completed"
            data = {"agent": name, "output": output_data, "status": status, "durationMs": duration_ms}
    else:
        # Generic activity
        if status == "running":
            event_type = "activity_started"
        else:
            event_type = "activity_completed"
        data = {"activity": name, "status": status, "input": input_data, "output": output_data, "durationMs": duration_ms}

    # Use local import to avoid circular dependency
    try:
        publish_workflow_event(workflow_id, event_type, data)
    except Exception as e:
        logger.debug(f"Could not publish activity event: {e}")


def reset_workflow_context(workflow_id: str) -> None:
    """Initialize workflow context for a new workflow run.

    Creates a new WorkflowExecutionContext and sets it as the current context.
    """
    ctx = WorkflowExecutionContext(
        workflow_id=workflow_id,
        metadata={"workflow_id": workflow_id},
    )
    set_workflow_context(ctx)


# ============================================================================
# Legacy Interceptor Support
# ============================================================================
#
# The ActivityTrackingInterceptor is now imported from durable_runner.py.
# This provides backward compatibility while using the new interceptor framework.
#
# For new code, prefer using DurableAgentRunner which automatically wraps tools
# with the interceptor chain. The @tracked_tool decorator below provides a
# simpler alternative for individual tools.

# Global interceptor instance using the new framework (can be configured/replaced)
# ActivityTrackingInterceptor is imported from durable_runner.py
_tool_interceptor = ActivityTrackingInterceptor(update_callback=None)


def _serialize_arg(arg: Any, max_length: int = 500) -> Any:
    """Serialize argument for activity tracking, truncating large values."""
    if isinstance(arg, str):
        return arg[:max_length] if len(arg) > max_length else arg
    if isinstance(arg, (int, float, bool, type(None))):
        return arg
    if isinstance(arg, (list, tuple)):
        return [_serialize_arg(x, max_length) for x in arg[:10]]
    if isinstance(arg, dict):
        return {k: _serialize_arg(v, max_length) for k, v in list(arg.items())[:10]}
    return str(arg)[:200]


def _serialize_output(result: Any) -> dict:
    """Serialize function output for activity tracking."""
    if isinstance(result, dict):
        output = {}
        for key in list(result.keys())[:10]:
            val = result[key]
            if isinstance(val, str) and len(val) > 100:
                output[key] = f"<{len(val)} chars>"
            elif isinstance(val, list):
                output[key] = f"<{len(val)} items>"
            elif isinstance(val, (int, float, bool, type(None))):
                output[key] = val
            else:
                output[key] = str(val)[:100]
        return output
    if isinstance(result, str):
        return {"length": len(result), "preview": result[:100] if len(result) > 100 else result}
    return {"type": type(result).__name__}


def tracked_tool(func: Callable) -> FunctionTool:
    """Interceptor decorator that wraps a function with automatic activity tracking.

    This decorator implements the interceptor pattern from dapr/python-sdk PR #827,
    adapted for the OpenAI Agents SDK. It:

    1. Creates an ExecuteToolRequest with input metadata (like PR #827's request objects)
    2. Passes the request through the interceptor chain
    3. The interceptor handles tracking start/completion/failure

    Usage:
        @tracked_tool
        def my_tool(arg1: str, arg2: int) -> dict:
            '''Tool description for the LLM.'''
            return {"result": arg1 * arg2}

    The decorator applies @function_tool internally, so you only need @tracked_tool.

    To customize interception, replace the global _tool_interceptor.

    Note: For new code, prefer using DurableAgentRunner which provides full
    durability support. This decorator provides a simpler alternative for
    backward compatibility.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Build input data from function signature (like PR #827's request.input)
        input_data = {"args": args, "kwargs": kwargs}
        try:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            serialized = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    serialized[param_names[i]] = _serialize_arg(arg)
            serialized.update({k: _serialize_arg(v) for k, v in kwargs.items()})
            input_data["serialized"] = serialized
        except Exception:
            pass

        # Get workflow context for metadata (like PR #827's metadata envelope)
        ctx = get_workflow_context()

        # Create request object (mirrors PR #827's ExecuteActivityRequest)
        # Using the new ExecuteToolRequest structure with call_id and workflow_id
        request = ExecuteToolRequest(
            tool_name=func.__name__,
            input=input_data,
            metadata=ctx.metadata if ctx else None,
            call_id=str(uuid.uuid4()),
            workflow_id=ctx.workflow_id if ctx else None,
        )

        # Execute through interceptor (the actual function call)
        def execute_func(req: ExecuteToolRequest) -> Any:
            return func(*args, **kwargs)

        return _tool_interceptor.execute_tool(request, execute_func)

    # Apply function_tool decorator to the wrapper
    return function_tool(wrapper)


class ActivityTrackingHooks(RunHooks):
    """RunHooks implementation that tracks LLM calls and agent lifecycle.

    This follows PR #827's interceptor pattern by using the workflow context
    for state management instead of global variables.

    OpenAI Trace Schema Compatibility:
    - Captures GenerationSpanData equivalent for each LLM call
    - Tracks token usage and aggregates totals
    - Creates span hierarchy with agent as parent

    Note: Tool tracking is handled by the ActivityTrackingInterceptor via
    the @tracked_tool decorator. These hooks capture LLM calls which provide
    additional visibility into the agent's reasoning process.
    """

    def on_agent_start(self, context, agent) -> None:
        """Called when agent execution starts."""
        ctx = get_workflow_context()

        # Set workflow name from agent
        if ctx:
            ctx.workflow_name = agent.name if hasattr(agent, 'name') else "Planner"

        track_activity(
            name="agent:run",
            status="running",
            input_data={
                "agent_name": agent.name if hasattr(agent, 'name') else "Planner",
                "model": agent.model if hasattr(agent, 'model') else "unknown",
                # OpenAI AgentSpanData fields
                "tools": [getattr(t, 'name', str(t)) for t in (agent.tools or [])[:10]],
            },
        )
        logger.info(f"Agent started: {agent.name if hasattr(agent, 'name') else 'Planner'}")

    def on_agent_end(self, context, agent, output) -> None:
        """Called when agent execution completes."""
        ctx = get_workflow_context()

        # Summarize output
        output_summary = {}
        if output:
            if isinstance(output, str):
                output_summary = {"output_length": len(output), "output_preview": output[:200]}
            elif isinstance(output, dict):
                output_summary = {"output_keys": list(output.keys())[:10]}
            else:
                output_summary = {"output_type": type(output).__name__}

        # Include aggregated usage if available
        if ctx and ctx.usage:
            output_summary["total_usage"] = ctx.usage

        track_activity(
            name="agent:run",
            status="completed",
            output_data=output_summary,
        )
        logger.info("Agent completed")

    def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        """Called when an LLM call starts."""
        ctx = get_workflow_context()
        if ctx:
            ctx.llm_call_count += 1
            llm_call_num = ctx.llm_call_count
        else:
            llm_call_num = 1

        # Count message types
        message_count = len(input_items) if input_items else 0

        track_activity(
            name=f"llm:call_{llm_call_num}",
            status="running",
            input_data={
                "model": agent.model if hasattr(agent, 'model') else "unknown",
                "message_count": message_count,
                "has_system_prompt": bool(system_prompt),
                # OpenAI GenerationSpanData fields
                "span_type": "generation",
            },
        )
        logger.debug(f"LLM call {llm_call_num} started with {message_count} messages")

    def on_llm_end(self, context, agent, response) -> None:
        """Called when an LLM call completes."""
        ctx = get_workflow_context()
        llm_call_num = ctx.llm_call_count if ctx else 1

        # Extract response metadata
        output_data = {
            "response_type": type(response).__name__,
            "span_type": "generation",
        }

        # Try to get token usage if available
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', None) or getattr(response.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(response.usage, 'output_tokens', None) or getattr(response.usage, 'completion_tokens', 0) or 0

            output_data["usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

            # Aggregate usage in context
            if ctx:
                ctx.add_usage(input_tokens=input_tokens, output_tokens=output_tokens)

        track_activity(
            name=f"llm:call_{llm_call_num}",
            status="completed",
            output_data=output_data,
        )
        logger.debug(f"LLM call {llm_call_num} completed")


# ============================================================================
# Tools with @function_tool decorator
# ============================================================================


@tracked_tool
def create_task(subject: str, description: str, blocked_by: Optional[List[str]] = None) -> dict:
    """Create a planning task with dependencies.

    Args:
        subject: Brief task title
        description: Detailed task description
        blocked_by: List of task IDs that must complete before this task

    Returns:
        Created task info with id, subject, and status
    """
    ctx = get_workflow_context()
    if not ctx:
        return {"error": "No workflow context", "id": "0", "subject": subject, "status": "error"}

    ctx.task_counter += 1
    task_id = str(ctx.task_counter)
    blocked_by = blocked_by or []

    task = {
        "id": task_id,
        "subject": subject,
        "description": description,
        "status": "pending",
        "blockedBy": blocked_by,
        "blocks": [],
    }
    ctx.tasks.append(task)

    # Update blocks for dependent tasks
    for dep_id in blocked_by:
        for t in ctx.tasks:
            if t["id"] == dep_id:
                t["blocks"].append(task_id)

    logger.info(f"Created task {task_id}: {subject}")
    return {"id": task_id, "subject": subject, "status": "pending"}


@tracked_tool
def list_tasks() -> str:
    """List all created tasks.

    Returns:
        Formatted string of all tasks with their IDs and subjects
    """
    ctx = get_workflow_context()
    tasks = ctx.tasks if ctx else []

    if not tasks:
        return "No tasks created yet."

    return "\n".join(f"[{t['id']}] {t['subject']}" for t in tasks)


@tracked_tool
def get_tasks_json() -> dict:
    """Get all tasks as JSON for the workflow response.

    Returns:
        Dictionary with tasks array and count
    """
    ctx = get_workflow_context()
    tasks = ctx.tasks if ctx else []
    return {"tasks": tasks, "count": len(tasks)}


@tracked_tool
def read_file(file_path: str) -> dict:
    """Read file contents from workspace.

    Args:
        file_path: Path relative to workspace directory

    Returns:
        Dictionary with content and exists flag
    """
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, file_path)

    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()[:10000]  # Limit to 10KB
        return {"content": content, "exists": True}
    else:
        return {"content": "", "exists": False}


@tracked_tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the workspace.

    Args:
        file_path: Path relative to workspace directory
        content: Content to write

    Returns:
        Success message or error
    """
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, file_path)

    # Create parent directories if needed
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, 'w') as f:
        f.write(content)

    return f"Successfully wrote {len(content)} bytes to {file_path}"


@tracked_tool
def list_directory(path: str = ".") -> dict:
    """List files and directories in workspace.

    Args:
        path: Path relative to workspace directory (default: root)

    Returns:
        Dictionary with files, directories, and count
    """
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, path)

    items = glob.glob(os.path.join(full_path, "*"))
    files = [os.path.relpath(p, workspace) for p in items if os.path.isfile(p)]
    dirs = [os.path.relpath(p, workspace) for p in items if os.path.isdir(p)]

    return {"files": files[:50], "directories": dirs[:20], "count": len(items)}


@tracked_tool
def run_shell_command(command: str) -> str:
    """Execute a shell command in the workspace.

    Args:
        command: Shell command to execute

    Returns:
        Command output or error message
    """
    workspace = get_workspace_dir()
    result = subprocess.run(
        command,
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,  # 1 minute timeout
    )
    output = result.stdout + result.stderr
    output = output[:5000]  # Limit output size

    return output if output else f"Command completed with exit code {result.returncode}"


@tracked_tool
def search_code(pattern: str, path: str = ".") -> str:
    """Search for a pattern in code files using grep.

    Args:
        pattern: Regex pattern to search for
        path: Path relative to workspace (default: root)

    Returns:
        Matching lines or message if no matches
    """
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, path)

    result = subprocess.run(
        ["grep", "-r", "-n", "--include=*.py", "--include=*.js", "--include=*.ts",
         "--include=*.json", "--include=*.yaml", "--include=*.yml", "--include=*.md",
         pattern, full_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout[:5000]  # Limit output

    if output:
        return output
    else:
        return f"No matches found for pattern: {pattern}"


# ============================================================================
# Agent Configuration
# ============================================================================


def create_agent() -> Agent:
    """Create the planner agent with all tools."""
    return Agent(
        name="Planner",
        instructions="""You are a software planning assistant with tools to explore codebases and create implementation plans.

Available tools:
- create_task: Create planning tasks with dependencies (use blocked_by to define task order)
- list_tasks: Show all created tasks
- get_tasks_json: Get tasks as JSON (call this after creating all tasks)
- read_file: Read file contents
- write_file: Write/create files
- list_directory: Explore project structure
- run_shell_command: Execute shell commands
- search_code: Search codebase for patterns

Guidelines:
1. Start by exploring the codebase with list_directory and read_file to understand the structure
2. Use search_code to find relevant patterns and implementations
3. Break down the request into 3-8 specific implementation tasks
4. For EACH task, call create_task with:
   - subject: Brief title
   - description: Detailed implementation steps
   - blocked_by: List of task IDs that must complete first
5. After creating ALL tasks, call get_tasks_json to return the complete plan
6. Provide a brief summary of the plan you created

Task Dependencies:
- Use blocked_by to define execution order
- blocked_by=['1'] means task 1 must complete before this task
- Leave blocked_by empty for tasks that can start immediately
- Build a proper DAG of dependencies for complex plans""",
        model=get_config("OPENAI_MODEL", "gpt-4o"),
        tools=[
            create_task,
            list_tasks,
            get_tasks_json,
            read_file,
            write_file,
            list_directory,
            run_shell_command,
            search_code,
        ],
    )


# ============================================================================
# Workflow Index Operations (for ai-chatbot)
# ============================================================================


def register_workflow_in_index(
    workflow_id: str,
    workflow_name: str,
    message: str,
) -> bool:
    """Register a workflow in the workflow-patterns index for ai-chatbot listing.

    Uses the ai-chatbot WorkflowEntry format for compatibility with the UI.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Use ai-chatbot compatible WorkflowEntry format
    entry = {
        # Core identifiers (ai-chatbot uses both 'id' and 'instanceId')
        "id": workflow_id,
        "instanceId": workflow_id,
        "workflowName": workflow_name,
        "workflowType": "orchestrator",
        "appId": AGENT_ID,
        "status": "RUNNING",  # Uppercase for ai-chatbot compatibility

        # Request info (ai-chatbot format)
        "request": {
            "prompt": message,
            "submittedAt": now,
        },

        # Legacy format for backwards compatibility
        "input": {"message": message},

        # Execution state (ai-chatbot format)
        "execution": {
            "currentTaskIndex": 0,
            "completedTasks": [],
            "failedTasks": [],
            "skippedTasks": [],
            "logs": [],
        },

        # Activities for our agent's detailed tracking
        "activities": [],

        # Timestamps
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


def update_workflow_status(
    workflow_id: str,
    status: str,
    output: Optional[dict] = None,
    error: Optional[str] = None,
) -> bool:
    """Update workflow status in the index.

    Transforms data to ai-chatbot WorkflowEntry format for UI compatibility.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        with DaprClient() as client:
            entry_data = client.get_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
            )

            if entry_data.data:
                entry = json.loads(entry_data.data.decode('utf-8'))
            else:
                logger.warning(f"Workflow {workflow_id} not found in index")
                return False

            # Map status to uppercase for ai-chatbot compatibility
            status_map = {
                "running": "RUNNING",
                "completed": "COMPLETED",
                "failed": "FAILED",
                "terminated": "TERMINATED",
                "cancelled": "CANCELLED",
            }
            entry["status"] = status_map.get(status.lower(), status.upper())
            entry["updatedAt"] = now

            if output:
                # Store raw output for our detailed tracking
                entry["output"] = output

                # Transform to ai-chatbot plan format if tasks are present
                tasks = output.get("tasks", [])
                if tasks:
                    entry["plan"] = {
                        "id": workflow_id,
                        "title": output.get("plan", "")[:100] if output.get("plan") else "Planning Tasks",
                        "summary": output.get("plan", "")[:500] if output.get("plan") else "",
                        "tasks": [
                            {
                                "id": task.get("id", str(i)),
                                "title": task.get("subject", task.get("title", f"Task {i+1}")),
                                "description": task.get("description", ""),
                                "status": task.get("status", "pending"),
                                "dependsOn": task.get("blockedBy", []),
                            }
                            for i, task in enumerate(tasks)
                        ],
                    }

                # Add trace metadata if present
                if output.get("trace"):
                    entry["trace"] = output["trace"]

                # Add usage if present
                if output.get("usage"):
                    entry["usage"] = output["usage"]

            if error:
                entry["error"] = error

            if status.lower() in ("completed", "failed", "terminated"):
                entry["completedAt"] = now

            # Convert activities to execution.logs format for ai-chatbot
            activities = entry.get("activities", [])
            if activities:
                logs = []
                for activity in activities:
                    # Map activity status to execution log event
                    activity_status = activity.get("status", "")
                    if activity_status == "running":
                        event = "started"
                    elif activity_status == "completed":
                        event = "completed"
                    elif activity_status == "failed":
                        event = "failed"
                    else:
                        event = "started"

                    logs.append({
                        "timestamp": activity.get("startTime", now),
                        "taskId": activity.get("activityName", "unknown"),
                        "event": event,
                        "message": f"{event.capitalize()}: {activity.get('activityName', 'Activity')}",
                        "details": activity.get("output") or activity.get("input"),
                    })

                entry["execution"] = {
                    "currentTaskIndex": len([a for a in activities if a.get("status") == "completed"]),
                    "completedTasks": [a.get("activityName") for a in activities if a.get("status") == "completed"],
                    "failedTasks": [a.get("activityName") for a in activities if a.get("status") == "failed"],
                    "skippedTasks": [],
                    "logs": logs,
                }

            client.save_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
                value=_safe_json_dumps(entry),
            )

        logger.info(f"Updated workflow {workflow_id} status to {status}")
        return True
    except Exception as e:
        logger.warning(f"Failed to update workflow status: {e}")
        return False


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles complex SDK types."""

    def default(self, obj):
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(obj)
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            result = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            result['__type__'] = type(obj).__name__
            return result
        # Fallback
        return str(obj)


def _safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, cls=SafeJSONEncoder)
    except Exception as e:
        logger.warning(f"JSON serialization fallback: {e}")
        # Fallback: convert any remaining non-serializable objects to strings
        def sanitize(o, depth=0):
            if depth > 5:
                return str(o)
            if o is None or isinstance(o, (bool, int, float, str)):
                return o
            if isinstance(o, (list, tuple)):
                return [sanitize(x, depth + 1) for x in o[:100]]
            if isinstance(o, dict):
                return {str(k): sanitize(v, depth + 1) for k, v in list(o.items())[:50]}
            return str(o)
        return json.dumps(sanitize(obj))


def update_workflow_activities(workflow_id: str, activities: List[dict]) -> bool:
    """Update activities array for workflow visualization.

    Also updates execution.logs for ai-chatbot compatibility.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        with DaprClient() as client:
            entry_data = client.get_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
            )

            if entry_data.data:
                entry = json.loads(entry_data.data.decode('utf-8'))
            else:
                return False

            # Store raw activities for detailed tracking
            entry["activities"] = activities
            entry["updatedAt"] = now

            # Convert to ai-chatbot execution.logs format
            logs = []
            for activity in activities:
                activity_status = activity.get("status", "")
                if activity_status == "running":
                    event = "started"
                elif activity_status == "completed":
                    event = "completed"
                elif activity_status == "failed":
                    event = "failed"
                else:
                    event = "started"

                logs.append({
                    "timestamp": activity.get("startTime", now),
                    "taskId": activity.get("activityName", "unknown"),
                    "event": event,
                    "message": f"{event.capitalize()}: {activity.get('activityName', 'Activity')}",
                    "details": {
                        "input": activity.get("input"),
                        "output": activity.get("output"),
                        "durationMs": activity.get("durationMs"),
                        "span_id": activity.get("span_id"),
                    },
                })

            entry["execution"] = {
                "currentTaskIndex": len([a for a in activities if a.get("status") == "completed"]),
                "completedTasks": [a.get("activityName") for a in activities if a.get("status") == "completed"],
                "failedTasks": [a.get("activityName") for a in activities if a.get("status") == "failed"],
                "skippedTasks": [],
                "logs": logs,
            }

            client.save_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
                value=_safe_json_dumps(entry),
            )

        return True
    except Exception as e:
        logger.warning(f"Failed to update workflow activities: {e}")
        return False


def get_workflow_from_index(workflow_id: str) -> Optional[dict]:
    """Get workflow entry from index."""
    try:
        with DaprClient() as client:
            entry_data = client.get_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=f"{WORKFLOW_KEY_PREFIX}{workflow_id}",
            )
            if entry_data.data:
                return json.loads(entry_data.data.decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to get workflow from index: {e}")
    return None


def get_workflows_from_index(limit: int = 20) -> List[dict]:
    """Get list of workflows from index."""
    try:
        with DaprClient() as client:
            index_data = client.get_state(
                store_name=WORKFLOW_INDEX_STORE,
                key=WORKFLOW_INDEX_KEY,
            )

            if not index_data.data:
                return []

            ids = json.loads(index_data.data.decode('utf-8'))[:limit]
            workflows = []

            for wf_id in ids:
                entry_data = client.get_state(
                    store_name=WORKFLOW_INDEX_STORE,
                    key=f"{WORKFLOW_KEY_PREFIX}{wf_id}",
                )
                if entry_data.data:
                    workflows.append(json.loads(entry_data.data.decode('utf-8')))

            return workflows
    except Exception as e:
        logger.warning(f"Failed to get workflows from index: {e}")
        return []


# ============================================================================
# Pub/Sub Events
# ============================================================================


def publish_workflow_event(
    workflow_id: str,
    event_type: str,
    data: dict,
    task_id: Optional[str] = None,
) -> bool:
    """Publish a workflow event to the Dapr pub/sub topic for ai-chatbot.

    Also buffers the event for replay when SSE streams connect (solves race condition).
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

    # Buffer the event for SSE replay (don't buffer heartbeats)
    if event_type not in ("heartbeat", "ping"):
        _buffer_event(workflow_id, event)

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


# ============================================================================
# Workflow Execution
# ============================================================================


async def execute_workflow(workflow_id: str, task: str) -> dict:
    """Execute the planning workflow using OpenAI Agents SDK.

    This follows PR #827's pattern of setting up context before execution
    and cleaning up after completion.
    """
    # Initialize context for this workflow (like PR #827's context setup)
    reset_workflow_context(workflow_id)
    ctx = get_workflow_context()

    # Track initial activity
    track_activity("agent:planning", "running", {"task": task[:200]})

    try:
        # Create agent
        agent = create_agent()

        # Get OpenAI API key
        api_key = get_secret_value("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        # Set environment variable for OpenAI client
        os.environ["OPENAI_API_KEY"] = api_key

        # Run agent
        # Note: RunHooks disabled due to async compatibility issue with current SDK version
        # Tool tracking is handled by @tracked_tool interceptor decorator
        logger.info(f"Starting Runner.run for workflow {workflow_id}")
        result = await Runner.run(
            starting_agent=agent,
            input=task,
        )
        logger.info(f"Runner.run completed for workflow {workflow_id}")

        # Get final output
        final_output = result.final_output

        # Get tasks from context (not globals)
        tasks = ctx.tasks if ctx else []

        # Get message count (RunResult might have new_items or raw_responses)
        message_count = len(result.new_items) if hasattr(result, 'new_items') else 0

        # Track completion with usage data
        track_activity("agent:planning", "completed", output_data={
            "messages": message_count,
            "tasks_created": len(tasks),
            "usage": ctx.usage if ctx else {},
        })

        # Get trace metadata for OpenAI-compatible output
        trace_metadata = ctx.get_trace_metadata() if ctx else {}

        # Update workflow status with usage data
        update_workflow_status(
            workflow_id=workflow_id,
            status="completed",
            output={
                "plan": final_output,
                "tasks": tasks,
                "message_count": message_count,
                "usage": ctx.usage if ctx else {},
                "trace": trace_metadata,
            },
        )

        # Publish completion event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_completed",
            data={
                "status": "completed",
                "progress": 100,
                "metadata": {"tasks": tasks, "plan": final_output},
                "usage": ctx.usage if ctx else {},
            },
        )

        logger.info(f"Workflow {workflow_id} completed with {len(tasks)} tasks")

        return {
            "status": "completed",
            "plan": final_output,
            "tasks": tasks,
            "usage": ctx.usage if ctx else {},
            "trace": trace_metadata,
        }

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")

        track_activity("agent:planning", "failed", output_data={"error": str(e)})

        update_workflow_status(
            workflow_id=workflow_id,
            status="failed",
            error=str(e),
        )

        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_failed",
            data={"error": str(e)},
        )

        return {
            "status": "failed",
            "error": str(e),
        }
    finally:
        # Clean up context (like PR #827's finally block)
        set_workflow_context(None)


async def execute_workflow_durable(workflow_id: str, task: str) -> dict:
    """Execute the planning workflow using DurableAgentRunner.

    This is an alternative implementation that uses the full interceptor framework
    from PR #827, providing:
    - Durable tool call recording (for replay/recovery)
    - Activity tracking (for ai-chatbot visualization)
    - Metadata envelope propagation

    The DurableAgentRunner wraps all tools with the interceptor chain automatically,
    so you don't need to use @tracked_tool on individual tools.

    Args:
        workflow_id: Unique identifier for this workflow
        task: The planning task/request

    Returns:
        Dictionary with status, plan, and tasks
    """
    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    # Set environment variable for OpenAI client
    os.environ["OPENAI_API_KEY"] = api_key

    # Track initial activity (outside of interceptor chain)
    track_activity("agent:planning", "running", {"task": task[:200]})

    try:
        # Create base agent (tools will be wrapped by DurableAgentRunner)
        # Note: We use non-tracked versions of tools here since the runner wraps them
        base_agent = Agent(
            name="Planner",
            instructions=create_agent().instructions,  # Reuse instructions from create_agent
            model=get_config("OPENAI_MODEL", "gpt-4o"),
            tools=[
                # These are the raw functions, not wrapped with @tracked_tool
                function_tool(create_task_impl),
                function_tool(list_tasks_impl),
                function_tool(get_tasks_json_impl),
                function_tool(read_file_impl),
                function_tool(write_file_impl),
                function_tool(list_directory_impl),
                function_tool(run_shell_command_impl),
                function_tool(search_code_impl),
            ],
        )

        # Create DurableAgentRunner with interceptor chain
        # Hooks are optional - they track LLM calls for usage statistics
        # If hooks cause issues, the runner works without them (tool tracking still works)
        try:
            hooks = ActivityTrackingHooks()
            logger.info("ActivityTrackingHooks created successfully")
        except Exception as e:
            logger.warning(f"Could not create hooks: {e}")
            hooks = None

        runner = DurableAgentRunner(
            agent=base_agent,
            state_store="statestore",
            activity_update_callback=update_workflow_activities,
            run_hooks=hooks,  # May be None if hooks creation failed
        )

        # Run with durability
        logger.info(f"Starting DurableAgentRunner for workflow {workflow_id}")
        durable_result = await runner.run(
            workflow_id=workflow_id,
            input=task,
            metadata={"source": "api", "version": "1.0"},
        )
        logger.info(f"DurableAgentRunner completed for workflow {workflow_id}")

        # Extract data from DurableRunResult (context data captured before cleanup)
        tasks = durable_result.tasks
        usage = durable_result.usage
        trace_metadata = durable_result.trace_metadata
        message_count = len(durable_result.new_items)

        # Track completion with usage data
        track_activity("agent:planning", "completed", output_data={
            "messages": message_count,
            "tasks_created": len(tasks),
            "usage": usage,
            "llm_calls": durable_result.llm_call_count,
        })

        # Update workflow status with usage data
        update_workflow_status(
            workflow_id=workflow_id,
            status="completed",
            output={
                "plan": durable_result.final_output,
                "tasks": tasks,
                "message_count": message_count,
                "usage": usage,
                "trace": trace_metadata,
            },
        )

        # Publish completion event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_completed",
            data={
                "status": "completed",
                "progress": 100,
                "metadata": {"tasks": tasks, "plan": durable_result.final_output},
                "usage": usage,
            },
        )

        logger.info(f"Workflow {workflow_id} completed with {len(tasks)} tasks")

        return {
            "status": "completed",
            "plan": durable_result.final_output,
            "tasks": tasks,
            "usage": usage,
            "trace": trace_metadata,
        }

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")

        track_activity("agent:planning", "failed", output_data={"error": str(e)})

        update_workflow_status(
            workflow_id=workflow_id,
            status="failed",
            error=str(e),
        )

        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_failed",
            data={"error": str(e)},
        )

        return {
            "status": "failed",
            "error": str(e),
        }


async def execute_workflow_session(workflow_id: str, task: str) -> dict:
    """Execute the planning workflow using WorkflowContext (clean pattern).

    This execution path provides activity tracking and state persistence
    using WorkflowContext, following OpenAI Agents SDK patterns:

    1. Agent defined with standard @function_tool and Agent() in agent.py
    2. WorkflowContext provides activity tracking and state management
    3. Runner.run() executes the agent normally

    Args:
        workflow_id: Unique identifier for this workflow
        task: The planning task/request

    Returns:
        Dictionary with status, plan, tasks, usage, and trace metadata
    """
    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        # Create the planner agent (clean SDK pattern from agent.py)
        agent = create_planner_agent(model=get_config("OPENAI_MODEL", "gpt-4o"))

        # Execute with WorkflowContext for durability
        async with WorkflowContext(
            workflow_id=workflow_id,
            state_store="statestore",
            workflow_name="planner_workflow",
            metadata={"source": "api", "version": "2.0"},
            activity_callback=update_workflow_activities,
        ) as session:
            # Track agent start
            session.track_activity(
                name="agent:planning",
                status="running",
                input_data={"task": task[:200], "model": agent.model},
            )

            logger.info(f"Starting Runner.run for workflow {workflow_id}")

            # Run the agent using standard SDK pattern
            result = await Runner.run(
                starting_agent=agent,
                input=task,
            )

            logger.info(f"Runner.run completed for workflow {workflow_id}")

            # Track agent completion
            session.track_activity(
                name="agent:planning",
                status="completed",
                output_data={
                    "tasks_created": len(session.tasks),
                    "usage": session.usage,
                },
            )

        # Session context has exited - data is captured in session object
        tasks = session.tasks
        usage = session.usage
        trace_metadata = session.trace_metadata
        message_count = len(result.new_items) if hasattr(result, 'new_items') else 0

        # Update workflow status
        update_workflow_status(
            workflow_id=workflow_id,
            status="completed",
            output={
                "plan": result.final_output,
                "tasks": tasks,
                "message_count": message_count,
                "usage": usage,
                "trace": trace_metadata,
            },
        )

        # Publish completion event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_completed",
            data={
                "status": "completed",
                "progress": 100,
                "metadata": {"tasks": tasks, "plan": result.final_output},
                "usage": usage,
            },
        )

        logger.info(f"Workflow {workflow_id} completed with {len(tasks)} tasks (session mode)")

        return {
            "status": "completed",
            "plan": result.final_output,
            "tasks": tasks,
            "usage": usage,
            "trace": trace_metadata,
        }

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")

        update_workflow_status(
            workflow_id=workflow_id,
            status="failed",
            error=str(e),
        )

        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_failed",
            data={"error": str(e)},
        )

        return {
            "status": "failed",
            "error": str(e),
        }


async def execute_workflow_v2(workflow_id: str, task: str) -> dict:
    """Execute workflow using official DaprSession with multi-turn support.

    This is the recommended execution path that uses:
    1. Official DaprSession from agents.extensions.memory for conversation memory
    2. WorkflowContext for activity tracking and workflow state
    3. Standard Runner.run() with session parameter

    The official DaprSession stores conversation history (messages) in Dapr state,
    enabling multi-turn conversations where the agent remembers previous context.

    Args:
        workflow_id: Unique identifier for this workflow
        task: The planning task/request

    Returns:
        Dictionary with status, plan, tasks, usage, and trace metadata
    """
    if not OFFICIAL_DAPR_SESSION_AVAILABLE:
        logger.warning("Official DaprSession not available, falling back to session mode")
        return await execute_workflow_session(workflow_id, task)

    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        # Create the planner agent (clean SDK pattern from agent.py)
        agent = create_planner_agent(model=get_config("OPENAI_MODEL", "gpt-4o"))

        # Create official DaprSession for conversation memory
        # This stores messages in Dapr state store for multi-turn support
        session = OfficialDaprSession.from_address(
            session_id=workflow_id,
            state_store_name="statestore",
        )

        # Execute with WorkflowContext for activity tracking
        async with WorkflowContext(
            workflow_id=workflow_id,
            state_store="statestore",
            workflow_name="planner_workflow",
            metadata={"source": "api", "version": "2.0", "mode": "v2"},
            activity_callback=update_workflow_activities,
        ) as ctx:
            # Track agent start
            ctx.track_activity(
                name="agent:planning",
                status="running",
                input_data={"task": task[:200], "model": agent.model, "session_mode": "v2"},
            )

            logger.info(f"Starting Runner.run for workflow {workflow_id} (v2 mode with official DaprSession)")

            # Run the agent with official DaprSession for conversation memory
            result = await Runner.run(
                starting_agent=agent,
                input=task,
                session=session,  # Official DaprSession for multi-turn support
            )

            logger.info(f"Runner.run completed for workflow {workflow_id}")

            # Track agent completion
            ctx.track_activity(
                name="agent:planning",
                status="completed",
                output_data={
                    "tasks_created": len(ctx.tasks),
                    "usage": ctx.usage,
                },
            )

        # Context has exited - data is captured in ctx object
        tasks = ctx.tasks
        usage = ctx.usage
        trace_metadata = ctx.trace_metadata
        message_count = len(result.new_items) if hasattr(result, 'new_items') else 0

        # Update workflow status
        update_workflow_status(
            workflow_id=workflow_id,
            status="completed",
            output={
                "plan": result.final_output,
                "tasks": tasks,
                "message_count": message_count,
                "usage": usage,
                "trace": trace_metadata,
                "mode": "v2",
            },
        )

        # Publish completion event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_completed",
            data={
                "status": "completed",
                "progress": 100,
                "metadata": {"tasks": tasks, "plan": result.final_output},
                "usage": usage,
            },
        )

        logger.info(f"Workflow {workflow_id} completed with {len(tasks)} tasks (v2 mode)")

        return {
            "status": "completed",
            "plan": result.final_output,
            "tasks": tasks,
            "usage": usage,
            "trace": trace_metadata,
            "mode": "v2",
        }

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")

        update_workflow_status(
            workflow_id=workflow_id,
            status="failed",
            error=str(e),
        )

        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_failed",
            data={"error": str(e)},
        )

        return {
            "status": "failed",
            "error": str(e),
        }


async def continue_workflow(workflow_id: str, message: str) -> dict:
    """Continue an existing workflow conversation.

    This uses the official DaprSession to maintain conversation context
    across multiple turns. The agent will have access to all previous
    messages and can build on prior work.

    Args:
        workflow_id: Existing workflow ID to continue
        message: Follow-up message/question

    Returns:
        Dictionary with status, response, and updated tasks
    """
    if not OFFICIAL_DAPR_SESSION_AVAILABLE:
        return {
            "status": "error",
            "error": "Official DaprSession not available. Install openai-agents[dapr]>=0.7.0",
        }

    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        # Create the planner agent
        agent = create_planner_agent(model=get_config("OPENAI_MODEL", "gpt-4o"))

        # Reuse existing session (conversation history is persisted in DaprSession)
        session = OfficialDaprSession.from_address(
            session_id=workflow_id,
            state_store_name="statestore",
        )

        # Execute with WorkflowContext for activity tracking
        async with WorkflowContext(
            workflow_id=workflow_id,
            state_store="statestore",
            workflow_name="planner_workflow",
            metadata={"source": "api", "version": "2.0", "mode": "continue"},
            activity_callback=update_workflow_activities,
        ) as ctx:
            # Track continuation
            ctx.track_activity(
                name="agent:continue",
                status="running",
                input_data={"message": message[:200], "model": agent.model},
            )

            logger.info(f"Continuing workflow {workflow_id} with follow-up message")

            # Run agent with existing session (has previous context)
            result = await Runner.run(
                starting_agent=agent,
                input=message,
                session=session,  # Session has previous conversation history
            )

            logger.info(f"Continue completed for workflow {workflow_id}")

            # Track completion
            ctx.track_activity(
                name="agent:continue",
                status="completed",
                output_data={
                    "tasks_created": len(ctx.tasks),
                    "usage": ctx.usage,
                },
            )

        # Update workflow with new results
        update_workflow_status(
            workflow_id=workflow_id,
            status="completed",
            output={
                "response": result.final_output,
                "tasks": ctx.tasks,
                "usage": ctx.usage,
                "trace": ctx.trace_metadata,
                "mode": "continue",
            },
        )

        logger.info(f"Workflow {workflow_id} continuation completed")

        return {
            "status": "completed",
            "response": result.final_output,
            "tasks": ctx.tasks,
            "usage": ctx.usage,
            "trace": ctx.trace_metadata,
        }

    except Exception as e:
        logger.error(f"Workflow {workflow_id} continuation failed: {e}")

        return {
            "status": "failed",
            "error": str(e),
        }


async def execute_workflow_dapr(workflow_id: str, task: str) -> dict:
    """Execute workflow using DaprOpenAIRunner with true workflow durability.

    This execution path provides true Dapr workflow-level durability:
    1. Agent execution is wrapped as a Dapr workflow
    2. Tool execution becomes a durable activity
    3. Workflows survive crashes and restart from last completed activity

    This is the most durable execution mode, based on patterns from dapr-agents:
    - DurableAgent: @workflow_entry + ctx.call_activity()
    - AgentRunner: Workflow lifecycle management

    Args:
        workflow_id: Unique identifier for this workflow
        task: The planning task/request

    Returns:
        Dictionary with status, plan, tasks, usage, and trace metadata
    """
    if not DAPR_WORKFLOW_RUNNER_AVAILABLE:
        logger.warning("DaprOpenAIRunner not available, falling back to v2 mode")
        return await execute_workflow_v2(workflow_id, task)

    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        # Create the planner agent (clean SDK pattern from agent.py)
        agent = create_planner_agent(model=get_config("OPENAI_MODEL", "gpt-4o"))

        # Run with DaprOpenAIRunner for true workflow durability
        logger.info(f"Starting DaprOpenAIRunner for workflow {workflow_id}")

        result = await run_durable_agent(
            agent=agent,
            input=task,
            workflow_id=workflow_id,
            activity_callback=update_workflow_activities,
        )

        logger.info(f"DaprOpenAIRunner completed for workflow {workflow_id}")

        # Extract data from WorkflowResult
        if result.status in ("COMPLETED", "completed"):
            # Update workflow status
            update_workflow_status(
                workflow_id=workflow_id,
                status="completed",
                output={
                    "plan": result.output.get("output") if result.output else None,
                    "tasks": result.tasks,
                    "usage": result.usage,
                    "trace": result.trace_metadata,
                    "mode": "workflow",
                },
            )

            # Publish completion event
            publish_workflow_event(
                workflow_id=workflow_id,
                event_type="execution_completed",
                data={
                    "status": "completed",
                    "progress": 100,
                    "metadata": {
                        "tasks": result.tasks,
                        "plan": result.output.get("output") if result.output else None,
                    },
                    "usage": result.usage,
                },
            )

            logger.info(f"Workflow {workflow_id} completed with {len(result.tasks)} tasks (workflow mode)")

            return {
                "status": "completed",
                "plan": result.output.get("output") if result.output else None,
                "tasks": result.tasks,
                "usage": result.usage,
                "trace": result.trace_metadata,
                "mode": "workflow",
            }
        else:
            # Workflow failed or timed out
            error = result.error or f"Workflow ended with status: {result.status}"

            update_workflow_status(
                workflow_id=workflow_id,
                status="failed",
                error=error,
            )

            publish_workflow_event(
                workflow_id=workflow_id,
                event_type="execution_failed",
                data={"error": error},
            )

            return {
                "status": "failed",
                "error": error,
                "mode": "workflow",
            }

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")

        update_workflow_status(
            workflow_id=workflow_id,
            status="failed",
            error=str(e),
        )

        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_failed",
            data={"error": str(e)},
        )

        return {
            "status": "failed",
            "error": str(e),
            "mode": "workflow",
        }


# Tool implementation functions (for use with DurableAgentRunner - legacy)
# These are the raw implementations without @tracked_tool decoration.
# DurableAgentRunner wraps them automatically with the interceptor chain.

def create_task_impl(subject: str, description: str, blocked_by: Optional[List[str]] = None) -> dict:
    """Create a planning task with dependencies."""
    ctx = get_workflow_context()
    logger.info(f"create_task_impl called with subject={subject}, ctx exists={ctx is not None}, workflow_id={ctx.workflow_id if ctx else 'N/A'}")
    if not ctx:
        logger.warning("create_task_impl: No workflow context! Check contextvars propagation.")
        return {"error": "No workflow context", "id": "0", "subject": subject, "status": "error"}

    ctx.task_counter += 1
    task_id = str(ctx.task_counter)
    blocked_by = blocked_by or []

    task = {
        "id": task_id,
        "subject": subject,
        "description": description,
        "status": "pending",
        "blockedBy": blocked_by,
        "blocks": [],
    }
    ctx.tasks.append(task)
    logger.info(f"create_task_impl: Added task {task_id}, total tasks now: {len(ctx.tasks)}")

    for dep_id in blocked_by:
        for t in ctx.tasks:
            if t["id"] == dep_id:
                t["blocks"].append(task_id)

    return {"id": task_id, "subject": subject, "status": "pending"}


def list_tasks_impl() -> str:
    """List all created tasks."""
    ctx = get_workflow_context()
    tasks = ctx.tasks if ctx else []

    if not tasks:
        return "No tasks created yet."

    return "\n".join(f"[{t['id']}] {t['subject']}" for t in tasks)


def get_tasks_json_impl() -> dict:
    """Get all tasks as JSON for the workflow response."""
    ctx = get_workflow_context()
    tasks = ctx.tasks if ctx else []
    return {"tasks": tasks, "count": len(tasks)}


def read_file_impl(file_path: str) -> dict:
    """Read file contents from workspace."""
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, file_path)

    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()[:10000]
        return {"content": content, "exists": True}
    else:
        return {"content": "", "exists": False}


def write_file_impl(file_path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, file_path)
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, 'w') as f:
        f.write(content)

    return f"Successfully wrote {len(content)} bytes to {file_path}"


def list_directory_impl(path: str = ".") -> dict:
    """List files and directories in workspace."""
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, path)

    items = glob.glob(os.path.join(full_path, "*"))
    files = [os.path.relpath(p, workspace) for p in items if os.path.isfile(p)]
    dirs = [os.path.relpath(p, workspace) for p in items if os.path.isdir(p)]

    return {"files": files[:50], "directories": dirs[:20], "count": len(items)}


def run_shell_command_impl(command: str) -> str:
    """Execute a shell command in the workspace."""
    workspace = get_workspace_dir()
    result = subprocess.run(
        command,
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    output = output[:5000]

    return output if output else f"Command completed with exit code {result.returncode}"


def search_code_impl(pattern: str, path: str = ".") -> str:
    """Search for a pattern in code files using grep."""
    workspace = get_workspace_dir()
    full_path = os.path.join(workspace, path)

    result = subprocess.run(
        ["grep", "-r", "-n", "--include=*.py", "--include=*.js", "--include=*.ts",
         "--include=*.json", "--include=*.yaml", "--include=*.yml", "--include=*.md",
         pattern, full_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout[:5000]

    if output:
        return output
    else:
        return f"No matches found for pattern: {pattern}"


# ============================================================================
# FastAPI Application
# ============================================================================


class RunRequest(BaseModel):
    """Request model for workflow invocation."""
    # Support both 'task' (new) and 'message' (legacy) fields
    task: Optional[str] = Field(default=None, description="The planning task/request")
    message: Optional[str] = Field(default=None, description="Alias for task (legacy)")
    # Execution mode selection
    durable: bool = Field(default=False, description="Use DurableAgentRunner with PR #827 interceptors (legacy)")
    mode: Optional[str] = Field(
        default=None,
        description=(
            "Execution mode: "
            "'workflow' (Dapr workflow durability - most durable), "
            "'v2' (official DaprSession - recommended), "
            "'session' (WorkflowContext), "
            "'durable' (interceptors), "
            "'standard' (basic)"
        )
    )


class ContinueRequest(BaseModel):
    """Request model for continuing a workflow conversation."""
    message: str = Field(..., description="Follow-up message/question to continue the conversation")


class TargetRepository(BaseModel):
    """Target repository for workflow cloning."""
    owner: str = Field(..., description="Repository owner (user or organization)")
    repo: str = Field(..., description="Repository name")
    branch: str = Field(default="main", description="Branch to clone")
    token: Optional[str] = Field(default=None, description="GitHub access token for private repos")


class MultiStepWorkflowRequest(BaseModel):
    """Request model for multi-step workflow (clone → planning → approval → execution → testing)."""
    task: str = Field(..., description="The task description for the workflow")
    model: str = Field(default="gpt-5.2-codex", description="OpenAI model to use")
    max_turns: int = Field(default=20, description="Max iterations per phase")
    max_test_retries: int = Field(default=3, description="Max retries if tests fail")
    repository: Optional[TargetRepository] = Field(default=None, description="Repository to clone before planning")
    auto_approve: bool = Field(default=False, description="Skip approval gate and auto-approve the plan")


class ApprovalRequest(BaseModel):
    """Request model for workflow approval."""
    approved: bool = Field(..., description="Whether the plan is approved")
    reason: Optional[str] = Field(default=None, description="Reason for approval/rejection")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan for startup/shutdown."""
    global PUBSUB_NAME, PUBSUB_TOPIC, WORKFLOW_INDEX_STORE

    # Initialize configuration and secrets from Dapr
    await initialize_config_and_secrets()

    # Update configuration values
    PUBSUB_NAME = get_config("PUBSUB_NAME", "pubsub")
    PUBSUB_TOPIC = get_config("PUBSUB_TOPIC", "workflow.stream")
    WORKFLOW_INDEX_STORE = get_config("WORKFLOW_INDEX_STORE", "ai-chatbot-statestore")

    if is_dapr_enabled():
        logger.info("[ConfigProvider] Using Dapr for configuration and secrets")
    else:
        logger.info("[ConfigProvider] Using environment variables (Dapr not available)")

    # Initialize OpenTelemetry tracing with OpenInference instrumentation (optional)
    try:
        from tracing import setup_tracing
        tracer = setup_tracing(
            project_name="planner-dapr-agent",
            enable_openai_instrumentation=True,
            trace_include_sensitive_data=True,
        )
        if tracer:
            logger.info("[Tracing] OpenTelemetry tracing enabled with OpenInference")
        else:
            logger.info("[Tracing] OpenTelemetry not available, using internal tracing only")
    except ImportError as e:
        logger.debug(f"[Tracing] Tracing module not available: {e}")

    # Start Dapr workflow runtime for multi-step workflows
    # Note: We start this regardless of is_dapr_enabled() because the Dapr sidecar
    # might not be ready at startup but will be ready when workflows are scheduled
    workflow_runtime = None
    if DAPR_MULTI_STEP_WORKFLOW_AVAILABLE:
        try:
            workflow_runtime = get_workflow_runtime()
            workflow_runtime.start()
            logger.info("[DaprWorkflow] Workflow runtime started for multi_step_workflow")
        except Exception as e:
            logger.warning(f"[DaprWorkflow] Failed to start workflow runtime: {e}")
            workflow_runtime = None

    yield

    # Shutdown workflow runtime
    if workflow_runtime:
        try:
            workflow_runtime.shutdown()
            logger.info("[DaprWorkflow] Workflow runtime stopped")
        except Exception as e:
            logger.warning(f"[DaprWorkflow] Error stopping workflow runtime: {e}")


app = FastAPI(
    title="Planner Agent (OpenAI Agents SDK)",
    description="Planning agent using OpenAI Agents SDK with Dapr integration",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint for Kubernetes."""
    return {"status": "healthy"}


# ============================================================================
# Dapr Streaming Subscription SSE Endpoint
# ============================================================================

# Active streaming subscriptions per workflow
_active_streams: dict[str, Queue] = {}
_stream_lock = threading.Lock()


def _dapr_event_handler(workflow_id: str, queue: Queue):
    """Create a Dapr pub/sub message handler for a specific workflow."""
    from dapr.clients.grpc._response import TopicEventResponse

    def handler(message):
        try:
            # Parse the message data
            data = message.data()
            if isinstance(data, bytes):
                data = json.loads(data.decode())
            elif isinstance(data, str):
                data = json.loads(data)

            # Filter by workflow ID
            msg_workflow_id = data.get("workflowId", "")
            if msg_workflow_id == workflow_id:
                logger.debug(f"[Stream] Event for {workflow_id}: {data.get('type')}")
                queue.put(data)

            return TopicEventResponse('success')
        except Exception as e:
            logger.warning(f"[Stream] Error handling message: {e}")
            return TopicEventResponse('success')  # Acknowledge anyway to not block

    return handler


async def _stream_workflow_events(workflow_id: str, request: Request):
    """Generator that streams workflow events via SSE using Dapr streaming subscription.

    Architecture:
    1. On connect: Replay historical events from Dapr workflow state (eliminates Redis)
    2. Real-time: Stream new events via Dapr streaming subscription

    This removes the need for Redis as an intermediary because:
    - Dapr workflow state already stores activity history
    - Streaming subscription provides real-time events directly
    """
    from dapr_config import get_config

    pubsub_name = get_config("PUBSUB_NAME", "pubsub")
    pubsub_topic = get_config("PUBSUB_TOPIC", "workflow.stream")

    # Create a queue for this stream
    event_queue: Queue = Queue()
    close_fn = None
    workflow_completed = False

    with _stream_lock:
        _active_streams[workflow_id] = event_queue

    try:
        # ============================================================
        # Phase 1: Send historical events from Dapr workflow state
        # This replaces the need for Redis event store
        # ============================================================
        initial_event = {
            "id": f"init-{uuid.uuid4().hex[:8]}",
            "type": "initial",
            "workflowId": workflow_id,
            "data": {"status": "RUNNING", "content": "Connected to workflow stream..."},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        yield f"data: {json.dumps(initial_event)}\n\n"
        await asyncio.sleep(0)  # Force flush to client

        # ============================================================
        # Replay buffered events (solves race condition)
        # Events published before SSE connect are buffered and replayed here
        # ============================================================
        buffered_events = _get_buffered_events(workflow_id)
        if buffered_events:
            logger.info(f"[Stream] Replaying {len(buffered_events)} buffered events for {workflow_id}")
            for event in buffered_events:
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)  # Force flush

        # Fetch workflow state which includes activity history
        try:
            from dapr.ext.workflow import DaprWorkflowClient
            wf_client = DaprWorkflowClient()
            state = wf_client.get_workflow_state(workflow_id)

            if state:
                runtime_status = state.runtime_status.name if state.runtime_status else "UNKNOWN"

                # Parse custom status for phase/progress info
                custom_status = {}
                if state.serialized_custom_status:
                    try:
                        custom_status = json.loads(state.serialized_custom_status)
                    except:
                        pass

                # Send current status
                status_event = {
                    "id": f"status-{uuid.uuid4().hex[:8]}",
                    "type": "status",
                    "workflowId": workflow_id,
                    "data": {
                        "status": runtime_status,
                        "phase": custom_status.get("phase", "running"),
                        "progress": custom_status.get("progress", 0),
                        "message": custom_status.get("message", ""),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                yield f"data: {json.dumps(status_event)}\n\n"
                await asyncio.sleep(0)  # Force flush to client

                # Check if workflow is already completed
                if runtime_status in ["COMPLETED", "FAILED", "TERMINATED"]:
                    workflow_completed = True

                    # Send completion event
                    done_event = {
                        "id": f"done-{uuid.uuid4().hex[:8]}",
                        "type": "stream_done",
                        "workflowId": workflow_id,
                        "data": {"status": runtime_status},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    yield f"data: {json.dumps(done_event)}\n\n"
                    await asyncio.sleep(0)  # Force flush to client
                    return  # Stream ends for completed workflows

        except Exception as e:
            logger.debug(f"[Stream] Could not fetch initial workflow state: {e}")

        # ============================================================
        # Phase 2: Start Dapr streaming subscription for real-time events
        # ============================================================
        def start_subscription():
            nonlocal close_fn
            try:
                with DaprClient() as client:
                    handler = _dapr_event_handler(workflow_id, event_queue)
                    close_fn = client.subscribe_with_handler(
                        pubsub_name=pubsub_name,
                        topic=pubsub_topic,
                        handler_fn=handler,
                    )
                    logger.info(f"[Stream] Started Dapr subscription for {workflow_id}")

                    # Keep subscription alive until queue is removed
                    while workflow_id in _active_streams:
                        import time
                        time.sleep(0.5)

                    if close_fn:
                        close_fn()
                        logger.info(f"[Stream] Closed Dapr subscription for {workflow_id}")
            except Exception as e:
                logger.error(f"[Stream] Subscription error for {workflow_id}: {e}")
                event_queue.put({"type": "error", "data": {"error": str(e)}})

        # Start subscription thread
        sub_thread = threading.Thread(target=start_subscription, daemon=True)
        sub_thread.start()

        # Stream events from queue
        timeout_count = 0
        max_timeout_count = 1800  # 30 minutes at 1 second intervals

        while timeout_count < max_timeout_count:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info(f"[Stream] Client disconnected for {workflow_id}")
                break

            try:
                # Get event from queue with timeout
                event = event_queue.get(timeout=1.0)
                timeout_count = 0  # Reset on activity

                # Check for terminal events
                event_type = event.get("type", "")
                if event_type in ["execution_completed", "execution_failed", "phase_completed"]:
                    phase = event.get("data", {}).get("phase", "")
                    if phase == "completed" or event_type == "execution_completed":
                        yield f"data: {json.dumps(event)}\n\n"
                        await asyncio.sleep(0)  # Force flush to client
                        # Send done marker
                        done_event = {
                            "id": f"done-{uuid.uuid4().hex[:8]}",
                            "type": "stream_done",
                            "workflowId": workflow_id,
                            "data": {"status": "COMPLETED"},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        yield f"data: {json.dumps(done_event)}\n\n"
                        await asyncio.sleep(0)  # Force flush to client
                        break

                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)  # Force flush to client

            except Empty:
                timeout_count += 1
                # Send heartbeat every 15 seconds
                if timeout_count % 15 == 0:
                    heartbeat = {
                        "id": f"hb-{uuid.uuid4().hex[:8]}",
                        "type": "heartbeat",
                        "workflowId": workflow_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                    await asyncio.sleep(0)  # Force flush to client

    finally:
        # Cleanup
        with _stream_lock:
            if workflow_id in _active_streams:
                del _active_streams[workflow_id]
        logger.info(f"[Stream] Ended stream for {workflow_id}")


@app.get("/workflows/{workflow_id}/stream")
async def stream_workflow(workflow_id: str, request: Request):
    """SSE endpoint for streaming workflow events using Dapr streaming subscriptions.

    This uses Dapr's pull-based streaming subscription (Dapr 1.15+) to receive
    events directly without the webhook + Redis intermediary.

    Args:
        workflow_id: The workflow instance ID (e.g., wf-abc123)

    Returns:
        SSE stream of workflow events
    """
    logger.info(f"[Stream] Starting SSE stream for workflow {workflow_id}")

    return StreamingResponse(
        _stream_workflow_events(workflow_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/run")
async def start_workflow(request: RunRequest, background_tasks: BackgroundTasks):
    """Start a new planning workflow.

    Returns:
        instance_id: Workflow ID for tracking
        status: Initial status (started)
    """
    # Support both 'task' and 'message' fields for compatibility
    task = request.task or request.message
    if not task:
        return JSONResponse(
            status_code=400,
            content={"error": "Either 'task' or 'message' field is required"},
        )

    # Generate workflow ID with wf- prefix for ai-chatbot compatibility
    workflow_id = f"wf-{uuid.uuid4().hex[:12]}"

    # Register in ai-chatbot index
    register_workflow_in_index(
        workflow_id=workflow_id,
        workflow_name="planner_workflow",
        message=task,
    )

    # Publish initial event
    publish_workflow_event(
        workflow_id=workflow_id,
        event_type="initial",
        data={
            "status": "started",
            "metadata": {"task": task},
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

    # Execute workflow in background
    # Determine execution mode (mode parameter takes precedence over durable flag)
    if request.mode:
        mode = request.mode
    elif request.durable:
        mode = "durable"
    else:
        mode = "v2"  # Default to v2 mode (official DaprSession - recommended)

    # Dispatch to appropriate execution function
    if mode == "workflow":
        background_tasks.add_task(execute_workflow_dapr, workflow_id, task)
    elif mode == "v2":
        background_tasks.add_task(execute_workflow_v2, workflow_id, task)
    elif mode == "durable":
        background_tasks.add_task(execute_workflow_durable, workflow_id, task)
    elif mode == "session":
        background_tasks.add_task(execute_workflow_session, workflow_id, task)
    else:  # "standard" or any other value
        background_tasks.add_task(execute_workflow, workflow_id, task)

    return {
        "instance_id": workflow_id,
        "status": "started",
        "mode": mode,
    }


@app.get("/status/{instance_id}")
async def get_status(instance_id: str):
    """Get workflow status by instance ID.

    Returns:
        instance_id: Workflow ID
        status: Current status
        output: Workflow output (if completed)
        created_at: Creation timestamp
    """
    entry = get_workflow_from_index(instance_id)

    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": f"Workflow {instance_id} not found"},
        )

    return {
        "instance_id": instance_id,
        "status": entry.get("status", "unknown"),
        "output": entry.get("output"),
        "created_at": entry.get("createdAt"),
    }


@app.get("/workflows/{workflow_id}")
async def get_workflow_details(workflow_id: str):
    """Get detailed workflow information including activities.

    Returns:
        instanceId: Workflow ID
        status: Current status
        phase: Current workflow phase (from custom_status)
        progress: Progress percentage (from custom_status)
        message: Status message (from custom_status)
        plan: Plan data if available (from custom_status)
        activities: Array of activity executions
        output: Workflow output
        createdAt: Creation timestamp
        updatedAt: Last update timestamp
    """
    entry = get_workflow_from_index(workflow_id)

    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": f"Workflow {workflow_id} not found"},
        )

    # Build base response from index
    response = {
        "instanceId": workflow_id,
        "workflowName": entry.get("workflowName"),
        "status": entry.get("status"),
        "phase": None,
        "progress": 0,
        "message": None,
        "plan": None,
        "activities": entry.get("activities", []),
        "input": entry.get("input"),
        "output": entry.get("output"),
        "error": entry.get("error"),
        "createdAt": entry.get("createdAt"),
        "updatedAt": entry.get("updatedAt"),
        "completedAt": entry.get("completedAt"),
    }

    # Try to get custom_status from Dapr workflow runtime for live phase/progress
    if DAPR_MULTI_STEP_WORKFLOW_AVAILABLE and workflow_id.startswith("wf-"):
        try:
            client = DaprWorkflowClient()
            state = client.get_workflow_state(instance_id=workflow_id)
            if state and state.serialized_custom_status:
                custom_status = json.loads(state.serialized_custom_status)
                # Handle double-encoded JSON
                if isinstance(custom_status, str):
                    custom_status = json.loads(custom_status)
                if isinstance(custom_status, dict):
                    response["phase"] = custom_status.get("phase")
                    response["progress"] = custom_status.get("progress", 0)
                    response["message"] = custom_status.get("message")
                    response["plan"] = custom_status.get("plan")
                    # Update status from Dapr state if available
                    if state.runtime_status:
                        response["status"] = state.runtime_status.name
        except Exception as e:
            logger.debug(f"Could not fetch Dapr workflow state for {workflow_id}: {e}")

    return response


@app.get("/workflows")
async def list_workflows(limit: int = 20):
    """List all workflows.

    Returns:
        workflows: Array of workflow entries
        total: Total count
    """
    workflows = get_workflows_from_index(limit)
    return {
        "workflows": workflows,
        "total": len(workflows),
    }


@app.post("/workflow/{workflow_id}/approve")
async def approve_workflow(workflow_id: str, request: ApprovalRequest):
    """Approve or reject a workflow plan.

    This endpoint raises the approval event to a Dapr workflow that is
    waiting at wait_for_external_event("approval").

    Args:
        workflow_id: The workflow instance ID
        request: ApprovalRequest with approved boolean and optional reason

    Returns:
        success: Whether the event was raised
        workflow_id: The workflow ID
        message: Status message
    """
    if not DAPR_MULTI_STEP_WORKFLOW_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Dapr multi-step workflow not available",
                "hint": "Ensure dapr-ext-workflow is installed and Dapr sidecar is running",
            },
        )

    try:
        # Raise the approval event to the waiting workflow
        client = DaprWorkflowClient()
        client.raise_workflow_event(
            instance_id=workflow_id,
            event_name="approval",
            data={
                "approved": request.approved,
                "reason": request.reason,
            },
        )

        logger.info(
            f"Raised approval event for workflow {workflow_id}: "
            f"approved={request.approved}, reason={request.reason}"
        )

        # Update workflow status in index
        if request.approved:
            update_workflow_status(
                workflow_id=workflow_id,
                status="RUNNING",
            )
        else:
            update_workflow_status(
                workflow_id=workflow_id,
                status="REJECTED",
                error=request.reason or "Plan rejected",
            )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": f"Plan {'approved' if request.approved else 'rejected'}",
        }

    except Exception as e:
        logger.error(f"Failed to raise approval event for workflow {workflow_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
            },
        )


@app.post("/continue/{workflow_id}")
async def continue_workflow_endpoint(workflow_id: str, request: ContinueRequest):
    """Continue an existing workflow conversation.

    This endpoint enables multi-turn conversations by continuing an existing
    workflow with a follow-up message. The agent will have access to all
    previous context through the official DaprSession.

    Args:
        workflow_id: ID of the existing workflow to continue
        request: ContinueRequest with the follow-up message

    Returns:
        status: Completion status
        response: Agent's response
        tasks: Updated task list
        usage: Token usage for this turn
    """
    # Verify workflow exists
    entry = get_workflow_from_index(workflow_id)
    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": f"Workflow {workflow_id} not found"},
        )

    # Run continuation (synchronous to return response directly)
    result = await continue_workflow(workflow_id, request.message)

    if result.get("status") == "error":
        return JSONResponse(
            status_code=400,
            content=result,
        )

    return result


@app.post("/workflow")
async def run_multi_step(request: MultiStepWorkflowRequest, background_tasks: BackgroundTasks):
    """Run multi-step planning → execution → testing workflow.

    This endpoint runs a three-phase agent workflow:
    1. Planning Phase - Agent researches and creates a detailed plan with test cases
    2. Execution Phase - Agent executes the plan using tools
    3. Testing Phase - Agent verifies the implementation meets requirements

    Each phase terminates when the agent outputs the appropriate structured type:
    - Planning: outputs Plan (tasks + test cases)
    - Execution: outputs ExecutionResult (completed tasks)
    - Testing: outputs TestResult (pass/fail summary)

    Args:
        request: MultiStepWorkflowRequest with task description and configuration

    Returns:
        status: "completed" if all tests pass, "failed" otherwise
        plan: The generated plan with tasks and test cases
        execution: Results from the execution phase
        testing: Results from the testing phase
    """
    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENAI_API_KEY not configured"},
        )

    # Set environment variable for OpenAI client
    os.environ["OPENAI_API_KEY"] = api_key

    # Generate workflow ID
    workflow_id = f"wf-{uuid.uuid4().hex[:12]}"

    # Register workflow in index for ai-chatbot visibility
    register_workflow_in_index(
        workflow_id=workflow_id,
        workflow_name="multi_step_workflow",
        message=request.task,
    )
    logger.info(f"Registered multi-step workflow {workflow_id} in index")

    # Publish initial event
    publish_workflow_event(
        workflow_id=workflow_id,
        event_type="execution_started",
        data={"phase": "planning", "task": request.task[:200]},
    )

    logger.info(f"Starting multi-step workflow {workflow_id} for task: {request.task[:100]}...")

    # Track activities for each phase
    activities = []

    def add_activity(name: str, status: str, output: dict = None):
        """Add or update activity in the list."""
        now = datetime.now(timezone.utc).isoformat()
        # Find existing activity
        existing = None
        for act in activities:
            if act["activityName"] == name and act["status"] == "running":
                existing = act
                break

        if existing and status in ("completed", "failed"):
            existing["status"] = status
            existing["endTime"] = now
            existing["durationMs"] = int((datetime.fromisoformat(now.replace('Z', '+00:00')) -
                                          datetime.fromisoformat(existing["startTime"].replace('Z', '+00:00'))).total_seconds() * 1000)
            if output:
                existing["output"] = output
        else:
            activity = {
                "activityName": name,
                "status": status,
                "startTime": now,
            }
            if output:
                activity["output"] = output
            activities.append(activity)

        # Update workflow activities in index
        update_workflow_activities(workflow_id, activities)

    try:
        # Import workflow functions
        from workflow_agent import (
            create_planning_agent, create_execution_agent, create_testing_agent,
            Plan, ExecutionResult, TestResult, Task
        )
        from agents import Runner

        # ==================== PHASE 1: PLANNING ====================
        add_activity("phase:planning", "running")
        publish_workflow_event(workflow_id, "phase_started", {"phase": "planning"})

        planning_agent = create_planning_agent(request.model)
        plan_result = await Runner.run(
            planning_agent,
            input=request.task,
            max_turns=request.max_turns,
        )
        plan: Plan = plan_result.final_output

        # Auto-populate blocks based on blockedBy
        task_map = {t.id: t for t in plan.tasks}
        for task in plan.tasks:
            for blocked_by_id in task.blockedBy:
                if blocked_by_id in task_map:
                    if task.id not in task_map[blocked_by_id].blocks:
                        task_map[blocked_by_id].blocks.append(task.id)

        add_activity("phase:planning", "completed", {
            "tasks_created": len(plan.tasks),
            "tests_created": len(plan.tests),
            "summary": plan.summary[:200],
        })
        publish_workflow_event(workflow_id, "phase_completed", {
            "phase": "planning",
            "tasks_created": len(plan.tasks),
        })

        # ==================== PHASE 2: EXECUTION ====================
        add_activity("phase:execution", "running")
        publish_workflow_event(workflow_id, "phase_started", {"phase": "execution"})

        execution_agent = create_execution_agent(request.model)
        exec_prompt = f"""Execute this plan:

Summary: {plan.summary}

Tasks:
{chr(10).join(f"- [{t.id}] {t.subject}: {t.description} (blockedBy: {t.blockedBy})" for t in plan.tasks)}

Reasoning: {plan.reasoning}"""

        exec_result = await Runner.run(
            execution_agent,
            input=exec_prompt,
            max_turns=request.max_turns,
        )
        execution: ExecutionResult = exec_result.final_output

        add_activity("phase:execution", "completed", {
            "success": execution.success,
            "completed_tasks": execution.completed_tasks,
            "errors": execution.errors,
        })
        publish_workflow_event(workflow_id, "phase_completed", {
            "phase": "execution",
            "success": execution.success,
            "completed_tasks": len(execution.completed_tasks),
        })

        # ==================== PHASE 3: TESTING ====================
        add_activity("phase:testing", "running")
        publish_workflow_event(workflow_id, "phase_started", {"phase": "testing"})

        testing_agent = create_testing_agent(request.model)
        test_prompt = f"""Verify the implementation:

Plan Summary: {plan.summary}

Test Cases:
{chr(10).join(f"- [{tc.id}] {tc.description} (type: {tc.test_type}, command: {tc.command})" for tc in plan.tests)}

Execution Summary: {execution.output}
Completed Tasks: {execution.completed_tasks}"""

        test: TestResult = TestResult(
            passed=False, tests_run=0, tests_passed=0, tests_failed=0,
            failures=[], summary="Tests not yet run"
        )

        for attempt in range(request.max_test_retries):
            test_result = await Runner.run(
                testing_agent,
                input=test_prompt,
                max_turns=request.max_turns,
            )
            test = test_result.final_output
            if test.passed:
                break

        test_status = "completed" if test.passed else "failed"
        add_activity("phase:testing", test_status, {
            "passed": test.passed,
            "tests_run": test.tests_run,
            "tests_passed": test.tests_passed,
            "tests_failed": test.tests_failed,
            "failures": test.failures,
        })
        publish_workflow_event(workflow_id, "phase_completed", {
            "phase": "testing",
            "passed": test.passed,
            "tests_passed": test.tests_passed,
            "tests_failed": test.tests_failed,
        })

        # Build result
        result = {
            "workflow_id": workflow_id,
            "plan": plan.model_dump(),
            "execution": execution.model_dump(),
            "testing": test.model_dump(),
            "status": "completed" if test.passed else "failed",
        }

        # Update workflow status in index
        final_status = "COMPLETED" if result["status"] == "completed" else "FAILED"
        update_workflow_status(
            workflow_id=workflow_id,
            status=final_status,
            output=result,
        )

        # Publish completion event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type="execution_completed",
            data={
                "status": final_status,
                "tasks_planned": len(plan.tasks),
                "tasks_executed": len(execution.completed_tasks),
                "tests_passed": test.tests_passed,
                "tests_failed": test.tests_failed,
            },
        )

        logger.info(f"Multi-step workflow {workflow_id} completed with status: {result['status']}")

        return result

    except Exception as e:
        logger.error(f"Multi-step workflow {workflow_id} failed: {e}")

        # Mark current activity as failed
        for act in activities:
            if act["status"] == "running":
                act["status"] = "failed"
                act["endTime"] = datetime.now(timezone.utc).isoformat()
                act["output"] = {"error": str(e)}
        update_workflow_activities(workflow_id, activities)

        # Update workflow status to failed
        update_workflow_status(
            workflow_id=workflow_id,
            status="FAILED",
            output={"error": str(e)},
        )

        return JSONResponse(
            status_code=500,
            content={
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
            },
        )


@app.post("/workflow/dapr")
async def run_dapr_multi_step(request: MultiStepWorkflowRequest, background_tasks: BackgroundTasks):
    """Run multi-step workflow using Dapr workflow SDK.

    This endpoint runs a multi-phase agent workflow using Dapr's native
    workflow SDK so activities appear in the ai-chatbot UI workflow graph.

    Phases:
    0. Clone (optional) - Clone repository if provided
    1. Planning - Creates detailed plan with tasks and test cases
    2. Approval - Wait for human approval (unless auto_approve=true)
    3. Execution - Executes the plan
    4. Testing - Verifies the implementation

    Args:
        request: MultiStepWorkflowRequest with task, repository, and configuration

    Returns:
        workflow_id: Dapr workflow instance ID
        status: Current status ("running" or "pending" if awaiting approval)
    """
    if not DAPR_MULTI_STEP_WORKFLOW_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Dapr multi-step workflow not available",
                "hint": "Ensure dapr-ext-workflow is installed and Dapr sidecar is running",
            },
        )

    # Get OpenAI API key
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENAI_API_KEY not configured"},
        )

    # Set environment variable for OpenAI client
    os.environ["OPENAI_API_KEY"] = api_key

    # Generate workflow ID
    workflow_id = f"wf-{uuid.uuid4().hex[:12]}"

    # Build workflow input
    workflow_input = {
        "task": request.task,
        "model": request.model,
        "max_turns": request.max_turns,
        "max_test_retries": request.max_test_retries,
        "auto_approve": request.auto_approve,
    }

    # Add repository config if provided
    if request.repository:
        workflow_input["repository"] = {
            "owner": request.repository.owner,
            "repo": request.repository.repo,
            "branch": request.repository.branch,
            "token": request.repository.token,
        }
        logger.info(
            f"Workflow {workflow_id} will clone {request.repository.owner}/"
            f"{request.repository.repo}@{request.repository.branch}"
        )

    # Register workflow in index for ai-chatbot visibility
    register_workflow_in_index(
        workflow_id=workflow_id,
        workflow_name="multi_step_workflow",
        message=request.task,
    )
    logger.info(f"Registered Dapr multi-step workflow {workflow_id} in index")

    try:
        # Start the Dapr workflow
        client = DaprWorkflowClient()
        instance_id = client.schedule_new_workflow(
            workflow=multi_step_workflow,
            instance_id=workflow_id,
            input=workflow_input,
        )

        logger.info(f"Started Dapr multi-step workflow: {instance_id}")

        # If auto_approve is enabled, wait for completion synchronously
        # Otherwise return immediately (workflow will pause at approval gate)
        if request.auto_approve:
            # Wait for the workflow to complete
            state = client.wait_for_workflow_completion(
                instance_id=instance_id,
                timeout_in_seconds=600,  # 10 minute timeout
            )

            # Get the result
            if state.runtime_status.name == "COMPLETED":
                result = state.serialized_output
                logger.info(f"Workflow completed. serialized_output type: {type(result)}")
                if isinstance(result, str):
                    result = json.loads(result)

                # Update the workflow index with completion status
                _update_workflow_completion(workflow_id, request.task, result)
                return result

            elif state.runtime_status.name == "FAILED":
                error_msg = state.failure_details.message if state.failure_details else "Unknown error"
                update_workflow_status(workflow_id, status="failed", error=error_msg)
                return JSONResponse(
                    status_code=500,
                    content={
                        "workflow_id": instance_id,
                        "status": "failed",
                        "error": error_msg,
                    },
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "workflow_id": instance_id,
                        "status": state.runtime_status.name,
                        "error": f"Workflow ended with status: {state.runtime_status.name}",
                    },
                )
        else:
            # Return immediately - workflow will pause at approval gate
            # Background task monitors and updates status
            background_tasks.add_task(_monitor_workflow, workflow_id, request.task)

            return {
                "workflow_id": instance_id,
                "status": "running",
                "message": "Workflow started. Will pause at planning phase for approval.",
                "approval_endpoint": f"/workflow/{instance_id}/approve",
            }

    except Exception as e:
        logger.error(f"Failed to start Dapr workflow: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
            },
        )


def _update_workflow_completion(workflow_id: str, task: str, result: dict) -> None:
    """Update workflow index with completion status and activities."""
    try:
        now = datetime.now(timezone.utc).isoformat()

        # Create activities for the phases
        activities = []

        # Add clone activity if present
        if result.get("phase") == "cloning" or "repository" in result:
            activities.append({
                "activityName": "clone_repository",
                "status": "completed",
                "startTime": now,
                "endTime": now,
                "input": {"repository": result.get("repository", {})},
                "output": {"path": result.get("workspace_path", "")},
            })

        # Add planning activity
        activities.append({
            "activityName": "planning",
            "status": "completed",
            "startTime": now,
            "endTime": now,
            "input": {"task": task},
            "output": result.get("plan", {}),
        })

        # Add execution activity
        activities.append({
            "activityName": "execution",
            "status": "completed",
            "startTime": now,
            "endTime": now,
            "input": {"plan": result.get("plan", {})},
            "output": result.get("execution", {}),
        })

        # Add testing activity
        activities.append({
            "activityName": "testing",
            "status": "completed",
            "startTime": now,
            "endTime": now,
            "input": {"plan": result.get("plan", {}), "execution": result.get("execution", {})},
            "output": result.get("testing", {}),
        })

        update_workflow_activities(workflow_id, activities)
        update_workflow_status(workflow_id, status="completed", output=result)
        logger.info(f"Updated workflow index for {workflow_id}: COMPLETED")
    except Exception as e:
        logger.warning(f"Failed to update workflow index: {e}")


async def _monitor_workflow(workflow_id: str, task: str) -> None:
    """Background task to monitor workflow progress and update status."""
    try:
        client = DaprWorkflowClient()
        max_wait_seconds = 86400  # 24 hours (matches approval timeout)
        poll_interval = 5  # Check every 5 seconds

        import time
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            try:
                state = client.get_workflow_state(instance_id=workflow_id)

                if not state:
                    logger.warning(f"Workflow {workflow_id} not found")
                    break

                # Check custom status for phase updates
                if state.serialized_custom_status:
                    try:
                        custom_status = json.loads(state.serialized_custom_status)
                        # Handle double-encoded JSON (Dapr workflow serializes twice)
                        if isinstance(custom_status, str):
                            custom_status = json.loads(custom_status)
                        phase = custom_status.get("phase", "")
                        progress = custom_status.get("progress", 0)
                        message = custom_status.get("message", "")

                        # Update workflow status with phase info
                        # This allows UI to show progress
                        logger.debug(f"Workflow {workflow_id} phase: {phase}, progress: {progress}%")

                    except json.JSONDecodeError:
                        pass

                # Check if workflow is complete
                if state.runtime_status.name == "COMPLETED":
                    result = state.serialized_output
                    if isinstance(result, str):
                        result = json.loads(result)
                    _update_workflow_completion(workflow_id, task, result)
                    logger.info(f"Workflow {workflow_id} completed successfully")
                    break

                elif state.runtime_status.name == "FAILED":
                    error_msg = state.failure_details.message if state.failure_details else "Unknown error"
                    update_workflow_status(workflow_id, status="failed", error=error_msg)
                    logger.error(f"Workflow {workflow_id} failed: {error_msg}")
                    break

                elif state.runtime_status.name in ("TERMINATED", "CANCELED"):
                    update_workflow_status(workflow_id, status=state.runtime_status.name.lower())
                    logger.info(f"Workflow {workflow_id} {state.runtime_status.name.lower()}")
                    break

            except Exception as e:
                logger.warning(f"Error checking workflow {workflow_id} state: {e}")

            await asyncio.sleep(poll_interval)

    except Exception as e:
        logger.error(f"Workflow monitor for {workflow_id} failed: {e}")


@app.get("/capabilities")
async def get_capabilities():
    """Get agent capabilities and available execution modes.

    Returns information about available features, including whether
    the official DaprSession is available for v2 mode and whether
    the Dapr workflow extension is available for workflow mode.
    """
    return {
        "official_dapr_session_available": OFFICIAL_DAPR_SESSION_AVAILABLE,
        "dapr_workflow_runner_available": DAPR_WORKFLOW_RUNNER_AVAILABLE,
        "dapr_multi_step_workflow_available": DAPR_MULTI_STEP_WORKFLOW_AVAILABLE,
        "execution_modes": {
            "multi_step_dapr": {
                "available": DAPR_MULTI_STEP_WORKFLOW_AVAILABLE,
                "endpoint": "/workflow/dapr",
                "description": "Three-phase workflow (planning→execution→testing) using Dapr workflow SDK",
                "features": ["dapr-activities", "ui-graph-visualization", "workflow-durability", "phase-tracking"],
            },
            "multi_step": {
                "available": True,
                "endpoint": "/workflow",
                "description": "Three-phase workflow (planning→execution→testing) with state-store tracking",
                "features": ["planning", "execution", "testing", "activity-tracking"],
            },
            "workflow": {
                "available": DAPR_WORKFLOW_RUNNER_AVAILABLE,
                "description": "DaprOpenAIRunner with true Dapr workflow durability (most durable)",
                "features": ["crash-recovery", "workflow-durability", "activity-checkpointing", "replay"],
            },
            "v2": {
                "available": OFFICIAL_DAPR_SESSION_AVAILABLE,
                "description": "Official DaprSession for conversation memory (recommended)",
                "features": ["multi-turn", "conversation-history", "activity-tracking"],
            },
            "session": {
                "available": True,
                "description": "WorkflowContext for activity tracking",
                "features": ["activity-tracking", "state-persistence"],
            },
            "durable": {
                "available": True,
                "description": "DurableAgentRunner with interceptors",
                "features": ["durability", "replay", "activity-tracking"],
            },
            "standard": {
                "available": True,
                "description": "Basic execution without durability",
                "features": ["activity-tracking"],
            },
            "multi-step": {
                "available": True,
                "description": "Three-phase workflow: Planning → Execution → Testing",
                "features": ["structured-output", "test-verification", "dependency-ordering"],
                "endpoint": "/workflow",
            },
        },
        "default_mode": "workflow" if DAPR_WORKFLOW_RUNNER_AVAILABLE else ("v2" if OFFICIAL_DAPR_SESSION_AVAILABLE else "session"),
    }


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Main entry point."""
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Planner Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
