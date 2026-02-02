"""Planner Agent definition using OpenAI Agents SDK.

This module defines the planner agent using the standard OpenAI Agents SDK patterns:
- @function_tool decorator for tool definitions
- Agent class for agent configuration
- Runner.run() for execution

Dapr integration is handled separately via WorkflowContext and interceptors,
keeping the agent definition clean and SDK-native.
"""

import functools
import glob
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from agents import Agent, function_tool

logger = logging.getLogger(__name__)

# Default workspace directory
DEFAULT_CWD = os.getenv("PLANNER_CWD", "/app/workspace")


def _publish_tool_event(event_type: str, tool_name: str, data: dict) -> None:
    """Publish a tool event to pub/sub for real-time SSE streaming.

    This is called by tool wrappers to publish tool_call and tool_result events.
    """
    try:
        # Get workflow context to get workflow_id
        ctx = _get_context()
        if not ctx:
            return

        workflow_id = getattr(ctx, 'workflow_id', None)
        if not workflow_id:
            return

        # Import and call publish function
        from app import publish_workflow_event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type=event_type,
            data=data,
        )
        logger.debug(f"Published {event_type} event for tool {tool_name}")
    except Exception as e:
        logger.debug(f"Could not publish {event_type} event: {e}")


def tracked_tool(func: Callable) -> Any:
    """Decorator that wraps a tool function to publish events.

    Publishes tool_call when the tool starts and tool_result when it completes.
    Falls back gracefully if publishing fails.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        tool_name = func.__name__

        # Publish tool_call event
        _publish_tool_event(
            "tool_call",
            tool_name,
            {
                "toolName": tool_name,
                "toolInput": _safe_serialize(kwargs or args),
                "content": f"Running: {tool_name}",
            }
        )

        try:
            # Execute the tool
            result = func(*args, **kwargs)

            # Publish tool_result event
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            _publish_tool_event(
                "tool_result",
                tool_name,
                {
                    "toolName": tool_name,
                    "toolOutput": result_str,
                    "status": "completed",
                    "content": f"Result: {tool_name}",
                }
            )

            return result
        except Exception as e:
            # Publish error event
            _publish_tool_event(
                "tool_result",
                tool_name,
                {
                    "toolName": tool_name,
                    "toolOutput": str(e),
                    "status": "failed",
                    "content": f"Error: {tool_name}",
                }
            )
            raise

    # Apply @function_tool decorator
    return function_tool(wrapper)


def _safe_serialize(value: Any, max_depth: int = 3) -> Any:
    """Safely serialize a value for logging/publishing."""
    if max_depth <= 0:
        return str(value)[:200]

    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500] if len(value) > 500 else value

    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            return f"<{len(value)} items>"
        return [_safe_serialize(v, max_depth - 1) for v in value]

    if isinstance(value, dict):
        result = {}
        for k, v in list(value.items())[:10]:
            result[str(k)] = _safe_serialize(v, max_depth - 1)
        return result

    return str(value)[:200]


# =============================================================================
# Tool Definitions using @function_tool decorator
# =============================================================================


def _get_context():
    """Get the current workflow/session context.

    Tries WorkflowContext first (new pattern), falls back to durable_runner (legacy).
    """
    try:
        from workflow_context import get_context_state
        ctx = get_context_state()
        if ctx:
            return ctx
    except ImportError:
        pass

    # Fallback to durable_runner
    from durable_runner import get_workflow_context
    return get_workflow_context()


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
    ctx = _get_context()
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

    return {"id": task_id, "subject": subject, "status": "pending"}


@tracked_tool
def list_tasks() -> str:
    """List all created tasks.

    Returns:
        Formatted string of all tasks with their IDs and subjects
    """
    ctx = _get_context()
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
    ctx = _get_context()
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
    full_path = os.path.join(DEFAULT_CWD, file_path)

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
    full_path = os.path.join(DEFAULT_CWD, file_path)

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
    full_path = os.path.join(DEFAULT_CWD, path)

    items = glob.glob(os.path.join(full_path, "*"))
    files = [os.path.relpath(p, DEFAULT_CWD) for p in items if os.path.isfile(p)]
    dirs = [os.path.relpath(p, DEFAULT_CWD) for p in items if os.path.isdir(p)]

    return {"files": files[:50], "directories": dirs[:20], "count": len(items)}


@tracked_tool
def run_shell_command(command: str) -> str:
    """Execute a shell command in the workspace.

    Args:
        command: Shell command to execute

    Returns:
        Command output or error message
    """
    result = subprocess.run(
        command,
        shell=True,
        cwd=DEFAULT_CWD,
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
    full_path = os.path.join(DEFAULT_CWD, path)

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


# =============================================================================
# Agent Definition
# =============================================================================

PLANNER_INSTRUCTIONS = """You are a software planning assistant with tools to explore codebases and create implementation plans.

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
- Build a proper DAG of dependencies for complex plans"""


def create_planner_agent(model: str = "gpt-4o") -> Agent:
    """Create the planner agent with all tools.

    This creates a standard OpenAI Agents SDK Agent instance.
    Dapr integration (durability, state persistence) is handled
    separately by wrapping the runner execution.

    Args:
        model: The model to use (default: gpt-4o)

    Returns:
        Configured Agent instance
    """
    return Agent(
        name="Planner",
        instructions=PLANNER_INSTRUCTIONS,
        model=model,
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
