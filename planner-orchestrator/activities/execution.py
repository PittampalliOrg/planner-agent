"""Execution activity - invokes the execution agent service via Dapr."""

from __future__ import annotations

import json
import os
from typing import Any

import requests


EXEC_SERVICE_APP_ID = os.environ.get("EXEC_SERVICE_APP_ID", "planner-agent-exec")
DAPR_HTTP_PORT = os.environ.get("DAPR_HTTP_PORT", "3500")
EXEC_TIMEOUT_SECONDS = 1800


def run_execution(ctx, input_data: dict[str, Any]) -> dict[str, Any]:
    """Invoke the execution agent service via Dapr service invocation.

    Sends tasks to the execution service, which restores them to the
    native filesystem and runs the Claude Agent SDK in bypassPermissions mode.
    """
    workflow_id = input_data["workflow_id"]
    tasks = input_data.get("tasks", [])
    cwd = input_data.get("cwd", os.getcwd())

    prompt = _build_execution_prompt(tasks)

    url = f"http://localhost:{DAPR_HTTP_PORT}/v1.0/invoke/{EXEC_SERVICE_APP_ID}/method/execute"

    try:
        response = requests.post(
            url,
            json={"prompt": prompt, "cwd": cwd, "tasks": tasks, "workflow_id": workflow_id},
            headers={"dapr-app-timeout": str(EXEC_TIMEOUT_SECONDS)},
            timeout=EXEC_TIMEOUT_SECONDS + 30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.Timeout:
        return {
            "success": False,
            "error": f"Execution timed out after {EXEC_TIMEOUT_SECONDS} seconds",
            "workflow_id": workflow_id,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:2000],
            "workflow_id": workflow_id,
        }

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Unknown error"),
            "workflow_id": workflow_id,
        }

    return {
        "success": True,
        "workflow_id": workflow_id,
    }


def _build_execution_prompt(tasks: list[dict[str, Any]]) -> str:
    """Format tasks into a prompt for the execution agent."""
    if not tasks:
        return "Execute the planned tasks. Check your task list and implement each task in order."

    lines = [
        "Execute the following implementation tasks in dependency order.",
        "Mark each task as completed using TaskUpdate as you finish it.",
        "",
        "Tasks:",
    ]

    for task in tasks:
        status = task.get("status", "pending")
        subject = task.get("subject", "Untitled")
        description = task.get("description", "")
        blocked_by = task.get("blockedBy", [])

        lines.append(f"- [{status}] {subject}")
        if description:
            lines.append(f"  Description: {description}")
        if blocked_by:
            lines.append(f"  Blocked by: {', '.join(str(b) for b in blocked_by)}")
        lines.append("")

    return "\n".join(lines)
