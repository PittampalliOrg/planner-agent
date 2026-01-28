"""Planning activity - invokes the planning agent service via Dapr."""

from __future__ import annotations

import json
import os
from typing import Any

import requests


PLAN_SERVICE_APP_ID = os.environ.get("PLAN_SERVICE_APP_ID", "planner-agent-plan")
DAPR_HTTP_PORT = os.environ.get("DAPR_HTTP_PORT", "3500")
PLAN_TIMEOUT_SECONDS = 600


def run_planning(ctx, input_data: dict[str, Any]) -> dict[str, Any]:
    """Invoke the planning agent service via Dapr service invocation.

    The planning service runs the Claude Agent SDK in plan mode,
    reads the native task files, and returns them in the HTTP response.
    """
    workflow_id = input_data["workflow_id"]
    feature_request = input_data["feature_request"]
    cwd = input_data.get("cwd", os.getcwd())

    prompt = (
        f"Plan the implementation for the following feature request. "
        f"Create a detailed task list using TaskCreate with descriptions and dependencies.\n\n"
        f"Feature request: {feature_request}"
    )

    url = f"http://localhost:{DAPR_HTTP_PORT}/v1.0/invoke/{PLAN_SERVICE_APP_ID}/method/plan"

    try:
        response = requests.post(
            url,
            json={"prompt": prompt, "cwd": cwd, "workflow_id": workflow_id},
            headers={"dapr-app-timeout": str(PLAN_TIMEOUT_SECONDS)},
            timeout=PLAN_TIMEOUT_SECONDS + 30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.Timeout:
        return {
            "success": False,
            "error": f"Planning timed out after {PLAN_TIMEOUT_SECONDS} seconds",
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

    tasks = result.get("tasks", [])

    return {
        "success": True,
        "workflow_id": workflow_id,
        "task_count": len(tasks),
        "tasks": tasks,
    }
