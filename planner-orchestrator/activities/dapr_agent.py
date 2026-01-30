"""Dapr Agent activity - invokes the planner-dapr-agent DurableAgent via Dapr."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests


DAPR_AGENT_APP_ID = os.environ.get("DAPR_AGENT_APP_ID", "planner-dapr-agent")
DAPR_HTTP_PORT = os.environ.get("DAPR_HTTP_PORT", "3500")
AGENT_TIMEOUT_SECONDS = 600  # 10 minutes
POLL_INTERVAL_SECONDS = 5  # How often to check workflow status

logger = logging.getLogger(__name__)


def run_dapr_agent(ctx, input_data: dict[str, Any]) -> dict[str, Any]:
    """Invoke the planner-dapr-agent DurableAgent via Dapr service invocation.

    The DurableAgent runs asynchronously with workflow durability.
    This activity:
      1. Starts the workflow via POST /run
      2. Polls GET /run/{instance_id} until completion
      3. Returns the workflow result with tasks
    """
    workflow_id = input_data.get("workflow_id", "")
    prompt = input_data.get("prompt", "")
    cwd = input_data.get("cwd", "")

    logger.info(f"Invoking planner-dapr-agent DurableAgent for workflow {workflow_id}")
    logger.info(f"Prompt: {prompt[:200]}...")

    base_url = f"http://localhost:{DAPR_HTTP_PORT}/v1.0/invoke/{DAPR_AGENT_APP_ID}/method"

    # Step 1: Start the workflow
    try:
        response = requests.post(
            f"{base_url}/run",
            json={"message": prompt},
            headers={"dapr-app-timeout": "60"},
            timeout=120,
        )
        response.raise_for_status()
        start_result = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to start DurableAgent workflow: {e}")
        return {
            "success": False,
            "error": f"Failed to start workflow: {str(e)[:1000]}",
            "workflow_id": workflow_id,
        }

    instance_id = start_result.get("instance_id")
    if not instance_id:
        logger.error(f"No instance_id in response: {start_result}")
        return {
            "success": False,
            "error": "No instance_id returned from /run endpoint",
            "workflow_id": workflow_id,
        }

    logger.info(f"DurableAgent workflow started: instance_id={instance_id}")

    # Step 2: Poll for completion
    status_url = f"{base_url}/run/{instance_id}"
    start_time = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start_time
        if elapsed > AGENT_TIMEOUT_SECONDS:
            logger.error(f"DurableAgent workflow timed out after {elapsed:.0f}s")
            return {
                "success": False,
                "error": f"Workflow timed out after {AGENT_TIMEOUT_SECONDS} seconds",
                "workflow_id": workflow_id,
                "instance_id": instance_id,
            }

        try:
            response = requests.get(
                status_url,
                headers={"dapr-app-timeout": "30"},
                timeout=60,
            )
            response.raise_for_status()
            status = response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to get workflow status (will retry): {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        runtime_status = status.get("runtime_status", "UNKNOWN")

        if runtime_status != last_status:
            logger.info(f"DurableAgent workflow status: {runtime_status} (elapsed: {elapsed:.0f}s)")
            last_status = runtime_status

        if runtime_status == "COMPLETED":
            logger.info(f"DurableAgent workflow completed in {elapsed:.0f}s")
            break
        elif runtime_status in ("FAILED", "TERMINATED"):
            error_msg = status.get("failure_details", {}).get("message", "Unknown error")
            logger.error(f"DurableAgent workflow {runtime_status}: {error_msg}")
            return {
                "success": False,
                "error": f"Workflow {runtime_status}: {error_msg}",
                "workflow_id": workflow_id,
                "instance_id": instance_id,
            }

        time.sleep(POLL_INTERVAL_SECONDS)

    # Step 3: Extract result
    # The workflow output is in status["serialized_output"] as a JSON string
    output_str = status.get("serialized_output", "{}")
    try:
        import json
        output = json.loads(output_str) if isinstance(output_str, str) else output_str
    except (json.JSONDecodeError, TypeError):
        output = {"raw_output": output_str}

    # Extract tasks from the output
    # The DurableAgent's response may contain tasks in different formats
    tasks = output.get("tasks", [])
    response_text = output.get("response", str(output))

    logger.info(f"DurableAgent completed: {len(tasks)} tasks, instance_id={instance_id}")

    return {
        "success": True,
        "workflow_id": workflow_id,
        "instance_id": instance_id,
        "response": response_text,
        "tasks": tasks,
        "task_count": len(tasks),
    }
