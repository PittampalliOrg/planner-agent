"""Dapr Agent workflow - orchestrates calling the planner-dapr-agent DurableAgent."""

from __future__ import annotations

import json

import dapr.ext.workflow as wf

from activities.dapr_agent import run_dapr_agent
from activities.publish_event import publish_event

# Use the same workflow runtime as the main planner workflow
from workflows.planner_workflow import wfr


@wfr.workflow(name="dapr_agent_workflow")
def dapr_agent_workflow(ctx: wf.DaprWorkflowContext, input_data: dict):
    """Simple workflow that invokes the planner-dapr-agent DurableAgent.

    This workflow:
      1. Publishes a start event
      2. Calls the DurableAgent via Dapr service invocation
      3. Publishes the result

    The DurableAgent internally handles:
      - Conversation memory (via Dapr state)
      - Tool execution (file reading, code search, task creation)
      - LLM interactions (via Anthropic Claude)
    """
    workflow_id = ctx.instance_id
    prompt = input_data.get("prompt", "")
    cwd = input_data.get("cwd", "")

    # --- Publish: workflow started ---
    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "initial",
        "data": {
            "status": "started",
            "metadata": {"prompt": prompt[:200], "agent_type": "dapr_agent"},
        },
    })

    # --- Phase 1: Run the DurableAgent ---
    ctx.set_custom_status(json.dumps({
        "phase": "running_agent",
        "progress": 10,
        "message": "Running DurableAgent...",
    }))

    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "task_progress",
        "data": {
            "status": "running_agent",
            "progress": 10,
            "metadata": {"phase": "running_agent"},
        },
    })

    agent_input = {
        "workflow_id": workflow_id,
        "prompt": prompt,
        "cwd": cwd,
    }
    agent_result = yield ctx.call_activity(run_dapr_agent, input=agent_input)

    if not agent_result.get("success"):
        error_msg = agent_result.get("error", "Unknown error")
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Agent failed: {error_msg}",
        }))
        yield ctx.call_activity(publish_event, input={
            "workflow_id": workflow_id,
            "event_type": "execution_failed",
            "data": {"error": f"Agent failed: {error_msg}"},
        })
        return {"success": False, "phase": "agent", "error": error_msg}

    # --- Completed ---
    tasks = agent_result.get("tasks", [])
    response = agent_result.get("response", "")
    session_id = agent_result.get("session_id", "")

    ctx.set_custom_status(json.dumps({
        "phase": "completed",
        "progress": 100,
        "message": f"Agent completed with {len(tasks)} tasks",
        "task_count": len(tasks),
    }))

    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "execution_completed",
        "agent_id": "planner-dapr-agent",
        "data": {
            "status": "completed",
            "progress": 100,
            "metadata": {
                "task_count": len(tasks),
                "session_id": session_id,
            },
        },
    })

    return {
        "success": True,
        "workflow_id": workflow_id,
        "session_id": session_id,
        "response": response,
        "tasks": tasks,
        "task_count": len(tasks),
    }
