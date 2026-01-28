"""Unified planner workflow - orchestrates planning, persistence, approval, and execution."""

from __future__ import annotations

import json
from datetime import timedelta

import dapr.ext.workflow as wf

from activities.planning import run_planning
from activities.persist_tasks import persist_tasks
from activities.execution import run_execution
from activities.publish_event import publish_event

wfr = wf.WorkflowRuntime()


@wfr.workflow(name="unified_planner_workflow")
def unified_planner_workflow(ctx: wf.DaprWorkflowContext, input_data: dict):
    """Single workflow: plan → persist → approve → execute.

    Phases:
      1. Planning:   Calls planning agent service to create tasks
      2. Persist:    Saves tasks to Dapr statestore
      3. Approval:   Waits for external event (human-in-the-loop gate)
      4. Execution:  Calls execution agent service to implement tasks

    Each phase transition publishes events to the workflow.stream pub/sub
    topic so the ai-chatbot UI can show real-time updates via SSE.
    """
    workflow_id = ctx.instance_id
    feature_request = input_data.get("feature_request", "")
    cwd = input_data.get("cwd", "")

    # --- Publish: workflow started ---
    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "initial",
        "data": {
            "status": "started",
            "metadata": {"feature_request": feature_request},
        },
    })

    # --- Phase 1: Planning ---
    ctx.set_custom_status(json.dumps({
        "phase": "planning",
        "progress": 10,
        "message": "Creating implementation plan...",
    }))

    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "task_progress",
        "data": {
            "status": "planning",
            "progress": 10,
            "metadata": {"phase": "planning"},
        },
    })

    planning_input = {
        "workflow_id": workflow_id,
        "feature_request": feature_request,
        "cwd": cwd,
    }
    planning_result = yield ctx.call_activity(run_planning, input=planning_input)

    if not planning_result.get("success"):
        error_msg = planning_result.get("error", "Unknown error")
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Planning failed: {error_msg}",
        }))
        yield ctx.call_activity(publish_event, input={
            "workflow_id": workflow_id,
            "event_type": "execution_failed",
            "data": {"error": f"Planning failed: {error_msg}"},
        })
        return {"success": False, "phase": "planning", "error": error_msg}

    # --- Phase 2: Persist tasks to statestore ---
    ctx.set_custom_status(json.dumps({
        "phase": "persisting",
        "progress": 30,
        "message": "Persisting tasks to statestore...",
    }))

    persist_input = {
        "workflow_id": workflow_id,
        "tasks": planning_result.get("tasks", []),
    }
    persist_result = yield ctx.call_activity(persist_tasks, input=persist_input)

    tasks = persist_result.get("tasks", [])

    ctx.set_custom_status(json.dumps({
        "phase": "awaiting_approval",
        "progress": 50,
        "message": f"Plan ready with {len(tasks)} tasks. Waiting for approval.",
        "task_count": len(tasks),
    }))

    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "task_progress",
        "data": {
            "status": "awaiting_approval",
            "progress": 50,
            "metadata": {"phase": "awaiting_approval", "task_count": len(tasks)},
        },
    })

    # --- Phase 3: Approval gate ---
    approval_event = ctx.wait_for_external_event(f"plan_approval_{workflow_id}")
    timeout_timer = ctx.create_timer(timedelta(hours=24))

    completed_task = yield wf.when_any([approval_event, timeout_timer])

    if completed_task == timeout_timer:
        ctx.set_custom_status(json.dumps({
            "phase": "timed_out",
            "progress": 0,
            "message": "Approval timed out after 24 hours",
        }))
        yield ctx.call_activity(publish_event, input={
            "workflow_id": workflow_id,
            "event_type": "execution_failed",
            "data": {"error": "Approval timed out after 24 hours"},
        })
        return {"success": False, "phase": "approval", "error": "Timed out waiting for approval"}

    approval = approval_event.get_result()
    if not approval or not approval.get("approved"):
        reason = approval.get("reason", "No reason provided") if approval else "No response"
        ctx.set_custom_status(json.dumps({
            "phase": "rejected",
            "progress": 0,
            "message": f"Plan rejected: {reason}",
        }))
        yield ctx.call_activity(publish_event, input={
            "workflow_id": workflow_id,
            "event_type": "execution_failed",
            "data": {"error": f"Plan rejected: {reason}"},
        })
        return {"success": False, "phase": "approval", "error": f"Plan rejected: {reason}"}

    # --- Phase 4: Execution ---
    ctx.set_custom_status(json.dumps({
        "phase": "executing",
        "progress": 60,
        "message": "Executing implementation tasks...",
    }))

    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "execution_started",
        "agent_id": "claude-code-agent",
        "data": {
            "status": "executing",
            "progress": 60,
            "metadata": {"phase": "executing", "task_count": len(tasks)},
        },
    })

    execution_input = {
        "workflow_id": workflow_id,
        "tasks": tasks,
        "cwd": cwd,
    }
    execution_result = yield ctx.call_activity(run_execution, input=execution_input)

    if not execution_result.get("success"):
        error_msg = execution_result.get("error", "Unknown error")
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Execution failed: {error_msg}",
        }))
        yield ctx.call_activity(publish_event, input={
            "workflow_id": workflow_id,
            "event_type": "execution_failed",
            "agent_id": "claude-code-agent",
            "data": {"error": f"Execution failed: {error_msg}"},
        })
        return {"success": False, "phase": "execution", "error": error_msg}

    ctx.set_custom_status(json.dumps({
        "phase": "completed",
        "progress": 100,
        "message": "Workflow completed successfully",
    }))

    yield ctx.call_activity(publish_event, input={
        "workflow_id": workflow_id,
        "event_type": "execution_completed",
        "agent_id": "claude-code-agent",
        "data": {
            "status": "completed",
            "progress": 100,
            "metadata": {"task_count": len(tasks)},
        },
    })

    return {
        "success": True,
        "workflow_id": workflow_id,
        "task_count": len(tasks),
        "tasks": tasks,
    }
