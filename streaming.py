"""
Streaming Module for Execution Events

Publishes execution events to Dapr pub/sub for real-time streaming to the UI.
Events flow: Python -> Dapr pub/sub -> webhook -> SSE -> UI
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal

import httpx


# =============================================================================
# Configuration
# =============================================================================

DAPR_HTTP_PORT = os.getenv("DAPR_HTTP_PORT", "3500")
PUBSUB_NAME = os.getenv("PUBSUB_NAME", "pubsub")
STREAM_TOPIC = "workflow.stream"


# =============================================================================
# Event Types
# =============================================================================

EventType = Literal[
    "execution_started",
    "task_started",
    "task_completed",
    "task_failed",
    "tool_call",
    "tool_result",
    "file_changed",
    "execution_completed",
    "execution_failed",
    "llm_chunk",
    "thinking",  # Claude's extended thinking (internal reasoning)
    "phase_changed",
    "native_task_created",
    "native_task_updated",
    "clarification_requested",
    "plan_ready",
]


# =============================================================================
# Event Publishing
# =============================================================================

async def publish_execution_event(
    workflow_id: str,
    event_type: EventType,
    data: dict[str, Any],
) -> bool:
    """
    Publish an execution event to Dapr pub/sub for real-time streaming.

    Args:
        workflow_id: The Dapr workflow instance ID
        event_type: Type of execution event
        data: Event payload data

    Returns:
        True if published successfully, False otherwise
    """
    url = f"http://localhost:{DAPR_HTTP_PORT}/v1.0/publish/{PUBSUB_NAME}/{STREAM_TOPIC}"

    event = {
        "type": event_type,
        "workflowId": workflow_id,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=event,
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )

            if response.status_code in (200, 204):
                print(f"[Streaming] Published {event_type} event for workflow {workflow_id}")
                return True
            else:
                print(f"[Streaming] Failed to publish event: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        # Don't fail execution if streaming fails - log and continue
        print(f"[Streaming] Error publishing event (non-fatal): {e}")
        return False


def publish_execution_event_sync(
    workflow_id: str,
    event_type: EventType,
    data: dict[str, Any],
) -> bool:
    """
    Synchronous version of publish_execution_event for use in workflow context.

    Args:
        workflow_id: The Dapr workflow instance ID
        event_type: Type of execution event
        data: Event payload data

    Returns:
        True if published successfully, False otherwise
    """
    url = f"http://localhost:{DAPR_HTTP_PORT}/v1.0/publish/{PUBSUB_NAME}/{STREAM_TOPIC}"

    event = {
        "type": event_type,
        "workflowId": workflow_id,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        response = httpx.post(
            url,
            json=event,
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )

        if response.status_code in (200, 204):
            print(f"[Streaming] Published {event_type} event for workflow {workflow_id}")
            return True
        else:
            print(f"[Streaming] Failed to publish event: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        # Don't fail execution if streaming fails - log and continue
        print(f"[Streaming] Error publishing event (non-fatal): {e}")
        return False


async def stream_tool_call(
    workflow_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    task_id: str | None = None,
    call_id: str | None = None,
) -> bool:
    """Stream a tool call event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="tool_call",
        data={
            "toolName": tool_name,
            "toolInput": tool_input,
            "taskId": task_id,
            "callId": call_id,
        },
    )


async def stream_tool_result(
    workflow_id: str,
    tool_name: str,
    result: str,
    is_error: bool = False,
    task_id: str | None = None,
    call_id: str | None = None,
) -> bool:
    """Stream a tool result event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="tool_result",
        data={
            "toolName": tool_name,
            "result": result[:500] if len(result) > 500 else result,  # Truncate large results
            "isError": is_error,
            "taskId": task_id,
            "callId": call_id,
        },
    )


async def stream_task_progress(
    workflow_id: str,
    task_id: str,
    status: Literal["started", "completed", "failed"],
    task_subject: str,
    error: str | None = None,
) -> bool:
    """Stream a task progress event."""
    event_type: EventType = f"task_{status}"  # type: ignore
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type=event_type,
        data={
            "taskId": task_id,
            "subject": task_subject,
            "error": error,
        },
    )


async def stream_file_changed(
    workflow_id: str,
    file_path: str,
    operation: Literal["create", "modify", "delete"],
    task_id: str | None = None,
) -> bool:
    """Stream a file changed event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="file_changed",
        data={
            "filePath": file_path,
            "operation": operation,
            "taskId": task_id,
        },
    )


async def stream_execution_started(
    workflow_id: str,
    plan_id: str,
    total_tasks: int,
) -> bool:
    """Stream execution started event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="execution_started",
        data={
            "planId": plan_id,
            "totalTasks": total_tasks,
        },
    )


async def stream_execution_completed(
    workflow_id: str,
    tasks_completed: int,
    files_changed: list[str],
) -> bool:
    """Stream execution completed event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="execution_completed",
        data={
            "tasksCompleted": tasks_completed,
            "filesChanged": files_changed,
        },
    )


async def stream_execution_failed(
    workflow_id: str,
    error: str,
    tasks_completed: int,
) -> bool:
    """Stream execution failed event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="execution_failed",
        data={
            "error": error,
            "tasksCompleted": tasks_completed,
        },
    )


async def stream_llm_chunk(
    workflow_id: str,
    text: str,
    task_id: str | None = None,
) -> bool:
    """Stream LLM text chunk event."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="llm_chunk",
        data={
            "text": text[:1000] if len(text) > 1000 else text,  # Truncate very long chunks
            "taskId": task_id,
        },
    )


async def stream_thinking(
    workflow_id: str,
    text: str,
    task_id: str | None = None,
) -> bool:
    """Stream a thinking block event (Claude's internal reasoning)."""
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="thinking",
        data={
            "text": text[:5000] if len(text) > 5000 else text,  # Truncate very long thinking
            "taskId": task_id,
        },
    )


async def stream_phase_changed(
    workflow_id: str,
    phase: str,
    status: str,
    progress: int,
    plan_id: str | None = None,
    extra_data: dict[str, Any] | None = None,
) -> bool:
    """Stream phase change event for UI detection."""
    data = {
        "phase": phase,
        "status": status,
        "progress": progress,
    }
    if plan_id:
        data["planId"] = plan_id
    if extra_data:
        data.update(extra_data)

    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="phase_changed",
        data=data,
    )


def stream_phase_changed_sync(
    workflow_id: str,
    phase: str,
    status: str,
    progress: int,
    plan_id: str | None = None,
    extra_data: dict[str, Any] | None = None,
) -> bool:
    """Synchronous version of stream_phase_changed for workflow context."""
    data = {
        "phase": phase,
        "status": status,
        "progress": progress,
    }
    if plan_id:
        data["planId"] = plan_id
    if extra_data:
        data.update(extra_data)

    return publish_execution_event_sync(
        workflow_id=workflow_id,
        event_type="phase_changed",
        data=data,
    )


# =============================================================================
# Native Task Events (Claude Code's TaskCreate/TaskUpdate tracking)
# =============================================================================

async def stream_native_task_created(
    workflow_id: str,
    subject: str,
    description: str,
    task_id: str | None = None,
) -> bool:
    """
    Stream event when Claude uses TaskCreate to create a native task.

    Passes through native tool data as-is for UI display.
    """
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="native_task_created",
        data={
            "subject": subject,
            "description": description[:500] if len(description) > 500 else description,
            "taskId": task_id,
        },
    )


async def stream_native_task_updated(
    workflow_id: str,
    task_id: str,
    status: str,
    subject: str | None = None,
) -> bool:
    """
    Stream event when Claude uses TaskUpdate to update a native task.

    Passes through native tool data as-is for UI display.
    """
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="native_task_updated",
        data={
            "taskId": task_id,
            "status": status,
            "subject": subject,
        },
    )


def stream_native_task_created_sync(
    workflow_id: str,
    subject: str,
    description: str,
    task_id: str | None = None,
) -> bool:
    """Synchronous version for workflow context."""
    return publish_execution_event_sync(
        workflow_id=workflow_id,
        event_type="native_task_created",
        data={
            "subject": subject,
            "description": description[:500] if len(description) > 500 else description,
            "taskId": task_id,
        },
    )


def stream_native_task_updated_sync(
    workflow_id: str,
    task_id: str,
    status: str,
    subject: str | None = None,
) -> bool:
    """Synchronous version for workflow context."""
    return publish_execution_event_sync(
        workflow_id=workflow_id,
        event_type="native_task_updated",
        data={
            "taskId": task_id,
            "status": status,
            "subject": subject,
        },
    )


# =============================================================================
# Plan Mode Events (Clarification and Plan Ready)
# =============================================================================

async def stream_clarification_requested(
    workflow_id: str,
    questions: list[dict],
    question_id: str,
    plan_id: str,
) -> bool:
    """
    Stream event when Claude uses AskUserQuestion in plan mode.

    This signals that planning is paused waiting for user clarification.
    """
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="clarification_requested",
        data={
            "questionId": question_id,
            "questions": questions,
            "planId": plan_id,
        },
    )


async def stream_plan_ready(
    workflow_id: str,
    plan_id: str,
    plan_content: str,
    critical_files: list[str],
    tasks_created: int,
) -> bool:
    """
    Stream event when Claude uses ExitPlanMode, signaling plan is ready.

    This signals that planning is complete and ready for user approval.
    """
    return await publish_execution_event(
        workflow_id=workflow_id,
        event_type="plan_ready",
        data={
            "planId": plan_id,
            "planContent": plan_content[:2000] if plan_content else "",
            "criticalFiles": critical_files,
            "tasksCreated": tasks_created,
        },
    )


def stream_clarification_requested_sync(
    workflow_id: str,
    questions: list[dict],
    question_id: str,
    plan_id: str,
) -> bool:
    """Synchronous version for workflow context."""
    return publish_execution_event_sync(
        workflow_id=workflow_id,
        event_type="clarification_requested",
        data={
            "questionId": question_id,
            "questions": questions,
            "planId": plan_id,
        },
    )


def stream_plan_ready_sync(
    workflow_id: str,
    plan_id: str,
    plan_content: str,
    critical_files: list[str],
    tasks_created: int,
) -> bool:
    """Synchronous version for workflow context."""
    return publish_execution_event_sync(
        workflow_id=workflow_id,
        event_type="plan_ready",
        data={
            "planId": plan_id,
            "planContent": plan_content[:2000] if plan_content else "",
            "criticalFiles": critical_files,
            "tasksCreated": tasks_created,
        },
    )
