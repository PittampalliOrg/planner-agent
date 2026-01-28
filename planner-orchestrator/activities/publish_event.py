"""Publish event activity - publishes workflow events to Dapr pub/sub."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from dapr.clients import DaprClient


logger = logging.getLogger(__name__)

PUBSUB_NAME = os.environ.get("PUBSUB_NAME", "pubsub")
PUBSUB_TOPIC = os.environ.get("PUBSUB_TOPIC", "workflow.stream")


def publish_event(ctx, input_data: dict[str, Any]) -> dict[str, Any]:
    """Publish a workflow stream event to the Dapr pub/sub topic.

    This activity is called at key workflow phase transitions to push
    real-time updates through the SSE pipeline:
      orchestrator → pub/sub → ai-chatbot webhook → Redis → SSE → browser

    Event format matches ai-chatbot's WorkflowStreamEvent interface.
    """
    workflow_id = input_data["workflow_id"]
    event_type = input_data["event_type"]
    data = input_data.get("data", {})
    task_id = input_data.get("task_id")
    agent_id = input_data.get("agent_id", "claude-planner")

    event = {
        "id": f"orch-{workflow_id}-{uuid.uuid4().hex[:8]}",
        "type": event_type,
        "workflowId": workflow_id,
        "agentId": agent_id,
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
        logger.info(
            f"Published {event_type} event for workflow {workflow_id} "
            f"to {PUBSUB_NAME}/{PUBSUB_TOPIC}"
        )
        return {"success": True, "event_id": event["id"]}
    except Exception as e:
        # Non-fatal: event publishing failure shouldn't break the workflow
        logger.warning(
            f"Failed to publish {event_type} event for workflow {workflow_id}: {e}"
        )
        return {"success": False, "error": str(e)}
