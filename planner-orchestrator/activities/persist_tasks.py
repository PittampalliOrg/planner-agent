"""Persist tasks activity - saves tasks to Dapr statestore."""

from __future__ import annotations

import json
from typing import Any

from dapr.clients import DaprClient


STATESTORE_NAME = "statestore"


def persist_tasks(ctx, input_data: dict[str, Any]) -> dict[str, Any]:
    """Save tasks received from the planning service to the Dapr statestore."""
    workflow_id = input_data["workflow_id"]
    tasks = input_data.get("tasks", [])

    if not tasks:
        return {"success": True, "workflow_id": workflow_id, "count": 0, "tasks": []}

    with DaprClient() as client:
        client.save_state(
            store_name=STATESTORE_NAME,
            key=f"tasks:{workflow_id}",
            value=json.dumps(tasks),
        )

    return {
        "success": True,
        "workflow_id": workflow_id,
        "count": len(tasks),
        "tasks": tasks,
    }
