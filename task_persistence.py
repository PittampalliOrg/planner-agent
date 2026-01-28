"""
Task Persistence Module - Passive Observer

Records native Claude Code task tool events (TaskCreate, TaskUpdate, TaskList)
as-is to JSON files for cross-session durability.

This module does NOT implement task logic. The native Claude Code SDK manages:
- Task creation and ID assignment
- Dependency tracking (addBlockedBy, addBlocks)
- Dependency resolution on completion
- Task state transitions

This module simply persists snapshots of what the native tools do,
so state survives across Dapr workflow activities.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


class TaskStore:
    """
    Passive persistence layer for native Claude Code task events.

    Records native tool call inputs as-is to JSON files.
    Does NOT implement any task logic — the native SDK handles all of that.

    File structure:
        workspace/tasks/{workflow_id}/
        ├── _index.json          # Summary snapshot
        ├── _events.jsonl        # Append-only event journal
        ├── task-1.json          # Latest snapshot per task
        ├── task-2.json
        └── task-3.json
    """

    def __init__(
        self,
        base_path: str | Path,
        workflow_id: str | None = None,
        plan_id: str | None = None,
    ):
        self.base_path = Path(base_path)
        self.workflow_id = workflow_id or "default"
        self.plan_id = plan_id
        self.tasks_dir = self.base_path / self.workflow_id
        self._next_local_seq = 1
        self._ensure_directory()
        self._load_next_seq()

    def _ensure_directory(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _load_next_seq(self) -> None:
        """Load next local sequence number from existing files."""
        max_seq = 0
        for task_file in self.tasks_dir.glob("task-*.json"):
            try:
                seq = int(task_file.stem.replace("task-", ""))
                max_seq = max(max_seq, seq)
            except ValueError:
                continue
        self._next_local_seq = max_seq + 1

    # -------------------------------------------------------------------------
    # Event Journal — append-only log of native tool calls
    # -------------------------------------------------------------------------

    def _append_event(self, event: dict) -> None:
        """Append an event to the journal (events.jsonl)."""
        events_path = self.tasks_dir / "_events.jsonl"
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"
        event["workflow_id"] = self.workflow_id
        event["plan_id"] = self.plan_id
        with open(events_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # -------------------------------------------------------------------------
    # Record native tool calls — pass-through, no business logic
    # -------------------------------------------------------------------------

    def record_task_created(
        self,
        tool_input: dict,
        tool_call_id: str | None = None,
    ) -> dict:
        """
        Record a native TaskCreate/TodoWrite tool call.

        Saves the tool_input as-is. Assigns a local sequence number
        for file naming only (not overriding the SDK's internal ID).

        Returns:
            The recorded task snapshot dict.
        """
        now = datetime.utcnow().isoformat() + "Z"
        local_seq = str(self._next_local_seq)
        self._next_local_seq += 1

        # Pass through native input faithfully
        snapshot = {
            "local_seq": local_seq,
            "tool_call_id": tool_call_id,
            "tool_name": "TaskCreate",
            "tool_input": tool_input,
            "workflow_id": self.workflow_id,
            "plan_id": self.plan_id,
            "recorded_at": now,
        }

        # Save snapshot
        self._atomic_write(self.tasks_dir / f"task-{local_seq}.json", snapshot)

        # Journal
        self._append_event({
            "event": "task_created",
            "local_seq": local_seq,
            "tool_call_id": tool_call_id,
            "tool_input": tool_input,
        })

        self._update_index()
        print(f"[TaskStore] Recorded TaskCreate (seq {local_seq}): {tool_input.get('subject', '')}")
        return snapshot

    def record_task_updated(
        self,
        tool_input: dict,
        tool_call_id: str | None = None,
    ) -> dict | None:
        """
        Record a native TaskUpdate/TodoUpdate tool call.

        Merges the tool_input into the existing task snapshot.
        Passes through addBlockedBy, addBlocks, status, etc. as-is.
        Does NOT interpret or process dependencies — the native SDK does that.

        Returns:
            The updated snapshot dict, or None if task not found.
        """
        task_id = tool_input.get("taskId")
        if not task_id:
            return None

        # Find the snapshot file for this task
        snapshot = self._load_snapshot_by_native_id(task_id)
        if snapshot is None:
            # Task might have been created by the SDK without us seeing it,
            # or the ID mapping differs. Record as a new event anyway.
            snapshot = self._load_snapshot_by_seq(task_id)

        if snapshot is None:
            # Record as orphan update event
            self._append_event({
                "event": "task_updated",
                "task_id": task_id,
                "tool_call_id": tool_call_id,
                "tool_input": tool_input,
                "note": "no_matching_snapshot",
            })
            print(f"[TaskStore] Recorded TaskUpdate for unknown task {task_id}")
            return None

        # Merge update into snapshot — pass through all fields as-is
        now = datetime.utcnow().isoformat() + "Z"
        if "updates" not in snapshot:
            snapshot["updates"] = []
        snapshot["updates"].append({
            "tool_call_id": tool_call_id,
            "tool_input": tool_input,
            "recorded_at": now,
        })

        # Save updated snapshot
        local_seq = snapshot.get("local_seq", task_id)
        self._atomic_write(self.tasks_dir / f"task-{local_seq}.json", snapshot)

        # Journal
        self._append_event({
            "event": "task_updated",
            "local_seq": local_seq,
            "task_id": task_id,
            "tool_call_id": tool_call_id,
            "tool_input": tool_input,
        })

        self._update_index()
        print(f"[TaskStore] Recorded TaskUpdate (task {task_id})")
        return snapshot

    def record_tool_result(
        self,
        tool_name: str,
        tool_call_id: str | None,
        result_content: str,
        is_error: bool,
    ) -> None:
        """
        Record a tool result for correlation with the originating call.
        """
        self._append_event({
            "event": "tool_result",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result_content": result_content[:1000],
            "is_error": is_error,
        })

    # -------------------------------------------------------------------------
    # Read operations
    # -------------------------------------------------------------------------

    def list_tasks(self) -> list[dict]:
        """List all task snapshots from disk."""
        tasks = []
        for task_file in self.tasks_dir.glob("task-*.json"):
            try:
                with open(task_file, "r") as f:
                    tasks.append(json.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[TaskStore] Error loading {task_file}: {e}")
                continue

        # Sort by local sequence number
        tasks.sort(key=lambda t: int(t.get("local_seq", 0)))
        return tasks

    def get_task(self, task_id: str) -> dict | None:
        """Get a task snapshot by local sequence or native task ID."""
        # Try direct local_seq lookup first
        snapshot = self._load_snapshot_by_seq(task_id)
        if snapshot:
            return snapshot
        # Try native ID lookup
        return self._load_snapshot_by_native_id(task_id)

    def get_events(self) -> list[dict]:
        """Read all events from the journal."""
        events_path = self.tasks_dir / "_events.jsonl"
        if not events_path.exists():
            return []

        events = []
        with open(events_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def get_index(self) -> dict | None:
        """Get the index file data."""
        index_path = self.tasks_dir / "_index.json"
        if not index_path.exists():
            return None
        with open(index_path, "r") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_snapshot_by_seq(self, local_seq: str) -> dict | None:
        """Load snapshot by local sequence number."""
        path = self.tasks_dir / f"task-{local_seq}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _load_snapshot_by_native_id(self, native_id: str) -> dict | None:
        """Load snapshot by searching for a matching native task ID in tool_input."""
        for task_file in self.tasks_dir.glob("task-*.json"):
            try:
                with open(task_file, "r") as f:
                    snapshot = json.load(f)
                # Check if any update references this native ID
                task_input = snapshot.get("tool_input", {})
                if task_input.get("taskId") == native_id:
                    return snapshot
                # Check updates for taskId match
                for update in snapshot.get("updates", []):
                    if update.get("tool_input", {}).get("taskId") == native_id:
                        return snapshot
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def _update_index(self) -> None:
        """Update the index file with a summary of recorded tasks."""
        tasks = self.list_tasks()

        task_summaries = []
        for task in tasks:
            tool_input = task.get("tool_input", {})
            # Extract last known status from updates
            status = "pending"
            for update in task.get("updates", []):
                s = update.get("tool_input", {}).get("status")
                if s:
                    status = s

            task_summaries.append({
                "local_seq": task.get("local_seq"),
                "subject": tool_input.get("subject", tool_input.get("content", "")),
                "status": status,
            })

        index_data = {
            "workflow_id": self.workflow_id,
            "plan_id": self.plan_id,
            "total": len(tasks),
            "tasks": task_summaries,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        self._atomic_write(self.tasks_dir / "_index.json", index_data)

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write JSON atomically via temp-file-then-rename."""
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            dir=self.tasks_dir,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
