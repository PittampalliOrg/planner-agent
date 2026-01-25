"""
Task Management System for Planner Agent

Provides a task system with dependencies (blocks/blockedBy), status tracking,
and persistence - replicating Claude Code's task management capabilities.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    """Task status values matching Claude Code's task system."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """
    A task with dependency tracking.

    Attributes:
        id: Unique identifier for the task
        subject: Brief title for the task (imperative form, e.g., "Implement auth module")
        description: Detailed description of what needs to be done
        status: Current status (pending, in_progress, completed)
        active_form: Present continuous form shown during execution (e.g., "Implementing auth module")
        owner: Agent or entity responsible for the task
        blocks: List of task IDs that this task blocks (cannot start until this completes)
        blocked_by: List of task IDs that must complete before this task can start
        metadata: Arbitrary metadata attached to the task
        created_at: Timestamp when the task was created
        updated_at: Timestamp when the task was last updated
    """
    id: str
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    active_form: str | None = None
    owner: str | None = None
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_blocked(self) -> bool:
        """Check if this task is blocked by any incomplete tasks."""
        return len(self.blocked_by) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status.value,
            "active_form": self.active_form,
            "owner": self.owner,
            "blocks": self.blocks,
            "blocked_by": self.blocked_by,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a Task from a dictionary."""
        return cls(
            id=data["id"],
            subject=data["subject"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            active_form=data.get("active_form"),
            owner=data.get("owner"),
            blocks=data.get("blocks", []),
            blocked_by=data.get("blocked_by", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class TaskManager:
    """
    Manages a collection of tasks with dependency tracking.

    This class provides Claude Code-style task management with:
    - Task creation with automatic ID generation
    - Dependency management (blocks/blockedBy relationships)
    - Status tracking and transitions
    - Persistence to JSON files
    """

    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize the task manager.

        Args:
            storage_path: Path to store task data. If None, uses ./plans/tasks.json
        """
        self.tasks: dict[str, Task] = {}
        self.storage_path = Path(storage_path) if storage_path else Path("./plans/tasks.json")
        self._next_id = 1

    def create_task(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """
        Create a new task.

        Args:
            subject: Brief title (imperative form, e.g., "Implement auth module")
            description: Detailed description of what needs to be done
            active_form: Present continuous form (e.g., "Implementing auth module")
            owner: Agent or entity responsible
            metadata: Optional metadata to attach

        Returns:
            The newly created Task
        """
        task_id = str(self._next_id)
        self._next_id += 1

        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form or f"{subject}...",
            owner=owner,
            metadata=metadata or {},
        )

        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by its ID."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        """Get all tasks."""
        return list(self.tasks.values())

    def get_available_tasks(self) -> list[Task]:
        """
        Get tasks that are available to work on.

        A task is available if:
        - Status is 'pending'
        - Has no owner (not claimed)
        - All blockedBy dependencies are completed
        """
        available = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if task.owner:
                continue
            # Check if all blocking tasks are completed
            all_blockers_done = all(
                self.tasks.get(blocker_id) is None
                or self.tasks[blocker_id].status == TaskStatus.COMPLETED
                for blocker_id in task.blocked_by
            )
            if all_blockers_done:
                available.append(task)
        return available

    def update_task(
        self,
        task_id: str,
        *,
        subject: str | None = None,
        description: str | None = None,
        status: TaskStatus | str | None = None,
        active_form: str | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> Task | None:
        """
        Update an existing task.

        Args:
            task_id: ID of the task to update
            subject: New subject (optional)
            description: New description (optional)
            status: New status (optional)
            active_form: New active form (optional)
            owner: New owner (optional)
            metadata: Metadata to merge (keys with None values are deleted)
            add_blocks: Task IDs to add to blocks list
            add_blocked_by: Task IDs to add to blocked_by list

        Returns:
            The updated Task or None if not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return None

        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if status is not None:
            if isinstance(status, str):
                status = TaskStatus(status)
            task.status = status
            # When a task completes, remove it from other tasks' blocked_by lists
            if status == TaskStatus.COMPLETED:
                self._resolve_dependencies(task_id)
        if active_form is not None:
            task.active_form = active_form
        if owner is not None:
            task.owner = owner
        if metadata is not None:
            for key, value in metadata.items():
                if value is None:
                    task.metadata.pop(key, None)
                else:
                    task.metadata[key] = value
        if add_blocks:
            for blocked_id in add_blocks:
                if blocked_id not in task.blocks:
                    task.blocks.append(blocked_id)
                # Also update the blocked task's blocked_by list
                blocked_task = self.tasks.get(blocked_id)
                if blocked_task and task_id not in blocked_task.blocked_by:
                    blocked_task.blocked_by.append(task_id)
        if add_blocked_by:
            for blocker_id in add_blocked_by:
                if blocker_id not in task.blocked_by:
                    task.blocked_by.append(blocker_id)
                # Also update the blocker task's blocks list
                blocker_task = self.tasks.get(blocker_id)
                if blocker_task and task_id not in blocker_task.blocks:
                    blocker_task.blocks.append(task_id)

        task.updated_at = datetime.now().isoformat()
        return task

    def _resolve_dependencies(self, completed_task_id: str) -> None:
        """Remove a completed task from other tasks' blocked_by lists."""
        for task in self.tasks.values():
            if completed_task_id in task.blocked_by:
                task.blocked_by.remove(completed_task_id)

    def claim_task(self, task_id: str, owner: str) -> Task | None:
        """
        Claim a task for an owner and set it to in_progress.

        Args:
            task_id: ID of the task to claim
            owner: Name of the agent/entity claiming the task

        Returns:
            The claimed Task or None if not found/not available
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
        if task.status != TaskStatus.PENDING:
            return None
        if task.owner:
            return None

        task.owner = owner
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now().isoformat()
        return task

    def complete_task(self, task_id: str) -> Task | None:
        """
        Mark a task as completed.

        Args:
            task_id: ID of the task to complete

        Returns:
            The completed Task or None if not found
        """
        return self.update_task(task_id, status=TaskStatus.COMPLETED)

    def get_task_stats(self) -> dict[str, int]:
        """Get statistics about task statuses."""
        stats = {
            "total": len(self.tasks),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
        }
        for task in self.tasks.values():
            stats[task.status.value] += 1
        return stats

    def save(self, path: Path | str | None = None) -> None:
        """
        Save tasks to a JSON file.

        Args:
            path: Optional path override. Uses storage_path if not provided.
        """
        save_path = Path(path) if path else self.storage_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "tasks": [task.to_dict() for task in self.tasks.values()],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path | str | None = None) -> None:
        """
        Load tasks from a JSON file.

        Args:
            path: Optional path override. Uses storage_path if not provided.
        """
        load_path = Path(path) if path else self.storage_path

        if not load_path.exists():
            return

        with open(load_path, "r") as f:
            data = json.load(f)

        self._next_id = data.get("next_id", 1)
        self.tasks = {}
        for task_data in data.get("tasks", []):
            task = Task.from_dict(task_data)
            self.tasks[task.id] = task

    def clear(self) -> None:
        """Clear all tasks."""
        self.tasks = {}
        self._next_id = 1

    def format_task_list(self) -> str:
        """Format all tasks as a human-readable string."""
        if not self.tasks:
            return "No tasks."

        lines = []
        for task in self.tasks.values():
            status_icon = {
                TaskStatus.PENDING: "[ ]",
                TaskStatus.IN_PROGRESS: "[~]",
                TaskStatus.COMPLETED: "[x]",
            }[task.status]

            blocked_info = ""
            if task.blocked_by:
                blocked_info = f" (blocked by: {', '.join(task.blocked_by)})"

            owner_info = f" @{task.owner}" if task.owner else ""

            lines.append(f"{status_icon} {task.id}. {task.subject}{owner_info}{blocked_info}")

        return "\n".join(lines)
