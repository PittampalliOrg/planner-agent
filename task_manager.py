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


class EventType(str, Enum):
    """Types of task execution events."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRIED = "retried"
    BLOCKED = "blocked"
    UNBLOCKED = "unblocked"


@dataclass
class TaskEvent:
    """A single execution event for a task."""
    event_type: EventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEvent:
        """Create a TaskEvent from a dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            duration_ms=data.get("duration_ms"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskMetrics:
    """Computed metrics for a single task."""
    total_duration_ms: float
    attempt_count: int
    first_started_at: str | None
    last_completed_at: str | None
    error_count: int
    blocked_duration_ms: float


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
    events: list[TaskEvent] = field(default_factory=list)
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
            "events": [e.to_dict() for e in self.events],
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
            events=[TaskEvent.from_dict(e) for e in data.get("events", [])],
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
            old_status = task.status
            task.status = status
            self._record_status_event(task, old_status, status)
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
        old_status = task.status
        task.status = TaskStatus.IN_PROGRESS
        self._record_status_event(task, old_status, TaskStatus.IN_PROGRESS)
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

    def _record_status_event(
        self,
        task: Task,
        old_status: TaskStatus,
        new_status: TaskStatus,
    ) -> None:
        """Append a TaskEvent based on a status transition."""
        now = datetime.now().isoformat()

        if new_status == TaskStatus.IN_PROGRESS and old_status == TaskStatus.PENDING:
            task.events.append(TaskEvent(event_type=EventType.STARTED, timestamp=now))
        elif new_status == TaskStatus.IN_PROGRESS and old_status == TaskStatus.COMPLETED:
            task.events.append(TaskEvent(event_type=EventType.RETRIED, timestamp=now))
        elif new_status == TaskStatus.COMPLETED:
            duration_ms = self._compute_duration_ms(task)
            task.events.append(TaskEvent(
                event_type=EventType.COMPLETED,
                timestamp=now,
                duration_ms=duration_ms,
            ))
        elif new_status == TaskStatus.PENDING and old_status == TaskStatus.IN_PROGRESS:
            task.events.append(TaskEvent(event_type=EventType.FAILED, timestamp=now))

    @staticmethod
    def _compute_duration_ms(task: Task) -> float | None:
        """Compute ms elapsed since the last started/retried event."""
        for event in reversed(task.events):
            if event.event_type in (EventType.STARTED, EventType.RETRIED):
                try:
                    start = datetime.fromisoformat(event.timestamp)
                    end = datetime.now()
                    return (end - start).total_seconds() * 1000
                except (ValueError, TypeError):
                    return None
        return None

    def record_task_event(
        self,
        task_id: str,
        event_type: EventType | str,
        *,
        duration_ms: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskEvent | None:
        """Manually record an event on a task."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        event = TaskEvent(
            event_type=event_type,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )
        task.events.append(event)
        return event

    def get_task_metrics(self, task_id: str) -> TaskMetrics | None:
        """Compute metrics for a single task from its event history."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        events = task.events
        total_duration_ms: float = 0.0
        attempt_count = 0
        first_started_at: str | None = None
        last_completed_at: str | None = None
        error_count = 0
        blocked_duration_ms: float = 0.0

        last_blocked_at: str | None = None

        for ev in events:
            if ev.event_type in (EventType.STARTED, EventType.RETRIED):
                attempt_count += 1
                if first_started_at is None:
                    first_started_at = ev.timestamp
            elif ev.event_type == EventType.COMPLETED:
                last_completed_at = ev.timestamp
                if ev.duration_ms is not None:
                    total_duration_ms += ev.duration_ms
            elif ev.event_type == EventType.FAILED:
                error_count += 1
            elif ev.event_type == EventType.BLOCKED:
                last_blocked_at = ev.timestamp
            elif ev.event_type == EventType.UNBLOCKED:
                if last_blocked_at:
                    try:
                        b_start = datetime.fromisoformat(last_blocked_at)
                        b_end = datetime.fromisoformat(ev.timestamp)
                        blocked_duration_ms += (b_end - b_start).total_seconds() * 1000
                    except (ValueError, TypeError):
                        pass
                    last_blocked_at = None

        return TaskMetrics(
            total_duration_ms=total_duration_ms,
            attempt_count=attempt_count,
            first_started_at=first_started_at,
            last_completed_at=last_completed_at,
            error_count=error_count,
            blocked_duration_ms=blocked_duration_ms,
        )

    def get_plan_metrics(self) -> dict[str, Any]:
        """Return aggregate metrics across all tasks."""
        total = len(self.tasks)
        completed = 0
        failed = 0
        durations: list[float] = []
        longest_task: str | None = None
        longest_duration: float = 0.0
        earliest_start: str | None = None
        latest_end: str | None = None

        for task in self.tasks.values():
            metrics = self.get_task_metrics(task.id)
            if metrics is None:
                continue
            if task.status == TaskStatus.COMPLETED:
                completed += 1
            if metrics.error_count > 0:
                failed += 1
            if metrics.total_duration_ms > 0:
                durations.append(metrics.total_duration_ms)
                if metrics.total_duration_ms > longest_duration:
                    longest_duration = metrics.total_duration_ms
                    longest_task = task.id
            if metrics.first_started_at:
                if earliest_start is None or metrics.first_started_at < earliest_start:
                    earliest_start = metrics.first_started_at
            if metrics.last_completed_at:
                if latest_end is None or metrics.last_completed_at > latest_end:
                    latest_end = metrics.last_completed_at

        avg_duration = sum(durations) / len(durations) if durations else 0.0

        total_wall_clock_ms: float = 0.0
        if earliest_start and latest_end:
            try:
                wall_start = datetime.fromisoformat(earliest_start)
                wall_end = datetime.fromisoformat(latest_end)
                total_wall_clock_ms = (wall_end - wall_start).total_seconds() * 1000
            except (ValueError, TypeError):
                pass

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "avg_duration_ms": avg_duration,
            "longest_task_id": longest_task,
            "longest_duration_ms": longest_duration,
            "total_wall_clock_ms": total_wall_clock_ms,
        }

    def get_task_timeline(self, task_id: str) -> str:
        """Return a formatted string showing the event timeline for a task."""
        task = self.tasks.get(task_id)
        if not task:
            return f"Task {task_id} not found."
        if not task.events:
            return f"Task {task_id} ({task.subject}): no events recorded."

        lines = [f"Timeline for task {task_id}: {task.subject}"]
        for ev in task.events:
            parts = [f"  [{ev.timestamp}] {ev.event_type.value}"]
            if ev.duration_ms is not None:
                parts.append(f"duration={ev.duration_ms:.1f}ms")
            if ev.error:
                parts.append(f"error={ev.error!r}")
            if ev.metadata:
                parts.append(f"metadata={ev.metadata}")
            lines.append("  ".join(parts))
        return "\n".join(lines)

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
