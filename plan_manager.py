"""
Plan Manager for Planner Agent

Handles plan creation, persistence, and formatting - replicating
Claude Code's plan mode functionality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from task_manager import TaskManager, Task, TaskStatus


@dataclass
class PlanStep:
    """A single step in an implementation plan."""
    number: int
    title: str
    description: str
    files_affected: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high


@dataclass
class Plan:
    """
    An implementation plan for a feature request.

    Attributes:
        id: Unique identifier for the plan
        title: Brief title of the feature/change
        summary: High-level summary of what will be implemented
        context: Background context gathered during exploration
        steps: Ordered list of implementation steps
        critical_files: Key files that will be modified
        considerations: Architectural considerations or trade-offs
        status: Current plan status (draft, approved, implementing, completed)
        created_at: When the plan was created
        approved_at: When the plan was approved (if applicable)
    """
    id: str
    title: str
    summary: str
    context: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    critical_files: list[str] = field(default_factory=list)
    considerations: list[str] = field(default_factory=list)
    status: str = "draft"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    approved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "context": self.context,
            "steps": [
                {
                    "number": s.number,
                    "title": s.title,
                    "description": s.description,
                    "files_affected": s.files_affected,
                    "estimated_complexity": s.estimated_complexity,
                }
                for s in self.steps
            ],
            "critical_files": self.critical_files,
            "considerations": self.considerations,
            "status": self.status,
            "created_at": self.created_at,
            "approved_at": self.approved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Create a Plan from a dictionary."""
        steps = [
            PlanStep(
                number=s["number"],
                title=s["title"],
                description=s["description"],
                files_affected=s.get("files_affected", []),
                estimated_complexity=s.get("estimated_complexity", "medium"),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            id=data["id"],
            title=data["title"],
            summary=data["summary"],
            context=data.get("context", ""),
            steps=steps,
            critical_files=data.get("critical_files", []),
            considerations=data.get("considerations", []),
            status=data.get("status", "draft"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            approved_at=data.get("approved_at"),
        )

    def format_markdown(self) -> str:
        """Format the plan as Markdown."""
        lines = [
            f"# Plan: {self.title}",
            "",
            f"**Status:** {self.status}",
            f"**Created:** {self.created_at}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
        ]

        if self.context:
            lines.extend([
                "## Context",
                "",
                self.context,
                "",
            ])

        if self.critical_files:
            lines.extend([
                "## Critical Files",
                "",
            ])
            for f in self.critical_files:
                lines.append(f"- `{f}`")
            lines.append("")

        if self.steps:
            lines.extend([
                "## Implementation Steps",
                "",
            ])
            for step in self.steps:
                lines.append(f"### Step {step.number}: {step.title}")
                lines.append("")
                lines.append(step.description)
                lines.append("")
                if step.files_affected:
                    lines.append("**Files affected:**")
                    for f in step.files_affected:
                        lines.append(f"- `{f}`")
                    lines.append("")
                lines.append(f"**Complexity:** {step.estimated_complexity}")
                lines.append("")

        if self.considerations:
            lines.extend([
                "## Considerations",
                "",
            ])
            for c in self.considerations:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)


class PlanManager:
    """
    Manages implementation plans with persistence.

    This class provides:
    - Plan creation and modification
    - Plan approval workflow
    - Conversion of plans to tasks
    - Persistence to disk
    """

    def __init__(self, storage_dir: Path | str | None = None):
        """
        Initialize the plan manager.

        Args:
            storage_dir: Directory to store plans. Defaults to ./plans/
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./plans")
        self.current_plan: Plan | None = None
        self._next_id = 1

    def create_plan(
        self,
        title: str,
        summary: str,
        context: str = "",
        steps: list[dict[str, Any]] | None = None,
        critical_files: list[str] | None = None,
        considerations: list[str] | None = None,
    ) -> Plan:
        """
        Create a new implementation plan.

        Args:
            title: Brief title of the feature/change
            summary: High-level summary
            context: Background context
            steps: List of implementation steps
            critical_files: Key files to be modified
            considerations: Architectural considerations

        Returns:
            The newly created Plan
        """
        plan_id = f"plan_{self._next_id}"
        self._next_id += 1

        plan_steps = []
        if steps:
            for i, step_data in enumerate(steps, 1):
                plan_steps.append(PlanStep(
                    number=i,
                    title=step_data.get("title", f"Step {i}"),
                    description=step_data.get("description", ""),
                    files_affected=step_data.get("files_affected", []),
                    estimated_complexity=step_data.get("estimated_complexity", "medium"),
                ))

        plan = Plan(
            id=plan_id,
            title=title,
            summary=summary,
            context=context,
            steps=plan_steps,
            critical_files=critical_files or [],
            considerations=considerations or [],
        )

        self.current_plan = plan
        return plan

    def approve_plan(self, plan: Plan | None = None) -> Plan | None:
        """
        Mark a plan as approved.

        Args:
            plan: The plan to approve. Uses current_plan if not provided.

        Returns:
            The approved Plan or None if no plan found
        """
        target = plan or self.current_plan
        if not target:
            return None

        target.status = "approved"
        target.approved_at = datetime.now().isoformat()
        return target

    def convert_plan_to_tasks(
        self,
        task_manager: TaskManager,
        plan: Plan | None = None,
    ) -> list[Task]:
        """
        Convert plan steps to tasks with proper dependencies.

        Args:
            task_manager: TaskManager instance to create tasks in
            plan: The plan to convert. Uses current_plan if not provided.

        Returns:
            List of created Tasks
        """
        target = plan or self.current_plan
        if not target:
            return []

        tasks = []
        previous_task_id: str | None = None

        for step in target.steps:
            # Create the task
            task = task_manager.create_task(
                subject=step.title,
                description=step.description,
                active_form=f"Working on: {step.title}",
                metadata={
                    "plan_id": target.id,
                    "step_number": step.number,
                    "files_affected": step.files_affected,
                    "complexity": step.estimated_complexity,
                },
            )

            # Set up dependency chain - each task depends on the previous
            if previous_task_id:
                task_manager.update_task(
                    task.id,
                    add_blocked_by=[previous_task_id],
                )

            tasks.append(task)
            previous_task_id = task.id

        # Update plan status
        target.status = "implementing"

        return tasks

    def save_plan(self, plan: Plan | None = None, filename: str | None = None) -> Path:
        """
        Save a plan to disk as both JSON and Markdown.

        Args:
            plan: The plan to save. Uses current_plan if not provided.
            filename: Custom filename (without extension). Defaults to plan ID.

        Returns:
            Path to the saved JSON file
        """
        target = plan or self.current_plan
        if not target:
            raise ValueError("No plan to save")

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        base_name = filename or target.id

        # Save JSON
        json_path = self.storage_dir / f"{base_name}.json"
        with open(json_path, "w") as f:
            json.dump(target.to_dict(), f, indent=2)

        # Save Markdown
        md_path = self.storage_dir / f"{base_name}.md"
        with open(md_path, "w") as f:
            f.write(target.format_markdown())

        return json_path

    def load_plan(self, filename: str) -> Plan:
        """
        Load a plan from disk.

        Args:
            filename: Filename (with or without .json extension)

        Returns:
            The loaded Plan
        """
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        path = self.storage_dir / filename

        with open(path, "r") as f:
            data = json.load(f)

        plan = Plan.from_dict(data)
        self.current_plan = plan
        return plan

    def list_plans(self) -> list[str]:
        """List all saved plan files."""
        if not self.storage_dir.exists():
            return []
        return [f.stem for f in self.storage_dir.glob("*.json")]
