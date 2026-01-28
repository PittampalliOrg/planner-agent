"""
Durable Planner Agent - Dapr Workflow with Claude CLI Integration

Uses Dapr Workflow for durability with Claude CLI providing native
Claude Code tools (Read, Write, Edit, Bash, TodoWrite, etc).

Architecture:
- Dapr Workflow: Orchestrates activities, persists state, handles retries
- CLIPlannerAgent: Uses Claude CLI in plan/execution modes with streaming JSON
- No custom plan/task managers - uses Claude's native TodoWrite tool

This approach gives us:
- Native Claude Code tools with full capabilities (latest CLI version)
- Dapr durability at workflow level
- Human-in-the-loop approval via wait_for_external_event
- Fault tolerance with activity retries
- Session resumption for clarification loops
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Import Dapr Workflow for durable orchestration
try:
    from dapr.ext.workflow import (
        WorkflowRuntime,
        DaprWorkflowClient,
        DaprWorkflowContext,
        WorkflowActivityContext,
    )
    DAPR_WORKFLOW_AVAILABLE = True
except ImportError:
    DAPR_WORKFLOW_AVAILABLE = False
    WorkflowRuntime = None
    DaprWorkflowClient = None
    DaprWorkflowContext = None
    WorkflowActivityContext = None

# Claude CLI is used directly via CLIPlannerAgent in planner_agent.py
# No SDK import needed - we spawn the 'claude' CLI process

from streaming import (
    stream_phase_changed_sync,
    stream_native_task_created_sync,
)


# =============================================================================
# Configuration
# =============================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
PLANS_DIR = Path(os.getenv("PLANS_DIR", "/plans"))


# =============================================================================
# Activity Input/Output Models
# =============================================================================

class CloneInput(BaseModel):
    """Input for repository cloning activity."""
    owner: str
    repo: str
    branch: str = "main"
    token: str | None = None
    workspace_dir: str = "/workspace"
    workflow_id: str | None = None


class CloneOutput(BaseModel):
    """Output from repository cloning activity."""
    success: bool
    path: str = ""
    file_count: int = 0
    error: str | None = None


class PlanningInput(BaseModel):
    """Input for planning activity."""
    cwd: str
    feature_request: str
    workflow_id: str | None = None
    session_id: str | None = None  # CLI session ID for resumption
    clarification_response: str | None = None  # Response to a previous clarification request


class PlanningOutput(BaseModel):
    """Output from planning activity."""
    success: bool
    state: str = "plan_ready"  # exploring, awaiting_clarification, plan_ready
    plan_id: str | None = None
    session_id: str | None = None  # CLI session ID for resumption
    tasks_created: int = 0
    task_subjects: list[str] = Field(default_factory=list)
    tasks_dir: str | None = None  # Path to recorded task snapshots
    files_explored: list[str] = Field(default_factory=list)
    clarification_request: dict | None = None  # AskUserQuestion format
    plan_content: str | None = None  # Plan summary/content
    critical_files: list[str] = Field(default_factory=list)
    error: str | None = None


class ExecutionInput(BaseModel):
    """Input for execution activity."""
    cwd: str
    workflow_id: str | None = None
    task_subjects: list[str] = Field(default_factory=list)
    tasks_dir: str | None = None  # Path to recorded task snapshots


class ExecutionOutput(BaseModel):
    """Output from execution activity."""
    success: bool
    tasks_completed: int = 0
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None


# =============================================================================
# Workflow Activities
# =============================================================================

def clone_repository_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Clone a GitHub repository.

    Clones the repository to the workspace directory and returns the path.
    """
    import shutil
    import subprocess

    input_data = CloneInput.model_validate_json(input_json)
    workspace = Path(input_data.workspace_dir)
    clone_path = workspace / f"{input_data.owner}_{input_data.repo}"

    print(f"[Activity] Cloning {input_data.owner}/{input_data.repo} to {clone_path}")

    try:
        # Remove existing if present
        if clone_path.exists():
            print(f"[Activity] Removing existing directory: {clone_path}")
            shutil.rmtree(clone_path)

        # Build clone command
        cmd = [
            "git", "clone",
            "--depth", "1",
            "--single-branch",
            "--branch", input_data.branch,
        ]

        # Add auth if token provided
        if input_data.token:
            auth_url = f"https://{input_data.token}@github.com/{input_data.owner}/{input_data.repo}.git"
            cmd.append(auth_url)
        else:
            cmd.append(f"https://github.com/{input_data.owner}/{input_data.repo}.git")

        cmd.append(str(clone_path))

        # Execute clone
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            print(f"[Activity] Clone failed: {error_msg}")
            return CloneOutput(
                success=False,
                error=f"Clone failed: {error_msg}",
            ).model_dump()

        # Count files
        file_count = sum(1 for _ in clone_path.rglob("*") if _.is_file())
        print(f"[Activity] Clone success: {file_count} files")

        return CloneOutput(
            success=True,
            path=str(clone_path),
            file_count=file_count,
        ).model_dump()

    except subprocess.TimeoutExpired:
        return CloneOutput(
            success=False,
            error="Clone timed out after 5 minutes",
        ).model_dump()
    except Exception as e:
        print(f"[Activity] Clone error: {e}")
        return CloneOutput(
            success=False,
            error=str(e),
        ).model_dump()


def planning_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Run planning phase with native Claude Code tools.

    Uses NativePlannerAgent which leverages Claude's native TaskCreate,
    TaskList, and TaskUpdate tools for plan/task management.

    Returns PlanningOutput with state indicating next action:
    - "exploring": Still in progress
    - "awaiting_clarification": Needs user input (AskUserQuestion detected)
    - "plan_ready": Plan complete (ExitPlanMode detected or implicit)
    """
    from planner_agent import NativePlannerAgent, PlanningResult
    import uuid

    input_data = PlanningInput.model_validate_json(input_json)
    workflow_id = input_data.workflow_id

    # Generate plan_id for task association
    plan_id = f"plan-{uuid.uuid4().hex[:8]}"

    print(f"[Activity] Planning for: {input_data.feature_request[:100]}...")
    print(f"[Activity] Plan ID: {plan_id}")
    if input_data.clarification_response:
        print(f"[Activity] Resuming with clarification response")

    try:
        # Run async planning in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            agent = NativePlannerAgent(
                cwd=input_data.cwd,
                workflow_id=workflow_id,
                plan_id=plan_id,
            )
            result: PlanningResult = loop.run_until_complete(
                agent.run_planning(
                    input_data.feature_request,
                    session_id=input_data.session_id,
                    clarification_response=input_data.clarification_response,
                )
            )
        finally:
            loop.close()

        # Convert PlanningResult to PlanningOutput
        clarification_request_dict = None
        if result.clarification_request:
            clarification_request_dict = {
                "question_id": result.clarification_request.question_id,
                "questions": result.clarification_request.questions,
                "timestamp": result.clarification_request.timestamp.isoformat(),
            }

        return PlanningOutput(
            success=result.success,
            state=result.state.value,
            plan_id=result.plan_id,
            session_id=result.session_id,
            tasks_created=result.tasks_created,
            task_subjects=result.task_subjects,
            tasks_dir=result.tasks_dir,
            files_explored=result.files_explored,
            clarification_request=clarification_request_dict,
            plan_content=result.plan_content,
            critical_files=result.critical_files,
            error=result.error,
        ).model_dump()

    except Exception as e:
        print(f"[Activity] Planning failed: {e}")
        return PlanningOutput(
            success=False,
            state="exploring",
            error=str(e),
        ).model_dump()


def execution_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Run execution phase with native Claude Code tools.

    Uses NativePlannerAgent to implement the tasks identified during planning.
    The native SDK handles all task logic; we just pass context through.
    """
    from planner_agent import NativePlannerAgent
    from task_persistence import TaskStore

    input_data = ExecutionInput.model_validate_json(input_json)
    workflow_id = input_data.workflow_id
    task_subjects = input_data.task_subjects
    tasks_dir = input_data.tasks_dir

    print(f"[Activity] Executing tasks in: {input_data.cwd}")
    print(f"[Activity] Tasks to implement: {task_subjects}")
    if tasks_dir:
        print(f"[Activity] Tasks dir: {tasks_dir}")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            agent = NativePlannerAgent(
                cwd=input_data.cwd,
                workflow_id=workflow_id,
            )

            # Point task store at the existing snapshots directory
            if tasks_dir:
                from pathlib import Path
                tasks_path = Path(tasks_dir)
                if tasks_path.exists():
                    agent.task_store = TaskStore(
                        base_path=tasks_path.parent,
                        workflow_id=workflow_id,
                    )
                    print(f"[Activity] Task store loaded from: {tasks_dir}")

            result = loop.run_until_complete(agent.run_execution(task_subjects))
        finally:
            loop.close()

        return ExecutionOutput(
            success=result.get("success", False),
            tasks_completed=result.get("tasks_completed", 0),
            files_changed=result.get("files_changed", []),
            error=result.get("error"),
        ).model_dump()

    except Exception as e:
        print(f"[Activity] Execution failed: {e}")
        return ExecutionOutput(
            success=False,
            error=str(e),
        ).model_dump()


# =============================================================================
# Dapr Workflow - Unified Planning and Execution
# =============================================================================

def unified_workflow(ctx: DaprWorkflowContext, input_json: str) -> dict:
    """
    Dapr Workflow: Unified planning and execution with approval gate.

    This is the SINGLE ORCHESTRATOR workflow that handles:
    0. Clone repository (optional, if repository info provided)
    1. Planning phase - Claude uses TaskCreate to create implementation tasks
    2. Wait for approval (Dapr external event)
    3. Execution phase - Claude uses TaskList/TaskUpdate to execute tasks

    Uses wait_for_external_event for the approval gate - the workflow
    properly pauses without polling until the approval event is raised.

    The workflow is durable - it survives crashes and restarts.

    Phases:
    - clone: Cloning repository
    - planning: Creating implementation tasks
    - awaiting_approval: Tasks ready, waiting for user approval
    - executing: Running implementation
    - completed: Workflow finished successfully
    - failed: Workflow encountered an error
    """
    input_data = json.loads(input_json)
    cwd = input_data.get("cwd", str(WORKSPACE_DIR))
    feature_request = input_data.get("feature_request", "")
    workflow_id = input_data.get("workflow_id") or ctx.instance_id

    # Repository info for cloning (optional)
    repository = input_data.get("repository")

    # Phase 0: Clone repository (if repository info provided)
    if repository:
        ctx.set_custom_status(json.dumps({
            "phase": "clone",
            "progress": 5,
            "message": f"Cloning {repository.get('owner')}/{repository.get('repo')}...",
        }))

        stream_phase_changed_sync(
            workflow_id=ctx.instance_id,
            phase="clone",
            status=f"Cloning {repository.get('owner')}/{repository.get('repo')}",
            progress=5,
        )

        clone_input = CloneInput(
            owner=repository.get("owner", ""),
            repo=repository.get("repo", ""),
            branch=repository.get("branch", "main"),
            token=repository.get("token"),
            workspace_dir=str(WORKSPACE_DIR),
            workflow_id=workflow_id,
        )

        clone_result = yield ctx.call_activity(
            clone_repository_activity,
            input=clone_input.model_dump_json(),
        )
        clone_output = CloneOutput.model_validate(clone_result)

        if not clone_output.success:
            ctx.set_custom_status(json.dumps({
                "phase": "failed",
                "progress": 0,
                "message": f"Clone failed: {clone_output.error}",
            }))
            return {
                "success": False,
                "error": f"Clone failed: {clone_output.error}",
                "phase": "clone",
            }

        # Update cwd to cloned repository path
        cwd = clone_output.path
        print(f"[Workflow] Repository cloned to: {cwd}")

    # Phase 1: Planning with native tools (with clarification loop)
    ctx.set_custom_status(json.dumps({
        "phase": "planning",
        "progress": 20,
        "message": "Creating implementation tasks...",
    }))

    stream_phase_changed_sync(
        workflow_id=ctx.instance_id,
        phase="planning",
        status="Creating implementation tasks",
        progress=20,
    )

    # Planning phase with clarification loop
    clarification_response = None
    session_id = None  # CLI session ID for resumption
    max_clarifications = 5
    clarification_count = 0
    plan_output = None

    while clarification_count <= max_clarifications:
        plan_input = PlanningInput(
            cwd=cwd,
            feature_request=feature_request,
            workflow_id=workflow_id,
            session_id=session_id,
            clarification_response=clarification_response,
        )

        plan_result = yield ctx.call_activity(
            planning_activity,
            input=plan_input.model_dump_json(),
        )
        plan_output = PlanningOutput.model_validate(plan_result)

        if not plan_output.success:
            ctx.set_custom_status(json.dumps({
                "phase": "failed",
                "progress": 0,
                "message": f"Planning failed: {plan_output.error}",
            }))
            return {
                "success": False,
                "error": f"Planning failed: {plan_output.error}",
                "phase": "planning",
            }

        # Check if clarification is needed
        if plan_output.state == "awaiting_clarification":
            ctx.set_custom_status(json.dumps({
                "phase": "awaiting_clarification",
                "progress": 30,
                "message": "Waiting for user clarification",
                "plan_id": plan_output.plan_id,
                "clarification_request": plan_output.clarification_request,
                "clarification_index": clarification_count,
            }))

            stream_phase_changed_sync(
                workflow_id=ctx.instance_id,
                phase="awaiting_clarification",
                status="Waiting for user clarification",
                progress=30,
                plan_id=plan_output.plan_id,
                extra_data={
                    "clarification_request": plan_output.clarification_request,
                    "clarification_index": clarification_count,
                },
            )

            # Wait for clarification response via external event
            event_name = f"clarification_{plan_output.plan_id}_{clarification_count}"
            print(f"[Workflow] Waiting for clarification event: {event_name}")
            clarification_event = yield ctx.wait_for_external_event(event_name)
            print(f"[Workflow] Received clarification event: {clarification_event}")

            clarification_response = clarification_event.get("response", "") if clarification_event else ""
            session_id = plan_output.session_id  # Preserve session for resumption
            clarification_count += 1

            # Update status and continue loop
            ctx.set_custom_status(json.dumps({
                "phase": "planning",
                "progress": 25,
                "message": f"Continuing planning with clarification ({clarification_count}/{max_clarifications})",
            }))
            continue

        # Plan is ready
        if plan_output.state == "plan_ready":
            break

    # Ensure we have valid plan output
    if plan_output is None:
        return {
            "success": False,
            "error": "Planning loop completed without result",
            "phase": "planning",
        }

    plan_id = plan_output.plan_id
    print(f"[Workflow] Planning completed: {plan_output.tasks_created} tasks created")

    # Phase 2: Await approval
    ctx.set_custom_status(json.dumps({
        "phase": "awaiting_approval",
        "progress": 50,
        "message": "Tasks ready for approval",
        "plan_id": plan_id,
        "tasks_created": plan_output.tasks_created,
        "task_subjects": plan_output.task_subjects,
    }))

    stream_phase_changed_sync(
        workflow_id=ctx.instance_id,
        phase="awaiting_approval",
        status="Waiting for plan approval",
        progress=50,
        plan_id=plan_id,
        extra_data={
            "tasks_created": plan_output.tasks_created,
            "task_subjects": plan_output.task_subjects,
        },
    )

    print(f"[Workflow] Waiting for approval event: plan_approval_{plan_id}")
    approval = yield ctx.wait_for_external_event(f"plan_approval_{plan_id}")
    print(f"[Workflow] Received approval event: {approval}")

    # Check approval result
    if not approval or not approval.get("approved"):
        reason = approval.get("reason", "No reason provided") if approval else "No response"
        ctx.set_custom_status(json.dumps({
            "phase": "rejected",
            "progress": 0,
            "message": f"Plan rejected: {reason}",
        }))
        return {
            "success": False,
            "plan_id": plan_id,
            "tasks_created": plan_output.tasks_created,
            "status": "rejected",
            "error": f"Plan rejected: {reason}",
            "phase": "approval",
        }

    # Phase 3: Execution with native tools
    ctx.set_custom_status(json.dumps({
        "phase": "executing",
        "progress": 60,
        "message": "Executing implementation tasks...",
        "plan_id": plan_id,
    }))

    stream_phase_changed_sync(
        workflow_id=ctx.instance_id,
        phase="executing",
        status="Executing implementation tasks",
        progress=60,
        plan_id=plan_id,
    )

    exec_input = ExecutionInput(
        cwd=cwd,
        workflow_id=workflow_id,
        task_subjects=plan_output.task_subjects,
        tasks_dir=plan_output.tasks_dir,
    )

    exec_result = yield ctx.call_activity(
        execution_activity,
        input=exec_input.model_dump_json(),
    )
    exec_output = ExecutionOutput.model_validate(exec_result)

    # Set final status
    if exec_output.success:
        ctx.set_custom_status(json.dumps({
            "phase": "completed",
            "progress": 100,
            "message": "Workflow completed successfully",
            "tasks_completed": exec_output.tasks_completed,
            "files_changed": exec_output.files_changed,
        }))

        stream_phase_changed_sync(
            workflow_id=ctx.instance_id,
            phase="completed",
            status="Workflow completed",
            progress=100,
        )
    else:
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Execution failed: {exec_output.error}",
        }))

    return {
        "success": exec_output.success,
        "plan_id": plan_id,
        "tasks_created": plan_output.tasks_created,
        "tasks_completed": exec_output.tasks_completed,
        "files_changed": exec_output.files_changed,
        "status": "completed" if exec_output.success else "failed",
        "error": exec_output.error,
        "phase": "complete" if exec_output.success else "execution",
    }


# Keep old name for backward compatibility
planning_and_execution_workflow = unified_workflow


# =============================================================================
# Workflow Runtime Management
# =============================================================================

_workflow_runtime: WorkflowRuntime | None = None
_workflow_client: DaprWorkflowClient | None = None


def get_workflow_runtime() -> WorkflowRuntime | None:
    """Get or create the workflow runtime."""
    global _workflow_runtime

    if not DAPR_WORKFLOW_AVAILABLE:
        return None

    if _workflow_runtime is None:
        _workflow_runtime = WorkflowRuntime()

        # Register the unified workflow
        _workflow_runtime.register_workflow(unified_workflow)

        # Register activities
        _workflow_runtime.register_activity(clone_repository_activity)
        _workflow_runtime.register_activity(planning_activity)
        _workflow_runtime.register_activity(execution_activity)

    return _workflow_runtime


def get_workflow_client() -> DaprWorkflowClient | None:
    """Get or create the workflow client."""
    global _workflow_client

    if not DAPR_WORKFLOW_AVAILABLE:
        return None

    if _workflow_client is None:
        _workflow_client = DaprWorkflowClient()

    return _workflow_client


async def start_workflow_runtime() -> bool:
    """Start the workflow runtime."""
    runtime = get_workflow_runtime()
    if runtime is None:
        return False

    try:
        runtime.start()
        print("[Workflow] Runtime started")
        return True
    except Exception as e:
        print(f"[Workflow] Failed to start runtime: {e}")
        return False


async def stop_workflow_runtime() -> None:
    """Stop the workflow runtime."""
    global _workflow_runtime, _workflow_client

    if _workflow_runtime:
        try:
            _workflow_runtime.shutdown()
            print("[Workflow] Runtime stopped")
        except Exception as e:
            print(f"[Workflow] Error stopping runtime: {e}")

    _workflow_runtime = None
    _workflow_client = None


# =============================================================================
# Convenience Functions
# =============================================================================

def is_durable_agents_available() -> bool:
    """Check if durable workflows are available."""
    return DAPR_WORKFLOW_AVAILABLE


def get_durable_agent_status() -> dict[str, Any]:
    """Get status of durable agent dependencies."""
    return {
        "dapr_workflow_available": DAPR_WORKFLOW_AVAILABLE,
        "claude_cli_available": True,  # CLI is always available (spawned as subprocess)
        "workspace_dir": str(WORKSPACE_DIR),
        "plans_dir": str(PLANS_DIR),
    }


# Legacy compatibility aliases
DAPR_AGENTS_AVAILABLE = DAPR_WORKFLOW_AVAILABLE


async def create_durable_plan(
    cwd: str,
    feature_request: str,
    session_id: str | None = None,
    workflow_id: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Create a plan using the unified workflow.

    This is a convenience function that starts the workflow and waits
    for the planning phase to complete. The workflow will then be
    paused awaiting approval.
    """
    from planner_agent import NativePlannerAgent

    # For non-durable execution, just run planning directly
    agent = NativePlannerAgent(cwd=cwd, workflow_id=workflow_id)
    return await agent.run_planning(feature_request)


async def execute_durable_plan(
    cwd: str,
    plan_id: str,
    workflow_id: str,
    session_id: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Execute a plan using NativePlannerAgent.

    This is a convenience function for direct execution without
    the full workflow orchestration.
    """
    from planner_agent import NativePlannerAgent

    agent = NativePlannerAgent(cwd=cwd, workflow_id=workflow_id)
    return await agent.run_execution()


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Test the durable workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Durable Planner Agent")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory")
    parser.add_argument("prompt", nargs="?", type=str, help="Feature request")

    args = parser.parse_args()

    print("=" * 60)
    print("DURABLE PLANNER AGENT (Native Claude Code Tools)")
    print("=" * 60)
    print(f"\nStatus: {get_durable_agent_status()}")

    if args.prompt:
        # Start runtime
        await start_workflow_runtime()

        try:
            result = await create_durable_plan(
                cwd=args.cwd,
                feature_request=args.prompt,
            )
            print(f"\nResult: {json.dumps(result, indent=2)}")
        finally:
            await stop_workflow_runtime()
    else:
        print("\nNo prompt provided. Use: python durable_agent.py --cwd /path 'feature request'")


if __name__ == "__main__":
    asyncio.run(main())
