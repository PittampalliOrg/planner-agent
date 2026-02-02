"""Dapr workflow for multi-step planning → execution → testing.

This module implements a proper Dapr workflow using dapr-ext-workflow so that
the workflow phases (planning, execution, testing) appear as activities in the
ai-chatbot UI workflow graph.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import dapr.ext.workflow as wf
from agents import Runner
from dapr.clients import DaprClient
from pydantic import BaseModel, Field

# Import config provider for runtime configuration
from dapr_config import get_config

logger = logging.getLogger(__name__)


def _get_pubsub_name() -> str:
    """Get pub/sub name from Dapr config store (with fallback to env var)."""
    return get_config("PUBSUB_NAME", "pubsub")


def _get_pubsub_topic() -> str:
    """Get pub/sub topic from Dapr config store (with fallback to env var)."""
    return get_config("PUBSUB_TOPIC", "workflow.stream")


def publish_workflow_event(
    workflow_id: str,
    event_type: str,
    data: dict,
    task_id: Optional[str] = None,
) -> bool:
    """Publish a workflow event to the Dapr pub/sub topic for ai-chatbot."""
    event = {
        "id": f"workflow-{workflow_id}-{uuid.uuid4().hex[:8]}",
        "type": event_type,
        "workflowId": workflow_id,
        "agentId": "planner-dapr-agent",
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if task_id:
        event["taskId"] = task_id

    try:
        pubsub_name = _get_pubsub_name()
        pubsub_topic = _get_pubsub_topic()
        with DaprClient() as client:
            client.publish_event(
                pubsub_name=pubsub_name,
                topic_name=pubsub_topic,
                data=json.dumps(event),
                data_content_type="application/json",
            )
        logger.info(f"Published {event_type} event for workflow {workflow_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to publish {event_type} event: {e}")
        return False


# ============================================================================
# Clone Activity Models
# ============================================================================

class CloneInput(BaseModel):
    """Input for repository cloning activity."""
    owner: str
    repo: str
    branch: str = "main"
    token: Optional[str] = None
    workspace_dir: str = "/app/workspace"
    workflow_id: Optional[str] = None


class CloneOutput(BaseModel):
    """Output from repository cloning activity."""
    success: bool
    path: str = ""
    file_count: int = 0
    error: Optional[str] = None

# Initialize workflow runtime
wfr = wf.WorkflowRuntime()


# ============================================================================
# Activity: Clone Repository Phase
# ============================================================================

@wfr.activity(name="clone_repository")
def clone_repository_activity(ctx: wf.WorkflowActivityContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clone a GitHub repository with optional token authentication.

    This activity:
    1. Validates the input parameters
    2. Builds the git URL (with token if provided)
    3. Clones the repository with --depth 1 for speed
    4. Counts files for tracking
    5. Returns the clone path and file count
    """
    clone_input = CloneInput(**input_data)
    workflow_id = clone_input.workflow_id or "unknown"

    logger.info(f"Clone activity started for {clone_input.owner}/{clone_input.repo}@{clone_input.branch}")

    # Publish clone started event
    publish_workflow_event(workflow_id, "phase_started", {
        "phase": "cloning",
        "status": f"Cloning {clone_input.owner}/{clone_input.repo}@{clone_input.branch}...",
        "progress": 5,
    })

    # Build repository path
    repo_path = os.path.join(clone_input.workspace_dir, clone_input.repo)

    # Remove existing directory if present
    if os.path.exists(repo_path):
        logger.info(f"Removing existing directory: {repo_path}")
        subprocess.run(["rm", "-rf", repo_path], check=True)

    # Build git URL with token if provided
    if clone_input.token:
        git_url = f"https://{clone_input.token}@github.com/{clone_input.owner}/{clone_input.repo}.git"
        logger.info(f"Using token authentication for clone")
    else:
        git_url = f"https://github.com/{clone_input.owner}/{clone_input.repo}.git"
        logger.info(f"Using public clone (no token)")

    try:
        # Clone with depth 1 for speed
        result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", clone_input.branch,
                git_url,
                repo_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown git error"
            # Sanitize token from error message
            if clone_input.token:
                error_msg = error_msg.replace(clone_input.token, "***")
            logger.error(f"Git clone failed: {error_msg}")
            return CloneOutput(
                success=False,
                error=f"Git clone failed: {error_msg}",
            ).model_dump()

        # Count files (excluding .git)
        file_count = 0
        for root, dirs, files in os.walk(repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            file_count += len(files)

        logger.info(f"Clone completed: {repo_path} with {file_count} files")

        # Publish clone completed event
        publish_workflow_event(workflow_id, "phase_completed", {
            "phase": "cloning",
            "status": f"Repository cloned: {file_count} files",
            "progress": 10,
            "file_count": file_count,
            "repo_path": repo_path,
        })

        return CloneOutput(
            success=True,
            path=repo_path,
            file_count=file_count,
        ).model_dump()

    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out after 5 minutes")
        return CloneOutput(
            success=False,
            error="Git clone timed out after 5 minutes",
        ).model_dump()
    except Exception as e:
        error_msg = str(e)
        # Sanitize token from error message
        if clone_input.token:
            error_msg = error_msg.replace(clone_input.token, "***")
        logger.error(f"Clone failed: {error_msg}")
        return CloneOutput(
            success=False,
            error=error_msg,
        ).model_dump()


# ============================================================================
# Activity: Planning Phase
# ============================================================================

@wfr.activity(name="planning")
def planning_activity(ctx: wf.WorkflowActivityContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the planning phase using OpenAI agents.

    This activity:
    1. Creates a planning agent
    2. Runs it to generate a plan with tasks and test cases
    3. Returns the plan data
    """
    import asyncio
    from workflow_agent import create_planning_agent, Plan, Task

    task = input_data.get("task", "")
    model = input_data.get("model", "gpt-5.2-codex")
    max_turns = input_data.get("max_turns", 20)
    workflow_id = input_data.get("workflow_id", "unknown")

    logger.info(f"Planning activity started for task: {task[:100]}...")

    # Publish phase started event
    publish_workflow_event(workflow_id, "phase_started", {
        "phase": "planning",
        "status": "Creating implementation plan...",
        "progress": 10,
    })

    async def run_planning():
        planning_agent = create_planning_agent(model)
        plan_result = await Runner.run(
            planning_agent,
            input=task,
            max_turns=max_turns,
        )
        plan: Plan = plan_result.final_output

        # Auto-populate blocks based on blockedBy
        task_map = {t.id: t for t in plan.tasks}
        for t in plan.tasks:
            for blocked_by_id in t.blockedBy:
                if blocked_by_id in task_map:
                    if t.id not in task_map[blocked_by_id].blocks:
                        task_map[blocked_by_id].blocks.append(t.id)

        return plan.model_dump()

    try:
        plan_data = asyncio.run(run_planning())
        logger.info(f"Planning completed with {len(plan_data.get('tasks', []))} tasks")

        # Publish phase completed event
        publish_workflow_event(workflow_id, "phase_completed", {
            "phase": "planning",
            "status": f"Plan created with {len(plan_data.get('tasks', []))} tasks",
            "progress": 30,
            "tasks_count": len(plan_data.get("tasks", [])),
            "tests_count": len(plan_data.get("tests", [])),
        })

        return {
            "success": True,
            "plan": plan_data,
        }
    except Exception as e:
        logger.error(f"Planning failed: {e}")

        # Publish failure event
        publish_workflow_event(workflow_id, "phase_failed", {
            "phase": "planning",
            "status": f"Planning failed: {str(e)}",
            "error": str(e),
        })

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Activity: Execution Phase
# ============================================================================

@wfr.activity(name="execution")
def execution_activity(ctx: wf.WorkflowActivityContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the execution phase using OpenAI agents.

    This activity:
    1. Creates an execution agent
    2. Runs it with the plan to execute tasks
    3. Returns the execution result
    """
    import asyncio
    from workflow_agent import create_execution_agent, ExecutionResult

    plan = input_data.get("plan", {})
    model = input_data.get("model", "gpt-5.2-codex")
    max_turns = input_data.get("max_turns", 20)
    workflow_id = input_data.get("workflow_id", "unknown")
    tasks = plan.get("tasks", [])

    logger.info(f"Execution activity started with {len(tasks)} tasks")

    # Publish execution started event
    publish_workflow_event(workflow_id, "execution_started", {
        "phase": "execution",
        "status": f"Executing {len(tasks)} tasks...",
        "progress": 50,
        "tasks_count": len(tasks),
    })

    async def run_execution():
        execution_agent = create_execution_agent(model)

        exec_prompt = f"""Execute this plan:

Summary: {plan.get('summary', '')}

Tasks:
{chr(10).join(f"- [{t['id']}] {t['subject']}: {t['description']} (blockedBy: {t.get('blockedBy', [])})" for t in tasks)}

Reasoning: {plan.get('reasoning', '')}"""

        exec_result = await Runner.run(
            execution_agent,
            input=exec_prompt,
            max_turns=max_turns,
        )
        execution: ExecutionResult = exec_result.final_output
        return execution.model_dump()

    try:
        execution_data = asyncio.run(run_execution())
        logger.info(f"Execution completed: success={execution_data.get('success')}")

        # Publish execution completed event
        publish_workflow_event(workflow_id, "execution_completed", {
            "phase": "execution",
            "status": "Execution completed",
            "progress": 80,
            "success": execution_data.get("success", False),
            "completed_tasks": execution_data.get("completed_tasks", []),
        })

        return {
            "success": True,
            "execution": execution_data,
        }
    except Exception as e:
        logger.error(f"Execution failed: {e}")

        # Publish execution failed event
        publish_workflow_event(workflow_id, "execution_failed", {
            "phase": "execution",
            "status": f"Execution failed: {str(e)}",
            "error": str(e),
        })

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Activity: Testing Phase
# ============================================================================

@wfr.activity(name="testing")
def testing_activity(ctx: wf.WorkflowActivityContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the testing phase using OpenAI agents.

    This activity:
    1. Creates a testing agent
    2. Runs it to verify the implementation
    3. Returns the test result
    """
    import asyncio
    from workflow_agent import create_testing_agent, TestResult

    plan = input_data.get("plan", {})
    execution = input_data.get("execution", {})
    model = input_data.get("model", "gpt-5.2-codex")
    max_turns = input_data.get("max_turns", 20)
    max_test_retries = input_data.get("max_test_retries", 3)
    workflow_id = input_data.get("workflow_id", "unknown")

    tests = plan.get("tests", [])
    logger.info(f"Testing activity started with {len(tests)} test cases")

    # Publish testing started event
    publish_workflow_event(workflow_id, "phase_started", {
        "phase": "testing",
        "status": f"Running {len(tests)} test cases...",
        "progress": 85,
        "tests_count": len(tests),
    })

    async def run_testing():
        testing_agent = create_testing_agent(model)

        test_prompt = f"""Verify the implementation:

Plan Summary: {plan.get('summary', '')}

Test Cases:
{chr(10).join(f"- [{tc['id']}] {tc['description']} (type: {tc.get('test_type', '')}, command: {tc.get('command', '')})" for tc in tests)}

Execution Summary: {execution.get('output', '')}
Completed Tasks: {execution.get('completed_tasks', [])}"""

        test: TestResult = TestResult(
            passed=False, tests_run=0, tests_passed=0, tests_failed=0,
            failures=[], summary="Tests not yet run"
        )

        for attempt in range(max_test_retries):
            test_result = await Runner.run(
                testing_agent,
                input=test_prompt,
                max_turns=max_turns,
            )
            test = test_result.final_output
            if test.passed:
                break

        return test.model_dump()

    try:
        test_data = asyncio.run(run_testing())
        logger.info(f"Testing completed: passed={test_data.get('passed')}")

        # Publish testing completed event
        publish_workflow_event(workflow_id, "phase_completed", {
            "phase": "testing",
            "status": "All tests passed!" if test_data.get("passed") else f"Tests failed: {test_data.get('tests_failed', 0)} failures",
            "progress": 95,
            "passed": test_data.get("passed", False),
            "tests_run": test_data.get("tests_run", 0),
            "tests_passed": test_data.get("tests_passed", 0),
            "tests_failed": test_data.get("tests_failed", 0),
        })

        return {
            "success": True,
            "testing": test_data,
        }
    except Exception as e:
        logger.error(f"Testing failed: {e}")

        # Publish testing failed event
        publish_workflow_event(workflow_id, "phase_failed", {
            "phase": "testing",
            "status": f"Testing failed: {str(e)}",
            "error": str(e),
        })

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Workflow: Multi-Step Clone → Planning → Approval → Execution → Testing
# ============================================================================

@wfr.workflow(name="multi_step_workflow")
def multi_step_workflow(ctx: wf.DaprWorkflowContext, input_data: Dict[str, Any]):
    """Dapr workflow for multi-step clone → planning → approval → execution → testing.

    This workflow:
    0. Clone Phase (optional) - Clone repository if provided
    1. Planning Phase - Creates detailed plan with tasks and test cases
    2. Approval Phase - Wait for human approval (can be auto-approved)
    3. Execution Phase - Executes the plan
    4. Testing Phase - Verifies the implementation

    Each phase is a separate activity that appears in the UI workflow graph.
    """
    workflow_id = ctx.instance_id
    task = input_data.get("task", "")
    model = input_data.get("model", "gpt-5.2-codex")
    max_turns = input_data.get("max_turns", 20)
    max_test_retries = input_data.get("max_test_retries", 3)

    # Repository cloning configuration (optional)
    repository = input_data.get("repository")
    auto_approve = input_data.get("auto_approve", False)

    # Track workspace path - may be updated by clone
    workspace_path = input_data.get("workspace_dir", "/app/workspace")

    # --- Phase 0: Clone Repository (optional) ---
    if repository:
        ctx.set_custom_status(json.dumps({
            "phase": "cloning",
            "progress": 5,
            "message": f"Cloning {repository.get('owner')}/{repository.get('repo')}@{repository.get('branch', 'main')}...",
        }))

        clone_result = yield ctx.call_activity(
            clone_repository_activity,
            input={
                "owner": repository.get("owner"),
                "repo": repository.get("repo"),
                "branch": repository.get("branch", "main"),
                "token": repository.get("token"),
                "workspace_dir": workspace_path,
                "workflow_id": workflow_id,
            }
        )

        if not clone_result.get("success"):
            error = clone_result.get("error", "Unknown clone error")
            ctx.set_custom_status(json.dumps({
                "phase": "failed",
                "progress": 0,
                "message": f"Clone failed: {error}",
            }))
            return {
                "success": False,
                "workflow_id": workflow_id,
                "phase": "cloning",
                "error": error,
            }

        # Update workspace path to cloned repo
        workspace_path = clone_result.get("path", workspace_path)
        logger.info(f"Repository cloned to {workspace_path} ({clone_result.get('file_count', 0)} files)")

    # --- Phase 1: Planning ---
    ctx.set_custom_status(json.dumps({
        "phase": "planning",
        "progress": 10,
        "message": "Creating implementation plan with tasks and test cases...",
    }))

    planning_result = yield ctx.call_activity(
        planning_activity,
        input={
            "task": task,
            "model": model,
            "max_turns": max_turns,
            "workspace_path": workspace_path,
            "workflow_id": workflow_id,
        }
    )

    if not planning_result.get("success"):
        error = planning_result.get("error", "Unknown error")
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Planning failed: {error}",
        }))
        return {
            "success": False,
            "workflow_id": workflow_id,
            "phase": "planning",
            "error": error,
        }

    plan = planning_result.get("plan", {})

    # --- Phase 2: Approval Gate ---
    if not auto_approve:
        ctx.set_custom_status(json.dumps({
            "phase": "awaiting_approval",
            "progress": 40,
            "message": f"Plan ready for review. {len(plan.get('tasks', []))} tasks pending approval...",
            "plan": plan,
        }))

        # Wait for external approval event (24h timeout handled by Dapr)
        logger.info(f"Workflow {workflow_id} waiting for approval event")
        approval = yield ctx.wait_for_external_event("approval")
        logger.info(f"Workflow {workflow_id} received approval event: {approval}")

        if not approval.get("approved", False):
            reason = approval.get("reason", "Plan rejected by user")
            ctx.set_custom_status(json.dumps({
                "phase": "rejected",
                "progress": 0,
                "message": f"Plan rejected: {reason}",
            }))
            return {
                "success": False,
                "workflow_id": workflow_id,
                "phase": "rejected",
                "reason": reason,
                "plan": plan,
            }

    # --- Phase 3: Execution ---
    ctx.set_custom_status(json.dumps({
        "phase": "execution",
        "progress": 50,
        "message": f"Executing {len(plan.get('tasks', []))} tasks...",
    }))

    execution_result = yield ctx.call_activity(
        execution_activity,
        input={
            "plan": plan,
            "model": model,
            "max_turns": max_turns,
            "workspace_path": workspace_path,
            "workflow_id": workflow_id,
        }
    )

    if not execution_result.get("success"):
        error = execution_result.get("error", "Unknown error")
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Execution failed: {error}",
        }))
        return {
            "success": False,
            "workflow_id": workflow_id,
            "phase": "execution",
            "error": error,
            "plan": plan,
        }

    execution = execution_result.get("execution", {})

    # --- Phase 4: Testing ---
    ctx.set_custom_status(json.dumps({
        "phase": "testing",
        "progress": 85,
        "message": f"Running {len(plan.get('tests', []))} test cases...",
    }))

    testing_result = yield ctx.call_activity(
        testing_activity,
        input={
            "plan": plan,
            "execution": execution,
            "model": model,
            "max_turns": max_turns,
            "max_test_retries": max_test_retries,
            "workspace_path": workspace_path,
            "workflow_id": workflow_id,
        }
    )

    if not testing_result.get("success"):
        error = testing_result.get("error", "Unknown error")
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Testing failed: {error}",
        }))
        return {
            "success": False,
            "workflow_id": workflow_id,
            "phase": "testing",
            "error": error,
            "plan": plan,
            "execution": execution,
        }

    testing = testing_result.get("testing", {})
    passed = testing.get("passed", False)

    # --- Completed ---
    ctx.set_custom_status(json.dumps({
        "phase": "completed" if passed else "tests_failed",
        "progress": 100,
        "message": "All tests passed!" if passed else f"Tests failed: {testing.get('tests_failed', 0)} failures",
    }))

    return {
        "success": passed,
        "workflow_id": workflow_id,
        "status": "completed" if passed else "failed",
        "plan": plan,
        "execution": execution,
        "testing": testing,
    }


def get_workflow_runtime() -> wf.WorkflowRuntime:
    """Get the workflow runtime instance."""
    return wfr
