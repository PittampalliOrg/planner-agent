"""
Durable Planner Agent - Dapr Workflow with Claude Agent SDK Native Tools

Uses Dapr Workflow for durability and fault tolerance, with ClaudeSDKClient
providing native Claude Code tools (Read, Write, Edit, Bash, Glob, Grep, etc.)

Architecture:
- Dapr Workflow: Orchestrates activities, persists state, handles retries
- ClaudeSDKClient: Runs inside activities with ALL native Claude Code tools
- No custom tools needed - uses claude_code preset for full tool access

This approach gives us:
- Native Claude tools with full capabilities
- Dapr durability at workflow level
- Claude native OTEL tracing
- Fault tolerance with activity retries
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
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

# Import Claude Agent SDK for native tools
try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        ToolResultBlock,  # For proper tool result correlation
    )
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    ClaudeSDKClient = None
    ClaudeAgentOptions = None

from plan_manager import PlanManager
from streaming import (
    stream_tool_call,
    stream_tool_result,
    stream_llm_chunk,
    stream_execution_started,
    stream_execution_completed,
    stream_execution_failed,
    stream_phase_changed,
    stream_phase_changed_sync,  # Sync version for workflow context
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


class ExploreInput(BaseModel):
    """Input for codebase exploration activity."""
    cwd: str
    query: str


class ExploreOutput(BaseModel):
    """Output from codebase exploration activity."""
    success: bool
    findings: str
    error: str | None = None


class PlanInput(BaseModel):
    """Input for plan creation activity."""
    cwd: str
    feature_request: str
    plans_dir: str
    workflow_id: str | None = None


class PlanOutput(BaseModel):
    """Output from plan creation activity."""
    success: bool
    plan_id: str | None = None
    title: str | None = None
    summary: str | None = None
    steps_count: int = 0
    status: str | None = None
    error: str | None = None


class ExecuteInput(BaseModel):
    """Input for plan execution activity."""
    cwd: str
    plan_id: str
    workflow_id: str
    plans_dir: str


class ExecuteOutput(BaseModel):
    """Output from plan execution activity."""
    success: bool
    tasks_completed: int = 0
    tasks_total: int = 0
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None


# =============================================================================
# Workflow Activities (use ClaudeSDKClient with native tools)
# =============================================================================

async def _run_claude_session(
    cwd: str,
    prompt: str,
    system_prompt: str,
    workflow_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """
    Run a Claude SDK session with native tools.

    This is the core function that leverages ClaudeSDKClient with the
    claude_code preset for ALL native tools.

    Args:
        cwd: Working directory for the session
        prompt: The prompt to send to Claude
        system_prompt: System prompt for the session
        workflow_id: Optional workflow ID for streaming events to UI
        task_id: Optional task ID for event correlation
    """
    if not CLAUDE_SDK_AVAILABLE:
        raise RuntimeError("Claude Agent SDK not available")

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        # Use claude_code preset for ALL native tools:
        # Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, Task, etc.
        tools={"type": "preset", "preset": "claude_code"},
        # Full permissions - equivalent to --dangerously-skip-permissions
        permission_mode="bypassPermissions",
        cwd=cwd,
    )

    results = []
    # Track pending tool calls for correlation with results
    # Use a list as a queue since ResultMessage doesn't have tool_use_id
    pending_tool_calls: list[tuple[str, str]] = []  # [(tool_id, tool_name), ...]

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            # Debug: Log message type and attributes
            msg_type = type(message).__name__
            msg_attrs = [attr for attr in dir(message) if not attr.startswith('_')]
            print(f"[Claude Session] Message type: {msg_type}, attrs: {msg_attrs[:10]}")

            # Handle AssistantMessage with content blocks
            if hasattr(message, 'content'):
                for block in message.content:
                    # Handle TextBlock - LLM reasoning/text
                    if hasattr(block, 'text') and not hasattr(block, 'tool_use_id'):
                        results.append(block.text)
                        # Stream LLM text to UI
                        if workflow_id:
                            await stream_llm_chunk(workflow_id, block.text, task_id)

                    # Handle ToolUseBlock - tool calls (has 'name' and 'input')
                    elif hasattr(block, 'name') and hasattr(block, 'input'):
                        tool_name = block.name
                        tool_input = block.input if hasattr(block, 'input') else {}
                        tool_id = block.id if hasattr(block, 'id') and block.id else f"fallback-{uuid.uuid4().hex[:12]}"

                        # Track for correlation with ToolResultBlock
                        pending_tool_calls.append((tool_id, tool_name))

                        # Stream tool call to UI
                        if workflow_id:
                            await stream_tool_call(
                                workflow_id,
                                tool_name,
                                tool_input if isinstance(tool_input, dict) else {},
                                task_id,
                                call_id=tool_id,
                            )
                        print(f"[Claude Session] Tool call: {tool_name} ({tool_id[:20]}...)")

                    # Handle ToolResultBlock - individual tool results with tool_use_id
                    elif hasattr(block, 'tool_use_id'):
                        tool_use_id = block.tool_use_id
                        result_content = block.content if hasattr(block, 'content') else ""
                        is_error = block.is_error if hasattr(block, 'is_error') else False

                        # Convert content to string if needed
                        if isinstance(result_content, list):
                            result_content = str(result_content)
                        elif result_content is None:
                            result_content = ""

                        # Find matching tool call by tool_use_id
                        matching_idx = None
                        for idx, (tid, tname) in enumerate(pending_tool_calls):
                            if tid == tool_use_id:
                                matching_idx = idx
                                break

                        if matching_idx is not None:
                            tool_id, tool_name = pending_tool_calls.pop(matching_idx)
                        else:
                            # No match - use the tool_use_id directly
                            tool_id, tool_name = tool_use_id, "tool"

                        # Stream individual tool result with proper correlation
                        if workflow_id:
                            await stream_tool_result(
                                workflow_id,
                                tool_name,
                                str(result_content),
                                is_error or False,
                                task_id,
                                call_id=tool_id,
                            )
                        print(f"[Claude Session] Tool result: {tool_name} ({tool_id[:20]}...), pending: {len(pending_tool_calls)}")

            # Handle ResultMessage - session summary/final result
            # Note: Individual tool results come through ToolResultBlock in message.content
            # ResultMessage is typically a session-level summary
            elif hasattr(message, 'result'):
                result_text = str(message.result) if message.result else ""
                is_error = getattr(message, 'is_error', False)
                print(f"[Claude Session] ResultMessage received (session summary), pending: {len(pending_tool_calls)}")

                # If there are still pending tool calls, they may have been handled by
                # ToolResultBlock already, or this is a consolidated summary
                if pending_tool_calls:
                    print(f"[Claude Session] Draining {len(pending_tool_calls)} remaining tool calls")
                    remaining_calls = pending_tool_calls.copy()
                    pending_tool_calls.clear()

                    for remaining_id, remaining_name in remaining_calls:
                        if workflow_id:
                            await stream_tool_result(
                                workflow_id,
                                remaining_name,
                                f"[Completed - see session summary]",
                                is_error,
                                task_id,
                                call_id=remaining_id,
                            )
                        print(f"[Claude Session] Drained {remaining_name} ({remaining_id[:20]}...)")

    # Log any unmatched tool calls at session end
    if pending_tool_calls:
        print(f"[Claude Session] WARNING: {len(pending_tool_calls)} unmatched tool calls remaining")
        for tid, tname in pending_tool_calls:
            print(f"  - {tname}: {tid}")

    return "\n".join(results) if results else "Session complete"


class ExploreInputWithWorkflow(BaseModel):
    """Input for codebase exploration activity with optional workflow ID."""
    cwd: str
    query: str
    workflow_id: str | None = None


def clone_repository_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Clone a GitHub repository.

    Clones the repository to the workspace directory and returns the path.
    Streams progress to UI if workflow_id is provided.
    """
    import shutil
    import subprocess

    input_data = CloneInput.model_validate_json(input_json)
    workflow_id = input_data.workflow_id

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


def explore_codebase_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Explore codebase using Claude SDK native tools.

    Uses ClaudeSDKClient with claude_code preset for intelligent exploration
    using native Read, Glob, Grep, and other tools.
    Streams tool calls and LLM chunks to UI if workflow_id is provided.
    """
    # Try new model with workflow_id, fall back to old model
    try:
        input_data = ExploreInputWithWorkflow.model_validate_json(input_json)
        workflow_id = input_data.workflow_id
    except Exception:
        input_data = ExploreInput.model_validate_json(input_json)
        workflow_id = None

    print(f"[Activity] Exploring codebase at {input_data.cwd}: {input_data.query[:100]}...")

    system_prompt = """You are a code exploration assistant.
Your task is to explore the codebase and understand its structure.
Use the available tools (Read, Glob, Grep, Bash) to find relevant files and patterns.
Be thorough but concise in your findings.
Focus on understanding the architecture, key files, and patterns."""

    try:
        # Run the async Claude session in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            findings = loop.run_until_complete(
                _run_claude_session(
                    cwd=input_data.cwd,
                    prompt=f"Explore the codebase: {input_data.query}",
                    system_prompt=system_prompt,
                    workflow_id=workflow_id,
                    task_id="exploration",
                )
            )
        finally:
            loop.close()

        output = ExploreOutput(success=True, findings=findings)

    except Exception as e:
        print(f"[Activity] Exploration failed: {e}")
        output = ExploreOutput(success=False, findings="", error=str(e))

    return output.model_dump()


def create_plan_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Create implementation plan using Claude SDK native tools.

    Uses ClaudeSDKClient with claude_code preset for intelligent planning
    with access to Read, Write, Edit, and other native tools.
    Streams tool calls and LLM chunks to UI if workflow_id is provided.
    """
    input_data = PlanInput.model_validate_json(input_json)
    workflow_id = input_data.workflow_id
    print(f"[Activity] Creating plan for: {input_data.feature_request[:100]}...")

    system_prompt = f"""You are a software planning assistant.
Your task is to create a detailed implementation plan for the requested feature.

Use the available tools to:
1. Explore the codebase structure (Glob, Grep, Read)
2. Understand existing patterns and architecture
3. Create a comprehensive plan

After exploring, create a plan file at {input_data.plans_dir}/plan_1.json with this structure:
{{
    "id": "plan_1",
    "title": "Brief title",
    "summary": "Detailed summary of what will be implemented",
    "status": "draft",
    "steps": [
        {{"number": 1, "title": "Step title", "description": "What to do"}}
    ],
    "critical_files": ["file1.py", "file2.py"],
    "considerations": ["Note about approach"],
    "created_at": "ISO timestamp"
}}

Be thorough in exploration and detailed in planning."""

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _run_claude_session(
                    cwd=input_data.cwd,
                    prompt=f"""Create an implementation plan for this feature request:

{input_data.feature_request}

Steps:
1. First explore the codebase at {input_data.cwd} to understand its structure
2. Then create a detailed implementation plan and save it to {input_data.plans_dir}/plan_1.json""",
                    system_prompt=system_prompt,
                    workflow_id=workflow_id,
                    task_id="planning",
                )
            )
        finally:
            loop.close()

        # Check if plan was created
        plan_manager = PlanManager(input_data.plans_dir)
        plans = plan_manager.list_plans()

        if plans:
            plan = plan_manager.load_plan(plans[-1])
            if plan:
                output = PlanOutput(
                    success=True,
                    plan_id=plan.id,
                    title=plan.title,
                    summary=plan.summary,
                    steps_count=len(plan.steps),
                    status=plan.status,
                )
            else:
                output = PlanOutput(success=False, error="Plan file exists but could not be loaded")
        else:
            output = PlanOutput(success=False, error="No plan file was created")

    except Exception as e:
        print(f"[Activity] Plan creation failed: {e}")
        output = PlanOutput(success=False, error=str(e))

    return output.model_dump()


def execute_plan_activity(ctx: WorkflowActivityContext, input_json: str) -> dict:
    """
    Dapr Workflow Activity: Execute approved plan using Claude SDK native tools.

    Uses ClaudeSDKClient with claude_code preset for intelligent execution
    with access to Read, Write, Edit, Bash, and other native tools.
    Streams tool calls, results, and LLM chunks to the UI via pub/sub.
    """
    input_data = ExecuteInput.model_validate_json(input_json)
    workflow_id = input_data.workflow_id
    print(f"[Activity] Executing plan {input_data.plan_id} (workflow: {workflow_id})...")

    # Load the plan
    plan_manager = PlanManager(input_data.plans_dir)
    plan = plan_manager.load_plan(input_data.plan_id)

    if not plan:
        # Stream failure event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(stream_execution_failed(workflow_id, f"Plan {input_data.plan_id} not found", 0))
        finally:
            loop.close()
        return ExecuteOutput(success=False, error=f"Plan {input_data.plan_id} not found").model_dump()

    if plan.status != "approved":
        # Stream failure event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(stream_execution_failed(workflow_id, f"Plan status is '{plan.status}', not 'approved'", 0))
        finally:
            loop.close()
        return ExecuteOutput(success=False, error=f"Plan status is '{plan.status}', not 'approved'").model_dump()

    # Build execution prompt from plan steps
    steps_text = "\n".join([
        f"{i+1}. {step.title}: {step.description}"
        for i, step in enumerate(plan.steps)
    ])

    system_prompt = """You are a software implementation assistant.
Your task is to execute the approved implementation plan step by step.

Use the available tools to:
1. Read existing files to understand context
2. Write new files or Edit existing files as needed
3. Run Bash commands for testing or setup
4. Verify your changes work correctly

Execute each step carefully and verify the results before moving to the next step.
If a step fails, report the error and stop."""

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Stream execution started
            loop.run_until_complete(stream_execution_started(workflow_id, input_data.plan_id, len(plan.steps)))

            result = loop.run_until_complete(
                _run_claude_session(
                    cwd=input_data.cwd,
                    prompt=f"""Execute this approved implementation plan:

Plan: {plan.title}
Summary: {plan.summary}

Steps to execute:
{steps_text}

Execute each step in order, using the available tools (Read, Write, Edit, Bash, etc.).
Verify each step completes successfully before proceeding.""",
                    system_prompt=system_prompt,
                    workflow_id=workflow_id,
                    task_id=input_data.plan_id,
                )
            )

            # Stream execution completed
            loop.run_until_complete(stream_execution_completed(workflow_id, len(plan.steps), []))
        finally:
            loop.close()

        # Mark plan as completed
        plan.status = "completed"
        plan.updated_at = datetime.now().isoformat()
        plan_manager.save_plan()

        output = ExecuteOutput(
            success=True,
            tasks_completed=len(plan.steps),
            tasks_total=len(plan.steps),
            files_changed=[],  # TODO: Track changed files
        )

    except Exception as e:
        print(f"[Activity] Execution failed: {e}")
        # Stream failure event
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(stream_execution_failed(workflow_id, str(e), 0))
        finally:
            loop.close()
        output = ExecuteOutput(success=False, error=str(e))

    return output.model_dump()


# =============================================================================
# Dapr Workflows
# =============================================================================

def planning_workflow(ctx: DaprWorkflowContext, input_json: str) -> dict:
    """
    Dapr Workflow: Create an implementation plan.

    This workflow orchestrates:
    1. Codebase exploration (optional, for context)
    2. Plan creation using Claude SDK native tools

    The workflow is durable - it survives crashes and restarts.
    Streams tool calls and LLM chunks to UI via ctx.instance_id.
    """
    input_data = json.loads(input_json)
    cwd = input_data.get("cwd", str(WORKSPACE_DIR))
    feature_request = input_data.get("feature_request", "")
    plans_dir = input_data.get("plans_dir", str(PLANS_DIR))
    skip_exploration = input_data.get("skip_exploration", False)
    # Prefer UI workflow ID for streaming events, fallback to Dapr instance ID
    workflow_id = input_data.get("workflow_id") or ctx.instance_id

    # Set initial status
    ctx.set_custom_status(json.dumps({
        "phase": "exploration" if not skip_exploration else "planning",
        "progress": 0,
        "message": "Starting planning workflow...",
    }))

    # Step 1: Explore codebase (optional)
    if not skip_exploration:
        ctx.set_custom_status(json.dumps({
            "phase": "exploration",
            "progress": 10,
            "message": "Exploring codebase...",
        }))

        explore_input = ExploreInputWithWorkflow(
            cwd=cwd,
            query=f"Understand the codebase structure for implementing: {feature_request[:200]}",
            workflow_id=workflow_id,
        )
        explore_result = yield ctx.call_activity(
            explore_codebase_activity,
            input=explore_input.model_dump_json(),
        )
        # Result is already a dict from the activity
        explore_output = ExploreOutput.model_validate(explore_result)

        if not explore_output.success:
            ctx.set_custom_status(json.dumps({
                "phase": "failed",
                "progress": 0,
                "message": f"Exploration failed: {explore_output.error}",
            }))
            return PlanOutput(success=False, error=f"Exploration failed: {explore_output.error}").model_dump()

    # Step 2: Create plan
    ctx.set_custom_status(json.dumps({
        "phase": "planning",
        "progress": 50,
        "message": "Creating implementation plan...",
    }))
    plan_input = PlanInput(
        cwd=cwd,
        feature_request=feature_request,
        plans_dir=plans_dir,
        workflow_id=workflow_id,
    )
    plan_result = yield ctx.call_activity(
        create_plan_activity,
        input=plan_input.model_dump_json(),
    )

    # Set final status
    plan_output = PlanOutput.model_validate(plan_result)
    if plan_output.success:
        ctx.set_custom_status(json.dumps({
            "phase": "completed",
            "progress": 100,
            "message": "Plan created successfully",
            "plan_id": plan_output.plan_id,
        }))
    else:
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Planning failed: {plan_output.error}",
        }))

    # Result is already a dict from the activity
    return plan_result


def execution_workflow(ctx: DaprWorkflowContext, input_json: str) -> dict:
    """
    Dapr Workflow: Execute an approved plan.

    This workflow orchestrates plan execution using Claude SDK native tools.
    The workflow is durable - it survives crashes and restarts.
    """
    input_data = json.loads(input_json)
    cwd = input_data.get("cwd", str(WORKSPACE_DIR))
    plan_id = input_data.get("plan_id", "")
    workflow_id = input_data.get("workflow_id", "")
    plans_dir = input_data.get("plans_dir", str(PLANS_DIR))

    # Set initial status
    ctx.set_custom_status(json.dumps({
        "phase": "executing",
        "progress": 10,
        "message": f"Starting execution of plan {plan_id}...",
        "plan_id": plan_id,
    }))

    execute_input = ExecuteInput(
        cwd=cwd,
        plan_id=plan_id,
        workflow_id=workflow_id,
        plans_dir=plans_dir,
    )

    execute_result = yield ctx.call_activity(
        execute_plan_activity,
        input=execute_input.model_dump_json(),
    )

    # Set final status
    execute_output = ExecuteOutput.model_validate(execute_result)
    if execute_output.success:
        ctx.set_custom_status(json.dumps({
            "phase": "completed",
            "progress": 100,
            "message": "Execution completed successfully",
            "tasks_completed": execute_output.tasks_completed,
            "tasks_total": execute_output.tasks_total,
        }))
    else:
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Execution failed: {execute_output.error}",
        }))

    # Result is already a dict from the activity
    return execute_result


def planning_and_execution_workflow(ctx: DaprWorkflowContext, input_json: str) -> dict:
    """
    Dapr Workflow: Combined planning and execution with approval gate.

    This workflow orchestrates:
    0. Clone repository (optional, if repository info provided)
    1. Codebase exploration (optional)
    2. Plan creation using Claude SDK native tools
    3. Wait for external approval event (Dapr native pattern)
    4. Execute the approved plan

    Uses wait_for_external_event for the approval gate - the workflow
    properly pauses without polling until the approval event is raised.

    The workflow is durable - it survives crashes and restarts.
    Streams tool calls and LLM chunks to UI via ctx.instance_id.

    Custom status is set at each phase for deterministic phase tracking:
    - clone: Cloning repository
    - exploration: Exploring codebase structure
    - planning: Creating implementation plan
    - awaiting_approval: Plan ready, waiting for user approval
    - executing: Running implementation
    - completed: Workflow finished successfully
    - failed: Workflow encountered an error
    """
    input_data = json.loads(input_json)
    cwd = input_data.get("cwd", str(WORKSPACE_DIR))
    feature_request = input_data.get("feature_request", "")
    plans_dir = input_data.get("plans_dir", str(PLANS_DIR))
    skip_exploration = input_data.get("skip_exploration", False)
    approval_timeout_minutes = input_data.get("approval_timeout_minutes", 60)
    # Prefer UI workflow ID for streaming events, fallback to Dapr instance ID
    workflow_id = input_data.get("workflow_id") or ctx.instance_id

    # Repository info for cloning (optional)
    repository = input_data.get("repository")

    # Step 0: Clone repository (if repository info provided)
    if repository:
        ctx.set_custom_status(json.dumps({
            "phase": "clone",
            "progress": 5,
            "message": f"Cloning {repository.get('owner')}/{repository.get('repo')}...",
        }))

        # Stream phase change for UI
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

    # Set initial status for exploration/planning
    ctx.set_custom_status(json.dumps({
        "phase": "exploration" if not skip_exploration else "planning",
        "progress": 10 if repository else 0,
        "message": "Starting workflow...",
    }))

    # Step 1: Explore codebase (optional)
    if not skip_exploration:
        ctx.set_custom_status(json.dumps({
            "phase": "exploration",
            "progress": 10,
            "message": "Exploring codebase structure...",
        }))

        explore_input = ExploreInputWithWorkflow(
            cwd=cwd,
            query=f"Understand the codebase structure for implementing: {feature_request[:200]}",
            workflow_id=workflow_id,
        )
        explore_result = yield ctx.call_activity(
            explore_codebase_activity,
            input=explore_input.model_dump_json(),
        )
        explore_output = ExploreOutput.model_validate(explore_result)

        if not explore_output.success:
            ctx.set_custom_status(json.dumps({
                "phase": "failed",
                "progress": 0,
                "message": f"Exploration failed: {explore_output.error}",
            }))
            return {
                "success": False,
                "error": f"Exploration failed: {explore_output.error}",
                "phase": "exploration",
            }

    # Step 2: Create plan
    ctx.set_custom_status(json.dumps({
        "phase": "planning",
        "progress": 25,
        "message": "Creating implementation plan...",
    }))
    plan_input = PlanInput(
        cwd=cwd,
        feature_request=feature_request,
        plans_dir=plans_dir,
        workflow_id=workflow_id,
    )
    plan_result = yield ctx.call_activity(
        create_plan_activity,
        input=plan_input.model_dump_json(),
    )

    plan_output = PlanOutput.model_validate(plan_result)

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

    plan_id = plan_output.plan_id
    print(f"[Workflow] Plan created: id={plan_id}, title={plan_output.title}")

    # Update status: Plan ready, awaiting approval
    print(f"[Workflow] Setting awaiting_approval status...")
    ctx.set_custom_status(json.dumps({
        "phase": "awaiting_approval",
        "progress": 50,
        "message": "Plan ready for approval",
        "plan_id": plan_id,
        "plan_title": plan_output.title,
    }))

    # Stream phase change event for UI detection
    # This is critical - the UI falls back to event-based phase detection
    # when the status endpoint times out (Dapr state lookups can be slow)
    print(f"[Workflow] Streaming awaiting_approval phase change...")
    stream_phase_changed_sync(
        workflow_id=ctx.instance_id,
        phase="awaiting_approval",
        status="Waiting for plan approval",  # UI detects this text
        progress=50,
        plan_id=plan_id,
        extra_data={"plan_title": plan_output.title},
    )

    # Step 3: Wait for approval (Dapr native external event pattern)
    # The workflow properly pauses here until the event is raised
    # No polling needed - Dapr handles this efficiently
    # Note: Dapr SDK 1.17+ doesn't support timeout on wait_for_external_event,
    # approval will wait indefinitely until the event is raised
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
            "title": plan_output.title,
            "summary": plan_output.summary,
            "status": "rejected",
            "error": f"Plan rejected: {reason}",
            "phase": "approval",
        }

    # Step 4: Update plan status to approved
    ctx.set_custom_status(json.dumps({
        "phase": "executing",
        "progress": 60,
        "message": "Plan approved, starting execution...",
        "plan_id": plan_id,
    }))
    plan_manager = PlanManager(plans_dir)
    plan = plan_manager.load_plan(plan_id)
    if plan:
        plan.status = "approved"
        plan.approved_at = datetime.now().isoformat()
        plan_manager.save_plan()

    # Step 5: Execute the approved plan
    execute_input = ExecuteInput(
        cwd=cwd,
        plan_id=plan_id,
        workflow_id=ctx.instance_id,
        plans_dir=plans_dir,
    )

    execute_result = yield ctx.call_activity(
        execute_plan_activity,
        input=execute_input.model_dump_json(),
    )

    execute_output = ExecuteOutput.model_validate(execute_result)

    # Set final status
    if execute_output.success:
        ctx.set_custom_status(json.dumps({
            "phase": "completed",
            "progress": 100,
            "message": "Workflow completed successfully",
            "tasks_completed": execute_output.tasks_completed,
            "tasks_total": execute_output.tasks_total,
        }))
    else:
        ctx.set_custom_status(json.dumps({
            "phase": "failed",
            "progress": 0,
            "message": f"Execution failed: {execute_output.error}",
        }))

    return {
        "success": execute_output.success,
        "plan_id": plan_id,
        "title": plan_output.title,
        "summary": plan_output.summary,
        "status": "completed" if execute_output.success else "failed",
        "tasks_completed": execute_output.tasks_completed,
        "tasks_total": execute_output.tasks_total,
        "files_changed": execute_output.files_changed,
        "error": execute_output.error,
        "phase": "execution" if not execute_output.success else "complete",
    }


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

        # Register workflows
        _workflow_runtime.register_workflow(planning_workflow)
        _workflow_runtime.register_workflow(execution_workflow)
        _workflow_runtime.register_workflow(planning_and_execution_workflow)

        # Register activities
        _workflow_runtime.register_activity(clone_repository_activity)
        _workflow_runtime.register_activity(explore_codebase_activity)
        _workflow_runtime.register_activity(create_plan_activity)
        _workflow_runtime.register_activity(execute_plan_activity)

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
# High-Level API (compatible with existing code)
# =============================================================================

class DurablePlanningWorkflow:
    """
    High-level workflow interface for durable planning.

    This class provides a simple API that wraps Dapr Workflows,
    maintaining compatibility with the existing codebase.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str | Path,
        plans_dir: str | Path | None = None,
        workflow_id: str | None = None,
        **kwargs,  # Ignore other args for compatibility
    ):
        self.session_id = session_id
        self.cwd = Path(cwd)
        self.plans_dir = Path(plans_dir) if plans_dir else PLANS_DIR
        self.workflow_id = workflow_id  # UI workflow ID for streaming

    async def run_planning(self, feature_request: str) -> dict[str, Any]:
        """Run the planning workflow."""
        client = get_workflow_client()

        if client is None:
            # Fallback to direct execution without durability
            return await self._run_planning_direct(feature_request)

        try:
            instance_id = f"plan-{self.session_id}-{datetime.now().strftime('%H%M%S')}"

            workflow_input = json.dumps({
                "cwd": str(self.cwd),
                "feature_request": feature_request,
                "plans_dir": str(self.plans_dir),
                "workflow_id": self.workflow_id,  # UI workflow ID for streaming
            })

            # Start the workflow
            client.schedule_new_workflow(
                workflow=planning_workflow,
                input=workflow_input,
                instance_id=instance_id,
            )

            # Wait for completion
            state = client.wait_for_workflow_completion(
                instance_id=instance_id,
                timeout_in_seconds=300,
            )

            if state and state.runtime_status.name == "COMPLETED":
                # serialized_output is a JSON string, parse it
                result_data = json.loads(state.serialized_output)
                # If it's still a string (double-encoded), parse again
                if isinstance(result_data, str):
                    result_data = json.loads(result_data)
                return result_data
            else:
                return {"success": False, "error": f"Workflow failed: {state.failure_details if state else 'unknown'}"}

        except Exception as e:
            print(f"[DurableWorkflow] Workflow failed: {e}, falling back to direct execution")
            return await self._run_planning_direct(feature_request)

    async def _run_planning_direct(self, feature_request: str) -> dict[str, Any]:
        """Direct planning without workflow durability."""
        print("[DurableWorkflow] Running direct planning (no durability)")

        try:
            result = await _run_claude_session(
                cwd=str(self.cwd),
                prompt=f"""Create an implementation plan for this feature request:

{feature_request}

Steps:
1. First explore the codebase to understand its structure
2. Then create a detailed implementation plan and save it to {self.plans_dir}/plan_1.json""",
                system_prompt="""You are a software planning assistant.
Create a detailed implementation plan. Save the plan as JSON to the plans directory.""",
            )

            # Check for created plan
            plan_manager = PlanManager(str(self.plans_dir))
            plans = plan_manager.list_plans()

            if plans:
                plan = plan_manager.load_plan(plans[-1])
                if plan:
                    return {
                        "success": True,
                        "plan_id": plan.id,
                        "title": plan.title,
                        "summary": plan.summary,
                        "steps_count": len(plan.steps),
                        "status": plan.status,
                    }

            return {"success": False, "error": "No plan was created"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_execution(self, plan_id: str, workflow_id: str) -> dict[str, Any]:
        """Run the execution workflow."""
        client = get_workflow_client()

        if client is None:
            return await self._run_execution_direct(plan_id, workflow_id)

        try:
            instance_id = f"exec-{plan_id}-{datetime.now().strftime('%H%M%S')}"

            workflow_input = json.dumps({
                "cwd": str(self.cwd),
                "plan_id": plan_id,
                "workflow_id": workflow_id,
                "plans_dir": str(self.plans_dir),
            })

            client.schedule_new_workflow(
                workflow=execution_workflow,
                input=workflow_input,
                instance_id=instance_id,
            )

            state = client.wait_for_workflow_completion(
                instance_id=instance_id,
                timeout_in_seconds=600,
            )

            if state and state.runtime_status.name == "COMPLETED":
                # serialized_output is a JSON string, parse it
                result_data = json.loads(state.serialized_output)
                # If it's still a string (double-encoded), parse again
                if isinstance(result_data, str):
                    result_data = json.loads(result_data)
                return result_data
            else:
                return {"success": False, "error": f"Workflow failed: {state.failure_details if state else 'unknown'}"}

        except Exception as e:
            print(f"[DurableWorkflow] Execution workflow failed: {e}")
            return await self._run_execution_direct(plan_id, workflow_id)

    async def _run_execution_direct(self, plan_id: str, workflow_id: str) -> dict[str, Any]:
        """Direct execution without workflow durability."""
        print("[DurableWorkflow] Running direct execution (no durability)")

        plan_manager = PlanManager(str(self.plans_dir))
        plan = plan_manager.load_plan(plan_id)

        if not plan:
            return {"success": False, "error": f"Plan {plan_id} not found"}

        steps_text = "\n".join([
            f"{i+1}. {step.title}: {step.description}"
            for i, step in enumerate(plan.steps)
        ])

        try:
            await _run_claude_session(
                cwd=str(self.cwd),
                prompt=f"""Execute this implementation plan:

{plan.title}
{plan.summary}

Steps:
{steps_text}""",
                system_prompt="You are a software implementation assistant. Execute the plan step by step.",
            )

            return {
                "success": True,
                "tasks_completed": len(plan.steps),
                "tasks_total": len(plan.steps),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Convenience Functions (maintain compatibility)
# =============================================================================

async def create_durable_plan(
    cwd: str,
    feature_request: str,
    session_id: str | None = None,
    workflow_id: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a plan using the durable workflow."""
    session_id = session_id or f"plan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    workflow = DurablePlanningWorkflow(session_id=session_id, cwd=cwd, workflow_id=workflow_id)
    return await workflow.run_planning(feature_request)


async def execute_durable_plan(
    cwd: str,
    plan_id: str,
    workflow_id: str,
    session_id: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Execute a plan using the durable workflow."""
    session_id = session_id or f"exec-{plan_id}"
    workflow = DurablePlanningWorkflow(session_id=session_id, cwd=cwd)
    return await workflow.run_execution(plan_id, workflow_id)


def is_durable_agents_available() -> bool:
    """Check if durable workflows are available."""
    return DAPR_WORKFLOW_AVAILABLE and CLAUDE_SDK_AVAILABLE


def get_durable_agent_status() -> dict[str, Any]:
    """Get status of durable agent dependencies."""
    return {
        "dapr_workflow_available": DAPR_WORKFLOW_AVAILABLE,
        "claude_sdk_available": CLAUDE_SDK_AVAILABLE,
        "workspace_dir": str(WORKSPACE_DIR),
        "plans_dir": str(PLANS_DIR),
    }


# Legacy compatibility - these are no longer used but kept for imports
def create_durable_planner_agent(*args, **kwargs):
    """Legacy function - returns None as DurableAgent is no longer used."""
    return None


DAPR_AGENTS_AVAILABLE = DAPR_WORKFLOW_AVAILABLE  # Compatibility alias


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Test the durable workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Durable Planner Agent")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory")
    parser.add_argument("--session-id", type=str, help="Session ID")
    parser.add_argument("prompt", nargs="?", type=str, help="Feature request")

    args = parser.parse_args()

    print("=" * 60)
    print("DURABLE PLANNER AGENT (Claude SDK Native Tools)")
    print("=" * 60)
    print(f"\nStatus: {get_durable_agent_status()}")

    if args.prompt:
        # Start runtime
        await start_workflow_runtime()

        try:
            result = await create_durable_plan(
                cwd=args.cwd,
                feature_request=args.prompt,
                session_id=args.session_id,
            )
            print(f"\nResult: {json.dumps(result, indent=2)}")
        finally:
            await stop_workflow_runtime()
    else:
        print("\nNo prompt provided. Use: python durable_agent.py --cwd /path 'feature request'")


if __name__ == "__main__":
    asyncio.run(main())
