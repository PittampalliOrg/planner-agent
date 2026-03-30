"""
Planner Agent - Claude Code Plan Mode Replica

This agent replicates Claude Code's plan mode functionality:
1. User provides a feature request/prompt
2. Agent explores the codebase to understand context
3. Agent creates an implementation plan
4. User reviews and approves the plan
5. Agent converts plan to tasks and begins implementation

Uses the Claude Agent SDK with custom MCP tools for task/plan management.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
    SystemMessage,
)

from task_manager import TaskManager, TaskStatus
from plan_manager import PlanManager
from streaming import (
    stream_execution_started,
    stream_execution_completed,
    stream_execution_failed,
    stream_task_progress,
    stream_tool_call,
    stream_tool_result,
    stream_file_changed,
)
from version import __version__


# =============================================================================
# Custom MCP Tools for Task Management
# =============================================================================

# Global instances for the tools to access
_task_manager: TaskManager | None = None
_plan_manager: PlanManager | None = None


@tool(
    "task_create",
    "Create a new task in the task list. Tasks track implementation work.",
    {
        "subject": str,
        "description": str,
        "active_form": str,
    }
)
async def task_create(args: dict[str, Any]) -> dict[str, Any]:
    """Create a new task."""
    if not _task_manager:
        return {"content": [{"type": "text", "text": "Error: Task manager not initialized"}], "is_error": True}

    task = _task_manager.create_task(
        subject=args["subject"],
        description=args["description"],
        active_form=args.get("active_form", f"{args['subject']}..."),
    )
    _task_manager.save()

    return {
        "content": [{
            "type": "text",
            "text": f"Created task {task.id}: {task.subject}"
        }]
    }


@tool(
    "task_update",
    "Update an existing task's status, dependencies, or other fields.",
    {
        "task_id": str,
        "status": str,  # "pending", "in_progress", or "completed"
        "add_blocked_by": str,  # Comma-separated task IDs
    }
)
async def task_update(args: dict[str, Any]) -> dict[str, Any]:
    """Update a task."""
    if not _task_manager:
        return {"content": [{"type": "text", "text": "Error: Task manager not initialized"}], "is_error": True}

    task_id = args["task_id"]
    status = args.get("status")
    add_blocked_by_str = args.get("add_blocked_by", "")

    add_blocked_by = [x.strip() for x in add_blocked_by_str.split(",") if x.strip()] if add_blocked_by_str else None

    task = _task_manager.update_task(
        task_id,
        status=TaskStatus(status) if status else None,
        add_blocked_by=add_blocked_by,
    )
    _task_manager.save()

    if not task:
        return {"content": [{"type": "text", "text": f"Task {task_id} not found"}], "is_error": True}

    return {
        "content": [{
            "type": "text",
            "text": f"Updated task {task.id}: status={task.status.value}"
        }]
    }


@tool(
    "task_list",
    "List all tasks in the task list with their status and dependencies.",
    {}
)
async def task_list(args: dict[str, Any]) -> dict[str, Any]:
    """List all tasks."""
    if not _task_manager:
        return {"content": [{"type": "text", "text": "Error: Task manager not initialized"}], "is_error": True}

    output = _task_manager.format_task_list()
    stats = _task_manager.get_task_stats()

    result = f"{output}\n\nStats: {stats['completed']}/{stats['total']} completed, {stats['in_progress']} in progress, {stats['pending']} pending"

    return {"content": [{"type": "text", "text": result}]}


@tool(
    "task_get",
    "Get detailed information about a specific task.",
    {"task_id": str}
)
async def task_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get a specific task."""
    if not _task_manager:
        return {"content": [{"type": "text", "text": "Error: Task manager not initialized"}], "is_error": True}

    task = _task_manager.get_task(args["task_id"])
    if not task:
        return {"content": [{"type": "text", "text": f"Task {args['task_id']} not found"}], "is_error": True}

    info = f"""Task {task.id}: {task.subject}
Status: {task.status.value}
Description: {task.description}
Active Form: {task.active_form}
Owner: {task.owner or 'unassigned'}
Blocked By: {', '.join(task.blocked_by) if task.blocked_by else 'none'}
Blocks: {', '.join(task.blocks) if task.blocks else 'none'}
Created: {task.created_at}
Updated: {task.updated_at}"""

    return {"content": [{"type": "text", "text": info}]}


@tool(
    "plan_create",
    "Create an implementation plan for the feature request. The plan should include a summary, implementation steps, critical files, and considerations.",
    {
        "title": str,
        "summary": str,
        "context": str,
        "steps_json": str,  # JSON array of steps with title, description, files_affected, complexity
        "critical_files_json": str,  # JSON array of file paths
        "considerations_json": str,  # JSON array of considerations
    }
)
async def plan_create(args: dict[str, Any]) -> dict[str, Any]:
    """Create an implementation plan."""
    if not _plan_manager:
        return {"content": [{"type": "text", "text": "Error: Plan manager not initialized"}], "is_error": True}

    try:
        steps = json.loads(args.get("steps_json", "[]"))
        critical_files = json.loads(args.get("critical_files_json", "[]"))
        considerations = json.loads(args.get("considerations_json", "[]"))
    except json.JSONDecodeError as e:
        return {"content": [{"type": "text", "text": f"JSON parse error: {e}"}], "is_error": True}

    plan = _plan_manager.create_plan(
        title=args["title"],
        summary=args["summary"],
        context=args.get("context", ""),
        steps=steps,
        critical_files=critical_files,
        considerations=considerations,
    )
    _plan_manager.save_plan()

    return {
        "content": [{
            "type": "text",
            "text": f"Created plan: {plan.id}\n\n{plan.format_markdown()}"
        }]
    }


@tool(
    "plan_get",
    "Get the current plan as formatted Markdown.",
    {}
)
async def plan_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get the current plan."""
    if not _plan_manager:
        return {"content": [{"type": "text", "text": "Error: Plan manager not initialized"}], "is_error": True}

    if not _plan_manager.current_plan:
        return {"content": [{"type": "text", "text": "No plan has been created yet."}]}

    return {
        "content": [{
            "type": "text",
            "text": _plan_manager.current_plan.format_markdown()
        }]
    }


@tool(
    "plan_convert_to_tasks",
    "Convert the approved plan into tasks. Each step becomes a task with proper dependencies.",
    {}
)
async def plan_convert_to_tasks(args: dict[str, Any]) -> dict[str, Any]:
    """Convert plan steps to tasks."""
    if not _plan_manager or not _task_manager:
        return {"content": [{"type": "text", "text": "Error: Managers not initialized"}], "is_error": True}

    if not _plan_manager.current_plan:
        return {"content": [{"type": "text", "text": "No plan exists to convert."}], "is_error": True}

    if _plan_manager.current_plan.status != "approved":
        return {"content": [{"type": "text", "text": "Plan must be approved before converting to tasks. Current status: " + _plan_manager.current_plan.status}], "is_error": True}

    tasks = _plan_manager.convert_plan_to_tasks(_task_manager)
    _task_manager.save()
    _plan_manager.save_plan()

    task_list = "\n".join([f"  - Task {t.id}: {t.subject}" for t in tasks])
    return {
        "content": [{
            "type": "text",
            "text": f"Created {len(tasks)} tasks from plan:\n{task_list}"
        }]
    }


# =============================================================================
# Planner Agent Class
# =============================================================================

class PlannerAgent:
    """
    Planner Agent that replicates Claude Code's plan mode.

    The agent follows this workflow:
    1. Receive feature request from user
    2. Explore codebase to understand context (using Read, Glob, Grep)
    3. Create an implementation plan
    4. Present plan to user for approval
    5. On approval, convert plan to tasks
    6. Execute tasks sequentially
    """

    def __init__(
        self,
        cwd: str | Path | None = None,
        plans_dir: str | Path | None = None,
    ):
        """
        Initialize the planner agent.

        Args:
            cwd: Working directory for the agent (the git repo to work in)
            plans_dir: Directory to store plans and tasks
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.plans_dir = Path(plans_dir) if plans_dir else self.cwd / "plans"

        # Initialize managers
        global _task_manager, _plan_manager
        _task_manager = TaskManager(self.plans_dir / "tasks.json")
        _plan_manager = PlanManager(self.plans_dir)

        self.task_manager = _task_manager
        self.plan_manager = _plan_manager

        # Create MCP server with our tools
        self.mcp_server = create_sdk_mcp_server(
            name="planner",
            version=__version__,
            tools=[
                task_create,
                task_update,
                task_list,
                task_get,
                plan_create,
                plan_get,
                plan_convert_to_tasks,
            ]
        )

        self.client: ClaudeSDKClient | None = None
        self.session_id: str | None = None

    def _get_planning_system_prompt(self) -> str:
        """Get the system prompt for planning mode."""
        return """You are a software planning agent that helps users implement new features in their codebase.

## Your Workflow

You operate in a structured planning workflow:

### Phase 1: Exploration
When the user provides a feature request, first explore the codebase to understand:
- The existing code structure and patterns
- Relevant files that may need modification
- Dependencies and architectural considerations

Use the Read, Glob, and Grep tools to explore the codebase.

### Phase 2: Planning
Once you understand the codebase, create an implementation plan using the plan_create tool.
Your plan should include:
- A clear summary of what will be implemented
- Context you gathered during exploration
- Ordered implementation steps (each with title, description, affected files, complexity)
- List of critical files that will be modified
- Architectural considerations or trade-offs

### Phase 3: Review
After creating the plan, present it to the user and wait for their approval.
Ask if they want to modify anything before proceeding.

### Phase 4: Task Creation
Once the user approves the plan, use plan_convert_to_tasks to convert the plan into actionable tasks.
Each step becomes a task with proper dependencies (each task depends on the previous one).

### Phase 5: Implementation
After tasks are created, work through them sequentially:
1. Use task_list to see available tasks
2. Use task_update to mark a task as in_progress when starting
3. Implement the task using Read, Write, Edit, and Bash tools
4. Use task_update to mark the task as completed when done
5. Move to the next unblocked task

## Guidelines

- Always explore first before planning - don't make assumptions about the codebase
- Create plans that are detailed enough for any developer to follow
- Each step should be atomic and testable
- Consider edge cases and error handling in your plan
- Ask clarifying questions if the feature request is ambiguous
- Use the AskUserQuestion tool for interactive decisions

## Available Tools

For planning:
- mcp__planner__plan_create: Create an implementation plan
- mcp__planner__plan_get: View the current plan
- mcp__planner__plan_convert_to_tasks: Convert approved plan to tasks

For task management:
- mcp__planner__task_create: Create a task manually
- mcp__planner__task_update: Update task status/dependencies
- mcp__planner__task_list: List all tasks with status
- mcp__planner__task_get: Get details of a specific task

For code exploration:
- Read: Read file contents
- Glob: Find files by pattern
- Grep: Search file contents
- Bash: Run shell commands

For implementation:
- Write: Create new files
- Edit: Modify existing files
- Bash: Run commands (tests, builds, etc.)

For user interaction:
- AskUserQuestion: Ask the user clarifying questions with options
"""

    async def run_planning_only(self, feature_request: str) -> None:
        """
        Run exploration and planning phases only (no implementation).

        Used by the Dapr workflow service to create a plan without
        user interaction or implementation.

        Args:
            feature_request: The user's feature request/prompt
        """
        options = ClaudeAgentOptions(
            system_prompt=self._get_planning_system_prompt(),
            mcp_servers={"planner": self.mcp_server},
            allowed_tools=[
                # Planning tools
                "mcp__planner__plan_create",
                "mcp__planner__plan_get",
                # Code exploration tools
                "Read",
                "Glob",
                "Grep",
                "Bash",
            ],
            permission_mode="acceptEdits",
            cwd=str(self.cwd),
        )

        print(f"[Planner Agent] Starting planning-only session")
        print(f"[Planner Agent] CWD: {self.cwd}")

        async with ClaudeSDKClient(options=options) as client:
            planning_prompt = f"""I need you to create an implementation plan for:

{feature_request}

IMPORTANT INSTRUCTIONS:
1. Do a QUICK exploration of the codebase (limit to 5-6 tool calls max for exploration)
2. Focus on finding the most relevant files for this feature
3. Then IMMEDIATELY create a detailed implementation plan using the plan_create tool

You MUST call plan_create within your first 15 tool uses. Do not explore exhaustively - gather enough context to make a good plan, then create it.

When you call plan_create, include:
- title: A descriptive title for the feature
- summary: What the feature does
- context: Key findings from your exploration
- steps_json: JSON array of implementation steps
- critical_files_json: JSON array of files to modify
- considerations_json: JSON array of things to consider

Example steps_json format: [{{"title": "Step 1", "description": "...", "files_affected": ["file1.ts"], "complexity": "low"}}]"""

            await client.query(planning_prompt)

            # Process until plan is created or we run out of turns
            max_turns = 30
            turn = 0
            plan_create_called = False

            async for message in client.receive_response():
                turn += 1
                if turn > max_turns:
                    print(f"[Planner Agent] Max turns reached without plan creation")
                    break

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"[Planner Agent] {block.text[:200]}...")
                        elif isinstance(block, ToolUseBlock):
                            print(f"[Planner Agent] Using tool: {block.name}")
                            if block.name == "mcp__planner__plan_create":
                                plan_create_called = True
                                print(f"[Planner Agent] Plan create tool called")

                # Check if plan was created after tool execution
                if plan_create_called and self.plan_manager.current_plan:
                    print(f"[Planner Agent] Plan created successfully: {self.plan_manager.current_plan.id}")
                    return  # Plan created, we're done

        print(f"[Planner Agent] Planning session completed")

    async def run_execution_only(
        self,
        plan_id: str,
        workflow_id: str,
    ) -> dict[str, Any]:
        """
        Execute an approved plan (implementation only, no planning).

        Used by the Dapr workflow service to execute a plan after user approval.
        Streams progress events to Dapr pub/sub for real-time UI updates.

        Args:
            plan_id: The ID of the approved plan to execute
            workflow_id: The Dapr workflow instance ID for streaming

        Returns:
            dict with success, tasks_completed, tasks_total, files_changed
        """
        print(f"[Planner Agent] Starting execution-only session")
        print(f"[Planner Agent] Plan ID: {plan_id}, Workflow ID: {workflow_id}")

        # Load the plan if not already loaded
        if not self.plan_manager.current_plan or self.plan_manager.current_plan.id != plan_id:
            loaded = self.plan_manager.load_plan(plan_id)
            if not loaded:
                error = f"Plan {plan_id} not found"
                await stream_execution_failed(workflow_id, error, 0)
                return {
                    "success": False,
                    "error": error,
                    "tasks_completed": 0,
                    "tasks_total": 0,
                    "files_changed": [],
                }

        plan = self.plan_manager.current_plan

        # Mark plan as approved if not already
        if plan.status != "approved":
            self.plan_manager.approve_plan()
            self.plan_manager.save_plan()

        # Convert plan to tasks
        tasks = self.plan_manager.convert_plan_to_tasks(self.task_manager)
        self.task_manager.save()
        self.plan_manager.save_plan()

        total_tasks = len(tasks)
        print(f"[Planner Agent] Created {total_tasks} tasks from plan")

        # Stream execution started
        await stream_execution_started(workflow_id, plan_id, total_tasks)

        # Track files changed
        files_changed: list[str] = []
        tasks_completed = 0

        # Set up execution options
        options = ClaudeAgentOptions(
            system_prompt=self._get_execution_system_prompt(),
            mcp_servers={"planner": self.mcp_server},
            allowed_tools=[
                # Task tools
                "mcp__planner__task_update",
                "mcp__planner__task_list",
                "mcp__planner__task_get",
                # Code exploration tools
                "Read",
                "Glob",
                "Grep",
                "Bash",
                # Code modification tools
                "Write",
                "Edit",
            ],
            permission_mode="acceptEdits",
            cwd=str(self.cwd),
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                # Execute each task
                for task in tasks:
                    # Skip already completed tasks
                    if task.status == TaskStatus.COMPLETED:
                        tasks_completed += 1
                        continue

                    # Mark task as in progress
                    self.task_manager.update_task(task.id, status=TaskStatus.IN_PROGRESS)
                    self.task_manager.save()
                    await stream_task_progress(workflow_id, task.id, "started", task.subject)

                    # Create prompt for this task
                    task_prompt = f"""Execute the following task:

Task ID: {task.id}
Subject: {task.subject}
Description: {task.description}

INSTRUCTIONS:
1. Implement exactly what the task describes
2. Use the Write tool to create new files
3. Use the Edit tool to modify existing files
4. Use Bash to run any necessary commands (build, format, etc.)
5. When the task is complete, use task_update to mark it as completed

Focus only on this task. Do not modify any other parts of the codebase."""

                    await client.query(task_prompt)

                    # Process responses and track changes
                    current_task_files: list[str] = []
                    task_failed = False
                    error_message = None

                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, ToolUseBlock):
                                    tool_name = block.name
                                    tool_input = block.input if hasattr(block, 'input') else {}

                                    # Stream tool call
                                    await stream_tool_call(
                                        workflow_id,
                                        tool_name,
                                        tool_input if isinstance(tool_input, dict) else {},
                                        task.id,
                                    )

                                    # Track file changes
                                    if tool_name in ("Write", "Edit"):
                                        file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
                                        if file_path:
                                            operation = "create" if tool_name == "Write" else "modify"
                                            if file_path not in current_task_files:
                                                current_task_files.append(file_path)
                                            await stream_file_changed(
                                                workflow_id,
                                                file_path,
                                                operation,
                                                task.id,
                                            )

                                    print(f"[Planner Agent] Tool: {tool_name}")

                        elif isinstance(message, ResultMessage):
                            result_text = str(message.result) if message.result else ""
                            if message.is_error:
                                error_message = result_text
                                task_failed = True
                            await stream_tool_result(
                                workflow_id,
                                "execution",
                                result_text,
                                message.is_error,
                                task.id,
                            )

                    # Update task status
                    if task_failed:
                        # Don't mark as failed, leave in progress for potential retry
                        await stream_task_progress(
                            workflow_id, task.id, "failed", task.subject, error_message
                        )
                    else:
                        self.task_manager.update_task(task.id, status=TaskStatus.COMPLETED)
                        self.task_manager.save()
                        tasks_completed += 1
                        files_changed.extend(current_task_files)
                        await stream_task_progress(workflow_id, task.id, "completed", task.subject)

                    print(f"[Planner Agent] Task {task.id} {'completed' if not task_failed else 'failed'}")

            # Update plan status
            self.plan_manager.current_plan.status = "completed"
            self.plan_manager.save_plan()

            # Stream execution completed
            await stream_execution_completed(workflow_id, tasks_completed, files_changed)

            print(f"[Planner Agent] Execution completed: {tasks_completed}/{total_tasks} tasks")

            return {
                "success": tasks_completed == total_tasks,
                "tasks_completed": tasks_completed,
                "tasks_total": total_tasks,
                "files_changed": list(set(files_changed)),  # Deduplicate
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[Planner Agent] Execution failed: {error_msg}")
            await stream_execution_failed(workflow_id, error_msg, tasks_completed)

            return {
                "success": False,
                "error": error_msg,
                "tasks_completed": tasks_completed,
                "tasks_total": total_tasks,
                "files_changed": list(set(files_changed)),
            }

    def _get_execution_system_prompt(self) -> str:
        """Get the system prompt for execution mode."""
        return """You are a software implementation agent that executes pre-planned tasks.

## Your Role

You are given specific tasks from an approved implementation plan. Your job is to:
1. Read and understand the task requirements
2. Implement exactly what the task describes
3. Make minimal, focused changes
4. Mark the task as completed when done

## Guidelines

- Focus ONLY on the current task - do not make additional changes
- Use Read, Glob, and Grep to understand existing code before modifying
- Use Write to create new files, Edit to modify existing files
- Use Bash for running commands (npm, build tools, formatters, etc.)
- Test your changes if possible (run build, lint, type-check)
- Mark the task as completed using task_update when done

## Code Quality

- Follow existing code patterns and style
- Add appropriate error handling
- Keep changes minimal and focused
- Don't refactor unrelated code

## Available Tools

For task management:
- mcp__planner__task_update: Update task status (mark completed when done)
- mcp__planner__task_list: List all tasks
- mcp__planner__task_get: Get task details

For code exploration:
- Read: Read file contents
- Glob: Find files by pattern
- Grep: Search file contents
- Bash: Run shell commands

For implementation:
- Write: Create new files
- Edit: Modify existing files
- Bash: Run commands (tests, builds, etc.)
"""

    async def run_planning_session(self, feature_request: str) -> None:
        """
        Run a complete planning session for a feature request.

        Args:
            feature_request: The user's feature request/prompt
        """
        options = ClaudeAgentOptions(
            system_prompt=self._get_planning_system_prompt(),
            mcp_servers={"planner": self.mcp_server},
            allowed_tools=[
                # Planning tools
                "mcp__planner__plan_create",
                "mcp__planner__plan_get",
                "mcp__planner__plan_convert_to_tasks",
                # Task tools
                "mcp__planner__task_create",
                "mcp__planner__task_update",
                "mcp__planner__task_list",
                "mcp__planner__task_get",
                # Code exploration tools
                "Read",
                "Glob",
                "Grep",
                "Bash",
                # Code modification tools
                "Write",
                "Edit",
                # User interaction
                "AskUserQuestion",
            ],
            permission_mode="acceptEdits",
            cwd=str(self.cwd),
        )

        print(f"\n{'='*60}")
        print("PLANNER AGENT - Planning Session")
        print(f"{'='*60}")
        print(f"\nWorking directory: {self.cwd}")
        print(f"Plans directory: {self.plans_dir}")
        print(f"\nFeature Request:\n{feature_request}")
        print(f"\n{'='*60}\n")

        async with ClaudeSDKClient(options=options) as client:
            self.client = client

            # Initial planning prompt
            planning_prompt = f"""I need you to help me implement the following feature:

{feature_request}

Please follow your workflow:
1. First, explore the codebase to understand its structure and relevant files
2. Then create a detailed implementation plan
3. Present the plan for my approval
4. After I approve, convert it to tasks

Start by exploring the codebase."""

            await client.query(planning_prompt)

            # Main conversation loop
            while True:
                plan_approved = False
                implementation_started = False

                # Process responses
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"\nAssistant: {block.text}")
                            elif isinstance(block, ToolUseBlock):
                                print(f"\n[Using tool: {block.name}]")

                                # Check if plan was approved (user would have said yes)
                                if block.name == "mcp__planner__plan_convert_to_tasks":
                                    plan_approved = True
                                    implementation_started = True

                    elif isinstance(message, ResultMessage):
                        if message.is_error:
                            print(f"\n[Error: {message.result}]")
                        else:
                            print(f"\n[Session completed]")

                        # Check if we should continue
                        if message.result and "approve" in message.result.lower():
                            continue

                # If implementation started and all tasks completed, we're done
                if implementation_started:
                    stats = self.task_manager.get_task_stats()
                    if stats["completed"] == stats["total"] and stats["total"] > 0:
                        print(f"\n{'='*60}")
                        print("All tasks completed!")
                        print(f"{'='*60}")
                        break

                # Get user input for continuation
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nEnding session.")
                    break

                # Check for plan approval
                if user_input.lower() in ["approve", "yes", "approved", "lgtm", "looks good"]:
                    if self.plan_manager.current_plan and self.plan_manager.current_plan.status == "draft":
                        self.plan_manager.approve_plan()
                        self.plan_manager.save_plan()
                        user_input = "The plan is approved. Please convert it to tasks and begin implementation."

                # Send follow-up
                await client.query(user_input)

    async def run_interactive(self) -> None:
        """Run the planner agent in interactive mode."""
        print(f"\n{'='*60}")
        print("PLANNER AGENT - Interactive Mode")
        print(f"{'='*60}")
        print("\nDescribe the feature you want to implement.")
        print("The agent will explore your codebase and create an implementation plan.")
        print("\nType 'quit' to exit.\n")

        feature_request = input("Feature Request: ").strip()
        if not feature_request or feature_request.lower() in ["quit", "exit", "q"]:
            return

        await self.run_planning_session(feature_request)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main(
    cwd: str | None = None,
    plans_dir: str | None = None,
    prompt: str | None = None,
) -> None:
    """Main entry point for the planner agent."""
    agent = PlannerAgent(
        cwd=cwd,
        plans_dir=plans_dir,
    )

    if prompt:
        await agent.run_planning_session(prompt)
    else:
        await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
