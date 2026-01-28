"""
Native Planner Agent - Claude CLI Integration

This agent uses the Claude CLI directly in non-interactive mode with
streaming JSON output. This provides access to the latest Claude Code
tools without SDK version dependencies.

Key CLI flags:
- `-p` / `--print`: Non-interactive mode
- `--permission-mode plan`: Plan mode (read-only with task tools)
- `--permission-mode bypassPermissions`: Execution mode (full access)
- `--output-format stream-json --verbose`: Streaming JSON output
- `--resume <session_id>`: Resume a previous session
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel

from task_persistence import TaskStore

from streaming import (
    stream_execution_started,
    stream_execution_completed,
    stream_execution_failed,
    stream_tool_call,
    stream_tool_result,
    stream_file_changed,
    stream_llm_chunk,
    stream_native_task_created,
    stream_native_task_updated,
    stream_clarification_requested,
    stream_plan_ready,
)


# =============================================================================
# Planning State Models
# =============================================================================

class PlanningState(str, Enum):
    """States the planning phase can be in."""
    EXPLORING = "exploring"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PLAN_READY = "plan_ready"


class ClarificationRequest(BaseModel):
    """Represents a request for user clarification."""
    question_id: str
    questions: list[dict]  # AskUserQuestion format
    timestamp: datetime


class PlanningResult(BaseModel):
    """Result from a planning phase run."""
    success: bool
    state: PlanningState
    plan_id: str | None = None
    session_id: str | None = None  # CLI session ID for resumption
    clarification_request: ClarificationRequest | None = None
    plan_content: str | None = None
    critical_files: list[str] = []
    tasks_created: int = 0
    task_subjects: list[str] = []
    tasks_dir: str | None = None
    files_explored: list[str] = []
    error: str | None = None


class CLIPlannerAgent:
    """
    Planner Agent using Claude CLI directly.

    Uses the Claude CLI in non-interactive mode with streaming JSON output.
    This gives us access to the latest Claude Code tools without SDK dependencies.

    Requires: CLAUDE_CODE_ENABLE_TASKS=true for native task tools.

    Available tools in plan mode:
    - TaskCreate: Create tasks with IDs and metadata
    - TaskUpdate: Update tasks, set dependencies (addBlockedBy/addBlocks)
    - TaskList: List all tasks with status
    - TaskGet: Get task details
    - Read, Glob, Grep: Explore codebase
    - AskUserQuestion: Request clarification
    - ExitPlanMode: Signal plan completion
    """

    def __init__(
        self,
        cwd: str | Path,
        workflow_id: str | None = None,
        plan_id: str | None = None,
    ):
        self.cwd = Path(cwd)
        self.workflow_id = workflow_id
        self.plan_id = plan_id

        # Passive task store for recording native tool events
        self.task_store = TaskStore(
            base_path=self.cwd / "workspace" / "tasks",
            workflow_id=workflow_id,
            plan_id=plan_id,
        )

    def _build_cli_args(
        self,
        prompt: str,
        mode: str = "plan",
        session_id: str | None = None,
    ) -> list[str]:
        """Build CLI arguments for claude command.

        Note: Working directory is set via subprocess cwd parameter,
        not via CLI flag (Claude CLI uses current working directory).
        """
        args = [
            "claude",
            "-p",  # Non-interactive print mode
            "--permission-mode", mode,
            "--output-format", "stream-json",
            "--verbose",
        ]

        if session_id:
            args.extend(["--resume", session_id])

        args.append(prompt)
        return args

    async def _run_cli_streaming(
        self,
        args: list[str],
    ) -> AsyncIterator[dict]:
        """Run CLI and yield parsed JSON messages."""
        print(f"[CLIPlannerAgent] Running: {' '.join(args[:6])}...")

        # Enable native task tools (TaskCreate, TaskUpdate, TaskList, TaskGet)
        import os
        env = os.environ.copy()
        env["CLAUDE_CODE_ENABLE_TASKS"] = "true"

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd),
            env=env,
        )

        async for line in process.stdout:
            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                yield msg
            except json.JSONDecodeError as e:
                print(f"[CLIPlannerAgent] JSON parse error: {e}")
                continue

        # Wait for process to complete
        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            print(f"[CLIPlannerAgent] CLI error (code {process.returncode}): {stderr.decode()}")

    async def _process_tool_use(
        self,
        tool_name: str,
        tool_input: dict,
        tool_call_id: str | None,
        tasks_created: list[str],
        files_explored: list[str],
    ) -> dict | None:
        """
        Process a tool use block.

        Returns clarification request if AskUserQuestion was called.
        Records TaskCreate/TaskUpdate calls to task store.
        """
        print(f"[CLIPlannerAgent] Tool: {tool_name} (id: {tool_call_id})")

        # Stream tool call
        if self.workflow_id:
            await stream_tool_call(
                self.workflow_id,
                tool_name,
                tool_input,
                self.plan_id,
                tool_call_id,
            )

        # Record TaskCreate calls (native task tool with IDs and dependencies)
        if tool_name == "TaskCreate":
            self.task_store.record_task_created(tool_input, tool_call_id)

            # Extract task subject for tracking
            subject = tool_input.get("subject", "")
            description = tool_input.get("description", "")
            active_form = tool_input.get("activeForm", "")

            if subject:
                tasks_created.append(subject)
                if self.workflow_id:
                    await stream_native_task_created(
                        self.workflow_id,
                        subject,
                        active_form or description,
                    )

            print(f"[TaskStore] Recorded TaskCreate: {subject}")

        # Record TaskUpdate calls (for dependency tracking)
        if tool_name == "TaskUpdate":
            self.task_store.record_task_updated(tool_input, tool_call_id)
            task_id = tool_input.get("taskId", "")
            status = tool_input.get("status", "")
            blocks = tool_input.get("addBlocks", [])
            blocked_by = tool_input.get("addBlockedBy", [])
            print(f"[TaskStore] Recorded TaskUpdate: task {task_id} status={status} blocks={blocks} blockedBy={blocked_by}")

        # Track exploration tools
        if tool_name in ("Read", "Glob", "Grep"):
            path = (
                tool_input.get("file_path") or
                tool_input.get("path") or
                tool_input.get("pattern", "")
            )
            if path and path not in files_explored:
                files_explored.append(path)

        # Track file changes
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if file_path and self.workflow_id:
                operation = "create" if tool_name == "Write" else "modify"
                await stream_file_changed(self.workflow_id, file_path, operation)

        return None

    def _extract_permission_denial(
        self,
        result: dict,
        denial_type: str,
    ) -> dict | None:
        """Extract a specific permission denial from result."""
        denials = result.get("permission_denials", [])
        for denial in denials:
            if denial.get("tool_name") == denial_type:
                return denial.get("tool_input", {})
        return None

    async def run_planning(
        self,
        feature_request: str,
        session_id: str | None = None,
        clarification_response: str | None = None,
    ) -> PlanningResult:
        """
        Run planning phase using Claude CLI in plan mode.

        Args:
            feature_request: The user's feature request
            session_id: Optional session ID to resume
            clarification_response: Optional response to previous clarification

        Returns:
            PlanningResult with state indicating next action
        """
        print(f"[CLIPlannerAgent] Starting planning session")
        print(f"[CLIPlannerAgent] CWD: {self.cwd}")

        plan_id = self.plan_id or f"plan-{uuid.uuid4().hex[:8]}"

        # Build prompt
        if clarification_response and session_id:
            prompt = f"""Continue planning based on user's clarification.

User's response: {clarification_response}

Now create tasks using TaskCreate and set up dependencies with TaskUpdate, then call ExitPlanMode when ready."""
        else:
            prompt = f"""You are in PLAN MODE. Create a task list for implementing this feature.

Feature Request:
{feature_request}

Plan ID: {plan_id}

CRITICAL: You MUST call TaskCreate to create tasks. This is required.

INSTRUCTIONS:

1. EXPLORE BRIEFLY: Use Glob to find relevant files (1-2 calls max)

2. CREATE TASKS: Call TaskCreate for EACH implementation task
   - REQUIRED: You must create at least 1 task
   - Tasks get assigned IDs (1, 2, 3, etc.)
   - Example: TaskCreate({{"subject": "Create hello.py file", "description": "Create a new Python file", "activeForm": "Creating hello.py"}})

3. SET UP DEPENDENCIES: Use TaskUpdate to establish task order
   - Example: TaskUpdate({{"taskId": "2", "addBlockedBy": ["1"]}}) - task 2 waits for task 1
   - Example: TaskUpdate({{"taskId": "3", "addBlockedBy": ["1", "2"]}}) - task 3 waits for tasks 1 and 2

4. THEN: Call ExitPlanMode to signal the plan is ready for approval

IMPORTANT:
- Do NOT use the Task tool - it spawns subagents
- Do NOT skip TaskCreate - it is mandatory
- Call TaskCreate BEFORE ExitPlanMode
- Use TaskUpdate to set up dependencies between tasks

TOOLS:
- Glob, Read: Explore codebase briefly
- TaskCreate: Create tasks with subject/description/activeForm (REQUIRED)
- TaskUpdate: Set task dependencies with addBlockedBy/addBlocks
- ExitPlanMode: Signal plan complete (only after creating tasks)
"""

        args = self._build_cli_args(prompt, mode="plan", session_id=session_id)

        tasks_created = []
        files_explored = []
        result_session_id = session_id
        final_result = None

        try:
            async for msg in self._run_cli_streaming(args):
                msg_type = msg.get("type")

                # Capture session ID from init
                if msg_type == "system" and msg.get("subtype") == "init":
                    result_session_id = msg.get("session_id")
                    tools = msg.get("tools", [])
                    print(f"[CLIPlannerAgent] Session: {result_session_id}")
                    print(f"[CLIPlannerAgent] Available tools: {tools}")

                # Process assistant messages (tool calls and text)
                elif msg_type == "assistant":
                    message = msg.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        block_type = block.get("type")

                        if block_type == "text":
                            text = block.get("text", "")
                            if text and self.workflow_id:
                                await stream_llm_chunk(self.workflow_id, text, plan_id)
                            if text:
                                preview = text[:100].replace('\n', ' ')
                                print(f"[CLIPlannerAgent] {preview}...")

                        elif block_type == "tool_use":
                            tool_name = block.get("name")
                            tool_input = block.get("input", {})
                            tool_call_id = block.get("id")

                            await self._process_tool_use(
                                tool_name,
                                tool_input,
                                tool_call_id,
                                tasks_created,
                                files_explored,
                            )

                # Process tool results
                elif msg_type == "user":
                    message = msg.get("message", {})
                    content = message.get("content", [])
                    tool_result = msg.get("tool_use_result")

                    for block in content:
                        if block.get("type") == "tool_result":
                            is_error = block.get("is_error", False)
                            result_content = block.get("content", "")
                            tool_use_id = block.get("tool_use_id")

                            if self.workflow_id:
                                await stream_tool_result(
                                    self.workflow_id,
                                    "tool",
                                    str(result_content)[:500],
                                    is_error,
                                    plan_id,
                                    tool_use_id,
                                )

                # Process final result
                elif msg_type == "result":
                    final_result = msg
                    result_session_id = msg.get("session_id", result_session_id)

            # Analyze final result for permission denials
            # Priority: ExitPlanMode > AskUserQuestion
            # If LLM called ExitPlanMode, plan is ready regardless of earlier questions
            if final_result:
                # Check for ExitPlanMode denial FIRST (plan ready takes priority)
                exit_denial = self._extract_permission_denial(final_result, "ExitPlanMode")
                if exit_denial:
                    plan_content = exit_denial.get("plan", "")

                    if self.workflow_id:
                        await stream_plan_ready(
                            self.workflow_id,
                            plan_id,
                            plan_content,
                            [],
                            len(tasks_created),
                        )

                    return PlanningResult(
                        success=True,
                        state=PlanningState.PLAN_READY,
                        plan_id=plan_id,
                        session_id=result_session_id,
                        plan_content=plan_content,
                        tasks_created=len(tasks_created),
                        task_subjects=tasks_created,
                        tasks_dir=str(self.task_store.tasks_dir),
                        files_explored=files_explored,
                    )

                # Check for AskUserQuestion denial (clarification needed)
                # Only if ExitPlanMode was NOT called
                ask_denial = self._extract_permission_denial(final_result, "AskUserQuestion")
                if ask_denial:
                    questions = ask_denial.get("questions", [])
                    question_id = f"q-{uuid.uuid4().hex[:8]}"

                    if self.workflow_id:
                        await stream_clarification_requested(
                            self.workflow_id,
                            questions,
                            question_id,
                            plan_id,
                        )

                    return PlanningResult(
                        success=True,
                        state=PlanningState.AWAITING_CLARIFICATION,
                        plan_id=plan_id,
                        session_id=result_session_id,
                        clarification_request=ClarificationRequest(
                            question_id=question_id,
                            questions=questions,
                            timestamp=datetime.now(),
                        ),
                        tasks_created=len(tasks_created),
                        task_subjects=tasks_created,
                        tasks_dir=str(self.task_store.tasks_dir),
                        files_explored=files_explored,
                    )

            # Default: plan completed without explicit exit
            print(f"[CLIPlannerAgent] Planning completed: {len(tasks_created)} tasks")

            return PlanningResult(
                success=True,
                state=PlanningState.PLAN_READY,
                plan_id=plan_id,
                session_id=result_session_id,
                tasks_created=len(tasks_created),
                task_subjects=tasks_created,
                tasks_dir=str(self.task_store.tasks_dir),
                files_explored=files_explored,
            )

        except Exception as e:
            print(f"[CLIPlannerAgent] Planning failed: {e}")
            return PlanningResult(
                success=False,
                state=PlanningState.EXPLORING,
                plan_id=plan_id,
                session_id=result_session_id,
                error=str(e),
                tasks_created=len(tasks_created),
                tasks_dir=str(self.task_store.tasks_dir),
            )

    async def run_execution(
        self,
        task_subjects: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run execution phase using Claude CLI in bypassPermissions mode.

        Args:
            task_subjects: List of task subjects from planning phase
            session_id: Optional session ID to resume

        Returns:
            dict with success, tasks_completed, files_changed
        """
        print(f"[CLIPlannerAgent] Starting execution session")
        print(f"[CLIPlannerAgent] CWD: {self.cwd}")

        # Build context about what to implement
        if task_subjects and any(t for t in task_subjects):
            tasks_context = "Tasks to implement:\n" + "\n".join(f"- {t}" for t in task_subjects if t)
        else:
            tasks_context = "Implement a hello world function in a new file called hello.py"

        prompt = f"""Execute the implementation now. You MUST use the Write tool to create files.

{tasks_context}

Instructions:
1. Implement each task in order
2. Use Write tool to create new files
3. Use Edit tool to modify existing files
4. Run tests or verification commands as needed

Start implementing now.
"""

        args = self._build_cli_args(prompt, mode="bypassPermissions", session_id=session_id)

        files_changed = []
        tasks_completed = 0

        try:
            if self.workflow_id:
                await stream_execution_started(self.workflow_id, "cli-execution", 0)

            async for msg in self._run_cli_streaming(args):
                msg_type = msg.get("type")

                if msg_type == "system" and msg.get("subtype") == "init":
                    session_id = msg.get("session_id")
                    print(f"[CLIPlannerAgent] Execution session: {session_id}")

                elif msg_type == "assistant":
                    message = msg.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        block_type = block.get("type")

                        if block_type == "text":
                            text = block.get("text", "")
                            if text and self.workflow_id:
                                await stream_llm_chunk(self.workflow_id, text, None)

                        elif block_type == "tool_use":
                            tool_name = block.get("name")
                            tool_input = block.get("input", {})
                            tool_call_id = block.get("id")

                            print(f"[CLIPlannerAgent] Tool: {tool_name}")

                            if self.workflow_id:
                                await stream_tool_call(
                                    self.workflow_id,
                                    tool_name,
                                    tool_input,
                                    None,
                                    tool_call_id,
                                )

                            # Track file changes
                            if tool_name in ("Write", "Edit"):
                                file_path = tool_input.get("file_path", "")
                                if file_path and file_path not in files_changed:
                                    files_changed.append(file_path)
                                    operation = "create" if tool_name == "Write" else "modify"
                                    if self.workflow_id:
                                        await stream_file_changed(
                                            self.workflow_id,
                                            file_path,
                                            operation,
                                        )

                elif msg_type == "user":
                    message = msg.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        if block.get("type") == "tool_result":
                            is_error = block.get("is_error", False)
                            result_content = block.get("content", "")
                            tool_use_id = block.get("tool_use_id")

                            if self.workflow_id:
                                await stream_tool_result(
                                    self.workflow_id,
                                    "tool",
                                    str(result_content)[:500],
                                    is_error,
                                    None,
                                    tool_use_id,
                                )

                elif msg_type == "result":
                    is_error = msg.get("is_error", False)
                    if not is_error:
                        tasks_completed = len(task_subjects) if task_subjects else 1

            if self.workflow_id:
                await stream_execution_completed(
                    self.workflow_id,
                    tasks_completed,
                    files_changed,
                )

            print(f"[CLIPlannerAgent] Execution completed: {len(files_changed)} files changed")

            return {
                "success": True,
                "tasks_completed": tasks_completed,
                "files_changed": files_changed,
                "session_id": session_id,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[CLIPlannerAgent] Execution failed: {error_msg}")

            if self.workflow_id:
                await stream_execution_failed(self.workflow_id, error_msg, tasks_completed)

            return {
                "success": False,
                "error": error_msg,
                "tasks_completed": tasks_completed,
                "files_changed": files_changed,
            }

    async def run_interactive(self) -> None:
        """Run the planner agent in interactive mode."""
        print(f"\n{'='*60}")
        print("CLI PLANNER AGENT - Claude CLI Integration")
        print(f"{'='*60}")
        print(f"\nWorking directory: {self.cwd}")
        print("Using Claude CLI with streaming JSON output.")
        print("Type 'quit' to exit.\n")

        feature_request = input("Feature Request: ").strip()
        if not feature_request or feature_request.lower() in ["quit", "exit", "q"]:
            return

        # Run planning with clarification loop
        print("\n--- Planning Phase ---")
        session_id = None
        clarification_response = None
        max_clarifications = 5

        for _ in range(max_clarifications + 1):
            plan_result = await self.run_planning(
                feature_request,
                session_id=session_id,
                clarification_response=clarification_response,
            )
            print(f"\nPlanning result: {plan_result.model_dump_json(indent=2)}")

            if not plan_result.success:
                print("Planning failed, exiting.")
                return

            session_id = plan_result.session_id

            # Handle clarification request
            if plan_result.state == PlanningState.AWAITING_CLARIFICATION:
                if plan_result.clarification_request:
                    print("\n--- Clarification Needed ---")
                    for q in plan_result.clarification_request.questions:
                        print(f"\nQuestion: {q.get('question', 'Unknown')}")
                        if 'options' in q:
                            for i, opt in enumerate(q['options'], 1):
                                print(f"  {i}. {opt.get('label', '')} - {opt.get('description', '')}")
                    clarification_response = input("\nYour response: ").strip()
                    if not clarification_response:
                        print("No response provided, exiting.")
                        return
                    continue

            # Plan is ready
            if plan_result.state == PlanningState.PLAN_READY:
                break

        # Display plan summary
        print("\n--- Plan Summary ---")
        print(f"Plan ID: {plan_result.plan_id}")
        print(f"Session ID: {plan_result.session_id}")
        print(f"Tasks created: {plan_result.tasks_created}")
        if plan_result.task_subjects:
            print("Tasks:")
            for i, task in enumerate(plan_result.task_subjects, 1):
                print(f"  {i}. {task}")

        # Ask for approval
        approve = input("\nApprove plan and execute? (y/n): ").strip().lower()
        if approve != "y":
            print("Plan not approved, exiting.")
            return

        # Run execution
        print("\n--- Execution Phase ---")
        exec_result = await self.run_execution(plan_result.task_subjects)
        print(f"\nExecution result: {json.dumps(exec_result, indent=2)}")


# Aliases for compatibility
NativePlannerAgent = CLIPlannerAgent
PlannerAgent = CLIPlannerAgent


async def main():
    """Main entry point for the planner agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI Planner Agent - Claude CLI Integration"
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (git repository) to work in",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        default=None,
        help="Feature request (if not provided, runs interactively)",
    )

    args = parser.parse_args()

    cwd = args.cwd or Path.cwd()
    agent = CLIPlannerAgent(cwd=cwd)

    if args.prompt:
        plan_result = await agent.run_planning(args.prompt)
        print(f"\nPlanning result: {plan_result.model_dump_json(indent=2)}")

        if plan_result.success and plan_result.state == PlanningState.PLAN_READY:
            print("To execute, call run_execution()")
    else:
        await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
