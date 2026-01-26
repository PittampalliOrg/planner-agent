"""
Planner Agent - Claude Code Native Tools

This agent uses the Claude Agent SDK with ALL native Claude Code tools
and full permissions (equivalent to --dangerously-skip-permissions).

Uses native Claude Code tools:
- Read, Write, Edit (file operations)
- Glob, Grep (file search)
- Bash (command execution)
- Task (agent spawning)
- And all other native Claude Code tools
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

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


# =============================================================================
# Planner Agent Class - Native Claude Code Tools Only
# =============================================================================

class PlannerAgent:
    """
    Planner Agent using native Claude Code tools.

    Uses ALL native Claude Code tools with bypassPermissions mode
    (equivalent to --dangerously-skip-permissions flag).
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
            plans_dir: Directory to store plans
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.plans_dir = Path(plans_dir) if plans_dir else self.cwd / "plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)

        # Initialize plan manager for storing plans
        self.plan_manager = PlanManager(self.plans_dir)

        self.client: ClaudeSDKClient | None = None

    def _get_planning_system_prompt(self) -> str:
        """Get the system prompt for planning mode."""
        return """You are a software planning agent that helps users implement new features.

## Your Workflow

### Phase 1: Exploration
When given a feature request, first explore the codebase using:
- Read: Read file contents
- Glob: Find files by pattern
- Grep: Search file contents
- Bash: Run shell commands

### Phase 2: Planning
Create a detailed implementation plan including:
- Summary of what will be implemented
- Ordered implementation steps
- Critical files to modify
- Considerations and trade-offs

### Phase 3: Implementation
Execute the plan using:
- Write: Create new files
- Edit: Modify existing files
- Bash: Run builds, tests, formatters

## Guidelines
- Always explore first before making changes
- Make minimal, focused changes
- Follow existing code patterns
- Test changes when possible
- Handle errors gracefully

You have access to ALL Claude Code tools with full permissions."""

    def _get_execution_system_prompt(self) -> str:
        """Get the system prompt for execution mode."""
        return """You are a software implementation agent that executes implementation plans.

## Your Role
1. Read and understand the plan/task requirements
2. Implement exactly what is described
3. Make minimal, focused changes
4. Test changes when possible

## Available Tools
You have access to ALL Claude Code native tools:
- Read, Write, Edit (file operations)
- Glob, Grep (file search)
- Bash (command execution)
- Task (spawn sub-agents)
- And all other native tools

## Guidelines
- Focus on the current task
- Follow existing code patterns
- Add appropriate error handling
- Don't refactor unrelated code
- Test your changes if possible

You have full permissions (bypassPermissions mode)."""

    def _get_agent_options(self, system_prompt: str) -> ClaudeAgentOptions:
        """Get agent options with all native tools and full permissions."""
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            # Use claude_code preset for ALL native tools
            tools={"type": "preset", "preset": "claude_code"},
            # Full permissions - equivalent to --dangerously-skip-permissions
            permission_mode="bypassPermissions",
            cwd=str(self.cwd),
        )

    async def run_planning_only(self, feature_request: str) -> None:
        """
        Run exploration and planning phases only.

        Args:
            feature_request: The user's feature request/prompt
        """
        options = self._get_agent_options(self._get_planning_system_prompt())

        print(f"[Planner Agent] Starting planning-only session")
        print(f"[Planner Agent] CWD: {self.cwd}")
        print(f"[Planner Agent] Using ALL native Claude Code tools")
        print(f"[Planner Agent] Permission mode: bypassPermissions")

        async with ClaudeSDKClient(options=options) as client:
            planning_prompt = f"""I need you to create an implementation plan for:

{feature_request}

INSTRUCTIONS:
1. Quickly explore the codebase to understand its structure (5-6 tool calls max)
2. Create a detailed implementation plan
3. Save the plan to a file at {self.plans_dir}/plan.md

Your plan should include:
- Title and summary
- Implementation steps with descriptions
- Critical files to modify
- Considerations and trade-offs

Start by exploring the codebase."""

            await client.query(planning_prompt)

            # Process responses
            max_turns = 30
            turn = 0
            plan_created = False

            async for message in client.receive_response():
                turn += 1
                if turn > max_turns:
                    print(f"[Planner Agent] Max turns reached")
                    break

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Check for plan creation indicators
                            text = block.text[:200]
                            print(f"[Planner Agent] {text}...")
                            if "plan" in block.text.lower() and ("created" in block.text.lower() or "saved" in block.text.lower()):
                                plan_created = True
                        elif isinstance(block, ToolUseBlock):
                            print(f"[Planner Agent] Using tool: {block.name}")
                            # Check for Write tool creating plan file
                            if block.name == "Write":
                                tool_input = block.input if hasattr(block, 'input') else {}
                                file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
                                if "plan" in file_path.lower():
                                    plan_created = True
                                    print(f"[Planner Agent] Plan created at: {file_path}")

            # Try to load the plan from file
            plan_file = self.plans_dir / "plan.md"
            if plan_file.exists():
                print(f"[Planner Agent] Plan file found: {plan_file}")
                # Create a simple plan object for compatibility
                plan_content = plan_file.read_text()
                self.plan_manager.create_plan(
                    title=feature_request[:50],
                    summary=feature_request,
                    context=plan_content,
                    steps=[{"title": "Implementation", "description": plan_content}],
                )
                self.plan_manager.save_plan()

        print(f"[Planner Agent] Planning session completed")

    async def run_execution_only(
        self,
        plan_id: str,
        workflow_id: str,
    ) -> dict[str, Any]:
        """
        Execute an approved plan.

        Args:
            plan_id: The ID of the approved plan to execute
            workflow_id: The Dapr workflow instance ID for streaming

        Returns:
            dict with success, tasks_completed, tasks_total, files_changed
        """
        print(f"[Planner Agent] Starting execution session")
        print(f"[Planner Agent] Plan ID: {plan_id}, Workflow ID: {workflow_id}")

        # Load plan
        plan = self.plan_manager.load_plan(plan_id)
        if not plan:
            error = f"Plan {plan_id} not found"
            await stream_execution_failed(workflow_id, error, 0)
            return {
                "success": False,
                "error": error,
                "tasks_completed": 0,
                "tasks_total": 0,
                "files_changed": [],
            }

        # Mark plan as approved
        self.plan_manager.approve_plan()
        self.plan_manager.save_plan()

        total_steps = len(plan.steps)
        print(f"[Planner Agent] Executing plan with {total_steps} steps")

        # Stream execution started
        await stream_execution_started(workflow_id, plan_id, total_steps)

        # Track progress
        files_changed: list[str] = []
        steps_completed = 0

        options = self._get_agent_options(self._get_execution_system_prompt())

        try:
            async with ClaudeSDKClient(options=options) as client:
                # Execute the plan
                execution_prompt = f"""Execute the following implementation plan:

Title: {plan.title}
Summary: {plan.summary}

Context:
{plan.context}

Steps to implement:
{json.dumps([{"title": s.title, "description": s.description} for s in plan.steps], indent=2)}

INSTRUCTIONS:
1. Implement each step in order
2. Use Write to create new files
3. Use Edit to modify existing files
4. Use Bash to run any necessary commands
5. Report progress as you complete each step

Begin implementation."""

                await client.query(execution_prompt)

                # Process responses
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
                                    plan_id,
                                )

                                # Track file changes
                                if tool_name in ("Write", "Edit"):
                                    file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
                                    if file_path and file_path not in files_changed:
                                        files_changed.append(file_path)
                                        operation = "create" if tool_name == "Write" else "modify"
                                        await stream_file_changed(workflow_id, file_path, operation, plan_id)

                                print(f"[Planner Agent] Tool: {tool_name}")

                            elif isinstance(block, TextBlock):
                                # Check for step completion indicators
                                if "completed" in block.text.lower() or "done" in block.text.lower():
                                    steps_completed += 1
                                    await stream_task_progress(
                                        workflow_id,
                                        f"step_{steps_completed}",
                                        "completed",
                                        f"Step {steps_completed}"
                                    )

                    elif isinstance(message, ResultMessage):
                        result_text = str(message.result) if message.result else ""
                        await stream_tool_result(
                            workflow_id,
                            "execution",
                            result_text,
                            message.is_error,
                            plan_id,
                        )

            # Update plan status
            self.plan_manager.current_plan.status = "completed"
            self.plan_manager.save_plan()

            # Stream execution completed
            await stream_execution_completed(workflow_id, steps_completed, files_changed)

            print(f"[Planner Agent] Execution completed: {steps_completed} steps, {len(files_changed)} files changed")

            return {
                "success": True,
                "tasks_completed": steps_completed,
                "tasks_total": total_steps,
                "files_changed": files_changed,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[Planner Agent] Execution failed: {error_msg}")
            await stream_execution_failed(workflow_id, error_msg, steps_completed)

            return {
                "success": False,
                "error": error_msg,
                "tasks_completed": steps_completed,
                "tasks_total": total_steps,
                "files_changed": files_changed,
            }

    async def run_planning_session(self, feature_request: str) -> None:
        """
        Run a complete planning session for a feature request.

        Args:
            feature_request: The user's feature request/prompt
        """
        options = self._get_agent_options(self._get_planning_system_prompt())

        print(f"\n{'='*60}")
        print("PLANNER AGENT - Native Claude Code Tools")
        print(f"{'='*60}")
        print(f"\nWorking directory: {self.cwd}")
        print(f"Permission mode: bypassPermissions (full access)")
        print(f"\nFeature Request:\n{feature_request}")
        print(f"\n{'='*60}\n")

        async with ClaudeSDKClient(options=options) as client:
            self.client = client

            planning_prompt = f"""I need you to help me implement the following feature:

{feature_request}

Please:
1. Explore the codebase to understand its structure
2. Create a detailed implementation plan
3. Then implement the changes

You have full access to all Claude Code tools."""

            await client.query(planning_prompt)

            # Main conversation loop
            while True:
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"\nAssistant: {block.text}")
                            elif isinstance(block, ToolUseBlock):
                                print(f"\n[Using tool: {block.name}]")

                    elif isinstance(message, ResultMessage):
                        if message.is_error:
                            print(f"\n[Error: {message.result}]")
                        else:
                            print(f"\n[Result received]")

                # Get user input
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nEnding session.")
                    break

                await client.query(user_input)

    async def run_interactive(self) -> None:
        """Run the planner agent in interactive mode."""
        print(f"\n{'='*60}")
        print("PLANNER AGENT - Interactive Mode")
        print(f"{'='*60}")
        print("\nUsing ALL native Claude Code tools with full permissions.")
        print("Describe the feature you want to implement.")
        print("\nType 'quit' to exit.\n")

        feature_request = input("Feature Request: ").strip()
        if not feature_request or feature_request.lower() in ["quit", "exit", "q"]:
            return

        await self.run_planning_session(feature_request)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point for the planner agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Planner Agent - Native Claude Code Tools"
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (git repository) to work in",
    )
    parser.add_argument(
        "--plans-dir",
        type=str,
        default=None,
        help="Directory to store plans",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        default=None,
        help="Feature request (if not provided, runs interactively)",
    )

    args = parser.parse_args()

    agent = PlannerAgent(
        cwd=args.cwd,
        plans_dir=args.plans_dir,
    )

    if args.prompt:
        await agent.run_planning_session(args.prompt)
    else:
        await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
