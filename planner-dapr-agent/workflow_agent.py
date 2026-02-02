"""
Multi-step agent workflow: Planning → Execution → Testing
Minimal implementation using OpenAI Agents SDK best practices.

This module provides a three-phase workflow:
1. Planning Phase - Agent loop that researches, thinks, and writes a plan
2. Execution Phase - Agent loop that executes the plan using tools
3. Testing Phase - Agent loop that verifies the implementation

Key design decisions:
- Structured Output for Loop Control: output_type with Pydantic models signals phase completion
- Minimal Tool Set: Only essential tools, each with clear purpose
- Code-Based Orchestration: Deterministic phase sequencing (not LLM-decided)
"""

import asyncio
import contextvars
import functools
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool

logger = logging.getLogger(__name__)

# Default workspace directory
DEFAULT_CWD = os.getenv("PLANNER_CWD", "/app/workspace")

# Context variable to store current workflow_id for tool event publishing
_current_workflow_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'workflow_agent_workflow_id', default=None
)


def set_workflow_id(workflow_id: str) -> contextvars.Token:
    """Set the current workflow ID for tool event publishing."""
    return _current_workflow_id.set(workflow_id)


def get_workflow_id() -> Optional[str]:
    """Get the current workflow ID."""
    return _current_workflow_id.get()


def reset_workflow_id(token: contextvars.Token) -> None:
    """Reset the workflow ID context."""
    _current_workflow_id.reset(token)


def _publish_tool_event(event_type: str, tool_name: str, data: dict) -> None:
    """Publish a tool event to pub/sub for real-time SSE streaming."""
    workflow_id = get_workflow_id()
    if not workflow_id:
        return

    try:
        from dapr_multi_step_workflow import publish_workflow_event
        publish_workflow_event(
            workflow_id=workflow_id,
            event_type=event_type,
            data=data,
        )
        logger.debug(f"Published {event_type} event for tool {tool_name}")
    except Exception as e:
        logger.debug(f"Could not publish {event_type} event: {e}")


def _safe_serialize(value: Any, max_depth: int = 3) -> Any:
    """Safely serialize a value for logging/publishing."""
    if max_depth <= 0:
        return str(value)[:200]

    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500] if len(value) > 500 else value

    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            return f"<{len(value)} items>"
        return [_safe_serialize(v, max_depth - 1) for v in value]

    if isinstance(value, dict):
        result = {}
        for k, v in list(value.items())[:10]:
            result[str(k)] = _safe_serialize(v, max_depth - 1)
        return result

    return str(value)[:200]


def tracked_tool(func: Callable) -> Any:
    """Decorator that wraps a tool function to publish events.

    Publishes tool_call when the tool starts and tool_result when it completes.
    Falls back gracefully if publishing fails.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        tool_name = func.__name__

        # Publish tool_call event
        _publish_tool_event(
            "tool_call",
            tool_name,
            {
                "toolName": tool_name,
                "toolInput": _safe_serialize(kwargs or args),
                "content": f"Running: {tool_name}",
            }
        )

        try:
            # Execute the tool
            result = await func(*args, **kwargs)

            # Publish tool_result event
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            _publish_tool_event(
                "tool_result",
                tool_name,
                {
                    "toolName": tool_name,
                    "toolOutput": result_str,
                    "status": "completed",
                    "content": f"Result: {tool_name}",
                }
            )

            return result
        except Exception as e:
            # Publish error event
            _publish_tool_event(
                "tool_result",
                tool_name,
                {
                    "toolName": tool_name,
                    "toolOutput": str(e),
                    "status": "failed",
                    "content": f"Error: {tool_name}",
                }
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        tool_name = func.__name__

        # Publish tool_call event
        _publish_tool_event(
            "tool_call",
            tool_name,
            {
                "toolName": tool_name,
                "toolInput": _safe_serialize(kwargs or args),
                "content": f"Running: {tool_name}",
            }
        )

        try:
            # Execute the tool
            result = func(*args, **kwargs)

            # Publish tool_result event
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            _publish_tool_event(
                "tool_result",
                tool_name,
                {
                    "toolName": tool_name,
                    "toolOutput": result_str,
                    "status": "completed",
                    "content": f"Result: {tool_name}",
                }
            )

            return result
        except Exception as e:
            # Publish error event
            _publish_tool_event(
                "tool_result",
                tool_name,
                {
                    "toolName": tool_name,
                    "toolOutput": str(e),
                    "status": "failed",
                    "content": f"Error: {tool_name}",
                }
            )
            raise

    # Handle both async and sync functions
    if asyncio.iscoroutinefunction(func):
        return function_tool(async_wrapper)
    else:
        return function_tool(sync_wrapper)


# ============================================================================
# Structured Output Types (Control Loop Termination)
# ============================================================================

class Task(BaseModel):
    """A single task in the plan - matches existing planner-dapr-agent format."""
    id: str
    subject: str
    description: str
    status: str = "pending"
    blockedBy: List[str] = Field(default_factory=list)  # Tasks that must complete first
    blocks: List[str] = Field(default_factory=list)      # Tasks blocked by this one


class TestCase(BaseModel):
    """A test case to verify task completion."""
    id: str
    task_id: str  # Which task this tests
    description: str
    test_type: str  # "command", "file_exists", "output_contains"
    command: Optional[str] = None  # For command-based tests
    expected: Optional[str] = None  # Expected output or file path


class Plan(BaseModel):
    """Structured plan output - terminates planning loop."""
    summary: str
    tasks: List[Task]
    tests: List[TestCase]  # Verification tests for each task
    reasoning: str


class ExecutionResult(BaseModel):
    """Execution result - terminates execution loop."""
    success: bool
    completed_tasks: List[str]
    output: str
    errors: List[str] = Field(default_factory=list)


class TestResult(BaseModel):
    """Test result - terminates testing loop."""
    passed: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    failures: List[str] = Field(default_factory=list)
    summary: str


# ============================================================================
# Planning Phase Tools
# ============================================================================

@tracked_tool
async def research(query: str) -> str:
    """Search codebase or documentation for information needed to plan.

    Args:
        query: Search term to look for in the codebase

    Returns:
        List of files containing the query or a no-match message
    """
    try:
        result = subprocess.run(
            ["grep", "-r", "-l", query, DEFAULT_CWD],
            capture_output=True, text=True, timeout=30
        )
        files = result.stdout.strip().split("\n")[:10]
        if files[0]:
            # Make paths relative for readability
            rel_files = [os.path.relpath(f, DEFAULT_CWD) for f in files if f]
            return f"Found {len(rel_files)} relevant files: {rel_files}"
        return "No matches found"
    except subprocess.TimeoutExpired:
        return "Search timed out after 30s"
    except Exception as e:
        return f"Search error: {e}"


@tracked_tool
async def think(thought: str) -> str:
    """Record a reasoning step. Use this to think through the problem.

    Args:
        thought: Your reasoning or analysis to record

    Returns:
        Acknowledgment that the thought was recorded
    """
    return f"Noted: {thought}"


@tracked_tool
async def draft_plan(summary: str, tasks_json: str) -> str:
    """Draft or refine the plan. Call multiple times to iterate.

    Args:
        summary: High-level summary of the plan
        tasks_json: JSON string containing list of task objects with id, subject, description, blockedBy

    Returns:
        Status message about the draft
    """
    import json
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in tasks_json: {e}"

    if not isinstance(tasks, list):
        return "Error: tasks_json must be a JSON array"

    # Validate task structure
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            return f"Error: Task {i} is not an object"
        if not task.get("id"):
            return f"Error: Task {i} missing 'id' field"
        if not task.get("subject"):
            return f"Error: Task {i} missing 'subject' field"
        if not task.get("description"):
            return f"Error: Task {i} missing 'description' field"

    return f"Draft plan with {len(tasks)} tasks recorded. Continue refining or finalize by outputting the Plan."


# ============================================================================
# Execution Phase Tools
# ============================================================================

@tracked_tool
async def read_file(file_path: str) -> str:
    """Read contents of a file.

    Args:
        file_path: Path to file (relative to workspace or absolute)

    Returns:
        File contents or error message
    """
    # Handle both relative and absolute paths
    if os.path.isabs(file_path):
        full_path = file_path
    else:
        full_path = os.path.join(DEFAULT_CWD, file_path)

    try:
        with open(full_path, "r") as f:
            content = f.read()
            if len(content) > 10000:
                return content[:10000] + f"\n\n... (truncated, {len(content)} total bytes)"
            return content
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading {file_path}: {e}"


@tracked_tool
async def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates directories if needed.

    Args:
        file_path: Path to file (relative to workspace or absolute)
        content: Content to write

    Returns:
        Success message or error
    """
    # Handle both relative and absolute paths
    if os.path.isabs(file_path):
        full_path = file_path
    else:
        full_path = os.path.join(DEFAULT_CWD, file_path)

    try:
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


@tracked_tool
async def run_command(command: str) -> str:
    """Run a shell command. Use for builds, tests, etc.

    Args:
        command: Shell command to execute

    Returns:
        Command output or error message
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=DEFAULT_CWD
        )
        output = result.stdout + result.stderr
        if not output:
            return f"Command completed with exit code {result.returncode}"
        if len(output) > 5000:
            return output[:5000] + f"\n\n... (truncated, {len(output)} total chars)"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out after 60s"
    except Exception as e:
        return f"Error: {e}"


@tracked_tool
async def mark_task_complete(task_id: str, notes: str = "") -> str:
    """Mark a task as complete. Call after finishing each task.

    Args:
        task_id: ID of the task to mark complete
        notes: Optional notes about what was done

    Returns:
        Confirmation message
    """
    return f"Task {task_id} marked complete. {notes}"


# ============================================================================
# Testing Phase Tools
# ============================================================================

@tracked_tool
async def run_tests(command: str) -> str:
    """Run test command (pytest, npm test, etc.).

    Args:
        command: Test command to execute

    Returns:
        Test output or error message
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=DEFAULT_CWD
        )
        output = result.stdout + result.stderr
        if not output:
            return f"Tests completed with exit code {result.returncode}"
        if len(output) > 8000:
            return output[:8000] + f"\n\n... (truncated, {len(output)} total chars)"
        return output
    except subprocess.TimeoutExpired:
        return "Tests timed out after 120s"
    except Exception as e:
        return f"Error running tests: {e}"


@tracked_tool
async def verify_output(task_id: str, expected: str, actual: str) -> str:
    """Verify a task's output matches expectations.

    Args:
        task_id: ID of the task being verified
        expected: Expected substring in the output
        actual: Actual output to check

    Returns:
        PASS or FAIL message
    """
    matches = expected.strip() in actual.strip()
    if matches:
        return f"Task {task_id}: PASS - Expected '{expected[:100]}' found in output"
    return f"Task {task_id}: FAIL - Expected '{expected[:100]}' not found in output"


@tracked_tool
async def check_file_exists(file_path: str) -> str:
    """Check if a file was created/modified as expected.

    Args:
        file_path: Path to check (relative to workspace or absolute)

    Returns:
        PASS or FAIL message with file info
    """
    # Handle both relative and absolute paths
    if os.path.isabs(file_path):
        full_path = file_path
    else:
        full_path = os.path.join(DEFAULT_CWD, file_path)

    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        return f"PASS: {file_path} exists ({size} bytes)"
    return f"FAIL: {file_path} does not exist"


# ============================================================================
# Agent Definitions
# ============================================================================

def create_planning_agent(model: str = "gpt-5.2-codex") -> Agent:
    """Create the planning agent.

    Args:
        model: OpenAI model to use

    Returns:
        Configured planning Agent
    """
    return Agent(
        name="Planner",
        model=model,
        instructions="""You are a planning agent. Your job is to create a detailed plan WITH test cases.

WORKFLOW:
1. Use `research` to understand the codebase/requirements
2. Use `think` to reason through the approach
3. Use `draft_plan` to iterate on the plan
4. When ready, output the final Plan as structured JSON

OUTPUT REQUIREMENTS:
- tasks: List of implementation tasks with blockedBy/blocks dependencies
- tests: List of test cases to verify EACH task works
  - test_type: "command" (run a command), "file_exists", "output_contains"
  - For each task, create at least one test case

RULES:
- Break work into small, concrete tasks
- Use blockedBy to define which tasks must complete first
- The blocks array will be auto-populated based on blockedBy
- Each task MUST have a corresponding test case
- Keep iterating until the plan is complete and testable""",
        tools=[research, think, draft_plan],
        output_type=Plan,  # Loop terminates when Plan is output
    )


def create_execution_agent(model: str = "gpt-5.2-codex") -> Agent:
    """Create the execution agent.

    Args:
        model: OpenAI model to use

    Returns:
        Configured execution Agent
    """
    return Agent(
        name="Executor",
        model=model,
        instructions="""You are an execution agent. Execute the plan you receive.

WORKFLOW:
1. Review the plan and tasks
2. Execute tasks in dependency order (respect blockedBy)
3. Use tools to complete each task
4. Call `mark_task_complete` after each task
5. Output ExecutionResult when all tasks are done

RULES:
- Follow the plan exactly
- Handle errors gracefully
- Report what was accomplished""",
        tools=[read_file, write_file, run_command, mark_task_complete],
        output_type=ExecutionResult,  # Loop terminates when Result is output
    )


def create_testing_agent(model: str = "gpt-5.2-codex") -> Agent:
    """Create the testing agent.

    Args:
        model: OpenAI model to use

    Returns:
        Configured testing Agent
    """
    return Agent(
        name="Tester",
        model=model,
        instructions="""You are a testing agent. Verify the implementation works.

WORKFLOW:
1. Review the test cases from the plan
2. Run each test using appropriate tools
3. If tests fail, report what went wrong
4. Output TestResult with pass/fail summary

RULES:
- Run ALL test cases from the plan
- Be thorough in verification
- Report specific failures for debugging""",
        tools=[run_tests, verify_output, check_file_exists, read_file, run_command],
        output_type=TestResult,  # Loop terminates when TestResult is output
    )


# ============================================================================
# Workflow Orchestration
# ============================================================================

async def run_workflow(
    prompt: str,
    model: str = "gpt-5.2-codex",
    max_turns: int = 20,
    max_test_retries: int = 3
) -> dict:
    """
    Run the complete planning → execution → testing workflow.

    Args:
        prompt: User's task description
        model: OpenAI model to use
        max_turns: Max iterations per phase (safety limit)
        max_test_retries: Max times to retry execution if tests fail

    Returns:
        dict with plan, execution, and test results
    """
    # Phase 1: Planning
    planning_agent = create_planning_agent(model)
    plan_result = await Runner.run(
        planning_agent,
        input=prompt,
        max_turns=max_turns,
    )
    plan: Plan = plan_result.final_output

    # Auto-populate blocks based on blockedBy
    task_map = {t.id: t for t in plan.tasks}
    for task in plan.tasks:
        for blocked_by_id in task.blockedBy:
            if blocked_by_id in task_map:
                if task.id not in task_map[blocked_by_id].blocks:
                    task_map[blocked_by_id].blocks.append(task.id)

    # Phase 2: Execution
    execution_agent = create_execution_agent(model)
    exec_prompt = f"""Execute this plan:

Summary: {plan.summary}

Tasks:
{chr(10).join(f"- [{t.id}] {t.subject}: {t.description} (blockedBy: {t.blockedBy})" for t in plan.tasks)}

Reasoning: {plan.reasoning}"""

    exec_result = await Runner.run(
        execution_agent,
        input=exec_prompt,
        max_turns=max_turns,
    )
    execution: ExecutionResult = exec_result.final_output

    # Phase 3: Testing (loops until pass or max retries)
    testing_agent = create_testing_agent(model)
    test_prompt = f"""Verify the implementation:

Plan Summary: {plan.summary}

Test Cases:
{chr(10).join(f"- [{tc.id}] {tc.description} (type: {tc.test_type}, command: {tc.command})" for tc in plan.tests)}

Execution Summary: {execution.output}
Completed Tasks: {execution.completed_tasks}"""

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
            break  # All tests passed!

        # Tests failed - could add re-execution logic here
        # For now, just report the failure

    return {
        "plan": plan.model_dump(),
        "execution": execution.model_dump(),
        "testing": test.model_dump(),
        "status": "completed" if test.passed else "failed",
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    import json

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Create a hello world Python script"

    print(f"Running workflow for: {prompt}")
    print("=" * 60)

    result = asyncio.run(run_workflow(prompt))

    print("\n=== PLAN ===")
    print(json.dumps(result["plan"], indent=2))
    print("\n=== EXECUTION ===")
    print(json.dumps(result["execution"], indent=2))
    print("\n=== TESTING ===")
    print(json.dumps(result["testing"], indent=2))
    print("\n=== STATUS ===")
    print(f"Workflow {'PASSED' if result['status'] == 'completed' else 'FAILED'}")
