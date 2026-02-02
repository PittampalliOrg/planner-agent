"""Dapr workflow for multi-step planning → execution → testing.

This module implements a proper Dapr workflow using dapr-ext-workflow so that
the workflow phases (planning, execution, testing) appear as activities in the
ai-chatbot UI workflow graph.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import dapr.ext.workflow as wf
from agents import Runner

logger = logging.getLogger(__name__)

# Initialize workflow runtime
wfr = wf.WorkflowRuntime()


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

    logger.info(f"Planning activity started for task: {task[:100]}...")

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
        return {
            "success": True,
            "plan": plan_data,
        }
    except Exception as e:
        logger.error(f"Planning failed: {e}")
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

    logger.info(f"Execution activity started with {len(plan.get('tasks', []))} tasks")

    async def run_execution():
        execution_agent = create_execution_agent(model)

        tasks = plan.get("tasks", [])
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
        return {
            "success": True,
            "execution": execution_data,
        }
    except Exception as e:
        logger.error(f"Execution failed: {e}")
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

    tests = plan.get("tests", [])
    logger.info(f"Testing activity started with {len(tests)} test cases")

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
        return {
            "success": True,
            "testing": test_data,
        }
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Workflow: Multi-Step Planning → Execution → Testing
# ============================================================================

@wfr.workflow(name="multi_step_workflow")
def multi_step_workflow(ctx: wf.DaprWorkflowContext, input_data: Dict[str, Any]):
    """Dapr workflow for multi-step planning → execution → testing.

    This workflow:
    1. Planning Phase - Creates detailed plan with tasks and test cases
    2. Execution Phase - Executes the plan
    3. Testing Phase - Verifies the implementation

    Each phase is a separate activity that appears in the UI workflow graph.
    """
    workflow_id = ctx.instance_id
    task = input_data.get("task", "")
    model = input_data.get("model", "gpt-5.2-codex")
    max_turns = input_data.get("max_turns", 20)
    max_test_retries = input_data.get("max_test_retries", 3)

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

    # --- Phase 2: Execution ---
    ctx.set_custom_status(json.dumps({
        "phase": "execution",
        "progress": 40,
        "message": f"Executing {len(plan.get('tasks', []))} tasks...",
    }))

    execution_result = yield ctx.call_activity(
        execution_activity,
        input={
            "plan": plan,
            "model": model,
            "max_turns": max_turns,
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

    # --- Phase 3: Testing ---
    ctx.set_custom_status(json.dumps({
        "phase": "testing",
        "progress": 70,
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
