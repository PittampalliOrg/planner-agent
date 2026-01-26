"""
Durable Planner Agent - Dapr DurableAgent wrapper for fault-tolerant planning

Combines Dapr Agents' durable execution capabilities with Claude Agent SDK's
reasoning and tool-use. This provides:
- Fault-tolerant execution that survives crashes/restarts
- Persistent state management via Dapr state stores
- Automatic retry and recovery mechanisms
- Workflow-backed agent orchestration

Architecture:
- DurableAgent: Manages durable state, orchestration, and fault tolerance
- Claude SDK: Handles LLM reasoning and tool execution inside activities
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Import Dapr Agents for durable agent capabilities
try:
    from dapr_agents import Agent, DurableAgent, tool as dapr_tool
    from dapr_agents.agents.configs import AgentMemoryConfig
    from dapr_agents.memory import ConversationDaprStateMemory, ConversationListMemory
    from dapr_agents.workflow.runners import AgentRunner
    DAPR_AGENTS_AVAILABLE = True
except ImportError:
    DAPR_AGENTS_AVAILABLE = False
    Agent = None
    DurableAgent = None
    dapr_tool = None
    AgentMemoryConfig = None
    ConversationDaprStateMemory = None
    ConversationListMemory = None
    AgentRunner = None

# Import custom Anthropic LLM client (uses native SDK with tool support)
try:
    from anthropic_llm import AnthropicChatClient, create_anthropic_client
    ANTHROPIC_LLM_AVAILABLE = True
except ImportError:
    ANTHROPIC_LLM_AVAILABLE = False
    AnthropicChatClient = None
    create_anthropic_client = None

# Import Claude Agent SDK for LLM capabilities
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

from planner_agent import PlannerAgent
from plan_manager import PlanManager


# =============================================================================
# Configuration
# =============================================================================

DAPR_STATE_STORE = os.getenv("DAPR_STATE_STORE", "statestore")
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
PLANS_DIR = Path(os.getenv("PLANS_DIR", "/plans"))


# =============================================================================
# Pydantic Models for Tool Inputs
# =============================================================================

class ExploreCodebaseInput(BaseModel):
    """Input for codebase exploration."""
    cwd: str = Field(description="Working directory path")
    query: str = Field(description="What to explore or search for")


class CreatePlanInput(BaseModel):
    """Input for plan creation."""
    cwd: str = Field(description="Working directory path")
    feature_request: str = Field(description="The feature request to plan")


class ExecutePlanInput(BaseModel):
    """Input for plan execution."""
    cwd: str = Field(description="Working directory path")
    plan_id: str = Field(description="Plan ID to execute")
    workflow_id: str = Field(description="Dapr workflow ID for streaming")


class PlanApprovalInput(BaseModel):
    """Input for plan approval status check."""
    plan_id: str = Field(description="Plan ID to check")


# =============================================================================
# Dapr Agent Tools (wrap Claude SDK operations)
# =============================================================================

if DAPR_AGENTS_AVAILABLE:

    @dapr_tool(args_model=ExploreCodebaseInput)
    async def explore_codebase(cwd: str, query: str) -> str:
        """
        Explore the codebase to understand its structure and find relevant files.
        Uses Claude SDK for intelligent code exploration.
        """
        print(f"[DurableAgent] Exploring codebase at {cwd}: {query}")

        options = ClaudeAgentOptions(
            system_prompt="""You are a code exploration assistant.
            Explore the codebase to find relevant files and understand the structure.
            Be concise in your findings.""",
            # Use claude_code preset for ALL native tools
            tools={"type": "preset", "preset": "claude_code"},
            # Full permissions - equivalent to --dangerously-skip-permissions
            permission_mode="bypassPermissions",
            cwd=cwd,
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(f"Explore the codebase: {query}")

                findings = []
                async for message in client.receive_response():
                    if hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                findings.append(block.text)

                return "\n".join(findings) if findings else "Exploration complete"
        except Exception as e:
            return f"Exploration error: {str(e)}"


    @dapr_tool(args_model=CreatePlanInput)
    async def create_implementation_plan(cwd: str, feature_request: str) -> str:
        """
        Create an implementation plan for a feature request.
        Uses Claude SDK planner agent for intelligent plan creation.
        """
        print(f"[DurableAgent] Creating plan for: {feature_request[:100]}...")

        agent = PlannerAgent(cwd=cwd, plans_dir=str(PLANS_DIR))

        try:
            await agent.run_planning_only(feature_request)

            if agent.plan_manager.current_plan:
                plan = agent.plan_manager.current_plan
                return json.dumps({
                    "success": True,
                    "plan_id": plan.id,
                    "title": plan.title,
                    "summary": plan.summary,
                    "steps_count": len(plan.steps),
                    "status": plan.status,
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "Failed to create plan",
                })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
            })


    @dapr_tool(args_model=ExecutePlanInput)
    async def execute_approved_plan(cwd: str, plan_id: str, workflow_id: str) -> str:
        """
        Execute an approved implementation plan.
        Uses Claude SDK for task execution with streaming progress.
        """
        print(f"[DurableAgent] Executing plan {plan_id}")

        agent = PlannerAgent(cwd=cwd, plans_dir=str(PLANS_DIR))

        try:
            result = await agent.run_execution_only(plan_id, workflow_id)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
            })


    @dapr_tool(args_model=PlanApprovalInput)
    def check_plan_approval(plan_id: str) -> str:
        """
        Check if a plan has been approved by the user.
        """
        plan_manager = PlanManager(PLANS_DIR)
        loaded = plan_manager.load_plan(plan_id)

        if not loaded:
            return json.dumps({"approved": False, "error": "Plan not found"})

        return json.dumps({
            "approved": plan_manager.current_plan.status == "approved",
            "status": plan_manager.current_plan.status,
        })


# =============================================================================
# Durable Planner Agent Factory
# =============================================================================

def create_durable_planner_agent(
    session_id: str,
    use_durable_memory: bool = True,
) -> DurableAgent | None:
    """
    Create a DurableAgent for fault-tolerant planning workflows.

    Args:
        session_id: Unique session ID for state persistence
        use_durable_memory: Whether to use Dapr state store for memory

    Returns:
        DurableAgent instance or None if dapr-agents not available
    """
    if not DAPR_AGENTS_AVAILABLE:
        print("[DurableAgent] dapr-agents not available, falling back to standard agent")
        return None

    # Configure memory based on environment
    if use_durable_memory:
        memory_config = AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name=DAPR_STATE_STORE,
                session_id=session_id,
            )
        )
    else:
        memory_config = AgentMemoryConfig(
            store=ConversationListMemory()
        )

    # Configure LLM using native Anthropic SDK with full tool support
    # This replaces DaprChatClient which doesn't support tool calling
    if not ANTHROPIC_LLM_AVAILABLE:
        print("[DurableAgent] AnthropicChatClient not available")
        return None

    llm_client = create_anthropic_client(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=4096,
    )

    # Create the DurableAgent with planning tools and Anthropic LLM
    planner = DurableAgent(
        name="DurablePlannerAgent",
        role="Software Planning Agent",
        goal="Help users plan and implement features in their codebase with fault tolerance",
        instructions=[
            "You are a durable software planning agent with fault-tolerant execution.",
            "Your workflow persists across crashes and restarts.",
            "",
            "## Workflow",
            "1. Use explore_codebase to understand the codebase structure",
            "2. Use create_implementation_plan to generate a detailed plan",
            "3. Wait for user approval (check with check_plan_approval)",
            "4. Use execute_approved_plan to implement the approved plan",
            "",
            "## Guidelines",
            "- Always explore before planning to understand context",
            "- Create detailed, actionable plans with clear steps",
            "- Wait for explicit user approval before execution",
            "- Handle failures gracefully - your state is persisted",
        ],
        tools=[
            explore_codebase,
            create_implementation_plan,
            execute_approved_plan,
            check_plan_approval,
        ],
        memory=memory_config,
        llm=llm_client,
    )

    return planner


# =============================================================================
# Durable Planning Workflow
# =============================================================================

class DurablePlanningWorkflow:
    """
    High-level workflow for durable planning using Dapr Agents.

    This class orchestrates the planning workflow with:
    - Durable state management
    - Fault-tolerant execution
    - Automatic retry and recovery
    """

    def __init__(
        self,
        session_id: str,
        cwd: str | Path,
        plans_dir: str | Path | None = None,
        runner: "AgentRunner | None" = None,
        agent: "DurableAgent | None" = None,
    ):
        self.session_id = session_id
        self.cwd = Path(cwd)
        self.plans_dir = Path(plans_dir) if plans_dir else PLANS_DIR
        # Use provided agent (global) or create a new one
        self.agent = agent if agent is not None else create_durable_planner_agent(session_id)
        # Use provided runner (global) or create a new one
        self.runner = runner if runner is not None else (AgentRunner() if DAPR_AGENTS_AVAILABLE else None)

    async def run_planning(self, feature_request: str) -> dict[str, Any]:
        """
        Run the planning phase durably.

        Args:
            feature_request: The feature request to plan

        Returns:
            Dict with plan details or error
        """
        if not self.agent or not self.runner:
            # Fallback to standard planner if dapr-agents not available
            print("[DurableWorkflow] Falling back to standard PlannerAgent")
            agent = PlannerAgent(cwd=str(self.cwd), plans_dir=str(self.plans_dir))
            await agent.run_planning_only(feature_request)

            if agent.plan_manager.current_plan:
                plan = agent.plan_manager.current_plan
                return {
                    "success": True,
                    "plan_id": plan.id,
                    "title": plan.title,
                    "summary": plan.summary,
                    "steps": [s.title for s in plan.steps],
                    "status": plan.status,
                }
            return {"success": False, "error": "Failed to create plan"}

        # Run with DurableAgent
        print(f"[DurableWorkflow] Starting durable planning for: {feature_request[:100]}...")

        try:
            # Run the DurableAgent to create a plan
            await self.runner.run(
                self.agent,
                payload={
                    "task": f"""Create an implementation plan for this feature:

{feature_request}

Steps:
1. First, explore the codebase at {self.cwd} to understand its structure
2. Then create a detailed implementation plan using create_implementation_plan
""",
                },
            )

            # Check for created plans in the plans directory
            # The DurableAgent creates plans via the CreateImplementationPlan tool
            # which writes to the plans directory
            plan_manager = PlanManager(str(self.plans_dir))
            plans = plan_manager.list_plans()

            if plans:
                # Get the most recently created plan
                latest_plan_filename = plans[-1]  # Plans are ordered by creation
                plan = plan_manager.load_plan(latest_plan_filename)
                if plan:
                    print(f"[DurableWorkflow] DurableAgent created plan: {plan.id}")
                    return {
                        "success": True,
                        "plan_id": plan.id,
                        "title": plan.title,
                        "summary": plan.summary,
                        "steps": [s.title for s in plan.steps],
                        "steps_count": len(plan.steps),
                        "status": plan.status,
                    }

            # No plan found, fall back
            print("[DurableWorkflow] DurableAgent did not produce a plan, falling back to PlannerAgent")
            return await self._fallback_planning(feature_request)

        except Exception as e:
            print(f"[DurableWorkflow] DurableAgent failed: {e}, falling back to PlannerAgent")
            return await self._fallback_planning(feature_request)

    async def _fallback_planning(self, feature_request: str) -> dict[str, Any]:
        """Fallback to standard PlannerAgent for planning."""
        print("[DurableWorkflow] Using standard PlannerAgent for planning")
        agent = PlannerAgent(cwd=str(self.cwd), plans_dir=str(self.plans_dir))
        await agent.run_planning_only(feature_request)

        if agent.plan_manager.current_plan:
            plan = agent.plan_manager.current_plan
            return {
                "success": True,
                "plan_id": plan.id,
                "title": plan.title,
                "summary": plan.summary,
                "steps": [s.title for s in plan.steps],
                "steps_count": len(plan.steps),
                "status": plan.status,
            }
        return {"success": False, "error": "Failed to create plan"}

    async def run_execution(
        self,
        plan_id: str,
        workflow_id: str,
    ) -> dict[str, Any]:
        """
        Run the execution phase durably.

        Args:
            plan_id: The approved plan ID to execute
            workflow_id: Dapr workflow ID for streaming events

        Returns:
            Dict with execution results
        """
        if not self.agent or not self.runner:
            # Fallback to standard execution
            print("[DurableWorkflow] Falling back to standard PlannerAgent execution")
            agent = PlannerAgent(cwd=str(self.cwd), plans_dir=str(self.plans_dir))
            return await agent.run_execution_only(plan_id, workflow_id)

        # Run with DurableAgent
        print(f"[DurableWorkflow] Starting durable execution for plan: {plan_id}")

        try:
            result = await self.runner.run(
                self.agent,
                payload={
                    "task": f"Execute the approved plan {plan_id} using execute_approved_plan with workflow_id {workflow_id} and cwd {self.cwd}",
                },
            )

            parsed = self._parse_result(result)

            # Check if DurableAgent produced execution results
            # If not (no tasks completed or just a text response), fall back to standard agent
            if parsed.get("tasks_completed", 0) == 0 and parsed.get("tasks_total", 0) == 0:
                if not parsed.get("files_changed"):
                    print("[DurableWorkflow] DurableAgent did not execute tasks, falling back to PlannerAgent")
                    return await self._fallback_execution(plan_id, workflow_id)

            return parsed

        except Exception as e:
            print(f"[DurableWorkflow] DurableAgent execution failed: {e}, falling back to PlannerAgent")
            return await self._fallback_execution(plan_id, workflow_id)

    async def _fallback_execution(self, plan_id: str, workflow_id: str) -> dict[str, Any]:
        """Fallback to standard PlannerAgent for execution."""
        print("[DurableWorkflow] Using standard PlannerAgent for execution")
        agent = PlannerAgent(cwd=str(self.cwd), plans_dir=str(self.plans_dir))
        return await agent.run_execution_only(plan_id, workflow_id)

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Parse the agent result into a standardized format."""
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"success": True, "result": result}
        return {"success": True, "result": str(result)}


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_durable_plan(
    cwd: str,
    feature_request: str,
    session_id: str | None = None,
    runner: "AgentRunner | None" = None,
    agent: "DurableAgent | None" = None,
) -> dict[str, Any]:
    """
    Create a plan using the durable workflow.

    Args:
        cwd: Working directory
        feature_request: Feature to plan
        session_id: Optional session ID (auto-generated if not provided)
        runner: Optional AgentRunner instance (uses global runner if available)
        agent: Optional DurableAgent instance (uses global agent if available)

    Returns:
        Plan creation result
    """
    session_id = session_id or f"plan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    workflow = DurablePlanningWorkflow(session_id=session_id, cwd=cwd, runner=runner, agent=agent)
    return await workflow.run_planning(feature_request)


async def execute_durable_plan(
    cwd: str,
    plan_id: str,
    workflow_id: str,
    session_id: str | None = None,
    runner: "AgentRunner | None" = None,
    agent: "DurableAgent | None" = None,
) -> dict[str, Any]:
    """
    Execute a plan using the durable workflow.

    Args:
        cwd: Working directory
        plan_id: Plan ID to execute
        workflow_id: Dapr workflow ID
        session_id: Optional session ID
        runner: Optional AgentRunner instance (uses global runner if available)
        agent: Optional DurableAgent instance (uses global agent if available)

    Returns:
        Execution result
    """
    session_id = session_id or f"exec-{plan_id}"
    workflow = DurablePlanningWorkflow(session_id=session_id, cwd=cwd, runner=runner, agent=agent)
    return await workflow.run_execution(plan_id, workflow_id)


# =============================================================================
# Check Availability
# =============================================================================

def is_durable_agents_available() -> bool:
    """Check if Dapr Agents and Anthropic LLM are available."""
    return DAPR_AGENTS_AVAILABLE and ANTHROPIC_LLM_AVAILABLE


def get_durable_agent_status() -> dict[str, Any]:
    """Get status of durable agent dependencies."""
    return {
        "dapr_agents_available": DAPR_AGENTS_AVAILABLE,
        "anthropic_llm_available": ANTHROPIC_LLM_AVAILABLE,
        "state_store": DAPR_STATE_STORE,
        "workspace_dir": str(WORKSPACE_DIR),
        "plans_dir": str(PLANS_DIR),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Test the durable agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Durable Planner Agent")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory")
    parser.add_argument("--session-id", type=str, help="Session ID")
    parser.add_argument("prompt", nargs="?", type=str, help="Feature request")

    args = parser.parse_args()

    print("=" * 60)
    print("DURABLE PLANNER AGENT")
    print("=" * 60)
    print(f"\nStatus: {get_durable_agent_status()}")

    if args.prompt:
        result = await create_durable_plan(
            cwd=args.cwd,
            feature_request=args.prompt,
            session_id=args.session_id,
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        print("\nNo prompt provided. Use: python durable_agent.py --cwd /path 'feature request'")


if __name__ == "__main__":
    asyncio.run(main())
