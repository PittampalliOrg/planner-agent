"""DaprOpenAIRunner - Bridge OpenAI Agents SDK with Dapr Workflow durability.

This module provides true workflow-level durability by:
- Converting OpenAI Agent tools to Dapr workflow activities
- Using ctx.call_activity() for durable tool execution
- Enabling crash recovery with automatic replay

Based on patterns from dapr-agents DurableAgent and AgentRunner:
- https://github.com/dapr/dapr-agents/blob/main/dapr_agents/agents/durable.py
- https://github.com/dapr/dapr-agents/blob/main/dapr_agents/workflow/runners/agent.py

Architecture:
```
┌─────────────────────────────────────────────────────────────────────┐
│                         DaprOpenAIRunner                             │
├─────────────────────────────────────────────────────────────────────┤
│  build_runtime() ────► WorkflowRuntime                              │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  @workflow_entry                                              │   │
│  │  def agent_workflow(ctx: DaprWorkflowContext, input):        │   │
│  │      # Each tool call becomes a durable activity             │   │
│  │      result = yield ctx.call_activity(tool_activity, input)  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Registered Activities (one per tool):                        │   │
│  │  - run_agent_activity (main agent execution)                  │   │
│  │  - tool activities (auto-generated from agent.tools)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import Dapr workflow extension
try:
    from dapr.ext.workflow import (
        WorkflowRuntime,
        DaprWorkflowContext,
        DaprWorkflowClient,
        WorkflowActivityContext,
    )
    DAPR_WORKFLOW_AVAILABLE = True
except ImportError:
    DAPR_WORKFLOW_AVAILABLE = False
    WorkflowRuntime = None
    DaprWorkflowContext = None
    DaprWorkflowClient = None
    WorkflowActivityContext = None
    logger.warning("dapr-ext-workflow not installed. DaprOpenAIRunner will not be available.")

# Try to import Dapr client for state operations
try:
    from dapr.clients import DaprClient
    DAPR_CLIENT_AVAILABLE = True
except ImportError:
    DAPR_CLIENT_AVAILABLE = False
    DaprClient = None


def build_runtime() -> Optional['WorkflowRuntime']:
    """Build a Dapr workflow runtime.

    Returns:
        WorkflowRuntime configured for agent workflows, or None if not available
    """
    if not DAPR_WORKFLOW_AVAILABLE:
        logger.error("Cannot build runtime: dapr-ext-workflow not installed")
        return None
    return WorkflowRuntime()


@dataclass
class DaprOpenAIRunnerConfig:
    """Configuration for DaprOpenAIRunner.

    Attributes:
        state_store: Dapr state store name for durability
        timeout_seconds: Maximum time to wait for workflow completion
        retry_max_attempts: Maximum retry attempts for activities
        retry_first_delay_seconds: Initial delay before first retry
        retry_max_delay_seconds: Maximum delay between retries
    """
    state_store: str = "statestore"
    timeout_seconds: int = 600
    retry_max_attempts: int = 3
    retry_first_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 30.0


@dataclass
class WorkflowResult:
    """Result from a workflow execution.

    Attributes:
        instance_id: Workflow instance ID
        status: Workflow status (scheduled, running, completed, failed, etc.)
        output: Workflow output data
        tasks: Tasks created during execution
        activities: Activities tracked during execution
        usage: Token usage statistics
        trace_metadata: Trace metadata for observability
        error: Error message if failed
    """
    instance_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class DaprOpenAIRunner:
    """Runner that bridges OpenAI Agents SDK with Dapr Workflow durability.

    This class provides workflow-level durability for OpenAI Agents by:
    1. Registering agent execution as a Dapr workflow
    2. Running the agent as a durable activity (crash-safe)
    3. Storing workflow state for recovery

    Usage:
        rt = build_runtime()
        runner = DaprOpenAIRunner(
            workflow_runtime=rt,
            auto_convert_tools_to_activities=True,
        )
        await runner.start()

        result = await runner.run(agent, "Create a plan...")

        await runner.stop()

    Based on dapr-agents patterns:
    - DurableAgent: @workflow_entry + ctx.call_activity()
    - AgentRunner: Workflow lifecycle management

    Note: This implementation runs the entire agent execution as a single
    durable activity. A more advanced implementation would intercept individual
    tool calls and run each as a separate activity for finer-grained durability.
    """

    def __init__(
        self,
        workflow_runtime: Optional['WorkflowRuntime'],
        auto_convert_tools_to_activities: bool = True,
        config: Optional[DaprOpenAIRunnerConfig] = None,
        activity_callback: Optional[Callable[[str, List[Dict]], None]] = None,
    ):
        """Initialize the runner.

        Args:
            workflow_runtime: Dapr WorkflowRuntime instance
            auto_convert_tools_to_activities: If True, automatically convert
                agent tools to workflow activities (reserved for future use)
            config: Optional configuration
            activity_callback: Optional callback for activity updates
        """
        if not DAPR_WORKFLOW_AVAILABLE:
            raise RuntimeError(
                "dapr-ext-workflow not installed. "
                "Install with: pip install dapr-ext-workflow>=1.14.0"
            )

        self._runtime = workflow_runtime
        self._auto_convert = auto_convert_tools_to_activities
        self._config = config or DaprOpenAIRunnerConfig()
        self._activity_callback = activity_callback

        self._wf_client: Optional['DaprWorkflowClient'] = None
        self._registered_workflows: Dict[str, bool] = {}
        self._registered_activities: Dict[str, Callable] = {}
        self._started = False

        # Current agent for workflow execution
        self._current_agent: Any = None
        self._current_workflow_id: Optional[str] = None

    async def start(self) -> None:
        """Start the workflow runtime and client."""
        if self._started:
            return

        if self._runtime is None:
            raise RuntimeError("Workflow runtime is None. Call build_runtime() first.")

        # Register the agent workflow and activity
        self._register_workflows()

        # Start the runtime
        self._runtime.start()

        # Create workflow client
        self._wf_client = DaprWorkflowClient()

        self._started = True
        logger.info("DaprOpenAIRunner started")

    async def stop(self) -> None:
        """Stop the workflow runtime and client."""
        if not self._started:
            return

        if self._wf_client:
            self._wf_client.close()
            self._wf_client = None

        if self._runtime:
            self._runtime.shutdown()

        self._started = False
        logger.info("DaprOpenAIRunner stopped")

    def _register_workflows(self) -> None:
        """Register the agent workflow and activities with the runtime."""
        if self._runtime is None:
            return

        # Register the main agent workflow
        self._runtime.register_workflow(self._agent_workflow)
        self._registered_workflows["agent_workflow"] = True

        # Register the agent execution activity
        self._runtime.register_activity(self._run_agent_activity)
        self._registered_activities["run_agent"] = True

        logger.info("Registered agent_workflow and run_agent_activity")

    def _agent_workflow(
        self,
        ctx: 'DaprWorkflowContext',
        input: str,
    ):
        """Main agent workflow (generator function for Dapr).

        This workflow:
        1. Parses input to get the user prompt and agent name
        2. Calls the agent execution activity durably
        3. Returns the result

        The workflow is durable - if the process crashes and restarts,
        the workflow will resume from the last completed activity.

        Args:
            ctx: Dapr workflow context
            input: JSON string with {"input": str, "agent_name": str}

        Yields:
            Activity call that runs the agent
        """
        # Parse input
        try:
            data = json.loads(input)
            user_input = data.get("input", "")
            agent_name = data.get("agent_name", "Agent")
        except (json.JSONDecodeError, TypeError):
            user_input = input
            agent_name = "Agent"

        if not ctx.is_replaying:
            logger.info(f"Starting workflow for {agent_name}")

        # Call the agent execution activity durably
        # This is the key durability feature - if we crash after this completes,
        # the result is cached and we won't re-execute the agent
        result = yield ctx.call_activity(
            self._run_agent_activity,
            input=json.dumps({
                "input": user_input,
                "workflow_id": ctx.instance_id,
            }),
        )

        if not ctx.is_replaying:
            logger.info(f"Workflow completed for {agent_name}")

        return result

    def _run_agent_activity(
        self,
        ctx: 'WorkflowActivityContext',
        input: str,
    ) -> str:
        """Activity that runs the OpenAI agent.

        This activity executes the agent synchronously. Since it's wrapped
        as a Dapr activity, it benefits from:
        - Result caching (replay returns cached result)
        - Automatic retries (configurable)
        - State persistence

        Args:
            ctx: Dapr activity context
            input: JSON string with {"input": str, "workflow_id": str}

        Returns:
            JSON string with agent result
        """
        try:
            data = json.loads(input)
            user_input = data.get("input", "")
            workflow_id = data.get("workflow_id", "")

            if self._current_agent is None:
                return json.dumps({
                    "status": "failed",
                    "error": "No agent configured",
                })

            # Import here to avoid circular imports
            from agents import Runner

            # Import workflow context for task tracking
            from workflow_context import WorkflowContext

            # Create async function to run the agent
            async def run_agent():
                # Execute with WorkflowContext for activity tracking
                async with WorkflowContext(
                    workflow_id=workflow_id,
                    state_store=self._config.state_store,
                    workflow_name="planner_workflow",
                    metadata={"source": "workflow_mode", "version": "3.0"},
                    activity_callback=self._activity_callback,
                ) as session:
                    # Track agent start
                    session.track_activity(
                        name="agent:planning",
                        status="running",
                        input_data={"task": user_input[:200], "mode": "workflow"},
                    )

                    # Run the agent using standard SDK pattern
                    result = await Runner.run(
                        starting_agent=self._current_agent,
                        input=user_input,
                    )

                    # Track agent completion
                    session.track_activity(
                        name="agent:planning",
                        status="completed",
                        output_data={
                            "tasks_created": len(session.tasks),
                            "usage": session.usage,
                        },
                    )

                    return result, session

            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result, session = loop.run_until_complete(run_agent())
            finally:
                loop.close()

            # Build response
            return json.dumps({
                "status": "completed",
                "output": result.final_output,
                "tasks": session.tasks,
                "usage": session.usage,
                "trace": session.trace_metadata,
            })

        except Exception as e:
            logger.error(f"Agent activity failed: {e}")
            return json.dumps({
                "status": "failed",
                "error": str(e),
            })

    async def run(
        self,
        agent: Any,
        input: str,
        workflow_id: Optional[str] = None,
        wait: bool = True,
    ) -> WorkflowResult:
        """Run an OpenAI Agent with Dapr workflow durability.

        Args:
            agent: OpenAI Agents SDK Agent instance
            input: User input/prompt
            workflow_id: Optional workflow ID (auto-generated if not provided)
            wait: If True, wait for completion; else return immediately

        Returns:
            WorkflowResult with workflow status, output, tasks, and trace metadata
        """
        if not self._started:
            await self.start()

        workflow_id = workflow_id or f"wf-{uuid.uuid4().hex[:12]}"

        # Store agent reference for the activity to use
        self._current_agent = agent
        self._current_workflow_id = workflow_id

        # Schedule the workflow
        instance_id = self._wf_client.schedule_new_workflow(
            workflow=self._agent_workflow,
            input=json.dumps({
                "input": input,
                "agent_name": agent.name if hasattr(agent, 'name') else "Agent",
            }),
            instance_id=workflow_id,
        )

        logger.info(f"Scheduled workflow {instance_id}")

        if not wait:
            return WorkflowResult(
                instance_id=instance_id,
                status="scheduled",
            )

        # Wait for completion
        state = self._wf_client.wait_for_workflow_completion(
            instance_id,
            fetch_payloads=True,
            timeout_in_seconds=self._config.timeout_seconds,
        )

        if state:
            try:
                output = json.loads(state.serialized_output) if state.serialized_output else {}
                # Activity returns JSON string, so output might be a string needing another parse
                if isinstance(output, str):
                    output = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                output = {"raw": state.serialized_output}

            # Extract tasks, usage, trace from output
            tasks = output.get("tasks", []) if isinstance(output, dict) else []
            usage = output.get("usage", {}) if isinstance(output, dict) else {}
            trace = output.get("trace", {}) if isinstance(output, dict) else {}
            error = output.get("error") if isinstance(output, dict) else None

            return WorkflowResult(
                instance_id=instance_id,
                status=state.runtime_status.name if state.runtime_status else "UNKNOWN",
                output=output,
                tasks=tasks,
                usage=usage,
                trace_metadata=trace,
                error=error,
            )

        return WorkflowResult(
            instance_id=instance_id,
            status="TIMEOUT",
            error="Workflow timed out",
        )

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get the status of a workflow.

        Args:
            workflow_id: The workflow instance ID

        Returns:
            WorkflowResult with current status, or None if not found
        """
        if not self._started or not self._wf_client:
            return None

        try:
            state = self._wf_client.get_workflow_state(
                workflow_id,
                fetch_payloads=True,
            )

            if state:
                try:
                    output = json.loads(state.serialized_output) if state.serialized_output else {}
                except (json.JSONDecodeError, TypeError):
                    output = {}

                return WorkflowResult(
                    instance_id=workflow_id,
                    status=state.runtime_status.name if state.runtime_status else "UNKNOWN",
                    output=output,
                    tasks=output.get("tasks", []),
                    usage=output.get("usage", {}),
                    trace_metadata=output.get("trace", {}),
                )
        except Exception as e:
            logger.warning(f"Failed to get workflow status: {e}")

        return None

    async def terminate_workflow(self, workflow_id: str) -> bool:
        """Terminate a running workflow.

        Args:
            workflow_id: The workflow instance ID

        Returns:
            True if terminated successfully
        """
        if not self._started or not self._wf_client:
            return False

        try:
            self._wf_client.terminate_workflow(workflow_id)
            logger.info(f"Terminated workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate workflow: {e}")
            return False


# Global runner instance (initialized on first use)
_global_runner: Optional[DaprOpenAIRunner] = None


async def get_or_create_runner(
    activity_callback: Optional[Callable[[str, List[Dict]], None]] = None,
) -> Optional[DaprOpenAIRunner]:
    """Get or create the global DaprOpenAIRunner instance.

    Args:
        activity_callback: Optional callback for activity updates

    Returns:
        DaprOpenAIRunner instance, or None if not available
    """
    global _global_runner

    if not DAPR_WORKFLOW_AVAILABLE:
        logger.warning("Dapr workflow extension not available")
        return None

    if _global_runner is None:
        rt = build_runtime()
        if rt is None:
            return None

        _global_runner = DaprOpenAIRunner(
            workflow_runtime=rt,
            auto_convert_tools_to_activities=True,
            activity_callback=activity_callback,
        )
        await _global_runner.start()

    return _global_runner


async def run_durable_agent(
    agent: Any,
    input: str,
    workflow_id: Optional[str] = None,
    activity_callback: Optional[Callable[[str, List[Dict]], None]] = None,
) -> WorkflowResult:
    """Run an OpenAI Agent with Dapr workflow durability.

    This is a convenience function that handles runtime lifecycle.

    Args:
        agent: OpenAI Agents SDK Agent instance
        input: User input/prompt
        workflow_id: Optional workflow ID
        activity_callback: Optional callback for activity updates

    Returns:
        WorkflowResult with workflow status, output, tasks, and trace metadata

    Raises:
        RuntimeError: If Dapr workflow extension is not available
    """
    runner = await get_or_create_runner(activity_callback)

    if runner is None:
        raise RuntimeError(
            "Cannot run durable agent: Dapr workflow extension not available. "
            "Install with: pip install dapr-ext-workflow>=1.14.0"
        )

    return await runner.run(agent, input, workflow_id)


# Availability check
def is_dapr_workflow_available() -> bool:
    """Check if Dapr workflow extension is available.

    Returns:
        True if dapr-ext-workflow is installed and can be imported
    """
    return DAPR_WORKFLOW_AVAILABLE
