"""Workflow Context for durable agent execution.

This module provides a WorkflowContext context manager that wraps OpenAI Agents SDK
execution with Dapr durability features:
- State persistence for recovery
- Activity tracking for visualization
- Trace metadata for observability

Note: This is separate from the official `agents.extensions.memory.dapr_session.DaprSession`
which provides conversation memory. WorkflowContext provides activity tracking and workflow
state management that complements the official DaprSession.

Usage:
    from agents import Agent, Runner
    from agents.extensions.memory.dapr_session import DaprSession
    from workflow_context import WorkflowContext

    agent = Agent(name="Planner", instructions="...", tools=[...])

    # Create official DaprSession for conversation memory
    session = DaprSession.from_address(session_id="wf-123", state_store_name="statestore")

    # Create WorkflowContext for activity tracking
    async with WorkflowContext(workflow_id="wf-123", state_store="statestore") as ctx:
        result = await Runner.run(starting_agent=agent, input="Create a plan...", session=session)
        # Context automatically tracks activities and persists state

    # Access context data after execution
    print(ctx.tasks)
    print(ctx.usage)
    print(ctx.trace_metadata)
"""

import json
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from dapr.clients import DaprClient

logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI-compatible ID Generation
# =============================================================================


def generate_trace_id() -> str:
    """Generate a trace ID in OpenAI format: trace_<32_hex>."""
    return f"trace_{secrets.token_hex(16)}"


def generate_span_id() -> str:
    """Generate a span ID in OpenAI format: span_<24_hex>."""
    return f"span_{secrets.token_hex(12)}"


def generate_group_id() -> str:
    """Generate a group ID for linking related traces."""
    return f"group_{secrets.token_hex(12)}"


# =============================================================================
# Session State
# =============================================================================


@dataclass
class SessionState:
    """Persistent state for a workflow context.

    This is stored in Dapr state store and can be recovered after restart.
    """
    workflow_id: str
    trace_id: str
    agent_span_id: str
    group_id: str = ""
    workflow_name: str = ""
    status: str = "running"
    started_at: str = ""
    completed_at: str = ""
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    task_counter: int = 0
    llm_call_count: int = 0
    usage: Dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Add token usage to aggregated totals."""
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["total_tokens"] = (
            self.usage["input_tokens"] + self.usage["output_tokens"]
        )

    def get_trace_metadata(self) -> Dict[str, Any]:
        """Get OpenAI-compatible trace metadata."""
        return {
            "trace_id": self.trace_id,
            "agent_span_id": self.agent_span_id,
            "workflow_name": self.workflow_name,
            "group_id": self.group_id,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata,
            "usage": self.usage,
        }


# =============================================================================
# Context Variable for Current Session
# =============================================================================

import contextvars

_current_context: contextvars.ContextVar[Optional['WorkflowContext']] = \
    contextvars.ContextVar('workflow_context', default=None)


def get_current_context() -> Optional['WorkflowContext']:
    """Get the current WorkflowContext from context."""
    return _current_context.get()


def get_context_state() -> Optional[SessionState]:
    """Get the current context state (for use in tools)."""
    ctx = get_current_context()
    return ctx.state if ctx else None


# Backward compatibility aliases
def get_current_session() -> Optional['WorkflowContext']:
    """Alias for get_current_context() for backward compatibility."""
    return get_current_context()


def get_session_state() -> Optional[SessionState]:
    """Alias for get_context_state() for backward compatibility."""
    return get_context_state()


# =============================================================================
# Workflow Context
# =============================================================================


class WorkflowContext:
    """Context manager for durable agent execution with Dapr.

    Provides:
    - Automatic state persistence to Dapr state store
    - Activity tracking for visualization
    - OpenAI-compatible trace metadata
    - Recovery support for interrupted workflows

    This is separate from the official DaprSession which handles conversation
    memory. WorkflowContext handles workflow-level state and activity tracking.

    Usage:
        async with WorkflowContext(workflow_id="wf-123") as ctx:
            result = await Runner.run(agent, input="...")

        # Access results
        print(ctx.tasks)
        print(ctx.usage)
    """

    def __init__(
        self,
        workflow_id: str,
        state_store: str = "statestore",
        workflow_name: str = "planner_workflow",
        metadata: Optional[Dict[str, Any]] = None,
        activity_callback: Optional[Callable[[str, List[Dict]], None]] = None,
    ):
        """Initialize a workflow context.

        Args:
            workflow_id: Unique identifier for this workflow
            state_store: Dapr state store name
            workflow_name: Logical workflow name for tracing
            metadata: Additional metadata to include in trace
            activity_callback: Optional callback when activities are updated
        """
        self.workflow_id = workflow_id
        self.state_store = state_store
        self.activity_callback = activity_callback

        # Initialize session state
        self.state = SessionState(
            workflow_id=workflow_id,
            trace_id=generate_trace_id(),
            agent_span_id=generate_span_id(),
            workflow_name=workflow_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        self._token: Optional[contextvars.Token] = None

    async def __aenter__(self) -> 'WorkflowContext':
        """Enter the context."""
        # Set as current context
        self._token = _current_context.set(self)

        # Try to recover existing state
        existing = self._load_state()
        if existing:
            logger.info(f"Recovered existing context state for {self.workflow_id}")
            self.state = existing

        # Persist initial state
        self._save_state()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        # Update completion status
        self.state.completed_at = datetime.now(timezone.utc).isoformat()
        self.state.status = "failed" if exc_type else "completed"

        # Persist final state
        self._save_state()

        # Clear context
        if self._token:
            _current_context.reset(self._token)

        return False  # Don't suppress exceptions

    # -------------------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------------------

    def _load_state(self) -> Optional[SessionState]:
        """Load context state from Dapr state store."""
        key = f"workflow-context-{self.workflow_id}"
        try:
            with DaprClient() as client:
                data = client.get_state(self.state_store, key)
                if data.data:
                    state_dict = json.loads(data.data.decode('utf-8'))
                    return SessionState(**state_dict)
        except Exception as e:
            logger.warning(f"Failed to load context state: {e}")
        return None

    def _save_state(self) -> None:
        """Save context state to Dapr state store."""
        key = f"workflow-context-{self.workflow_id}"
        try:
            with DaprClient() as client:
                client.save_state(self.state_store, key, json.dumps(asdict(self.state)))
        except Exception as e:
            logger.warning(f"Failed to save context state: {e}")

    # -------------------------------------------------------------------------
    # Activity Tracking
    # -------------------------------------------------------------------------

    def track_activity(
        self,
        name: str,
        status: str,
        input_data: Optional[Dict] = None,
        output_data: Optional[Dict] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track an activity execution.

        Args:
            name: Activity name (e.g., "tool:create_task")
            status: Status ("running", "completed", "failed")
            input_data: Optional input data
            output_data: Optional output data
            span_id: Optional span ID (generated if not provided)

        Returns:
            The activity record
        """
        now = datetime.now(timezone.utc).isoformat()
        span_id = span_id or generate_span_id()

        # Find existing activity or create new one
        existing = None
        for activity in self.state.activities:
            if activity["activityName"] == name and activity["status"] == "running":
                existing = activity
                break

        if existing and status in ("completed", "failed"):
            # Update existing activity
            existing["status"] = status
            existing["endTime"] = now
            if existing.get("startTime"):
                start = datetime.fromisoformat(existing["startTime"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(now.replace("Z", "+00:00"))
                existing["durationMs"] = int((end - start).total_seconds() * 1000)
            if output_data:
                existing["output"] = output_data
            activity = existing
        else:
            # Create new activity
            activity = {
                "activityName": name,
                "status": status,
                "startTime": now,
                "span_id": span_id,
                "parent_id": self.state.agent_span_id,
                "span_type": "function" if name.startswith("tool:") else "agent",
            }
            if input_data:
                activity["input"] = input_data
            if output_data:
                activity["output"] = output_data
            self.state.activities.append(activity)

        # Notify callback
        if self.activity_callback and status in ("completed", "failed"):
            self.activity_callback(self.workflow_id, self.state.activities)

        return activity

    # -------------------------------------------------------------------------
    # Activity Hooks for Runner
    # -------------------------------------------------------------------------

    @property
    def activity_hooks(self):
        """Get RunHooks-compatible hooks for activity tracking.

        Returns a hooks object that can be passed to Runner.run() for
        tracking LLM calls and agent lifecycle events.
        """
        # Return None for now - hooks can cause issues with some SDK versions
        # Activity tracking is handled by track_activity() method
        return None

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        """Get the tasks created during this context."""
        return self.state.tasks

    @property
    def activities(self) -> List[Dict[str, Any]]:
        """Get the activities tracked during this context."""
        return self.state.activities

    @property
    def usage(self) -> Dict[str, int]:
        """Get the aggregated token usage."""
        return self.state.usage

    @property
    def trace_metadata(self) -> Dict[str, Any]:
        """Get the OpenAI-compatible trace metadata."""
        return self.state.get_trace_metadata()

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self.state.trace_id

    @property
    def agent_span_id(self) -> str:
        """Get the agent span ID."""
        return self.state.agent_span_id


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Alias for backward compatibility with code that imports DaprSession
DaprSession = WorkflowContext


# =============================================================================
# Compatibility with durable_runner.py
# =============================================================================
# These functions provide backward compatibility with the existing
# WorkflowExecutionContext pattern used in durable_runner.py

def get_workflow_context():
    """Get workflow context (compatibility shim for durable_runner)."""
    ctx = get_current_context()
    if ctx:
        return ctx.state
    # Fallback to durable_runner context
    from durable_runner import get_workflow_context as _get_workflow_context
    return _get_workflow_context()
