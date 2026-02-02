"""Durable Agent Runner with Dapr state store integration.

This module provides durability for OpenAI Agents SDK tool calls by:
- Recording tool call inputs/outputs to Dapr state store
- Supporting replay/recovery from checkpoints
- Tracking activities for visualization
- Preserving context across async boundaries

Follows the patterns from dapr/python-sdk PR #827:
- ExecuteToolRequest with metadata envelope (adapted from ExecuteActivityRequest)
- Interceptor chain for composable middleware (via compose_tool_chain)
- Context variables for state propagation
- Generator wrapper for yield point context preservation

When PR #827 is merged, this code can be migrated to use the official
dapr-ext-workflow interceptors by:
1. Replacing ExecuteToolRequest with ExecuteActivityRequest
2. Using BaseRuntimeInterceptor instead of BaseToolInterceptor
3. Using compose_runtime_chain instead of compose_tool_chain
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

from dapr.clients import DaprClient

from interceptors import (
    BaseToolInterceptor,
    ExecuteToolRequest,
    InterceptorChain,
    compose_tool_chain,
    wrap_payload_with_metadata,
    unwrap_payload_with_metadata,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Durable Run Result
# =============================================================================


@dataclass
class DurableRunResult:
    """Result from DurableAgentRunner.run() with context data.

    This wraps the SDK's RunResult and includes additional data from
    the workflow context that would otherwise be lost after context cleanup.

    Attributes:
        result: The original RunResult from the SDK
        tasks: List of tasks created during execution
        activities: List of activities tracked during execution
        usage: Aggregated token usage (OpenAI-compatible)
        trace_metadata: OpenAI-compatible trace metadata
        llm_call_count: Number of LLM calls made
    """
    result: Any  # RunResult from SDK
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    llm_call_count: int = 0

    @property
    def final_output(self) -> Any:
        """Proxy to result.final_output for compatibility."""
        return getattr(self.result, 'final_output', None)

    @property
    def new_items(self) -> List[Any]:
        """Proxy to result.new_items for compatibility."""
        return getattr(self.result, 'new_items', [])


# =============================================================================
# Workflow Context (Context Variables) - PR #827 Pattern
# =============================================================================
# Uses contextvars for thread-safe context propagation, matching PR #827's
# approach for maintaining execution state.


def _generate_trace_id() -> str:
    """Generate a trace ID in OpenAI format: trace_<32_hex>.

    OpenAI uses lowercase hex characters only (0-9, a-f).
    """
    import secrets
    return f"trace_{secrets.token_hex(16)}"


def _generate_group_id() -> str:
    """Generate a group ID for linking traces from same conversation."""
    import secrets
    return f"group_{secrets.token_hex(12)}"


@dataclass
class WorkflowExecutionContext:
    """Execution context for a workflow run with OpenAI-compatible trace fields.

    This mirrors PR #827's pattern of maintaining execution state
    in context variables rather than global state.

    OpenAI Trace Schema Compatibility:
    - trace_id: Format trace_<32_hex>
    - span_id: Format span_<24_hex>
    - group_id: Links traces from same conversation
    - workflow_name: Logical workflow name

    Attributes:
        workflow_id: Unique identifier for this workflow instance
        instance_id: Alias for workflow_id (Dapr terminology)
        metadata: Additional context metadata (propagated via envelope)
        activities: List of activity executions for visualization
        tasks: List of planning tasks created
        task_counter: Counter for generating task IDs
        llm_call_count: Counter for LLM calls
        trace_id: OpenAI-format trace ID
        group_id: Links related traces
        workflow_name: Logical workflow name
        agent_span_id: Agent span ID (parent for function/response spans)
        usage: Aggregated token usage
    """
    workflow_id: str
    instance_id: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_counter: int = 0
    llm_call_count: int = 0
    # OpenAI-compatible trace fields
    trace_id: str = ""
    group_id: str = ""
    workflow_name: str = ""
    agent_span_id: str = ""  # Parent span for function/response spans
    usage: Dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = self.workflow_id
        # Generate OpenAI-format trace ID if not provided
        if not self.trace_id:
            self.trace_id = _generate_trace_id()
        # Generate agent span ID for parent hierarchy
        if not self.agent_span_id:
            self.agent_span_id = _generate_span_id()

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
            "workflow_name": self.workflow_name or "planner_workflow",
            "group_id": self.group_id,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata,
            "usage": self.usage,
        }


# Context variable for current workflow execution (PR #827 pattern)
_workflow_context: contextvars.ContextVar[Optional[WorkflowExecutionContext]] = \
    contextvars.ContextVar('workflow_context', default=None)


def get_workflow_context() -> Optional[WorkflowExecutionContext]:
    """Get the current workflow execution context."""
    return _workflow_context.get()


def set_workflow_context(ctx: Optional[WorkflowExecutionContext]) -> contextvars.Token:
    """Set the current workflow execution context. Returns token for reset."""
    return _workflow_context.set(ctx)


def clear_workflow_context() -> None:
    """Clear the current workflow execution context."""
    _workflow_context.set(None)


# =============================================================================
# Tool Call Recording (Durability)
# =============================================================================


@dataclass
class ToolCallRecord:
    """Record of a tool call for replay/recovery.

    When a workflow is interrupted and restarted, tool calls that
    have already completed can be replayed from these records
    instead of re-executing.

    This is analogous to how Dapr Workflow records activity results
    for replay during workflow recovery.

    Attributes:
        call_id: Unique identifier for this call
        tool_name: Name of the tool being called
        input: Input arguments to the tool (wrapped with metadata envelope)
        output: Output from the tool (if completed)
        status: Current status (pending, running, completed, failed)
        started_at: Timestamp when execution started
        completed_at: Timestamp when execution completed
        error: Error message if failed
        attempt: Retry attempt number
    """
    call_id: str
    tool_name: str
    input: Dict[str, Any]
    output: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed
    started_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    attempt: int = 1


# =============================================================================
# Durability Interceptor
# =============================================================================


class DurabilityInterceptor(BaseToolInterceptor):
    """Records tool calls to Dapr state store for durability.

    This interceptor provides:
    - Checkpointing: Records tool call start before execution
    - Recovery: Returns cached result if call already completed
    - Replay: On restart, completed calls return cached results

    This is analogous to how Dapr Workflow's durable task framework
    records activity executions for replay.

    State store keys:
    - tool-call-{workflow_id}-{call_id}: Individual tool call record
    - tool-calls-{workflow_id}: List of all call IDs for this workflow

    Usage:
        interceptors = [DurabilityInterceptor(state_store="statestore")]
        chain_fn = compose_tool_chain(interceptors, tool_function)
        result = chain_fn(request)
    """

    def __init__(
        self,
        state_store: str = "statestore",
        key_prefix: str = "tool-call"
    ):
        self.state_store = state_store
        self.key_prefix = key_prefix

    def execute_tool(
        self,
        input: ExecuteToolRequest,
        next: Callable[[ExecuteToolRequest], Any],
    ) -> Any:
        workflow_id = input.workflow_id
        call_id = input.call_id or str(uuid.uuid4())
        tool_name = input.tool_name

        if not workflow_id:
            # No workflow context, execute without durability
            return next(input)

        # Check if already completed (replay scenario)
        existing = self._get_tool_call(workflow_id, call_id)
        if existing and existing.status == "completed":
            logger.info(f"Replaying cached result for {tool_name} ({call_id})")
            return existing.output

        # Wrap input with metadata envelope (PR #827 pattern)
        ctx = get_workflow_context()
        metadata = ctx.metadata if ctx else None
        wrapped_input = wrap_payload_with_metadata(input.input, metadata)

        # Record start
        record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            input=self._serialize_input(wrapped_input),
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save_tool_call(workflow_id, record)

        try:
            # Execute tool
            result = next(input)

            # Record completion
            record.output = self._serialize_output(result)
            record.status = "completed"
            record.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_tool_call(workflow_id, record)

            return result

        except Exception as e:
            # Record failure
            record.status = "failed"
            record.error = str(e)
            record.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_tool_call(workflow_id, record)
            raise

    def _get_tool_call(
        self,
        workflow_id: str,
        call_id: str
    ) -> Optional[ToolCallRecord]:
        """Get a tool call record from state store."""
        key = f"{self.key_prefix}-{workflow_id}-{call_id}"
        try:
            with DaprClient() as client:
                data = client.get_state(self.state_store, key)
                if data.data:
                    record_dict = json.loads(data.data.decode('utf-8'))
                    return ToolCallRecord(**record_dict)
        except Exception as e:
            logger.warning(f"Failed to get tool call record: {e}")
        return None

    def _save_tool_call(self, workflow_id: str, record: ToolCallRecord) -> None:
        """Save a tool call record to state store."""
        key = f"{self.key_prefix}-{workflow_id}-{record.call_id}"
        try:
            with DaprClient() as client:
                client.save_state(
                    self.state_store,
                    key,
                    json.dumps(asdict(record)),
                )
        except Exception as e:
            logger.warning(f"Failed to save tool call record: {e}")

    def _serialize_input(self, payload: Any) -> Dict[str, Any]:
        """Serialize input for storage, handling complex SDK types."""
        return {"data": self._safe_serialize(payload)}

    def _serialize_output(self, result: Any) -> Any:
        """Serialize output for storage, handling complex SDK types."""
        return self._safe_serialize(result)

    def _safe_serialize(self, value: Any, max_depth: int = 5) -> Any:
        """Safely serialize any value to JSON-compatible format.

        Handles complex types from OpenAI SDK like InputTokensDetails, ToolContext, etc.

        Args:
            value: Value to serialize
            max_depth: Maximum recursion depth

        Returns:
            JSON-serializable value
        """
        if max_depth <= 0:
            return str(value)[:200]

        # Handle None and primitives
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate(value)

        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            if len(value) > 100:
                return [self._safe_serialize(v, max_depth - 1) for v in value[:100]] + [f"... ({len(value)} total items)"]
            return [self._safe_serialize(v, max_depth - 1) for v in value]

        # Handle dicts
        if isinstance(value, dict):
            return {str(k): self._safe_serialize(v, max_depth - 1) for k, v in list(value.items())[:50]}

        # Handle objects with __dict__ (like SDK models)
        if hasattr(value, '__dict__'):
            try:
                obj_dict = {}
                for k, v in value.__dict__.items():
                    if not k.startswith('_'):
                        obj_dict[k] = self._safe_serialize(v, max_depth - 1)
                obj_dict['__type__'] = type(value).__name__
                return obj_dict
            except Exception:
                pass

        # Handle Pydantic models
        if hasattr(value, 'model_dump'):
            try:
                return self._safe_serialize(value.model_dump(), max_depth - 1)
            except Exception:
                pass

        # Handle dataclasses
        if hasattr(value, '__dataclass_fields__'):
            try:
                from dataclasses import asdict
                return self._safe_serialize(asdict(value), max_depth - 1)
            except Exception:
                pass

        # Fallback: convert to string
        return str(value)[:500]

    def _truncate(self, value: Any, max_length: int = 1000) -> Any:
        """Truncate large values for storage."""
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + f"... ({len(value)} total chars)"
        if isinstance(value, (list, tuple)) and len(value) > 100:
            return list(value[:100]) + [f"... ({len(value)} total items)"]
        return value

    def get_all_tool_calls(self, workflow_id: str) -> List[ToolCallRecord]:
        """Get all tool call records for a workflow."""
        records = []
        try:
            with DaprClient() as client:
                # Get the index of call IDs
                index_key = f"{self.key_prefix}s-{workflow_id}"
                index_data = client.get_state(self.state_store, index_key)
                if index_data.data:
                    call_ids = json.loads(index_data.data.decode('utf-8'))
                    for call_id in call_ids:
                        record = self._get_tool_call(workflow_id, call_id)
                        if record:
                            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to get all tool calls: {e}")
        return records


# =============================================================================
# Activity Tracking Interceptor
# =============================================================================


def _generate_span_id() -> str:
    """Generate a span ID in OpenAI format: span_<24_hex>.

    OpenAI uses lowercase hex characters only (0-9, a-f).
    """
    import secrets
    return f"span_{secrets.token_hex(12)}"


class ActivityTrackingInterceptor(BaseToolInterceptor):
    """Tracks tool calls as activities for visualization with OpenAI-compatible spans.

    This interceptor records activity start/completion to the workflow
    context for display in ai-chatbot. It also persists activities
    to the state store via the update_callback.

    This is analogous to how PR #827's ContextRuntimeInterceptor tracks
    workflow and activity execution for observability.

    Activity format (compatible with ai-chatbot + OpenAI tracing schema):
        {
            "activityName": "tool:read_file",
            "status": "running" | "completed" | "failed",
            "startTime": "2024-01-15T10:30:00Z",
            "endTime": "2024-01-15T10:30:01Z",
            "durationMs": 1000,
            "input": {...},
            "output": {...},
            # OpenAI-compatible fields:
            "span_id": "span_abc123...",
            "parent_id": "span_parent...",  # optional
            "span_type": "function",
            "function_span": {
                "name": "read_file",
                "input": {...},  # full input (configurable)
                "output": {...},  # full output (configurable)
            }
        }

    Attributes:
        update_callback: Optional callback for persisting activities
        capture_full_data: If True, capture full input/output (not truncated)
        parent_span_id: Optional parent span for hierarchy
    """

    def __init__(
        self,
        update_callback: Optional[Callable[[str, List[Dict]], None]] = None,
        capture_full_data: bool = False,
        parent_span_id: Optional[str] = None,
    ):
        """Initialize with optional callback for persisting activities.

        Args:
            update_callback: Function(workflow_id, activities) to persist activities
            capture_full_data: If True, don't truncate input/output in function_span
            parent_span_id: Optional parent span ID for hierarchy
        """
        self.update_callback = update_callback
        self.capture_full_data = capture_full_data
        self.parent_span_id = parent_span_id

    def execute_tool(
        self,
        input: ExecuteToolRequest,
        next: Callable[[ExecuteToolRequest], Any],
    ) -> Any:
        ctx = get_workflow_context()
        if not ctx:
            return next(input)

        tool_name = f"tool:{input.tool_name}"
        now = datetime.now(timezone.utc).isoformat()

        # Generate OpenAI-format span ID
        span_id = _generate_span_id()

        # Extract tool arguments from the request
        # The _wrap_tool method stores actual tool args under "_tool_args" key
        raw_input = input.input or {}
        if "_tool_args" in raw_input:
            # New format: tool args stored explicitly
            clean_input = raw_input["_tool_args"]
        else:
            # Fallback: filter out internal keys
            clean_input = {k: v for k, v in raw_input.items()
                          if not k.startswith('_')}

        # Create activity record with OpenAI-compatible fields
        activity = {
            "activityName": tool_name,
            "status": "running",
            "startTime": now,
            "input": self._serialize_input(clean_input),
            # OpenAI-compatible fields
            "span_id": span_id,
            "span_type": "function",
        }

        # Add parent_id for hierarchy (links to agent span)
        if self.parent_span_id:
            activity["parent_id"] = self.parent_span_id
        elif ctx.agent_span_id:
            # Use agent span as parent for function spans
            activity["parent_id"] = ctx.agent_span_id

        # Also store call_id for durability tracking
        if input.call_id:
            activity["call_id"] = input.call_id

        # Add function span data (OpenAI FunctionSpanData equivalent)
        activity["function_span"] = {
            "name": input.tool_name,
            "input": self._serialize_full(clean_input) if self.capture_full_data
                     else self._serialize_input(clean_input),
        }

        ctx.activities.append(activity)

        # Helper to complete the activity
        def complete_activity(result: Any, error: Optional[str] = None):
            activity["endTime"] = datetime.now(timezone.utc).isoformat()
            activity["durationMs"] = self._calculate_duration(
                activity["startTime"],
                activity["endTime"]
            )

            if error:
                activity["status"] = "failed"
                activity["output"] = {"error": error}
                activity["error"] = error
                activity["function_span"]["output"] = {"error": error}
            else:
                activity["status"] = "completed"
                activity["output"] = self._serialize_output(result)
                activity["function_span"]["output"] = (
                    self._serialize_full(result) if self.capture_full_data
                    else self._serialize_output(result)
                )

            if self.update_callback:
                self.update_callback(ctx.workflow_id, ctx.activities)

        try:
            result = next(input)

            # Handle async results - wrap in a coroutine that completes the activity
            if asyncio.iscoroutine(result):
                async def await_and_complete():
                    try:
                        actual_result = await result
                        complete_activity(actual_result)
                        return actual_result
                    except Exception as e:
                        complete_activity(None, str(e))
                        raise

                return await_and_complete()
            else:
                # Sync result - complete immediately
                complete_activity(result)
                return result

        except Exception as e:
            complete_activity(None, str(e))
            raise

    def _serialize_input(self, payload: Any) -> Dict[str, Any]:
        """Serialize input for display (truncate large values)."""
        return self._safe_serialize_display(payload, max_str_len=200)

    def _serialize_output(self, result: Any) -> Dict[str, Any]:
        """Serialize output for display."""
        return self._safe_serialize_display(result, max_str_len=100)

    def _safe_serialize_display(self, value: Any, max_depth: int = 3, max_str_len: int = 200) -> Any:
        """Safely serialize value for display, handling complex SDK types.

        Args:
            value: Value to serialize
            max_depth: Maximum recursion depth
            max_str_len: Maximum string length before truncation

        Returns:
            JSON-serializable representation suitable for display
        """
        if max_depth <= 0:
            return str(value)[:max_str_len]

        # Handle None and primitives
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            if len(value) > max_str_len:
                return f"<{len(value)} chars>"
            return value

        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            if len(value) > 10:
                return f"<{len(value)} items>"
            return [self._safe_serialize_display(v, max_depth - 1, max_str_len) for v in value]

        # Handle dicts
        if isinstance(value, dict):
            result = {}
            for k, v in list(value.items())[:10]:
                result[str(k)] = self._safe_serialize_display(v, max_depth - 1, max_str_len)
            return result

        # Handle Pydantic models
        if hasattr(value, 'model_dump'):
            try:
                return self._safe_serialize_display(value.model_dump(), max_depth - 1, max_str_len)
            except Exception:
                pass

        # Handle dataclasses
        if hasattr(value, '__dataclass_fields__'):
            try:
                from dataclasses import asdict
                return self._safe_serialize_display(asdict(value), max_depth - 1, max_str_len)
            except Exception:
                pass

        # Handle objects with __dict__
        if hasattr(value, '__dict__'):
            try:
                obj_dict = {}
                for k, v in value.__dict__.items():
                    if not k.startswith('_'):
                        obj_dict[k] = self._safe_serialize_display(v, max_depth - 1, max_str_len)
                return {"__type__": type(value).__name__, **obj_dict}
            except Exception:
                pass

        # Fallback
        return {"type": type(value).__name__}

    def _serialize_full(self, value: Any, max_depth: int = 5) -> Any:
        """Serialize value without truncation (for full capture mode).

        Args:
            value: Value to serialize
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            JSON-serializable representation
        """
        if max_depth <= 0:
            return str(value)[:1000]

        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value

        if isinstance(value, (list, tuple)):
            return [self._serialize_full(v, max_depth - 1) for v in value[:100]]

        if isinstance(value, dict):
            return {
                str(k): self._serialize_full(v, max_depth - 1)
                for k, v in list(value.items())[:50]
            }

        # Handle Pydantic models
        if hasattr(value, 'model_dump'):
            try:
                return self._serialize_full(value.model_dump(), max_depth - 1)
            except Exception:
                pass

        # Handle dataclasses
        if hasattr(value, '__dataclass_fields__'):
            try:
                from dataclasses import asdict
                return self._serialize_full(asdict(value), max_depth - 1)
            except Exception:
                pass

        # Handle objects with __dict__
        if hasattr(value, '__dict__'):
            try:
                obj_dict = {}
                for k, v in value.__dict__.items():
                    if not k.startswith('_'):
                        obj_dict[k] = self._serialize_full(v, max_depth - 1)
                return {"__type__": type(value).__name__, **obj_dict}
            except Exception:
                pass

        # Fallback
        return str(value)[:1000]

    def _calculate_duration(self, start: str, end: str) -> int:
        """Calculate duration in milliseconds."""
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return int((end_dt - start_dt).total_seconds() * 1000)
        except Exception:
            return 0


# =============================================================================
# Async Workflow Context (PR #827 Pattern)
# =============================================================================


@dataclass
class AsyncWorkflowContext:
    """Async context for workflow operations following PR #827 pattern.

    Provides methods for durable workflow operations:
    - call_activity: Durably call an activity function
    - when_all: Wait for multiple tasks
    - wait_for_external_event: Wait for external signals

    This mirrors the DaprWorkflowContext from PR #827's async workflow support.

    Usage:
        @async_workflow("my_workflow")
        async def my_workflow(ctx: AsyncWorkflowContext, input: dict):
            result1 = await ctx.call_activity(activity1, {"data": "value"})
            result2 = await ctx.call_activity(activity2, result1)
            return result2
    """
    workflow_id: str
    instance_id: str
    metadata: Dict[str, str] = field(default_factory=dict)
    _dapr_client: Optional[DaprClient] = field(default=None, repr=False)
    _activity_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = self.workflow_id

    async def call_activity(
        self,
        activity_fn: Callable,
        input: Any,
        activity_id: Optional[str] = None
    ) -> Any:
        """Durably call an activity function.

        If the activity has already been executed (during replay),
        returns the cached result. Otherwise executes the activity
        and caches the result.

        This mirrors PR #827's async workflow call_activity behavior.

        Args:
            activity_fn: The activity function to call
            input: Input to pass to the activity
            activity_id: Optional ID for the activity (for replay matching)

        Returns:
            The activity result
        """
        call_id = activity_id or str(uuid.uuid4())

        # Check for cached result (replay)
        if call_id in self._activity_results:
            return self._activity_results[call_id]

        # Check state store for persisted result
        stored_result = await self._get_activity_result(call_id)
        if stored_result is not None:
            self._activity_results[call_id] = stored_result
            return stored_result

        # Execute activity
        if asyncio.iscoroutinefunction(activity_fn):
            result = await activity_fn(input)
        else:
            result = activity_fn(input)

        # Cache and persist result
        self._activity_results[call_id] = result
        await self._save_activity_result(call_id, result)

        return result

    async def when_all(self, *tasks) -> List[Any]:
        """Wait for all tasks to complete (PR #827 pattern).

        Args:
            *tasks: Coroutines or tasks to wait for

        Returns:
            List of results in the same order as inputs
        """
        return await asyncio.gather(*tasks)

    async def when_any(self, *tasks) -> Any:
        """Wait for any task to complete.

        Args:
            *tasks: Coroutines or tasks to wait for

        Returns:
            Result of the first completed task
        """
        done, pending = await asyncio.wait(
            [asyncio.ensure_future(t) for t in tasks],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        # Return result of first completed
        return done.pop().result()

    async def wait_for_external_event(
        self,
        event_name: str,
        timeout: Optional[timedelta] = None
    ) -> Any:
        """Wait for an external event (approval, callback, etc.).

        This mirrors PR #827's wait_for_external_event for human approval
        and external signal patterns.

        Args:
            event_name: Name of the event to wait for
            timeout: Maximum time to wait (default: no timeout)

        Returns:
            Event data when received

        Raises:
            TimeoutError: If timeout is exceeded
        """
        event_key = f"event-{self.workflow_id}-{event_name}"
        poll_interval = 1.0  # seconds
        elapsed = 0.0

        timeout_seconds = timeout.total_seconds() if timeout else float('inf')

        while elapsed < timeout_seconds:
            # Check for event in state store
            try:
                with DaprClient() as client:
                    data = client.get_state("statestore", event_key)
                    if data.data:
                        event_data = json.loads(data.data.decode('utf-8'))
                        # Clear the event
                        client.delete_state("statestore", event_key)
                        return event_data
            except Exception as e:
                logger.warning(f"Error polling for event: {e}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Timeout waiting for event: {event_name}")

    async def _get_activity_result(self, call_id: str) -> Optional[Any]:
        """Get cached activity result from state store."""
        key = f"activity-{self.workflow_id}-{call_id}"
        try:
            with DaprClient() as client:
                data = client.get_state("statestore", key)
                if data.data:
                    return json.loads(data.data.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to get activity result: {e}")
        return None

    async def _save_activity_result(self, call_id: str, result: Any) -> None:
        """Save activity result to state store."""
        key = f"activity-{self.workflow_id}-{call_id}"
        try:
            with DaprClient() as client:
                client.save_state("statestore", key, json.dumps(result))
        except Exception as e:
            logger.warning(f"Failed to save activity result: {e}")


def async_workflow(name: str):
    """Decorator for async workflows following PR #827 pattern.

    Usage:
        @async_workflow("my_planning_workflow")
        async def plan_implementation(ctx: AsyncWorkflowContext, input: dict):
            # Workflow logic here
            return result
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(ctx: AsyncWorkflowContext, input: Any):
            return await fn(ctx, input)

        wrapper._workflow_name = name
        wrapper._is_workflow = True
        return wrapper

    return decorator


# =============================================================================
# Durable Agent Runner
# =============================================================================


class DurableAgentRunner:
    """Wraps OpenAI Agents SDK Runner with Dapr durability.

    This runner wraps all tool calls with the interceptor chain,
    providing durability, activity tracking, and metadata propagation.

    Uses the interceptor patterns from PR #827:
    - compose_tool_chain for building the interceptor chain
    - ExecuteToolRequest for metadata envelope
    - Context variables for workflow state

    Features:
    - Tool calls recorded to state store for replay/recovery
    - Activity tracking for ai-chatbot visualization
    - Metadata envelope propagation through tool calls
    - Session state persistence for recovery

    When PR #827 is merged, migration path:
    1. Replace BaseToolInterceptor with BaseRuntimeInterceptor
    2. Replace ExecuteToolRequest with ExecuteActivityRequest
    3. Use dapr.ext.workflow.interceptors directly

    Usage:
        from agents import Agent

        agent = Agent(
            name="Planner",
            instructions="...",
            tools=[tool1, tool2],
        )

        runner = DurableAgentRunner(
            agent=agent,
            interceptors=[CustomInterceptor()],
            state_store="statestore",
        )

        result = await runner.run(
            workflow_id="wf-123",
            input="Create a plan for...",
        )
    """

    def __init__(
        self,
        agent: Any,  # Agent from openai-agents
        interceptors: Optional[List[BaseToolInterceptor]] = None,
        state_store: str = "statestore",
        activity_update_callback: Optional[Callable[[str, List[Dict]], None]] = None,
        run_hooks: Optional[Any] = None,  # RunHooks for LLM tracking
    ):
        """Initialize the durable runner.

        Args:
            agent: OpenAI Agents SDK Agent instance
            interceptors: Additional interceptors to add to the chain
            state_store: Dapr state store name for durability
            activity_update_callback: Callback for persisting activities
            run_hooks: Optional RunHooks for LLM call tracking (usage capture)
        """
        self.agent = agent
        self.state_store = state_store
        self.run_hooks = run_hooks
        self._interceptors: List[BaseToolInterceptor] = []

        # Add default interceptors (PR #827 pattern)
        self._interceptors.append(DurabilityInterceptor(state_store))
        self._interceptors.append(ActivityTrackingInterceptor(activity_update_callback))

        # Add custom interceptors
        if interceptors:
            self._interceptors.extend(interceptors)

    async def run(
        self,
        workflow_id: str,
        input: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> 'DurableRunResult':
        """Run the agent with durability and interceptors.

        Args:
            workflow_id: Unique identifier for this workflow
            input: User input/prompt for the agent
            metadata: Additional metadata to propagate (via envelope pattern)

        Returns:
            DurableRunResult containing the SDK result plus context data
        """
        from agents import Runner

        # Set up workflow context (PR #827 pattern)
        ctx = WorkflowExecutionContext(
            workflow_id=workflow_id,
            metadata=metadata or {},
            workflow_name=self.agent.name if hasattr(self.agent, 'name') else "Planner",
        )
        token = set_workflow_context(ctx)

        try:
            # Check for existing session (recovery)
            session_state = self._load_session(workflow_id)

            # Wrap tools with interceptor chain (using compose_tool_chain)
            wrapped_tools = [
                self._wrap_tool(tool) for tool in self.agent.tools
            ]

            # Create agent with wrapped tools (preserving other attributes)
            from agents import Agent
            durable_agent = Agent(
                name=self.agent.name,
                instructions=self.agent.instructions,
                model=self.agent.model,
                tools=wrapped_tools,
            )

            # Run the agent with optional hooks for LLM tracking
            logger.info(f"Starting DurableAgentRunner for workflow {workflow_id}")
            run_kwargs = {
                "starting_agent": durable_agent,
                "input": input,
            }

            # Add hooks if provided (for LLM usage tracking)
            if self.run_hooks:
                logger.info(f"Adding RunHooks to Runner.run: {type(self.run_hooks).__name__}")
                run_kwargs["hooks"] = self.run_hooks

            try:
                result = await Runner.run(**run_kwargs)
            except TypeError as e:
                # If hooks cause issues, try without them
                if self.run_hooks and "awaitable" in str(e).lower():
                    logger.warning(f"Hooks caused error, retrying without: {e}")
                    run_kwargs.pop("hooks", None)
                    result = await Runner.run(**run_kwargs)
                else:
                    raise
            logger.info(f"DurableAgentRunner completed for workflow {workflow_id}")

            # Save session for recovery
            self._save_session(workflow_id, result)

            # Capture context data before cleanup
            logger.info(f"Capturing context: tasks={len(ctx.tasks)}, activities={len(ctx.activities)}, usage={ctx.usage}")
            durable_result = DurableRunResult(
                result=result,
                tasks=ctx.tasks.copy(),
                activities=ctx.activities.copy(),
                usage=ctx.usage.copy(),
                trace_metadata=ctx.get_trace_metadata(),
                llm_call_count=ctx.llm_call_count,
            )

            return durable_result

        finally:
            # Clean up context (PR #827 pattern)
            _workflow_context.reset(token)

    def _wrap_tool(self, tool: Any) -> Any:
        """Wrap a tool function with the interceptor chain.

        Uses compose_tool_chain to build the interceptor chain,
        matching PR #827's compose_runtime_chain pattern.

        The OpenAI Agents SDK invokes tools via on_invoke_tool(ctx, input_dict) where:
        - ctx: RunContextWrapper with usage stats and context
        - input_dict: Parsed JSON arguments for the tool

        We extract just the tool arguments for logging/tracking.

        Args:
            tool: FunctionTool from openai-agents SDK

        Returns:
            New FunctionTool with wrapped function
        """
        from agents import function_tool
        from agents.tool import FunctionTool

        is_function_tool = isinstance(tool, FunctionTool)

        # Get tool name
        tool_name = getattr(tool, 'name', None) or 'unknown'

        # Build the interceptor chain (PR #827 pattern)
        # Terminal executes the original on_invoke_tool which is async
        original_on_invoke = tool.on_invoke_tool if is_function_tool else tool

        def terminal(request: ExecuteToolRequest) -> Any:
            """Terminal handler that executes the actual tool function."""
            # Call the original on_invoke_tool with SDK context and parsed args
            args = request.input.get("_args", ())
            kwargs = request.input.get("_kwargs", {})
            return original_on_invoke(*args, **kwargs)

        chain_fn = compose_tool_chain(self._interceptors, terminal)

        # Create async wrapper (FunctionTool.on_invoke_tool is always async)
        @functools.wraps(original_on_invoke)
        async def wrapped_on_invoke(ctx, input_dict):
            """Wrapped tool handler with interceptor chain."""
            wf_ctx = get_workflow_context()

            # Extract tool arguments from input_dict (the parsed JSON args)
            # The SDK passes tool arguments as a dict, e.g., {"path": "."} for list_directory
            # Handle cases where input_dict might be None, a string (JSON), or dict
            if input_dict is None:
                tool_args = {}
            elif isinstance(input_dict, dict):
                tool_args = input_dict.copy()  # Copy to avoid mutation
            elif isinstance(input_dict, str):
                # SDK sometimes passes JSON string - try to parse it
                try:
                    import json
                    tool_args = json.loads(input_dict)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {"value": input_dict}
            else:
                # If it's a primitive, wrap it
                tool_args = {"value": input_dict}

            # Log for debugging
            logger.debug(f"Tool {tool_name} called with args: {tool_args}")

            request = ExecuteToolRequest(
                tool_name=tool_name,
                input={
                    # Store tool args under a clear key so interceptor can find them
                    "_tool_args": tool_args,
                    "_args": (ctx, input_dict),  # Original args for terminal
                    "_kwargs": {},
                },
                metadata=wf_ctx.metadata if wf_ctx else None,
                call_id=str(uuid.uuid4()),
                workflow_id=wf_ctx.workflow_id if wf_ctx else None,
            )

            # Execute through interceptor chain
            result = chain_fn(request)

            # The terminal returns a coroutine from on_invoke_tool, await it
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Return new FunctionTool with wrapped on_invoke_tool
        if is_function_tool:
            return FunctionTool(
                name=tool.name,
                description=tool.description,
                params_json_schema=tool.params_json_schema,
                on_invoke_tool=wrapped_on_invoke,
            )

        # Fallback: try to wrap as function_tool
        return function_tool(wrapped_on_invoke)

    def _extract_tool_args(self, args: tuple, kwargs: dict) -> dict:
        """Extract tool arguments from SDK call pattern.

        The SDK calls on_invoke_tool(ctx, input_dict) where:
        - ctx is RunContextWrapper (has 'usage', 'context' attributes)
        - input_dict is the parsed tool arguments

        Args:
            args: Positional arguments from the SDK call
            kwargs: Keyword arguments

        Returns:
            Dictionary of tool arguments (excluding SDK context)
        """
        tool_args = {}

        if len(args) >= 2:
            first_arg = args[0]
            second_arg = args[1]

            # Check if first arg is SDK context
            if hasattr(first_arg, 'usage') or hasattr(first_arg, 'context'):
                tool_args = second_arg if isinstance(second_arg, dict) else {}
            else:
                tool_args = {"arg_0": args[0], "arg_1": args[1]}
        elif len(args) == 1:
            first_arg = args[0]
            if hasattr(first_arg, 'usage'):
                tool_args = kwargs
            elif isinstance(first_arg, dict):
                tool_args = first_arg
            else:
                tool_args = {"arg": first_arg}
        else:
            tool_args = kwargs

        return tool_args

    def _load_session(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load session state from state store for recovery."""
        key = f"session-{workflow_id}"
        try:
            with DaprClient() as client:
                data = client.get_state(self.state_store, key)
                if data.data:
                    return json.loads(data.data.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
        return None

    def _save_session(self, workflow_id: str, result: Any) -> None:
        """Save session state to state store for recovery."""
        key = f"session-{workflow_id}"
        try:
            session = {
                "workflow_id": workflow_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "final_output_type": type(result.final_output).__name__
                    if hasattr(result, 'final_output') else "unknown",
            }
            with DaprClient() as client:
                client.save_state(self.state_store, key, json.dumps(session))
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def add_interceptor(self, interceptor: BaseToolInterceptor) -> 'DurableAgentRunner':
        """Add an interceptor to the chain. Returns self for fluent API."""
        self._interceptors.append(interceptor)
        return self

    def add_interceptor_first(self, interceptor: BaseToolInterceptor) -> 'DurableAgentRunner':
        """Add an interceptor at the beginning of the chain."""
        self._interceptors.insert(0, interceptor)
        return self


# =============================================================================
# Helper Functions
# =============================================================================


def create_durable_agent(
    name: str,
    instructions: str,
    model: str,
    tools: List[Any],
    state_store: str = "statestore",
    activity_callback: Optional[Callable] = None,
) -> DurableAgentRunner:
    """Helper to create a DurableAgentRunner.

    Args:
        name: Agent name
        instructions: Agent instructions/system prompt
        model: Model to use (e.g., "gpt-4o")
        tools: List of tool functions
        state_store: Dapr state store name
        activity_callback: Callback for persisting activities

    Returns:
        Configured DurableAgentRunner
    """
    from agents import Agent

    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
    )

    return DurableAgentRunner(
        agent=agent,
        state_store=state_store,
        activity_update_callback=activity_callback,
    )


async def send_external_event(
    workflow_id: str,
    event_name: str,
    data: Any,
    state_store: str = "statestore",
) -> bool:
    """Send an external event to a waiting workflow.

    This is the counterpart to AsyncWorkflowContext.wait_for_external_event(),
    allowing external systems to signal workflows (e.g., human approval).

    Args:
        workflow_id: The workflow to send the event to
        event_name: Name of the event
        data: Event data
        state_store: Dapr state store name

    Returns:
        True if successful
    """
    event_key = f"event-{workflow_id}-{event_name}"
    try:
        with DaprClient() as client:
            client.save_state(state_store, event_key, json.dumps(data))
        return True
    except Exception as e:
        logger.error(f"Failed to send event: {e}")
        return False
