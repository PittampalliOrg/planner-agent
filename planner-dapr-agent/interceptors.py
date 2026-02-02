"""Interceptor framework for OpenAI Agents SDK with Dapr durability.

This module implements the interceptor pattern from dapr/python-sdk PR #827:
https://github.com/dapr/python-sdk/pull/827

The API is designed to be compatible with PR #827 so that when the PR is merged,
this code can be easily migrated to use the official dapr-ext-workflow interceptors.

Key patterns from PR #827:
- Request dataclasses with metadata envelope (__dapr_meta__, __dapr_payload__)
- Protocol-based interceptor interfaces (ClientInterceptor, RuntimeInterceptor, WorkflowOutboundInterceptor)
- Base classes with default pass-through implementations
- Chain composition functions (compose_client_chain, compose_runtime_chain, etc.)
- Metadata propagation across workflow boundaries

Adapted for OpenAI Agents SDK:
- ExecuteToolRequest instead of ExecuteActivityRequest (tools are like activities)
- Tool interceptor chain wraps tool function calls
- Same metadata envelope pattern for propagating context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar
import contextvars
import functools
import logging

logger = logging.getLogger(__name__)

# Type variables for generic interceptor payload typing (from PR #827)
TInput = TypeVar('TInput')
TWorkflowInput = TypeVar('TWorkflowInput')
TActivityInput = TypeVar('TActivityInput')
TToolInput = TypeVar('TToolInput')


# =============================================================================
# Metadata Envelope Pattern (PR #827)
# =============================================================================
# "metadata" is a durable, string-only map. It is serialized on the wire and
# propagates across boundaries (client → runtime → activity/child), surviving
# replays/retries. Use it when downstream components must observe the value.

_META_KEY = '__dapr_meta__'
_META_VERSION = 1
_PAYLOAD_KEY = '__dapr_payload__'


def wrap_payload_with_metadata(payload: Any, metadata: Dict[str, str] | None) -> Any:
    """Wrap payload in an envelope with metadata for durable persistence.

    The envelope structure allows metadata to be propagated across workflow boundaries
    (client → workflow → activity → child workflow) and persisted durably alongside the
    payload. This metadata survives replays, retries, and continues-as-new operations.

    Envelope structure (when metadata is present):
    ```python
    {
        '__dapr_meta__': {
            'v': 1,  # Version for future compatibility
            'metadata': {
                'tenant': 'acme-corp',
                'request_id': 'req-12345',
                # ... other metadata
            }
        },
        '__dapr_payload__': <original_payload>
    }
    ```

    Args:
        payload: The actual data to be passed (can be any JSON-serializable type)
        metadata: Optional string-only dictionary with cross-cutting concerns

    Returns:
        If metadata is provided and non-empty, returns the envelope dict.
        Otherwise returns payload unchanged (backward compatible).
    """
    if metadata:
        return {
            _META_KEY: {
                'v': _META_VERSION,
                'metadata': metadata,
            },
            _PAYLOAD_KEY: payload,
        }
    return payload


def unwrap_payload_with_metadata(obj: Any) -> tuple[Any, Dict[str, str] | None]:
    """Extract payload and metadata from envelope if present.

    This function is called by the runtime before executing workflows/activities to
    separate the user payload from the metadata.

    Args:
        obj: The potentially-wrapped input (may be an envelope or raw payload)

    Returns:
        A tuple of (payload, metadata_dict_or_none)
    """
    try:
        if isinstance(obj, dict) and _META_KEY in obj and _PAYLOAD_KEY in obj:
            meta = obj.get(_META_KEY) or {}
            md = meta.get('metadata') if isinstance(meta, dict) else None
            return obj.get(_PAYLOAD_KEY), md if isinstance(md, dict) else None
    except Exception:
        # Be robust: on any error, treat as raw payload
        pass
    return obj, None


# =============================================================================
# Request Dataclasses (PR #827 Pattern)
# =============================================================================


@dataclass
class ScheduleWorkflowRequest(Generic[TInput]):
    """Request to schedule a new workflow (PR #827 compatible).

    Used by ClientInterceptor.schedule_new_workflow().
    """
    workflow_name: str
    input: TInput
    instance_id: str | None = None
    start_at: Any | None = None
    reuse_id_policy: Any | None = None
    # Durable context serialized and propagated across boundaries
    metadata: Dict[str, str] | None = None


@dataclass
class ExecuteWorkflowRequest(Generic[TInput]):
    """Request to execute a workflow (PR #827 compatible).

    Used by RuntimeInterceptor.execute_workflow().
    """
    ctx: Any  # WorkflowContext
    input: TInput
    # Durable metadata (runtime chain only; not injected into user code)
    metadata: Dict[str, str] | None = None


@dataclass
class ExecuteActivityRequest(Generic[TInput]):
    """Request to execute an activity (PR #827 compatible).

    Used by RuntimeInterceptor.execute_activity().
    """
    ctx: Any  # WorkflowActivityContext
    input: TInput
    # Durable metadata (runtime chain only; not injected into user code)
    metadata: Dict[str, str] | None = None


@dataclass
class CallActivityRequest(Generic[TInput]):
    """Request to call an activity from within a workflow (PR #827 compatible).

    Used by WorkflowOutboundInterceptor.call_activity().
    """
    activity_name: str
    input: TInput
    retry_policy: Any | None = None
    app_id: str | None = None
    # Optional workflow context for outbound calls made inside workflows
    workflow_ctx: Any | None = None
    # Durable context serialized and propagated across boundaries
    metadata: Dict[str, str] | None = None


@dataclass
class CallChildWorkflowRequest(Generic[TInput]):
    """Request to call a child workflow (PR #827 compatible).

    Used by WorkflowOutboundInterceptor.call_child_workflow().
    """
    workflow_name: str
    input: TInput
    instance_id: str | None = None
    retry_policy: Any | None = None
    app_id: str | None = None
    workflow_ctx: Any | None = None
    metadata: Dict[str, str] | None = None


@dataclass
class ContinueAsNewRequest(Generic[TInput]):
    """Request to continue workflow as new (PR #827 compatible).

    Used by WorkflowOutboundInterceptor.continue_as_new().
    """
    input: TInput
    workflow_ctx: Any | None = None
    metadata: Dict[str, str] | None = None


# =============================================================================
# OpenAI Agents SDK Adapted Request (Tool Execution)
# =============================================================================


@dataclass
class ExecuteToolRequest(Generic[TInput]):
    """Request to execute a tool in OpenAI Agents SDK.

    This is the adapted version of ExecuteActivityRequest for tools.
    Tools in OpenAI Agents SDK are analogous to activities in Dapr Workflows.

    Attributes:
        tool_name: Name of the tool being called
        input: Input arguments to the tool
        metadata: Durable metadata for propagation
        call_id: Unique identifier for this call (for replay matching)
        workflow_id: ID of the containing workflow
    """
    tool_name: str
    input: TInput
    metadata: Dict[str, str] | None = None
    call_id: str | None = None
    workflow_id: str | None = None

    def with_metadata(self, **kwargs) -> 'ExecuteToolRequest[TInput]':
        """Create a new request with additional metadata."""
        new_meta = {**(self.metadata or {}), **kwargs}
        return ExecuteToolRequest(
            tool_name=self.tool_name,
            input=self.input,
            metadata=new_meta,
            call_id=self.call_id,
            workflow_id=self.workflow_id,
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default."""
        return (self.metadata or {}).get(key, default)


# =============================================================================
# Interceptor Protocols (PR #827 Pattern)
# =============================================================================


class ClientInterceptor(Protocol, Generic[TInput]):
    """Protocol for client-side interceptors (PR #827 compatible).

    Client interceptors wrap outbound calls from the client, such as
    scheduling new workflows.
    """
    def schedule_new_workflow(
        self,
        input: ScheduleWorkflowRequest[TInput],
        next: Callable[[ScheduleWorkflowRequest[TInput]], Any],
    ) -> Any: ...


class RuntimeInterceptor(Protocol, Generic[TWorkflowInput, TActivityInput]):
    """Protocol for runtime-side interceptors (PR #827 compatible).

    Runtime interceptors wrap the execution of workflows and activities.
    """
    def execute_workflow(
        self,
        input: ExecuteWorkflowRequest[TWorkflowInput],
        next: Callable[[ExecuteWorkflowRequest[TWorkflowInput]], Any],
    ) -> Any: ...

    def execute_activity(
        self,
        input: ExecuteActivityRequest[TActivityInput],
        next: Callable[[ExecuteActivityRequest[TActivityInput]], Any],
    ) -> Any: ...


class WorkflowOutboundInterceptor(Protocol, Generic[TWorkflowInput, TActivityInput]):
    """Protocol for workflow outbound interceptors (PR #827 compatible).

    Outbound interceptors wrap calls made from within workflows, such as
    calling activities or child workflows.
    """
    def call_child_workflow(
        self,
        input: CallChildWorkflowRequest[TWorkflowInput],
        next: Callable[[CallChildWorkflowRequest[TWorkflowInput]], Any],
    ) -> Any: ...

    def continue_as_new(
        self,
        input: ContinueAsNewRequest[TWorkflowInput],
        next: Callable[[ContinueAsNewRequest[TWorkflowInput]], Any],
    ) -> Any: ...

    def call_activity(
        self,
        input: CallActivityRequest[TActivityInput],
        next: Callable[[CallActivityRequest[TActivityInput]], Any],
    ) -> Any: ...


class ToolInterceptor(Protocol, Generic[TToolInput]):
    """Protocol for tool interceptors (OpenAI Agents SDK adapted).

    Tool interceptors wrap tool function calls in the OpenAI Agents SDK.
    This is the adapted version of RuntimeInterceptor for tools.
    """
    def execute_tool(
        self,
        input: ExecuteToolRequest[TToolInput],
        next: Callable[[ExecuteToolRequest[TToolInput]], Any],
    ) -> Any: ...


# =============================================================================
# Base Interceptor Classes (PR #827 Pattern)
# =============================================================================


class BaseClientInterceptor(Generic[TInput]):
    """Subclass this to get method name completion and safe defaults.

    Override any of the methods to customize behavior. By default, these
    methods simply call `next` unchanged.
    """

    def schedule_new_workflow(
        self,
        input: ScheduleWorkflowRequest[TInput],
        next: Callable[[ScheduleWorkflowRequest[TInput]], Any],
    ) -> Any:
        return next(input)


class BaseRuntimeInterceptor(Generic[TWorkflowInput, TActivityInput]):
    """Subclass this to get method name completion and safe defaults."""

    def execute_workflow(
        self,
        input: ExecuteWorkflowRequest[TWorkflowInput],
        next: Callable[[ExecuteWorkflowRequest[TWorkflowInput]], Any],
    ) -> Any:
        return next(input)

    def execute_activity(
        self,
        input: ExecuteActivityRequest[TActivityInput],
        next: Callable[[ExecuteActivityRequest[TActivityInput]], Any],
    ) -> Any:
        return next(input)


class BaseWorkflowOutboundInterceptor(Generic[TWorkflowInput, TActivityInput]):
    """Subclass this to get method name completion and safe defaults."""

    def call_child_workflow(
        self,
        input: CallChildWorkflowRequest[TWorkflowInput],
        next: Callable[[CallChildWorkflowRequest[TWorkflowInput]], Any],
    ) -> Any:
        return next(input)

    def continue_as_new(
        self,
        input: ContinueAsNewRequest[TWorkflowInput],
        next: Callable[[ContinueAsNewRequest[TWorkflowInput]], Any],
    ) -> Any:
        return next(input)

    def call_activity(
        self,
        input: CallActivityRequest[TActivityInput],
        next: Callable[[CallActivityRequest[TActivityInput]], Any],
    ) -> Any:
        return next(input)


class BaseToolInterceptor(Generic[TToolInput]):
    """Subclass this for tool interception in OpenAI Agents SDK.

    This is the adapted version of BaseRuntimeInterceptor for tools.
    """

    def execute_tool(
        self,
        input: ExecuteToolRequest[TToolInput],
        next: Callable[[ExecuteToolRequest[TToolInput]], Any],
    ) -> Any:
        return next(input)


# =============================================================================
# Chain Composition Functions (PR #827 Pattern)
# =============================================================================


def compose_client_chain(
    interceptors: List[ClientInterceptor],
    terminal: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Compose client interceptors into a single callable (PR #827 compatible).

    Interceptors are applied in list order; each receives a ``next``.
    The ``terminal`` callable is the final handler invoked after all interceptors.
    """
    next_fn = terminal
    for icpt in reversed(interceptors or []):

        def make_next(curr_icpt: ClientInterceptor, nxt: Callable[[Any], Any]):
            def runner(input: Any) -> Any:
                if isinstance(input, ScheduleWorkflowRequest):
                    return curr_icpt.schedule_new_workflow(input, nxt)
                return nxt(input)

            return runner

        next_fn = make_next(icpt, next_fn)
    return next_fn


def compose_runtime_chain(
    interceptors: List[RuntimeInterceptor],
    final_handler: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Compose runtime interceptors into a single callable (PR #827 compatible).

    The ``final_handler`` callable is the final handler invoked after all interceptors.
    """
    next_fn = final_handler
    for icpt in reversed(interceptors or []):

        def make_next(curr_icpt: RuntimeInterceptor, nxt: Callable[[Any], Any]):
            def runner(input: Any) -> Any:
                if isinstance(input, ExecuteWorkflowRequest):
                    return curr_icpt.execute_workflow(input, nxt)
                if isinstance(input, ExecuteActivityRequest):
                    return curr_icpt.execute_activity(input, nxt)
                return nxt(input)

            return runner

        next_fn = make_next(icpt, next_fn)
    return next_fn


def compose_workflow_outbound_chain(
    interceptors: List[WorkflowOutboundInterceptor],
    terminal: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """Compose workflow outbound interceptors into a single callable (PR #827 compatible).

    Interceptors are applied in list order; each receives a ``next``.
    """
    next_fn = terminal
    for icpt in reversed(interceptors or []):

        def make_next(curr_icpt: WorkflowOutboundInterceptor, nxt: Callable[[Any], Any]):
            def runner(input: Any) -> Any:
                if isinstance(input, CallActivityRequest):
                    return curr_icpt.call_activity(input, nxt)
                if isinstance(input, CallChildWorkflowRequest):
                    return curr_icpt.call_child_workflow(input, nxt)
                if isinstance(input, ContinueAsNewRequest):
                    return curr_icpt.continue_as_new(input, nxt)
                return nxt(input)

            return runner

        next_fn = make_next(icpt, next_fn)
    return next_fn


def compose_tool_chain(
    interceptors: List[ToolInterceptor],
    terminal: Callable[[ExecuteToolRequest], Any]
) -> Callable[[ExecuteToolRequest], Any]:
    """Compose tool interceptors into a single callable (OpenAI Agents SDK adapted).

    This is the adapted version of compose_runtime_chain for tools.
    """
    next_fn = terminal
    for icpt in reversed(interceptors or []):

        def make_next(curr_icpt: ToolInterceptor, nxt: Callable[[ExecuteToolRequest], Any]):
            def runner(input: ExecuteToolRequest) -> Any:
                return curr_icpt.execute_tool(input, nxt)

            return runner

        next_fn = make_next(icpt, next_fn)
    return next_fn


# =============================================================================
# Interceptor Chain Class (Alternative API)
# =============================================================================


class InterceptorChain(Generic[TInput]):
    """Chain of interceptors applied in order.

    This provides an alternative, more object-oriented API to the compose_*_chain
    functions. Use whichever style fits your codebase better.

    Interceptors are executed in the order they were added:
    - Request flows forward through the chain (first to last)
    - Response flows backward through the chain (last to first)

    Example:
        chain = InterceptorChain()
        chain.add(LoggingInterceptor())
        chain.add(DurabilityInterceptor())
        chain.add(ActivityTrackingInterceptor())

        result = chain.execute(request, actual_function)
    """

    def __init__(self):
        self._interceptors: List[BaseToolInterceptor] = []

    def add(self, interceptor: BaseToolInterceptor) -> 'InterceptorChain':
        """Add an interceptor to the chain. Returns self for fluent API."""
        self._interceptors.append(interceptor)
        return self

    def add_first(self, interceptor: BaseToolInterceptor) -> 'InterceptorChain':
        """Add an interceptor at the beginning of the chain."""
        self._interceptors.insert(0, interceptor)
        return self

    def remove(self, interceptor_type: type) -> bool:
        """Remove the first interceptor of the given type. Returns True if found."""
        for i, interceptor in enumerate(self._interceptors):
            if isinstance(interceptor, interceptor_type):
                self._interceptors.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all interceptors from the chain."""
        self._interceptors.clear()

    def execute(
        self,
        request: ExecuteToolRequest[TInput],
        final: Callable[[ExecuteToolRequest[TInput]], Any]
    ) -> Any:
        """Execute the interceptor chain, calling final function at the end."""
        chain_fn = compose_tool_chain(self._interceptors, final)
        return chain_fn(request)

    def __len__(self) -> int:
        return len(self._interceptors)

    def __iter__(self):
        return iter(self._interceptors)


# =============================================================================
# Generator Wrapper Pattern (PR #827)
# =============================================================================
#
# IMPORTANT: Generator wrappers for async workflows
# --------------------------------------------------
# When writing runtime interceptors that touch workflow execution, be careful with
# generator handling. If an interceptor obtains a workflow generator from user code
# it must not manually iterate it using a for-loop and yield the produced items.
# Doing so breaks send()/throw() propagation back into the inner generator.
#
# Best practices:
# - If the interceptor must wrap the generator, always use "yield from inner_gen"
#   so that send()/throw() are forwarded correctly.


class GeneratorWrapper(Generic[TInput]):
    """Preserves context during generator execution (PR #827 pattern).

    When using generators (like Dapr Workflow's replay mechanism),
    context can be lost across yield points. This wrapper ensures that context
    variables are restored before each iteration.
    """

    def __init__(self, gen: Any, context: Any, context_var: contextvars.ContextVar):
        self._gen = gen
        self._context = context
        self._context_var = context_var

    def __iter__(self):
        return self

    def __next__(self) -> TInput:
        """Get next value, restoring context first."""
        token = self._context_var.set(self._context)
        try:
            return next(self._gen)
        finally:
            self._context_var.reset(token)

    def send(self, value: Any) -> TInput:
        """Send value to generator, restoring context first."""
        token = self._context_var.set(self._context)
        try:
            return self._gen.send(value)
        finally:
            self._context_var.reset(token)

    def throw(self, exc_type, exc_val=None, exc_tb=None) -> TInput:
        """Throw exception into generator."""
        return self._gen.throw(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        """Close the generator."""
        self._gen.close()


def wrap_generator(context_var: contextvars.ContextVar):
    """Decorator to wrap generator functions with context preservation.

    Usage:
        _workflow_ctx = contextvars.ContextVar('workflow_ctx')

        @wrap_generator(_workflow_ctx)
        def my_workflow_generator():
            yield activity1()
            yield activity2()
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = context_var.get()
            gen = fn(*args, **kwargs)
            return GeneratorWrapper(gen, ctx, context_var)
        return wrapper
    return decorator


# =============================================================================
# Concrete Interceptor Implementations
# =============================================================================


class LoggingToolInterceptor(BaseToolInterceptor):
    """Interceptor that logs tool execution for debugging."""

    def __init__(self, log_level: int = logging.DEBUG, prefix: str = ""):
        self.log_level = log_level
        self.prefix = prefix

    def execute_tool(
        self,
        input: ExecuteToolRequest,
        next: Callable[[ExecuteToolRequest], Any],
    ) -> Any:
        call_id = input.call_id or 'unknown'
        tool_name = input.tool_name

        logger.log(
            self.log_level,
            f"{self.prefix}[{call_id}] -> {tool_name} input={_summarize(input.input)}"
        )

        try:
            result = next(input)
            logger.log(
                self.log_level,
                f"{self.prefix}[{call_id}] <- {tool_name} output={_summarize(result)}"
            )
            return result
        except Exception as e:
            logger.log(
                self.log_level,
                f"{self.prefix}[{call_id}] <- {tool_name} error={e}"
            )
            raise


class MetadataPropagationInterceptor(BaseToolInterceptor):
    """Interceptor that propagates metadata through the chain."""

    def __init__(self, default_metadata: Optional[Dict[str, str]] = None):
        self.default_metadata = default_metadata or {}

    def execute_tool(
        self,
        input: ExecuteToolRequest,
        next: Callable[[ExecuteToolRequest], Any],
    ) -> Any:
        # Merge default metadata (request metadata takes precedence)
        merged = {**self.default_metadata, **(input.metadata or {})}
        enriched = ExecuteToolRequest(
            tool_name=input.tool_name,
            input=input.input,
            metadata=merged,
            call_id=input.call_id,
            workflow_id=input.workflow_id,
        )
        return next(enriched)


class RetryToolInterceptor(BaseToolInterceptor):
    """Interceptor that retries failed tool calls with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    def execute_tool(
        self,
        input: ExecuteToolRequest,
        next: Callable[[ExecuteToolRequest], Any],
    ) -> Any:
        import time

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return next(input)
            except self.retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Tool {input.tool_name} retry {attempt + 1}/{self.max_retries} "
                        f"after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

        raise last_exception


# =============================================================================
# Helper Functions
# =============================================================================


def _summarize(value: Any, max_length: int = 100) -> str:
    """Create a summary of a value for logging."""
    if value is None:
        return "None"
    if isinstance(value, str):
        if len(value) > max_length:
            return f"str({len(value)} chars)"
        return repr(value)
    if isinstance(value, dict):
        return f"dict({len(value)} keys: {list(value.keys())[:3]})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}({len(value)} items)"
    return f"{type(value).__name__}"


def create_tool_request(
    tool_name: str,
    input: Any,
    workflow_id: Optional[str] = None,
    call_id: Optional[str] = None,
    **extra_metadata
) -> ExecuteToolRequest:
    """Helper to create an ExecuteToolRequest with common metadata."""
    import uuid

    metadata = {**extra_metadata}

    return ExecuteToolRequest(
        tool_name=tool_name,
        input=input,
        metadata=metadata if metadata else None,
        call_id=call_id or str(uuid.uuid4()),
        workflow_id=workflow_id,
    )
