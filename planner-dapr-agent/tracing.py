"""OpenAI-compatible tracing module for capturing LLM generation data.

This module provides span data classes that match OpenAI's tracing schema at
https://platform.openai.com/logs?api=traces

Key span types:
- TraceSpan: Base span with OpenAI-format IDs
- GenerationSpanData: LLM call details (input/output messages, model, tokens)
- AgentSpanData: Agent execution data (handoffs, tools, output_type)
- FunctionSpanData: Tool/function call data

The tracing can be integrated with:
1. OpenInference instrumentation for automatic OpenAI SDK tracing
2. Phoenix/Arize for LLM-specific observability
3. OTEL collectors for distributed tracing

Usage:
    from tracing import setup_tracing, GenerationSpanData

    # Initialize tracing at app startup
    tracer = setup_tracing(project_name="planner-dapr-agent")

    # Create generation spans when LLM calls occur
    span_data = GenerationSpanData(
        input=messages,
        output=response.messages,
        model="gpt-4o",
        model_config={"temperature": 0.7},
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
"""

from __future__ import annotations

import logging
import os
import secrets
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI-Compatible ID Generation
# =============================================================================


def generate_trace_id() -> str:
    """Generate a trace ID in OpenAI format: trace_<32_alphanumeric>.

    OpenAI uses base62-like IDs (lowercase letters + digits).
    """
    chars = string.ascii_lowercase + string.digits
    suffix = ''.join(secrets.choice(chars) for _ in range(32))
    return f"trace_{suffix}"


def generate_span_id() -> str:
    """Generate a span ID in OpenAI format: span_<26_alphanumeric>.

    OpenAI uses shorter IDs for spans than traces.
    """
    chars = string.ascii_lowercase + string.digits
    suffix = ''.join(secrets.choice(chars) for _ in range(26))
    return f"span_{suffix}"


def generate_group_id() -> str:
    """Generate a group ID for linking traces from the same conversation."""
    chars = string.ascii_lowercase + string.digits
    suffix = ''.join(secrets.choice(chars) for _ in range(24))
    return f"group_{suffix}"


# =============================================================================
# Base Span Data Class
# =============================================================================


@dataclass
class TraceSpan:
    """Base span matching OpenAI's span structure.

    OpenAI traces have a hierarchical structure:
    - Trace contains multiple spans
    - Each span has a unique span_id and optional parent_id
    - Spans have start/end timestamps and optional error info

    Attributes:
        span_id: Unique identifier in format span_<26_alphanumeric>
        trace_id: Parent trace ID in format trace_<32_alphanumeric>
        parent_id: Optional parent span ID for hierarchy
        started_at: ISO timestamp when span started
        ended_at: ISO timestamp when span ended
        error: Error info if span failed
        span_type: Type of span (generation, agent, function, custom)
    """
    span_id: str = field(default_factory=generate_span_id)
    trace_id: str = ""
    parent_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    error: Optional[str] = None
    span_type: str = "custom"

    def start(self) -> 'TraceSpan':
        """Mark span as started with current timestamp."""
        self.started_at = datetime.now(timezone.utc).isoformat()
        return self

    def end(self, error: Optional[str] = None) -> 'TraceSpan':
        """Mark span as ended with current timestamp."""
        self.ended_at = datetime.now(timezone.utc).isoformat()
        if error:
            self.error = error
        return self

    def duration_ms(self) -> Optional[int]:
        """Calculate duration in milliseconds."""
        if not self.started_at or not self.ended_at:
            return None
        try:
            start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.ended_at.replace("Z", "+00:00"))
            return int((end - start).total_seconds() * 1000)
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# =============================================================================
# Generation Span (LLM Calls)
# =============================================================================


@dataclass
class GenerationSpanData:
    """LLM call span matching OpenAI's GenerationSpanData.

    Captures complete information about a single LLM API call including:
    - Input messages sent to the model
    - Output messages received
    - Model identifier and configuration
    - Token usage statistics

    This data is essential for:
    - Cost tracking (token usage)
    - Debugging (input/output inspection)
    - Performance monitoring (latency correlation with token counts)

    Attributes:
        input: List of messages sent to the model
        output: List of response messages
        model: Model identifier (e.g., "gpt-4o", "claude-3-opus")
        model_config: Hyperparameters (temperature, max_tokens, etc.)
        usage: Token counts {input_tokens, output_tokens, total_tokens}
    """
    input: List[Dict[str, Any]] = field(default_factory=list)
    output: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)

    def add_input_message(self, role: str, content: str) -> None:
        """Add an input message."""
        self.input.append({"role": role, "content": content})

    def add_output_message(self, role: str, content: str, tool_calls: Optional[List] = None) -> None:
        """Add an output message."""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.output.append(msg)

    def set_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Set token usage statistics."""
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or (input_tokens + output_tokens),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# =============================================================================
# Agent Span
# =============================================================================


@dataclass
class AgentSpanData:
    """Agent span matching OpenAI's AgentSpanData.

    Captures information about an agent execution including:
    - Agent name and configuration
    - Available handoff targets (other agents)
    - Available tools
    - Output type

    Attributes:
        name: Agent name
        handoffs: List of agent names that can receive handoffs
        tools: List of available tool names
        output_type: Output type name (if structured output)
    """
    name: str = ""
    handoffs: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    output_type: Optional[str] = None

    @classmethod
    def from_agent(cls, agent: Any) -> 'AgentSpanData':
        """Create AgentSpanData from an OpenAI Agents SDK Agent."""
        name = getattr(agent, 'name', 'unknown')
        tools = []
        handoffs = []

        # Extract tool names
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                tool_name = getattr(tool, 'name', None) or getattr(tool, '__name__', 'unknown')
                tools.append(tool_name)

        # Extract handoff targets
        if hasattr(agent, 'handoffs') and agent.handoffs:
            for h in agent.handoffs:
                handoff_name = getattr(h, 'name', None) or str(h)
                handoffs.append(handoff_name)

        return cls(
            name=name,
            tools=tools,
            handoffs=handoffs,
            output_type=getattr(agent, 'output_type', None).__name__
                if hasattr(agent, 'output_type') and agent.output_type else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# =============================================================================
# Function/Tool Span
# =============================================================================


@dataclass
class FunctionSpanData:
    """Function/tool call span matching OpenAI's FunctionSpanData.

    Captures complete information about a tool/function call including:
    - Function name
    - Input arguments (full, not truncated)
    - Output result (full, not truncated)
    - MCP server metadata (if applicable)

    Attributes:
        name: Tool/function name
        input: Input arguments (can be any JSON-serializable type)
        output: Output result
        mcp_data: Optional MCP server metadata
    """
    name: str = ""
    input: Any = None
    output: Any = None
    mcp_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"name": self.name}
        if self.input is not None:
            result["input"] = self.input
        if self.output is not None:
            result["output"] = self.output
        if self.mcp_data:
            result["mcp_data"] = self.mcp_data
        return result


# =============================================================================
# Trace Context
# =============================================================================


@dataclass
class TraceContext:
    """Context for a complete trace, linking all spans.

    A trace represents a complete workflow execution with:
    - Unique trace_id
    - Workflow name
    - Group ID for linking related traces
    - Metadata
    - Collection of spans

    Attributes:
        trace_id: Unique trace identifier
        workflow_name: Logical workflow name
        group_id: Links traces from same conversation
        metadata: Arbitrary metadata
        spans: List of spans in this trace
    """
    trace_id: str = field(default_factory=generate_trace_id)
    workflow_name: str = ""
    group_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    spans: List[TraceSpan] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    def start(self) -> 'TraceContext':
        """Mark trace as started."""
        self.started_at = datetime.now(timezone.utc).isoformat()
        return self

    def end(self) -> 'TraceContext':
        """Mark trace as ended."""
        self.ended_at = datetime.now(timezone.utc).isoformat()
        return self

    def create_span(self, span_type: str = "custom", parent_id: Optional[str] = None) -> TraceSpan:
        """Create a new span in this trace."""
        span = TraceSpan(
            trace_id=self.trace_id,
            parent_id=parent_id,
            span_type=span_type,
        )
        self.spans.append(span)
        return span

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "workflow_name": self.workflow_name,
            "group_id": self.group_id,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# =============================================================================
# OpenInference / Phoenix Integration
# =============================================================================


def setup_tracing(
    project_name: str = "planner-dapr-agent",
    endpoint: Optional[str] = None,
    enable_openai_instrumentation: bool = True,
    trace_include_sensitive_data: bool = True,
) -> Optional[Any]:
    """Initialize tracing with optional OpenInference instrumentation.

    This function sets up:
    1. OpenTelemetry tracer provider
    2. OpenInference instrumentation for OpenAI SDK (if installed)
    3. Export to OTEL collector or Phoenix

    Args:
        project_name: Project name for grouping traces
        endpoint: OTEL endpoint (defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var)
        enable_openai_instrumentation: Auto-instrument OpenAI SDK calls
        trace_include_sensitive_data: Include full message content in traces

    Returns:
        TracerProvider if setup succeeds, None otherwise
    """
    try:
        # Try to import OpenInference packages
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        # Get endpoint from env if not provided
        endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://otel-collector.observability.svc.cluster.local:4317"
        )

        # Create resource with service name
        resource = Resource.create({
            "service.name": project_name,
            "service.version": "1.0.0",
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Try to set up OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"OTEL tracing enabled, exporting to {endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not available, traces won't be exported")

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Try to instrument OpenAI SDK
        if enable_openai_instrumentation:
            try:
                from openinference.instrumentation.openai import OpenAIInstrumentor
                OpenAIInstrumentor().instrument(tracer_provider=provider)
                logger.info("OpenAI SDK instrumentation enabled")
            except ImportError:
                logger.debug("OpenInference OpenAI instrumentation not available")

        return provider

    except ImportError as e:
        logger.debug(f"OpenTelemetry not available: {e}")
        return None


# =============================================================================
# LLM Tracing Interceptor Hook
# =============================================================================


class LLMTracingHook:
    """Hook for capturing LLM call data within agents.

    This class provides methods that can be called from RunHooks to capture
    generation span data. It stores captured data for later retrieval.

    Usage with OpenAI Agents SDK RunHooks:
        hook = LLMTracingHook(trace_include_sensitive_data=True)

        class MyHooks(RunHooks):
            def on_llm_start(self, context, agent, system_prompt, input_items):
                hook.on_llm_start(agent, system_prompt, input_items)

            def on_llm_end(self, context, agent, response):
                span_data = hook.on_llm_end(agent, response)
                # span_data now contains full GenerationSpanData
    """

    def __init__(self, trace_include_sensitive_data: bool = True):
        """Initialize the hook.

        Args:
            trace_include_sensitive_data: If True, capture full message content.
                If False, only capture metadata (for privacy).
        """
        self.trace_include_sensitive_data = trace_include_sensitive_data
        self._current_span: Optional[GenerationSpanData] = None
        self._spans: List[GenerationSpanData] = []

    def on_llm_start(
        self,
        agent: Any,
        system_prompt: Optional[str],
        input_items: List[Any],
    ) -> GenerationSpanData:
        """Called before LLM API call.

        Args:
            agent: The agent making the call
            system_prompt: System prompt if any
            input_items: Input messages/items

        Returns:
            New GenerationSpanData being populated
        """
        model = getattr(agent, 'model', 'unknown')
        model_config = {}

        # Try to extract model config
        if hasattr(agent, 'model_settings') and agent.model_settings:
            settings = agent.model_settings
            model_config = {
                "temperature": getattr(settings, 'temperature', None),
                "max_tokens": getattr(settings, 'max_tokens', None),
                "top_p": getattr(settings, 'top_p', None),
            }
            model_config = {k: v for k, v in model_config.items() if v is not None}

        self._current_span = GenerationSpanData(
            model=model,
            model_config=model_config,
        )

        if self.trace_include_sensitive_data:
            # Capture system prompt
            if system_prompt:
                self._current_span.add_input_message("system", system_prompt)

            # Capture input items
            for item in (input_items or []):
                if hasattr(item, 'role') and hasattr(item, 'content'):
                    self._current_span.add_input_message(
                        str(item.role),
                        str(item.content) if item.content else "",
                    )

        return self._current_span

    def on_llm_end(
        self,
        agent: Any,
        response: Any,
    ) -> GenerationSpanData:
        """Called after LLM API response.

        Args:
            agent: The agent that made the call
            response: The LLM response

        Returns:
            Completed GenerationSpanData
        """
        if not self._current_span:
            self._current_span = GenerationSpanData()

        # Extract usage
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self._current_span.set_usage(
                input_tokens=getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0),
                output_tokens=getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0),
                total_tokens=getattr(usage, 'total_tokens', None),
            )

        # Capture output if enabled
        if self.trace_include_sensitive_data:
            # Try to extract output messages
            if hasattr(response, 'output'):
                for item in response.output:
                    if hasattr(item, 'role') and hasattr(item, 'content'):
                        tool_calls = None
                        if hasattr(item, 'tool_calls'):
                            tool_calls = [
                                {"name": tc.name, "arguments": tc.arguments}
                                for tc in (item.tool_calls or [])
                            ]
                        self._current_span.add_output_message(
                            str(item.role),
                            str(item.content) if item.content else "",
                            tool_calls,
                        )

        # Store completed span
        completed = self._current_span
        self._spans.append(completed)
        self._current_span = None

        return completed

    def get_spans(self) -> List[GenerationSpanData]:
        """Get all captured generation spans."""
        return self._spans.copy()

    def clear_spans(self) -> None:
        """Clear captured spans."""
        self._spans.clear()
        self._current_span = None


# =============================================================================
# Convenience Exports
# =============================================================================


__all__ = [
    # ID generation
    "generate_trace_id",
    "generate_span_id",
    "generate_group_id",
    # Span data classes
    "TraceSpan",
    "GenerationSpanData",
    "AgentSpanData",
    "FunctionSpanData",
    "TraceContext",
    # Integration
    "setup_tracing",
    "LLMTracingHook",
]
