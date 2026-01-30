"""
Custom Anthropic LLM Client for Dapr Agents

This module provides an AnthropicChatClient that implements the dapr-agents
ChatClientBase interface while using the native Anthropic Python SDK.
This enables full tool/function calling support that the Dapr Conversation
API doesn't provide.

Usage:
    from anthropic_llm import AnthropicChatClient

    client = AnthropicChatClient(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Use with DurableAgent
    agent = DurableAgent(
        name="MyAgent",
        llm=client,
        tools=[...],
    )
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

# Import Anthropic SDK
try:
    import anthropic
    from anthropic.types import Message, ContentBlock, ToolUseBlock, TextBlock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    Message = None
    ContentBlock = None
    ToolUseBlock = None
    TextBlock = None

# Import dapr-agents base classes
try:
    from dapr_agents.llm.base import ChatClientBase, LLMClientBase
    from dapr_agents.types import AgentTool
    DAPR_AGENTS_AVAILABLE = True
except ImportError:
    DAPR_AGENTS_AVAILABLE = False
    ChatClientBase = object
    LLMClientBase = object
    AgentTool = None

T = TypeVar("T", bound=BaseModel)


class FunctionCall(BaseModel):
    """Function call details within a tool call."""
    name: str = ""
    arguments: str = ""

    @property
    def arguments_dict(self) -> Dict[str, Any]:
        """Parse arguments JSON string into a dict."""
        if not self.arguments:
            return {}
        try:
            return json.loads(self.arguments)
        except (json.JSONDecodeError, TypeError):
            return {}


class ToolCall(BaseModel):
    """Tool call object compatible with dapr-agents expectations."""
    id: str = ""
    type: str = "function"
    function: FunctionCall = Field(default_factory=FunctionCall)


class AssistantMessage(BaseModel):
    """
    Assistant message model for dapr-agents compatibility.

    dapr-agents expects get_message() to return a Pydantic model with model_dump(),
    not a plain dict. The Agent class also calls has_tool_calls() on the message.
    """
    role: str = "assistant"
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "allow"}

    def has_tool_calls(self) -> bool:
        """Check if the message contains tool calls."""
        return bool(self.tool_calls and len(self.tool_calls) > 0)

    def get_tool_calls(self) -> List[ToolCall]:
        """Get the list of tool calls as ToolCall objects."""
        if not self.tool_calls:
            return []

        result = []
        for tc in self.tool_calls:
            if isinstance(tc, ToolCall):
                result.append(tc)
            elif isinstance(tc, dict):
                # Convert dict to ToolCall object
                func_data = tc.get("function", {})
                result.append(ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=func_data.get("name", ""),
                        arguments=func_data.get("arguments", "{}"),
                    )
                ))
        return result


class LLMChatResponse(BaseModel):
    """Response from LLM chat completion."""
    content: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = Field(default_factory=dict)
    raw_response: Optional[Any] = None

    def get_message(self) -> AssistantMessage:
        """
        Get the response as an AssistantMessage model (for dapr-agents compatibility).

        dapr-agents DurableAgent.call_llm() calls response.get_message().model_dump(),
        so this must return a Pydantic model, not a plain dict.
        """
        return AssistantMessage(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls if self.tool_calls else None,
        )

    def get_content(self) -> str:
        """Get the text content of the response."""
        return self.content

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get list of tool calls from the response."""
        return self.tool_calls

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class AnthropicChatClient(BaseModel):
    """
    Anthropic Chat Client for dapr-agents with full tool support.

    Uses the native Anthropic Python SDK to enable tool/function calling,
    which the Dapr Conversation API doesn't support.

    Args:
        model: Claude model to use (default: claude-sonnet-4-20250514)
        api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
        max_tokens: Maximum tokens in response (default: 4096)
        temperature: Sampling temperature (default: 0.7)
    """

    model: str = Field(default="claude-sonnet-4-20250514")
    api_key: Optional[str] = Field(default=None)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)

    # Required by ChatClientBase interface
    prompty: Optional[Any] = Field(default=None)
    prompt_template: Optional[Any] = Field(default=None)

    # Internal client (set after initialization)
    _client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        # Get API key from env if not provided
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter is required")

        # Initialize the Anthropic client
        object.__setattr__(self, '_client', anthropic.Anthropic(api_key=api_key))

    @classmethod
    def from_prompty(cls, prompty_source, timeout=1500):
        """Load from Prompty spec (not implemented - returns default client)."""
        return cls()

    def generate(
        self,
        messages: Union[str, Dict[str, Any], Any, Iterable[Union[Dict[str, Any], Any]]] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[Any, Dict[str, Any]]]] = None,
        response_format: Optional[Type[T]] = None,
        structured_mode: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMChatResponse, T, List[T]]:
        """
        Generate a chat completion using Anthropic's API.

        Args:
            messages: Conversation messages (string or list of dicts)
            input_data: Variables for template formatting
            model: Override model for this request
            tools: List of tools available to the model
            response_format: Pydantic model for structured output
            structured_mode: "json" or "function_call"
            stream: Whether to stream the response
            **kwargs: Additional API parameters

        Returns:
            LLMChatResponse with content and any tool calls
        """
        # Debug logging
        print(f"[AnthropicChatClient] generate called with messages type={type(messages)}, messages={str(messages)[:500] if messages else 'None'}")
        print(f"[AnthropicChatClient] kwargs={list(kwargs.keys())}")

        # Normalize messages and extract system messages
        anthropic_messages, system_content = self._normalize_messages(messages)
        print(f"[AnthropicChatClient] After normalize: {len(anthropic_messages)} messages, system={bool(system_content)}")

        # Convert tools to Anthropic format
        anthropic_tools = self._convert_tools(tools) if tools else None

        # Build request parameters
        request_params = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }

        if anthropic_tools:
            request_params["tools"] = anthropic_tools

        # Add system message - prefer extracted from messages, then kwargs
        if system_content:
            request_params["system"] = system_content
        elif "system" in kwargs:
            request_params["system"] = kwargs.pop("system")

        try:
            # Call Anthropic API
            response = self._client.messages.create(**request_params)

            # Parse response
            return self._parse_response(response, response_format)

        except Exception as e:
            print(f"[AnthropicChatClient] API error: {e}")
            raise

    def _normalize_messages(
        self,
        messages: Union[str, Dict[str, Any], Any, Iterable[Union[Dict[str, Any], Any]]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert messages to Anthropic format.

        Returns:
            Tuple of (messages_list, system_content)
            System messages are extracted and returned separately since
            Anthropic requires them as a top-level parameter.
        """
        if messages is None:
            return [], None

        if isinstance(messages, str):
            return [{"role": "user", "content": messages}], None

        if isinstance(messages, dict):
            if messages.get("role") == "system":
                return [], messages.get("content", "")
            return [messages], None

        # Handle iterable - extract system messages
        result = []
        system_parts = []

        for msg in messages:
            if isinstance(msg, str):
                result.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                # Extract system messages
                if msg.get("role") == "system":
                    system_parts.append(msg.get("content", ""))
                    continue

                # Convert OpenAI-style "assistant" content with tool_calls
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    # Convert to Anthropic tool_use format
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    for tc in msg.get("tool_calls", []):
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        })
                    result.append({"role": "assistant", "content": content})
                elif msg.get("role") == "tool":
                    # Convert tool result to Anthropic format
                    result.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }]
                    })
                else:
                    result.append(msg)
            elif hasattr(msg, 'model_dump') and callable(getattr(msg, 'model_dump', None)):
                # Pydantic model - convert to dict
                try:
                    dumped = msg.model_dump()
                    if dumped.get("role") == "system":
                        system_parts.append(dumped.get("content", ""))
                    else:
                        result.append(dumped)
                except (AttributeError, TypeError):
                    # Fallback if model_dump fails
                    result.append({"role": "user", "content": str(msg)})
            else:
                result.append({"role": "user", "content": str(msg)})

        # Combine system parts
        system_content = "\n\n".join(system_parts) if system_parts else None

        # Anthropic requires at least one user message
        # If we only have system messages, add a minimal user message
        if not result and system_content:
            result.append({"role": "user", "content": "Please proceed with your instructions."})

        return result, system_content

    def _convert_tools(
        self,
        tools: List[Union[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Convert dapr-agents tools to Anthropic tool format."""
        anthropic_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                # Already in dict format
                if "function" in tool:
                    # OpenAI format
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    })
                else:
                    # Assume already Anthropic format
                    anthropic_tools.append(tool)
            elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                # AgentTool or similar object
                input_schema = {"type": "object", "properties": {}}
                if hasattr(tool, 'args_model') and tool.args_model:
                    input_schema = tool.args_model.model_json_schema()
                elif hasattr(tool, 'parameters'):
                    input_schema = tool.parameters

                anthropic_tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": input_schema,
                })
            elif hasattr(tool, 'to_dict'):
                anthropic_tools.append(tool.to_dict())

        return anthropic_tools

    def _parse_response(
        self,
        response: Any,
        response_format: Optional[Type[T]] = None
    ) -> Union[LLMChatResponse, T, List[T]]:
        """Parse Anthropic response into LLMChatResponse."""
        content_text = ""
        tool_calls = []

        for block in response.content:
            if hasattr(block, 'text'):
                content_text += block.text
            elif hasattr(block, 'type') and block.type == "tool_use":
                # Convert to OpenAI-compatible format for dapr-agents
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    }
                })

        llm_response = LLMChatResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
            },
            raw_response=response,
        )

        # Handle structured output
        if response_format and content_text:
            try:
                data = json.loads(content_text)
                return response_format(**data)
            except (json.JSONDecodeError, Exception):
                pass

        return llm_response


def create_anthropic_client(
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> AnthropicChatClient:
    """
    Factory function to create an AnthropicChatClient.

    Args:
        model: Claude model to use
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Configured AnthropicChatClient instance
    """
    return AnthropicChatClient(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
