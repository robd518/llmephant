from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from llmephant.models.memory_model import MemoryPolicy


class ToolDefinition(BaseModel):
    """
    A tool's declared interface, independent of any specific provider/protocol.
    - `name` is the model-facing tool name (must be stable).
    - `provider` is optional and helps disambiguate collisions when multiple tool
      backends provide the same `name`.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None

    # Optional metadata
    provider: Optional[str] = None
    version: Optional[str] = None

    # Controls whether/how tool output should be stored as memory.
    memory_policy: Optional[MemoryPolicy] = None


class ToolCall(BaseModel):
    """
    A single model-requested invocation of a tool.
    This is *not* the same thing as a ToolDefinition.

    Notes:
    - `name` is the tool name the model requested.
    - `arguments` should be the decoded JSON arguments.
    - `call_id` is optional but useful for correlating results (OpenAI uses "id").
    """

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None

    # Optional fields to support multi-provider routing and debugging
    provider: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class ToolResult(BaseModel):
    """
    The result of executing a ToolCall.
    """

    name: str
    result: Any = None

    call_id: Optional[str] = None
    is_error: bool = False
    error: Optional[str] = None

    # Optional raw provider response (useful for provenance/debugging)
    raw: Optional[Dict[str, Any]] = None
