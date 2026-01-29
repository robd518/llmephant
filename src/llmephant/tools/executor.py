# tools/executor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from llmephant.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool execution result consumed by chat_runtime.

    - result: what should be sent back to the LLM as the tool output (often a string)
    - is_error: whether the tool call failed
    - error: optional human-readable error message
    - raw: optional raw provider payload (useful for debugging / memory extraction)
    """

    result: Any
    is_error: bool
    error: Optional[str] = None
    raw: Optional[Any] = None


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any] | None
    ) -> ToolResult:
        # Prefer returning ToolResult over raising, so callers have one consistent shape.
        if not isinstance(tool_name, str):
            return ToolResult(
                result=None,
                is_error=True,
                error=f"tool_name must be str, got {type(tool_name).__name__}",
                raw={"tool_name": tool_name, "arguments": arguments},
            )

        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return ToolResult(
                result=None,
                is_error=True,
                error=f"arguments must be dict, got {type(arguments).__name__}",
                raw={"tool_name": tool_name, "arguments": arguments},
            )

        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                result=None,
                is_error=True,
                error=f"Unknown tool: {tool_name}",
                raw=None,
            )

        provider = self.registry.get_provider(tool.provider_name)
        if provider is None:
            return ToolResult(
                result=None,
                is_error=True,
                error=f"Unknown provider: {tool.provider_name}",
                raw={"tool_name": tool_name, "provider_name": tool.provider_name},
            )

        # MCP provider typically returns ToolCallResult(text=..., raw=..., is_error=...)
        provider_result = await provider.call_tool(tool.provider_tool_name, arguments)

        # Normalize provider return into our canonical ToolResult shape.
        # Supports:
        #   - ToolResult
        #   - attribute-style: .text / .result / .is_error / .raw / .error
        #   - dict-style: {text|result, is_error, raw, error}
        #   - bare payload (treat as successful result)
        if isinstance(provider_result, ToolResult):
            return provider_result

        text: Any = getattr(provider_result, "text", None)
        if text is None:
            text = getattr(provider_result, "result", None)

        is_error: Any = getattr(provider_result, "is_error", False)
        raw: Any = getattr(provider_result, "raw", None)
        error: Optional[str] = getattr(provider_result, "error", None)

        if isinstance(provider_result, dict):
            # Common dict keys we may see from providers/adapters
            text = provider_result.get("text", provider_result.get("result", text))
            is_error = provider_result.get("is_error", is_error)
            raw = provider_result.get("raw", raw)
            error = provider_result.get("error", error)

        # If provider didn't supply a dedicated raw payload, keep the whole object for debugging.
        if raw is None:
            raw = provider_result

        # If provider returned a bare payload (no wrapper fields), treat that as the result.
        if text is None and provider_result is not None and not bool(is_error):
            text = provider_result

        if error is None and bool(is_error):
            error = "Tool call returned is_error=true"

        return ToolResult(
            result=text,
            is_error=bool(is_error),
            error=error,
            raw=raw,
        )
