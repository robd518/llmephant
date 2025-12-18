# tools/executor.py

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from llmephant.tools.registry import ToolRegistry


class ToolExecutionResult(TypedDict):
    """Normalized tool execution result consumed by chat_runtime.

    - result: what should be sent back to the LLM as the tool output (often a string)
    - is_error: whether the tool call failed
    - error: optional human-readable error message
    - raw: optional raw provider payload (useful for debugging / memory extraction)
    """

    result: Any
    is_error: bool
    error: Optional[str]
    raw: Optional[Any]


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {
                "result": None,
                "is_error": True,
                "error": f"Unknown tool: {tool_name}",
                "raw": None,
            }

        provider = self.registry.get_provider(tool.provider_name)

        # MCP provider returns ToolCallResult(text=..., raw=..., is_error=...)
        result = await provider.call_tool(tool.provider_tool_name, arguments)

        return {
            "result": result.text,
            "is_error": bool(result.is_error),
            "error": None if not result.is_error else "Tool call returned is_error=true",
            "raw": result.raw,
        }