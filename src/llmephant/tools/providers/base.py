

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from llmephant.models.tool_model import ToolDefinition


@runtime_checkable
class ToolProvider(Protocol):
    """
    Protocol for any tool backend (MCP, local python, HTTP APIs, etc.).
    """

    async def list_tools(self) -> List[ToolDefinition]:
        ...

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        ...