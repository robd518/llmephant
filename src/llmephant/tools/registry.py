from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RegisteredTool:
    # name exposed to the LLM (often prefixed to avoid collisions)
    name: str
    description: str
    parameters: Dict[str, Any]

    # routing
    provider_name: str
    provider_tool_name: str  # the tool name at the provider (e.g. MCP "get_weather")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}
        self._providers: Dict[str, Any] = {}  # keep loose typing to fit your codebase

    def register_provider(self, provider: Any) -> None:
        self._providers[provider.name] = provider

    def get_provider(self, provider_name: str) -> Any:
        return self._providers[provider_name]

    def register_tool(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def openai_tools(self) -> List[Dict[str, Any]]:
        # Adapt to whatever your upstream LLM expects; this is OpenAI-style.
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

async def import_mcp_tools(registry: ToolRegistry, mcp_provider) -> None:
    registry.register_provider(mcp_provider)
    tools = await mcp_provider.list_tools()

    for t in tools:
        # Expose to the LLM using a stable prefixed name
        llm_name = mcp_provider._full_name(t.name)  # or a public helper if you prefer

        registry.register_tool(
            RegisteredTool(
                name=llm_name,
                description=t.description or (t.title or t.name),
                parameters=t.input_schema,
                provider_name=mcp_provider.name,
                provider_tool_name=t.name,
            )
        )