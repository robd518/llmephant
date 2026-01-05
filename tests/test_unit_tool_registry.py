# FILE: test_unit_tool_registry.py

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest


@dataclass
class FakeMCPTool:
    name: str
    description: str
    title: str
    input_schema: Dict[str, Any]


class FakeMCPProvider:
    """A minimal MCP-like provider for registry unit tests."""

    def __init__(self, provider_name: str, tools: List[FakeMCPTool]):
        self.name = provider_name
        self.provider_name = provider_name
        self._tools = tools

    def _full_name(self, tool_name: str) -> str:
        return f"{self.provider_name}:{tool_name}"

    async def list_tools(self) -> List[FakeMCPTool]:
        return self._tools


def _call_register_provider(registry: Any, provider_name: str, provider: Any) -> None:
    fn = getattr(registry, "register_provider")
    params = list(inspect.signature(fn).parameters.values())

    # Common:
    #   register_provider(name, provider)
    #   register_provider(provider)
    if len(params) == 1:
        fn(provider)
        return
    if len(params) == 2:
        fn(provider_name, provider)
        return

    try:
        fn(provider_name, provider)
    except TypeError:
        fn(provider)


def _call_register_tool(registry: Any, **tool_fields: Any) -> None:
    fields = dict(tool_fields)
    if "parameters" not in fields and "input_schema" in fields:
        fields["parameters"] = fields["input_schema"]

    fn = getattr(registry, "register_tool")
    params = list(inspect.signature(fn).parameters.values())

    def _make_tool_obj() -> Any:
        try:
            from llmephant.tools.registry import RegisteredTool  # type: ignore

            ctor_sig = inspect.signature(RegisteredTool)
            ctor_params = set(ctor_sig.parameters.keys())

            kwargs: Dict[str, Any] = {}
            for k, v in fields.items():
                if k in ctor_params:
                    kwargs[k] = v

            if "parameters" in ctor_params and "parameters" not in kwargs and "input_schema" in fields:
                kwargs["parameters"] = fields["input_schema"]
            if "input_schema" in ctor_params and "input_schema" not in kwargs and "parameters" in fields:
                kwargs["input_schema"] = fields["parameters"]

            tool = RegisteredTool(**kwargs)
            return tool
        except Exception:
            class _ToolObj:
                pass

            obj = _ToolObj()
            for k, v in fields.items():
                setattr(obj, k, v)
            return obj

    if len(params) == 1:
        fn(_make_tool_obj())
        return

    try:
        fn(**fields)
    except TypeError:
        fn(_make_tool_obj())


@pytest.fixture
def registry():
    from llmephant.tools.registry import ToolRegistry

    return ToolRegistry()


def test_register_and_get_provider(registry):
    provider = FakeMCPProvider("mcp", tools=[])
    _call_register_provider(registry, "mcp", provider)
    assert registry.get_provider("mcp") is provider


def test_register_and_get_tool_roundtrip(registry):
    provider = FakeMCPProvider("mcp", tools=[])
    _call_register_provider(registry, "mcp", provider)

    _call_register_tool(
        registry,
        name="mcp:echo",
        description="Echo tool",
        title="Echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        provider_name="mcp",
        provider_tool_name="echo",
    )

    tool = registry.get_tool("mcp:echo")
    assert tool is not None
    assert getattr(tool, "name", None) == "mcp:echo"
    assert getattr(tool, "provider_name", None) == "mcp"
    assert getattr(tool, "provider_tool_name", None) == "echo"


def test_openai_tools_shape(registry):
    provider = FakeMCPProvider("mcp", tools=[])
    _call_register_provider(registry, "mcp", provider)

    _call_register_tool(
        registry,
        name="mcp:echo",
        description="Echo tool",
        title="Echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        provider_name="mcp",
        provider_tool_name="echo",
    )

    tools = registry.openai_tools()
    assert isinstance(tools, list)

    by_name = {t.get("function", {}).get("name"): t for t in tools}
    assert "mcp:echo" in by_name

    entry = by_name["mcp:echo"]
    assert entry.get("type") == "function"
    fn = entry.get("function", {})
    assert fn.get("name") == "mcp:echo"
    assert isinstance(fn.get("parameters"), dict)


@pytest.mark.asyncio
async def test_import_mcp_tools_registers_prefixed_names(registry):
    try:
        from llmephant.tools.registry import import_mcp_tools as import_fn
    except Exception:
        pytest.skip("import_mcp_tools helper not found in llmephant.tools.registry")

    tools = [
        FakeMCPTool(
            name="echo",
            title="Echo",
            description="Echo tool",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
        FakeMCPTool(
            name="ping",
            title="Ping",
            description="Ping tool",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    provider = FakeMCPProvider("mcp", tools=tools)

    await import_fn(registry, provider)

    assert registry.get_provider("mcp") is provider
    assert registry.get_tool("mcp:echo") is not None
    assert registry.get_tool("mcp:ping") is not None

    openai = registry.openai_tools()
    names = {t.get("function", {}).get("name") for t in openai}
    assert {"mcp:echo", "mcp:ping"}.issubset(names)