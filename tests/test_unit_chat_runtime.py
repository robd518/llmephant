# FILE: test_unit_chat_runtime.py

import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pytest


# ----------------------------
# Item 1: ToolExecutor.execute() ToolResult contract
# ----------------------------


def _import_executor_symbols() -> Tuple[Any, Any]:
    """Import ToolExecutor/ToolResult from the most likely locations."""
    candidates = [
        ("llmephant.tools.executor", ("ToolExecutor", "ToolResult")),
        ("llmephant.tools.tool_executor", ("ToolExecutor", "ToolResult")),
    ]

    last_err: Optional[Exception] = None
    for mod_name, (exec_name, res_name) in candidates:
        try:
            mod = __import__(mod_name, fromlist=[exec_name, res_name])
            return getattr(mod, exec_name), getattr(mod, res_name)
        except Exception as e:  # pragma: no cover
            last_err = e

    raise ImportError(f"Could not import ToolExecutor/ToolResult: {last_err}")


def _import_registry_symbols() -> Tuple[Any, Any]:
    candidates = [
        ("llmephant.tools.registry", ("ToolRegistry", "RegisteredTool")),
        ("llmephant.tools.tool_registry", ("ToolRegistry", "RegisteredTool")),
    ]

    last_err: Optional[Exception] = None
    for mod_name, (reg_name, tool_name) in candidates:
        try:
            mod = __import__(mod_name, fromlist=[reg_name, tool_name])
            return getattr(mod, reg_name), getattr(mod, tool_name)
        except Exception as e:  # pragma: no cover
            last_err = e

    raise ImportError(f"Could not import ToolRegistry/RegisteredTool: {last_err}")


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


def _call_register_tool(registry: Any, tool_obj: Any) -> None:
    fn = getattr(registry, "register_tool")
    params = list(inspect.signature(fn).parameters.values())

    # Shape: register_tool(tool)
    if len(params) == 1:
        fn(tool_obj)
        return

    schema = getattr(tool_obj, "parameters", None) or {}
    try:
        fn(
            name=getattr(tool_obj, "name"),
            description=getattr(tool_obj, "description", ""),
            title=getattr(tool_obj, "title", ""),
            parameters=schema,
            provider_name=getattr(tool_obj, "provider_name"),
            provider_tool_name=getattr(tool_obj, "provider_tool_name"),
        )
    except TypeError:
        fn(tool_obj)


def _make_registered_tool(
    name: str, provider_name: str, provider_tool_name: str
) -> Any:
    _ToolRegistry, RegisteredTool = _import_registry_symbols()

    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    ctor_sig = inspect.signature(RegisteredTool)
    ctor_params = set(ctor_sig.parameters.keys())

    # NOTE: We no longer assume an input_schema field exists. Use parameters only.
    fields: Dict[str, Any] = {
        "name": name,
        "description": "test",
        "title": "Test",
        "parameters": parameters,
        "provider_name": provider_name,
        "provider_tool_name": provider_tool_name,
    }

    kwargs = {k: v for k, v in fields.items() if k in ctor_params}
    return RegisteredTool(**kwargs)


@dataclass
class _AttrProviderResult:
    result: Any
    is_error: bool
    error: Optional[str] = None


class FakeProvider:
    """Provider double covering common call method names."""

    def __init__(self, name: str, return_shape: str = "dict"):
        self.name = name
        self.provider_name = name
        self.return_shape = return_shape
        self.last_tool: Optional[str] = None
        self.last_args: Optional[Dict[str, Any]] = None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        return await self._do(tool_name, arguments)

    async def execute(self, tool_name: str, arguments: Dict[str, Any]):
        return await self._do(tool_name, arguments)

    async def run_tool(self, tool_name: str, arguments: Dict[str, Any]):
        return await self._do(tool_name, arguments)

    async def invoke(self, tool_name: str, arguments: Dict[str, Any]):
        return await self._do(tool_name, arguments)

    async def _do(self, tool_name: str, arguments: Dict[str, Any]):
        self.last_tool = tool_name
        self.last_args = dict(arguments)

        payload = {"echo": arguments, "tool": tool_name}
        if self.return_shape == "dict":
            return {"result": payload, "is_error": False, "error": None}
        if self.return_shape == "attr":
            return _AttrProviderResult(result=payload, is_error=False, error=None)
        if self.return_shape == "error_dict":
            return {"result": None, "is_error": True, "error": "boom"}
        return payload


@pytest.fixture
def registry():
    ToolRegistry, _RegisteredTool = _import_registry_symbols()
    return ToolRegistry()


@pytest.fixture
def executor(registry):
    ToolExecutor, _ToolResult = _import_executor_symbols()
    return ToolExecutor(registry)


@pytest.mark.asyncio
async def test_executor_unknown_tool_returns_error_toolresult(executor):
    _ToolExecutor, ToolResult = _import_executor_symbols()

    tr = await executor.execute("nope:missing", {})
    assert isinstance(tr, ToolResult)
    assert tr.is_error is True
    assert tr.error


@pytest.mark.asyncio
async def test_executor_args_none_coerces_to_empty_dict(registry, executor):
    _ToolExecutor, ToolResult = _import_executor_symbols()

    provider = FakeProvider("mcp", return_shape="dict")
    _call_register_provider(registry, "mcp", provider)

    tool = _make_registered_tool("mcp:echo", "mcp", "echo")
    _call_register_tool(registry, tool)

    tr = await executor.execute("mcp:echo", None)
    assert isinstance(tr, ToolResult)
    assert tr.is_error is False
    assert provider.last_args == {}


@pytest.mark.asyncio
async def test_executor_normalizes_dict_provider_return_to_toolresult(
    registry, executor
):
    _ToolExecutor, ToolResult = _import_executor_symbols()

    provider = FakeProvider("mcp", return_shape="dict")
    _call_register_provider(registry, "mcp", provider)

    tool = _make_registered_tool("mcp:echo", "mcp", "echo")
    _call_register_tool(registry, tool)

    tr = await executor.execute("mcp:echo", {"text": "hi"})
    assert isinstance(tr, ToolResult)
    assert tr.is_error is False
    assert isinstance(tr.result, dict)
    assert tr.result.get("echo") == {"text": "hi"}


@pytest.mark.asyncio
async def test_executor_normalizes_attr_provider_return_to_toolresult(
    registry, executor
):
    _ToolExecutor, ToolResult = _import_executor_symbols()

    provider = FakeProvider("mcp", return_shape="attr")
    _call_register_provider(registry, "mcp", provider)

    tool = _make_registered_tool("mcp:echo", "mcp", "echo")
    _call_register_tool(registry, tool)

    tr = await executor.execute("mcp:echo", {"text": "hi"})
    assert isinstance(tr, ToolResult)
    assert tr.is_error is False
    assert isinstance(tr.result, dict)
    assert tr.result.get("echo") == {"text": "hi"}


@pytest.mark.asyncio
async def test_executor_validates_tool_name_type(registry, executor):
    _ToolExecutor, ToolResult = _import_executor_symbols()

    provider = FakeProvider("mcp", return_shape="dict")
    _call_register_provider(registry, "mcp", provider)

    tool = _make_registered_tool("mcp:echo", "mcp", "echo")
    _call_register_tool(registry, tool)

    tr = await executor.execute("echo", {"text": "hi"})  # type: ignore[arg-type]
    assert isinstance(tr, ToolResult)
    assert tr.is_error is True
    assert tr.error


# ----------------------------
# Item 2: chat_runtime helper tests
# ----------------------------


def _import_chat_runtime_helpers():
    candidates = [
        "llmephant.services.chat_runtime",
        "llmephant.service.chat_runtime",
    ]

    last_err: Optional[Exception] = None
    for mod_name in candidates:
        try:
            mod = __import__(
                mod_name, fromlist=["_tool_result_to_content", "_require_tooling"]
            )
            return getattr(mod, "_tool_result_to_content"), getattr(
                mod, "_require_tooling"
            )
        except Exception as e:
            last_err = e

    raise ImportError(f"Could not import chat_runtime helpers: {last_err}")


def _make_request_with_state(**state_items: Any):
    from fastapi import FastAPI
    from starlette.requests import Request

    app = FastAPI()
    for k, v in state_items.items():
        setattr(app.state, k, v)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "app": app,
    }
    return Request(scope)


def test_tool_result_to_content_passes_through_string():
    _tool_result_to_content, _require_tooling = _import_chat_runtime_helpers()

    class TR:
        def __init__(self):
            self.result = "hello"
            self.is_error = False
            self.error = None

    assert _tool_result_to_content(TR()) == "hello"


def test_tool_result_to_content_json_dumps_non_string():
    _tool_result_to_content, _require_tooling = _import_chat_runtime_helpers()

    class TR:
        def __init__(self):
            self.result = {"a": 1}
            self.is_error = False
            self.error = None

    out = _tool_result_to_content(TR())
    assert json.loads(out) == {"a": 1}


def test_tool_result_to_content_error_serializes_error_json():
    _tool_result_to_content, _require_tooling = _import_chat_runtime_helpers()

    class TR:
        def __init__(self):
            self.result = {"ignored": True}
            self.is_error = True
            self.error = "nope"

    out = _tool_result_to_content(TR())
    parsed = json.loads(out)
    assert "error" in parsed
    assert parsed["error"]


def test_require_tooling_raises_and_includes_state_keys():
    _tool_result_to_content, _require_tooling = _import_chat_runtime_helpers()

    req = _make_request_with_state(registry=object())
    with pytest.raises(RuntimeError) as e:
        _require_tooling(req)

    msg = str(e.value)
    assert "executor" in msg
    assert "registry" in msg


def test_require_tooling_passes_when_both_present():
    _tool_result_to_content, _require_tooling = _import_chat_runtime_helpers()

    req = _make_request_with_state(registry=object(), executor=object())
    _require_tooling(req)


# ----------------------------
# Item 3: tool advertisement gating (no tools/tool_choice when disabled)
# ----------------------------


def _import_chat_runtime_runners():
    candidates = [
        "llmephant.services.chat_runtime",
        "llmephant.service.chat_runtime",
    ]

    last_err: Optional[Exception] = None
    for mod_name in candidates:
        try:
            mod = __import__(
                mod_name, fromlist=["run_chat_runtime", "run_chat_runtime_stream"]
            )
            return (
                mod,
                getattr(mod, "run_chat_runtime"),
                getattr(mod, "run_chat_runtime_stream"),
            )
        except Exception as e:
            last_err = e

    raise ImportError(f"Could not import chat_runtime runners: {last_err}")


# Inserted dataclasses for object-based mock returns


@dataclass
class _Msg:
    role: str
    content: str


@dataclass
class _UpstreamChoice:
    message: _Msg


@dataclass
class _UpstreamResp:
    choices: list[_UpstreamChoice]


@dataclass
class _DummyChatReq:
    model: str = "test-model"
    messages: Any = None
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 16
    tools: Any = None
    tool_choice: Any = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = [_Msg(role="user", content="hi")]


class _StubRegistry:
    def __init__(self, tools: Optional[list] = None):
        self._tools = tools or []

    def openai_tools(self):
        return list(self._tools)


@pytest.mark.asyncio
async def test_tools_disabled_omits_tools_and_tool_choice_non_streaming(monkeypatch):
    mod, run_chat_runtime, _run_chat_runtime_stream = _import_chat_runtime_runners()

    captured: Dict[str, Any] = {}

    class CapturingChatRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    async def fake_chat_upstream(chat_req):
        captured["kwargs"] = getattr(chat_req, "kwargs", {})
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(mod, "ChatRequest", CapturingChatRequest)
    monkeypatch.setattr(mod, "chat_upstream", fake_chat_upstream)
    monkeypatch.setattr(mod, "search_relevant_memories", lambda _user_id, _query: [])

    request = _make_request_with_state(
        registry=_StubRegistry(tools=[{"type": "function", "function": {"name": "x"}}]),
        executor=object(),
        tools_enabled=False,
    )
    req = _DummyChatReq()

    async for event in run_chat_runtime(req, user_id="test-user", request=request):
        if event.get("type") == "done":
            break

    assert "tools" not in captured.get("kwargs", {})
    assert "tool_choice" not in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_tools_disabled_omits_tools_and_tool_choice_streaming(monkeypatch):
    mod, _run_chat_runtime, run_chat_runtime_stream = _import_chat_runtime_runners()

    captured: Dict[str, Any] = {}

    class CapturingChatRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    async def fake_chat_upstream_stream(chat_req):
        captured["kwargs"] = getattr(chat_req, "kwargs", {})
        # One content chunk; no tool deltas.
        yield {"choices": [{"delta": {"content": "ok"}}]}

    monkeypatch.setattr(mod, "ChatRequest", CapturingChatRequest)
    monkeypatch.setattr(mod, "chat_upstream_stream", fake_chat_upstream_stream)
    monkeypatch.setattr(mod, "search_relevant_memories", lambda _user_id, _query: [])

    request = _make_request_with_state(
        registry=_StubRegistry(tools=[{"type": "function", "function": {"name": "x"}}]),
        executor=object(),
        tools_enabled=False,
    )
    req = _DummyChatReq()

    async for event in run_chat_runtime_stream(
        req, user_id="test-user", request=request
    ):
        if event.get("type") == "done":
            break

    assert "tools" not in captured.get("kwargs", {})
    assert "tool_choice" not in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_tools_enabled_but_empty_list_omits_tools_and_tool_choice(monkeypatch):
    mod, run_chat_runtime, _run_chat_runtime_stream = _import_chat_runtime_runners()

    captured: Dict[str, Any] = {}

    class CapturingChatRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    async def fake_chat_upstream(chat_req):
        captured["kwargs"] = getattr(chat_req, "kwargs", {})
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(mod, "ChatRequest", CapturingChatRequest)
    monkeypatch.setattr(mod, "chat_upstream", fake_chat_upstream)
    monkeypatch.setattr(mod, "search_relevant_memories", lambda _user_id, _query: [])

    request = _make_request_with_state(
        registry=_StubRegistry(tools=[]),
        executor=object(),
        tools_enabled=True,
    )
    req = _DummyChatReq()

    async for event in run_chat_runtime(req, user_id="test-user", request=request):
        if event.get("type") == "done":
            break

    assert "tools" not in captured.get("kwargs", {})
    assert "tool_choice" not in captured.get("kwargs", {})
