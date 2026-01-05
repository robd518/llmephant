import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict

import pytest


def _import_chat_service_module():
    # Expected location
    return __import__("llmephant.services.chat_service", fromlist=["dispatch_chat_request"])


async def _fake_run_chat_runtime_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
    """Emit tool-ish events plus output events.

    The SSE layer must forward only output and must not leak tool-call JSON.
    """

    # Tool-ish events that must NOT reach the client
    yield {
        "type": "tool_call_delta",
        "delta": '{"name":"mcp:echo","arguments":"{\\"text\\":\\"leak?\\"}"}',
    }
    yield {"type": "tool", "name": "mcp:echo", "arguments": {"text": "leak?"}}

    # Output that SHOULD reach the client
    yield {
        "type": "chunk",
        "chunk": {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        },
    }
    yield {
        "type": "chunk",
        "chunk": {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
        },
    }

    yield {"type": "done"}


def _make_request():
    from fastapi import FastAPI
    from starlette.requests import Request

    app = FastAPI()
    # dispatch_chat_request now preflights tooling on app.state even if tools are optional per-turn.
    # Provide a minimal registry/executor so the streaming path can be exercised.
    try:
        from llmephant.tools.registry import ToolRegistry
        from llmephant.tools.executor import ToolExecutor
    except Exception:
        pytest.skip("ToolRegistry/ToolExecutor imports unavailable for integration test")

    app.state.registry = ToolRegistry()
    app.state.executor = ToolExecutor(app.state.registry)
    app.state.tools_enabled = False

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "app": app,
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_streaming_sse_does_not_forward_tool_events(monkeypatch):
    chat_service = _import_chat_service_module()

    if not hasattr(chat_service, "dispatch_chat_request"):
        pytest.skip("chat_service.dispatch_chat_request not found")

    # Patch the runtime stream source used by dispatch_chat_request.
    # Be tolerant about the attribute name in case it changed.
    for stream_name in (
        "run_chat_runtime_stream",
        "run_chat_runtime_streaming",
        "_run_chat_runtime_stream",
    ):
        monkeypatch.setattr(chat_service, stream_name, _fake_run_chat_runtime_stream, raising=False)

    raw_req = _make_request()

    # dispatch_chat_request typically reads .stream and .messages (and sometimes .model/.user).
    req = SimpleNamespace(
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
        model="dummy",
        user=None,
    )

    resp = await chat_service.dispatch_chat_request(req, raw_req)

    # Ensure we actually went down the streaming path.
    assert getattr(resp, "media_type", None) == "text/event-stream"

    chunks: list[str] = []
    async for item in resp.body_iterator:
        if isinstance(item, (bytes, bytearray)):
            chunks.append(item.decode("utf-8", errors="replace"))
        else:
            chunks.append(str(item))

    stream_text = "".join(chunks)

    # Must contain the output content. Parse SSE frames and reconstruct delta content.
    content_parts: list[str] = []
    for frame in stream_text.split("\n\n"):
        if not frame:
            continue
        if not frame.startswith("data: "):
            continue
        payload = frame.removeprefix("data: ").strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        # OpenAI-style streaming chunk
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)

    final_text = "".join(content_parts)
    assert "Hello" in final_text
    assert "world" in final_text

    # Must not leak tool-call identifiers/JSON.
    assert "mcp:echo" not in stream_text
    assert "tool_call_delta" not in stream_text
    assert "arguments" not in stream_text

    # Optional sanity: any 'data: ' lines should contain either JSON or [DONE].
    for frame in stream_text.split("\n\n"):
        if not frame:
            continue
        if not frame.startswith("data: "):
            continue
        payload = frame.removeprefix("data: ")
        if payload.strip() == "[DONE]":
            continue
        # If it's JSON, it should parse; if not JSON, that's fine (some servers stream plain text).
        try:
            json.loads(payload)
        except Exception:
            pass
